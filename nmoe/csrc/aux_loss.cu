// nmoe/csrc/aux_loss.cu
//
// Fused MoE auxiliary loss computation.
//
// Replaces ~12 separate PyTorch kernel launches in model.py _compute_aux_loss()
// with a single fused CUDA kernel.
//
// The auxiliary loss (GShard-style load balancing) is:
//   aux_loss = alpha * E * sum(f * P)
// where:
//   f[e] = count(expert_ids == e) / (T * K)   -- dispatch fraction
//   P[e] = sum(gates where expert_ids == e) / (T * K)  -- mean gate probability
//
// This kernel computes f and P in a single pass over the data using atomic
// adds, then computes the final dot product and scalar loss.
//
// Kernel design:
//   - Phase 1: Each thread block processes a chunk of [T*K] elements
//   - Atomic adds to global f[] and P[] accumulators (FP32)
//   - Phase 2: Single-block reduction to compute dot(f, P)
//
// Target: sm_80+ (Ampere, Hopper, Blackwell) for FP32 atomicAdd.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace nmoe {
namespace aux_loss {

// ============================================================================
// Helpers
// ============================================================================

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// ============================================================================
// Constants
// ============================================================================

// Threads per block for accumulation kernel
constexpr int BLOCK_SIZE_ACCUM = 256;

// Threads per block for reduction kernel
constexpr int BLOCK_SIZE_REDUCE = 256;
constexpr int MAX_BLOCKS_FUSED = 256;
constexpr size_t DEFAULT_DYNAMIC_SMEM_LIMIT_BYTES = 48 * 1024;

// ============================================================================
// Kernel 1: Accumulate f[] and P[] counts
//
// Each thread processes one (expert_id, gate) pair and atomically adds to
// the global accumulators.
//
// Memory layout:
//   expert_ids [T*K] int32  -- flattened expert indices
//   gates      [T*K] float  -- flattened gate values (FP32 or BF16)
//   f_accum    [E]   float  -- count accumulator (zeroed by caller)
//   P_accum    [E]   float  -- gate sum accumulator (zeroed by caller)
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE_ACCUM)
k_aux_loss_accumulate_f32(
    const int32_t* __restrict__ expert_ids,  // [T*K]
    const float*   __restrict__ gates,       // [T*K] FP32
    float*         __restrict__ f_accum,     // [E]
    float*         __restrict__ P_accum,     // [E]
    int TK)  // T * K
{
    const int idx = blockIdx.x * BLOCK_SIZE_ACCUM + threadIdx.x;

    if (idx < TK) {
        const int eid = expert_ids[idx];
        const float gate = gates[idx];

        // Atomic add 1.0 to f[eid] and gate to P[eid]
        atomicAdd(&f_accum[eid], 1.0f);
        atomicAdd(&P_accum[eid], gate);
    }
}

// BF16 gates variant
__global__ void __launch_bounds__(BLOCK_SIZE_ACCUM)
k_aux_loss_accumulate_bf16(
    const int32_t*       __restrict__ expert_ids,  // [T*K]
    const __nv_bfloat16* __restrict__ gates,       // [T*K] BF16
    float*               __restrict__ f_accum,     // [E]
    float*               __restrict__ P_accum,     // [E]
    int TK)  // T * K
{
    const int idx = blockIdx.x * BLOCK_SIZE_ACCUM + threadIdx.x;

    if (idx < TK) {
        const int eid = expert_ids[idx];
        const float gate = bf16_to_f32(gates[idx]);

        atomicAdd(&f_accum[eid], 1.0f);
        atomicAdd(&P_accum[eid], gate);
    }
}

// FP16 gates variant
__global__ void __launch_bounds__(BLOCK_SIZE_ACCUM)
k_aux_loss_accumulate_f16(
    const int32_t* __restrict__ expert_ids,  // [T*K]
    const half*    __restrict__ gates,       // [T*K] FP16
    float*         __restrict__ f_accum,     // [E]
    float*         __restrict__ P_accum,     // [E]
    int TK)  // T * K
{
    const int idx = blockIdx.x * BLOCK_SIZE_ACCUM + threadIdx.x;

    if (idx < TK) {
        const int eid = expert_ids[idx];
        const float gate = __half2float(gates[idx]);

        atomicAdd(&f_accum[eid], 1.0f);
        atomicAdd(&P_accum[eid], gate);
    }
}

// ============================================================================
// Kernel 2: Compute dot(f, P) and final loss
//
// Single block reduction. After accumulation, f[] and P[] contain raw counts
// and gate sums. This kernel:
//   1. Normalizes: f[e] /= TK, P[e] /= TK
//   2. Computes: dot_product = sum(f[e] * P[e])
//   3. Returns: loss = alpha * E * dot_product
//
// The result is written to a single scalar output.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE_REDUCE)
k_aux_loss_reduce(
    const float* __restrict__ f_accum,    // [E] raw counts
    const float* __restrict__ P_accum,    // [E] raw gate sums
    float*       __restrict__ loss_out,   // [1] scalar output
    int E,
    int TK,      // T * K for normalization
    float alpha) // aux_loss_alpha
{
    __shared__ float smem[BLOCK_SIZE_REDUCE];

    const int tid = threadIdx.x;
    const float inv_TK = 1.0f / (float)TK;

    // Each thread accumulates partial sum over experts it handles
    float local_sum = 0.0f;
    for (int e = tid; e < E; e += BLOCK_SIZE_REDUCE) {
        float f_e = f_accum[e] * inv_TK;  // normalized dispatch fraction
        float P_e = P_accum[e] * inv_TK;  // normalized mean gate
        local_sum += f_e * P_e;
    }

    smem[tid] = local_sum;
    __syncthreads();

    // Block reduction (assumes BLOCK_SIZE_REDUCE is power of 2)
    for (int stride = BLOCK_SIZE_REDUCE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        loss_out[0] = alpha * (float)E * smem[0];
    }
}

// ============================================================================
// Fused single-kernel variant (for small E that fits in shared memory)
//
// Combines accumulation and reduction in a single kernel launch when E is
// small enough to fit per-expert accumulators in shared memory.
//
// Uses block-local shared memory accumulation with atomics, then inter-block
// atomics to global accumulators, followed by a grid sync and final reduction.
//
// This is more complex but reduces kernel launches from 2 to 1.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE_ACCUM)
k_aux_loss_fused(
    const int32_t* __restrict__ expert_ids,  // [T*K]
    const float*   __restrict__ gates,       // [T*K] FP32
    float*         __restrict__ f_global,    // [E] global accumulator (zero-initialized once)
    float*         __restrict__ P_global,    // [E] global accumulator (zero-initialized once)
    float*         __restrict__ loss_out,    // [1] scalar output
    int*           __restrict__ block_done,  // [1] counter for grid sync
    int E,
    int TK,
    float alpha,
    int total_blocks)
{
    extern __shared__ float smem[];  // [2*E]: f_local[E] and P_local[E]

    float* f_local = smem;
    float* P_local = smem + E;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Initialize shared memory accumulators to 0
    for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
        f_local[e] = 0.0f;
        P_local[e] = 0.0f;
    }
    __syncthreads();

    // Phase 1: Accumulate into shared memory (block-local)
    const int start_idx = bid * BLOCK_SIZE_ACCUM + tid;
    const int stride = gridDim.x * BLOCK_SIZE_ACCUM;

    for (int idx = start_idx; idx < TK; idx += stride) {
        const int eid = expert_ids[idx];
        const float gate = gates[idx];

        atomicAdd(&f_local[eid], 1.0f);
        atomicAdd(&P_local[eid], gate);
    }
    __syncthreads();

    // Phase 2: Flush shared memory to global accumulators
    for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
        if (f_local[e] > 0.0f) {
            atomicAdd(&f_global[e], f_local[e]);
        }
        if (P_local[e] != 0.0f) {
            atomicAdd(&P_global[e], P_local[e]);
        }
    }
    __threadfence();

    // Phase 3: Grid synchronization via atomic counter
    __shared__ bool is_last_block;
    if (tid == 0) {
        int finished = atomicAdd(block_done, 1);
        is_last_block = (finished == total_blocks - 1);
    }
    __syncthreads();

    // Phase 4: Last block computes final reduction
    if (is_last_block) {
        const float inv_TK = 1.0f / (float)TK;
        float local_sum = 0.0f;

        for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
            float f_e = f_global[e] * inv_TK;
            float P_e = P_global[e] * inv_TK;
            local_sum += f_e * P_e;
        }

        // Block reduction in shared memory (reuse smem)
        smem[tid] = local_sum;
        __syncthreads();

        for (int stride = BLOCK_SIZE_ACCUM / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            loss_out[0] = alpha * (float)E * smem[0];
        }

        // Keep scratch state reset for the next launch on this stream.
        for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
            f_global[e] = 0.0f;
            P_global[e] = 0.0f;
        }
        if (tid == 0) {
            block_done[0] = 0;
        }
    }
}

// BF16 variant of fused kernel
__global__ void __launch_bounds__(BLOCK_SIZE_ACCUM)
k_aux_loss_fused_bf16(
    const int32_t*       __restrict__ expert_ids,  // [T*K]
    const __nv_bfloat16* __restrict__ gates,       // [T*K] BF16
    float*               __restrict__ f_global,    // [E] global accumulator
    float*               __restrict__ P_global,    // [E] global accumulator
    float*               __restrict__ loss_out,    // [1] scalar output
    int*                 __restrict__ block_done,  // [1] counter
    int E,
    int TK,
    float alpha,
    int total_blocks)
{
    extern __shared__ float smem[];

    float* f_local = smem;
    float* P_local = smem + E;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
        f_local[e] = 0.0f;
        P_local[e] = 0.0f;
    }
    __syncthreads();

    const int start_idx = bid * BLOCK_SIZE_ACCUM + tid;
    const int stride = gridDim.x * BLOCK_SIZE_ACCUM;

    for (int idx = start_idx; idx < TK; idx += stride) {
        const int eid = expert_ids[idx];
        const float gate = bf16_to_f32(gates[idx]);

        atomicAdd(&f_local[eid], 1.0f);
        atomicAdd(&P_local[eid], gate);
    }
    __syncthreads();

    for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
        if (f_local[e] > 0.0f) {
            atomicAdd(&f_global[e], f_local[e]);
        }
        if (P_local[e] != 0.0f) {
            atomicAdd(&P_global[e], P_local[e]);
        }
    }
    __threadfence();

    __shared__ bool is_last_block;
    if (tid == 0) {
        int finished = atomicAdd(block_done, 1);
        is_last_block = (finished == total_blocks - 1);
    }
    __syncthreads();

    if (is_last_block) {
        const float inv_TK = 1.0f / (float)TK;
        float local_sum = 0.0f;

        for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
            float f_e = f_global[e] * inv_TK;
            float P_e = P_global[e] * inv_TK;
            local_sum += f_e * P_e;
        }

        smem[tid] = local_sum;
        __syncthreads();

        for (int stride = BLOCK_SIZE_ACCUM / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            loss_out[0] = alpha * (float)E * smem[0];
        }

        for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
            f_global[e] = 0.0f;
            P_global[e] = 0.0f;
        }
        if (tid == 0) {
            block_done[0] = 0;
        }
    }
}

// FP16 variant of fused kernel
__global__ void __launch_bounds__(BLOCK_SIZE_ACCUM)
k_aux_loss_fused_f16(
    const int32_t* __restrict__ expert_ids,  // [T*K]
    const half*    __restrict__ gates,       // [T*K] FP16
    float*         __restrict__ f_global,    // [E] global accumulator
    float*         __restrict__ P_global,    // [E] global accumulator
    float*         __restrict__ loss_out,    // [1] scalar output
    int*           __restrict__ block_done,  // [1] counter
    int E,
    int TK,
    float alpha,
    int total_blocks)
{
    extern __shared__ float smem[];

    float* f_local = smem;
    float* P_local = smem + E;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
        f_local[e] = 0.0f;
        P_local[e] = 0.0f;
    }
    __syncthreads();

    const int start_idx = bid * BLOCK_SIZE_ACCUM + tid;
    const int stride = gridDim.x * BLOCK_SIZE_ACCUM;

    for (int idx = start_idx; idx < TK; idx += stride) {
        const int eid = expert_ids[idx];
        const float gate = __half2float(gates[idx]);

        atomicAdd(&f_local[eid], 1.0f);
        atomicAdd(&P_local[eid], gate);
    }
    __syncthreads();

    for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
        if (f_local[e] > 0.0f) {
            atomicAdd(&f_global[e], f_local[e]);
        }
        if (P_local[e] != 0.0f) {
            atomicAdd(&P_global[e], P_local[e]);
        }
    }
    __threadfence();

    __shared__ bool is_last_block;
    if (tid == 0) {
        int finished = atomicAdd(block_done, 1);
        is_last_block = (finished == total_blocks - 1);
    }
    __syncthreads();

    if (is_last_block) {
        const float inv_TK = 1.0f / (float)TK;
        float local_sum = 0.0f;

        for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
            float f_e = f_global[e] * inv_TK;
            float P_e = P_global[e] * inv_TK;
            local_sum += f_e * P_e;
        }

        smem[tid] = local_sum;
        __syncthreads();

        for (int stride = BLOCK_SIZE_ACCUM / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            loss_out[0] = alpha * (float)E * smem[0];
        }

        for (int e = tid; e < E; e += BLOCK_SIZE_ACCUM) {
            f_global[e] = 0.0f;
            P_global[e] = 0.0f;
        }
        if (tid == 0) {
            block_done[0] = 0;
        }
    }
}

} // namespace aux_loss
} // namespace nmoe

// ============================================================================
// C API -- called from Python via ctypes
// ============================================================================

template <typename KernelT>
static inline cudaError_t configure_dynamic_smem_if_needed(
    KernelT kernel,
    size_t smem_bytes,
    const char* fn_name)
{
    if (smem_bytes <= nmoe::aux_loss::DEFAULT_DYNAMIC_SMEM_LIMIT_BYTES) {
        return cudaSuccess;
    }

    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;

    // Template-local cache (one cache per kernel symbol/type).
    thread_local int cached_device = -1;
    thread_local int cached_max_optin_smem = -1;
    thread_local int configured_device = -1;
    thread_local size_t configured_smem = 0;

    if (device != cached_device || cached_max_optin_smem <= 0) {
        int max_optin_smem = 0;
        err = cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (err != cudaSuccess) return err;
        if (max_optin_smem <= 0) {
            max_optin_smem = static_cast<int>(nmoe::aux_loss::DEFAULT_DYNAMIC_SMEM_LIMIT_BYTES);
        }
        cached_device = device;
        cached_max_optin_smem = max_optin_smem;
        configured_device = -1;
        configured_smem = 0;
    }
    const int max_optin_smem = cached_max_optin_smem;

    if (smem_bytes > static_cast<size_t>(max_optin_smem)) {
        fprintf(
            stderr,
            "%s: requested dynamic shared memory %zu bytes exceeds device opt-in limit %d bytes\n",
            fn_name,
            smem_bytes,
            max_optin_smem);
        return cudaErrorInvalidValue;
    }

    if (configured_device != device || configured_smem != smem_bytes) {
        err = cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (err != cudaSuccess) {
            fprintf(
                stderr,
                "%s: cudaFuncSetAttribute(max_dynamic_smem=%zu) failed: %s\n",
                fn_name,
                smem_bytes,
                cudaGetErrorString(err));
            return err;
        }
        configured_device = device;
        configured_smem = smem_bytes;
    }
    return cudaSuccess;
}

extern "C" {

/// Fused MoE auxiliary loss computation (two-kernel variant).
///
/// Computes: aux_loss = alpha * E * sum((f_e / TK) * (P_e / TK))
/// where f_e = count of tokens routed to expert e,
///       P_e = sum of gates for tokens routed to expert e.
///
/// Args:
///   expert_ids: [TK] int32 flattened expert indices
///   gates: [TK] float32 flattened gate values
///   f_accum: [E] float32 count accumulator (zeroed by caller)
///   P_accum: [E] float32 gate sum accumulator (zeroed by caller)
///   loss_out: [1] float32 scalar output
///   E: number of experts
///   TK: T * K (total token-expert assignments)
///   alpha: aux_loss_alpha coefficient
///   stream: CUDA stream
///
/// Returns cudaError_t.
cudaError_t fused_aux_loss_f32(
    const void* expert_ids,    // [TK] int32
    const void* gates,         // [TK] FP32
    void*       f_accum,       // [E] FP32 (zeroed)
    void*       P_accum,       // [E] FP32 (zeroed)
    void*       loss_out,      // [1] FP32
    int E,
    int TK,
    float alpha,
    cudaStream_t stream)
{
    using namespace nmoe::aux_loss;

    if (TK <= 0) {
        // No tokens, loss is 0
        cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
        return cudaSuccess;
    }

    // Kernel 1: Accumulate f and P
    int blocks = ceil_div(TK, BLOCK_SIZE_ACCUM);
    k_aux_loss_accumulate_f32<<<blocks, BLOCK_SIZE_ACCUM, 0, stream>>>(
        reinterpret_cast<const int32_t*>(expert_ids),
        reinterpret_cast<const float*>(gates),
        reinterpret_cast<float*>(f_accum),
        reinterpret_cast<float*>(P_accum),
        TK);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Kernel 2: Reduce to scalar loss
    k_aux_loss_reduce<<<1, BLOCK_SIZE_REDUCE, 0, stream>>>(
        reinterpret_cast<const float*>(f_accum),
        reinterpret_cast<const float*>(P_accum),
        reinterpret_cast<float*>(loss_out),
        E, TK, alpha);

    return cudaGetLastError();
}

/// BF16 gates variant.
cudaError_t fused_aux_loss_bf16(
    const void* expert_ids,    // [TK] int32
    const void* gates,         // [TK] BF16
    void*       f_accum,       // [E] FP32 (zeroed)
    void*       P_accum,       // [E] FP32 (zeroed)
    void*       loss_out,      // [1] FP32
    int E,
    int TK,
    float alpha,
    cudaStream_t stream)
{
    using namespace nmoe::aux_loss;

    if (TK <= 0) {
        cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
        return cudaSuccess;
    }

    int blocks = ceil_div(TK, BLOCK_SIZE_ACCUM);
    k_aux_loss_accumulate_bf16<<<blocks, BLOCK_SIZE_ACCUM, 0, stream>>>(
        reinterpret_cast<const int32_t*>(expert_ids),
        reinterpret_cast<const __nv_bfloat16*>(gates),
        reinterpret_cast<float*>(f_accum),
        reinterpret_cast<float*>(P_accum),
        TK);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    k_aux_loss_reduce<<<1, BLOCK_SIZE_REDUCE, 0, stream>>>(
        reinterpret_cast<const float*>(f_accum),
        reinterpret_cast<const float*>(P_accum),
        reinterpret_cast<float*>(loss_out),
        E, TK, alpha);

    return cudaGetLastError();
}

/// FP16 gates variant.
cudaError_t fused_aux_loss_f16(
    const void* expert_ids,    // [TK] int32
    const void* gates,         // [TK] FP16
    void*       f_accum,       // [E] FP32 (zeroed)
    void*       P_accum,       // [E] FP32 (zeroed)
    void*       loss_out,      // [1] FP32
    int E,
    int TK,
    float alpha,
    cudaStream_t stream)
{
    using namespace nmoe::aux_loss;

    if (TK <= 0) {
        cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
        return cudaSuccess;
    }

    int blocks = ceil_div(TK, BLOCK_SIZE_ACCUM);
    k_aux_loss_accumulate_f16<<<blocks, BLOCK_SIZE_ACCUM, 0, stream>>>(
        reinterpret_cast<const int32_t*>(expert_ids),
        reinterpret_cast<const half*>(gates),
        reinterpret_cast<float*>(f_accum),
        reinterpret_cast<float*>(P_accum),
        TK);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    k_aux_loss_reduce<<<1, BLOCK_SIZE_REDUCE, 0, stream>>>(
        reinterpret_cast<const float*>(f_accum),
        reinterpret_cast<const float*>(P_accum),
        reinterpret_cast<float*>(loss_out),
        E, TK, alpha);

    return cudaGetLastError();
}

/// Single-kernel fused variant (FP32 gates).
/// More efficient when E is small enough for shared memory.
/// Requires: smem_bytes = 2 * E * sizeof(float) <= device dynamic-smem limit.
cudaError_t fused_aux_loss_single_f32(
    const void* expert_ids,    // [TK] int32
    const void* gates,         // [TK] FP32
    void*       f_global,      // [E] FP32 accumulator scratch (zero-init once; reset internally per launch)
    void*       P_global,      // [E] FP32 accumulator scratch (zero-init once; reset internally per launch)
    void*       loss_out,      // [1] FP32
    void*       block_done,    // [1] int32 scratch (reset internally)
    int E,
    int TK,
    float alpha,
    cudaStream_t stream)
{
    using namespace nmoe::aux_loss;

    if (TK <= 0 || E <= 0) {
        cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
        return cudaSuccess;
    }

    // Shared memory: 2 * E floats for f_local and P_local
    size_t smem_bytes = 2ull * static_cast<size_t>(E) * sizeof(float);

    cudaError_t err = configure_dynamic_smem_if_needed(
        k_aux_loss_fused, smem_bytes, "fused_aux_loss_single_f32");
    if (err != cudaSuccess) return err;

    int blocks = ceil_div(TK, BLOCK_SIZE_ACCUM);
    if (blocks > MAX_BLOCKS_FUSED) {
        blocks = MAX_BLOCKS_FUSED;  // Limit blocks for grid-sync counter efficiency.
    }

    k_aux_loss_fused<<<blocks, BLOCK_SIZE_ACCUM, smem_bytes, stream>>>(
        reinterpret_cast<const int32_t*>(expert_ids),
        reinterpret_cast<const float*>(gates),
        reinterpret_cast<float*>(f_global),
        reinterpret_cast<float*>(P_global),
        reinterpret_cast<float*>(loss_out),
        reinterpret_cast<int*>(block_done),
        E, TK, alpha, blocks);

    return cudaGetLastError();
}

/// Single-kernel fused variant (BF16 gates).
cudaError_t fused_aux_loss_single_bf16(
    const void* expert_ids,    // [TK] int32
    const void* gates,         // [TK] BF16
    void*       f_global,      // [E] FP32 accumulator scratch (zero-init once; reset internally per launch)
    void*       P_global,      // [E] FP32 accumulator scratch (zero-init once; reset internally per launch)
    void*       loss_out,      // [1] FP32
    void*       block_done,    // [1] int32 scratch (reset internally)
    int E,
    int TK,
    float alpha,
    cudaStream_t stream)
{
    using namespace nmoe::aux_loss;

    if (TK <= 0 || E <= 0) {
        cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
        return cudaSuccess;
    }

    size_t smem_bytes = 2ull * static_cast<size_t>(E) * sizeof(float);

    cudaError_t err = configure_dynamic_smem_if_needed(
        k_aux_loss_fused_bf16, smem_bytes, "fused_aux_loss_single_bf16");
    if (err != cudaSuccess) return err;

    int blocks = ceil_div(TK, BLOCK_SIZE_ACCUM);
    if (blocks > MAX_BLOCKS_FUSED) {
        blocks = MAX_BLOCKS_FUSED;
    }

    k_aux_loss_fused_bf16<<<blocks, BLOCK_SIZE_ACCUM, smem_bytes, stream>>>(
        reinterpret_cast<const int32_t*>(expert_ids),
        reinterpret_cast<const __nv_bfloat16*>(gates),
        reinterpret_cast<float*>(f_global),
        reinterpret_cast<float*>(P_global),
        reinterpret_cast<float*>(loss_out),
        reinterpret_cast<int*>(block_done),
        E, TK, alpha, blocks);

    return cudaGetLastError();
}

/// Single-kernel fused variant (FP16 gates).
cudaError_t fused_aux_loss_single_f16(
    const void* expert_ids,    // [TK] int32
    const void* gates,         // [TK] FP16
    void*       f_global,      // [E] FP32 accumulator scratch (zero-init once; reset internally per launch)
    void*       P_global,      // [E] FP32 accumulator scratch (zero-init once; reset internally per launch)
    void*       loss_out,      // [1] FP32
    void*       block_done,    // [1] int32 scratch (reset internally)
    int E,
    int TK,
    float alpha,
    cudaStream_t stream)
{
    using namespace nmoe::aux_loss;

    if (TK <= 0 || E <= 0) {
        cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
        return cudaSuccess;
    }

    size_t smem_bytes = 2ull * static_cast<size_t>(E) * sizeof(float);

    cudaError_t err = configure_dynamic_smem_if_needed(
        k_aux_loss_fused_f16, smem_bytes, "fused_aux_loss_single_f16");
    if (err != cudaSuccess) return err;

    int blocks = ceil_div(TK, BLOCK_SIZE_ACCUM);
    if (blocks > MAX_BLOCKS_FUSED) {
        blocks = MAX_BLOCKS_FUSED;
    }

    k_aux_loss_fused_f16<<<blocks, BLOCK_SIZE_ACCUM, smem_bytes, stream>>>(
        reinterpret_cast<const int32_t*>(expert_ids),
        reinterpret_cast<const half*>(gates),
        reinterpret_cast<float*>(f_global),
        reinterpret_cast<float*>(P_global),
        reinterpret_cast<float*>(loss_out),
        reinterpret_cast<int*>(block_done),
        E, TK, alpha, blocks);

    return cudaGetLastError();
}

/// Aux-loss single-kernel capability flags for Python runtime contract checks.
int fused_aux_loss_single_caps() {
    constexpr int kHasInternalReset = 1 << 0;
    constexpr int kNoTwoKernelFallback = 1 << 1;
    return kHasInternalReset | kNoTwoKernelFallback;
}

} // extern "C"
