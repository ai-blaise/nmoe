// nmoe/csrc/router_bwd.cu
//
// Fused MoE router backward pass.
//
// Replaces ~18 separate PyTorch kernel launches in fused_router.py backward
// with a single fused CUDA kernel that exploits the sparsity of TopK routing,
// plus a small transpose+cast helper.
//
// Key insight: d_logits[n, e] is non-zero ONLY at the K selected expert
// positions per token. For K=8, E=256 this is 3% density -- the dense
// [N, E] intermediate is never materialized. Everything stays sparse at
// K positions per token, saving both memory and compute.
//
// Kernel design (k_fused_router_bwd):
//   One warp (32 threads) per token. Per token the warp:
//     1. Loads expert_ids[n, 0..K-1], pre_probs[n, 0..K-1],
//        gates[n, 0..K-1], grad_gates[n, 0..K-1]
//     2. Computes normalization VJP + sigmoid derivative from saved pre_probs
//        (no backward logit recompute pass)
//     3. Streams over D once to scatter grad_router_weight (FP32 atomics)
//        and optionally accumulate grad_hidden (BF16).
//
// Since multiple tokens route to the same expert, grad_router_weight uses
// FP32 atomicAdd for correctness.
//
// Complexity: O(N * K * D)  vs  O(N * E * D) in the dense formulation.
//   For K=8, E=256: 32x reduction. For K=8, E=128: 16x.
//
// Target: sm_80+ (Ampere, Hopper, Blackwell) for BF16 and FP32 atomicAdd.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace nmoe {
namespace router_bwd {

// ============================================================================
// Helpers
// ============================================================================

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float scalar_to_f32(float v) {
    return v;
}

__device__ __forceinline__ float scalar_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// ============================================================================
// Constants
// ============================================================================

// Maximum TopK we support via static register arrays.
// K is typically 1-8 for MoE; 16 covers all practical configurations.
constexpr int MAX_K = 16;

// Maximum warps per thread-block. Runtime launch picks 4 or 8 based on shape.
constexpr int MAX_WARPS_PER_BLOCK = 8;

// BF16 elements loaded per vectorized 128-bit transaction.
constexpr int VEC_BF16 = 8;  // sizeof(uint4) / sizeof(__nv_bfloat16)

// Transpose/cast tile shape.
constexpr int TRANSPOSE_TILE = 32;
constexpr int TRANSPOSE_BLOCK_ROWS_DEFAULT = 8;
constexpr int TRANSPOSE_BLOCK_ROWS_SMALL = 4;

__host__ __forceinline__ int select_router_bwd_warps_per_block(int N, int D, int K) {
    int w = (D >= 3072 || K >= 8) ? 8 : 4;
    if (N > 0 && w > N) w = N;
    if (w < 1) w = 1;
    if (w > MAX_WARPS_PER_BLOCK) w = MAX_WARPS_PER_BLOCK;
    return w;
}

__host__ __forceinline__ int select_router_bwd_transpose_block_rows(int E, int D) {
    // Single-tile transforms are launch-latency dominated; use fewer threads.
    if (E <= TRANSPOSE_TILE && D <= TRANSPOSE_TILE) {
        return TRANSPOSE_BLOCK_ROWS_SMALL;
    }
    return TRANSPOSE_BLOCK_ROWS_DEFAULT;
}

// ============================================================================
// Kernel 1: Fused Router Backward  (main computational kernel)
// ============================================================================
//
// Memory layout expectations (all row-major, contiguous):
//   hidden         [N, D]  BF16
//   router_weight  [D, E]  BF16   (element (d,e) at offset d*E + e)
//   expert_ids     [N, K]  int32
//   pre_probs      [N, K]  BF16/FP32 pre-normalized probs from forward
//   gates          [N, K]  BF16/FP32 normalized gates
//   grad_gates     [N, K]  BF16/FP32 incoming gradient
//   grad_rw        [E, D]  FP32   (output, zeroed by caller)
//   grad_hidden    [N, D]  BF16   (output, optional)
//
// Algorithm per token n:
//   For k = 0..K-1:
//     e_k = expert_ids[n, k]
//     prob_k = pre_probs[n, k]                                  // saved forward sigmoid
//   sig_deriv_k = prob_k * (1 - prob_k)
//   gate_sum = sum_k prob_k
//   dot_gs = sum_k grad_gates[n,k] * gates[n,k]                // VJP constant
//   grad_unnorm_k = (grad_gates[n,k] - dot_gs) / gate_sum
//   d_logit_k = grad_unnorm_k * sig_deriv_k * route_scale       // chain through logit*scale
//
//   grad_rw[e_k, d] += d_logit_k * hidden[n, d]                // atomic
//   grad_hidden[n, d] = sum_k d_logit_k * router_weight[d, e_k] // optional
//
// Implementation:
//   Single pass over D after scalar d_logit[k] setup from saved forward values.

template <bool kComputeGradHidden, bool kComputeGradW, typename GatesT, typename GradGatesT>
__global__ void __launch_bounds__(32 * MAX_WARPS_PER_BLOCK, 2)
k_fused_router_bwd(
    // Inputs
    const __nv_bfloat16* __restrict__ hidden,          // [N, D]
    const __nv_bfloat16* __restrict__ router_weight,   // [D, E]
    const int32_t*       __restrict__ expert_ids,      // [N, K]
    const GatesT*        __restrict__ pre_probs,       // [N, K] pre-normalized probs
    const GatesT*        __restrict__ gates,           // [N, K]
    const GradGatesT*    __restrict__ grad_gates,      // [N, K]
    // Outputs
    __nv_bfloat16*       __restrict__ grad_hidden,     // [N, D]  (NULL when !kComputeGradHidden)
    float*               __restrict__ grad_rw,         // [E, D] FP32 (NULL when !kComputeGradW)
    // Scalars
    float route_scale,
    int N, int E, int K, int D)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    const int lane = threadIdx.x;                       // 0..31
    const int warp_id = threadIdx.y;                    // 0..blockDim.y-1
    const int warps_per_block = blockDim.y;
    const int token_n = (int)blockIdx.x * warps_per_block + warp_id;

    if (token_n >= N) return;

    // ====================================================================
    // Phase 0: Load per-token metadata (K values).  Lane 0 reads, then
    //          broadcast via __shfl_sync to all 32 lanes.
    // ====================================================================

    int   eid[MAX_K];
    float prob_k[MAX_K];    // pre-normalized probs p_k = sigmoid(logit*scale)
    float gate_s[MAX_K];    // normalized gate (s_k) from forward
    float grad_s[MAX_K];    // incoming grad d(loss)/d(s_k)

    #pragma unroll
    for (int k = 0; k < MAX_K; k++) {
        int   e_tmp = 0;
        float p_tmp = 0.0f;
        float g_tmp = 0.0f;
        float dg_tmp = 0.0f;
        if (k < K && lane == 0) {
            const int idx = token_n * K + k;
            e_tmp  = expert_ids[idx];
            p_tmp  = scalar_to_f32(pre_probs[idx]);
            g_tmp  = scalar_to_f32(gates[idx]);
            dg_tmp = scalar_to_f32(grad_gates[idx]);
        }
        eid[k]    = __shfl_sync(0xFFFFFFFF, e_tmp,  0);
        prob_k[k] = __shfl_sync(0xFFFFFFFF, p_tmp,  0);
        gate_s[k] = __shfl_sync(0xFFFFFFFF, g_tmp,  0);
        grad_s[k] = __shfl_sync(0xFFFFFFFF, dg_tmp, 0);
    }

    // ====================================================================
    // Phase 1: Compute sigmoid derivative and normalization constants from
    //          pre-normalized probabilities captured in forward.
    // ====================================================================

    // ====================================================================
    // Phase 1+2: Scalar setup for this token (lane 0 only), then broadcast.
    //
    //   Forward: s_k = prob_k / S,  where S = sum_j prob_j (over K selected).
    //   VJP:     grad_unnorm_k = (grad_s_k - dot(grad_s, s)) / S
    //   Chain:   d_logit_k = grad_unnorm_k * sigmoid'(logit_k) * route_scale
    // ====================================================================
    float d_logit[MAX_K];
    if (lane == 0) {
        float gate_sum = 0.0f;
        float sig_d_k[MAX_K];
        #pragma unroll
        for (int k = 0; k < MAX_K; k++) {
            if (k < K) {
                const float p = prob_k[k];
                sig_d_k[k] = p * (1.0f - p);
                gate_sum += p;
            } else {
                sig_d_k[k] = 0.0f;
            }
        }
        gate_sum = fmaxf(gate_sum, 1e-12f);

        float dot_gs = 0.0f;
        #pragma unroll
        for (int k = 0; k < MAX_K; k++) {
            if (k < K) dot_gs += grad_s[k] * gate_s[k];
        }

        #pragma unroll
        for (int k = 0; k < MAX_K; k++) {
            if (k < K) {
                const float grad_unnorm = (grad_s[k] - dot_gs) / gate_sum;
                d_logit[k] = grad_unnorm * sig_d_k[k] * route_scale;
            } else {
                d_logit[k] = 0.0f;
            }
        }
    }
    #pragma unroll
    for (int k = 0; k < MAX_K; k++) {
        d_logit[k] = __shfl_sync(0xFFFFFFFFu, d_logit[k], 0);
    }

    // ====================================================================
    // Phase 2: Scatter into grad_router_weight and (optionally) accumulate
    //          grad_hidden.
    //
    //   grad_rw[eid[k], d] += d_logit[k] * hidden[n, d]       (atomicAdd)
    //   grad_hidden[n, d]   = sum_k d_logit[k] * rw[d, eid[k]] (direct store)
    // ====================================================================

    // Pre-check which experts have non-zero d_logit (common for sparse grads).
    bool dk_nonzero[MAX_K];
    bool any_dk = false;
    #pragma unroll
    for (int k = 0; k < MAX_K; k++) {
        dk_nonzero[k] = (k < K) && (d_logit[k] != 0.0f);
        any_dk = any_dk || dk_nonzero[k];
    }
    int64_t rw_base[MAX_K];
    if constexpr (kComputeGradW) {
        #pragma unroll
        for (int k = 0; k < MAX_K; k++) {
            rw_base[k] = static_cast<int64_t>(eid[k]) * D;
        }
    }
    __nv_bfloat16* gh_row_base = nullptr;
    if constexpr (kComputeGradHidden) {
        gh_row_base = grad_hidden + (int64_t)token_n * D;
    }
    const __nv_bfloat16* rw_col_base[MAX_K];
    if constexpr (kComputeGradHidden) {
        #pragma unroll
        for (int k = 0; k < MAX_K; k++) {
            rw_col_base[k] = router_weight + eid[k];
        }
    }
    const int D_vec = (D / VEC_BF16) * VEC_BF16;
    // If all expert-side logits are zero, fast-exit:
    // - grad_rw-only path: no work
    // - grad_hidden path: write zeros for this token and return
    if (!any_dk) {
        if constexpr (kComputeGradHidden) {
            for (int d0 = lane * VEC_BF16; d0 < D_vec; d0 += 32 * VEC_BF16) {
                uint4 z{};
                *reinterpret_cast<uint4*>(gh_row_base + d0) = z;
            }
            for (int d = D_vec + lane; d < D; d += 32) {
                gh_row_base[d] = __float2bfloat16(0.0f);
            }
        }
        return;
    }
    if (!kComputeGradHidden && !kComputeGradW) return;
    const __nv_bfloat16* h_row = nullptr;
    if constexpr (kComputeGradW) {
        h_row = hidden + (int64_t)token_n * D;
    }

    // --- Vectorized loop over D ---

    for (int d0 = lane * VEC_BF16; d0 < D_vec; d0 += 32 * VEC_BF16) {

        if constexpr (kComputeGradW) {
            // Re-load hidden from (likely L2-cached) global memory.
            float hv[VEC_BF16];
            {
                const uint4 ld = *reinterpret_cast<const uint4*>(h_row + d0);
                const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&ld);
                #pragma unroll
                for (int v = 0; v < VEC_BF16; v++) hv[v] = bf16_to_f32(bp[v]);
            }

            // Scatter into grad_rw via atomicAdd.
            #pragma unroll
            for (int k = 0; k < MAX_K; k++) {
                if (!dk_nonzero[k]) continue;
                float dk = d_logit[k];
                float* rw_row = grad_rw + rw_base[k] + d0;
                #pragma unroll
                for (int v = 0; v < VEC_BF16; v++) {
                    atomicAdd(&rw_row[v], dk * hv[v]);
                }
            }
        }

        // Gather for grad_hidden.
        if constexpr (kComputeGradHidden) {
            float gh[VEC_BF16];
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) gh[v] = 0.0f;

            #pragma unroll
            for (int k = 0; k < MAX_K; k++) {
                if (!dk_nonzero[k]) continue;
                float dk = d_logit[k];
                const __nv_bfloat16* rw_col = rw_col_base[k];
                #pragma unroll
                for (int v = 0; v < VEC_BF16; v++) {
                    gh[v] += dk * bf16_to_f32(rw_col[(int64_t)(d0 + v) * E]);
                }
            }

            // Store grad_hidden -- one warp exclusively owns this row.
            __nv_bfloat16* gh_row = gh_row_base + d0;
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) {
                gh_row[v] = f32_to_bf16(gh[v]);
            }
        }
    }

    // --- Scalar tail ---
    for (int d = D_vec + lane; d < D; d += 32) {
        if constexpr (kComputeGradW) {
            float h_val = bf16_to_f32(h_row[d]);
            #pragma unroll
            for (int k = 0; k < MAX_K; k++) {
                if (!dk_nonzero[k]) continue;
                atomicAdd(&grad_rw[rw_base[k] + d], d_logit[k] * h_val);
            }
        }

        if constexpr (kComputeGradHidden) {
            float gh_val = 0.0f;
            #pragma unroll
            for (int k = 0; k < MAX_K; k++) {
                if (!dk_nonzero[k]) continue;
                gh_val += d_logit[k] * bf16_to_f32(rw_col_base[k][(int64_t)d * E]);
            }
            gh_row_base[d] = f32_to_bf16(gh_val);
        }
    }

#endif // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Kernel 2: Transpose + cast  [E, D] FP32 row-major  -->  [D, E] BF16 row-major
//
// Tiled 32x32 transpose via shared memory for full coalescing on both
// the read and write sides. Launch uses blockDim = (32, block_rows) with
// block_rows in {4, 8}; each thread handles TILE / block_rows rows.
// ============================================================================

template <int BLOCK_ROWS>
__global__ void __launch_bounds__(32 * BLOCK_ROWS)
k_transpose_cast_fp32_to_bf16_full(
    const float*    __restrict__ src,   // [E, D] FP32 row-major
    __nv_bfloat16*  __restrict__ dst,   // [D, E] BF16 row-major
    int E, int D)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int TILE = TRANSPOSE_TILE;
    __shared__ float smem[TILE][TILE + 1]; // +1 avoids bank conflicts

    const int tile_e = (int)blockIdx.y * TILE;
    const int tile_d = (int)blockIdx.x * TILE;
    const int tx = threadIdx.x; // 0..31
    const int ty = threadIdx.y; // 0..BLOCK_ROWS-1

    #pragma unroll
    for (int i = 0; i < TILE; i += BLOCK_ROWS) {
        const int r = tile_e + ty + i;
        const int c = tile_d + tx;
        smem[ty + i][tx] = src[(int64_t)r * D + c];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < TILE; i += BLOCK_ROWS) {
        const int dst_d = tile_d + ty + i;
        const int dst_e = tile_e + tx;
        dst[(int64_t)dst_d * E + dst_e] = f32_to_bf16(smem[tx][ty + i]);
    }
#endif // __CUDA_ARCH__ >= 800
}

template <int BLOCK_ROWS>
__global__ void __launch_bounds__(32 * BLOCK_ROWS)
k_transpose_cast_fp32_to_bf16_masked(
    const float*    __restrict__ src,   // [E, D] FP32 row-major
    __nv_bfloat16*  __restrict__ dst,   // [D, E] BF16 row-major
    int E, int D)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int TILE = TRANSPOSE_TILE;
    __shared__ float smem[TILE][TILE + 1]; // +1 avoids bank conflicts

    const int tile_e = (int)blockIdx.y * TILE;
    const int tile_d = (int)blockIdx.x * TILE;
    const int tx = threadIdx.x; // 0..31
    const int ty = threadIdx.y; // 0..BLOCK_ROWS-1

    const bool full_tile = (tile_e + TILE <= E) && (tile_d + TILE <= D);
    if (full_tile) {
        #pragma unroll
        for (int i = 0; i < TILE; i += BLOCK_ROWS) {
            const int r = tile_e + ty + i;
            const int c = tile_d + tx;
            smem[ty + i][tx] = src[(int64_t)r * D + c];
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i += BLOCK_ROWS) {
            const int dst_d = tile_d + ty + i;
            const int dst_e = tile_e + tx;
            dst[(int64_t)dst_d * E + dst_e] = f32_to_bf16(smem[tx][ty + i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < TILE; i += BLOCK_ROWS) {
            const int r = tile_e + ty + i;
            const int c = tile_d + tx;
            float val = 0.0f;
            if (r < E && c < D) {
                val = src[(int64_t)r * D + c];
            }
            smem[ty + i][tx] = val;
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i += BLOCK_ROWS) {
            const int dst_d = tile_d + ty + i;
            const int dst_e = tile_e + tx;
            if (dst_d < D && dst_e < E) {
                dst[(int64_t)dst_d * E + dst_e] = f32_to_bf16(smem[tx][ty + i]);
            }
        }
    }
#endif // __CUDA_ARCH__ >= 800
}

template <int BLOCK_ROWS>
__host__ __forceinline__ void launch_transpose_cast_fp32_to_bf16(
    const float* src,
    __nv_bfloat16* dst,
    int E,
    int D,
    cudaStream_t stream)
{
    constexpr int TILE = TRANSPOSE_TILE;
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((unsigned)ceil_div(D, TILE), (unsigned)ceil_div(E, TILE));
    if ((D % TILE) == 0 && (E % TILE) == 0) {
        k_transpose_cast_fp32_to_bf16_full<BLOCK_ROWS><<<grid, block, 0, stream>>>(src, dst, E, D);
    } else {
        k_transpose_cast_fp32_to_bf16_masked<BLOCK_ROWS><<<grid, block, 0, stream>>>(src, dst, E, D);
    }
}

__host__ __forceinline__ void launch_router_bwd_transpose_cast(
    const float* src,
    __nv_bfloat16* dst,
    int E,
    int D,
    cudaStream_t stream)
{
    const int block_rows = select_router_bwd_transpose_block_rows(E, D);
    if (block_rows == TRANSPOSE_BLOCK_ROWS_SMALL) {
        launch_transpose_cast_fp32_to_bf16<TRANSPOSE_BLOCK_ROWS_SMALL>(src, dst, E, D, stream);
    } else {
        launch_transpose_cast_fp32_to_bf16<TRANSPOSE_BLOCK_ROWS_DEFAULT>(src, dst, E, D, stream);
    }
}

} // namespace router_bwd
} // namespace nmoe

// ============================================================================
// C API  --  called from Python via ctypes or pybind11
// ============================================================================

static cudaError_t require_sm80_or_newer(const char* fn_name) {
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    thread_local int cached_device = -1;
    thread_local cudaError_t cached_result = cudaSuccess;
    thread_local bool cached = false;
    if (cached && device == cached_device) {
        return cached_result;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return err;
    cached = true;
    cached_device = device;
    if (prop.major < 8) {
        fprintf(stderr, "%s requires sm_80+; got sm_%d%d\n", fn_name, prop.major, prop.minor);
        cached_result = cudaErrorNotSupported;
        return cached_result;
    }
    cached_result = cudaSuccess;
    return cached_result;
}

extern "C" {

/// Fused MoE router backward pass (sparse formulation).
///
/// Computes grad_router_weight (and optionally grad_hidden) from the incoming
/// gate gradients without materializing any dense [N, E] intermediates.
///
/// Caller responsibilities:
///   - grad_rw_fp32 must be zeroed before this call (it accumulates via atomicAdd).
///   - gates/grad_gates can be FP32 (this entrypoint) or BF16
///     (use fused_router_backward_bf16).
///   - pre_probs must contain pre-normalized selected probabilities from forward.
///   - hidden and router_weight must be BF16, contiguous, row-major.
///
/// After this call, grad_rw_fp32 is [E, D] FP32 row-major.  Call
/// fused_router_bwd_transpose() to convert to [D, E] BF16 matching the
/// router_weight storage layout.
cudaError_t fused_router_backward(
    const void*  hidden,           // [N, D] BF16
    const void*  router_weight,    // [D, E] BF16
    const void*  expert_ids,       // [N, K] int32
    const void*  pre_probs_f32,    // [N, K] FP32  (pre-normalized probs)
    const void*  gates_f32,        // [N, K] FP32  (normalized gates)
    const void*  grad_gates_f32,   // [N, K] FP32  (incoming gradient)
    void*        grad_rw_fp32,     // [E, D] FP32  (zeroed, output)
    void*        grad_hidden,      // [N, D] BF16  (output, or NULL)
    float        route_scale,
    int N, int E, int K, int D,
    cudaStream_t stream)
{
    using namespace nmoe::router_bwd;
    cudaError_t arch_err = require_sm80_or_newer("fused_router_backward");
    if (arch_err != cudaSuccess) return arch_err;

    if (N <= 0 || K <= 0 || D <= 0) return cudaSuccess;
    if (K > MAX_K) {
        fprintf(stderr, "fused_router_backward: K=%d > MAX_K=%d\n", K, MAX_K);
        return cudaErrorInvalidValue;
    }

    const int warps_per_block = select_router_bwd_warps_per_block(N, D, K);
    dim3 block(32, warps_per_block);
    dim3 grid((unsigned)ceil_div(N, warps_per_block));

    auto h   = reinterpret_cast<const __nv_bfloat16*>(hidden);
    auto rw  = reinterpret_cast<const __nv_bfloat16*>(router_weight);
    auto eid = reinterpret_cast<const int32_t*>(expert_ids);
    auto pp  = reinterpret_cast<const float*>(pre_probs_f32);
    auto gf  = reinterpret_cast<const float*>(gates_f32);
    auto dgf = reinterpret_cast<const float*>(grad_gates_f32);
    auto grw = reinterpret_cast<float*>(grad_rw_fp32);

    if (grad_hidden != nullptr) {
        auto gh = reinterpret_cast<__nv_bfloat16*>(grad_hidden);
        k_fused_router_bwd<true, true, float, float><<<grid, block, 0, stream>>>(
            h, rw, eid, pp, gf, dgf, gh, grw, route_scale, N, E, K, D);
    } else {
        k_fused_router_bwd<false, true, float, float><<<grid, block, 0, stream>>>(
            h, rw, eid, pp, gf, dgf, nullptr, grw, route_scale, N, E, K, D);
    }

    return cudaGetLastError();
}

/// BF16 gate/grad variant of fused router backward.
///
/// This avoids Python-side gates/grad_gates casts for the default BF16 router
/// path and performs BF16->FP32 conversion inside the fused CUDA kernel.
cudaError_t fused_router_backward_bf16(
    const void*  hidden,              // [N, D] BF16
    const void*  router_weight,       // [D, E] BF16
    const void*  expert_ids,          // [N, K] int32
    const void*  pre_probs_bf16,      // [N, K] BF16
    const void*  gates_bf16,          // [N, K] BF16
    const void*  grad_gates_bf16,     // [N, K] BF16
    void*        grad_rw_fp32,        // [E, D] FP32 (zeroed, output)
    void*        grad_hidden,         // [N, D] BF16 (output, or NULL)
    float        route_scale,
    int N, int E, int K, int D,
    cudaStream_t stream)
{
    using namespace nmoe::router_bwd;
    cudaError_t arch_err = require_sm80_or_newer("fused_router_backward_bf16");
    if (arch_err != cudaSuccess) return arch_err;

    if (N <= 0 || K <= 0 || D <= 0) return cudaSuccess;
    if (K > MAX_K) {
        fprintf(stderr, "fused_router_backward_bf16: K=%d > MAX_K=%d\n", K, MAX_K);
        return cudaErrorInvalidValue;
    }

    const int warps_per_block = select_router_bwd_warps_per_block(N, D, K);
    dim3 block(32, warps_per_block);
    dim3 grid((unsigned)ceil_div(N, warps_per_block));

    auto h   = reinterpret_cast<const __nv_bfloat16*>(hidden);
    auto rw  = reinterpret_cast<const __nv_bfloat16*>(router_weight);
    auto eid = reinterpret_cast<const int32_t*>(expert_ids);
    auto pp  = reinterpret_cast<const __nv_bfloat16*>(pre_probs_bf16);
    auto gf  = reinterpret_cast<const __nv_bfloat16*>(gates_bf16);
    auto dgf = reinterpret_cast<const __nv_bfloat16*>(grad_gates_bf16);
    auto grw = reinterpret_cast<float*>(grad_rw_fp32);

    if (grad_hidden != nullptr) {
        auto gh = reinterpret_cast<__nv_bfloat16*>(grad_hidden);
        k_fused_router_bwd<true, true, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
            h, rw, eid, pp, gf, dgf, gh, grw, route_scale, N, E, K, D);
    } else {
        k_fused_router_bwd<false, true, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
            h, rw, eid, pp, gf, dgf, nullptr, grw, route_scale, N, E, K, D);
    }

    return cudaGetLastError();
}

/// BF16 fused epilogue variant:
///   1) zero grad_rw_fp32
///   2) launch fused_router_backward_bf16 core kernel
///   3) transpose+cast [E,D] FP32 -> [D,E] BF16
///
/// This removes one host-side call boundary in the Python hot path.
cudaError_t fused_router_backward_bf16_fused(
    const void*  hidden,              // [N, D] BF16
    const void*  router_weight,       // [D, E] BF16
    const void*  expert_ids,          // [N, K] int32
    const void*  pre_probs_bf16,      // [N, K] BF16
    const void*  gates_bf16,          // [N, K] BF16
    const void*  grad_gates_bf16,     // [N, K] BF16
    void*        grad_rw_fp32,        // [E, D] FP32 scratch
    void*        grad_rw_bf16,        // [D, E] BF16 output
    void*        grad_hidden,         // [N, D] BF16 (output, or NULL)
    float        route_scale,
    int N, int E, int K, int D,
    cudaStream_t stream)
{
    using namespace nmoe::router_bwd;
    cudaError_t arch_err = require_sm80_or_newer("fused_router_backward_bf16_fused");
    if (arch_err != cudaSuccess) return arch_err;

    if (N <= 0 || K <= 0 || D <= 0) {
        if (E > 0 && D > 0) {
            cudaError_t zerr = cudaMemsetAsync(
                grad_rw_bf16,
                0,
                static_cast<size_t>(E) * static_cast<size_t>(D) * sizeof(__nv_bfloat16),
                stream);
            if (zerr != cudaSuccess) return zerr;
            zerr = cudaMemsetAsync(
                grad_rw_fp32,
                0,
                static_cast<size_t>(E) * static_cast<size_t>(D) * sizeof(float),
                stream);
            if (zerr != cudaSuccess) return zerr;
        }
        return cudaSuccess;
    }
    if (K > MAX_K) {
        fprintf(stderr, "fused_router_backward_bf16_fused: K=%d > MAX_K=%d\n", K, MAX_K);
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaMemsetAsync(
        grad_rw_fp32,
        0,
        static_cast<size_t>(E) * static_cast<size_t>(D) * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;

    const int warps_per_block = select_router_bwd_warps_per_block(N, D, K);
    dim3 block(32, warps_per_block);
    dim3 grid((unsigned)ceil_div(N, warps_per_block));

    auto h   = reinterpret_cast<const __nv_bfloat16*>(hidden);
    auto rw  = reinterpret_cast<const __nv_bfloat16*>(router_weight);
    auto eid = reinterpret_cast<const int32_t*>(expert_ids);
    auto pp  = reinterpret_cast<const __nv_bfloat16*>(pre_probs_bf16);
    auto gf  = reinterpret_cast<const __nv_bfloat16*>(gates_bf16);
    auto dgf = reinterpret_cast<const __nv_bfloat16*>(grad_gates_bf16);
    auto grw = reinterpret_cast<float*>(grad_rw_fp32);

    if (grad_hidden != nullptr) {
        auto gh = reinterpret_cast<__nv_bfloat16*>(grad_hidden);
        k_fused_router_bwd<true, true, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
            h, rw, eid, pp, gf, dgf, gh, grw, route_scale, N, E, K, D);
    } else {
        k_fused_router_bwd<false, true, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
            h, rw, eid, pp, gf, dgf, nullptr, grw, route_scale, N, E, K, D);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    launch_router_bwd_transpose_cast(
        reinterpret_cast<const float*>(grad_rw_fp32),
        reinterpret_cast<__nv_bfloat16*>(grad_rw_bf16),
        E, D,
        stream);
    return cudaGetLastError();
}

/// BF16 hidden-only variant:
///   - computes grad_hidden only (no grad_router_weight accumulation or transpose).
///   - intended for frozen-router runs where wgrad is not required.
cudaError_t fused_router_backward_bf16_hidden_only(
    const void*  hidden,              // [N, D] BF16
    const void*  router_weight,       // [D, E] BF16
    const void*  expert_ids,          // [N, K] int32
    const void*  pre_probs_bf16,      // [N, K] BF16
    const void*  gates_bf16,          // [N, K] BF16
    const void*  grad_gates_bf16,     // [N, K] BF16
    void*        grad_hidden,         // [N, D] BF16 (output)
    float        route_scale,
    int N, int E, int K, int D,
    cudaStream_t stream)
{
    using namespace nmoe::router_bwd;
    cudaError_t arch_err = require_sm80_or_newer("fused_router_backward_bf16_hidden_only");
    if (arch_err != cudaSuccess) return arch_err;

    if (N <= 0 || K <= 0 || D <= 0) return cudaSuccess;
    if (K > MAX_K) {
        fprintf(stderr, "fused_router_backward_bf16_hidden_only: K=%d > MAX_K=%d\n", K, MAX_K);
        return cudaErrorInvalidValue;
    }
    if (grad_hidden == nullptr) {
        fprintf(stderr, "fused_router_backward_bf16_hidden_only: grad_hidden is null\n");
        return cudaErrorInvalidValue;
    }

    const int warps_per_block = select_router_bwd_warps_per_block(N, D, K);
    dim3 block(32, warps_per_block);
    dim3 grid((unsigned)ceil_div(N, warps_per_block));

    auto h   = reinterpret_cast<const __nv_bfloat16*>(hidden);
    auto rw  = reinterpret_cast<const __nv_bfloat16*>(router_weight);
    auto eid = reinterpret_cast<const int32_t*>(expert_ids);
    auto pp  = reinterpret_cast<const __nv_bfloat16*>(pre_probs_bf16);
    auto gf  = reinterpret_cast<const __nv_bfloat16*>(gates_bf16);
    auto dgf = reinterpret_cast<const __nv_bfloat16*>(grad_gates_bf16);
    auto gh  = reinterpret_cast<__nv_bfloat16*>(grad_hidden);

    k_fused_router_bwd<true, false, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
        h, rw, eid, pp, gf, dgf, gh, nullptr, route_scale, N, E, K, D);
    return cudaGetLastError();
}

/// Transpose + cast: [E, D] FP32 row-major  ->  [D, E] BF16 row-major.
///
/// Converts the FP32 accumulation buffer into the same layout and dtype as
/// router_weight so the optimizer can apply the gradient directly.
cudaError_t fused_router_bwd_transpose(
    const void*  grad_rw_fp32,     // [E, D] FP32 row-major
    void*        grad_rw_bf16,     // [D, E] BF16 row-major
    int E, int D,
    cudaStream_t stream)
{
    using namespace nmoe::router_bwd;
    cudaError_t arch_err = require_sm80_or_newer("fused_router_bwd_transpose");
    if (arch_err != cudaSuccess) return arch_err;

    if (E <= 0 || D <= 0) return cudaSuccess;

    launch_router_bwd_transpose_cast(
        reinterpret_cast<const float*>(grad_rw_fp32),
        reinterpret_cast<__nv_bfloat16*>(grad_rw_bf16),
        E, D,
        stream);

    return cudaGetLastError();
}

} // extern "C"
