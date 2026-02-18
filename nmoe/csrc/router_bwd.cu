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
//     1. Loads expert_ids[n, 0..K-1], gates[n, 0..K-1], grad_gates[n, 0..K-1]
//     2. Streams over D dimension once, computing:
//        - K dot products (logits at selected experts) -- warp-reduced
//        - After reduction: sigmoid, sigmoid derivative, normalization VJP
//        - K scattered atomicAdds into grad_router_weight  (FP32)
//        - Optionally: K gathers + sum for grad_hidden      (BF16)
//     3. The streaming single-pass over D avoids re-reading hidden twice.
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

// Full-warp sum reduction.  All 32 lanes receive the result.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Constants
// ============================================================================

// Maximum TopK we support via static register arrays.
// K is typically 1-8 for MoE; 16 covers all practical configurations.
constexpr int MAX_K = 16;

// Warps (= tokens) per thread-block.  4 warps = 128 threads -- good
// occupancy without excessive register pressure.
constexpr int WARPS_PER_BLOCK = 4;

// BF16 elements loaded per vectorized 128-bit transaction.
constexpr int VEC_BF16 = 8;  // sizeof(uint4) / sizeof(__nv_bfloat16)

// FP32 elements loaded per vectorized 128-bit transaction.
constexpr int VEC_F32 = 4;   // sizeof(uint4) / sizeof(float)

// ============================================================================
// Kernel 1: Fused Router Backward  (main computational kernel)
// ============================================================================
//
// Memory layout expectations (all row-major, contiguous):
//   hidden         [N, D]  BF16
//   router_weight  [D, E]  BF16   (element (d,e) at offset d*E + e)
//   expert_ids     [N, K]  int32
//   gates_f32      [N, K]  FP32   (normalized gates -- Python pre-casts)
//   grad_gates_f32 [N, K]  FP32   (incoming gradient)
//   grad_rw        [E, D]  FP32   (output, zeroed by caller)
//   grad_hidden    [N, D]  BF16   (output, optional)
//
// Algorithm per token n:
//   For k = 0..K-1:
//     e_k = expert_ids[n, k]
//     logit_k = sum_d hidden[n,d] * router_weight[d, e_k]       // dot product
//   prob_k = sigmoid(logit_k * route_scale)
//   sig_deriv_k = prob_k * (1 - prob_k)
//   gate_sum = sum_k prob_k
//   dot_gs = sum_k grad_gates[n,k] * gates[n,k]                // VJP constant
//   grad_unnorm_k = (grad_gates[n,k] - dot_gs) / gate_sum
//   d_logit_k = grad_unnorm_k * sig_deriv_k * route_scale
//
//   grad_rw[e_k, d] += d_logit_k * hidden[n, d]                // atomic
//   grad_hidden[n, d] = sum_k d_logit_k * router_weight[d, e_k] // optional
//
// Implementation:
//   Two-pass over D.
//     Pass 1: accumulate K partial dot-products (logits) across D.
//     -- warp-reduce to get final logit values, compute d_logit[k] --
//     Pass 2: scatter d_logit_k * h into grad_rw; gather rw into grad_hidden.
//
//   We accept the cost of reading hidden twice (from L2 cache the second time)
//   to avoid an O(D) shared-memory buffer.  For D=7168 BF16 that would be
//   14 KB per warp = 56 KB per block, too much for good occupancy.

template <bool kComputeGradHidden>
__global__ void __launch_bounds__(32 * WARPS_PER_BLOCK, 2)
k_fused_router_bwd(
    // Inputs
    const __nv_bfloat16* __restrict__ hidden,          // [N, D]
    const __nv_bfloat16* __restrict__ router_weight,   // [D, E]
    const int32_t*       __restrict__ expert_ids,      // [N, K]
    const float*         __restrict__ gates_f32,       // [N, K] FP32
    const float*         __restrict__ grad_gates_f32,  // [N, K] FP32
    // Outputs
    __nv_bfloat16*       __restrict__ grad_hidden,     // [N, D]  (NULL when !kComputeGradHidden)
    float*               __restrict__ grad_rw,         // [E, D] FP32
    // Scalars
    float route_scale,
    int N, int E, int K, int D)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    const int lane = threadIdx.x;                       // 0..31
    const int warp_id = threadIdx.y;                    // 0..WARPS_PER_BLOCK-1
    const int token_n = (int)blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (token_n >= N) return;

    // ====================================================================
    // Phase 0: Load per-token metadata (K values).  Lane 0 reads, then
    //          broadcast via __shfl_sync to all 32 lanes.
    // ====================================================================

    int   eid[MAX_K];
    float gate_s[MAX_K];   // normalized gate (s_k) from forward
    float grad_s[MAX_K];   // incoming grad d(loss)/d(s_k)

    for (int k = 0; k < K && k < MAX_K; k++) {
        int   e_tmp = 0;
        float g_tmp = 0.0f;
        float dg_tmp = 0.0f;
        if (lane == 0) {
            const int idx = token_n * K + k;
            e_tmp  = expert_ids[idx];
            g_tmp  = gates_f32[idx];
            dg_tmp = grad_gates_f32[idx];
        }
        eid[k]    = __shfl_sync(0xFFFFFFFF, e_tmp,  0);
        gate_s[k] = __shfl_sync(0xFFFFFFFF, g_tmp,  0);
        grad_s[k] = __shfl_sync(0xFFFFFFFF, dg_tmp, 0);
    }

    // ====================================================================
    // Phase 1: Compute K dot-products (logits at selected experts).
    //
    //   logit_k = sum_d hidden[n,d] * router_weight[d, eid[k]]
    //
    //   router_weight is [D, E] row-major, so rw(d, e) = rw[d*E + e].
    //   Accessing rw[:, e] for a fixed e is a stride-E gather -- not
    //   coalesced, but the working set is K*D*2 bytes ~ 112 KB for
    //   K=8, D=7168 which fits comfortably in L2.
    //
    //   hidden is contiguous in D -- coalesced BF16 vector loads.
    // ====================================================================

    const __nv_bfloat16* h_row = hidden + (int64_t)token_n * D;

    float logit_partial[MAX_K];
    #pragma unroll
    for (int k = 0; k < MAX_K; k++) logit_partial[k] = 0.0f;

    // Vectorized loop body: 32 threads * VEC_BF16(8) = 256 BF16 elems per iter.
    const int D_full = (D / (32 * VEC_BF16)) * (32 * VEC_BF16);

    for (int base = 0; base < D_full; base += 32 * VEC_BF16) {
        const int d0 = base + lane * VEC_BF16;

        // Coalesced 128-bit load of 8 consecutive BF16 from hidden.
        float hv[VEC_BF16];
        {
            const uint4 ld = *reinterpret_cast<const uint4*>(h_row + d0);
            const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&ld);
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) hv[v] = bf16_to_f32(bp[v]);
        }

        // Accumulate into each expert's logit partial sum.
        for (int k = 0; k < K && k < MAX_K; k++) {
            // Stride-E scalar loads from router_weight column eid[k].
            const __nv_bfloat16* rw_col = router_weight + eid[k];
            float acc = 0.0f;
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) {
                acc += hv[v] * bf16_to_f32(rw_col[(int64_t)(d0 + v) * E]);
            }
            logit_partial[k] += acc;
        }
    }

    // Scalar tail for remaining D elements.
    for (int d = D_full + lane; d < D; d += 32) {
        float h_val = bf16_to_f32(h_row[d]);
        for (int k = 0; k < K && k < MAX_K; k++) {
            logit_partial[k] += h_val * bf16_to_f32(router_weight[(int64_t)d * E + eid[k]]);
        }
    }

    // Warp-reduce each logit, then sigmoid + sigmoid derivative.
    float prob_k[MAX_K];
    float sig_d_k[MAX_K];
    float gate_sum = 0.0f;

    for (int k = 0; k < K && k < MAX_K; k++) {
        float logit_k = warp_reduce_sum(logit_partial[k]);
        float s = logit_k * route_scale;
        float p = 1.0f / (1.0f + expf(-s));
        prob_k[k]  = p;
        sig_d_k[k] = p * (1.0f - p);
        gate_sum  += p;
    }

    // ====================================================================
    // Phase 2: Normalization VJP + chain rule to get d_logit[k].
    //
    //   Forward: s_k = prob_k / S,  where S = sum_j prob_j (over K selected).
    //   VJP:     grad_unnorm_k = (grad_s_k - dot(grad_s, s)) / S
    //   Chain:   d_logit_k = grad_unnorm_k * sigmoid'(logit_k) * route_scale
    // ====================================================================

    gate_sum = fmaxf(gate_sum, 1e-12f);

    float dot_gs = 0.0f;
    for (int k = 0; k < K && k < MAX_K; k++) {
        dot_gs += grad_s[k] * gate_s[k];
    }

    float d_logit[MAX_K];
    for (int k = 0; k < K && k < MAX_K; k++) {
        float grad_unnorm = (grad_s[k] - dot_gs) / gate_sum;
        d_logit[k] = grad_unnorm * sig_d_k[k] * route_scale;
    }

    // ====================================================================
    // Phase 3: Scatter into grad_router_weight and (optionally) accumulate
    //          grad_hidden.  Second pass over D -- hidden will typically
    //          be served from L2 cache.
    //
    //   grad_rw[eid[k], d] += d_logit[k] * hidden[n, d]       (atomicAdd)
    //   grad_hidden[n, d]   = sum_k d_logit[k] * rw[d, eid[k]] (direct store)
    // ====================================================================

    // Pre-check which experts have non-zero d_logit (common for sparse grads).
    bool dk_nonzero[MAX_K];
    for (int k = 0; k < K && k < MAX_K; k++) {
        dk_nonzero[k] = (d_logit[k] != 0.0f);
    }

    // --- Vectorized loop over D (same chunking as Phase 1) ---

    for (int base = 0; base < D_full; base += 32 * VEC_BF16) {
        const int d0 = base + lane * VEC_BF16;

        // Re-load hidden from (likely L2-cached) global memory.
        float hv[VEC_BF16];
        {
            const uint4 ld = *reinterpret_cast<const uint4*>(h_row + d0);
            const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&ld);
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) hv[v] = bf16_to_f32(bp[v]);
        }

        // Scatter into grad_rw via atomicAdd.
        for (int k = 0; k < K && k < MAX_K; k++) {
            if (!dk_nonzero[k]) continue;
            float dk = d_logit[k];
            float* rw_row = grad_rw + (int64_t)eid[k] * D + d0;
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) {
                atomicAdd(&rw_row[v], dk * hv[v]);
            }
        }

        // Gather for grad_hidden.
        if (kComputeGradHidden) {
            float gh[VEC_BF16];
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) gh[v] = 0.0f;

            for (int k = 0; k < K && k < MAX_K; k++) {
                if (!dk_nonzero[k]) continue;
                float dk = d_logit[k];
                const __nv_bfloat16* rw_col = router_weight + eid[k];
                #pragma unroll
                for (int v = 0; v < VEC_BF16; v++) {
                    gh[v] += dk * bf16_to_f32(rw_col[(int64_t)(d0 + v) * E]);
                }
            }

            // Store grad_hidden -- one warp exclusively owns this row.
            __nv_bfloat16* gh_row = grad_hidden + (int64_t)token_n * D + d0;
            #pragma unroll
            for (int v = 0; v < VEC_BF16; v++) {
                gh_row[v] = f32_to_bf16(gh[v]);
            }
        }
    }

    // --- Scalar tail ---
    for (int d = D_full + lane; d < D; d += 32) {
        float h_val = bf16_to_f32(h_row[d]);

        for (int k = 0; k < K && k < MAX_K; k++) {
            if (!dk_nonzero[k]) continue;
            atomicAdd(&grad_rw[(int64_t)eid[k] * D + d], d_logit[k] * h_val);
        }

        if (kComputeGradHidden) {
            float gh_val = 0.0f;
            for (int k = 0; k < K && k < MAX_K; k++) {
                if (!dk_nonzero[k]) continue;
                gh_val += d_logit[k] * bf16_to_f32(router_weight[(int64_t)d * E + eid[k]]);
            }
            grad_hidden[(int64_t)token_n * D + d] = f32_to_bf16(gh_val);
        }
    }

#endif // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Kernel 2: Transpose + cast  [E, D] FP32 row-major  -->  [D, E] BF16 row-major
//
// Tiled 32x32 transpose via shared memory for full coalescing on both
// the read and write sides.  blockDim = (32, 8), each thread handles 4 rows.
// ============================================================================

__global__ void __launch_bounds__(256)
k_transpose_cast_fp32_to_bf16(
    const float*    __restrict__ src,   // [E, D] FP32 row-major
    __nv_bfloat16*  __restrict__ dst,   // [D, E] BF16 row-major
    int E, int D)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int TILE = 32;
    __shared__ float smem[TILE][TILE + 1]; // +1 avoids bank conflicts

    const int tile_e = (int)blockIdx.y * TILE;
    const int tile_d = (int)blockIdx.x * TILE;
    const int tx = threadIdx.x; // 0..31
    const int ty = threadIdx.y; // 0..7

    // Load src[tile_e..+32, tile_d..+32] -> smem (coalesced in D).
    #pragma unroll
    for (int i = 0; i < TILE; i += 8) {
        int r = tile_e + ty + i;
        int c = tile_d + tx;
        float val = 0.0f;
        if (r < E && c < D) {
            val = src[(int64_t)r * D + c];
        }
        smem[ty + i][tx] = val;
    }

    __syncthreads();

    // Write transposed: dst[tile_d..+32, tile_e..+32] (coalesced in E).
    #pragma unroll
    for (int i = 0; i < TILE; i += 8) {
        int dst_d = tile_d + ty + i;
        int dst_e = tile_e + tx;
        if (dst_d < D && dst_e < E) {
            dst[(int64_t)dst_d * E + dst_e] = f32_to_bf16(smem[tx][ty + i]);
        }
    }
#endif // __CUDA_ARCH__ >= 800
}

} // namespace router_bwd
} // namespace nmoe

// ============================================================================
// C API  --  called from Python via ctypes or pybind11
// ============================================================================

extern "C" {

/// Fused MoE router backward pass (sparse formulation).
///
/// Computes grad_router_weight (and optionally grad_hidden) from the incoming
/// gate gradients without materializing any dense [N, E] intermediates.
///
/// Caller responsibilities:
///   - grad_rw_fp32 must be zeroed before this call (it accumulates via atomicAdd).
///   - gates_f32 and grad_gates_f32 must be FP32 (Python side casts if needed).
///   - hidden and router_weight must be BF16, contiguous, row-major.
///
/// After this call, grad_rw_fp32 is [E, D] FP32 row-major.  Call
/// fused_router_bwd_transpose() to convert to [D, E] BF16 matching the
/// router_weight storage layout.
cudaError_t fused_router_backward(
    const void*  hidden,           // [N, D] BF16
    const void*  router_weight,    // [D, E] BF16
    const void*  expert_ids,       // [N, K] int32
    const void*  gates_f32,        // [N, K] FP32  (normalized gates)
    const void*  grad_gates_f32,   // [N, K] FP32  (incoming gradient)
    void*        grad_rw_fp32,     // [E, D] FP32  (zeroed, output)
    void*        grad_hidden,      // [N, D] BF16  (output, or NULL)
    float        route_scale,
    int N, int E, int K, int D,
    cudaStream_t stream)
{
    using namespace nmoe::router_bwd;

    if (N <= 0 || K <= 0 || D <= 0) return cudaSuccess;
    if (K > MAX_K) {
        fprintf(stderr, "fused_router_backward: K=%d > MAX_K=%d\n", K, MAX_K);
        return cudaErrorInvalidValue;
    }

    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((unsigned)ceil_div(N, WARPS_PER_BLOCK));

    auto h   = reinterpret_cast<const __nv_bfloat16*>(hidden);
    auto rw  = reinterpret_cast<const __nv_bfloat16*>(router_weight);
    auto eid = reinterpret_cast<const int32_t*>(expert_ids);
    auto gf  = reinterpret_cast<const float*>(gates_f32);
    auto dgf = reinterpret_cast<const float*>(grad_gates_f32);
    auto grw = reinterpret_cast<float*>(grad_rw_fp32);

    if (grad_hidden != nullptr) {
        auto gh = reinterpret_cast<__nv_bfloat16*>(grad_hidden);
        k_fused_router_bwd<true><<<grid, block, 0, stream>>>(
            h, rw, eid, gf, dgf, gh, grw, route_scale, N, E, K, D);
    } else {
        k_fused_router_bwd<false><<<grid, block, 0, stream>>>(
            h, rw, eid, gf, dgf, nullptr, grw, route_scale, N, E, K, D);
    }

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

    if (E <= 0 || D <= 0) return cudaSuccess;

    constexpr int TILE = 32;
    dim3 block(32, 8);
    dim3 grid((unsigned)ceil_div(D, TILE), (unsigned)ceil_div(E, TILE));

    k_transpose_cast_fp32_to_bf16<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(grad_rw_fp32),
        reinterpret_cast<__nv_bfloat16*>(grad_rw_bf16),
        E, D);

    return cudaGetLastError();
}

} // extern "C"
