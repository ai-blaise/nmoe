// nmoe/eco_adam.cu
//
// Fused ECO AdamW update for NVFP4 primary weights with FP8 optimizer states.
//
// Replaces Python fused_update() in eco.py. Zero FP32 global memory materialization.
//
// Memory per element:
//   Read:  NVFP4 (0.5B) + E4M3 group scale + FP8 m (1B) + FP8 v (1B) + FP32 grad (4B) ≈ 7B
//   Write: NVFP4 (0.5B) + E4M3 group scale + FP8 m (1B) + FP8 v (1B) ≈ 2.5B
//   vs Python: ~68B read + ~68B write (full FP32 materialization)
//
// Layout overview:
//   NVFP4 weights: HF layout [E, out_dim, in_dim] (compressed_tensors format)
//   FP8 m/v:       nmoe layout [E, in_dim, out_dim] with per-row FP32 scales
//   Gradient:      nmoe layout [E, in_dim, out_dim], FP32
//
// The kernel tiles over the weight's HF layout (out_dim × in_dim) and maps
// each element to the transposed position in the gradient/m/v arrays.
//
// Thread mapping:
//   256 threads: threadIdx.x = out_local [0,31], threadIdx.y = in_iter [0,7].
//   Each thread processes 4 elements across in_dim: in_local = t*THREADS_Y + in_iter.
//   For packed NVFP4 write-back, nibbles are collected in shared memory and
//   written as full bytes to avoid read-modify-write races.
//
// Target: NVIDIA Blackwell B200 (sm_100a).

#include "ptx.cu"
#include "swizzle.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nmoe {
namespace eco_adam {

constexpr float NVFP4_MAX = 6.0f;

// Tile: 32 out_dim rows × 32 in_dim columns.
// 256 threads: threadIdx.x = out_local [0,31], threadIdx.y = in_iter [0,7].
// Each thread processes 4 elements across in_dim (32/8 = 4).
constexpr int TILE_OUT = 32;
constexpr int TILE_IN = 32;
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int ELEMS_PER_THREAD = TILE_IN / THREADS_Y;  // 4

__host__ __device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Choose a bounded grid for grid-stride kernels to reduce launch overhead.
inline cudaError_t choose_stride_grid_x(int64_t total, int threads, uint32_t* grid_x_out) {
    if (grid_x_out == nullptr || threads <= 0) return cudaErrorInvalidValue;
    if (total <= 0) {
        *grid_x_out = 0;
        return cudaSuccess;
    }

    int dev = -1;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;

    static thread_local int cached_dev = -2;
    static thread_local int cached_sm_count = 0;
    if (dev != cached_dev) {
        int sm_count = 0;
        err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess) return err;
        cached_dev = dev;
        cached_sm_count = sm_count;
    }

    constexpr int64_t kBlocksPerSm = 8;
    const int64_t blocks64 = (total + threads - 1) / threads;
    const int64_t cap64 = static_cast<int64_t>(cached_sm_count) * kBlocksPerSm;
    int64_t grid64 = blocks64;
    if (cap64 > 0 && grid64 > cap64) grid64 = cap64;
    if (grid64 > 2147483647LL) grid64 = 2147483647LL;
    if (grid64 < 1) grid64 = 1;
    *grid_x_out = static_cast<uint32_t>(grid64);
    return cudaSuccess;
}

template <typename GradT>
__device__ __forceinline__ float load_grad_elem(const GradT* grad, int64_t idx);

template <>
__device__ __forceinline__ float load_grad_elem<float>(const float* grad, int64_t idx) {
    return grad[idx];
}

template <>
__device__ __forceinline__ float load_grad_elem<__nv_bfloat16>(const __nv_bfloat16* grad, int64_t idx) {
    return __bfloat162float(grad[idx]);
}

// ============================================================================
// Factored-v reduction kernels (Kernel A).
//
// Compute Adafactor-style factored second moment update BEFORE the main
// ECO Adam kernel. The main kernel then reads v_row/v_col/v_rms to
// reconstruct v on-the-fly instead of loading full FP8 v_data.
//
// Two sub-kernels:
//   k_factored_v_row: Launch [E * in_dim] threads. Each sums grad^2 across
//     out_dim, updates v_row[e, i] EMA, and atomicAdds into v_rms[e].
//   k_factored_v_col: Launch [E * out_dim] threads. Each sums grad^2 across
//     in_dim, updates v_col[e, j] EMA.
// ============================================================================

// One BLOCK per (expert, row) pair. Threads collaboratively reduce across out_dim.
template <typename GradT>
__global__ void k_factored_v_row(
    const GradT* __restrict__ grad,    // [E, in_dim, out_dim] nmoe layout
    float* __restrict__ v_row,         // [E, in_dim]
    float* __restrict__ v_rms,         // [E] (output: mean of v_row per expert)
    int E,
    int in_dim,
    int out_dim,
    float beta2)
{
    // blockIdx.x = (expert * in_dim + row) index
    const int idx = blockIdx.x;
    const int total = E * in_dim;
    if (idx >= total) return;

    const int e = idx / in_dim;
    const int i = idx % in_dim;

    const float one_minus_b2 = 1.0f - beta2;

    // Each thread sums a strided subset of out_dim
    const int64_t row_base = static_cast<int64_t>(e) * in_dim * out_dim
                           + static_cast<int64_t>(i) * out_dim;
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < out_dim; j += blockDim.x) {
        float g = load_grad_elem<GradT>(grad, row_base + j);
        thread_sum += g * g;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    // Shared memory reduction across warps
    __shared__ float sdata[8];  // max 256 threads = 8 warps
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) sdata[warp_id] = thread_sum;
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) {
            const float grad_sq_row_mean = val / static_cast<float>(out_dim);
            const float inv_in_dim = 1.0f / static_cast<float>(in_dim);
            const int64_t vr_idx = static_cast<int64_t>(e) * in_dim + i;
            float vr = beta2 * v_row[vr_idx] + one_minus_b2 * grad_sq_row_mean;
            v_row[vr_idx] = vr;
            // One atomic per (expert,row) block. This avoids extra warp-level
            // atomics and removes unnecessary shuffle work in the hot path.
            atomicAdd(&v_rms[e], vr * inv_in_dim);
        }
    }
}

// One BLOCK per (expert, col) pair. Threads collaboratively reduce across in_dim.
template <typename GradT>
__global__ void k_factored_v_col(
    const GradT* __restrict__ grad,    // [E, in_dim, out_dim] nmoe layout
    float* __restrict__ v_col,         // [E, out_dim]
    int E,
    int in_dim,
    int out_dim,
    float beta2)
{
    // blockIdx.x = (expert * out_dim + col) index
    const int idx = blockIdx.x;
    const int total = E * out_dim;
    if (idx >= total) return;

    const int e = idx / out_dim;
    const int j = idx % out_dim;

    const float one_minus_b2 = 1.0f - beta2;

    // Each thread sums a strided subset of in_dim
    const int64_t expert_base = static_cast<int64_t>(e) * in_dim * out_dim;
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float g = load_grad_elem<GradT>(grad, expert_base + static_cast<int64_t>(i) * out_dim + j);
        thread_sum += g * g;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    // Shared memory reduction across warps
    __shared__ float sdata[8];  // max 256 threads = 8 warps
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) sdata[warp_id] = thread_sum;
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) {
            float grad_sq_col_mean = val / static_cast<float>(in_dim);
            const int64_t vc_idx = static_cast<int64_t>(e) * out_dim + j;
            v_col[vc_idx] = beta2 * v_col[vc_idx] + one_minus_b2 * grad_sq_col_mean;
        }
    }
}

// Launch wrapper for all factored-v reduction sub-kernels.
template <typename GradT>
inline cudaError_t launch_factored_v_update(
    const GradT* grad,
    float* v_row, float* v_col, float* v_rms,
    int E, int in_dim, int out_dim,
    float beta2,
    cudaStream_t stream)
{
    if (E < 0 || in_dim < 0 || out_dim < 0) return cudaErrorInvalidValue;
    if (E == 0 || in_dim == 0 || out_dim == 0) return cudaSuccess;

    // Zero v_rms before atomicAdd accumulation
    auto err = cudaMemsetAsync(v_rms, 0, E * sizeof(float), stream);
    if (err != cudaSuccess) return err;

    constexpr int THREADS = 256;

    // Sub-kernel 1: row means + v_row update + v_rms accumulation
    // One block per (expert, row) pair; threads reduce across out_dim in parallel
    {
        const int64_t total64 = static_cast<int64_t>(E) * in_dim;
        if (total64 > 2147483647LL) return cudaErrorInvalidValue;
        const dim3 grid(static_cast<uint32_t>(total64));
        k_factored_v_row<GradT><<<grid, THREADS, 0, stream>>>(
            grad, v_row, v_rms, E, in_dim, out_dim, beta2);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Sub-kernel 2: column means + v_col update
    // One block per (expert, col) pair; threads reduce across in_dim in parallel
    {
        const int64_t total64 = static_cast<int64_t>(E) * out_dim;
        if (total64 > 2147483647LL) return cudaErrorInvalidValue;
        const dim3 grid(static_cast<uint32_t>(total64));
        k_factored_v_col<GradT><<<grid, THREADS, 0, stream>>>(
            grad, v_col, E, in_dim, out_dim, beta2);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}


// ============================================================================
// ECO Adam kernel.
//
// Weight NVFP4 buffers:     [E, out_dim, in_dim]  (HF layout)
//   W_packed:  [E, out_dim, in_dim/2]             uint8 (2 nibbles/byte)
//   W_scale:   [E, out_dim, in_dim/group_size]    uint8 (E4M3 per-group scale)
//   W_gs:      [E]                                float32 (global scale ≈ 1.0)
//
// Optimizer states:         [E, in_dim, out_dim]   (nmoe layout)
//   m_data:    [E, in_dim, out_dim]               uint8 (E5M2)
//   m_scale:   [E, in_dim, 1]                     float32 (per-row scale)
//   v_data:    [E, in_dim, out_dim]               uint8 (E4M3)       [kFactoredV=false only]
//   v_scale:   [E, in_dim, 1]                     float32 (per-row)  [kFactoredV=false only]
//   v_row:     [E, in_dim]                        float32            [kFactoredV=true only]
//   v_col:     [E, out_dim]                       float32            [kFactoredV=true only]
//   v_rms:     [E]                                float32            [kFactoredV=true only]
//
// Gradient:                 [E, in_dim, out_dim]   (nmoe layout, FP32)
// ============================================================================

template <bool kStochasticRounding, bool kErrorFeedback, bool kFactoredV = false, typename GradT = float>
__global__ void k_eco_adam_nvfp4_update(
    // NVFP4 weight triplet (HF layout)
    uint8_t* __restrict__ W_packed,       // [E, out_dim, in_dim/2]
    uint8_t* __restrict__ W_scale,        // [E, out_dim, in_dim/group_size]
    float* __restrict__ W_gs,             // [E] (written to 1.0 after requant)

    // FP8 optimizer states (nmoe layout: [E, in_dim, out_dim])
    uint8_t* __restrict__ m_data,         // [E, in_dim, out_dim] E5M2
    float* __restrict__ m_scale,          // [E, in_dim, 1] per-row FP32 scale
    uint8_t* __restrict__ v_data,         // [E, in_dim, out_dim] E4M3 (kFactoredV=false)
    float* __restrict__ v_scale,          // [E, in_dim, 1] per-row (kFactoredV=false)

    // Factored-v pointers (kFactoredV=true only; nullptr when kFactoredV=false)
    const float* __restrict__ v_row,      // [E, in_dim] (pre-updated by Kernel A)
    const float* __restrict__ v_col,      // [E, out_dim] (pre-updated by Kernel A)
    const float* __restrict__ v_rms,      // [E] mean(v_row) per expert

    // Gradient (nmoe layout)
    const GradT* __restrict__ grad,       // [E, in_dim, out_dim]

    // Dimensions
    int E,
    int out_dim,
    int in_dim,
    int group_size,

    // Hyperparameters
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bc2_sqrt,
    float eco_alpha,

    // Philox PRNG seed
    uint32_t prng_seed0,
    uint32_t prng_seed1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Grid: blockIdx.x = out_tile, blockIdx.y = in_tile, blockIdx.z = expert
    const int e = static_cast<int>(blockIdx.z);
    const int out0 = static_cast<int>(blockIdx.x) * TILE_OUT;
    const int in0 = static_cast<int>(blockIdx.y) * TILE_IN;

    const int out_local = static_cast<int>(threadIdx.x);  // [0, 31]
    const int in_iter = static_cast<int>(threadIdx.y);     // [0, 7]
    const int out_row = out0 + out_local;
    const int group_shift = __ffs(group_size) - 1;
    const int groups_per_row = in_dim >> group_shift;

    // Strides for layout indexing
    const int64_t hf_stride_e = static_cast<int64_t>(out_dim) * in_dim;       // per expert in HF
    const int64_t nmoe_stride_e = static_cast<int64_t>(in_dim) * out_dim;     // per expert in nmoe
    const int64_t hf_packed_stride_e = static_cast<int64_t>(out_dim) * (in_dim / 2);
    const int64_t hf_scale_stride_e = static_cast<int64_t>(out_dim) * groups_per_row;

    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;
    const float effective_eps = fmaxf(eps, 1e-6f);

    // ============================================================
    // Phase 0: Cache W_gs[e] in shared memory
    // ============================================================
    // W_gs is read by all blocks for expert e, but Phase 5 writes W_gs[e]=1.0
    // from block (0,0,e). CUDA does not guarantee inter-block ordering within
    // a kernel launch, so a late-starting block could read the updated 1.0
    // instead of the original value. We cache it in shared memory at block
    // start, before any other block's Phase 5 can execute.
    __shared__ float sh_gs;
    if (out_local == 0 && in_iter == 0) {
        sh_gs = W_gs[e];
    }
    __syncthreads();
    const float gs = sh_gs;
    const float inv_gs = (gs > 0.0f) ? (1.0f / gs) : 1.0f;

    // ============================================================
    // Phase 1: Load all inputs, do AdamW update, store w to shared
    // ============================================================
    __shared__ float sh_w[TILE_OUT][TILE_IN + 1];   // +1 padding to avoid bank conflicts
    __shared__ uint8_t sh_nibble[TILE_OUT][TILE_IN]; // E2M1 nibbles for packed write
    float m_reg[ELEMS_PER_THREAD];              // Updated momentum (for ECO + requant)
    float v_reg[ELEMS_PER_THREAD];              // Updated variance (for requant)
    float denom_reg[ELEMS_PER_THREAD];          // Denominator (for ECO error injection)

    #pragma unroll
    for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
        const int in_local = t * THREADS_Y + in_iter;
        const int in_col = in0 + in_local;

        // --- Load NVFP4 weight (HF layout: [e, out_row, in_col]) ---
        const int64_t packed_idx = static_cast<int64_t>(e) * hf_packed_stride_e
                                 + static_cast<int64_t>(out_row) * (in_dim / 2)
                                 + (in_col / 2);
        uint8_t packed_byte = W_packed[packed_idx];
        uint8_t nibble = (in_col & 1) ? (packed_byte >> 4) : (packed_byte & 0x0F);
        float w_raw = ptx::e2m1_nibble_to_f32(nibble);

        // Per-group scale
        const int64_t wscale_idx = static_cast<int64_t>(e) * hf_scale_stride_e
                                 + static_cast<int64_t>(out_row) * groups_per_row
                                 + (in_col >> group_shift);
        float group_sc = ptx::e4m3_byte_to_f32(W_scale[wscale_idx]);
        float w = w_raw * group_sc * inv_gs;

        // --- Load gradient (nmoe layout: [e, in_col, out_row]) ---
        const int64_t grad_idx = static_cast<int64_t>(e) * nmoe_stride_e
                               + static_cast<int64_t>(in_col) * out_dim
                               + out_row;
        float g = load_grad_elem<GradT>(grad, grad_idx);

        // --- Load FP8 m (nmoe layout: [e, in_col, out_row]) ---
        const int64_t mv_idx = grad_idx;  // same layout as gradient
        uint8_t m_byte = m_data[mv_idx];

        // Per-row scale for m: scale[e, in_col, 0]
        const int64_t mv_scale_idx = static_cast<int64_t>(e) * in_dim + in_col;
        float m_sc = m_scale[mv_scale_idx];
        float m_val = ptx::e5m2_byte_to_f32(m_byte) * m_sc;

        // --- Load v (full FP8 or factored) ---
        float v_val;
        if constexpr (!kFactoredV) {
            // Standard path: load FP8 E4M3 v
            uint8_t v_byte = v_data[mv_idx];
            float v_sc = v_scale[mv_scale_idx];
            v_val = ptx::e4m3_byte_to_f32(v_byte) * v_sc;
        }

        // --- AdamW update ---
        // Decoupled weight decay
        if (weight_decay != 0.0f) {
            w -= lr * weight_decay * w;
        }

        // EMA updates
        m_val = beta1 * m_val + one_minus_b1 * g;
        if constexpr (!kFactoredV) {
            v_val = beta2 * v_val + one_minus_b2 * g * g;
        }

        // Denominator
        float denom;
        if constexpr (kFactoredV) {
            // Reconstruct v from factored components (already updated by Kernel A):
            //   v = v_row[e, in_col] * v_col[e, out_row] / v_rms[e]
            float vr = v_row[static_cast<int64_t>(e) * in_dim + in_col];
            float vc = v_col[static_cast<int64_t>(e) * out_dim + out_row];
            float vrms = fmaxf(v_rms[e], 1e-30f);
            v_val = (vr * vc) / vrms;
            denom = sqrtf(v_val) * inv_bc2_sqrt + effective_eps;
        } else {
            denom = sqrtf(v_val) * inv_bc2_sqrt + effective_eps;
        }

        // Weight update
        w -= step_size * m_val / denom;

        // Store to shared and registers
        sh_w[out_local][in_local] = w;
        m_reg[t] = m_val;
        v_reg[t] = v_val;
        denom_reg[t] = denom;
    }

    __syncthreads();

    // ============================================================
    // Phase 2: Compute true per-group NVFP4 scales (no extra kernel pass)
    // ============================================================
    // Previous versions quantized with one scale per 32-wide tile, then ran a
    // second kernel to tighten scales to group_size. We now compute per-group
    // scales directly in this kernel so the external recompute pass is not
    // needed on the hot path.
    const int groups_per_tile = TILE_IN >> group_shift;
    __shared__ float sh_group_amax[TILE_OUT][THREADS_Y];
    __shared__ float sh_group_scale[TILE_OUT][TILE_IN];

    for (int g = 0; g < groups_per_tile; ++g) {
        const int group_start = g * group_size;
        const int group_end = group_start + group_size;
        float thread_amax = 0.0f;

        #pragma unroll
        for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
            const int in_local = t * THREADS_Y + in_iter;
            if (in_local >= group_start && in_local < group_end) {
                thread_amax = fmaxf(thread_amax, fabsf(sh_w[out_local][in_local]));
            }
        }

        sh_group_amax[out_local][in_iter] = thread_amax;
        __syncthreads();

        if (in_iter == 0) {
            float group_amax = 0.0f;
            #pragma unroll
            for (int y = 0; y < THREADS_Y; ++y) {
                group_amax = fmaxf(group_amax, sh_group_amax[out_local][y]);
            }

            float scale_f = group_amax / NVFP4_MAX;
            if (!(scale_f > 0.0f)) scale_f = 1e-12f;
            const uint8_t e4m3_scale = ptx::f32_to_e4m3_byte(scale_f);
            const float decoded_scale = ptx::e4m3_byte_to_f32(e4m3_scale);
            sh_group_scale[out_local][g] = decoded_scale;

            const int in_col_group_start = in0 + group_start;
            if (in_col_group_start < in_dim) {
                const int64_t wscale_idx = static_cast<int64_t>(e) * hf_scale_stride_e
                                         + static_cast<int64_t>(out_row) * groups_per_row
                                         + (in_col_group_start >> group_shift);
                W_scale[wscale_idx] = e4m3_scale;
            }
        }
        __syncthreads();
    }

    // ============================================================
    // Phase 3: Requant weight to E2M1 + ECO error injection + write back
    // ============================================================
    //
    // NVFP4 packed bytes: each byte holds 2 nibbles (even=low, odd=high).
    // The thread mapping (in_local = t*THREADS_Y + in_iter) means two different
    // threads may own the even and odd nibble of the same byte. To avoid a
    // race condition on the packed byte write, each thread stores its computed
    // nibble into sh_nibble[out_local][in_local]. After a barrier, threads
    // with even in_local pack both nibbles and write the byte.
    //
    // FP8 m/v per-row scale: the row spans all out_dim elements but this tile
    // only covers TILE_OUT. We reuse the existing scale for write-back; a
    // separate k_fp8_recompute_row_scale kernel corrects it afterwards.

    #pragma unroll
    for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
        const int in_local = t * THREADS_Y + in_iter;
        const int in_col = in0 + in_local;

        float w = sh_w[out_local][in_local];
        const int group_idx = in_local >> group_shift;
        const float group_scale = sh_group_scale[out_local][group_idx];
        const float inv_scale = (group_scale > 0.0f) ? (1.0f / group_scale) : 1.0f;

        // Stochastic-round to E2M1
        float w_scaled = w * inv_scale;
        float ax = fabsf(w_scaled);
        uint8_t sign_bit = (w < 0.0f) ? 0x8u : 0x0u;

        uint8_t nibble;
        if constexpr (kStochasticRounding) {
            // Full 128-bit Philox counter for collision-free PRNG across
            // arbitrarily large tensors. counter[0:1] = 64-bit element index,
            // counter[2:3] = seeds (step/param identity baked in by caller).
            // The key is fixed; all per-launch variation is in the counter.
            uint64_t elem_idx64 = static_cast<uint64_t>(e) * hf_stride_e
                                + static_cast<uint64_t>(out_row) * in_dim
                                + in_col;
            uint32_t counter[4] = {
                static_cast<uint32_t>(elem_idx64),
                static_cast<uint32_t>(elem_idx64 >> 32),
                prng_seed0,
                prng_seed1
            };
            uint32_t rand_out[4];
            // Fixed Philox key; per-launch uniqueness is in the counter.
            ptx::philox4x32_10(rand_out, counter,
                               0x9E3779B9u, 0xBB67AE85u);
            float rand_u = ptx::uint32_to_uniform(rand_out[0]);
            nibble = ptx::e2m1_stochastic_round(ax, sign_bit, rand_u);
        } else {
            nibble = ptx::e2m1_round_nearest(ax, sign_bit);
        }

        // Dequant w_hat for ECO error
        float w_hat = ptx::e2m1_nibble_to_f32(nibble) * group_scale;

        // ECO error injection into momentum
        float m_val = m_reg[t];
        if constexpr (kErrorFeedback) {
            if (eco_alpha != 0.0f) {
                float error = w - w_hat;
                m_val += eco_alpha * denom_reg[t] * error;
            }
        }

        // Store nibble to shared memory for race-free packed write
        sh_nibble[out_local][in_local] = nibble;

        // --- Write FP8 m (always) and v (only when !kFactoredV) ---
        const int64_t mv_idx = static_cast<int64_t>(e) * nmoe_stride_e
                             + static_cast<int64_t>(in_col) * out_dim
                             + out_row;
        const int64_t mv_scale_idx = static_cast<int64_t>(e) * in_dim + in_col;

        // Read per-row scale for m
        float m_sc = m_scale[mv_scale_idx];

        // Clamp and quantize. If value exceeds scale * FP8_MAX, it clips.
        // The separate scale-recompute pass will fix this.
        float m_inv_sc = (m_sc > 0.0f) ? (1.0f / m_sc) : 1.0f;
        m_data[mv_idx] = ptx::f32_to_e5m2_byte(m_val * m_inv_sc);

        if constexpr (!kFactoredV) {
            // Write FP8 v only when using full v matrix
            float v_sc = v_scale[mv_scale_idx];
            float v_inv_sc = (v_sc > 0.0f) ? (1.0f / v_sc) : 1.0f;
            v_data[mv_idx] = ptx::f32_to_e4m3_byte(v_reg[t] * v_inv_sc);
        }
    }

    // ============================================================
    // Phase 4: Race-free packed byte write
    // ============================================================
    // All nibbles are in sh_nibble[][]. Threads with even in_local indices
    // pack both nibbles and write the full byte.
    __syncthreads();

    #pragma unroll
    for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
        const int in_local = t * THREADS_Y + in_iter;
        const int in_col = in0 + in_local;

        // Only even-column threads write the packed byte (max even in_local is
        // 30, and 30+1=31 < TILE_IN=32, so neighbor always exists within tile).
        if ((in_col & 1) == 0) {
            uint8_t lo_nibble = sh_nibble[out_local][in_local];      // even
            uint8_t hi_nibble = sh_nibble[out_local][in_local + 1];  // odd
            uint8_t packed_byte = (hi_nibble << 4) | (lo_nibble & 0x0F);

            const int64_t packed_idx = static_cast<int64_t>(e) * hf_packed_stride_e
                                     + static_cast<int64_t>(out_row) * (in_dim / 2)
                                     + (in_col / 2);
            W_packed[packed_idx] = packed_byte;
        }
    }

    // ============================================================
    // Phase 5: Write W_gs[e] = 1.0
    // ============================================================
    // After requantization, the weights are stored with the new E8M0 block
    // scales. The global scale factor is no longer meaningful — set it to 1.0
    // so subsequent dequant operations see: value = nibble * group_scale / 1.0.
    // Only one thread per expert needs to do this.
    if (out_local == 0 && in_iter == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        W_gs[e] = 1.0f;
    }

#else
    (void)W_packed; (void)W_scale; (void)W_gs;
    (void)m_data; (void)m_scale; (void)v_data; (void)v_scale;
    (void)v_row; (void)v_col; (void)v_rms;
    (void)grad; (void)E; (void)out_dim; (void)in_dim; (void)group_size;
    (void)lr; (void)beta1; (void)beta2; (void)weight_decay; (void)eps;
    (void)step_size; (void)inv_bc2_sqrt; (void)eco_alpha;
    (void)prng_seed0; (void)prng_seed1;
    __trap();
#endif
}


// ============================================================================
// FP8 per-row scale recomputation kernel
// ============================================================================
// After the main ECO Adam kernel writes FP8 m/v values using the old per-row
// scale, this kernel recomputes the correct per-row scale by scanning all
// out_dim elements per row, then rescales the FP8 values accordingly.
//
// Grid: one block per row = (E * in_dim) blocks, 256 threads per block.
// Threads collaboratively reduce across out_dim for amax, then rescale in parallel.

template <bool kIsE5M2>
__global__ void k_fp8_recompute_row_scale(
    uint8_t* __restrict__ data,       // [E, in_dim, out_dim]
    float* __restrict__ row_scale,    // [E, in_dim, 1]
    int E,
    int in_dim,
    int out_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // One block per row. blockIdx.x = row index.
    const int row = blockIdx.x;
    const int total_rows = E * in_dim;
    if (row >= total_rows) return;

    constexpr float FP8_MAX = kIsE5M2 ? 57344.0f : 448.0f;

    const float old_scale = row_scale[row];
    const int64_t row_base = static_cast<int64_t>(row) * out_dim;

    // Pass 1: parallel amax reduction across out_dim
    float thread_amax = 0.0f;
    for (int j = threadIdx.x; j < out_dim; j += blockDim.x) {
        uint8_t byte = data[row_base + j];
        float val;
        if constexpr (kIsE5M2) {
            val = ptx::e5m2_byte_to_f32(byte) * old_scale;
        } else {
            val = ptx::e4m3_byte_to_f32(byte) * old_scale;
        }
        thread_amax = fmaxf(thread_amax, fabsf(val));
    }

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_amax = fmaxf(thread_amax, __shfl_down_sync(0xFFFFFFFF, thread_amax, offset));

    // Cross-warp reduction via shared memory
    __shared__ float sdata[8];  // max 256 threads = 8 warps
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) sdata[warp_id] = thread_amax;
    __syncthreads();

    // Final reduction + broadcast scale
    __shared__ float sh_ratio;
    if (warp_id == 0) {
        float val = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        if (lane == 0) {
            float amax = val;
            float new_scale = fmaxf(amax / FP8_MAX, 1e-30f);
            row_scale[row] = new_scale;
            sh_ratio = old_scale / new_scale;
        }
    }
    __syncthreads();

    // Pass 2: parallel rescale
    float ratio = sh_ratio;
    for (int j = threadIdx.x; j < out_dim; j += blockDim.x) {
        uint8_t byte = data[row_base + j];
        float val;
        if constexpr (kIsE5M2) {
            val = ptx::e5m2_byte_to_f32(byte) * ratio;
            data[row_base + j] = ptx::f32_to_e5m2_byte(val);
        } else {
            val = ptx::e4m3_byte_to_f32(byte) * ratio;
            data[row_base + j] = ptx::f32_to_e4m3_byte(val);
        }
    }
#else
    (void)data; (void)row_scale; (void)E; (void)in_dim; (void)out_dim;
    __trap();
#endif
}

// Combined m/v row-scale recompute for non-factored FP8 optimizer states.
// This fuses two post-passes into one launch:
// - m_data (E5M2) + m_scale
// - v_data (E4M3) + v_scale
// and reuses the row traversal for both tensors.
__global__ void k_fp8_recompute_row_scale_pair(
    uint8_t* __restrict__ m_data,       // [E, in_dim, out_dim] E5M2
    float* __restrict__ m_scale,        // [E, in_dim]
    uint8_t* __restrict__ v_data,       // [E, in_dim, out_dim] E4M3
    float* __restrict__ v_scale,        // [E, in_dim]
    int E,
    int in_dim,
    int out_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row = blockIdx.x;
    const int total_rows = E * in_dim;
    if (row >= total_rows) return;

    constexpr float E5M2_MAX = 57344.0f;
    constexpr float E4M3_MAX = 448.0f;

    const float old_m_scale = m_scale[row];
    const float old_v_scale = v_scale[row];
    const int64_t row_base = static_cast<int64_t>(row) * out_dim;

    float thread_amax_m = 0.0f;
    float thread_amax_v = 0.0f;
    for (int j = threadIdx.x; j < out_dim; j += blockDim.x) {
        const uint8_t m_byte = m_data[row_base + j];
        const uint8_t v_byte = v_data[row_base + j];
        const float m_val = ptx::e5m2_byte_to_f32(m_byte) * old_m_scale;
        const float v_val = ptx::e4m3_byte_to_f32(v_byte) * old_v_scale;
        thread_amax_m = fmaxf(thread_amax_m, fabsf(m_val));
        thread_amax_v = fmaxf(thread_amax_v, fabsf(v_val));
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_amax_m = fmaxf(thread_amax_m, __shfl_down_sync(0xFFFFFFFF, thread_amax_m, offset));
        thread_amax_v = fmaxf(thread_amax_v, __shfl_down_sync(0xFFFFFFFF, thread_amax_v, offset));
    }

    __shared__ float sdata_m[8];
    __shared__ float sdata_v[8];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        sdata_m[warp_id] = thread_amax_m;
        sdata_v[warp_id] = thread_amax_v;
    }
    __syncthreads();

    __shared__ float sh_ratio_m;
    __shared__ float sh_ratio_v;
    if (warp_id == 0) {
        float max_m = (lane < (blockDim.x >> 5)) ? sdata_m[lane] : 0.0f;
        float max_v = (lane < (blockDim.x >> 5)) ? sdata_v[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_m = fmaxf(max_m, __shfl_down_sync(0xFFFFFFFF, max_m, offset));
            max_v = fmaxf(max_v, __shfl_down_sync(0xFFFFFFFF, max_v, offset));
        }
        if (lane == 0) {
            const float new_m_scale = fmaxf(max_m / E5M2_MAX, 1e-30f);
            const float new_v_scale = fmaxf(max_v / E4M3_MAX, 1e-30f);
            m_scale[row] = new_m_scale;
            v_scale[row] = new_v_scale;
            sh_ratio_m = old_m_scale / new_m_scale;
            sh_ratio_v = old_v_scale / new_v_scale;
        }
    }
    __syncthreads();

    const float ratio_m = sh_ratio_m;
    const float ratio_v = sh_ratio_v;
    for (int j = threadIdx.x; j < out_dim; j += blockDim.x) {
        const int64_t idx = row_base + j;
        const float m_val = ptx::e5m2_byte_to_f32(m_data[idx]) * ratio_m;
        const float v_val = ptx::e4m3_byte_to_f32(v_data[idx]) * ratio_v;
        m_data[idx] = ptx::f32_to_e5m2_byte(m_val);
        v_data[idx] = ptx::f32_to_e4m3_byte(v_val);
    }
#else
    (void)m_data; (void)m_scale; (void)v_data; (void)v_scale;
    (void)E; (void)in_dim; (void)out_dim;
    __trap();
#endif
}


// ============================================================================
// Launch wrapper (non-factored-v — backward compatible)
// ============================================================================

template <typename GradT>
inline cudaError_t launch_eco_adam_nvfp4_update_t(
    uint8_t* W_packed, uint8_t* W_scale, float* W_gs,
    uint8_t* m_data, float* m_scale_f32,
    uint8_t* v_data, float* v_scale_f32,
    const GradT* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt, float eco_alpha,
    bool stochastic_rounding, bool error_feedback,
    uint32_t prng_seed0, uint32_t prng_seed1,
    cudaStream_t stream)
{
    if (E < 0 || out_dim < 0 || in_dim < 0) return cudaErrorInvalidValue;
    if (E == 0 || out_dim == 0 || in_dim == 0) return cudaSuccess;
    if ((out_dim & 31) != 0) return cudaErrorInvalidValue;
    if ((in_dim & 31) != 0) return cudaErrorInvalidValue;
    if (group_size <= 0 || group_size > TILE_IN || (TILE_IN % group_size) != 0) {
        return cudaErrorInvalidValue;
    }

    // Main ECO Adam kernel (kFactoredV=false)
    const dim3 block(THREADS_X, THREADS_Y, 1);
    const dim3 grid(ceil_div(out_dim, TILE_OUT), ceil_div(in_dim, TILE_IN), E);

    // v_row/v_col/v_rms are nullptr for the non-factored path
    const float* v_row_null = nullptr;
    const float* v_col_null = nullptr;
    const float* v_rms_null = nullptr;

    if (stochastic_rounding && error_feedback) {
        k_eco_adam_nvfp4_update<true, true, false, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (stochastic_rounding) {
        k_eco_adam_nvfp4_update<true, false, false, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (error_feedback) {
        k_eco_adam_nvfp4_update<false, true, false, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else {
        k_eco_adam_nvfp4_update<false, false, false, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation pass for m and v (fused pair kernel).
    const int64_t total_rows64 = static_cast<int64_t>(E) * in_dim;
    if (total_rows64 > 2147483647LL) return cudaErrorInvalidValue;
    const int total_rows = static_cast<int>(total_rows64);
    constexpr int RECOMP_THREADS = 256;

    k_fp8_recompute_row_scale_pair<<<total_rows, RECOMP_THREADS, 0, stream>>>(
        m_data, m_scale_f32, v_data, v_scale_f32, E, in_dim, out_dim);
    return cudaGetLastError();
}

inline cudaError_t launch_eco_adam_nvfp4_update(
    uint8_t* W_packed, uint8_t* W_scale, float* W_gs,
    uint8_t* m_data, float* m_scale_f32,
    uint8_t* v_data, float* v_scale_f32,
    const float* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt, float eco_alpha,
    bool stochastic_rounding, bool error_feedback,
    uint32_t prng_seed0, uint32_t prng_seed1,
    cudaStream_t stream)
{
    return launch_eco_adam_nvfp4_update_t<float>(
        W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32, grad,
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding, error_feedback, prng_seed0, prng_seed1, stream);
}

inline cudaError_t launch_eco_adam_nvfp4_update_bf16(
    uint8_t* W_packed, uint8_t* W_scale, float* W_gs,
    uint8_t* m_data, float* m_scale_f32,
    uint8_t* v_data, float* v_scale_f32,
    const __nv_bfloat16* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt, float eco_alpha,
    bool stochastic_rounding, bool error_feedback,
    uint32_t prng_seed0, uint32_t prng_seed1,
    cudaStream_t stream)
{
    return launch_eco_adam_nvfp4_update_t<__nv_bfloat16>(
        W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32, grad,
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding, error_feedback, prng_seed0, prng_seed1, stream);
}

// ============================================================================
// Launch wrapper (factored-v path)
// ============================================================================

template <typename GradT>
inline cudaError_t launch_eco_adam_nvfp4_fv_update_t(
    uint8_t* W_packed, uint8_t* W_scale, float* W_gs,
    uint8_t* m_data, float* m_scale_f32,
    float* v_row, float* v_col, float* v_rms,
    const GradT* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt, float eco_alpha,
    bool stochastic_rounding, bool error_feedback,
    uint32_t prng_seed0, uint32_t prng_seed1,
    cudaStream_t stream)
{
    if (E < 0 || out_dim < 0 || in_dim < 0) return cudaErrorInvalidValue;
    if (E == 0 || out_dim == 0 || in_dim == 0) return cudaSuccess;
    if ((out_dim & 31) != 0) return cudaErrorInvalidValue;
    if ((in_dim & 31) != 0) return cudaErrorInvalidValue;
    if (group_size <= 0 || group_size > TILE_IN || (TILE_IN % group_size) != 0) {
        return cudaErrorInvalidValue;
    }

    // Step 1: Run factored-v reduction kernels (updates v_row, v_col, v_rms)
    auto err = launch_factored_v_update<GradT>(
        grad, v_row, v_col, v_rms, E, in_dim, out_dim, beta2, stream);
    if (err != cudaSuccess) return err;

    // Step 2: Main ECO Adam kernel (kFactoredV=true)
    const dim3 block(THREADS_X, THREADS_Y, 1);
    const dim3 grid(ceil_div(out_dim, TILE_OUT), ceil_div(in_dim, TILE_IN), E);

    // v_data/v_scale are nullptr for the factored path
    uint8_t* v_data_null = nullptr;
    float* v_scale_null = nullptr;

    if (stochastic_rounding && error_feedback) {
        k_eco_adam_nvfp4_update<true, true, true, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (stochastic_rounding) {
        k_eco_adam_nvfp4_update<true, false, true, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (error_feedback) {
        k_eco_adam_nvfp4_update<false, true, true, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else {
        k_eco_adam_nvfp4_update<false, false, true, GradT><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation pass for m only (one block per row, parallel reduction)
    const int64_t total_rows64 = static_cast<int64_t>(E) * in_dim;
    if (total_rows64 > 2147483647LL) return cudaErrorInvalidValue;
    const int total_rows = static_cast<int>(total_rows64);
    constexpr int RECOMP_THREADS = 256;

    k_fp8_recompute_row_scale<true><<<total_rows, RECOMP_THREADS, 0, stream>>>(
        m_data, m_scale_f32, E, in_dim, out_dim);
    return cudaGetLastError();
}

inline cudaError_t launch_eco_adam_nvfp4_fv_update(
    uint8_t* W_packed, uint8_t* W_scale, float* W_gs,
    uint8_t* m_data, float* m_scale_f32,
    float* v_row, float* v_col, float* v_rms,
    const float* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt, float eco_alpha,
    bool stochastic_rounding, bool error_feedback,
    uint32_t prng_seed0, uint32_t prng_seed1,
    cudaStream_t stream)
{
    return launch_eco_adam_nvfp4_fv_update_t<float>(
        W_packed, W_scale, W_gs, m_data, m_scale_f32, v_row, v_col, v_rms, grad,
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding, error_feedback, prng_seed0, prng_seed1, stream);
}

inline cudaError_t launch_eco_adam_nvfp4_fv_update_bf16(
    uint8_t* W_packed, uint8_t* W_scale, float* W_gs,
    uint8_t* m_data, float* m_scale_f32,
    float* v_row, float* v_col, float* v_rms,
    const __nv_bfloat16* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt, float eco_alpha,
    bool stochastic_rounding, bool error_feedback,
    uint32_t prng_seed0, uint32_t prng_seed1,
    cudaStream_t stream)
{
    return launch_eco_adam_nvfp4_fv_update_t<__nv_bfloat16>(
        W_packed, W_scale, W_gs, m_data, m_scale_f32, v_row, v_col, v_rms, grad,
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding, error_feedback, prng_seed0, prng_seed1, stream);
}

// ============================================================================
// AdamA m/v accumulation kernel (lightweight, no weight update)
// ============================================================================
//
// For gradient accumulation via Adam Accumulation (AdamA, arXiv:2305.19982):
// instead of storing separate gradient accumulation buffers (~5.85 GiB),
// we update m/v directly each micro-step with fractional betas.
//
// After K micro-steps with β^(1/K):
//   m_K = β * m_0 + Σ (1-β^(1/K)) * β^((K-i)/K) * gᵢ
//
// Bias corrections remain identical: 1-(β^(1/K))^(t*K) = 1-β^t
//
// This kernel handles NON-FINAL micro-steps only. It updates m/v without
// touching NVFP4 weight buffers — no weight update, no ECO error injection.
// The final micro-step uses the existing k_eco_adam_nvfp4_update kernel
// (with fractional betas passed as the beta1/beta2 arguments).
//
// Memory: zero additional buffers. m/v states already exist.
//
// Layout: gradient [E, in_dim, out_dim] (nmoe), m/v same layout.
// ============================================================================

// Per-element m/v EMA update with fractional betas.
// One thread per element in the (E, in_dim, out_dim) tensor.
template <bool kFactoredV, typename GradT = float>
__global__ void k_eco_mv_accumulate(
    // FP8 optimizer states (nmoe layout: [E, in_dim, out_dim])
    uint8_t* __restrict__ m_data,         // [E, in_dim, out_dim] E5M2
    float* __restrict__ m_scale,          // [E * in_dim] per-row FP32 scale
    uint8_t* __restrict__ v_data,         // [E, in_dim, out_dim] E4M3 (kFactoredV=false only)
    float* __restrict__ v_scale,          // [E * in_dim] per-row (kFactoredV=false only)

    // Gradient (nmoe layout)
    const GradT* __restrict__ grad,       // [E, in_dim, out_dim]

    // Dimensions
    int E,
    int in_dim,
    int out_dim,

    // Fractional betas: β^(1/K)
    float beta1_frac,
    float beta2_frac)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int64_t total = static_cast<int64_t>(E) * in_dim * out_dim;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float one_minus_b1 = 1.0f - beta1_frac;
    const float one_minus_b2 = 1.0f - beta2_frac;

    for (int64_t idx = tid; idx < total; idx += stride) {
        // Compute row index for per-row scale: row = idx / out_dim
        const int64_t row = idx / out_dim;

        // Load gradient
        float g = load_grad_elem<GradT>(grad, idx);

        // --- Update m ---
        float m_sc = m_scale[row];
        float m_val = ptx::e5m2_byte_to_f32(m_data[idx]) * m_sc;
        m_val = beta1_frac * m_val + one_minus_b1 * g;

        // Write m back with old scale (scale recompute pass follows)
        float m_inv_sc = (m_sc > 0.0f) ? (1.0f / m_sc) : 1.0f;
        m_data[idx] = ptx::f32_to_e5m2_byte(m_val * m_inv_sc);

        // --- Update v (full FP8 path only; factored-v handled separately) ---
        if constexpr (!kFactoredV) {
            float v_sc = v_scale[row];
            float v_val = ptx::e4m3_byte_to_f32(v_data[idx]) * v_sc;
            v_val = beta2_frac * v_val + one_minus_b2 * g * g;
            float v_inv_sc = (v_sc > 0.0f) ? (1.0f / v_sc) : 1.0f;
            v_data[idx] = ptx::f32_to_e4m3_byte(v_val * v_inv_sc);
        }
    }
#else
    (void)m_data; (void)m_scale; (void)v_data; (void)v_scale;
    (void)grad; (void)E; (void)in_dim; (void)out_dim;
    (void)beta1_frac; (void)beta2_frac;
    __trap();
#endif
}


// Launch wrapper for AdamA m/v accumulation (non-factored v)
template <typename GradT>
inline cudaError_t launch_eco_mv_accumulate_t(
    uint8_t* m_data, float* m_scale,
    uint8_t* v_data, float* v_scale,
    const GradT* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    if (E < 0 || in_dim < 0 || out_dim < 0) return cudaErrorInvalidValue;
    const int64_t total = static_cast<int64_t>(E) * in_dim * out_dim;
    constexpr int THREADS = 256;
    if (total <= 0) return cudaSuccess;

    uint32_t grid_x = 0;
    auto err = choose_stride_grid_x(total, THREADS, &grid_x);
    if (err != cudaSuccess) return err;
    const dim3 grid(grid_x);
    const dim3 block(THREADS);

    k_eco_mv_accumulate<false, GradT><<<grid, block, 0, stream>>>(
        m_data, m_scale, v_data, v_scale, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation for m and v (fused pair kernel).
    const int64_t total_rows64 = static_cast<int64_t>(E) * in_dim;
    if (total_rows64 > 2147483647LL) return cudaErrorInvalidValue;
    const int total_rows = static_cast<int>(total_rows64);
    constexpr int RECOMP_THREADS = 256;

    k_fp8_recompute_row_scale_pair<<<total_rows, RECOMP_THREADS, 0, stream>>>(
        m_data, m_scale, v_data, v_scale, E, in_dim, out_dim);
    return cudaGetLastError();
}

inline cudaError_t launch_eco_mv_accumulate(
    uint8_t* m_data, float* m_scale,
    uint8_t* v_data, float* v_scale,
    const float* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return launch_eco_mv_accumulate_t<float>(
        m_data, m_scale, v_data, v_scale, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac, stream);
}

inline cudaError_t launch_eco_mv_accumulate_bf16(
    uint8_t* m_data, float* m_scale,
    uint8_t* v_data, float* v_scale,
    const __nv_bfloat16* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return launch_eco_mv_accumulate_t<__nv_bfloat16>(
        m_data, m_scale, v_data, v_scale, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac, stream);
}


// Launch wrapper for AdamA m/v accumulation (factored-v)
template <typename GradT>
inline cudaError_t launch_eco_mv_accumulate_fv_t(
    uint8_t* m_data, float* m_scale,
    float* v_row, float* v_col, float* v_rms,
    const GradT* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    if (E < 0 || in_dim < 0 || out_dim < 0) return cudaErrorInvalidValue;
    if (E == 0 || in_dim == 0 || out_dim == 0) return cudaSuccess;

    // Step 1: Run factored-v reduction kernels to update v_row, v_col, v_rms
    // (these use beta2_frac instead of beta2)
    auto err = launch_factored_v_update<GradT>(
        grad, v_row, v_col, v_rms, E, in_dim, out_dim, beta2_frac, stream);
    if (err != cudaSuccess) return err;

    // Step 2: Update m only (kFactoredV=true skips v_data update)
    const int64_t total = static_cast<int64_t>(E) * in_dim * out_dim;
    constexpr int THREADS = 256;
    uint32_t grid_x = 0;
    err = choose_stride_grid_x(total, THREADS, &grid_x);
    if (err != cudaSuccess) return err;
    const dim3 grid(grid_x);
    const dim3 block(THREADS);

    k_eco_mv_accumulate<true, GradT><<<grid, block, 0, stream>>>(
        m_data, m_scale, nullptr, nullptr, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation for m only (one block per row, parallel reduction)
    const int64_t total_rows64 = static_cast<int64_t>(E) * in_dim;
    if (total_rows64 > 2147483647LL) return cudaErrorInvalidValue;
    const int total_rows = static_cast<int>(total_rows64);
    constexpr int RECOMP_THREADS = 256;

    k_fp8_recompute_row_scale<true><<<total_rows, RECOMP_THREADS, 0, stream>>>(
        m_data, m_scale, E, in_dim, out_dim);
    return cudaGetLastError();
}

inline cudaError_t launch_eco_mv_accumulate_fv(
    uint8_t* m_data, float* m_scale,
    float* v_row, float* v_col, float* v_rms,
    const float* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return launch_eco_mv_accumulate_fv_t<float>(
        m_data, m_scale, v_row, v_col, v_rms, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac, stream);
}

inline cudaError_t launch_eco_mv_accumulate_fv_bf16(
    uint8_t* m_data, float* m_scale,
    float* v_row, float* v_col, float* v_rms,
    const __nv_bfloat16* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return launch_eco_mv_accumulate_fv_t<__nv_bfloat16>(
        m_data, m_scale, v_row, v_col, v_rms, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac, stream);
}


// ============================================================================
// NVFP4 per-group scale recomputation kernel
// ============================================================================
// After the ECO Adam kernel requantizes NVFP4 weights with per-tile E8M0
// block scales, the per-group E4M3 scale factors (group_size=16) may be
// stale because the kernel writes the same block scale (computed over
// TILE_IN=32 elements) to all groups within a tile.
//
// This kernel recomputes correct per-group E4M3 scales by:
//   1. Unpacking E2M1 nibbles from W_packed
//   2. Dequantizing to FP32 using the current (stale) E4M3 group scale
//   3. Computing group amax over group_size elements
//   4. Encoding the new scale as E4M3
//   5. Re-quantizing E2M1 nibbles against the new scale (round-to-nearest)
//   6. Repacking nibbles and writing both W_packed and W_scale in-place
//
// Replaces ~33 separate PyTorch kernel launches in eco.py
// _recompute_nvfp4_group_scales with a single fused kernel.
//
// Thread mapping (matches k_eco_adam_nvfp4_update):
//   blockDim = (32, 8, 1) = 256 threads
//   gridDim  = (ceil(out_dim, 32), ceil(in_dim, 32), E)
//   Each thread processes 4 elements across in_dim (TILE_IN/THREADS_Y = 32/8).
//
// For group_size=16 (half a warp), the 16 elements within a group are handled
// by threads across both threadIdx.x and threadIdx.y dimensions. Within each
// tile, groups are processed independently — shared memory gathers nibbles
// for race-free packed byte write-back.
// ============================================================================

template <int kGroupSize = 16>
__global__ void k_nvfp4_recompute_group_scales(
    uint8_t* __restrict__ W_packed,     // [E, out_dim, in_dim/2]
    uint8_t* __restrict__ W_scale,      // [E, out_dim, in_dim/group_size]
    const float* __restrict__ W_gs,     // [E]
    int E, int out_dim, int in_dim,
    int group_size)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // E2M1 magnitude LUT (indexed by 3-bit unsigned code, sign ignored)
    // Code: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
    constexpr float kE2M1Mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    // Grid: blockIdx.x = out_tile, blockIdx.y = in_tile, blockIdx.z = expert
    const int e = static_cast<int>(blockIdx.z);
    const int out0 = static_cast<int>(blockIdx.x) * TILE_OUT;
    const int in0 = static_cast<int>(blockIdx.y) * TILE_IN;

    const int out_local = static_cast<int>(threadIdx.x);  // [0, 31]
    const int in_iter = static_cast<int>(threadIdx.y);     // [0, 7]
    const int out_row = out0 + out_local;

    // Bounds check: skip threads beyond tensor dimensions
    const bool out_valid = (out_row < out_dim);

    // Strides for HF layout indexing
    const int64_t hf_packed_stride_e = static_cast<int64_t>(out_dim) * (in_dim / 2);
    const int64_t hf_scale_stride_e = static_cast<int64_t>(out_dim) * (in_dim / group_size);

    // Cache global scale for this expert
    __shared__ float sh_gs;
    if (out_local == 0 && in_iter == 0) {
        sh_gs = W_gs[e];
    }
    __syncthreads();
    const float gs = sh_gs;

    // Shared memory for nibble collection (race-free packed byte write-back)
    __shared__ uint8_t sh_nibble[TILE_OUT][TILE_IN];
    // Shared memory for dequantized real values (needed for group amax + requant)
    __shared__ float sh_real[TILE_OUT][TILE_IN + 1];  // +1 to avoid bank conflicts
    // Shared memory for cross-warp amax reduction (one slot per row per warp)
    __shared__ float sh_amax[TILE_OUT][THREADS_Y];
    // Shared memory for broadcasting new scale to all threads
    __shared__ float sh_new_scale_f32[TILE_OUT];

    // ================================================================
    // Phase 1: Unpack, dequantize, store to shared memory
    // ================================================================
    // Initialize nibble buffer to zero (for out-of-bounds elements that may
    // still be read during packed byte write-back in Phase 3).
    #pragma unroll
    for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
        const int in_local = t * THREADS_Y + in_iter;
        sh_nibble[out_local][in_local] = 0;
    }

    #pragma unroll
    for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
        const int in_local = t * THREADS_Y + in_iter;
        const int in_col = in0 + in_local;
        const bool in_valid = (in_col < in_dim);

        float real_val = 0.0f;
        if (out_valid && in_valid) {
            // Load packed byte and extract nibble
            const int64_t packed_idx = static_cast<int64_t>(e) * hf_packed_stride_e
                                     + static_cast<int64_t>(out_row) * (in_dim / 2)
                                     + (in_col / 2);
            uint8_t packed_byte = W_packed[packed_idx];
            uint8_t nibble = (in_col & 1) ? (packed_byte >> 4) : (packed_byte & 0x0F);

            // Decode E2M1 magnitude (absolute value only for amax)
            float magnitude = kE2M1Mag[nibble & 0x7];

            // Load current E4M3 group scale
            const int64_t wscale_idx = static_cast<int64_t>(e) * hf_scale_stride_e
                                     + static_cast<int64_t>(out_row) * (in_dim / group_size)
                                     + (in_col / group_size);
            float old_scale = ptx::e4m3_byte_to_f32(W_scale[wscale_idx]);

            // Dequantize to true FP32 value (signed)
            float sign = (nibble & 0x8) ? -1.0f : 1.0f;
            real_val = sign * magnitude * old_scale * gs;
        }
        sh_real[out_local][in_local] = real_val;
    }

    __syncthreads();

    // ================================================================
    // Phase 2: Compute per-group amax, new E4M3 scale, re-quantize
    // ================================================================
    // groups_per_tile = TILE_IN / group_size.
    // For group_size=16, TILE_IN=32: 2 groups per tile row.
    // For group_size=32, TILE_IN=32: 1 group per tile row.
    //
    // Strategy: each thread iterates over its elements. For each element,
    // determine which group it belongs to. Use shared memory to collect
    // the group amax, then re-quantize.
    //
    // We process groups sequentially within each row.  threadIdx.y threads
    // (8 threads) collaborate to reduce group_size elements.

    const int groups_per_tile = TILE_IN / group_size;

    // For each group in this tile row:
    for (int g = 0; g < groups_per_tile; ++g) {
        const int group_start = g * group_size;  // local offset within tile
        const int group_start_global = in0 + group_start;

        // Skip if this group is entirely out of bounds
        if (group_start_global >= in_dim) break;

        // Each thread computes partial amax over elements it owns within this group
        float thread_amax = 0.0f;
        #pragma unroll
        for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
            const int in_local = t * THREADS_Y + in_iter;
            // Check if this element belongs to the current group
            if (in_local >= group_start && in_local < group_start + group_size) {
                if (out_valid && (in0 + in_local) < in_dim) {
                    thread_amax = fmaxf(thread_amax, fabsf(sh_real[out_local][in_local]));
                }
            }
        }

        // Cross-warp amax reduction.
        // threadIdx.y threads span different warps (warp_id = threadIdx.y for
        // blockDim.x=32), so we reduce via shared memory rather than shuffles.
        sh_amax[out_local][in_iter] = thread_amax;
        __syncthreads();

        // threadIdx.y == 0 reduces across all THREADS_Y entries for this row
        float group_amax = 0.0f;
        if (in_iter == 0 && out_valid) {
            #pragma unroll
            for (int y = 0; y < THREADS_Y; ++y) {
                group_amax = fmaxf(group_amax, sh_amax[out_local][y]);
            }

            // Compute new scale: amax / FP4_MAX
            float new_scale = group_amax / NVFP4_MAX;
            // Clamp to avoid zero scale (E4M3 can't represent 0 as a scale)
            new_scale = fmaxf(new_scale, 1e-12f);
            // Encode as E4M3
            uint8_t new_e4m3 = ptx::f32_to_e4m3_byte(new_scale);
            // Decode back to get the actual represented value
            float new_scale_decoded = ptx::e4m3_byte_to_f32(new_e4m3);

            sh_new_scale_f32[out_local] = new_scale_decoded;

            // Write new E4M3 scale to W_scale
            if (group_start_global < in_dim) {
                const int64_t wscale_idx = static_cast<int64_t>(e) * hf_scale_stride_e
                                         + static_cast<int64_t>(out_row) * (in_dim / group_size)
                                         + (group_start_global / group_size);
                W_scale[wscale_idx] = new_e4m3;
            }
        }

        __syncthreads();

        // Re-quantize: each thread re-quantizes its elements in this group
        #pragma unroll
        for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
            const int in_local = t * THREADS_Y + in_iter;
            if (in_local >= group_start && in_local < group_start + group_size) {
                const int in_col = in0 + in_local;
                if (out_valid && in_col < in_dim) {
                    float real_val = sh_real[out_local][in_local];
                    float decoded_scale = sh_new_scale_f32[out_local];

                    // Compute the denominator for requantization
                    float denom = decoded_scale * gs;
                    float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

                    // Scale the value and round to nearest E2M1
                    float ax = fabsf(real_val) * inv_denom;
                    uint8_t sign_bit = (real_val < 0.0f) ? 0x8u : 0x0u;
                    uint8_t nibble = ptx::e2m1_round_nearest(ax, sign_bit);

                    sh_nibble[out_local][in_local] = nibble;
                }
            }
        }

        __syncthreads();
    }  // end for each group

    // ================================================================
    // Phase 3: Race-free packed byte write-back
    // ================================================================
    // All nibbles are in sh_nibble[][]. Threads with even in_local
    // pack both nibbles and write the full byte.

    #pragma unroll
    for (int t = 0; t < ELEMS_PER_THREAD; ++t) {
        const int in_local = t * THREADS_Y + in_iter;
        const int in_col = in0 + in_local;

        if ((in_col & 1) == 0 && out_valid && in_col < in_dim) {
            uint8_t lo_nibble = sh_nibble[out_local][in_local];
            uint8_t hi_nibble = sh_nibble[out_local][in_local + 1];
            uint8_t packed_byte = (hi_nibble << 4) | (lo_nibble & 0x0F);

            const int64_t packed_idx = static_cast<int64_t>(e) * hf_packed_stride_e
                                     + static_cast<int64_t>(out_row) * (in_dim / 2)
                                     + (in_col / 2);
            W_packed[packed_idx] = packed_byte;
        }
    }

#else
    (void)W_packed; (void)W_scale; (void)W_gs;
    (void)E; (void)out_dim; (void)in_dim; (void)group_size;
    __trap();
#endif
}


// Launch wrapper for NVFP4 per-group scale recomputation.
inline cudaError_t launch_nvfp4_recompute_group_scales(
    uint8_t* W_packed, uint8_t* W_scale, const float* W_gs,
    int E, int out_dim, int in_dim, int group_size,
    cudaStream_t stream)
{
    if (E < 0 || out_dim < 0 || in_dim < 0) return cudaErrorInvalidValue;
    if (E == 0 || out_dim == 0 || in_dim == 0) return cudaSuccess;
    if (group_size <= 0 || group_size > TILE_IN || (TILE_IN % group_size) != 0 || (in_dim % group_size) != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(THREADS_X, THREADS_Y, 1);
    const dim3 grid(ceil_div(out_dim, TILE_OUT), ceil_div(in_dim, TILE_IN), E);

    k_nvfp4_recompute_group_scales<16><<<grid, block, 0, stream>>>(
        W_packed, W_scale, W_gs, E, out_dim, in_dim, group_size);

    return cudaGetLastError();
}


}  // namespace eco_adam
}  // namespace nmoe


// ============================================================================
// C API
// ============================================================================

// Original non-factored-v entry point (backward compatible)
extern "C" cudaError_t eco_adam_nvfp4_update(
    void* W_packed, void* W_scale, void* W_gs,
    void* m_data, void* m_scale, void* v_data, void* v_scale,
    const void* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    float eco_alpha,
    int stochastic_rounding, int error_feedback,
    unsigned int prng_seed0, unsigned int prng_seed1,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_adam_nvfp4_update(
        reinterpret_cast<uint8_t*>(W_packed),
        reinterpret_cast<uint8_t*>(W_scale),
        reinterpret_cast<float*>(W_gs),
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<uint8_t*>(v_data),
        reinterpret_cast<float*>(v_scale),
        reinterpret_cast<const float*>(grad),
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding != 0, error_feedback != 0,
        prng_seed0, prng_seed1,
        stream);
}

extern "C" cudaError_t eco_adam_nvfp4_update_bf16(
    void* W_packed, void* W_scale, void* W_gs,
    void* m_data, void* m_scale, void* v_data, void* v_scale,
    const void* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    float eco_alpha,
    int stochastic_rounding, int error_feedback,
    unsigned int prng_seed0, unsigned int prng_seed1,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_adam_nvfp4_update_bf16(
        reinterpret_cast<uint8_t*>(W_packed),
        reinterpret_cast<uint8_t*>(W_scale),
        reinterpret_cast<float*>(W_gs),
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<uint8_t*>(v_data),
        reinterpret_cast<float*>(v_scale),
        reinterpret_cast<const __nv_bfloat16*>(grad),
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding != 0, error_feedback != 0,
        prng_seed0, prng_seed1,
        stream);
}

// Factored-v entry point (new)
extern "C" cudaError_t eco_adam_nvfp4_fv_update(
    void* W_packed, void* W_scale, void* W_gs,
    void* m_data, void* m_scale,
    void* v_row, void* v_col, void* v_rms,
    const void* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    float eco_alpha,
    int stochastic_rounding, int error_feedback,
    unsigned int prng_seed0, unsigned int prng_seed1,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_adam_nvfp4_fv_update(
        reinterpret_cast<uint8_t*>(W_packed),
        reinterpret_cast<uint8_t*>(W_scale),
        reinterpret_cast<float*>(W_gs),
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<float*>(v_row),
        reinterpret_cast<float*>(v_col),
        reinterpret_cast<float*>(v_rms),
        reinterpret_cast<const float*>(grad),
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding != 0, error_feedback != 0,
        prng_seed0, prng_seed1,
        stream);
}

extern "C" cudaError_t eco_adam_nvfp4_fv_update_bf16(
    void* W_packed, void* W_scale, void* W_gs,
    void* m_data, void* m_scale,
    void* v_row, void* v_col, void* v_rms,
    const void* grad,
    int E, int out_dim, int in_dim, int group_size,
    float lr, float beta1, float beta2,
    float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    float eco_alpha,
    int stochastic_rounding, int error_feedback,
    unsigned int prng_seed0, unsigned int prng_seed1,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_adam_nvfp4_fv_update_bf16(
        reinterpret_cast<uint8_t*>(W_packed),
        reinterpret_cast<uint8_t*>(W_scale),
        reinterpret_cast<float*>(W_gs),
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<float*>(v_row),
        reinterpret_cast<float*>(v_col),
        reinterpret_cast<float*>(v_rms),
        reinterpret_cast<const __nv_bfloat16*>(grad),
        E, out_dim, in_dim, group_size,
        lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
        stochastic_rounding != 0, error_feedback != 0,
        prng_seed0, prng_seed1,
        stream);
}

// AdamA m/v accumulation: update FP8 m/v with fractional betas (non-factored v)
extern "C" cudaError_t eco_mv_accumulate(
    void* m_data, void* m_scale,
    void* v_data, void* v_scale,
    const void* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_mv_accumulate(
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<uint8_t*>(v_data),
        reinterpret_cast<float*>(v_scale),
        reinterpret_cast<const float*>(grad),
        E, in_dim, out_dim,
        beta1_frac, beta2_frac, stream);
}

extern "C" cudaError_t eco_mv_accumulate_bf16(
    void* m_data, void* m_scale,
    void* v_data, void* v_scale,
    const void* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_mv_accumulate_bf16(
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<uint8_t*>(v_data),
        reinterpret_cast<float*>(v_scale),
        reinterpret_cast<const __nv_bfloat16*>(grad),
        E, in_dim, out_dim,
        beta1_frac, beta2_frac, stream);
}

// AdamA m/v accumulation: update FP8 m + factored v with fractional betas
extern "C" cudaError_t eco_mv_accumulate_fv(
    void* m_data, void* m_scale,
    void* v_row, void* v_col, void* v_rms,
    const void* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_mv_accumulate_fv(
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<float*>(v_row),
        reinterpret_cast<float*>(v_col),
        reinterpret_cast<float*>(v_rms),
        reinterpret_cast<const float*>(grad),
        E, in_dim, out_dim,
        beta1_frac, beta2_frac, stream);
}

extern "C" cudaError_t eco_mv_accumulate_fv_bf16(
    void* m_data, void* m_scale,
    void* v_row, void* v_col, void* v_rms,
    const void* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_eco_mv_accumulate_fv_bf16(
        reinterpret_cast<uint8_t*>(m_data),
        reinterpret_cast<float*>(m_scale),
        reinterpret_cast<float*>(v_row),
        reinterpret_cast<float*>(v_col),
        reinterpret_cast<float*>(v_rms),
        reinterpret_cast<const __nv_bfloat16*>(grad),
        E, in_dim, out_dim,
        beta1_frac, beta2_frac, stream);
}

// NVFP4 per-group scale recomputation entry point
extern "C" cudaError_t nvfp4_recompute_group_scales(
    void* W_packed, void* W_scale, const void* W_gs,
    int E, int out_dim, int in_dim, int group_size,
    cudaStream_t stream)
{
    return nmoe::eco_adam::launch_nvfp4_recompute_group_scales(
        reinterpret_cast<uint8_t*>(W_packed),
        reinterpret_cast<uint8_t*>(W_scale),
        reinterpret_cast<const float*>(W_gs),
        E, out_dim, in_dim, group_size,
        stream);
}
