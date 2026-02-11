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

__global__ void k_factored_v_row(
    const float* __restrict__ grad,    // [E, in_dim, out_dim] nmoe layout
    float* __restrict__ v_row,         // [E, in_dim]
    float* __restrict__ v_rms,         // [E] (output: mean of v_row per expert)
    int E,
    int in_dim,
    int out_dim,
    float beta2)
{
    // One thread per (expert, row) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = E * in_dim;
    if (idx >= total) return;

    const int e = idx / in_dim;
    const int i = idx % in_dim;

    const float one_minus_b2 = 1.0f - beta2;

    // Sum grad^2 across out_dim for row i of expert e
    const int64_t row_base = static_cast<int64_t>(e) * in_dim * out_dim
                           + static_cast<int64_t>(i) * out_dim;
    float sum_sq = 0.0f;
    for (int j = 0; j < out_dim; ++j) {
        float g = grad[row_base + j];
        sum_sq += g * g;
    }
    float grad_sq_row_mean = sum_sq / static_cast<float>(out_dim);

    // EMA update: v_row[e, i] = beta2 * v_row[e, i] + (1 - beta2) * mean
    const int64_t vr_idx = static_cast<int64_t>(e) * in_dim + i;
    float vr = beta2 * v_row[vr_idx] + one_minus_b2 * grad_sq_row_mean;
    v_row[vr_idx] = vr;

    // Accumulate into v_rms[e] via atomicAdd (will divide by in_dim later)
    atomicAdd(&v_rms[e], vr);
}

__global__ void k_factored_v_col(
    const float* __restrict__ grad,    // [E, in_dim, out_dim] nmoe layout
    float* __restrict__ v_col,         // [E, out_dim]
    int E,
    int in_dim,
    int out_dim,
    float beta2)
{
    // One thread per (expert, col) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = E * out_dim;
    if (idx >= total) return;

    const int e = idx / out_dim;
    const int j = idx % out_dim;

    const float one_minus_b2 = 1.0f - beta2;

    // Sum grad^2 across in_dim for column j of expert e
    const int64_t expert_base = static_cast<int64_t>(e) * in_dim * out_dim;
    float sum_sq = 0.0f;
    for (int i = 0; i < in_dim; ++i) {
        float g = grad[expert_base + static_cast<int64_t>(i) * out_dim + j];
        sum_sq += g * g;
    }
    float grad_sq_col_mean = sum_sq / static_cast<float>(in_dim);

    // EMA update: v_col[e, j] = beta2 * v_col[e, j] + (1 - beta2) * mean
    const int64_t vc_idx = static_cast<int64_t>(e) * out_dim + j;
    v_col[vc_idx] = beta2 * v_col[vc_idx] + one_minus_b2 * grad_sq_col_mean;
}

// Finalize v_rms: divide accumulated sum by in_dim to get mean
__global__ void k_factored_v_rms_finalize(
    float* __restrict__ v_rms,  // [E]
    int E,
    int in_dim)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    v_rms[e] = fmaxf(v_rms[e] / static_cast<float>(in_dim), 1e-30f);
}

// Launch wrapper for all factored-v reduction sub-kernels.
inline cudaError_t launch_factored_v_update(
    const float* grad,
    float* v_row, float* v_col, float* v_rms,
    int E, int in_dim, int out_dim,
    float beta2,
    cudaStream_t stream)
{
    // Zero v_rms before atomicAdd accumulation
    auto err = cudaMemsetAsync(v_rms, 0, E * sizeof(float), stream);
    if (err != cudaSuccess) return err;

    constexpr int THREADS = 256;

    // Sub-kernel 1: row means + v_row update + v_rms accumulation
    {
        const int total = E * in_dim;
        const dim3 grid(ceil_div(total, THREADS));
        k_factored_v_row<<<grid, THREADS, 0, stream>>>(
            grad, v_row, v_rms, E, in_dim, out_dim, beta2);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Sub-kernel 2: column means + v_col update
    {
        const int total = E * out_dim;
        const dim3 grid(ceil_div(total, THREADS));
        k_factored_v_col<<<grid, THREADS, 0, stream>>>(
            grad, v_col, E, in_dim, out_dim, beta2);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Sub-kernel 3: finalize v_rms (divide by in_dim)
    {
        const dim3 grid(ceil_div(E, THREADS));
        k_factored_v_rms_finalize<<<grid, THREADS, 0, stream>>>(
            v_rms, E, in_dim);
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

template <bool kStochasticRounding, bool kErrorFeedback, bool kFactoredV = false>
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

    // Gradient (nmoe layout, FP32)
    const float* __restrict__ grad,       // [E, in_dim, out_dim]

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

    // Strides for layout indexing
    const int64_t hf_stride_e = static_cast<int64_t>(out_dim) * in_dim;       // per expert in HF
    const int64_t nmoe_stride_e = static_cast<int64_t>(in_dim) * out_dim;     // per expert in nmoe
    const int64_t hf_packed_stride_e = static_cast<int64_t>(out_dim) * (in_dim / 2);
    const int64_t hf_scale_stride_e = static_cast<int64_t>(out_dim) * (in_dim / group_size);

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
    __shared__ float sh_w[TILE_OUT][TILE_IN];      // Updated weights for requant
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
                                 + static_cast<int64_t>(out_row) * (in_dim / group_size)
                                 + (in_col / group_size);
        float group_sc = ptx::e4m3_byte_to_f32(W_scale[wscale_idx]);
        float w = w_raw * group_sc * inv_gs;

        // --- Load gradient (nmoe layout: [e, in_col, out_row]) ---
        const int64_t grad_idx = static_cast<int64_t>(e) * nmoe_stride_e
                               + static_cast<int64_t>(in_col) * out_dim
                               + out_row;
        float g = grad[grad_idx];

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
            float vrms = v_rms[e];  // already clamped to >= 1e-30
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
    // Phase 2: Compute E8M0 block scale for NVFP4 requantization
    // ============================================================
    // Each row of 32 elements in the in_dim direction = one SF_VEC block.
    // threadIdx.y == 0 computes amax.
    __shared__ uint8_t sh_e8m0[TILE_OUT];
    __shared__ float sh_block_scale[TILE_OUT];

    if (in_iter == 0) {
        float amax = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_IN; ++k) {
            amax = fmaxf(amax, fabsf(sh_w[out_local][k]));
        }
        float scale_f = amax / NVFP4_MAX;
        if (!(scale_f > 0.0f)) scale_f = 1.17549435e-38f;
        uint8_t scale_byte = ptx::e8m0_encode_from_pos_f32(scale_f);
        sh_e8m0[out_local] = scale_byte;
        sh_block_scale[out_local] = ptx::e8m0_decode_to_f32(scale_byte);
    }

    __syncthreads();

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
        float block_scale = sh_block_scale[out_local];
        float inv_scale = ptx::e8m0_inv_decode_to_f32(sh_e8m0[out_local]);

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
        float w_hat = ptx::e2m1_nibble_to_f32(nibble) * block_scale;

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

        // --- Write NVFP4 per-group scale ---
        // Each group of group_size consecutive elements shares one E4M3 scale.
        // The block scale is the E8M0 scale converted to E4M3 for storage.
        if ((in_col % group_size) == 0) {
            const int64_t wscale_idx = static_cast<int64_t>(e) * hf_scale_stride_e
                                     + static_cast<int64_t>(out_row) * (in_dim / group_size)
                                     + (in_col / group_size);
            W_scale[wscale_idx] = ptx::f32_to_e4m3_byte(block_scale);
        }

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
// Grid: one thread per row = (E * in_dim) threads, each handles out_dim elements.
// This is fast because it's just O(E * in_dim * out_dim) scalar ops with no
// inter-thread dependencies.

template <bool kIsE5M2>
__global__ void k_fp8_recompute_row_scale(
    uint8_t* __restrict__ data,       // [E, in_dim, out_dim]
    float* __restrict__ row_scale,    // [E, in_dim, 1]
    int E,
    int in_dim,
    int out_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rows = E * in_dim;
    if (row >= total_rows) return;

    constexpr float FP8_MAX = kIsE5M2 ? 57344.0f : 448.0f;

    const float old_scale = row_scale[row];
    const int64_t row_base = static_cast<int64_t>(row) * out_dim;

    // Pass 1: dequant all elements, find new amax
    float amax = 0.0f;
    for (int j = 0; j < out_dim; ++j) {
        uint8_t byte = data[row_base + j];
        float val;
        if constexpr (kIsE5M2) {
            val = ptx::e5m2_byte_to_f32(byte) * old_scale;
        } else {
            val = ptx::e4m3_byte_to_f32(byte) * old_scale;
        }
        amax = fmaxf(amax, fabsf(val));
    }

    float new_scale = amax / FP8_MAX;
    new_scale = fmaxf(new_scale, 1e-30f);
    row_scale[row] = new_scale;

    // Pass 2: rescale all elements with new scale
    float ratio = old_scale / new_scale;
    for (int j = 0; j < out_dim; ++j) {
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


// ============================================================================
// Launch wrapper (non-factored-v — backward compatible)
// ============================================================================

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
    if ((out_dim & 31) != 0) return cudaErrorInvalidValue;
    if ((in_dim & 31) != 0) return cudaErrorInvalidValue;

    // Main ECO Adam kernel (kFactoredV=false)
    const dim3 block(THREADS_X, THREADS_Y, 1);
    const dim3 grid(ceil_div(out_dim, TILE_OUT), ceil_div(in_dim, TILE_IN), E);

    // v_row/v_col/v_rms are nullptr for the non-factored path
    const float* v_row_null = nullptr;
    const float* v_col_null = nullptr;
    const float* v_rms_null = nullptr;

    if (stochastic_rounding && error_feedback) {
        k_eco_adam_nvfp4_update<true, true, false><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (stochastic_rounding) {
        k_eco_adam_nvfp4_update<true, false, false><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (error_feedback) {
        k_eco_adam_nvfp4_update<false, true, false><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else {
        k_eco_adam_nvfp4_update<false, false, false><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data, v_scale_f32,
            v_row_null, v_col_null, v_rms_null,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation pass for m and v
    const int total_rows = E * in_dim;
    constexpr int RECOMP_THREADS = 256;
    const dim3 recomp_grid(ceil_div(total_rows, RECOMP_THREADS));
    const dim3 recomp_block(RECOMP_THREADS);

    k_fp8_recompute_row_scale<true><<<recomp_grid, recomp_block, 0, stream>>>(
        m_data, m_scale_f32, E, in_dim, out_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    k_fp8_recompute_row_scale<false><<<recomp_grid, recomp_block, 0, stream>>>(
        v_data, v_scale_f32, E, in_dim, out_dim);
    return cudaGetLastError();
}

// ============================================================================
// Launch wrapper (factored-v path)
// ============================================================================

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
    if ((out_dim & 31) != 0) return cudaErrorInvalidValue;
    if ((in_dim & 31) != 0) return cudaErrorInvalidValue;

    // Step 1: Run factored-v reduction kernels (updates v_row, v_col, v_rms)
    auto err = launch_factored_v_update(grad, v_row, v_col, v_rms,
                                        E, in_dim, out_dim, beta2, stream);
    if (err != cudaSuccess) return err;

    // Step 2: Main ECO Adam kernel (kFactoredV=true)
    const dim3 block(THREADS_X, THREADS_Y, 1);
    const dim3 grid(ceil_div(out_dim, TILE_OUT), ceil_div(in_dim, TILE_IN), E);

    // v_data/v_scale are nullptr for the factored path
    uint8_t* v_data_null = nullptr;
    float* v_scale_null = nullptr;

    if (stochastic_rounding && error_feedback) {
        k_eco_adam_nvfp4_update<true, true, true><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (stochastic_rounding) {
        k_eco_adam_nvfp4_update<true, false, true><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else if (error_feedback) {
        k_eco_adam_nvfp4_update<false, true, true><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    } else {
        k_eco_adam_nvfp4_update<false, false, true><<<grid, block, 0, stream>>>(
            W_packed, W_scale, W_gs, m_data, m_scale_f32, v_data_null, v_scale_null,
            v_row, v_col, v_rms,
            grad, E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps, step_size, inv_bc2_sqrt, eco_alpha,
            prng_seed0, prng_seed1);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation pass for m only (v is factored, no FP8 v)
    const int total_rows = E * in_dim;
    constexpr int RECOMP_THREADS = 256;
    const dim3 recomp_grid(ceil_div(total_rows, RECOMP_THREADS));
    const dim3 recomp_block(RECOMP_THREADS);

    k_fp8_recompute_row_scale<true><<<recomp_grid, recomp_block, 0, stream>>>(
        m_data, m_scale_f32, E, in_dim, out_dim);
    return cudaGetLastError();
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
template <bool kFactoredV>
__global__ void k_eco_mv_accumulate(
    // FP8 optimizer states (nmoe layout: [E, in_dim, out_dim])
    uint8_t* __restrict__ m_data,         // [E, in_dim, out_dim] E5M2
    float* __restrict__ m_scale,          // [E * in_dim] per-row FP32 scale
    uint8_t* __restrict__ v_data,         // [E, in_dim, out_dim] E4M3 (kFactoredV=false only)
    float* __restrict__ v_scale,          // [E * in_dim] per-row (kFactoredV=false only)

    // Gradient (nmoe layout, FP32)
    const float* __restrict__ grad,       // [E, in_dim, out_dim]

    // Dimensions
    int E,
    int in_dim,
    int out_dim,

    // Fractional betas: β^(1/K)
    float beta1_frac,
    float beta2_frac)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(E) * in_dim * out_dim;
    if (idx >= total) return;

    // Compute row index for per-row scale: row = idx / out_dim
    const int row = idx / out_dim;

    const float one_minus_b1 = 1.0f - beta1_frac;
    const float one_minus_b2 = 1.0f - beta2_frac;

    // Load gradient
    float g = grad[idx];

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
#else
    (void)m_data; (void)m_scale; (void)v_data; (void)v_scale;
    (void)grad; (void)E; (void)in_dim; (void)out_dim;
    (void)beta1_frac; (void)beta2_frac;
    __trap();
#endif
}


// Launch wrapper for AdamA m/v accumulation (non-factored v)
inline cudaError_t launch_eco_mv_accumulate(
    uint8_t* m_data, float* m_scale,
    uint8_t* v_data, float* v_scale,
    const float* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    const int64_t total = static_cast<int64_t>(E) * in_dim * out_dim;
    constexpr int THREADS = 256;
    const dim3 grid(ceil_div(static_cast<int>(total), THREADS));
    const dim3 block(THREADS);

    k_eco_mv_accumulate<false><<<grid, block, 0, stream>>>(
        m_data, m_scale, v_data, v_scale, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation for m and v
    const int total_rows = E * in_dim;
    constexpr int RECOMP_THREADS = 256;
    const dim3 recomp_grid(ceil_div(total_rows, RECOMP_THREADS));

    k_fp8_recompute_row_scale<true><<<recomp_grid, RECOMP_THREADS, 0, stream>>>(
        m_data, m_scale, E, in_dim, out_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    k_fp8_recompute_row_scale<false><<<recomp_grid, RECOMP_THREADS, 0, stream>>>(
        v_data, v_scale, E, in_dim, out_dim);
    return cudaGetLastError();
}


// Launch wrapper for AdamA m/v accumulation (factored-v)
inline cudaError_t launch_eco_mv_accumulate_fv(
    uint8_t* m_data, float* m_scale,
    float* v_row, float* v_col, float* v_rms,
    const float* grad,
    int E, int in_dim, int out_dim,
    float beta1_frac, float beta2_frac,
    cudaStream_t stream)
{
    // Step 1: Run factored-v reduction kernels to update v_row, v_col, v_rms
    // (these use beta2_frac instead of beta2)
    auto err = launch_factored_v_update(grad, v_row, v_col, v_rms,
                                        E, in_dim, out_dim, beta2_frac, stream);
    if (err != cudaSuccess) return err;

    // Step 2: Update m only (kFactoredV=true skips v_data update)
    const int64_t total = static_cast<int64_t>(E) * in_dim * out_dim;
    constexpr int THREADS = 256;
    const dim3 grid(ceil_div(static_cast<int>(total), THREADS));

    k_eco_mv_accumulate<true><<<grid, THREADS, 0, stream>>>(
        m_data, m_scale, nullptr, nullptr, grad,
        E, in_dim, out_dim, beta1_frac, beta2_frac);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // FP8 scale recomputation for m only (v is factored)
    const int total_rows = E * in_dim;
    constexpr int RECOMP_THREADS = 256;
    const dim3 recomp_grid(ceil_div(total_rows, RECOMP_THREADS));

    k_fp8_recompute_row_scale<true><<<recomp_grid, RECOMP_THREADS, 0, stream>>>(
        m_data, m_scale, E, in_dim, out_dim);
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
