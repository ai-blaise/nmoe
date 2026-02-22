// nmoe/csrc/adamw_fused_step.cu
//
// Fused AdamW optimizer step for ZeRO-2 dense parameter shards.
//
// Replaces 9 separate PyTorch kernel launches per chunk with a single
// vectorized fused kernel.  Supports FP32 (float4 vectorized) and BF16
// (__nv_bfloat162 vectorized) paths.  The Python side selects the
// appropriate C entry point based on tensor dtype.
//
// Target: sm_80+ (Ampere and later for BF16), sm_70+ for FP32-only.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace nmoe {
namespace dense_adamw {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

constexpr int BLOCK = 256;

// ---------------------------------------------------------------------------
// Scalar kernel -- used for tail elements or as a fallback
// ---------------------------------------------------------------------------
template <typename T>
__global__ void k_fused_adamw_scalar(
    T*       __restrict__ param,
    const T* __restrict__ grad,
    T*       __restrict__ exp_avg,
    T*       __restrict__ exp_avg_sq,
    int N,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bc2_sqrt);

// FP32 specialization (trivial: no conversion needed)
template <>
__global__ void k_fused_adamw_scalar<float>(
    float*       __restrict__ param,
    const float* __restrict__ grad,
    float*       __restrict__ exp_avg,
    float*       __restrict__ exp_avg_sq,
    int N,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bc2_sqrt)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  float p = param[idx];
  float g = grad[idx];
  float m = exp_avg[idx];
  float v = exp_avg_sq[idx];

  // Decoupled weight decay (applied before moment update, matching PyTorch)
  if (weight_decay != 0.0f) {
    p *= (1.0f - lr * weight_decay);
  }

  // Moment updates
  m = beta1 * m + (1.0f - beta1) * g;
  v = beta2 * v + (1.0f - beta2) * g * g;

  // Bias-corrected parameter update
  float denom = sqrtf(v) * inv_bc2_sqrt + eps;
  p -= step_size * m / denom;

  param[idx]      = p;
  exp_avg[idx]    = m;
  exp_avg_sq[idx] = v;
}

// BF16 specialization (convert to FP32 for arithmetic, store back as BF16)
template <>
__global__ void k_fused_adamw_scalar<__nv_bfloat16>(
    __nv_bfloat16*       __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16*       __restrict__ exp_avg,
    __nv_bfloat16*       __restrict__ exp_avg_sq,
    int N,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bc2_sqrt)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  float p = __bfloat162float(param[idx]);
  float g = __bfloat162float(grad[idx]);
  float m = __bfloat162float(exp_avg[idx]);
  float v = __bfloat162float(exp_avg_sq[idx]);

  if (weight_decay != 0.0f) {
    p *= (1.0f - lr * weight_decay);
  }

  m = beta1 * m + (1.0f - beta1) * g;
  v = beta2 * v + (1.0f - beta2) * g * g;

  float denom = sqrtf(v) * inv_bc2_sqrt + eps;
  p -= step_size * m / denom;

  param[idx]      = __float2bfloat16(p);
  exp_avg[idx]    = __float2bfloat16(m);
  exp_avg_sq[idx] = __float2bfloat16(v);
#endif
}

// ---------------------------------------------------------------------------
// Vectorized FP32 kernel -- float4 (4 elements per thread)
// ---------------------------------------------------------------------------
__global__ void k_fused_adamw_fp32_vec4(
    float*       __restrict__ param,
    const float* __restrict__ grad,
    float*       __restrict__ exp_avg,
    float*       __restrict__ exp_avg_sq,
    int N4,    // number of float4 groups = N / 4
    int tail,  // leftover scalar elements (N % 4)
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bc2_sqrt)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N4 + tail) return;

  const float decay_factor = (weight_decay != 0.0f) ? (1.0f - lr * weight_decay) : 1.0f;
  const float one_m_b1 = 1.0f - beta1;
  const float one_m_b2 = 1.0f - beta2;

  if (idx < N4) {
    // Vectorized loads (128-bit aligned)
    float4 p4 = reinterpret_cast<float4*>(param)[idx];
    float4 g4 = reinterpret_cast<const float4*>(grad)[idx];
    float4 m4 = reinterpret_cast<float4*>(exp_avg)[idx];
    float4 v4 = reinterpret_cast<float4*>(exp_avg_sq)[idx];

    // Component 0
    {
      float p = p4.x * decay_factor;
      float g = g4.x;
      float m = beta1 * m4.x + one_m_b1 * g;
      float v = beta2 * v4.x + one_m_b2 * g * g;
      p -= step_size * m / (sqrtf(v) * inv_bc2_sqrt + eps);
      p4.x = p;  m4.x = m;  v4.x = v;
    }
    // Component 1
    {
      float p = p4.y * decay_factor;
      float g = g4.y;
      float m = beta1 * m4.y + one_m_b1 * g;
      float v = beta2 * v4.y + one_m_b2 * g * g;
      p -= step_size * m / (sqrtf(v) * inv_bc2_sqrt + eps);
      p4.y = p;  m4.y = m;  v4.y = v;
    }
    // Component 2
    {
      float p = p4.z * decay_factor;
      float g = g4.z;
      float m = beta1 * m4.z + one_m_b1 * g;
      float v = beta2 * v4.z + one_m_b2 * g * g;
      p -= step_size * m / (sqrtf(v) * inv_bc2_sqrt + eps);
      p4.z = p;  m4.z = m;  v4.z = v;
    }
    // Component 3
    {
      float p = p4.w * decay_factor;
      float g = g4.w;
      float m = beta1 * m4.w + one_m_b1 * g;
      float v = beta2 * v4.w + one_m_b2 * g * g;
      p -= step_size * m / (sqrtf(v) * inv_bc2_sqrt + eps);
      p4.w = p;  m4.w = m;  v4.w = v;
    }

    // Vectorized stores
    reinterpret_cast<float4*>(param)[idx]      = p4;
    reinterpret_cast<float4*>(exp_avg)[idx]    = m4;
    reinterpret_cast<float4*>(exp_avg_sq)[idx] = v4;
  } else {
    // Tail idx only when N % 4 != 0; keep arithmetic identical to scalar path.
    const int tail_idx = N4 * 4 + (idx - N4);
    float p = param[tail_idx] * decay_factor;
    float g = grad[tail_idx];
    float m = beta1 * exp_avg[tail_idx] + one_m_b1 * g;
    float v = beta2 * exp_avg_sq[tail_idx] + one_m_b2 * g * g;
    p -= step_size * m / (sqrtf(v) * inv_bc2_sqrt + eps);

    param[tail_idx]      = p;
    exp_avg[tail_idx]    = m;
    exp_avg_sq[tail_idx] = v;
  }
}

// ---------------------------------------------------------------------------
// Vectorized BF16 kernel -- __nv_bfloat162 (2 elements per thread)
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
// Device-only helper: update a pair of BF16 values in FP32 arithmetic.
__device__ __forceinline__ void adamw_bf162_update(
    __nv_bfloat162& p2,
    __nv_bfloat162  g2,
    __nv_bfloat162& m2,
    __nv_bfloat162& v2,
    float decay_factor,
    float beta1, float one_m_b1,
    float beta2, float one_m_b2,
    float step_size, float inv_bc2_sqrt, float eps)
{
  float p_lo = __bfloat162float(__low2bfloat16(p2));
  float p_hi = __bfloat162float(__high2bfloat16(p2));
  float g_lo = __bfloat162float(__low2bfloat16(g2));
  float g_hi = __bfloat162float(__high2bfloat16(g2));
  float m_lo = __bfloat162float(__low2bfloat16(m2));
  float m_hi = __bfloat162float(__high2bfloat16(m2));
  float v_lo = __bfloat162float(__low2bfloat16(v2));
  float v_hi = __bfloat162float(__high2bfloat16(v2));

  p_lo *= decay_factor;
  p_hi *= decay_factor;

  m_lo = beta1 * m_lo + one_m_b1 * g_lo;
  m_hi = beta1 * m_hi + one_m_b1 * g_hi;
  v_lo = beta2 * v_lo + one_m_b2 * g_lo * g_lo;
  v_hi = beta2 * v_hi + one_m_b2 * g_hi * g_hi;

  p_lo -= step_size * m_lo / (sqrtf(v_lo) * inv_bc2_sqrt + eps);
  p_hi -= step_size * m_hi / (sqrtf(v_hi) * inv_bc2_sqrt + eps);

  p2 = __halves2bfloat162(__float2bfloat16(p_lo), __float2bfloat16(p_hi));
  m2 = __halves2bfloat162(__float2bfloat16(m_lo), __float2bfloat16(m_hi));
  v2 = __halves2bfloat162(__float2bfloat16(v_lo), __float2bfloat16(v_hi));
}
#endif

__global__ void k_fused_adamw_bf16_vec2(
    __nv_bfloat16*       __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16*       __restrict__ exp_avg,
    __nv_bfloat16*       __restrict__ exp_avg_sq,
    int N2,    // number of bfloat162 pairs = N / 2
    int tail,  // leftover scalar elements (N % 2)
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step_size,
    float inv_bc2_sqrt)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N2 + tail) return;

  const float decay_factor = (weight_decay != 0.0f) ? (1.0f - lr * weight_decay) : 1.0f;
  const float one_m_b1 = 1.0f - beta1;
  const float one_m_b2 = 1.0f - beta2;

  if (idx < N2) {
    __nv_bfloat162 p2 = reinterpret_cast<__nv_bfloat162*>(param)[idx];
    __nv_bfloat162 g2 = reinterpret_cast<const __nv_bfloat162*>(grad)[idx];
    __nv_bfloat162 m2 = reinterpret_cast<__nv_bfloat162*>(exp_avg)[idx];
    __nv_bfloat162 v2 = reinterpret_cast<__nv_bfloat162*>(exp_avg_sq)[idx];

    adamw_bf162_update(p2, g2, m2, v2,
                       decay_factor, beta1, one_m_b1, beta2, one_m_b2,
                       step_size, inv_bc2_sqrt, eps);

    reinterpret_cast<__nv_bfloat162*>(param)[idx]      = p2;
    reinterpret_cast<__nv_bfloat162*>(exp_avg)[idx]    = m2;
    reinterpret_cast<__nv_bfloat162*>(exp_avg_sq)[idx] = v2;
  } else {
    // Tail idx only when N is odd; keep arithmetic identical to scalar BF16 path.
    const int tail_idx = N2 * 2;
    float p = __bfloat162float(param[tail_idx]);
    float g = __bfloat162float(grad[tail_idx]);
    float m = __bfloat162float(exp_avg[tail_idx]);
    float v = __bfloat162float(exp_avg_sq[tail_idx]);

    p *= decay_factor;
    m = beta1 * m + one_m_b1 * g;
    v = beta2 * v + one_m_b2 * g * g;
    p -= step_size * m / (sqrtf(v) * inv_bc2_sqrt + eps);

    param[tail_idx]      = __float2bfloat16(p);
    exp_avg[tail_idx]    = __float2bfloat16(m);
    exp_avg_sq[tail_idx] = __float2bfloat16(v);
  }
#endif
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

inline cudaError_t launch_fused_adamw_fp32(
    float*       param,
    const float* grad,
    float*       exp_avg,
    float*       exp_avg_sq,
    int N,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    cudaStream_t stream)
{
  if (N < 0) return cudaErrorInvalidValue;
  if (N == 0) return cudaSuccess;

  // Vectorized path: process groups of 4 via float4.
  // Tail elements (N % 4 != 0) are handled in-kernel to avoid a second launch.
  const int N4   = N / 4;
  const int tail = N - N4 * 4;
  const int total = N4 + tail;
  if (total > 0) {
    const int grid = ceil_div(total, BLOCK);
    k_fused_adamw_fp32_vec4<<<grid, BLOCK, 0, stream>>>(
        param, grad, exp_avg, exp_avg_sq,
        N4, tail, beta1, beta2, lr, weight_decay, eps,
        step_size, inv_bc2_sqrt);
  }

  return cudaGetLastError();
}

inline cudaError_t launch_fused_adamw_bf16(
    __nv_bfloat16*       param,
    const __nv_bfloat16* grad,
    __nv_bfloat16*       exp_avg,
    __nv_bfloat16*       exp_avg_sq,
    int N,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    cudaStream_t stream)
{
  if (N < 0) return cudaErrorInvalidValue;
  if (N == 0) return cudaSuccess;

  // Vectorized path: process pairs via __nv_bfloat162.
  const int N2   = N / 2;
  const int tail = N - N2 * 2;

  const int total = N2 + tail;
  if (total > 0) {
    const int grid = ceil_div(total, BLOCK);
    k_fused_adamw_bf16_vec2<<<grid, BLOCK, 0, stream>>>(
        param, grad, exp_avg, exp_avg_sq,
        N2, tail, beta1, beta2, lr, weight_decay, eps,
        step_size, inv_bc2_sqrt);
  }

  return cudaGetLastError();
}

inline cudaError_t current_device_supports_bf16(bool* supports_out) {
  if (supports_out == nullptr) return cudaErrorInvalidValue;
  static thread_local int cached_dev = -2;
  static thread_local bool cached_support = false;
  int dev = -1;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) {
    return err;
  }
  if (dev == cached_dev) {
    *supports_out = cached_support;
    return cudaSuccess;
  }
  int cc_major = 0;
  err = cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
  if (err != cudaSuccess) {
    return err;
  }
  cached_dev = dev;
  cached_support = (cc_major >= 8);
  *supports_out = cached_support;
  return cudaSuccess;
}

}  // namespace dense_adamw
}  // namespace nmoe

// ===========================================================================
// C API wrappers (callable from pybind11 bindings)
// ===========================================================================

extern "C" cudaError_t fused_adamw_step_fp32(
    void* param,
    const void* grad,
    void* exp_avg,
    void* exp_avg_sq,
    int N,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    cudaStream_t stream)
{
  if (N < 0) {
    return cudaErrorInvalidValue;
  }
  if (N > 0 && (param == nullptr || grad == nullptr || exp_avg == nullptr || exp_avg_sq == nullptr)) {
    return cudaErrorInvalidValue;
  }
  return nmoe::dense_adamw::launch_fused_adamw_fp32(
      static_cast<float*>(param),
      static_cast<const float*>(grad),
      static_cast<float*>(exp_avg),
      static_cast<float*>(exp_avg_sq),
      N, beta1, beta2, lr, weight_decay, eps,
      step_size, inv_bc2_sqrt, stream);
}

extern "C" cudaError_t fused_adamw_step_bf16(
    void* param,
    const void* grad,
    void* exp_avg,
    void* exp_avg_sq,
    int N,
    float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float step_size, float inv_bc2_sqrt,
    cudaStream_t stream)
{
  if (N < 0) {
    return cudaErrorInvalidValue;
  }
  if (N > 0 && (param == nullptr || grad == nullptr || exp_avg == nullptr || exp_avg_sq == nullptr)) {
    return cudaErrorInvalidValue;
  }
  bool supports_bf16 = false;
  cudaError_t probe = nmoe::dense_adamw::current_device_supports_bf16(&supports_bf16);
  if (probe != cudaSuccess) {
    return probe;
  }
  if (!supports_bf16) {
    return cudaErrorInvalidDeviceFunction;
  }
  return nmoe::dense_adamw::launch_fused_adamw_bf16(
      static_cast<__nv_bfloat16*>(param),
      static_cast<const __nv_bfloat16*>(grad),
      static_cast<__nv_bfloat16*>(exp_avg),
      static_cast<__nv_bfloat16*>(exp_avg_sq),
      N, beta1, beta2, lr, weight_decay, eps,
      step_size, inv_bc2_sqrt, stream);
}
