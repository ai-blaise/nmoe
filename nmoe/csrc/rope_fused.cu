// Fused Rotary Position Embedding (RoPE) CUDA kernel
//
// Replaces ~8 separate PyTorch kernel launches (slice, mul, sub, mul, mul, add,
// unsqueeze x2) with a single fused kernel. Called 122 times per forward pass
// = 854 unnecessary kernel launches eliminated.
//
// Forward:  out[..., :half] = x1*cos - x2*sin
//           out[..., half:] = x2*cos + x1*sin
//
// Backward: grad_x[..., :half] =  grad_out1*cos + grad_out2*sin
//           grad_x[..., half:] = -grad_out1*sin + grad_out2*cos
//
// All computation in FP32 for numerical stability; BF16 I/O for memory bandwidth.
// Uses __nv_bfloat162 vectorized loads/stores (2 elements per thread) where the
// half_dim is even, falling back to scalar for odd half_dim.
//
// Requires sm_80+ (Ampere or later) for native BF16 support.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <climits>

namespace nmoe {
namespace rope {

// ============================================================================
// Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

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

// ============================================================================
// Forward kernel — vectorized path (2 elements per thread via bfloat162)
// ============================================================================
//
// Thread indexing:
//   Each thread handles 2 element pairs from the first/second half of head_dim.
//   Global linear index maps to (vec_idx, pair_idx) where:
//     - vec_idx  = which flattened vector (batch * seq_len * n_heads)
//     - pair_idx = which pair of elements within half_dim (processes 2 at a time)
//
// cos/sin indexing:
//   cos_buf and sin_buf are [seq_len, half_dim] in BF16.
//   seq_pos = (vec_idx / n_heads) % seq_len  handles [B, T, H, D] layout.

__global__ void k_fused_rope_forward_vec2(
    const __nv_bfloat16* __restrict__ x,        // [total_vecs, head_dim]
    const __nv_bfloat16* __restrict__ cos_buf,   // [seq_len, half_dim]
    const __nv_bfloat16* __restrict__ sin_buf,   // [seq_len, half_dim]
    __nv_bfloat16* __restrict__ out,             // [total_vecs, head_dim]
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int half_dim)                                 // head_dim / 2
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  // Each thread processes 2 consecutive elements within half_dim.
  // half_dim_pairs = half_dim / 2  (caller guarantees half_dim is even for this path)
  const int half_dim_pairs = half_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_idx = idx % half_dim_pairs;   // which pair within the half
  const int vec_idx  = idx / half_dim_pairs;   // which vector

  if (vec_idx >= total_vecs) return;

  // Compute sequence position for cos/sin lookup.
  // Layout is [B, T, H, D], so within the flattened (B*T*H) space:
  //   vec_idx = b * (seq_len * n_heads) + t * n_heads + h
  //   => t = (vec_idx / n_heads) % seq_len
  const int seq_pos = (vec_idx / n_heads) % seq_len;

  // Element offset within the vector (2 elements starting at pair_idx*2)
  const int d = pair_idx * 2;

  // Pointers into x and out for this vector
  const __nv_bfloat16* x_vec   = x   + (int64_t)vec_idx * head_dim;
  __nv_bfloat16*       out_vec = out + (int64_t)vec_idx * head_dim;

  // Vectorized load: 2 elements from first half and second half
  const __nv_bfloat162* x1_ptr = reinterpret_cast<const __nv_bfloat162*>(x_vec + d);
  const __nv_bfloat162* x2_ptr = reinterpret_cast<const __nv_bfloat162*>(x_vec + half_dim + d);
  __nv_bfloat162 x1_pair = *x1_ptr;
  __nv_bfloat162 x2_pair = *x2_ptr;

  // Load cos/sin (also vectorized)
  const __nv_bfloat162* cs_ptr = reinterpret_cast<const __nv_bfloat162*>(cos_buf + (int64_t)seq_pos * half_dim + d);
  const __nv_bfloat162* sn_ptr = reinterpret_cast<const __nv_bfloat162*>(sin_buf + (int64_t)seq_pos * half_dim + d);
  __nv_bfloat162 c_pair = *cs_ptr;
  __nv_bfloat162 s_pair = *sn_ptr;

  // Unpack to float for FP32 computation
  float x1_lo = __bfloat162float(__low2bfloat16(x1_pair));
  float x1_hi = __bfloat162float(__high2bfloat16(x1_pair));
  float x2_lo = __bfloat162float(__low2bfloat16(x2_pair));
  float x2_hi = __bfloat162float(__high2bfloat16(x2_pair));
  float c_lo  = __bfloat162float(__low2bfloat16(c_pair));
  float c_hi  = __bfloat162float(__high2bfloat16(c_pair));
  float s_lo  = __bfloat162float(__low2bfloat16(s_pair));
  float s_hi  = __bfloat162float(__high2bfloat16(s_pair));

  // Fused rotation in FP32
  float out1_lo = x1_lo * c_lo - x2_lo * s_lo;
  float out1_hi = x1_hi * c_hi - x2_hi * s_hi;
  float out2_lo = x2_lo * c_lo + x1_lo * s_lo;
  float out2_hi = x2_hi * c_hi + x1_hi * s_hi;

  // Pack back to bfloat162 and store
  __nv_bfloat162 out1_pair = __halves2bfloat162(__float2bfloat16(out1_lo), __float2bfloat16(out1_hi));
  __nv_bfloat162 out2_pair = __halves2bfloat162(__float2bfloat16(out2_lo), __float2bfloat16(out2_hi));

  __nv_bfloat162* out1_ptr = reinterpret_cast<__nv_bfloat162*>(out_vec + d);
  __nv_bfloat162* out2_ptr = reinterpret_cast<__nv_bfloat162*>(out_vec + half_dim + d);
  *out1_ptr = out1_pair;
  *out2_ptr = out2_pair;
#endif  // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Forward kernel — scalar fallback (1 element per thread)
// ============================================================================

__global__ void k_fused_rope_forward_scalar(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_buf,
    const __nv_bfloat16* __restrict__ sin_buf,
    __nv_bfloat16* __restrict__ out,
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int half_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int d       = idx % half_dim;
  const int vec_idx = idx / half_dim;

  if (vec_idx >= total_vecs) return;

  const int seq_pos = (vec_idx / n_heads) % seq_len;

  // Load
  float x1 = __bfloat162float(x[(int64_t)vec_idx * head_dim + d]);
  float x2 = __bfloat162float(x[(int64_t)vec_idx * head_dim + half_dim + d]);
  float c  = __bfloat162float(cos_buf[(int64_t)seq_pos * half_dim + d]);
  float s  = __bfloat162float(sin_buf[(int64_t)seq_pos * half_dim + d]);

  // Fused rotation in FP32
  float out1 = x1 * c - x2 * s;
  float out2 = x2 * c + x1 * s;

  // Store
  out[(int64_t)vec_idx * head_dim + d]            = __float2bfloat16(out1);
  out[(int64_t)vec_idx * head_dim + half_dim + d] = __float2bfloat16(out2);
#endif  // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Backward kernel — vectorized path
// ============================================================================
//
// RoPE backward is the transpose of the rotation matrix:
//   grad_x[..., :half] =  grad_out1 * cos + grad_out2 * sin
//   grad_x[..., half:] = -grad_out1 * sin + grad_out2 * cos
//
// (The rotation matrix R is orthogonal, so R^{-1} = R^T.)

__global__ void k_fused_rope_backward_vec2(
    const __nv_bfloat16* __restrict__ grad_out,  // [total_vecs, head_dim]
    const __nv_bfloat16* __restrict__ cos_buf,   // [seq_len, half_dim]
    const __nv_bfloat16* __restrict__ sin_buf,   // [seq_len, half_dim]
    __nv_bfloat16* __restrict__ grad_x,          // [total_vecs, head_dim]
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int half_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int half_dim_pairs = half_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_idx = idx % half_dim_pairs;
  const int vec_idx  = idx / half_dim_pairs;

  if (vec_idx >= total_vecs) return;

  const int seq_pos = (vec_idx / n_heads) % seq_len;
  const int d = pair_idx * 2;

  const __nv_bfloat16* go_vec = grad_out + (int64_t)vec_idx * head_dim;
  __nv_bfloat16*       gx_vec = grad_x   + (int64_t)vec_idx * head_dim;

  // Vectorized load
  __nv_bfloat162 go1_pair = *reinterpret_cast<const __nv_bfloat162*>(go_vec + d);
  __nv_bfloat162 go2_pair = *reinterpret_cast<const __nv_bfloat162*>(go_vec + half_dim + d);
  __nv_bfloat162 c_pair   = *reinterpret_cast<const __nv_bfloat162*>(cos_buf + (int64_t)seq_pos * half_dim + d);
  __nv_bfloat162 s_pair   = *reinterpret_cast<const __nv_bfloat162*>(sin_buf + (int64_t)seq_pos * half_dim + d);

  // Unpack
  float go1_lo = __bfloat162float(__low2bfloat16(go1_pair));
  float go1_hi = __bfloat162float(__high2bfloat16(go1_pair));
  float go2_lo = __bfloat162float(__low2bfloat16(go2_pair));
  float go2_hi = __bfloat162float(__high2bfloat16(go2_pair));
  float c_lo   = __bfloat162float(__low2bfloat16(c_pair));
  float c_hi   = __bfloat162float(__high2bfloat16(c_pair));
  float s_lo   = __bfloat162float(__low2bfloat16(s_pair));
  float s_hi   = __bfloat162float(__high2bfloat16(s_pair));

  // Backward rotation (transpose of forward rotation matrix)
  float gx1_lo =  go1_lo * c_lo + go2_lo * s_lo;
  float gx1_hi =  go1_hi * c_hi + go2_hi * s_hi;
  float gx2_lo = -go1_lo * s_lo + go2_lo * c_lo;
  float gx2_hi = -go1_hi * s_hi + go2_hi * c_hi;

  // Pack and store
  __nv_bfloat162 gx1_pair = __halves2bfloat162(__float2bfloat16(gx1_lo), __float2bfloat16(gx1_hi));
  __nv_bfloat162 gx2_pair = __halves2bfloat162(__float2bfloat16(gx2_lo), __float2bfloat16(gx2_hi));

  *reinterpret_cast<__nv_bfloat162*>(gx_vec + d)            = gx1_pair;
  *reinterpret_cast<__nv_bfloat162*>(gx_vec + half_dim + d) = gx2_pair;
#endif  // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Backward kernel — scalar fallback
// ============================================================================

__global__ void k_fused_rope_backward_scalar(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ cos_buf,
    const __nv_bfloat16* __restrict__ sin_buf,
    __nv_bfloat16* __restrict__ grad_x,
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int half_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int d       = idx % half_dim;
  const int vec_idx = idx / half_dim;

  if (vec_idx >= total_vecs) return;

  const int seq_pos = (vec_idx / n_heads) % seq_len;

  float go1 = __bfloat162float(grad_out[(int64_t)vec_idx * head_dim + d]);
  float go2 = __bfloat162float(grad_out[(int64_t)vec_idx * head_dim + half_dim + d]);
  float c   = __bfloat162float(cos_buf[(int64_t)seq_pos * half_dim + d]);
  float s   = __bfloat162float(sin_buf[(int64_t)seq_pos * half_dim + d]);

  float gx1 =  go1 * c + go2 * s;
  float gx2 = -go1 * s + go2 * c;

  grad_x[(int64_t)vec_idx * head_dim + d]            = __float2bfloat16(gx1);
  grad_x[(int64_t)vec_idx * head_dim + half_dim + d] = __float2bfloat16(gx2);
#endif  // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Partial RoPE Forward kernel — vectorized path (in-place, rope portion only)
// ============================================================================
//
// Applies RoPE only to elements [nope_dim : head_dim] within each head.
// Elements [0 : nope_dim] are left unchanged.
// Operates IN-PLACE on the input tensor.
//
// Thread indexing same as full RoPE, but offset by nope_dim.

__global__ void k_fused_rope_forward_partial_vec2(
    __nv_bfloat16* __restrict__ x,           // IN-PLACE [total_vecs, head_dim]
    const __nv_bfloat16* __restrict__ cos_buf,   // [seq_len, half_dim]
    const __nv_bfloat16* __restrict__ sin_buf,   // [seq_len, half_dim]
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int nope_dim)                                 // elements [0:nope_dim] are unchanged
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int rope_dim = head_dim - nope_dim;
  const int half_dim = rope_dim / 2;
  const int half_dim_pairs = half_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_idx = idx % half_dim_pairs;
  const int vec_idx  = idx / half_dim_pairs;

  if (vec_idx >= total_vecs) return;

  const int seq_pos = (vec_idx / n_heads) % seq_len;
  const int d = pair_idx * 2;

  // Pointer to start of this vector
  __nv_bfloat16* x_vec = x + (int64_t)vec_idx * head_dim;

  // x1 = x[nope_dim + d], x2 = x[nope_dim + half_dim + d]
  const __nv_bfloat162* x1_ptr = reinterpret_cast<const __nv_bfloat162*>(x_vec + nope_dim + d);
  const __nv_bfloat162* x2_ptr = reinterpret_cast<const __nv_bfloat162*>(x_vec + nope_dim + half_dim + d);
  __nv_bfloat162 x1_pair = *x1_ptr;
  __nv_bfloat162 x2_pair = *x2_ptr;

  // Load cos/sin
  const __nv_bfloat162* cs_ptr = reinterpret_cast<const __nv_bfloat162*>(cos_buf + (int64_t)seq_pos * half_dim + d);
  const __nv_bfloat162* sn_ptr = reinterpret_cast<const __nv_bfloat162*>(sin_buf + (int64_t)seq_pos * half_dim + d);
  __nv_bfloat162 c_pair = *cs_ptr;
  __nv_bfloat162 s_pair = *sn_ptr;

  // Unpack to float for FP32 computation
  float x1_lo = __bfloat162float(__low2bfloat16(x1_pair));
  float x1_hi = __bfloat162float(__high2bfloat16(x1_pair));
  float x2_lo = __bfloat162float(__low2bfloat16(x2_pair));
  float x2_hi = __bfloat162float(__high2bfloat16(x2_pair));
  float c_lo  = __bfloat162float(__low2bfloat16(c_pair));
  float c_hi  = __bfloat162float(__high2bfloat16(c_pair));
  float s_lo  = __bfloat162float(__low2bfloat16(s_pair));
  float s_hi  = __bfloat162float(__high2bfloat16(s_pair));

  // Fused rotation in FP32
  float out1_lo = x1_lo * c_lo - x2_lo * s_lo;
  float out1_hi = x1_hi * c_hi - x2_hi * s_hi;
  float out2_lo = x2_lo * c_lo + x1_lo * s_lo;
  float out2_hi = x2_hi * c_hi + x1_hi * s_hi;

  // Pack back to bfloat162 and store IN-PLACE
  __nv_bfloat162 out1_pair = __halves2bfloat162(__float2bfloat16(out1_lo), __float2bfloat16(out1_hi));
  __nv_bfloat162 out2_pair = __halves2bfloat162(__float2bfloat16(out2_lo), __float2bfloat16(out2_hi));

  __nv_bfloat162* out1_ptr = reinterpret_cast<__nv_bfloat162*>(x_vec + nope_dim + d);
  __nv_bfloat162* out2_ptr = reinterpret_cast<__nv_bfloat162*>(x_vec + nope_dim + half_dim + d);
  *out1_ptr = out1_pair;
  *out2_ptr = out2_pair;
#endif  // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Partial RoPE Forward kernel — scalar fallback
// ============================================================================

__global__ void k_fused_rope_forward_partial_scalar(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_buf,
    const __nv_bfloat16* __restrict__ sin_buf,
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int nope_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int rope_dim = head_dim - nope_dim;
  const int half_dim = rope_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int d       = idx % half_dim;
  const int vec_idx = idx / half_dim;

  if (vec_idx >= total_vecs) return;

  const int seq_pos = (vec_idx / n_heads) % seq_len;

  // Load from rope portion
  float x1 = __bfloat162float(x[(int64_t)vec_idx * head_dim + nope_dim + d]);
  float x2 = __bfloat162float(x[(int64_t)vec_idx * head_dim + nope_dim + half_dim + d]);
  float c  = __bfloat162float(cos_buf[(int64_t)seq_pos * half_dim + d]);
  float s  = __bfloat162float(sin_buf[(int64_t)seq_pos * half_dim + d]);

  // Fused rotation in FP32
  float out1 = x1 * c - x2 * s;
  float out2 = x2 * c + x1 * s;

  // Store IN-PLACE
  x[(int64_t)vec_idx * head_dim + nope_dim + d]            = __float2bfloat16(out1);
  x[(int64_t)vec_idx * head_dim + nope_dim + half_dim + d] = __float2bfloat16(out2);
#endif  // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Partial RoPE Backward kernel — vectorized path (in-place)
// ============================================================================

__global__ void k_fused_rope_backward_partial_vec2(
    __nv_bfloat16* __restrict__ grad_x,      // IN-PLACE [total_vecs, head_dim]
    const __nv_bfloat16* __restrict__ cos_buf,   // [seq_len, half_dim]
    const __nv_bfloat16* __restrict__ sin_buf,   // [seq_len, half_dim]
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int nope_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int rope_dim = head_dim - nope_dim;
  const int half_dim = rope_dim / 2;
  const int half_dim_pairs = half_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_idx = idx % half_dim_pairs;
  const int vec_idx  = idx / half_dim_pairs;

  if (vec_idx >= total_vecs) return;

  const int seq_pos = (vec_idx / n_heads) % seq_len;
  const int d = pair_idx * 2;

  __nv_bfloat16* gx_vec = grad_x + (int64_t)vec_idx * head_dim;

  // Vectorized load (grad_out values at rope portion)
  __nv_bfloat162 go1_pair = *reinterpret_cast<const __nv_bfloat162*>(gx_vec + nope_dim + d);
  __nv_bfloat162 go2_pair = *reinterpret_cast<const __nv_bfloat162*>(gx_vec + nope_dim + half_dim + d);
  __nv_bfloat162 c_pair   = *reinterpret_cast<const __nv_bfloat162*>(cos_buf + (int64_t)seq_pos * half_dim + d);
  __nv_bfloat162 s_pair   = *reinterpret_cast<const __nv_bfloat162*>(sin_buf + (int64_t)seq_pos * half_dim + d);

  // Unpack
  float go1_lo = __bfloat162float(__low2bfloat16(go1_pair));
  float go1_hi = __bfloat162float(__high2bfloat16(go1_pair));
  float go2_lo = __bfloat162float(__low2bfloat16(go2_pair));
  float go2_hi = __bfloat162float(__high2bfloat16(go2_pair));
  float c_lo   = __bfloat162float(__low2bfloat16(c_pair));
  float c_hi   = __bfloat162float(__high2bfloat16(c_pair));
  float s_lo   = __bfloat162float(__low2bfloat16(s_pair));
  float s_hi   = __bfloat162float(__high2bfloat16(s_pair));

  // Backward rotation (transpose of forward rotation matrix)
  float gx1_lo =  go1_lo * c_lo + go2_lo * s_lo;
  float gx1_hi =  go1_hi * c_hi + go2_hi * s_hi;
  float gx2_lo = -go1_lo * s_lo + go2_lo * c_lo;
  float gx2_hi = -go1_hi * s_hi + go2_hi * c_hi;

  // Pack and store IN-PLACE
  __nv_bfloat162 gx1_pair = __halves2bfloat162(__float2bfloat16(gx1_lo), __float2bfloat16(gx1_hi));
  __nv_bfloat162 gx2_pair = __halves2bfloat162(__float2bfloat16(gx2_lo), __float2bfloat16(gx2_hi));

  *reinterpret_cast<__nv_bfloat162*>(gx_vec + nope_dim + d)            = gx1_pair;
  *reinterpret_cast<__nv_bfloat162*>(gx_vec + nope_dim + half_dim + d) = gx2_pair;
#endif  // __CUDA_ARCH__ >= 800
}

// ============================================================================
// Partial RoPE Backward kernel — scalar fallback
// ============================================================================

__global__ void k_fused_rope_backward_partial_scalar(
    __nv_bfloat16* __restrict__ grad_x,
    const __nv_bfloat16* __restrict__ cos_buf,
    const __nv_bfloat16* __restrict__ sin_buf,
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int nope_dim)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int rope_dim = head_dim - nope_dim;
  const int half_dim = rope_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int d       = idx % half_dim;
  const int vec_idx = idx / half_dim;

  if (vec_idx >= total_vecs) return;

  const int seq_pos = (vec_idx / n_heads) % seq_len;

  float go1 = __bfloat162float(grad_x[(int64_t)vec_idx * head_dim + nope_dim + d]);
  float go2 = __bfloat162float(grad_x[(int64_t)vec_idx * head_dim + nope_dim + half_dim + d]);
  float c   = __bfloat162float(cos_buf[(int64_t)seq_pos * half_dim + d]);
  float s   = __bfloat162float(sin_buf[(int64_t)seq_pos * half_dim + d]);

  float gx1 =  go1 * c + go2 * s;
  float gx2 = -go1 * s + go2 * c;

  grad_x[(int64_t)vec_idx * head_dim + nope_dim + d]            = __float2bfloat16(gx1);
  grad_x[(int64_t)vec_idx * head_dim + nope_dim + half_dim + d] = __float2bfloat16(gx2);
#endif  // __CUDA_ARCH__ >= 800
}

}  // namespace rope
}  // namespace nmoe

// ============================================================================
// C API wrappers
// ============================================================================
//
// These are the entry points called from Python (via pybind11 in bindings.cpp).
// Production path is vectorized-only (bfloat162). Odd half_dim is rejected to
// avoid silently falling back to scalar kernels on B200 training runs.

extern "C" cudaError_t fused_rope_forward(
    const void* x,           // [total_vecs, head_dim] BF16
    const void* cos_buf,     // [seq_len, half_dim] BF16
    const void* sin_buf,     // [seq_len, half_dim] BF16
    void* out,               // [total_vecs, head_dim] BF16
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    cudaStream_t stream)
{
  if (total_vecs < 0 || head_dim < 0 || seq_len < 0 || n_heads < 0) {
    return cudaErrorInvalidValue;
  }
  if (total_vecs == 0 || head_dim == 0) {
    return cudaSuccess;  // Nothing to do
  }
  if ((head_dim % 4) != 0 || seq_len <= 0 || n_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  if (x == nullptr || cos_buf == nullptr || sin_buf == nullptr || out == nullptr) {
    return cudaErrorInvalidValue;
  }
  bool supports_bf16 = false;
  cudaError_t probe = nmoe::rope::current_device_supports_bf16(&supports_bf16);
  if (probe != cudaSuccess) {
    return probe;
  }
  if (!supports_bf16) {
    return cudaErrorInvalidDeviceFunction;
  }
  const int half_dim = head_dim / 2;

  const auto* x_bf16   = static_cast<const __nv_bfloat16*>(x);
  const auto* cos_bf16  = static_cast<const __nv_bfloat16*>(cos_buf);
  const auto* sin_bf16  = static_cast<const __nv_bfloat16*>(sin_buf);
  auto*       out_bf16  = static_cast<__nv_bfloat16*>(out);

  const int half_dim_pairs = half_dim / 2;
  const int64_t total_threads = (int64_t)total_vecs * half_dim_pairs;
  if (total_threads <= 0 || total_threads > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int64_t grid64 = (total_threads + nmoe::rope::BLOCK_SIZE - 1) / nmoe::rope::BLOCK_SIZE;
  if (grid64 <= 0 || grid64 > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int grid = static_cast<int>(grid64);
  nmoe::rope::k_fused_rope_forward_vec2<<<grid, nmoe::rope::BLOCK_SIZE, 0, stream>>>(
      x_bf16, cos_bf16, sin_bf16, out_bf16,
      total_vecs, seq_len, n_heads, head_dim, half_dim);

  return cudaGetLastError();
}

extern "C" cudaError_t fused_rope_backward(
    const void* grad_out,    // [total_vecs, head_dim] BF16
    const void* cos_buf,     // [seq_len, half_dim] BF16
    const void* sin_buf,     // [seq_len, half_dim] BF16
    void* grad_x,            // [total_vecs, head_dim] BF16
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    cudaStream_t stream)
{
  if (total_vecs < 0 || head_dim < 0 || seq_len < 0 || n_heads < 0) {
    return cudaErrorInvalidValue;
  }
  if (total_vecs == 0 || head_dim == 0) {
    return cudaSuccess;
  }
  if ((head_dim % 4) != 0 || seq_len <= 0 || n_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  if (grad_out == nullptr || cos_buf == nullptr || sin_buf == nullptr || grad_x == nullptr) {
    return cudaErrorInvalidValue;
  }
  bool supports_bf16 = false;
  cudaError_t probe = nmoe::rope::current_device_supports_bf16(&supports_bf16);
  if (probe != cudaSuccess) {
    return probe;
  }
  if (!supports_bf16) {
    return cudaErrorInvalidDeviceFunction;
  }
  const int half_dim = head_dim / 2;

  const auto* go_bf16  = static_cast<const __nv_bfloat16*>(grad_out);
  const auto* cos_bf16 = static_cast<const __nv_bfloat16*>(cos_buf);
  const auto* sin_bf16 = static_cast<const __nv_bfloat16*>(sin_buf);
  auto*       gx_bf16  = static_cast<__nv_bfloat16*>(grad_x);

  const int half_dim_pairs = half_dim / 2;
  const int64_t total_threads = (int64_t)total_vecs * half_dim_pairs;
  if (total_threads <= 0 || total_threads > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int64_t grid64 = (total_threads + nmoe::rope::BLOCK_SIZE - 1) / nmoe::rope::BLOCK_SIZE;
  if (grid64 <= 0 || grid64 > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int grid = static_cast<int>(grid64);
  nmoe::rope::k_fused_rope_backward_vec2<<<grid, nmoe::rope::BLOCK_SIZE, 0, stream>>>(
      go_bf16, cos_bf16, sin_bf16, gx_bf16,
      total_vecs, seq_len, n_heads, head_dim, half_dim);

  return cudaGetLastError();
}

// ============================================================================
// Partial RoPE C API wrappers (in-place, rope portion only)
// ============================================================================
//
// These kernels apply RoPE only to elements [nope_dim : head_dim] within each
// head, leaving [0 : nope_dim] unchanged. They operate IN-PLACE, eliminating
// the need for torch.split + rotate_pe + torch.cat (3 kernels -> 1 kernel).
//
// In MLA attention (61 layers), this saves 4 kernel launches per layer
// (2 for Q split+cat, 2 for K split+cat) = 244 kernel launches per step.

extern "C" cudaError_t fused_rope_forward_partial(
    void* x,                 // IN-PLACE [total_vecs, head_dim] BF16
    const void* cos_buf,     // [seq_len, half_dim] BF16  (half_dim = rope_dim/2)
    const void* sin_buf,     // [seq_len, half_dim] BF16
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int nope_dim,            // elements [0:nope_dim] are unchanged
    cudaStream_t stream)
{
  if (total_vecs < 0 || head_dim < 0 || seq_len < 0 || n_heads < 0 || nope_dim < 0) {
    return cudaErrorInvalidValue;
  }
  if (nope_dim > head_dim) {
    return cudaErrorInvalidValue;
  }
  if (total_vecs == 0) {
    return cudaSuccess;  // Nothing to do
  }
  if (head_dim == 0) {
    return cudaSuccess;
  }
  if (seq_len <= 0 || n_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  const int rope_dim = head_dim - nope_dim;
  if (rope_dim <= 0) {
    return cudaSuccess;
  }
  if ((rope_dim % 4) != 0) {
    return cudaErrorInvalidValue;
  }
  if (x == nullptr || cos_buf == nullptr || sin_buf == nullptr) {
    return cudaErrorInvalidValue;
  }
  bool supports_bf16 = false;
  cudaError_t probe = nmoe::rope::current_device_supports_bf16(&supports_bf16);
  if (probe != cudaSuccess) {
    return probe;
  }
  if (!supports_bf16) {
    return cudaErrorInvalidDeviceFunction;
  }
  const int half_dim = rope_dim / 2;

  auto* x_bf16     = static_cast<__nv_bfloat16*>(x);
  const auto* cos_bf16 = static_cast<const __nv_bfloat16*>(cos_buf);
  const auto* sin_bf16 = static_cast<const __nv_bfloat16*>(sin_buf);

  const int half_dim_pairs = half_dim / 2;
  const int64_t total_threads = (int64_t)total_vecs * half_dim_pairs;
  if (total_threads <= 0 || total_threads > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int64_t grid64 = (total_threads + nmoe::rope::BLOCK_SIZE - 1) / nmoe::rope::BLOCK_SIZE;
  if (grid64 <= 0 || grid64 > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int grid = static_cast<int>(grid64);
  nmoe::rope::k_fused_rope_forward_partial_vec2<<<grid, nmoe::rope::BLOCK_SIZE, 0, stream>>>(
      x_bf16, cos_bf16, sin_bf16,
      total_vecs, seq_len, n_heads, head_dim, nope_dim);

  return cudaGetLastError();
}

extern "C" cudaError_t fused_rope_backward_partial(
    void* grad_x,            // IN-PLACE [total_vecs, head_dim] BF16
    const void* cos_buf,     // [seq_len, half_dim] BF16
    const void* sin_buf,     // [seq_len, half_dim] BF16
    int total_vecs,
    int seq_len,
    int n_heads,
    int head_dim,
    int nope_dim,            // elements [0:nope_dim] gradients are unchanged
    cudaStream_t stream)
{
  if (total_vecs < 0 || head_dim < 0 || seq_len < 0 || n_heads < 0 || nope_dim < 0) {
    return cudaErrorInvalidValue;
  }
  if (nope_dim > head_dim) {
    return cudaErrorInvalidValue;
  }
  if (total_vecs == 0) {
    return cudaSuccess;
  }
  if (head_dim == 0) {
    return cudaSuccess;
  }
  if (seq_len <= 0 || n_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  const int rope_dim = head_dim - nope_dim;
  if (rope_dim <= 0) {
    return cudaSuccess;
  }
  if ((rope_dim % 4) != 0) {
    return cudaErrorInvalidValue;
  }
  if (grad_x == nullptr || cos_buf == nullptr || sin_buf == nullptr) {
    return cudaErrorInvalidValue;
  }
  bool supports_bf16 = false;
  cudaError_t probe = nmoe::rope::current_device_supports_bf16(&supports_bf16);
  if (probe != cudaSuccess) {
    return probe;
  }
  if (!supports_bf16) {
    return cudaErrorInvalidDeviceFunction;
  }
  const int half_dim = rope_dim / 2;

  auto* gx_bf16    = static_cast<__nv_bfloat16*>(grad_x);
  const auto* cos_bf16 = static_cast<const __nv_bfloat16*>(cos_buf);
  const auto* sin_bf16 = static_cast<const __nv_bfloat16*>(sin_buf);

  const int half_dim_pairs = half_dim / 2;
  const int64_t total_threads = (int64_t)total_vecs * half_dim_pairs;
  if (total_threads <= 0 || total_threads > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int64_t grid64 = (total_threads + nmoe::rope::BLOCK_SIZE - 1) / nmoe::rope::BLOCK_SIZE;
  if (grid64 <= 0 || grid64 > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidValue;
  }
  const int grid = static_cast<int>(grid64);
  nmoe::rope::k_fused_rope_backward_partial_vec2<<<grid, nmoe::rope::BLOCK_SIZE, 0, stream>>>(
      gx_bf16, cos_bf16, sin_bf16,
      total_vecs, seq_len, n_heads, head_dim, nope_dim);

  return cudaGetLastError();
}
