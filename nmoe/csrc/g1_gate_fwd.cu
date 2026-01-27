// G1 Gate forward kernel - sm_100a (Blackwell)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace nmoe {

struct alignas(16) bf16x8 { __nv_bfloat162 v[4]; };

__device__ __forceinline__
bf16x8 load_bf16x8(const __nv_bfloat16* __restrict__ p) {
  bf16x8 result;
  *reinterpret_cast<float4*>(&result) = __ldg(reinterpret_cast<const float4*>(p));
  return result;
}

__device__ __forceinline__
void store_bf16x8(__nv_bfloat16* __restrict__ p, const bf16x8& x) {
  *reinterpret_cast<bf16x8*>(p) = x;
}

__device__ __forceinline__
float fast_sigmoid(float x) {
  return 1.f / (1.f + __expf(-x));
}

__device__ __forceinline__
void compute_fwd(__nv_bfloat162 lin, __nv_bfloat162 attn,
                 __nv_bfloat162& out, __nv_bfloat162& gate) {
  float2 fl = __bfloat1622float2(lin);
  float2 fa = __bfloat1622float2(attn);
  float2 fg = {fast_sigmoid(fl.x), fast_sigmoid(fl.y)};
  gate = __float22bfloat162_rn(fg);
  out = __float22bfloat162_rn({fa.x * fg.x, fa.y * fg.y});
}

template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
g1_gate_fwd_kernel(
    const __nv_bfloat16* __restrict__ linear_out,
    const __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ gate,
    int64_t n_total
) {
  const int64_t tid = int64_t(blockIdx.x) * BLOCK + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * BLOCK;
  const int64_t n_vec8 = n_total / 8;
  const int64_t rem_start = n_vec8 * 8;

  for (int64_t i = tid; i < n_vec8; i += stride) {
    const int64_t off = i * 8;
    bf16x8 lin = load_bf16x8(linear_out + off);
    bf16x8 attn = load_bf16x8(attn_out + off);
    bf16x8 o, g;
    #pragma unroll
    for (int j = 0; j < 4; j++)
      compute_fwd(lin.v[j], attn.v[j], o.v[j], g.v[j]);
    store_bf16x8(output + off, o);
    store_bf16x8(gate + off, g);
  }

  for (int64_t i = rem_start + tid; i < n_total; i += stride) {
    float fl = __bfloat162float(linear_out[i]);
    float fa = __bfloat162float(attn_out[i]);
    float fg = fast_sigmoid(fl);
    gate[i] = __float2bfloat16(fg);
    output[i] = __float2bfloat16(fa * fg);
  }
}

}  // namespace nmoe

extern "C" cudaError_t g1_gate_fwd(
    const void* linear_out,
    const void* attn_out,
    void* output,
    void* gate,
    int64_t n,
    cudaStream_t stream
) {
  if (n <= 0) return cudaSuccess;

  constexpr int BLOCK = 256;
  int64_t n_vec8 = n / 8;

  int dev, sm;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
  const int64_t total_threads = (n + 7) / 8 * 8;
  int blocks = min(sm * 4, int((total_threads + BLOCK - 1) / BLOCK));
  if (blocks < 1) blocks = 1;

  nmoe::g1_gate_fwd_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
      (const __nv_bfloat16*)linear_out,
      (const __nv_bfloat16*)attn_out,
      (__nv_bfloat16*)output,
      (__nv_bfloat16*)gate,
      n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  return cudaStreamSynchronize(stream);
}
