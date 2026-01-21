// G1 Gate backward kernel - sm_100a (Blackwell)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

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
void compute_bwd(__nv_bfloat162 d_o, __nv_bfloat162 o_u, __nv_bfloat162 g,
                 __nv_bfloat162& d_o_u, __nv_bfloat162& d_g_l) {
  float2 fd = __bfloat1622float2(d_o);
  float2 fo = __bfloat1622float2(o_u);
  float2 fg = __bfloat1622float2(g);
  d_o_u = __float22bfloat162_rn({fd.x * fg.x, fd.y * fg.y});
  d_g_l = __float22bfloat162_rn({fd.x * fo.x * fg.x * (1.f - fg.x),
                                  fd.y * fo.y * fg.y * (1.f - fg.y)});
}

template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
g1_gate_bwd_kernel(
    const __nv_bfloat16* __restrict__ d_out,
    const __nv_bfloat16* __restrict__ out_ungated,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ d_out_ungated,
    __nv_bfloat16* __restrict__ d_gate_linear,
    int64_t n_vec8,
    int64_t n_total
) {
  const int64_t tid = int64_t(blockIdx.x) * BLOCK + threadIdx.x;
  const int64_t stride = int64_t(gridDim.x) * BLOCK;
  const int64_t rem_start = n_vec8 * 8;

  for (int64_t i = tid; i < n_vec8; i += stride) {
    const int64_t off = i * 8;
    bf16x8 d_o = load_bf16x8(d_out + off);
    bf16x8 o_u = load_bf16x8(out_ungated + off);
    bf16x8 g = load_bf16x8(gate + off);
    bf16x8 d_o_u, d_g_l;
    #pragma unroll
    for (int j = 0; j < 4; j++)
      compute_bwd(d_o.v[j], o_u.v[j], g.v[j], d_o_u.v[j], d_g_l.v[j]);
    store_bf16x8(d_out_ungated + off, d_o_u);
    store_bf16x8(d_gate_linear + off, d_g_l);
  }

  for (int64_t i = rem_start + tid; i < n_total; i += stride) {
    float fd = __bfloat162float(d_out[i]);
    float fo = __bfloat162float(out_ungated[i]);
    float fg = __bfloat162float(gate[i]);
    d_out_ungated[i] = __float2bfloat16(fd * fg);
    d_gate_linear[i] = __float2bfloat16(fd * fo * fg * (1.f - fg));
  }
}

}  // namespace nmoe

extern "C" cudaError_t g1_gate_bwd(
    const void* d_out,
    const void* out_ungated,
    const void* gate,
    void* d_out_ungated,
    void* d_gate_linear,
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

  nmoe::g1_gate_bwd_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
      (const __nv_bfloat16*)d_out,
      (const __nv_bfloat16*)out_ungated,
      (const __nv_bfloat16*)gate,
      (__nv_bfloat16*)d_out_ungated,
      (__nv_bfloat16*)d_gate_linear,
      n_vec8, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  return cudaStreamSynchronize(stream);
}
