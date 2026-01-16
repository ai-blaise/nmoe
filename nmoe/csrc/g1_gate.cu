// G1 Gate backward kernel - sm_100a (Blackwell)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm100_tma.hpp"

using namespace cute;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "requires sm_100a or newer (Blackwell)."
#endif

namespace nmoe {

constexpr int TILE = 256;
constexpr int THREADS = 128;
constexpr int SMEM_TILE = TILE * sizeof(__nv_bfloat16);

struct G1GateSmem {
    alignas(128) __nv_bfloat16 d_out[2][TILE];
    alignas(128) __nv_bfloat16 gate[2][TILE];
    alignas(128) __nv_bfloat16 out_ung[2][TILE];
    alignas(128) __nv_bfloat16 d_gate_lin[TILE];
    alignas(128) __nv_bfloat16 d_out_ung[TILE];
    alignas(8) uint64_t mbar[2];
};

__device__ __forceinline__ void mbar_init(uint64_t* m, int n) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(__cvta_generic_to_shared(m)), "r"(n));
}

__device__ __forceinline__ void mbar_arrive_tx(uint64_t* m, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(__cvta_generic_to_shared(m)), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* m, int phase) {
    asm volatile(
        "{.reg .pred p; WAIT: mbarrier.try_wait.parity.shared.b64 p, [%0], %1; @!p bra WAIT;}"
        :: "r"(__cvta_generic_to_shared(m)), "r"(phase));
}

__device__ __forceinline__ void tma_load_1d(void* dst, const CUtensorMap* tmap, int coord, uint64_t* mbar) {
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
        :: "r"(__cvta_generic_to_shared(dst)), "l"(tmap), "r"(coord), "r"(__cvta_generic_to_shared(mbar)) : "memory");
}

__device__ __forceinline__ void tma_store_1d(const CUtensorMap* tmap, int coord, const void* src) {
    asm volatile(
        "cp.async.bulk.tensor.1d.global.shared::cta.tile [%0, {%1}], [%2];"
        :: "l"(tmap), "r"(coord), "r"(__cvta_generic_to_shared(src)) : "memory");
}

__global__ __launch_bounds__(THREADS)
void g1_gate_bwd_kernel(
    const CUtensorMap* __restrict__ tm_d_out,
    const CUtensorMap* __restrict__ tm_gate,
    const CUtensorMap* __restrict__ tm_out_ung,
    const CUtensorMap* __restrict__ tm_d_gate_lin,
    const CUtensorMap* __restrict__ tm_d_out_ung,
    int ntiles
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ G1GateSmem s[];
    int tid = threadIdx.x;

    if (tid == 0) { mbar_init(&s->mbar[0], 1); mbar_init(&s->mbar[1], 1); }
    __syncthreads();

    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        int b = t & 1;
        int c = t * TILE;

        if (tid == 0) {
            mbar_arrive_tx(&s->mbar[b], 3 * SMEM_TILE);
            tma_load_1d(s->d_out[b], tm_d_out, c, &s->mbar[b]);
            tma_load_1d(s->gate[b], tm_gate, c, &s->mbar[b]);
            tma_load_1d(s->out_ung[b], tm_out_ung, c, &s->mbar[b]);
        }
        mbar_wait(&s->mbar[b], b);
        __syncthreads();

        #pragma unroll
        for (int i = tid; i < TILE; i += THREADS) {
            float d = __bfloat162float(s->d_out[b][i]);
            float g = __bfloat162float(s->gate[b][i]);
            float u = __bfloat162float(s->out_ung[b][i]);
            s->d_out_ung[i] = __float2bfloat16(d * g);
            s->d_gate_lin[i] = __float2bfloat16(d * u * g * (1.f - g));
        }
        __syncthreads();

        if (tid == 0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            tma_store_1d(tm_d_gate_lin, c, s->d_gate_lin);
            tma_store_1d(tm_d_out_ung, c, s->d_out_ung);
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
    }
    if (tid == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    __syncthreads();
#endif
}

}  // namespace nmoe

extern "C" cudaError_t g1_gate_bwd(
    const void* tm_d_out, const void* tm_gate, const void* tm_out_ung,
    const void* tm_d_gate_lin, const void* tm_d_out_ung,
    int ntiles, cudaStream_t stream
) {
    int blocks = min(ntiles, 132);
    size_t smem = sizeof(nmoe::G1GateSmem);
    cudaFuncSetAttribute(nmoe::g1_gate_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    nmoe::g1_gate_bwd_kernel<<<blocks, nmoe::THREADS, smem, stream>>>(
        (const CUtensorMap*)tm_d_out, (const CUtensorMap*)tm_gate, (const CUtensorMap*)tm_out_ung,
        (const CUtensorMap*)tm_d_gate_lin, (const CUtensorMap*)tm_d_out_ung, ntiles);
    return cudaGetLastError();
}
