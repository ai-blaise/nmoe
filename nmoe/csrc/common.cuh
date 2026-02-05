#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

namespace nmoe{

constexpr uint32_t sm100bitmask = 0xFEFFFFF;

struct SM100Arch{
    static constexpr int smem_capacity = 232448;

    static bool is_block_size_legal(const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const int &m, const int &n, const int &k,
                                    const int& block_m, const int& block_n, const int& block_k) {
        if (block_n % 16 != 0)
            return false;
        
        // for a small K, fewer store blocks improve store/compuet overlap and reduce epilogue bottleneck
        if (k < 256 and (block_n > 128 or block_m > 128))
            return false;

        return true;
    }
};

struct SharedMemConfig{
    int smem_size;
    int swizzle_a_mode;
    int swizzle_b_mode;
    int swizzle_cd_mode;
};

struct ThreadConfig {
    int num_threads;
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    static ThreadConfig sm100(const int& num_non_epilogue_threads, const int& num_epilogue_threads){
        auto config = ThreadConfig();
        config.num_threads = num_epilogue_threads + num_non_epilogue_threads;
        config.num_non_epilogue_threads = num_non_epilogue_threads;
        config.num_epilogue_threads = num_epilogue_threads;
        return config;
    }
};

struct tma_2sm_load_2d {
    __device__ __host__ static void copy_tma_2d(
        const void *desc_ptr, uint64_t* mbar_ptr,
        uint64_t cache_hint, void* smem_ptr,
        int32_t const& crd0, int32_t const& crd1) {
        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        // set peer bit to 0 so that transaction bytes will update CTA0's barrier.
        uint64_t smem_int_mbar = __cvta_generic_to_shared(mbar_ptr) & sm100bitmask;
        uint64_t smem_int_ptr  = __cvta_generic_to_shared(smem_ptr);
        asm volatile (
          "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
          " [%0], [%1, {%3, %4}], [%2], %5;"
          :
          : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
           "r"(crd0), "r"(crd1), "l"(cache_hint)
          : "memory");
    }
};

struct sm100_tma_2sm_load_multicast_2d {
    __device__ __host__ static void copy_tma_2d_multicast(
        const void* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
        uint64_t cache_hint, void* smem_ptr, int32_t const& crd0,
        int32_t const& crd1) {
        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        uint32_t smem_int_mbar = __cvta_generic_to_shared(mbar_ptr) & sm100bitmask;
        uint32_t smem_int_ptr  = __cvta_generic_to_shared(smem_ptr);
        asm volatile (
          "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
          " [%0], [%1, {%4, %5}], [%2], %3, %6;"
          :
          : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "h"(multicast_mask),
            "r"(crd0), "r"(crd1), "l"(cache_hint)
          : "memory");
    }
};

template <int CTAGroup>
struct Allocator{
    static_assert(CTAGroup == 1 | CTAGroup == 2, "CTAGroup must either be 1 or 2");
    using impl = std::conditional_t<CTAGroup == 1, cute::TMEM::Allocator1SM, cute::TMEM::Allocator2SM>;

    __device__ static void allocate(int num_columns, uint32_t* dst_ptr){
        impl{}.allocate(num_columns,dst_ptr);
    }

    __device__ static void free(uint32_t tmem_ptr, int num_columns) {
        impl{}.free(tmem_ptr, num_columns);
    }

    __device__ static void release_allocation_lock(){
        impl{}.release_allocation_lock();
    }
}

void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *error_msg_ptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
    error_msg_ptr = "unable to get error string";
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

} // namespace nmoe
