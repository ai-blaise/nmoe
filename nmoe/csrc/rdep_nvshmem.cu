// NVSHMEM implementation for RDEP hybrid mode
//
// Provides hybrid dispatch/return that uses:
//   - CUDA IPC for intra-node communication (faster, lower latency)
//   - NVSHMEM for inter-node communication (required for multi-node)
//
// Architecture:
//   rank = rdma_rank * local_world + nvl_rank
//   - rdma_rank: node index
//   - nvl_rank: GPU within node
//
// Only compiled when WITH_NVSHMEM is defined.

#ifdef WITH_NVSHMEM

#include "rdep_nvshmem.cuh"
#include "ptx.cu"
#include "swizzle.cuh"

// Vendored DeepEP primitives - proper PTX semantics + IBGDA WQE support
#include "rdep/configs.cuh"
#include "rdep/utils.cuh"
#include "rdep/ibgda_device.cuh"

#include <cub/cub.cuh>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

using namespace nmoe::ptx;

// Forward declaration for swizzle_sf_strided (defined in quant.cu)
extern "C" cudaError_t swizzle_sf_strided(
    const void* sf_mkl,
    void* sf_mma,
    const int32_t* offs,
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream);

// NVSHMEM error checking macro
#define NVSHMEM_CHECK(call)                                                   \
    do {                                                                      \
        int status = call;                                                    \
        if (status != 0) {                                                    \
            fprintf(stderr, "NVSHMEM error at %s:%d: %s returned %d\n",        \
                    __FILE__, __LINE__, #call, status);                       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Helper for vectorized non-allocating store (int2)
__device__ __forceinline__ void st_na_v2_s32(int2* ptr, int2 val) {
    asm volatile(
        "st.global.relaxed.gpu.v2.s32 [%0], {%1, %2};"
        :
        : "l"(ptr), "r"(val.x), "r"(val.y)
        : "memory"
    );
}

namespace rdep {
namespace nvshmem {

// Import vendored DeepEP primitives for use in hybrid kernels
using nmoe::rdep::memory_fence;
using nmoe::rdep::memory_fence_gpu;
using nmoe::rdep::memory_fence_cta;
using nmoe::rdep::st_release_sys_global;
using nmoe::rdep::ld_acquire_sys_global;
using nmoe::rdep::st_na_release;
using nmoe::rdep::ld_na_relaxed;
using nmoe::rdep::st_na_relaxed;
using nmoe::rdep::ld_nc_global;
using nmoe::rdep::st_na_global;
using nmoe::rdep::barrier_block;
using nmoe::rdep::ibgda_put_nbi_warp;
using nmoe::rdep::ibgda_quiet;
using nmoe::rdep::ibgda_get_state;
using nmoe::rdep::ibgda_get_rc;
using nmoe::rdep::get_lane_id;
using nmoe::rdep::warp_reduce_sum;
using nmoe::rdep::ceil_div;
using nmoe::rdep::align_up;

// ============================================================================
// Global State
// ============================================================================

NvshmemState g_nvshmem = {};
static std::atomic<uint32_t> g_bwd_phase{0};
static std::atomic<uintptr_t> g_bwd_stream_slot{0};

// ============================================================================
// Constants
// ============================================================================

constexpr int SF_VEC = 32;
constexpr float FP8_MAX = 448.0f;
constexpr float FP4_MAX = 6.0f;
constexpr uint64_t TIMEOUT_CYCLES = 200000000000ull;  // ~100s at 2GHz

__host__ __forceinline__ bool host_ptr_is_pinned(const void* ptr) {
    if (ptr == nullptr) return false;
    unsigned int flags = 0;
    cudaError_t st = cudaHostGetFlags(&flags, const_cast<void*>(ptr));
    if (st == cudaSuccess) {
        return true;
    }
    if (st == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        return false;
    }
    fprintf(stderr, "RDEP ERROR: cudaHostGetFlags failed: %s\n", cudaGetErrorString(st));
    return false;
}

__host__ __forceinline__ bool validate_pinned_host_int(const int* ptr, const char* name) {
    if (ptr == nullptr) {
        fprintf(stderr, "RDEP ERROR: %s (host scratch) is null\n", name);
        return false;
    }
    struct CachedPinnedPtrs {
        const int* ptrs[8];
        int used = 0;
        int next = 0;
    };
    thread_local CachedPinnedPtrs cache{};
    for (int i = 0; i < cache.used; ++i) {
        if (cache.ptrs[i] == ptr) {
            return true;
        }
    }
    if (!host_ptr_is_pinned(ptr)) {
        fprintf(stderr, "RDEP ERROR: %s must reference pinned host memory\n", name);
        return false;
    }
    if (cache.used < 8) {
        cache.ptrs[cache.used++] = ptr;
    } else {
        cache.ptrs[cache.next] = ptr;
        cache.next = (cache.next + 1) & 7;
    }
    return true;
}

struct AsyncDeviceIntReadSlot {
    int* h_value = nullptr;
    cudaEvent_t ready = nullptr;
    const int* d_ptr = nullptr;
    uintptr_t stream_key = 0;
    bool pending = false;
    int last_value = 0;
};

struct AsyncDeviceIntReadTable {
    static constexpr int kMaxSlots = 128;
    AsyncDeviceIntReadSlot slots[kMaxSlots];
    std::vector<AsyncDeviceIntReadSlot> spill_slots;
    int next_evict = 0;
};
static_assert((AsyncDeviceIntReadTable::kMaxSlots & (AsyncDeviceIntReadTable::kMaxSlots - 1)) == 0,
              "AsyncDeviceIntReadTable::kMaxSlots must be power-of-two");

__host__ __forceinline__ AsyncDeviceIntReadSlot* get_async_read_slot(const int* d_ptr, cudaStream_t stream) {
    if (d_ptr == nullptr) return nullptr;
    const uintptr_t stream_key = reinterpret_cast<uintptr_t>(stream) + 1;
    thread_local AsyncDeviceIntReadTable table;
    thread_local AsyncDeviceIntReadSlot* hot_slot = nullptr;
    thread_local const int* hot_d_ptr = nullptr;
    thread_local uintptr_t hot_stream_key = 0;
    auto clear_hot = [&]() {
        hot_slot = nullptr;
        hot_d_ptr = nullptr;
        hot_stream_key = 0;
    };

    if (hot_slot != nullptr &&
        hot_d_ptr == d_ptr &&
        hot_stream_key == stream_key &&
        hot_slot->d_ptr == d_ptr &&
        hot_slot->stream_key == stream_key) {
        return hot_slot;
    }

    for (int i = 0; i < AsyncDeviceIntReadTable::kMaxSlots; ++i) {
        if (table.slots[i].d_ptr == d_ptr && table.slots[i].stream_key == stream_key) {
            hot_slot = &table.slots[i];
            hot_d_ptr = d_ptr;
            hot_stream_key = stream_key;
            return hot_slot;
        }
    }
    for (auto& spill : table.spill_slots) {
        if (spill.d_ptr == d_ptr && spill.stream_key == stream_key) {
            clear_hot();
            return &spill;
        }
    }
    for (int i = 0; i < AsyncDeviceIntReadTable::kMaxSlots; ++i) {
        if (table.slots[i].d_ptr == nullptr) {
            table.slots[i].d_ptr = d_ptr;
            table.slots[i].stream_key = stream_key;
            hot_slot = &table.slots[i];
            hot_d_ptr = d_ptr;
            hot_stream_key = stream_key;
            return hot_slot;
        }
    }
    for (auto& spill : table.spill_slots) {
        if (spill.d_ptr == nullptr) {
            spill.d_ptr = d_ptr;
            spill.stream_key = stream_key;
            spill.pending = false;
            spill.last_value = 0;
            clear_hot();
            return &spill;
        }
    }

    auto refresh_slot = [](AsyncDeviceIntReadSlot* cand) {
        if (cand->pending && cand->ready != nullptr) {
            cudaError_t q = cudaEventQuery(cand->ready);
            if (q == cudaSuccess) {
                if (cand->h_value != nullptr) cand->last_value = *cand->h_value;
                cand->pending = false;
            } else if (q != cudaErrorNotReady) {
                (void)cudaGetLastError();
                cand->pending = false;
            }
        }
    };

    for (int i = 0; i < AsyncDeviceIntReadTable::kMaxSlots; ++i) {
        const int idx = (table.next_evict + i) & (AsyncDeviceIntReadTable::kMaxSlots - 1);
        AsyncDeviceIntReadSlot* cand = &table.slots[idx];
        refresh_slot(cand);
        if (!cand->pending) {
            table.next_evict = (idx + 1) & (AsyncDeviceIntReadTable::kMaxSlots - 1);
            cand->d_ptr = d_ptr;
            cand->stream_key = stream_key;
            cand->last_value = 0;
            hot_slot = cand;
            hot_d_ptr = d_ptr;
            hot_stream_key = stream_key;
            return cand;
        }
    }
    for (auto& spill : table.spill_slots) {
        refresh_slot(&spill);
        if (!spill.pending) {
            spill.d_ptr = d_ptr;
            spill.stream_key = stream_key;
            spill.last_value = 0;
            clear_hot();
            return &spill;
        }
    }

    // Keep stream-ordered semantics without forced host waits when slot count spikes.
    table.spill_slots.emplace_back();
    AsyncDeviceIntReadSlot* spill = &table.spill_slots.back();
    spill->d_ptr = d_ptr;
    spill->stream_key = stream_key;
    spill->pending = false;
    spill->last_value = 0;
    clear_hot();
    return spill;
}

__host__ __forceinline__ bool complete_device_int_read_blocking(
    AsyncDeviceIntReadSlot* st,
    bool* ok_out,
    const char* context) {
    auto counter_wait_timeout_ms = []() -> int {
        static int cached = -1;
        if (cached < 0) {
            int parsed = 180000;  // 3 minutes
            const char* raw = std::getenv("NMOE_RDEP_COUNTER_WAIT_TIMEOUT_MS");
            if (raw && raw[0] != '\0') {
                parsed = std::atoi(raw);
                if (parsed < 0) parsed = 0;
            }
            cached = parsed;
        }
        return cached;
    };
    if (st == nullptr || st->ready == nullptr || st->h_value == nullptr) {
        fprintf(stderr, "RDEP ERROR: %s invalid async read slot state\n", context);
        return false;
    }
    cudaError_t q = cudaEventQuery(st->ready);
    if (q == cudaSuccess) {
        st->last_value = *st->h_value;
        st->pending = false;
        if (ok_out) *ok_out = true;
        return true;
    }
    if (q == cudaErrorNotReady) {
        const int timeout_ms = counter_wait_timeout_ms();
        const auto t0 = std::chrono::steady_clock::now();
        while (q == cudaErrorNotReady) {
            if (timeout_ms > 0) {
                const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0
                ).count();
                if (elapsed_ms > timeout_ms) {
                    fprintf(
                        stderr,
                        "RDEP ERROR: %s counter wait timed out after %lld ms (set NMOE_RDEP_COUNTER_WAIT_TIMEOUT_MS=0 to disable)\n",
                        context,
                        static_cast<long long>(elapsed_ms)
                    );
                    st->pending = false;
                    if (ok_out) *ok_out = false;
                    return false;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            q = cudaEventQuery(st->ready);
        }
        if (q != cudaSuccess) {
            fprintf(stderr, "RDEP ERROR: %s cudaEventQuery failed after wait: %s\n", context, cudaGetErrorString(q));
            (void)cudaGetLastError();
            st->pending = false;
            if (ok_out) *ok_out = false;
            return false;
        }
        st->last_value = *st->h_value;
        st->pending = false;
        if (ok_out) *ok_out = true;
        return true;
    }
    fprintf(stderr, "RDEP ERROR: %s cudaEventQuery failed: %s\n", context, cudaGetErrorString(q));
    (void)cudaGetLastError();
    st->pending = false;
    return false;
}

__host__ __forceinline__ int read_device_int_blocking(
    const int* d_ptr,
    cudaStream_t stream,
    bool* ok_out = nullptr) {
    if (ok_out) *ok_out = false;
    if (d_ptr == nullptr) {
        fprintf(stderr, "RDEP ERROR: read_device_int_blocking received null device pointer\n");
        return 0;
    }
    AsyncDeviceIntReadSlot* st = get_async_read_slot(d_ptr, stream);
    if (st == nullptr) return 0;

    if (st->h_value == nullptr) {
        cudaError_t alloc = cudaHostAlloc(reinterpret_cast<void**>(&st->h_value), sizeof(int), cudaHostAllocDefault);
        if (alloc != cudaSuccess) {
            fprintf(stderr, "RDEP ERROR: read_device_int_blocking cudaHostAlloc failed: %s\n",
                    cudaGetErrorString(alloc));
            (void)cudaGetLastError();
            return st->last_value;
        }
        *st->h_value = 0;
    }
    if (st->ready == nullptr) {
        cudaError_t create = cudaEventCreateWithFlags(&st->ready, cudaEventDisableTiming);
        if (create != cudaSuccess) {
            fprintf(stderr, "RDEP ERROR: read_device_int_blocking cudaEventCreateWithFlags failed: %s\n",
                    cudaGetErrorString(create));
            (void)cudaGetLastError();
            return st->last_value;
        }
    }

    if (st->pending) {
        (void)complete_device_int_read_blocking(st, ok_out, "read_device_int_blocking(pending)");
        return st->last_value;
    }

    cudaError_t cpy = cudaMemcpyAsync(st->h_value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (cpy != cudaSuccess) {
        fprintf(stderr, "RDEP ERROR: read_device_int_blocking cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(cpy));
        (void)cudaGetLastError();
        return st->last_value;
    }
    cudaError_t rec = cudaEventRecord(st->ready, stream);
    if (rec != cudaSuccess) {
        fprintf(stderr, "RDEP ERROR: read_device_int_blocking cudaEventRecord failed: %s\n",
                cudaGetErrorString(rec));
        (void)cudaGetLastError();
        st->pending = false;
        return st->last_value;
    }
    st->pending = true;
    (void)complete_device_int_read_blocking(st, ok_out, "read_device_int_blocking(fresh)");
    return st->last_value;
}

__host__ __forceinline__ int poll_device_int_async(const int* d_ptr, cudaStream_t stream) {
    if (d_ptr == nullptr) return 0;
    AsyncDeviceIntReadSlot* st = get_async_read_slot(d_ptr, stream);
    if (st == nullptr) return 0;

    if (st->h_value == nullptr) {
        if (cudaHostAlloc(reinterpret_cast<void**>(&st->h_value), sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            return st->last_value;
        }
        *st->h_value = 0;
    }
    if (st->ready == nullptr) {
        if (cudaEventCreateWithFlags(&st->ready, cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError();
            return st->last_value;
        }
    }

    if (st->pending) {
        cudaError_t q = cudaEventQuery(st->ready);
        if (q == cudaSuccess) {
            st->last_value = *st->h_value;
            st->pending = false;
        } else if (q != cudaErrorNotReady) {
            (void)cudaGetLastError();
            st->pending = false;
        }
    }
    if (!st->pending) {
        cudaError_t cpy = cudaMemcpyAsync(st->h_value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (cpy != cudaSuccess) {
            (void)cudaGetLastError();
            return st->last_value;
        }
        cudaError_t rec = cudaEventRecord(st->ready, stream);
        if (rec != cudaSuccess) {
            (void)cudaGetLastError();
            return st->last_value;
        }
        st->pending = true;
    }
    return st->last_value;
}

__host__ __forceinline__ int warp_stride_launch_block_cap() {
    thread_local int cached_cap = -1;
    if (cached_cap > 0) return cached_cap;
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        cached_cap = 1024;
        return cached_cap;
    }
    int sms = 0;
    err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess || sms <= 0) {
        (void)cudaGetLastError();
        cached_cap = 1024;
        return cached_cap;
    }
    cached_cap = std::max(1, std::min(4096, sms * 8));
    return cached_cap;
}

__host__ __forceinline__ int cap_warp_stride_blocks(int blocks_by_work) {
    return std::max(1, std::min(blocks_by_work, warp_stride_launch_block_cap()));
}

__host__ __forceinline__ int radix_sort_end_bit_for_range(int key_range) {
    if (key_range <= 1) return 1;
    return 32 - __builtin_clz(static_cast<unsigned int>(key_range - 1));
}

// ============================================================================
// Metadata (same as IPC version in rdep.cu)
// ============================================================================

struct alignas(16) Meta {
    int64_t row_id;      // encodes (src_rank, tok, slot)
    int32_t local_eid;   // expert index on owner GPU
    float   gate;        // gating weight
};
static_assert(sizeof(Meta) == 16, "Meta must be 16 bytes");

// Hybrid internode routing uses per-node proxies (DeepEP pattern): inter-node sends
// target the peer with the same `nvl_rank` on the destination node, then proxy
// forwards intra-node over IPC. We pack the final destination `dest_nvl_rank` in
// the high 16 bits of Meta.local_eid when writing to the proxy.
static constexpr int META_DEST_NVL_SHIFT = 16;
__device__ __host__ __forceinline__ int meta_pack_local_eid_dest_nvl(int local_eid, int dest_nvl_rank) {
    return (dest_nvl_rank << META_DEST_NVL_SHIFT) | (local_eid & 0xFFFF);
}
__device__ __host__ __forceinline__ int meta_unpack_local_eid(int packed) {
    return packed & 0xFFFF;
}
__device__ __host__ __forceinline__ int meta_unpack_dest_nvl(int packed) {
    return (packed >> META_DEST_NVL_SHIFT) & 0xFFFF;
}

__device__ __forceinline__ void nvshmem_meta_p(
    Meta* meta, int pe, int64_t row_id, int32_t local_eid, float gate) {
    unsigned long long* dst = reinterpret_cast<unsigned long long*>(meta);
    unsigned int gate_bits = __float_as_uint(gate);
    unsigned long long w0 = static_cast<unsigned long long>(row_id);
    unsigned long long w1 =
        static_cast<unsigned long long>(static_cast<unsigned int>(local_eid)) |
        (static_cast<unsigned long long>(gate_bits) << 32);
    nvshmem_ulonglong_p(dst + 0, w0, pe);
    nvshmem_ulonglong_p(dst + 1, w1, pe);
}

// Row ID encoding: (rank * T + tok) * K + slot
__device__ __host__ __forceinline__
int64_t encode_rid(int rank, int tok, int slot, int T, int K) {
    return (static_cast<int64_t>(rank) * T + tok) * K + slot;
}

__device__ __host__ __forceinline__ bool nmoe_is_pow2(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

__device__ __host__ __forceinline__ int nmoe_pow2_shift(int x) {
#ifdef __CUDA_ARCH__
    return __ffs(x) - 1;
#else
    return __builtin_ctz(static_cast<unsigned int>(x));
#endif
}

__device__ __host__ __forceinline__
void decode_rid_fast(
    int64_t rid,
    int T,
    int K,
    int* rank,
    int* tok,
    int* slot,
    bool k_pow2,
    int k_shift,
    bool t_pow2,
    int t_shift) {
    int64_t tmp = k_pow2 ? (rid >> k_shift) : (rid / K);
    *slot = static_cast<int>(rid - tmp * K);
    int64_t rank64 = t_pow2 ? (tmp >> t_shift) : (tmp / T);
    *rank = static_cast<int>(rank64);
    *tok = static_cast<int>(tmp - rank64 * T);
}

__device__ __host__ __forceinline__
void decode_rid(int64_t rid, int T, int K, int* rank, int* tok, int* slot) {
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;
    decode_rid_fast(rid, T, K, rank, tok, slot, k_pow2, k_shift, t_pow2, t_shift);
}

// ============================================================================
// Quantization Helpers (same as rdep.cu)
// ============================================================================

__device__ __forceinline__ uint8_t to_fp8(float v) {
    return f32_to_e4m3_byte(v);
}

__device__ __forceinline__ uint16_t to_fp4x4(float x0, float x1, float x2, float x3) {
    return f32x4_to_e2m1x4_packed(x0, x1, x2, x3);
}

__device__ __forceinline__ uint8_t e8m0_encode(float scale) {
    return e8m0_encode_from_pos_f32(scale);
}

__device__ __forceinline__ float e8m0_decode(uint8_t byte) {
    return e8m0_decode_to_f32(byte);
}

__device__ __forceinline__ int nmoe_expert_dest_fast(
    int eid,
    int n_local,
    bool n_local_pow2,
    int n_local_shift) {
    return n_local_pow2 ? (eid >> n_local_shift) : (eid / n_local);
}

__device__ __forceinline__ int nmoe_expert_local_fast(
    int eid,
    int n_local,
    bool n_local_pow2,
    int n_local_mask) {
    return n_local_pow2 ? (eid & n_local_mask) : (eid % n_local);
}

__device__ __forceinline__ int nmoe_rank_node_fast(
    int rank,
    int local_world,
    bool local_world_pow2,
    int local_world_shift) {
    return local_world_pow2 ? (rank >> local_world_shift) : (rank / local_world);
}

__device__ __forceinline__ int nmoe_rank_local_fast(
    int rank,
    int local_world,
    bool local_world_pow2,
    int local_world_mask) {
    return local_world_pow2 ? (rank & local_world_mask) : (rank % local_world);
}

// ============================================================================
// Intra-node IPC barrier (CUDA IPC + system atomics)
//
// Inter-node synchronization uses NVSHMEM host collectives (e.g. nvshmemx_barrier_all_on_stream).
// ============================================================================

__device__ __forceinline__ void ipc_barrier_dynamic(int** barrier_ptrs, int nvl_rank, int local_world) {
    int thread_id = static_cast<int>(threadIdx.x);

    fence_acq_rel_sys();
    __syncthreads();

    if (thread_id < local_world) {
        atomicAdd_sys(barrier_ptrs[nvl_rank] + thread_id, RDMA_BARRIER_TAG);
        atomicSub_sys(barrier_ptrs[thread_id] + nvl_rank, RDMA_BARRIER_TAG);
    }

    uint64_t start_time = clock64();
    uint32_t spins = 0;
    while (true) {
        int value = (thread_id < local_world) ? ld_volatile_s32(barrier_ptrs[nvl_rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0)) break;
        if (thread_id < local_world && (((++spins) & 0x3Fu) == 0u)) {
            __nanosleep(64);
        }
        if (clock64() - start_time > TIMEOUT_CYCLES && thread_id < local_world) {
            printf("nmoe IPC barrier timeout\n");
            trap();
        }
    }
    fence_acq_rel_sys();
    __syncthreads();
}

// ============================================================================
// Initialization
// ============================================================================

void get_uid(void* uid_out) {
    nvshmemx_uniqueid_t uid;
    nvshmemx_get_uniqueid(&uid);
    memcpy(uid_out, &uid, sizeof(nvshmemx_uniqueid_t));
}

int get_uid_size() {
    return sizeof(nvshmemx_uniqueid_t);
}

void init(const void* uid, int rank, int world, int local_world) {
    if (g_nvshmem.initialized) return;

    // NOTE: Do NOT call cudaSetDevice before nvshmem init
    // DeepEP doesn't do this and it causes problems with PyTorch

    // Initialize with UID using the proper helper function
    nvshmemx_uniqueid_t nvshmem_uid;
    memcpy(&nvshmem_uid, uid, sizeof(nvshmemx_uniqueid_t));

    nvshmemx_init_attr_t attr = {};
    // Use the helper to set up UID args properly
    int status = nvshmemx_set_attr_uniqueid_args(rank, world, &nvshmem_uid, &attr);
    if (status != 0) {
        fprintf(stderr, "RDEP: nvshmemx_set_attr_uniqueid_args failed with status %d\n", status);
        return;
    }

    status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (status != 0) {
        fprintf(stderr, "RDEP: nvshmemx_init_attr failed with status %d\n", status);
        return;
    }

    // Call nvshmem_barrier_all like DeepEP does after init
    nvshmem_barrier_all();

    g_nvshmem.rank = rank;
    g_nvshmem.world = world;
    g_nvshmem.local_world = local_world;
    g_nvshmem.num_nodes = world / local_world;
    g_nvshmem.rdma_rank = rank / local_world;
    g_nvshmem.nvl_rank = rank % local_world;
    g_nvshmem.initialized = true;
    g_bwd_phase.store(0u, std::memory_order_relaxed);
    g_bwd_stream_slot.store(0u, std::memory_order_relaxed);

    fprintf(stderr, "RDEP: NVSHMEM initialized (rank=%d, world=%d, local_world=%d, nodes=%d)\n",
            rank, world, local_world, g_nvshmem.num_nodes);
}

void finalize() {
    if (!g_nvshmem.initialized) {
        g_bwd_phase.store(0u, std::memory_order_relaxed);
        g_bwd_stream_slot.store(0u, std::memory_order_relaxed);
        return;
    }

    // Free symmetric allocations (NVSHMEM buffers).
    //
    // DeepEP pattern: prefer a single aligned symmetric allocation and slice it.
    // If the aligned base pointers are set, free them; otherwise fall back to the
    // legacy per-buffer frees.
    if (g_nvshmem.sym_bf16_base) {
        nvshmem_free(g_nvshmem.sym_bf16_base);
    } else {
        if (g_nvshmem.x_buf_bf16) nvshmem_free(g_nvshmem.x_buf_bf16);
        if (g_nvshmem.tok_y) nvshmem_free(g_nvshmem.tok_y);
        if (g_nvshmem.tok_gate) nvshmem_free(g_nvshmem.tok_gate);
        if (g_nvshmem.tok_tag) nvshmem_free(g_nvshmem.tok_tag);
        if (g_nvshmem.meta) nvshmem_free(g_nvshmem.meta);
        if (g_nvshmem.counter) nvshmem_free(g_nvshmem.counter);
        if (g_nvshmem.dropped) nvshmem_free(g_nvshmem.dropped);
        if (g_nvshmem.barrier_signals) nvshmem_free(g_nvshmem.barrier_signals);
    }
    if (g_nvshmem.sym_block_base) {
        nvshmem_free(g_nvshmem.sym_block_base);
    } else {
        if (g_nvshmem.x_buf_block) nvshmem_free(g_nvshmem.x_buf_block);
        if (g_nvshmem.sfa_buf) nvshmem_free(g_nvshmem.sfa_buf);
        if (g_nvshmem.y_buf) nvshmem_free(g_nvshmem.y_buf);
    }

    // Close IPC handles for remote buffers (not own buffer)
    for (int r = 0; r < g_nvshmem.local_world; r++) {
        if (r != g_nvshmem.nvl_rank && g_nvshmem.ipc_buffer_ptrs[r]) {
            cudaIpcCloseMemHandle(g_nvshmem.ipc_buffer_ptrs[r]);
        }
    }

    // Free local IPC buffer (cudaMalloc'd)
    if (g_nvshmem.ipc_buffer) cudaFree(g_nvshmem.ipc_buffer);

    // Free local work buffers
    if (g_nvshmem.local_eid) cudaFree(g_nvshmem.local_eid);
    if (g_nvshmem.order) cudaFree(g_nvshmem.order);
    if (g_nvshmem.offsets) cudaFree(g_nvshmem.offsets);
    if (g_nvshmem.dest) cudaFree(g_nvshmem.dest);
    if (g_nvshmem.M_pad_dev) cudaFree(g_nvshmem.M_pad_dev);
    if (g_nvshmem.meta_copy) cudaFree(g_nvshmem.meta_copy);
    if (g_nvshmem.sort_temp) cudaFree(g_nvshmem.sort_temp);
    if (g_nvshmem.d_ipc_buffer_ptrs) cudaFree(g_nvshmem.d_ipc_buffer_ptrs);
    if (g_nvshmem.d_ipc_barrier_signal_ptrs) cudaFree(g_nvshmem.d_ipc_barrier_signal_ptrs);

    nvshmem_finalize();
    g_nvshmem = {};
    g_bwd_phase.store(0u, std::memory_order_relaxed);
    g_bwd_stream_slot.store(0u, std::memory_order_relaxed);
}

// Helper function to compute IPC buffer layout (same as rdep.cu)
static inline size_t align_up(size_t x, size_t align) {
    return ((x + align - 1) / align) * align;
}

static void compute_ipc_buffer_layout_bf16(
    size_t capacity, int Ha, int world,
    size_t* x_off, size_t* meta_off, size_t* counter_off,
    size_t* dropped_off, size_t* barrier_off,
    size_t* tok_y_off, size_t* tok_gate_off,
    size_t* total_size)
{
    constexpr size_t BUFFER_ALIGNMENT = 128;
    *x_off = 0;
    *meta_off = capacity * Ha * sizeof(uint16_t);
    *counter_off = *meta_off + capacity * sizeof(Meta);
    *dropped_off = *counter_off + sizeof(int);
    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
    const size_t ptrs_end = align_up(*barrier_off + MAX_LOCAL_GPUS * sizeof(int), BUFFER_ALIGNMENT);

    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
    *tok_y_off = ptrs_end;
    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);
    *total_size = align_up(*tok_gate_off + tok_slots * sizeof(float), BUFFER_ALIGNMENT);
}

static void compute_ipc_buffer_layout_blockscaled(
    size_t capacity, int H, int Hp, int Hsf, int world,
    size_t* x_off, size_t* sfa_off, size_t* y_off,
    size_t* meta_off, size_t* counter_off,
    size_t* dropped_off, size_t* barrier_off,
    size_t* tok_y_off, size_t* tok_gate_off,
    size_t* total_size)
{
    constexpr size_t BUFFER_ALIGNMENT = 128;
    *x_off = 0;
    *sfa_off = capacity * static_cast<size_t>(Hp) * sizeof(uint16_t);
    *y_off = *sfa_off + capacity * static_cast<size_t>(Hsf) * sizeof(uint8_t);
    *meta_off = *y_off + capacity * static_cast<size_t>(H) * sizeof(uint16_t);
    *counter_off = *meta_off + capacity * sizeof(Meta);
    *dropped_off = *counter_off + sizeof(int);
    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
    const size_t ptrs_end = align_up(*barrier_off + MAX_LOCAL_GPUS * sizeof(int), BUFFER_ALIGNMENT);

    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
    const int tok_Ha = ((H + 7) / 8) * 8;
    *tok_y_off = ptrs_end;
    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(tok_Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);
    *total_size = align_up(*tok_gate_off + tok_slots * sizeof(float), BUFFER_ALIGNMENT);
}

void alloc_bf16(size_t capacity, int H, int n_local) {
    if (n_local <= 0) {
        fprintf(stderr, "RDEP FATAL: hybrid alloc_bf16 requires n_local > 0, got %d\n", n_local);
        abort();
    }
    int Ha = ((H + 7) / 8) * 8;
    const size_t tok_slots = (g_nvshmem.world > 0) ? (capacity / static_cast<size_t>(g_nvshmem.world)) : 0;

    // Free old NVSHMEM allocations
    if (g_nvshmem.sym_bf16_base) {
        nvshmem_free(g_nvshmem.sym_bf16_base);
    } else {
        if (g_nvshmem.x_buf_bf16) nvshmem_free(g_nvshmem.x_buf_bf16);
        if (g_nvshmem.tok_y) nvshmem_free(g_nvshmem.tok_y);
        if (g_nvshmem.tok_gate) nvshmem_free(g_nvshmem.tok_gate);
        if (g_nvshmem.tok_tag) nvshmem_free(g_nvshmem.tok_tag);
        if (g_nvshmem.meta) nvshmem_free(g_nvshmem.meta);
        if (g_nvshmem.counter) nvshmem_free(g_nvshmem.counter);
        if (g_nvshmem.dropped) nvshmem_free(g_nvshmem.dropped);
        if (g_nvshmem.barrier_signals) nvshmem_free(g_nvshmem.barrier_signals);
    }
    g_nvshmem.sym_bf16_base = nullptr;
    g_nvshmem.sym_bf16_bytes = 0;

    // Allocate symmetric heap for INTER-NODE communication.
    // DeepEP pattern: one aligned symmetric allocation + slicing into sub-buffers.
    // NOTE: NVSHMEM uses a fixed symmetric heap sized by NVSHMEM_SYMMETRIC_SIZE (default: 1GiB).
    // For moonlight-scale configs, this must be increased or nvshmem_malloc will fail and later
    // CUDA ops will surface as illegal memory access.
    const size_t x_bytes = capacity * static_cast<size_t>(Ha) * sizeof(uint16_t);
    const size_t tok_y_bytes = tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t);
    const size_t tok_gate_bytes = tok_slots * sizeof(float);
    const size_t tok_tag_bytes = tok_slots * sizeof(int);
    const size_t meta_bytes = capacity * sizeof(Meta);
    const size_t counter_bytes = sizeof(int);
    const size_t dropped_bytes = sizeof(int);
    const size_t barrier_bytes = MAX_NODES * sizeof(int);
    const size_t sym_total =
        x_bytes + tok_y_bytes + tok_gate_bytes + tok_tag_bytes +
        meta_bytes + counter_bytes + dropped_bytes + barrier_bytes;

    (void)sym_total;
    constexpr size_t kAlign = 128;
    const size_t x_off = 0;
    const size_t tok_y_off = align_up(x_off + x_bytes, kAlign);
    const size_t tok_gate_off = align_up(tok_y_off + tok_y_bytes, kAlign);
    const size_t tok_tag_off = align_up(tok_gate_off + tok_gate_bytes, kAlign);
    const size_t meta_off = align_up(tok_tag_off + tok_tag_bytes, kAlign);
    const size_t counter_off = align_up(meta_off + meta_bytes, kAlign);
    const size_t dropped_off = align_up(counter_off + counter_bytes, kAlign);
    const size_t barrier_off = align_up(dropped_off + dropped_bytes, kAlign);
    const size_t total_bytes = align_up(barrier_off + barrier_bytes, kAlign);

    g_nvshmem.sym_bf16_base = nvshmem_align(kAlign, total_bytes);
    g_nvshmem.sym_bf16_bytes = total_bytes;
    if (!g_nvshmem.sym_bf16_base) {
        fprintf(stderr,
            "RDEP ERROR: nvshmem_align failed (bf16). capacity=%zu H=%d Ha=%d tok_slots=%zu "
            "sym_bytes_total=%zu. Increase NVSHMEM_SYMMETRIC_SIZE and retry.\n",
            capacity, H, Ha, tok_slots, total_bytes);
        exit(EXIT_FAILURE);
    }
    char* base = static_cast<char*>(g_nvshmem.sym_bf16_base);
    g_nvshmem.x_buf_bf16 = reinterpret_cast<uint16_t*>(base + x_off);
    g_nvshmem.tok_y = reinterpret_cast<uint16_t*>(base + tok_y_off);
    g_nvshmem.tok_gate = reinterpret_cast<float*>(base + tok_gate_off);
    g_nvshmem.tok_tag = reinterpret_cast<int*>(base + tok_tag_off);
    g_nvshmem.meta = reinterpret_cast<void*>(base + meta_off);
    g_nvshmem.counter = reinterpret_cast<int*>(base + counter_off);
    g_nvshmem.dropped = reinterpret_cast<int*>(base + dropped_off);
    g_nvshmem.barrier_signals = reinterpret_cast<int*>(base + barrier_off);

    // Initialize counters and barriers
    CUDA_CHECK(cudaMemset(g_nvshmem.counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.dropped, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.barrier_signals, 0, MAX_NODES * sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.tok_tag, 0, tok_slots * sizeof(int)));

    // CRITICAL: Barrier + sync after NVSHMEM allocations, before any cudaMalloc
    // This matches DeepEP's pattern and prevents CUDA context corruption
    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // Allocate SEPARATE IPC buffer for INTRA-NODE communication (via cudaMalloc)
    // This buffer CAN be used with cudaIpcGetMemHandle/cudaIpcOpenMemHandle
    // because it's allocated with cudaMalloc, NOT nvshmem_malloc
    // =========================================================================
    if (g_nvshmem.ipc_buffer) cudaFree(g_nvshmem.ipc_buffer);

    // Compute IPC buffer layout
    compute_ipc_buffer_layout_bf16(
        capacity, Ha, g_nvshmem.world,
        &g_nvshmem.ipc_x_off,
        &g_nvshmem.ipc_meta_off,
        &g_nvshmem.ipc_counter_off,
        &g_nvshmem.ipc_dropped_off,
        &g_nvshmem.ipc_barrier_off,
        &g_nvshmem.ipc_tok_y_off,
        &g_nvshmem.ipc_tok_gate_off,
        &g_nvshmem.ipc_buffer_size);
    g_nvshmem.ipc_sfa_off = 0;
    g_nvshmem.ipc_y_off = 0;

    // Allocate IPC buffer with cudaMalloc
    CUDA_CHECK(cudaMalloc(&g_nvshmem.ipc_buffer, g_nvshmem.ipc_buffer_size));
    CUDA_CHECK(cudaMemset(g_nvshmem.ipc_buffer, 0, g_nvshmem.ipc_buffer_size));

    // Reset local pointer arrays; open_ipc_handles_* will populate peers.
    for (int r = 0; r < g_nvshmem.local_world; r++) {
        g_nvshmem.ipc_buffer_ptrs[r] = nullptr;
        g_nvshmem.ipc_barrier_signal_ptrs[r] = nullptr;
    }

    // Set local IPC buffer pointer
    g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank] = g_nvshmem.ipc_buffer;
    char* local_ipc = static_cast<char*>(g_nvshmem.ipc_buffer);
    g_nvshmem.ipc_barrier_signal_ptrs[g_nvshmem.nvl_rank] =
        reinterpret_cast<int*>(local_ipc + g_nvshmem.ipc_barrier_off);

    fprintf(stderr, "RDEP: Allocated IPC buffer (size=%zu bytes) for intra-node communication\n",
            g_nvshmem.ipc_buffer_size);

    // Allocate local work buffers
    if (g_nvshmem.local_eid) cudaFree(g_nvshmem.local_eid);
    if (g_nvshmem.order) cudaFree(g_nvshmem.order);
    if (g_nvshmem.offsets) cudaFree(g_nvshmem.offsets);
    if (g_nvshmem.dest) cudaFree(g_nvshmem.dest);
    if (g_nvshmem.M_pad_dev) cudaFree(g_nvshmem.M_pad_dev);
    if (g_nvshmem.sort_temp) cudaFree(g_nvshmem.sort_temp);

    CUDA_CHECK(cudaMalloc(&g_nvshmem.local_eid, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.order, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.offsets, (n_local + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.dest, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.M_pad_dev, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.meta_copy, capacity * sizeof(Meta)));

    // CUB sort temp storage
    g_nvshmem.sort_temp = nullptr;
    g_nvshmem.sort_temp_bytes = 0;
    if (n_local > 1 && capacity > 1) {
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g_nvshmem.sort_temp_bytes,
            g_nvshmem.local_eid, g_nvshmem.local_eid, g_nvshmem.order, g_nvshmem.order,
            static_cast<int>(capacity), 0, 32));
        if (g_nvshmem.sort_temp_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes));
        }
    }

    g_nvshmem.capacity = capacity;
    g_nvshmem.H = H;
    g_nvshmem.Ha = Ha;
    g_nvshmem.tok_Ha = Ha;
    g_nvshmem.n_local = n_local;
    g_nvshmem.align = 128;  // Match blockscaled for consistent padding
    g_nvshmem.profile = -1;  // BF16 mode
}

// ============================================================================
// IPC Buffer Management Functions
// These use the SEPARATE cudaMalloc'd IPC buffer, NOT the NVSHMEM buffers
// ============================================================================

void get_ipc_handle_bf16(void* handle_out) {
    if (!g_nvshmem.initialized || !g_nvshmem.ipc_buffer) {
        fprintf(stderr, "RDEP: get_ipc_handle_bf16 called but IPC buffer not allocated!\n");
        return;
    }
    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, g_nvshmem.ipc_buffer);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP: cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        return;
    }
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

void get_ipc_handle_blockscaled(void* handle_out) {
    if (!g_nvshmem.initialized || !g_nvshmem.ipc_buffer) {
        fprintf(stderr, "RDEP: get_ipc_handle_blockscaled called but IPC buffer not allocated!\n");
        return;
    }
    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, g_nvshmem.ipc_buffer);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP: cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        return;
    }
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

void open_ipc_handles_bf16(const void* handles, int local_world) {
    if (local_world < 1 || local_world > MAX_LOCAL_GPUS) {
        fprintf(stderr,
                "RDEP ERROR: invalid local_world=%d for NVSHMEM IPC open (MAX_LOCAL_GPUS=%d)\n",
                local_world, MAX_LOCAL_GPUS);
        return;
    }
    const cudaIpcMemHandle_t* all_handles = static_cast<const cudaIpcMemHandle_t*>(handles);
    int my_nvl_rank = g_nvshmem.nvl_rank;

    for (int r = 0; r < local_world; r++) {
        if (r == my_nvl_rank) {
            // Local buffer already set in alloc_bf16
            continue;
        }
        // Open remote IPC buffer
        CUDA_CHECK(cudaIpcOpenMemHandle(
            &g_nvshmem.ipc_buffer_ptrs[r],
            all_handles[r],
            cudaIpcMemLazyEnablePeerAccess));
        // Set barrier signal pointer for remote buffer
        char* remote_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[r]);
        g_nvshmem.ipc_barrier_signal_ptrs[r] =
            reinterpret_cast<int*>(remote_buf + g_nvshmem.ipc_barrier_off);
    }
    fprintf(stderr, "RDEP: Opened %d IPC handles for intra-node communication\n", local_world);
}

void open_ipc_handles_blockscaled(const void* handles, int local_world) {
    open_ipc_handles_bf16(handles, local_world);
}

void sync_ipc_buffer_ptrs_bf16() {
    // Copy IPC buffer pointers to device memory so kernels can access them
    // The host arrays ipc_buffer_ptrs and ipc_barrier_signal_ptrs cannot be
    // dereferenced from GPU code, so we need device copies

    int local_world = g_nvshmem.local_world;
    if (local_world < 1 || local_world > MAX_LOCAL_GPUS) {
        fprintf(stderr,
                "RDEP ERROR: invalid local_world=%d for pointer sync (MAX_LOCAL_GPUS=%d)\n",
                local_world, MAX_LOCAL_GPUS);
        return;
    }

    // Allocate device arrays if not already done
    if (g_nvshmem.d_ipc_buffer_ptrs == nullptr) {
        CUDA_CHECK(cudaMalloc(&g_nvshmem.d_ipc_buffer_ptrs, MAX_LOCAL_GPUS * sizeof(void*)));
    }
    if (g_nvshmem.d_ipc_barrier_signal_ptrs == nullptr) {
        CUDA_CHECK(cudaMalloc(&g_nvshmem.d_ipc_barrier_signal_ptrs, MAX_LOCAL_GPUS * sizeof(int*)));
    }

    for (int r = 0; r < local_world; ++r) {
        if (g_nvshmem.ipc_buffer_ptrs[r] == nullptr || g_nvshmem.ipc_barrier_signal_ptrs[r] == nullptr) {
            fprintf(stderr,
                    "RDEP ERROR: IPC pointer sync found null peer entry at local rank %d/%d. "
                    "Ensure open_ipc_handles_* completed before sync.\n",
                    r, local_world);
            abort();
        }
    }

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(g_nvshmem.d_ipc_buffer_ptrs, g_nvshmem.ipc_buffer_ptrs,
                          local_world * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_nvshmem.d_ipc_barrier_signal_ptrs, g_nvshmem.ipc_barrier_signal_ptrs,
                          local_world * sizeof(int*), cudaMemcpyHostToDevice));

    fprintf(stderr, "RDEP: Synced %d IPC buffer pointers to device\n", local_world);
}

void sync_ipc_buffer_ptrs_blockscaled() {
    sync_ipc_buffer_ptrs_bf16();
}

void alloc_blockscaled(size_t capacity, int H, int n_local, int profile) {
    if (n_local <= 0) {
        fprintf(stderr, "RDEP FATAL: hybrid alloc_blockscaled requires n_local > 0, got %d\n", n_local);
        abort();
    }
    int pack_factor = (profile == 0) ? 2 : 4;  // FP8: 2, NVFP4: 4
    int Hp = H / pack_factor;
    int Hsf = (H + SF_VEC - 1) / SF_VEC;
    int Ha = H;  // Return buffers are BF16, unpadded
    int tok_Ha = ((H + 7) / 8) * 8;
    const size_t tok_slots = (g_nvshmem.world > 0) ? (capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    constexpr int align = 128;

    // Free old allocations
    if (g_nvshmem.sym_block_base) {
        nvshmem_free(g_nvshmem.sym_block_base);
    } else {
        if (g_nvshmem.x_buf_block) nvshmem_free(g_nvshmem.x_buf_block);
        if (g_nvshmem.sfa_buf) nvshmem_free(g_nvshmem.sfa_buf);
        if (g_nvshmem.y_buf) nvshmem_free(g_nvshmem.y_buf);
        if (g_nvshmem.tok_y) nvshmem_free(g_nvshmem.tok_y);
        if (g_nvshmem.tok_gate) nvshmem_free(g_nvshmem.tok_gate);
        if (g_nvshmem.tok_tag) nvshmem_free(g_nvshmem.tok_tag);
        if (g_nvshmem.meta) nvshmem_free(g_nvshmem.meta);
        if (g_nvshmem.counter) nvshmem_free(g_nvshmem.counter);
        if (g_nvshmem.dropped) nvshmem_free(g_nvshmem.dropped);
        if (g_nvshmem.barrier_signals) nvshmem_free(g_nvshmem.barrier_signals);
    }
    g_nvshmem.sym_block_base = nullptr;
    g_nvshmem.sym_block_bytes = 0;

    // Allocate symmetric heap.
    // DeepEP pattern: one aligned symmetric allocation + slicing into sub-buffers.
    const size_t x_bytes = capacity * static_cast<size_t>(Hp) * sizeof(uint16_t);
    const size_t sfa_bytes = capacity * static_cast<size_t>(Hsf) * sizeof(uint8_t);
    const size_t y_bytes = capacity * static_cast<size_t>(H) * sizeof(uint16_t);
    const size_t tok_y_bytes = tok_slots * static_cast<size_t>(tok_Ha) * sizeof(uint16_t);
    const size_t tok_gate_bytes = tok_slots * sizeof(float);
    const size_t tok_tag_bytes = tok_slots * sizeof(int);
    const size_t meta_bytes = capacity * sizeof(Meta);
    const size_t counter_bytes = sizeof(int);
    const size_t dropped_bytes = sizeof(int);
    const size_t barrier_bytes = MAX_NODES * sizeof(int);
    const size_t sym_total =
        x_bytes + sfa_bytes + y_bytes + tok_y_bytes + tok_gate_bytes + tok_tag_bytes +
        meta_bytes + counter_bytes + dropped_bytes + barrier_bytes;

    (void)sym_total;
    constexpr size_t kAlignBuf = 128;
    const size_t x_off = 0;
    const size_t sfa_off = align_up(x_off + x_bytes, kAlignBuf);
    const size_t y_off = align_up(sfa_off + sfa_bytes, kAlignBuf);
    const size_t tok_y_off = align_up(y_off + y_bytes, kAlignBuf);
    const size_t tok_gate_off = align_up(tok_y_off + tok_y_bytes, kAlignBuf);
    const size_t tok_tag_off = align_up(tok_gate_off + tok_gate_bytes, kAlignBuf);
    const size_t meta_off = align_up(tok_tag_off + tok_tag_bytes, kAlignBuf);
    const size_t counter_off = align_up(meta_off + meta_bytes, kAlignBuf);
    const size_t dropped_off = align_up(counter_off + counter_bytes, kAlignBuf);
    const size_t barrier_off = align_up(dropped_off + dropped_bytes, kAlignBuf);
    const size_t total_bytes = align_up(barrier_off + barrier_bytes, kAlignBuf);

    g_nvshmem.sym_block_base = nvshmem_align(kAlignBuf, total_bytes);
    g_nvshmem.sym_block_bytes = total_bytes;
    if (!g_nvshmem.sym_block_base) {
        fprintf(stderr,
            "RDEP ERROR: nvshmem_align failed (blockscaled). profile=%d capacity=%zu H=%d Hp=%d Hsf=%d tok_slots=%zu "
            "sym_bytes_total=%zu. Increase NVSHMEM_SYMMETRIC_SIZE and retry.\n",
            profile, capacity, H, Hp, Hsf, tok_slots, total_bytes);
        exit(EXIT_FAILURE);
    }
    char* base = static_cast<char*>(g_nvshmem.sym_block_base);
    g_nvshmem.x_buf_block = reinterpret_cast<uint16_t*>(base + x_off);
    g_nvshmem.sfa_buf = reinterpret_cast<uint8_t*>(base + sfa_off);
    g_nvshmem.y_buf = reinterpret_cast<uint16_t*>(base + y_off);
    g_nvshmem.tok_y = reinterpret_cast<uint16_t*>(base + tok_y_off);
    g_nvshmem.tok_gate = reinterpret_cast<float*>(base + tok_gate_off);
    g_nvshmem.tok_tag = reinterpret_cast<int*>(base + tok_tag_off);
    g_nvshmem.meta = reinterpret_cast<void*>(base + meta_off);
    g_nvshmem.counter = reinterpret_cast<int*>(base + counter_off);
    g_nvshmem.dropped = reinterpret_cast<int*>(base + dropped_off);
    g_nvshmem.barrier_signals = reinterpret_cast<int*>(base + barrier_off);

    // Initialize
    CUDA_CHECK(cudaMemset(g_nvshmem.counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.dropped, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.barrier_signals, 0, MAX_NODES * sizeof(int)));
    CUDA_CHECK(cudaMemset(g_nvshmem.tok_tag, 0, tok_slots * sizeof(int)));

    // CRITICAL: Barrier + sync after NVSHMEM allocations, before any cudaMalloc
    // This matches DeepEP's pattern and prevents CUDA context corruption
    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // Allocate SEPARATE IPC buffer for INTRA-NODE communication (via cudaMalloc)
    // =========================================================================
    if (g_nvshmem.ipc_buffer) cudaFree(g_nvshmem.ipc_buffer);

    compute_ipc_buffer_layout_blockscaled(
        capacity, H, Hp, Hsf, g_nvshmem.world,
        &g_nvshmem.ipc_x_off,
        &g_nvshmem.ipc_sfa_off,
        &g_nvshmem.ipc_y_off,
        &g_nvshmem.ipc_meta_off,
        &g_nvshmem.ipc_counter_off,
        &g_nvshmem.ipc_dropped_off,
        &g_nvshmem.ipc_barrier_off,
        &g_nvshmem.ipc_tok_y_off,
        &g_nvshmem.ipc_tok_gate_off,
        &g_nvshmem.ipc_buffer_size);

    CUDA_CHECK(cudaMalloc(&g_nvshmem.ipc_buffer, g_nvshmem.ipc_buffer_size));
    CUDA_CHECK(cudaMemset(g_nvshmem.ipc_buffer, 0, g_nvshmem.ipc_buffer_size));

    // Reset local pointer arrays; open_ipc_handles_* will populate peers.
    for (int r = 0; r < g_nvshmem.local_world; r++) {
        g_nvshmem.ipc_buffer_ptrs[r] = nullptr;
        g_nvshmem.ipc_barrier_signal_ptrs[r] = nullptr;
    }

    g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank] = g_nvshmem.ipc_buffer;
    char* local_ipc = static_cast<char*>(g_nvshmem.ipc_buffer);
    g_nvshmem.ipc_barrier_signal_ptrs[g_nvshmem.nvl_rank] =
        reinterpret_cast<int*>(local_ipc + g_nvshmem.ipc_barrier_off);

    fprintf(stderr, "RDEP: Allocated IPC buffer (size=%zu bytes) for intra-node communication\n",
            g_nvshmem.ipc_buffer_size);

    // Allocate local work buffers
    if (g_nvshmem.local_eid) cudaFree(g_nvshmem.local_eid);
    if (g_nvshmem.order) cudaFree(g_nvshmem.order);
    if (g_nvshmem.offsets) cudaFree(g_nvshmem.offsets);
    if (g_nvshmem.dest) cudaFree(g_nvshmem.dest);
    if (g_nvshmem.M_pad_dev) cudaFree(g_nvshmem.M_pad_dev);
    if (g_nvshmem.meta_copy) cudaFree(g_nvshmem.meta_copy);
    if (g_nvshmem.sort_temp) cudaFree(g_nvshmem.sort_temp);

    CUDA_CHECK(cudaMalloc(&g_nvshmem.local_eid, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.order, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.offsets, (n_local + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.dest, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.M_pad_dev, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_nvshmem.meta_copy, capacity * sizeof(Meta)));

    g_nvshmem.sort_temp = nullptr;
    g_nvshmem.sort_temp_bytes = 0;
    if (n_local > 1 && capacity > 1) {
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g_nvshmem.sort_temp_bytes,
            g_nvshmem.local_eid, g_nvshmem.local_eid, g_nvshmem.order, g_nvshmem.order,
            static_cast<int>(capacity), 0, 32));
        if (g_nvshmem.sort_temp_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes));
        }
    }

    g_nvshmem.capacity = capacity;
    g_nvshmem.H = H;
    g_nvshmem.Ha = Ha;
    g_nvshmem.Hp = Hp;
    g_nvshmem.Hsf = Hsf;
    g_nvshmem.n_local = n_local;
    g_nvshmem.tok_Ha = tok_Ha;
    g_nvshmem.align = align;  // Blockscaled alignment
    g_nvshmem.profile = profile;
}

void reset_dispatch_counters(cudaStream_t stream) {
    cudaMemsetAsync(g_nvshmem.counter, 0, sizeof(int), stream);
    cudaMemsetAsync(g_nvshmem.dropped, 0, sizeof(int), stream);
}

void reset_counter_only(cudaStream_t stream) {
    cudaMemsetAsync(g_nvshmem.counter, 0, sizeof(int), stream);
}

// ============================================================================
// Synchronization (Host API)
// ============================================================================

void quiet() {
    nvshmem_quiet();
}

void quiet_on_stream(cudaStream_t stream) {
    nvshmemx_quiet_on_stream(stream);
}

__global__ void k_ipc_barrier(
    int** nvl_barrier_ptrs,
    int nvl_rank,
    int local_world)
{
    ipc_barrier_dynamic(nvl_barrier_ptrs, nvl_rank, local_world);
}

static inline void hybrid_barrier_on_stream(cudaStream_t stream) {
    // Intra-node (IPC) completion + inter-node (NVSHMEM) completion.
    // Skip no-op components to avoid unnecessary global rendezvous.
    if (g_nvshmem.local_world > 1) {
        thread_local int cached_world = -1;
        thread_local int cached_threads = 32;
        if (cached_world != g_nvshmem.local_world) {
            int threads = 32;
            while (threads < g_nvshmem.local_world && threads < 256) threads <<= 1;
            cached_world = g_nvshmem.local_world;
            cached_threads = threads;
        }
        k_ipc_barrier<<<1, cached_threads, 0, stream>>>(
            g_nvshmem.d_ipc_barrier_signal_ptrs,
            g_nvshmem.nvl_rank,
            g_nvshmem.local_world);
        CUDA_CHECK(cudaGetLastError());
    }
    if (g_nvshmem.num_nodes > 1) {
        nvshmemx_barrier_all_on_stream(stream);
    }
}

static inline void ipc_barrier_on_stream(cudaStream_t stream) {
    if (g_nvshmem.local_world > 1) {
        thread_local int cached_world = -1;
        thread_local int cached_threads = 32;
        if (cached_world != g_nvshmem.local_world) {
            int threads = 32;
            while (threads < g_nvshmem.local_world && threads < 256) threads <<= 1;
            cached_world = g_nvshmem.local_world;
            cached_threads = threads;
        }
        k_ipc_barrier<<<1, cached_threads, 0, stream>>>(
            g_nvshmem.d_ipc_barrier_signal_ptrs,
            g_nvshmem.nvl_rank,
            g_nvshmem.local_world);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Hybrid Dispatch Kernel (BF16)
// ============================================================================

__global__ void k_dispatch_hybrid_bf16(
    const __nv_bfloat16* __restrict__ x,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    const float* __restrict__ gates,        // [T, K]
    int my_rank, int T, int H, int Ha, int K,
    int n_local, int capacity,
    int local_world, int num_nodes, int rdma_rank, int nvl_rank,
    // NVSHMEM buffers (for inter-node)
    uint16_t* nvshmem_x_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    int* nvshmem_dropped,
    // IPC buffers (for intra-node)
    void** ipc_buffer_ptrs,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;
    int M = T * K;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const int n_local_mask = n_local - 1;
    const int world = local_world * num_nodes;
    const int64_t rid_base = static_cast<int64_t>(my_rank) * static_cast<int64_t>(T) * static_cast<int64_t>(K);

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);

        int eid = eids[i];
        if (eid < 0) continue;
        float gate = gates[i];
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        int local_eid = nmoe_expert_local_fast(eid, n_local, n_local_pow2, n_local_mask);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;
        if (local_eid < 0 || local_eid >= n_local) continue;

        int dest_rdma_rank = nmoe_rank_node_fast(dest, local_world, local_world_pow2, local_world_shift);
        int dest_nvl_rank = nmoe_rank_local_fast(dest, local_world, local_world_pow2, local_world_mask);
        bool is_remote_node = (dest_rdma_rank != rdma_rank);
        bool is_remote_gpu = (!is_remote_node) && (dest_nvl_rank != nvl_rank);
        int slot_r;

        if (is_remote_node) {
            // Inter-node: send to proxy peer (same nvl_rank) on destination node.
            const int proxy_pe = dest_rdma_rank * local_world + nvl_rank;
            if (lane == 0) {
                // Atomic increment counter on destination node
                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, proxy_pe);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) {
                    nvshmem_int_atomic_add(nvshmem_dropped, 1, proxy_pe);  // Fire and forget
                }
                continue;
            }

            // Write metadata via NVSHMEM
            if (lane == 0) {
                int64_t row_id = rid_base + static_cast<int64_t>(i);
                const int local_eid_packed = meta_pack_local_eid_dest_nvl(local_eid, dest_nvl_rank);
                nvshmem_meta_p(nvshmem_meta + slot_r, proxy_pe, row_id, local_eid_packed, gate);
            }

            // Write BF16 payload via NVSHMEM (warp-cooperative)
            const __nv_bfloat16* row = x + (int64_t)tok * H;
            uint16_t* dst = nvshmem_x_buf + (int64_t)slot_r * Ha;

            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(row + h),
                                      1, proxy_pe);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh, reinterpret_cast<const uint16_t*>(row) + hh, 1, proxy_pe);
                    }
                }
            }
        } else {
            // Intra-node: use IPC
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl_rank]);
            uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) {
                    atomicAdd(nvshmem_dropped, 1);
                }
                continue;
            }

            // Write metadata
            if (lane == 0) {
                Meta m{rid_base + static_cast<int64_t>(i), local_eid, gate};
                if (is_remote_gpu) {
                    int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                    int4 meta_val = *reinterpret_cast<const int4*>(&m);
                    st_na_v4_s32(meta_dst, meta_val);
                } else {
                    meta_buf[slot_r] = m;
                }
            }

            // Write BF16 payload
            const __nv_bfloat16* row = x + (int64_t)tok * H;
            uint16_t* dst = x_buf + (int64_t)slot_r * Ha;

            if (is_remote_gpu) {
                // Vectorized non-allocating stores for P2P
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        st_na_v2_s32(d, v);
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(row)[hh]);
                        }
                    }
                }
            } else {
                // Local write
                const uint16_t* row_u16 = reinterpret_cast<const uint16_t*>(row);
                const bool vec_ok =
                    ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(row_u16)) & 0x7u) == 0u;
                if (vec_ok) {
                    for (int h = lane * 4; h < H; h += 32 * 4) {
                        if (h + 4 <= H) {
                            *reinterpret_cast<int2*>(dst + h) =
                                ld_nc_v2_s32(reinterpret_cast<const int2*>(row_u16 + h));
                        } else {
                            for (int hh = h; hh < H && hh < h + 4; hh++) {
                                dst[hh] = row_u16[hh];
                            }
                        }
                    }
                } else {
                    for (int h = lane; h < H; h += 32) {
                        dst[h] = row_u16[h];
                    }
                }
            }
        }
    }
    // Ensure producer writes are globally ordered before subsequent barrier kernels.
    fence_acq_rel_sys();
}

__global__ void k_forward_nvshmem_dispatch_to_ipc_bf16(
    const uint16_t* __restrict__ nv_x_buf,   // [nv_count, Ha]
    const Meta* __restrict__ nv_meta,        // [nv_count]
    int nv_count,
    int H, int Ha,
    int capacity,
    int nvl_rank,
    int local_world,
    void** ipc_buffer_ptrs,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(nv_meta + i));
        const Meta in_m = mv.m;

        const int dest_nvl = meta_unpack_dest_nvl(in_m.local_eid);
        const int local_eid = meta_unpack_local_eid(in_m.local_eid);
        if (dest_nvl < 0 || dest_nvl >= local_world) continue;

        char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

        int slot_r;
        if (lane == 0) {
            slot_r = atomicAdd(counter, 1);
        }
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r < 0 || slot_r >= capacity) continue;

        // Metadata (strip packed dest_nvl).
        if (lane == 0) {
            Meta out_m{in_m.row_id, local_eid, in_m.gate};
            if (dest_nvl != nvl_rank) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&out_m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = out_m;
            }
        }

        const bool remote_gpu = (dest_nvl != nvl_rank);
        const uint16_t* src = nv_x_buf + (int64_t)i * Ha;
        uint16_t* dst = x_buf + (int64_t)slot_r * Ha;
        for (int h = lane * 4; h < H; h += 32 * 4) {
            if (h + 4 <= H) {
                int2 v = ld_nc_v2_s32(reinterpret_cast<const int2*>(src + h));
                int2* d = reinterpret_cast<int2*>(dst + h);
                if (remote_gpu) {
                    st_na_v2_s32(d, v);
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 4; hh++) {
                    uint16_t u = src[hh];
                    if (remote_gpu) {
                        st_na_relaxed_gpu_b16(dst + hh, u);
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }
    fence_acq_rel_sys();
}

// Dynamic variant that reads nv_count from device memory to avoid host D2H
// count read on the dispatch hot path.
__global__ void k_forward_nvshmem_dispatch_to_ipc_bf16_dynamic(
    const uint16_t* __restrict__ nv_x_buf,   // [capacity, Ha]
    const Meta* __restrict__ nv_meta,        // [capacity]
    const int* __restrict__ nv_count_ptr,    // device counter
    int* __restrict__ dropped_ptr,           // device dropped counter
    int H, int Ha,
    int capacity,
    int nvl_rank,
    int local_world,
    void** ipc_buffer_ptrs,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    int nv_count = ld_nc_s32(nv_count_ptr);
    if (nv_count <= 0) return;
    if (nv_count > capacity) {
        if (dropped_ptr && threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd(dropped_ptr, nv_count - capacity);
        }
        nv_count = capacity;
    }

    for (int i = warp_id; i < nv_count; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(nv_meta + i));
        const Meta in_m = mv.m;

        const int dest_nvl = meta_unpack_dest_nvl(in_m.local_eid);
        const int local_eid = meta_unpack_local_eid(in_m.local_eid);
        if (dest_nvl < 0 || dest_nvl >= local_world) continue;

        char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

        int slot_r;
        if (lane == 0) {
            slot_r = atomicAdd(counter, 1);
        }
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r < 0 || slot_r >= capacity) {
            if (lane == 0 && dropped_ptr) atomicAdd(dropped_ptr, 1);
            continue;
        }

        if (lane == 0) {
            Meta out_m{in_m.row_id, local_eid, in_m.gate};
            if (dest_nvl != nvl_rank) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&out_m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = out_m;
            }
        }

        const bool remote_gpu = (dest_nvl != nvl_rank);
        const uint16_t* src = nv_x_buf + (int64_t)i * Ha;
        uint16_t* dst = x_buf + (int64_t)slot_r * Ha;
        for (int h = lane * 4; h < H; h += 32 * 4) {
            if (h + 4 <= H) {
                int2 v = ld_nc_v2_s32(reinterpret_cast<const int2*>(src + h));
                int2* d = reinterpret_cast<int2*>(dst + h);
                if (remote_gpu) {
                    st_na_v2_s32(d, v);
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 4; hh++) {
                    uint16_t u = src[hh];
                    if (remote_gpu) {
                        st_na_relaxed_gpu_b16(dst + hh, u);
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_merge_nvshmem_into_ipc_bf16(
    const uint16_t* __restrict__ nv_x_buf,
    const Meta* __restrict__ nv_meta,
    uint16_t* __restrict__ ipc_x_buf,
    Meta* __restrict__ ipc_meta,
    int ipc_base,
    int nv_count,
    int H, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        int out_i = ipc_base + i;
        if (lane == 0) {
            ipc_meta[out_i] = nv_meta[i];
        }
        const uint16_t* src = nv_x_buf + (int64_t)i * Ha;
        uint16_t* dst = ipc_x_buf + (int64_t)out_i * Ha;
        for (int h = lane; h < H; h += 32) {
            dst[h] = src[h];
        }
    }
}

// ============================================================================
// Hybrid Dispatch Kernel (Blockscaled: FP8/NVFP4)
// ============================================================================

template <bool kFP8>
__global__ void k_dispatch_hybrid_blockscaled(
    const __nv_bfloat16* __restrict__ x,   // [T, H]
    const int* __restrict__ eids,          // [T, K]
    const float* __restrict__ gates,       // [T, K]
    int my_rank, int T, int H, int Hp, int Hsf, int K,
    int n_local, int capacity,
    int local_world, int num_nodes, int rdma_rank, int nvl_rank,
    // NVSHMEM buffers (inter-node)
    uint16_t* nvshmem_x_buf,
    uint8_t* nvshmem_sfa_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    int* nvshmem_dropped,
    // IPC buffers (intra-node)
    void** ipc_buffer_ptrs,
    size_t ipc_sfa_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const int n_local_mask = n_local - 1;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;
    const int world = local_world * num_nodes;
    const int64_t rid_base = static_cast<int64_t>(my_rank) * static_cast<int64_t>(T) * static_cast<int64_t>(K);

    constexpr float dtype_max = kFP8 ? FP8_MAX : FP4_MAX;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);

        int eid = eids[i];
        if (eid < 0) continue;
        float gate = gates[i];
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        int local_eid = nmoe_expert_local_fast(eid, n_local, n_local_pow2, n_local_mask);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;
        if (local_eid < 0 || local_eid >= n_local) continue;

        int dest_rdma_rank = nmoe_rank_node_fast(dest, local_world, local_world_pow2, local_world_shift);
        int dest_nvl_rank = nmoe_rank_local_fast(dest, local_world, local_world_pow2, local_world_mask);
        bool is_remote_node = (dest_rdma_rank != rdma_rank);
        bool is_remote_gpu = (!is_remote_node) && (dest_nvl_rank != nvl_rank);

        int slot_r;

        const __nv_bfloat16* row = x + (int64_t)tok * H;

        if (is_remote_node) {
            if (lane == 0) {
                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, dest);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) {
                    nvshmem_int_atomic_add(nvshmem_dropped, 1, dest);
                }
                continue;
            }

            if (lane == 0) {
                int64_t row_id = rid_base + static_cast<int64_t>(i);
                nvshmem_meta_p(nvshmem_meta + slot_r, dest, row_id, local_eid, gate);
            }

            uint16_t* dst_pack = nvshmem_x_buf + (int64_t)slot_r * Hp;
            uint8_t* dst_sfa = nvshmem_sfa_buf + (int64_t)slot_r * Hsf;

            for (int blk = 0; blk < Hsf; blk++) {
                int h0 = blk * SF_VEC;
                int h_end = min(h0 + SF_VEC, H);
                int blk_size = h_end - h0;

                float val = 0.0f;
                if (lane < blk_size) val = __bfloat162float(row[h0 + lane]);

                float blk_amax = warp_reduce_max(fabsf(val));
                float scale = blk_amax / dtype_max;
                if (!(scale > 0.0f)) scale = 1.0f;
                uint8_t scale_byte = e8m0_encode(scale);
                float inv_scale = e8m0_inv_decode_to_f32(scale_byte);

                if (lane == 0) {
                    nvshmem_uint8_p(dst_sfa + blk, scale_byte, dest);
                }

                float qf = val * inv_scale;
                if constexpr (kFP8) {
                    uint8_t q8 = to_fp8(qf);
                    uint8_t q8_neighbor = __shfl_xor_sync(0xFFFFFFFF, q8, 1);
                    if ((lane & 1) == 0 && lane < blk_size) {
                        uint16_t packed = (uint16_t)q8 | ((uint16_t)q8_neighbor << 8);
                        int pack_idx = blk * (SF_VEC / 2) + (lane / 2);
                        nvshmem_uint16_p(dst_pack + pack_idx, packed, dest);
                    }
                } else {
                    float qf0 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 0);
                    float qf1 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 1);
                    float qf2 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 2);
                    float qf3 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 3);
                    if ((lane & 3) == 0 && lane < blk_size) {
                        uint16_t packed = to_fp4x4(qf0, qf1, qf2, qf3);
                        int pack_idx = blk * (SF_VEC / 4) + (lane / 4);
                        nvshmem_uint16_p(dst_pack + pack_idx, packed, dest);
                    }
                }
            }
        } else {
            // Intra-node: use IPC
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl_rank]);
            uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
            uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(dest_buf + ipc_sfa_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) {
                    atomicAdd(nvshmem_dropped, 1);
                }
                continue;
            }

            if (lane == 0) {
                Meta m{rid_base + static_cast<int64_t>(i), local_eid, gate};
                if (is_remote_gpu) {
                    int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                    int4 meta_val = *reinterpret_cast<const int4*>(&m);
                    st_na_v4_s32(meta_dst, meta_val);
                } else {
                    meta_buf[slot_r] = m;
                }
            }

            uint16_t* dst_pack = x_buf + (int64_t)slot_r * Hp;
            uint8_t* dst_sfa = sfa_buf + (int64_t)slot_r * Hsf;

            for (int blk = 0; blk < Hsf; blk++) {
                int h0 = blk * SF_VEC;
                int h_end = min(h0 + SF_VEC, H);
                int blk_size = h_end - h0;

                float val = 0.0f;
                if (lane < blk_size) val = __bfloat162float(row[h0 + lane]);

                float blk_amax = warp_reduce_max(fabsf(val));
                float scale = blk_amax / dtype_max;
                if (!(scale > 0.0f)) scale = 1.0f;
                uint8_t scale_byte = e8m0_encode(scale);
                float inv_scale = e8m0_inv_decode_to_f32(scale_byte);

                if (lane == 0) {
                    if (is_remote_gpu) st_na_relaxed_gpu_b8(dst_sfa + blk, scale_byte);
                    else dst_sfa[blk] = scale_byte;
                }

                float qf = val * inv_scale;
                if constexpr (kFP8) {
                    uint8_t q8 = to_fp8(qf);
                    uint8_t q8_neighbor = __shfl_xor_sync(0xFFFFFFFF, q8, 1);
                    if ((lane & 1) == 0 && lane < blk_size) {
                        uint16_t packed = (uint16_t)q8 | ((uint16_t)q8_neighbor << 8);
                        int pack_idx = blk * (SF_VEC / 2) + (lane / 2);
                        if (is_remote_gpu) st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                        else dst_pack[pack_idx] = packed;
                    }
                } else {
                    float qf0 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 0);
                    float qf1 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 1);
                    float qf2 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 2);
                    float qf3 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 3);
                    if ((lane & 3) == 0 && lane < blk_size) {
                        uint16_t packed = to_fp4x4(qf0, qf1, qf2, qf3);
                        int pack_idx = blk * (SF_VEC / 4) + (lane / 4);
                        if (is_remote_gpu) st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                        else dst_pack[pack_idx] = packed;
                    }
                }
            }
        }
    }
    // Ensure producer writes are globally ordered before subsequent barrier kernels.
    fence_acq_rel_sys();
}

__global__ void k_merge_nvshmem_into_ipc_blockscaled(
    const uint16_t* __restrict__ nv_x_buf,
    const uint8_t* __restrict__ nv_sfa,
    const Meta* __restrict__ nv_meta,
    uint16_t* __restrict__ ipc_x_buf,
    uint8_t* __restrict__ ipc_sfa,
    Meta* __restrict__ ipc_meta,
    int ipc_base,
    int nv_count,
    int Hp, int Hsf)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        int out_i = ipc_base + i;
        if (lane == 0) {
            ipc_meta[out_i] = nv_meta[i];
        }
        const uint16_t* src = nv_x_buf + (int64_t)i * Hp;
        uint16_t* dst = ipc_x_buf + (int64_t)out_i * Hp;
        for (int hp = lane; hp < Hp; hp += 32) {
            dst[hp] = src[hp];
        }

        const uint8_t* sfa_src = nv_sfa + (int64_t)i * Hsf;
        uint8_t* sfa_dst = ipc_sfa + (int64_t)out_i * Hsf;
        for (int sf = lane; sf < Hsf; sf += 32) {
            sfa_dst[sf] = sfa_src[sf];
        }
    }
}

// Dynamic merge: reads both IPC/NVSHMEM counters on device and appends
// NVSHMEM rows after current IPC rows without host-side count reads.
__global__ void k_merge_nvshmem_into_ipc_blockscaled_dynamic(
    const uint16_t* __restrict__ nv_x_buf,
    const uint8_t* __restrict__ nv_sfa,
    const Meta* __restrict__ nv_meta,
    uint16_t* __restrict__ ipc_x_buf,
    uint8_t* __restrict__ ipc_sfa,
    Meta* __restrict__ ipc_meta,
    int* __restrict__ ipc_counter_ptr,
    const int* __restrict__ nv_counter_ptr,
    int* __restrict__ dropped_ptr,
    int capacity,
    int Hp,
    int Hsf)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    int ipc_base = ld_nc_s32(ipc_counter_ptr);
    if (ipc_base < 0) ipc_base = 0;
    if (ipc_base > capacity) ipc_base = capacity;
    if (warp_id == 0 && lane == 0) {
        // Clamp destination IPC counter eagerly so all early-exit paths leave it bounded.
        *ipc_counter_ptr = ipc_base;
    }
    int nv_count = ld_nc_s32(nv_counter_ptr);
    if (nv_count <= 0) return;
    if (nv_count > capacity) {
        nv_count = capacity;
    }
    const int free_slots = capacity - ipc_base;
    if (free_slots <= 0) {
        if (warp_id == 0 && lane == 0 && dropped_ptr) atomicAdd(dropped_ptr, nv_count);
        return;
    }
    int dropped_rows = 0;
    if (nv_count > free_slots) {
        dropped_rows = nv_count - free_slots;
        nv_count = free_slots;
    }
    if (dropped_rows > 0 && warp_id == 0 && lane == 0 && dropped_ptr) {
        atomicAdd(dropped_ptr, dropped_rows);
    }

    for (int i = warp_id; i < nv_count; i += num_warps) {
        int out_i = ipc_base + i;
        if (lane == 0) {
            ipc_meta[out_i] = nv_meta[i];
        }
        const uint16_t* src = nv_x_buf + (int64_t)i * Hp;
        uint16_t* dst = ipc_x_buf + (int64_t)out_i * Hp;
        for (int hp = lane; hp < Hp; hp += 32) {
            dst[hp] = src[hp];
        }

        const uint8_t* sfa_src = nv_sfa + (int64_t)i * Hsf;
        uint8_t* sfa_dst = ipc_sfa + (int64_t)out_i * Hsf;
        for (int sf = lane; sf < Hsf; sf += 32) {
            sfa_dst[sf] = sfa_src[sf];
        }
    }
    if (warp_id == 0 && lane == 0) {
        int total = ipc_base + nv_count;
        if (total > capacity) total = capacity;
        *ipc_counter_ptr = total;
    }
}

__global__ void k_gather_blockscaled_hybrid(
    const uint16_t* __restrict__ x_recv,
    const uint8_t* __restrict__ sfa_recv,
    const int* __restrict__ order,
    const int* __restrict__ dest,
    uint16_t* __restrict__ Xe_out,
    uint8_t* __restrict__ sfa_out_mma,
    int M, int M_pad, int Hp, int Hsf)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        int out_i = (dest != nullptr) ? dest[sorted_i] : sorted_i;

        const uint16_t* src = x_recv + (int64_t)orig_i * Hp;
        uint16_t* dst = Xe_out + (int64_t)out_i * Hp;
        for (int hp = lane; hp < Hp; hp += 32) {
            dst[hp] = src[hp];
        }

        const uint8_t* sfa_src = sfa_recv + (int64_t)orig_i * Hsf;
        for (int sf = lane; sf < Hsf; sf += 32) {
            const size_t dst_off = ::nmoe::cutlass_sf_swizzle_offset(
                static_cast<size_t>(out_i),
                static_cast<size_t>(sf),
                static_cast<uint32_t>(M_pad),
                static_cast<uint32_t>(Hsf));
            sfa_out_mma[dst_off] = sfa_src[sf];
        }
    }
}

__global__ void k_gather_meta_sorted_hybrid(
    const Meta* __restrict__ meta,
    const int* __restrict__ order,
    int64_t* __restrict__ row_id_out,
    float* __restrict__ gate_out,
    int M)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M;
         i += blockDim.x * gridDim.x) {
        int orig_i = order[i];
        Meta m = meta[orig_i];
        row_id_out[i] = m.row_id;
        gate_out[i] = m.gate;
    }
}

// ============================================================================
// Helper Kernels for Sort/Gather (same as rdep.cu)
// ============================================================================

__global__ void k_extract_local_eid_hybrid(
    const Meta* meta, int* local_eid, int* order, int M)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M;
         i += blockDim.x * gridDim.x) {
        local_eid[i] = meta[i].local_eid;
        order[i] = i;
    }
}

__global__ void k_compute_offsets_hybrid(
    const int* sorted_eid, int* offsets, int M, int n_local)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x;
         e <= n_local;
         e += blockDim.x * gridDim.x) {
        int lo = 0;
        int hi = M;
        while (lo < hi) {
            const int mid = lo + ((hi - lo) >> 1);
            if (sorted_eid[mid] < e) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        offsets[e] = lo;
    }
}

__global__ void k_compute_padded_prefix_hybrid(
    const int* __restrict__ offsets,
    int* __restrict__ offs_pad,
    int* __restrict__ M_pad_out,
    int n_local, int align,
    int override_total)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int sum = 0;
    for (int e = 0; e < n_local; ++e) {
        const int c = offsets[e + 1] - offsets[e];
        const int c_pad = ((c + align - 1) / align) * align;
        sum += c_pad;
        offs_pad[e] = sum;
    }
    if (override_total > 0 && n_local > 0) {
        offs_pad[n_local - 1] = override_total;
        sum = override_total;
    }
    *M_pad_out = sum;
}

__global__ void k_compute_offsets_and_padded_prefix_hybrid(
    const int* __restrict__ sorted_eid,
    int* __restrict__ offsets,
    int* __restrict__ offs_pad,
    int* __restrict__ M_pad_out,
    int M, int n_local, int align,
    int override_total)
{
    if (blockIdx.x != 0) return;
    for (int e = threadIdx.x; e <= n_local; e += blockDim.x) {
        int lo = 0;
        int hi = M;
        while (lo < hi) {
            const int mid = lo + ((hi - lo) >> 1);
            if (sorted_eid[mid] < e) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        offsets[e] = lo;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int e = 0; e < n_local; ++e) {
            const int c = offsets[e + 1] - offsets[e];
            const int c_pad = ((c + align - 1) / align) * align;
            sum += c_pad;
            offs_pad[e] = sum;
        }
        if (override_total > 0 && n_local > 0) {
            offs_pad[n_local - 1] = override_total;
            sum = override_total;
        }
        if (M_pad_out != nullptr) {
            *M_pad_out = sum;
        }
    }
}

__global__ void k_fill_dest_from_sorted_eid_hybrid(
    const int* __restrict__ sorted_eid,
    const int* __restrict__ offsets,
    const int* __restrict__ offs_pad,
    int* __restrict__ dest,
    int M, int n_local)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M;
         i += blockDim.x * gridDim.x) {
        const int e = sorted_eid[i];
        if (e >= 0 && e < n_local) {
            const int pad_start = (e == 0) ? 0 : offs_pad[e - 1];
            dest[i] = pad_start + (i - offsets[e]);
        }
    }
}

__global__ void k_init_single_expert_layout_hybrid(
    int* __restrict__ order,
    int* __restrict__ dest,
    int* __restrict__ offsets,
    int* __restrict__ offs_pad,
    int* __restrict__ M_pad_out,
    int M_recv,
    int M_pad)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M_recv;
         i += blockDim.x * gridDim.x) {
        order[i] = i;
        dest[i] = i;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        offsets[0] = 0;
        offsets[1] = M_recv;
        offs_pad[0] = M_pad;
        if (M_pad_out != nullptr) {
            *M_pad_out = M_pad;
        }
    }
}

// Clear only padded gap rows introduced by per-expert alignment.
__global__ void k_zero_bf16_padding_rows_hybrid(
    const int* __restrict__ offsets,   // [n_local + 1]
    const int* __restrict__ offs_pad,  // [n_local]
    __nv_bfloat16* __restrict__ out,   // [M_pad, H]
    int n_local,
    int H)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const __nv_bfloat16 z = __float2bfloat16(0.0f);

    for (int e = warp_id; e < n_local; e += num_warps) {
        const int cnt = offsets[e + 1] - offsets[e];
        const int pad_start = (e == 0) ? 0 : offs_pad[e - 1];
        const int gap_start = pad_start + cnt;
        const int pad_end = offs_pad[e];
        for (int row = gap_start; row < pad_end; ++row) {
            __nv_bfloat16* row_ptr = out + static_cast<int64_t>(row) * H;
            for (int h = lane; h < H; h += 32) {
                row_ptr[h] = z;
            }
        }
    }
}

// Fill only blockscaled padding rows:
//   q: zeros
//   sf: optional 127 (E8M0 scale=1.0) when sf_out != nullptr
__global__ void k_fill_blockscaled_padding_rows_hybrid(
    const int* __restrict__ offsets,   // [n_local + 1]
    const int* __restrict__ offs_pad,  // [n_local]
    uint16_t* __restrict__ q_out,      // [M_pad, Hp]
    uint8_t* __restrict__ sf_out,      // [M_pad, Hsf]
    int n_local,
    int Hp,
    int Hsf)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int e = warp_id; e < n_local; e += num_warps) {
        const int cnt = offsets[e + 1] - offsets[e];
        const int pad_start = (e == 0) ? 0 : offs_pad[e - 1];
        const int gap_start = pad_start + cnt;
        const int pad_end = offs_pad[e];
        for (int row = gap_start; row < pad_end; ++row) {
            uint16_t* q_row = q_out + static_cast<int64_t>(row) * Hp;
            for (int h = lane; h < Hp; h += 32) {
                q_row[h] = 0;
            }
            if (sf_out != nullptr) {
                uint8_t* sf_row = sf_out + static_cast<int64_t>(row) * Hsf;
                for (int sf = lane; sf < Hsf; sf += 32) {
                    sf_row[sf] = static_cast<uint8_t>(127);
                }
            }
        }
    }
}

// Fill only blockscaled SF padding rows in CUTLASS MMA-swizzled layout.
__global__ void k_fill_blockscaled_padding_sf_swizzled_hybrid(
    const int* __restrict__ offsets,   // [n_local + 1]
    const int* __restrict__ offs_pad,  // [n_local]
    uint8_t* __restrict__ sf_out_mma,  // [M_pad, Hsf] (swizzled layout)
    int n_local,
    int M_pad,
    int Hsf)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const uint8_t one_sf = static_cast<uint8_t>(127);

    for (int e = warp_id; e < n_local; e += num_warps) {
        const int cnt = offsets[e + 1] - offsets[e];
        const int pad_start = (e == 0) ? 0 : offs_pad[e - 1];
        const int gap_start = pad_start + cnt;
        const int pad_end = offs_pad[e];
        for (int row = gap_start; row < pad_end; ++row) {
            for (int sf = lane; sf < Hsf; sf += 32) {
                const size_t dst_off = ::nmoe::cutlass_sf_swizzle_offset(
                    static_cast<size_t>(row),
                    static_cast<size_t>(sf),
                    static_cast<uint32_t>(M_pad),
                    static_cast<uint32_t>(Hsf));
                sf_out_mma[dst_off] = one_sf;
            }
        }
    }
}

__host__ __forceinline__ void zero_bf16_padding_rows_hybrid_async(
    const int* offsets,
    const int* offs_pad,
    void* out,
    int n_local,
    int M_recv,
    int M_pad,
    int H,
    cudaStream_t stream)
{
    if (out == nullptr || offsets == nullptr || offs_pad == nullptr) return;
    if (n_local <= 0 || M_pad <= M_recv || H <= 0) return;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_by_work = std::max(1, (n_local + warps_per_block - 1) / warps_per_block);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_zero_bf16_padding_rows_hybrid<<<blocks, threads, 0, stream>>>(
        offsets,
        offs_pad,
        static_cast<__nv_bfloat16*>(out),
        n_local,
        H);
    CUDA_CHECK(cudaGetLastError());
}

__host__ __forceinline__ void fill_blockscaled_padding_rows_hybrid_async(
    const int* offsets,
    const int* offs_pad,
    void* q_out,
    void* sf_out,
    int n_local,
    int M_recv,
    int M_pad,
    int Hp,
	    int Hsf,
	    cudaStream_t stream)
{
	    if (q_out == nullptr || offsets == nullptr || offs_pad == nullptr) return;
	    if (n_local <= 0 || M_pad <= M_recv || Hp <= 0 || Hsf <= 0) return;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_by_work = std::max(1, (n_local + warps_per_block - 1) / warps_per_block);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_fill_blockscaled_padding_rows_hybrid<<<blocks, threads, 0, stream>>>(
        offsets,
        offs_pad,
        static_cast<uint16_t*>(q_out),
        static_cast<uint8_t*>(sf_out),
        n_local,
        Hp,
        Hsf);
    CUDA_CHECK(cudaGetLastError());
}

__host__ __forceinline__ void fill_blockscaled_padding_sf_swizzled_hybrid_async(
    const int* offsets,
    const int* offs_pad,
    void* sf_out,
    int n_local,
    int M_recv,
    int M_pad,
    int Hsf,
    cudaStream_t stream)
{
    if (sf_out == nullptr || offsets == nullptr || offs_pad == nullptr) return;
    if (n_local <= 0 || M_pad <= M_recv || Hsf <= 0) return;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_by_work = std::max(1, (n_local + warps_per_block - 1) / warps_per_block);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_fill_blockscaled_padding_sf_swizzled_hybrid<<<blocks, threads, 0, stream>>>(
        offsets,
        offs_pad,
        static_cast<uint8_t*>(sf_out),
        n_local,
        M_pad,
        Hsf);
    CUDA_CHECK(cudaGetLastError());
}

void zero_bf16_padding_rows_hybrid(
    void* out,
    const int* offs_pad,
    int M_recv,
    int M_pad,
    int H,
    cudaStream_t stream)
{
    zero_bf16_padding_rows_hybrid_async(
        g_nvshmem.offsets,
        offs_pad,
        out,
        g_nvshmem.n_local,
        M_recv,
        M_pad,
        H,
        stream);
}

__global__ void k_gather_bf16_hybrid(
    const uint16_t* src_buf, const int* order, const int* dest,
    __nv_bfloat16* out, int M, int H, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_idx = order[i];
        int dst_idx = dest[i];

        const uint16_t* src_row = src_buf + (int64_t)src_idx * Ha;
        __nv_bfloat16* dst_row = out + (int64_t)dst_idx * H;
        uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(dst_row);
        const bool vec_ok =
            ((reinterpret_cast<uintptr_t>(src_row) | reinterpret_cast<uintptr_t>(dst_u16)) & 0x7u) == 0u;

        if (vec_ok) {
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    *reinterpret_cast<int2*>(dst_u16 + h) =
                        ld_nc_v2_s32(reinterpret_cast<const int2*>(src_row + h));
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        dst_u16[hh] = src_row[hh];
                    }
                }
            }
        } else {
            for (int h = lane; h < H; h += 32) {
                dst_u16[h] = src_row[h];
            }
        }
    }
}

// ============================================================================
// Host API: Hybrid Dispatch (BF16)
// ============================================================================

int dispatch_hybrid_bf16(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K,
    int align,
    void* Xe_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    void** ipc_buffer_ptrs,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return -1;
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid bf16 requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_bf16()\n");
        return -2;
    }
    g_nvshmem.offs_pad = offs_pad_out;

    int M = T * K;
    int capacity = static_cast<int>(g_nvshmem.capacity);

    // Reset NVSHMEM counters
    reset_dispatch_counters(stream);

    // Reset IPC counter (local buffer)
    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    CUDA_CHECK(cudaMemsetAsync(local_counter, 0, sizeof(int), stream));

    // Global barrier: ensure all ranks reset counters before any remote atomicAdd.
    hybrid_barrier_on_stream(stream);

    // Launch hybrid dispatch kernel
    int threads = 256;
    int warps_needed = M;
    int warps_per_block = threads / 32;
    int blocks_by_work = std::max(1, (warps_needed + warps_per_block - 1) / warps_per_block);
    int blocks = cap_warp_stride_blocks(blocks_by_work);

    // CRITICAL: Pass device pointer array (d_ipc_buffer_ptrs) to kernel, NOT host array.
    // The kernel cannot dereference host pointers - it needs device-accessible pointers.
    if (M > 0) {
        k_dispatch_hybrid_bf16<<<blocks, threads, 0, stream>>>(
            x, eids, gates,
            g_nvshmem.rank, T, g_nvshmem.H, g_nvshmem.Ha, K,
            g_nvshmem.n_local, capacity,
            g_nvshmem.local_world, g_nvshmem.num_nodes, g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
            g_nvshmem.x_buf_bf16,
            static_cast<Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.dropped,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_meta_off,
            ipc_counter_off);
        CUDA_CHECK(cudaGetLastError());
    }

		    // Ensure all NVSHMEM puts are complete and globally visible before counting/sorting.
    hybrid_barrier_on_stream(stream);

    // Forward inter-node receives (NVSHMEM proxy mailbox) to their true
    // destination GPU via IPC. Skip this stage on single-node jobs.
    if (g_nvshmem.num_nodes > 1) {
        const int64_t recv_upper_i64 = std::min<int64_t>(
            static_cast<int64_t>(capacity),
            static_cast<int64_t>(T) * static_cast<int64_t>(K) * static_cast<int64_t>(g_nvshmem.world));
        int f_threads = 256;
        int f_warps_per_block = f_threads / 32;
        int f_work = std::max(1, static_cast<int>(recv_upper_i64));
        int f_blocks_by_work = std::max(1, (f_work + f_warps_per_block - 1) / f_warps_per_block);
        int f_blocks = cap_warp_stride_blocks(f_blocks_by_work);
        k_forward_nvshmem_dispatch_to_ipc_bf16_dynamic<<<f_blocks, f_threads, 0, stream>>>(
            g_nvshmem.x_buf_bf16,
            static_cast<const Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.dropped,
            g_nvshmem.H, g_nvshmem.Ha,
            capacity,
            g_nvshmem.nvl_rank,
            g_nvshmem.local_world,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_meta_off,
            ipc_counter_off);
        CUDA_CHECK(cudaGetLastError());
        // Ensure local IPC mailbox writes are globally visible before count read.
        ipc_barrier_on_stream(stream);
    }

            // Pre-arm async D2H read so ready-query may complete without blocking.
            (void)poll_device_int_async(local_counter, stream);

		    // Count total received rows (IPC only; inter-node rows were forwarded into IPC buffers).
		    if (M_pad_out == nullptr) {
		        fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
		        return -3;
		    }
		    bool recv_ok = false;
		    int M_recv = read_device_int_blocking(local_counter, stream, &recv_ok);
		    if (!recv_ok) {
		        return -3;
		    }
		    *M_pad_out = M_recv;
		    if (M_recv <= 0) {
		        CUDA_CHECK(cudaMemsetAsync(offs_pad_out, 0, g_nvshmem.n_local * sizeof(int), stream));
		        *M_pad_out = 0;
		        return 0;
		    }
		    if (M_recv > capacity) {
		        fprintf(stderr, "RDEP ERROR: hybrid BF16 dispatch overflow: M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
		        return -3;
		    }

		    Meta* meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
		    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf);

		    // Extract/sort offsets + padded mapping.
		    int extract_threads = 256;
		    int extract_blocks_by_work = std::max(1, (M_recv + extract_threads - 1) / extract_threads);
		    int extract_blocks = cap_warp_stride_blocks(extract_blocks_by_work);
	    int M_pad = 0;
	    if (g_nvshmem.n_local == 1) {
	        M_pad = ((M_recv + align - 1) / align) * align;
	        k_init_single_expert_layout_hybrid<<<extract_blocks, extract_threads, 0, stream>>>(
	            g_nvshmem.order, g_nvshmem.dest,
	            g_nvshmem.offsets, offs_pad_out, g_nvshmem.M_pad_dev,
	            M_recv, M_pad);
	        CUDA_CHECK(cudaGetLastError());
	    } else {
	        k_extract_local_eid_hybrid<<<extract_blocks, extract_threads, 0, stream>>>(
	            meta_buf, g_nvshmem.local_eid, g_nvshmem.order, M_recv);
	        CUDA_CHECK(cudaGetLastError());

	        if (g_nvshmem.n_local > 1 && M_recv > 1) {
	            const int sort_end_bit = radix_sort_end_bit_for_range(g_nvshmem.n_local);
	            cub::DeviceRadixSort::SortPairs(g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes,
	                g_nvshmem.local_eid, g_nvshmem.local_eid,
	                g_nvshmem.order, g_nvshmem.order,
	                M_recv, 0, sort_end_bit, stream);
	        }

	        int M_pad_bound = M_recv + g_nvshmem.n_local * (align - 1);
	        M_pad = (M_pad_bound / align) * align;
	        k_compute_offsets_and_padded_prefix_hybrid<<<1, 256, 0, stream>>>(
	            g_nvshmem.local_eid, g_nvshmem.offsets, offs_pad_out, g_nvshmem.M_pad_dev,
	            M_recv, g_nvshmem.n_local, align, M_pad);
	        k_fill_dest_from_sorted_eid_hybrid<<<extract_blocks, extract_threads, 0, stream>>>(
	            g_nvshmem.local_eid, g_nvshmem.offsets, offs_pad_out, g_nvshmem.dest, M_recv, g_nvshmem.n_local);
	        CUDA_CHECK(cudaGetLastError());
	    }

	    // Avoid a second host sync for exact M_pad. Use deterministic aligned bound.
	    // Prefix kernel writes bound to offs_pad[n_local - 1].
    *M_pad_out = M_pad;
    if (Xe_out != nullptr) {
        if (g_nvshmem.n_local == 1 && M_pad > M_recv) {
            const size_t tail_rows = static_cast<size_t>(M_pad - M_recv);
            __nv_bfloat16* tail_ptr =
                static_cast<__nv_bfloat16*>(Xe_out) + static_cast<size_t>(M_recv) * static_cast<size_t>(g_nvshmem.H);
            CUDA_CHECK(cudaMemsetAsync(
                tail_ptr,
                0,
                tail_rows * static_cast<size_t>(g_nvshmem.H) * sizeof(__nv_bfloat16),
                stream));
        } else {
            zero_bf16_padding_rows_hybrid_async(
                g_nvshmem.offsets,
                offs_pad_out,
                Xe_out,
                g_nvshmem.n_local,
                M_recv,
                M_pad,
                g_nvshmem.H,
                stream);
        }

        // Gather sorted rows into output
        int gather_threads = 256;
        int gather_warps_per_block = gather_threads / 32;
        int gather_blocks_by_work = std::max(1, (M_recv + gather_warps_per_block - 1) / gather_warps_per_block);
        int gather_blocks = cap_warp_stride_blocks(gather_blocks_by_work);
        k_gather_bf16_hybrid<<<gather_blocks, gather_threads, 0, stream>>>(
            x_buf, g_nvshmem.order, g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(Xe_out),
            M_recv, g_nvshmem.H, g_nvshmem.Ha);
        CUDA_CHECK(cudaGetLastError());
    }

    // Copy dest to output if requested
    if (dest_out) {
        CUDA_CHECK(cudaMemcpyAsync(dest_out, g_nvshmem.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }

	    if (row_id_out && gate_out) {
	        int meta_threads = 256;
	        int meta_blocks_by_work = std::max(1, (M_recv + meta_threads - 1) / meta_threads);
	        int meta_blocks = cap_warp_stride_blocks(meta_blocks_by_work);
	        k_gather_meta_sorted_hybrid<<<meta_blocks, meta_threads, 0, stream>>>(
	            meta_buf, g_nvshmem.order, row_id_out, gate_out, M_recv);
	        CUDA_CHECK(cudaGetLastError());
	    }

    return M_recv;
}

// ============================================================================
// Blockscaled Hybrid Dispatch (stub - similar structure)
// ============================================================================

int dispatch_hybrid_blockscaled(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K,
    void* Xe_q_out,
    void* Xe_sf_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    void** ipc_buffer_ptrs,
    size_t ipc_x_off,
    size_t ipc_sfa_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) return -1;

    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid blockscaled requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_blockscaled()\n");
        return -2;
    }
    g_nvshmem.offs_pad = offs_pad_out;

    int M = T * K;
    int capacity = static_cast<int>(g_nvshmem.capacity);

    // Reset counters
    reset_dispatch_counters(stream);

    // Reset IPC counter (local buffer)
    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + ipc_x_off);
    uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(local_ipc_buf + ipc_sfa_off);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
    cudaMemsetAsync(local_counter, 0, sizeof(int), stream);

    // Global barrier: ensure all ranks reset counters before any remote atomicAdd.
    hybrid_barrier_on_stream(stream);

    // Launch hybrid dispatch kernel
    int threads = 256;
    int warps_needed = M;
    int warps_per_block = threads / 32;
    int blocks_by_work = std::max(1, (warps_needed + warps_per_block - 1) / warps_per_block);
    int blocks = cap_warp_stride_blocks(blocks_by_work);

    if (M > 0) {
        if (g_nvshmem.profile == 0) {
            k_dispatch_hybrid_blockscaled<true><<<blocks, threads, 0, stream>>>(
                x, eids, gates,
                g_nvshmem.rank, T, g_nvshmem.H, g_nvshmem.Hp, g_nvshmem.Hsf, K,
                g_nvshmem.n_local, capacity,
                g_nvshmem.local_world, g_nvshmem.num_nodes, g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
                g_nvshmem.x_buf_block,
                g_nvshmem.sfa_buf,
                static_cast<Meta*>(g_nvshmem.meta),
                g_nvshmem.counter,
                g_nvshmem.dropped,
                g_nvshmem.d_ipc_buffer_ptrs,
                ipc_sfa_off,
                ipc_meta_off,
                ipc_counter_off);
        } else if (g_nvshmem.profile == 1) {
            k_dispatch_hybrid_blockscaled<false><<<blocks, threads, 0, stream>>>(
                x, eids, gates,
                g_nvshmem.rank, T, g_nvshmem.H, g_nvshmem.Hp, g_nvshmem.Hsf, K,
                g_nvshmem.n_local, capacity,
                g_nvshmem.local_world, g_nvshmem.num_nodes, g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
                g_nvshmem.x_buf_block,
                g_nvshmem.sfa_buf,
                static_cast<Meta*>(g_nvshmem.meta),
                g_nvshmem.counter,
                g_nvshmem.dropped,
                g_nvshmem.d_ipc_buffer_ptrs,
                ipc_sfa_off,
                ipc_meta_off,
                ipc_counter_off);
        } else {
            fprintf(stderr, "RDEP ERROR: invalid hybrid blockscaled profile=%d\n", g_nvshmem.profile);
            return -3;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Ensure all NVSHMEM puts are complete and globally visible before counting/sorting.
    hybrid_barrier_on_stream(stream);

    // Merge NVSHMEM receives into local IPC buffer [ipc_recv .. ipc_recv+nv_recv)
    // with device-side count reads and capacity clamping (no host D2H counts).
    if (g_nvshmem.num_nodes > 1) {
        const int64_t recv_upper_i64 = std::min<int64_t>(
            static_cast<int64_t>(capacity),
            static_cast<int64_t>(T) * static_cast<int64_t>(K) * static_cast<int64_t>(g_nvshmem.world));
        int merge_threads = 256;
        int merge_warps_per_block = merge_threads / 32;
        int merge_work = std::max(1, static_cast<int>(recv_upper_i64));
        int merge_blocks_by_work = std::max(1, (merge_work + merge_warps_per_block - 1) / merge_warps_per_block);
        int merge_blocks = cap_warp_stride_blocks(merge_blocks_by_work);
        k_merge_nvshmem_into_ipc_blockscaled_dynamic<<<merge_blocks, merge_threads, 0, stream>>>(
            g_nvshmem.x_buf_block,
            g_nvshmem.sfa_buf,
            static_cast<const Meta*>(g_nvshmem.meta),
            x_buf,
            sfa_buf,
            meta_buf,
            local_counter,
            g_nvshmem.counter,
            g_nvshmem.dropped,
            capacity,
            g_nvshmem.Hp,
            g_nvshmem.Hsf);
        CUDA_CHECK(cudaGetLastError());
    }

            // Pre-arm async D2H read so ready-query may complete without blocking.
            (void)poll_device_int_async(local_counter, stream);

		    if (M_pad_out == nullptr) {
		        fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
		        return -3;
		    }
		    bool recv_ok = false;
		    int M_recv = read_device_int_blocking(local_counter, stream, &recv_ok);
		    if (!recv_ok) {
		        return -3;
		    }
		    *M_pad_out = M_recv;
	    if (M_recv <= 0) {
	        CUDA_CHECK(cudaMemsetAsync(offs_pad_out, 0, g_nvshmem.n_local * sizeof(int), stream));
	        *M_pad_out = 0;
        return 0;
    }
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP ERROR: hybrid blockscaled dispatch overflow: M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        return -3;
    }

	    int extract_threads = 256;
	    int extract_blocks_by_work = std::max(1, (M_recv + extract_threads - 1) / extract_threads);
	    int extract_blocks = cap_warp_stride_blocks(extract_blocks_by_work);
	    int M_pad = 0;
	    if (g_nvshmem.n_local == 1) {
	        M_pad = ((M_recv + g_nvshmem.align - 1) / g_nvshmem.align) * g_nvshmem.align;
	        k_init_single_expert_layout_hybrid<<<extract_blocks, extract_threads, 0, stream>>>(
	            g_nvshmem.order, g_nvshmem.dest,
	            g_nvshmem.offsets, offs_pad_out, g_nvshmem.M_pad_dev,
	            M_recv, M_pad);
	        CUDA_CHECK(cudaGetLastError());
	    } else {
	        k_extract_local_eid_hybrid<<<extract_blocks, extract_threads, 0, stream>>>(
	            meta_buf, g_nvshmem.local_eid, g_nvshmem.order, M_recv);

	        if (g_nvshmem.n_local > 1 && M_recv > 1) {
	            const int sort_end_bit = radix_sort_end_bit_for_range(g_nvshmem.n_local);
	            cub::DeviceRadixSort::SortPairs(g_nvshmem.sort_temp, g_nvshmem.sort_temp_bytes,
	                g_nvshmem.local_eid, g_nvshmem.local_eid,
	                g_nvshmem.order, g_nvshmem.order,
	                M_recv, 0, sort_end_bit, stream);
	        }

	        int M_pad_bound = M_recv + g_nvshmem.n_local * (g_nvshmem.align - 1);
	        M_pad = (M_pad_bound / g_nvshmem.align) * g_nvshmem.align;
	        k_compute_offsets_and_padded_prefix_hybrid<<<1, 256, 0, stream>>>(
	            g_nvshmem.local_eid, g_nvshmem.offsets, offs_pad_out, g_nvshmem.M_pad_dev,
	            M_recv, g_nvshmem.n_local, g_nvshmem.align, M_pad);
	        k_fill_dest_from_sorted_eid_hybrid<<<extract_blocks, extract_threads, 0, stream>>>(
	            g_nvshmem.local_eid, g_nvshmem.offsets, offs_pad_out, g_nvshmem.dest, M_recv, g_nvshmem.n_local);
	        CUDA_CHECK(cudaGetLastError());
	    }

	    // Avoid exact M_pad D2H/sync in hot path; use deterministic aligned bound.
	    // Prefix kernel writes bound to offs_pad[n_local - 1].
    *M_pad_out = M_pad;
    if (dest_out) {
        CUDA_CHECK(cudaMemcpyAsync(dest_out, g_nvshmem.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    }

    // Optional output materialization: meta-only mode passes Xe_q_out/Xe_sf_out as nullptr.
    if (Xe_q_out && Xe_sf_out) {
        if (g_nvshmem.n_local == 1 && M_pad > M_recv) {
            const size_t tail_rows = static_cast<size_t>(M_pad - M_recv);
            uint16_t* q_tail =
                static_cast<uint16_t*>(Xe_q_out) + static_cast<size_t>(M_recv) * static_cast<size_t>(g_nvshmem.Hp);
            uint8_t* sf_tail =
                static_cast<uint8_t*>(Xe_sf_out) + static_cast<size_t>(M_recv) * static_cast<size_t>(g_nvshmem.Hsf);
            CUDA_CHECK(cudaMemsetAsync(
                q_tail,
                0,
                tail_rows * static_cast<size_t>(g_nvshmem.Hp) * sizeof(uint16_t),
                stream));
            CUDA_CHECK(cudaMemsetAsync(
                sf_tail,
                0,
                tail_rows * static_cast<size_t>(g_nvshmem.Hsf) * sizeof(uint8_t),
                stream));
        } else {
            fill_blockscaled_padding_rows_hybrid_async(
                g_nvshmem.offsets,
                offs_pad_out,
                Xe_q_out,
                nullptr,
                g_nvshmem.n_local,
                M_recv,
                M_pad,
                g_nvshmem.Hp,
                g_nvshmem.Hsf,
                stream);
            fill_blockscaled_padding_sf_swizzled_hybrid_async(
                g_nvshmem.offsets,
                offs_pad_out,
                Xe_sf_out,
                g_nvshmem.n_local,
                M_recv,
                M_pad,
                g_nvshmem.Hsf,
                stream);
        }

        // Gather packed activations and write SF directly in CUTLASS MMA layout.
        int gather_threads = 256;
        int gather_warps_per_block = gather_threads / 32;
        int gather_blocks_by_work = std::max(1, (M_recv + gather_warps_per_block - 1) / gather_warps_per_block);
        int gather_blocks = cap_warp_stride_blocks(gather_blocks_by_work);
        k_gather_blockscaled_hybrid<<<gather_blocks, gather_threads, 0, stream>>>(
            x_buf, sfa_buf, g_nvshmem.order, g_nvshmem.dest,
            static_cast<uint16_t*>(Xe_q_out), static_cast<uint8_t*>(Xe_sf_out),
            M_recv, M_pad, g_nvshmem.Hp, g_nvshmem.Hsf);
        CUDA_CHECK(cudaGetLastError());
    }

    if (row_id_out && gate_out) {
        int meta_threads = 256;
        int meta_blocks_by_work = std::max(1, (M_recv + meta_threads - 1) / meta_threads);
        int meta_blocks = cap_warp_stride_blocks(meta_blocks_by_work);
        k_gather_meta_sorted_hybrid<<<meta_blocks, meta_threads, 0, stream>>>(
            meta_buf, g_nvshmem.order, row_id_out, gate_out, M_recv);
    }

    return M_recv;
}

void gather_xe_hybrid_blockscaled(
    void* Xe_q_out,
    void* Xe_sf_out,
    int M_recv,
    int M_pad,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP FATAL: gather_xe_hybrid_blockscaled called before NVSHMEM init\n");
        abort();
    }
    if (g_nvshmem.profile < 0) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled requires blockscaled NVSHMEM state\n");
        abort();
    }
    if (!Xe_q_out || !Xe_sf_out) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled requires non-null outputs\n");
        abort();
    }
    if (M_recv <= 0) return;
    if (M_pad <= 0) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled invalid M_pad=%d for M_recv=%d\n", M_pad, M_recv);
        abort();
    }
    const int capacity = static_cast<int>(g_nvshmem.capacity);
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        abort();
    }
    const int max_pad = capacity + g_nvshmem.n_local * (g_nvshmem.align - 1);
    if (M_pad > max_pad) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled M_pad=%d exceeds max_pad=%d\n", M_pad, max_pad);
        abort();
    }

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + g_nvshmem.ipc_x_off);
    uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(local_ipc_buf + g_nvshmem.ipc_sfa_off);
    if (g_nvshmem.offs_pad == nullptr) {
        fprintf(stderr, "RDEP ERROR: gather_xe_hybrid_blockscaled missing offs_pad mapping from prior dispatch\n");
        abort();
    }

    fill_blockscaled_padding_rows_hybrid_async(
        g_nvshmem.offsets,
        g_nvshmem.offs_pad,
        Xe_q_out,
        nullptr,
        g_nvshmem.n_local,
        M_recv,
        M_pad,
        g_nvshmem.Hp,
        g_nvshmem.Hsf,
        stream);
    fill_blockscaled_padding_sf_swizzled_hybrid_async(
        g_nvshmem.offsets,
        g_nvshmem.offs_pad,
        Xe_sf_out,
        g_nvshmem.n_local,
        M_recv,
        M_pad,
        g_nvshmem.Hsf,
        stream);

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks_by_work = std::max(1, (M_recv + warps_per_block - 1) / warps_per_block);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_gather_blockscaled_hybrid<<<blocks, threads, 0, stream>>>(
        x_buf, sfa_buf, g_nvshmem.order, g_nvshmem.dest,
        static_cast<uint16_t*>(Xe_q_out), static_cast<uint8_t*>(Xe_sf_out),
        M_recv, M_pad, g_nvshmem.Hp, g_nvshmem.Hsf);
}

// ============================================================================
// Hybrid Return Scatter Kernel (BF16)
// ============================================================================

__global__ void k_return_scatter_hybrid_bf16(
    const __nv_bfloat16* __restrict__ Ye,    // [M_recv, H] expert outputs (sorted order)
    const int* __restrict__ order,            // [M_recv] original indices
    const Meta* __restrict__ meta,            // [capacity] metadata from dispatch
    float* __restrict__ out,                  // [T, H] local output accumulator
    int M_recv, int H, int Ha, int T, int K,
    const int my_rank, const int local_world, const int num_nodes, const int rdma_rank, const int nvl_rank,
    int capacity,
    // NVSHMEM buffers (for inter-node return)
    uint16_t* nvshmem_y_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    int* nvshmem_dropped,
    // IPC buffers (for intra-node return)
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        const int world = local_world * num_nodes;
        if (static_cast<unsigned>(src_rank) >= static_cast<unsigned>(world)) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        int src_rdma_rank = nmoe_rank_node_fast(src_rank, local_world, local_world_pow2, local_world_shift);
        int src_nvl_rank = nmoe_rank_local_fast(src_rank, local_world, local_world_pow2, local_world_mask);
        if (static_cast<unsigned>(src_nvl_rank) >= static_cast<unsigned>(local_world)) continue;
        bool is_remote_node = (src_rdma_rank != rdma_rank);
        bool is_local = (src_rank == my_rank);

        if (is_local) {
            // Local: scatter directly with gate weighting
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32) {
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
            }
        } else if (is_remote_node) {
            // Inter-node: write to proxy peer (same nvl_rank) on source node, then proxy forwards via IPC.
            const int proxy_pe = src_rdma_rank * local_world + nvl_rank;
            int slot_r;
            if (lane == 0) {
                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, proxy_pe);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) nvshmem_int_atomic_add(nvshmem_dropped, 1, proxy_pe);
                continue;
            }

            // Write metadata via NVSHMEM
            if (lane == 0) {
                const int local_eid_packed = meta_pack_local_eid_dest_nvl(/*local_eid=*/0, src_nvl_rank);
                nvshmem_meta_p(nvshmem_meta + slot_r, proxy_pe, m.row_id, local_eid_packed, m.gate);
            }

            // Write BF16 payload via NVSHMEM
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = nvshmem_y_buf + (int64_t)slot_r * Ha;

            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(y_row + h),
                                      1, proxy_pe);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh, reinterpret_cast<const uint16_t*>(y_row) + hh, 1, proxy_pe);
                    }
                }
            }
        } else {
            // Intra-node (different GPU, same node): use IPC
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + ipc_y_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            int slot_r;
            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) atomicAdd(nvshmem_dropped, 1);
                continue;
            }

            // Write metadata
            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_na_v4_s32(meta_dst, meta_val);
            }

            // Write BF16 payload via IPC
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;

            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    int2* d = reinterpret_cast<int2*>(dst + h);
                    int2 v = *reinterpret_cast<const int2*>(y_row + h);
                    st_na_v2_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_return_scatter_hybrid_bf16_from_pad(
    const __nv_bfloat16* __restrict__ Ye_pad, // [M_pad, H] expert outputs (padded)
    const int* __restrict__ dest,             // [M_recv] sorted_i -> pad_i
    const int* __restrict__ order,            // [M_recv] original indices
    const Meta* __restrict__ meta,            // [capacity] metadata from dispatch
    float* __restrict__ out,                  // [T, H] local output accumulator
    int M_recv, int H, int Ha, int T, int K,
    const int my_rank, const int local_world, const int num_nodes, const int rdma_rank, const int nvl_rank,
    int capacity,
    // NVSHMEM buffers (for inter-node return)
    uint16_t* nvshmem_y_buf,
    Meta* nvshmem_meta,
    int* nvshmem_counter,
    int* nvshmem_dropped,
    // IPC buffers (for intra-node return)
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    (void)num_nodes;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        const int pad_i = dest[sorted_i];
        if (pad_i < 0) continue;

        int orig_i = order[sorted_i];
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        const int world = local_world * num_nodes;
        if (static_cast<unsigned>(src_rank) >= static_cast<unsigned>(world)) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        int src_rdma_rank = nmoe_rank_node_fast(src_rank, local_world, local_world_pow2, local_world_shift);
        int src_nvl_rank = nmoe_rank_local_fast(src_rank, local_world, local_world_pow2, local_world_mask);
        if (static_cast<unsigned>(src_nvl_rank) >= static_cast<unsigned>(local_world)) continue;
        bool is_remote_node = (src_rdma_rank != rdma_rank);
        bool is_local = (src_rank == my_rank);

        const __nv_bfloat16* y_row = Ye_pad + (int64_t)pad_i * H;

        if (is_local) {
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32) {
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
            }
        } else if (is_remote_node) {
            const int proxy_pe = src_rdma_rank * local_world + nvl_rank;
            int slot_r;
            if (lane == 0) {
                slot_r = nvshmem_int_atomic_fetch_add(nvshmem_counter, 1, proxy_pe);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) nvshmem_int_atomic_add(nvshmem_dropped, 1, proxy_pe);
                continue;
            }

            if (lane == 0) {
                const int local_eid_packed = meta_pack_local_eid_dest_nvl(/*local_eid=*/0, src_nvl_rank);
                nvshmem_meta_p(nvshmem_meta + slot_r, proxy_pe, m.row_id, local_eid_packed, m.gate);
            }

            uint16_t* dst = nvshmem_y_buf + (int64_t)slot_r * Ha;
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(y_row + h),
                                      1, proxy_pe);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh, reinterpret_cast<const uint16_t*>(y_row) + hh, 1, proxy_pe);
                    }
                }
            }
        } else {
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + ipc_y_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

            int slot_r;
            if (lane == 0) {
                slot_r = atomicAdd(counter, 1);
            }
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) {
                if (lane == 0) atomicAdd(nvshmem_dropped, 1);
                continue;
            }

            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_na_v4_s32(meta_dst, meta_val);
            }

            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    int2 v = *reinterpret_cast<const int2*>(y_row + h);
                    int2* d = reinterpret_cast<int2*>(dst + h);
                    st_na_v2_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_forward_nvshmem_return_to_ipc_bf16(
    const uint16_t* __restrict__ nv_y_buf,   // [nv_count, Ha]
    const Meta* __restrict__ nv_meta,        // [nv_count]
    int nv_count,
    int H, int Ha,
    int capacity,
    int nvl_rank,
    int local_world,
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < nv_count; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(nv_meta + i));
        const Meta in_m = mv.m;

        const int dest_nvl = meta_unpack_dest_nvl(in_m.local_eid);
        if (dest_nvl < 0 || dest_nvl >= local_world) continue;

        char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
        uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + ipc_y_off);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

        int slot_r;
        if (lane == 0) {
            slot_r = atomicAdd(counter, 1);
        }
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r < 0 || slot_r >= capacity) continue;

        if (lane == 0) {
            Meta out_m{in_m.row_id, 0, in_m.gate};
            if (dest_nvl != nvl_rank) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&out_m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = out_m;
            }
        }

        const bool remote_gpu = (dest_nvl != nvl_rank);
        const uint16_t* src = nv_y_buf + (int64_t)i * Ha;
        uint16_t* dst = y_buf + (int64_t)slot_r * Ha;
        for (int h = lane * 4; h < H; h += 32 * 4) {
            if (h + 4 <= H) {
                int2 v = ld_nc_v2_s32(reinterpret_cast<const int2*>(src + h));
                int2* d = reinterpret_cast<int2*>(dst + h);
                if (remote_gpu) {
                    st_na_v2_s32(d, v);
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 4; hh++) {
                    uint16_t u = src[hh];
                    if (remote_gpu) {
                        st_na_relaxed_gpu_b16(dst + hh, u);
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }
    fence_acq_rel_sys();
}

// Dynamic variant that reads nv_count from device memory to avoid host D2H
// reads in the return-scatter hot path.
__global__ void k_forward_nvshmem_return_to_ipc_bf16_dynamic(
    const uint16_t* __restrict__ nv_y_buf,   // [capacity, Ha]
    const Meta* __restrict__ nv_meta,        // [capacity]
    const int* __restrict__ nv_count_ptr,    // device counter
    int* __restrict__ dropped_ptr,           // device dropped counter
    int H, int Ha,
    int capacity,
    int nvl_rank,
    int local_world,
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    int nv_count = ld_nc_s32(nv_count_ptr);
    if (nv_count <= 0) return;
    if (nv_count > capacity) {
        if (dropped_ptr && threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd(dropped_ptr, nv_count - capacity);
        }
        nv_count = capacity;
    }

    for (int i = warp_id; i < nv_count; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(nv_meta + i));
        const Meta in_m = mv.m;

        const int dest_nvl = meta_unpack_dest_nvl(in_m.local_eid);
        if (dest_nvl < 0 || dest_nvl >= local_world) continue;

        char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
        uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + ipc_y_off);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + ipc_meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + ipc_counter_off);

        int slot_r;
        if (lane == 0) {
            slot_r = atomicAdd(counter, 1);
        }
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r < 0 || slot_r >= capacity) {
            if (lane == 0 && dropped_ptr) atomicAdd(dropped_ptr, 1);
            continue;
        }

        if (lane == 0) {
            Meta out_m{in_m.row_id, 0, in_m.gate};
            if (dest_nvl != nvl_rank) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&out_m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = out_m;
            }
        }

        const bool remote_gpu = (dest_nvl != nvl_rank);
        const uint16_t* src = nv_y_buf + (int64_t)i * Ha;
        uint16_t* dst = y_buf + (int64_t)slot_r * Ha;
        for (int h = lane * 4; h < H; h += 32 * 4) {
            if (h + 4 <= H) {
                int2 v = ld_nc_v2_s32(reinterpret_cast<const int2*>(src + h));
                int2* d = reinterpret_cast<int2*>(dst + h);
                if (remote_gpu) {
                    st_na_v2_s32(d, v);
                } else {
                    *d = v;
                }
            } else {
                for (int hh = h; hh < H && hh < h + 4; hh++) {
                    uint16_t u = src[hh];
                    if (remote_gpu) {
                        st_na_relaxed_gpu_b16(dst + hh, u);
                    } else {
                        dst[hh] = u;
                    }
                }
            }
        }
    }
    fence_acq_rel_sys();
}

// Kernel to scatter received rows from return buffer to output
__global__ void k_scatter_received_hybrid_bf16(
    const uint16_t* __restrict__ y_buf,
    const Meta* __restrict__ meta,
    float* __restrict__ out,
    int M_ret, int H, int Ha, int T, int K)
{
    int lane = threadIdx.x % 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M_ret; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta + i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const uint16_t* y_row = y_buf + (int64_t)i * Ha;
        float* out_row = out + (int64_t)tok * H;

        // NOTE: y_row/meta may be written by peer GPUs (IPC) or by remote nodes
        // (NVSHMEM). Receiver-side L2 is not coherent with those writes, so use
        // non-caching loads to observe updates.
        for (int h = lane * 8; h < Ha; h += 32 * 8) {
            int4 v = ld_nc_v4_s32(reinterpret_cast<const int4*>(y_row + h));
            union BF16x8 {
                int4 v;
                uint16_t u[8];
            };
            BF16x8 x;
            x.v = v;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh < H) {
                    const __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&x.u[j]);
                    atomicAdd(out_row + hh, __bfloat162float(bf) * m.gate);
                }
            }
        }
    }
}

// Dynamic variant that reads M_ret from device memory to avoid host D2H reads
// in return-scatter hot path.
__global__ void k_scatter_received_hybrid_bf16_dynamic(
    const uint16_t* __restrict__ y_buf,
    const Meta* __restrict__ meta,
    float* __restrict__ out,
    const int* __restrict__ M_ret_ptr,
    int capacity,
    int H, int Ha, int T, int K)
{
    int lane = threadIdx.x % 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    int M_ret = ld_nc_s32(M_ret_ptr);
    if (M_ret <= 0) return;
    if (M_ret > capacity) M_ret = capacity;

    for (int i = warp_id; i < M_ret; i += num_warps) {
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta + i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const uint16_t* y_row = y_buf + (int64_t)i * Ha;
        float* out_row = out + (int64_t)tok * H;

        for (int h = lane * 8; h < Ha; h += 32 * 8) {
            int4 v = ld_nc_v4_s32(reinterpret_cast<const int4*>(y_row + h));
            union BF16x8 {
                int4 v;
                uint16_t u[8];
            };
            BF16x8 x;
            x.v = v;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh < H) {
                    const __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&x.u[j]);
                    atomicAdd(out_row + hh, __bfloat162float(bf) * m.gate);
                }
            }
        }
    }
}

// ============================================================================
// Host API: Hybrid Return Scatter (BF16)
// ============================================================================

static void return_scatter_hybrid_impl(
    const __nv_bfloat16* Ye,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    uint16_t* nvshmem_y_buf,
    int H, int Ha,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP FATAL: hybrid return_scatter called before NVSHMEM init\n");
        abort();
    }

    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid return_scatter requires synced IPC pointers\n");
        abort();
    }

    int capacity = static_cast<int>(g_nvshmem.capacity);

    // Reset counters
    reset_counter_only(stream);

    // Reset local IPC counter
    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    cudaMemsetAsync(local_counter, 0, sizeof(int), stream);

    // Snapshot dispatch metadata before return writes reuse the same IPC meta buffer.
    Meta* local_meta = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
    if (M_recv > 0) {
        cudaMemcpyAsync(g_nvshmem.meta_copy, local_meta,
                        static_cast<size_t>(M_recv) * sizeof(Meta),
                        cudaMemcpyDeviceToDevice, stream);
    }

	    // Launch return scatter kernel
	    int threads = 256;
	    int warps_needed = std::max(M_recv, 1);
	    int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
	    int blocks = cap_warp_stride_blocks(blocks_by_work);

	    // Global barrier: ensure all ranks reset counters *and* snapshot their
	    // dispatch metadata before any remote atomicAdd()/writes begin.
		    hybrid_barrier_on_stream(stream);

    if (M_recv > 0) {
        k_return_scatter_hybrid_bf16<<<blocks, threads, 0, stream>>>(
            Ye, g_nvshmem.order, g_nvshmem.meta_copy, out,
            M_recv, H, Ha, T, K,
            g_nvshmem.rank, g_nvshmem.local_world, g_nvshmem.num_nodes,
            g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
            capacity,
            nvshmem_y_buf,
            static_cast<Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.dropped,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_y_off,
            ipc_meta_off,
            ipc_counter_off);
        CUDA_CHECK(cudaGetLastError());
    }

    // Wait for all NVSHMEM puts to complete, then forward proxy mailbox to true destination GPUs via IPC.
    hybrid_barrier_on_stream(stream);

    if (g_nvshmem.num_nodes > 1) {
        const int ret_upper = std::max(1, std::min(capacity, T * K));
        int f_threads = 256;
        int f_warps_per_block = f_threads / 32;
        int f_blocks_by_work = std::max(1, (ret_upper + f_warps_per_block - 1) / f_warps_per_block);
        int f_blocks = cap_warp_stride_blocks(f_blocks_by_work);
        k_forward_nvshmem_return_to_ipc_bf16_dynamic<<<f_blocks, f_threads, 0, stream>>>(
            nvshmem_y_buf,
            static_cast<const Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.dropped,
            H, Ha,
            capacity,
            g_nvshmem.nvl_rank,
            g_nvshmem.local_world,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_y_off,
            ipc_meta_off,
            ipc_counter_off);
        CUDA_CHECK(cudaGetLastError());
        // Ensure local IPC mailbox writes are visible before scatter.
        ipc_barrier_on_stream(stream);
    }

    uint16_t* ipc_y_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + ipc_y_off);
    Meta* ipc_meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
    int scatter_threads = 256;
    int scatter_warps_per_block = scatter_threads / 32;
    int scatter_work = std::max(1, std::min(capacity, T * K));
    int scatter_blocks_by_work = std::max(1, (scatter_work + scatter_warps_per_block - 1) / scatter_warps_per_block);
    int scatter_blocks = cap_warp_stride_blocks(scatter_blocks_by_work);
    k_scatter_received_hybrid_bf16_dynamic<<<scatter_blocks, scatter_threads, 0, stream>>>(
        ipc_y_buf,
        ipc_meta_buf,
        out,
        local_counter,
        capacity,
        H, Ha, T, K);
    CUDA_CHECK(cudaGetLastError());
}

static void return_scatter_hybrid_from_pad_impl(
    const __nv_bfloat16* Ye_pad,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    size_t ipc_y_off,
    size_t ipc_meta_off,
    size_t ipc_counter_off,
    uint16_t* nvshmem_y_buf,
    int H, int Ha,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP FATAL: hybrid return_scatter_from_pad called before NVSHMEM init\n");
        abort();
    }

    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid return_scatter requires synced IPC pointers\n");
        abort();
    }

    int capacity = static_cast<int>(g_nvshmem.capacity);

    reset_counter_only(stream);

    char* local_ipc_buf = static_cast<char*>(ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    int* local_counter = reinterpret_cast<int*>(local_ipc_buf + ipc_counter_off);
    cudaMemsetAsync(local_counter, 0, sizeof(int), stream);

    Meta* local_meta = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
    if (M_recv > 0) {
        cudaMemcpyAsync(g_nvshmem.meta_copy, local_meta,
                        static_cast<size_t>(M_recv) * sizeof(Meta),
                        cudaMemcpyDeviceToDevice, stream);
    }

    int threads = 256;
    int warps_needed = std::max(M_recv, 1);
    int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);

    hybrid_barrier_on_stream(stream);

    if (M_recv > 0) {
        k_return_scatter_hybrid_bf16_from_pad<<<blocks, threads, 0, stream>>>(
            Ye_pad,
            g_nvshmem.dest,
            g_nvshmem.order,
            g_nvshmem.meta_copy,
            out,
            M_recv, H, Ha, T, K,
            g_nvshmem.rank, g_nvshmem.local_world, g_nvshmem.num_nodes,
            g_nvshmem.rdma_rank, g_nvshmem.nvl_rank,
            capacity,
            nvshmem_y_buf,
            static_cast<Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.dropped,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_y_off,
            ipc_meta_off,
            ipc_counter_off);
        CUDA_CHECK(cudaGetLastError());
    }

    hybrid_barrier_on_stream(stream);

    if (g_nvshmem.num_nodes > 1) {
        const int ret_upper = std::max(1, std::min(capacity, T * K));
        int f_threads = 256;
        int f_warps_per_block = f_threads / 32;
        int f_blocks_by_work = std::max(1, (ret_upper + f_warps_per_block - 1) / f_warps_per_block);
        int f_blocks = cap_warp_stride_blocks(f_blocks_by_work);
        k_forward_nvshmem_return_to_ipc_bf16_dynamic<<<f_blocks, f_threads, 0, stream>>>(
            nvshmem_y_buf,
            static_cast<const Meta*>(g_nvshmem.meta),
            g_nvshmem.counter,
            g_nvshmem.dropped,
            H, Ha,
            capacity,
            g_nvshmem.nvl_rank,
            g_nvshmem.local_world,
            g_nvshmem.d_ipc_buffer_ptrs,
            ipc_y_off,
            ipc_meta_off,
            ipc_counter_off);
        CUDA_CHECK(cudaGetLastError());
        ipc_barrier_on_stream(stream);
    }

    uint16_t* ipc_y_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + ipc_y_off);
    Meta* ipc_meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + ipc_meta_off);
    int scatter_threads = 256;
    int scatter_warps_per_block = scatter_threads / 32;
    int scatter_work = std::max(1, std::min(capacity, T * K));
    int scatter_blocks_by_work = std::max(1, (scatter_work + scatter_warps_per_block - 1) / scatter_warps_per_block);
    int scatter_blocks = cap_warp_stride_blocks(scatter_blocks_by_work);
    k_scatter_received_hybrid_bf16_dynamic<<<scatter_blocks, scatter_threads, 0, stream>>>(
        ipc_y_buf,
        ipc_meta_buf,
        out,
        local_counter,
        capacity,
        H, Ha, T, K);
    CUDA_CHECK(cudaGetLastError());
}

void return_scatter_hybrid_bf16(
    const __nv_bfloat16* Ye,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_impl(
        Ye,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        /*ipc_y_off=*/0,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.x_buf_bf16,  // BF16 reuses x_buf for return
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

void return_scatter_hybrid_bf16_from_pad(
    const __nv_bfloat16* Ye_pad,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_from_pad_impl(
        Ye_pad,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        /*ipc_y_off=*/0,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.x_buf_bf16,
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

// ============================================================================
// Host API: Hybrid Return Scatter (Blockscaled)
// ============================================================================

void return_scatter_hybrid_blockscaled(
    const __nv_bfloat16* Ye,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_impl(
        Ye,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        g_nvshmem.ipc_y_off,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.y_buf,
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

void return_scatter_hybrid_blockscaled_from_pad(
    const __nv_bfloat16* Ye_pad,
    float* out,
    int M_recv, int T, int K,
    void** ipc_buffer_ptrs,
    int** ipc_barrier_ptrs,
    cudaStream_t stream)
{
    (void)ipc_barrier_ptrs;
    return_scatter_hybrid_from_pad_impl(
        Ye_pad,
        out,
        M_recv, T, K,
        ipc_buffer_ptrs,
        g_nvshmem.ipc_y_off,
        g_nvshmem.ipc_meta_off,
        g_nvshmem.ipc_counter_off,
        g_nvshmem.y_buf,
        g_nvshmem.H,
        g_nvshmem.Ha,
        stream);
}

// ============================================================================
// Hybrid Backward (BF16 payload)
// ============================================================================

static inline void enforce_single_bwd_stream(cudaStream_t stream, const char* tag) {
    const uintptr_t key = reinterpret_cast<uintptr_t>(stream) + 1u;
    uintptr_t expected = 0u;
    if (g_bwd_stream_slot.compare_exchange_strong(expected, key, std::memory_order_relaxed)) {
        return;
    }
    if (expected != key) {
        fprintf(stderr,
                "RDEP FATAL: %s called on multiple CUDA streams in hybrid backward. "
                "Use a single stream for hybrid backward tok-slot phases.\n",
                tag);
        abort();
    }
}

static inline uint32_t next_bwd_phase(cudaStream_t stream, int tok_slots) {
    enforce_single_bwd_stream(stream, "next_bwd_phase");
    uint32_t phase = g_bwd_phase.fetch_add(1u, std::memory_order_relaxed) + 1u;
    if (phase != 0u) {
        return phase;
    }
    // Wraparound: clear phase tags and restart from 1.
    CUDA_CHECK(cudaMemsetAsync(g_nvshmem.tok_tag, 0, static_cast<size_t>(tok_slots) * sizeof(int), stream));
    hybrid_barrier_on_stream(stream);
    g_bwd_phase.store(1u, std::memory_order_relaxed);
    return 1u;
}

__global__ void k_stage_dy_push_hybrid(
    const __nv_bfloat16* __restrict__ dY,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    int my_rank, int T, int H, int stage_stride, int K,
    int n_local, int capacity,
    int local_world, int rdma_rank, int world,
    uint16_t* nv_stage,
    void** ipc_buffer_ptrs,
    size_t ipc_stage_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int64_t rid_base = static_cast<int64_t>(my_rank) * static_cast<int64_t>(T) * static_cast<int64_t>(K);

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);
        int slot = i - tok * K;

        int eid = eids[i];
        if (eid < 0) continue;
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;
        int dest_rdma = nmoe_rank_node_fast(dest, local_world, local_world_pow2, local_world_shift);
        int dest_nvl = dest - dest_rdma * local_world;
        if (static_cast<unsigned>(dest_nvl) >= static_cast<unsigned>(local_world)) continue;
        bool remote_node = (dest_rdma != rdma_rank);
        bool remote_gpu = (dest != my_rank);

        int64_t rid = rid_base + static_cast<int64_t>(i);
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const __nv_bfloat16* row = dY + (int64_t)tok * H;

        if (remote_node) {
            uint16_t* dst = nv_stage + rid * static_cast<int64_t>(stage_stride);
            // Vectorized NVSHMEM put: 64-bit (4 BF16 values).
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(row + h),
                                      1, dest);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh,
                                          reinterpret_cast<const uint16_t*>(row) + hh,
                                          1, dest);
                    }
                }
            }
        } else {
            char* dest_buf = static_cast<char*>(ipc_buffer_ptrs[dest_nvl]);
            uint16_t* stage = reinterpret_cast<uint16_t*>(dest_buf + ipc_stage_off);
            uint16_t* dst = stage + rid * static_cast<int64_t>(stage_stride);
            if (remote_gpu) {
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        st_na_v2_s32(d, v);
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                            st_na_relaxed_gpu_b16(dst + hh, u);
                        }
                    }
                }
            } else {
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        *d = v;
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                            dst[hh] = u;
                        }
                    }
                }
            }
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_gather_dy_from_stage_and_send_gate_hybrid(
    const __nv_bfloat16* __restrict__ Ye_pad,      // [M_pad, H]
    const int* __restrict__ dest,                  // [M] sorted_i -> pad_i
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage_ipc,        // [capacity, stage_stride]
    const uint16_t* __restrict__ stage_nv,         // [capacity, stage_stride]
    int stage_stride,
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_sorted_out,          // [M]
    int M, int T, int H, int K,
    int capacity,
    int my_rank, int local_world, int rdma_rank, int world,
    void** ipc_buffer_ptrs,
    size_t ipc_tok_gate_off,
    float* nv_tok_gate,
    int* nv_tok_tag,
    int phase)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        int src_rank, tok, slot;
        decode_rid_fast(rid, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;
        const int src_rdma = nmoe_rank_node_fast(src_rank, local_world, local_world_pow2, local_world_shift);
        const bool src_same_node = (src_rdma == rdma_rank);

        const uint16_t* dy_u16 = (src_same_node ? stage_ipc : stage_nv) + rid * static_cast<int64_t>(stage_stride);
        const int pad_i = dest[i];
        if (pad_i < 0) continue;
        const uint16_t* ye_u16 = reinterpret_cast<const uint16_t*>(Ye_pad + (int64_t)pad_i * H);
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        const float g = gate_sorted[i];
        float dot = 0.0f;

        for (int h = lane * 8; h < stage_stride; h += 32 * 8) {
            int4 dy_v = ld_nc_v4_s32(reinterpret_cast<const int4*>(dy_u16 + h));
            union U16x8 { int4 v; uint16_t u[8]; };
            U16x8 dy8; dy8.v = dy_v;

            U16x8 ye8;
            if (h + 8 <= H) {
                ye8.v = *reinterpret_cast<const int4*>(ye_u16 + h);
            }

#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh >= H) break;
                const __nv_bfloat16 dy_bf = *reinterpret_cast<const __nv_bfloat16*>(&dy8.u[j]);
                const __nv_bfloat16 ye_bf = (h + 8 <= H) ? *reinterpret_cast<const __nv_bfloat16*>(&ye8.u[j])
                                                         : *reinterpret_cast<const __nv_bfloat16*>(ye_u16 + hh);
                float dy = __bfloat162float(dy_bf);
                float ye = __bfloat162float(ye_bf);
                dot += ye * dy;
                dye_row[hh] = __float2bfloat16(dy * g);
            }
        }

        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            dGate_sorted_out[i] = dot;

            // idx = tok*K + slot within the source rank's tok buffers.
            const int64_t idx = (int64_t)tok * K + slot;
            if (idx < 0 || idx >= static_cast<int64_t>(T) * K) continue;

            if (src_same_node) {
                const int src_nvl = nmoe_rank_local_fast(src_rank, local_world, local_world_pow2, local_world_mask);
                char* src_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl]);
                float* tok_gate = reinterpret_cast<float*>(src_buf + ipc_tok_gate_off);
                if (src_rank == my_rank) {
                    tok_gate[idx] = dot;
                } else {
                    st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(dot));
                }
            } else {
                nvshmem_float_p(nv_tok_gate + idx, dot, src_rank);
                nvshmem_int_p(nv_tok_tag + idx, phase, src_rank);
            }
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_collect_tok_gate_hybrid(
    const float* __restrict__ ipc_tok_gate,
    const float* __restrict__ nv_tok_gate,
    const int* __restrict__ nv_tok_tag,
    float* __restrict__ dGates_tk_out,
    int tok_slots,
    int phase)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < tok_slots;
         i += blockDim.x * gridDim.x) {
        int tag = ld_nc_s32(nv_tok_tag + i);
        float v = (tag == phase) ? ld_nc_f32(nv_tok_gate + i) : ld_nc_f32(ipc_tok_gate + i);
        dGates_tk_out[i] = v;
    }
}

__global__ void k_collect_tok_gate_hybrid_bf16(
    const float* __restrict__ ipc_tok_gate,
    const float* __restrict__ nv_tok_gate,
    const int* __restrict__ nv_tok_tag,
    __nv_bfloat16* __restrict__ dGates_tk_out,
    int tok_slots,
    int phase)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < tok_slots;
         i += blockDim.x * gridDim.x) {
        int tag = ld_nc_s32(nv_tok_tag + i);
        float v = (tag == phase) ? ld_nc_f32(nv_tok_gate + i) : ld_nc_f32(ipc_tok_gate + i);
        dGates_tk_out[i] = __float2bfloat16(v);
    }
}

__global__ void k_gather_dy_from_stage_nogate_hybrid(
    const int* __restrict__ dest,                  // [M] sorted_i -> pad_i
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage_ipc,        // [capacity, stage_stride]
    const uint16_t* __restrict__ stage_nv,         // [capacity, stage_stride]
    int stage_stride,
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    int M, int T, int H, int K,
    int capacity,
    int local_world, int rdma_rank, int world)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        int src_rank, tok, slot;
        decode_rid_fast(rid, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;
        const int src_rdma = nmoe_rank_node_fast(src_rank, local_world, local_world_pow2, local_world_shift);
        const bool src_same_node = (src_rdma == rdma_rank);

        const uint16_t* dy_u16 = (src_same_node ? stage_ipc : stage_nv) + rid * static_cast<int64_t>(stage_stride);
        const int pad_i = dest[i];
        if (pad_i < 0) continue;
        __nv_bfloat16* dye_row = dYe_out + static_cast<int64_t>(i) * H;
        const float g = gate_sorted[i];

        for (int h = lane * 8; h < stage_stride; h += 32 * 8) {
            int4 dy_v = ld_nc_v4_s32(reinterpret_cast<const int4*>(dy_u16 + h));
            union U16x8 { int4 v; uint16_t u[8]; };
            U16x8 dy8;
            dy8.v = dy_v;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh >= H) break;
                const __nv_bfloat16 dy_bf = *reinterpret_cast<const __nv_bfloat16*>(&dy8.u[j]);
                const float dy = __bfloat162float(dy_bf);
                dye_row[hh] = __float2bfloat16(dy * g);
            }
        }
    }
}

__global__ void k_send_dgate_tokslot_hybrid(
    const int64_t* __restrict__ row_id,   // [M]
    const float* __restrict__ dGate_sorted, // [M]
    int M, int T, int K,
    int capacity,
    int my_rank, int local_world, int rdma_rank, int world,
    void** ipc_buffer_ptrs,
    size_t ipc_tok_gate_off,
    float* nv_tok_gate,
    int* nv_tok_tag,
    int phase)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (; i < M; i += stride) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        int src_rank, tok, slot;
        decode_rid_fast(rid, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;
        const int64_t idx = static_cast<int64_t>(tok) * K + slot;
        if (idx < 0 || idx >= static_cast<int64_t>(T) * K) continue;
        const float dg = dGate_sorted[i];

        const int src_rdma = nmoe_rank_node_fast(src_rank, local_world, local_world_pow2, local_world_shift);
        const bool src_same_node = (src_rdma == rdma_rank);
        if (src_same_node) {
            const int src_nvl = nmoe_rank_local_fast(src_rank, local_world, local_world_pow2, local_world_mask);
            char* src_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl]);
            float* tok_gate = reinterpret_cast<float*>(src_buf + ipc_tok_gate_off);
            if (src_rank == my_rank) {
                tok_gate[idx] = dg;
            } else {
                st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(dg));
            }
        } else {
            nvshmem_float_p(nv_tok_gate + idx, dg, src_rank);
            nvshmem_int_p(nv_tok_tag + idx, phase, src_rank);
        }
    }
    fence_acq_rel_sys();
}

void gather_dy_hybrid_bf16(
    const __nv_bfloat16* dY_local,
    const int* eids,
    const __nv_bfloat16* Ye_pad,
    const int64_t* row_id,
    const float* gate_sorted,
    __nv_bfloat16* dYe_out,
    float* dGate_sorted_out,
    float* dGates_tk_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP ERROR: hybrid gather_dy called before NVSHMEM init\n");
        abort();
    }
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid gather_dy requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        abort();
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        abort();
    }
    if (H != g_nvshmem.H) {
        fprintf(stderr, "RDEP ERROR: gather_dy H mismatch: got H=%d, state H=%d\n", H, g_nvshmem.H);
        abort();
    }

    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;

    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        abort();
    }
    const int phase = static_cast<int>(next_bwd_phase(stream, tok_slots));

    const bool bf16_profile = (g_nvshmem.profile == -1);
    const uint16_t* nv_stage = bf16_profile ? g_nvshmem.x_buf_bf16 : g_nvshmem.y_buf;
    const size_t ipc_stage_off = bf16_profile ? g_nvshmem.ipc_x_off : g_nvshmem.ipc_y_off;
    const int stage_stride = bf16_profile ? g_nvshmem.Ha : g_nvshmem.H;

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    const uint16_t* stage_ipc = reinterpret_cast<const uint16_t*>(local_ipc_buf + ipc_stage_off);
    const float* ipc_tok_gate = reinterpret_cast<const float*>(local_ipc_buf + g_nvshmem.ipc_tok_gate_off);
    ipc_barrier_on_stream(stream);
    float* ipc_tok_gate_mut = reinterpret_cast<float*>(local_ipc_buf + g_nvshmem.ipc_tok_gate_off);
    CUDA_CHECK(cudaMemsetAsync(
        ipc_tok_gate_mut,
        0,
        static_cast<size_t>(tok_slots) * sizeof(float),
        stream));
    ipc_barrier_on_stream(stream);

	    // Stage dY (push) to expert owners.
    const int threads = 256;
    const int warps_needed = std::max(1, tok_slots);
    const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_stage_dy_push_hybrid<<<blocks, threads, 0, stream>>>(
        dY_local,
        eids,
        my_rank, T, H, stage_stride, K,
        g_nvshmem.n_local, static_cast<int>(g_nvshmem.capacity),
        local_world, rdma_rank, g_nvshmem.world,
        const_cast<uint16_t*>(nv_stage),
        g_nvshmem.d_ipc_buffer_ptrs,
        ipc_stage_off);
    CUDA_CHECK(cudaGetLastError());

    hybrid_barrier_on_stream(stream);

	    // Compute dYe/dGate locally and return dGate to token owners.
    if (M > 0) {
        const int g_threads = 256;
        const int g_blocks_by_work = std::max(1, (M * 32 + g_threads - 1) / g_threads);
        const int g_blocks = cap_warp_stride_blocks(g_blocks_by_work);
        k_gather_dy_from_stage_and_send_gate_hybrid<<<g_blocks, g_threads, 0, stream>>>(
            Ye_pad,
            g_nvshmem.dest,
            row_id,
            gate_sorted,
            stage_ipc,
            nv_stage,
            stage_stride,
            dYe_out,
            dGate_sorted_out,
            M, T, H, K,
            static_cast<int>(g_nvshmem.capacity),
            my_rank, local_world, rdma_rank, g_nvshmem.world,
            g_nvshmem.d_ipc_buffer_ptrs,
            g_nvshmem.ipc_tok_gate_off,
            g_nvshmem.tok_gate,
            g_nvshmem.tok_tag,
            phase);
        CUDA_CHECK(cudaGetLastError());
    }

    hybrid_barrier_on_stream(stream);

    // Collect per-(tok,slot) dGate into output tensor.
    const int c_threads = 256;
    const int c_blocks_by_work = std::max(1, (tok_slots + c_threads - 1) / c_threads);
    const int c_blocks = cap_warp_stride_blocks(c_blocks_by_work);
    k_collect_tok_gate_hybrid<<<c_blocks, c_threads, 0, stream>>>(
        ipc_tok_gate,
        g_nvshmem.tok_gate,
        g_nvshmem.tok_tag,
        dGates_tk_out,
        tok_slots,
        phase);
    CUDA_CHECK(cudaGetLastError());
}

void gather_dy_nogate_hybrid_bf16(
    const __nv_bfloat16* dY_local,
    const int* eids,
    const int64_t* row_id,
    const float* gate_sorted,
    __nv_bfloat16* dYe_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP ERROR: hybrid gather_dy_nogate called before NVSHMEM init\n");
        abort();
    }
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid gather_dy_nogate requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        abort();
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        abort();
    }
    if (H != g_nvshmem.H) {
        fprintf(stderr, "RDEP ERROR: gather_dy_nogate H mismatch: got H=%d, state H=%d\n", H, g_nvshmem.H);
        abort();
    }

    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;

    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        abort();
    }

    const bool bf16_profile = (g_nvshmem.profile == -1);
    const uint16_t* nv_stage = bf16_profile ? g_nvshmem.x_buf_bf16 : g_nvshmem.y_buf;
    const size_t ipc_stage_off = bf16_profile ? g_nvshmem.ipc_x_off : g_nvshmem.ipc_y_off;
    const int stage_stride = bf16_profile ? g_nvshmem.Ha : g_nvshmem.H;
    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    const uint16_t* stage_ipc = reinterpret_cast<const uint16_t*>(local_ipc_buf + ipc_stage_off);

    // Stage dY (push) to expert owners.
    const int threads = 256;
    const int warps_needed = std::max(1, tok_slots);
    const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_stage_dy_push_hybrid<<<blocks, threads, 0, stream>>>(
        dY_local,
        eids,
        my_rank, T, H, stage_stride, K,
        g_nvshmem.n_local, static_cast<int>(g_nvshmem.capacity),
        local_world, rdma_rank, g_nvshmem.world,
        const_cast<uint16_t*>(nv_stage),
        g_nvshmem.d_ipc_buffer_ptrs,
        ipc_stage_off);
    CUDA_CHECK(cudaGetLastError());

    hybrid_barrier_on_stream(stream);

    // Gather dYe only (split dGate path).
    if (M > 0) {
        const int g_threads = 256;
        const int g_blocks_by_work = std::max(1, (M * 32 + g_threads - 1) / g_threads);
        const int g_blocks = cap_warp_stride_blocks(g_blocks_by_work);
        k_gather_dy_from_stage_nogate_hybrid<<<g_blocks, g_threads, 0, stream>>>(
            g_nvshmem.dest,
            row_id,
            gate_sorted,
            stage_ipc,
            nv_stage,
            stage_stride,
            dYe_out,
            M, T, H, K,
            static_cast<int>(g_nvshmem.capacity),
            local_world, rdma_rank, g_nvshmem.world);
        CUDA_CHECK(cudaGetLastError());
    }
}

void send_dgate_hybrid_bf16(
    const int64_t* row_id,
    const float* dGate_sorted,
    float* dGates_tk_out,
    int M, int T, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP ERROR: hybrid send_dgate called before NVSHMEM init\n");
        abort();
    }
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid send_dgate requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        abort();
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        abort();
    }

    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;
    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        abort();
    }
    const int phase = static_cast<int>(next_bwd_phase(stream, tok_slots));

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    float* ipc_tok_gate_mut = reinterpret_cast<float*>(local_ipc_buf + g_nvshmem.ipc_tok_gate_off);
    const float* ipc_tok_gate = reinterpret_cast<const float*>(ipc_tok_gate_mut);
    ipc_barrier_on_stream(stream);
    CUDA_CHECK(cudaMemsetAsync(
        ipc_tok_gate_mut,
        0,
        static_cast<size_t>(tok_slots) * sizeof(float),
        stream));
    ipc_barrier_on_stream(stream);

    if (M > 0) {
        const int threads = 256;
        const int blocks_by_work = std::max(1, (M + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_send_dgate_tokslot_hybrid<<<blocks, threads, 0, stream>>>(
            row_id,
            dGate_sorted,
            M, T, K,
            static_cast<int>(g_nvshmem.capacity),
            my_rank, local_world, rdma_rank, g_nvshmem.world,
            g_nvshmem.d_ipc_buffer_ptrs,
            g_nvshmem.ipc_tok_gate_off,
            g_nvshmem.tok_gate,
            g_nvshmem.tok_tag,
            phase);
        CUDA_CHECK(cudaGetLastError());
    }

    hybrid_barrier_on_stream(stream);

    const int c_threads = 256;
    const int c_blocks_by_work = std::max(1, (tok_slots + c_threads - 1) / c_threads);
    const int c_blocks = cap_warp_stride_blocks(c_blocks_by_work);
    k_collect_tok_gate_hybrid<<<c_blocks, c_threads, 0, stream>>>(
        ipc_tok_gate,
        g_nvshmem.tok_gate,
        g_nvshmem.tok_tag,
        dGates_tk_out,
        tok_slots,
        phase);
    CUDA_CHECK(cudaGetLastError());
}

void send_dgate_hybrid_bf16_out_bf16(
    const int64_t* row_id,
    const float* dGate_sorted,
    __nv_bfloat16* dGates_tk_out,
    int M, int T, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP ERROR: hybrid send_dgate_out_bf16 called before NVSHMEM init\n");
        abort();
    }
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid send_dgate_out_bf16 requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        abort();
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        abort();
    }

    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;
    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        abort();
    }
    const int phase = static_cast<int>(next_bwd_phase(stream, tok_slots));

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    float* ipc_tok_gate_mut = reinterpret_cast<float*>(local_ipc_buf + g_nvshmem.ipc_tok_gate_off);
    const float* ipc_tok_gate = reinterpret_cast<const float*>(ipc_tok_gate_mut);
    ipc_barrier_on_stream(stream);
    CUDA_CHECK(cudaMemsetAsync(
        ipc_tok_gate_mut,
        0,
        static_cast<size_t>(tok_slots) * sizeof(float),
        stream));
    ipc_barrier_on_stream(stream);

    if (M > 0) {
        const int threads = 256;
        const int blocks_by_work = std::max(1, (M + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_send_dgate_tokslot_hybrid<<<blocks, threads, 0, stream>>>(
            row_id,
            dGate_sorted,
            M, T, K,
            static_cast<int>(g_nvshmem.capacity),
            my_rank, local_world, rdma_rank, g_nvshmem.world,
            g_nvshmem.d_ipc_buffer_ptrs,
            g_nvshmem.ipc_tok_gate_off,
            g_nvshmem.tok_gate,
            g_nvshmem.tok_tag,
            phase);
        CUDA_CHECK(cudaGetLastError());
    }

    hybrid_barrier_on_stream(stream);

    const int c_threads = 256;
    const int c_blocks_by_work = std::max(1, (tok_slots + c_threads - 1) / c_threads);
    const int c_blocks = cap_warp_stride_blocks(c_blocks_by_work);
    k_collect_tok_gate_hybrid_bf16<<<c_blocks, c_threads, 0, stream>>>(
        ipc_tok_gate,
        g_nvshmem.tok_gate,
        g_nvshmem.tok_tag,
        dGates_tk_out,
        tok_slots,
        phase);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_send_dx_tokslot_hybrid(
    const __nv_bfloat16* __restrict__ dXe_sorted,  // [M, H]
    const int64_t* __restrict__ row_id,            // [M]
    int M, int T, int H, int K,
    int tok_Ha,
    int capacity,
    int my_rank, int local_world, int rdma_rank, int world,
    void** ipc_buffer_ptrs,
    size_t ipc_tok_y_off,
    size_t ipc_tok_gate_off,
    uint16_t* nv_tok_y,
    int* nv_tok_tag,
    int phase)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        int src_rank, tok, slot;
        decode_rid_fast(rid, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const int src_rdma = nmoe_rank_node_fast(src_rank, local_world, local_world_pow2, local_world_shift);
        const bool same_node = (src_rdma == rdma_rank);
        const int64_t idx = (int64_t)tok * K + slot;
        if (idx < 0 || idx >= static_cast<int64_t>(T) * K) continue;

        const __nv_bfloat16* row = dXe_sorted + (int64_t)i * H;

        if (same_node) {
            const int src_nvl = nmoe_rank_local_fast(src_rank, local_world, local_world_pow2, local_world_mask);
            char* src_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl]);
            uint16_t* tok_y = reinterpret_cast<uint16_t*>(src_buf + ipc_tok_y_off);
            float* tok_gate = reinterpret_cast<float*>(src_buf + ipc_tok_gate_off);
            uint16_t* dst = tok_y + idx * static_cast<int64_t>(tok_Ha);

            const bool remote_gpu = (src_rank != my_rank);
            if (remote_gpu) {
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        st_na_v2_s32(d, v);
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                            st_na_relaxed_gpu_b16(dst + hh, u);
                        }
                    }
                }
                if (lane == 0) {
                    st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(1.0f));
                }
            } else {
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        *d = v;
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                            dst[hh] = u;
                        }
                    }
                }
                if (lane == 0) {
                    tok_gate[idx] = 1.0f;
                }
            }
        } else {
            uint16_t* dst = nv_tok_y + idx * static_cast<int64_t>(tok_Ha);
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(row + h),
                                      1, src_rank);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh,
                                          reinterpret_cast<const uint16_t*>(row) + hh,
                                          1, src_rank);
                    }
                }
            }
            if (lane == 0) {
                nvshmem_int_p(nv_tok_tag + idx, phase, src_rank);
            }
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_send_dx_tokslot_hybrid_from_pad(
    const __nv_bfloat16* __restrict__ dXe_pad,     // [M_pad, H]
    const int* __restrict__ dest,                  // [M] sorted->pad index
    const int64_t* __restrict__ row_id,            // [M]
    int M, int T, int H, int K,
    int tok_Ha,
    int capacity,
    int my_rank, int local_world, int rdma_rank, int world,
    void** ipc_buffer_ptrs,
    size_t ipc_tok_y_off,
    size_t ipc_tok_gate_off,
    uint16_t* nv_tok_y,
    int* nv_tok_tag,
    int phase)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool local_world_pow2 = nmoe_is_pow2(local_world);
    const int local_world_shift = local_world_pow2 ? (__ffs(local_world) - 1) : 0;
    const int local_world_mask = local_world - 1;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        int src_rank, tok, slot;
        decode_rid_fast(rid, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const int src_rdma = nmoe_rank_node_fast(src_rank, local_world, local_world_pow2, local_world_shift);
        const bool same_node = (src_rdma == rdma_rank);
        const int64_t idx = (int64_t)tok * K + slot;
        if (idx < 0 || idx >= static_cast<int64_t>(T) * K) continue;

        const int pad_i = dest[i];
        if (pad_i < 0) continue;
        const __nv_bfloat16* row = dXe_pad + static_cast<int64_t>(pad_i) * H;

        if (same_node) {
            const int src_nvl = nmoe_rank_local_fast(src_rank, local_world, local_world_pow2, local_world_mask);
            char* src_buf = static_cast<char*>(ipc_buffer_ptrs[src_nvl]);
            uint16_t* tok_y = reinterpret_cast<uint16_t*>(src_buf + ipc_tok_y_off);
            float* tok_gate = reinterpret_cast<float*>(src_buf + ipc_tok_gate_off);
            uint16_t* dst = tok_y + idx * static_cast<int64_t>(tok_Ha);

            const bool remote_gpu = (src_rank != my_rank);
            if (remote_gpu) {
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        st_na_v2_s32(d, v);
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                            st_na_relaxed_gpu_b16(dst + hh, u);
                        }
                    }
                }
                if (lane == 0) {
                    st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(1.0f));
                }
            } else {
                for (int h = lane * 4; h < H; h += 32 * 4) {
                    if (h + 4 <= H) {
                        int2 v = *reinterpret_cast<const int2*>(row + h);
                        int2* d = reinterpret_cast<int2*>(dst + h);
                        *d = v;
                    } else {
                        for (int hh = h; hh < H && hh < h + 4; hh++) {
                            uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                            dst[hh] = u;
                        }
                    }
                }
                if (lane == 0) {
                    tok_gate[idx] = 1.0f;
                }
            }
        } else {
            uint16_t* dst = nv_tok_y + idx * static_cast<int64_t>(tok_Ha);
            for (int h = lane * 4; h < H; h += 32 * 4) {
                if (h + 4 <= H) {
                    nvshmem_put64_nbi(reinterpret_cast<uint64_t*>(dst + h),
                                      reinterpret_cast<const uint64_t*>(row + h),
                                      1, src_rank);
                } else {
                    for (int hh = h; hh < H && hh < h + 4; hh++) {
                        nvshmem_put16_nbi(dst + hh,
                                          reinterpret_cast<const uint16_t*>(row) + hh,
                                          1, src_rank);
                    }
                }
            }
            if (lane == 0) {
                nvshmem_int_p(nv_tok_tag + idx, phase, src_rank);
            }
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_reduce_dx_tokslot_hybrid(
    const uint16_t* __restrict__ ipc_tok_y,
    const float* __restrict__ ipc_tok_gate,
    const uint16_t* __restrict__ nv_tok_y,
    const int* __restrict__ nv_tok_tag,
    float* __restrict__ dX_out,
    int T, int H, int tok_Ha, int K,
    int phase)
{
    if (K <= 0 || K > 32) return;
    __shared__ uint8_t slot_src[32];  // 0=none, 1=ipc, 2=nvshmem

    for (int tok = static_cast<int>(blockIdx.x); tok < T; tok += static_cast<int>(gridDim.x)) {
        if (static_cast<int>(threadIdx.x) < K) {
            const int slot = static_cast<int>(threadIdx.x);
            const int64_t idx = static_cast<int64_t>(tok) * K + slot;
            const bool remote = (ld_nc_s32(nv_tok_tag + idx) == phase);
            slot_src[slot] = remote ? static_cast<uint8_t>(2u)
                                    : ((ld_nc_f32(ipc_tok_gate + idx) > 0.0f) ? static_cast<uint8_t>(1u)
                                                                               : static_cast<uint8_t>(0u));
        }
        __syncthreads();

        int vec = static_cast<int>(threadIdx.x);
        for (int h0 = vec * 8; h0 < H; h0 += static_cast<int>(blockDim.x) * 8) {
            float acc[8] = {0};
            for (int slot = 0; slot < K; ++slot) {
                const int64_t idx = (int64_t)tok * K + slot;
                const uint8_t src_sel = slot_src[slot];
                if (src_sel == 0u) continue;
                const uint16_t* base = (src_sel == 2u) ? nv_tok_y : ipc_tok_y;
                const uint16_t* y_row = base + idx * static_cast<int64_t>(tok_Ha) + h0;

                int4 v = ld_nc_v4_s32(reinterpret_cast<const int4*>(y_row));
                union BF16x8 {
                    int4 v;
                    uint16_t u[8];
                };
                BF16x8 x;
                x.v = v;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int hh = h0 + j;
                    if (hh < H) {
                        const __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&x.u[j]);
                        acc[j] += __bfloat162float(bf);
                    }
                }
            }

            float* out_row = dX_out + (int64_t)tok * H + h0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int hh = h0 + j;
                if (hh < H) out_row[j] = acc[j];
            }
        }
        __syncthreads();
    }
}

void scatter_dx_hybrid_bf16(
    const __nv_bfloat16* dXe_sorted,
    const int64_t* row_id,
    float* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP ERROR: hybrid scatter_dx called before NVSHMEM init\n");
        abort();
    }
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid scatter_dx requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        abort();
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        abort();
    }
    if (H != g_nvshmem.H) {
        fprintf(stderr, "RDEP ERROR: scatter_dx H mismatch: got H=%d, state H=%d\n", H, g_nvshmem.H);
        abort();
    }
    if ((H & 7) != 0) {
        fprintf(stderr, "RDEP ERROR: scatter_dx requires H multiple of 8 (H=%d)\n", H);
        abort();
    }

    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;

    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        abort();
    }
    const int phase = static_cast<int>(next_bwd_phase(stream, tok_slots));

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    const uint16_t* ipc_tok_y = reinterpret_cast<const uint16_t*>(local_ipc_buf + g_nvshmem.ipc_tok_y_off);
    ipc_barrier_on_stream(stream);
    float* ipc_tok_gate_mut = reinterpret_cast<float*>(local_ipc_buf + g_nvshmem.ipc_tok_gate_off);
    const float* ipc_tok_gate = reinterpret_cast<const float*>(ipc_tok_gate_mut);
    CUDA_CHECK(cudaMemsetAsync(
        ipc_tok_gate_mut,
        0,
        static_cast<size_t>(tok_slots) * sizeof(float),
        stream));
    ipc_barrier_on_stream(stream);

    if (M > 0) {
        const int threads = 256;
        const int warps_needed = std::max(1, M);
        const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_send_dx_tokslot_hybrid<<<blocks, threads, 0, stream>>>(
            dXe_sorted,
            row_id,
            M, T, H, K,
            g_nvshmem.tok_Ha,
            static_cast<int>(g_nvshmem.capacity),
            my_rank, local_world, rdma_rank, g_nvshmem.world,
            g_nvshmem.d_ipc_buffer_ptrs,
            g_nvshmem.ipc_tok_y_off,
            g_nvshmem.ipc_tok_gate_off,
            g_nvshmem.tok_y,
            g_nvshmem.tok_tag,
            phase);
        CUDA_CHECK(cudaGetLastError());
    }

    hybrid_barrier_on_stream(stream);

    const int reduce_threads = 256;
    const int reduce_blocks_by_work = std::max(1, T);
    const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
    k_reduce_dx_tokslot_hybrid<<<reduce_blocks, reduce_threads, 0, stream>>>(
        ipc_tok_y,
        ipc_tok_gate,
        g_nvshmem.tok_y,
        g_nvshmem.tok_tag,
        dX_out,
        T, H, g_nvshmem.tok_Ha, K,
        phase);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_dx_hybrid_bf16_from_pad(
    const __nv_bfloat16* dXe_pad,
    const int64_t* row_id,
    float* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_nvshmem.initialized) {
        fprintf(stderr, "RDEP ERROR: hybrid scatter_dx_from_pad called before NVSHMEM init\n");
        abort();
    }
    if (!g_nvshmem.d_ipc_buffer_ptrs || !g_nvshmem.d_ipc_barrier_signal_ptrs) {
        fprintf(stderr, "RDEP ERROR: hybrid scatter_dx_from_pad requires synced IPC pointers; call nvshmem_sync_ipc_buffer_ptrs_*\n");
        abort();
    }
    if (K <= 0 || K > 32) {
        fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
        abort();
    }
    if (H != g_nvshmem.H) {
        fprintf(stderr, "RDEP ERROR: scatter_dx_from_pad H mismatch: got H=%d, state H=%d\n", H, g_nvshmem.H);
        abort();
    }
    if ((H & 7) != 0) {
        fprintf(stderr, "RDEP ERROR: scatter_dx_from_pad requires H multiple of 8 (H=%d)\n", H);
        abort();
    }

    const int my_rank = g_nvshmem.rank;
    const int local_world = g_nvshmem.local_world;
    const int rdma_rank = g_nvshmem.rdma_rank;

    const size_t tok_cap = (g_nvshmem.world > 0) ? (g_nvshmem.capacity / static_cast<size_t>(g_nvshmem.world)) : 0;
    const int tok_slots = T * K;
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_nvshmem.capacity, g_nvshmem.world);
        abort();
    }
    const int phase = static_cast<int>(next_bwd_phase(stream, tok_slots));

    char* local_ipc_buf = static_cast<char*>(g_nvshmem.ipc_buffer_ptrs[g_nvshmem.nvl_rank]);
    const uint16_t* ipc_tok_y = reinterpret_cast<const uint16_t*>(local_ipc_buf + g_nvshmem.ipc_tok_y_off);
    ipc_barrier_on_stream(stream);
    float* ipc_tok_gate_mut = reinterpret_cast<float*>(local_ipc_buf + g_nvshmem.ipc_tok_gate_off);
    const float* ipc_tok_gate = reinterpret_cast<const float*>(ipc_tok_gate_mut);
    CUDA_CHECK(cudaMemsetAsync(
        ipc_tok_gate_mut,
        0,
        static_cast<size_t>(tok_slots) * sizeof(float),
        stream));
    ipc_barrier_on_stream(stream);

    if (M > 0) {
        const int threads = 256;
        const int warps_needed = std::max(1, M);
        const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_send_dx_tokslot_hybrid_from_pad<<<blocks, threads, 0, stream>>>(
            dXe_pad,
            g_nvshmem.dest,
            row_id,
            M, T, H, K,
            g_nvshmem.tok_Ha,
            static_cast<int>(g_nvshmem.capacity),
            my_rank, local_world, rdma_rank, g_nvshmem.world,
            g_nvshmem.d_ipc_buffer_ptrs,
            g_nvshmem.ipc_tok_y_off,
            g_nvshmem.ipc_tok_gate_off,
            g_nvshmem.tok_y,
            g_nvshmem.tok_tag,
            phase);
        CUDA_CHECK(cudaGetLastError());
    }

    hybrid_barrier_on_stream(stream);

    const int reduce_threads = 256;
    const int reduce_blocks_by_work = std::max(1, T);
    const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
    k_reduce_dx_tokslot_hybrid<<<reduce_blocks, reduce_threads, 0, stream>>>(
        ipc_tok_y,
        ipc_tok_gate,
        g_nvshmem.tok_y,
        g_nvshmem.tok_tag,
        dX_out,
        T, H, g_nvshmem.tok_Ha, K,
        phase);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace nvshmem
}  // namespace rdep

// ============================================================================
// C API wrappers for Python bindings
// Use rdep_ prefix to avoid conflicts with NVSHMEM's own functions
// ============================================================================

extern "C" {

void rdep_nvshmem_get_uid(void* uid_out) {
    rdep::nvshmem::get_uid(uid_out);
}

int rdep_nvshmem_get_uid_size() {
    return rdep::nvshmem::get_uid_size();
}

void rdep_nvshmem_init_with_uid(const void* uid, int rank, int world, int local_world) {
    rdep::nvshmem::init(uid, rank, world, local_world);
}

void rdep_nvshmem_finalize() {
    rdep::nvshmem::finalize();
}

void rdep_nvshmem_alloc_bf16(size_t capacity, int H, int n_local) {
    rdep::nvshmem::alloc_bf16(capacity, H, n_local);
}

void rdep_nvshmem_alloc_blockscaled(size_t capacity, int H, int n_local, int profile) {
    rdep::nvshmem::alloc_blockscaled(capacity, H, n_local, profile);
}

void rdep_nvshmem_quiet() {
    rdep::nvshmem::quiet();
}

// IPC buffer management functions
void rdep_nvshmem_get_ipc_handle_bf16(void* handle_out) {
    rdep::nvshmem::get_ipc_handle_bf16(handle_out);
}

void rdep_nvshmem_open_ipc_handles_bf16(const void* handles, int local_world) {
    rdep::nvshmem::open_ipc_handles_bf16(handles, local_world);
}

void rdep_nvshmem_sync_ipc_buffer_ptrs_bf16() {
    rdep::nvshmem::sync_ipc_buffer_ptrs_bf16();
}

void rdep_nvshmem_get_ipc_handle_blockscaled(void* handle_out) {
    rdep::nvshmem::get_ipc_handle_blockscaled(handle_out);
}

void rdep_nvshmem_open_ipc_handles_blockscaled(const void* handles, int local_world) {
    rdep::nvshmem::open_ipc_handles_blockscaled(handles, local_world);
}

void rdep_nvshmem_sync_ipc_buffer_ptrs_blockscaled() {
    rdep::nvshmem::sync_ipc_buffer_ptrs_blockscaled();
}

}  // extern "C"

#endif  // WITH_NVSHMEM
