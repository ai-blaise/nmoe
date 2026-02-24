// RDEP: Expert-parallel dispatch/return for MoE
//
// Three modes:
//   - MODE_SINGLE: world=1, local sort/pad/quant only
//   - MODE_IPC: world=local_world, CUDA IPC for intra-node NVLink
//   - MODE_HYBRID: world>local_world, IPC intra-node + NVSHMEM inter-node
//
// Bootstrap (one-time at init):
//   - IPC: NCCL all_gather to exchange cudaIpcMemHandle_t
//   - NVSHMEM: NCCL broadcast to share NVSHMEM UID
//
// Hot path (zero NCCL):
//   - GPU-side atomics for sync (atomicAdd_system/atomicSub_system)
//   - Direct P2P writes via IPC handles or NVSHMEM puts

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <type_traits>
#include <vector>
#include "swizzle.cuh"

// NVSHMEM support (optional, for multi-node)
#ifdef WITH_NVSHMEM
#include "rdep_nvshmem.cuh"
#endif

namespace rdep {

// ============================================================================
// Constants (following DeepEP configs.cuh)
// ============================================================================

constexpr int SF_VEC = 32;        // Scale factor granularity
constexpr float FP8_MAX = 448.0f;
constexpr float FP4_MAX = 6.0f;
constexpr int MAX_RANKS = 8;      // Max NVLink peers (like DeepEP)
constexpr int BUFFER_ALIGNMENT = 128;  // DeepEP's NUM_BUFFER_ALIGNMENT_BYTES
constexpr int BARRIER_TAG = 1024;      // DeepEP's FINISHED_SUM_TAG
constexpr uint64_t TIMEOUT_CYCLES = 200000000000ull;  // ~100s at 2GHz

__host__ __forceinline__ bool env_truthy(const char* value) {
    if (value == nullptr || value[0] == '\0') return false;
    return
        value[0] == '1' ||
        value[0] == 'y' || value[0] == 'Y' ||
        value[0] == 't' || value[0] == 'T';
}

__host__ __forceinline__ bool rdep_barrier_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = env_truthy(std::getenv("NMOE_RDEP_BARRIER_TRACE")) ? 1 : 0;
    }
    return cached != 0;
}

__host__ __forceinline__ int rdep_barrier_watchdog_ms() {
    static int cached = -1;
    if (cached < 0) {
        const char* raw = std::getenv("NMOE_RDEP_BARRIER_WATCHDOG_MS");
        int parsed = 0;
        if (raw && raw[0] != '\0') {
            parsed = std::atoi(raw);
            if (parsed < 0) parsed = 0;
        }
        cached = parsed;
    }
    return cached;
}

__host__ __forceinline__ int rdep_env_global_rank() {
    static bool initialized = false;
    static int rank = -1;
    if (!initialized) {
        const char* raw = std::getenv("RANK");
        rank = (raw && raw[0] != '\0') ? std::atoi(raw) : -1;
        initialized = true;
    }
    return rank;
}

__host__ __forceinline__ int rdep_counter_wait_timeout_ms() {
    static int cached = -1;
    if (cached < 0) {
        // Counter read wait should normally resolve quickly once the stream
        // records the event. Keep a finite default so silent hangs become
        // actionable errors with context.
        int parsed = 180000;  // 3 minutes
        const char* raw = std::getenv("NMOE_RDEP_COUNTER_WAIT_TIMEOUT_MS");
        if (raw && raw[0] != '\0') {
            parsed = std::atoi(raw);
            if (parsed < 0) parsed = 0;
        }
        cached = parsed;
    }
    return cached;
}

__host__ __forceinline__ void rdep_trace_barrier_launch(
    const char* kind,
    const char* site,
    int ep_rank,
    int world,
    int phase) {
    if (!rdep_barrier_trace_enabled()) return;
    const int global_rank = rdep_env_global_rank();
    fprintf(
        stderr,
        "RDEP TRACE barrier launch kind=%s site=%s global_rank=%d ep_rank=%d world=%d phase=%d\n",
        kind,
        site ? site : "-",
        global_rank,
        ep_rank,
        world,
        phase
    );
    fflush(stderr);
}

__host__ __forceinline__ void rdep_watch_barrier_completion(
    cudaStream_t stream,
    const char* kind,
    const char* site,
    int ep_rank,
    int world,
    int phase) {
    const int timeout_ms = rdep_barrier_watchdog_ms();
    if (timeout_ms <= 0) return;

    cudaEvent_t done = nullptr;
    cudaError_t create = cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
    if (create != cudaSuccess) {
        fprintf(
            stderr,
            "RDEP ERROR: barrier watchdog create failed kind=%s site=%s err=%s\n",
            kind,
            site ? site : "-",
            cudaGetErrorString(create)
        );
        (void)cudaGetLastError();
        return;
    }

    cudaError_t rec = cudaEventRecord(done, stream);
    if (rec != cudaSuccess) {
        fprintf(
            stderr,
            "RDEP ERROR: barrier watchdog event record failed kind=%s site=%s err=%s\n",
            kind,
            site ? site : "-",
            cudaGetErrorString(rec)
        );
        (void)cudaGetLastError();
        (void)cudaEventDestroy(done);
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    while (true) {
        cudaError_t q = cudaEventQuery(done);
        if (q == cudaSuccess) break;
        if (q != cudaErrorNotReady) {
            fprintf(
                stderr,
                "RDEP ERROR: barrier watchdog query failed kind=%s site=%s err=%s\n",
                kind,
                site ? site : "-",
                cudaGetErrorString(q)
            );
            (void)cudaGetLastError();
            (void)cudaEventDestroy(done);
            abort();
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms > timeout_ms) {
            const int global_rank = rdep_env_global_rank();
            fprintf(
                stderr,
                "RDEP WATCHDOG timeout kind=%s site=%s global_rank=%d ep_rank=%d world=%d phase=%d timeout_ms=%d\n",
                kind,
                site ? site : "-",
                global_rank,
                ep_rank,
                world,
                phase,
                timeout_ms
            );
            fflush(stderr);
            (void)cudaEventDestroy(done);
            abort();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    (void)cudaEventDestroy(done);
}

// ============================================================================
// PTX Primitives - imported from ptx.cu
// ============================================================================
// Use nmoe::ptx:: namespace for all PTX primitives.
// See ptx.cu for the full list of IPC/P2P memory ordering primitives.

#include "ptx.cu"

using namespace nmoe::ptx;

// Forward declaration for swizzle_sf_strided (defined in quant.cu)
extern "C" cudaError_t swizzle_sf_strided(
    const void* sf_mkl,
    void* sf_mma,
    const int32_t* offs,
    int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
    cudaStream_t stream);

// Forward decls (defined below).
extern "C" void rdep_sync_buffer_ptrs_bf16();
static int dispatch_2phase_bf16(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K, int M,
    size_t meta_off, size_t counter_off, size_t dropped_off, size_t recv_counts_off, size_t recv_offsets_off,
    int capacity,
    int* recv_out_host,
    cudaStream_t stream);

// ============================================================================
// Helpers
// ============================================================================

__device__ __host__ __forceinline__ int H_aligned(int H) {
    return ((H + 7) / 8) * 8;
}

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

__host__ __forceinline__ bool host_ptr_is_pinned(const void* ptr) {
    if (ptr == nullptr) return false;
    unsigned int flags = 0;
    cudaError_t st = cudaHostGetFlags(&flags, const_cast<void*>(ptr));
    if (st == cudaSuccess) {
        return true;
    }
    if (st == cudaErrorInvalidValue) {
        // Clear sticky error from validation probe.
        (void)cudaGetLastError();
        return false;
    }
    fprintf(stderr, "RDEP ERROR: cudaHostGetFlags failed: %s\n", cudaGetErrorString(st));
    return false;
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
    // Warp-stride kernels scale well with a moderate CTA cap.
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

__host__ __forceinline__ bool validate_pinned_host_int(const int* ptr, const char* name) {
    if (ptr == nullptr) {
        fprintf(stderr, "RDEP ERROR: %s (host scratch) is null\n", name);
        return false;
    }
    // Steady-state hot path uses stable pinned scratch pointers; avoid repeated driver probes.
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

struct AsyncDeviceIntPollSlot {
    int* h_value = nullptr;
    cudaEvent_t ready = nullptr;
    const int* d_ptr = nullptr;
    uintptr_t stream_key = 0;
    bool pending = false;
    int last_value = 0;
};

struct AsyncDeviceIntPollTable {
    static constexpr int kMaxSlots = 128;
    AsyncDeviceIntPollSlot slots[kMaxSlots];
    std::vector<AsyncDeviceIntPollSlot> spill_slots;
    int next_evict = 0;
};
static_assert((AsyncDeviceIntPollTable::kMaxSlots & (AsyncDeviceIntPollTable::kMaxSlots - 1)) == 0,
              "AsyncDeviceIntPollTable::kMaxSlots must be power-of-two");

__host__ __forceinline__ AsyncDeviceIntPollSlot* get_async_poll_slot(const int* d_ptr, cudaStream_t stream) {
    if (d_ptr == nullptr) return nullptr;
    const uintptr_t stream_key = reinterpret_cast<uintptr_t>(stream) + 1;
    thread_local AsyncDeviceIntPollTable table;

    for (int i = 0; i < AsyncDeviceIntPollTable::kMaxSlots; ++i) {
        if (table.slots[i].d_ptr == d_ptr && table.slots[i].stream_key == stream_key) {
            return &table.slots[i];
        }
    }
    for (auto& spill : table.spill_slots) {
        if (spill.d_ptr == d_ptr && spill.stream_key == stream_key) {
            return &spill;
        }
    }
    for (int i = 0; i < AsyncDeviceIntPollTable::kMaxSlots; ++i) {
        if (table.slots[i].d_ptr == nullptr) {
            table.slots[i].d_ptr = d_ptr;
            table.slots[i].stream_key = stream_key;
            return &table.slots[i];
        }
    }
    for (auto& spill : table.spill_slots) {
        if (spill.d_ptr == nullptr) {
            spill.d_ptr = d_ptr;
            spill.stream_key = stream_key;
            spill.pending = false;
            spill.last_value = 0;
            return &spill;
        }
    }

    auto refresh_slot = [](AsyncDeviceIntPollSlot* cand) {
        if (cand->pending && cand->ready != nullptr) {
            cudaError_t q = cudaEventQuery(cand->ready);
            if (q == cudaSuccess) {
                if (cand->h_value != nullptr) {
                    cand->last_value = *cand->h_value;
                }
                cand->pending = false;
            } else if (q != cudaErrorNotReady) {
                (void)cudaGetLastError();
                cand->pending = false;
            }
        }
    };

    // Prefer reusing a non-pending slot to avoid reassigning while a D2H copy
    // is still in flight.
    for (int i = 0; i < AsyncDeviceIntPollTable::kMaxSlots; ++i) {
        const int idx = (table.next_evict + i) & (AsyncDeviceIntPollTable::kMaxSlots - 1);
        AsyncDeviceIntPollSlot* cand = &table.slots[idx];
        refresh_slot(cand);
        if (!cand->pending) {
            table.next_evict = (idx + 1) & (AsyncDeviceIntPollTable::kMaxSlots - 1);
            cand->d_ptr = d_ptr;
            cand->stream_key = stream_key;
            cand->last_value = 0;
            return cand;
        }
    }
    for (auto& spill : table.spill_slots) {
        refresh_slot(&spill);
        if (!spill.pending) {
            spill.d_ptr = d_ptr;
            spill.stream_key = stream_key;
            spill.last_value = 0;
            return &spill;
        }
    }

    // Avoid host-side blocking when the fixed table is saturated: extend with
    // per-thread spill slots and keep reads stream-ordered.
    table.spill_slots.emplace_back();
    AsyncDeviceIntPollSlot* spill = &table.spill_slots.back();
    spill->d_ptr = d_ptr;
    spill->stream_key = stream_key;
    spill->pending = false;
    spill->last_value = 0;
    return spill;
}

__host__ __forceinline__ int poll_device_int_async(const int* d_ptr, cudaStream_t stream) {
    if (d_ptr == nullptr) return 0;
    AsyncDeviceIntPollSlot* st = get_async_poll_slot(d_ptr, stream);
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

__host__ __forceinline__ bool complete_device_int_read_blocking(
    AsyncDeviceIntPollSlot* st,
    bool* ok_out,
    const char* context) {
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
        const int timeout_ms = rdep_counter_wait_timeout_ms();
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
    AsyncDeviceIntPollSlot* st = get_async_poll_slot(d_ptr, stream);
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

    // If a prior async read is already in flight for this pointer, complete it
    // instead of issuing another memcpy.
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

// Strict host-blocking device-int read used on dispatch hot paths.
// We intentionally avoid cudaStreamSynchronize(stream) here: event-scoped wait
// preserves ordering while avoiding an explicit full-stream host sync primitive.
__host__ __forceinline__ int read_device_int_stream_sync(
    const int* d_ptr,
    cudaStream_t stream,
    bool* ok_out = nullptr) {
    return read_device_int_blocking(d_ptr, stream, ok_out);
}

// ============================================================================
// FP8 E4M3 / FP4 E2M1 Conversion - use ptx.cu versions
// ============================================================================
// Aliases to match existing code using local names

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

// ============================================================================
// GPU-Side Barrier (DeepEP pattern)
// ============================================================================
// Cross-GPU barrier using atomicAdd_system/atomicSub_system.
// Each rank adds to its own signal array and subtracts from other ranks' arrays.
// When all signals reach zero, barrier is complete.
//
// Pattern: barrier_signal_ptrs[rank][i] tracks arrivals from rank i.
// - Add BARRIER_TAG to own array: barrier_signal_ptrs[my_rank][thread_id]
// - Sub BARRIER_TAG from other array: barrier_signal_ptrs[thread_id][my_rank]
// - Poll until all values <= 0

template <int kNumRanks, bool kSyncOnly = false>
__device__ __forceinline__ void
barrier_block(int** barrier_signal_ptrs, int rank) {
    int thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, flush all P2P writes to system scope
    if constexpr (!kSyncOnly) {
        fence_acq_rel_sys();
        __syncthreads();
    }

    // Add to self signals, subtract from others
    // This ensures all ranks must have arrived before any can proceed
    if (thread_id < kNumRanks) {
        atomicAdd_sys(barrier_signal_ptrs[rank] + thread_id, BARRIER_TAG);
        atomicSub_sys(barrier_signal_ptrs[thread_id] + rank, BARRIER_TAG);
    }

    // Wait for all signals to reach zero (all ranks have arrived)
    uint64_t start_time = clock64();
    while (true) {
        int value = (thread_id < kNumRanks) ? ld_volatile_s32(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0))
            break;

        // Timeout check
        if (clock64() - start_time > TIMEOUT_CYCLES && thread_id < kNumRanks) {
            printf("nmoe barrier timeout: rank=%d, thread=%d, value=%d\n", rank, thread_id, value);
            trap();
        }
    }
    // Acquire remote writes that happened-before peers signaled arrival.
    fence_acq_rel_sys();
    __syncthreads();
}

// ============================================================================
// Metadata (16-byte aligned)
// ============================================================================

struct alignas(16) Meta {
    int64_t row_id;
    int32_t local_eid;
    float   gate;
};
static_assert(sizeof(Meta) == 16, "Meta must be 16 bytes");

// ============================================================================
// Mode Selection
// ============================================================================

enum RdepMode {
    MODE_SINGLE = 0,   // world=1, local only
    MODE_IPC = 1,      // world=local_world, CUDA IPC
    MODE_HYBRID = 2    // world>local_world, IPC intra-node + NVSHMEM inter-node
};

static RdepMode g_mode = MODE_SINGLE;

// ============================================================================
// Global State with IPC Buffer Pointers
// ============================================================================

	struct StateBF16 {
	    // IPC buffer pointers - [rank] -> remote buffer on that rank
	    // buffer_ptrs[my_rank] is local cudaMalloc, others are IPC-opened
	    void* buffer_ptrs[MAX_RANKS];

    // Barrier signal pointers - [rank] -> signal array on that rank
    // Each rank's buffer has MAX_RANKS ints for barrier signals
    int* barrier_signal_ptrs[MAX_RANKS];

	    // Buffer layout within each rank's allocation (IPC, BF16):
	    // [capacity * Ha * sizeof(uint16_t)]          - x_buf (BF16 activations, dispatch receive)
	    // [capacity * sizeof(Meta)]                   - meta (dispatch metadata)
	    // [sizeof(int)]                               - counter (legacy append counter; avoided on hot paths where possible)
	    // [sizeof(int)]                               - dropped (dispatch overflow counter)
	    // [MAX_RANKS * sizeof(int)]                   - barrier_signals (GPU-side sync)
	    // [MAX_RANKS * sizeof(void*)]                 - buffer_ptrs_gpu (pointers on GPU)
	    // [MAX_RANKS * sizeof(int*)]                  - barrier_signal_ptrs_gpu
	    // [tok_slots * Ha * sizeof(uint16_t)]         - tok_y (BF16 per-(tok,slot) buffer, used for return/dX)
	    // [tok_slots * sizeof(float)]                 - tok_gate (float per-(tok,slot) buffer, used for return gating / scratch)
	    //
	    // Where tok_slots = capacity / world (must be >= T*K).

    // Local work buffers
    int*      local_eid;
    int*      order;
    int*      offsets;
    int*      offs_pad;
    int*      offs_pad_last;  // non-owning: last caller-provided offs_pad_out
    int*      dest;
    int*      M_pad_dev;
    Meta*     meta_copy;
    void*     sort_temp;
    size_t    sort_temp_bytes;

    // 2-phase dispatch: local atomic counters for ordering within rank's sends
    int* local_counters;  // [MAX_RANKS] - local atomics for 2-phase dispatch

    // Dimensions
    size_t capacity;
    size_t buffer_size;  // Total bytes per rank
    int M_pad;
    int H, Ha;
    int world, rank;
    int n_local;
    int align;
    bool initialized;

};

		struct StateBlockscaled {
    void* buffer_ptrs[MAX_RANKS];
    int* barrier_signal_ptrs[MAX_RANKS];

    // Buffer layout:
    // [capacity * Hp * sizeof(uint16_t)]  - x_buf (packed)
    // [capacity * Hsf * sizeof(uint8_t)]  - sfa_buf
    // [capacity * H * sizeof(uint16_t)]   - y_buf (return BF16)
    // [capacity * sizeof(Meta)]           - meta
    // [sizeof(int)]                        - counter
    // [sizeof(int)]                        - dropped
    // [MAX_RANKS * sizeof(int)]           - barrier_signals
    // [MAX_RANKS * sizeof(void*)]         - buffer_ptrs_gpu
    // [MAX_RANKS * sizeof(int*)]          - barrier_signal_ptrs_gpu
    // [tok_slots * Ha * sizeof(uint16_t)] - tok_y (BF16 per-(tok,slot) scratch for return/dX)
    // [tok_slots * sizeof(float)]         - tok_gate (float per-(tok,slot) scratch for return/dGate)
    // [MAX_RANKS * sizeof(int)]           - recv_counts[src]  (2-phase dispatch)
    // [MAX_RANKS * sizeof(int)]           - recv_offsets[src] (2-phase dispatch)

    int*      local_eid;
    int*      order;
    int*      offsets;
    int*      offs_pad;
    int*      offs_pad_last;  // non-owning: last caller-provided offs_pad_out
    int*      dest;
    int*      M_pad_dev;
    void*     sort_temp;
    size_t    sort_temp_bytes;
    int*      local_counters;  // [MAX_RANKS] local atomics for 2-phase dispatch

    size_t capacity;
    size_t buffer_size;
    int M_pad;
    int H, Ha, Hp, Hsf;
    int world, rank;
    int n_local;
	    int align;
	    int profile;
		    bool initialized;
		};

static StateBF16 g_bf16 = {};
static StateBlockscaled g_block = {};

// IPC handles stored for cleanup
static cudaIpcMemHandle_t g_ipc_handles_bf16[MAX_RANKS];
static cudaIpcMemHandle_t g_ipc_handles_block[MAX_RANKS];

// Single IPC allocation shared between BF16 and blockscaled buffer layouts.
// We never need both payload layouts live at the same time; the mode/profile selects
// which offsets are active. This avoids allocating two large per-rank staging buffers.
static void* g_ipc_shared_buf = nullptr;
static size_t g_ipc_shared_bytes = 0;
static bool g_ipc_handles_opened = false;

static std::atomic<int> g_ipc_phase_bf16{0};
static std::atomic<int> g_ipc_phase_block{0};
static std::atomic<uintptr_t> g_ipc_stream_bf16{0};
static std::atomic<uintptr_t> g_ipc_stream_block{0};

// 2-phase dispatch is always enabled for IPC mode in production (eliminates remote atomics).
constexpr bool k_use_2phase_dispatch = true;

// ============================================================================
// Helper: Get buffer offsets (following DeepEP layout pattern)
// ============================================================================

__host__ __device__ __forceinline__
size_t align_up(size_t x, size_t align) {
    return ((x + align - 1) / align) * align;
}

	__host__ __device__ __forceinline__
	void bf16_buffer_offsets(size_t capacity, int Ha, int world,
	                         size_t* x_off, size_t* meta_off,
	                         size_t* counter_off, size_t* dropped_off,
	                         size_t* barrier_off, size_t* buf_ptrs_off, size_t* sig_ptrs_off,
	                         size_t* tok_y_off, size_t* tok_gate_off,
	                         size_t* total_size,
	                         // 2-phase dispatch areas (optional, can be nullptr)
	                         size_t* recv_counts_off = nullptr,
	                         size_t* recv_offsets_off = nullptr) {
	    *x_off = 0;
	    *meta_off = capacity * Ha * sizeof(uint16_t);
	    *counter_off = *meta_off + capacity * sizeof(Meta);
	    *dropped_off = *counter_off + sizeof(int);
	    // Align barrier signals for atomic operations
	    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
	    *buf_ptrs_off = *barrier_off + MAX_RANKS * sizeof(int);
	    *sig_ptrs_off = *buf_ptrs_off + MAX_RANKS * sizeof(void*);
	    const size_t ptrs_end = *sig_ptrs_off + MAX_RANKS * sizeof(int*);

	    // Token-slot buffers (fixed size per rank; used by IPC return/dX to avoid append counters).
	    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
	    *tok_y_off = align_up(ptrs_end, BUFFER_ALIGNMENT);
	    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);

	    // 2-phase dispatch: count exchange area (MAX_RANKS ints per rank for recv_from[src] counts)
	    // Layout: recv_counts[src] = how many tokens I receive from rank src
	    //         recv_offsets[src] = where rank src's data starts in my buffer (prefix sum)
	    const size_t tok_gate_end = *tok_gate_off + tok_slots * sizeof(float);
	    const size_t _recv_counts_off = align_up(tok_gate_end, BUFFER_ALIGNMENT);
	    const size_t _recv_offsets_off = _recv_counts_off + MAX_RANKS * sizeof(int);

	    if (recv_counts_off) *recv_counts_off = _recv_counts_off;
	    if (recv_offsets_off) *recv_offsets_off = _recv_offsets_off;

	    *total_size = align_up(_recv_offsets_off + MAX_RANKS * sizeof(int), BUFFER_ALIGNMENT);
	}

__host__ __device__ __forceinline__
void blockscaled_buffer_offsets(size_t capacity, int H, int Hp, int Hsf,
                                int world,
                                size_t* x_off, size_t* sfa_off, size_t* y_off,
                                size_t* meta_off, size_t* counter_off, size_t* dropped_off,
                                size_t* barrier_off, size_t* buf_ptrs_off, size_t* sig_ptrs_off,
                                size_t* tok_y_off, size_t* tok_gate_off,
                                size_t* total_size,
                                // 2-phase dispatch areas (optional, can be nullptr)
                                size_t* recv_counts_off = nullptr,
                                size_t* recv_offsets_off = nullptr) {
    *x_off = 0;
    *sfa_off = capacity * Hp * sizeof(uint16_t);
    *y_off = *sfa_off + capacity * Hsf * sizeof(uint8_t);
    *meta_off = *y_off + capacity * H * sizeof(uint16_t);
    *counter_off = *meta_off + capacity * sizeof(Meta);
    *dropped_off = *counter_off + sizeof(int);
    *barrier_off = align_up(*dropped_off + sizeof(int), BUFFER_ALIGNMENT);
    *buf_ptrs_off = *barrier_off + MAX_RANKS * sizeof(int);
    *sig_ptrs_off = *buf_ptrs_off + MAX_RANKS * sizeof(void*);
    const size_t ptrs_end = *sig_ptrs_off + MAX_RANKS * sizeof(int*);

    // Token-slot buffers (fixed size per rank; used by IPC return/dX).
    const size_t tok_slots = (world > 0) ? (capacity / static_cast<size_t>(world)) : 0;
    const int Ha = H_aligned(H);
    *tok_y_off = align_up(ptrs_end, BUFFER_ALIGNMENT);
    *tok_gate_off = align_up(*tok_y_off + tok_slots * static_cast<size_t>(Ha) * sizeof(uint16_t), BUFFER_ALIGNMENT);

    const size_t tok_gate_end = *tok_gate_off + tok_slots * sizeof(float);
    const size_t _recv_counts_off = align_up(tok_gate_end, BUFFER_ALIGNMENT);
    const size_t _recv_offsets_off = _recv_counts_off + MAX_RANKS * sizeof(int);

    if (recv_counts_off) *recv_counts_off = _recv_counts_off;
    if (recv_offsets_off) *recv_offsets_off = _recv_offsets_off;

    *total_size = align_up(_recv_offsets_off + MAX_RANKS * sizeof(int), BUFFER_ALIGNMENT);
}

static void rdep_ensure_ipc_shared_buffer(size_t bytes_needed) {
    if (bytes_needed <= g_ipc_shared_bytes) return;
    if (g_ipc_handles_opened) {
        fprintf(stderr,
                "RDEP FATAL: cannot resize IPC shared buffer after IPC handles are opened "
                "(requested=%zu current=%zu). This indicates a configuration change between "
                "alloc() and the current dispatch. Possible causes:\n"
                "  1. rdep_alloc_bf16/blockscaled called with different capacity after IPC open\n"
                "  2. Model configuration changed between init and training\n"
                "Fix: ensure alloc() is called with final capacity BEFORE open_ipc_handles().\n",
                bytes_needed, g_ipc_shared_bytes);
        abort();
    }

    if (g_ipc_shared_buf) {
        cudaFree(g_ipc_shared_buf);
        g_ipc_shared_buf = nullptr;
        g_ipc_shared_bytes = 0;
    }

    cudaMalloc(&g_ipc_shared_buf, bytes_needed);
    cudaMemset(g_ipc_shared_buf, 0, bytes_needed);
    g_ipc_shared_bytes = bytes_needed;
}

// ============================================================================
// Init / Alloc - IPC Setup
// ============================================================================

// Get local IPC handle (call on each rank after alloc)
extern "C" void rdep_get_ipc_handle_bf16(void* handle_out) {
    if (!g_bf16.initialized || !g_bf16.buffer_ptrs[g_bf16.rank]) return;
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, g_bf16.buffer_ptrs[g_bf16.rank]);
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

extern "C" void rdep_get_ipc_handle_blockscaled(void* handle_out) {
    if (!g_block.initialized || !g_block.buffer_ptrs[g_block.rank]) return;
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, g_block.buffer_ptrs[g_block.rank]);
    memcpy(handle_out, &handle, sizeof(cudaIpcMemHandle_t));
}

// Initialize with rank info and determine mode
// Mode is auto-selected based on world vs local_world:
//   - world=1: MODE_SINGLE (local only)
//   - world=local_world: MODE_IPC (CUDA IPC for intra-node)
//   - world>local_world: MODE_HYBRID (IPC intra-node + NVSHMEM inter-node)
extern "C" void rdep_init(int rank, int world, int local_world) {
    g_bf16.rank = rank;
    g_bf16.world = world;
    g_block.rank = rank;
    g_block.world = world;

    // Mode selection
    if (world == 1) {
        g_mode = MODE_SINGLE;
        return;
    } else if (world == local_world) {
        g_mode = MODE_IPC;
    } else {
        g_mode = MODE_HYBRID;
#ifndef WITH_NVSHMEM
        fprintf(stderr, "RDEP ERROR: Multi-node (world=%d > local_world=%d) requires NVSHMEM.\n", world, local_world);
        fprintf(stderr, "           Rebuild with NVSHMEM support or use single-node configuration.\n");
        exit(1);
#endif
    }

    // For IPC mode: check P2P access for local peers only
    // For HYBRID mode: check P2P for local peers, NVSHMEM handles inter-node
    int my_device = -1;
    cudaError_t dev_err = cudaGetDevice(&my_device);
    if (dev_err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaGetDevice failed in rdep_init: %s\n", cudaGetErrorString(dev_err));
        abort();
    }
    int local_rank = rank % local_world;

    for (int peer = 0; peer < local_world; peer++) {
        if (peer == local_rank) continue;
        int can_access = 0;
        cudaError_t p2p_err = cudaDeviceCanAccessPeer(&can_access, my_device, peer);
        if (p2p_err != cudaSuccess) {
            fprintf(stderr, "RDEP FATAL: cudaDeviceCanAccessPeer(%d,%d) failed: %s\n",
                    my_device, peer, cudaGetErrorString(p2p_err));
            abort();
        }
        if (!can_access) {
            fprintf(stderr, "RDEP FATAL: GPU %d cannot access local peer GPU %d\n", my_device, peer);
            abort();
        }
        int native_atomic = 0;
        cudaError_t attr_err = cudaDeviceGetP2PAttribute(
            &native_atomic, cudaDevP2PAttrNativeAtomicSupported, my_device, peer);
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "RDEP FATAL: cudaDeviceGetP2PAttribute(native atomic) failed for %d->%d: %s\n",
                    my_device, peer, cudaGetErrorString(attr_err));
            abort();
        }
        if (!native_atomic) {
            fprintf(stderr, "RDEP FATAL: Native atomics not supported between GPU %d and %d\n", my_device, peer);
            abort();
        }
    }

#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        // NVSHMEM initialization will be done separately after UID broadcast
    }
#endif
}

// Query current mode
extern "C" int rdep_get_mode() {
    return static_cast<int>(g_mode);
}

// Check if NVSHMEM support is compiled in
extern "C" bool rdep_has_nvshmem() {
#ifdef WITH_NVSHMEM
    return true;
#else
    return false;
#endif
}

// Open remote IPC handles after all_gather
extern "C" void rdep_open_ipc_handles_bf16(const void* handles, int world) {
    if (world < 1 || world > MAX_RANKS) {
        fprintf(stderr, "RDEP FATAL: invalid world=%d for IPC handle open (MAX_RANKS=%d)\n", world, MAX_RANKS);
        g_ipc_handles_opened = false;
        abort();
    }
    const cudaIpcMemHandle_t* all_handles = static_cast<const cudaIpcMemHandle_t*>(handles);
    int my_rank = g_bf16.rank;

    // Calculate barrier offset
    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    g_ipc_handles_opened = false;
    bool opened_remote[MAX_RANKS] = {false};
    for (int r = 0; r < world; ++r) {
        if (r != my_rank) {
            g_bf16.buffer_ptrs[r] = nullptr;
            g_bf16.barrier_signal_ptrs[r] = nullptr;
        }
    }

    for (int r = 0; r < world; r++) {
        if (r == my_rank) {
            // Local buffer already allocated
            memcpy(&g_ipc_handles_bf16[r], &all_handles[r], sizeof(cudaIpcMemHandle_t));
            if (g_bf16.buffer_ptrs[r] == nullptr) {
                fprintf(stderr, "RDEP FATAL: local IPC buffer pointer is null for rank %d\n", r);
                abort();
            }
            char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[r]);
            g_bf16.barrier_signal_ptrs[r] = reinterpret_cast<int*>(local_buf + barrier_off);
        } else {
            // Open remote buffer
            memcpy(&g_ipc_handles_bf16[r], &all_handles[r], sizeof(cudaIpcMemHandle_t));
            cudaError_t err = cudaIpcOpenMemHandle(&g_bf16.buffer_ptrs[r], g_ipc_handles_bf16[r],
                                 cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                fprintf(stderr, "RDEP FATAL: cudaIpcOpenMemHandle failed for rank %d: %s (err=%d)\n",
                        r, cudaGetErrorString(err), (int)err);
                g_bf16.buffer_ptrs[r] = nullptr;
                g_bf16.barrier_signal_ptrs[r] = nullptr;
                for (int rr = 0; rr < world; ++rr) {
                    if (opened_remote[rr] && g_bf16.buffer_ptrs[rr]) {
                        cudaIpcCloseMemHandle(g_bf16.buffer_ptrs[rr]);
                        g_bf16.buffer_ptrs[rr] = nullptr;
                        g_bf16.barrier_signal_ptrs[rr] = nullptr;
                    }
                }
                abort();
            }
            opened_remote[r] = true;
            // Set barrier signal pointer for remote buffer
            char* remote_buf = static_cast<char*>(g_bf16.buffer_ptrs[r]);
            g_bf16.barrier_signal_ptrs[r] = remote_buf ? reinterpret_cast<int*>(remote_buf + barrier_off) : nullptr;
        }
    }

    for (int r = 0; r < world; ++r) {
        if (g_bf16.buffer_ptrs[r] == nullptr || g_bf16.barrier_signal_ptrs[r] == nullptr) {
            fprintf(stderr,
                    "RDEP FATAL: incomplete IPC mapping for rank %d (buf=%p barrier=%p)\n",
                    r, g_bf16.buffer_ptrs[r], g_bf16.barrier_signal_ptrs[r]);
            for (int rr = 0; rr < world; ++rr) {
                if (opened_remote[rr] && g_bf16.buffer_ptrs[rr]) {
                    cudaIpcCloseMemHandle(g_bf16.buffer_ptrs[rr]);
                    g_bf16.buffer_ptrs[rr] = nullptr;
                    g_bf16.barrier_signal_ptrs[rr] = nullptr;
                }
            }
            abort();
        }
    }
    g_ipc_handles_opened = true;
}

extern "C" void rdep_open_ipc_handles_blockscaled(const void* handles, int world) {
    if (world < 1 || world > MAX_RANKS) {
        fprintf(stderr, "RDEP FATAL: invalid world=%d for blockscaled IPC handle open (MAX_RANKS=%d)\n", world, MAX_RANKS);
        abort();
    }
    // Blockscaled payload shares the same underlying IPC buffer mapping as BF16.
    // We only need to compute blockscaled barrier signal pointers and alias base pointers.
    if (!g_ipc_handles_opened) {
        fprintf(stderr,
                "RDEP FATAL: open_ipc_handles_blockscaled requires BF16 IPC handles to be opened first\n");
        abort();
    }
    // Calculate barrier offset
    [[maybe_unused]] size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    size_t barrier_off;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);

    const cudaIpcMemHandle_t* all_handles = static_cast<const cudaIpcMemHandle_t*>(handles);
    for (int r = 0; r < world; r++) {
        if (memcmp(&all_handles[r], &g_ipc_handles_bf16[r], sizeof(cudaIpcMemHandle_t)) != 0) {
            fprintf(stderr,
                    "RDEP FATAL: BF16 and blockscaled IPC handles differ for rank %d; "
                    "expected shared buffer allocation\n",
                    r);
            abort();
        }
        if (g_bf16.buffer_ptrs[r] == nullptr) {
            fprintf(stderr, "RDEP FATAL: missing BF16 IPC buffer pointer for rank %d in blockscaled open\n", r);
            abort();
        }
        g_block.buffer_ptrs[r] = g_bf16.buffer_ptrs[r];
        char* buf = static_cast<char*>(g_block.buffer_ptrs[r]);
        if (!buf) {
            fprintf(stderr, "RDEP FATAL: missing IPC buffer pointer for rank %d in blockscaled open\n", r);
            abort();
        }
        g_block.barrier_signal_ptrs[r] = reinterpret_cast<int*>(buf + barrier_off);
        memcpy(&g_ipc_handles_block[r], &all_handles[r], sizeof(cudaIpcMemHandle_t));
    }
}

// Allocate local buffer (BF16 path)
extern "C" void rdep_alloc_bf16(size_t capacity, int H, int n_local) {
    if (n_local <= 0) {
        fprintf(stderr, "RDEP FATAL: rdep_alloc_bf16 requires n_local > 0, got %d\n", n_local);
        abort();
    }
    int Ha = H_aligned(H);

    // Calculate buffer layout
    [[maybe_unused]] size_t x_off, meta_off, counter_off, dropped_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off;
    size_t barrier_off, total_size;
    bf16_buffer_offsets(capacity, Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    // Free old per-mode workspaces (shared IPC buffer is managed separately).
    if (g_bf16.local_eid) cudaFree(g_bf16.local_eid);
    if (g_bf16.order) cudaFree(g_bf16.order);
    if (g_bf16.offsets) cudaFree(g_bf16.offsets);
    if (g_bf16.offs_pad) cudaFree(g_bf16.offs_pad);
    if (g_bf16.dest) cudaFree(g_bf16.dest);
    if (g_bf16.M_pad_dev) cudaFree(g_bf16.M_pad_dev);
    if (g_bf16.meta_copy) cudaFree(g_bf16.meta_copy);
    if (g_bf16.sort_temp) cudaFree(g_bf16.sort_temp);
    if (g_bf16.local_counters) cudaFree(g_bf16.local_counters);

    rdep_ensure_ipc_shared_buffer(total_size);
    g_bf16.buffer_ptrs[g_bf16.rank] = g_ipc_shared_buf;
    // Blockscaled and BF16 payloads share the same base IPC allocation.
    // If blockscaled was allocated first with a smaller buffer and we just
    // grew the shared buffer, refresh blockscaled's pointer to the new one.
    g_block.buffer_ptrs[g_block.rank] = g_ipc_shared_buf;
    g_block.buffer_size = g_ipc_shared_bytes;
    if (g_block.initialized) {
        int block_Hp = g_block.Hp;
        int block_Hsf = g_block.Hsf;
        [[maybe_unused]] size_t bk_x_off, bk_sfa_off, bk_y_off, bk_meta_off, bk_counter_off, bk_dropped_off, bk_buf_ptrs_off, bk_sig_ptrs_off, bk_tok_y_off, bk_tok_gate_off;
        size_t bk_barrier_off, bk_total_size;
        blockscaled_buffer_offsets(g_block.capacity, g_block.H, block_Hp, block_Hsf, g_block.world,
                                   &bk_x_off, &bk_sfa_off, &bk_y_off,
                                   &bk_meta_off, &bk_counter_off, &bk_dropped_off,
                                   &bk_barrier_off, &bk_buf_ptrs_off, &bk_sig_ptrs_off,
                                   &bk_tok_y_off, &bk_tok_gate_off,
                                   &bk_total_size);
        char* bk_local = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
        g_block.barrier_signal_ptrs[g_block.rank] = reinterpret_cast<int*>(bk_local + bk_barrier_off);
    }

    // Set local barrier signal pointer
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    g_bf16.barrier_signal_ptrs[g_bf16.rank] = reinterpret_cast<int*>(local_buf + barrier_off);
    // Reset local barrier slots so phase reset cannot observe stale signals.
    cudaMemset(local_buf + barrier_off, 0, MAX_RANKS * sizeof(int));

    // Allocate work buffers
    cudaMalloc(&g_bf16.local_eid, capacity * sizeof(int));
    cudaMalloc(&g_bf16.order, capacity * sizeof(int));
    cudaMalloc(&g_bf16.offsets, (n_local + 1) * sizeof(int));
    cudaMalloc(&g_bf16.offs_pad, n_local * sizeof(int));
    cudaMalloc(&g_bf16.dest, capacity * sizeof(int));
    cudaMalloc(&g_bf16.M_pad_dev, sizeof(int));
    cudaMalloc(&g_bf16.meta_copy, capacity * sizeof(Meta));

    g_bf16.sort_temp = nullptr;
    g_bf16.sort_temp_bytes = 0;
    if (n_local > 1 && capacity > 1) {
        cub::DeviceRadixSort::SortPairs(nullptr, g_bf16.sort_temp_bytes,
            g_bf16.local_eid, g_bf16.local_eid, g_bf16.order, g_bf16.order, (int)capacity);
        if (g_bf16.sort_temp_bytes > 0) {
            cudaMalloc(&g_bf16.sort_temp, g_bf16.sort_temp_bytes);
        }
    }

    // 2-phase dispatch: local atomic counters (one per destination rank)
    cudaMalloc(&g_bf16.local_counters, MAX_RANKS * sizeof(int));

    g_bf16.capacity = capacity;
    g_bf16.buffer_size = g_ipc_shared_bytes;
    g_bf16.H = H;
    g_bf16.Ha = Ha;
    g_bf16.n_local = n_local;
    g_bf16.align = 128;  // Match blockscaled for consistent padding
    g_bf16.initialized = true;
    g_ipc_phase_bf16.store(0, std::memory_order_relaxed);
    g_ipc_stream_bf16.store(0, std::memory_order_relaxed);
}

// Allocate local buffer (Blockscaled path)
	extern "C" void rdep_alloc_blockscaled(size_t capacity, int H, int n_local, int profile) {
    if (n_local <= 0) {
        fprintf(stderr, "RDEP FATAL: rdep_alloc_blockscaled requires n_local > 0, got %d\n", n_local);
        abort();
    }
    if (profile < 0 || profile > 1) {
        fprintf(stderr, "RDEP FATAL: invalid profile=%d in rdep_alloc_blockscaled (must be 0=fp8 or 1=nvfp4)\n", profile);
        abort();
    }
    int pack_factor = (profile == 0) ? 2 : 4;
    if ((H % pack_factor) != 0) {
        fprintf(stderr,
                "RDEP FATAL: H=%d must be divisible by pack_factor=%d for profile=%s\n",
                H, pack_factor, (profile == 0 ? "fp8" : "nvfp4"));
        abort();
    }
    int Hp = H / pack_factor;
    int Hsf = (H + SF_VEC - 1) / SF_VEC;
    int Ha = H_aligned(H);
    const int align = 128;

    [[maybe_unused]] size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off;
    size_t barrier_off, total_size;
    blockscaled_buffer_offsets(capacity, H, Hp, Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off,
                               &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);

    // Free old per-mode workspaces (shared IPC buffer is managed separately).
    if (g_block.local_eid) cudaFree(g_block.local_eid);
    if (g_block.order) cudaFree(g_block.order);
    if (g_block.offsets) cudaFree(g_block.offsets);
    if (g_block.offs_pad) cudaFree(g_block.offs_pad);
	    if (g_block.dest) cudaFree(g_block.dest);
	    if (g_block.M_pad_dev) cudaFree(g_block.M_pad_dev);
	    if (g_block.sort_temp) cudaFree(g_block.sort_temp);
	    if (g_block.local_counters) cudaFree(g_block.local_counters);

    rdep_ensure_ipc_shared_buffer(total_size);
    g_block.buffer_ptrs[g_block.rank] = g_ipc_shared_buf;
    // BF16 and blockscaled payloads share the same base IPC allocation.
    g_bf16.buffer_ptrs[g_bf16.rank] = g_ipc_shared_buf;
    g_bf16.buffer_size = g_ipc_shared_bytes;
    if (g_bf16.initialized) {
        [[maybe_unused]] size_t bf16_x_off, bf16_meta_off, bf16_counter_off, bf16_dropped_off, bf16_buf_ptrs_off, bf16_sig_ptrs_off, bf16_tok_y_off, bf16_tok_gate_off;
        size_t bf16_barrier_off, bf16_total_size;
        bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                            &bf16_x_off, &bf16_meta_off, &bf16_counter_off, &bf16_dropped_off,
                            &bf16_barrier_off, &bf16_buf_ptrs_off, &bf16_sig_ptrs_off,
                            &bf16_tok_y_off, &bf16_tok_gate_off,
                            &bf16_total_size);
        char* bf16_local = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
        g_bf16.barrier_signal_ptrs[g_bf16.rank] = reinterpret_cast<int*>(bf16_local + bf16_barrier_off);
    }

    // Set local barrier signal pointer
    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    g_block.barrier_signal_ptrs[g_block.rank] = reinterpret_cast<int*>(local_buf + barrier_off);
    // Reset local barrier slots so phase reset cannot observe stale signals.
    cudaMemset(local_buf + barrier_off, 0, MAX_RANKS * sizeof(int));

    cudaMalloc(&g_block.local_eid, capacity * sizeof(int));
    cudaMalloc(&g_block.order, capacity * sizeof(int));
    cudaMalloc(&g_block.offsets, (n_local + 1) * sizeof(int));
    cudaMalloc(&g_block.offs_pad, n_local * sizeof(int));
    cudaMalloc(&g_block.dest, capacity * sizeof(int));
    cudaMalloc(&g_block.M_pad_dev, sizeof(int));

    g_block.sort_temp = nullptr;
    g_block.sort_temp_bytes = 0;
    if (n_local > 1 && capacity > 1) {
        cub::DeviceRadixSort::SortPairs(nullptr, g_block.sort_temp_bytes,
            g_block.local_eid, g_block.local_eid, g_block.order, g_block.order, (int)capacity);
        if (g_block.sort_temp_bytes > 0) {
            cudaMalloc(&g_block.sort_temp, g_block.sort_temp_bytes);
        }
    }
    cudaMalloc(&g_block.local_counters, MAX_RANKS * sizeof(int));

    g_block.capacity = capacity;
    g_block.buffer_size = g_ipc_shared_bytes;
    g_block.H = H;
    g_block.Ha = Ha;
    g_block.Hp = Hp;
    g_block.Hsf = Hsf;
    g_block.n_local = n_local;
	    g_block.align = align;
	    g_block.profile = profile;
	    g_block.initialized = true;
	    g_ipc_phase_block.store(0, std::memory_order_relaxed);
	    g_ipc_stream_block.store(0, std::memory_order_relaxed);
	    if (g_mode == MODE_SINGLE) {
	        // rdep.py syncs BF16 pointers before alloc_blockscaled in MODE_SINGLE.
	        // If alloc_blockscaled resized the shared buffer, refresh device symbols here.
	        rdep_sync_buffer_ptrs_bf16();
	    }
	}

// Read dropped tokens from local receive buffer (BF16 path).
// Returns the last completed async sample (non-blocking telemetry helper).
extern "C" int rdep_get_dropped_bf16(cudaStream_t stream) {
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) return 0;
        // Async telemetry read: return last completed sample without stalling the caller stream.
        return poll_device_int_async(nvshmem::g_nvshmem.dropped, stream);
    }
#endif
    if (!g_bf16.initialized) return 0;
    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off, &total_size);
    // Read local dropped counter only (tokens dropped trying to write to OUR buffer)
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    const int* dropped_dev = reinterpret_cast<const int*>(local_buf + dropped_off);
    // Async telemetry read: return last completed sample without stalling the caller stream.
    return poll_device_int_async(dropped_dev, stream);
}

// Read dropped tokens from local receive buffer (blockscaled path).
// Returns the last completed async sample (non-blocking telemetry helper).
extern "C" int rdep_get_dropped_blockscaled(cudaStream_t stream) {
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) return 0;
        // Async telemetry read: return last completed sample without stalling the caller stream.
        return poll_device_int_async(nvshmem::g_nvshmem.dropped, stream);
    }
#endif
    if (!g_block.initialized) return 0;
    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off, &total_size);
    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    const int* dropped_dev = reinterpret_cast<const int*>(local_buf + dropped_off);
    // Async telemetry read: return last completed sample without stalling the caller stream.
    return poll_device_int_async(dropped_dev, stream);
}

// ============================================================================
// IPC Dispatch Kernel - Direct P2P writes via IPC pointers
// ============================================================================

// Device pointers to all ranks' buffers (DeepEP pattern: kernel-accessible arrays)
__device__ void* d_buffer_ptrs_bf16[MAX_RANKS];
__device__ int*  d_barrier_signal_ptrs_bf16[MAX_RANKS];
__device__ void* d_buffer_ptrs_block[MAX_RANKS];
__device__ int*  d_barrier_signal_ptrs_block[MAX_RANKS];
__device__ int   d_my_rank_bf16;
__device__ int   d_my_rank_block;
__device__ int   d_world_bf16;
__device__ int   d_world_block;

// One-CTA, system-scope cross-GPU barriers (IPC mode).
// Declared here for use in forward dispatch/return; defined below with other IPC helpers.
__global__ void k_ipc_barrier_phase_bf16(int phase);
__global__ void k_ipc_barrier_phase_block(int phase);

__host__ __forceinline__ int ipc_barrier_threads_cached(int world) {
    thread_local int cached_world = -1;
    thread_local int cached_threads = 32;
    if (cached_world == world) {
        return cached_threads;
    }
    int threads = 32;
    while (threads < world && threads < 256) {
        threads <<= 1;
    }
    cached_world = world;
    cached_threads = threads;
    return threads;
}

__host__ __forceinline__ void enforce_single_ipc_stream(
    std::atomic<uintptr_t>& stream_slot,
    cudaStream_t stream,
    const char* tag) {
    const uintptr_t key = reinterpret_cast<uintptr_t>(stream) + 1;
    const uintptr_t seen = stream_slot.load(std::memory_order_relaxed);
    if (seen == key) {
        return;
    }
    uintptr_t expected = 0;
    if (stream_slot.compare_exchange_strong(expected, key, std::memory_order_relaxed)) {
        return;
    }
    if (expected != key) {
        fprintf(stderr,
                "RDEP FATAL: %s called on multiple CUDA streams. "
                "Use a single stream for IPC dispatch/return barriers.\n",
                tag);
        abort();
    }
}

__host__ __forceinline__ void ipc_barrier_bf16_site(cudaStream_t stream, const char* site) {
    if (g_mode != MODE_IPC) return;
    if (g_bf16.world <= 1) return;
    if (!g_ipc_handles_opened) {
        fprintf(stderr, "RDEP FATAL: ipc_barrier_bf16 called before IPC handles are opened\n");
        abort();
    }
    enforce_single_ipc_stream(g_ipc_stream_bf16, stream, "ipc_barrier_bf16");
    const int phase = g_ipc_phase_bf16.fetch_add(1, std::memory_order_relaxed) + 1;
    rdep_trace_barrier_launch("bf16", site, g_bf16.rank, g_bf16.world, phase);
    k_ipc_barrier_phase_bf16<<<1, ipc_barrier_threads_cached(g_bf16.world), 0, stream>>>(phase);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: ipc_barrier_bf16 launch failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    rdep_watch_barrier_completion(stream, "bf16", site, g_bf16.rank, g_bf16.world, phase);
}

__host__ __forceinline__ void ipc_barrier_bf16(cudaStream_t stream) {
    ipc_barrier_bf16_site(stream, nullptr);
}

__host__ __forceinline__ void ipc_barrier_block_site(cudaStream_t stream, const char* site) {
    if (g_mode != MODE_IPC) return;
    if (g_block.world <= 1) return;
    if (!g_ipc_handles_opened) {
        fprintf(stderr, "RDEP FATAL: ipc_barrier_block called before IPC handles are opened\n");
        abort();
    }
    enforce_single_ipc_stream(g_ipc_stream_block, stream, "ipc_barrier_block");
    const int phase = g_ipc_phase_block.fetch_add(1, std::memory_order_relaxed) + 1;
    rdep_trace_barrier_launch("block", site, g_block.rank, g_block.world, phase);
    k_ipc_barrier_phase_block<<<1, ipc_barrier_threads_cached(g_block.world), 0, stream>>>(phase);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: ipc_barrier_block launch failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    rdep_watch_barrier_completion(stream, "block", site, g_block.rank, g_block.world, phase);
}

__host__ __forceinline__ void ipc_barrier_block(cudaStream_t stream) {
    ipc_barrier_block_site(stream, nullptr);
}

__host__ __forceinline__ void maybe_zero_tokslot_buffers(
    char* local_buf,
    size_t tok_y_off,
    size_t tok_gate_off,
    int tok_slots,
    int Ha,
    bool zero_tok_y,
    bool zero_tok_gate,
    cudaStream_t stream) {
    if (tok_slots <= 0) return;
    if (zero_tok_y) {
        cudaMemsetAsync(local_buf + tok_y_off, 0, static_cast<size_t>(tok_slots) * static_cast<size_t>(Ha) * sizeof(uint16_t), stream);
    }
    if (zero_tok_gate) {
        cudaMemsetAsync(local_buf + tok_gate_off, 0, static_cast<size_t>(tok_slots) * sizeof(float), stream);
    }
}

// Copy buffer and barrier signal pointers to device
extern "C" void rdep_sync_buffer_ptrs_bf16() {
    if (g_bf16.world < 1 || g_bf16.world > MAX_RANKS) {
        fprintf(stderr, "RDEP FATAL: invalid g_bf16.world=%d (MAX_RANKS=%d)\n", g_bf16.world, MAX_RANKS);
        abort();
    }
    for (int r = 0; r < g_bf16.world; ++r) {
        if (g_bf16.buffer_ptrs[r] == nullptr || g_bf16.barrier_signal_ptrs[r] == nullptr) {
            fprintf(stderr,
                    "RDEP FATAL: cannot sync BF16 IPC pointers; rank %d has buf=%p barrier=%p\n",
                    r, g_bf16.buffer_ptrs[r], g_bf16.barrier_signal_ptrs[r]);
            abort();
        }
    }
    cudaError_t err = cudaMemcpyToSymbol(
        d_buffer_ptrs_bf16, g_bf16.buffer_ptrs,
        g_bf16.world * sizeof(void*), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_buffer_ptrs_bf16) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemcpyToSymbol(
        d_barrier_signal_ptrs_bf16, g_bf16.barrier_signal_ptrs,
        g_bf16.world * sizeof(int*), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_barrier_signal_ptrs_bf16) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemcpyToSymbol(d_my_rank_bf16, &g_bf16.rank, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_my_rank_bf16) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemcpyToSymbol(d_world_bf16, &g_bf16.world, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_world_bf16) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
}

extern "C" void rdep_sync_buffer_ptrs_blockscaled() {
    if (g_block.world < 1 || g_block.world > MAX_RANKS) {
        fprintf(stderr, "RDEP FATAL: invalid g_block.world=%d (MAX_RANKS=%d)\n", g_block.world, MAX_RANKS);
        abort();
    }
    for (int r = 0; r < g_block.world; ++r) {
        if (g_block.buffer_ptrs[r] == nullptr || g_block.barrier_signal_ptrs[r] == nullptr) {
            fprintf(stderr,
                    "RDEP FATAL: cannot sync blockscaled IPC pointers; rank %d has buf=%p barrier=%p\n",
                    r, g_block.buffer_ptrs[r], g_block.barrier_signal_ptrs[r]);
            abort();
        }
    }
    cudaError_t err = cudaMemcpyToSymbol(
        d_buffer_ptrs_block, g_block.buffer_ptrs,
        g_block.world * sizeof(void*), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_buffer_ptrs_block) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemcpyToSymbol(
        d_barrier_signal_ptrs_block, g_block.barrier_signal_ptrs,
        g_block.world * sizeof(int*), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_barrier_signal_ptrs_block) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemcpyToSymbol(d_my_rank_block, &g_block.rank, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_my_rank_block) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
    err = cudaMemcpyToSymbol(d_world_block, &g_block.world, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: cudaMemcpyToSymbol(d_world_block) failed: %s\n", cudaGetErrorString(err));
        abort();
    }
}

// ============================================================================
// Direct IPC Write - Copy data directly to peer's IPC buffer
// Used by rdep_adapter.py for low-level direct buffer access
// ============================================================================
extern "C" void rdep_direct_ipc_write(
    void* src_ptr,      // Source data pointer on local GPU
    int target_rank,    // Destination rank
    size_t offset,      // Offset in peer buffer (in bytes)
    size_t n_bytes,     // Number of bytes to copy
    cudaStream_t stream // CUDA stream for async copy
) {
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP ERROR: direct_ipc_write called before initialization\n");
        return;
    }

    if (target_rank < 0 || target_rank >= g_bf16.world) {
        fprintf(stderr, "RDEP ERROR: invalid target_rank %d (world=%d)\n",
                target_rank, g_bf16.world);
        return;
    }

    // Get destination buffer pointer (already mapped via IPC)
    void* dest_buf = g_bf16.buffer_ptrs[target_rank];
    if (!dest_buf) {
        fprintf(stderr, "RDEP ERROR: peer buffer for rank %d not initialized\n", target_rank);
        return;
    }

    // Calculate destination address
    char* dest_ptr = static_cast<char*>(dest_buf) + offset;

    // Perform async copy to peer buffer
    // Since IPC handles were opened with cudaIpcMemLazyEnablePeerAccess,
    // this uses P2P transfer when available, otherwise falls back to host staging
    cudaMemcpyAsync(dest_ptr, src_ptr, n_bytes, cudaMemcpyDeviceToDevice, stream ? stream : 0);
}

// ============================================================================
// BF16 Dispatch Kernel
// Each warp handles one (token, slot) pair
// ============================================================================
__global__ void k_dispatch_bf16(
    const __nv_bfloat16* __restrict__ x,  // [T, H] - NOT expanded
    const int* __restrict__ eids,          // [T, K] - expert IDs
    const float* __restrict__ gates,       // [T, K] - gate values
    int my_rank, int T, int H, int Ha, int K,
    int n_local, int capacity,
    size_t meta_off, size_t counter_off, size_t dropped_off)
{
    // Each warp processes one (tok, slot) pair
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const int n_local_mask = n_local - 1;
    const int world = d_world_bf16;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);
        int slot = i - tok * K;

        int eid = eids[i];
        if (eid < 0) continue;
        float gate = gates[i];
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        int local_eid = nmoe_expert_local_fast(eid, n_local, n_local_pow2, n_local_mask);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;
        if (local_eid < 0 || local_eid >= n_local) continue;
        bool is_remote = (dest != my_rank);

        // Get destination buffer pointer
        char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + counter_off);
        int* dropped = reinterpret_cast<int*>(dest_buf + dropped_off);

        // One warp, one slot - leader does atomic
        int slot_r;
        if (lane == 0)
            slot_r = atomicAdd(counter, 1);
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

        if (slot_r >= capacity) {
            if (lane == 0)
                atomicAdd(dropped, 1);
            continue;
        }

        // Write metadata
        if (lane == 0) {
            Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
            if (is_remote) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&m);
                st_relaxed_sys_v4_s32(meta_dst, meta_val);  // sys-scope for cross-GPU
            } else {
                meta_buf[slot_r] = m;
            }
        }

        // Write BF16 payload - each lane handles H/32 elements
        const __nv_bfloat16* row = x + (int64_t)tok * H;  // Read from original [T,H]
        uint16_t* dst = x_buf + (int64_t)slot_r * Ha;

        if (is_remote) {
            // Vectorized P2P writes - sys-scope for cross-GPU visibility
            // H is validated as a multiple of 8 at the dispatch API boundary.
            for (int h = lane * 8; h < H; h += 32 * 8) {
                int4* d = reinterpret_cast<int4*>(dst + h);
                int4 v = *reinterpret_cast<const int4*>(row + h);
                st_relaxed_sys_v4_s32(d, v);
            }
        } else {
            // Local copy - vectorized int4 (same pattern as remote, but regular stores)
            for (int h = lane * 8; h < H; h += 32 * 8) {
                int4* d = reinterpret_cast<int4*>(dst + h);
                int4 v = *reinterpret_cast<const int4*>(row + h);
                *d = v;  // Regular store (no NA hint needed for local)
            }
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

// ============================================================================
// 2-Phase Dispatch: Count tokens per destination (Phase 1)
// No remote atomics - just counts locally, then writes counts to each dest
// ============================================================================
__global__ void k_count_dispatch_bf16(
    const int* __restrict__ eids,          // [T, K] - expert IDs
    int M, int n_local,
    int* __restrict__ send_counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int world = d_world_bf16;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;

    // Shared memory for per-destination counts
    extern __shared__ int shared_counts[];

    // Initialize shared counts
    if (threadIdx.x < world) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int i = tid; i < M; i += gridDim.x * blockDim.x) {
        int dest = -1;
        bool valid = false;
        int eid = eids[i];
        if (eid >= 0) {
            dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
            valid = (static_cast<unsigned>(dest) < static_cast<unsigned>(world));
        }
        unsigned valid_mask = __ballot_sync(0xFFFFFFFFu, valid);
        if (valid) {
            const unsigned group = __match_any_sync(valid_mask, dest);
            const int leader = __ffs(static_cast<int>(group)) - 1;
            if (lane == leader) {
                atomicAdd(&shared_counts[dest], __popc(group));  // Local atomic only.
            }
        }
    }
    __syncthreads();

    // Block leader writes to per-rank local send-count staging.
    if (threadIdx.x < world) {
        int c = shared_counts[threadIdx.x];
        if (c != 0) {
            atomicAdd(&send_counts[threadIdx.x], c);
        }
    }
}

// Write send counts to each destination's recv_counts area (after k_count_dispatch completes)
__global__ void k_write_counts_to_dests_bf16(const int* __restrict__ send_counts, size_t recv_counts_off) {
    int my_rank = d_my_rank_bf16;
    int world = d_world_bf16;
    int dest = threadIdx.x;

    if (dest >= world) return;

    // Read my staged send count for this destination.
    int count = send_counts[dest];

    // Write to dest's recv_counts[my_rank]
    char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
    int* dest_recv_counts = reinterpret_cast<int*>(dest_buf + recv_counts_off);

    if (dest != my_rank) {
        // P2P write - sys-scope for cross-GPU visibility
        st_relaxed_sys_s32(&dest_recv_counts[my_rank], count);
    } else {
        dest_recv_counts[my_rank] = count;
    }

    // Fence to ensure writes are sys-visible before kernel completes
    fence_acq_rel_sys();
}

// Compute prefix sums from recv_counts and write offsets back to sources
__global__ void k_compute_and_write_offsets_bf16(
    size_t recv_counts_off,
    size_t recv_offsets_off,
    size_t counter_off) {
    int my_rank = d_my_rank_bf16;
    int world = d_world_bf16;
    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    char* my_buf = static_cast<char*>(d_buffer_ptrs_bf16[my_rank]);
    int* recv_counts = reinterpret_cast<int*>(my_buf + recv_counts_off);
    constexpr unsigned mask = 0xFFFFFFFFu;
    int c = (lane < world) ? recv_counts[lane] : 0;
    int scan = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int n = __shfl_up_sync(mask, scan, off);
        if (lane >= off) scan += n;
    }
    int src_base = scan - c;
    if (lane < world) {
        char* src_buf = static_cast<char*>(d_buffer_ptrs_bf16[lane]);
        int* src_recv_offsets = reinterpret_cast<int*>(src_buf + recv_offsets_off);
        if (lane != my_rank) {
            // P2P write - sys-scope for cross-GPU visibility.
            st_relaxed_sys_s32(&src_recv_offsets[my_rank], src_base);
        } else {
            src_recv_offsets[my_rank] = src_base;
        }
    }
    int offset = (world > 0) ? __shfl_sync(mask, scan, world - 1) : 0;

    // Persist total received rows for host readback (single int D2H).
    if (lane == 0) {
        int* recv_total = reinterpret_cast<int*>(my_buf + counter_off);
        *recv_total = offset;
    }

    // Fence to ensure writes are sys-visible before kernel completes
    fence_acq_rel_sys();
}

// 2-phase count/exchange kernels for blockscaled path.
__global__ void k_count_dispatch_block(
    const int* __restrict__ eids,
    int M, int n_local,
    int* __restrict__ send_counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int world = d_world_block;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;

    extern __shared__ int shared_counts[];
    if (threadIdx.x < world) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int i = tid; i < M; i += gridDim.x * blockDim.x) {
        int dest = -1;
        bool valid = false;
        int eid = eids[i];
        if (eid >= 0) {
            dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
            valid = (static_cast<unsigned>(dest) < static_cast<unsigned>(world));
        }
        unsigned valid_mask = __ballot_sync(0xFFFFFFFFu, valid);
        if (valid) {
            const unsigned group = __match_any_sync(valid_mask, dest);
            const int leader = __ffs(static_cast<int>(group)) - 1;
            if (lane == leader) {
                atomicAdd(&shared_counts[dest], __popc(group));
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < world) {
        int c = shared_counts[threadIdx.x];
        if (c != 0) {
            atomicAdd(&send_counts[threadIdx.x], c);
        }
    }
}

__global__ void k_write_counts_to_dests_block(const int* __restrict__ send_counts, size_t recv_counts_off) {
    int my_rank = d_my_rank_block;
    int world = d_world_block;
    int dest = threadIdx.x;
    if (dest >= world) return;

    int count = send_counts[dest];

    char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[dest]);
    int* dest_recv_counts = reinterpret_cast<int*>(dest_buf + recv_counts_off);

    if (dest != my_rank) {
        st_relaxed_sys_s32(&dest_recv_counts[my_rank], count);
    } else {
        dest_recv_counts[my_rank] = count;
    }
    fence_acq_rel_sys();
}

__global__ void k_compute_and_write_offsets_block(
    size_t recv_counts_off,
    size_t recv_offsets_off,
    size_t counter_off) {
    int my_rank = d_my_rank_block;
    int world = d_world_block;
    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    char* my_buf = static_cast<char*>(d_buffer_ptrs_block[my_rank]);
    int* recv_counts = reinterpret_cast<int*>(my_buf + recv_counts_off);
    constexpr unsigned mask = 0xFFFFFFFFu;
    int c = (lane < world) ? recv_counts[lane] : 0;
    int scan = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int n = __shfl_up_sync(mask, scan, off);
        if (lane >= off) scan += n;
    }
    int src_base = scan - c;
    if (lane < world) {
        char* src_buf = static_cast<char*>(d_buffer_ptrs_block[lane]);
        int* src_recv_offsets = reinterpret_cast<int*>(src_buf + recv_offsets_off);
        if (lane != my_rank) {
            st_relaxed_sys_s32(&src_recv_offsets[my_rank], src_base);
        } else {
            src_recv_offsets[my_rank] = src_base;
        }
    }
    int offset = (world > 0) ? __shfl_sync(mask, scan, world - 1) : 0;

    if (lane == 0) {
        int* recv_total = reinterpret_cast<int*>(my_buf + counter_off);
        *recv_total = offset;
    }
    fence_acq_rel_sys();
}

// 2-Phase Dispatch: Deterministic write (Phase 2)
// Uses pre-computed offsets, LOCAL atomics only for ordering within a rank's batch
__global__ void k_dispatch_2phase_bf16(
    const __nv_bfloat16* __restrict__ x,  // [T, H]
    const int* __restrict__ eids,          // [T, K]
    const float* __restrict__ gates,       // [T, K]
    int* __restrict__ local_counters,      // [world] - local atomic counters (device memory)
    int T, int H, int Ha, int K,
    int n_local, int capacity,
    size_t meta_off, size_t dropped_off, size_t recv_offsets_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    int M = T * K;
    int my_rank = d_my_rank_bf16;
    int world = d_world_bf16;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const int n_local_mask = n_local - 1;

    // Read my starting offsets from each destination (written by k_compute_and_write_offsets)
    char* my_buf = static_cast<char*>(d_buffer_ptrs_bf16[my_rank]);
    int* my_recv_offsets = reinterpret_cast<int*>(my_buf + recv_offsets_off);
    __shared__ int s_recv_offsets[MAX_RANKS];
    for (int r = threadIdx.x; r < world; r += blockDim.x) {
        s_recv_offsets[r] = my_recv_offsets[r];
    }
    __syncthreads();

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);
        int slot = i - tok * K;

        int eid = eids[i];
        if (eid < 0) continue;
        float gate = gates[i];
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        int local_eid = nmoe_expert_local_fast(eid, n_local, n_local_pow2, n_local_mask);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;
        if (local_eid < 0 || local_eid >= n_local) continue;
        bool is_remote = (dest != my_rank);

        // Get local offset within this rank's batch to dest (LOCAL atomic only)
        int local_idx;
        if (lane == 0) {
            local_idx = atomicAdd(&local_counters[dest], 1);
        }
        local_idx = __shfl_sync(0xFFFFFFFF, local_idx, 0);

        // Read where this rank's data starts at dest.
        int base_offset = 0;
        if (lane == 0) {
            base_offset = s_recv_offsets[dest];
        }
        base_offset = __shfl_sync(0xFFFFFFFFu, base_offset, 0);
        int slot_r = base_offset + local_idx;

        if (slot_r >= capacity) {
            if (lane == 0) {
                char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
                int* dropped = reinterpret_cast<int*>(dest_buf + dropped_off);
                atomicAdd(dropped, 1);
            }
            continue;  // Overflow protection
        }

        // Get destination buffer
        char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);

        // Write metadata
        if (lane == 0) {
            Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
            if (is_remote) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&m);
                st_relaxed_sys_v4_s32(meta_dst, meta_val);  // sys-scope for cross-GPU visibility
            } else {
                meta_buf[slot_r] = m;
            }
        }

        // Write BF16 payload
        const __nv_bfloat16* row = x + (int64_t)tok * H;
        uint16_t* dst = x_buf + (int64_t)slot_r * Ha;

        if (is_remote) {
            // H is validated as a multiple of 8 at the dispatch API boundary.
            for (int h = lane * 8; h < H; h += 32 * 8) {
                int4* d = reinterpret_cast<int4*>(dst + h);
                int4 v = *reinterpret_cast<const int4*>(row + h);
                st_relaxed_sys_v4_s32(d, v);  // sys-scope for cross-GPU visibility
            }
        } else {
            for (int h = lane * 8; h < H; h += 32 * 8) {
                int4* d = reinterpret_cast<int4*>(dst + h);
                int4 v = *reinterpret_cast<const int4*>(row + h);
                *d = v;
            }
        }
    }

    // Ensure all sys-scope writes are visible before kernel completes.
    // The barrier kernel's fence cannot order writes from this kernel since
    // they are in different threads' program order. This fence must happen
    // HERE, in the kernel that did the writes.
    fence_acq_rel_sys();
}

// 2-Phase Dispatch (blockscaled): deterministic destination offsets, no remote atomics.
template <bool kFP8>
__global__ void k_dispatch_2phase_blockscaled(
    const __nv_bfloat16* __restrict__ x,  // [T, H]
    const int* __restrict__ eids,         // [T, K]
    const float* __restrict__ gates,      // [T, K]
    int* __restrict__ local_counters,     // [world]
    int T, int H, int Hp, int Hsf, int K,
    int n_local, int capacity,
    size_t sfa_off, size_t meta_off, size_t recv_offsets_off, size_t dropped_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    int my_rank = d_my_rank_block;
    int world = d_world_block;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const int n_local_mask = n_local - 1;

    constexpr float dtype_max = kFP8 ? FP8_MAX : FP4_MAX;

    char* my_buf = static_cast<char*>(d_buffer_ptrs_block[my_rank]);
    int* my_recv_offsets = reinterpret_cast<int*>(my_buf + recv_offsets_off);
    __shared__ int s_recv_offsets[MAX_RANKS];
    for (int r = threadIdx.x; r < world; r += blockDim.x) {
        s_recv_offsets[r] = my_recv_offsets[r];
    }
    __syncthreads();

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);
        int slot = i - tok * K;

        int eid = eids[i];
        if (eid < 0) continue;
        float gate = gates[i];
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        int local_eid = nmoe_expert_local_fast(eid, n_local, n_local_pow2, n_local_mask);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;
        if (local_eid < 0 || local_eid >= n_local) continue;
        bool is_remote = (dest != my_rank);

        int local_idx;
        if (lane == 0) {
            local_idx = atomicAdd(&local_counters[dest], 1);
        }
        local_idx = __shfl_sync(0xFFFFFFFF, local_idx, 0);

        int base_offset = 0;
        if (lane == 0) {
            base_offset = s_recv_offsets[dest];
        }
        base_offset = __shfl_sync(0xFFFFFFFFu, base_offset, 0);
        int slot_r = base_offset + local_idx;
        if (slot_r >= capacity) {
            if (lane == 0) {
                char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[dest]);
                int* dropped = reinterpret_cast<int*>(dest_buf + dropped_off);
                atomicAdd(dropped, 1);
            }
            continue;
        }

        char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[dest]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(dest_buf + sfa_off);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);

        if (lane == 0) {
            Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
            if (is_remote) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = m;
            }
        }

        const __nv_bfloat16* row = x + static_cast<int64_t>(tok) * H;
        uint16_t* dst_pack = x_buf + static_cast<int64_t>(slot_r) * Hp;
        uint8_t* dst_sfa = sfa_buf + static_cast<int64_t>(slot_r) * Hsf;

        for (int blk = 0; blk < Hsf; blk++) {
            int h0 = blk * SF_VEC;
            int h_end = min(h0 + SF_VEC, H);
            int blk_size = h_end - h0;

            float val = 0.0f;
            if (lane < blk_size) {
                val = __bfloat162float(row[h0 + lane]);
            }

            float local_amax = fabsf(val);
            float blk_amax = warp_reduce_max(local_amax);
            float scale = blk_amax / dtype_max;
            if (!(scale > 0.0f)) scale = 1.0f;
            uint8_t scale_byte = e8m0_encode(scale);
            float s = e8m0_decode(scale_byte);
            float inv_scale = (s > 0.0f) ? (1.0f / s) : 1.0f;

            if (lane == 0) {
                if (is_remote) {
                    st_na_relaxed_gpu_b8(dst_sfa + blk, scale_byte);
                } else {
                    dst_sfa[blk] = scale_byte;
                }
            }

            float qf = val * inv_scale;
            if constexpr (kFP8) {
                uint8_t q8 = to_fp8(qf);
                uint8_t q8_neighbor = __shfl_xor_sync(0xFFFFFFFF, q8, 1);
                if ((lane & 1) == 0 && lane < blk_size) {
                    uint16_t packed = static_cast<uint16_t>(q8) | (static_cast<uint16_t>(q8_neighbor) << 8);
                    int pack_idx = blk * (SF_VEC / 2) + (lane / 2);
                    if (is_remote) {
                        st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                    } else {
                        dst_pack[pack_idx] = packed;
                    }
                }
            } else {
                float qf0 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 0);
                float qf1 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 1);
                float qf2 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 2);
                float qf3 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 3);
                if ((lane & 3) == 0 && lane < blk_size) {
                    uint16_t packed = to_fp4x4(qf0, qf1, qf2, qf3);
                    int pack_idx = blk * (SF_VEC / 4) + (lane / 4);
                    if (is_remote) {
                        st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                    } else {
                        dst_pack[pack_idx] = packed;
                    }
                }
            }
        }
    }

    // Match BF16 2-phase ordering semantics: ensure remote writes from this
    // kernel are sys-visible before the subsequent barrier/reduction stages.
    fence_acq_rel_sys();
}

// ============================================================================
// Blockscaled Dispatch Kernel
// Each warp handles one (token, slot) pair
// Quantization happens in registers, writes directly to remote buffer
// ============================================================================
template <bool kFP8>
__global__ void k_dispatch_blockscaled(
    const __nv_bfloat16* __restrict__ x,  // [T, H] - NOT expanded
    const int* __restrict__ eids,          // [T, K] - expert IDs
    const float* __restrict__ gates,       // [T, K] - gate values
    int my_rank, int T, int H, int Hp, int Hsf, int K,
    int n_local, int capacity,
    size_t sfa_off, size_t meta_off, size_t counter_off, size_t dropped_off)
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
    const int world = d_world_block;

    constexpr float dtype_max = kFP8 ? FP8_MAX : FP4_MAX;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);
        int slot = i - tok * K;

        int eid = eids[i];
        if (eid < 0) continue;
        float gate = gates[i];
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        int local_eid = nmoe_expert_local_fast(eid, n_local, n_local_pow2, n_local_mask);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;
        if (local_eid < 0 || local_eid >= n_local) continue;
        bool is_remote = (dest != my_rank);

        // Get destination buffer pointers
        char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[dest]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(dest_buf);
        uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(dest_buf + sfa_off);
        Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
        int* counter = reinterpret_cast<int*>(dest_buf + counter_off);
        int* dropped = reinterpret_cast<int*>(dest_buf + dropped_off);

        // One warp, one slot - leader does atomic
        int slot_r;
        if (lane == 0)
            slot_r = atomicAdd(counter, 1);
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

        if (slot_r >= capacity) {
            if (lane == 0)
                atomicAdd(dropped, 1);
            continue;
        }

        // Write metadata
        if (lane == 0) {
            Meta m{encode_rid(my_rank, tok, slot, T, K), local_eid, gate};
            if (is_remote) {
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&m);
                st_na_v4_s32(meta_dst, meta_val);
            } else {
                meta_buf[slot_r] = m;
            }
        }

        // Pointers to destination row
        const __nv_bfloat16* row = x + (int64_t)tok * H;
        uint16_t* dst_pack = x_buf + (int64_t)slot_r * Hp;
        uint8_t* dst_sfa = sfa_buf + (int64_t)slot_r * Hsf;

        // Process each 32-element block
        // Lanes 0-31 each handle one element within the block
        for (int blk = 0; blk < Hsf; blk++) {
            int h0 = blk * SF_VEC;  // SF_VEC = 32
            int h_end = min(h0 + SF_VEC, H);
            int blk_size = h_end - h0;

            // Each lane loads its element (if within bounds)
            float val = 0.0f;
            if (lane < blk_size) {
                val = __bfloat162float(row[h0 + lane]);
            }

            // Warp reduction for max absolute value
            float local_amax = fabsf(val);
            float blk_amax = warp_reduce_max(local_amax);

            // Compute and broadcast scale
            float scale = blk_amax / dtype_max;
            if (!(scale > 0.0f)) scale = 1.0f;
            uint8_t scale_byte = e8m0_encode(scale);
            float s = e8m0_decode(scale_byte);
            float inv_scale = (s > 0.0f) ? (1.0f / s) : 1.0f;

            // Write scale factor (lane 0 only)
            if (lane == 0) {
                if (is_remote) {
                    st_na_relaxed_gpu_b8(dst_sfa + blk, scale_byte);
                } else {
                    dst_sfa[blk] = scale_byte;
                }
            }

            // Quantize value in register
            float qf = val * inv_scale;

            if constexpr (kFP8) {
                // FP8 E4M3: pack 2 values into uint16
                // Lanes 0,1 -> pack[0], lanes 2,3 -> pack[1], etc.
                uint8_t q8 = to_fp8(qf);

                // Get neighbor's quantized value via shuffle
                uint8_t q8_neighbor = __shfl_xor_sync(0xFFFFFFFF, q8, 1);

                // Even lanes pack [self, neighbor], odd lanes idle
                if ((lane & 1) == 0 && lane < blk_size) {
                    uint16_t packed = (uint16_t)q8 | ((uint16_t)q8_neighbor << 8);
                    int pack_idx = blk * (SF_VEC / 2) + (lane / 2);
                    if (is_remote) {
                        st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                    } else {
                        dst_pack[pack_idx] = packed;
                    }
                }
            } else {
                // NVFP4 E2M1: pack 4 values into uint16
                // Lanes 0,1,2,3 -> pack[0], lanes 4,5,6,7 -> pack[1], etc.
                // First get all 4 quantized values via shuffles
                float qf0 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 0);
                float qf1 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 1);
                float qf2 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 2);
                float qf3 = __shfl_sync(0xFFFFFFFF, qf, (lane / 4) * 4 + 3);

                // Only lane 0 of each group of 4 writes
                if ((lane & 3) == 0 && lane < blk_size) {
                    uint16_t packed = to_fp4x4(qf0, qf1, qf2, qf3);
                    int pack_idx = blk * (SF_VEC / 4) + (lane / 4);
                    if (is_remote) {
                        st_na_relaxed_gpu_b16(dst_pack + pack_idx, packed);
                    } else {
                        dst_pack[pack_idx] = packed;
                    }
                }
            }
        }
    }

    // Ensure remote metadata/payload writes are sys-visible before any
    // downstream barrier or consumer kernel observes this dispatch.
    fence_acq_rel_sys();
}

// ============================================================================
// Sort + Gather Kernels
// ============================================================================

__global__ void k_extract_local_eid(
    const Meta* __restrict__ meta,
    int* __restrict__ local_eid,
    int* __restrict__ order,
    int M)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M;
         i += blockDim.x * gridDim.x) {
        local_eid[i] = meta[i].local_eid;
        order[i] = i;
    }
}

__global__ void k_compute_offsets(
    const int* __restrict__ sorted_eid,
    int* __restrict__ offsets,
    int M, int n_local)
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

__global__ void k_compute_padded_prefix(
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

__global__ void k_compute_offsets_and_padded_prefix(
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

__global__ void k_fill_dest_from_sorted_eid(
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

__global__ void k_init_single_expert_layout(
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

// Zero only padded rows (gaps introduced by per-expert alignment), not the full
// [M_pad, H] buffer. This avoids large full-buffer memsets on hot dispatch paths.
__global__ void k_zero_bf16_padding_rows(
    const int* __restrict__ offsets,    // [n_local + 1]
    const int* __restrict__ offs_pad,   // [n_local]
    __nv_bfloat16* __restrict__ out,    // [M_pad, H]
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

// Fill only padded rows for blockscaled gather staging:
//   - q rows with 0
//   - optional sf rows with E8M0 byte 127 (scale=1.0) when sf_out != nullptr
__global__ void k_fill_blockscaled_padding_rows(
    const int* __restrict__ offsets,    // [n_local + 1]
    const int* __restrict__ offs_pad,   // [n_local]
    uint16_t* __restrict__ q_out,       // [M_pad, Hp]
    uint8_t* __restrict__ sf_out,       // [M_pad, Hsf]
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
__global__ void k_fill_blockscaled_padding_sf_swizzled(
    const int* __restrict__ offsets,    // [n_local + 1]
    const int* __restrict__ offs_pad,   // [n_local]
    uint8_t* __restrict__ sf_out_mma,   // [M_pad, Hsf] swizzled layout
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

__host__ __forceinline__ void zero_bf16_padding_rows_async(
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
    k_zero_bf16_padding_rows<<<blocks, threads, 0, stream>>>(
        offsets,
        offs_pad,
        static_cast<__nv_bfloat16*>(out),
        n_local,
        H);
}

__host__ __forceinline__ void fill_blockscaled_padding_rows_async(
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
    k_fill_blockscaled_padding_rows<<<blocks, threads, 0, stream>>>(
        offsets,
        offs_pad,
        static_cast<uint16_t*>(q_out),
        static_cast<uint8_t*>(sf_out),
        n_local,
        Hp,
        Hsf);
}

__host__ __forceinline__ void fill_blockscaled_padding_sf_swizzled_async(
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
    k_fill_blockscaled_padding_sf_swizzled<<<blocks, threads, 0, stream>>>(
        offsets,
        offs_pad,
        static_cast<uint8_t*>(sf_out),
        n_local,
        M_pad,
        Hsf);
}

__global__ void k_gather_bf16(
    const uint16_t* __restrict__ x_recv,
    const int* __restrict__ order,
    const int* __restrict__ dest,
    __nv_bfloat16* __restrict__ Xe_out,
    int M, int H, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    // Use int4 vectorized copy for better bandwidth (H/8 int4s = H/2 BF16s per int4)
    int hidden_int4 = H / 8;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        const int4* src = reinterpret_cast<const int4*>(x_recv + (int64_t)orig_i * Ha);
        int out_i = (dest != nullptr) ? dest[sorted_i] : sorted_i;
        int4* dst = reinterpret_cast<int4*>(Xe_out + (int64_t)out_i * H);

        // Use UNROLLED_WARP_COPY for efficient vectorized copy
        UNROLLED_WARP_COPY(4, lane, hidden_int4, dst, src, ld_nc_v4_s32, st_na_global);

        // Handle remaining elements if H not divisible by 8
        int remaining_start = hidden_int4 * 8;
        const __nv_bfloat16* src_bf16 = reinterpret_cast<const __nv_bfloat16*>(x_recv + (int64_t)orig_i * Ha);
        __nv_bfloat16* dst_bf16 = Xe_out + (int64_t)out_i * H;
        for (int h = remaining_start + lane; h < H; h += 32)
            dst_bf16[h] = src_bf16[h];
    }
}

__global__ void k_gather_blockscaled(
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
        for (int hp = lane; hp < Hp; hp += 32)
            dst[hp] = src[hp];

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

__global__ void k_gather_meta_sorted(
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

__global__ void k_gather_from_pad_bf16(
    const __nv_bfloat16* __restrict__ in_pad,
    const int* __restrict__ dest,
    __nv_bfloat16* __restrict__ out_sorted,
    int M, int H)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    // Use int4 vectorized copy for better bandwidth (H/8 int4s = H/2 BF16s per int4)
    int hidden_int4 = H / 8;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int pad_i = dest[sorted_i];
        const int4* src = reinterpret_cast<const int4*>(in_pad + (int64_t)pad_i * H);
        int4* dst = reinterpret_cast<int4*>(out_sorted + (int64_t)sorted_i * H);

        UNROLLED_WARP_COPY(4, lane, hidden_int4, dst, src, ld_nc_v4_s32, st_na_global);

        // Handle remaining elements if H not divisible by 8
        int remaining_start = hidden_int4 * 8;
        const __nv_bfloat16* src_bf16 = in_pad + (int64_t)pad_i * H;
        __nv_bfloat16* dst_bf16 = out_sorted + (int64_t)sorted_i * H;
        for (int h = remaining_start + lane; h < H; h += 32)
            dst_bf16[h] = src_bf16[h];
    }
}

__global__ void k_scatter_sorted_to_pad_bf16(
    const __nv_bfloat16* __restrict__ in_sorted,
    const int* __restrict__ dest,
    __nv_bfloat16* __restrict__ out_pad,
    int M, int H)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    int hidden_int4 = H / 8;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        int pad_i = dest[sorted_i];
        const int4* src = reinterpret_cast<const int4*>(in_sorted + (int64_t)sorted_i * H);
        int4* dst = reinterpret_cast<int4*>(out_pad + (int64_t)pad_i * H);

        UNROLLED_WARP_COPY(4, lane, hidden_int4, dst, src, ld_nc_v4_s32, st_na_global);

        int remaining_start = hidden_int4 * 8;
        const __nv_bfloat16* src_bf16 = in_sorted + (int64_t)sorted_i * H;
        __nv_bfloat16* dst_bf16 = out_pad + (int64_t)pad_i * H;
        for (int h = remaining_start + lane; h < H; h += 32)
            dst_bf16[h] = src_bf16[h];
    }
}

// ============================================================================
// Host API: BF16 Dispatch
// ============================================================================

extern "C" int rdep_dispatch_meta_bf16(
    const void* x,           // [T, H] - NOT expanded
    const int* eids,         // [T, K] - expert IDs (NOT flattened)
    const float* gates,      // [T, K] - gate values (NOT flattened)
    int T, int K,
    int align,               // Per-expert row padding (8 for BF16, 128 for blockscaled)
    int* offs_pad_out,       // [n_local] device int32
    int* M_pad_out,          // host int32 (pinned recommended). Used as a host scratch for M_recv.
    cudaStream_t stream)
{
    g_bf16.offs_pad_last = offs_pad_out;
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return -1;
        }
        if (nvshmem::g_nvshmem.profile != -1) {
            fprintf(stderr, "RDEP ERROR: rdep_dispatch_meta_bf16 requires BF16 NVSHMEM state (profile=-1)\n");
            return -2;
        }

        // Reuse the hybrid dispatch pipeline but skip Xe_out materialization.
        return nvshmem::dispatch_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(x),
            eids,
            gates,
            T, K,
            align,
            /*Xe_out=*/nullptr,
            offs_pad_out,
            /*dest_out=*/nullptr,
            /*row_id_out=*/nullptr,
            /*gate_out=*/nullptr,
            M_pad_out,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_meta_off,
            nvshmem::g_nvshmem.ipc_counter_off,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
    }
#endif

    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP ERROR: BF16 buffers not initialized\n");
        return -1;
    }

    if (g_bf16.H % 8 != 0) {
        fprintf(stderr, "RDEP ERROR: H=%d must be multiple of 8 for vectorized copies\n", g_bf16.H);
        return -2;
    }
    if (align <= 0 || (align % 8) != 0) {
        fprintf(stderr, "RDEP ERROR: align must be positive and multiple of 8, got align=%d\n", align);
        return -2;
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    size_t recv_counts_off, recv_offsets_off;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size,
                        &recv_counts_off, &recv_offsets_off);

    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    int capacity = static_cast<int>(g_bf16.capacity);
    int M = T * K;
    int M_recv = 0;

    if (k_use_2phase_dispatch && g_bf16.world > 1) {
        M_recv = dispatch_2phase_bf16(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K, M,
            meta_off, counter_off, dropped_off, recv_counts_off, recv_offsets_off,
            capacity, M_pad_out, stream);
    } else {
        cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
        cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);

        if (g_bf16.world > 1) {
            ipc_barrier_bf16(stream);
        }

        int warps_needed = M;
        int threads = 256;
        int warps_per_block = threads / 32;
        int blocks_by_work = std::max(1, (warps_needed + warps_per_block - 1) / warps_per_block);
        int blocks = cap_warp_stride_blocks(blocks_by_work);

        k_dispatch_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            g_bf16.rank, T, g_bf16.H, g_bf16.Ha, K,
            g_bf16.n_local, capacity,
            meta_off, counter_off, dropped_off);

	        if (g_bf16.world > 1) {
	            ipc_barrier_bf16(stream);
	        }
	        const int* local_counter = reinterpret_cast<const int*>(local_buf + counter_off);
	        if (M_pad_out == nullptr) {
            fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
            return -3;
        }
        bool recv_ok = false;
        M_recv = read_device_int_stream_sync(local_counter, stream, &recv_ok);
        if (!recv_ok) {
            return -3;
        }
        *M_pad_out = M_recv;
    }

    if (M_recv < 0) {
        return M_recv;
    }
    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_bf16.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP ERROR: BF16 dispatch overflow: M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        return -3;
    }

    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    int extract_threads = 256;
    int extract_blocks_by_work = std::max(1, (M_recv + extract_threads - 1) / extract_threads);
    int extract_blocks = cap_warp_stride_blocks(extract_blocks_by_work);
    if (g_bf16.n_local == 1) {
        const int M_pad = ((M_recv + align - 1) / align) * align;
        k_init_single_expert_layout<<<extract_blocks, extract_threads, 0, stream>>>(
            g_bf16.order, g_bf16.dest,
            g_bf16.offsets, offs_pad_out, g_bf16.M_pad_dev,
            M_recv, M_pad);
    } else {
        k_extract_local_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            meta_buf, g_bf16.local_eid, g_bf16.order, M_recv);

        if (g_bf16.n_local > 1 && M_recv > 1) {
            const int sort_end_bit = radix_sort_end_bit_for_range(g_bf16.n_local);
            cub::DeviceRadixSort::SortPairs(g_bf16.sort_temp, g_bf16.sort_temp_bytes,
                g_bf16.local_eid, g_bf16.local_eid, g_bf16.order, g_bf16.order, M_recv, 0, sort_end_bit, stream);
        }

        k_compute_offsets_and_padded_prefix<<<1, 256, 0, stream>>>(
            g_bf16.local_eid, g_bf16.offsets, offs_pad_out, g_bf16.M_pad_dev,
            M_recv, g_bf16.n_local, align, -1);
        k_fill_dest_from_sorted_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            g_bf16.local_eid, g_bf16.offsets, offs_pad_out, g_bf16.dest, M_recv, g_bf16.n_local);
    }
    return M_recv;
}

extern "C" void rdep_gather_xe_bf16(
    void* Xe_out,           // [M_pad, H] BF16
    int M_recv,
    int M_pad,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP FATAL: NVSHMEM not initialized for hybrid mode\n");
            abort();
        }
        if (nvshmem::g_nvshmem.profile != -1) {
            fprintf(stderr, "RDEP FATAL: rdep_gather_xe_bf16 requires BF16 NVSHMEM state (profile=-1)\n");
            abort();
        }
        if (M_recv <= 0 || M_pad <= 0) return;
        const int capacity = static_cast<int>(nvshmem::g_nvshmem.capacity);
        if (M_recv > capacity) {
            fprintf(stderr, "RDEP FATAL: gather_xe_bf16 M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
            abort();
        }
        const int max_pad = capacity + nvshmem::g_nvshmem.n_local * (nvshmem::g_nvshmem.align - 1);
        if (M_pad > max_pad) {
            fprintf(stderr, "RDEP FATAL: gather_xe_bf16 M_pad=%d exceeds max_pad=%d\n", M_pad, max_pad);
            abort();
        }

        const int H = nvshmem::g_nvshmem.H;
        const int Ha = nvshmem::g_nvshmem.Ha;
        char* local_ipc_buf = static_cast<char*>(nvshmem::g_nvshmem.ipc_buffer_ptrs[nvshmem::g_nvshmem.nvl_rank]);
        uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_ipc_buf + nvshmem::g_nvshmem.ipc_x_off);
        if (nvshmem::g_nvshmem.offs_pad == nullptr) {
            fprintf(stderr, "RDEP FATAL: gather_xe_bf16 missing offs_pad mapping from prior hybrid dispatch\n");
            abort();
        }

        zero_bf16_padding_rows_async(
            nvshmem::g_nvshmem.offsets,
            nvshmem::g_nvshmem.offs_pad,
            Xe_out,
            nvshmem::g_nvshmem.n_local,
            M_recv,
            M_pad,
            H,
            stream);

        int gather_threads = 256;
        int gather_blocks_by_work = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
        int gather_blocks = cap_warp_stride_blocks(gather_blocks_by_work);
        k_gather_bf16<<<gather_blocks, gather_threads, 0, stream>>>(
            x_buf, nvshmem::g_nvshmem.order, nvshmem::g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(Xe_out),
            M_recv, H, Ha);
        return;
    }
#endif
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: BF16 state not initialized for rdep_gather_xe_bf16\n");
        abort();
    }
    if (M_recv <= 0 || M_pad <= 0) return;
    const int capacity = static_cast<int>(g_bf16.capacity);
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP FATAL: gather_xe_bf16 M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        abort();
    }
    const int max_pad = capacity + g_bf16.n_local * (g_bf16.align - 1);
    if (M_pad > max_pad) {
        fprintf(stderr, "RDEP FATAL: gather_xe_bf16 M_pad=%d exceeds max_pad=%d\n", M_pad, max_pad);
        abort();
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);
    if (g_bf16.offs_pad_last == nullptr) {
        fprintf(stderr, "RDEP FATAL: gather_xe_bf16 missing offs_pad mapping from prior dispatch\n");
        abort();
    }

    zero_bf16_padding_rows_async(
        g_bf16.offsets,
        g_bf16.offs_pad_last,
        Xe_out,
        g_bf16.n_local,
        M_recv,
        M_pad,
        g_bf16.H,
        stream);

    int gather_threads = 256;
    int gather_blocks_by_work = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
    int gather_blocks = cap_warp_stride_blocks(gather_blocks_by_work);
    k_gather_bf16<<<gather_blocks, gather_threads, 0, stream>>>(
        x_buf, g_bf16.order, g_bf16.dest,
        static_cast<__nv_bfloat16*>(Xe_out),
        M_recv, g_bf16.H, g_bf16.Ha);
}

extern "C" void rdep_gather_meta_sorted_bf16(
    int64_t* row_id_out,     // [M_recv] int64 (device)
    float* gate_out,         // [M_recv] float32 (device)
    int M_recv,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (M_recv <= 0) return;

        char* local_ipc_buf = static_cast<char*>(nvshmem::g_nvshmem.ipc_buffer_ptrs[nvshmem::g_nvshmem.nvl_rank]);
        Meta* meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + nvshmem::g_nvshmem.ipc_meta_off);

        int t = 256;
        int b_by_work = std::max(1, (M_recv + t - 1) / t);
        int b = cap_warp_stride_blocks(b_by_work);
        k_gather_meta_sorted<<<b, t, 0, stream>>>(
            meta_buf, nvshmem::g_nvshmem.order, row_id_out, gate_out, M_recv);
        return;
    }
#endif
    if (!g_bf16.initialized) return;
    if (M_recv <= 0) return;

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    int t = 256;
    int b_by_work = std::max(1, (M_recv + t - 1) / t);
    int b = cap_warp_stride_blocks(b_by_work);
    k_gather_meta_sorted<<<b, t, 0, stream>>>(
        meta_buf, g_bf16.order, row_id_out, gate_out, M_recv);
}

extern "C" void rdep_gather_meta_sorted_blockscaled(
    int64_t* row_id_out,     // [M_recv] int64 (device)
    float* gate_out,         // [M_recv] float32 (device)
    int M_recv,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (M_recv <= 0) return;

        char* local_ipc_buf = static_cast<char*>(nvshmem::g_nvshmem.ipc_buffer_ptrs[nvshmem::g_nvshmem.nvl_rank]);
        Meta* meta_buf = reinterpret_cast<Meta*>(local_ipc_buf + nvshmem::g_nvshmem.ipc_meta_off);

        int t = 256;
        int b_by_work = std::max(1, (M_recv + t - 1) / t);
        int b = cap_warp_stride_blocks(b_by_work);
        k_gather_meta_sorted<<<b, t, 0, stream>>>(
            meta_buf, nvshmem::g_nvshmem.order, row_id_out, gate_out, M_recv);
        return;
    }
#endif
    if (!g_block.initialized) return;
    if (M_recv <= 0) return;

    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);
    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    int t = 256;
    int b_by_work = std::max(1, (M_recv + t - 1) / t);
    int b = cap_warp_stride_blocks(b_by_work);
    k_gather_meta_sorted<<<b, t, 0, stream>>>(
        meta_buf, g_block.order, row_id_out, gate_out, M_recv);
}

extern "C" void rdep_gather_from_pad_bf16(
    const void* in_pad,      // [M_pad, H] BF16
    void* out_sorted,        // [M_recv, H] BF16
    int M_recv,
    int H,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (H != nvshmem::g_nvshmem.H) {
            fprintf(stderr, "RDEP ERROR: rdep_gather_from_pad_bf16 H mismatch: got H=%d state H=%d\n",
                    H, nvshmem::g_nvshmem.H);
            return;
        }
        if (M_recv <= 0 || H <= 0) return;

        int threads = 256;
        int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_gather_from_pad_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(in_pad),
            nvshmem::g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(out_sorted),
            M_recv, H);
        return;
    }
#endif
    if (!g_bf16.initialized) return;
    if (M_recv <= 0 || H <= 0) return;

    int threads = 256;
    int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_gather_from_pad_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_pad),
        g_bf16.dest,
        static_cast<__nv_bfloat16*>(out_sorted),
        M_recv, H);
}

extern "C" void rdep_scatter_sorted_to_pad_bf16(
    const void* in_sorted,   // [M_recv, H] BF16
    void* out_pad,           // [M_pad, H] BF16
    int M_recv,
    int H,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return;
        }
        if (H != nvshmem::g_nvshmem.H) {
            fprintf(stderr, "RDEP ERROR: rdep_scatter_sorted_to_pad_bf16 H mismatch: got H=%d state H=%d\n",
                    H, nvshmem::g_nvshmem.H);
            return;
        }
        if (M_recv <= 0 || H <= 0) return;

        int threads = 256;
        int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_scatter_sorted_to_pad_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(in_sorted),
            nvshmem::g_nvshmem.dest,
            static_cast<__nv_bfloat16*>(out_pad),
            M_recv, H);
        return;
    }
#endif
    if (!g_bf16.initialized) return;
    if (M_recv <= 0 || H <= 0) return;

    int threads = 256;
    int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_scatter_sorted_to_pad_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_sorted),
        g_bf16.dest,
        static_cast<__nv_bfloat16*>(out_pad),
        M_recv, H);
}

extern "C" void rdep_zero_padding_rows_bf16(
    void* out_pad,             // [M_pad, H] BF16
    const int* offs_pad,       // [n_local]
    int M_recv,
    int M_pad,
    int H,
    cudaStream_t stream)
{
    if (out_pad == nullptr || offs_pad == nullptr) {
        fprintf(stderr, "RDEP FATAL: rdep_zero_padding_rows_bf16 received null pointer(s)\n");
        abort();
    }
    if (M_recv < 0 || M_pad < M_recv || H <= 0) {
        fprintf(stderr,
                "RDEP FATAL: invalid zero-padding args (M_recv=%d M_pad=%d H=%d)\n",
                M_recv, M_pad, H);
        abort();
    }
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP FATAL: NVSHMEM not initialized for hybrid mode\n");
            abort();
        }
        if (H != nvshmem::g_nvshmem.H) {
            fprintf(stderr, "RDEP FATAL: rdep_zero_padding_rows_bf16 H mismatch: got H=%d state H=%d\n",
                    H, nvshmem::g_nvshmem.H);
            abort();
        }
        nvshmem::zero_bf16_padding_rows_hybrid(
            out_pad, offs_pad, M_recv, M_pad, H, stream);
        return;
    }
#endif
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: BF16 state not initialized for rdep_zero_padding_rows_bf16\n");
        abort();
    }
    if (H != g_bf16.H) {
        fprintf(stderr, "RDEP FATAL: rdep_zero_padding_rows_bf16 H mismatch: got H=%d state H=%d\n",
                H, g_bf16.H);
        abort();
    }
    zero_bf16_padding_rows_async(
        g_bf16.offsets,
        offs_pad,
        out_pad,
        g_bf16.n_local,
        M_recv,
        M_pad,
        H,
        stream);
}

// ============================================================================
// 2-Phase Dispatch Implementation (IPC mode only)
// Eliminates remote atomics by exchanging counts before writing data
// ============================================================================
static int dispatch_2phase_bf16(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K, int M,
    size_t meta_off, size_t counter_off, size_t dropped_off, size_t recv_counts_off, size_t recv_offsets_off,
    int capacity,
    int* recv_out_host,
    cudaStream_t stream)
{
    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);

    // Phase 1: Count tokens per destination
    // Reset local send-count staging and dropped counter.
    cudaMemsetAsync(g_bf16.local_counters, 0, MAX_RANKS * sizeof(int), stream);
    cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);

    // Count tokens per destination
    int count_threads = 256;
    int count_blocks_by_work = std::max(1, (M + count_threads - 1) / count_threads);
    int count_blocks = cap_warp_stride_blocks(count_blocks_by_work);
    k_count_dispatch_bf16<<<count_blocks, count_threads, MAX_RANKS * sizeof(int), stream>>>(
        eids, M, g_bf16.n_local, g_bf16.local_counters);

    // Write counts to each destination's buffer
    k_write_counts_to_dests_bf16<<<1, MAX_RANKS, 0, stream>>>(g_bf16.local_counters, recv_counts_off);

    // Barrier: ensure all counts are visible
    ipc_barrier_bf16_site(stream, "dispatch_2phase_bf16/counts");

    // Phase 2: Compute prefix sums and exchange offsets
    k_compute_and_write_offsets_bf16<<<1, 32, 0, stream>>>(
        recv_counts_off, recv_offsets_off, counter_off);

    // Barrier: ensure all offsets are visible
    ipc_barrier_bf16_site(stream, "dispatch_2phase_bf16/offsets");
    const int* local_counter = reinterpret_cast<const int*>(local_buf + counter_off);
    // Prime async D2H now; phase-3 dispatch/barrier latency overlaps this copy.
    (void)poll_device_int_async(local_counter, stream);

    // Reset local_counters again (they may have been corrupted by multi-block counting)
    cudaMemsetAsync(g_bf16.local_counters, 0, MAX_RANKS * sizeof(int), stream);

    // Phase 3: Write data at deterministic offsets
    int dispatch_threads = 256;
    int dispatch_warps = M;
    int dispatch_blocks_by_work = std::max(1, (dispatch_warps * 32 + dispatch_threads - 1) / dispatch_threads);
    int dispatch_blocks = cap_warp_stride_blocks(dispatch_blocks_by_work);
    k_dispatch_2phase_bf16<<<dispatch_blocks, dispatch_threads, 0, stream>>>(
        x, eids, gates,
        g_bf16.local_counters,
        T, g_bf16.H, g_bf16.Ha, K,
        g_bf16.n_local, capacity,
        meta_off, dropped_off, recv_offsets_off);

    // Barrier: ensure all data writes are visible
    ipc_barrier_bf16_site(stream, "dispatch_2phase_bf16/writes");

    // Read M_recv (single int) produced by k_compute_and_write_offsets_bf16.
    if (recv_out_host == nullptr) {
        fprintf(stderr, "RDEP ERROR: recv_out_host (host scratch) is null\n");
        return -3;
    }
    bool recv_ok = false;
    int M_recv = read_device_int_stream_sync(local_counter, stream, &recv_ok);
    if (!recv_ok) {
        return -3;
    }
    *recv_out_host = M_recv;
    return M_recv;
}

static int dispatch_2phase_blockscaled(
    const __nv_bfloat16* x,
    const int* eids,
    const float* gates,
    int T, int K, int M,
    size_t sfa_off, size_t meta_off, size_t counter_off, size_t dropped_off,
    size_t recv_counts_off, size_t recv_offsets_off,
    int capacity,
    int* recv_out_host,
    cudaStream_t stream)
{
    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);

    cudaMemsetAsync(g_block.local_counters, 0, MAX_RANKS * sizeof(int), stream);
    cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);

    int count_threads = 256;
    int count_blocks_by_work = std::max(1, (M + count_threads - 1) / count_threads);
    int count_blocks = cap_warp_stride_blocks(count_blocks_by_work);
    k_count_dispatch_block<<<count_blocks, count_threads, MAX_RANKS * sizeof(int), stream>>>(
        eids, M, g_block.n_local, g_block.local_counters);

    k_write_counts_to_dests_block<<<1, MAX_RANKS, 0, stream>>>(g_block.local_counters, recv_counts_off);
    ipc_barrier_block_site(stream, "dispatch_2phase_blockscaled/counts");

    k_compute_and_write_offsets_block<<<1, 32, 0, stream>>>(
        recv_counts_off, recv_offsets_off, counter_off);
    ipc_barrier_block_site(stream, "dispatch_2phase_blockscaled/offsets");
    const int* local_counter = reinterpret_cast<const int*>(local_buf + counter_off);
    // Prime async D2H now; phase-3 dispatch/barrier latency overlaps this copy.
    (void)poll_device_int_async(local_counter, stream);

    cudaMemsetAsync(g_block.local_counters, 0, MAX_RANKS * sizeof(int), stream);

    int dispatch_threads = 256;
    int dispatch_warps = M;
    int dispatch_blocks_by_work = std::max(1, (dispatch_warps * 32 + dispatch_threads - 1) / dispatch_threads);
    int dispatch_blocks = cap_warp_stride_blocks(dispatch_blocks_by_work);
    if (g_block.profile == 0) {
        k_dispatch_2phase_blockscaled<true><<<dispatch_blocks, dispatch_threads, 0, stream>>>(
            x, eids, gates,
            g_block.local_counters,
            T, g_block.H, g_block.Hp, g_block.Hsf, K,
            g_block.n_local, capacity,
            sfa_off, meta_off, recv_offsets_off, dropped_off);
    } else if (g_block.profile == 1) {
        k_dispatch_2phase_blockscaled<false><<<dispatch_blocks, dispatch_threads, 0, stream>>>(
            x, eids, gates,
            g_block.local_counters,
            T, g_block.H, g_block.Hp, g_block.Hsf, K,
            g_block.n_local, capacity,
            sfa_off, meta_off, recv_offsets_off, dropped_off);
    } else {
        fprintf(stderr, "RDEP ERROR: invalid blockscaled profile=%d in dispatch_2phase_blockscaled\n", g_block.profile);
        return -3;
    }

    ipc_barrier_block_site(stream, "dispatch_2phase_blockscaled/writes");

    if (recv_out_host == nullptr) {
        fprintf(stderr, "RDEP ERROR: recv_out_host (host scratch) is null\n");
        return -3;
    }
    bool recv_ok = false;
    int M_recv = read_device_int_stream_sync(local_counter, stream, &recv_ok);
    if (!recv_ok) {
        return -3;
    }
    *recv_out_host = M_recv;
    return M_recv;
}

extern "C" int rdep_dispatch(
    const void* x,           // [T, H] - NOT expanded
    const int* eids,         // [T, K] - expert IDs (NOT flattened)
    const float* gates,      // [T, K] - gate values (NOT flattened)
    int T, int K,
    void* Xe_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    cudaStream_t stream)
{
    g_bf16.offs_pad_last = offs_pad_out;
#ifdef WITH_NVSHMEM
    // Hybrid mode: use NVSHMEM for inter-node + IPC for intra-node
    // CRITICAL: Check for hybrid mode FIRST before g_bf16.initialized,
    // because hybrid mode uses g_nvshmem state, not g_bf16 state
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return -1;
        }
        // Alignment check for hybrid mode
        if (nvshmem::g_nvshmem.H % 8 != 0) {
            fprintf(stderr, "RDEP ERROR: H=%d must be multiple of 8 for vectorized copies\n", nvshmem::g_nvshmem.H);
            return -2;
        }
        return nvshmem::dispatch_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K,
            nvshmem::g_nvshmem.align,  // Use NVSHMEM state's alignment
            Xe_out, offs_pad_out,
            dest_out, row_id_out, gate_out,
            M_pad_out,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_meta_off,
            nvshmem::g_nvshmem.ipc_counter_off,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
    }
#endif

    // Single-GPU and IPC modes: use local IPC path
    // Check g_bf16.initialized for non-hybrid mode
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP ERROR: BF16 buffers not initialized\n");
        return -1;
    }

    // Alignment assertion: H must be multiple of 8 for vectorized int4 copies
    // (8 BF16 = 16 bytes = sizeof(int4))
    if (g_bf16.H % 8 != 0) {
        fprintf(stderr, "RDEP ERROR: H=%d must be multiple of 8 for vectorized copies\n", g_bf16.H);
        return -2;
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    size_t recv_counts_off, recv_offsets_off;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size,
                        &recv_counts_off, &recv_offsets_off);

    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    int capacity = static_cast<int>(g_bf16.capacity);
    int M = T * K;

    int M_recv = 0;

    // Use 2-phase dispatch for IPC mode with world > 1 (eliminates remote atomics)
    if (k_use_2phase_dispatch && g_bf16.world > 1) {
        M_recv = dispatch_2phase_bf16(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K, M,
            meta_off, counter_off, dropped_off, recv_counts_off, recv_offsets_off,
            capacity, M_pad_out, stream);
    } else {
        // Legacy atomic-counter dispatch (single GPU or disabled 2-phase)
        // Reset counter (async, no sync needed before dispatch starts)
        cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
        cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);

        // IPC mode requires a global barrier to avoid counter races:
        // all ranks must reset their receive counters before any rank begins
        // remote atomicAdd() into those counters.
        if (g_bf16.world > 1) {
            ipc_barrier_bf16_site(stream, "dispatch_meta_bf16/pre");
        }

        // Launch fused dispatch kernel (reads [T,H] directly)
        // One warp per (tok, slot) pair
        int warps_needed = M;
        int threads = 256;
        int warps_per_block = threads / 32;
        int blocks_by_work = std::max(1, (warps_needed + warps_per_block - 1) / warps_per_block);
        int blocks = cap_warp_stride_blocks(blocks_by_work);

        k_dispatch_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            g_bf16.rank, T, g_bf16.H, g_bf16.Ha, K,
            g_bf16.n_local, capacity,
            meta_off, counter_off, dropped_off);

        // IPC mode requires a global barrier to ensure all remote writes are
        // complete before reading the local counter and sorting.
	        if (g_bf16.world > 1) {
	            ipc_barrier_bf16_site(stream, "dispatch_meta_bf16/post_dispatch");
	        }
	        const int* local_counter = reinterpret_cast<const int*>(local_buf + counter_off);
	        if (M_pad_out == nullptr) {
            fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
            return -3;
        }
        bool recv_ok = false;
        M_recv = read_device_int_stream_sync(local_counter, stream, &recv_ok);
        if (!recv_ok) {
            return -3;
        }
        *M_pad_out = M_recv;
    }

    if (M_recv < 0) {
        return M_recv;
    }
    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_bf16.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP ERROR: BF16 dispatch overflow: M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        return -3;
    }

    // Sort and gather pipeline - all async
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);

    int extract_threads = 256;
    int extract_blocks_by_work = std::max(1, (M_recv + extract_threads - 1) / extract_threads);
    int extract_blocks = cap_warp_stride_blocks(extract_blocks_by_work);
    int M_pad = 0;
    if (g_bf16.n_local == 1) {
        M_pad = ((M_recv + g_bf16.align - 1) / g_bf16.align) * g_bf16.align;
        k_init_single_expert_layout<<<extract_blocks, extract_threads, 0, stream>>>(
            g_bf16.order, g_bf16.dest,
            g_bf16.offsets, offs_pad_out, g_bf16.M_pad_dev,
            M_recv, M_pad);
    } else {
        k_extract_local_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            meta_buf, g_bf16.local_eid, g_bf16.order, M_recv);

        if (g_bf16.n_local > 1 && M_recv > 1) {
            const int sort_end_bit = radix_sort_end_bit_for_range(g_bf16.n_local);
            cub::DeviceRadixSort::SortPairs(g_bf16.sort_temp, g_bf16.sort_temp_bytes,
                g_bf16.local_eid, g_bf16.local_eid, g_bf16.order, g_bf16.order, M_recv, 0, sort_end_bit, stream);
        }

        int M_pad_bound = M_recv + g_bf16.n_local * (g_bf16.align - 1);
        M_pad = (M_pad_bound / g_bf16.align) * g_bf16.align;
        k_compute_offsets_and_padded_prefix<<<1, 256, 0, stream>>>(
            g_bf16.local_eid, g_bf16.offsets, offs_pad_out, g_bf16.M_pad_dev,
            M_recv, g_bf16.n_local, g_bf16.align, M_pad);
        k_fill_dest_from_sorted_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            g_bf16.local_eid, g_bf16.offsets, offs_pad_out, g_bf16.dest, M_recv, g_bf16.n_local);
    }

    if (dest_out)
        cudaMemcpyAsync(dest_out, g_bf16.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // Avoid a second host sync for exact M_pad:
    // keep per-expert alignment and over-approximate to a deterministic bound.
    // The prefix kernel already writes this bound to offs_pad[n_local - 1].
    *M_pad_out = M_pad;

    if (g_bf16.n_local == 1 && M_pad > M_recv) {
        const size_t tail_rows = static_cast<size_t>(M_pad - M_recv);
        __nv_bfloat16* tail_ptr =
            static_cast<__nv_bfloat16*>(Xe_out) + static_cast<size_t>(M_recv) * static_cast<size_t>(g_bf16.H);
        cudaMemsetAsync(
            tail_ptr,
            0,
            tail_rows * static_cast<size_t>(g_bf16.H) * sizeof(__nv_bfloat16),
            stream);
    } else {
        zero_bf16_padding_rows_async(
            g_bf16.offsets,
            offs_pad_out,
            Xe_out,
            g_bf16.n_local,
            M_recv,
            M_pad,
            g_bf16.H,
            stream);
    }

    int gather_threads = 256;
    int gather_blocks_by_work = std::max(1, (M_recv * 32 + gather_threads - 1) / gather_threads);
    int gather_blocks = cap_warp_stride_blocks(gather_blocks_by_work);
    k_gather_bf16<<<gather_blocks, gather_threads, 0, stream>>>(
        x_buf, g_bf16.order, g_bf16.dest,
        static_cast<__nv_bfloat16*>(Xe_out),
        M_recv, g_bf16.H, g_bf16.Ha);

    if (row_id_out != nullptr && gate_out != nullptr) {
        int t = 256;
        int b_by_work = std::max(1, (M_recv + t - 1) / t);
        int b = cap_warp_stride_blocks(b_by_work);
        k_gather_meta_sorted<<<b, t, 0, stream>>>(
            meta_buf, g_bf16.order, row_id_out, gate_out, M_recv);
    }

    g_bf16.M_pad = M_pad;
    return M_recv;
}

// ============================================================================
// Host API: Blockscaled Dispatch
// ============================================================================

extern "C" int rdep_dispatch_meta_blockscaled(
    const void* x,          // [T, H] BF16 - NOT expanded
    const int* eids,        // [T, K] expert IDs
    const float* gates,     // [T, K] gate values
    int T, int K,
    int* offs_pad_out,      // [n_local] device int32
    int* M_pad_out,         // host int32 (pinned recommended). Used as a host scratch for M_recv/M_pad.
    cudaStream_t stream)
{
    g_block.offs_pad_last = offs_pad_out;
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return -1;
        }
        if (nvshmem::g_nvshmem.profile < 0) {
            fprintf(stderr, "RDEP ERROR: hybrid blockscaled dispatch requires blockscaled NVSHMEM profile state\n");
            return -2;
        }
        return nvshmem::dispatch_hybrid_blockscaled(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K,
            /*Xe_q_out=*/nullptr,
            /*Xe_sf_out=*/nullptr,
            offs_pad_out,
            /*dest_out=*/nullptr,
            /*row_id_out=*/nullptr,
            /*gate_out=*/nullptr,
            M_pad_out,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_x_off,
            nvshmem::g_nvshmem.ipc_sfa_off,
            nvshmem::g_nvshmem.ipc_meta_off,
            nvshmem::g_nvshmem.ipc_counter_off,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
    }
#endif

    if (!g_block.initialized) return -1;

    int M = T * K;
    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    size_t recv_counts_off, recv_offsets_off;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size,
                               &recv_counts_off, &recv_offsets_off);

    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    int capacity = static_cast<int>(g_block.capacity);

    int M_recv = 0;
    // Backward dispatch-meta is latency-sensitive but correctness-critical.
    // Keep it on the direct single-phase IPC path to avoid 2-phase barrier
    // ordering deadlocks under large distributed autograd graphs.
    constexpr bool k_use_2phase_dispatch_meta_blockscaled = false;
    if (k_use_2phase_dispatch_meta_blockscaled && k_use_2phase_dispatch && g_block.world > 1) {
        M_recv = dispatch_2phase_blockscaled(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K, M,
            sfa_off, meta_off, counter_off, dropped_off,
            recv_counts_off, recv_offsets_off,
            capacity, M_pad_out, stream);
    } else {
        cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
        cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);
        if (g_block.world > 1) {
            ipc_barrier_block_site(stream, "dispatch_meta_blockscaled/pre");
        }

        int threads = 256;
        int warps_per_block = threads / 32;
        int blocks_by_work = std::max(1, (M + warps_per_block - 1) / warps_per_block);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        if (g_block.profile == 0) {
            k_dispatch_blockscaled<true><<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x), eids, gates,
                g_block.rank, T, g_block.H, g_block.Hp, g_block.Hsf, K,
                g_block.n_local, capacity,
                sfa_off, meta_off, counter_off, dropped_off);
        } else if (g_block.profile == 1) {
            k_dispatch_blockscaled<false><<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x), eids, gates,
                g_block.rank, T, g_block.H, g_block.Hp, g_block.Hsf, K,
                g_block.n_local, capacity,
                sfa_off, meta_off, counter_off, dropped_off);
        } else {
            fprintf(stderr, "RDEP ERROR: invalid blockscaled profile=%d in rdep_dispatch_meta_blockscaled\n", g_block.profile);
            return -3;
        }

		        if (g_block.world > 1) {
		            ipc_barrier_block_site(stream, "dispatch_meta_blockscaled/post_dispatch");
		        }
	        const int* local_counter = reinterpret_cast<const int*>(local_buf + counter_off);
	        if (M_pad_out == nullptr) {
	            fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
	            return -3;
        }
        bool recv_ok = false;
        M_recv = read_device_int_stream_sync(local_counter, stream, &recv_ok);
        if (!recv_ok) {
            return -3;
        }
        *M_pad_out = M_recv;
    }

    if (M_recv < 0) {
        return M_recv;
    }
    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_block.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP ERROR: blockscaled dispatch-meta overflow: M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        return -3;
    }

    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    int extract_threads = 256;
    int extract_blocks_by_work = std::max(1, (M_recv + extract_threads - 1) / extract_threads);
    int extract_blocks = cap_warp_stride_blocks(extract_blocks_by_work);
    int M_pad = 0;
    if (g_block.n_local == 1) {
        M_pad = ((M_recv + g_block.align - 1) / g_block.align) * g_block.align;
        k_init_single_expert_layout<<<extract_blocks, extract_threads, 0, stream>>>(
            g_block.order, g_block.dest,
            g_block.offsets, offs_pad_out, g_block.M_pad_dev,
            M_recv, M_pad);
    } else {
        k_extract_local_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            meta_buf, g_block.local_eid, g_block.order, M_recv);

        if (g_block.n_local > 1 && M_recv > 1) {
            const int sort_end_bit = radix_sort_end_bit_for_range(g_block.n_local);
            cub::DeviceRadixSort::SortPairs(g_block.sort_temp, g_block.sort_temp_bytes,
                g_block.local_eid, g_block.local_eid, g_block.order, g_block.order, M_recv, 0, sort_end_bit, stream);
        }

        int M_pad_bound = M_recv + g_block.n_local * (g_block.align - 1);
        M_pad = (M_pad_bound / g_block.align) * g_block.align;
        k_compute_offsets_and_padded_prefix<<<1, 256, 0, stream>>>(
            g_block.local_eid, g_block.offsets, offs_pad_out, g_block.M_pad_dev,
            M_recv, g_block.n_local, g_block.align, M_pad);
        k_fill_dest_from_sorted_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            g_block.local_eid, g_block.offsets, offs_pad_out, g_block.dest, M_recv, g_block.n_local);
    }

    // Deterministic upper bound on M_pad (>= exact) while preserving per-expert alignment.
    // The prefix kernel already writes this bound to offs_pad[n_local - 1].
    *M_pad_out = M_pad;
    g_block.M_pad = M_pad;
    return M_recv;
}

extern "C" void rdep_gather_xe_blockscaled(
    void* Xe_q_out,     // [M_pad, Hp] uint16
    void* Xe_sf_out,    // [M_pad, Hsf] uint8 (packed MMA layout)
    int M_recv,
    int M_pad,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP FATAL: NVSHMEM not initialized for hybrid mode\n");
            abort();
        }
        if (M_recv > 0 && (M_pad % nvshmem::g_nvshmem.align) != 0) {
            fprintf(stderr,
                    "RDEP FATAL: gather_xe_blockscaled requires M_pad aligned to %d (got M_pad=%d)\n",
                    nvshmem::g_nvshmem.align, M_pad);
            abort();
        }
        nvshmem::gather_xe_hybrid_blockscaled(
            Xe_q_out,
            Xe_sf_out,
            M_recv,
            M_pad,
            stream);
        return;
    }
#endif

    if (!g_block.initialized) {
        fprintf(stderr, "RDEP FATAL: blockscaled state not initialized for rdep_gather_xe_blockscaled\n");
        abort();
    }
    if (M_recv <= 0 || M_pad <= 0) return;
    if ((M_pad % g_block.align) != 0) {
        fprintf(stderr,
                "RDEP FATAL: gather_xe_blockscaled requires M_pad aligned to %d (got M_pad=%d)\n",
                g_block.align, M_pad);
        abort();
    }
    if (M_pad < M_recv) {
        fprintf(stderr,
                "RDEP FATAL: gather_xe_blockscaled requires M_pad >= M_recv (M_pad=%d M_recv=%d)\n",
                M_pad, M_recv);
        abort();
    }
    const int capacity = static_cast<int>(g_block.capacity);
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP FATAL: gather_xe_blockscaled M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        abort();
    }
    const int max_pad = capacity + g_block.n_local * (g_block.align - 1);
    if (M_pad > max_pad) {
        fprintf(stderr, "RDEP FATAL: gather_xe_blockscaled M_pad=%d exceeds max_pad=%d\n", M_pad, max_pad);
        abort();
    }

    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);
    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);
    uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(local_buf + sfa_off);
    if (g_block.offs_pad_last == nullptr) {
        fprintf(stderr, "RDEP FATAL: gather_xe_blockscaled missing offs_pad mapping from prior dispatch\n");
        abort();
    }

    fill_blockscaled_padding_rows_async(
        g_block.offsets,
        g_block.offs_pad_last,
        Xe_q_out,
        nullptr,
        g_block.n_local,
        M_recv,
        M_pad,
        g_block.Hp,
        g_block.Hsf,
        stream);
    fill_blockscaled_padding_sf_swizzled_async(
        g_block.offsets,
        g_block.offs_pad_last,
        Xe_sf_out,
        g_block.n_local,
        M_recv,
        M_pad,
        g_block.Hsf,
        stream);

    const int threads = 256;
    const int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_gather_blockscaled<<<blocks, threads, 0, stream>>>(
        x_buf, sfa_buf, g_block.order, g_block.dest,
        static_cast<uint16_t*>(Xe_q_out), static_cast<uint8_t*>(Xe_sf_out),
        M_recv, M_pad, g_block.Hp, g_block.Hsf);
    g_block.M_pad = M_pad;
}

extern "C" void rdep_copy_blockscaled_layout(
    int* dest_out,        // [M_recv] int32 (device)
    int* offsets_out,     // [n_local + 1] int32 (device)
    int M_recv,
    cudaStream_t stream)
{
    if (!g_block.initialized) {
        fprintf(stderr, "RDEP FATAL: rdep_copy_blockscaled_layout requires initialized blockscaled state\n");
        abort();
    }
    if (M_recv < 0 || M_recv > static_cast<int>(g_block.capacity)) {
        fprintf(stderr,
                "RDEP FATAL: rdep_copy_blockscaled_layout invalid M_recv=%d (capacity=%zu)\n",
                M_recv, g_block.capacity);
        abort();
    }
    if (dest_out == nullptr || offsets_out == nullptr) {
        fprintf(stderr, "RDEP FATAL: rdep_copy_blockscaled_layout received null output pointer(s)\n");
        abort();
    }
    if (M_recv > 0) {
        cudaError_t e = cudaMemcpyAsync(
            dest_out, g_block.dest, static_cast<size_t>(M_recv) * sizeof(int),
            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "RDEP FATAL: rdep_copy_blockscaled_layout dest memcpy failed: %s\n",
                    cudaGetErrorString(e));
            abort();
        }
    }
    cudaError_t e = cudaMemcpyAsync(
        offsets_out, g_block.offsets, static_cast<size_t>(g_block.n_local + 1) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "RDEP FATAL: rdep_copy_blockscaled_layout offsets memcpy failed: %s\n",
                cudaGetErrorString(e));
        abort();
    }
}

extern "C" void rdep_restore_layout_from_saved(
    const int* dest_in,      // [M_recv] int32 (device)
    const int* offsets_in,   // [n_local + 1] int32 (device)
    const int* offs_pad_in,  // [n_local] int32 (device)
    int M_recv,
    cudaStream_t stream)
{
    if (M_recv < 0) {
        fprintf(stderr, "RDEP FATAL: rdep_restore_layout_from_saved invalid M_recv=%d\n", M_recv);
        abort();
    }
    if (offsets_in == nullptr || offs_pad_in == nullptr) {
        fprintf(stderr, "RDEP FATAL: rdep_restore_layout_from_saved received null offsets/offs_pad pointer\n");
        abort();
    }
    if (M_recv > 0 && dest_in == nullptr) {
        fprintf(stderr, "RDEP FATAL: rdep_restore_layout_from_saved received null dest pointer for M_recv=%d\n", M_recv);
        abort();
    }

    const bool has_bf16 = g_bf16.initialized;
    const bool has_block = g_block.initialized;
    if (!has_bf16 && !has_block) {
        fprintf(stderr, "RDEP FATAL: rdep_restore_layout_from_saved requires initialized BF16 or blockscaled state\n");
        abort();
    }
    if (has_bf16 && M_recv > static_cast<int>(g_bf16.capacity)) {
        fprintf(stderr,
                "RDEP FATAL: rdep_restore_layout_from_saved BF16 M_recv=%d exceeds capacity=%zu\n",
                M_recv, g_bf16.capacity);
        abort();
    }
    if (has_block && M_recv > static_cast<int>(g_block.capacity)) {
        fprintf(stderr,
                "RDEP FATAL: rdep_restore_layout_from_saved blockscaled M_recv=%d exceeds capacity=%zu\n",
                M_recv, g_block.capacity);
        abort();
    }
    if (has_bf16 && has_block && g_bf16.n_local != g_block.n_local) {
        fprintf(stderr,
                "RDEP FATAL: rdep_restore_layout_from_saved n_local mismatch bf16=%d blockscaled=%d\n",
                g_bf16.n_local, g_block.n_local);
        abort();
    }

    auto copy_layout = [&](int* dest_dst, int* offsets_dst, int n_local) {
        if (M_recv > 0) {
            cudaError_t e = cudaMemcpyAsync(
                dest_dst, dest_in, static_cast<size_t>(M_recv) * sizeof(int),
                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) {
                fprintf(stderr, "RDEP FATAL: rdep_restore_layout_from_saved dest memcpy failed: %s\n",
                        cudaGetErrorString(e));
                abort();
            }
        }
        cudaError_t e = cudaMemcpyAsync(
            offsets_dst, offsets_in, static_cast<size_t>(n_local + 1) * sizeof(int),
            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "RDEP FATAL: rdep_restore_layout_from_saved offsets memcpy failed: %s\n",
                    cudaGetErrorString(e));
            abort();
        }
    };

    if (has_bf16) {
        copy_layout(g_bf16.dest, g_bf16.offsets, g_bf16.n_local);
        g_bf16.offs_pad_last = const_cast<int*>(offs_pad_in);
    }
    if (has_block) {
        copy_layout(g_block.dest, g_block.offsets, g_block.n_local);
        g_block.offs_pad_last = const_cast<int*>(offs_pad_in);
    }
}

extern "C" int rdep_dispatch_blockscaled(
    const void* x,          // [T, H] BF16 - NOT expanded
    const int* eids,        // [T, K] expert IDs
    const float* gates,     // [T, K] gate values
    int T, int K,
    void* Xe_q_out,
    void* Xe_sf_out,
    int* offs_pad_out,
    int* dest_out,
    int64_t* row_id_out,
    float* gate_out,
    int* M_pad_out,
    cudaStream_t stream)
{
    g_block.offs_pad_last = offs_pad_out;
#ifdef WITH_NVSHMEM
    // Hybrid mode: use NVSHMEM for inter-node + IPC for intra-node
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP ERROR: NVSHMEM not initialized for hybrid mode\n");
            return -1;
        }
        if (nvshmem::g_nvshmem.profile < 0) {
            fprintf(stderr, "RDEP ERROR: hybrid blockscaled dispatch requires blockscaled NVSHMEM profile state\n");
            return -2;
        }
        return nvshmem::dispatch_hybrid_blockscaled(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K, Xe_q_out, Xe_sf_out,
            offs_pad_out, dest_out,
            row_id_out, gate_out,
            M_pad_out,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_x_off,
            nvshmem::g_nvshmem.ipc_sfa_off,
            nvshmem::g_nvshmem.ipc_meta_off,
            nvshmem::g_nvshmem.ipc_counter_off,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
    }
#endif

    if (!g_block.initialized) return -1;

    // Single-GPU and IPC modes: use local IPC path
    int M = T * K;
    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    size_t recv_counts_off, recv_offsets_off;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size,
                               &recv_counts_off, &recv_offsets_off);

    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    int capacity = static_cast<int>(g_block.capacity);

    int M_recv = 0;
    if (k_use_2phase_dispatch && g_block.world > 1) {
        M_recv = dispatch_2phase_blockscaled(
            static_cast<const __nv_bfloat16*>(x), eids, gates,
            T, K, M,
            sfa_off, meta_off, counter_off, dropped_off,
            recv_counts_off, recv_offsets_off,
            capacity, M_pad_out, stream);
    } else {
        cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);
        cudaMemsetAsync(local_buf + dropped_off, 0, sizeof(int), stream);
        if (g_block.world > 1) {
            ipc_barrier_block(stream);
        }

        int threads = 256;
        int warps_per_block = threads / 32;
        int blocks_by_work = std::max(1, (M + warps_per_block - 1) / warps_per_block);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        if (g_block.profile == 0) {
            k_dispatch_blockscaled<true><<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x), eids, gates,
                g_block.rank, T, g_block.H, g_block.Hp, g_block.Hsf, K,
                g_block.n_local, capacity,
                sfa_off, meta_off, counter_off, dropped_off);
        } else if (g_block.profile == 1) {
            k_dispatch_blockscaled<false><<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x), eids, gates,
                g_block.rank, T, g_block.H, g_block.Hp, g_block.Hsf, K,
                g_block.n_local, capacity,
                sfa_off, meta_off, counter_off, dropped_off);
        } else {
            fprintf(stderr, "RDEP ERROR: invalid blockscaled profile=%d in rdep_dispatch_blockscaled\n", g_block.profile);
            return -3;
        }

	        if (g_block.world > 1) {
	            ipc_barrier_block(stream);
	        }
	        const int* local_counter = reinterpret_cast<const int*>(local_buf + counter_off);
	        if (M_pad_out == nullptr) {
	            fprintf(stderr, "RDEP ERROR: M_pad_out (host scratch) is null\n");
	            return -3;
        }
        bool recv_ok = false;
        M_recv = read_device_int_stream_sync(local_counter, stream, &recv_ok);
        if (!recv_ok) {
            return -3;
        }
        *M_pad_out = M_recv;
    }

    if (M_recv < 0) {
        return M_recv;
    }
    if (M_recv <= 0) {
        cudaMemsetAsync(offs_pad_out, 0, g_block.n_local * sizeof(int), stream);
        *M_pad_out = 0;
        return 0;
    }
    if (M_recv > capacity) {
        fprintf(stderr, "RDEP ERROR: blockscaled dispatch overflow: M_recv=%d exceeds capacity=%d\n", M_recv, capacity);
        return -3;
    }

    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);
    uint8_t* sfa_buf = reinterpret_cast<uint8_t*>(local_buf + sfa_off);

    int extract_threads = 256;
    int extract_blocks_by_work = std::max(1, (M_recv + extract_threads - 1) / extract_threads);
    int extract_blocks = cap_warp_stride_blocks(extract_blocks_by_work);
    int M_pad = 0;
    if (g_block.n_local == 1) {
        M_pad = ((M_recv + g_block.align - 1) / g_block.align) * g_block.align;
        k_init_single_expert_layout<<<extract_blocks, extract_threads, 0, stream>>>(
            g_block.order, g_block.dest,
            g_block.offsets, offs_pad_out, g_block.M_pad_dev,
            M_recv, M_pad);
    } else {
        k_extract_local_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            meta_buf, g_block.local_eid, g_block.order, M_recv);

        if (g_block.n_local > 1 && M_recv > 1) {
            const int sort_end_bit = radix_sort_end_bit_for_range(g_block.n_local);
            cub::DeviceRadixSort::SortPairs(g_block.sort_temp, g_block.sort_temp_bytes,
                g_block.local_eid, g_block.local_eid, g_block.order, g_block.order, M_recv, 0, sort_end_bit, stream);
        }

        int M_pad_bound = M_recv + g_block.n_local * (g_block.align - 1);
        M_pad = (M_pad_bound / g_block.align) * g_block.align;
        k_compute_offsets_and_padded_prefix<<<1, 256, 0, stream>>>(
            g_block.local_eid, g_block.offsets, offs_pad_out, g_block.M_pad_dev,
            M_recv, g_block.n_local, g_block.align, M_pad);
        k_fill_dest_from_sorted_eid<<<extract_blocks, extract_threads, 0, stream>>>(
            g_block.local_eid, g_block.offsets, offs_pad_out, g_block.dest, M_recv, g_block.n_local);
    }

    if (dest_out)
        cudaMemcpyAsync(dest_out, g_block.dest, M_recv * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // Avoid a second host sync for exact M_pad:
    // - Exact padded total is sum_e align_up(cnt_e, align) and depends on routing.
    // - For blockscaled grouped GEMM we only need per-expert offsets to be aligned.
    // - Over-approximate to a deterministic upper bound and extend the last expert.
    //
    // Upper bound (aligned, >= exact): floor((M_recv + n_local*(align-1)) / align) * align.
    // The prefix kernel already writes this bound to offs_pad[n_local - 1].
    *M_pad_out = M_pad;

    if (g_block.n_local == 1 && M_pad > M_recv) {
        const size_t tail_rows = static_cast<size_t>(M_pad - M_recv);
        uint16_t* q_tail =
            static_cast<uint16_t*>(Xe_q_out) + static_cast<size_t>(M_recv) * static_cast<size_t>(g_block.Hp);
        uint8_t* sf_tail =
            static_cast<uint8_t*>(Xe_sf_out) + static_cast<size_t>(M_recv) * static_cast<size_t>(g_block.Hsf);
        cudaMemsetAsync(
            q_tail,
            0,
            tail_rows * static_cast<size_t>(g_block.Hp) * sizeof(uint16_t),
            stream);
        cudaMemsetAsync(
            sf_tail,
            0,
            tail_rows * static_cast<size_t>(g_block.Hsf) * sizeof(uint8_t),
            stream);
    } else {
        fill_blockscaled_padding_rows_async(
            g_block.offsets,
            offs_pad_out,
            Xe_q_out,
            nullptr,
            g_block.n_local,
            M_recv,
            M_pad,
            g_block.Hp,
            g_block.Hsf,
            stream);
        fill_blockscaled_padding_sf_swizzled_async(
            g_block.offsets,
            offs_pad_out,
            Xe_sf_out,
            g_block.n_local,
            M_recv,
            M_pad,
            g_block.Hsf,
            stream);
    }

    int warps_needed = M_recv;
    int gather_threads = 256;
    int gather_warps_per_block = gather_threads / 32;
    int gather_blocks_by_work = std::max(1, (warps_needed + gather_warps_per_block - 1) / gather_warps_per_block);
    int gather_blocks = cap_warp_stride_blocks(gather_blocks_by_work);
    // Gather packed activations and write SF directly in CUTLASS MMA layout.
    k_gather_blockscaled<<<gather_blocks, gather_threads, 0, stream>>>(
        x_buf, sfa_buf, g_block.order, g_block.dest,
        static_cast<uint16_t*>(Xe_q_out), static_cast<uint8_t*>(Xe_sf_out),
        M_recv, M_pad, g_block.Hp, g_block.Hsf);

    g_block.M_pad = M_pad;

    if (row_id_out != nullptr && gate_out != nullptr) {
        int t = 256;
        int b_by_work = std::max(1, (M_recv + t - 1) / t);
        int b = cap_warp_stride_blocks(b_by_work);
        k_gather_meta_sorted<<<b, t, 0, stream>>>(
            meta_buf, g_block.order, row_id_out, gate_out, M_recv);
    }

    return M_recv;
}

// ============================================================================
// Return Scatter (IPC version)
// ============================================================================

// BF16 return scatter kernel (uses d_buffer_ptrs_bf16)
__global__ void k_return_scatter_bf16(
    const __nv_bfloat16* __restrict__ Ye,
    const int* __restrict__ order,
    const Meta* __restrict__ meta,
    float* __restrict__ out,
    int M_recv, int H, int Ha, int T, int K,
    int my_rank, int world, int capacity,
    size_t meta_off, size_t counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        // Bounds check on orig_i
        if (orig_i < 0 || orig_i >= capacity) {
            if (lane == 0) printf("RDEP BUG k_return_scatter: orig_i=%d out of bounds [0,%d)\n", orig_i, capacity);
            continue;
        }
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

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG k_return_scatter: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)m.row_id);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG k_return_scatter: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        if (src_rank == my_rank) {
            // Local: scatter directly
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32)
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
        } else {
            // Remote: write to source rank's buffer via IPC (BF16 buffers)
            char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + counter_off);

            int slot_r;
            if (lane == 0)
                slot_r = atomicAdd(counter, 1);
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            // Write metadata (16B) - sys-scope for cross-GPU visibility.
            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_relaxed_sys_v4_s32(meta_dst, meta_val);
            }

            // Write BF16 payload - sys-scope for cross-GPU visibility.
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;

            int h = lane * 8;
            for (; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(y_row + h);
                    st_relaxed_sys_v4_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        st_relaxed_sys_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

// Blockscaled return scatter kernel (uses d_buffer_ptrs_block)
__global__ void k_return_scatter_blockscaled_bf16(
    const __nv_bfloat16* __restrict__ Ye,
    const int* __restrict__ order,
    const Meta* __restrict__ meta,
    float* __restrict__ out,
    int M_recv, int H, int Ha, int T, int K,
    int my_rank, int world, int capacity,
    size_t y_off, size_t meta_off, size_t counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        if (orig_i < 0 || orig_i >= capacity) continue;
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
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        if (src_rank == my_rank) {
            // Local: scatter directly
            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            float* out_row = out + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32)
                atomicAdd(out_row + h, __bfloat162float(y_row[h]) * m.gate);
        } else {
            // Remote: write to source rank's buffer via IPC (blockscaled buffers)
            char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
            uint16_t* y_buf = reinterpret_cast<uint16_t*>(dest_buf + y_off);
            Meta* meta_buf = reinterpret_cast<Meta*>(dest_buf + meta_off);
            int* counter = reinterpret_cast<int*>(dest_buf + counter_off);

            int slot_r;
            if (lane == 0)
                slot_r = atomicAdd(counter, 1);
            slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);

            if (slot_r >= capacity) continue;

            if (lane == 0) {
                Meta mr{m.row_id, 0, m.gate};
                int4* meta_dst = reinterpret_cast<int4*>(&meta_buf[slot_r]);
                int4 meta_val = *reinterpret_cast<const int4*>(&mr);
                st_na_v4_s32(meta_dst, meta_val);
            }

            const __nv_bfloat16* y_row = Ye + (int64_t)sorted_i * H;
            uint16_t* dst = y_buf + (int64_t)slot_r * Ha;

            int h = lane * 8;
            for (; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    int4 v = *reinterpret_cast<const int4*>(y_row + h);
                    st_na_v4_s32(d, v);
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        st_na_relaxed_gpu_b16(dst + hh, reinterpret_cast<const uint16_t*>(y_row)[hh]);
                    }
                }
            }
        }
    }
}

__global__ void k_scatter_received_bf16(
    const uint16_t* __restrict__ y_recv,
    const Meta* __restrict__ meta_recv,
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
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_recv + i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        const uint16_t* y_row = y_recv + (int64_t)i * Ha;
        float* out_row = out + (int64_t)tok * H;

        // NOTE: y_row is written by peer GPUs via IPC. Receiver-side L2 is not
        // coherent with peer writes; use non-caching loads to observe updates.
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

// Device-counter driven variant: removes host D2H sync for M_ret in legacy IPC return.
__global__ void k_scatter_received_bf16_dynamic(
    const uint16_t* __restrict__ y_recv,
    const Meta* __restrict__ meta_recv,
    const int* __restrict__ m_ret_dev,
    float* __restrict__ out,
    int capacity, int H, int Ha, int T, int K)
{
    __shared__ int s_m_ret;
    if (threadIdx.x == 0) {
        int m = 0;
        if (m_ret_dev != nullptr) {
            m = *m_ret_dev;
        }
        if (m < 0) m = 0;
        if (m > capacity) m = capacity;
        s_m_ret = m;
    }
    __syncthreads();

    const int M_ret = s_m_ret;
    if (M_ret <= 0) return;

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
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_recv + i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        const uint16_t* y_row = y_recv + (int64_t)i * Ha;
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
// IPC Token-Slot Return / dX (BF16)
//
// For operations where every (tok,slot) exists exactly once on the source rank
// (forward return, backward dX), avoid dynamic append+metadata:
// write directly into a per-(tok,slot) buffer on the source rank:
//   idx = tok*K + slot
// This eliminates receive counters, meta overwrites, and atomicAdd scatter.
// ============================================================================

__global__ void k_return_write_tokslot_bf16(
    const __nv_bfloat16* __restrict__ Ye_sorted,
    const int* __restrict__ order,
    const Meta* __restrict__ meta_buf,
    int M_recv, int H, int Ha, int T, int K,
    int world, int capacity,
    size_t tok_y_off, size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        // Bounds check on orig_i
        if (orig_i < 0 || orig_i >= capacity) {
            if (lane == 0) printf("RDEP BUG: orig_i=%d out of bounds [0,%d)\n", orig_i, capacity);
            continue;
        }
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_buf + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)m.row_id);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;

        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        // Gate is a scalar; one lane writes. Use sys-scope for cross-GPU visibility.
        if (lane == 0) st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(m.gate));

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(Ye_sorted + (int64_t)sorted_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // H is required to be multiple of 8 (int4 = 8 BF16). Sys-scope for cross-GPU.
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

__global__ void k_return_write_tokslot_blockscaled(
    const __nv_bfloat16* __restrict__ Ye_sorted,
    const int* __restrict__ order,
    const Meta* __restrict__ meta_buf,
    int M_recv, int H, int Ha, int T, int K,
    int world, int capacity,
    size_t tok_y_off, size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        if (orig_i < 0 || orig_i >= capacity) {
            if (lane == 0) printf("RDEP BUG: orig_i=%d out of bounds [0,%d)\n", orig_i, capacity);
            continue;
        }
        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec {
            Meta m;
            int4 v;
        };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_buf + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)m.row_id);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;

        char* dst_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        // Gate is a scalar; one lane writes. Use sys-scope for cross-GPU visibility.
        if (lane == 0) st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(m.gate));

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(Ye_sorted + (int64_t)sorted_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // Sys-scope for cross-GPU visibility.
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
    }

    // Fence to ensure all sys-scope writes are visible before kernel completes
    fence_acq_rel_sys();
}

__global__ void k_return_write_tokslot_from_pad_bf16(
    const __nv_bfloat16* __restrict__ Ye_pad,      // [M_pad, H]
    const int* __restrict__ dest,                  // [M_recv] sorted_i -> pad_i
    const int* __restrict__ order,                 // [M_recv] sorted_i -> orig_i
    const Meta* __restrict__ meta_buf,             // [capacity]
    int M_recv, int H, int Ha, int T, int K,
    int world, int capacity, int M_pad_cap,
    size_t tok_y_off, size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const int my_rank = d_my_rank_bf16;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        if (orig_i < 0 || orig_i >= capacity) continue;

        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec { Meta m; int4 v; };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_buf + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const int pad_i = dest[sorted_i];
        if (pad_i < 0 || pad_i >= M_pad_cap) continue;

        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        if (lane == 0) {
            if (src_rank == my_rank) tok_gate[idx] = m.gate;
            else st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(m.gate));
        }

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(Ye_pad + (int64_t)pad_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            if (src_rank == my_rank) *d = v;
            else st_relaxed_sys_v4_s32(d, v);
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_return_write_tokslot_from_pad_blockscaled(
    const __nv_bfloat16* __restrict__ Ye_pad,      // [M_pad, H]
    const int* __restrict__ dest,                  // [M_recv] sorted_i -> pad_i
    const int* __restrict__ order,                 // [M_recv] sorted_i -> orig_i
    const Meta* __restrict__ meta_buf,             // [capacity]
    int M_recv, int H, int Ha, int T, int K,
    int world, int capacity, int M_pad_cap,
    size_t tok_y_off, size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const int my_rank = d_my_rank_block;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        int orig_i = order[sorted_i];
        if (orig_i < 0 || orig_i >= capacity) continue;

        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec { Meta m; int4 v; };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_buf + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const int pad_i = dest[sorted_i];
        if (pad_i < 0 || pad_i >= M_pad_cap) continue;

        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        if (lane == 0) {
            if (src_rank == my_rank) tok_gate[idx] = m.gate;
            else st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(m.gate));
        }

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(Ye_pad + (int64_t)pad_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            if (src_rank == my_rank) *d = v;
            else st_relaxed_sys_v4_s32(d, v);
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_return_scatter_from_pad_atomic(
    const __nv_bfloat16* __restrict__ Ye_pad,  // [M_pad, H]
    const int* __restrict__ dest,              // [M_recv]
    const int* __restrict__ order,             // [M_recv]
    const Meta* __restrict__ meta_buf,         // [capacity]
    float* __restrict__ out,                   // [T, H]
    int M_recv, int H, int T, int K,
    int capacity,
    int M_pad_cap)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int sorted_i = warp_id; sorted_i < M_recv; sorted_i += num_warps) {
        const int orig_i = order[sorted_i];
        if (orig_i < 0 || orig_i >= capacity) continue;

        static_assert(sizeof(Meta) == sizeof(int4), "Meta must be 16B");
        union MetaVec { Meta m; int4 v; };
        MetaVec mv;
        mv.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(meta_buf + orig_i));
        const Meta m = mv.m;

        int src_rank, tok, slot;
        decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const int pad_i = dest[sorted_i];
        if (pad_i < 0 || pad_i >= M_pad_cap) continue;

        const uint16_t* y_u16 = reinterpret_cast<const uint16_t*>(Ye_pad + (int64_t)pad_i * H);
        float* out_row = out + (int64_t)tok * H;
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(y_u16 + h);
            union U16x8 { int4 v; uint16_t u[8]; };
            U16x8 y8;
            y8.v = v;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int hh = h + j;
                if (hh >= H) break;
                const __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&y8.u[j]);
                atomicAdd(out_row + hh, __bfloat162float(bf) * m.gate);
            }
        }
    }
}

__global__ void k_reduce_tokslot_gate_bf16(
    const uint16_t* __restrict__ tok_y,
    const float* __restrict__ tok_gate,
    float* __restrict__ out,
    int T, int H, int Ha, int K)
{
    if (K <= 0 || K > 32) return;

    __shared__ float g_shared[32];
    for (int tok = static_cast<int>(blockIdx.x); tok < T; tok += static_cast<int>(gridDim.x)) {
        if (threadIdx.x < K) {
            g_shared[threadIdx.x] = ld_nc_f32(tok_gate + (int64_t)tok * K + threadIdx.x);
        }
        __syncthreads();

        int vec = static_cast<int>(threadIdx.x);
        for (int h0 = vec * 8; h0 < H; h0 += static_cast<int>(blockDim.x) * 8) {
            float acc[8] = {0};
            for (int slot = 0; slot < K; ++slot) {
                const float g = g_shared[slot];
                if (g == 0.0f) continue;
                const uint16_t* y_row = tok_y + (int64_t)(tok * K + slot) * Ha + h0;
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
                        acc[j] += __bfloat162float(bf) * g;
                    }
                }
            }

            float* out_row = out + (int64_t)tok * H + h0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int hh = h0 + j;
                if (hh < H) out_row[j] = acc[j];
            }
        }
        __syncthreads();
    }
}

__global__ void k_send_dx_tokslot_bf16(
    const __nv_bfloat16* __restrict__ dXe_sorted,
    const int64_t* __restrict__ row_id,
    int M, int T, int H, int Ha, int K,
    int world,
    size_t tok_y_off,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG k_send_dx: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)row_id[i]);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG k_send_dx: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(dXe_sorted + (int64_t)i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // Sys-scope for cross-GPU visibility
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
        if (lane == 0) {
            st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(1.0f));
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_send_dx_tokslot_blockscaled(
    const __nv_bfloat16* __restrict__ dXe_sorted,
    const int64_t* __restrict__ row_id,
    int M, int T, int H, int Ha, int K,
    int world,
    size_t tok_y_off,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        // Bounds check on decoded values
        if (src_rank < 0 || src_rank >= world) {
            if (lane == 0) printf("RDEP BUG k_send_dx_block: src_rank=%d out of bounds [0,%d), row_id=%lld\n", src_rank, world, (long long)row_id[i]);
            continue;
        }
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) {
            if (lane == 0) printf("RDEP BUG k_send_dx_block: tok=%d slot=%d out of bounds T=%d K=%d\n", tok, slot, T, K);
            continue;
        }

        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(dXe_sorted + (int64_t)i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        // Sys-scope for cross-GPU visibility
        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
        if (lane == 0) {
            st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(1.0f));
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_send_dx_tokslot_from_pad_bf16(
    const __nv_bfloat16* __restrict__ dXe_pad,
    const int* __restrict__ dest,
    const int64_t* __restrict__ row_id,
    int M, int T, int H, int Ha, int K,
    int world, int M_pad_cap,
    size_t tok_y_off,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const int pad_i = dest[i];
        if (pad_i < 0 || pad_i >= M_pad_cap) continue;
        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(dXe_pad + (int64_t)pad_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
        if (lane == 0) {
            st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(1.0f));
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_send_dx_tokslot_from_pad_blockscaled(
    const __nv_bfloat16* __restrict__ dXe_pad,
    const int* __restrict__ dest,
    const int64_t* __restrict__ row_id,
    int M, int T, int H, int Ha, int K,
    int world, int M_pad_cap,
    size_t tok_y_off,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;

        const int pad_i = dest[i];
        if (pad_i < 0 || pad_i >= M_pad_cap) continue;
        const int64_t idx = (int64_t)tok * K + slot;
        char* dst_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
        uint16_t* tok_y = reinterpret_cast<uint16_t*>(dst_buf + tok_y_off);
        float* tok_gate = reinterpret_cast<float*>(dst_buf + tok_gate_off);

        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(dXe_pad + (int64_t)pad_i * H);
        uint16_t* dst_u16 = tok_y + idx * Ha;

        for (int h = lane * 8; h < H; h += 32 * 8) {
            int4 v = *reinterpret_cast<const int4*>(src_u16 + h);
            int4* d = reinterpret_cast<int4*>(dst_u16 + h);
            st_relaxed_sys_v4_s32(d, v);
        }
        if (lane == 0) {
            st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(1.0f));
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_reduce_tokslot_sum_bf16(
    const uint16_t* __restrict__ tok_y,
    const float* __restrict__ tok_tag,
    float* __restrict__ out,
    int T, int H, int Ha, int K)
{
    if (K <= 0 || K > 32) return;

    __shared__ float tag_shared[32];
    for (int tok = static_cast<int>(blockIdx.x); tok < T; tok += static_cast<int>(gridDim.x)) {
        if (threadIdx.x < K) {
            tag_shared[threadIdx.x] = ld_nc_f32(tok_tag + (int64_t)tok * K + threadIdx.x);
        }
        __syncthreads();

        int vec = static_cast<int>(threadIdx.x);
        for (int h0 = vec * 8; h0 < H; h0 += static_cast<int>(blockDim.x) * 8) {
            float acc[8] = {0};
            for (int slot = 0; slot < K; ++slot) {
                const float present = tag_shared[slot];
                if (present <= 0.0f) continue;
                const uint16_t* y_row = tok_y + (int64_t)(tok * K + slot) * Ha + h0;
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

            float* out_row = out + (int64_t)tok * H + h0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int hh = h0 + j;
                if (hh < H) out_row[j] = acc[j];
            }
        }
        __syncthreads();
    }
}

extern "C" void rdep_return_scatter(
    const void* Ye,
    void* out,
    int M_recv, int T, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    // Hybrid mode: use NVSHMEM for inter-node + IPC for intra-node
    // CRITICAL: Use g_nvshmem.ipc_buffer_ptrs (populated by nvshmem_open_ipc_handles_bf16),
    // NOT g_bf16.buffer_ptrs (which is NOT populated in hybrid mode)
    // Check hybrid mode FIRST before g_bf16.initialized (hybrid doesn't use g_bf16)
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP FATAL: NVSHMEM not initialized for hybrid mode\n");
            abort();
        }
        nvshmem::return_scatter_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(Ye),
            static_cast<float*>(out),
            M_recv, T, K,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
        return;
    }
#endif

    // Single-GPU and IPC modes require g_bf16 to be initialized
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: BF16 state not initialized for rdep_return_scatter\n");
        abort();
    }

    // Single-GPU and IPC modes: use local IPC path
    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    // IPC fast path: direct per-(tok,slot) writes + deterministic reduction.
    if (g_mode == MODE_IPC && g_bf16.world > 1) {
        const int H = g_bf16.H;
        const int Ha = g_bf16.Ha;
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }
        ipc_barrier_bf16(stream);
        maybe_zero_tokslot_buffers(local_buf, tok_y_off, tok_gate_off, tok_slots, Ha, false, true, stream);
        ipc_barrier_bf16(stream);

        const int warps_needed = M_recv;
        const int threads = 256;
        const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        if (M_recv > 0) {
            k_return_write_tokslot_bf16<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(Ye),
                g_bf16.order,
                meta_buf,
                M_recv, H, Ha, T, K,
                g_bf16.world, static_cast<int>(g_bf16.capacity),
                tok_y_off, tok_gate_off);
        }

        // Ensure all ranks finished writing tok-slot buffers before reduction.
        ipc_barrier_bf16(stream);

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int reduce_threads = 256;
        const int reduce_blocks_by_work = std::max(1, T);
        const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
        k_reduce_tokslot_gate_bf16<<<reduce_blocks, reduce_threads, 0, stream>>>(
            tok_y,
            tok_gate,
            static_cast<float*>(out),
            T, H, Ha, K);
        return;
    }

    // Legacy path (single-GPU / non-IPC): append+meta + atomic scatter.
    uint16_t* x_buf = reinterpret_cast<uint16_t*>(local_buf + x_off);
    cudaMemsetAsync(local_buf + counter_off, 0, sizeof(int), stream);

    // Snapshot dispatch metadata before return writes reuse the same IPC buffer.
    if (M_recv > 0) {
        cudaMemcpyAsync(g_bf16.meta_copy, meta_buf,
                        static_cast<size_t>(M_recv) * sizeof(Meta),
                        cudaMemcpyDeviceToDevice, stream);
    }

    ipc_barrier_bf16(stream);

    int threads = 256;
    int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    int capacity = static_cast<int>(g_bf16.capacity);

    if (M_recv > 0) {
        k_return_scatter_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(Ye), g_bf16.order, g_bf16.meta_copy,
            static_cast<float*>(out),
            M_recv, g_bf16.H, g_bf16.Ha, T, K,
            g_bf16.rank, g_bf16.world, capacity,
            meta_off, counter_off);
    }

    // Single-rank legacy mode has no remote return mailbox to drain.
    if (g_bf16.world == 1) {
        return;
    }

    ipc_barrier_bf16(stream);
    const int* m_ret_dev = reinterpret_cast<const int*>(local_buf + counter_off);
    const int threads_scatter = 256;
    const int warps_per_block = threads_scatter / 32;
    const int scatter_work = std::max(1, std::min(capacity, T * K));
    const int scatter_blocks_by_work = std::max(1, (scatter_work + warps_per_block - 1) / warps_per_block);
    const int blocks_scatter = cap_warp_stride_blocks(scatter_blocks_by_work);
    k_scatter_received_bf16_dynamic<<<blocks_scatter, threads_scatter, 0, stream>>>(
        x_buf, meta_buf, m_ret_dev,
        static_cast<float*>(out),
        capacity, g_bf16.H, g_bf16.Ha, T, K);
}

extern "C" void rdep_return_scatter_from_pad_bf16(
    const void* Ye_pad,
    void* out,
    int M_recv, int T, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::return_scatter_hybrid_bf16_from_pad(
            static_cast<const __nv_bfloat16*>(Ye_pad),
            static_cast<float*>(out),
            M_recv, T, K,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
        return;
    }
#endif

    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: BF16 state not initialized for rdep_return_scatter_from_pad_bf16\n");
        abort();
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);
    const int capacity = static_cast<int>(g_bf16.capacity);
    if (M_recv < 0 || M_recv > capacity) {
        fprintf(stderr,
                "RDEP FATAL: rdep_return_scatter_from_pad_bf16 requires 0 <= M_recv <= capacity (M_recv=%d capacity=%d)\n",
                M_recv, capacity);
        abort();
    }

    char* local_buf = static_cast<char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    // IPC fast path: tok-slot write + deterministic reduction.
    if (g_mode == MODE_IPC && g_bf16.world > 1) {
        const int H = g_bf16.H;
        const int Ha = g_bf16.Ha;
        const int M_pad_cap = static_cast<int>(g_bf16.capacity + g_bf16.n_local * (g_bf16.align - 1));
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }
        ipc_barrier_bf16(stream);
        maybe_zero_tokslot_buffers(local_buf, tok_y_off, tok_gate_off, tok_slots, Ha, false, true, stream);
        ipc_barrier_bf16(stream);

        const int threads = 256;
        const int warps_needed = std::max(M_recv, 1);
        const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        if (M_recv > 0) {
            k_return_write_tokslot_from_pad_bf16<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(Ye_pad),
                g_bf16.dest,
                g_bf16.order,
                meta_buf,
                M_recv, H, Ha, T, K,
                g_bf16.world, capacity, M_pad_cap,
                tok_y_off, tok_gate_off);
        }

        ipc_barrier_bf16(stream);

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int reduce_threads = 256;
        const int reduce_blocks_by_work = std::max(1, T);
        const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
        k_reduce_tokslot_gate_bf16<<<reduce_blocks, reduce_threads, 0, stream>>>(
            tok_y,
            tok_gate,
            static_cast<float*>(out),
            T, H, Ha, K);
        return;
    }

    // Single-GPU path: atomic scatter directly from Ye_pad[dest[sorted_i]].
    if (M_recv <= 0) return;
    const int threads = 256;
    const int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_return_scatter_from_pad_atomic<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Ye_pad),
        g_bf16.dest,
        g_bf16.order,
        meta_buf,
        static_cast<float*>(out),
        M_recv, g_bf16.H, T, K,
        capacity,
        static_cast<int>(g_bf16.capacity + g_bf16.n_local * (g_bf16.align - 1)));
}

extern "C" void rdep_return_scatter_from_pad_blockscaled(
    const void* Ye_pad,
    void* out,
    int M_recv, int T, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP FATAL: NVSHMEM not initialized for hybrid mode\n");
            abort();
        }
        nvshmem::return_scatter_hybrid_blockscaled_from_pad(
            static_cast<const __nv_bfloat16*>(Ye_pad),
            static_cast<float*>(out),
            M_recv, T, K,
            nvshmem::g_nvshmem.ipc_buffer_ptrs,
            nvshmem::g_nvshmem.ipc_barrier_signal_ptrs,
            stream);
        return;
    }
#endif

    if (!g_block.initialized) {
        fprintf(stderr, "RDEP FATAL: blockscaled state not initialized for rdep_return_scatter_from_pad_blockscaled\n");
        abort();
    }

    size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                               &x_off, &sfa_off, &y_off, &meta_off, &counter_off, &dropped_off,
                               &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                               &tok_y_off, &tok_gate_off,
                               &total_size);
    const int capacity = static_cast<int>(g_block.capacity);
    if (M_recv < 0 || M_recv > capacity) {
        fprintf(stderr,
                "RDEP FATAL: rdep_return_scatter_from_pad_blockscaled requires 0 <= M_recv <= capacity (M_recv=%d capacity=%d)\n",
                M_recv, capacity);
        abort();
    }

    char* local_buf = static_cast<char*>(g_block.buffer_ptrs[g_block.rank]);
    Meta* meta_buf = reinterpret_cast<Meta*>(local_buf + meta_off);

    // IPC fast path: tok-slot write + deterministic reduction.
    if (g_mode == MODE_IPC && g_block.world > 1) {
        const int H = g_block.H;
        const int Ha = g_block.Ha;
        const int M_pad_cap = static_cast<int>(g_block.capacity + g_block.n_local * (g_block.align - 1));
        const int tok_slots = T * K;
        const size_t tok_cap = g_block.capacity / static_cast<size_t>(g_block.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_block.capacity, g_block.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }
        ipc_barrier_block(stream);
        maybe_zero_tokslot_buffers(local_buf, tok_y_off, tok_gate_off, tok_slots, Ha, false, true, stream);
        ipc_barrier_block(stream);

        const int threads = 256;
        const int warps_needed = std::max(M_recv, 1);
        const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        if (M_recv > 0) {
            k_return_write_tokslot_from_pad_blockscaled<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(Ye_pad),
                g_block.dest,
                g_block.order,
                meta_buf,
                M_recv, H, Ha, T, K,
                g_block.world, capacity, M_pad_cap,
                tok_y_off, tok_gate_off);
        }

        ipc_barrier_block(stream);

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int reduce_threads = 256;
        const int reduce_blocks_by_work = std::max(1, T);
        const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
        k_reduce_tokslot_gate_bf16<<<reduce_blocks, reduce_threads, 0, stream>>>(
            tok_y,
            tok_gate,
            static_cast<float*>(out),
            T, H, Ha, K);
        return;
    }

    // Single-GPU path: atomic scatter directly from Ye_pad[dest[sorted_i]].
    if (M_recv <= 0) return;
    const int threads = 256;
    const int blocks_by_work = std::max(1, (M_recv * 32 + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_return_scatter_from_pad_atomic<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Ye_pad),
        g_block.dest,
        g_block.order,
        meta_buf,
        static_cast<float*>(out),
        M_recv, g_block.H, T, K,
        capacity,
        static_cast<int>(g_block.capacity + g_block.n_local * (g_block.align - 1)));
}

__global__ void k_gather_dy_bf16(
    const __nv_bfloat16* __restrict__ dY,          // [T, H]
    const __nv_bfloat16* __restrict__ Ye_pad,      // [M_pad, H]
    const int* __restrict__ dest,                  // [M] sorted_i -> pad_i
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate,                // [M]
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_out,                 // [M]
    int M, int T, int H, int K)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        const __nv_bfloat16* dy_row = dY + (int64_t)tok * H;
        const int pad_i = dest[i];
        const __nv_bfloat16* ye_row = Ye_pad + (int64_t)pad_i * H;
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        float g = gate[i];
        float dot = 0.0f;
        for (int h = lane; h < H; h += 32) {
            float dy = __bfloat162float(dy_row[h]);
            float ye = __bfloat162float(ye_row[h]);
            dot += ye * dy;
            dye_row[h] = __float2bfloat16(dy * g);
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            dGate_out[i] = dot;
        }
    }
}

__global__ void k_scatter_gate_bf16(
    const float* __restrict__ dGate_sorted,
    const int64_t* __restrict__ row_id,
    float* __restrict__ dGates_tk,
    int M, int T, int K)
{
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M;
         i += blockDim.x * gridDim.x) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        dGates_tk[tok * K + slot] = dGate_sorted[i];
    }
}

__global__ void k_scatter_gate_bf16_out_bf16(
    const float* __restrict__ dGate_sorted,
    const int64_t* __restrict__ row_id,
    __nv_bfloat16* __restrict__ dGates_tk,
    int M, int T, int K)
{
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M;
         i += blockDim.x * gridDim.x) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        dGates_tk[tok * K + slot] = __float2bfloat16(dGate_sorted[i]);
    }
}

// ============================================================================
// SonicMoE dGate Identity: dGate = ⟨A, dA'⟩ instead of ⟨dOut, Ye⟩
// ============================================================================
// This eliminates the need to recompute or store Ye in backward.
// A = SwiGLU output (post-activation), dA' = dOut @ W2.T (ungated gradient)

// Gather dY with gate scaling, but defer dGate computation to dgate_from_adA
// dYe[i] = dY[tok] * gate[i]  (standard chain rule for MoE combine)
// dGate is NOT computed here - use dgate_from_adA_bf16 after computing A and dA
__global__ void k_gather_dy_nogate_bf16(
    const __nv_bfloat16* __restrict__ dY,          // [T, H]
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate,                // [M] gate values for scaling
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    int M, int T, int H, int K)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        const __nv_bfloat16* dy_row = dY + (int64_t)tok * H;
        __nv_bfloat16* out_row = dYe_out + (int64_t)i * H;
        float g = gate[i];
        for (int h = lane; h < H; h += 32) {
            float dy = __bfloat162float(dy_row[h]);
            out_row[h] = __float2bfloat16(dy * g);
        }
    }
}

// Compute dGate via SonicMoE identity: dGate[i] = sum_d(A[pad_i, d] * dA[pad_i, d])
// This avoids needing Ye entirely - uses post-SwiGLU activations A and ungated gradient dA
__global__ void k_dgate_from_adA_bf16(
    const __nv_bfloat16* __restrict__ A_pad,       // [M_pad, D] post-SwiGLU activation
    const __nv_bfloat16* __restrict__ dA_pad,      // [M_pad, D] ungated gradient (dYe @ W2.T)
    const int* __restrict__ dest,                  // [M] sorted -> pad index
    float* __restrict__ dGate_sorted_out,          // [M]
    int M, int D)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int sorted_i = warp_id; sorted_i < M; sorted_i += num_warps) {
        const int pad_i = dest[sorted_i];
        const __nv_bfloat16* a_row = A_pad + (int64_t)pad_i * D;
        const __nv_bfloat16* da_row = dA_pad + (int64_t)pad_i * D;

        float dot = 0.0f;
        for (int j = lane; j < D; j += 32) {
            float a = __bfloat162float(a_row[j]);
            float da = __bfloat162float(da_row[j]);
            dot += a * da;
        }

        dot = warp_reduce_sum(dot);
        if (lane == 0) dGate_sorted_out[sorted_i] = dot;
    }
}

__global__ void k_scatter_dx_bf16(
    const __nv_bfloat16* __restrict__ dXe_pad,   // [M_pad, H]
    const int* __restrict__ dest,                // [M]
    const int64_t* __restrict__ row_id,           // [M]
    float* __restrict__ dX,                       // [T, H] float32 accum
    int M, int T, int H, int K)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        int pad_i = dest[i];
        const __nv_bfloat16* dxe_row = dXe_pad + (int64_t)pad_i * H;
        float* dx_row = dX + (int64_t)tok * H;
        for (int h = lane; h < H; h += 32) {
            atomicAdd(dx_row + h, __bfloat162float(dxe_row[h]));
        }
    }
}

extern "C" void rdep_gather_dy_bf16(
    const void* dY,
    const void* Ye_pad,
    const int64_t* row_id,
    const float* gate,
    void* dYe_out,
    float* dGate_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    const int* dest = nullptr;
    if (g_bf16.initialized) {
        dest = g_bf16.dest;
    } else if (g_block.initialized) {
        dest = g_block.dest;
    } else {
        return;
    }
    if (M <= 0 || T <= 0 || H <= 0 || K <= 0) return;
    int threads = 256;
    int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_gather_dy_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dY),
        static_cast<const __nv_bfloat16*>(Ye_pad),
        dest,
        row_id,
        gate,
        static_cast<__nv_bfloat16*>(dYe_out),
        dGate_out,
        M, T, H, K);
}

extern "C" void rdep_scatter_gate_bf16(
    const float* dGate_sorted,
    const int64_t* row_id,
    float* dGates_tk,
    int M, int T, int K,
    cudaStream_t stream)
{
    if (M <= 0 || T <= 0 || K <= 0) return;
    int threads = 256;
    int blocks_by_work = std::max(1, (M + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_scatter_gate_bf16<<<blocks, threads, 0, stream>>>(
        dGate_sorted, row_id, dGates_tk, M, T, K);
}

extern "C" void rdep_scatter_gate_bf16_out_bf16(
    const float* dGate_sorted,
    const int64_t* row_id,
    void* dGates_tk_bf16,
    int M, int T, int K,
    cudaStream_t stream)
{
    if (M <= 0 || T <= 0 || K <= 0) return;
    int threads = 256;
    int blocks_by_work = std::max(1, (M + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_scatter_gate_bf16_out_bf16<<<blocks, threads, 0, stream>>>(
        dGate_sorted, row_id, static_cast<__nv_bfloat16*>(dGates_tk_bf16), M, T, K);
}

// ============================================================================
// SonicMoE dGate Host Wrappers
// ============================================================================

extern "C" void rdep_gather_dy_nogate_bf16(
    const void* dY,
    const int64_t* row_id,
    const float* gate,
    void* dYe_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (M <= 0 || H <= 0) return;
    int threads = 256;
    int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_gather_dy_nogate_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dY),
        row_id,
        gate,
        static_cast<__nv_bfloat16*>(dYe_out),
        M, T, H, K);
}

extern "C" void rdep_dgate_from_adA_bf16(
    const void* A_pad,
    const void* dA_pad,
    float* dGate_sorted_out,
    int M, int D,
    cudaStream_t stream)
{
    if (M <= 0 || D <= 0) return;

#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        if (!nvshmem::g_nvshmem.initialized) {
            fprintf(stderr, "RDEP FATAL: NVSHMEM not initialized for hybrid dGate path\n");
            abort();
        }
        int threads = 256;
        int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_dgate_from_adA_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(A_pad),
            static_cast<const __nv_bfloat16*>(dA_pad),
            nvshmem::g_nvshmem.dest,
            dGate_sorted_out,
            M, D);
        return;
    }
#endif
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: BF16 state not initialized for dGate path\n");
        abort();
    }

    int threads = 256;
    int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_dgate_from_adA_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(A_pad),
        static_cast<const __nv_bfloat16*>(dA_pad),
        g_bf16.dest,
        dGate_sorted_out,
        M, D);
}

extern "C" void rdep_scatter_dx_bf16(
    const void* dXe_pad,
    const int* dest,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (M <= 0 || T <= 0 || H <= 0 || K <= 0) return;
    int threads = 256;
    int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_scatter_dx_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dXe_pad),
        dest,
        row_id,
        static_cast<float*>(dX_out),
        M, T, H, K);
}

extern "C" void rdep_scatter_dx_bf16_internal(
    const void* dXe_pad,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
    if (!g_bf16.initialized) return;
    if (M <= 0 || T <= 0 || H <= 0 || K <= 0) return;
    int threads = 256;
    int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
    int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_scatter_dx_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dXe_pad),
        g_bf16.dest,
        row_id,
        static_cast<float*>(dX_out),
        M, T, H, K);
}

// ============================================================================
// IPC Backward Helpers (single-node, BF16)
//
// These helpers implement the "flipped" RDEP backward using IPC memory only.
// They intentionally avoid relying on global mutable forward metadata across
// layers by using the materialized per-dispatch row_id tensors.
// ============================================================================

__global__ void k_ipc_barrier_phase_bf16(int phase) {
    // One-CTA IPC barrier using per-peer phase slots (DeepEP-style direction):
    // - Each rank writes `phase` into each peer's signal[my_rank] (sys-scope release store).
    // - Each rank waits on its local signal[peer] (sys-scope acquire load).
    //
    // This avoids remote polling traffic: waiting spins only on local memory.
    int tid = threadIdx.x;
    int world = d_world_bf16;
    int my_rank = d_my_rank_bf16;

    // Fence to ensure all prior writes (from previous kernels) are sys-visible
    // before we signal completion. Without this, the release store only orders
    // writes within THIS kernel, not writes from previous dispatch kernels.
    fence_acq_rel_sys();
    if (blockDim.x <= 32) __syncwarp();
    else __syncthreads();

    if (tid < world) {
        int* peer_sig = d_barrier_signal_ptrs_bf16[tid] + my_rank;
        st_release_sys_s32(peer_sig, phase);
    }
    if (blockDim.x <= 32) __syncwarp();
    else __syncthreads();

    uint64_t start_time = clock64();
    if (tid < world) {
        const int* local_sig = d_barrier_signal_ptrs_bf16[my_rank] + tid;
        uint32_t spins = 0;
        while (ld_acquire_sys_s32(local_sig) < phase) {
            if (((++spins) & 0x3Fu) == 0u) {
                __nanosleep(64);
            }
            if (clock64() - start_time > TIMEOUT_CYCLES) {
                printf("nmoe phase barrier timeout: rank=%d wait_rank=%d phase=%d\n",
                       my_rank, tid, phase);
                trap();
            }
        }
    }
    if (blockDim.x <= 32) __syncwarp();
    else __syncthreads();
}

__global__ void k_ipc_barrier_phase_block(int phase) {
    int tid = threadIdx.x;
    int world = d_world_block;
    int my_rank = d_my_rank_block;

    // Fence to ensure all prior writes (from previous kernels) are sys-visible
    fence_acq_rel_sys();
    if (blockDim.x <= 32) __syncwarp();
    else __syncthreads();

    if (tid < world) {
        int* peer_sig = d_barrier_signal_ptrs_block[tid] + my_rank;
        st_release_sys_s32(peer_sig, phase);
    }
    if (blockDim.x <= 32) __syncwarp();
    else __syncthreads();

    uint64_t start_time = clock64();
    if (tid < world) {
        const int* local_sig = d_barrier_signal_ptrs_block[my_rank] + tid;
        uint32_t spins = 0;
        while (ld_acquire_sys_s32(local_sig) < phase) {
            if (((++spins) & 0x3Fu) == 0u) {
                __nanosleep(64);
            }
            if (clock64() - start_time > TIMEOUT_CYCLES) {
                printf("nmoe phase barrier timeout (blockscaled): rank=%d wait_rank=%d phase=%d\n",
                       my_rank, tid, phase);
                trap();
            }
        }
    }
    if (blockDim.x <= 32) __syncwarp();
    else __syncthreads();
}

__global__ void k_stage_dy_to_xbuf_bf16(
    const __nv_bfloat16* __restrict__ dY,
    uint16_t* __restrict__ x_buf,
    int T, int H, int Ha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * H) {
        int t = i / H;
        int h = i - t * H;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(dY + (int64_t)t * H);
        x_buf[(int64_t)t * Ha + h] = src[h];
    }
}

// ============================================================================
// IPC Backward (push staging) - BF16 payload
//
// Contract:
//  - Stage dY by row_id into destination buffers (push, no remote reads).
//  - Compute local dYe/dGate from staged dY and Ye_sorted.
//  - Return dGate via fixed (tok,slot) writes into src-rank tok_gate buffer.
//
// This path is used for BF16 *and* blockscaled profiles (STE backward), with
// the active runtime state selecting the underlying buffer layout.
// ============================================================================

__global__ void k_push_stage_dy_ipc_bf16(
    const __nv_bfloat16* __restrict__ dY,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    int T, int H, int Ha, int K,
    int n_local, int capacity,
    size_t x_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    const int my_rank = d_my_rank_bf16;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const int world = d_world_bf16;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);
        int slot = i - tok * K;
        int eid = eids[i];
        if (eid < 0) continue;
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;

        const int64_t rid = encode_rid(my_rank, tok, slot, T, K);
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        char* dest_buf = static_cast<char*>(d_buffer_ptrs_bf16[dest]);
        uint16_t* stage = reinterpret_cast<uint16_t*>(dest_buf + x_off);
        uint16_t* dst = stage + rid * Ha;
        const __nv_bfloat16* row = dY + (int64_t)tok * H;

        const bool is_remote = (dest != my_rank);
        if (is_remote) {
            for (int h = lane * 8; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    st_relaxed_sys_v4_s32(d, v);  // sys-scope for cross-GPU
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        const uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                        st_relaxed_sys_b16(dst + hh, u);  // sys-scope for cross-GPU
                    }
                }
            }
        } else {
            for (int h = lane * 8; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    *d = v;
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        const uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                        dst[hh] = u;
                    }
                }
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_push_stage_dy_ipc_blockscaled(
    const __nv_bfloat16* __restrict__ dY,   // [T, H]
    const int* __restrict__ eids,           // [T, K]
    int T, int H, int K,
    int n_local, int capacity,
    size_t y_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    int M = T * K;
    const int my_rank = d_my_rank_block;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool n_local_pow2 = nmoe_is_pow2(n_local);
    const int n_local_shift = n_local_pow2 ? (__ffs(n_local) - 1) : 0;
    const int world = d_world_block;

    for (int i = warp_id; i < M; i += num_warps) {
        int tok = k_pow2 ? (i >> k_shift) : (i / K);
        int slot = i - tok * K;
        int eid = eids[i];
        if (eid < 0) continue;
        int dest = nmoe_expert_dest_fast(eid, n_local, n_local_pow2, n_local_shift);
        if (static_cast<unsigned>(dest) >= static_cast<unsigned>(world)) continue;

        const int64_t rid = encode_rid(my_rank, tok, slot, T, K);
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        char* dest_buf = static_cast<char*>(d_buffer_ptrs_block[dest]);
        uint16_t* stage = reinterpret_cast<uint16_t*>(dest_buf + y_off);
        uint16_t* dst = stage + rid * static_cast<int64_t>(H);
        const __nv_bfloat16* row = dY + (int64_t)tok * H;

        const bool is_remote = (dest != my_rank);
        if (is_remote) {
            for (int h = lane * 8; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    st_relaxed_sys_v4_s32(d, v);  // sys-scope for cross-GPU
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        const uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                        st_relaxed_sys_b16(dst + hh, u);  // sys-scope for cross-GPU
                    }
                }
            }
        } else {
            for (int h = lane * 8; h < H; h += 32 * 8) {
                if (h + 8 <= H) {
                    int4 v = *reinterpret_cast<const int4*>(row + h);
                    int4* d = reinterpret_cast<int4*>(dst + h);
                    *d = v;
                } else {
                    for (int hh = h; hh < H && hh < h + 8; hh++) {
                        const uint16_t u = reinterpret_cast<const uint16_t*>(row)[hh];
                        dst[hh] = u;
                    }
                }
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_gather_dy_from_stage_and_send_gate_ipc_bf16(
    const __nv_bfloat16* __restrict__ Ye_pad,      // [M_pad, H]
    const int* __restrict__ dest,                  // [M] sorted_i -> pad_i
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage,            // [capacity, Ha]
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_sorted_out,          // [M]
    int M, int T, int H, int Ha, int K,
    int capacity,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const int my_rank = d_my_rank_bf16;
    const int world = d_world_bf16;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const uint16_t* dy_u16 = stage + rid * Ha;
        const int pad_i = dest[i];
        const uint16_t* ye_u16 = reinterpret_cast<const uint16_t*>(Ye_pad + (int64_t)pad_i * H);
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        const float g = gate_sorted[i];
        float dot = 0.0f;

        for (int h = lane * 8; h < Ha; h += 32 * 8) {
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

            // Return dGate to source rank via fixed tok-slot write.
            const int64_t tmp = k_pow2 ? (rid >> k_shift) : (rid / K);
            int slot = static_cast<int>(rid - tmp * K);
            int src_rank = static_cast<int>(t_pow2 ? (tmp >> t_shift) : (tmp / T));
            int tok = static_cast<int>(tmp - static_cast<int64_t>(src_rank) * T);
            if (src_rank < 0 || src_rank >= world) continue;
            if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;
            const int64_t idx = (int64_t)tok * K + slot;
            if (idx < 0 || idx >= static_cast<int64_t>(capacity)) continue;

            char* src_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
            float* tok_gate = reinterpret_cast<float*>(src_buf + tok_gate_off);
            if (src_rank == my_rank) {
                tok_gate[idx] = dot;
            } else {
                // Sys-scope for cross-GPU visibility
                st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(dot));
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_gather_dy_from_stage_and_send_gate_ipc_blockscaled(
    const __nv_bfloat16* __restrict__ Ye_pad,      // [M_pad, H]
    const int* __restrict__ dest,                  // [M] sorted_i -> pad_i
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage,            // [capacity, H]
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    float* __restrict__ dGate_sorted_out,          // [M]
    int M, int T, int H, int K,
    int capacity,
    size_t tok_gate_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const int my_rank = d_my_rank_block;
    const int world = d_world_block;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const uint16_t* dy_u16 = stage + rid * static_cast<int64_t>(H);
        const int pad_i = dest[i];
        const uint16_t* ye_u16 = reinterpret_cast<const uint16_t*>(Ye_pad + (int64_t)pad_i * H);
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        const float g = gate_sorted[i];
        float dot = 0.0f;

        for (int h = lane * 8; h < H; h += 32 * 8) {
            union U16x8 { int4 v; uint16_t u[8]; };
            U16x8 dy8;
            U16x8 ye8;
            const bool full = (h + 8 <= H);
            if (full) {
                dy8.v = ld_nc_v4_s32(reinterpret_cast<const int4*>(dy_u16 + h));
                ye8.v = *reinterpret_cast<const int4*>(ye_u16 + h);
            } else {
                // Tail: load scalar BF16 values.
                for (int j = 0; j < 8; j++) {
                    int hh = h + j;
                    if (hh < H) {
                        dy8.u[j] = reinterpret_cast<const uint16_t*>(dy_u16)[hh];
                        ye8.u[j] = reinterpret_cast<const uint16_t*>(ye_u16)[hh];
                    } else {
                        dy8.u[j] = 0;
                        ye8.u[j] = 0;
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh >= H) break;
                const __nv_bfloat16 dy_bf = *reinterpret_cast<const __nv_bfloat16*>(&dy8.u[j]);
                const __nv_bfloat16 ye_bf = *reinterpret_cast<const __nv_bfloat16*>(&ye8.u[j]);
                float dy = __bfloat162float(dy_bf);
                float ye = __bfloat162float(ye_bf);
                dot += ye * dy;
                dye_row[hh] = __float2bfloat16(dy * g);
            }
        }

        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            dGate_sorted_out[i] = dot;

            const int64_t tmp = k_pow2 ? (rid >> k_shift) : (rid / K);
            int slot = static_cast<int>(rid - tmp * K);
            int src_rank = static_cast<int>(t_pow2 ? (tmp >> t_shift) : (tmp / T));
            int tok = static_cast<int>(tmp - static_cast<int64_t>(src_rank) * T);
            if (src_rank < 0 || src_rank >= world) continue;
            if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;
            const int64_t idx = (int64_t)tok * K + slot;
            if (idx < 0 || idx >= static_cast<int64_t>(capacity)) continue;

            char* src_buf = static_cast<char*>(d_buffer_ptrs_block[src_rank]);
            float* tok_gate = reinterpret_cast<float*>(src_buf + tok_gate_off);
            if (src_rank == my_rank) {
                tok_gate[idx] = dot;
            } else {
                // Sys-scope for cross-GPU visibility
                st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(dot));
            }
        }
    }

    fence_acq_rel_sys();
}

__global__ void k_collect_tok_gate_ipc(
    const float* __restrict__ tok_gate,
    float* __restrict__ dGates_tk_out,
    int tok_slots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < tok_slots; i += stride) {
        // tok_gate may be written by peer GPUs; use non-caching loads.
        dGates_tk_out[i] = ld_nc_f32(tok_gate + i);
    }
}

__global__ void k_collect_tok_gate_ipc_bf16(
    const float* __restrict__ tok_gate,
    __nv_bfloat16* __restrict__ dGates_tk_out,
    int tok_slots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < tok_slots; i += stride) {
        dGates_tk_out[i] = __float2bfloat16(ld_nc_f32(tok_gate + i));
    }
}

// ============================================================================
// SonicMoE Distributed dGate: gather dYe without dGate, then send dGate later
// ============================================================================

// Gather dYe from IPC stage buffer with gate scaling, but do NOT compute dGate
// (dGate will be computed via ⟨A, dA⟩ identity after expert backward)
__global__ void k_gather_dy_from_stage_nogate_ipc_bf16(
    const int* __restrict__ dest,                  // [M] sorted_i -> pad_i
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ gate_sorted,         // [M]
    const uint16_t* __restrict__ stage,            // [capacity, Ha]
    __nv_bfloat16* __restrict__ dYe_out,           // [M, H]
    int M, int T, int H, int Ha, int K,
    int capacity)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    for (int i = warp_id; i < M; i += num_warps) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const uint16_t* dy_u16 = stage + rid * Ha;
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        const float g = gate_sorted[i];

        for (int h = lane * 8; h < Ha; h += 32 * 8) {
            int4 dy_v = ld_nc_v4_s32(reinterpret_cast<const int4*>(dy_u16 + h));
            union U16x8 { int4 v; uint16_t u[8]; };
            U16x8 dy8; dy8.v = dy_v;

#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hh = h + j;
                if (hh >= H) break;
                const __nv_bfloat16 dy_bf = *reinterpret_cast<const __nv_bfloat16*>(&dy8.u[j]);
                float dy = __bfloat162float(dy_bf);
                dye_row[hh] = __float2bfloat16(dy * g);
            }
        }
    }
}

// Send locally-computed dGate back to source ranks via tok-slot IPC buffer
// Called after computing dGate = ⟨A, dA⟩ locally
__global__ void k_send_dgate_ipc_bf16(
    const int64_t* __restrict__ row_id,            // [M]
    const float* __restrict__ dGate_sorted,        // [M]
    int M, int T, int K,
    int capacity,
    size_t tok_gate_off)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const int my_rank = d_my_rank_bf16;
    const int world = d_world_bf16;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = tid; i < M; i += stride) {
        const int64_t rid = row_id[i];
        if (rid < 0 || rid >= static_cast<int64_t>(capacity)) continue;

        const float dg = dGate_sorted[i];

        // Decode source rank and token slot from row_id
        const int64_t tmp = k_pow2 ? (rid >> k_shift) : (rid / K);
        int slot = static_cast<int>(rid - tmp * K);
        int src_rank = static_cast<int>(t_pow2 ? (tmp >> t_shift) : (tmp / T));
        int tok = static_cast<int>(tmp - static_cast<int64_t>(src_rank) * T);
        if (src_rank < 0 || src_rank >= world) continue;
        if (tok < 0 || tok >= T || slot < 0 || slot >= K) continue;
        const int64_t idx = (int64_t)tok * K + slot;

        if (idx < 0 || idx >= static_cast<int64_t>(capacity)) continue;

        char* src_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        float* tok_gate = reinterpret_cast<float*>(src_buf + tok_gate_off);

        if (src_rank == my_rank) {
            tok_gate[idx] = dg;
        } else {
            // Sys-scope for cross-GPU visibility
            st_relaxed_sys_s32(reinterpret_cast<int*>(tok_gate + idx), __float_as_int(dg));
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_gather_dy_remote_ipc_bf16(
    const __nv_bfloat16* __restrict__ Ye,
    const int64_t* __restrict__ row_id,
    const float* __restrict__ gate,
    __nv_bfloat16* __restrict__ dYe_out,
    float* __restrict__ dGate_out,
    int M, int T, int H, int K, int Ha)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
        const char* src_buf = static_cast<const char*>(d_buffer_ptrs_bf16[src_rank]);
        const uint16_t* src_x = reinterpret_cast<const uint16_t*>(src_buf);
        const __nv_bfloat16* dy_row = reinterpret_cast<const __nv_bfloat16*>(src_x + (int64_t)tok * Ha);

        const __nv_bfloat16* ye_row = Ye + (int64_t)i * H;
        __nv_bfloat16* dye_row = dYe_out + (int64_t)i * H;

        float g = gate[i];
        float dot = 0.0f;
        for (int h = lane; h < H; h += 32) {
            float dy = __bfloat162float(dy_row[h]);
            float ye = __bfloat162float(ye_row[h]);
            dot += ye * dy;
            dye_row[h] = __float2bfloat16(dy * g);
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) dGate_out[i] = dot;
    }
}

__global__ void k_send_dgate_ipc_bf16(
    const int64_t* __restrict__ row_id,
    const float* __restrict__ dGate_sorted,
    float* __restrict__ dGates_tk_local,
    int M, int T, int K,
    int capacity,
    size_t meta_off, size_t counter_off)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;
    for (; i < M; i += stride) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        if (src_rank == d_my_rank_bf16) {
            atomicAdd(dGates_tk_local + tok * K + slot, dGate_sorted[i]);
            continue;
        }

        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        Meta* meta_buf = reinterpret_cast<Meta*>(dst_buf + meta_off);
        int* counter = reinterpret_cast<int*>(dst_buf + counter_off);

        int slot_r = atomicAdd(counter, 1);
        if (slot_r < capacity) {
            meta_buf[slot_r] = Meta{row_id[i], 0, dGate_sorted[i]};
        }
    }
    fence_acq_rel_sys();
}

__global__ void k_scatter_received_gate_ipc_bf16(
    const Meta* __restrict__ meta_recv,
    float* __restrict__ dGates_tk,
    int M_ret, int T, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;
    if (i >= M_ret) return;
    const Meta& m = meta_recv[i];
    int src_rank, tok, slot;
    decode_rid_fast(m.row_id, T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);
    atomicAdd(dGates_tk + tok * K + slot, m.gate);
}

__global__ void k_send_dx_ipc_bf16(
    const __nv_bfloat16* __restrict__ dXe_sorted,
    const int64_t* __restrict__ row_id,
    float* __restrict__ dX_local,
    int M, int T, int H, int K, int Ha,
    int capacity,
    size_t meta_off, size_t counter_off)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;
    const bool k_pow2 = nmoe_is_pow2(K);
    const int k_shift = k_pow2 ? nmoe_pow2_shift(K) : 0;
    const bool t_pow2 = nmoe_is_pow2(T);
    const int t_shift = t_pow2 ? nmoe_pow2_shift(T) : 0;

    for (int i = warp_id; i < M; i += num_warps) {
        int src_rank, tok, slot;
        decode_rid_fast(row_id[i], T, K, &src_rank, &tok, &slot, k_pow2, k_shift, t_pow2, t_shift);

        const __nv_bfloat16* src_row = dXe_sorted + (int64_t)i * H;

        if (src_rank == d_my_rank_bf16) {
            float* out_row = dX_local + (int64_t)tok * H;
            for (int h = lane; h < H; h += 32) atomicAdd(out_row + h, __bfloat162float(src_row[h]));
            continue;
        }

        char* dst_buf = static_cast<char*>(d_buffer_ptrs_bf16[src_rank]);
        uint16_t* y_buf = reinterpret_cast<uint16_t*>(dst_buf);  // x_off = 0
        Meta* meta_buf = reinterpret_cast<Meta*>(dst_buf + meta_off);
        int* counter = reinterpret_cast<int*>(dst_buf + counter_off);

        int slot_r;
        if (lane == 0) slot_r = atomicAdd(counter, 1);
        slot_r = __shfl_sync(0xFFFFFFFF, slot_r, 0);
        if (slot_r >= capacity) continue;

        if (lane == 0) meta_buf[slot_r] = Meta{row_id[i], 0, 1.0f};

        uint16_t* dst = y_buf + (int64_t)slot_r * Ha;
        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(src_row);
        for (int h = lane; h < H; h += 32) dst[h] = src_u16[h];
    }
}



extern "C" void rdep_gather_dy_dist_bf16(
    const void* dY_local,
    const int* eids,
    const void* Ye_pad,
    const int64_t* row_id,
    const float* gate_sorted,
    void* dYe_out,
    float* dGate_sorted_out,
    float* dGates_tk_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::gather_dy_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(dY_local),
            eids,
            static_cast<const __nv_bfloat16*>(Ye_pad),
            row_id,
            gate_sorted,
            static_cast<__nv_bfloat16*>(dYe_out),
            dGate_sorted_out,
            dGates_tk_out,
            M, T, H, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) {
        fprintf(stderr, "RDEP FATAL: rdep_gather_dy_dist_bf16 requires MODE_IPC or MODE_HYBRID\n");
        abort();
    }

    // When blockscaled is active both states may be initialized; prefer the
    // blockscaled branch so barrier/layout domains match dispatch_meta_blockscaled.
    if (g_bf16.initialized && !(g_block.initialized && (g_block.profile == 0 || g_block.profile == 1))) {
        if (g_bf16.world <= 1) return;
        if (g_bf16.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
            abort();
        }
        size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                            &x_off, &meta_off, &counter_off, &dropped_off,
                            &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                            &tok_y_off, &tok_gate_off,
                            &total_size);
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }
        const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
        ipc_barrier_bf16_site(stream, "gather_dy_dist_bf16/pre_zero");
        maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, g_bf16.Ha, false, true, stream);
        ipc_barrier_bf16_site(stream, "gather_dy_dist_bf16/post_zero");

        const int threads = 256;
        const int warps_needed = std::max(1, tok_slots);
        const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_push_stage_dy_ipc_bf16<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_local),
            eids,
            T, H, g_bf16.Ha, K,
            g_bf16.n_local, static_cast<int>(g_bf16.capacity),
            x_off);

        ipc_barrier_bf16_site(stream, "gather_dy_dist_bf16/post_stage_push");

        const uint16_t* stage = reinterpret_cast<const uint16_t*>(local_buf + x_off);
        const int g_threads = 256;
        const int g_blocks_by_work = std::max(1, (M * 32 + g_threads - 1) / g_threads);
        const int g_blocks = cap_warp_stride_blocks(g_blocks_by_work);
        if (M > 0) {
            k_gather_dy_from_stage_and_send_gate_ipc_bf16<<<g_blocks, g_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(Ye_pad),
                g_bf16.dest,
                row_id,
                gate_sorted,
                stage,
                static_cast<__nv_bfloat16*>(dYe_out),
                dGate_sorted_out,
                M, T, H, g_bf16.Ha, K,
                static_cast<int>(g_bf16.capacity),
                tok_gate_off);
        }

        ipc_barrier_bf16_site(stream, "gather_dy_dist_bf16/post_gather");

        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int t_threads = 256;
        const int t_blocks_by_work = std::max(1, (tok_slots + t_threads - 1) / t_threads);
        const int t_blocks = cap_warp_stride_blocks(t_blocks_by_work);
        k_collect_tok_gate_ipc<<<t_blocks, t_threads, 0, stream>>>(
            tok_gate,
            dGates_tk_out,
            tok_slots);
        return;
    }

    if (g_block.initialized) {
        if (g_block.world <= 1) return;
        if (g_block.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_block.world, MAX_RANKS);
            abort();
        }
        size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                                   &x_off, &sfa_off, &y_off,
                                   &meta_off, &counter_off, &dropped_off,
                                   &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                                   &tok_y_off, &tok_gate_off,
                                   &total_size);
        const int tok_slots = T * K;
        const size_t tok_cap = g_block.capacity / static_cast<size_t>(g_block.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_block.capacity, g_block.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }
        const char* local_buf = static_cast<const char*>(g_block.buffer_ptrs[g_block.rank]);
        ipc_barrier_block_site(stream, "gather_dy_dist_blockscaled/pre_zero");
        maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, g_block.Ha, false, true, stream);
        ipc_barrier_block_site(stream, "gather_dy_dist_blockscaled/post_zero");

        const int threads = 256;
        const int warps_needed = std::max(1, tok_slots);
        const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
        const int blocks = cap_warp_stride_blocks(blocks_by_work);
        k_push_stage_dy_ipc_blockscaled<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dY_local),
            eids,
            T, H, K,
            g_block.n_local, static_cast<int>(g_block.capacity),
            y_off);

        ipc_barrier_block_site(stream, "gather_dy_dist_blockscaled/post_stage_push");

        const uint16_t* stage = reinterpret_cast<const uint16_t*>(local_buf + y_off);
        const int g_threads = 256;
        const int g_blocks_by_work = std::max(1, (M * 32 + g_threads - 1) / g_threads);
        const int g_blocks = cap_warp_stride_blocks(g_blocks_by_work);
        if (M > 0) {
            k_gather_dy_from_stage_and_send_gate_ipc_blockscaled<<<g_blocks, g_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(Ye_pad),
                g_block.dest,
                row_id,
                gate_sorted,
                stage,
                static_cast<__nv_bfloat16*>(dYe_out),
                dGate_sorted_out,
                M, T, H, K,
                static_cast<int>(g_block.capacity),
                tok_gate_off);
        }

        ipc_barrier_block_site(stream, "gather_dy_dist_blockscaled/post_gather");

        const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int t_threads = 256;
        const int t_blocks_by_work = std::max(1, (tok_slots + t_threads - 1) / t_threads);
        const int t_blocks = cap_warp_stride_blocks(t_blocks_by_work);
        k_collect_tok_gate_ipc<<<t_blocks, t_threads, 0, stream>>>(
            tok_gate,
            dGates_tk_out,
            tok_slots);
        return;
    }

    fprintf(stderr, "RDEP FATAL: rdep_gather_dy_dist_bf16 requires initialized IPC/blockscaled state\n");
    abort();
}

// ============================================================================
// SonicMoE Distributed dGate Host Wrappers
// ============================================================================

// Gather dYe from remote ranks with gate scaling, but do NOT compute dGate
// (Use with send_dgate_dist_bf16 after computing dGate = ⟨A, dA⟩)
extern "C" void rdep_gather_dy_nogate_dist_bf16(
    const void* dY_local,
    const int* eids,
    const int64_t* row_id,
    const float* gate_sorted,
    void* dYe_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::gather_dy_nogate_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(dY_local),
            eids,
            row_id,
            gate_sorted,
            static_cast<__nv_bfloat16*>(dYe_out),
            M, T, H, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) {
        fprintf(stderr, "RDEP FATAL: rdep_gather_dy_nogate_dist_bf16 requires MODE_IPC or MODE_HYBRID\n");
        abort();
    }
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: rdep_gather_dy_nogate_dist_bf16 requires initialized BF16 IPC state\n");
        abort();
    }
    if (g_bf16.world <= 1) {
        // Single-GPU: just use the local nogate path
        rdep_gather_dy_nogate_bf16(dY_local, row_id, gate_sorted, dYe_out, M, T, H, K, stream);
        return;
    }
    if (g_bf16.world > MAX_RANKS) {
        fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
        abort();
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    const int tok_slots = T * K;
    const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu\n",
                tok_slots, tok_cap);
        abort();
    }

    // Step 1: Push dY to remote ranks
    const int threads = 256;
    const int warps_needed = std::max(1, tok_slots);
    const int blocks_by_work = std::max(1, (warps_needed * 32 + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    k_push_stage_dy_ipc_bf16<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dY_local),
        eids,
        T, H, g_bf16.Ha, K,
        g_bf16.n_local, static_cast<int>(g_bf16.capacity),
        x_off);

    ipc_barrier_bf16_site(stream, "gather_dy_nogate_dist_bf16/post_stage_push");

    // Step 2: Gather dYe with gate scaling (no dGate computation)
    const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    const uint16_t* stage = reinterpret_cast<const uint16_t*>(local_buf + x_off);
    if (M > 0) {
        const int g_threads = 256;
        const int g_blocks_by_work = std::max(1, (M * 32 + g_threads - 1) / g_threads);
        const int g_blocks = cap_warp_stride_blocks(g_blocks_by_work);
        k_gather_dy_from_stage_nogate_ipc_bf16<<<g_blocks, g_threads, 0, stream>>>(
            g_bf16.dest,
            row_id,
            gate_sorted,
            stage,
            static_cast<__nv_bfloat16*>(dYe_out),
            M, T, H, g_bf16.Ha, K,
            static_cast<int>(g_bf16.capacity));
    }
}

// Send locally-computed dGate back to source ranks and collect into dGates_tk
// Called after computing dGate = ⟨A, dA⟩ locally
extern "C" void rdep_send_dgate_dist_bf16(
    const int64_t* row_id,
    const float* dGate_sorted,
    float* dGates_tk_out,
    int M, int T, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::send_dgate_hybrid_bf16(
            row_id,
            dGate_sorted,
            dGates_tk_out,
            M, T, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) {
        fprintf(stderr, "RDEP FATAL: rdep_send_dgate_dist_bf16 requires MODE_IPC or MODE_HYBRID\n");
        abort();
    }
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: rdep_send_dgate_dist_bf16 requires initialized BF16 IPC state\n");
        abort();
    }
    if (g_bf16.world <= 1) {
        // Single-GPU: just scatter directly
        rdep_scatter_gate_bf16(dGate_sorted, row_id, dGates_tk_out, M, T, K, stream);
        return;
    }
    if (g_bf16.world > MAX_RANKS) {
        fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
        abort();
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    const int tok_slots = T * K;
    const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
        abort();
    }
    const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    ipc_barrier_bf16_site(stream, "send_dgate_dist_bf16/pre_zero");
    maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, g_bf16.Ha, false, true, stream);
    ipc_barrier_bf16_site(stream, "send_dgate_dist_bf16/post_zero");

    // Step 1: Send dGate to source ranks via tok-slot IPC buffer
    const int threads = 256;
    const int blocks_by_work = std::max(1, (M + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    if (M > 0) {
        k_send_dgate_ipc_bf16<<<blocks, threads, 0, stream>>>(
            row_id,
            dGate_sorted,
            M, T, K,
            static_cast<int>(g_bf16.capacity),
            tok_gate_off);
    }

    ipc_barrier_bf16_site(stream, "send_dgate_dist_bf16/post_send");

    // Step 2: Collect dGate from tok-slot buffer
    const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
    const int t_threads = 256;
    const int t_blocks_by_work = std::max(1, (tok_slots + t_threads - 1) / t_threads);
    const int t_blocks = cap_warp_stride_blocks(t_blocks_by_work);
    k_collect_tok_gate_ipc<<<t_blocks, t_threads, 0, stream>>>(
        tok_gate,
        dGates_tk_out,
        tok_slots);
}

extern "C" void rdep_send_dgate_dist_bf16_out_bf16(
    const int64_t* row_id,
    const float* dGate_sorted,
    void* dGates_tk_out_bf16,
    int M, int T, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::send_dgate_hybrid_bf16_out_bf16(
            row_id,
            dGate_sorted,
            static_cast<__nv_bfloat16*>(dGates_tk_out_bf16),
            M, T, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) {
        fprintf(stderr, "RDEP FATAL: rdep_send_dgate_dist_bf16_out_bf16 requires MODE_IPC or MODE_HYBRID\n");
        abort();
    }
    if (!g_bf16.initialized) {
        fprintf(stderr, "RDEP FATAL: rdep_send_dgate_dist_bf16_out_bf16 requires initialized BF16 IPC state\n");
        abort();
    }
    if (g_bf16.world <= 1) {
        rdep_scatter_gate_bf16_out_bf16(dGate_sorted, row_id, dGates_tk_out_bf16, M, T, K, stream);
        return;
    }
    if (g_bf16.world > MAX_RANKS) {
        fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
        abort();
    }

    size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
    bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                        &x_off, &meta_off, &counter_off, &dropped_off,
                        &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                        &tok_y_off, &tok_gate_off,
                        &total_size);

    const int tok_slots = T * K;
    const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
    if (static_cast<size_t>(tok_slots) > tok_cap) {
        fprintf(stderr,
                "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
        abort();
    }
    const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
    ipc_barrier_bf16_site(stream, "send_dgate_dist_bf16_out_bf16/pre_zero");
    maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, g_bf16.Ha, false, true, stream);
    ipc_barrier_bf16_site(stream, "send_dgate_dist_bf16_out_bf16/post_zero");

    const int threads = 256;
    const int blocks_by_work = std::max(1, (M + threads - 1) / threads);
    const int blocks = cap_warp_stride_blocks(blocks_by_work);
    if (M > 0) {
        k_send_dgate_ipc_bf16<<<blocks, threads, 0, stream>>>(
            row_id,
            dGate_sorted,
            M, T, K,
            static_cast<int>(g_bf16.capacity),
            tok_gate_off);
    }

    ipc_barrier_bf16_site(stream, "send_dgate_dist_bf16_out_bf16/post_send");

    const float* tok_gate = reinterpret_cast<const float*>(local_buf + tok_gate_off);
    const int t_threads = 256;
    const int t_blocks_by_work = std::max(1, (tok_slots + t_threads - 1) / t_threads);
    const int t_blocks = cap_warp_stride_blocks(t_blocks_by_work);
    k_collect_tok_gate_ipc_bf16<<<t_blocks, t_threads, 0, stream>>>(
        tok_gate,
        static_cast<__nv_bfloat16*>(dGates_tk_out_bf16),
        tok_slots);
}

extern "C" void rdep_scatter_dx_dist_bf16(
    const void* dXe_sorted,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::scatter_dx_hybrid_bf16(
            static_cast<const __nv_bfloat16*>(dXe_sorted),
            row_id,
            static_cast<float*>(dX_out),
            M, T, H, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) {
        fprintf(stderr, "RDEP FATAL: rdep_scatter_dx_dist_bf16 requires MODE_IPC or MODE_HYBRID\n");
        abort();
    }
    if (H <= 0 || (H & 7) != 0) {
        fprintf(stderr, "RDEP FATAL: rdep_scatter_dx_dist_bf16 requires H multiple of 8, got H=%d\n", H);
        abort();
    }

    // When blockscaled is active both states may be initialized; prefer the
    // blockscaled branch so barrier/layout domains match dispatch_meta_blockscaled.
    if (g_bf16.initialized && !(g_block.initialized && (g_block.profile == 0 || g_block.profile == 1))) {
        if (g_bf16.world <= 1) return;
        if (H != g_bf16.H) {
            fprintf(stderr, "RDEP FATAL: BF16 H mismatch in scatter_dx_dist_bf16: got H=%d state H=%d\n",
                    H, g_bf16.H);
            abort();
        }
        if (g_bf16.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
            abort();
        }
        size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                            &x_off, &meta_off, &counter_off, &dropped_off,
                            &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                            &tok_y_off, &tok_gate_off,
                            &total_size);

        const int Ha = g_bf16.Ha;
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }

        // Validate all buffer pointers before launching kernel
        for (int r = 0; r < g_bf16.world; r++) {
            if (g_bf16.buffer_ptrs[r] == nullptr) {
                fprintf(stderr, "RDEP FATAL: buffer_ptrs[%d] is NULL (rank=%d world=%d). IPC setup failed.\n",
                        r, g_bf16.rank, g_bf16.world);
                abort();
            }
        }

        int threads = 256;
        int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
        // Previous iteration already synchronized remote writers before local reduce.
        // We only need: (1) zero complete on all ranks before send, and
        // (2) send complete on all ranks before local reduce.
        maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, Ha, false, true, stream);
        ipc_barrier_bf16_site(stream, "scatter_dx_dist_bf16/post_zero");

        if (M > 0) {
            k_send_dx_tokslot_bf16<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(dXe_sorted),
                row_id,
                M, T, H, Ha, K,
                g_bf16.world,
                tok_y_off,
                tok_gate_off);
        }

        ipc_barrier_bf16_site(stream, "scatter_dx_dist_bf16/post_send");

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_tag = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int reduce_threads = 256;
        const int reduce_blocks_by_work = std::max(1, T);
        const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
        k_reduce_tokslot_sum_bf16<<<reduce_blocks, reduce_threads, 0, stream>>>(
            tok_y,
            tok_tag,
            static_cast<float*>(dX_out),
            T, H, Ha, K);

        return;
    }

    if (g_block.initialized) {
        if (g_block.world <= 1) return;
        if (H != g_block.H) {
            fprintf(stderr, "RDEP FATAL: blockscaled H mismatch in scatter_dx_dist_bf16: got H=%d state H=%d\n",
                    H, g_block.H);
            abort();
        }
        if (g_block.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_block.world, MAX_RANKS);
            abort();
        }
        size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                                   &x_off, &sfa_off, &y_off,
                                   &meta_off, &counter_off, &dropped_off,
                                   &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                                   &tok_y_off, &tok_gate_off,
                                   &total_size);

        const int Ha = g_block.Ha;
        const int tok_slots = T * K;
        const size_t tok_cap = g_block.capacity / static_cast<size_t>(g_block.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_block.capacity, g_block.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }

        for (int r = 0; r < g_block.world; r++) {
            if (g_block.buffer_ptrs[r] == nullptr) {
                fprintf(stderr, "RDEP FATAL: blockscaled buffer_ptrs[%d] is NULL (rank=%d world=%d). IPC setup failed.\n",
                        r, g_block.rank, g_block.world);
                abort();
            }
        }

        int threads = 256;
        int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        const char* local_buf = static_cast<const char*>(g_block.buffer_ptrs[g_block.rank]);
        // Mirror BF16 path: drop redundant pre-zero barrier in steady-state.
        maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, Ha, false, true, stream);
        ipc_barrier_block_site(stream, "scatter_dx_dist_blockscaled/post_zero");
        if (M > 0) {
            k_send_dx_tokslot_blockscaled<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(dXe_sorted),
                row_id,
                M, T, H, Ha, K,
                g_block.world,
                tok_y_off,
                tok_gate_off);
        }

        ipc_barrier_block_site(stream, "scatter_dx_dist_blockscaled/post_send");

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_tag = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int reduce_threads = 256;
        const int reduce_blocks_by_work = std::max(1, T);
        const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
        k_reduce_tokslot_sum_bf16<<<reduce_blocks, reduce_threads, 0, stream>>>(
            tok_y,
            tok_tag,
            static_cast<float*>(dX_out),
            T, H, Ha, K);
        return;
    }

    fprintf(stderr, "RDEP FATAL: rdep_scatter_dx_dist_bf16 requires initialized IPC/blockscaled state\n");
    abort();
}

extern "C" void rdep_scatter_dx_dist_from_pad_bf16(
    const void* dXe_pad,
    const int64_t* row_id,
    void* dX_out,
    int M, int T, int H, int K,
    cudaStream_t stream)
{
#ifdef WITH_NVSHMEM
    if (g_mode == MODE_HYBRID) {
        nvshmem::scatter_dx_hybrid_bf16_from_pad(
            static_cast<const __nv_bfloat16*>(dXe_pad),
            row_id,
            static_cast<float*>(dX_out),
            M, T, H, K,
            stream);
        return;
    }
#endif
    if (g_mode != MODE_IPC) {
        fprintf(stderr, "RDEP FATAL: rdep_scatter_dx_dist_from_pad_bf16 requires MODE_IPC\n");
        abort();
    }
    if (H <= 0 || (H & 7) != 0) {
        fprintf(stderr, "RDEP FATAL: rdep_scatter_dx_dist_from_pad_bf16 requires H multiple of 8, got H=%d\n", H);
        abort();
    }

    // When blockscaled is active both states may be initialized; prefer the
    // blockscaled branch so barrier/layout domains match dispatch_meta_blockscaled.
    if (g_bf16.initialized && !(g_block.initialized && (g_block.profile == 0 || g_block.profile == 1))) {
        if (g_bf16.world <= 1) return;
        if (H != g_bf16.H) {
            fprintf(stderr, "RDEP FATAL: BF16 H mismatch in scatter_dx_dist_from_pad_bf16: got H=%d state H=%d\n",
                    H, g_bf16.H);
            abort();
        }
        if (M < 0 || M > static_cast<int>(g_bf16.capacity)) {
            fprintf(stderr,
                    "RDEP FATAL: BF16 scatter_dx_dist_from_pad_bf16 requires 0 <= M <= capacity (M=%d capacity=%zu)\n",
                    M, g_bf16.capacity);
            abort();
        }
        if (g_bf16.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_bf16.world, MAX_RANKS);
            abort();
        }
        size_t x_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        bf16_buffer_offsets(g_bf16.capacity, g_bf16.Ha, g_bf16.world,
                            &x_off, &meta_off, &counter_off, &dropped_off,
                            &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                            &tok_y_off, &tok_gate_off,
                            &total_size);

        const int Ha = g_bf16.Ha;
        const int M_pad_cap = static_cast<int>(g_bf16.capacity + g_bf16.n_local * (g_bf16.align - 1));
        const int tok_slots = T * K;
        const size_t tok_cap = g_bf16.capacity / static_cast<size_t>(g_bf16.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_bf16.capacity, g_bf16.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }

        for (int r = 0; r < g_bf16.world; r++) {
            if (g_bf16.buffer_ptrs[r] == nullptr) {
                fprintf(stderr, "RDEP FATAL: buffer_ptrs[%d] is NULL (rank=%d world=%d). IPC setup failed.\n",
                        r, g_bf16.rank, g_bf16.world);
                abort();
            }
        }

        int threads = 256;
        int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        const char* local_buf = static_cast<const char*>(g_bf16.buffer_ptrs[g_bf16.rank]);
        // Same ordering contract as scatter_dx_dist_bf16(): pre-zero barrier is redundant.
        maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, Ha, false, true, stream);
        ipc_barrier_bf16_site(stream, "scatter_dx_dist_from_pad_bf16/post_zero");

        if (M > 0) {
            k_send_dx_tokslot_from_pad_bf16<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(dXe_pad),
                g_bf16.dest,
                row_id,
                M, T, H, Ha, K,
                g_bf16.world, M_pad_cap,
                tok_y_off,
                tok_gate_off);
        }

        ipc_barrier_bf16_site(stream, "scatter_dx_dist_from_pad_bf16/post_send");

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_tag = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int reduce_threads = 256;
        const int reduce_blocks_by_work = std::max(1, T);
        const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
        k_reduce_tokslot_sum_bf16<<<reduce_blocks, reduce_threads, 0, stream>>>(
            tok_y,
            tok_tag,
            static_cast<float*>(dX_out),
            T, H, Ha, K);
        return;
    }

    if (g_block.initialized) {
        if (g_block.world <= 1) return;
        if (H != g_block.H) {
            fprintf(stderr, "RDEP FATAL: blockscaled H mismatch in scatter_dx_dist_from_pad_bf16: got H=%d state H=%d\n",
                    H, g_block.H);
            abort();
        }
        if (M < 0 || M > static_cast<int>(g_block.capacity)) {
            fprintf(stderr,
                    "RDEP FATAL: blockscaled scatter_dx_dist_from_pad_bf16 requires 0 <= M <= capacity (M=%d capacity=%zu)\n",
                    M, g_block.capacity);
            abort();
        }
        if (g_block.world > MAX_RANKS) {
            fprintf(stderr, "RDEP ERROR: world=%d exceeds MAX_RANKS=%d\n", g_block.world, MAX_RANKS);
            abort();
        }
        size_t x_off, sfa_off, y_off, meta_off, counter_off, dropped_off, barrier_off, buf_ptrs_off, sig_ptrs_off, tok_y_off, tok_gate_off, total_size;
        blockscaled_buffer_offsets(g_block.capacity, g_block.H, g_block.Hp, g_block.Hsf, g_block.world,
                                   &x_off, &sfa_off, &y_off,
                                   &meta_off, &counter_off, &dropped_off,
                                   &barrier_off, &buf_ptrs_off, &sig_ptrs_off,
                                   &tok_y_off, &tok_gate_off,
                                   &total_size);

        const int Ha = g_block.Ha;
        const int M_pad_cap = static_cast<int>(g_block.capacity + g_block.n_local * (g_block.align - 1));
        const int tok_slots = T * K;
        const size_t tok_cap = g_block.capacity / static_cast<size_t>(g_block.world);
        if (static_cast<size_t>(tok_slots) > tok_cap) {
            fprintf(stderr,
                    "RDEP ERROR: tok-slot buffer too small: tok_slots=%d > capacity/world=%zu (capacity=%zu world=%d)\n",
                    tok_slots, tok_cap, g_block.capacity, g_block.world);
            abort();
        }
        if (K <= 0 || K > 32) {
            fprintf(stderr, "RDEP ERROR: K=%d out of supported range (1..32)\n", K);
            abort();
        }

        int threads = 256;
        int blocks_by_work = std::max(1, (M * 32 + threads - 1) / threads);
        int blocks = cap_warp_stride_blocks(blocks_by_work);
        const char* local_buf = static_cast<const char*>(g_block.buffer_ptrs[g_block.rank]);
        // Same ordering contract as BF16 branch.
        maybe_zero_tokslot_buffers(const_cast<char*>(local_buf), tok_y_off, tok_gate_off, tok_slots, Ha, false, true, stream);
        ipc_barrier_block_site(stream, "scatter_dx_dist_from_pad_blockscaled/post_zero");

        if (M > 0) {
            k_send_dx_tokslot_from_pad_blockscaled<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(dXe_pad),
                g_block.dest,
                row_id,
                M, T, H, Ha, K,
                g_block.world, M_pad_cap,
                tok_y_off,
                tok_gate_off);
        }

        ipc_barrier_block_site(stream, "scatter_dx_dist_from_pad_blockscaled/post_send");

        const uint16_t* tok_y = reinterpret_cast<const uint16_t*>(local_buf + tok_y_off);
        const float* tok_tag = reinterpret_cast<const float*>(local_buf + tok_gate_off);
        const int reduce_threads = 256;
        const int reduce_blocks_by_work = std::max(1, T);
        const int reduce_blocks = cap_warp_stride_blocks(reduce_blocks_by_work);
        k_reduce_tokslot_sum_bf16<<<reduce_blocks, reduce_threads, 0, stream>>>(
            tok_y,
            tok_tag,
            static_cast<float*>(dX_out),
            T, H, Ha, K);
        return;
    }

    fprintf(stderr, "RDEP FATAL: rdep_scatter_dx_dist_from_pad_bf16 requires initialized IPC/blockscaled state\n");
    abort();
}

} // namespace rdep
