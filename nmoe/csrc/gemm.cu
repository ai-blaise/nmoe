#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
 
#include <cstddef>
#include <cstdint>
#include <vector>
 
namespace {
 
inline cudaError_t cuda_from_cublas(cublasStatus_t s) {
  return (s == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}
 
struct CublasState {
  cublasHandle_t handle = nullptr;
  cudaStream_t last_stream = nullptr;
};
 
CublasState& cublas_state() {
  thread_local CublasState s;
  return s;
}

struct OffsHostScratch {
  int32_t* data = nullptr;
  size_t capacity = 0;
  cudaEvent_t ready_event = nullptr;
  bool ready_pending = false;

  ~OffsHostScratch() {
    if (ready_event != nullptr) {
      if (ready_pending) {
        (void)cudaEventSynchronize(ready_event);
        ready_pending = false;
      }
      (void)cudaEventDestroy(ready_event);
      ready_event = nullptr;
    }
    if (data != nullptr) {
      cudaFreeHost(data);
      data = nullptr;
      capacity = 0;
    }
  }

  cudaError_t ensure_event() {
    if (ready_event != nullptr) return cudaSuccess;
    return cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
  }

  cudaError_t wait_if_pending() {
    if (!ready_pending || ready_event == nullptr) return cudaSuccess;
    auto werr = cudaEventSynchronize(ready_event);
    if (werr != cudaSuccess) return werr;
    ready_pending = false;
    return cudaSuccess;
  }

  cudaError_t wait_if_pending_for(const int32_t* ptr) {
    if (ptr != data) return cudaSuccess;
    return wait_if_pending();
  }

  cudaError_t record_ready(cudaStream_t stream) {
    auto eerr = ensure_event();
    if (eerr != cudaSuccess) return eerr;
    auto rerr = cudaEventRecord(ready_event, stream);
    if (rerr != cudaSuccess) return rerr;
    ready_pending = true;
    return cudaSuccess;
  }

  cudaError_t ensure(size_t n) {
    auto werr = wait_if_pending();
    if (werr != cudaSuccess) return werr;
    if (capacity >= n && data != nullptr) return cudaSuccess;
    if (data != nullptr) {
      auto ferr = cudaFreeHost(data);
      if (ferr != cudaSuccess) return ferr;
      data = nullptr;
      capacity = 0;
    }
    if (n == 0) return cudaSuccess;
    auto aerr = cudaMallocHost(reinterpret_cast<void**>(&data), n * sizeof(int32_t));
    if (aerr != cudaSuccess) return aerr;
    capacity = n;
    return cudaSuccess;
  }
};

struct OffsHostScratchSlot {
  OffsHostScratch scratch;
  uintptr_t stream_key = 0;
};

struct OffsHostScratchTable {
  static constexpr int kMaxSlots = 128;
  OffsHostScratchSlot slots[kMaxSlots];
  int next_evict = 0;
};

OffsHostScratchTable& offs_host_scratch_table() {
  thread_local OffsHostScratchTable t;
  return t;
}

uintptr_t encode_stream_key(cudaStream_t stream) {
  return reinterpret_cast<uintptr_t>(stream) + 1;
}

OffsHostScratch* find_offs_host_scratch_for_ptr(const int32_t* ptr) {
  if (ptr == nullptr) return nullptr;
  auto& table = offs_host_scratch_table();
  for (int i = 0; i < OffsHostScratchTable::kMaxSlots; ++i) {
    if (table.slots[i].stream_key != 0 && table.slots[i].scratch.data == ptr) {
      return &table.slots[i].scratch;
    }
  }
  return nullptr;
}

OffsHostScratch* find_offs_host_scratch_for_stream(cudaStream_t stream) {
  auto& table = offs_host_scratch_table();
  const uintptr_t key = encode_stream_key(stream);
  for (int i = 0; i < OffsHostScratchTable::kMaxSlots; ++i) {
    if (table.slots[i].stream_key == key) {
      return &table.slots[i].scratch;
    }
  }
  return nullptr;
}

OffsHostScratch& offs_host_scratch(cudaStream_t stream) {
  auto& table = offs_host_scratch_table();
  const uintptr_t key = encode_stream_key(stream);
  for (int i = 0; i < OffsHostScratchTable::kMaxSlots; ++i) {
    if (table.slots[i].stream_key == key) {
      return table.slots[i].scratch;
    }
  }
  for (int i = 0; i < OffsHostScratchTable::kMaxSlots; ++i) {
    if (table.slots[i].stream_key == 0) {
      table.slots[i].stream_key = key;
      return table.slots[i].scratch;
    }
  }
  const int idx = table.next_evict;
  table.next_evict = (table.next_evict + 1) & (OffsHostScratchTable::kMaxSlots - 1);
  table.slots[idx].stream_key = key;
  return table.slots[idx].scratch;
}

cudaError_t resolve_offs_pad_host_view(const int32_t* offs_pad,
                                       int E,
                                       cudaStream_t stream,
                                       const int32_t** offs_pad_host_out,
                                       bool block_for_device_copy) {
  if (offs_pad == nullptr || offs_pad_host_out == nullptr || E < 0) {
    return cudaErrorInvalidValue;
  }
  if (E == 0) {
    *offs_pad_host_out = offs_pad;
    return cudaSuccess;
  }

  if (auto* prepared = find_offs_host_scratch_for_ptr(offs_pad)) {
    if (prepared->capacity < static_cast<size_t>(E)) return cudaErrorInvalidValue;
    auto werr = prepared->wait_if_pending_for(offs_pad);
    if (werr != cudaSuccess) return werr;
    *offs_pad_host_out = offs_pad;
    return cudaSuccess;
  }

  auto& scratch = offs_host_scratch(stream);

  cudaPointerAttributes attr{};
  cudaError_t attr_err = cudaPointerGetAttributes(&attr, offs_pad);
  bool is_device_backed = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_backed = (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
    if (is_device_backed) {
      int cur_dev = -1;
      auto derr = cudaGetDevice(&cur_dev);
      if (derr != cudaSuccess) return derr;
      if (attr.device >= 0 && attr.device != cur_dev) return cudaErrorInvalidDevice;
    }
#else
    is_device_backed = (attr.memoryType == cudaMemoryTypeDevice) || (attr.isManaged != 0);
#endif
  } else if (attr_err == cudaErrorInvalidValue) {
    // Unregistered host pointer path.
    (void)cudaGetLastError();
    is_device_backed = false;
  } else {
    return attr_err;
  }

  if (!is_device_backed) {
    *offs_pad_host_out = offs_pad;
    return cudaSuccess;
  }

  if (block_for_device_copy) {
    // No-fallback contract: grouped wgrad callers must pre-stage device offs via
    // bf16_prepare_offs_pad_host(...) to avoid hidden full-stream host sync here.
    return cudaErrorInvalidValue;
  }

  auto serr = scratch.ensure(static_cast<size_t>(E));
  if (serr != cudaSuccess) return serr;
  auto cpy = cudaMemcpyAsync(
      scratch.data,
      offs_pad,
      static_cast<size_t>(E) * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream);
  if (cpy != cudaSuccess) return cpy;

  // Async prepare path: record an event so the first grouped-wgrad launch can
  // wait only on this tiny D2H copy rather than a full stream synchronize.
  auto rerr = scratch.record_ready(stream);
  if (rerr != cudaSuccess) return rerr;

  *offs_pad_host_out = scratch.data;
  return cudaSuccess;
}

struct GroupedGemmLaunchScratch {
  std::vector<cublasOperation_t> op_a;
  std::vector<cublasOperation_t> op_b;
  std::vector<int> m;
  std::vector<int> n;
  std::vector<int> k;
  std::vector<int> lda;
  std::vector<int> ldb;
  std::vector<int> ldc;
  std::vector<const void*> a;
  std::vector<const void*> b;
  std::vector<void*> c;

  void reset(size_t reserve_count) {
    op_a.clear();
    op_b.clear();
    m.clear();
    n.clear();
    k.clear();
    lda.clear();
    ldb.clear();
    ldc.clear();
    a.clear();
    b.clear();
    c.clear();

    if (op_a.capacity() < reserve_count) op_a.reserve(reserve_count);
    if (op_b.capacity() < reserve_count) op_b.reserve(reserve_count);
    if (m.capacity() < reserve_count) m.reserve(reserve_count);
    if (n.capacity() < reserve_count) n.reserve(reserve_count);
    if (k.capacity() < reserve_count) k.reserve(reserve_count);
    if (lda.capacity() < reserve_count) lda.reserve(reserve_count);
    if (ldb.capacity() < reserve_count) ldb.reserve(reserve_count);
    if (ldc.capacity() < reserve_count) ldc.reserve(reserve_count);
    if (a.capacity() < reserve_count) a.reserve(reserve_count);
    if (b.capacity() < reserve_count) b.reserve(reserve_count);
    if (c.capacity() < reserve_count) c.reserve(reserve_count);
  }
};

GroupedGemmLaunchScratch& grouped_gemm_scratch() {
  thread_local GroupedGemmLaunchScratch s;
  return s;
}

cudaError_t grouped_gemm_bf16_f32accum(
    const std::vector<cublasOperation_t>& opA,
    const std::vector<cublasOperation_t>& opB,
    const std::vector<int>& m,
    const std::vector<int>& n,
    const std::vector<int>& k,
    const std::vector<int>& lda,
    const std::vector<int>& ldb,
    const std::vector<int>& ldc,
    const std::vector<const void*>& Aarray,
    const std::vector<const void*>& Barray,
    const std::vector<void*>& Carray,
    cudaStream_t stream) {
  if (m.empty()) return cudaSuccess;
  const int group_count = static_cast<int>(m.size());
  if (
      static_cast<int>(opA.size()) != group_count ||
      static_cast<int>(opB.size()) != group_count ||
      static_cast<int>(n.size()) != group_count ||
      static_cast<int>(k.size()) != group_count ||
      static_cast<int>(lda.size()) != group_count ||
      static_cast<int>(ldb.size()) != group_count ||
      static_cast<int>(ldc.size()) != group_count ||
      static_cast<int>(Aarray.size()) != group_count ||
      static_cast<int>(Barray.size()) != group_count ||
      static_cast<int>(Carray.size()) != group_count) {
    return cudaErrorInvalidValue;
  }

  auto& s = cublas_state();
  if (s.handle == nullptr) {
    cublasStatus_t st_create = cublasCreate(&s.handle);
    if (st_create != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st_create);
  }
  cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
  if (s.last_stream != stream) {
    st = cublasSetStream(s.handle, stream);
    if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
    s.last_stream = stream;
  }

  // Reuse small host-side arrays across calls to reduce allocator churn on hot path.
  thread_local std::vector<float> alpha;
  thread_local std::vector<float> beta;
  thread_local std::vector<int> group_size;
  if (static_cast<int>(alpha.size()) < group_count) alpha.resize(group_count, 1.0f);
  if (static_cast<int>(beta.size()) < group_count) beta.resize(group_count, 0.0f);
  if (static_cast<int>(group_size.size()) < group_count) group_size.resize(group_count, 1);

  st = cublasGemmGroupedBatchedEx(
      s.handle,
      opA.data(),
      opB.data(),
      m.data(),
      n.data(),
      k.data(),
      alpha.data(),
      Aarray.data(),
      CUDA_R_16BF,
      lda.data(),
      Barray.data(),
      CUDA_R_16BF,
      ldb.data(),
      beta.data(),
      Carray.data(),
      CUDA_R_16BF,
      ldc.data(),
      group_count,
      group_size.data(),
      CUBLAS_COMPUTE_32F);
  return cuda_from_cublas(st);
}

cudaError_t ensure_host_offs_ready(const int32_t* offs_pad_host, int E, cudaStream_t stream) {
  if (E < 0) return cudaErrorInvalidValue;
  if (E == 0) return cudaSuccess;
  if (offs_pad_host == nullptr) return cudaErrorInvalidValue;

  // Hot-path: stream-key lookup first (same stream as prepare call).
  OffsHostScratch* scratch = find_offs_host_scratch_for_stream(stream);
  if (scratch != nullptr && scratch->data == offs_pad_host) {
    if (scratch->capacity < static_cast<size_t>(E)) return cudaErrorInvalidValue;
    return scratch->wait_if_pending_for(offs_pad_host);
  }

  // Slow-path correctness guard for cross-stream pointer handoff.
  OffsHostScratch* scratch_ptr = find_offs_host_scratch_for_ptr(offs_pad_host);
  if (scratch_ptr == nullptr) return cudaSuccess;
  if (scratch_ptr->capacity < static_cast<size_t>(E)) return cudaErrorInvalidValue;
  return scratch_ptr->wait_if_pending_for(offs_pad_host);
}

cudaError_t bf16_wgrad_w2_cublaslt_impl(const void* A,
                                        const void* dY,
                                        void* dW2_out,
                                        const int32_t* offs_pad_host,
                                        int E,
                                        int H,
                                        int Dff,
                                        cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;
  auto ready_err = ensure_host_offs_ready(offs_pad_host, E, stream);
  if (ready_err != cudaSuccess) return ready_err;

  const auto* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A);
  const auto* dY_bf16 = reinterpret_cast<const __nv_bfloat16*>(dY);
  auto* dW2_bf16 = reinterpret_cast<__nv_bfloat16*>(dW2_out);

  // Column-major cuBLAS mapping for row-major inputs:
  // dW2_rm[Dff,H] = A_rm[m,Dff]^T @ dY_rm[m,H]
  // == C_cm[H,Dff] = dY_cm[H,m] @ A_cm[Dff,m]^T
  const int lda = H;    // dY interpreted as column-major [H, m]
  const int ldb = Dff;  // A interpreted as column-major [Dff, m]
  const int ldc = H;    // C column-major [H, Dff] (same bytes as row-major [Dff, H])
  const int stride_A = Dff;   // Row-major contiguous stride for A[m, Dff]
  const int stride_dY = H;    // Row-major contiguous stride for dY[m, H]
  const size_t expert_elems = static_cast<size_t>(Dff) * static_cast<size_t>(H);
  const size_t total_bytes = static_cast<size_t>(E) * expert_elems * sizeof(__nv_bfloat16);

  int zero_experts = 0;
  int32_t start_scan = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad_host[e];
    const int m = static_cast<int>(end - start_scan);
    if (m < 0) return cudaErrorInvalidValue;
    if (m == 0) ++zero_experts;
    start_scan = end;
  }
  if (zero_experts == E) {
    auto zerr = cudaMemsetAsync(dW2_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
    return cudaSuccess;
  }
  const bool memset_all = (zero_experts * 2 >= E);
  const bool do_zero_run_memset = (!memset_all) && (zero_experts > 0);
  if (memset_all) {
    auto zerr = cudaMemsetAsync(dW2_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
  }

  auto& scratch = grouped_gemm_scratch();
  scratch.reset(static_cast<size_t>(E));
  auto& op_a = scratch.op_a;
  auto& op_b = scratch.op_b;
  auto& m_array = scratch.m;
  auto& n_array = scratch.n;
  auto& k_array = scratch.k;
  auto& lda_array = scratch.lda;
  auto& ldb_array = scratch.ldb;
  auto& ldc_array = scratch.ldc;
  auto& a_array = scratch.a;
  auto& b_array = scratch.b;
  auto& c_array = scratch.c;

  int32_t start = 0;
  int zero_run_begin = -1;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad_host[e];
    const int m = static_cast<int>(end - start);
    if (m < 0) return cudaErrorInvalidValue;
    __nv_bfloat16* dW2e = dW2_bf16 + static_cast<size_t>(e) * expert_elems;
    if (m > 0) {
      if (do_zero_run_memset && zero_run_begin >= 0) {
        const size_t run_elems = static_cast<size_t>(e - zero_run_begin) * expert_elems;
        __nv_bfloat16* run_ptr = dW2_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
        auto zerr = cudaMemsetAsync(run_ptr, 0, run_elems * sizeof(__nv_bfloat16), stream);
        if (zerr != cudaSuccess) return zerr;
        zero_run_begin = -1;
      }
      const __nv_bfloat16* Ae = A_bf16 + static_cast<size_t>(start) * static_cast<size_t>(stride_A);
      const __nv_bfloat16* dYe = dY_bf16 + static_cast<size_t>(start) * static_cast<size_t>(stride_dY);
      op_a.push_back(CUBLAS_OP_N);
      op_b.push_back(CUBLAS_OP_T);
      m_array.push_back(H);
      n_array.push_back(Dff);
      k_array.push_back(m);
      lda_array.push_back(lda);
      ldb_array.push_back(ldb);
      ldc_array.push_back(ldc);
      a_array.push_back(dYe);
      b_array.push_back(Ae);
      c_array.push_back(dW2e);
    } else if (do_zero_run_memset) {
      if (zero_run_begin < 0) zero_run_begin = e;
    }
    start = end;
  }
  if (do_zero_run_memset && zero_run_begin >= 0) {
    const size_t run_elems = static_cast<size_t>(E - zero_run_begin) * expert_elems;
    __nv_bfloat16* run_ptr = dW2_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
    auto zerr = cudaMemsetAsync(run_ptr, 0, run_elems * sizeof(__nv_bfloat16), stream);
    if (zerr != cudaSuccess) return zerr;
  }
  auto err = grouped_gemm_bf16_f32accum(
      op_a,
      op_b,
      m_array,
      n_array,
      k_array,
      lda_array,
      ldb_array,
      ldc_array,
      a_array,
      b_array,
      c_array,
      stream);
  if (err != cudaSuccess) return err;
  return cudaSuccess;
}

cudaError_t bf16_wgrad_w13_pair_cublaslt_impl(const void* X,
                                              const void* dH1,
                                              const void* dH3,
                                              void* dW1_out,
                                              void* dW3_out,
                                              const int32_t* offs_pad_host,
                                              int E,
                                              int H,
                                              int Dff,
                                              cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;
  auto ready_err = ensure_host_offs_ready(offs_pad_host, E, stream);
  if (ready_err != cudaSuccess) return ready_err;

  const auto* X_bf16 = reinterpret_cast<const __nv_bfloat16*>(X);
  const auto* dH1_bf16 = reinterpret_cast<const __nv_bfloat16*>(dH1);
  const auto* dH3_bf16 = reinterpret_cast<const __nv_bfloat16*>(dH3);
  auto* dW1_bf16 = reinterpret_cast<__nv_bfloat16*>(dW1_out);
  auto* dW3_bf16 = reinterpret_cast<__nv_bfloat16*>(dW3_out);

  // Same mapping as bf16_wgrad_w13_cublaslt, for both dH1 and dH3.
  const int lda = Dff;  // dH* interpreted as column-major [Dff, m]
  const int ldb = H;    // X interpreted as column-major [H, m]
  const int ldc = Dff;  // C column-major [Dff, H]
  const int stride_X = H;      // Row-major contiguous stride for X[m, H]
  const int stride_dH = Dff;   // Row-major contiguous stride for dH[m, Dff]
  const size_t expert_elems = static_cast<size_t>(H) * static_cast<size_t>(Dff);
  const size_t total_bytes = static_cast<size_t>(E) * expert_elems * sizeof(__nv_bfloat16);

  int zero_experts = 0;
  int32_t start_scan = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad_host[e];
    const int m = static_cast<int>(end - start_scan);
    if (m < 0) return cudaErrorInvalidValue;
    if (m == 0) ++zero_experts;
    start_scan = end;
  }
  if (zero_experts == E) {
    auto zerr = cudaMemsetAsync(dW1_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
    zerr = cudaMemsetAsync(dW3_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
    return cudaSuccess;
  }
  const bool memset_all = (zero_experts * 2 >= E);
  const bool do_zero_run_memset = (!memset_all) && (zero_experts > 0);
  if (memset_all) {
    auto zerr = cudaMemsetAsync(dW1_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
    zerr = cudaMemsetAsync(dW3_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
  }

  const size_t pair_ops = static_cast<size_t>(E) * 2;
  auto& scratch = grouped_gemm_scratch();
  scratch.reset(pair_ops);
  auto& op_a = scratch.op_a;
  auto& op_b = scratch.op_b;
  auto& m_array = scratch.m;
  auto& n_array = scratch.n;
  auto& k_array = scratch.k;
  auto& lda_array = scratch.lda;
  auto& ldb_array = scratch.ldb;
  auto& ldc_array = scratch.ldc;
  auto& a_array = scratch.a;
  auto& b_array = scratch.b;
  auto& c_array = scratch.c;

  int32_t start = 0;
  int zero_run_begin = -1;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad_host[e];
    const int m = static_cast<int>(end - start);
    if (m < 0) return cudaErrorInvalidValue;
    __nv_bfloat16* dW1e = dW1_bf16 + static_cast<size_t>(e) * expert_elems;
    __nv_bfloat16* dW3e = dW3_bf16 + static_cast<size_t>(e) * expert_elems;
    if (m > 0) {
      if (do_zero_run_memset && zero_run_begin >= 0) {
        const size_t run_elems = static_cast<size_t>(e - zero_run_begin) * expert_elems;
        __nv_bfloat16* run_ptr_w1 = dW1_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
        __nv_bfloat16* run_ptr_w3 = dW3_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
        auto zerr = cudaMemsetAsync(run_ptr_w1, 0, run_elems * sizeof(__nv_bfloat16), stream);
        if (zerr != cudaSuccess) return zerr;
        zerr = cudaMemsetAsync(run_ptr_w3, 0, run_elems * sizeof(__nv_bfloat16), stream);
        if (zerr != cudaSuccess) return zerr;
        zero_run_begin = -1;
      }
      const __nv_bfloat16* Xe = X_bf16 + static_cast<size_t>(start) * static_cast<size_t>(stride_X);
      const __nv_bfloat16* dH1e = dH1_bf16 + static_cast<size_t>(start) * static_cast<size_t>(stride_dH);
      const __nv_bfloat16* dH3e = dH3_bf16 + static_cast<size_t>(start) * static_cast<size_t>(stride_dH);
      op_a.push_back(CUBLAS_OP_N);
      op_b.push_back(CUBLAS_OP_T);
      m_array.push_back(Dff);
      n_array.push_back(H);
      k_array.push_back(m);
      lda_array.push_back(lda);
      ldb_array.push_back(ldb);
      ldc_array.push_back(ldc);
      a_array.push_back(dH1e);
      b_array.push_back(Xe);
      c_array.push_back(dW1e);

      op_a.push_back(CUBLAS_OP_N);
      op_b.push_back(CUBLAS_OP_T);
      m_array.push_back(Dff);
      n_array.push_back(H);
      k_array.push_back(m);
      lda_array.push_back(lda);
      ldb_array.push_back(ldb);
      ldc_array.push_back(ldc);
      a_array.push_back(dH3e);
      b_array.push_back(Xe);
      c_array.push_back(dW3e);
    } else if (do_zero_run_memset) {
      if (zero_run_begin < 0) zero_run_begin = e;
    }
    start = end;
  }
  if (do_zero_run_memset && zero_run_begin >= 0) {
    const size_t run_elems = static_cast<size_t>(E - zero_run_begin) * expert_elems;
    __nv_bfloat16* run_ptr_w1 = dW1_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
    __nv_bfloat16* run_ptr_w3 = dW3_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
    auto zerr = cudaMemsetAsync(run_ptr_w1, 0, run_elems * sizeof(__nv_bfloat16), stream);
    if (zerr != cudaSuccess) return zerr;
    zerr = cudaMemsetAsync(run_ptr_w3, 0, run_elems * sizeof(__nv_bfloat16), stream);
    if (zerr != cudaSuccess) return zerr;
  }
  auto err = grouped_gemm_bf16_f32accum(
      op_a,
      op_b,
      m_array,
      n_array,
      k_array,
      lda_array,
      ldb_array,
      ldc_array,
      a_array,
      b_array,
      c_array,
      stream);
  if (err != cudaSuccess) return err;
  return cudaSuccess;
}
 
}  // namespace

extern "C" cudaError_t bf16_prepare_offs_pad_host(const int32_t* offs_pad,
                                                   int E,
                                                   cudaStream_t stream,
                                                   const int32_t** offs_pad_host_out) {
  return resolve_offs_pad_host_view(
      offs_pad,
      E,
      stream,
      offs_pad_host_out,
      /*block_for_device_copy=*/false);
}

extern "C" cudaError_t bf16_wgrad_w2_cublaslt(const void* A,
                                             const void* dY,
                                             void* dW2_out,
                                             const int32_t* offs_pad,
                                             int E,
                                             int H,
                                             int Dff,
                                             cudaStream_t stream) {
  const int32_t* offs_pad_host = nullptr;
  auto offs_err = resolve_offs_pad_host_view(
      offs_pad,
      E,
      stream,
      &offs_pad_host,
      /*block_for_device_copy=*/true);
  if (offs_err != cudaSuccess) return offs_err;
  return bf16_wgrad_w2_cublaslt_impl(
      A,
      dY,
      dW2_out,
      offs_pad_host,
      E,
      H,
      Dff,
      stream);
}

extern "C" cudaError_t bf16_wgrad_w2_cublaslt_host_offs(const void* A,
                                                       const void* dY,
                                                       void* dW2_out,
                                                       const int32_t* offs_pad_host,
                                                       int E,
                                                       int H,
                                                       int Dff,
                                                       cudaStream_t stream) {
  return bf16_wgrad_w2_cublaslt_impl(
      A,
      dY,
      dW2_out,
      offs_pad_host,
      E,
      H,
      Dff,
      stream);
}

extern "C" cudaError_t bf16_wgrad_w13_cublaslt(const void* X,
                                              const void* dH,
                                              void* dW_out,
                                              const int32_t* offs_pad,
                                              int E,
                                              int H,
                                              int Dff,
                                              cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;

  const int32_t* offs_pad_host = nullptr;
  auto offs_err = resolve_offs_pad_host_view(
      offs_pad,
      E,
      stream,
      &offs_pad_host,
      /*block_for_device_copy=*/true);
  if (offs_err != cudaSuccess) return offs_err;

  const auto* X_bf16 = reinterpret_cast<const __nv_bfloat16*>(X);
  const auto* dH_bf16 = reinterpret_cast<const __nv_bfloat16*>(dH);
  auto* dW_bf16 = reinterpret_cast<__nv_bfloat16*>(dW_out);

  // Column-major cuBLAS mapping for row-major inputs:
  // dW_rm[H,Dff] = X_rm[m,H]^T @ dH_rm[m,Dff]
  // == C_cm[Dff,H] = dH_cm[Dff,m] @ X_cm[H,m]^T
  const int lda = Dff;  // dH interpreted as column-major [Dff, m]
  const int ldb = H;    // X interpreted as column-major [H, m]
  const int ldc = Dff;  // C column-major [Dff, H] (same bytes as row-major [H, Dff])
  const int stride_X = H;      // Row-major contiguous stride for X[m, H]
  const int stride_dH = Dff;   // Row-major contiguous stride for dH[m, Dff]
  const size_t expert_elems = static_cast<size_t>(H) * static_cast<size_t>(Dff);
  const size_t total_bytes = static_cast<size_t>(E) * expert_elems * sizeof(__nv_bfloat16);

  int zero_experts = 0;
  int32_t start_scan = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad_host[e];
    const int m = static_cast<int>(end - start_scan);
    if (m < 0) return cudaErrorInvalidValue;
    if (m == 0) ++zero_experts;
    start_scan = end;
  }
  if (zero_experts == E) {
    auto zerr = cudaMemsetAsync(dW_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
    return cudaSuccess;
  }
  const bool memset_all = (zero_experts * 2 >= E);
  const bool do_zero_run_memset = (!memset_all) && (zero_experts > 0);
  if (memset_all) {
    auto zerr = cudaMemsetAsync(dW_bf16, 0, total_bytes, stream);
    if (zerr != cudaSuccess) return zerr;
  }

  auto& scratch = grouped_gemm_scratch();
  scratch.reset(static_cast<size_t>(E));
  auto& op_a = scratch.op_a;
  auto& op_b = scratch.op_b;
  auto& m_array = scratch.m;
  auto& n_array = scratch.n;
  auto& k_array = scratch.k;
  auto& lda_array = scratch.lda;
  auto& ldb_array = scratch.ldb;
  auto& ldc_array = scratch.ldc;
  auto& a_array = scratch.a;
  auto& b_array = scratch.b;
  auto& c_array = scratch.c;

  int32_t start = 0;
  int zero_run_begin = -1;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad_host[e];
    const int m = static_cast<int>(end - start);
    if (m < 0) return cudaErrorInvalidValue;
    __nv_bfloat16* dWe = dW_bf16 + static_cast<size_t>(e) * expert_elems;
    if (m > 0) {
      if (do_zero_run_memset && zero_run_begin >= 0) {
        const size_t run_elems = static_cast<size_t>(e - zero_run_begin) * expert_elems;
        __nv_bfloat16* run_ptr = dW_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
        auto zerr = cudaMemsetAsync(run_ptr, 0, run_elems * sizeof(__nv_bfloat16), stream);
        if (zerr != cudaSuccess) return zerr;
        zero_run_begin = -1;
      }
      const __nv_bfloat16* Xe = X_bf16 + static_cast<size_t>(start) * static_cast<size_t>(stride_X);
      const __nv_bfloat16* dHe = dH_bf16 + static_cast<size_t>(start) * static_cast<size_t>(stride_dH);
      op_a.push_back(CUBLAS_OP_N);
      op_b.push_back(CUBLAS_OP_T);
      m_array.push_back(Dff);
      n_array.push_back(H);
      k_array.push_back(m);
      lda_array.push_back(lda);
      ldb_array.push_back(ldb);
      ldc_array.push_back(ldc);
      a_array.push_back(dHe);
      b_array.push_back(Xe);
      c_array.push_back(dWe);
    } else if (do_zero_run_memset) {
      if (zero_run_begin < 0) zero_run_begin = e;
    }
    start = end;
  }
  if (do_zero_run_memset && zero_run_begin >= 0) {
    const size_t run_elems = static_cast<size_t>(E - zero_run_begin) * expert_elems;
    __nv_bfloat16* run_ptr = dW_bf16 + static_cast<size_t>(zero_run_begin) * expert_elems;
    auto zerr = cudaMemsetAsync(run_ptr, 0, run_elems * sizeof(__nv_bfloat16), stream);
    if (zerr != cudaSuccess) return zerr;
  }
  auto err = grouped_gemm_bf16_f32accum(
      op_a,
      op_b,
      m_array,
      n_array,
      k_array,
      lda_array,
      ldb_array,
      ldc_array,
      a_array,
      b_array,
      c_array,
      stream);
  if (err != cudaSuccess) return err;
  return cudaSuccess;
}

extern "C" cudaError_t bf16_wgrad_w13_pair_cublaslt(const void* X,
                                                   const void* dH1,
                                                   const void* dH3,
                                                   void* dW1_out,
                                                   void* dW3_out,
                                                   const int32_t* offs_pad,
                                                   int E,
                                                   int H,
                                                   int Dff,
                                                   cudaStream_t stream) {
  const int32_t* offs_pad_host = nullptr;
  auto offs_err = resolve_offs_pad_host_view(
      offs_pad,
      E,
      stream,
      &offs_pad_host,
      /*block_for_device_copy=*/true);
  if (offs_err != cudaSuccess) return offs_err;
  return bf16_wgrad_w13_pair_cublaslt_impl(
      X,
      dH1,
      dH3,
      dW1_out,
      dW3_out,
      offs_pad_host,
      E,
      H,
      Dff,
      stream);
}

extern "C" cudaError_t bf16_wgrad_w13_pair_cublaslt_host_offs(const void* X,
                                                             const void* dH1,
                                                             const void* dH3,
                                                             void* dW1_out,
                                                             void* dW3_out,
                                                             const int32_t* offs_pad_host,
                                                             int E,
                                                             int H,
                                                             int Dff,
                                                             cudaStream_t stream) {
  return bf16_wgrad_w13_pair_cublaslt_impl(
      X,
      dH1,
      dH3,
      dW1_out,
      dW3_out,
      offs_pad_host,
      E,
      H,
      Dff,
      stream);
}
