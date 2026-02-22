// Pybind11 bindings for RDEP
// Module name: rdep

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

namespace py = pybind11;

static inline std::string cuda_err(cudaError_t err) {
  return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
}

// RDEP C symbols from rdep.cu
#ifdef BUILD_RDEP
extern "C" {
  // Init
  void rdep_init(int rank, int world, int local_world);
  int  rdep_get_mode();
  bool rdep_has_nvshmem();

#ifdef WITH_NVSHMEM
  // NVSHMEM functions (from rdep_nvshmem.cu) - use rdep_ prefix to avoid conflicts
  void rdep_nvshmem_get_uid(void* uid_out);
  int  rdep_nvshmem_get_uid_size();
  void rdep_nvshmem_init_with_uid(const void* uid, int rank, int world, int local_world);
  void rdep_nvshmem_finalize();
  void rdep_nvshmem_alloc_bf16(size_t capacity, int H, int n_local);
  void rdep_nvshmem_alloc_blockscaled(size_t capacity, int H, int n_local, int profile);
  void rdep_nvshmem_quiet();
  // NVSHMEM IPC buffer functions (separate cudaMalloc'd buffer for intra-node IPC)
  void rdep_nvshmem_get_ipc_handle_bf16(void* handle_out);
  void rdep_nvshmem_open_ipc_handles_bf16(const void* handles, int local_world);
  void rdep_nvshmem_sync_ipc_buffer_ptrs_bf16();
  void rdep_nvshmem_get_ipc_handle_blockscaled(void* handle_out);
  void rdep_nvshmem_open_ipc_handles_blockscaled(const void* handles, int local_world);
  void rdep_nvshmem_sync_ipc_buffer_ptrs_blockscaled();
#endif
  void rdep_get_ipc_handle_bf16(void* handle_out);
  void rdep_get_ipc_handle_blockscaled(void* handle_out);
  void rdep_open_ipc_handles_bf16(const void* handles, int world);
  void rdep_open_ipc_handles_blockscaled(const void* handles, int world);
  void rdep_sync_buffer_ptrs_bf16();
  void rdep_sync_buffer_ptrs_blockscaled();
  void rdep_direct_ipc_write(void* src_ptr, int target_rank, size_t offset, size_t n_bytes, cudaStream_t stream);

  // BF16 path
  void rdep_alloc_bf16(size_t capacity, int H, int n_local);
  int  rdep_dispatch_meta_bf16(const void* x, const int* eids, const float* gates,
                               int T, int K, int align,
                               int* offs_pad_out, int* M_pad_out,
                               cudaStream_t stream);
  void rdep_gather_xe_bf16(void* Xe_out, int M_recv, int M_pad, cudaStream_t stream);
  void rdep_gather_meta_sorted_bf16(int64_t* row_id_out, float* gate_out, int M_recv, cudaStream_t stream);
  void rdep_gather_from_pad_bf16(const void* in_pad, void* out_sorted, int M_recv, int H, cudaStream_t stream);
  void rdep_scatter_sorted_to_pad_bf16(const void* in_sorted, void* out_pad, int M_recv, int H, cudaStream_t stream);
  void rdep_zero_padding_rows_bf16(void* out_pad, const int* offs_pad, int M_recv, int M_pad, int H, cudaStream_t stream);
  int  rdep_dispatch_meta_blockscaled(const void* x, const int* eids, const float* gates,
                                     int T, int K,
                                     int* offs_pad_out, int* M_pad_out,
                                     cudaStream_t stream);
  void rdep_gather_xe_blockscaled(void* Xe_q_out, void* Xe_sf_out, int M_recv, int M_pad, cudaStream_t stream);
  void rdep_return_scatter(const void* Ye, void* out, int M_recv, int T, int K,
                           cudaStream_t stream);
  void rdep_return_scatter_from_pad_bf16(const void* Ye_pad, void* out, int M_recv, int T, int K,
                                        cudaStream_t stream);
  void rdep_return_scatter_from_pad_blockscaled(const void* Ye_pad, void* out, int M_recv, int T, int K,
                                               cudaStream_t stream);
  void rdep_gather_dy_bf16(const void* dY, const void* Ye, const int64_t* row_id, const float* gate,
                           void* dYe_out, float* dGate_out,
                           int M, int T, int H, int K, cudaStream_t stream);
  void rdep_scatter_gate_bf16(const float* dGate_sorted, const int64_t* row_id,
                              float* dGates_tk, int M, int T, int K, cudaStream_t stream);
  void rdep_scatter_gate_bf16_out_bf16(const float* dGate_sorted, const int64_t* row_id,
                                       void* dGates_tk_bf16, int M, int T, int K, cudaStream_t stream);
  // SonicMoE dGate identity: gather dY with gate scaling, defer dGate to ⟨A, dA'⟩
  void rdep_gather_dy_nogate_bf16(const void* dY, const int64_t* row_id,
                                  const float* gate, void* dYe_out,
                                  int M, int T, int H, int K, cudaStream_t stream);
  void rdep_dgate_from_adA_bf16(const void* A_pad, const void* dA_pad,
                                float* dGate_sorted_out,
                                int M, int D, cudaStream_t stream);
  // SonicMoE distributed: gather dYe across ranks, then send dGate back
  void rdep_gather_dy_nogate_dist_bf16(const void* dY_local, const int* eids,
                                       const int64_t* row_id, const float* gate_sorted,
                                       void* dYe_out,
                                       int M, int T, int H, int K, cudaStream_t stream);
  void rdep_send_dgate_dist_bf16(const int64_t* row_id, const float* dGate_sorted,
                                 float* dGates_tk_out,
                                 int M, int T, int K, cudaStream_t stream);
  void rdep_send_dgate_dist_bf16_out_bf16(const int64_t* row_id, const float* dGate_sorted,
                                          void* dGates_tk_out_bf16,
                                          int M, int T, int K, cudaStream_t stream);
  void rdep_scatter_dx_bf16_internal(const void* dXe_pad, const int64_t* row_id,
                                     void* dX_out, int M, int T, int H, int K, cudaStream_t stream);
	  void rdep_gather_dy_dist_bf16(const void* dY_local, const int* eids, const void* Ye_pad,
	                                const int64_t* row_id, const float* gate_sorted,
	                                void* dYe_out, float* dGate_sorted_out, float* dGates_tk_out,
	                                int M, int T, int H, int K, cudaStream_t stream);
  void rdep_scatter_dx_dist_bf16(const void* dXe_sorted, const int64_t* row_id,
                                 void* dX_out, int M, int T, int H, int K, cudaStream_t stream);
  void rdep_scatter_dx_dist_from_pad_bf16(const void* dXe_pad, const int64_t* row_id,
                                          void* dX_out, int M, int T, int H, int K, cudaStream_t stream);

  // Blockscaled path
  void rdep_alloc_blockscaled(size_t capacity, int H, int n_local, int profile);

  // Dropped token counters (for capacity overflow detection)
  int rdep_get_dropped_bf16(cudaStream_t stream);
  int rdep_get_dropped_blockscaled(cudaStream_t stream);

  // Quantization and swizzle (dense/weights usage only)
  cudaError_t quant_fp8(const void* x, int ldx,
                        void* out, int ld_out,
                        void* sfa, int ld_sf,
                        int M, int K, cudaStream_t stream);
  cudaError_t quant_nvfp4(const void* x, int ldx,
                          void* out, int ld_out,
                          void* sfa, int ld_sf,
                          int M, int K, cudaStream_t stream);
  cudaError_t quant_fp8_sf_strided_mma(const void* x, int ldx,
                                       void* out, int ld_out,
                                       void* sf_mma,
                                       const int32_t* offs,
                                       int E, int M_e_stride,
                                       int M_pad, int K, cudaStream_t stream);
  cudaError_t quant_nvfp4_sf_strided_mma(const void* x, int ldx,
                                         void* out, int ld_out,
                                         void* sf_mma,
                                         const int32_t* offs,
                                         int E, int M_e_stride,
                                         int M_pad, int K, cudaStream_t stream);
  cudaError_t ct_nvfp4_to_blockscaled_mma(
      const void* ct_packed, const void* ct_scale, const void* ct_gs,
      void* out, void* sf_mma,
      int E, int M_ct, int K_in, int M_stride, int group_size,
      cudaStream_t stream);
  cudaError_t ct_nvfp4_to_blockscaled_mma_interleaved(
      const void* ct_packed_a, const void* ct_scale_a, const void* ct_gs_a,
      const void* ct_packed_b, const void* ct_scale_b, const void* ct_gs_b,
      void* out, void* sf_mma,
      int E, int M_ct, int K_in, int M_stride, int group_size,
      cudaStream_t stream);
  cudaError_t ct_nvfp4_to_bf16(
      const void* ct_packed, const void* ct_scale, const void* ct_gs,
      void* out,
      int M, int K, int group_size,
      int gs_stride, int expert_rows, int transpose,
      cudaStream_t stream);
  cudaError_t swiglu_bwd_bf16(const void* h1, int ld_h1,
                              const void* h3, int ld_h3,
                              const void* dA, int ld_dA,
                              void* A_out, int ld_A,
                              void* dH1_out, int ld_dH1,
                              void* dH3_out, int ld_dH3,
                              int M, int K, cudaStream_t stream);
  cudaError_t bf16_wgrad_w2_cublaslt(const void* A,
                                     const void* dY,
                                     void* dW2_out,
                                     const int32_t* offs_pad,
                                     int E, int H, int Dff,
                                     cudaStream_t stream);
  cudaError_t bf16_wgrad_w2_cublaslt_host_offs(const void* A,
                                               const void* dY,
                                               void* dW2_out,
                                               const int32_t* offs_pad_host,
                                               int E, int H, int Dff,
                                               cudaStream_t stream);
  cudaError_t bf16_prepare_offs_pad_host(const int32_t* offs_pad,
                                         int E,
                                         cudaStream_t stream,
                                         const int32_t** offs_pad_host_out);
  cudaError_t bf16_wgrad_w13_cublaslt(const void* X,
                                      const void* dH,
                                      void* dW_out,
                                      const int32_t* offs_pad,
                                      int E, int H, int Dff,
                                      cudaStream_t stream);
  cudaError_t bf16_wgrad_w13_pair_cublaslt(const void* X,
                                           const void* dH1,
                                           const void* dH3,
                                           void* dW1_out,
                                           void* dW3_out,
                                           const int32_t* offs_pad,
                                           int E, int H, int Dff,
                                           cudaStream_t stream);
  cudaError_t bf16_wgrad_w13_pair_cublaslt_host_offs(const void* X,
                                                     const void* dH1,
                                                     const void* dH3,
                                                     void* dW1_out,
                                                     void* dW3_out,
                                                     const int32_t* offs_pad_host,
                                                     int E, int H, int Dff,
                                                     cudaStream_t stream);
  cudaError_t swiglu_quant_fp8_sf_strided_mma(const void* h13, int ld_h13,
                                              void* out, int ld_out,
                                              void* sf_mma,
                                              const int32_t* offs,
                                              int E, int M_e_stride,
                                              int M_pad, int K, cudaStream_t stream);
  cudaError_t swiglu_quant_nvfp4_sf_strided_mma(const void* h13, int ld_h13,
                                                void* out, int ld_out,
                                                void* sf_mma,
                                                const int32_t* offs,
                                                int E, int M_e_stride,
                                                int M_pad, int K, cudaStream_t stream);
  cudaError_t quant_fp8_with_sfa(const void* x, int ldx,
                                 void* out, int ld_out,
                                 const void* sfa, int ld_sf,
                                 int M, int K, cudaStream_t stream);
  cudaError_t quant_nvfp4_with_sfa(const void* x, int ldx,
                                   void* out, int ld_out,
                                   const void* sfa, int ld_sf,
                                   int M, int K, cudaStream_t stream);
  cudaError_t dequant_fp8_to_bf16(const void* q, int ldq,
                                  const void* sfa, int ld_sf,
                                  void* out, int ld_out,
                                  int M, int K, cudaStream_t stream);
  cudaError_t dequant_fp8_to_bf16_mma_sf(const void* q, int ldq,
                                         const void* sfa, int ld_sf,
                                         void* out, int ld_out,
                                         int M, int K, cudaStream_t stream);
  cudaError_t dequant_nvfp4_to_bf16(const void* q, int ldq,
                                    const void* sfa, int ld_sf,
                                    void* out, int ld_out,
                                    int M, int K, cudaStream_t stream);
  cudaError_t dequant_nvfp4_to_bf16_mma_sf(const void* q, int ldq,
                                           const void* sfa, int ld_sf,
                                           void* out, int ld_out,
                                           int M, int K, cudaStream_t stream);
  cudaError_t swizzle_sf_mkl_to_mma(const void* sf_mkl, void* sf_mma,
                                    int M, int sf_k, cudaStream_t stream);
  cudaError_t expert_adamw_step(
      int profile,
      void* W1, const void* dW1, void* m1, void* v1,
      void* W3, const void* dW3, void* m3, void* v3,
      void* W2, const void* dW2, void* m2, void* v2,
      void* W13_q, void* W13_sf_mma,
      void* W2_q, void* W2_sf_mma,
      int E, int H, int Dff,
      float lr, float beta1, float beta2,
      float weight_decay, float eps,
      float step_size, float inv_bias_correction2_sqrt,
      cudaStream_t stream);
  cudaError_t fused_adamw_step_fp32(
      void* param, const void* grad,
      void* exp_avg, void* exp_avg_sq,
      int N,
      float beta1, float beta2,
      float lr, float weight_decay, float eps,
      float step_size, float inv_bc2_sqrt,
      cudaStream_t stream);
  cudaError_t fused_adamw_step_bf16(
      void* param, const void* grad,
      void* exp_avg, void* exp_avg_sq,
      int N,
      float beta1, float beta2,
      float lr, float weight_decay, float eps,
      float step_size, float inv_bc2_sqrt,
      cudaStream_t stream);
  cudaError_t eco_adam_nvfp4_update(
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
      cudaStream_t stream);
  cudaError_t eco_adam_nvfp4_update_bf16(
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
      cudaStream_t stream);
  cudaError_t eco_adam_nvfp4_fv_update(
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
      cudaStream_t stream);
  cudaError_t eco_adam_nvfp4_fv_update_bf16(
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
      cudaStream_t stream);
  cudaError_t eco_mv_accumulate(
      void* m_data, void* m_scale,
      void* v_data, void* v_scale,
      const void* grad,
      int E, int in_dim, int out_dim,
      float beta1_frac, float beta2_frac,
      cudaStream_t stream);
  cudaError_t eco_mv_accumulate_bf16(
      void* m_data, void* m_scale,
      void* v_data, void* v_scale,
      const void* grad,
      int E, int in_dim, int out_dim,
      float beta1_frac, float beta2_frac,
      cudaStream_t stream);
  cudaError_t eco_mv_accumulate_fv(
      void* m_data, void* m_scale,
      void* v_row, void* v_col, void* v_rms,
      const void* grad,
      int E, int in_dim, int out_dim,
      float beta1_frac, float beta2_frac,
      cudaStream_t stream);
  cudaError_t eco_mv_accumulate_fv_bf16(
      void* m_data, void* m_scale,
      void* v_row, void* v_col, void* v_rms,
      const void* grad,
      int E, int in_dim, int out_dim,
      float beta1_frac, float beta2_frac,
      cudaStream_t stream);
	  cudaError_t build_grouped_gemm_metadata(
	      const int32_t* offs, int E,
	      int64_t A_base, int64_t A_row_bytes,
	      int64_t B_base, int64_t B_expert_bytes,
	      int64_t C_base, int64_t C_row_bytes,
	      int64_t SFA_base, int64_t SFA_row_bytes,
	      int64_t SFB_base, int64_t SFB_expert_bytes,
	      int32_t A_stride0_elem, int32_t A_stride1_elem,
	      int32_t B_stride0_elem, int32_t B_stride1_elem,
	      int32_t C_stride0_elem, int32_t C_stride1_elem,
	      int32_t N, int32_t K,
	      int32_t* sizes_mnkl, int32_t* strides_abc, int64_t* ptrs_abc, int64_t* ptrs_sfasfb,
	      cudaStream_t stream);

  // Fused RoPE (rope_fused.cu)
  cudaError_t fused_rope_forward(
      const void* x, const void* cos_buf, const void* sin_buf, void* out,
      int total_vecs, int seq_len, int n_heads, int head_dim,
      cudaStream_t stream);
  cudaError_t fused_rope_backward(
      const void* grad_out, const void* cos_buf, const void* sin_buf, void* grad_x,
      int total_vecs, int seq_len, int n_heads, int head_dim,
      cudaStream_t stream);
  // Partial RoPE (in-place, rope portion only - eliminates split/cat)
  cudaError_t fused_rope_forward_partial(
      void* x, const void* cos_buf, const void* sin_buf,
      int total_vecs, int seq_len, int n_heads, int head_dim, int nope_dim,
      cudaStream_t stream);
  cudaError_t fused_rope_backward_partial(
      void* grad_x, const void* cos_buf, const void* sin_buf,
      int total_vecs, int seq_len, int n_heads, int head_dim, int nope_dim,
      cudaStream_t stream);

  // NVFP4 scale recomputation (eco_adam.cu)
  cudaError_t nvfp4_recompute_group_scales(
      void* W_packed, void* W_scale, const void* W_gs,
      int E, int out_dim, int in_dim, int group_size,
      cudaStream_t stream);
}
#endif

static inline cudaStream_t to_stream(py::object s) {
  if (s.is_none()) return nullptr;
  if (!py::hasattr(s, "cuda_stream")) {
    throw py::type_error("stream must be torch.cuda.Stream or None");
  }
  auto uptr = s.attr("cuda_stream").cast<uintptr_t>();
  return reinterpret_cast<cudaStream_t>(uptr);
}

constexpr int NMOE_RDEP_ABI_VERSION = 1;
// IPC handle size is 64 bytes
constexpr int IPC_HANDLE_SIZE = 64;

static inline void validate_ipc_handles(const py::array_t<uint8_t>& handles, int n_ranks, const char* fn_name) {
  auto info = handles.request();
  if (info.ndim != 1 || info.strides[0] != 1) {
    throw py::value_error(std::string(fn_name) + ": handles must be contiguous 1-D uint8");
  }
  const ssize_t expected = static_cast<ssize_t>(n_ranks) * IPC_HANDLE_SIZE;
  if (info.size != expected) {
    throw py::value_error(
      std::string(fn_name) + ": expected " + std::to_string(expected) + " bytes, got " + std::to_string(info.size)
    );
  }
}

#ifdef BUILD_MUON
// Muon module (built separately)
extern "C" void* muon_plan_create(int Bmax, int M, int N);
extern "C" void  muon_plan_destroy(void* plan);
extern "C" void  muon_plan_run(void* plan, void* x_bf16, long long B, int M, int N, int steps, int coeff_mode, cudaStream_t stream);
extern "C" void  muon(void* x_bf16, long long B, int M, int N, int steps, int coeff_mode, cudaStream_t stream);
PYBIND11_MODULE(muon, m2) {
  m2.doc() = "Muon orthogonalization (CUDA/cuBLAS)";
  m2.def("plan_create", [](int Bmax, int M, int N){
    return reinterpret_cast<uintptr_t>(muon_plan_create(Bmax, M, N));
  }, py::arg("Bmax"), py::arg("M"), py::arg("N"));
  m2.def("plan_destroy", [](uintptr_t plan){ muon_plan_destroy(reinterpret_cast<void*>(plan)); }, py::arg("plan"));
  m2.def("plan_run", [](uintptr_t plan, uintptr_t x_ptr, long long B, int M, int N, int steps, int coeff_mode, py::object stream){
    muon_plan_run(reinterpret_cast<void*>(plan), reinterpret_cast<void*>(x_ptr), B, M, N, steps, coeff_mode, to_stream(stream));
  }, py::arg("plan"), py::arg("x"), py::arg("B"), py::arg("M"), py::arg("N"), py::arg("steps") = 5, py::arg("coeff_mode") = 1, py::arg("stream") = py::none());
  m2.def("muon", [](uintptr_t x_ptr, long long B, int M, int N, int steps, int coeff_mode, py::object stream) {
    muon(reinterpret_cast<void*>(x_ptr), B, M, N, steps, coeff_mode, to_stream(stream));
  }, py::arg("x"), py::arg("B"), py::arg("M"), py::arg("N"), py::arg("steps") = 5, py::arg("coeff_mode") = 1, py::arg("stream") = py::none());
}
#endif

#ifdef BUILD_RDEP
PYBIND11_MODULE(rdep, m) {
  m.doc() = "RDEP: Expert-parallel dispatch/return for MoE";
  m.def("abi_version", []() { return NMOE_RDEP_ABI_VERSION; });
  m.def("ipc_handle_size", []() { return IPC_HANDLE_SIZE; });

  // ========== Init ==========
  m.def("init", [](int rank, int world, int local_world) {
    rdep_init(rank, world, local_world);
  }, py::arg("rank"), py::arg("world"), py::arg("local_world"),
     "Initialize RDEP with rank, world size, and local world size");

  m.def("get_mode", &rdep_get_mode,
        "Get current mode (0=SINGLE, 1=IPC, 2=HYBRID)");

  m.def("has_nvshmem", &rdep_has_nvshmem,
        "Check if NVSHMEM support is compiled in");

  m.def("get_ipc_handle_bf16", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_get_ipc_handle_bf16(handle.mutable_data());
    return handle;
  }, "Get local IPC handle for BF16 buffer");

  m.def("get_ipc_handle_blockscaled", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_get_ipc_handle_blockscaled(handle.mutable_data());
    return handle;
  }, "Get local IPC handle for blockscaled buffer");

  m.def("open_ipc_handles_bf16", [](py::array_t<uint8_t> handles, int world) {
    validate_ipc_handles(handles, world, "open_ipc_handles_bf16");
    rdep_open_ipc_handles_bf16(handles.data(), world);
  }, py::arg("handles"), py::arg("world"),
     "Open remote IPC handles for BF16 path");

  m.def("open_ipc_handles_blockscaled", [](py::array_t<uint8_t> handles, int world) {
    validate_ipc_handles(handles, world, "open_ipc_handles_blockscaled");
    rdep_open_ipc_handles_blockscaled(handles.data(), world);
  }, py::arg("handles"), py::arg("world"),
     "Open remote IPC handles for blockscaled path");

  m.def("sync_buffer_ptrs_bf16", &rdep_sync_buffer_ptrs_bf16,
        "Sync BF16 buffer pointers to device");
  m.def("sync_buffer_ptrs_blockscaled", &rdep_sync_buffer_ptrs_blockscaled,
        "Sync blockscaled buffer pointers to device");

  // ========== BF16 Path ==========
  m.def("alloc_bf16", [](size_t capacity, int H, int n_local) {
    rdep_alloc_bf16(capacity, H, n_local);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"),
     "Allocate BF16 dispatch buffers");

  m.def("dispatch_meta_bf16", [](uintptr_t x_ptr, uintptr_t eids_ptr, uintptr_t gates_ptr,
                                 int T, int K, int align,
                                 uintptr_t offs_pad_ptr, uintptr_t M_pad_ptr,
                                 py::object stream) {
    return rdep_dispatch_meta_bf16(
        reinterpret_cast<const void*>(x_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const float*>(gates_ptr),
        T, K, align,
        reinterpret_cast<int*>(offs_pad_ptr),
        reinterpret_cast<int*>(M_pad_ptr),
        to_stream(stream));
  }, py::arg("x"), py::arg("eids"), py::arg("gates"),
     py::arg("T"), py::arg("K"), py::arg("align"),
     py::arg("offs_pad"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Dispatch BF16 tokens to experts (meta only: sort + pad mapping). "
     "align: per-expert row padding (8 for BF16, 128 for blockscaled). "
     "NOTE: `M_pad` is treated as a pinned host scratch (used to read back M_recv).");

  m.def("gather_xe_bf16", [](uintptr_t Xe_out_ptr, int M_recv, int M_pad, py::object stream) {
    rdep_gather_xe_bf16(
        reinterpret_cast<void*>(Xe_out_ptr),
        M_recv, M_pad,
        to_stream(stream));
  }, py::arg("Xe_out"), py::arg("M_recv"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Gather BF16 activations into padded expert layout (requires prior dispatch_meta_bf16)");

  m.def("gather_meta_sorted_bf16", [](uintptr_t row_id_ptr, uintptr_t gate_sorted_ptr, int M_recv, py::object stream) {
    rdep_gather_meta_sorted_bf16(
        reinterpret_cast<int64_t*>(row_id_ptr),
        reinterpret_cast<float*>(gate_sorted_ptr),
        M_recv,
        to_stream(stream));
  }, py::arg("row_id"), py::arg("gate_sorted"), py::arg("M_recv"),
     py::arg("stream") = py::none(),
     "Gather sorted row_id and gate (requires prior dispatch_meta_bf16)");

  m.def("gather_from_pad_bf16", [](uintptr_t in_pad_ptr, uintptr_t out_sorted_ptr, int M_recv, int H, py::object stream) {
    rdep_gather_from_pad_bf16(
        reinterpret_cast<const void*>(in_pad_ptr),
        reinterpret_cast<void*>(out_sorted_ptr),
        M_recv, H,
        to_stream(stream));
  }, py::arg("in_pad"), py::arg("out_sorted"), py::arg("M_recv"), py::arg("H"),
     py::arg("stream") = py::none(),
     "Gather BF16 rows from padded layout into sorted layout (requires prior dispatch_meta_bf16)");

  m.def("scatter_sorted_to_pad_bf16", [](uintptr_t in_sorted_ptr, uintptr_t out_pad_ptr, int M_recv, int H, py::object stream) {
    rdep_scatter_sorted_to_pad_bf16(
        reinterpret_cast<const void*>(in_sorted_ptr),
        reinterpret_cast<void*>(out_pad_ptr),
        M_recv, H,
        to_stream(stream));
  }, py::arg("in_sorted"), py::arg("out_pad"), py::arg("M_recv"), py::arg("H"),
     py::arg("stream") = py::none(),
     "Scatter BF16 rows from sorted layout into padded layout (requires prior dispatch_meta_bf16)");

  m.def("zero_padding_rows_bf16", [](uintptr_t out_pad_ptr, uintptr_t offs_pad_ptr,
                                     int M_recv, int M_pad, int H, py::object stream) {
    rdep_zero_padding_rows_bf16(
        reinterpret_cast<void*>(out_pad_ptr),
        reinterpret_cast<const int*>(offs_pad_ptr),
        M_recv, M_pad, H,
        to_stream(stream));
  }, py::arg("out_pad"), py::arg("offs_pad"), py::arg("M_recv"), py::arg("M_pad"), py::arg("H"),
     py::arg("stream") = py::none(),
     "Zero only per-expert padded rows in BF16 padded layout using dispatch offsets.");

  m.def("return_scatter_from_pad_bf16", [](uintptr_t Ye_pad_ptr, uintptr_t out_ptr,
                                          int M_recv, int T, int K,
                                          py::object stream) {
    rdep_return_scatter_from_pad_bf16(
        reinterpret_cast<const void*>(Ye_pad_ptr),
        reinterpret_cast<void*>(out_ptr),
        M_recv, T, K, to_stream(stream));
  }, py::arg("Ye_pad"), py::arg("out"),
     py::arg("M_recv"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Scatter expert outputs from padded layout Ye_pad[M_pad,H] back to tokens (BF16 path).");

  // ========== BF16 Backward Helpers (single-GPU milestone) ==========
  m.def("scatter_gate_bf16", [](uintptr_t dGate_sorted_ptr, uintptr_t row_id_ptr,
                               uintptr_t dGates_tk_ptr,
                               int M, int T, int K,
                               py::object stream) {
    rdep_scatter_gate_bf16(
        reinterpret_cast<const float*>(dGate_sorted_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<float*>(dGates_tk_ptr),
        M, T, K,
        to_stream(stream));
  }, py::arg("dGate_sorted"), py::arg("row_id"), py::arg("dGates_tk"),
     py::arg("M"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Scatter dGate[M] back to [T,K] (BF16 path)");

  m.def("scatter_gate_bf16_out_bf16", [](uintptr_t dGate_sorted_ptr, uintptr_t row_id_ptr,
                                         uintptr_t dGates_tk_ptr,
                                         int M, int T, int K,
                                         py::object stream) {
    rdep_scatter_gate_bf16_out_bf16(
        reinterpret_cast<const float*>(dGate_sorted_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dGates_tk_ptr),
        M, T, K,
        to_stream(stream));
  }, py::arg("dGate_sorted"), py::arg("row_id"), py::arg("dGates_tk"),
     py::arg("M"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Scatter dGate[M] back to [T,K] directly as BF16.");

  // SonicMoE dGate identity: gather dY with gate scaling, defer dGate
  m.def("gather_dy_nogate_bf16", [](uintptr_t dY_ptr, uintptr_t row_id_ptr,
                                    uintptr_t gate_ptr, uintptr_t dYe_out_ptr,
                                    int M, int T, int H, int K,
                                    py::object stream) {
    rdep_gather_dy_nogate_bf16(
        reinterpret_cast<const void*>(dY_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(gate_ptr),
        reinterpret_cast<void*>(dYe_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dY"), py::arg("row_id"), py::arg("gate"), py::arg("dYe_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Gather dY[T,H] -> dYe[M,H] with gate scaling (defer dGate to dgate_from_adA)");

  // SonicMoE dGate identity: dGate = ⟨A, dA'⟩ instead of ⟨dOut, Ye⟩
  m.def("dgate_from_adA_bf16", [](uintptr_t A_pad_ptr, uintptr_t dA_pad_ptr,
                                  uintptr_t dGate_sorted_ptr,
                                  int M, int D,
                                  py::object stream) {
    rdep_dgate_from_adA_bf16(
        reinterpret_cast<const void*>(A_pad_ptr),
        reinterpret_cast<const void*>(dA_pad_ptr),
        reinterpret_cast<float*>(dGate_sorted_ptr),
        M, D,
        to_stream(stream));
  }, py::arg("A_pad"), py::arg("dA_pad"), py::arg("dGate_sorted"),
     py::arg("M"), py::arg("D"),
     py::arg("stream") = py::none(),
     "SonicMoE dGate: dGate[i] = sum_d(A[pad_i,d] * dA[pad_i,d])");

  // SonicMoE distributed: gather dYe across ranks with gate scaling (no dGate)
  m.def("gather_dy_nogate_dist_bf16", [](uintptr_t dY_ptr, uintptr_t eids_ptr,
                                         uintptr_t row_id_ptr, uintptr_t gate_ptr,
                                         uintptr_t dYe_ptr,
                                         int M, int T, int H, int K,
                                         py::object stream) {
    rdep_gather_dy_nogate_dist_bf16(
        reinterpret_cast<const void*>(dY_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(gate_ptr),
        reinterpret_cast<void*>(dYe_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dY"), py::arg("eids"), py::arg("row_id"), py::arg("gate"), py::arg("dYe_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "SonicMoE distributed: gather dYe with gate scaling (defer dGate to send_dgate_dist)");

  // SonicMoE distributed: send locally-computed dGate back to source ranks
  m.def("send_dgate_dist_bf16", [](uintptr_t row_id_ptr, uintptr_t dGate_sorted_ptr,
                                   uintptr_t dGates_tk_ptr,
                                   int M, int T, int K,
                                   py::object stream) {
    rdep_send_dgate_dist_bf16(
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(dGate_sorted_ptr),
        reinterpret_cast<float*>(dGates_tk_ptr),
        M, T, K,
        to_stream(stream));
  }, py::arg("row_id"), py::arg("dGate_sorted"), py::arg("dGates_tk"),
     py::arg("M"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "SonicMoE distributed: send dGate to source ranks via IPC");

  m.def("send_dgate_dist_bf16_out_bf16", [](uintptr_t row_id_ptr, uintptr_t dGate_sorted_ptr,
                                            uintptr_t dGates_tk_ptr,
                                            int M, int T, int K,
                                            py::object stream) {
    rdep_send_dgate_dist_bf16_out_bf16(
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(dGate_sorted_ptr),
        reinterpret_cast<void*>(dGates_tk_ptr),
        M, T, K,
        to_stream(stream));
  }, py::arg("row_id"), py::arg("dGate_sorted"), py::arg("dGates_tk"),
     py::arg("M"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "SonicMoE distributed: send dGate and collect [T,K] directly as BF16.");

  m.def("scatter_dx_bf16_internal", [](uintptr_t dXe_pad_ptr, uintptr_t row_id_ptr,
                                      uintptr_t dX_out_ptr,
                                      int M, int T, int H, int K,
                                      py::object stream) {
    rdep_scatter_dx_bf16_internal(
        reinterpret_cast<const void*>(dXe_pad_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dX_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dXe_pad"), py::arg("row_id"), py::arg("dX_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Backward scatter: dXe_pad[M_pad,H] -> dX[T,H] using internal dest mapping (float32 accum)");

  m.def("gather_dy_dist_bf16", [](uintptr_t dY_ptr, uintptr_t eids_ptr, uintptr_t Ye_ptr,
                                 uintptr_t row_id_ptr, uintptr_t gate_ptr,
                                 uintptr_t dYe_ptr, uintptr_t dGate_sorted_ptr, uintptr_t dGates_tk_ptr,
                                 int M, int T, int H, int K,
                                 py::object stream) {
    rdep_gather_dy_dist_bf16(
        reinterpret_cast<const void*>(dY_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const void*>(Ye_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(gate_ptr),
        reinterpret_cast<void*>(dYe_ptr),
        reinterpret_cast<float*>(dGate_sorted_ptr),
        reinterpret_cast<float*>(dGates_tk_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dY"), py::arg("eids"), py::arg("Ye"),
     py::arg("row_id"), py::arg("gate_sorted"),
     py::arg("dYe_out"), py::arg("dGate_sorted_out"), py::arg("dGates_tk_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Distributed backward gather (IPC+hybrid): push-stages dY, computes dYe, returns dGate to [T,K]");

  m.def("scatter_dx_dist_bf16", [](uintptr_t dXe_sorted_ptr, uintptr_t row_id_ptr,
                                  uintptr_t dX_out_ptr,
                                  int M, int T, int H, int K,
                                  py::object stream) {
    rdep_scatter_dx_dist_bf16(
        reinterpret_cast<const void*>(dXe_sorted_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dX_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dXe_sorted"), py::arg("row_id"), py::arg("dX_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Distributed backward scatter (IPC+hybrid): sends dXe rows to sources and reduces into dX[T,H]");

  m.def("scatter_dx_dist_from_pad_bf16", [](uintptr_t dXe_pad_ptr, uintptr_t row_id_ptr,
                                            uintptr_t dX_out_ptr,
                                            int M, int T, int H, int K,
                                            py::object stream) {
    rdep_scatter_dx_dist_from_pad_bf16(
        reinterpret_cast<const void*>(dXe_pad_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dX_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dXe_pad"), py::arg("row_id"), py::arg("dX_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Distributed backward scatter (IPC-only fast path): sends dXe directly from padded layout and reduces into dX[T,H].");

  // ========== Blockscaled Path ==========
  m.def("alloc_blockscaled", [](size_t capacity, int H, int n_local, int profile) {
    rdep_alloc_blockscaled(capacity, H, n_local, profile);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"), py::arg("profile"),
     "Allocate blockscaled dispatch buffers (profile: 0=fp8, 1=nvfp4)");

  m.def("get_dropped_bf16", [](py::object stream) {
    return rdep_get_dropped_bf16(to_stream(stream));
  }, py::arg("stream") = py::none(),
     "Get count of tokens dropped due to capacity overflow (BF16 path)");

  m.def("get_dropped_blockscaled", [](py::object stream) {
    return rdep_get_dropped_blockscaled(to_stream(stream));
  }, py::arg("stream") = py::none(),
     "Get count of tokens dropped due to capacity overflow (blockscaled path)");

  m.def("dispatch_meta_blockscaled", [](uintptr_t x_ptr, uintptr_t eids_ptr, uintptr_t gates_ptr,
                                       int T, int K,
                                       uintptr_t offs_pad_ptr, uintptr_t M_pad_ptr,
                                       py::object stream) {
    return rdep_dispatch_meta_blockscaled(
        reinterpret_cast<const void*>(x_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const float*>(gates_ptr),
        T, K,
        reinterpret_cast<int*>(offs_pad_ptr),
        reinterpret_cast<int*>(M_pad_ptr),
        to_stream(stream));
  }, py::arg("x"), py::arg("eids"), py::arg("gates"),
     py::arg("T"), py::arg("K"),
     py::arg("offs_pad"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Dispatch blockscaled tokens (meta only): quantized dispatch + sort + padded mapping. "
     "NOTE: `M_pad` is treated as a pinned host scratch (used to read back M_recv/M_pad).");

  m.def("gather_xe_blockscaled", [](uintptr_t Xe_q_ptr, uintptr_t Xe_sf_ptr,
                                   int M_recv, int M_pad,
                                   py::object stream) {
    rdep_gather_xe_blockscaled(
        reinterpret_cast<void*>(Xe_q_ptr),
        reinterpret_cast<void*>(Xe_sf_ptr),
        M_recv,
        M_pad,
        to_stream(stream));
  }, py::arg("Xe_q_out"), py::arg("Xe_sf_out"),
     py::arg("M_recv"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Gather blockscaled activations into padded expert layout: Xe_q[M_pad,Hp] + Xe_sf[M_pad,Hsf] (packed SF).");

  m.def("return_scatter_from_pad_blockscaled", [](uintptr_t Ye_pad_ptr, uintptr_t out_ptr,
                                                 int M_recv, int T, int K,
                                                 py::object stream) {
    rdep_return_scatter_from_pad_blockscaled(
        reinterpret_cast<const void*>(Ye_pad_ptr),
        reinterpret_cast<void*>(out_ptr),
        M_recv, T, K,
        to_stream(stream));
  }, py::arg("Ye_pad"), py::arg("out"),
     py::arg("M_recv"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Scatter expert outputs from padded layout Ye_pad[M_pad,H] back to token output (blockscaled path).");

  // ========== Expert Optimizer (AdamW + cache emit) ==========
  m.def(
      "expert_adamw_step",
      [](int profile,
         uintptr_t W1_ptr, uintptr_t dW1_ptr, uintptr_t m1_ptr, uintptr_t v1_ptr,
         uintptr_t W3_ptr, uintptr_t dW3_ptr, uintptr_t m3_ptr, uintptr_t v3_ptr,
         uintptr_t W2_ptr, uintptr_t dW2_ptr, uintptr_t m2_ptr, uintptr_t v2_ptr,
         uintptr_t W13_q_ptr, uintptr_t W13_sf_ptr,
         uintptr_t W2_q_ptr, uintptr_t W2_sf_ptr,
         int E, int H, int Dff,
         float lr, float beta1, float beta2,
         float weight_decay, float eps,
         float step_size, float inv_bias_correction2_sqrt,
         py::object stream) {
        auto err = expert_adamw_step(
            profile,
            reinterpret_cast<void*>(W1_ptr), reinterpret_cast<const void*>(dW1_ptr),
            reinterpret_cast<void*>(m1_ptr), reinterpret_cast<void*>(v1_ptr),
            reinterpret_cast<void*>(W3_ptr), reinterpret_cast<const void*>(dW3_ptr),
            reinterpret_cast<void*>(m3_ptr), reinterpret_cast<void*>(v3_ptr),
            reinterpret_cast<void*>(W2_ptr), reinterpret_cast<const void*>(dW2_ptr),
            reinterpret_cast<void*>(m2_ptr), reinterpret_cast<void*>(v2_ptr),
            reinterpret_cast<void*>(W13_q_ptr), reinterpret_cast<void*>(W13_sf_ptr),
            reinterpret_cast<void*>(W2_q_ptr), reinterpret_cast<void*>(W2_sf_ptr),
            E, H, Dff,
            lr, beta1, beta2,
            weight_decay, eps,
            step_size, inv_bias_correction2_sqrt,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("expert_adamw_step failed");
      },
      py::arg("profile"),
      py::arg("W1"),
      py::arg("dW1"),
      py::arg("m1"),
      py::arg("v1"),
      py::arg("W3"),
      py::arg("dW3"),
      py::arg("m3"),
      py::arg("v3"),
      py::arg("W2"),
      py::arg("dW2"),
      py::arg("m2"),
      py::arg("v2"),
      py::arg("W13_q"),
      py::arg("W13_sf_mma"),
      py::arg("W2_q"),
      py::arg("W2_sf_mma"),
      py::arg("E"),
      py::arg("H"),
      py::arg("Dff"),
      py::arg("lr"),
      py::arg("beta1"),
      py::arg("beta2"),
      py::arg("weight_decay"),
      py::arg("eps"),
      py::arg("step_size"),
      py::arg("inv_bias_correction2_sqrt"),
      py::arg("stream") = py::none(),
      "Fused expert AdamW update + packed weight cache emission (FP8/NVFP4)");

  // ========== Fused Dense AdamW (ZeRO-2 shard update, single kernel) ==========
  m.def(
      "fused_adamw_step_fp32",
      [](uintptr_t param_ptr, uintptr_t grad_ptr,
         uintptr_t exp_avg_ptr, uintptr_t exp_avg_sq_ptr,
         int N,
         float beta1, float beta2,
         float lr, float weight_decay, float eps,
         float step_size, float inv_bc2_sqrt,
         py::object stream) {
        auto err = fused_adamw_step_fp32(
            reinterpret_cast<void*>(param_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            reinterpret_cast<void*>(exp_avg_ptr),
            reinterpret_cast<void*>(exp_avg_sq_ptr),
            N, beta1, beta2, lr, weight_decay, eps,
            step_size, inv_bc2_sqrt,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("fused_adamw_step_fp32 failed: " + cuda_err(err));
      },
      py::arg("param"), py::arg("grad"),
      py::arg("exp_avg"), py::arg("exp_avg_sq"),
      py::arg("N"),
      py::arg("beta1"), py::arg("beta2"),
      py::arg("lr"), py::arg("weight_decay"), py::arg("eps"),
      py::arg("step_size"), py::arg("inv_bc2_sqrt"),
      py::arg("stream") = py::none(),
      "Fused dense AdamW step (FP32): single vectorized kernel replacing 9 PyTorch launches");

  m.def(
      "fused_adamw_step_bf16",
      [](uintptr_t param_ptr, uintptr_t grad_ptr,
         uintptr_t exp_avg_ptr, uintptr_t exp_avg_sq_ptr,
         int N,
         float beta1, float beta2,
         float lr, float weight_decay, float eps,
         float step_size, float inv_bc2_sqrt,
         py::object stream) {
        auto err = fused_adamw_step_bf16(
            reinterpret_cast<void*>(param_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            reinterpret_cast<void*>(exp_avg_ptr),
            reinterpret_cast<void*>(exp_avg_sq_ptr),
            N, beta1, beta2, lr, weight_decay, eps,
            step_size, inv_bc2_sqrt,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("fused_adamw_step_bf16 failed: " + cuda_err(err));
      },
      py::arg("param"), py::arg("grad"),
      py::arg("exp_avg"), py::arg("exp_avg_sq"),
      py::arg("N"),
      py::arg("beta1"), py::arg("beta2"),
      py::arg("lr"), py::arg("weight_decay"), py::arg("eps"),
      py::arg("step_size"), py::arg("inv_bc2_sqrt"),
      py::arg("stream") = py::none(),
      "Fused dense AdamW step (BF16): single vectorized kernel replacing 9 PyTorch launches");

  // ========== ECO Adam NVFP4 Update (fused optimizer for NVFP4 primary weights) ==========
  m.def(
      "eco_adam_nvfp4_update",
      [](uintptr_t W_packed_ptr, uintptr_t W_scale_ptr, uintptr_t W_gs_ptr,
         uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_data_ptr, uintptr_t v_scale_ptr,
         uintptr_t grad_ptr,
         int E, int out_dim, int in_dim, int group_size,
         float lr, float beta1, float beta2,
         float weight_decay, float eps,
         float step_size, float inv_bc2_sqrt,
         float eco_alpha,
         int stochastic_rounding, int error_feedback,
         unsigned int prng_seed0, unsigned int prng_seed1,
         py::object stream) {
        if (E <= 0 || out_dim <= 0 || in_dim <= 0 || group_size <= 0)
            throw std::invalid_argument("eco_adam: E, out_dim, in_dim, group_size must be positive");
        if ((out_dim & 31) != 0 || (in_dim & 31) != 0)
            throw std::invalid_argument("eco_adam: out_dim and in_dim must be multiples of 32");
        if (in_dim % group_size != 0)
            throw std::invalid_argument("eco_adam: in_dim must be divisible by group_size");
        auto err = eco_adam_nvfp4_update(
            reinterpret_cast<void*>(W_packed_ptr),
            reinterpret_cast<void*>(W_scale_ptr),
            reinterpret_cast<void*>(W_gs_ptr),
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_data_ptr),
            reinterpret_cast<void*>(v_scale_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps,
            step_size, inv_bc2_sqrt, eco_alpha,
            stochastic_rounding, error_feedback,
            prng_seed0, prng_seed1,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_adam_nvfp4_update failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("W_packed"), py::arg("W_scale"), py::arg("W_gs"),
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_data"), py::arg("v_scale"),
      py::arg("grad"),
      py::arg("E"), py::arg("out_dim"), py::arg("in_dim"), py::arg("group_size"),
      py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
      py::arg("weight_decay"), py::arg("eps"),
      py::arg("step_size"), py::arg("inv_bc2_sqrt"),
      py::arg("eco_alpha"),
      py::arg("stochastic_rounding"), py::arg("error_feedback"),
      py::arg("prng_seed0"), py::arg("prng_seed1"),
      py::arg("stream") = py::none(),
      "Fused ECO AdamW update for NVFP4 primary weights with FP8 optimizer states. "
      "Zero FP32 global memory materialization — all computation in registers/shared.");

  m.def(
      "eco_adam_nvfp4_update_bf16",
      [](uintptr_t W_packed_ptr, uintptr_t W_scale_ptr, uintptr_t W_gs_ptr,
         uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_data_ptr, uintptr_t v_scale_ptr,
         uintptr_t grad_ptr,
         int E, int out_dim, int in_dim, int group_size,
         float lr, float beta1, float beta2,
         float weight_decay, float eps,
         float step_size, float inv_bc2_sqrt,
         float eco_alpha,
         int stochastic_rounding, int error_feedback,
         unsigned int prng_seed0, unsigned int prng_seed1,
         py::object stream) {
        if (E <= 0 || out_dim <= 0 || in_dim <= 0 || group_size <= 0)
            throw std::invalid_argument("eco_adam_bf16: E, out_dim, in_dim, group_size must be positive");
        if ((out_dim & 31) != 0 || (in_dim & 31) != 0)
            throw std::invalid_argument("eco_adam_bf16: out_dim and in_dim must be multiples of 32");
        if (in_dim % group_size != 0)
            throw std::invalid_argument("eco_adam_bf16: in_dim must be divisible by group_size");
        auto err = eco_adam_nvfp4_update_bf16(
            reinterpret_cast<void*>(W_packed_ptr),
            reinterpret_cast<void*>(W_scale_ptr),
            reinterpret_cast<void*>(W_gs_ptr),
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_data_ptr),
            reinterpret_cast<void*>(v_scale_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps,
            step_size, inv_bc2_sqrt, eco_alpha,
            stochastic_rounding, error_feedback,
            prng_seed0, prng_seed1,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_adam_nvfp4_update_bf16 failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("W_packed"), py::arg("W_scale"), py::arg("W_gs"),
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_data"), py::arg("v_scale"),
      py::arg("grad"),
      py::arg("E"), py::arg("out_dim"), py::arg("in_dim"), py::arg("group_size"),
      py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
      py::arg("weight_decay"), py::arg("eps"),
      py::arg("step_size"), py::arg("inv_bc2_sqrt"),
      py::arg("eco_alpha"),
      py::arg("stochastic_rounding"), py::arg("error_feedback"),
      py::arg("prng_seed0"), py::arg("prng_seed1"),
      py::arg("stream") = py::none(),
      "Fused ECO AdamW update for NVFP4 primary weights accepting BF16 gradients.");

  // ========== ECO Adam NVFP4 Factored-V Update (factored second moment) ==========
  m.def(
      "eco_adam_nvfp4_fv_update",
      [](uintptr_t W_packed_ptr, uintptr_t W_scale_ptr, uintptr_t W_gs_ptr,
         uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_row_ptr, uintptr_t v_col_ptr, uintptr_t v_rms_ptr,
         uintptr_t grad_ptr,
         int E, int out_dim, int in_dim, int group_size,
         float lr, float beta1, float beta2,
         float weight_decay, float eps,
         float step_size, float inv_bc2_sqrt,
         float eco_alpha,
         int stochastic_rounding, int error_feedback,
         unsigned int prng_seed0, unsigned int prng_seed1,
         py::object stream) {
        if (E <= 0 || out_dim <= 0 || in_dim <= 0 || group_size <= 0)
            throw std::invalid_argument("eco_adam_fv: E, out_dim, in_dim, group_size must be positive");
        if ((out_dim & 31) != 0 || (in_dim & 31) != 0)
            throw std::invalid_argument("eco_adam_fv: out_dim and in_dim must be multiples of 32");
        if (in_dim % group_size != 0)
            throw std::invalid_argument("eco_adam_fv: in_dim must be divisible by group_size");
        auto err = eco_adam_nvfp4_fv_update(
            reinterpret_cast<void*>(W_packed_ptr),
            reinterpret_cast<void*>(W_scale_ptr),
            reinterpret_cast<void*>(W_gs_ptr),
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_row_ptr),
            reinterpret_cast<void*>(v_col_ptr),
            reinterpret_cast<void*>(v_rms_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps,
            step_size, inv_bc2_sqrt, eco_alpha,
            stochastic_rounding, error_feedback,
            prng_seed0, prng_seed1,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_adam_nvfp4_fv_update failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("W_packed"), py::arg("W_scale"), py::arg("W_gs"),
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_row"), py::arg("v_col"), py::arg("v_rms"),
      py::arg("grad"),
      py::arg("E"), py::arg("out_dim"), py::arg("in_dim"), py::arg("group_size"),
      py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
      py::arg("weight_decay"), py::arg("eps"),
      py::arg("step_size"), py::arg("inv_bc2_sqrt"),
      py::arg("eco_alpha"),
      py::arg("stochastic_rounding"), py::arg("error_feedback"),
      py::arg("prng_seed0"), py::arg("prng_seed1"),
      py::arg("stream") = py::none(),
      "Fused ECO AdamW update with Adafactor-style factored second moment (v_row/v_col). "
      "Runs factored-v reduction kernels + modified main kernel in a single call.");

  m.def(
      "eco_adam_nvfp4_fv_update_bf16",
      [](uintptr_t W_packed_ptr, uintptr_t W_scale_ptr, uintptr_t W_gs_ptr,
         uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_row_ptr, uintptr_t v_col_ptr, uintptr_t v_rms_ptr,
         uintptr_t grad_ptr,
         int E, int out_dim, int in_dim, int group_size,
         float lr, float beta1, float beta2,
         float weight_decay, float eps,
         float step_size, float inv_bc2_sqrt,
         float eco_alpha,
         int stochastic_rounding, int error_feedback,
         unsigned int prng_seed0, unsigned int prng_seed1,
         py::object stream) {
        if (E <= 0 || out_dim <= 0 || in_dim <= 0 || group_size <= 0)
            throw std::invalid_argument("eco_adam_fv_bf16: E, out_dim, in_dim, group_size must be positive");
        if ((out_dim & 31) != 0 || (in_dim & 31) != 0)
            throw std::invalid_argument("eco_adam_fv_bf16: out_dim and in_dim must be multiples of 32");
        if (in_dim % group_size != 0)
            throw std::invalid_argument("eco_adam_fv_bf16: in_dim must be divisible by group_size");
        auto err = eco_adam_nvfp4_fv_update_bf16(
            reinterpret_cast<void*>(W_packed_ptr),
            reinterpret_cast<void*>(W_scale_ptr),
            reinterpret_cast<void*>(W_gs_ptr),
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_row_ptr),
            reinterpret_cast<void*>(v_col_ptr),
            reinterpret_cast<void*>(v_rms_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, out_dim, in_dim, group_size,
            lr, beta1, beta2, weight_decay, eps,
            step_size, inv_bc2_sqrt, eco_alpha,
            stochastic_rounding, error_feedback,
            prng_seed0, prng_seed1,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_adam_nvfp4_fv_update_bf16 failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("W_packed"), py::arg("W_scale"), py::arg("W_gs"),
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_row"), py::arg("v_col"), py::arg("v_rms"),
      py::arg("grad"),
      py::arg("E"), py::arg("out_dim"), py::arg("in_dim"), py::arg("group_size"),
      py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
      py::arg("weight_decay"), py::arg("eps"),
      py::arg("step_size"), py::arg("inv_bc2_sqrt"),
      py::arg("eco_alpha"),
      py::arg("stochastic_rounding"), py::arg("error_feedback"),
      py::arg("prng_seed0"), py::arg("prng_seed1"),
      py::arg("stream") = py::none(),
      "Fused ECO AdamW factored-v update accepting BF16 gradients.");

  // ========== AdamA m/v Accumulation (replaces gradient accumulation buffers) ==========
  m.def(
      "eco_mv_accumulate",
      [](uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_data_ptr, uintptr_t v_scale_ptr,
         uintptr_t grad_ptr,
         int E, int in_dim, int out_dim,
         float beta1_frac, float beta2_frac,
         py::object stream) {
        if (E <= 0 || in_dim <= 0 || out_dim <= 0)
            throw std::invalid_argument("eco_mv_accumulate: E, in_dim, out_dim must be positive");
        auto err = eco_mv_accumulate(
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_data_ptr),
            reinterpret_cast<void*>(v_scale_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, in_dim, out_dim,
            beta1_frac, beta2_frac,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_mv_accumulate failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_data"), py::arg("v_scale"),
      py::arg("grad"),
      py::arg("E"), py::arg("in_dim"), py::arg("out_dim"),
      py::arg("beta1_frac"), py::arg("beta2_frac"),
      py::arg("stream") = py::none(),
      "AdamA m/v accumulation: update FP8 m (E5M2) and v (E4M3) with fractional betas. "
      "Zero additional memory — eliminates ~5.85 GiB gradient accumulation buffers.");

  m.def(
      "eco_mv_accumulate_bf16",
      [](uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_data_ptr, uintptr_t v_scale_ptr,
         uintptr_t grad_ptr,
         int E, int in_dim, int out_dim,
         float beta1_frac, float beta2_frac,
         py::object stream) {
        if (E <= 0 || in_dim <= 0 || out_dim <= 0)
            throw std::invalid_argument("eco_mv_accumulate_bf16: E, in_dim, out_dim must be positive");
        auto err = eco_mv_accumulate_bf16(
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_data_ptr),
            reinterpret_cast<void*>(v_scale_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, in_dim, out_dim,
            beta1_frac, beta2_frac,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_mv_accumulate_bf16 failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_data"), py::arg("v_scale"),
      py::arg("grad"),
      py::arg("E"), py::arg("in_dim"), py::arg("out_dim"),
      py::arg("beta1_frac"), py::arg("beta2_frac"),
      py::arg("stream") = py::none(),
      "AdamA m/v accumulation accepting BF16 gradients.");

  m.def(
      "eco_mv_accumulate_fv",
      [](uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_row_ptr, uintptr_t v_col_ptr, uintptr_t v_rms_ptr,
         uintptr_t grad_ptr,
         int E, int in_dim, int out_dim,
         float beta1_frac, float beta2_frac,
         py::object stream) {
        if (E <= 0 || in_dim <= 0 || out_dim <= 0)
            throw std::invalid_argument("eco_mv_accumulate_fv: E, in_dim, out_dim must be positive");
        auto err = eco_mv_accumulate_fv(
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_row_ptr),
            reinterpret_cast<void*>(v_col_ptr),
            reinterpret_cast<void*>(v_rms_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, in_dim, out_dim,
            beta1_frac, beta2_frac,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_mv_accumulate_fv failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_row"), py::arg("v_col"), py::arg("v_rms"),
      py::arg("grad"),
      py::arg("E"), py::arg("in_dim"), py::arg("out_dim"),
      py::arg("beta1_frac"), py::arg("beta2_frac"),
      py::arg("stream") = py::none(),
      "AdamA m/v accumulation with factored second moment: update FP8 m (E5M2) + "
      "factored v_row/v_col with fractional betas. Zero additional memory.");

  m.def(
      "eco_mv_accumulate_fv_bf16",
      [](uintptr_t m_data_ptr, uintptr_t m_scale_ptr,
         uintptr_t v_row_ptr, uintptr_t v_col_ptr, uintptr_t v_rms_ptr,
         uintptr_t grad_ptr,
         int E, int in_dim, int out_dim,
         float beta1_frac, float beta2_frac,
         py::object stream) {
        if (E <= 0 || in_dim <= 0 || out_dim <= 0)
            throw std::invalid_argument("eco_mv_accumulate_fv_bf16: E, in_dim, out_dim must be positive");
        auto err = eco_mv_accumulate_fv_bf16(
            reinterpret_cast<void*>(m_data_ptr),
            reinterpret_cast<void*>(m_scale_ptr),
            reinterpret_cast<void*>(v_row_ptr),
            reinterpret_cast<void*>(v_col_ptr),
            reinterpret_cast<void*>(v_rms_ptr),
            reinterpret_cast<const void*>(grad_ptr),
            E, in_dim, out_dim,
            beta1_frac, beta2_frac,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("eco_mv_accumulate_fv_bf16 failed: " + std::string(cudaGetErrorString(err)));
      },
      py::arg("m_data"), py::arg("m_scale"),
      py::arg("v_row"), py::arg("v_col"), py::arg("v_rms"),
      py::arg("grad"),
      py::arg("E"), py::arg("in_dim"), py::arg("out_dim"),
      py::arg("beta1_frac"), py::arg("beta2_frac"),
      py::arg("stream") = py::none(),
      "AdamA m/factored-v accumulation accepting BF16 gradients.");

  // ========== Quantization + Swizzle (dense / weight cache only) ==========
  m.def("quant_fp8", [](uintptr_t x_ptr, int ldx,
                        uintptr_t out_ptr, int ld_out,
                        uintptr_t sfa_ptr, int ld_sf,
                        int M, int K, py::object stream) {
    auto err = quant_fp8(reinterpret_cast<const void*>(x_ptr), ldx,
                         reinterpret_cast<void*>(out_ptr), ld_out,
                         reinterpret_cast<void*>(sfa_ptr), ld_sf,
                         M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_fp8 failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
      "BF16 -> FP8 (packed u16) with SFA (rowwise, sf_vec=32)");

  m.def("quant_nvfp4", [](uintptr_t x_ptr, int ldx,
                           uintptr_t out_ptr, int ld_out,
                           uintptr_t sfa_ptr, int ld_sf,
                           int M, int K, py::object stream) {
    auto err = quant_nvfp4(reinterpret_cast<const void*>(x_ptr), ldx,
                           reinterpret_cast<void*>(out_ptr), ld_out,
                           reinterpret_cast<void*>(sfa_ptr), ld_sf,
                           M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_nvfp4 failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> NVFP4 (packed u16) with SFA (rowwise, sf_vec=32)");

  m.def("quant_fp8_sf_strided_mma", [](uintptr_t x_ptr, int ldx,
                                       uintptr_t out_ptr, int ld_out,
                                       uintptr_t sf_mma_ptr,
                                       uintptr_t offs_ptr,
                                       int E, int M_e_stride,
                                       int M_pad, int K,
                                       py::object stream) {
    auto err = quant_fp8_sf_strided_mma(reinterpret_cast<const void*>(x_ptr), ldx,
                                        reinterpret_cast<void*>(out_ptr), ld_out,
                                        reinterpret_cast<void*>(sf_mma_ptr),
                                        reinterpret_cast<const int32_t*>(offs_ptr),
                                        E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_fp8_sf_strided_mma failed");
  }, py::arg("x"), py::arg("ldx"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> FP8 (packed u16) with SFA written directly to per-expert MMA layout");

  m.def("quant_nvfp4_sf_strided_mma", [](uintptr_t x_ptr, int ldx,
                                         uintptr_t out_ptr, int ld_out,
                                         uintptr_t sf_mma_ptr,
                                         uintptr_t offs_ptr,
                                         int E, int M_e_stride,
                                         int M_pad, int K,
                                         py::object stream) {
    auto err = quant_nvfp4_sf_strided_mma(reinterpret_cast<const void*>(x_ptr), ldx,
                                          reinterpret_cast<void*>(out_ptr), ld_out,
                                          reinterpret_cast<void*>(sf_mma_ptr),
                                          reinterpret_cast<const int32_t*>(offs_ptr),
                                          E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_nvfp4_sf_strided_mma failed");
  }, py::arg("x"), py::arg("ldx"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> NVFP4 (packed u16) with SFA written directly to per-expert MMA layout");

  // Direct CT NVFP4 → blockscaled MMA (single source, no interleave)
  m.def("ct_nvfp4_to_blockscaled_mma", [](
      uintptr_t ct_packed_ptr, uintptr_t ct_scale_ptr, uintptr_t ct_gs_ptr,
      uintptr_t out_ptr, uintptr_t sf_mma_ptr,
      int E, int M_ct, int K_in, int M_stride, int group_size,
      py::object stream) {
    auto err = ct_nvfp4_to_blockscaled_mma(
        reinterpret_cast<const void*>(ct_packed_ptr),
        reinterpret_cast<const void*>(ct_scale_ptr),
        reinterpret_cast<const void*>(ct_gs_ptr),
        reinterpret_cast<void*>(out_ptr),
        reinterpret_cast<void*>(sf_mma_ptr),
        E, M_ct, K_in, M_stride, group_size,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("ct_nvfp4_to_blockscaled_mma failed");
  }, py::arg("ct_packed"), py::arg("ct_scale"), py::arg("ct_gs"),
     py::arg("out"), py::arg("sf_mma"),
     py::arg("E"), py::arg("M_ct"), py::arg("K_in"), py::arg("M_stride"), py::arg("group_size"),
     py::arg("stream") = py::none(),
     "Direct compressed_tensors NVFP4 -> blockscaled MMA (single source)");

  // Direct CT NVFP4 → blockscaled MMA (dual source, interleaved W1/W3)
  m.def("ct_nvfp4_to_blockscaled_mma_interleaved", [](
      uintptr_t ct_packed_a_ptr, uintptr_t ct_scale_a_ptr, uintptr_t ct_gs_a_ptr,
      uintptr_t ct_packed_b_ptr, uintptr_t ct_scale_b_ptr, uintptr_t ct_gs_b_ptr,
      uintptr_t out_ptr, uintptr_t sf_mma_ptr,
      int E, int M_ct, int K_in, int M_stride, int group_size,
      py::object stream) {
    auto err = ct_nvfp4_to_blockscaled_mma_interleaved(
        reinterpret_cast<const void*>(ct_packed_a_ptr),
        reinterpret_cast<const void*>(ct_scale_a_ptr),
        reinterpret_cast<const void*>(ct_gs_a_ptr),
        reinterpret_cast<const void*>(ct_packed_b_ptr),
        reinterpret_cast<const void*>(ct_scale_b_ptr),
        reinterpret_cast<const void*>(ct_gs_b_ptr),
        reinterpret_cast<void*>(out_ptr),
        reinterpret_cast<void*>(sf_mma_ptr),
        E, M_ct, K_in, M_stride, group_size,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("ct_nvfp4_to_blockscaled_mma_interleaved failed");
  }, py::arg("ct_packed_a"), py::arg("ct_scale_a"), py::arg("ct_gs_a"),
     py::arg("ct_packed_b"), py::arg("ct_scale_b"), py::arg("ct_gs_b"),
     py::arg("out"), py::arg("sf_mma"),
     py::arg("E"), py::arg("M_ct"), py::arg("K_in"), py::arg("M_stride"), py::arg("group_size"),
     py::arg("stream") = py::none(),
     "Direct compressed_tensors NVFP4 -> blockscaled MMA (dual source W1/W3 interleaved)");

  // Direct CT NVFP4 → BF16 dequantization on GPU
  m.def("ct_nvfp4_to_bf16", [](
      uintptr_t ct_packed_ptr, uintptr_t ct_scale_ptr, uintptr_t ct_gs_ptr,
      uintptr_t out_ptr,
      int M, int K, int group_size,
      int gs_stride, int expert_rows, int transpose,
      py::object stream) {
    auto err = ct_nvfp4_to_bf16(
        reinterpret_cast<const void*>(ct_packed_ptr),
        reinterpret_cast<const void*>(ct_scale_ptr),
        reinterpret_cast<const void*>(ct_gs_ptr),
        reinterpret_cast<void*>(out_ptr),
        M, K, group_size, gs_stride, expert_rows, transpose,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("ct_nvfp4_to_bf16 failed");
  }, py::arg("ct_packed"), py::arg("ct_scale"), py::arg("ct_gs"),
     py::arg("out"),
     py::arg("M"), py::arg("K"), py::arg("group_size"),
     py::arg("gs_stride") = 0, py::arg("expert_rows") = 0, py::arg("transpose") = 0,
     py::arg("stream") = py::none(),
     "Direct compressed_tensors NVFP4 -> BF16 on GPU (no CPU dequant)");
  // 0=normal, 1=legacy [K,M], 2=expert-major [E,K,M]
  m.attr("ct_nvfp4_to_bf16_max_transpose_mode") = py::int_(2);

  m.def("swiglu_bwd_bf16", [](uintptr_t h1_ptr, int ld_h1,
                              uintptr_t h3_ptr, int ld_h3,
                              uintptr_t dA_ptr, int ld_dA,
                              uintptr_t A_out_ptr, int ld_A,
                              uintptr_t dH1_out_ptr, int ld_dH1,
                              uintptr_t dH3_out_ptr, int ld_dH3,
                              int M, int K, py::object stream) {
    auto err = swiglu_bwd_bf16(reinterpret_cast<const void*>(h1_ptr), ld_h1,
                               reinterpret_cast<const void*>(h3_ptr), ld_h3,
                               reinterpret_cast<const void*>(dA_ptr), ld_dA,
                               reinterpret_cast<void*>(A_out_ptr), ld_A,
                               reinterpret_cast<void*>(dH1_out_ptr), ld_dH1,
                               reinterpret_cast<void*>(dH3_out_ptr), ld_dH3,
                               M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_bwd_bf16 failed");
  }, py::arg("h1"), py::arg("ld_h1"),
     py::arg("h3"), py::arg("ld_h3"),
     py::arg("dA"), py::arg("ld_dA"),
     py::arg("A_out"), py::arg("ld_A"),
     py::arg("dH1_out"), py::arg("ld_dH1"),
     py::arg("dH3_out"), py::arg("ld_dH3"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "SwiGLU backward (BF16): (h1, h3, dA) -> (A, dH1, dH3)");

  m.attr("bf16_wgrad_device_offs_supported") = py::bool_(true);
  m.attr("bf16_prepare_offs_pad_host_supported") = py::bool_(true);
  m.attr("bf16_wgrad_host_offs_supported") = py::bool_(true);

  m.def("bf16_prepare_offs_pad_host", [](uintptr_t offs_pad_ptr, int E, py::object stream) -> uintptr_t {
    const int32_t* host_ptr = nullptr;
    auto err = bf16_prepare_offs_pad_host(
        reinterpret_cast<const int32_t*>(offs_pad_ptr),
        E,
        to_stream(stream),
        &host_ptr);
    if (err != cudaSuccess) throw std::runtime_error("bf16_prepare_offs_pad_host failed");
    return reinterpret_cast<uintptr_t>(host_ptr);
  }, py::arg("offs_pad"),
     py::arg("E"),
     py::arg("stream") = py::none(),
     "Prepare and return host-visible offs_pad pointer (pinned scratch) for grouped BF16 wgrad launch metadata.");

  m.def("bf16_wgrad_w2_cublaslt", [](uintptr_t A_ptr,
                                     uintptr_t dY_ptr,
                                     uintptr_t dW2_out_ptr,
                                     uintptr_t offs_pad_ptr,
                                     int E, int H, int Dff,
                                     py::object stream) {
    auto err = bf16_wgrad_w2_cublaslt(reinterpret_cast<const void*>(A_ptr),
                                      reinterpret_cast<const void*>(dY_ptr),
                                      reinterpret_cast<void*>(dW2_out_ptr),
                                      reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                      E, H, Dff,
                                      to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w2_cublaslt failed");
  }, py::arg("A"),
     py::arg("dY"),
     py::arg("dW2_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 wgrad for W2 via cuBLAS grouped batched GEMM: dW2 = A^T @ dY (FP32 accum). offs_pad accepts device or host int32 pointer.");

  m.def("bf16_wgrad_w2_cublaslt_host_offs", [](uintptr_t A_ptr,
                                               uintptr_t dY_ptr,
                                               uintptr_t dW2_out_ptr,
                                               uintptr_t offs_pad_host_ptr,
                                               int E, int H, int Dff,
                                               py::object stream) {
    auto err = bf16_wgrad_w2_cublaslt_host_offs(reinterpret_cast<const void*>(A_ptr),
                                                reinterpret_cast<const void*>(dY_ptr),
                                                reinterpret_cast<void*>(dW2_out_ptr),
                                                reinterpret_cast<const int32_t*>(offs_pad_host_ptr),
                                                E, H, Dff,
                                                to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w2_cublaslt_host_offs failed");
  }, py::arg("A"),
     py::arg("dY"),
     py::arg("dW2_out"),
     py::arg("offs_pad_host"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 wgrad for W2 using pre-staged host offs_pad pointer (no pointer introspection on launch path).");

  m.def("bf16_wgrad_w13_cublaslt", [](uintptr_t X_ptr,
                                      uintptr_t dH_ptr,
                                      uintptr_t dW_out_ptr,
                                      uintptr_t offs_pad_ptr,
                                      int E, int H, int Dff,
                                      py::object stream) {
    auto err = bf16_wgrad_w13_cublaslt(reinterpret_cast<const void*>(X_ptr),
                                       reinterpret_cast<const void*>(dH_ptr),
                                       reinterpret_cast<void*>(dW_out_ptr),
                                       reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                       E, H, Dff,
                                       to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w13_cublaslt failed");
  }, py::arg("X"),
     py::arg("dH"),
     py::arg("dW_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 wgrad via cuBLAS grouped batched GEMM: dW = X^T @ dH (FP32 accum). offs_pad accepts device or host int32 pointer.");

  m.def("bf16_wgrad_w13_pair_cublaslt", [](uintptr_t X_ptr,
                                           uintptr_t dH1_ptr,
                                           uintptr_t dH3_ptr,
                                           uintptr_t dW1_out_ptr,
                                           uintptr_t dW3_out_ptr,
                                           uintptr_t offs_pad_ptr,
                                           int E, int H, int Dff,
                                           py::object stream) {
    auto err = bf16_wgrad_w13_pair_cublaslt(reinterpret_cast<const void*>(X_ptr),
                                            reinterpret_cast<const void*>(dH1_ptr),
                                            reinterpret_cast<const void*>(dH3_ptr),
                                            reinterpret_cast<void*>(dW1_out_ptr),
                                            reinterpret_cast<void*>(dW3_out_ptr),
                                            reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                            E, H, Dff,
                                            to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w13_pair_cublaslt failed");
  }, py::arg("X"),
     py::arg("dH1"),
     py::arg("dH3"),
     py::arg("dW1_out"),
     py::arg("dW3_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 paired wgrad via cuBLAS grouped batched GEMM: (dW1,dW3) = X^T @ (dH1,dH3) (FP32 accum). offs_pad accepts device or host int32 pointer.");

  m.def("bf16_wgrad_w13_pair_cublaslt_host_offs", [](uintptr_t X_ptr,
                                                     uintptr_t dH1_ptr,
                                                     uintptr_t dH3_ptr,
                                                     uintptr_t dW1_out_ptr,
                                                     uintptr_t dW3_out_ptr,
                                                     uintptr_t offs_pad_host_ptr,
                                                     int E, int H, int Dff,
                                                     py::object stream) {
    auto err = bf16_wgrad_w13_pair_cublaslt_host_offs(reinterpret_cast<const void*>(X_ptr),
                                                      reinterpret_cast<const void*>(dH1_ptr),
                                                      reinterpret_cast<const void*>(dH3_ptr),
                                                      reinterpret_cast<void*>(dW1_out_ptr),
                                                      reinterpret_cast<void*>(dW3_out_ptr),
                                                      reinterpret_cast<const int32_t*>(offs_pad_host_ptr),
                                                      E, H, Dff,
                                                      to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w13_pair_cublaslt_host_offs failed");
  }, py::arg("X"),
     py::arg("dH1"),
     py::arg("dH3"),
     py::arg("dW1_out"),
     py::arg("dW3_out"),
     py::arg("offs_pad_host"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 paired wgrad using pre-staged host offs_pad pointer (no pointer introspection on launch path).");

  m.def("swiglu_quant_fp8_sf_strided_mma", [](uintptr_t h13_ptr, int ld_h13,
                                              uintptr_t out_ptr, int ld_out,
                                              uintptr_t sf_mma_ptr,
                                              uintptr_t offs_ptr,
                                              int E, int M_e_stride,
                                              int M_pad, int K,
                                              py::object stream) {
    auto err = swiglu_quant_fp8_sf_strided_mma(reinterpret_cast<const void*>(h13_ptr), ld_h13,
                                               reinterpret_cast<void*>(out_ptr), ld_out,
                                               reinterpret_cast<void*>(sf_mma_ptr),
                                               reinterpret_cast<const int32_t*>(offs_ptr),
                                               E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_quant_fp8_sf_strided_mma failed: " + cuda_err(err));
  }, py::arg("h13"), py::arg("ld_h13"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused SwiGLU + FP8 quantization with SFA written directly to per-expert MMA layout");

  m.def("swiglu_quant_nvfp4_sf_strided_mma", [](uintptr_t h13_ptr, int ld_h13,
                                                uintptr_t out_ptr, int ld_out,
                                                uintptr_t sf_mma_ptr,
                                                uintptr_t offs_ptr,
                                                int E, int M_e_stride,
                                                int M_pad, int K,
                                                py::object stream) {
    auto err = swiglu_quant_nvfp4_sf_strided_mma(reinterpret_cast<const void*>(h13_ptr), ld_h13,
                                                 reinterpret_cast<void*>(out_ptr), ld_out,
                                                 reinterpret_cast<void*>(sf_mma_ptr),
                                                 reinterpret_cast<const int32_t*>(offs_ptr),
                                                 E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_quant_nvfp4_sf_strided_mma failed: " + cuda_err(err));
  }, py::arg("h13"), py::arg("ld_h13"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused SwiGLU + NVFP4 quantization with SFA written directly to per-expert MMA layout");

  m.def("quant_fp8_with_sfa", [](uintptr_t x_ptr, int ldx,
                                  uintptr_t out_ptr, int ld_out,
                                  uintptr_t sfa_ptr, int ld_sf,
                                  int M, int K, py::object stream) {
    auto err = quant_fp8_with_sfa(reinterpret_cast<const void*>(x_ptr), ldx,
                                  reinterpret_cast<void*>(out_ptr), ld_out,
                                  reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                  M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_fp8_with_sfa failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> FP8 using provided SFA (rowwise, sf_vec=32)");

  m.def("quant_nvfp4_with_sfa", [](uintptr_t x_ptr, int ldx,
                                     uintptr_t out_ptr, int ld_out,
                                     uintptr_t sfa_ptr, int ld_sf,
                                     int M, int K, py::object stream) {
    auto err = quant_nvfp4_with_sfa(reinterpret_cast<const void*>(x_ptr), ldx,
                                    reinterpret_cast<void*>(out_ptr), ld_out,
                                    reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                    M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_nvfp4_with_sfa failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> NVFP4 using provided SFA (rowwise, sf_vec=32)");

  m.def("dequant_fp8_to_bf16", [](uintptr_t q_ptr, int ldq,
                                   uintptr_t sfa_ptr, int ld_sf,
                                   uintptr_t out_ptr, int ld_out,
                                   int M, int K, py::object stream) {
    auto err = dequant_fp8_to_bf16(reinterpret_cast<const void*>(q_ptr), ldq,
                                   reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                   reinterpret_cast<void*>(out_ptr), ld_out,
                                   M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("dequant_fp8_to_bf16 failed");
  }, py::arg("q"), py::arg("ldq"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "FP8(blockscaled) -> BF16 dequantization using provided rowwise SFA");

  m.def("dequant_fp8_to_bf16_mma_sf", [](uintptr_t q_ptr, int ldq,
                                          uintptr_t sfa_ptr, int ld_sf,
                                          uintptr_t out_ptr, int ld_out,
                                          int M, int K, py::object stream) {
    auto err = dequant_fp8_to_bf16_mma_sf(reinterpret_cast<const void*>(q_ptr), ldq,
                                          reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                          reinterpret_cast<void*>(out_ptr), ld_out,
                                          M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("dequant_fp8_to_bf16_mma_sf failed");
  }, py::arg("q"), py::arg("ldq"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "FP8(blockscaled) -> BF16 dequantization using CUTLASS MMA-swizzled SFA");

  m.def("dequant_nvfp4_to_bf16", [](uintptr_t q_ptr, int ldq,
                                     uintptr_t sfa_ptr, int ld_sf,
                                     uintptr_t out_ptr, int ld_out,
                                     int M, int K, py::object stream) {
    auto err = dequant_nvfp4_to_bf16(reinterpret_cast<const void*>(q_ptr), ldq,
                                     reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                     reinterpret_cast<void*>(out_ptr), ld_out,
                                     M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("dequant_nvfp4_to_bf16 failed");
  }, py::arg("q"), py::arg("ldq"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "NVFP4(blockscaled) -> BF16 dequantization using provided rowwise SFA");

  m.def("dequant_nvfp4_to_bf16_mma_sf", [](uintptr_t q_ptr, int ldq,
                                            uintptr_t sfa_ptr, int ld_sf,
                                            uintptr_t out_ptr, int ld_out,
                                            int M, int K, py::object stream) {
    auto err = dequant_nvfp4_to_bf16_mma_sf(reinterpret_cast<const void*>(q_ptr), ldq,
                                            reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                            reinterpret_cast<void*>(out_ptr), ld_out,
                                            M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("dequant_nvfp4_to_bf16_mma_sf failed");
  }, py::arg("q"), py::arg("ldq"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "NVFP4(blockscaled) -> BF16 dequantization using CUTLASS MMA-swizzled SFA");

  m.def("swizzle_sf_mkl_to_mma", [](uintptr_t sf_mkl_ptr, uintptr_t sf_mma_ptr,
                                     int M, int sf_k, py::object stream) {
    auto err = swizzle_sf_mkl_to_mma(reinterpret_cast<const void*>(sf_mkl_ptr),
                                     reinterpret_cast<void*>(sf_mma_ptr),
                                     M, sf_k, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swizzle_sf_mkl_to_mma failed");
  }, py::arg("sf_mkl"), py::arg("sf_mma"), py::arg("M"), py::arg("sf_k"),
     py::arg("stream") = py::none(),
     "Rowwise SF swizzle to MMA layout (A-side)");

  // Build grouped GEMM metadata (GPU) for strided grouped interface
	  m.def("build_grouped_gemm_metadata", [](uintptr_t offs_ptr, int E,
	                                          int64_t A_base, int64_t A_row_bytes,
	                                          int64_t B_base, int64_t B_expert_bytes,
	                                          int64_t C_base, int64_t C_row_bytes,
	                                          int64_t SFA_base, int64_t SFA_row_bytes,
	                                          int64_t SFB_base, int64_t SFB_expert_bytes,
	                                          int32_t A_s0, int32_t A_s1,
	                                          int32_t B_s0, int32_t B_s1,
	                                          int32_t C_s0, int32_t C_s1,
	                                          int32_t N, int32_t K,
	                                          uintptr_t sizes_ptr, uintptr_t strides_ptr,
	                                          uintptr_t ptrs_abc_ptr, uintptr_t ptrs_sfasfb_ptr,
	                                          py::object stream) {
	    auto err = build_grouped_gemm_metadata(
	        reinterpret_cast<const int32_t*>(offs_ptr), E,
	        A_base, A_row_bytes,
	        B_base, B_expert_bytes,
	        C_base, C_row_bytes,
	        SFA_base, SFA_row_bytes,
	        SFB_base, SFB_expert_bytes,
	        A_s0, A_s1,
	        B_s0, B_s1,
	        C_s0, C_s1,
        N, K,
        reinterpret_cast<int32_t*>(sizes_ptr),
        reinterpret_cast<int32_t*>(strides_ptr),
        reinterpret_cast<int64_t*>(ptrs_abc_ptr),
        reinterpret_cast<int64_t*>(ptrs_sfasfb_ptr),
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("build_grouped_gemm_metadata failed");
	  }, py::arg("offs"), py::arg("E"),
	     py::arg("A_base"), py::arg("A_row_bytes"),
	     py::arg("B_base"), py::arg("B_expert_bytes"),
	     py::arg("C_base"), py::arg("C_row_bytes"),
	     py::arg("SFA_base"), py::arg("SFA_row_bytes"),
	     py::arg("SFB_base"), py::arg("SFB_expert_bytes"),
	     py::arg("A_s0"), py::arg("A_s1"),
	     py::arg("B_s0"), py::arg("B_s1"),
	     py::arg("C_s0"), py::arg("C_s1"),
     py::arg("N"), py::arg("K"),
     py::arg("sizes_mnkl"), py::arg("strides_abc"),
     py::arg("ptrs_abc"), py::arg("ptrs_sfasfb"),
     py::arg("stream") = py::none(),
     "Build grouped GEMM metadata on GPU for strided grouped interface");

  // ========== Fused RoPE (rope_fused.cu) ==========
  m.def("fused_rope_forward", [](uintptr_t x_ptr, uintptr_t cos_ptr, uintptr_t sin_ptr,
                                 uintptr_t out_ptr,
                                 int total_vecs, int seq_len, int n_heads, int head_dim,
                                 py::object stream) {
    auto err = fused_rope_forward(
        reinterpret_cast<const void*>(x_ptr),
        reinterpret_cast<const void*>(cos_ptr),
        reinterpret_cast<const void*>(sin_ptr),
        reinterpret_cast<void*>(out_ptr),
        total_vecs, seq_len, n_heads, head_dim,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("fused_rope_forward failed: " + cuda_err(err));
  }, py::arg("x"), py::arg("cos_buf"), py::arg("sin_buf"), py::arg("out"),
     py::arg("total_vecs"), py::arg("seq_len"), py::arg("n_heads"), py::arg("head_dim"),
     py::arg("stream") = py::none(),
     "Fused RoPE forward: out[..,:half]=x1*cos-x2*sin; out[..,half:]=x2*cos+x1*sin. "
     "BF16 I/O, FP32 compute. x=[total_vecs,head_dim], cos/sin=[seq_len,half_dim].");

  m.def("fused_rope_backward", [](uintptr_t grad_out_ptr, uintptr_t cos_ptr, uintptr_t sin_ptr,
                                  uintptr_t grad_x_ptr,
                                  int total_vecs, int seq_len, int n_heads, int head_dim,
                                  py::object stream) {
    auto err = fused_rope_backward(
        reinterpret_cast<const void*>(grad_out_ptr),
        reinterpret_cast<const void*>(cos_ptr),
        reinterpret_cast<const void*>(sin_ptr),
        reinterpret_cast<void*>(grad_x_ptr),
        total_vecs, seq_len, n_heads, head_dim,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("fused_rope_backward failed: " + cuda_err(err));
  }, py::arg("grad_out"), py::arg("cos_buf"), py::arg("sin_buf"), py::arg("grad_x"),
     py::arg("total_vecs"), py::arg("seq_len"), py::arg("n_heads"), py::arg("head_dim"),
     py::arg("stream") = py::none(),
     "Fused RoPE backward: grad_x[..,:half]=go1*cos+go2*sin; grad_x[..,half:]=-go1*sin+go2*cos. "
     "BF16 I/O, FP32 compute.");

  // ========== Partial RoPE (in-place, rope portion only) ==========
  m.def("fused_rope_forward_partial", [](uintptr_t x_ptr, uintptr_t cos_ptr, uintptr_t sin_ptr,
                                         int total_vecs, int seq_len, int n_heads, int head_dim,
                                         int nope_dim,
                                         py::object stream) {
    auto err = fused_rope_forward_partial(
        reinterpret_cast<void*>(x_ptr),
        reinterpret_cast<const void*>(cos_ptr),
        reinterpret_cast<const void*>(sin_ptr),
        total_vecs, seq_len, n_heads, head_dim, nope_dim,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("fused_rope_forward_partial failed: " + cuda_err(err));
  }, py::arg("x"), py::arg("cos_buf"), py::arg("sin_buf"),
     py::arg("total_vecs"), py::arg("seq_len"), py::arg("n_heads"), py::arg("head_dim"),
     py::arg("nope_dim"),
     py::arg("stream") = py::none(),
     "In-place partial RoPE forward: applies rotation only to x[...,nope_dim:]. "
     "Elements [0:nope_dim] are unchanged. Eliminates split+cat overhead in MLA. "
     "BF16 I/O, FP32 compute. cos/sin are [seq_len, half_dim] where half_dim=(head_dim-nope_dim)/2.");

  m.def("fused_rope_backward_partial", [](uintptr_t grad_x_ptr, uintptr_t cos_ptr, uintptr_t sin_ptr,
                                          int total_vecs, int seq_len, int n_heads, int head_dim,
                                          int nope_dim,
                                          py::object stream) {
    auto err = fused_rope_backward_partial(
        reinterpret_cast<void*>(grad_x_ptr),
        reinterpret_cast<const void*>(cos_ptr),
        reinterpret_cast<const void*>(sin_ptr),
        total_vecs, seq_len, n_heads, head_dim, nope_dim,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("fused_rope_backward_partial failed: " + cuda_err(err));
  }, py::arg("grad_x"), py::arg("cos_buf"), py::arg("sin_buf"),
     py::arg("total_vecs"), py::arg("seq_len"), py::arg("n_heads"), py::arg("head_dim"),
     py::arg("nope_dim"),
     py::arg("stream") = py::none(),
     "In-place partial RoPE backward: applies inverse rotation only to grad_x[...,nope_dim:]. "
     "Elements [0:nope_dim] gradients are unchanged. BF16 I/O, FP32 compute.");

  m.def("nvfp4_recompute_group_scales", [](uintptr_t packed_ptr, uintptr_t scale_ptr,
                                           uintptr_t gs_ptr,
                                           int E, int out_dim, int in_dim, int group_size,
                                           py::object stream) {
    auto err = nvfp4_recompute_group_scales(
        reinterpret_cast<void*>(packed_ptr),
        reinterpret_cast<void*>(scale_ptr),
        reinterpret_cast<const void*>(gs_ptr),
        E, out_dim, in_dim, group_size,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("nvfp4_recompute_group_scales failed: " + cuda_err(err));
  }, py::arg("W_packed"), py::arg("W_scale"), py::arg("W_gs"),
     py::arg("E"), py::arg("out_dim"), py::arg("in_dim"), py::arg("group_size"),
     py::arg("stream") = py::none(),
     "Recompute NVFP4 E4M3 per-group scales at correct group_size granularity. "
     "Tightens scales from 32-element tile amax to true group_size amax, "
     "re-quantizes E2M1 nibbles in-place. W_packed=[E,out_dim,in_dim/2] uint8, "
     "W_scale=[E,out_dim,in_dim/group_size] uint8(E4M3).");

#ifdef WITH_NVSHMEM
  // ========== NVSHMEM Functions ==========
  m.def("nvshmem_get_uid", []() {
    int size = rdep_nvshmem_get_uid_size();
    py::array_t<uint8_t> uid(size);
    rdep_nvshmem_get_uid(uid.mutable_data());
    return uid;
  }, "Get NVSHMEM UID for bootstrap (call on rank 0 only)");

  m.def("nvshmem_get_uid_size", &rdep_nvshmem_get_uid_size,
        "Get NVSHMEM UID size in bytes");

  m.def("nvshmem_init", [](py::array_t<uint8_t> uid, int rank, int world, int local_world) {
    rdep_nvshmem_init_with_uid(uid.data(), rank, world, local_world);
  }, py::arg("uid"), py::arg("rank"), py::arg("world"), py::arg("local_world"),
     "Initialize NVSHMEM with UID from rank 0");

  m.def("nvshmem_alloc_bf16", [](size_t capacity, int H, int n_local) {
    rdep_nvshmem_alloc_bf16(capacity, H, n_local);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"),
     "Allocate NVSHMEM symmetric buffers for BF16 path");
  m.def("nvshmem_alloc_blockscaled", [](size_t capacity, int H, int n_local, int profile) {
    rdep_nvshmem_alloc_blockscaled(capacity, H, n_local, profile);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"), py::arg("profile"),
     "Allocate NVSHMEM symmetric buffers for blockscaled path");

  // NVSHMEM IPC buffer functions (separate cudaMalloc'd buffer)
  // These are different from rdep get_ipc_handle_bf16 - they get handles
  // from the cudaMalloc'd IPC buffer, NOT from NVSHMEM symmetric heap
  m.def("nvshmem_get_ipc_handle_bf16", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_nvshmem_get_ipc_handle_bf16(handle.mutable_data());
    return handle;
  }, "Get IPC handle for NVSHMEM's separate IPC buffer (BF16)");

  m.def("nvshmem_open_ipc_handles_bf16", [](py::array_t<uint8_t> handles, int local_world) {
    validate_ipc_handles(handles, local_world, "nvshmem_open_ipc_handles_bf16");
    rdep_nvshmem_open_ipc_handles_bf16(handles.data(), local_world);
  }, py::arg("handles"), py::arg("local_world"),
     "Open remote IPC handles for NVSHMEM's IPC buffer (BF16)");

  m.def("nvshmem_sync_ipc_buffer_ptrs_bf16", &rdep_nvshmem_sync_ipc_buffer_ptrs_bf16,
        "Sync NVSHMEM IPC buffer pointers to device");

  m.def("nvshmem_get_ipc_handle_blockscaled", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_nvshmem_get_ipc_handle_blockscaled(handle.mutable_data());
    return handle;
  }, "Get IPC handle for NVSHMEM's separate IPC buffer (blockscaled)");

  m.def("nvshmem_open_ipc_handles_blockscaled", [](py::array_t<uint8_t> handles, int local_world) {
    validate_ipc_handles(handles, local_world, "nvshmem_open_ipc_handles_blockscaled");
    rdep_nvshmem_open_ipc_handles_blockscaled(handles.data(), local_world);
  }, py::arg("handles"), py::arg("local_world"),
     "Open remote IPC handles for NVSHMEM's IPC buffer (blockscaled)");

  m.def("nvshmem_sync_ipc_buffer_ptrs_blockscaled", &rdep_nvshmem_sync_ipc_buffer_ptrs_blockscaled,
        "Sync NVSHMEM blockscaled IPC buffer pointers to device");

#endif
}
#endif
