#include <torch/extension.h>

#include "sm100/prefill/dense/interface.h"

void dense_prefill_fwd(
    at::Tensor workspace_buffer,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor cumulative_seqlen_q,
    at::Tensor cumulative_seqlen_kv,
    at::Tensor o,
    at::Tensor lse,
    int mask_mode_code,
    double softmax_scale,
    int max_seqlen_q,
    int max_seqlen_kv,
    bool is_varlen) {
  FMHACutlassSM100FwdRun(
      workspace_buffer,
      q,
      k,
      v,
      cumulative_seqlen_q,
      cumulative_seqlen_kv,
      o,
      lse,
      mask_mode_code,
      static_cast<float>(softmax_scale),
      max_seqlen_q,
      max_seqlen_kv,
      is_varlen);
}

void dense_prefill_bwd(
    at::Tensor workspace_buffer,
    at::Tensor d_o,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor cumulative_seqlen_q,
    at::Tensor cumulative_seqlen_kv,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    int mask_mode_code,
    double softmax_scale,
    int max_seqlen_q,
    int max_seqlen_kv,
    bool is_varlen) {
  FMHACutlassSM100BwdRun(
      workspace_buffer,
      d_o,
      q,
      k,
      v,
      o,
      lse,
      cumulative_seqlen_q,
      cumulative_seqlen_kv,
      dq,
      dk,
      dv,
      mask_mode_code,
      static_cast<float>(softmax_scale),
      max_seqlen_q,
      max_seqlen_kv,
      is_varlen);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "FlashMLA dense SM100 prefill forward/backward (SM100)";
  m.def("dense_prefill_fwd", &dense_prefill_fwd);
  m.def("dense_prefill_bwd", &dense_prefill_bwd);
}
