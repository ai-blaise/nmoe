#include <torch/extension.h>

#include <cmath>

#include "sm100/prefill/dense/interface.h"

namespace {

void check_cuda_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

void check_same_device(const at::Tensor& a, const at::Tensor& b, const char* a_name, const char* b_name) {
  TORCH_CHECK(
      a.device() == b.device(),
      a_name,
      " and ",
      b_name,
      " must be on the same CUDA device (got ",
      a.device(),
      " vs ",
      b.device(),
      ")");
}

void check_fwd_contract(
    const at::Tensor& workspace_buffer,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cumulative_seqlen_q,
    const at::Tensor& cumulative_seqlen_kv,
    const at::Tensor& o,
    const at::Tensor& lse,
    double softmax_scale) {
  check_cuda_tensor(workspace_buffer, "workspace_buffer");
  check_cuda_tensor(q, "q");
  check_cuda_tensor(k, "k");
  check_cuda_tensor(v, "v");
  check_cuda_tensor(cumulative_seqlen_q, "cumulative_seqlen_q");
  check_cuda_tensor(cumulative_seqlen_kv, "cumulative_seqlen_kv");
  check_cuda_tensor(o, "o");
  check_cuda_tensor(lse, "lse");

  check_same_device(q, k, "q", "k");
  check_same_device(q, v, "q", "v");
  check_same_device(q, cumulative_seqlen_q, "q", "cumulative_seqlen_q");
  check_same_device(q, cumulative_seqlen_kv, "q", "cumulative_seqlen_kv");
  check_same_device(q, o, "q", "o");
  check_same_device(q, lse, "q", "lse");
  check_same_device(q, workspace_buffer, "q", "workspace_buffer");

  TORCH_CHECK(workspace_buffer.scalar_type() == at::kByte, "workspace_buffer must be uint8");
  TORCH_CHECK(workspace_buffer.is_contiguous(), "workspace_buffer must be contiguous");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bfloat16");
  TORCH_CHECK(o.scalar_type() == at::kBFloat16, "o must be bfloat16");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be float32");
  TORCH_CHECK(cumulative_seqlen_q.scalar_type() == at::kInt, "cumulative_seqlen_q must be int32");
  TORCH_CHECK(cumulative_seqlen_kv.scalar_type() == at::kInt, "cumulative_seqlen_kv must be int32");

  TORCH_CHECK(q.dim() == 3, "q must be rank-3 [T, H, Dqk]");
  TORCH_CHECK(k.dim() == 3, "k must be rank-3 [T, H, Dqk]");
  TORCH_CHECK(v.dim() == 3, "v must be rank-3 [T, H, Dv]");
  TORCH_CHECK(o.dim() == 3, "o must be rank-3 [T, H, Dv]");
  TORCH_CHECK(lse.dim() == 2, "lse must be rank-2 [T, H]");
  TORCH_CHECK(cumulative_seqlen_q.dim() == 1, "cumulative_seqlen_q must be rank-1");
  TORCH_CHECK(cumulative_seqlen_kv.dim() == 1, "cumulative_seqlen_kv must be rank-1");
  TORCH_CHECK(cumulative_seqlen_q.numel() >= 2, "cumulative_seqlen_q must have at least 2 elements");
  TORCH_CHECK(cumulative_seqlen_kv.numel() >= 2, "cumulative_seqlen_kv must have at least 2 elements");

  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
  TORCH_CHECK(lse.stride(0) == 1, "lse stride(0) must be 1");

  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have identical shapes");
  TORCH_CHECK(q.size(0) == v.size(0) && q.size(1) == v.size(1), "q and v must match in [T, H]");
  TORCH_CHECK(o.size(0) == v.size(0) && o.size(1) == v.size(1) && o.size(2) == v.size(2), "o must match v shape");
  TORCH_CHECK(lse.size(0) == q.size(0) && lse.size(1) == q.size(1), "lse must match [T, H]");

  TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");
  TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be > 0");
}

void check_bwd_contract(
    const at::Tensor& workspace_buffer,
    const at::Tensor& d_o,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& o,
    const at::Tensor& lse,
    const at::Tensor& cumulative_seqlen_q,
    const at::Tensor& cumulative_seqlen_kv,
    const at::Tensor& dq,
    const at::Tensor& dk,
    const at::Tensor& dv,
    double softmax_scale) {
  check_fwd_contract(
      workspace_buffer,
      q,
      k,
      v,
      cumulative_seqlen_q,
      cumulative_seqlen_kv,
      o,
      lse,
      softmax_scale);

  check_cuda_tensor(d_o, "d_o");
  check_cuda_tensor(dq, "dq");
  check_cuda_tensor(dk, "dk");
  check_cuda_tensor(dv, "dv");

  check_same_device(q, d_o, "q", "d_o");
  check_same_device(q, dq, "q", "dq");
  check_same_device(q, dk, "q", "dk");
  check_same_device(q, dv, "q", "dv");

  TORCH_CHECK(d_o.scalar_type() == at::kBFloat16, "d_o must be bfloat16");
  TORCH_CHECK(dq.scalar_type() == at::kBFloat16, "dq must be bfloat16");
  TORCH_CHECK(dk.scalar_type() == at::kBFloat16, "dk must be bfloat16");
  TORCH_CHECK(dv.scalar_type() == at::kBFloat16, "dv must be bfloat16");

  TORCH_CHECK(d_o.is_contiguous(), "d_o must be contiguous");
  TORCH_CHECK(dq.is_contiguous(), "dq must be contiguous");
  TORCH_CHECK(dk.is_contiguous(), "dk must be contiguous");
  TORCH_CHECK(dv.is_contiguous(), "dv must be contiguous");

  TORCH_CHECK(d_o.sizes() == o.sizes(), "d_o shape must match o");
  TORCH_CHECK(dq.sizes() == q.sizes(), "dq shape must match q");
  TORCH_CHECK(dk.sizes() == k.sizes(), "dk shape must match k");
  TORCH_CHECK(dv.sizes() == v.sizes(), "dv shape must match v");
}

}  // namespace

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
  check_fwd_contract(
      workspace_buffer,
      q,
      k,
      v,
      cumulative_seqlen_q,
      cumulative_seqlen_kv,
      o,
      lse,
      softmax_scale);
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
  check_bwd_contract(
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
      softmax_scale);
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
