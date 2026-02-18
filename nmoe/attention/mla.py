import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from nmoe.attention.rope import rotate_pe, rotate_pe_partial
from nmoe.config import Config
from nmoe.norm import RMSNorm


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise RuntimeError(msg)


def _sm100_only(device: torch.device) -> None:
  _require(torch.cuda.is_available(), "MLA requires CUDA (B200 / SM100).")
  major, minor = torch.cuda.get_device_capability(device)
  _require(major == 10, f"MLA requires SM100 (B200). Got compute capability {major}.{minor}.")


def _nvtx(tag: str):
  if os.getenv('NMOE_NVTX', '0') not in ('1', 'true', 'True'):
    return nullcontext()
  if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'):
    return torch.cuda.nvtx.range(tag)
  return nullcontext()


# Use PyTorch SDPA by default. FA4 cute-dsl has a bug with (192, 128) dimensions on SM100.
# Set NMOE_USE_FA4=1 to use FA4+FlashMLA (may produce NaN).
_USE_SDPA = os.getenv('NMOE_USE_FA4', '0') not in ('1', 'true', 'True')


# Module-level workspace cache for MLA backward pass.
#
# Design goals:
# - Avoid per-backward allocation churn (token/s stability, allocator pressure)
# - Bounded growth: one buffer per device, grown in-place as needed
# - Scratch-only: not part of module state / checkpoints
_mla_workspace_cache: dict[torch.device, torch.Tensor] = {}


def _get_mla_workspace(device: torch.device, workspace_bytes: int) -> torch.Tensor:
  """Get a cached workspace buffer for MLA backward.

  Returns a `torch.uint8` CUDA tensor with `numel() >= workspace_bytes`.
  """
  buf = _mla_workspace_cache.get(device)
  if buf is None or buf.numel() < workspace_bytes:
    buf = torch.empty((workspace_bytes,), device=device, dtype=torch.uint8)
    _mla_workspace_cache[device] = buf
  return buf


def _mla_sdpa_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> torch.Tensor:
  """MLA attention using PyTorch's native SDPA.

  This is the default implementation as FA4 cute-dsl has a bug with (192, 128) dimensions.
  Uses PyTorch's optimized scaled_dot_product_attention which works correctly on all GPUs.

  Args:
    q: Query tensor [B, S, H, D_qk]
    k: Key tensor [B, S, H, D_qk]
    v: Value tensor [B, S, H, D_v]
    softmax_scale: Softmax scaling factor

  Returns:
    Output tensor [B, S, H, D_v]
  """
  # SDPA expects [B, H, S, D]
  q = q.transpose(1, 2)  # [B, H, S, D_qk]
  k = k.transpose(1, 2)  # [B, H, S, D_qk]
  v = v.transpose(1, 2)  # [B, H, S, D_v]

  with _nvtx("attn/sdpa_fwd"):
    out = F.scaled_dot_product_attention(
        q, k, v,
        scale=softmax_scale,
        is_causal=True,
    )

  # Back to [B, S, H, D_v]
  return out.transpose(1, 2)


def _build_block_causal_mask(
    cu_seqlens: torch.Tensor,
    seqlen: int,
    device: torch.device,
) -> torch.Tensor:
  """Build block-diagonal causal mask from cumulative sequence lengths.

  Uses torch.searchsorted to assign document IDs without .item() calls,
  enabling fully GPU-resident mask construction.

  Args:
    cu_seqlens: [num_docs + 1] cumulative sequence lengths (int32)
    seqlen: Total sequence length
    device: Target device

  Returns:
    mask: [seqlen, seqlen] bool tensor where True = can attend
  """
  pos = torch.arange(seqlen, device=device, dtype=cu_seqlens.dtype)

  # Assign document ID to each position using searchsorted
  # cu_seqlens[1:] contains document end boundaries
  # searchsorted(right=True) ensures position i belongs to doc d
  # if cu_seqlens[d] <= i < cu_seqlens[d+1]
  doc_ids = torch.searchsorted(cu_seqlens[1:], pos, right=True)  # [seqlen]

  # Build 2D masks using broadcasting:
  # - Causal: query position >= key position
  # - Document isolation: query and key belong to same document
  q_idx = pos.unsqueeze(1)  # [seqlen, 1] - query positions (rows)
  k_idx = pos.unsqueeze(0)  # [1, seqlen] - key positions (cols)
  causal_mask = q_idx >= k_idx  # [seqlen, seqlen]

  q_doc = doc_ids.unsqueeze(1)  # [seqlen, 1]
  k_doc = doc_ids.unsqueeze(0)  # [1, seqlen]
  doc_mask = q_doc == k_doc  # [seqlen, seqlen]

  return causal_mask & doc_mask


def _mla_sdpa_packed_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    cu_seqlens_list: list[torch.Tensor],
) -> torch.Tensor:
  """MLA attention using SDPA for packed sequences (document-isolated causal).

  Uses a single SDPA call with a block-diagonal causal mask instead of
  per-document Python loops. This eliminates:
  - ~100 kernel launches per call (was 5 per doc × ~20 docs)
  - CPU↔GPU synchronization from .item() calls on cu_seqlens

  The mask is built using torch.searchsorted for document ID assignment,
  keeping all operations on GPU without host synchronization.

  Args:
    q: Query tensor [B, S, H, D_qk]
    k: Key tensor [B, S, H, D_qk]
    v: Value tensor [B, S, H, D_v]
    softmax_scale: Softmax scaling factor
    cu_seqlens_list: List of B int32 tensors, each [num_docs_i + 1] marking
                     document boundaries within batch element i.

  Returns:
    Output tensor [B, S, H, D_v]
  """
  bsz, seqlen, n_heads, _ = q.shape
  device = q.device

  with _nvtx("attn/sdpa_packed_fwd"):
    # Build block-diagonal causal masks for all batch elements
    # Each mask is [seqlen, seqlen] with True where attention is allowed
    masks = []
    for b in range(bsz):
      mask = _build_block_causal_mask(cu_seqlens_list[b], seqlen, device)
      masks.append(mask)

    # Stack masks: [B, S, S] -> [B, 1, S, S] (broadcasts over heads)
    mask = torch.stack(masks, dim=0).unsqueeze(1)

    # Convert bool mask to additive float mask for SDPA
    # True (attend) -> 0.0, False (block) -> -inf
    float_mask = torch.where(
        mask,
        torch.tensor(0.0, device=device, dtype=q.dtype),
        torch.tensor(float('-inf'), device=device, dtype=q.dtype),
    )

    # SDPA expects [B, H, S, D] layout
    q_t = q.transpose(1, 2)  # [B, H, S, D_qk]
    k_t = k.transpose(1, 2)  # [B, H, S, D_qk]
    v_t = v.transpose(1, 2)  # [B, H, S, D_v]

    # Single SDPA call with block-diagonal causal mask
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=float_mask,
        scale=softmax_scale,
    )

    # Back to [B, S, H, D_v]
    return out.transpose(1, 2)


class _MlaFlashMlaVarlenPacked(torch.autograd.Function):
  """MLA attention using FlashMLA varlen for packed sequences.

  Uses the FlashMLA dense_prefill_fwd/bwd with per-document cu_seqlens
  to achieve zero-mask-memory document-isolated causal attention.

  This processes each batch element independently through the varlen API,
  since each element may have a different number of packed documents.
  """

  @staticmethod
  def forward(ctx, q, k, v, softmax_scale, cu_seqlens_list):
    """Forward pass with FlashMLA varlen for packed sequences.

    Args:
      q: [B, S, H, D_qk] query tensor (BF16, CUDA)
      k: [B, S, H, D_qk] key tensor
      v: [B, S, H, D_v] value tensor
      softmax_scale: float scaling factor
      cu_seqlens_list: list of B int32 tensors, each [num_docs_i + 1]
    """
    from flash_mla import flash_attn_varlen_func

    bsz, seqlen, n_heads, d_qk = q.shape
    d_v = v.shape[-1]

    # Process each batch element through varlen API
    outputs = []
    all_lse = []
    all_cu = []
    max_seqlens = []

    for b in range(bsz):
      cu = cu_seqlens_list[b]  # [num_docs + 1] int32
      # Filter out zero-length padding "documents" at the end
      # (the packer may add a padding region as the last segment)
      num_docs = cu.shape[0] - 1
      total_tokens = cu[-1].item()

      # Extract this batch element's tokens
      q_b = q[b, :total_tokens].contiguous()  # [total_tokens, H, D_qk]
      k_b = k[b, :total_tokens].contiguous()
      v_b = v[b, :total_tokens].contiguous()

      # Compute max seqlen for this batch element
      doc_lengths = cu[1:] - cu[:-1]
      max_seqlen = int(doc_lengths.max().item()) if num_docs > 0 else 0

      out_b, lse_b = flash_attn_varlen_func(
          q_b, k_b, v_b,
          cu_seqlens_qo=cu,
          cu_seqlens_kv=cu,
          max_seqlen_qo=max_seqlen,
          max_seqlen_kv=max_seqlen,
          causal=True,
          softmax_scale=softmax_scale,
          is_varlen=True,
      )

      # Pad output back to seqlen
      if total_tokens < seqlen:
        pad_out = torch.zeros(seqlen - total_tokens, n_heads, d_v,
                              device=out_b.device, dtype=out_b.dtype)
        out_b = torch.cat([out_b, pad_out], dim=0)

      outputs.append(out_b)
      all_lse.append(lse_b)
      all_cu.append(cu)
      max_seqlens.append(max_seqlen)

    output = torch.stack(outputs, dim=0)  # [B, S, H, D_v]

    # Save for backward
    ctx.save_for_backward(q, k, v, output, *all_lse, *all_cu)
    ctx.softmax_scale = softmax_scale
    ctx.bsz = bsz
    ctx.seqlen = seqlen
    ctx.n_heads = n_heads
    ctx.d_qk = d_qk
    ctx.d_v = d_v
    ctx.max_seqlens = max_seqlens
    ctx.num_lse = bsz
    ctx.num_cu = bsz

    return output

  @staticmethod
  def backward(ctx, d_out):
    from flash_mla.flash_mla_interface import _flash_attn_varlen_backward

    saved = ctx.saved_tensors
    q = saved[0]
    k = saved[1]
    v = saved[2]
    output = saved[3]
    all_lse = saved[4:4 + ctx.num_lse]
    all_cu = saved[4 + ctx.num_lse:]

    bsz = ctx.bsz
    seqlen = ctx.seqlen
    n_heads = ctx.n_heads
    d_qk = ctx.d_qk
    d_v = ctx.d_v
    softmax_scale = ctx.softmax_scale
    max_seqlens = ctx.max_seqlens

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for b in range(bsz):
      cu = all_cu[b]
      lse = all_lse[b]
      total_tokens = cu[-1].item()
      max_seqlen = max_seqlens[b]

      if total_tokens == 0 or max_seqlen == 0:
        continue

      q_b = q[b, :total_tokens].contiguous()
      k_b = k[b, :total_tokens].contiguous()
      v_b = v[b, :total_tokens].contiguous()
      out_b = output[b, :total_tokens].contiguous()
      do_b = d_out[b, :total_tokens].contiguous()

      dq_b, dk_b, dv_b = _flash_attn_varlen_backward(
          do_b, q_b, k_b, v_b, out_b, lse,
          cu, cu, max_seqlen, max_seqlen,
          causal=True, softmax_scale=softmax_scale, is_varlen=True,
      )

      dq[b, :total_tokens] = dq_b
      dk[b, :total_tokens] = dk_b
      dv[b, :total_tokens] = dv_b

    return dq, dk, dv, None, None


class _MlaFa4FwdFlashMlaBwd(torch.autograd.Function):
  """MLA attention using FA4 forward + FlashMLA backward.

  WARNING: FA4 cute-dsl has a known bug with (192, 128) dimensions on SM100 that
  produces NaN outputs. Use PyTorch SDPA instead (default).
  """
  @staticmethod
  def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    _sm100_only(q.device)
    _require(q.is_cuda and k.is_cuda and v.is_cuda, "MLA FA4+FlashMLA requires CUDA tensors.")
    _require(q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16,
             "MLA FA4+FlashMLA requires BF16 inputs.")
    _require(q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Expected q/k/v with shape [B, S, H, D].")

    bsz, seqlen, n_heads, d_qk = q.shape
    _require(k.shape == (bsz, seqlen, n_heads, d_qk), "k must match q shape.")
    d_v = v.shape[-1]
    _require(v.shape == (bsz, seqlen, n_heads, d_v), "v must have shape [B, S, H, Dv].")
    _require(d_qk == 192 and d_v == 128, f"Only (d_qk, d_v) = (192, 128) is supported. Got ({d_qk}, {d_v}).")

    # Hard requirement: do not proceed without FA4 forward + FlashMLA backward.
    # Environment contract: `third_party/flash_attn` is on PYTHONPATH (so `flash_attn.cute` is importable),
    # and `nmoe.csrc.flashmla_sm100` is built.
    from flash_attn.cute.interface import _flash_attn_fwd  # type: ignore
    from nmoe.csrc import flashmla_sm100 as _flashmla  # type: ignore

    total = bsz * seqlen
    q_ = q.reshape(total, n_heads, d_qk).contiguous()
    k_ = k.reshape(total, n_heads, d_qk).contiguous()
    v_ = v.reshape(total, n_heads, d_v).contiguous()

    cu = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, device=q.device, dtype=torch.int32)

    with _nvtx("attn/fa4_fwd"):
      # Use m_block=128, n_block=64 for (d_qk=192, d_v=128) on SM100 (B200)
      # The default n_block=128 triggers a TMEM layout mismatch in cute-dsl
      out, lse = _flash_attn_fwd(
          q_,
          k_,
          v_,
          cu_seqlens_q=cu,
          cu_seqlens_k=cu,
          softmax_scale=float(softmax_scale),
          causal=True,
          return_lse=True,
          m_block_size=128,
          n_block_size=64,
      )

    # FlashMLA expects lse as [total, H] float32 with stride(0) == 1.
    lse_t = lse.T

    ctx.save_for_backward(q_, k_, v_, out, lse_t, cu)
    ctx.softmax_scale = float(softmax_scale)
    ctx.seqlen = int(seqlen)
    ctx._flashmla = _flashmla
    return out.reshape(bsz, seqlen, n_heads, d_v)

  @staticmethod
  def backward(ctx, d_out: torch.Tensor):
    q, k, v, out, lse_t, cu = ctx.saved_tensors
    softmax_scale = ctx.softmax_scale
    seqlen = ctx.seqlen
    flashmla = ctx._flashmla

    bsz = cu.numel() - 1
    total, n_heads, d_qk = q.shape
    d_v = v.shape[-1]

    d_o = d_out.reshape(total, n_heads, d_v).contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    max_seqlen_aligned = ((seqlen + 7) // 8) * 8
    workspace_bytes = 0
    workspace_bytes += 4 * bsz * max_seqlen_aligned * n_heads * d_qk  # dQ_acc
    workspace_bytes += 4 * max_seqlen_aligned * bsz * n_heads * 2  # sum_OdO + scaled_lse
    # Use cached workspace to avoid allocation churn
    workspace = _get_mla_workspace(q.device, workspace_bytes)

    with _nvtx("attn/flashmla_bwd"):
      flashmla.dense_prefill_bwd(
          workspace,
          d_o,
          q,
          k,
          v,
          out,
          lse_t,
          cu,
          cu,
          dq,
          dk,
          dv,
          1,  # causal
          softmax_scale,
          seqlen,
          seqlen,
          True,  # is_varlen
      )

    return dq.reshape(bsz, seqlen, n_heads, d_qk), dk.reshape(bsz, seqlen, n_heads, d_qk), dv.reshape(bsz, seqlen, n_heads, d_v), None


class MLA(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.dim = config.dim
    self.n_heads = config.n_heads
    self.q_lora_rank = config.q_lora_rank
    self.kv_lora_rank = config.kv_lora_rank
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    self.v_head_dim = config.v_head_dim
    self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
    self.q_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
    self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False, dtype=torch.bfloat16)
    self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, dtype=torch.bfloat16)
    self.kv_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
    self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, dtype=torch.bfloat16)
    self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False, dtype=torch.bfloat16)
    self.softmax_scale = self.qk_head_dim ** -0.5

  def init_weights(self, init_std: float = 0.02):
    for proj in [self.wq_a, self.wq_b, self.wkv_a, self.wkv_b]:
      nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
    self.q_norm.weight.data.fill_(1.0)
    self.kv_norm.weight.data.fill_(1.0)

  @record_function("attn")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              cu_seqlens: list[torch.Tensor] | None = None) -> torch.Tensor:
    """MLA forward pass.

    Args:
      x: Input tensor [B, S, D].
      cos, sin: RoPE rotation tensors.
      cu_seqlens: Optional list of B int32 tensors for packed sequence attention.
                  Each tensor has shape [num_docs_i + 1] marking document boundaries
                  within batch element i. When provided, attention is document-isolated:
                  tokens from different documents cannot attend to each other.
                  When None, standard causal attention is used.
    """
    bsz, seqlen, _ = x.size()
    q = self.wq_b(self.q_norm(self.wq_a(x)))
    q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
    # In-place partial RoPE: rotates q[..., nope_dim:] leaving q[..., :nope_dim] unchanged.
    # Eliminates split+cat overhead (3 kernels -> 1 kernel).
    rotate_pe_partial(q, cos, sin, nope_dim=self.qk_nope_head_dim)
    kv = self.wkv_a(x)
    kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = rotate_pe(k_pe.unsqueeze(2), cos, sin)
    kv = self.wkv_b(self.kv_norm(kv))
    kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
    with record_function("attn.kernel[mla]"):
      if cu_seqlens is not None:
        # Packed sequences: per-document SDPA with is_causal=True.
        # Uses PyTorch's built-in flash-attention kernel — zero external deps,
        # zero mask materialization (unlike FlexAttention's create_block_mask).
        output = _mla_sdpa_packed_forward(q, k, v, self.softmax_scale, cu_seqlens)
      else:
        # Standard causal attention (no packing)
        if _USE_SDPA:
          output = _mla_sdpa_forward(q, k, v, self.softmax_scale)
        else:
          output = _MlaFa4FwdFlashMlaBwd.apply(q, k, v, self.softmax_scale)
    output = output.contiguous().view(bsz, seqlen, self.n_heads * self.v_head_dim)
    return self.wo(output)
