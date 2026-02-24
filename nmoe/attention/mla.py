import os
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from nmoe.attention.fa4_import import get_fa4_flashmla_modules
from nmoe.attention.rope import rotate_pe, rotate_pe_partial
from nmoe.config import Config
from nmoe.norm import RMSNorm


def _env_flag(name: str, default: str = "0") -> bool:
  return os.getenv(name, default) in ("1", "true", "True")


_NVTX_ENABLED = (
  _env_flag("NMOE_NVTX", "0")
  and torch.cuda.is_available()
  and hasattr(torch.cuda, "nvtx")
  and hasattr(torch.cuda.nvtx, "range")
)


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise RuntimeError(msg)


def _nan_debug_active() -> bool:
  # Scoped by train.py to a specific failing step/micro-step.
  return _env_flag("NMOE_NAN_DEBUG_ACTIVE", "0") or _env_flag("NMOE_MLA_FINITE_DEBUG", "0")


def _require_finite(tag: str, tensor: torch.Tensor, *, force: bool = False) -> None:
  if (not force) and (not _nan_debug_active()):
    return
  if not (tensor.is_floating_point() or tensor.is_complex()):
    return
  with torch.no_grad():
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all().item()):
      return
    numel = int(tensor.numel())
    finite = int(finite_mask.sum().item())
    nan = int(torch.isnan(tensor).sum().item())
    inf = int(torch.isinf(tensor).sum().item())
  raise RuntimeError(
    f"[NANDBG][mla] non-finite {tag} "
    f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
    f"finite={finite}/{numel} nan={nan} inf={inf}"
  )


_sm100_validated_devices: set[int] = set()


def _sm100_only(device: torch.device) -> None:
  _require(torch.cuda.is_available(), "MLA requires CUDA (B200 / SM100).")
  dev_idx = device.index if device.index is not None else torch.cuda.current_device()
  if dev_idx in _sm100_validated_devices:
    return
  major, minor = torch.cuda.get_device_capability(device)
  _require(major == 10, f"MLA requires SM100 (B200). Got compute capability {major}.{minor}.")
  _sm100_validated_devices.add(dev_idx)


def _nvtx(tag: str):
  if _NVTX_ENABLED:
    return torch.cuda.nvtx.range(tag)
  return nullcontext()


# Production default is FlashMLA/FA4 only. SDPA is explicit debug fallback.
_USE_SDPA = os.getenv('NMOE_USE_FA4', '1') not in ('1', 'true', 'True')
_REQUIRE_FLASHMLA = _env_flag("NMOE_REQUIRE_FLASHMLA", "1")
_USE_FLASHMLA_VARLEN = _env_flag("NMOE_MLA_USE_FLASHMLA_VARLEN", "0")
_USE_FLASHMLA_SM100_FWD = _env_flag("NMOE_MLA_USE_FLASHMLA_SM100_FWD", "0")
_PACKED_ATTN_BACKEND = os.getenv("NMOE_PACKED_ATTN_BACKEND", "flashmla").strip().lower()
_VALIDATE_PACKED_CU = _env_flag("NMOE_VALIDATE_PACKED_CU_SEQLENS", "0")
_MLA_WORKSPACE_CACHE_MAX_BYTES = int(os.getenv("NMOE_MLA_WORKSPACE_CACHE_MAX_BYTES", str(512 * 1024 * 1024)))
_MLA_STREAM_CACHE_LIMIT = max(1, int(os.getenv("NMOE_MLA_STREAM_CACHE_LIMIT", "8")))


def _load_mla_softmax_scale_mult() -> float:
  raw = os.getenv("NMOE_MLA_SOFTMAX_SCALE_MULT", "1.0")
  try:
    mult = float(raw)
  except ValueError as e:
    raise RuntimeError(
      f"NMOE_MLA_SOFTMAX_SCALE_MULT must parse as float, got {raw!r}."
    ) from e
  if not math.isfinite(mult) or mult <= 0.0 or mult > 8.0:
    raise RuntimeError(
      "NMOE_MLA_SOFTMAX_SCALE_MULT must be finite and in (0, 8], "
      f"got {mult} (raw={raw!r})."
    )
  return mult


_MLA_SOFTMAX_SCALE_MULT = _load_mla_softmax_scale_mult()
_MLA_FAILFAST_STEP0 = _env_flag("NMOE_MLA_FAILFAST_STEP0", "1")
_mla_step0_finite_checks_remaining = 1


def _step0_finite_guard_active() -> bool:
  return _MLA_FAILFAST_STEP0 and (_mla_step0_finite_checks_remaining > 0)


def _mark_step0_finite_guard_consumed() -> None:
  global _mla_step0_finite_checks_remaining
  if _mla_step0_finite_checks_remaining > 0:
    _mla_step0_finite_checks_remaining -= 1


def _check_flashmla_varlen_available() -> tuple[bool, str]:
  try:
    from flash_mla import flash_attn_varlen_func as _flash_attn_varlen_func  # noqa: F401
    return True, ""
  except Exception as e:
    return False, str(e)


_FLASHMLA_VARLEN_AVAILABLE, _FLASHMLA_VARLEN_ERR = _check_flashmla_varlen_available()


# Module-level workspace cache for MLA backward pass.
#
# Design goals:
# - Avoid per-backward allocation churn (token/s stability, allocator pressure)
# - Bounded growth: one buffer per device, grown in-place as needed
# - Scratch-only: not part of module state / checkpoints
_mla_workspace_cache: dict[tuple[int, int], torch.Tensor] = {}
_mla_uniform_cu_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _is_compiling() -> bool:
  try:
    return bool(torch.compiler.is_compiling())
  except Exception:
    dyn = getattr(torch, "_dynamo", None)
    if dyn is None:
      return False
    fn = getattr(dyn, "is_compiling", None)
    return bool(fn()) if callable(fn) else False


def _stream_id_for_cache(device: torch.device) -> int:
  """Best-effort stable stream identifier for per-stream workspace caches."""
  stream = torch.cuda.current_stream(device)
  raw = getattr(stream, "cuda_stream", None)
  if raw is None:
    raw = getattr(stream, "stream_id", None)
    if callable(raw):
      raw = raw()
  try:
    return int(raw) if raw is not None else id(stream)
  except Exception:
    return id(stream)


def _get_mla_workspace(device: torch.device, workspace_bytes: int) -> torch.Tensor:
  """Get a cached workspace buffer for MLA backward.

  Returns a `torch.uint8` CUDA tensor with `numel() >= workspace_bytes`.
  """
  if workspace_bytes > _MLA_WORKSPACE_CACHE_MAX_BYTES:
    return torch.empty((workspace_bytes,), device=device, dtype=torch.uint8)
  dev_idx = device.index if device.index is not None else torch.cuda.current_device()
  stream_id = _stream_id_for_cache(device)
  key = (dev_idx, stream_id)
  buf = _mla_workspace_cache.get(key)
  if buf is None or buf.numel() < workspace_bytes:
    buf = torch.empty((workspace_bytes,), device=device, dtype=torch.uint8)
    _mla_workspace_cache[key] = buf
    # Keep only a bounded number of stream-keyed workspaces per device.
    dev_keys = [k for k in _mla_workspace_cache.keys() if k[0] == dev_idx]
    while len(dev_keys) > _MLA_STREAM_CACHE_LIMIT:
      _mla_workspace_cache.pop(dev_keys[0], None)
      dev_keys.pop(0)
  return buf


def _get_uniform_cu(device: torch.device, bsz: int, seqlen: int) -> torch.Tensor:
  """Get cached uniform cu-seqlens tensor [0, S, 2S, ..., B*S] on target device."""
  dev_idx = device.index if device.index is not None else torch.cuda.current_device()
  key = (dev_idx, int(bsz), int(seqlen))
  cu = _mla_uniform_cu_cache.get(key)
  if cu is None or cu.device != device:
    cu = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, device=device, dtype=torch.int32)
    _mla_uniform_cu_cache[key] = cu
  # Inductor/AOT functionalization cannot lower record_stream in compiled graphs.
  if not _is_compiling():
    cu.record_stream(torch.cuda.current_stream(device))
  return cu


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
    out: torch.Tensor | None = None,
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
  _require(cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 1, "cu_seqlens must be 1D with at least one element.")
  if cu_seqlens.dtype != torch.int32:
    cu_seqlens = cu_seqlens.to(dtype=torch.int32)
  if cu_seqlens.device != device:
    cu_seqlens = cu_seqlens.to(device=device, non_blocking=True)
  if not cu_seqlens.is_contiguous():
    cu_seqlens = cu_seqlens.contiguous()

  pos = torch.arange(seqlen, device=device, dtype=cu_seqlens.dtype)
  total_tokens = cu_seqlens[-1]

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
  valid_mask = (q_idx < total_tokens) & (k_idx < total_tokens)

  q_doc = doc_ids.unsqueeze(1)  # [seqlen, 1]
  k_doc = doc_ids.unsqueeze(0)  # [1, seqlen]
  doc_mask = q_doc == k_doc  # [seqlen, seqlen]

  mask = causal_mask & doc_mask & valid_mask
  if out is not None:
    out.copy_(mask)
    return out
  return mask


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
    # Build block-diagonal causal masks directly into final [B, 1, S, S] buffer.
    mask = torch.empty((bsz, 1, seqlen, seqlen), device=device, dtype=torch.bool)
    for b in range(bsz):
      _build_block_causal_mask(cu_seqlens_list[b], seqlen, device, out=mask[b, 0])

    # SDPA expects [B, H, S, D] layout
    q_t = q.transpose(1, 2)  # [B, H, S, D_qk]
    k_t = k.transpose(1, 2)  # [B, H, S, D_qk]
    v_t = v.transpose(1, 2)  # [B, H, S, D_v]

    # Single SDPA call with block-diagonal causal mask
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=mask,
        scale=softmax_scale,
    )

    # Back to [B, S, H, D_v]
    return out.transpose(1, 2)


def _mla_flashmla_packed_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    cu_seqlens_list: list[torch.Tensor],
) -> torch.Tensor:
  from flash_mla import flash_attn_varlen_func

  bsz, seqlen, n_heads, _ = q.shape
  d_v = v.shape[-1]
  device = q.device

  merged_cu_parts: list[torch.Tensor] = [torch.zeros((1,), device=device, dtype=torch.int32)]
  token_counts_t: list[torch.Tensor] = []
  start_checks: list[torch.Tensor] = []
  monotonic_checks: list[torch.Tensor] = []
  has_end_boundary: list[bool] = []
  has_any_end_boundary = False

  for b in range(bsz):
    cu_in = cu_seqlens_list[b]
    _require(cu_in.ndim == 1 and cu_in.numel() >= 1, f"cu_seqlens[{b}] must be 1D with at least one element.")

    cu = cu_in
    _require(
      cu.device == device and cu.dtype == torch.int32,
      f"cu_seqlens[{b}] must be CUDA int32 on device {device}, got {cu.device}/{cu.dtype}."
    )
    _require(cu.is_contiguous(), f"cu_seqlens[{b}] must be contiguous.")

    has_end = cu.numel() > 1
    has_end_boundary.append(has_end)
    has_any_end_boundary = has_any_end_boundary or has_end
    token_counts_t.append(cu[-1])
    if _VALIDATE_PACKED_CU:
      start_checks.append(cu[0] == 0)
    if has_end:
      deltas = cu[1:] - cu[:-1]
      if _VALIDATE_PACKED_CU:
        monotonic_checks.append(torch.all(deltas >= 0))

  token_counts = torch.stack(token_counts_t, dim=0)
  if _VALIDATE_PACKED_CU:
    _require(
      bool(torch.all((token_counts >= 0) & (token_counts <= seqlen)).item()),
      f"Packed cu_seqlens total_tokens must be in [0, {seqlen}].",
    )
    has_end = torch.tensor(has_end_boundary, device=device, dtype=torch.bool)
    _require(
      bool(torch.all(has_end | (token_counts == 0)).item()),
      "Packed cu_seqlens must include an end boundary when total_tokens > 0.",
    )
    if start_checks:
      _require(bool(torch.stack(start_checks).all().item()), "Packed cu_seqlens must start at 0.")
    if monotonic_checks:
      _require(bool(torch.stack(monotonic_checks).all().item()), "Packed cu_seqlens must be nondecreasing.")

  # Fast empty-pack path without CUDA scalar readback.
  # Canonical packed inputs represent empty sequences as cu_seqlens=[0].
  if not has_any_end_boundary:
    return torch.zeros((bsz, seqlen, n_heads, d_v), device=device, dtype=v.dtype)

  batch_offsets = torch.zeros((bsz + 1,), device=device, dtype=torch.int32)
  batch_offsets[1:] = token_counts.cumsum(dim=0)
  for b in range(bsz):
    cu = cu_seqlens_list[b]
    if cu.numel() > 1:
      merged_cu_parts.append(cu[1:] + batch_offsets[b])

  token_pos = torch.arange(seqlen, device=device, dtype=torch.int32).unsqueeze(0)
  valid_token = token_pos < token_counts.unsqueeze(1)
  q_merged = q[valid_token]
  k_merged = k[valid_token]
  v_merged = v[valid_token]
  merged_cu = torch.cat(merged_cu_parts, dim=0)
  max_seqlen = int(seqlen)

  out_merged, _ = flash_attn_varlen_func(
      q_merged,
      k_merged,
      v_merged,
      cu_seqlens_qo=merged_cu,
      cu_seqlens_kv=merged_cu,
      max_seqlen_qo=max_seqlen,
      max_seqlen_kv=max_seqlen,
      causal=True,
      softmax_scale=softmax_scale,
      is_varlen=True,
  )

  # Always scatter from merged output to avoid CUDA scalar .item() sync checks.
  out = torch.zeros((bsz, seqlen, n_heads, d_v), device=device, dtype=out_merged.dtype)
  out[valid_token] = out_merged
  return out


def _mla_flashmla_uniform_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
  """FlashMLA forward for non-packed causal attention (uniform sequence lengths)."""
  from flash_mla import flash_attn_varlen_func

  bsz, seqlen, n_heads, d_qk = q.shape
  d_v = v.shape[-1]
  total = bsz * seqlen
  q_ = q.reshape(total, n_heads, d_qk).contiguous()
  k_ = k.reshape(total, n_heads, d_qk).contiguous()
  v_ = v.reshape(total, n_heads, d_v).contiguous()
  _require_finite("flashmla_uniform.q", q_)
  _require_finite("flashmla_uniform.k", k_)
  _require_finite("flashmla_uniform.v", v_)

  cu = _get_uniform_cu(q.device, bsz, seqlen)
  out, _ = flash_attn_varlen_func(
      q_,
      k_,
      v_,
      cu_seqlens_qo=cu,
      cu_seqlens_kv=cu,
      max_seqlen_qo=int(seqlen),
      max_seqlen_kv=int(seqlen),
      causal=True,
      softmax_scale=softmax_scale,
      is_varlen=True,
  )
  _require_finite("flashmla_uniform.out", out)
  return out.reshape(bsz, seqlen, n_heads, d_v)


class _MlaFa4FwdFlashMlaBwd(torch.autograd.Function):
  """MLA attention using FA4 or FlashMLA-SM100 forward + FlashMLA backward.

  Production path uses CUDA kernels only (FA4 or FlashMLA-SM100 + FlashMLA backward);
  SDPA exists only as explicit debug fallback.
  """
  @staticmethod
  def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    _sm100_only(q.device)
    _require(q.is_cuda and k.is_cuda and v.is_cuda, "MLA FA4+FlashMLA requires CUDA tensors.")
    _require(
      q.device == k.device and q.device == v.device,
      f"MLA FA4+FlashMLA requires q/k/v on same device, got {q.device}, {k.device}, {v.device}.",
    )
    _require(q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16,
             "MLA FA4+FlashMLA requires BF16 inputs.")
    _require(q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Expected q/k/v with shape [B, S, H, D].")

    bsz, seqlen, n_heads, d_qk = q.shape
    _require(k.shape == (bsz, seqlen, n_heads, d_qk), "k must match q shape.")
    d_v = v.shape[-1]
    _require(v.shape == (bsz, seqlen, n_heads, d_v), "v must have shape [B, S, H, Dv].")
    _require(d_qk == 192 and d_v == 128, f"Only (d_qk, d_v) = (192, 128) is supported. Got ({d_qk}, {d_v}).")

    # Hard requirement: do not proceed without FA4 forward + FlashMLA backward.
    # Environment contract: `third_party/flash_attn` is on PYTHONPATH (so
    # `flash_attn.cute` is importable), and `nmoe.csrc.flashmla_sm100` is built.
    _flash_attn_fwd, _flashmla = get_fa4_flashmla_modules()

    total = bsz * seqlen
    q_ = q.reshape(total, n_heads, d_qk).contiguous()
    k_ = k.reshape(total, n_heads, d_qk).contiguous()
    v_ = v.reshape(total, n_heads, d_v).contiguous()
    force_finite = _step0_finite_guard_active()
    _require_finite("fa4.q", q_, force=force_finite)
    _require_finite("fa4.k", k_, force=force_finite)
    _require_finite("fa4.v", v_, force=force_finite)

    cu = _get_uniform_cu(q.device, bsz, seqlen)

    if _USE_FLASHMLA_SM100_FWD:
      _require(
        hasattr(_flashmla, "dense_prefill_fwd"),
        "NMOE_MLA_USE_FLASHMLA_SM100_FWD=1 requires nmoe.csrc.flashmla_sm100.dense_prefill_fwd; rebuild nmoe/csrc.",
      )
      with _nvtx("attn/flashmla_sm100_fwd"):
        # FlashMLA forward uses a fixed-size workspace scratch buffer.
        fwd_workspace = _get_mla_workspace(q.device, 32 * 1024 * 1024)
        out = torch.empty((total, n_heads, d_v), device=q.device, dtype=q_.dtype)
        # FlashMLA forward requires lse stride(0) == 1.
        lse = torch.empty((n_heads, total), device=q.device, dtype=torch.float32).transpose(0, 1)
        _flashmla.dense_prefill_fwd(
            fwd_workspace,
            q_,
            k_,
            v_,
            cu,
            cu,
            out,
            lse,
            1,  # causal
            float(softmax_scale),
            seqlen,
            seqlen,
            False,  # non-packed dense path: fixed-length [B, S]
        )
    else:
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
    _require(out.shape == (total, n_heads, d_v),
             f"FA4 output shape mismatch: expected {(total, n_heads, d_v)}, got {tuple(out.shape)}.")
    _require(lse.ndim == 2, f"FA4 lse must be 2D, got shape={tuple(lse.shape)}.")
    if _nan_debug_active():
      out_finite = torch.isfinite(out)
      if not bool(out_finite.all().item()):
        numel = int(out.numel())
        finite = int(out_finite.sum().item())
        nan = int(torch.isnan(out).sum().item())
        inf = int(torch.isinf(out).sum().item())
        q_abs = float(q_.abs().amax().item())
        k_abs = float(k_.abs().amax().item())
        v_abs = float(v_.abs().amax().item())
        q_rms = float(q_.float().pow(2).mean().sqrt().item())
        k_rms = float(k_.float().pow(2).mean().sqrt().item())
        v_rms = float(v_.float().pow(2).mean().sqrt().item())
        raise RuntimeError(
          f"[NANDBG][mla] non-finite fa4.out shape={tuple(out.shape)} dtype={out.dtype} "
          f"finite={finite}/{numel} nan={nan} inf={inf} "
          f"q_abs_max={q_abs:.6g} k_abs_max={k_abs:.6g} v_abs_max={v_abs:.6g} "
          f"q_rms={q_rms:.6g} k_rms={k_rms:.6g} v_rms={v_rms:.6g} "
          f"softmax_scale={float(softmax_scale):.6g}"
        )
    _require_finite("fa4.out", out, force=force_finite)
    _require_finite("fa4.lse", lse, force=force_finite)

    # FlashMLA expects lse as [total, H] float32 with stride(0) == 1.
    if lse.shape == (n_heads, total):
      lse_t = lse.transpose(0, 1)
    elif lse.shape == (total, n_heads):
      lse_t = lse
    else:
      raise RuntimeError(
        f"Unexpected FA4 lse shape {tuple(lse.shape)}; expected {(n_heads, total)} or {(total, n_heads)}."
      )
    if lse_t.stride(0) != 1:
      lse_t = lse_t.transpose(0, 1).contiguous().transpose(0, 1)
    _require(lse_t.shape == (total, n_heads),
             f"FlashMLA lse shape mismatch: expected {(total, n_heads)}, got {tuple(lse_t.shape)}.")
    _require(lse_t.stride(0) == 1, f"FlashMLA lse stride(0) must be 1, got {lse_t.stride(0)}.")

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
    _require_finite("flashmla_bwd.d_o", d_o)

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
          False,  # non-packed dense path: fixed-length [B, S]
      )
    _require_finite("flashmla_bwd.dq", dq)
    _require_finite("flashmla_bwd.dk", dk)
    _require_finite("flashmla_bwd.dv", dv)

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
    self.softmax_scale = (self.qk_head_dim ** -0.5) * _MLA_SOFTMAX_SCALE_MULT
    if not math.isfinite(self.softmax_scale) or self.softmax_scale <= 0.0:
      raise RuntimeError(
        f"Invalid MLA softmax_scale={self.softmax_scale}. "
        f"qk_head_dim={self.qk_head_dim}, scale_mult={_MLA_SOFTMAX_SCALE_MULT}."
      )

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
    _require_finite("mla.x", x)
    _require_finite("mla.cos", cos)
    _require_finite("mla.sin", sin)

    q = self.wq_b(self.q_norm(self.wq_a(x)))
    _require_finite("mla.q_pre_rope", q)
    q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
    # In-place partial RoPE: rotates q[..., nope_dim:] leaving q[..., :nope_dim] unchanged.
    # Eliminates split+cat overhead (3 kernels -> 1 kernel).
    rotate_pe_partial(q, cos, sin, nope_dim=self.qk_nope_head_dim)
    _require_finite("mla.q", q)
    # Rotate K rope slice in-place before split to avoid non-contiguous view copies.
    kv = self.wkv_a(x)
    _require_finite("mla.kv_pre_rope", kv)
    rotate_pe_partial(kv.unsqueeze(2), cos, sin, nope_dim=self.kv_lora_rank)
    _require_finite("mla.kv_post_rope", kv)
    kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.unsqueeze(2)
    kv = self.wkv_b(self.kv_norm(kv))
    _require_finite("mla.kv_proj", kv)
    kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
    _require_finite("mla.k", k)
    _require_finite("mla.v", v)
    with record_function("attn.kernel[mla]"):
      if cu_seqlens is not None:
        if _PACKED_ATTN_BACKEND != "flashmla":
          raise RuntimeError(
            f"Invalid NMOE_PACKED_ATTN_BACKEND={_PACKED_ATTN_BACKEND!r}. "
            "Production path requires: flashmla."
          )
        if not _FLASHMLA_VARLEN_AVAILABLE:
          raise RuntimeError(
            "Packed MLA requested FlashMLA backend but FlashMLA varlen is unavailable "
            f"({_FLASHMLA_VARLEN_ERR})."
          )
        output = _mla_flashmla_packed_forward(q, k, v, self.softmax_scale, cu_seqlens)
      else:
        # Standard causal attention (no packing)
        if _REQUIRE_FLASHMLA and _USE_SDPA:
          raise RuntimeError(
            "NMOE_REQUIRE_FLASHMLA=1 requires NMOE_USE_FA4=1."
          )
        if _USE_SDPA:
          raise RuntimeError(
            "Non-packed SDPA fallback is disabled in production. "
            "Set NMOE_USE_FA4=1 to force FA4+FlashMLA path."
          )
        else:
          if _USE_FLASHMLA_VARLEN:
            if not _FLASHMLA_VARLEN_AVAILABLE:
              raise RuntimeError(
                "FlashMLA varlen kernel is unavailable for non-packed MLA "
                f"({_FLASHMLA_VARLEN_ERR})."
              )
            output = _mla_flashmla_uniform_forward(q, k, v, self.softmax_scale)
          else:
            output = _MlaFa4FwdFlashMlaBwd.apply(q, k, v, self.softmax_scale)
    output = output.reshape(bsz, seqlen, self.n_heads * self.v_head_dim)
    force_finite = _step0_finite_guard_active()
    _require_finite("mla.out_attn", output, force=force_finite)
    out = self.wo(output)
    _require_finite("mla.out_wo", out, force=force_finite)
    if force_finite:
      _mark_step0_finite_guard_consumed()
    return out
