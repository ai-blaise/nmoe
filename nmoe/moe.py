"""MoE computation: grouped GEMM and fused autograd functions.

This module contains:
- _MoEBlockscaledFused: Autograd function for FP8/NVFP4 MoE forward/backward
- dequant_nvfp4_to_bf16_transient(): Helper for NVFP4 primary backward

The dispatch/combine infrastructure is in rdep.py.
The Router and MoE nn.Module classes are in model.py.
"""
from __future__ import annotations
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from nmoe.csrc import rdep as _C
from nmoe.blockscaled.grouped import expert_blockscaled
from nmoe.cuda_errors import cuda_error_context

if TYPE_CHECKING:
  from nmoe.rdep import Rdep


def _env_flag(name: str, default: str = "0") -> bool:
  return os.getenv(name, default) in ("1", "true", "True")


# Disabled by default on production hot path; opt-in when debugging capacity.
_CHECK_DROPPED_TOKENS = os.getenv('NMOE_CHECK_DROPPED_TOKENS', '0') in ('1', 'true', 'True')
_CHECK_DROPPED_TOKENS_INTERVAL = max(1, int(os.getenv('NMOE_CHECK_DROPPED_TOKENS_INTERVAL', '1000')))
_PER_STREAM_CACHE_MAX = max(4, int(os.getenv("NMOE_PER_STREAM_CACHE_MAX", "32")))
_NVTX_ENABLED = (
  _env_flag("NMOE_NVTX", "0")
  and torch.cuda.is_available()
  and hasattr(torch.cuda, 'nvtx')
  and hasattr(torch.cuda.nvtx, 'range')
)
_HAS_SEND_DGATE_DIST_BF16_OUT_BF16 = hasattr(_C, "send_dgate_dist_bf16_out_bf16")
_HAS_SCATTER_GATE_BF16_OUT_BF16 = hasattr(_C, "scatter_gate_bf16_out_bf16")
_HAS_SCATTER_DX_DIST_FROM_PAD_BF16 = hasattr(_C, "scatter_dx_dist_from_pad_bf16")
_HAS_ZERO_PADDING_ROWS_BF16 = hasattr(_C, "zero_padding_rows_bf16")
_HAS_BF16_WGRAD_W2_HOST_OFFS = hasattr(_C, "bf16_wgrad_w2_cublaslt_host_offs")
_HAS_BF16_WGRAD_W13_PAIR_HOST_OFFS = hasattr(_C, "bf16_wgrad_w13_pair_cublaslt_host_offs")
_HAS_BF16_WGRAD_HOST_OFFS = bool(getattr(_C, "bf16_wgrad_host_offs_supported", False))
_HAS_BF16_PREPARE_OFFS_PAD_HOST = bool(getattr(_C, "bf16_prepare_offs_pad_host_supported", False))
_HAS_DEQUANT_FP8_TO_BF16_MMA_SF = hasattr(_C, "dequant_fp8_to_bf16_mma_sf")
_HAS_DEQUANT_NVFP4_TO_BF16_MMA_SF = hasattr(_C, "dequant_nvfp4_to_bf16_mma_sf")
_CT_NVFP4_TO_BF16_MAX_TRANSPOSE_MODE = int(getattr(_C, "ct_nvfp4_to_bf16_max_transpose_mode", 1))
_NULL_NVTX_CTX = nullcontext()


def _nvtx(tag: str):
    if _NVTX_ENABLED:
        return torch.cuda.nvtx.range(tag)
    return _NULL_NVTX_CTX


def _cache_put_capped(cache: dict, key, value) -> None:
    """Insert into an insertion-ordered dict with bounded cardinality."""
    if key in cache:
        cache[key] = value
        return
    if len(cache) >= _PER_STREAM_CACHE_MAX:
        # Dict preserves insertion order; drop oldest stream entry.
        cache.pop(next(iter(cache)))
    cache[key] = value


def _get_cached_offs_pad(rdep: "Rdep", device: torch.device, n_local: int) -> torch.Tensor:
    """Return reusable per-stream device buffer for expert padded offsets."""
    stream = torch.cuda.current_stream(device)
    sid = int(stream.cuda_stream)
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (int(dev_idx), sid, int(n_local))
    cache = getattr(rdep, "_offs_pad_dev_cache", None)
    if cache is None:
        cache = {}
        rdep._offs_pad_dev_cache = cache
    offs_pad = cache.get(key)
    if offs_pad is None or offs_pad.device != device or offs_pad.dtype != torch.int32:
        offs_pad = torch.empty(int(n_local), device=device, dtype=torch.int32)
        _cache_put_capped(cache, key, offs_pad)
    return offs_pad


def _get_cached_pinned_m_host(rdep: "Rdep") -> torch.Tensor:
    """Return reusable per-stream pinned host scalar for dispatch M_pad/M_recv."""
    dev = torch.cuda.current_device()
    stream = torch.cuda.current_stream(dev)
    sid = int(stream.cuda_stream)
    key = (int(dev), sid)
    cache = getattr(rdep, "_pinned_M_host_cache", None)
    if cache is None:
        cache = {}
        rdep._pinned_M_host_cache = cache
    m_host = cache.get(key)
    if m_host is None or m_host.dtype != torch.int32 or int(m_host.numel()) != 1:
        m_host = torch.empty(1, dtype=torch.int32, device='cpu', pin_memory=True)
        _cache_put_capped(cache, key, m_host)
    return m_host


def _require_contiguous(t: torch.Tensor, name: str, *, dtype: torch.dtype | None = None) -> None:
    if dtype is not None and t.dtype != dtype:
        raise RuntimeError(f"{name} dtype mismatch: expected {dtype}, got {t.dtype}")
    if not t.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous for CUDA no-fallback path")


def _as_contiguous_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype == dtype:
        return t if t.is_contiguous() else t.contiguous()
    return t.to(dtype=dtype).contiguous()


_BWD_RUNTIME_CONTRACT_CACHE: set[tuple[bool, bool, str]] = set()


def _validate_backward_runtime_contract_once(
    *,
    is_dist: bool,
    use_dist_blockscaled_meta: bool,
    profile: str,
) -> None:
    """Validate required CUDA bindings once per backward runtime mode."""
    key = (bool(is_dist), bool(use_dist_blockscaled_meta), str(profile))
    if key in _BWD_RUNTIME_CONTRACT_CACHE:
        return

    missing: list[str] = []
    if not _HAS_ZERO_PADDING_ROWS_BF16:
        missing.append("zero_padding_rows_bf16")
    if not _HAS_BF16_WGRAD_W2_HOST_OFFS:
        missing.append("bf16_wgrad_w2_cublaslt_host_offs")
    if not _HAS_BF16_WGRAD_W13_PAIR_HOST_OFFS:
        missing.append("bf16_wgrad_w13_pair_cublaslt_host_offs")
    if not _HAS_BF16_WGRAD_HOST_OFFS:
        missing.append("bf16_wgrad_host_offs_supported")
    if not _HAS_BF16_PREPARE_OFFS_PAD_HOST:
        missing.append("bf16_prepare_offs_pad_host_supported")

    if is_dist:
        if not _HAS_SEND_DGATE_DIST_BF16_OUT_BF16:
            missing.append("send_dgate_dist_bf16_out_bf16")
        if not _HAS_SCATTER_DX_DIST_FROM_PAD_BF16:
            missing.append("scatter_dx_dist_from_pad_bf16")
    else:
        if not _HAS_SCATTER_GATE_BF16_OUT_BF16:
            missing.append("scatter_gate_bf16_out_bf16")

    if use_dist_blockscaled_meta:
        if profile == "fp8" and not _HAS_DEQUANT_FP8_TO_BF16_MMA_SF:
            missing.append("dequant_fp8_to_bf16_mma_sf")
        if profile == "nvfp4" and not _HAS_DEQUANT_NVFP4_TO_BF16_MMA_SF:
            missing.append("dequant_nvfp4_to_bf16_mma_sf")

    if missing:
        raise RuntimeError(
            "Missing required CUDA bindings/support for blockscaled MoE backward no-fallback path: "
            + ", ".join(missing)
            + ". Rebuild nmoe/csrc."
        )

    _BWD_RUNTIME_CONTRACT_CACHE.add(key)


def dequant_nvfp4_to_bf16_transient(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
    transpose: bool = True,
) -> torch.Tensor:
    """Dequantize NVFP4 compressed_tensors triplet to BF16 on GPU via CUDA kernel.

    Uses ct_nvfp4_to_bf16 CUDA kernel — all arithmetic in GPU registers,
    zero FP32 intermediates in global memory.  Only allocates the output
    BF16 tensor (1/8 of the FP32 intermediate that the old Python path needed).

    Used to transiently populate W1/W3/W2 parameters for backward pass
    when operating in NVFP4 primary mode.

    Args:
        packed: [E, out_dim, in_dim//2] uint8 (2 E2M1 nibbles per byte)
        scale: [E, out_dim, in_dim//group_size] float8_e4m3fn
        global_scale: [E, 1] or [E] float32
        group_size: Elements per scale group (default 16)
        transpose: If True, transpose last two dims (HF→nmoe layout)

    Returns:
        [E, in_dim, out_dim] bfloat16 if transpose else [E, out_dim, in_dim] bfloat16
    """
    E = packed.shape[0]
    M = packed.shape[1]       # out_dim (rows per expert)
    K = packed.shape[2] * 2   # in_dim (each byte holds 2 elements)
    total_M = E * M
    device = packed.device
    stream = torch.cuda.current_stream(device)

    # Flatten to [E*M, K/2] for the kernel.
    packed_flat = packed.reshape(total_M, K // 2)
    if not packed_flat.is_contiguous():
        packed_flat = packed_flat.contiguous()
    n_groups = K // group_size
    scale_u8 = scale.view(torch.uint8).reshape(total_M, n_groups)
    if not scale_u8.is_contiguous():
        scale_u8 = scale_u8.contiguous()
    scale_flat = scale_u8
    gs_flat = global_scale.reshape(E)
    if gs_flat.dtype != torch.float32:
        gs_flat = gs_flat.float()
    if not gs_flat.is_contiguous():
        gs_flat = gs_flat.contiguous()

    if transpose and _CT_NVFP4_TO_BF16_MAX_TRANSPOSE_MODE < 2:
        raise RuntimeError(
            "ct_nvfp4_to_bf16 transpose mode=2 is required for production no-fallback path. "
            "Rebuild nmoe/csrc extension."
        )
    # Allocate output in direct expert-major transpose layout for no-fallback path.
    if transpose:
        out_bf16 = torch.empty(E, K, M, dtype=torch.bfloat16, device=device)
        transpose_mode = 2  # Write [E, K, M] directly in-kernel.
    else:
        out_bf16 = torch.empty(total_M, K, dtype=torch.bfloat16, device=device)
        transpose_mode = 0

    # Run GPU kernel — zero FP32 global memory, all in registers
    _C.ct_nvfp4_to_bf16(
        packed_flat.data_ptr(),
        scale_flat.data_ptr(),
        gs_flat.data_ptr(),
        out_bf16.data_ptr(),
        total_M, K, group_size,
        1,  # gs_stride=1 (per-expert global scale)
        M,  # expert_rows=M (rows per expert for indexing ct_gs)
        transpose_mode,
        stream,
    )

    # Reshape to [E, ...] form
    if not transpose:
        out_bf16 = out_bf16.reshape(E, M, K)

    return out_bf16


class _MoEBlockscaledFused(torch.autograd.Function):
  @staticmethod
  def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
              W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor, W_cache,
              fused_eco=None, moe_ref=None) -> torch.Tensor:
    if fused_eco is not None and moe_ref is None:
      raise RuntimeError("fused_eco requires moe_ref in _MoEBlockscaledFused.forward")
    if fused_eco is not None and not getattr(moe_ref, "_nvfp4_primary", False):
      raise RuntimeError(
        "fused_eco is supported only with NVFP4-primary MoE weights "
        "(set NVFP4 buffers and _nvfp4_primary=true)."
      )
    if moe_ref is not None and moe_ref._nvfp4_primary and fused_eco is None:
      raise RuntimeError(
        "NVFP4 primary mode requires fused ECO for MoE forward/backward "
        "(eco_fused_backward=true and eco_require_cuda=true)."
      )

    device = x.device
    stream = torch.cuda.current_stream(device)

    x = _as_contiguous_dtype(x, torch.bfloat16)
    eid = _as_contiguous_dtype(eid, torch.int32)
    gates_fp32 = _as_contiguous_dtype(gates.detach(), torch.float32)

    T, H = x.shape
    K = int(eid.shape[1])
    E = int(rdep.n_local)
    is_dist = int(rdep.world) > 1
    if is_dist:
      need = int(T) * int(K) * int(rdep.world)
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={rdep.world}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    # Option A: Use BF16 dispatch + local quantization
    # This ensures Xe_pad (BF16) is available for backward STE
    offs_pad = _get_cached_offs_pad(rdep, device, E)
    # Use per-stream pinned host scalar to avoid cross-stream metadata races.
    M_host = _get_cached_pinned_m_host(rdep)

    with _nvtx("moe_bs/fwd_dispatch_meta"), cuda_error_context("dispatch_meta_blockscaled"):
      M_recv = _C.dispatch_meta_blockscaled(
        x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
        int(T), int(K),
        offs_pad.data_ptr(), M_host.data_ptr(),
        stream,
      )
    if moe_ref is not None and moe_ref.training:
      mode = rdep._mode
      if mode == 'ipc':
        moe_ref._runtime_rdep_dispatch_blockscaled_ipc_calls += 1
      elif mode == 'hybrid':
        moe_ref._runtime_rdep_dispatch_blockscaled_hybrid_calls += 1

    # Optional diagnostics: disabled by default to avoid host sync in production.
    if _CHECK_DROPPED_TOKENS:
      if not hasattr(_MoEBlockscaledFused, '_dispatch_count'):
        _MoEBlockscaledFused._dispatch_count = 0
      _MoEBlockscaledFused._dispatch_count += 1
      if (_MoEBlockscaledFused._dispatch_count - 1) % _CHECK_DROPPED_TOKENS_INTERVAL == 0:
        dropped = _C.get_dropped_blockscaled(stream)
        # Store dropped count on the MoE module for training loop monitoring
        if moe_ref is not None:
          moe_ref._last_dropped_count = dropped
        if dropped > 0:
          raise RuntimeError(
            f"[RDEP] dropped {dropped:,} tokens due to capacity overflow "
            f"(capacity={rdep.capacity:,}, T={T}, K={K}). "
            "Dropped-token continuation is disabled on the production no-fallback path. "
            "Increase rdep_capacity or reduce batch_size."
          )
      elif moe_ref is not None and not hasattr(moe_ref, '_last_dropped_count'):
        moe_ref._last_dropped_count = 0
    elif moe_ref is not None:
      moe_ref._last_dropped_count = 0

    out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
    if M_recv < 0:
      raise RuntimeError(
        f"dispatch_meta_blockscaled failed with error code {int(M_recv)} "
        f"(T={T}, K={K}, mode={rdep._mode}, world={rdep.world})."
      )
    if M_recv <= 0:
      # DeepEP collectiveness: every rank must participate in return_scatter
      if is_dist:
        dummy_ye_pad = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.return_scatter_from_pad_blockscaled(dummy_ye_pad.data_ptr(), out_f32.data_ptr(), 0, int(T), int(K), stream)
      ctx.rdep = rdep
      ctx.T = int(T)
      ctx.H = int(H)
      ctx.K = int(K)
      ctx.fused_eco = fused_eco
      ctx.moe_ref = moe_ref
      if fused_eco is not None:
        # Save empty placeholders — backward will dequant from NVFP4 via moe_ref
        _dev = x.device
        _e0 = torch.empty(0, dtype=torch.bfloat16, device=_dev)
        ctx.save_for_backward(x, eid, gates_fp32, _e0, _e0, _e0)
      else:
        ctx.save_for_backward(x, eid, gates_fp32, W1, W3, W2)
      return out_f32.to(dtype=torch.bfloat16)

    # Avoid host scalar sync on hot path: M_pad is deterministic from M_recv.
    align = 128
    M_pad = ((int(M_recv) + int(E) * (align - 1)) // align) * align
    max_pad = int(rdep.capacity) + int(E) * (align - 1)
    if M_pad > max_pad:
      M_pad = max_pad

    # Gather blockscaled activations into padded layout (quantized + packed SF)
    pack_factor = 2 if rdep.profile == 'fp8' else 4
    Hp = H // pack_factor
    sf_k = H // 32
    sf_k_pad = ((sf_k + 3) // 4) * 4
    Xe_q = torch.empty(int(M_pad), Hp, device=device, dtype=torch.uint16)
    Xe_sf = torch.empty(int(M_pad), sf_k_pad, device=device, dtype=torch.uint8)
    with _nvtx("moe_bs/fwd_dispatch"):
      _C.gather_xe_blockscaled(Xe_q.data_ptr(), Xe_sf.data_ptr(), int(M_recv), int(M_pad), stream)

    # Expert compute (blockscaled)
    with _nvtx("moe_bs/fwd_expert_compute"):
      Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad, capacity_rows=int(rdep.capacity))

    with _nvtx("moe_bs/fwd_combine"):
      _require_contiguous(Ye_pad, "Ye_pad", dtype=torch.bfloat16)
      _C.return_scatter_from_pad_blockscaled(
        Ye_pad.data_ptr(),
        out_f32.data_ptr(),
        int(M_recv), int(T), int(K),
        stream,
      )
    ctx.rdep = rdep
    ctx.T = int(T)
    ctx.H = int(H)
    ctx.K = int(K)
    ctx.fused_eco = fused_eco
    ctx.moe_ref = moe_ref
    if fused_eco is not None:
      # Save empty placeholders — backward will dequant from NVFP4 via moe_ref
      _dev = x.device
      _e0 = torch.empty(0, dtype=torch.bfloat16, device=_dev)
      ctx.save_for_backward(x, eid, gates_fp32, _e0, _e0, _e0)
    else:
      ctx.save_for_backward(x, eid, gates_fp32, W1, W3, W2)
    return out_f32.to(dtype=torch.bfloat16)

  @staticmethod
  def backward(ctx, dOut: torch.Tensor):
    x, eid, gates_fp32, _W1, _W3, _W2 = ctx.saved_tensors
    rdep: Rdep = ctx.rdep
    fused_eco = ctx.fused_eco
    moe_ref = ctx.moe_ref

    if fused_eco is not None and moe_ref is None:
      raise RuntimeError("fused_eco requires moe_ref in _MoEBlockscaledFused.backward")
    if fused_eco is not None and not getattr(moe_ref, "_nvfp4_primary", False):
      raise RuntimeError(
        "fused_eco backward requires NVFP4-primary MoE weights. "
        "Non-primary fused_eco path is disabled."
      )
    if moe_ref is not None and moe_ref._nvfp4_primary and fused_eco is None:
      raise RuntimeError(
        "NVFP4 primary mode requires fused ECO for MoE forward/backward "
        "(eco_fused_backward=true and eco_require_cuda=true)."
      )

    device = dOut.device
    stream = torch.cuda.current_stream(device)

    dOut = _as_contiguous_dtype(dOut, torch.bfloat16)
    x = _as_contiguous_dtype(x, torch.bfloat16)
    eid = _as_contiguous_dtype(eid, torch.int32)
    gates_fp32 = _as_contiguous_dtype(gates_fp32, torch.float32)

    T = int(ctx.T)
    H = int(ctx.H)
    K = int(ctx.K)
    E = int(rdep.n_local)
    is_dist = int(rdep.world) > 1
    mode = getattr(rdep, "_mode", "ipc")
    rdep_profile = str(getattr(rdep, "profile", "")).lower()
    use_dist_blockscaled_meta = bool(
      is_dist and mode == "hybrid" and rdep_profile in ("fp8", "nvfp4")
    )
    nvfp4_mode_ok = bool(mode in {"ipc", "hybrid"})
    if rdep_profile == "nvfp4" and not nvfp4_mode_ok:
      raise RuntimeError(
        "NVFP4 backward requires distributed IPC/hybrid RDEP mode. "
        f"Got mode={mode!r}, world={rdep.world}."
      )
    require_fused_eco = bool(moe_ref is not None and getattr(moe_ref, "_nvfp4_primary", False))
    if require_fused_eco:
      if fused_eco is None:
        raise RuntimeError(
          "MoE NVFP4 blockscaled backward requires fused ECO on production no-fallback path."
        )
      if rdep_profile != "nvfp4":
        raise RuntimeError(
          f"NVFP4 primary backward requires rdep profile='nvfp4', got profile={rdep_profile!r}."
        )
      if not nvfp4_mode_ok:
        raise RuntimeError(
          "NVFP4 primary backward requires distributed IPC/hybrid RDEP mode. "
          f"Got mode={mode!r}, world={rdep.world}."
        )
    _validate_backward_runtime_contract_once(
      is_dist=is_dist,
      use_dist_blockscaled_meta=use_dist_blockscaled_meta,
      profile=rdep_profile,
    )
    if is_dist:
      # Use rdep.world (EP group size) for consistency with forward pass and auto-compute
      need = int(T) * int(K) * int(rdep.world)
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={rdep.world}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    # Option A: Use BF16 dispatch to get correct Xe_pad from all ranks
    # This fixes the distributed bug where local x was used for remote rows
    offs_pad = _get_cached_offs_pad(rdep, device, E)
    # Use per-stream pinned host scalar to avoid cross-stream metadata races.
    M_host = _get_cached_pinned_m_host(rdep)
    align = 128  # Required for blockscaled SF swizzle

    with _nvtx("moe_bs/bwd_dispatch_meta"):
      if use_dist_blockscaled_meta:
        dispatch_fn = "dispatch_meta_blockscaled"
        M_recv = _C.dispatch_meta_blockscaled(
          x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
          int(T), int(K),
          offs_pad.data_ptr(), M_host.data_ptr(),
          stream,
        )
      else:
        dispatch_fn = "dispatch_meta_bf16"
        M_recv = _C.dispatch_meta_bf16(
          x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
          int(T), int(K), align,
          offs_pad.data_ptr(), M_host.data_ptr(),
          stream,
        )
    if moe_ref is not None and moe_ref.training:
      mode = rdep._mode
      if mode == 'ipc':
        if use_dist_blockscaled_meta:
          moe_ref._runtime_rdep_dispatch_blockscaled_ipc_calls += 1
        else:
          moe_ref._runtime_rdep_dispatch_bf16_ipc_calls += 1
      elif mode == 'hybrid':
        if use_dist_blockscaled_meta:
          moe_ref._runtime_rdep_dispatch_blockscaled_hybrid_calls += 1
        else:
          moe_ref._runtime_rdep_dispatch_bf16_hybrid_calls += 1

    if M_recv < 0:
      raise RuntimeError(
        f"{dispatch_fn} failed with error code {int(M_recv)} "
        f"(T={T}, K={K}, mode={rdep._mode}, world={rdep.world})."
      )
    if M_recv <= 0:
      dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)

      # DeepEP collectiveness: still run distributed gather/scatter
      if is_dist:
        dGates_tk_bf16 = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)
        dummy_row_id = torch.empty(1, device=device, dtype=torch.int64)
        dummy_gate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        dummy_dye_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        dummy_dgate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        with cuda_error_context("gather_dy_nogate_dist_bf16 (blockscaled backward, M_recv=0)"):
          _C.gather_dy_nogate_dist_bf16(
            dOut.data_ptr(),
            eid.data_ptr(),
            dummy_row_id.data_ptr(),
            dummy_gate_sorted.data_ptr(),
            dummy_dye_sorted.data_ptr(),
            0, int(T), int(H), int(K),
            stream,
          )
        with cuda_error_context("send_dgate_dist_bf16_out_bf16 (blockscaled backward, M_recv=0)"):
          _C.send_dgate_dist_bf16_out_bf16(
            dummy_row_id.data_ptr(),
            dummy_dgate_sorted.data_ptr(),
            dGates_tk_bf16.data_ptr(),
            0, int(T), int(K),
            stream,
          )
        dummy_dxe_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        with cuda_error_context("scatter_dx_dist_bf16 (blockscaled backward, M_recv=0)"):
          _C.scatter_dx_dist_bf16(
            dummy_dxe_sorted.data_ptr(),
            dummy_row_id.data_ptr(),
            dX.data_ptr(),
            0, int(T), int(H), int(K),
            stream,
          )
      else:
        dGates_tk_bf16 = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)
      dGates = dGates_tk_bf16

      if fused_eco is not None:
        # Keep DP collectives aligned even if this rank received zero routed tokens.
        # We use local zero gradients in the same W2/W1/W3 order as the regular path,
        # so every rank executes identical collective counts.
        fused_eco.fused_update_zero(moe_ref, 'W2')
        fused_eco.fused_update_zero(moe_ref, 'W1')
        fused_eco.fused_update_zero(moe_ref, 'W3')
        # Only invalidate cache when weights were actually updated.
        # Non-final accumulation micro-steps update m/v only.
        if fused_eco.is_final_microstep:
          fused_eco.refresh_layer_cache(moe_ref)
        return None, dX, None, dGates, None, None, None, None, None, None

      dW1 = torch.zeros_like(_W1)
      dW3 = torch.zeros_like(_W3)
      dW2 = torch.zeros_like(_W2)
      return None, dX, None, dGates, dW1, dW3, dW2, None, None, None

    # When fused_eco is active, saved W1/W3/W2 are empty placeholders.
    # Dequant from NVFP4 on-the-fly — NEVER hold more than one BF16 expert
    # weight at a time to avoid OOM (each local expert weight tensor is
    # large: hundreds of MiB to >1 GiB depending geometry).
    _nvfp4_stagger = bool(require_fused_eco and fused_eco is not None)
    if not _nvfp4_stagger:
      W1, W3, W2 = _W1, _W3, _W2

    # Keep backward padding bound in lockstep with dispatch_meta_{bf16,blockscaled}
    # so blockscaled SF swizzle never sees a non-128-aligned M_pad.
    max_pad = ((int(M_recv) + E * (align - 1)) // align) * align
    max_pad_limit = int(rdep.capacity) + E * (align - 1)
    if max_pad < int(M_recv) or max_pad > max_pad_limit or (max_pad % align) != 0:
      raise RuntimeError(
        f"Invalid max_pad={max_pad} for backward gather path "
        f"(M_recv={int(M_recv)}, align={align}, limit={max_pad_limit})."
      )
    offs_pad[-1] = int(max_pad)
    _require_contiguous(offs_pad, "offs_pad", dtype=torch.int32)
    if not offs_pad.is_cuda:
      raise RuntimeError("offs_pad must be CUDA int32 for no-host BF16 grouped wgrad path.")
    # Prepare host-visible offs once per backward and reuse across grouped wgrad launches.
    # The copy is queued early so it can overlap with later backward kernels.
    # Hot wgrad launches use host-offs entrypoints to skip pointer introspection.
    offs_wgrad_host_ptr = _C.bf16_prepare_offs_pad_host(offs_pad.data_ptr(), int(E), stream)

    # Gather activations for backward recompute.
    if use_dist_blockscaled_meta:
      pack_factor = 2 if rdep.profile == "fp8" else 4
      Hp = H // pack_factor
      sf_k = H // 32
      sf_k_pad = ((sf_k + 3) // 4) * 4
      Xe_q = torch.empty(int(max_pad), int(Hp), device=device, dtype=torch.uint16)
      Xe_sf = torch.empty(int(max_pad), int(sf_k_pad), device=device, dtype=torch.uint8)
      _C.gather_xe_blockscaled(Xe_q.data_ptr(), Xe_sf.data_ptr(), int(M_recv), int(max_pad), stream)
      Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
      if rdep.profile == "fp8":
        _C.dequant_fp8_to_bf16_mma_sf(
          Xe_q.data_ptr(), int(Hp),
          Xe_sf.data_ptr(), int(sf_k_pad),
          Xe_pad.data_ptr(), int(H),
          int(max_pad), int(H),
          stream,
        )
      elif rdep.profile == "nvfp4":
        _C.dequant_nvfp4_to_bf16_mma_sf(
          Xe_q.data_ptr(), int(Hp),
          Xe_sf.data_ptr(), int(sf_k_pad),
          Xe_pad.data_ptr(), int(H),
          int(max_pad), int(H),
          stream,
        )
      else:
        raise RuntimeError(f"Unsupported hybrid blockscaled profile for backward dequant: {rdep.profile}")
      del Xe_q, Xe_sf
    else:
      # BF16 gather path (single-GPU, IPC, and BF16-profile distributed).
      Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
      _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    # Get row_id and gate_sorted for backward computation
    row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
    gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

    dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    if is_dist:
      if mode not in ("ipc", "hybrid"):
        raise RuntimeError(f"Unsupported distributed RDEP mode for MoE backward: mode={mode}")
    dGates_tk_bf16 = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)

    with _nvtx("moe_bs/bwd_combine"):
      if is_dist:
        # Distributed IPC+hybrid path: gather dYe across ranks with gate scaling (no dGate yet).
        _C.gather_dy_nogate_dist_bf16(
          dOut.data_ptr(),
          eid.data_ptr(),
          row_id.data_ptr(),
          gate_sorted.data_ptr(),
          dYe_sorted.data_ptr(),
          int(M_recv), int(T), int(H), int(K),
          stream,
        )
      else:
        # Single-GPU: gather dY with gate scaling (no dGate yet)
        _C.gather_dy_nogate_bf16(
          dOut.data_ptr(),
          row_id.data_ptr(),
          gate_sorted.data_ptr(),
          dYe_sorted.data_ptr(),
          int(M_recv), int(T), int(H), int(K),
          stream,
        )

      dYe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
      _C.scatter_sorted_to_pad_bf16(
        dYe_sorted.data_ptr(),
        dYe_pad.data_ptr(),
        int(M_recv), int(H),
        stream,
      )
      _C.zero_padding_rows_bf16(
        dYe_pad.data_ptr(),
        offs_pad.data_ptr(),
        int(M_recv),
        int(max_pad),
        int(H),
        stream,
      )
    del dYe_sorted  # Consumed; free [M_recv, H] BF16

    with _nvtx("moe_bs/bwd_expert_grad"):
      # H1/H3 from W1/W3 (activation recompute for SwiGLU backward).
      # In NVFP4 fused mode, dequant W1/W3 once and reuse them later for dX.
      # This removes two extra dequant kernel launches per layer.
      if _nvfp4_stagger:
        gs = moe_ref._nvfp4_group_size
        W1 = dequant_nvfp4_to_bf16_transient(
          moe_ref._W1_packed, moe_ref._W1_scale, moe_ref._W1_gs, gs, transpose=True)
        H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)

        W3 = dequant_nvfp4_to_bf16_transient(
          moe_ref._W3_packed, moe_ref._W3_scale, moe_ref._W3_gs, gs, transpose=True)
        H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
      else:
        H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
        H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)

      # dA needs W2^T. In NVFP4 fused mode, dequantize directly to [E, H, Dff]
      # to avoid [E, Dff, H] materialization + transpose.
      if _nvfp4_stagger:
        W2_t = dequant_nvfp4_to_bf16_transient(
          moe_ref._W2_packed, moe_ref._W2_scale, moe_ref._W2_gs, gs, transpose=False)
        Dff = int(W2_t.size(2))
        dA = torch._grouped_mm(dYe_pad, W2_t, offs=offs_pad)
        del W2_t
      else:
        Dff = int(W2.size(1))
        dA = torch._grouped_mm(dYe_pad, W2.transpose(1, 2), offs=offs_pad)

      A = torch.empty_like(H1)
      dH1 = torch.empty_like(H1)
      dH3 = torch.empty_like(H3)
      _C.swiglu_bwd_bf16(
        H1.data_ptr(), int(Dff),
        H3.data_ptr(), int(Dff),
        dA.data_ptr(), int(Dff),
        A.data_ptr(), int(Dff),
        dH1.data_ptr(), int(Dff),
        dH3.data_ptr(), int(Dff),
        int(max_pad), int(Dff),
        stream,
      )

      # SonicMoE dGate identity: dGate = ⟨A, dA⟩ instead of ⟨dOut, Ye⟩
      # This avoids recomputing Ye_pad in both single-GPU and distributed modes.
      _C.dgate_from_adA_bf16(
        A.data_ptr(),
        dA.data_ptr(),
        dGate_sorted.data_ptr(),
        int(M_recv), int(Dff),
        stream,
      )
      if is_dist:
        # Distributed IPC+hybrid: send dGate back to source ranks.
        _C.send_dgate_dist_bf16_out_bf16(
          row_id.data_ptr(),
          dGate_sorted.data_ptr(),
          dGates_tk_bf16.data_ptr(),
          int(M_recv), int(T), int(K),
          stream,
        )
      else:
        # Single-GPU: scatter dGate directly.
        _C.scatter_gate_bf16_out_bf16(
          dGate_sorted.data_ptr(),
          row_id.data_ptr(),
          dGates_tk_bf16.data_ptr(),
          int(M_recv), int(T), int(K),
          stream,
        )

    if fused_eco is not None:
      with _nvtx("moe_bs/bwd_wgrad_eco"):
        # Fused backward for NVFP4 primary mode.
        # Reuse dequantized W1/W3 from the earlier recompute to avoid
        # redundant dequant kernel launches.
        # At this point live tensors: H1, H3, dA, A, dH1, dH3,
        # Xe_pad, dYe_pad. We free H1/H3/dA now (consumed by SwiGLU+dGate).
        #
        # Order: dW2 → free → dX_part1(W1) → free W1 →
        #        dW1 → free → dX_part2(W3) → free W3 → dW3 → free
        del H1, H3, dA  # Consumed by swiglu_bwd + dgate; free before wgrad

        # Phase A: dW2 = wgrad_w2(A, dYe) → fused_update → free dW2, A, dYe
        dW2 = torch.empty(int(E), int(Dff), int(H), device=device, dtype=torch.bfloat16)
        _require_contiguous(A, "A", dtype=torch.bfloat16)
        _require_contiguous(dYe_pad, "dYe_pad", dtype=torch.bfloat16)
        _require_contiguous(dW2, "dW2", dtype=torch.bfloat16)
        _C.bf16_wgrad_w2_cublaslt_host_offs(
          A.data_ptr(),
          dYe_pad.data_ptr(),
          dW2.data_ptr(),
          offs_wgrad_host_ptr,
          int(E), int(H), int(Dff),
          stream,
        )
        del A, dYe_pad  # Free ~(max_pad*Dff + max_pad*H)*2 bytes
        fused_eco.fused_update(moe_ref, 'W2', dW2)
        del dW2  # Free ~448 MiB

        # Phase B: dX from W1.T using already-dequantized W1.
        dX_pad = torch._grouped_mm(dH1, W1.transpose(1, 2), offs=offs_pad)
        del W1

        # Phase C: paired dW1/dW3 = wgrad_w13(Xe, dH1/dH3) → fused_update W1 → keep dW3 for later.
        dW1 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        dW3 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        _require_contiguous(Xe_pad, "Xe_pad", dtype=torch.bfloat16)
        _require_contiguous(dH1, "dH1", dtype=torch.bfloat16)
        _require_contiguous(dH3, "dH3", dtype=torch.bfloat16)
        _require_contiguous(dW1, "dW1", dtype=torch.bfloat16)
        _require_contiguous(dW3, "dW3", dtype=torch.bfloat16)
        _C.bf16_wgrad_w13_pair_cublaslt_host_offs(
          Xe_pad.data_ptr(),
          dH1.data_ptr(),
          dH3.data_ptr(),
          dW1.data_ptr(),
          dW3.data_ptr(),
          offs_wgrad_host_ptr,
          int(E), int(H), int(Dff),
          stream,
        )
        del dH1
        fused_eco.fused_update(moe_ref, 'W1', dW1)
        del dW1

        # Phase D: free Xe_pad (~1.2 GiB) before dX+=W3.T so grouped_mm has room.
        del Xe_pad  # Free ~1.2 GiB before grouped_mm allocation

        # Phase E: dX += W3.T contribution before fused_update modifies buffers.
        # Reuse already-dequantized W3.
        # Xe_pad freed above; now we have room for grouped_mm temp (~1.23 GiB).
        dX_pad.add_(torch._grouped_mm(dH3, W3.transpose(1, 2), offs=offs_pad))
        del W3, dH3

        # Phase F: fused_update for W3 — AFTER dX has been computed from original W3.
        fused_eco.fused_update(moe_ref, 'W3', dW3)
        del dW3
        # Only invalidate cache when this micro-step performed a real weight update.
        if fused_eco.is_final_microstep:
          fused_eco.refresh_layer_cache(moe_ref)

    else:
      with _nvtx("moe_bs/bwd_wgrad"):
        # Standard backward (no fused ECO): compute wgrad, dX, return gradients
        dW2 = torch.empty(int(E), int(Dff), int(H), device=device, dtype=torch.bfloat16)
        _require_contiguous(A, "A", dtype=torch.bfloat16)
        _require_contiguous(dYe_pad, "dYe_pad", dtype=torch.bfloat16)
        _require_contiguous(dW2, "dW2", dtype=torch.bfloat16)
        _C.bf16_wgrad_w2_cublaslt_host_offs(
          A.data_ptr(),
          dYe_pad.data_ptr(),
          dW2.data_ptr(),
          offs_wgrad_host_ptr,
          int(E), int(H), int(Dff),
          stream,
        )

        dW1 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        dW3 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        _require_contiguous(Xe_pad, "Xe_pad", dtype=torch.bfloat16)
        _require_contiguous(dH1, "dH1", dtype=torch.bfloat16)
        _require_contiguous(dH3, "dH3", dtype=torch.bfloat16)
        _require_contiguous(dW1, "dW1", dtype=torch.bfloat16)
        _require_contiguous(dW3, "dW3", dtype=torch.bfloat16)
        _C.bf16_wgrad_w13_pair_cublaslt_host_offs(
          Xe_pad.data_ptr(),
          dH1.data_ptr(),
          dH3.data_ptr(),
          dW1.data_ptr(),
          dW3.data_ptr(),
          offs_wgrad_host_ptr,
          int(E), int(H), int(Dff),
          stream,
        )

        dX_pad = torch._grouped_mm(dH1, W1.transpose(1, 2), offs=offs_pad)
        dX_pad.add_(torch._grouped_mm(dH3, W3.transpose(1, 2), offs=offs_pad))
        del W1, W3, A, dH1, dH3, Xe_pad, dYe_pad

    with _nvtx("moe_bs/bwd_input_grad"):
      dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
      _require_contiguous(dX_pad, "dX_pad", dtype=torch.bfloat16)
      if is_dist:
        if mode in ("ipc", "hybrid"):
          _C.scatter_dx_dist_from_pad_bf16(
            dX_pad.data_ptr(),
            row_id.data_ptr(),
            dX.data_ptr(),
            int(M_recv), int(T), int(H), int(K),
            stream,
          )
        else:
          raise RuntimeError(f"Unsupported distributed RDEP mode for dX scatter: mode={mode}")
      else:
        _C.scatter_dx_bf16_internal(
          dX_pad.data_ptr(),
          row_id.data_ptr(),
          dX.data_ptr(),
          int(M_recv), int(T), int(H), int(K),
          stream,
        )

    dGates = dGates_tk_bf16

    if fused_eco is not None:
      return None, dX, None, dGates, None, None, None, None, None, None
    return None, dX, None, dGates, dW1, dW3, dW2, None, None, None
