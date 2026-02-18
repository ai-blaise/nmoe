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
import torch.distributed as dist

from nmoe.csrc import rdep as _C
from nmoe.cuda_errors import cuda_error_context

if TYPE_CHECKING:
  from nmoe.rdep import Rdep


def _nvtx(tag: str):
    if os.getenv('NMOE_NVTX', '0') not in ('1', 'true', 'True'):
        return nullcontext()
    if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'):
        return torch.cuda.nvtx.range(tag)
    return nullcontext()


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

    # Flatten to [E*M, K/2] for the kernel
    packed_flat = packed.reshape(total_M, K // 2).contiguous()
    n_groups = K // group_size
    scale_flat = scale.view(torch.uint8).reshape(total_M, n_groups).contiguous()
    gs_flat = global_scale.reshape(E).contiguous().float()

    # Allocate output — transposed or normal
    if transpose:
        out_bf16 = torch.empty(K, total_M, dtype=torch.bfloat16, device=device)
    else:
        out_bf16 = torch.empty(total_M, K, dtype=torch.bfloat16, device=device)

    # Run GPU kernel — zero FP32 global memory, all in registers
    _C.ct_nvfp4_to_bf16(
        packed_flat.data_ptr(),
        scale_flat.data_ptr(),
        gs_flat.data_ptr(),
        out_bf16.data_ptr(),
        total_M, K, group_size,
        1,  # gs_stride=1 (per-expert global scale)
        M,  # expert_rows=M (rows per expert for indexing ct_gs)
        1 if transpose else 0,
        stream,
    )

    # Reshape to [E, ...] form
    if transpose:
        # [K, E*M] → [E, K, M]  (matches model's [E, in_dim, out_dim])
        out_bf16 = out_bf16.reshape(K, E, M).permute(1, 0, 2).contiguous()
    else:
        out_bf16 = out_bf16.reshape(E, M, K)

    return out_bf16


class _MoEBlockscaledFused(torch.autograd.Function):
  @staticmethod
  def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
              W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor, W_cache,
              fused_eco=None, moe_ref=None) -> torch.Tensor:
    device = x.device
    stream = torch.cuda.current_stream(device)

    x = x.contiguous().bfloat16()
    eid = eid.contiguous().int()
    gates = gates.contiguous().bfloat16()
    gates_fp32 = gates.detach().float()

    T, H = x.shape
    K = int(eid.shape[1])
    E = int(rdep.n_local)
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if is_dist:
      need = int(T) * int(K) * int(rdep.world)
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={rdep.world}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    # Option A: Use BF16 dispatch + local quantization
    # This ensures Xe_pad (BF16) is available for backward STE
    offs_pad = torch.empty(E, device=device, dtype=torch.int32)
    # P3.14: Use pre-allocated pinned memory from Rdep
    M_host = rdep._pinned_M_host
    M_host.zero_()
    align = 128  # Required for blockscaled SF swizzle

    with _nvtx("moe_bs/fwd_dispatch_meta"), cuda_error_context("dispatch_meta_blockscaled"):
      M_recv = _C.dispatch_meta_blockscaled(
        x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
        int(T), int(K),
        offs_pad.data_ptr(), M_host.data_ptr(),
        stream,
      )

    # Check for dropped tokens periodically (avoid per-call GPU-CPU sync)
    if not hasattr(_MoEBlockscaledFused, '_dispatch_count'):
      _MoEBlockscaledFused._dispatch_count = 0
    _MoEBlockscaledFused._dispatch_count += 1
    if _MoEBlockscaledFused._dispatch_count % 100 == 1:  # Check on 1st call and every 100th
      dropped = _C.get_dropped_blockscaled(stream)
      # Store dropped count on the MoE module for training loop monitoring
      if moe_ref is not None:
        moe_ref._last_dropped_count = dropped
      if dropped > 0:
        import logging
        logging.getLogger(__name__).warning(
          f"[RDEP] {dropped:,} tokens dropped due to capacity overflow "
          f"(capacity={rdep.capacity:,}, T={T}, K={K}). "
          f"Increase rdep_capacity or reduce batch_size."
        )
    else:
      # On non-check iterations, reset to 0 so stale counts don't persist
      if moe_ref is not None and not hasattr(moe_ref, '_last_dropped_count'):
        moe_ref._last_dropped_count = 0

    out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
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
        ctx.save_for_backward(x, eid, gates, _e0, _e0, _e0)
      else:
        ctx.save_for_backward(x, eid, gates, W1, W3, W2)
      return out_f32.to(dtype=torch.bfloat16)

    # P3.13: Use non-blocking query loop instead of blocking synchronize()
    # This allows overlapping with other work and reduces latency spikes
    sync_event = torch.cuda.Event()
    sync_event.record(stream)
    while not sync_event.query():
      pass  # Busy-wait is faster than sleep for short waits
    M_pad = int(M_host[0].item())  # Now safe to read

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
    from nmoe.blockscaled.grouped import expert_blockscaled
    with _nvtx("moe_bs/fwd_expert_compute"):
      Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad, capacity_rows=int(rdep.capacity))

    with _nvtx("moe_bs/fwd_combine"):
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
      ctx.save_for_backward(x, eid, gates, _e0, _e0, _e0)
    else:
      ctx.save_for_backward(x, eid, gates, W1, W3, W2)
    return out_f32.to(dtype=torch.bfloat16)

  @staticmethod
  def backward(ctx, dOut: torch.Tensor):
    x, eid, gates, _W1, _W3, _W2 = ctx.saved_tensors
    rdep: Rdep = ctx.rdep
    fused_eco = ctx.fused_eco
    moe_ref = ctx.moe_ref

    device = dOut.device
    stream = torch.cuda.current_stream(device)

    dOut = dOut.contiguous().bfloat16()
    x = x.contiguous().bfloat16()
    eid = eid.contiguous().int()
    gates = gates.contiguous().bfloat16()
    gates_fp32 = gates.detach().float()

    T = int(ctx.T)
    H = int(ctx.H)
    K = int(ctx.K)
    E = int(rdep.n_local)
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
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
    offs_pad = torch.empty(E, device=device, dtype=torch.int32)
    # P3.14: Use pre-allocated pinned memory from Rdep
    M_host = rdep._pinned_M_host
    M_host.zero_()
    align = 128  # Required for blockscaled SF swizzle

    with _nvtx("moe_bs/bwd_dispatch_meta"):
      M_recv = _C.dispatch_meta_bf16(
        x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
        int(T), int(K), align,
        offs_pad.data_ptr(), M_host.data_ptr(),
        stream,
      )

    if M_recv <= 0:
      dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)

      # DeepEP collectiveness: still run distributed gather/scatter
      if is_dist:
        dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)
        dummy_row_id = torch.empty(1, device=device, dtype=torch.int64)
        dummy_gate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        dummy_ye_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        dummy_dye_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        dummy_dgate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        with cuda_error_context("gather_dy_dist_bf16 (blockscaled backward, M_recv=0)"):
          _C.gather_dy_dist_bf16(
            dOut.data_ptr(),
            eid.data_ptr(),
            dummy_ye_sorted.data_ptr(),
            dummy_row_id.data_ptr(),
            dummy_gate_sorted.data_ptr(),
            dummy_dye_sorted.data_ptr(),
            dummy_dgate_sorted.data_ptr(),
            dGates_tk_f32.data_ptr(),
            0, int(T), int(H), int(K),
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
        dGates = dGates_tk_f32.to(dtype=torch.bfloat16)
      else:
        dGates = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)

      if fused_eco is not None:
        # No tokens routed to this rank — skip fused_update entirely.
        # Applying zero-gradient update would incorrectly apply weight decay and
        # stale momentum, corrupting weights on ranks with no routed tokens.
        # In practice M_recv=0 is unreachable with top-8 routing over 128 experts
        # (each rank's 16 experts would need to receive zero out of ~32k assignments).
        return None, dX, None, dGates, None, None, None, None, None, None

      dW1 = torch.zeros_like(_W1)
      dW3 = torch.zeros_like(_W3)
      dW2 = torch.zeros_like(_W2)
      return None, dX, None, dGates, dW1, dW3, dW2, None, None, None

    # When fused_eco is active, saved W1/W3/W2 are empty placeholders.
    # Dequant from NVFP4 on-the-fly — NEVER hold more than one BF16 weight
    # at a time to avoid OOM (~1.3 GiB each, holding 2+ simultaneously
    # exhausts GPU memory with activations+gradients).
    _nvfp4_stagger = (fused_eco is not None and moe_ref is not None)
    if not _nvfp4_stagger:
      W1, W3, W2 = _W1, _W3, _W2

    # Compute max_pad and extend last expert's padded region
    max_pad = (int(M_recv) + E * (align - 1) + (align - 1)) // align * align
    offs_pad[-1] = int(max_pad)

    # Gather BF16 activations (correct from all source ranks via IPC buffer!)
    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    # Get row_id and gate_sorted for backward computation
    row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
    gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

    dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)

    with _nvtx("moe_bs/bwd_combine"):
      if is_dist:
        # Distributed path: gather dYe across ranks with gate scaling (no dGate yet)
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

      dYe_pad = torch.zeros(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
      _C.scatter_sorted_to_pad_bf16(
        dYe_sorted.data_ptr(),
        dYe_pad.data_ptr(),
        int(M_recv), int(H),
        stream,
      )
    del dYe_sorted  # Consumed; free [M_recv, H] BF16

    # P3.14: Use pre-allocated pinned memory from Rdep for offs
    offs_pinned = rdep._pinned_offs
    offs_pinned.copy_(offs_pad, non_blocking=True)
    copy_event = torch.cuda.Event()
    copy_event.record(stream)

    with _nvtx("moe_bs/bwd_expert_grad"):
      # H1/H3 from W1/W3 (activation recompute for SwiGLU backward).
      # In NVFP4 stagger mode: dequant one weight at a time, compute, free.
      # Each BF16 weight is ~1.3 GiB; we never hold two simultaneously.
      if _nvfp4_stagger:
        gs = getattr(moe_ref, '_nvfp4_group_size', 16)
        W1 = dequant_nvfp4_to_bf16_transient(
          moe_ref._W1_packed, moe_ref._W1_scale, moe_ref._W1_gs, gs, transpose=True)
        H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
        del W1  # Free ~1.3 GiB

        W3 = dequant_nvfp4_to_bf16_transient(
          moe_ref._W3_packed, moe_ref._W3_scale, moe_ref._W3_gs, gs, transpose=True)
        H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
        del W3  # Free ~1.3 GiB
      else:
        H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
        H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)

      # dA needs W2. Dequant W2 now, compute dA, free W2 immediately.
      if _nvfp4_stagger:
        W2 = dequant_nvfp4_to_bf16_transient(
          moe_ref._W2_packed, moe_ref._W2_scale, moe_ref._W2_gs, gs, transpose=True)
      Dff = int(W2.size(1))
      dA = torch._grouped_mm(dYe_pad, W2.transpose(1, 2), offs=offs_pad)
      if _nvfp4_stagger:
        del W2  # Free ~1.3 GiB

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
        # Distributed: send dGate back to source ranks via IPC
        _C.send_dgate_dist_bf16(
          row_id.data_ptr(),
          dGate_sorted.data_ptr(),
          dGates_tk_f32.data_ptr(),
          int(M_recv), int(T), int(K),
          stream,
        )
      else:
        # Single-GPU: scatter dGate directly
        _C.scatter_gate_bf16(
          dGate_sorted.data_ptr(),
          row_id.data_ptr(),
          dGates_tk_f32.data_ptr(),
          int(M_recv), int(T), int(K),
          stream,
        )

    # P3.13: Use non-blocking query loop instead of blocking synchronize()
    while not copy_event.query():
      pass  # Busy-wait for short D2H copies
    offs_host = offs_pinned

    if fused_eco is not None:
      with _nvtx("moe_bs/bwd_wgrad_eco"):
        # Memory-optimized backward for NVFP4 primary mode.
        # Key insight: never hold more than one BF16 weight at a time.
        # W1/W3 were already freed after H1/H3 recompute (stagger mode).
        # At this point live tensors: H1, H3, dA, A, dH1, dH3,
        # Xe_pad, dYe_pad. We free H1/H3/dA now (consumed by SwiGLU+dGate).
        #
        # Order: dW2 → free → re-dequant W1 → dX_part1 → free W1 →
        #        dW1 → free → re-dequant W3 → dX_part2 → free W3 → dW3 → free
        del H1, H3, dA  # Consumed by swiglu_bwd + dgate; free before wgrad

        # Phase A: dW2 = wgrad_w2(A, dYe) → fused_update → free dW2, A, dYe
        dW2 = torch.empty(int(E), int(Dff), int(H), device=device, dtype=torch.bfloat16)
        _C.bf16_wgrad_w2_cublaslt(
          A.data_ptr(),
          dYe_pad.data_ptr(),
          dW2.data_ptr(),
          offs_host.data_ptr(),
          int(E), int(H), int(Dff),
          stream,
        )
        del A, dYe_pad  # Free ~(max_pad*Dff + max_pad*H)*2 bytes
        fused_eco.fused_update(moe_ref, 'W2', dW2)
        del dW2  # Free ~448 MiB

        # Phase B: dX from W1.T — re-dequant W1 (was freed after H1 recompute).
        # Dequant is fast (~0.5ms), saves 1.3 GiB vs keeping W1 alive.
        W1 = dequant_nvfp4_to_bf16_transient(
          moe_ref._W1_packed, moe_ref._W1_scale, moe_ref._W1_gs, gs, transpose=True)
        dX_pad = torch._grouped_mm(dH1, W1.transpose(1, 2), offs=offs_pad)
        del W1  # Free ~1.3 GiB

        # Phase C: dW1 = wgrad_w13(Xe, dH1) → fused_update → free
        dW1 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        _C.bf16_wgrad_w13_cublaslt(
          Xe_pad.data_ptr(),
          dH1.data_ptr(),
          dW1.data_ptr(),
          offs_host.data_ptr(),
          int(E), int(H), int(Dff),
          stream,
        )
        del dH1
        fused_eco.fused_update(moe_ref, 'W1', dW1)
        del dW1

        # Phase D: dW3 = wgrad_w13(Xe, dH3) → fused_update → free Xe_pad (~1.2 GiB)
        # Done BEFORE dX+=W3.T so Xe_pad is freed, making room for grouped_mm temp.
        dW3 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        _C.bf16_wgrad_w13_cublaslt(
          Xe_pad.data_ptr(),
          dH3.data_ptr(),
          dW3.data_ptr(),
          offs_host.data_ptr(),
          int(E), int(H), int(Dff),
          stream,
        )
        del Xe_pad  # Free ~1.2 GiB before grouped_mm allocation

        # Phase E: dX += W3.T contribution — re-dequant W3 BEFORE fused_update
        # modifies the NVFP4 buffers in-place, so dX uses the ORIGINAL W3.
        # Xe_pad freed above; now we have room for grouped_mm temp (~1.23 GiB).
        W3 = dequant_nvfp4_to_bf16_transient(
          moe_ref._W3_packed, moe_ref._W3_scale, moe_ref._W3_gs, gs, transpose=True)
        dX_pad.add_(torch._grouped_mm(dH3, W3.transpose(1, 2), offs=offs_pad))
        del W3, dH3  # Free ~1.3 GiB + dH3

        # Phase F: fused_update for W3 — AFTER dX has been computed from original W3.
        fused_eco.fused_update(moe_ref, 'W3', dW3)
        del dW3
        fused_eco.refresh_layer_cache(moe_ref)

    else:
      with _nvtx("moe_bs/bwd_wgrad"):
        # Standard backward (no fused ECO): compute wgrad, dX, return gradients
        dW2 = torch.empty(int(E), int(Dff), int(H), device=device, dtype=torch.bfloat16)
        _C.bf16_wgrad_w2_cublaslt(
          A.data_ptr(),
          dYe_pad.data_ptr(),
          dW2.data_ptr(),
          offs_host.data_ptr(),
          int(E), int(H), int(Dff),
          stream,
        )

        dW1 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        _C.bf16_wgrad_w13_cublaslt(
          Xe_pad.data_ptr(),
          dH1.data_ptr(),
          dW1.data_ptr(),
          offs_host.data_ptr(),
          int(E), int(H), int(Dff),
          stream,
        )

        dW3 = torch.empty(int(E), int(H), int(Dff), device=device, dtype=torch.bfloat16)
        _C.bf16_wgrad_w13_cublaslt(
          Xe_pad.data_ptr(),
          dH3.data_ptr(),
          dW3.data_ptr(),
          offs_host.data_ptr(),
          int(E), int(H), int(Dff),
          stream,
        )

        dX_pad = torch._grouped_mm(dH1, W1.transpose(1, 2), offs=offs_pad)
        dX_pad.add_(torch._grouped_mm(dH3, W3.transpose(1, 2), offs=offs_pad))
        del W1, W3, A, dH1, dH3, Xe_pad, dYe_pad

    with _nvtx("moe_bs/bwd_input_grad"):
      dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
      if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dX_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
        _C.gather_from_pad_bf16(dX_pad.data_ptr(), dX_sorted.data_ptr(), int(M_recv), int(H), stream)
        _C.scatter_dx_dist_bf16(
          dX_sorted.data_ptr(),
          row_id.data_ptr(),
          dX.data_ptr(),
          int(M_recv), int(T), int(H), int(K),
          stream,
        )
      else:
        _C.scatter_dx_bf16_internal(
          dX_pad.data_ptr(),
          row_id.data_ptr(),
          dX.data_ptr(),
          int(M_recv), int(T), int(H), int(K),
          stream,
        )

    dGates = dGates_tk_f32.to(dtype=torch.bfloat16)

    if fused_eco is not None:
      return None, dX, None, dGates, None, None, None, None, None, None
    return None, dX, None, dGates, dW1, dW3, dW2, None, None, None
