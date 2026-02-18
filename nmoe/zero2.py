"""ZeRO-2 for replicated (dense) parameters.

Elegant minimal implementation:
- step_dense_adamw: RS(AVG) grads → AdamW shard update → AG params

State lives in caller-provided dict to keep this module stateless.
Works seamlessly for single GPU (no-op) and multi-GPU (ZeRO-2).
"""
import logging
import os
from contextlib import nullcontext
import math
import torch
import torch.distributed as dist

from nmoe.nccl_watchdog import get_watchdog

_log = logging.getLogger(__name__)

# Check for multi-tensor copy op (available in PyTorch 2.0+).
# This allows batching ~100 copy kernels into a single launch.
_HAS_FOREACH_COPY = hasattr(torch, '_foreach_copy_')

# ---------------------------------------------------------------------------
# Fused CUDA kernel for dense AdamW (direct import, no fallback)
# ---------------------------------------------------------------------------
from nmoe.csrc import rdep as _C

_fused_adamw_fns = {
    torch.float32: _C.fused_adamw_step_fp32,
    torch.bfloat16: _C.fused_adamw_step_bf16,
}


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


def _rs_chunk_elems(*, dtype: torch.dtype, world: int, shard_size: int) -> tuple[int, int]:
  """Return (shard_chunk, total_chunk_elems) for chunked reduce-scatter buffers.

  total_chunk_elems == world * shard_chunk.
  """
  if world <= 0:
    raise RuntimeError(f"ZeRO-2: invalid world_size={world}")
  if shard_size <= 0:
    raise RuntimeError(f"ZeRO-2: invalid shard_size={shard_size}")

  # Default: 2 GiB per dtype/group. Net win vs allocating padded_total for flat_grad.
  chunk_mb = int(os.getenv("NMOE_ZERO2_RS_CHUNK_MB", "2048"))
  if chunk_mb <= 0:
    raise RuntimeError(f"NMOE_ZERO2_RS_CHUNK_MB must be > 0 (got {chunk_mb})")
  bytes_per_elem = torch.empty((), dtype=dtype).element_size()
  target_elems_total = (chunk_mb * 1024 * 1024) // max(1, bytes_per_elem)
  # Must be divisible by world, and at least 1 elem per rank.
  target_shard_chunk = max(1, int(target_elems_total) // int(world))
  # Prefer an alignment that is friendly to both memcpy and NCCL (elements, not bytes).
  align = 256
  if shard_size <= align:
    shard_chunk = shard_size
  else:
    shard_chunk = max(align, (target_shard_chunk // align) * align)
    shard_chunk = min(shard_chunk, shard_size)
  total_chunk_elems = int(shard_chunk) * int(world)
  return int(shard_chunk), int(total_chunk_elems)


def _get_or_init_flat_group(
  group: dict,
  *,
  params: list[torch.nn.Parameter],
  rank: int,
  world: int,
  dtype: torch.dtype,
) -> dict:
  """Return (and lazily initialize) a persistent flat buffer plan for a param group/dtype.

  Stored on the *group dict* (not in optimizer state) to avoid checkpointing large buffers.
  """
  cache = group.setdefault('_zero2_flat', {})
  key = (dtype, int(world))
  if key in cache:
    flat = cache[key]
    # Re-init if param list changed (should not happen in training).
    if flat.get('n_params') == len(params):
      return flat

  if not params:
    raise RuntimeError("ZeRO-2: empty param list")

  dev = params[0].device
  for p in params:
    if p.device != dev:
      raise RuntimeError("ZeRO-2: params must be on a single device per group")
    if p.dtype != dtype:
      raise RuntimeError("ZeRO-2: dtype grouping invariant violated")

  offsets: list[tuple[int, int]] = []
  total = 0
  for p in params:
    n = int(p.numel())
    offsets.append((total, total + n))
    total += n

  shard_size = _ceil_div(total, world)
  padded_total = shard_size * world

  flat_param = torch.empty(padded_total, dtype=dtype, device=dev)
  # Initialize from current param values.
  for p, (a, b) in zip(params, offsets):
    flat_param[a:b].copy_(p.detach().view(-1))
  if total < padded_total:
    flat_param[total:padded_total].zero_()

  # Re-point params to views into flat_param (no per-step unpack copies).
  for p, (a, b) in zip(params, offsets):
    p.data = flat_param[a:b].view_as(p)  # type: ignore[assignment]

  # Scratch (persistent):
  # - Chunked reduce-scatter buffers (avoid allocating flat_grad of size padded_total).
  # - Local param shard (the only shard we update).
  shard_chunk, total_chunk_elems = _rs_chunk_elems(dtype=dtype, world=world, shard_size=shard_size)
  rs_in = torch.empty(total_chunk_elems, dtype=dtype, device=dev)
  rs_out = torch.empty(shard_chunk, dtype=dtype, device=dev)
  # Double-buffer pair for pipelined reduce-scatter (overlap NCCL with grad packing).
  rs_in_b = torch.empty(total_chunk_elems, dtype=dtype, device=dev)
  rs_out_b = torch.empty(shard_chunk, dtype=dtype, device=dev)
  if world == 1:
    param_shard = flat_param[:shard_size]  # view (no comm)
  else:
    param_shard = torch.empty(shard_size, dtype=dtype, device=dev)
    a = rank * shard_size
    b = a + shard_size
    param_shard.copy_(flat_param[a:b])

  flat = {
    'n_params': len(params),
    'offsets': offsets,
    'total': total,
    'padded_total': padded_total,
    'shard_size': shard_size,
    'flat_param': flat_param,
    'param_shard': param_shard,
    'rs_in': rs_in,
    'rs_out': rs_out,
    'rs_in_b': rs_in_b,
    'rs_out_b': rs_out_b,
    'shard_chunk': shard_chunk,
  }
  cache[key] = flat
  return flat


@torch.no_grad()
def eager_init(
  param_groups: list[dict],
  *,
  pg: dist.ProcessGroup | None = None,
) -> None:
  """Pre-allocate ZeRO-2 flat buffers and re-point params before the first forward pass.

  Must be called BEFORE any forward/backward to ensure the flat_param buffer
  is allocated while GPU memory is still available (only model weights resident).
  If called during the optimizer step (lazy), the buffer competes with activations
  and gradients, causing OOM on high-utilization setups (e.g., 345B NVFP4 on 8×B200).

  Re-entrancy safe: if buffers are already initialized, this is a no-op.
  """
  world = dist.get_world_size(pg) if (dist.is_available() and dist.is_initialized()) else 1
  rank = dist.get_rank(pg) if (dist.is_available() and dist.is_initialized()) else 0

  for group in param_groups:
    params = list(group['params'])
    if not params:
      continue
    by_dtype: dict[torch.dtype, list[torch.nn.Parameter]] = {}
    for p in params:
      if p.dtype not in by_dtype:
        by_dtype[p.dtype] = []
      by_dtype[p.dtype].append(p)
    for dtype, group_params in by_dtype.items():
      _get_or_init_flat_group(group, params=group_params, rank=rank, world=world, dtype=dtype)


@torch.no_grad()
def step_dense_adamw(
  param_groups: list[dict],
  *,
  state: dict,
  pg: dist.ProcessGroup | None = None,
  betas: tuple[float, float] = (0.9, 0.95),
  eps: float = 1e-8,
) -> None:
  """ZeRO-2 shard step for dense param groups using AdamW.

  For each group:
  1. Flatten grads and params per dtype
  2. Reduce-scatter(AVG) grads → local shard
  3. AdamW update on shard with shard-local state (exp_avg/exp_avg_sq)
  4. All-gather updated param shards

  Single GPU: just runs AdamW on full params (no RS/AG).
  Multi-GPU: full ZeRO-2 with sharding.
  """
  timers_on = os.getenv('NMOE_TIMERS', '1') not in ('0', 'false', 'False')
  if timers_on:
    try:
      from nmoe.metrics import cuda_time as _cuda_time  # local import to avoid hard dependency
      time_ctx = _cuda_time
    except Exception:
      _log.debug("cuda_time import failed, CUDA timers disabled for ZeRO-2", exc_info=True)
      time_ctx = lambda _tag: nullcontext()
  else:
    time_ctx = lambda _tag: nullcontext()

  world = dist.get_world_size(pg) if (dist.is_available() and dist.is_initialized()) else 1

  rank = dist.get_rank(pg) if (dist.is_available() and dist.is_initialized()) else 0

  nccl_wd = get_watchdog()

  for group_idx, group in enumerate(param_groups):
    params = list(group['params'])
    if not params:
      continue

    # Group by dtype
    by_dtype = {}
    for p in params:
      if p.dtype not in by_dtype:
        by_dtype[p.dtype] = []
      by_dtype[p.dtype].append(p)

    for dtype, group_params in by_dtype.items():
      flat = _get_or_init_flat_group(group, params=group_params, rank=rank, world=world, dtype=dtype)

      # Wait for previous step's async all-gather to complete before
      # reading gradients (which depend on the current parameters).
      prev_ag = flat.get('_pending_ag')
      if prev_ag is not None:
        prev_ag.wait()
        nccl_wd.clear()
        flat['_pending_ag'] = None

      flat_param = flat['flat_param']
      param_shard = flat['param_shard']
      offsets: list[tuple[int, int]] = flat['offsets']
      total = int(flat['total'])
      padded_total = int(flat['padded_total'])
      shard_size = int(flat['shard_size'])
      shard_chunk = int(flat['shard_chunk'])

      # Double-buffer pairs for pipelined reduce-scatter.
      rs_in_a = flat['rs_in']
      rs_out_a = flat['rs_out']
      rs_in_b = flat['rs_in_b']
      rs_out_b = flat['rs_out_b']
      rs_in2d_a = rs_in_a.view(world, shard_chunk)
      rs_in2d_b = rs_in_b.view(world, shard_chunk)

      # Get or initialize shard state
      state_key = f"shard_{rank}_{group_idx}_{dtype}"
      if state_key not in state:
        state[state_key] = {
          'step': 0,
          'exp_avg': torch.zeros(shard_size, dtype=dtype, device=param_shard.device),
          'exp_avg_sq': torch.zeros(shard_size, dtype=dtype, device=param_shard.device),
        }

      s = state[state_key]
      s['step'] += 1

      # AdamW update on shard (streamed per reduce-scatter chunk).
      beta1, beta2 = betas
      exp_avg = s['exp_avg']
      exp_avg_sq = s['exp_avg_sq']

      # Bias correction
      bias_correction1 = 1 - beta1 ** s['step']
      bias_correction2 = 1 - beta2 ** s['step']
      lr = float(group['lr'])
      wd = float(group.get('weight_decay', 0.0))
      step_size = lr / bias_correction1
      inv_bc2_sqrt = 1.0 / math.sqrt(bias_correction2)

      # Local helper: apply AdamW on a single completed RS chunk.
      # Defined once to avoid duplicating the update logic for the
      # in-loop drain and the trailing drain after the loop.
      # Uses the fused CUDA kernel (single launch replacing 9 PyTorch kernels).
      _fused_fn = _fused_adamw_fns[dtype]

      def _adamw_chunk(rs_out_buf: torch.Tensor, clen: int, soff: int) -> None:
        _fused_fn(
            param_shard.data_ptr() + soff * param_shard.element_size(),
            rs_out_buf.data_ptr(),
            exp_avg.data_ptr() + soff * exp_avg.element_size(),
            exp_avg_sq.data_ptr() + soff * exp_avg_sq.element_size(),
            clen,
            beta1, beta2,
            lr, wd, eps,
            step_size, inv_bc2_sqrt,
            torch.cuda.current_stream(),
        )

      # Precompute flat views of grads (avoid repeated reshape work).
      grad_views: list[torch.Tensor | None] = []
      for p in group_params:
        if p.grad is None:
          grad_views.append(None)
        else:
          grad_views.append(p.grad.detach().reshape(-1))

      # Double-buffered reduce-scatter with pipelined packing.
      # Treat conceptual flat_grad as [world, shard_size] laid out row-major.
      # While NCCL runs the RS on the current buffer, we pack grads into the
      # alternate buffer for the next chunk.
      cursors = [0 for _ in range(world)]
      pending_work = None          # NCCL async handle for in-flight RS
      pending_rs_out = None        # Which rs_out holds the result
      pending_chunk_len = 0
      pending_shard_off = 0
      chunks = list(range(0, shard_size, shard_chunk))

      with time_ctx('time_ms/zero2_reduce_scatter'):
        for ci, shard_off in enumerate(chunks):
          chunk_len = min(shard_chunk, shard_size - shard_off)

          # Pick current buffers (alternating A/B).
          if ci % 2 == 0:
            cur_rs_in2d = rs_in2d_a
            cur_rs_in = rs_in_a
            cur_rs_out = rs_out_a
          else:
            cur_rs_in2d = rs_in2d_b
            cur_rs_in = rs_in_b
            cur_rs_out = rs_out_b

          # Pack gradients into the current buffer.
          # This can execute on the CPU/GPU compute stream while the NCCL
          # stream is still running the previous chunk's reduce-scatter.
          #
          # Optimization: collect all (dst, src) slice pairs and execute with
          # a single torch._foreach_copy_ call instead of ~100 individual
          # .copy_() calls. This reduces kernel launch overhead significantly.
          cur_rs_in2d.zero_()
          dst_slices: list[torch.Tensor] = []
          src_slices: list[torch.Tensor] = []
          for src_rank in range(world):
            g0 = src_rank * shard_size + shard_off
            if g0 >= total:
              continue
            g1 = min(g0 + chunk_len, total)
            row = cur_rs_in2d[src_rank]

            i = cursors[src_rank]
            while i < len(offsets) and offsets[i][1] <= g0:
              i += 1
            cursors[src_rank] = i

            while i < len(offsets) and offsets[i][0] < g1:
              a, b = offsets[i]
              o0 = g0 if g0 > a else a
              o1 = g1 if g1 < b else b
              if o1 > o0:
                gv = grad_views[i]
                if gv is not None:
                  dst_slices.append(row[(o0 - g0):(o1 - g0)])
                  src_slices.append(gv[(o0 - a):(o1 - a)])
              i += 1

          # Execute all gradient copies in a single batched kernel.
          if dst_slices:
            if _HAS_FOREACH_COPY:
              torch._foreach_copy_(dst_slices, src_slices)
            else:
              # Fallback for older PyTorch versions (sequential copies).
              for dst, src in zip(dst_slices, src_slices):
                dst.copy_(src)

          # Wait for the PREVIOUS chunk's RS to finish, then apply AdamW.
          if pending_work is not None:
            pending_work.wait()
            nccl_wd.clear()
          if pending_rs_out is not None:
            _adamw_chunk(pending_rs_out, pending_chunk_len, pending_shard_off)

          # Launch async reduce-scatter for the current chunk.
          if world == 1:
            cur_rs_out[:chunk_len].copy_(cur_rs_in2d[0, :chunk_len])
            pending_work = None
          else:
            pending_work = dist.reduce_scatter_tensor(
              cur_rs_out, cur_rs_in, op=dist.ReduceOp.AVG, group=pg, async_op=True
            )
            nccl_wd.watch(pending_work, f"reduce_scatter chunk {ci}")
          pending_rs_out = cur_rs_out
          pending_chunk_len = chunk_len
          pending_shard_off = shard_off

      # Drain the last pending chunk (outside the timer context).
      if pending_work is not None:
        pending_work.wait()
        nccl_wd.clear()
      if pending_rs_out is not None:
        _adamw_chunk(pending_rs_out, pending_chunk_len, pending_shard_off)

      # Async all-gather updated params to all ranks (no-op when world==1).
      # The work handle is stored in the flat dict so the next step waits
      # before reading gradients, allowing the current AG to overlap with
      # the next forward pass.
      if world > 1:
        with time_ctx('time_ms/zero2_all_gather'):
          ag_work = dist.all_gather_into_tensor(flat_param, param_shard, group=pg, async_op=True)
          nccl_wd.watch(ag_work, f"all_gather params dtype={dtype}")
        flat['_pending_ag'] = ag_work
      elif total < padded_total:
        # Keep padding clean for determinism.
        flat_param[total:padded_total].zero_()
