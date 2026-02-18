"""Optimizer construction, LR scheduling, and training step for MoE.

Two parameter types:
- expert: MoE expert MLPs (sharded via RDEP, optimized with Muon)
- dense: everything else - attention, embeddings, norms, router (replicated, ZeRO-2, AdamW)

Precision support: bf16, fp8, nvfp4 (default: nvfp4)
"""
import logging
import math
import torch
import torch.distributed as dist

from contextlib import contextmanager

from nmoe.config import Config
from nmoe import zero2

_log = logging.getLogger(__name__)


@contextmanager
def _nvtx(tag: str):
  """Emit an NVTX range for Nsight profiling.  No-op cost when no profiler is attached."""
  torch.cuda.nvtx.range_push(tag)
  try:
    yield
  finally:
    torch.cuda.nvtx.range_pop()


class _NullExpertOptimizer(torch.optim.Optimizer):
  """No-op optimizer placeholder for fused backward-optimizer mode.

  When FusedBackwardECO is active, expert optimization happens inside the
  backward pass. This placeholder satisfies the interface expected by step()
  and update_lr() without doing any work.

  Uses a single-element Parameter to satisfy Optimizer's param_groups
  requirement with minimal overhead. The 'lr' key is read/written by
  update_lr() to communicate the scheduled expert LR to FusedBackwardECO.
  """
  emits_weight_cache = True  # Suppress post-step refresh (fused_eco refreshes in backward)

  _DUMMY = None  # Shared across instances to avoid repeated allocation

  def __init__(self):
    if _NullExpertOptimizer._DUMMY is None:
      _NullExpertOptimizer._DUMMY = torch.nn.Parameter(
        torch.zeros(1), requires_grad=False,
      )
    super().__init__([_NullExpertOptimizer._DUMMY], {"lr": 0.0})

  @torch.no_grad()
  def step(self, closure=None):  # type: ignore[override]
    return None

  def state_dict(self) -> dict:
    # Minimal state — nothing to save for a no-op optimizer
    return {"param_groups": self.param_groups}

  def load_state_dict(self, state_dict: dict) -> None:
    # Accept any state dict gracefully (migration from real optimizer)
    if "param_groups" in state_dict and state_dict["param_groups"]:
      # Only restore lr (the only value we care about)
      for saved_g, g in zip(state_dict["param_groups"], self.param_groups):
        g["lr"] = saved_g.get("lr", g["lr"])


def build_optimizer(model: torch.nn.Module, cfg: Config) -> tuple[torch.optim.Optimizer, list[dict]]:
  """Build optimizer for experts and return dense groups for AdamW/ZeRO-2.

  Returns:
    (expert_optimizer, dense_groups): Expert optimizer and dense param groups for ZeRO-2
  """
  # Collect parameters by type (single source of truth when available)
  expert_params: list[torch.nn.Parameter] = []
  router_params: list[torch.nn.Parameter] = []  # Router gate weights (separate LR, no decay)
  dense_params_decay: list[torch.nn.Parameter] = []
  dense_params_no_decay: list[torch.nn.Parameter] = []

  expert_ids: set[int] = set()
  if hasattr(model, "param_sets"):
    eps, _ = model.param_sets()  # type: ignore[attr-defined]
    expert_ids = {id(p) for p in eps}

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue

    if id(param) in expert_ids:
      expert_params.append(param)
      continue

    # Router parameters: separate group with lr_router and weight_decay=0
    # Matches old_nmoe behavior for aux-free load balancing stability
    if 'router' in name:
      router_params.append(param)
      continue

    # Dense params: split decay vs no-decay by name pattern
    no_decay = name.endswith('.bias') or 'norm' in name or name.startswith('embedding.') or name.startswith('lm_head.') or 'bungee' in name
    if no_decay:
      dense_params_no_decay.append(param)
    else:
      dense_params_decay.append(param)

  # Build dense param groups (for ZeRO-2 AdamW)
  dense_groups = []
  if dense_params_decay:
    dense_groups.append({
      'name': 'dense_decay',
      'params': dense_params_decay,
      'lr': cfg.lr_dense,
      'weight_decay': cfg.weight_decay,
    })
  if dense_params_no_decay:
    dense_groups.append({
      'name': 'dense_no_decay',
      'params': dense_params_no_decay,
      'lr': cfg.lr_dense,
      'weight_decay': 0.0,
    })
  # Router group: separate LR, no weight decay (critical for aux-free load balancing)
  if router_params:
    dense_groups.append({
      'name': 'router',
      'params': router_params,
      'lr': cfg.lr_router,
      'weight_decay': 0.0,
    })

  # Expert optimizer selection:
  # 0. Fused backward-optimizer (eco_fused_backward=True): optimization happens in backward pass
  # 1. ECO mode (eco_enabled=True): NVFP4 primary weights + FP8 opt states + error feedback
  # 2. Fallback: standard PyTorch AdamW (debug / non-production only)
  eco_fused = getattr(cfg, "eco_fused_backward", False)
  eco_enabled = getattr(cfg, "eco_enabled", False)

  if eco_fused:
    # Expert optimization is handled by FusedBackwardECO inside backward pass.
    # Use a null optimizer as placeholder — no expert params need tracking here.
    expert_optimizer = _NullExpertOptimizer()
  elif eco_enabled and expert_params:
    # eco_enabled=True without eco_fused_backward=True is a configuration error.
    # The Python reference optimizer is ~10x slower and must not be used silently.
    raise RuntimeError(
        "eco_enabled=True requires eco_fused_backward=True for production use. "
        "The Python-only ECOAdamW reference optimizer (eco_reference.py) exists "
        "solely for unit testing and must not be used in training. "
        "Set eco_fused_backward=True in your config."
    )
  elif expert_params:
    # No ECO configured — this is a configuration error for MoE training with
    # NVFP4 weights. Standard AdamW does not emit blockscaled weight caches.
    raise RuntimeError(
        "Expert parameters found but neither eco_fused_backward nor eco_enabled "
        "is set. Standard PyTorch AdamW does not support NVFP4 quantized weights "
        "and will not emit blockscaled weight caches. "
        "Set eco_fused_backward=True and eco_enabled=True in your config."
    )
  else:
    # No expert params (e.g., dense-only model or all experts frozen).
    expert_optimizer = _NullExpertOptimizer()

  return expert_optimizer, dense_groups


def update_lr(optimizer: torch.optim.Optimizer, dense_groups: list[dict], step: int, tokens_seen: int, cfg: Config) -> float:
  """Update learning rate using a DeepSeek-style WSD schedule.

  WSD (Warmup-Sustain-Decay):
  1) Warmup: linear 0 → peak over warmup_steps (by step)
  2) Sustain: hold at peak until hold_tokens (by tokens)
  3) Cosine decay: decay over decay_tokens down to floor (absolute LR)
  4) Floor: hold at floor indefinitely

  Contract:
  - cfg.decay_floor is an *absolute* LR (not a ratio).
  - The same schedule shape is applied to expert, dense, and router peaks,
    with a shared absolute floor.
  """
  peak_expert = float(cfg.lr_expert)
  peak_dense = float(cfg.lr_dense)
  peak_router = float(cfg.lr_router)
  floor = float(cfg.decay_floor)

  if floor <= 0.0:
    raise RuntimeError(f"decay_floor must be > 0 (got {floor})")
  if floor > peak_expert or floor > peak_dense:
    raise RuntimeError(
      f"decay_floor ({floor}) must be <= lr_expert ({peak_expert}) and lr_dense ({peak_dense})"
    )

  # Compute schedule scale factor (0→1 during warmup, 1 during hold, 1→floor_ratio during decay)
  if step < cfg.warmup_steps:
    lr_scale = (step + 1) / max(1, int(cfg.warmup_steps))
  elif tokens_seen < cfg.hold_tokens:
    lr_scale = 1.0
  else:
    t = int(tokens_seen - cfg.hold_tokens)
    if t < cfg.decay_tokens:
      denom = float(max(1, int(cfg.decay_tokens)))
      lr_scale = 0.5 * (1.0 + math.cos(math.pi * float(t) / denom))
    else:
      lr_scale = 0.0

  # Apply schedule to each group's peak LR (floor applied at scale=0)
  floor_expert = floor
  floor_dense = floor
  floor_router = min(floor, peak_router)  # Router floor capped at its peak
  lr_expert = floor_expert + (peak_expert - floor_expert) * lr_scale
  lr_dense = floor_dense + (peak_dense - floor_dense) * lr_scale
  lr_router = floor_router + (peak_router - floor_router) * lr_scale

  for g in optimizer.param_groups:
    g['lr'] = lr_expert
  for g in dense_groups:
    # Router group uses lr_router, others use lr_dense
    if g.get('name') == 'router':
      g['lr'] = lr_router
    else:
      g['lr'] = lr_dense

  # Return the dense LR for logging (backward compatible with older single-LR configs).
  return float(lr_dense)


def _get_dp_group():
  """Get DP process group, or None if not initialized or DP=1."""
  try:
    from nmoe.distributed.init_groups import is_nmoe_parallel_initialized, get_data_parallel_group
    if is_nmoe_parallel_initialized():
      return get_data_parallel_group()
  except ImportError:
    pass
  return None


def _get_dp_size() -> int:
  """Get DP world size (1 if not initialized)."""
  try:
    from nmoe.distributed.init_groups import is_nmoe_parallel_initialized, get_dp_size
    if is_nmoe_parallel_initialized():
      return get_dp_size()
  except ImportError:
    pass
  return 1


def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, dense_groups: list[dict], zero2_state: dict, cfg: Config, world: int) -> None:
  """Optimizer step with ZeRO-2 and post-step hooks.

  Handles:
  - ZeRO-2 AdamW stepping for dense params (scoped to DP group when EP+DP)
  - Expert gradient AllReduce over DP group (when DP > 1)
  - Expert optimizer stepping
  - Post-step model updates (quantization rebuild, router bias)
  """
  dp_group = _get_dp_group()
  dp_size = _get_dp_size()

  # ZeRO-2 AdamW step for dense params.
  # When EP+DP is active, ZeRO-2 uses the DP group (not WORLD).
  # Dense params are replicated across DP ranks; ZeRO-2 shards within DP group.
  with _nvtx("opt/zero2_step"):
    if world > 1:
      pg = dp_group if dp_group is not None else dist.group.WORLD
      zero2.step_dense_adamw(
        dense_groups,
        state=zero2_state,
        pg=pg,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
      )
    else:
      zero2.step_dense_adamw(dense_groups, state=zero2_state)

  # Expert gradient AllReduce over DP group.
  # With EP+DP, each expert is replicated across dp_size ranks.
  # Their gradients must be averaged before the optimizer step.
  # Skip when fused backward-optimizer is active (AllReduce done inside backward).
  is_fused_eco = isinstance(optimizer, _NullExpertOptimizer)
  with _nvtx("opt/expert_allreduce"):
    if not is_fused_eco and dp_size > 1 and dp_group is not None:
      for p in optimizer.param_groups[0]['params']:
        if p.grad is not None:
          dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_group)

  # Expert optimizer step (params already sharded via RDEP across EP group).
  # _NullExpertOptimizer.step() is a no-op (fused backward-optimizer already ran).
  with _nvtx("opt/expert_step"):
    optimizer.step()

  # Post-step hooks (quantization rebuild, router bias update)
  with _nvtx("opt/post_hooks"), torch.no_grad():
    for blk in model.blocks:
      ffn = getattr(blk, 'ffn', None)
      if ffn is None:
        continue

      # Refresh quantized weight cache (FP8/NVFP4)
      # Skip when fused backward-optimizer is active (refresh done inside backward).
      if (not getattr(optimizer, "emits_weight_cache", False)) and hasattr(ffn, 'refresh_weight_cache'):
        try:
          ffn.refresh_weight_cache()
        except Exception:
          _log.warning("refresh_weight_cache() failed — stale quantized weights may be used", exc_info=True)

      # Update router bias for aux-free load balancing
      if hasattr(ffn, 'router') and hasattr(ffn, 'last_loads'):
        try:
          ffn.router.update_bias(ffn.last_loads, gamma=cfg.router_bias_update_rate)
        except Exception:
          _log.warning("router.update_bias() failed — load balancing may be affected", exc_info=True)
