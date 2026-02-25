r"""
nmoe: noumena's moe training library

   _ __   _ __ ___   ___   ___
  | '_ \ | '_ ` _ \ / _ \ / _ \
  | | | || | | | | | (_) |  __/
  |_| |_||_| |_| |_|\___/ \___|

Usage:
  python -m nmoe.train configs/moonlet.toml
  torchrun --nproc_per_node=8 -m nmoe.train configs/moonlight.toml
"""
import logging
import os
import socket
import sys
import ctypes
import time
from contextlib import nullcontext

import torch

from nmoe.config import Config, fingerprint, load_config_file
from nmoe.model import Transformer
from nmoe.data.loader import build_loader
from nmoe.data.sft_loader import build_sft_loader
from nmoe.opt import build_optimizer, update_lr, step
from nmoe.checkpoint import Checkpointer, load_checkpoint, save_checkpoint
from nmoe.metrics import init_metrics, start_metrics, log_training_step, stop_metrics, register_model_timers, cuda_time
from nmoe.attention.fa4_import import probe_fa4_flashmla_bindings
from nmoe.experiments import ExperimentTracker
from nmoe import runtime
from nmoe.eval.hooks import maybe_schedule_eval
from nmoe.fused_grad_clip import fused_clip_grad_norm_, fused_grad_norm_
from nmoe.fused_ce_loss import simple_fused_masked_ce_loss, chunked_projected_masked_ce_loss
from nmoe.nan_debug import nan_debug_active as _nan_debug_active
from nmoe.nan_debug import set_nan_debug_active

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _env_flag(name: str, default: str = "0") -> bool:
  return os.getenv(name, default) in ("1", "true", "True")


_USE_CHUNKED_PROJECTED_CE = _env_flag("NMOE_USE_CHUNKED_PROJECTED_CE", "1")
_CHUNKED_PROJECTED_CE_CHUNK = max(
  512,
  int(os.getenv("NMOE_CHUNKED_PROJECTED_CE_CHUNK", "4096")),
)


def _iter_tensors(x):
  if torch.is_tensor(x):
    yield x
  elif isinstance(x, (list, tuple)):
    for v in x:
      yield from _iter_tensors(v)
  elif isinstance(x, dict):
    for v in x.values():
      yield from _iter_tensors(v)


def _tensor_finite_stats(t: torch.Tensor) -> tuple[int, int, int, int]:
  nan = int(torch.isnan(t).sum().item())
  inf = int(torch.isinf(t).sum().item())
  numel = int(t.numel())
  finite = numel - nan - inf
  return numel, finite, nan, inf


def _collect_nonfinite_grad_details(
  named_params: list[tuple[str, torch.nn.Parameter]],
  max_items: int = 8,
) -> list[str]:
  """Collect first non-finite gradient parameter summaries."""
  details: list[str] = []
  for name, p in named_params:
    g = p.grad
    if g is None or not (g.is_floating_point() or g.is_complex()):
      continue
    finite_mask = torch.isfinite(g)
    if bool(finite_mask.all().item()):
      continue
    numel, finite, nan, inf = _tensor_finite_stats(g)
    g_f = g.detach().float()
    finite_vals = g_f[torch.isfinite(g_f)]
    if finite_vals.numel() > 0:
      abs_max = float(finite_vals.abs().amax().item())
      rms = float(finite_vals.square().mean().sqrt().item())
    else:
      abs_max = float("nan")
      rms = float("nan")
    details.append(
      f"{name} shape={tuple(g.shape)} dtype={g.dtype} "
      f"finite={finite}/{numel} nan={nan} inf={inf} "
      f"finite_absmax={abs_max:.6g} finite_rms={rms:.6g}"
    )
    if len(details) >= max_items:
      break
  return details


@torch.no_grad()
def _reference_grad_norm_fp64(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
  """Slow high-precision grad norm used only for NaN diagnostics."""
  if not parameters:
    return torch.tensor(0.0, device='cuda')

  device = parameters[0].device
  for p in parameters:
    if p.grad is not None:
      device = p.grad.device
      break

  total_sq = torch.zeros((), device=device, dtype=torch.float64)
  for p in parameters:
    g = p.grad
    if g is None or not (g.is_floating_point() or g.is_complex()):
      continue
    g_norm = torch.linalg.vector_norm(g, ord=2, dtype=torch.float32).to(dtype=torch.float64)
    total_sq += g_norm * g_norm
  return total_sq.sqrt().to(dtype=torch.float32)


def _register_nan_debug_hooks(model: torch.nn.Module):
  """Register module forward hooks that fail fast on non-finite outputs.

  Hooks are runtime-gated by process-local NaN debug scope to keep overhead negligible
  outside the scoped debug micro-step.
  """
  handles = []

  def _hook(name: str):
    def _fn(_module, _inputs, output):
      if not _nan_debug_active():
        return
      rank = int(os.getenv("RANK", "0"))
      # Keep hook diagnostics out of autograd graph so checkpoint recomputation
      # sees identical saved-tensor structure.
      with torch.no_grad():
        for idx, tensor in enumerate(_iter_tensors(output)):
          if not (torch.is_tensor(tensor) and (tensor.is_floating_point() or tensor.is_complex())):
            continue
          t = tensor.detach()
          if torch.isfinite(t).all():
            continue
          numel, finite, nan, inf = _tensor_finite_stats(t)
          raise RuntimeError(
            f"[NANDBG][rank={rank}] first failing module={name}[{idx}] "
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"finite={finite}/{numel} nan={nan} inf={inf}"
          )
    return _fn

  # Coarse and fine hooks: block outputs + attention/MoE submodule outputs.
  for i, block in enumerate(getattr(model, "blocks", [])):
    handles.append(block.register_forward_hook(_hook(f"transformer.blocks[{i}]")))
    attn = getattr(block, "attn", None)
    if attn is not None:
      handles.append(attn.register_forward_hook(_hook(f"transformer.blocks[{i}].attn")))
    ffn = getattr(block, "ffn", None)
    if ffn is not None:
      handles.append(ffn.register_forward_hook(_hook(f"transformer.blocks[{i}].ffn")))
  handles.append(model.register_forward_hook(_hook("transformer")))
  return handles


@torch.no_grad()
def _prime_nvfp4_weight_caches_for_checkpointing(
  model: torch.nn.Module,
  cfg: Config,
  rank: int,
  world: int,
) -> None:
  """Build NVFP4 MoE weight caches before first checkpointed forward.

  Lazy cache construction inside a checkpointed forward can make the recompute
  graph differ from the original forward (first pass builds cache, recompute does
  not), which triggers torch.utils.checkpoint.CheckpointError.
  """
  if not bool(getattr(cfg, "gradient_checkpointing", False)):
    return
  if str(getattr(cfg, "dtype", "bf16") or "bf16").lower() != "nvfp4":
    return

  targets: list[tuple[int, torch.nn.Module]] = []
  for idx, block in enumerate(getattr(model, "blocks", [])):
    moe = getattr(block, "ffn", None)
    if moe is None or not hasattr(moe, "refresh_weight_cache"):
      continue
    if not bool(getattr(moe, "_nvfp4_primary", False)):
      continue
    targets.append((idx, moe))

  if not targets:
    return

  if rank == 0:
    logger.info(
      "Priming NVFP4 MoE caches before first checkpointed step (%d layers)",
      len(targets),
    )

  rebuilt = 0
  for _, moe in targets:
    if getattr(moe, "_W_cache", None) is None:
      moe.refresh_weight_cache()
      rebuilt += 1
      # Flush transient allocations from cache-build kernels before next layer.
      torch.cuda.synchronize()

  if world > 1:
    import torch.distributed as _dist
    _dist.barrier()

  if rank == 0:
    logger.info(
      "NVFP4 MoE cache priming complete: rebuilt %d/%d layers",
      rebuilt,
      len(targets),
    )


def _validate_training_config(cfg: Config, world: int) -> None:
  """Validate training configuration parameters.

  Args:
      cfg: Training configuration
      world: World size for distributed training

  Raises:
      ValueError: If configuration is invalid
  """
  if cfg.batch_size <= 0:
    raise ValueError(f"batch_size must be > 0 (got {cfg.batch_size})")

  accum_steps = int(getattr(cfg, 'gradient_accumulation_steps', 1))
  if accum_steps <= 0:
    raise ValueError(
      f"gradient_accumulation_steps must be >= 1 (got {accum_steps})"
    )

  # For EP/DP training with gradient accumulation, the global batch is consumed
  # over `accum_steps` micro-steps. Require exact divisibility to avoid silent
  # floor division in local micro-batch construction.
  sft_mode = getattr(cfg, 'sft_enabled', False)
  ep_size = max(1, int(getattr(cfg, 'ep_size', 1)))
  if world % ep_size != 0:
    raise ValueError(
      f"world_size ({world}) must be divisible by ep_size ({ep_size}). "
      f"Cannot split {world} GPUs into groups of {ep_size}."
    )
  dp_size = int(getattr(cfg, 'dp_size', None) or max(1, world // ep_size))
  if dp_size <= 0:
    raise ValueError(f"dp_size must be >= 1 (got {dp_size})")
  if getattr(cfg, 'dp_size', None) is not None and (dp_size * ep_size) != world:
    raise ValueError(
      f"dp_size*ep_size must equal world_size when dp_size is set explicitly "
      f"(got dp_size={dp_size}, ep_size={ep_size}, world_size={world})"
    )

  denom = dp_size * accum_steps
  if denom <= 0:
    raise ValueError(f"Invalid batch denominator dp_size*accum_steps={denom}")

  if sft_mode:
    if (cfg.batch_size % denom) != 0:
      raise ValueError(
        f"batch_size ({cfg.batch_size}) must be divisible by dp_size*accum_steps "
        f"({dp_size}*{accum_steps}={denom}) in SFT mode. "
        "Each forward micro-batch uses batch_size/(dp_size*accum_steps)."
      )
  elif (cfg.batch_size % denom) != 0:
    raise ValueError(
      f"batch_size ({cfg.batch_size}) must be divisible by dp_size*accum_steps "
      f"({dp_size}*{accum_steps}={denom}). "
      "Uneven per-rank micro-batches break exact token accounting and DP averaging."
    )

  if cfg.seq_len <= 0:
    raise ValueError(f"seq_len must be > 0 (got {cfg.seq_len})")
  run_dtype = str(getattr(cfg, "dtype", "bf16") or "bf16").lower()
  if cfg.n_layers > cfg.n_dense_layers:
    if run_dtype not in {"fp8", "nvfp4"}:
      raise ValueError(
        f"MoE training requires blockscaled dtype ('fp8' or 'nvfp4'), got dtype={run_dtype!r}."
      )
  if run_dtype == "nvfp4":
    attn = str(getattr(cfg, "attn", "mla") or "mla").lower()
    attn_global_every = int(getattr(cfg, "attn_global_every", 1))
    if attn != "mla":
      raise ValueError(
        "dtype=nvfp4 production profile requires attn='mla'. "
        f"Got attn={attn!r}."
      )
    if attn_global_every != 1:
      raise ValueError(
        "dtype=nvfp4 production profile requires attn_global_every=1 "
        f"(pure MLA global attention). Got attn_global_every={attn_global_every}."
      )
  if cfg.n_activated_experts is not None and cfg.n_routed_experts is not None:
    if cfg.n_activated_experts <= 0:
      raise ValueError(f"n_activated_experts must be > 0 (got {cfg.n_activated_experts})")
    if cfg.n_activated_experts > cfg.n_routed_experts:
      raise ValueError(
        f"n_activated_experts ({cfg.n_activated_experts}) must be <= n_routed_experts ({cfg.n_routed_experts})"
      )


def _validate_required_cuda_bindings(cfg: Config) -> None:
  """Fail fast if required custom CUDA bindings are missing.

  This prevents expensive multi-node launches from silently drifting into
  partial or fallback kernel paths.
  """
  dtype = str(getattr(cfg, 'dtype', 'bf16') or 'bf16').lower()
  need_quant = dtype in {'fp8', 'nvfp4'}
  need_router = bool(getattr(cfg, 'use_fused_router', True))
  need_eco = bool(getattr(cfg, 'eco_fused_backward', False) and getattr(cfg, 'eco_require_cuda', True))
  use_fa4 = os.getenv('NMOE_USE_FA4', '1') in ('1', 'true', 'True')
  use_flashmla_sm100_fwd = os.getenv("NMOE_MLA_USE_FLASHMLA_SM100_FWD", "1") in ("1", "true", "True")
  require_flashmla = os.getenv('NMOE_REQUIRE_FLASHMLA', '1') in ('1', 'true', 'True')
  packed_attn_backend = os.getenv("NMOE_PACKED_ATTN_BACKEND", "flashmla").strip().lower()

  # All distributed MoE paths require the core RDEP extension.
  try:
    from nmoe.csrc import rdep as _rdep
  except Exception as e:
    raise RuntimeError(
      "Required CUDA extension nmoe.csrc.rdep is not importable. "
      "Build nmoe/csrc before launching multi-node training."
    ) from e

  missing: dict[str, list[str]] = {}

  core_required = [
    'init',
    'dispatch_meta_bf16',
    'dispatch_meta_blockscaled',
    'gather_meta_sorted_blockscaled',
    'gather_xe_blockscaled',
    'return_scatter_from_pad_blockscaled',
    'gather_dy_nogate_dist_bf16',
    'gather_dy_dist_bf16',
    'zero_padding_rows_bf16',
    'scatter_dx_dist_bf16',
    'scatter_dx_dist_from_pad_bf16',
    'scatter_gate_bf16_out_bf16',
    'send_dgate_dist_bf16_out_bf16',
    'bf16_wgrad_w2_cublaslt',
    'bf16_wgrad_w13_cublaslt',
    'bf16_wgrad_w13_pair_cublaslt',
    'bf16_prepare_offs_pad_host',
  ]
  missing_core = [name for name in core_required if not hasattr(_rdep, name)]
  if missing_core:
    missing['rdep_core'] = missing_core

  if need_quant:
    quant_required = [
      'build_grouped_gemm_metadata',
      'ct_nvfp4_to_bf16',
    ]
    if dtype == 'nvfp4':
      quant_required.extend([
        'quant_nvfp4',
        'quant_nvfp4_sf_strided_mma',
        'dequant_nvfp4_to_bf16_mma_sf',
      ])
    else:
      quant_required.extend([
        'quant_fp8',
        'quant_fp8_sf_strided_mma',
        'dequant_fp8_to_bf16_mma_sf',
      ])
    missing_quant = [name for name in quant_required if not hasattr(_rdep, name)]
    if missing_quant:
      missing['quant'] = missing_quant
    if dtype == 'nvfp4' and int(getattr(_rdep, 'ct_nvfp4_to_bf16_max_transpose_mode', 1)) < 2:
      missing.setdefault('quant', []).append('ct_nvfp4_to_bf16_max_transpose_mode>=2')

  if need_eco:
    eco_required = [
      'eco_adam_nvfp4_update',
      'eco_mv_accumulate',
      'nvfp4_recompute_group_scales',
    ]
    if bool(getattr(cfg, 'eco_factored_v', False)):
      eco_required.extend(['eco_adam_nvfp4_fv_update', 'eco_mv_accumulate_fv'])
    missing_eco = [name for name in eco_required if not hasattr(_rdep, name)]
    if missing_eco:
      missing['eco'] = missing_eco

  if need_router:
    try:
      lib = ctypes.CDLL(_rdep.__file__)
      router_required = [
        'fused_router_backward',
        'fused_router_backward_bf16',
        'fused_router_backward_bf16_fused',
        'fused_router_bwd_transpose',
      ]
      missing_router = [name for name in router_required if not hasattr(lib, name)]
    except OSError:
      missing_router = [
        'fused_router_backward',
        'fused_router_backward_bf16',
        'fused_router_backward_bf16_fused',
        'fused_router_bwd_transpose',
      ]
    if missing_router:
      missing['router'] = missing_router

  if require_flashmla and not use_fa4:
    missing['attention'] = ['NMOE_USE_FA4=1 required when NMOE_REQUIRE_FLASHMLA=1']
  if require_flashmla and packed_attn_backend != "flashmla":
    missing.setdefault('attention', []).append(
      "NMOE_PACKED_ATTN_BACKEND=flashmla required when NMOE_REQUIRE_FLASHMLA=1"
    )
  if use_fa4:
    attn_missing = probe_fa4_flashmla_bindings(
      require_fa4_fwd=not use_flashmla_sm100_fwd,
      require_flashmla_fwd=use_flashmla_sm100_fwd,
    )
    if attn_missing:
      missing['attention'] = attn_missing

  if missing:
    detail = "; ".join(f"{section}: {names}" for section, names in missing.items())
    raise RuntimeError(
      f"Missing required CUDA bindings for production launch ({detail}). "
      "Rebuild nmoe/csrc and redeploy before starting training."
    )


def train(cfg: Config):
  """Train MoE model. One clear path: forward → loss → backward → step → log → checkpoint."""
  # Force line-buffered stdout so all print() output is immediately visible
  # in log files (Python defaults to block-buffering when stdout is redirected)
  sys.stdout.reconfigure(line_buffering=True)
  _validate_required_cuda_bindings(cfg)
  run_dtype = str(getattr(cfg, 'dtype', 'bf16') or 'bf16').lower()
  if run_dtype == 'nvfp4' and not bool(getattr(cfg, 'resume', True)):
    raise RuntimeError(
      "dtype=nvfp4 requires resume=true with an imported NVFP4 checkpoint "
      "(no scratch/no BF16 fallback path)."
    )

  ep_size = getattr(cfg, 'ep_size', 1)
  rank, world = runtime.init(cfg.seed, ep_size=ep_size)
  _validate_training_config(cfg, world)
  # RDEP IPC phase barriers require deterministic barrier-call ordering across ranks.
  # PyTorch autograd worker-thread scheduling can reorder layer backward launches,
  # which can deadlock distributed dispatch-meta barriers on large MoE graphs.
  if (
    world > 1
    and str(getattr(cfg, "dtype", "bf16") or "bf16").lower() in {"fp8", "nvfp4"}
    and os.getenv("NMOE_RDEP_AUTOGRAD_SINGLE_THREAD", "1") in ("1", "true", "True")
    and hasattr(torch.autograd, "set_multithreading_enabled")
  ):
    torch.autograd.set_multithreading_enabled(False)
    if rank == 0:
      logger.info(
        "Autograd multithreading disabled for deterministic distributed RDEP barrier ordering "
        "(set NMOE_RDEP_AUTOGRAD_SINGLE_THREAD=0 to override)."
      )
  timers_on = os.getenv('NMOE_TIMERS', '0') not in ('0', 'false', 'False')
  time_ctx = cuda_time if timers_on else (lambda _tag: nullcontext())
  nvtx_on = os.getenv('NMOE_NVTX', '0') in ('1', 'true', 'True')
  nvtx_ok = bool(nvtx_on and torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'))
  nvtx_ctx = (torch.cuda.nvtx.range if nvtx_ok else (lambda _tag: nullcontext()))
  nan_debug_enabled = _env_flag("NMOE_NAN_DEBUG", "0")
  nan_debug_step = int(os.getenv("NMOE_NAN_DEBUG_STEP", "0"))
  nan_debug_micro = int(os.getenv("NMOE_NAN_DEBUG_MICRO", "0"))
  set_nan_debug_active(False)
  if nan_debug_enabled and rank == 0:
    print(
      f"[NANDBG] enabled for step={nan_debug_step}, micro={nan_debug_micro}",
      flush=True,
    )
  nan_trace_step0 = _env_flag("NMOE_NAN_TRACE_STEP0", "0")
  nan_trace_micro = int(os.getenv("NMOE_NAN_TRACE_MICRO", "0"))
  if nan_trace_step0 and rank == 0:
    print(
      f"[NANTRACE] detect_anomaly enabled for step=0, micro={nan_trace_micro} only",
      flush=True,
    )

  exp_tracker: ExperimentTracker | None = None
  run_id = os.getenv("NMOE_RUN", "")
  if rank == 0:
    exp_tracker = ExperimentTracker(cfg)
    run_id = exp_tracker.start_run(run_id=run_id or None)

  checkpointer = Checkpointer(
    base=cfg.checkpoint_dir,
    keep_last=getattr(cfg, 'checkpoint_keep_last_n', 5),
    async_io=True,
    async_max_queue=1,
  )

  accum_steps = int(getattr(cfg, 'gradient_accumulation_steps', 1))
  if accum_steps <= 0:
    raise RuntimeError(
      f"gradient_accumulation_steps must be >= 1, got {accum_steps}"
    )

  # Build components
  sft_mode = getattr(cfg, 'sft_enabled', False)
  if sft_mode:
    # SFT mode: use SFT loader with EP/DP-aware rank computation
    ep_size = getattr(cfg, 'ep_size', 1)
    dp_size = getattr(cfg, 'dp_size', None)
    if dp_size is None:
      dp_size = max(1, world // ep_size)
    dp_rank = rank // ep_size  # EP ranks are contiguous; DP rank = rank / ep_size
    if rank == 0:
      logger.info("SFT ep_size=%d dp_size=%d dp_rank=%d world=%d", ep_size, dp_size, dp_rank, world)
    loader = build_sft_loader(cfg, dp_rank=dp_rank, dp_world_size=dp_size, print_fn=print)
    plan = None  # SFT has no MixturePlan
  else:
    # Pretraining data is sharded by DP replicas, not by global world rank.
    ep_size = max(1, int(getattr(cfg, 'ep_size', 1)))
    dp_size = int(getattr(cfg, 'dp_size', None) or max(1, world // ep_size))
    dp_rank = rank // ep_size
    if rank == 0:
      logger.info(
        "Pretrain loader ep_size=%d dp_size=%d dp_rank=%d world=%d accum_steps=%d",
        ep_size, dp_size, dp_rank, world, accum_steps,
      )
    loader, plan = build_loader(
      cfg,
      dp_rank,
      dp_size,
      gradient_accumulation_steps=accum_steps,
    )

  # ── Epoch → steps resolution ──────────────────────────────────────────────
  if cfg.epochs is not None:
    if sft_mode:
      # SFT: one epoch = one pass over all SFT examples
      micro_batch = max(1, cfg.batch_size // dp_size)
      dataset_len = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader)
      steps_per_epoch = max(1, dataset_len // micro_batch)
      resolved_steps = cfg.epochs * steps_per_epoch
    else:
      # Pretraining: one epoch = one pass over all tokens
      # total_tokens estimated from dataset size * seq_len
      ep_size = max(1, int(getattr(cfg, 'ep_size', 1)))
      dp_size = int(getattr(cfg, 'dp_size', None) or max(1, world // ep_size))
      micro_batch = cfg.batch_size // max(1, dp_size * accum_steps)
      dataset_len = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader)
      resolved_steps = cfg.epochs * (dataset_len // micro_batch)

    if rank == 0:
      print(f"[TRAIN] Epoch mode: {cfg.epochs} epoch(s) × {dataset_len} samples "
            f"/ {micro_batch} micro_batch = {resolved_steps} steps")
      sys.stdout.flush()

    cfg.steps = resolved_steps

  with nvtx_ctx('train/model_init'):
    use_fused_router = bool(getattr(cfg, 'use_fused_router', True))
    if not use_fused_router:
      raise RuntimeError(
        "use_fused_router=False is disabled for production training. "
        "Set use_fused_router=true."
      )
    model = Transformer(cfg, use_fused_router=use_fused_router).cuda()
    model.init_weights()
    model.train()
  nan_debug_handles = _register_nan_debug_hooks(model) if nan_debug_enabled else []

  # Enable gradient checkpointing if configured (required for NVFP4 primary weights:
  # only 1 layer's BF16 scratch activations in GPU memory at a time)
  if getattr(cfg, 'gradient_checkpointing', False):
    model.gradient_checkpointing_enable()
    if rank == 0:
      logger.info("Gradient checkpointing enabled")
      if not bool(getattr(cfg, "checkpoint_moe_ffn", False)):
        logger.info(
          "Gradient checkpointing mode: attention-only (checkpoint_moe_ffn=false). "
          "MoE FFN checkpoint replay is disabled to avoid distributed RDEP recompute deadlocks."
        )

  if timers_on:
    register_model_timers(model)
  optimizer, dense_groups = build_optimizer(model, cfg)
  metrics_state = init_metrics(model, cfg.seq_len)
  metrics_ctx = start_metrics(run_id=run_id, metrics_dir=cfg.metrics_dir)

  # Upload a minimal safe config subset to local W&B run metadata (rank 0 only)
  if rank == 0 and metrics_ctx.wandb_run is not None:
    try:
      safe_keys = (
        "name",
        "steps",
        "batch_size",
        "seq_len",
        "ep_size",
        "dp_size",
        "dtype",
        "n_layers",
        "n_heads",
        "dim",
        "vocab_size",
        "n_routed_experts",
        "n_activated_experts",
        "moe_inter_dim",
        "lr_dense",
        "lr_expert",
        "lr_gate",
        "weight_decay",
      )
      safe_cfg = {k: getattr(cfg, k) for k in safe_keys if hasattr(cfg, k)}
      metrics_ctx.wandb_run.config.update(safe_cfg, allow_val_change=True)
    except Exception as e:
      raise RuntimeError("Local W&B config upload failed (no-fallback policy).") from e

  zero2_state = {}
  with nvtx_ctx('train/load_checkpoint'):
    start_step, tokens_seen, zero2_state = load_checkpoint(checkpointer, model, optimizer, loader, plan, cfg, rank, print)

  run_dtype = str(getattr(cfg, 'dtype', 'bf16') or 'bf16').lower()
  runtime_rdep_mode = "ipc"
  moe_layers = list(getattr(model, "_moe_layers", []) or [])
  if moe_layers:
    runtime_modes: set[str] = set()
    for idx, moe in enumerate(moe_layers):
      mode = str(getattr(getattr(moe, "_rdep", None), "_mode", "") or "").lower()
      if mode not in {"single", "ipc", "hybrid"}:
        raise RuntimeError(
          f"Invalid RDEP mode for MoE layer {idx}: {mode!r}. "
          "Expected one of {'single','ipc','hybrid'}."
        )
      runtime_modes.add(mode)
    if len(runtime_modes) != 1:
      raise RuntimeError(
        f"Inconsistent RDEP runtime modes across MoE layers: {sorted(runtime_modes)}"
      )
    runtime_rdep_mode = next(iter(runtime_modes))
  if run_dtype == 'nvfp4':
    if not bool(getattr(cfg, 'eco_fused_backward', False)):
      raise RuntimeError(
        "dtype=nvfp4 requires eco_fused_backward=true for production training. "
        "Non-fused NVFP4 expert execution is forbidden."
      )
    direct_loads = int(getattr(model, "_runtime_nvfp4_checkpoint_direct_expert_loads", 0))
    non_direct_loads = int(getattr(model, "_runtime_nvfp4_checkpoint_non_direct_expert_loads", 0))
    if non_direct_loads > 0:
      raise RuntimeError(
        "NVFP4 checkpoint fallback detected before training. "
        f"non_direct_expert_loads={non_direct_loads}. "
        "Production path requires direct NVFP4 expert buffer loading."
      )
    if int(start_step) > 0 and direct_loads <= 0:
      raise RuntimeError(
        "NVFP4 training requires direct expert NVFP4 checkpoint load, "
        "but no direct expert loads were recorded."
      )
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world <= 1 or runtime_rdep_mode not in {"ipc", "hybrid"}:
      raise RuntimeError(
        "dtype=nvfp4 production path requires distributed RDEP "
        "(world>1, mode in {'ipc','hybrid'}). "
        f"Got world={world}, mode={runtime_rdep_mode!r}."
      )
    if runtime_rdep_mode == "ipc" and int(ep_size) != int(local_world):
      raise RuntimeError(
        "dtype=nvfp4 IPC mode requires node-local EP grouping "
        "(ep_size == LOCAL_WORLD_SIZE) to avoid cross-node expert dispatch. "
        f"Got ep_size={ep_size}, LOCAL_WORLD_SIZE={local_world}."
      )
    non_primary_layers = [
      idx for idx, moe in enumerate(moe_layers)
      if not bool(getattr(moe, "_nvfp4_primary", False))
    ]
    if non_primary_layers:
      raise RuntimeError(
        "NVFP4 production training requires NVFP4-primary expert buffers on every MoE layer. "
        f"Missing _nvfp4_primary on layers: {non_primary_layers[:16]}"
      )

  dp_group = None
  dp_world_size = 1
  if world > 1:
    from nmoe.opt import _get_dp_group, _get_dp_size
    dp_group = _get_dp_group()
    dp_world_size = _get_dp_size()

  # Eagerly allocate ZeRO-2 flat buffers and re-point dense params into them.
  # This must happen BEFORE the first forward pass: at this point only model weights
  # are on GPU, leaving ~30+ GiB free.  If deferred to the optimizer step (lazy),
  # the allocation competes with activations and gradients, causing OOM.
  if world > 1 and dense_groups:
    from nmoe import zero2
    zero2.eager_init(dense_groups, pg=dp_group)
    if rank == 0:
      logger.info("ZeRO-2 flat buffers eagerly initialized for %d dense groups", len(dense_groups))

  # Fused backward-optimizer: create and attach after checkpoint load
  fused_eco = None
  if getattr(cfg, 'eco_fused_backward', False):
    from nmoe.eco import FusedBackwardECO
    fused_eco = FusedBackwardECO(model, cfg)
    # Set DP group for gradient AllReduce inside backward.
    if dp_group is not None and dp_world_size > 1:
      fused_eco.set_dp_group(dp_group, dp_world_size)

    expected_dp_size = getattr(cfg, 'dp_size', None)
    if expected_dp_size is None:
      expected_dp_size = max(1, world // max(1, ep_size))
    if expected_dp_size > 1 and (dp_group is None or dp_world_size <= 1):
      raise RuntimeError(
        "eco_fused_backward=True requires initialized DP process groups when dp_size > 1. "
        f"expected_dp_size={expected_dp_size}, detected_dp_size={dp_world_size}."
      )
    # Restore FusedBackwardECO state from checkpoint if available
    pending_state = getattr(model, '_pending_fused_eco_state', None)
    if pending_state is not None:
      fused_eco.load_state_dict(pending_state)
      del model._pending_fused_eco_state
      if rank == 0:
        logger.info("Restored FusedBackwardECO state from checkpoint")
    fused_eco.attach(model)
    if rank == 0:
      logger.info("FusedBackwardECO attached: %d MoE modules", len(fused_eco._moes))

  # Ensure first checkpointed forward and recompute see identical code paths.
  # NVFP4 lazy MoE cache construction inside forward can otherwise trigger
  # torch.utils.checkpoint tensor-count mismatch on backward.
  _prime_nvfp4_weight_caches_for_checkpointing(model, cfg, rank, world)

  last_loss: torch.Tensor | None = None
  config_fingerprint = fingerprint(cfg)
  checkpoint_every = getattr(cfg, 'checkpoint_every', 100)
  grad_norm_params = [p for p in model.parameters() if p.requires_grad]
  named_grad_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
  dense_clip_params: list[torch.nn.Parameter] = []
  if dense_groups:
    seen_dense: set[int] = set()
    for group in dense_groups:
      for p in group.get('params', []):
        if not isinstance(p, torch.nn.Parameter) or not p.requires_grad:
          continue
        pid = id(p)
        if pid in seen_dense:
          continue
        seen_dense.add(pid)
        dense_clip_params.append(p)

  # Gradient accumulation configuration
  step_heartbeat = _env_flag("NMOE_STEP_HEARTBEAT", "1")
  if step_heartbeat and rank == 0:
    logger.info("Step heartbeat enabled (NMOE_STEP_HEARTBEAT=1): per-micro-step stage markers will be printed")
  if accum_steps > 1 and rank == 0:
    logger.info("Gradient accumulation: %d micro-steps per optimizer step", accum_steps)
    logger.info("Effective batch size: %d (micro_batch=%d x %d)", cfg.batch_size, cfg.batch_size // accum_steps, accum_steps)

  # === Enable gradient anomaly detection for debugging (set NMOE_DETECT_ANOMALY=1) ===
  if os.getenv('NMOE_DETECT_ANOMALY', '0') == '1':
    torch.autograd.set_detect_anomaly(True)
    if rank == 0:
      logger.warning("[NAN] torch.autograd.detect_anomaly() ENABLED - expect 2-3x slower backward")

  # === One-time model weight sanity check (after checkpoint load) ===
  if rank == 0:
    nan_weight_params = []
    for name, p in model.named_parameters():
      nan_count = torch.isnan(p.data).sum().item()
      inf_count = torch.isinf(p.data).sum().item()
      if nan_count > 0 or inf_count > 0:
        nan_weight_params.append(f"{name}(nan={nan_count},inf={inf_count},numel={p.numel()})")
    if nan_weight_params:
      logger.warning("[NAN] Model weights contain NaN/Inf after checkpoint load:")
      print("[NAN] Model weights contain NaN/Inf after checkpoint load:", flush=True)
      for param_info in nan_weight_params[:20]:
        logger.warning(f"[NAN]   {param_info}")
        print(f"[NAN]   {param_info}", flush=True)
      if len(nan_weight_params) > 20:
        logger.warning(f"[NAN]   ... and {len(nan_weight_params) - 20} more parameters")
        print(f"[NAN]   ... and {len(nan_weight_params) - 20} more parameters", flush=True)
    else:
      logger.info("[NAN] Model weights sanity check passed: no NaN/Inf detected")
      print("[NAN] Model weights sanity check passed: no NaN/Inf detected", flush=True)

  # === Warmup NCCL communicators ===
  # Force NCCL ring/tree setup for all process groups BEFORE training.
  # The DP group spans all 16 nodes and its first collective would otherwise
  # happen during backward (ECO all_reduce), making hangs hard to debug.
  if world > 1:
    with nvtx_ctx('train/dist_init'):
      import torch.distributed as _dist
      _warmup_t = torch.ones(1, device='cuda')

      if rank == 0:
        print("[INIT] warming up NCCL communicators...", flush=True)

      # Global group
      _dist.all_reduce(_warmup_t, group=None)
      if rank == 0:
        print(f"[INIT] global all_reduce OK (world={world})", flush=True)

      # DP group
      try:
        from nmoe.distributed.init_groups import is_nmoe_parallel_initialized, get_data_parallel_group, get_dp_size
        if is_nmoe_parallel_initialized():
          dp_group = get_data_parallel_group()
          dp_size = get_dp_size()
          _dist.all_reduce(_warmup_t, group=dp_group)
          if rank == 0:
            print(f"[INIT] DP group all_reduce OK (dp_size={dp_size})", flush=True)
      except Exception as e:
        raise RuntimeError("DP group communicator warmup failed (no-fallback policy).") from e

      # EP group
      try:
        from nmoe.distributed.init_groups import get_ep_group, get_ep_size
        if is_nmoe_parallel_initialized():
          ep_group = get_ep_group()
          ep_size = get_ep_size()
          _dist.all_reduce(_warmup_t, group=ep_group)
          if rank == 0:
            print(f"[INIT] EP group all_reduce OK (ep_size={ep_size})", flush=True)
      except Exception as e:
        raise RuntimeError("EP group communicator warmup failed (no-fallback policy).") from e

      # Barrier to ensure all ranks complete warmup
      _dist.barrier()
      if rank == 0:
        print("[INIT] all NCCL warmup complete, starting training", flush=True)

      del _warmup_t

  try:
    with nvtx_ctx('train/run'):
      for step_num in range(start_step, cfg.steps):
        lr = update_lr(optimizer, dense_groups, step_num, tokens_seen, cfg)

        # Gradient accumulation: inner loop over micro-batches
        accumulated_loss = 0.0
        accumulated_loss_gpu = torch.tensor(0.0, device='cuda')
        t0 = time.perf_counter()

        for micro_step in range(accum_steps):
         with nvtx_ctx(f'train/micro_{micro_step}'):
          with nvtx_ctx('train/data_load'):
            cu_seqlens = None
            if sft_mode:
              loader_result = loader.next()
              if len(loader_result) == 4:
                # Packed mode: (inputs, targets, loss_mask, cu_seqlens)
                inputs, targets, loss_mask, cu_seqlens = loader_result
              else:
                # Unpacked mode: (inputs, targets, loss_mask)
                inputs, targets, loss_mask = loader_result
            else:
              inputs, targets = loader.next()
              loss_mask = None

          if step_heartbeat and rank == 0:
            print(f"[HB] step={step_num} micro={micro_step} stage=fwd_start", flush=True)
          nan_debug_scope = (
            nan_debug_enabled
            and step_num == nan_debug_step
            and micro_step == nan_debug_micro
          )
          nan_trace_scope = (
            nan_trace_step0
            and step_num == 0
            and micro_step == nan_trace_micro
          )
          set_nan_debug_active(nan_debug_scope)
          if nan_debug_scope:
            tok_min = int(inputs.min().item())
            tok_max = int(inputs.max().item())
            bad_tok = int(((inputs < 0) | (inputs >= cfg.vocab_size)).sum().item())
            mask_sum = float(loss_mask.sum().item()) if loss_mask is not None else -1.0
            print(
              f"[NANDBG][rank={rank}] step={step_num} micro={micro_step} "
              f"inputs shape={tuple(inputs.shape)} dtype={inputs.dtype} "
              f"min={tok_min} max={tok_max} bad_token_count={bad_tok} "
              f"mask_sum={mask_sum:.1f}",
              flush=True,
            )

          with nvtx_ctx('train/fwd_total'), time_ctx('time_ms/fwd_total'):
            logits = None
            hidden = None
            try:
              if _USE_CHUNKED_PROJECTED_CE:
                hidden = model(inputs, cu_seqlens=cu_seqlens, return_hidden=True)
              else:
                logits = model(inputs, cu_seqlens=cu_seqlens)
            except RuntimeError as e:
              if nan_debug_scope:
                raise RuntimeError(
                  f"[NANDBG][rank={rank}] forward failed at step={step_num}, micro={micro_step}: {e}"
                ) from e
              raise
          if step_heartbeat and rank == 0:
            print(f"[HB] step={step_num} micro={micro_step} stage=fwd_done", flush=True)

          if nan_debug_scope:
            dbg_tensor = hidden if _USE_CHUNKED_PROJECTED_CE else logits
            dbg_name = "hidden" if _USE_CHUNKED_PROJECTED_CE else "logits"
            numel, finite, nan, inf = _tensor_finite_stats(dbg_tensor)
            if finite == numel and numel > 0:
              dbg_f = dbg_tensor.float()
              lmin = float(dbg_f.amin().item())
              lmax = float(dbg_f.amax().item())
            else:
              lmin = float("nan")
              lmax = float("nan")
            print(
              f"[NANDBG][rank={rank}] step={step_num} micro={micro_step} "
              f"{dbg_name} shape={tuple(dbg_tensor.shape)} dtype={dbg_tensor.dtype} "
              f"finite={finite}/{numel} nan={nan} inf={inf} min={lmin:.6g} max={lmax:.6g}",
              flush=True,
            )
            if nan > 0 or inf > 0:
              raise RuntimeError(
                f"[NANDBG][rank={rank}] non-finite {dbg_name} at step={step_num}, micro={micro_step}"
              )

          # Fail-fast on token drops (indicates expert load collapse)
          _dropped = getattr(model, 'total_dropped_tokens', 0)
          if _dropped > 0:
            _drop_pct = _dropped / (inputs.numel()) * 100
            if rank == 0:
              print(f"[WARN] RDEP dropped {_dropped:,} tokens ({_drop_pct:.1f}%) "
                    f"at step {step_num} — expert load collapse detected. "
                    f"Consider enabling aux_loss_alpha or reducing route_scale.",
                    file=sys.stderr)
              sys.stderr.flush()
            # Hard fail if >5% tokens dropped (training is corrupted)
            if _drop_pct > 5.0:
              raise RuntimeError(
                f"RDEP dropped {_dropped:,} tokens ({_drop_pct:.1f}% > 5% threshold). "
                f"Expert routing has collapsed. Fix: set aux_loss_alpha > 0 "
                f"and/or reduce route_scale (currently may be too high)."
              )

          with nvtx_ctx('train/loss'), time_ctx('time_ms/loss'):
            if _USE_CHUNKED_PROJECTED_CE:
              if hidden is None:
                raise RuntimeError("Chunked projected CE requires hidden activations from model.forward.")
              loss = chunked_projected_masked_ce_loss(
                hidden=hidden,
                lm_head_weight=model.lm_head.weight,
                targets=targets,
                mask=loss_mask,
                eos_token_id=cfg.eos_token_id,
                logits_scale_factor=float(model.logits_scale_factor),
                chunk_size=_CHUNKED_PROJECTED_CE_CHUNK,
              )
            else:
              # Fused masked cross-entropy loss: replaces ~12 kernels with ~6-7 kernels
              # - FP32 cast for numerical stability (BF16 softmax over 129K vocab fails)
              # - PyTorch's optimized cross_entropy (cuDNN backend)
              # - torch.dot for fused mask*loss+sum (1 kernel vs 2)
              if logits is None:
                raise RuntimeError("simple_fused_masked_ce_loss requires logits from model.forward.")
              loss = simple_fused_masked_ce_loss(
                logits=logits,
                targets=targets,
                mask=loss_mask,
                vocab_size=cfg.vocab_size,
                eos_token_id=cfg.eos_token_id,
              )
            # Token-weight loss across DP ranks when supervised token counts differ.
            # Per-rank CE is normalized by local mask sum, while gradients are reduced
            # with DP AVG. Without this correction, sparse ranks are overweighted.
            rank_loss_scale = torch.ones((), device=loss.device, dtype=torch.float32)
            global_token_count = torch.ones((), device=loss.device, dtype=torch.float32)
            if world > 1 and dp_group is not None and dp_world_size > 1:
              import torch.distributed as _dist
              if loss_mask is not None:
                local_token_count = loss_mask.to(device=loss.device, dtype=torch.float32).sum()
              elif cfg.eos_token_id is not None:
                local_token_count = (targets != int(cfg.eos_token_id)).to(dtype=torch.float32).sum()
              else:
                local_token_count = torch.full(
                  (),
                  float(targets.numel()),
                  device=loss.device,
                  dtype=torch.float32,
                )
              global_token_count = local_token_count.clone()
              _dist.all_reduce(global_token_count, op=_dist.ReduceOp.SUM, group=dp_group)
              rank_loss_scale = (float(dp_world_size) * local_token_count) / global_token_count
              loss = loss * rank_loss_scale.to(dtype=loss.dtype)
            aux_loss = model.get_router_aux_loss()
            if aux_loss.device != loss.device:
              aux_loss = aux_loss.to(device=loss.device)
            loss = loss + aux_loss.to(dtype=loss.dtype)
            # Scale loss by accumulation steps so gradients are averaged
            if accum_steps > 1:
              loss = loss / accum_steps
            # Fail-fast on non-finite loss before backward. With fused ECO enabled,
            # expert optimizer updates happen inside backward, so we must abort first.
            diag_scalars = torch.stack((
              rank_loss_scale.to(dtype=torch.float32),
              aux_loss.detach().to(dtype=torch.float32),
              loss.detach().to(dtype=torch.float32),
            ))
            diag_finite = torch.isfinite(diag_scalars).to(dtype=torch.int32)
            micro_bad = torch.stack((
              (global_token_count <= 0).to(dtype=torch.int32),
              1 - diag_finite[0],
              1 - diag_finite[1],
              1 - diag_finite[2],
            ))
            if world > 1:
              import torch.distributed as _dist
              if _dist.is_available() and _dist.is_initialized():
                _dist.all_reduce(micro_bad, op=_dist.ReduceOp.MAX)

          if bool(micro_bad.any().item()):
            bad_token_count, bad_rank_scale, bad_aux_loss, bad_loss = (
              int(v) for v in micro_bad.detach().cpu().tolist()
            )
            token_count_v = float(global_token_count.detach().to(dtype=torch.float32).item())
            rank_loss_scale_v = float(diag_scalars[0].item())
            aux_loss_v = float(diag_scalars[1].item())
            loss_unscaled_v = float(diag_scalars[2].item()) * (accum_steps if accum_steps > 1 else 1)
            if bad_token_count != 0:
              raise RuntimeError(
                "Non-positive supervised token count across DP group during loss scaling."
              )
            if bad_rank_scale != 0:
              raise RuntimeError(
                f"Non-finite DP token-weight loss scale: {rank_loss_scale_v} (global_token_count={token_count_v})"
              )
            if bad_aux_loss != 0:
              raise RuntimeError(
                f"Non-finite router aux loss detected at step={step_num}, micro_step={micro_step}: "
                f"{aux_loss_v}"
              )
            if bad_loss != 0:
              raise RuntimeError(
                f"Non-finite loss detected at step={step_num}, micro_step={micro_step}. "
                f"loss_unscaled={loss_unscaled_v}. Aborting before backward/optimizer update."
              )
          if step_heartbeat and rank == 0:
            aux_loss_unscaled = float(diag_scalars[1].item())
            micro_loss_unscaled = float(diag_scalars[2].item()) * (accum_steps if accum_steps > 1 else 1)
            print(
              f"[HB] step={step_num} micro={micro_step} stage=loss_done "
              f"loss={micro_loss_unscaled:.6f} aux={aux_loss_unscaled:.6f}",
              flush=True,
            )

          # Zero gradients: on first micro-step, zero everything. On subsequent
          # micro-steps, only zero if no accumulation for dense params is needed.
          # With fused_eco, expert gradients are consumed in backward and don't
          # accumulate. Dense gradients DO accumulate across micro-steps via autograd.
          if micro_step == 0:
            model.zero_grad(set_to_none=True)
          # For micro_step > 0 with fused_eco: don't zero — dense grads should
          # accumulate. Expert grads are consumed in fused_update each micro-step.
          # For micro_step > 0 without fused_eco: don't zero — all grads accumulate.

          if fused_eco is not None:
            # update_lr() already set optimizer.param_groups[0]['lr'] to scheduled lr_expert
            lr_expert = float(optimizer.param_groups[0]['lr'])
            fused_eco.set_lr(lr_expert)
            fused_eco.set_microstep(micro_step, accum_steps)
            fused_eco.pre_backward(step_num)
          if step_heartbeat and rank == 0:
            print(f"[HB] step={step_num} micro={micro_step} stage=bwd_start", flush=True)
          if nan_trace_scope:
            print(
              f"[NANTRACE][rank={rank}] detect_anomaly active at step={step_num}, micro={micro_step}",
              flush=True,
            )
          with nvtx_ctx('train/bwd_total'), time_ctx('time_ms/bwd_total'):
            if nan_trace_scope:
              with torch.autograd.detect_anomaly(check_nan=True):
                loss.backward()
            else:
              loss.backward()
          if nan_debug_scope:
            _micro_grad_params = [p for p in grad_norm_params if p.grad is not None]
            micro_grad_norm = (
              fused_grad_norm_(_micro_grad_params)
              if _micro_grad_params else torch.tensor(0.0, device='cuda')
            )
            if not torch.isfinite(micro_grad_norm).item():
              bad_grad_details = _collect_nonfinite_grad_details(named_grad_params, max_items=8)
              detail = "; ".join(bad_grad_details) if bad_grad_details else "none"
              raise RuntimeError(
                f"[NANDBG][rank={rank}] non-finite micro grad_norm at "
                f"step={step_num}, micro={micro_step}: {float(micro_grad_norm)}. "
                f"First non-finite grads: {detail}"
              )
          set_nan_debug_active(False)
          if step_heartbeat and rank == 0:
            print(f"[HB] step={step_num} micro={micro_step} stage=bwd_done", flush=True)
          if fused_eco is not None and fused_eco.is_final_microstep:
            # Drain async ECO comm/update queue only once per optimizer step.
            # Non-final micro-steps only accumulate optimizer state (no weight update),
            # so forcing a full drain/sync here is unnecessary overhead.
            if step_heartbeat and rank == 0:
              qlen, qmb = fused_eco.pending_queue_state()
              print(
                f"[HB] step={step_num} stage=eco_drain_start pending={qlen} pending_mb={qmb:.1f}",
                flush=True,
              )
            eco_drain_t0 = time.perf_counter()
            fused_eco.post_backward()
            if step_heartbeat and rank == 0:
              qlen, qmb = fused_eco.pending_queue_state()
              eco_drain_ms = (time.perf_counter() - eco_drain_t0) * 1000.0
              print(
                f"[HB] step={step_num} stage=eco_drain_done ms={eco_drain_ms:.1f} "
                f"pending={qlen} pending_mb={qmb:.1f}",
                flush=True,
              )

          accumulated_loss_gpu += loss.detach() * (accum_steps if accum_steps > 1 else 1)

        # End of micro-batch loop. Now do optimizer step.
        loader_wait_ms = (time.perf_counter() - t0) * 1000.0

        # Lightweight NaN guard: compute grad_norm once and check for NaN/Inf
        # instead of iterating all 200K+ parameters with .item() calls.
        # Uses fused foreach op (1 kernel) instead of per-param norm computation (~N kernels)
        _grad_params = [p for p in grad_norm_params if p.grad is not None]
        grad_norm = fused_grad_norm_(_grad_params) if _grad_params else torch.tensor(0.0, device='cuda')
        local_grad_norm_finite = torch.isfinite(grad_norm).to(dtype=torch.int32)
        grad_norm_finite = local_grad_norm_finite.clone()
        if world > 1:
          import torch.distributed as _dist
          if _dist.is_available() and _dist.is_initialized():
            _dist.all_reduce(grad_norm_finite, op=_dist.ReduceOp.MIN)
        _grad_norm_bad = bool((grad_norm_finite == 0).item())
        if _grad_norm_bad:
            local_grad_norm_bad = bool((local_grad_norm_finite == 0).item())
            grad_norm_value = float(grad_norm.detach().to(dtype=torch.float32).item())
            bad_ranks: list[int] = []
            if world > 1:
              import torch.distributed as _dist
              if _dist.is_available() and _dist.is_initialized():
                local_bad_t = torch.tensor(
                  1 if local_grad_norm_bad else 0,
                  device=grad_norm.device,
                  dtype=torch.int32,
                )
                gathered = [torch.zeros_like(local_bad_t) for _ in range(_dist.get_world_size())]
                _dist.all_gather(gathered, local_bad_t)
                gathered_host = torch.stack(gathered).detach().cpu().tolist()
                bad_ranks = [i for i, t in enumerate(gathered_host) if int(t) == 1]

            bad_grad_details = _collect_nonfinite_grad_details(named_grad_params, max_items=8)
            detail = "; ".join(bad_grad_details) if bad_grad_details else "none"
            if local_grad_norm_bad and not bad_grad_details:
              ref_norm = _reference_grad_norm_fp64(_grad_params)
              ref_finite = bool(torch.isfinite(ref_norm).item())
              detail = (
                "no elementwise non-finite dense grads found; "
                "fused_grad_norm_ may have overflowed/underflowed. "
                f"fp64_ref_grad_norm={float(ref_norm):.6g} "
                f"fp64_ref_finite={int(ref_finite)}"
              )
            if not local_grad_norm_bad:
              detail = (
                f"remote rank reported non-finite dense grad_norm; local_grad_norm={float(grad_norm_value):.6g}; "
                f"local_non_finite_grads={detail}"
              )
            rank_detail = f"bad_ranks={bad_ranks}" if bad_ranks else "bad_ranks=unknown"
            raise RuntimeError(
              f"Non-finite dense grad_norm detected at step={step_num}: {float(grad_norm_value)}. "
              f"First non-finite grads: {detail}. {rank_detail}. Aborting before optimizer step."
            )
        else:
            # Only clip and step when gradients are clean
            # Gradient clipping (important for SFT stability with quantized training)
            # When fused_eco is active, expert grads are already consumed — only clip dense params.
            # Dense gradients accumulate across micro-steps (autograd adds them).
            # Gradient clipping using fused multi-tensor ops (2 kernels instead of ~2*N+3)
            # See nmoe/fused_grad_clip.py for implementation details
            grad_clip = getattr(cfg, 'grad_clip', 0.0)
            if grad_clip > 0:
              with nvtx_ctx('train/grad_clip'):
                if fused_eco is not None:
                  # Only clip dense parameters (expert grads already consumed in backward)
                  dense_params = [p for p in dense_clip_params if p.grad is not None]
                  if dense_params:
                    fused_clip_grad_norm_(dense_params, grad_clip)
                else:
                  if _grad_params:
                    fused_clip_grad_norm_(_grad_params, grad_clip)

            with nvtx_ctx('train/opt_step'), time_ctx('time_ms/opt_step'):
              step(model, optimizer, dense_groups, zero2_state, cfg, world)

        tokens_this_step = int(inputs.numel()) * accum_steps
        tokens_seen += cfg.batch_size * cfg.seq_len
        loss_scale = float(accum_steps) if accum_steps > 1 else 1.0
        last_loss = (accumulated_loss_gpu / loss_scale).detach()

        s = step_num + 1
        if step_heartbeat and rank == 0:
          print(f"[HB] step={step_num} stage=opt_done", flush=True)
        log_training_step(
          s,
          model=model,
          loss=last_loss,
          lr=lr,
          tokens_this_step=tokens_this_step,
          state=metrics_state,
          print_fn=print,
          ctx=metrics_ctx,
          loader_wait_ms=loader_wait_ms,
        )

        runtime_counts: dict[str, float] = {
          "fused_router_calls": 0.0,
          "rdep_dispatch_blockscaled_ipc_calls": 0.0,
          "rdep_dispatch_blockscaled_hybrid_calls": 0.0,
          "rdep_dispatch_bf16_ipc_calls": 0.0,
          "rdep_dispatch_bf16_hybrid_calls": 0.0,
          "nvfp4_forward_bf16_materialization_fallback_calls": 0.0,
          "nvfp4_checkpoint_direct_expert_loads": 0.0,
          "nvfp4_checkpoint_non_direct_expert_loads": 0.0,
          "eco_fused_update_calls": 0.0,
          "eco_cuda_fused_kernel_calls": 0.0,
        }
        if hasattr(model, "consume_runtime_counters"):
          model_counts = model.consume_runtime_counters()
          for key in runtime_counts:
            runtime_counts[key] += float(model_counts.get(key, 0.0))
        if fused_eco is not None and hasattr(fused_eco, "consume_runtime_counters"):
          eco_counts = fused_eco.consume_runtime_counters()
          for key in runtime_counts:
            runtime_counts[key] += float(eco_counts.get(key, 0.0))
        if getattr(metrics_ctx, "writer", None) is not None:
          try:
            runtime_items = [(f"runtime/r{rank}/{key}", value) for key, value in runtime_counts.items()]
            metrics_ctx.writer.insert_many(step=s, items=runtime_items)
          except Exception:
            logger.debug("metrics: runtime counter write failed", exc_info=True)
        if runtime_counts["nvfp4_forward_bf16_materialization_fallback_calls"] > 0:
          raise RuntimeError(
            "Forbidden NVFP4 forward BF16 materialization fallback was triggered. "
            "Aborting run to preserve precision-invariant execution."
          )
        if runtime_counts["nvfp4_checkpoint_non_direct_expert_loads"] > 0:
          raise RuntimeError(
            "Forbidden NVFP4 non-direct expert checkpoint load path was triggered. "
            "Aborting run to preserve precision-invariant execution."
          )
        if run_dtype in {"fp8", "nvfp4"} and world > 1 and runtime_rdep_mode == "hybrid":
          bf16_dispatch_calls = (
            runtime_counts["rdep_dispatch_bf16_ipc_calls"]
            + runtime_counts["rdep_dispatch_bf16_hybrid_calls"]
          )
          if bf16_dispatch_calls > 0:
            raise RuntimeError(
              "Forbidden BF16 distributed dispatch path was triggered during blockscaled training. "
              f"bf16_dispatch_calls={bf16_dispatch_calls} "
              "(ipc + hybrid). Rebuild/verify RDEP blockscaled kernels."
            )

        # Log expert load imbalance
        if s % cfg.log_every == 0 and rank == 0:
          _cv = getattr(model, 'mean_expert_load_cv', 0.0)
          if _cv > 0.5:
            print(f"[WARN] Expert load CV={_cv:.3f} (>0.5 indicates severe imbalance) "
                  f"at step {s}", file=sys.stderr)

        # Choice-based eval (fast forward-only scoring). All ranks participate.
        eval_every = int(getattr(cfg, "eval_every", 0) or 0)
        if eval_every > 0 and (s % eval_every) == 0:
          try:
            from nmoe.eval.choices import run_eval, format_results
            eval_results = run_eval(
              model,
              cfg,
              rank=rank,
              world=world,
              max_examples=int(getattr(cfg, "eval_budget_max_examples", 500)),
            )
            if rank == 0:
              logger.info("Eval step=%d %s", s, format_results(eval_results))
              if getattr(metrics_ctx, "writer", None) is not None:
                items = []
                for tname, r in eval_results.items():
                  items.append((f"eval_choices/{tname}/acc", float(r.get("acc", 0.0))))
                  items.append((f"eval_choices/{tname}/centered_acc", float(r.get("centered_acc", 0.0))))
                  items.append((f"eval_choices/{tname}/n", float(r.get("n", 0.0))))
                metrics_ctx.writer.insert_many(step=s, items=items)
          except Exception as e:
            if rank == 0:
              logger.exception("Eval failed")
          finally:
            # Prevent eval-only forwards from leaking into the next step's runtime counters.
            if hasattr(model, "consume_runtime_counters"):
              model.consume_runtime_counters()
            if fused_eco is not None and hasattr(fused_eco, "consume_runtime_counters"):
              fused_eco.consume_runtime_counters()

        # Opportunistically schedule evaluation (async or inline), if enabled
        try:
          maybe_schedule_eval(s, cfg, model, run_id, print)
        except Exception as e:
          if rank == 0:
            logger.warning("Eval scheduling failed: %s", e)

        save_checkpoint(
          checkpointer, s, tokens_seen, model, optimizer, loader, plan,
          zero2_state, cfg, rank, config_fingerprint, checkpoint_every, print
        )

    if rank == 0:
      logger.info("Training complete. %s tokens.", f"{tokens_seen:,}")

    final_loss = float(last_loss.item()) if last_loss is not None else 0.0
    results = {'final_loss': final_loss, 'tokens_seen': tokens_seen, 'steps_completed': cfg.steps}
    if exp_tracker is not None and rank == 0:
      exp_tracker.end_run(run_id, "completed", results)
    return results
  except Exception:
    if exp_tracker is not None and rank == 0:
      exp_tracker.end_run(run_id, "failed")
    raise
  finally:
    set_nan_debug_active(False)
    if 'loader' in locals() and hasattr(loader, 'close'):
      try:
        loader.close()
      except Exception:
        logger.debug("loader close failed", exc_info=True)
    checkpointer.close()
    stop_metrics(metrics_ctx)
    if exp_tracker is not None:
      exp_tracker.close()


def main():
  """Entry point. Loads config and starts training.

  Usage:
    python -m nmoe.train <config.toml> [--key=value ...]

  CLI overrides (applied after TOML):
    --dtype=fp8        Override precision (bf16, fp8, nvfp4)
    --steps=2000       Override training steps
    --batch_size=16    Override batch size
    --resume=false     Override resume behavior

  Environment overrides (lowest priority):
    NMOE_DTYPE, NMOE_STEPS, etc.
  """
  if '--help' in sys.argv or '-h' in sys.argv:
    print(__doc__)
    print(main.__doc__)
    sys.exit(0)

  if len(sys.argv) < 2:
    print("Usage: python -m nmoe.train <config.toml> [--key=value ...]", file=sys.stderr)
    sys.exit(1)

  config_path = sys.argv[1]
  if not config_path.lower().endswith(".toml"):
    print(
      f"Unsupported config format for production training: {config_path}. "
      "Only .toml is supported.",
      file=sys.stderr,
    )
    sys.exit(1)

  # Load base config from TOML.
  cfg_dict = load_config_file(config_path)

  # Apply environment variable overrides (NMOE_DTYPE, NMOE_STEPS, etc.)
  for key in ['dtype', 'steps', 'batch_size', 'seq_len', 'resume']:
    env_key = f'NMOE_{key.upper()}'
    if env_key in os.environ:
      val = os.environ[env_key]
      # Parse booleans and ints
      if val.lower() in ('true', 'false'):
        val = val.lower() == 'true'
      elif val.isdigit():
        val = int(val)
      cfg_dict[key] = val

  # Apply CLI overrides (--key=value)
  for arg in sys.argv[2:]:
    if not arg.startswith('--'):
      continue

    # Support boolean flags without '=' for common toggles.
    if '=' not in arg:
      if arg == '--resume':
        cfg_dict['resume'] = True
      elif arg == '--no-resume':
        cfg_dict['resume'] = False
      continue

    key, val = arg[2:].split('=', 1)
    # Parse booleans and ints
    if val.lower() in ('true', 'false'):
      val = val.lower() == 'true'
    elif val.lstrip('-').isdigit():
      val = int(val)
    elif val.replace('.', '', 1).lstrip('-').isdigit():
      val = float(val)
    cfg_dict[key] = val

  cfg = Config(**cfg_dict)

  # === Enhanced startup logging ===
  rank = int(os.environ.get("RANK", 0))
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  node_rank = int(os.environ.get("NODE_RANK", rank))
  world_size = int(os.environ.get('WORLD_SIZE', 1))
  host = socket.gethostname()
  print(f"[TRAIN] host={host} node_rank={node_rank} rank={rank} local_rank={local_rank} world_size={world_size}")
  master_addr = os.environ.get("MASTER_ADDR", "")
  master_port = os.environ.get("MASTER_PORT", "")
  if master_addr:
    print(f"[TRAIN] master_addr={master_addr} master_port={master_port}")
  print(f"[TRAIN] config: steps={cfg.steps} batch_size={cfg.batch_size} seq_len={cfg.seq_len}")
  print(f"[TRAIN] dtype={cfg.dtype} rdep_capacity={getattr(cfg, 'rdep_capacity', 'N/A')}")
  print(f"[TRAIN] n_routed_experts={cfg.n_routed_experts} n_activated_experts={cfg.n_activated_experts}")
  print(f"[TRAIN] gradient_accumulation_steps={cfg.gradient_accumulation_steps}")
  sys.stdout.flush()

  try:
    train(cfg)
  finally:
    runtime.finalize()


if __name__ == '__main__':
  main()
