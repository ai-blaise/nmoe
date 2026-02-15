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
import sys
import tomllib
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from nmoe.config import Config, fingerprint
from nmoe.model import Transformer
from nmoe.data.loader import build_loader
from nmoe.data.sft_loader import build_sft_loader
from nmoe.opt import build_optimizer, update_lr, step
from nmoe.checkpoint import Checkpointer, load_checkpoint, save_checkpoint
from nmoe.metrics import init_metrics, start_metrics, log_training_step, stop_metrics, register_model_timers, cuda_time
from nmoe.experiments import ExperimentTracker
from nmoe import runtime
from nmoe.eval.hooks import maybe_schedule_eval

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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

  # For SFT with EP/DP, batch_size must be divisible by dp_size (not world).
  # Example: EP=8, DP=16, world=128, batch_size=256 => 256 % 16 = 0
  sft_mode = getattr(cfg, 'sft_enabled', False)
  if sft_mode:
    ep_size = getattr(cfg, 'ep_size', 1)
    dp_size = getattr(cfg, 'dp_size', None) or max(1, world // ep_size)
    if dp_size > 1 and (cfg.batch_size % dp_size) != 0:
      raise ValueError(
        f"batch_size ({cfg.batch_size}) must be divisible by dp_size ({dp_size}) in SFT mode. "
        f"With EP={ep_size}, each DP replica gets batch_size/dp_size microbatch."
      )
    if ep_size > 1 and world % ep_size != 0:
      raise ValueError(
        f"world_size ({world}) must be divisible by ep_size ({ep_size}). "
        f"Cannot split {world} GPUs into groups of {ep_size}."
      )
  elif world > 1 and (cfg.batch_size % world) != 0:
    raise ValueError(
      f"batch_size ({cfg.batch_size}) must be divisible by world_size ({world}). "
      "Uneven per-rank microbatches break ZeRO-2 AVG semantics and exact token accounting."
    )

  if cfg.seq_len <= 0:
    raise ValueError(f"seq_len must be > 0 (got {cfg.seq_len})")
  if cfg.n_activated_experts is not None and cfg.n_routed_experts is not None:
    if cfg.n_activated_experts <= 0:
      raise ValueError(f"n_activated_experts must be > 0 (got {cfg.n_activated_experts})")
    if cfg.n_activated_experts > cfg.n_routed_experts:
      raise ValueError(
        f"n_activated_experts ({cfg.n_activated_experts}) must be <= n_routed_experts ({cfg.n_routed_experts})"
      )


def train(cfg: Config):
  """Train MoE model. One clear path: forward → loss → backward → step → log → checkpoint."""
  ep_size = getattr(cfg, 'ep_size', 1)
  rank, world = runtime.init(cfg.seed, ep_size=ep_size)
  _validate_training_config(cfg, world)
  timers_on = os.getenv('NMOE_TIMERS', '1') not in ('0', 'false', 'False')
  time_ctx = cuda_time if timers_on else (lambda _tag: nullcontext())
  nvtx_on = os.getenv('NMOE_NVTX', '0') in ('1', 'true', 'True')
  nvtx_ok = bool(nvtx_on and torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'))
  nvtx_ctx = (torch.cuda.nvtx.range if nvtx_ok else (lambda _tag: nullcontext()))

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
    loader, plan = build_loader(cfg, rank, world)

  model = Transformer(cfg).cuda()
  model.init_weights()
  model.train()

  # Enable gradient checkpointing if configured (required for NVFP4 primary weights:
  # only 1 layer's BF16 scratch activations in GPU memory at a time)
  if getattr(cfg, 'gradient_checkpointing', False):
    model.gradient_checkpointing_enable()
    if rank == 0:
      logger.info("Gradient checkpointing enabled")

  register_model_timers(model)
  optimizer, dense_groups = build_optimizer(model, cfg)
  metrics_state = init_metrics(model, cfg.seq_len)
  metrics_ctx = start_metrics(run_id=run_id, metrics_dir=cfg.metrics_dir)

  # Upload training config to W&B (rank 0 only)
  if rank == 0 and metrics_ctx.wandb_run is not None:
    import dataclasses
    try:
      metrics_ctx.wandb_run.config.update(dataclasses.asdict(cfg))
    except Exception:
      logger.debug("W&B config upload failed", exc_info=True)

  zero2_state = {}
  start_step, tokens_seen, zero2_state = load_checkpoint(checkpointer, model, optimizer, loader, plan, cfg, rank, print)

  # Eagerly allocate ZeRO-2 flat buffers and re-point dense params into them.
  # This must happen BEFORE the first forward pass: at this point only model weights
  # are on GPU, leaving ~30+ GiB free.  If deferred to the optimizer step (lazy),
  # the allocation competes with activations and gradients, causing OOM.
  if world > 1 and dense_groups:
    from nmoe import zero2
    from nmoe.opt import _get_dp_group
    zero2.eager_init(dense_groups, pg=_get_dp_group())
    if rank == 0:
      logger.info("ZeRO-2 flat buffers eagerly initialized for %d dense groups", len(dense_groups))

  # Fused backward-optimizer: create and attach after checkpoint load
  fused_eco = None
  if getattr(cfg, 'eco_fused_backward', False):
    from nmoe.eco import FusedBackwardECO
    fused_eco = FusedBackwardECO(model, cfg)
    # Set DP group for gradient AllReduce inside backward
    try:
      from nmoe.distributed.init_groups import is_nmoe_parallel_initialized, get_data_parallel_group, get_dp_size
      if is_nmoe_parallel_initialized():
        fused_eco.set_dp_group(get_data_parallel_group(), get_dp_size())
    except ImportError:
      pass
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

  last_loss: torch.Tensor | None = None
  config_fingerprint = fingerprint(cfg)
  checkpoint_every = getattr(cfg, 'checkpoint_every', 100)

  # Gradient accumulation configuration
  accum_steps = int(getattr(cfg, 'gradient_accumulation_steps', 1))
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
      for param_info in nan_weight_params[:20]:
        logger.warning(f"[NAN]   {param_info}")
      if len(nan_weight_params) > 20:
        logger.warning(f"[NAN]   ... and {len(nan_weight_params) - 20} more parameters")
    else:
      logger.info("[NAN] Model weights sanity check passed: no NaN/Inf detected")

  try:
    with nvtx_ctx('train/run'):
      for step_num in range(start_step, cfg.steps):
        lr = update_lr(optimizer, dense_groups, step_num, tokens_seen, cfg)

        # Gradient accumulation: inner loop over micro-batches
        accumulated_loss = 0.0
        t0 = time.perf_counter()

        for micro_step in range(accum_steps):
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

          with nvtx_ctx('train/fwd_total'), time_ctx('time_ms/fwd_total'):
            logits = model(inputs, cu_seqlens=cu_seqlens)

          # === NaN/Inf detection in logits (BEFORE loss computation) ===
          with torch.no_grad():
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            if nan_count > 0 or inf_count > 0:
              logits_max = logits[~torch.isnan(logits) & ~torch.isinf(logits)].abs().max().item() if (nan_count + inf_count) < logits.numel() else float('nan')
              logger.warning(f"[NAN] step={step_num} micro={micro_step}: logits contain NaN={nan_count} Inf={inf_count} (total={logits.numel()}, valid_max={logits_max:.4f})")

          with nvtx_ctx('train/loss'), time_ctx('time_ms/loss'):
            # Cast logits to FP32 before cross_entropy: BF16 softmax over vocab_size=129,280
            # causes catastrophic precision loss, producing NaN gradients in the backward pass.
            # The .float() is intentionally placed before .reshape() so the full tensor is in FP32
            # and both the forward softmax and backward gradient computation use FP32 precision.
            loss_unreduced = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), targets.reshape(-1), reduction='none')
            if loss_mask is not None:
              # SFT: use per-token loss mask from chat template (0=prompt, 1=response)
              mask = loss_mask.reshape(-1)
            else:
              # Pre-training: mask out EOS tokens
              mask = (targets != cfg.eos_token_id).reshape(-1).float()
            loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1.0)
            # Scale loss by accumulation steps so gradients are averaged
            if accum_steps > 1:
              loss = loss / accum_steps

          # === NaN/Inf detection in loss (BEFORE backward) ===
          if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[NAN] step={step_num} micro={micro_step}: loss={loss.item()} (mask_sum={mask.sum().item():.0f}, loss_unreduced_max={loss_unreduced.max().item():.4f})")

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
          with nvtx_ctx('train/bwd_total'), time_ctx('time_ms/bwd_total'):
            loss.backward()
          if fused_eco is not None:
            fused_eco.post_backward()

          accumulated_loss += loss.detach().item() * (accum_steps if accum_steps > 1 else 1)

        # End of micro-batch loop. Now do optimizer step.
        loader_wait_ms = (time.perf_counter() - t0) * 1000.0

        # === Enhanced NaN/Inf gradient detection (before clipping) ===
        nan_params = []
        total_nan_count = 0
        total_inf_count = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                nan_count = torch.isnan(p.grad).sum().item()
                inf_count = torch.isinf(p.grad).sum().item()
                if nan_count > 0 or inf_count > 0:
                    total_nan_count += nan_count
                    total_inf_count += inf_count
                    # Get gradient statistics for valid elements
                    valid_mask = ~torch.isnan(p.grad) & ~torch.isinf(p.grad)
                    valid_grads = p.grad[valid_mask]
                    grad_stats = ""
                    if valid_grads.numel() > 0:
                        grad_stats = f", valid_max={valid_grads.abs().max().item():.4e}, valid_mean={valid_grads.abs().mean().item():.4e}"
                    nan_params.append(f"{name}(nan={nan_count},inf={inf_count},numel={p.grad.numel()}{grad_stats})")
        if nan_params:
            logger.warning(f"[NAN] step={step_num}: total_nan={total_nan_count} total_inf={total_inf_count}")
            # Log first 10 affected params with details
            for param_info in nan_params[:10]:
                logger.warning(f"[NAN]   {param_info}")
            if len(nan_params) > 10:
                logger.warning(f"[NAN]   ... and {len(nan_params) - 10} more parameters with NaN/Inf gradients")

        # Gradient clipping (important for SFT stability with quantized training)
        # When fused_eco is active, expert grads are already consumed — only clip dense params.
        # Dense gradients accumulate across micro-steps (autograd adds them).
        grad_clip = getattr(cfg, 'grad_clip', 0.0)
        if grad_clip > 0:
          if fused_eco is not None:
            # Only clip dense parameters (expert grads already consumed in backward)
            dense_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if dense_params:
              torch.nn.utils.clip_grad_norm_(dense_params, grad_clip)
          else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        with nvtx_ctx('train/opt_step'), time_ctx('time_ms/opt_step'):
          step(model, optimizer, dense_groups, zero2_state, cfg, world)

        tokens_this_step = int(inputs.numel()) * accum_steps
        tokens_seen += cfg.batch_size * cfg.seq_len
        last_loss = torch.tensor(accumulated_loss / accum_steps if accum_steps > 1 else accumulated_loss)

        s = step_num + 1
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

  # Load base config from TOML
  with open(sys.argv[1], 'rb') as f:
    cfg_dict = tomllib.load(f)

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
  world_size = int(os.environ.get('WORLD_SIZE', 1))
  print(f"[TRAIN] rank={rank} local_rank={local_rank} world_size={world_size}")
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
