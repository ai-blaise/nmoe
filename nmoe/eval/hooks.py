"""
Evaluation hooks for seamless integration with the training loop.

Design goals:
- Async-capable: spawn a separate process or write a ticket for K8s.
- Deterministic: eval snapshot contains config; runner is pure function of snapshot+args.
- Minimal: no dependency on training internals beyond model/config/checkpointer.
"""
from __future__ import annotations

import os
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


import torch


def _save_eval_snapshot(snapshot_dir: Path, cfg, model: torch.nn.Module) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snapshot_dir / "eval_snapshot.pt"
    state = {
        "config": asdict(cfg),
        "model_state": model.state_dict(),
    }
    torch.save(state, str(snap_path))
    return snap_path


def maybe_schedule_eval(
    step: int,
    cfg,
    model: torch.nn.Module,
    run_id: str,
    print_fn=print,
) -> None:
    """Schedule evaluation according to cfg.*. No-ops unless eval is enabled.

    Modes:
      - inline: run synchronously on the current GPU (for smoke/dev)
      - reserved_gpu: spawn a subprocess bound to a specific GPU
      - k8s_job: write a ticket JSON; an external job picks it up
    """
    if not getattr(cfg, "eval_enabled", False):
        return

    every = int(getattr(cfg, "eval_every", 0) or 0)
    if every <= 0 or (step % every) != 0:
        return

    mode = str(getattr(cfg, "eval_mode", "inline")).lower()
    tasks = str(getattr(cfg, "eval_tasks", "core"))
    tasks_file = str(getattr(cfg, "eval_tasks_file", "configs/eval/tasks.toml"))
    budget_max_examples = int(getattr(cfg, "eval_budget_max_examples", 500))
    budget_max_time_s = int(getattr(cfg, "eval_budget_max_time_s", 300))

    # Write a minimal, self-contained snapshot for eval
    out_root = Path(getattr(cfg, "metrics_dir", "/data/metrics")) / "eval" / run_id / f"step_{step:07d}"
    snap_path = _save_eval_snapshot(out_root, cfg, model)
    print_fn(f"[eval] snapshot written: {snap_path}")

    if mode == "inline":
        # Synchronous, same GPU. Keep it brief via budgets.
        cmd = [
            "python", "-m", "nmoe.eval.runner",
            "--snapshot", str(snap_path),
            "--tasks", tasks,
            "--tasks-file", tasks_file,
            "--max-examples", str(budget_max_examples),
            "--max-time", str(budget_max_time_s),
        ]
        print_fn(f"[eval] inline: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        return

    if mode == "reserved_gpu":
        # Spawn a subprocess on a specific GPU ID; training continues.
        gpu_id = str(getattr(cfg, "eval_reserved_gpu_id", 0))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            "python", "-m", "nmoe.eval.runner",
            "--snapshot", str(snap_path),
            "--tasks", tasks,
            "--tasks-file", tasks_file,
            "--max-examples", str(budget_max_examples),
            "--max-time", str(budget_max_time_s),
        ]
        print_fn(f"[eval] reserved_gpu={gpu_id}: {' '.join(cmd)}")
        # Detach process; do not block the training loop
        subprocess.Popen(cmd, env=env)
        return

    if mode == "k8s_job":
        # Emit a ticket for an external job/cron to consume.
        ticket = {
            "run": run_id,
            "step": step,
            "snapshot": str(snap_path),
            "tasks": tasks,
            "budget": {"max_examples": budget_max_examples, "max_time": budget_max_time_s},
        }
        qdir = out_root.parent / "queue"
        qdir.mkdir(parents=True, exist_ok=True)
        tpath = qdir / f"ticket_{step:07d}.json"
        with tpath.open("w", encoding="utf-8") as f:
            json.dump(ticket, f)
        print_fn(f"[eval] queued ticket: {tpath}")
        return

    print_fn(f"[eval] warning: unknown eval_mode={mode}; skipping")
