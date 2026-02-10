"""Runtime initialization and cleanup for nmoe training.

Handles platform checks, distributed setup, seeds, and EP/DP process groups.
Seamlessly supports single GPU, single-node multi-GPU, and multi-node training.
"""
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist


def _require_b200():
  """Hard-target NVIDIA B200 (sm_100a). Off-target is not supported."""
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device required (B200, sm_100a). Off-target is not supported.")
  major, minor = torch.cuda.get_device_capability()
  if (major, minor) != (10, 0):
    raise RuntimeError(
      f"This repo targets NVIDIA B200 (sm_100a). Detected compute capability {major}.{minor}. "
      "Off-target is not supported."
    )


def _ensure_third_party_imports() -> None:
  """Ensure vendored deps are importable (container-first contract)."""
  root = Path(__file__).resolve().parents[1]
  flash_attn = root / "third_party" / "flash_attn"
  if flash_attn.exists():
    p = str(flash_attn)
    if p not in sys.path:
      sys.path.insert(0, p)


def init(seed: int = 42, ep_size: int = 1, tp_size: int = 1) -> tuple[int, int]:
  """Initialize runtime for training. Returns (rank, world).

  Handles:
  - Platform check (B200 required)
  - Seeds and TF32
  - Device assignment (LOCAL_RANK env var)
  - Distributed init (automatic for multi-GPU)
  - EP/DP process group creation (when ep_size > 1)

  Args:
    seed: Random seed for reproducibility.
    ep_size: Expert parallelism group size. Default 1 (no EP).
    tp_size: Tensor parallelism group size. Default 1 (no TP).

  Works seamlessly for:
  - Single GPU: rank=0, world=1
  - Single-node multi-GPU: torchrun sets LOCAL_RANK, init NCCL
  - Multi-node: same as single-node, world > local_world
  """
  _require_b200()
  _ensure_third_party_imports()

  # Seeds and TF32
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # Device assignment (torchrun sets LOCAL_RANK; single-process defaults to 0)
  local_rank = int(os.environ.get('LOCAL_RANK', '0'))
  torch.cuda.set_device(local_rank)

  # Distributed init (only when launched under torchrun)
  world_env = int(os.environ.get('WORLD_SIZE', '1'))
  if world_env > 1 and not dist.is_initialized():
    dist.init_process_group("nccl")

  # Get rank and world (or default to single GPU)
  rank = dist.get_rank() if dist.is_initialized() else 0
  world = dist.get_world_size() if dist.is_initialized() else 1

  # Initialize EP/TP/DP process groups when parallelism is configured.
  # This must happen after dist.init_process_group() and before model creation.
  if (ep_size > 1 or tp_size > 1) and world > 1:
    from nmoe.distributed.init_groups import init_nmoe_process_groups, is_nmoe_parallel_initialized
    if not is_nmoe_parallel_initialized():
      init_nmoe_process_groups(ep_size=ep_size, tp_size=tp_size)
      if rank == 0:
        dp_size = world // (ep_size * tp_size)
        print(f"[nmoe] Process groups: EP={ep_size}, TP={tp_size}, DP={dp_size}, world={world}")

  return rank, world


def finalize():
  """Cleanup distributed state (process groups and EP/DP groups)."""
  # Clean up nmoe process groups first (they are sub-groups of WORLD)
  try:
    from nmoe.distributed.init_groups import cleanup_process_groups, is_nmoe_parallel_initialized
    if is_nmoe_parallel_initialized():
      cleanup_process_groups()
  except Exception:
    pass

  if dist.is_initialized():
    dist.destroy_process_group()
