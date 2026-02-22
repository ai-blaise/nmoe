import logging
import os
import json
import hashlib
import pickle
import sys
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict, Set
import time
import math
import subprocess

import torch
import torch.distributed as dist

_log = logging.getLogger(__name__)

from nmoe.data.mixture import MixturePlan


# Checkpoint format versioning for backwards compatibility
# Version history:
#   1: Initial format (rd.pt + dp_rank_*.pt, run_info metadata)
#   2: Added checkpoint_version field and format versioning
#   3: Added EP sharding info (ep_size, ep_rank, expert_range) for resharding support
CHECKPOINT_FORMAT_VERSION = 3

TRACKER = "latest_checkpointed_iteration.txt"


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _ep_rank() -> int:
    """Get the EP rank for checkpoint loading.

    With EP sharding, each DP replica loads the same checkpoint shard.
    The EP rank determines which shard to load (0 to ep_size-1).

    Uses the nmoe distributed init_groups module if available, falling back
    to global rank if not (which works for EP=1 or non-EP configs).
    """
    try:
        from nmoe.distributed.init_groups import is_nmoe_parallel_initialized, get_ep_rank
        if is_nmoe_parallel_initialized():
            return get_ep_rank()
    except ImportError:
        pass

    # Fallback: global rank (works for EP=1 or when nmoe groups not initialized)
    return _rank()


def _ep_size() -> int:
    """Get EP size if nmoe process groups are initialized."""
    try:
        from nmoe.distributed.init_groups import is_nmoe_parallel_initialized, get_ep_size
        if is_nmoe_parallel_initialized():
            return max(1, int(get_ep_size()))
    except ImportError:
        pass
    return 1


def _use_local_ep_checkpoint_layout() -> bool:
    """Whether checkpoint files are stored per-node using EP-local shard IDs."""
    raw = os.getenv("NMOE_CHECKPOINT_LOCAL_EP", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _checkpoint_shard_rank() -> int:
    """Shard index used for dp_rank_XXX checkpoint file naming."""
    if _use_local_ep_checkpoint_layout():
        return _ep_rank()
    return _rank()


def _checkpoint_shard_count() -> int:
    """Number of dp_rank shards expected in a completed checkpoint iteration."""
    if _use_local_ep_checkpoint_layout():
        return _ep_size()
    return _world_size()


def _is_checkpoint_writer() -> bool:
    """Whether this rank is responsible for rd.pt + manifest/tracker writes."""
    if _use_local_ep_checkpoint_layout():
        return _ep_rank() == 0
    return _rank() == 0


def _validate_checkpoint_layout_safety() -> None:
    """Fail fast on known-unsafe layout combinations."""
    if not _use_local_ep_checkpoint_layout():
        return
    shared_raw = os.getenv("NMOE_CHECKPOINT_SHARED_FS", "").strip().lower()
    if shared_raw == "":
        raise RuntimeError(
            "NMOE_CHECKPOINT_LOCAL_EP=1 requires explicit NMOE_CHECKPOINT_SHARED_FS=0|1."
        )
    truthy = ("1", "true", "yes", "on")
    falsy = ("0", "false", "no", "off")
    if shared_raw not in truthy + falsy:
        raise RuntimeError(
            "Invalid NMOE_CHECKPOINT_SHARED_FS value. Expected one of "
            f"{truthy + falsy}, got {shared_raw!r}."
        )
    if shared_raw in truthy:
        raise RuntimeError(
            "NMOE_CHECKPOINT_LOCAL_EP=1 is incompatible with shared checkpoint storage. "
            "Use node-local checkpoint paths or disable local-EP checkpoint layout."
        )

    if _is_dist():
        ep_size = _ep_size()
        world = _world_size()
        if world > 1 and ep_size <= 1:
            raise RuntimeError(
                "NMOE_CHECKPOINT_LOCAL_EP=1 requires initialized EP groups (ep_size > 1)."
            )

        local_world_raw = os.getenv("LOCAL_WORLD_SIZE", "").strip()
        if world > 1:
            if not local_world_raw:
                raise RuntimeError(
                    "NMOE_CHECKPOINT_LOCAL_EP=1 requires LOCAL_WORLD_SIZE to be set."
                )
            try:
                local_world = int(local_world_raw)
            except ValueError as e:
                raise RuntimeError(
                    f"Invalid LOCAL_WORLD_SIZE={local_world_raw!r} for local-EP checkpoint mode"
                ) from e
            if local_world != ep_size:
                raise RuntimeError(
                    "NMOE_CHECKPOINT_LOCAL_EP=1 requires one full EP group per checkpoint "
                    f"directory. Got LOCAL_WORLD_SIZE={local_world}, ep_size={ep_size}."
                )


def iteration_dir(base: str | Path, step: int) -> str:
    return str(Path(base) / f"iter_{step:07d}")


def tracker_path(base: str | Path) -> str:
    return str(Path(base) / TRACKER)


def write_tracker(base: str | Path, step: int) -> None:
    os.makedirs(base, exist_ok=True)
    out = tracker_path(base)
    tmp = out + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(str(int(step)))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            _log.debug("fsync failed for tracker tmp file %s", tmp, exc_info=True)
    os.replace(tmp, out)
    try:
        _fsync_file(out)
        _fsync_dir(str(Path(base)))
    except Exception:
        _log.debug("fsync failed after writing tracker %s", out, exc_info=True)


def read_tracker(base: str | Path) -> int:
    try:
        with open(tracker_path(base), "r") as f:
            txt = f.read().strip()
            return int(txt)
    except FileNotFoundError:
        return -1
    except (OSError, IOError) as e:
        _log.warning("Failed to read tracker at %s: %s", tracker_path(base), e)
        return -1
    except (ValueError, TypeError) as e:
        _log.error("Corrupt tracker file at %s: %s", tracker_path(base), e)
        return -1


def _safe_load(path: str) -> Optional[dict]:
    """Best-effort torch.load for metadata reads. Logs errors instead of silencing."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except (OSError, IOError) as e:
        _log.warning("Failed to load checkpoint file %s: %s", path, e)
        return None
    except (pickle.UnpicklingError, RuntimeError) as e:
        _log.error("Corrupt checkpoint data in %s: %s", path, e)
        return None
    except Exception:
        _log.error("Unexpected error loading %s", path, exc_info=True)
        raise


def get_checkpoint_version(state: dict) -> int:
    """Get checkpoint format version from state dict.

    Returns:
        Version number (1 if not present, for legacy checkpoints).
    """
    return state.get('checkpoint_version', 1)


def validate_checkpoint_version(state: dict, max_version: int = CHECKPOINT_FORMAT_VERSION) -> None:
    """Validate that checkpoint version is compatible.

    Args:
        state: Loaded checkpoint state dict.
        max_version: Maximum supported version (default: current version).

    Raises:
        ValueError: If checkpoint version is newer than max_version.
    """
    version = get_checkpoint_version(state)
    if version > max_version:
        raise ValueError(
            f"Checkpoint format version {version} is newer than supported version {max_version}. "
            f"Please upgrade nmoe to load this checkpoint."
        )


# =============================================================================
# EP Sharding Info for Checkpoint
# =============================================================================

@dataclass
class EPShardInfo:
    """Expert Parallelism sharding information for checkpoints.

    This metadata is saved with checkpoints to enable resharding across
    different EP configurations (e.g., training with EP=4, resuming with EP=8).

    Attributes:
        ep_size: Number of EP ranks used during checkpoint creation.
        ep_rank: EP rank that created this shard (0 to ep_size-1).
        n_total_experts: Total number of experts in the model.
        n_local_experts: Number of experts in this shard.
        expert_start: First expert index in this shard (inclusive).
        expert_end: Last expert index in this shard (exclusive).
    """
    ep_size: int
    ep_rank: int
    n_total_experts: int
    n_local_experts: int
    expert_start: int
    expert_end: int

    def to_dict(self) -> Dict[str, int]:
        """Serialize to dictionary for checkpoint storage."""
        return {
            'ep_size': self.ep_size,
            'ep_rank': self.ep_rank,
            'n_total_experts': self.n_total_experts,
            'n_local_experts': self.n_local_experts,
            'expert_start': self.expert_start,
            'expert_end': self.expert_end,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'EPShardInfo':
        """Deserialize from dictionary."""
        return cls(
            ep_size=d['ep_size'],
            ep_rank=d['ep_rank'],
            n_total_experts=d['n_total_experts'],
            n_local_experts=d['n_local_experts'],
            expert_start=d['expert_start'],
            expert_end=d['expert_end'],
        )

    @classmethod
    def create(cls, ep_size: int, ep_rank: int, n_total_experts: int) -> 'EPShardInfo':
        """Create EPShardInfo from basic parameters.

        Args:
            ep_size: Total number of EP ranks.
            ep_rank: This rank's EP index.
            n_total_experts: Total number of experts.

        Returns:
            EPShardInfo with computed expert range.

        Raises:
            ValueError: If n_total_experts is not divisible by ep_size.
        """
        if n_total_experts % ep_size != 0:
            raise ValueError(
                f"n_total_experts ({n_total_experts}) must be divisible by "
                f"ep_size ({ep_size})"
            )
        n_local = n_total_experts // ep_size
        expert_start = ep_rank * n_local
        expert_end = expert_start + n_local
        return cls(
            ep_size=ep_size,
            ep_rank=ep_rank,
            n_total_experts=n_total_experts,
            n_local_experts=n_local,
            expert_start=expert_start,
            expert_end=expert_end,
        )


def get_ep_shard_info_from_checkpoint(state: dict) -> Optional[EPShardInfo]:
    """Extract EP sharding info from checkpoint state dict.

    Args:
        state: Checkpoint state dictionary.

    Returns:
        EPShardInfo if present, None for legacy checkpoints.
    """
    ep_info = state.get('ep_shard_info')
    if ep_info is None:
        return None
    return EPShardInfo.from_dict(ep_info)


def validate_ep_shard_compatibility(
    saved_info: Optional[EPShardInfo],
    current_ep_size: int,
    current_ep_rank: int,
    n_total_experts: int,
) -> Tuple[bool, str]:
    """Check if saved checkpoint EP sharding is compatible with current config.

    Args:
        saved_info: EP info from checkpoint (None for legacy).
        current_ep_size: Current EP size.
        current_ep_rank: Current EP rank.
        n_total_experts: Current total experts.

    Returns:
        (compatible, message) tuple. compatible=True if direct load is OK.
    """
    if saved_info is None:
        # Legacy checkpoint - assume world=1 (all experts in single shard)
        return (current_ep_size == 1,
                "Legacy checkpoint has no EP info. Can only load with EP=1.")

    if saved_info.n_total_experts != n_total_experts:
        return (False,
                f"Expert count mismatch: checkpoint has {saved_info.n_total_experts}, "
                f"current model has {n_total_experts}")

    if saved_info.ep_size != current_ep_size:
        return (False,
                f"EP size mismatch: checkpoint has EP={saved_info.ep_size}, "
                f"current config has EP={current_ep_size}. Resharding required.")

    if saved_info.ep_rank != current_ep_rank:
        return (False,
                f"EP rank mismatch: checkpoint shard is for EP rank {saved_info.ep_rank}, "
                f"but current rank is {current_ep_rank}.")

    return (True, "EP sharding compatible")


@dataclass
class AsyncTask:
    path: str
    base: str
    step: int
    state: dict[str, Any]


def _materialize_to_cpu_sync(state: dict[str, Any]) -> dict[str, Any]:
    """Materialize tensors in a nested state (dict/list/tuple) onto pinned CPU.

    Mirrors the async path but without spawning a thread; uses a dedicated CUDA
    stream when available. Non-tensor leaves are copied by reference.
    """
    @torch.no_grad()
    def _copy(obj: Any, ckpt_stream: Optional[torch.cuda.Stream]) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                dst = torch.empty_like(obj, device='cpu', pin_memory=False)
                if ckpt_stream is not None:
                    with torch.cuda.stream(ckpt_stream):
                        dst.copy_(obj, non_blocking=True)
                else:
                    dst.copy_(obj)
                return dst
            return obj
        if isinstance(obj, dict):
            return {k: _copy(v, ckpt_stream) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_copy(v, ckpt_stream) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_copy(v, ckpt_stream) for v in obj)
        return obj

    # Avoid creating a new CUDA stream per checkpoint. Stream creation can
    # allocate device resources and fail under tight memory, even though the
    # checkpoint copy itself is host-side.
    out = _copy(state, None)
    return out


def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rd_path(base: str | Path, step: int) -> str:
    return os.path.join(iteration_dir(base, step), 'rd.pt')


def _dp_path(base: str | Path, step: int, rank: int) -> str:
    return os.path.join(iteration_dir(base, step), f'dp_rank_{rank:03d}.pt')


def _read_git_sha_from_rd(base: str | Path, step: int) -> str:
    rd = _safe_load(_rd_path(base, step))
    if not rd:
        return 'unknown'
    info = rd.get('run_info', {}) if isinstance(rd, dict) else {}
    return str(info.get('git_sha', 'unknown'))


def _read_checkpoint_layout_from_rd(base: str | Path, step: int) -> tuple[int, int]:
    """Return (checkpoint_dp_count, world) from rd metadata."""
    rd = _safe_load(_rd_path(base, step))
    if not rd:
        return (-1, -1)
    info = rd.get('run_info', {}) if isinstance(rd, dict) else {}
    try:
        world = int(info.get('world', -1))
        checkpoint_dp_count = info.get('checkpoint_dp_count')
        if checkpoint_dp_count is not None:
            return (int(checkpoint_dp_count), world)
        return (world, world)
    except (ValueError, TypeError) as e:
        _log.warning("Invalid checkpoint dp_count metadata at step %d: %s", step, e)
        return (-1, -1)


def _read_checkpoint_layout_mode_from_rd(base: str | Path, step: int) -> Optional[bool]:
    """Read explicit checkpoint layout mode if present.

    Returns:
        True for local-EP layout, False for global-rank layout, None if unknown.
    """
    rd = _safe_load(_rd_path(base, step))
    if not rd:
        return None
    info = rd.get('run_info', {}) if isinstance(rd, dict) else {}
    mode = str(info.get('checkpoint_layout', '')).strip().lower()
    if mode in ('local_ep', 'ep_local', 'local'):
        return True
    if mode in ('global_rank', 'global', 'world'):
        return False
    return None


def _read_dp_count_from_rd(base: str | Path, step: int) -> int:
    dp_count, _ = _read_checkpoint_layout_from_rd(base, step)
    return dp_count


def _checkpoint_uses_local_layout(base: str | Path, step: int) -> Optional[bool]:
    """Infer checkpoint shard layout for a saved step.

    Returns:
        True:  EP-local shard layout (dp_count != world)
        False: global-rank shard layout (dp_count == world)
        None:  unknown (missing/invalid metadata)
    """
    explicit_layout = _read_checkpoint_layout_mode_from_rd(base, step)
    if explicit_layout is not None:
        return explicit_layout

    dp_count, world = _read_checkpoint_layout_from_rd(base, step)
    if dp_count <= 0:
        return None
    if world > 0 and dp_count == world:
        # Ambiguous legacy metadata case: DP=1 local-EP checkpoints can also
        # have dp_count==world. If current run is explicitly local-EP and EP
        # size matches world, treat it as local-EP.
        if _use_local_ep_checkpoint_layout() and _ep_size() == world:
            return True
        return False
    return True


def _shard_rank_for_checkpoint_step(base: str | Path, step: int) -> int:
    """Pick the correct shard rank for a specific checkpoint step layout."""
    local_layout = _checkpoint_uses_local_layout(base, step)
    if local_layout is None:
        return _checkpoint_shard_rank()
    return _ep_rank() if local_layout else _rank()


def _all_dp_present(base: str | Path, step: int, dp_count: int) -> bool:
    if dp_count <= 0:
        return False
    it = iteration_dir(base, step)
    if not os.path.isdir(it):
        return False
    for r in range(dp_count):
        if not os.path.exists(_dp_path(base, step, r)):
            return False
    return True


def _write_manifest(base: str | Path, step: int, dp_count: int) -> None:
    it_dir = iteration_dir(base, step)
    os.makedirs(it_dir, exist_ok=True)
    rd = _rd_path(base, step)
    files = [rd] + [_dp_path(base, step, r) for r in range(dp_count)]
    bytes_total = 0
    for p in files:
        try:
            bytes_total += os.path.getsize(p)
        except Exception:
            _log.debug("Could not stat file %s for manifest byte count", p, exc_info=True)
    git_sha = _read_git_sha_from_rd(base, step)
    manifest = {
        'step': int(step),
        'world': int(dp_count),
        'dp_count': int(dp_count),
        'bytes_total': int(bytes_total),
        'git_sha': git_sha,
        'files': [os.path.basename(p) for p in files],
    }
    tmp = os.path.join(it_dir, 'manifest.json.tmp')
    out = os.path.join(it_dir, 'manifest.json')
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    _fsync_file(out)
    _fsync_dir(it_dir)


def try_finalize_step(base: str | Path, step: int, dp_count: Optional[int] = None) -> bool:
    """Writer helper: finalize a step when rd.pt and all dp shards are present."""
    if not _is_checkpoint_writer():
        return False
    try:
        shard_count = int(dp_count) if dp_count is not None else _read_dp_count_from_rd(base, step)
        if shard_count <= 0:
            return False
        # Require rd.pt present
        if not os.path.exists(_rd_path(base, step)):
            return False
        if not _all_dp_present(base, step, shard_count):
            return False
        _write_manifest(base, step, shard_count)
        write_tracker(base, step)
        return True
    except Exception:
        _log.error("Failed to finalize checkpoint step=%d at %s", step, base, exc_info=True)
        return False


class _AsyncSaver(threading.Thread):
    def __init__(self, max_queue:int = 1, on_saved: Optional[Callable[[int, float, int], None]] = None) -> None:
        super().__init__(daemon=True)
        self.q: "queue.Queue[AsyncTask | None]" = queue.Queue(maxsize=max(1, int(max_queue)))
        self._on_saved = on_saved
        self._error: Exception | None = None
        self.start()

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            assert isinstance(item, AsyncTask)
            try:
                bytes_written, ms = self._atomic_save(item.path, item.state)
                # Do not flip tracker here; finalize step will update when complete
                if self._on_saved is not None:
                    try:
                        self._on_saved(bytes_written, ms, item.step)
                    except Exception:
                        _log.debug("on_saved callback failed for step=%d", item.step, exc_info=True)
            except Exception as e:
                if self._error is None:
                    self._error = e
                # Fail loud, but do not crash training from a background thread.
                try:
                    sys.stderr.write(f"[ckpt] async save failed step={item.step} path={item.path}: {e}\n")
                    sys.stderr.flush()
                except Exception:
                    _log.debug("Failed to write async save error to stderr", exc_info=True)
            finally:
                self.q.task_done()

    def submit(self, path: str, base: str, step: int, state: dict[str, Any]) -> None:
        if self._error is not None:
            raise RuntimeError(f"checkpoint async saver previously failed: {self._error}") from self._error
        # Reliability invariant: enqueue CPU materialized state only. The background thread must
        # never hold references to live CUDA tensors from the training step.
        cpu_state = _materialize_to_cpu_sync(state)
        try:
            self.q.put(AsyncTask(path, base, step, cpu_state), block=True, timeout=60.0)
        except queue.Full as e:
            raise RuntimeError(
                f"checkpoint async queue full (step={step}, path={path}). "
                "This indicates checkpoint I/O cannot keep up; refusing to drop checkpoints."
            ) from e

    def close(self) -> None:
        # Signal thread to exit and wait for a clean shutdown to avoid interpreter
        # teardown while the background thread still references Python objects.
        try:
            self.q.put(None)
        except Exception:
            _log.debug("Failed to enqueue shutdown sentinel for async saver", exc_info=True)
            return
        try:
            self.join(timeout=60.0)
        except Exception:
            _log.debug("Failed to join async saver thread during close", exc_info=True)

    def wait(self) -> None:
        # Wait until all queued tasks are processed
        self.q.join()
        if self._error is not None:
            raise RuntimeError(f"checkpoint async saver failed: {self._error}") from self._error

    def _atomic_save(self, path: str, state: dict[str, Any]) -> tuple[int, float]:
        t0 = time.perf_counter()
        tmp = path + ".tmp"
        base = os.path.dirname(path)
        os.makedirs(base, exist_ok=True)
        # Write temp file first
        torch.save(state, tmp)
        # Robust finalize: best-effort atomic replace; fall back to direct save on failure
        try:
            os.replace(tmp, path)
            try:
                _fsync_file(path)
                _fsync_dir(base)
            except Exception:
                _log.debug("fsync failed after atomic save of %s", path, exc_info=True)
        except FileNotFoundError:
            # Temp may not be visible yet on some backends; try a direct save as a fallback
            try:
                # Ensure directory still exists in fallback path
                os.makedirs(base, exist_ok=True)
                torch.save(state, path)
                try:
                    _fsync_file(path)
                    _fsync_dir(base)
                except Exception:
                    _log.debug("fsync failed in fallback save path for %s", path, exc_info=True)
            finally:
                # Cleanup temp if present
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    _log.debug("Failed to clean up temp file %s", tmp, exc_info=True)
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            size = os.path.getsize(path)
        except Exception:
            _log.debug("Could not stat saved checkpoint %s", path, exc_info=True)
            size = 0
        return size, ms


class _Purger(threading.Thread):
    def __init__(self, base: str, keep_last: int) -> None:
        super().__init__(daemon=True)
        self.base = base
        self.keep_last = keep_last
        self._req: "queue.Queue[None]" = queue.Queue()
        self.start()

    def run(self) -> None:
        while True:
            self._req.get()
            try:
                self._purge_once()
            finally:
                self._req.task_done()

    def trigger(self) -> None:
        # Collapse multiple triggers
        if self._req.empty():
            self._req.put(None)

    def _purge_once(self) -> None:
        if self.keep_last <= 0:
            return
        base = Path(self.base)
        if not base.exists():
            return
        iters = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("iter_")])
        if len(iters) <= self.keep_last:
            return
        for p in iters[:-self.keep_last]:
            try:
                # Remove directory tree for this iteration
                for sub in p.glob("**/*"):
                    try:
                        sub.unlink()
                    except IsADirectoryError:
                        pass
                for sub in sorted(p.glob("**/*"), reverse=True):
                    try:
                        sub.rmdir()
                    except Exception:
                        _log.debug("Failed to remove subdirectory %s during purge", sub, exc_info=True)
                p.rmdir()
            except Exception:
                _log.debug("Best-effort purge failed for %s", p, exc_info=True)


class Checkpointer:
    """Megatron/TorchTitan-style checkpoint manager (minimal).

    - Per-step directory: base/iter_0000123/
    - Rank-local files:   dp_rank_000.pt (per rank to capture sharded experts)
    - Tracker file:       latest_checkpointed_iteration.txt (integer step)
    - Keep-last rotation: background purger
    - Optional async rank-local torch.save via a background thread
    """

    def __init__(self, base: str, keep_last: int = 5, async_io: bool = False, async_max_queue:int = 1) -> None:
        # Normalize base to absolute to avoid CWD-related surprises in background threads
        _validate_checkpoint_layout_safety()
        self.base = str(Path(base).absolute())
        self.keep_last = keep_last
        self.async_io = async_io
        self._purger = _Purger(self.base, keep_last) if keep_last > 0 else None
        # metrics
        self.last_bytes: int = 0
        self.last_ms: float = 0.0
        self._last_requested_step: int = -1
        def _on_saved(bytes_written:int, ms:float, step:int):
            self.last_bytes = bytes_written
            self.last_ms = ms
            if self._purger:
                self._purger.trigger()
            # Writer rank opportunistically attempts to finalize this step.
            try:
                if _is_checkpoint_writer():
                    try_finalize_step(self.base, step, dp_count=_checkpoint_shard_count())
            except Exception:
                _log.debug("Opportunistic finalize failed for step=%d", step, exc_info=True)
        # Ensure capacity for rd.pt + dp_rank_XXX.pt without dropping.
        max_q = max(2, async_max_queue)
        self._saver = _AsyncSaver(max_queue=max_q, on_saved=_on_saved) if async_io else None

    def _raise_if_async_failed(self) -> None:
        if self._saver is None:
            return
        # wait() raises, but we also fail fast on submit paths.
        if getattr(self._saver, "_error", None) is not None:
            err = self._saver._error  # type: ignore[attr-defined]
            raise RuntimeError(f"checkpoint async saver failed: {err}") from err

    def save_rank_local(self, step: int, state: dict[str, Any]) -> str:
        self._raise_if_async_failed()
        self._last_requested_step = max(self._last_requested_step, int(step))
        it_dir = iteration_dir(self.base, step)
        os.makedirs(it_dir, exist_ok=True)
        path = os.path.join(it_dir, f"dp_rank_{_checkpoint_shard_rank():03d}.pt")

        if self._saver is not None:
            self._saver.submit(path, self.base, step, state)
        else:
            # Synchronous path: materialize and save inline (rare / tests)
            cpu_state = _materialize_to_cpu_sync(state)
            tmp = path + ".tmp"
            torch.save(cpu_state, tmp)
            os.replace(tmp, path)
            # Finalize may flip tracker if complete
            if _is_checkpoint_writer():
                try_finalize_step(self.base, step, dp_count=_checkpoint_shard_count())
            if self._purger is not None:
                self._purger.trigger()

        return path

    def save_dense(self, step: int, state: dict[str, Any]) -> str:
        """Save replicated (dense/router) parameters once per checkpoint writer."""
        self._raise_if_async_failed()
        self._last_requested_step = max(self._last_requested_step, int(step))
        it_dir = iteration_dir(self.base, step)
        os.makedirs(it_dir, exist_ok=True)
        path = os.path.join(it_dir, "rd.pt")

        if self._saver is not None:
            self._saver.submit(path, self.base, step, state)
        else:
            # Synchronous path
            cpu_state = _materialize_to_cpu_sync(state)
            tmp = path + ".tmp"
            torch.save(cpu_state, tmp)
            os.replace(tmp, path)
            if _is_checkpoint_writer():
                try_finalize_step(self.base, step, dp_count=_checkpoint_shard_count())
            if self._purger is not None:
                self._purger.trigger()

        return path

    def find_latest(self) -> tuple[int, str] | tuple[int, None]:
        """Find the latest checkpoint for this rank.

        Uses the active checkpoint shard layout:
        - default: global-rank dp shards
        - local EP mode (NMOE_CHECKPOINT_LOCAL_EP=1): EP-rank shards (0..ep_size-1)
        """
        def _candidate(step: int) -> str:
            shard_rank = _shard_rank_for_checkpoint_step(self.base, step)
            return os.path.join(iteration_dir(self.base, step), f"dp_rank_{shard_rank:03d}.pt")

        step = read_tracker(self.base)
        if step > 0:
            it = iteration_dir(self.base, step)
            if os.path.exists(os.path.join(it, "manifest.json")) and os.path.exists(_candidate(step)):
                return step, _candidate(step)

        base = Path(self.base)
        if not base.exists():
            return -1, None
        iters = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("iter_")], reverse=True)
        for p in iters:
            try:
                s = int(p.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if not (p / "manifest.json").exists():
                continue
            shard_rank = _shard_rank_for_checkpoint_step(self.base, s)
            path = str(p / f"dp_rank_{shard_rank:03d}.pt")
            if os.path.exists(path):
                return s, path
        return -1, None

    def path_for_step(self, step: int) -> Optional[str]:
        """Return local shard path for a known checkpoint step, if present."""
        if step < 0:
            return None
        shard_rank = _shard_rank_for_checkpoint_step(self.base, step)
        candidate = os.path.join(iteration_dir(self.base, step), f"dp_rank_{shard_rank:03d}.pt")
        if os.path.exists(candidate):
            return candidate
        return None

    def try_finalize(self, step: int) -> bool:
        """Attempt to finalize checkpoint step (writer rank only)."""
        if not _is_checkpoint_writer():
            return False
        return try_finalize_step(self.base, step, dp_count=_checkpoint_shard_count())

    def recommend_interval(self, step_time_ms: float, safety: float = 1.3) -> int:
        """Compute save_every given last measured save time and step time.
        Returns at least 1.
        """
        if self.last_ms <= 0 or step_time_ms <= 0:
            return 1
        n = math.ceil(safety * (self.last_ms / step_time_ms))
        return max(1, int(n))

    def close(self) -> None:
        if self._saver is not None:
            # Wait for all pending saves to complete before closing
            self._saver.wait()
            if _is_checkpoint_writer() and self._last_requested_step > 0:
                try:
                    try_finalize_step(
                        self.base,
                        int(self._last_requested_step),
                        dp_count=_checkpoint_shard_count(),
                    )
                except Exception:
                    _log.error(
                        "Failed to finalize step=%d during checkpointer close",
                        self._last_requested_step, exc_info=True,
                    )
            self._saver.close()


"""Split‑format checkpointing utilities (no legacy paths)."""


# =============================================================================
# NVFP4 Compressed-Tensors Dequantization (for imported checkpoints)
# =============================================================================
# The HuggingFace NVFP4 compressed_tensors format stores each quantized weight
# as a triplet:
#   - weight_packed:       [*, out_features, in_features // 2]  uint8  (2 E2M1 nibbles per byte)
#   - weight_scale:        [*, out_features, in_features // 16] float8_e4m3fn  (group_size=16)
#   - weight_global_scale: [*, 1]                               float32
#
# Dequantization formula:
#   bf16_weight = unpack_e2m1(weight_packed) * weight_scale * weight_global_scale
#
# E2M1 nibble encoding (4 bits: 1 sign + 2 exponent + 1 mantissa):
#   Positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
#   nibble -> value lookup table (unsigned):
#     0b0000=0, 0b0001=0.5, 0b0010=1.0, 0b0011=1.5,
#     0b0100=2.0, 0b0101=3.0, 0b0110=4.0, 0b0111=6.0
#   High bit is sign: 0b1xxx = negative

# E2M1 unsigned lookup table (nibble value 0-7 -> float)
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _unpack_e2m1_to_float(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed NVFP4 E2M1 nibbles to float32.

    Each uint8 byte contains 2 FP4 values:
      - Low nibble  (bits 0-3): first value
      - High nibble (bits 4-7): second value

    Within each nibble: bit 3 is sign, bits 0-2 are unsigned magnitude index.

    Args:
        packed: [..., N] uint8 tensor with 2 FP4 values per byte.

    Returns:
        [..., 2*N] float32 tensor with dequantized values.
    """
    lut = _E2M1_LUT.to(packed.device)

    # Extract nibbles
    lo = (packed & 0x0F).to(torch.int32)  # low nibble: first value
    hi = ((packed >> 4) & 0x0F).to(torch.int32)  # high nibble: second value

    # Separate sign (bit 3) and magnitude (bits 0-2)
    lo_sign = ((lo >> 3) & 1).float() * (-2.0) + 1.0  # 0 -> +1, 1 -> -1
    lo_mag = lo & 0x07
    hi_sign = ((hi >> 3) & 1).float() * (-2.0) + 1.0
    hi_mag = hi & 0x07

    # Look up magnitude values
    lo_val = lut[lo_mag.long()] * lo_sign
    hi_val = lut[hi_mag.long()] * hi_sign

    # Interleave: [lo_0, hi_0, lo_1, hi_1, ...]
    out_shape = list(packed.shape)
    out_shape[-1] *= 2
    out = torch.empty(out_shape, dtype=torch.float32, device=packed.device)
    out[..., 0::2] = lo_val
    out[..., 1::2] = hi_val

    return out


def dequantize_compressed_tensors_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Dequantize an NVFP4 compressed_tensors triplet to BF16.

    Formula (from compressed_tensors library):
        effective_scale = weight_scale / weight_global_scale
        bf16 = unpack_e2m1(packed) * effective_scale

    The global_scale is a divisor of the per-group scale (not a multiplier).
    This matches the compressed_tensors _dequantize() implementation:
        scale = scale / global_scale
        dequant_value = x_q * scale

    The scale is per-group (group_size elements share one E4M3 scale factor).
    The global_scale is a single FP32 scalar per tensor.

    Args:
        weight_packed: [..., out_dim, in_dim // 2] uint8 (2 FP4 per byte).
        weight_scale:  [..., out_dim, in_dim // group_size] float8_e4m3fn.
        weight_global_scale: [..., 1] float32.
        group_size: Number of elements per scale group (default 16).

    Returns:
        [..., out_dim, in_dim] bfloat16 weight tensor.
    """
    # Step 1: Unpack E2M1 nibbles to float32
    # packed shape: [..., out_dim, in_dim//2]
    # unpacked shape: [..., out_dim, in_dim]
    unpacked = _unpack_e2m1_to_float(weight_packed)

    # Step 2: Compute effective scale = weight_scale / weight_global_scale
    # weight_scale is float8_e4m3fn — convert to float32
    scale_f32 = weight_scale.float()  # [..., out_dim, in_dim // group_size]

    # Apply global scale as divisor (compressed_tensors convention)
    global_scale = weight_global_scale.float()
    # Broadcast global_scale to match scale dimensions
    while global_scale.ndim < scale_f32.ndim:
        global_scale = global_scale.unsqueeze(-1)
    effective_scale = scale_f32 / global_scale.clamp(min=1e-10)

    # Step 3: Apply per-group effective scales
    # Reshape unpacked into groups: [..., out_dim, n_groups, group_size]
    orig_shape = unpacked.shape
    n_groups = effective_scale.shape[-1]
    grouped = unpacked.reshape(*orig_shape[:-1], n_groups, group_size)
    # effective_scale: [..., out_dim, n_groups] -> [..., out_dim, n_groups, 1]
    scaled = grouped * effective_scale.unsqueeze(-1)
    # Reshape back: [..., out_dim, in_dim]
    dequantized = scaled.reshape(orig_shape)

    return dequantized.to(torch.bfloat16)


def dequantize_nvfp4_state_dict(
    state_dict: Dict[str, torch.Tensor],
    group_size: int = 16,
    print_fn=print,
) -> Dict[str, torch.Tensor]:
    """Dequantize all NVFP4 triplets in a state dict to BF16 parameters.

    DEPRECATED: CPU fallback path. Use dequantize_nvfp4_to_model_gpu() for
    GPU-native dequantization that avoids CPU float32 intermediates.

    Scans for keys ending in '.weight_packed', finds matching '.weight_scale'
    and '.weight_global_scale', dequantizes the triplet, and replaces with
    a single BF16 tensor under the base key (with '.weight' suffix for Linear
    layers, or without suffix for Parameter-style expert weights).

    Args:
        state_dict: State dict potentially containing NVFP4 triplets.
        group_size: NVFP4 group size (default 16).
        print_fn: Logging function.

    Returns:
        New state dict with NVFP4 triplets replaced by BF16 tensors.
    """
    # Find all weight_packed keys
    packed_keys = [k for k in state_dict if k.endswith('.weight_packed')]
    if not packed_keys:
        return state_dict  # No NVFP4 data

    result = {}
    consumed = set()
    dequant_count = 0

    for pk in packed_keys:
        base = pk[:-len('.weight_packed')]
        sk = base + '.weight_scale'
        gk = base + '.weight_global_scale'

        if sk not in state_dict or gk not in state_dict:
            print_fn(f"[nvfp4] WARNING: Incomplete triplet for {base}, "
                     f"missing scale={sk not in state_dict} global_scale={gk not in state_dict}")
            # Keep original keys
            result[pk] = state_dict[pk]
            continue

        # Dequantize
        bf16_weight = dequantize_compressed_tensors_nvfp4(
            state_dict[pk], state_dict[sk], state_dict[gk], group_size,
        )

        # Determine output key and whether transpose is needed:
        base_parts = base.split('.')
        last_part = base_parts[-1] if base_parts else ''

        if last_part in ('W1', 'W3', 'W2'):
            out_key = base
            if bf16_weight.ndim >= 2:
                bf16_weight = bf16_weight.transpose(-1, -2).contiguous()
        else:
            out_key = base + '.weight'

        result[out_key] = bf16_weight
        consumed.add(pk)
        consumed.add(sk)
        consumed.add(gk)
        dequant_count += 1

    # Copy non-triplet keys
    for k, v in state_dict.items():
        if k not in consumed and k not in result:
            result[k] = v

    if dequant_count > 0:
        print_fn(f"[nvfp4] Dequantized {dequant_count} NVFP4 triplets to BF16 (CPU fallback)")

    return result


def dequantize_nvfp4_state_dict_gpu(
    state_dict: Dict[str, torch.Tensor],
    group_size: int = 16,
    print_fn=print,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Dequantize all NVFP4 triplets in a state dict to BF16 on GPU.

    Like dequantize_nvfp4_state_dict but performs dequantization on GPU to avoid
    CPU float32 intermediates.  Returns a state dict with BF16 tensors (on GPU)
    suitable for resharding or direct load_state_dict.

    This is used for resharding paths where dequantize_nvfp4_to_model_gpu()
    cannot be used because the caller needs BF16 tensors in a dict for
    expert slicing before loading into the model.

    Args:
        state_dict: State dict potentially containing NVFP4 triplets (CPU tensors).
        group_size: NVFP4 group size (default 16).
        print_fn: Logging function.
        device: GPU device to use for dequantization. If None, uses current CUDA device.

    Returns:
        New state dict with NVFP4 triplets replaced by BF16 tensors (on GPU).
    """
    if device is None:
        device = torch.device('cuda', torch.cuda.current_device())

    packed_keys = [k for k in state_dict if k.endswith('.weight_packed')]
    if not packed_keys:
        return state_dict  # No NVFP4 data

    result = {}
    consumed = set()
    dequant_count = 0

    for pk in packed_keys:
        base = pk[:-len('.weight_packed')]
        sk = base + '.weight_scale'
        gk = base + '.weight_global_scale'

        if sk not in state_dict or gk not in state_dict:
            print_fn(f"[nvfp4] WARNING: Incomplete triplet for {base}, "
                     f"missing scale={sk not in state_dict} global_scale={gk not in state_dict}")
            result[pk] = state_dict[pk]
            continue

        # Move triplet to GPU for dequant (small: NVFP4 is 4x compressed)
        packed_gpu = state_dict[pk].to(device=device, non_blocking=True)
        scale_gpu = state_dict[sk].to(device=device, non_blocking=True)
        gs_gpu = state_dict[gk].to(device=device, non_blocking=True)

        # Dequantize on GPU
        bf16_weight = dequantize_compressed_tensors_nvfp4(
            packed_gpu, scale_gpu, gs_gpu, group_size,
        )

        # Free GPU temporaries for the compressed inputs
        del packed_gpu, scale_gpu, gs_gpu

        # Determine output key and whether transpose is needed
        base_parts = base.split('.')
        last_part = base_parts[-1] if base_parts else ''

        if last_part in ('W1', 'W3', 'W2'):
            out_key = base
            if bf16_weight.ndim >= 2:
                bf16_weight = bf16_weight.transpose(-1, -2).contiguous()
        else:
            out_key = base + '.weight'

        result[out_key] = bf16_weight
        consumed.add(pk)
        consumed.add(sk)
        consumed.add(gk)
        dequant_count += 1

    # Copy non-triplet keys
    for k, v in state_dict.items():
        if k not in consumed and k not in result:
            result[k] = v

    if dequant_count > 0:
        print_fn(f"[nvfp4] Dequantized {dequant_count} NVFP4 triplets to BF16 (GPU)")

    return result


def dequantize_nvfp4_to_model_gpu(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    group_size: int = 16,
    print_fn=print,
) -> Dict[str, torch.Tensor]:
    """Dequantize NVFP4 triplets directly on GPU via CUDA kernel.

    Instead of creating float32 intermediates on CPU, this function:
    1. Moves each NVFP4 triplet (packed, scale, global_scale) to GPU
    2. Runs ct_nvfp4_to_bf16 CUDA kernel to dequant in registers
    3. Writes BF16 result directly to the model parameter on GPU
    4. Returns remaining non-NVFP4 keys for standard load_state_dict

    Zero CPU float32 intermediates. All arithmetic happens in GPU registers.

    Args:
        state_dict: State dict with NVFP4 triplets (from torch.load on CPU).
        model: Model with parameters already on GPU (after .cuda()).
        group_size: NVFP4 group size (default 16).
        print_fn: Logging function.

    Returns:
        State dict with NVFP4 triplets removed (only non-NVFP4 keys remain).
        These should be loaded via model.load_state_dict(result, strict=False).
    """
    from nmoe.csrc import rdep as _rdep_ext
    if not hasattr(_rdep_ext, 'ct_nvfp4_to_bf16'):
        raise RuntimeError(
            "[nvfp4] ct_nvfp4_to_bf16 CUDA kernel not available in rdep extension. "
            "Rebuild nmoe with NMOE_ENABLE_PTX_E2M1=1 and sm_100a target. "
            "CPU dequant fallback has been removed."
        )

    packed_keys = [k for k in state_dict if k.endswith('.weight_packed')]
    if not packed_keys:
        return state_dict

    # Build name → parameter mapping for direct GPU writes
    param_map = {n: p for n, p in model.named_parameters()}
    # Also include buffers (some weights might be registered as buffers)
    for n, b in model.named_buffers():
        if n not in param_map:
            param_map[n] = b

    result = {}
    consumed = set()
    gpu_count = 0
    stream = torch.cuda.current_stream()

    for pk in packed_keys:
        base = pk[:-len('.weight_packed')]
        sk = base + '.weight_scale'
        gk = base + '.weight_global_scale'

        if sk not in state_dict or gk not in state_dict:
            print_fn(f"[nvfp4] WARNING: Incomplete triplet for {base}")
            result[pk] = state_dict[pk]
            continue

        packed_cpu = state_dict[pk]    # [..., out_dim, in_dim//2] uint8
        scale_cpu = state_dict[sk]     # [..., out_dim, n_groups] float8_e4m3fn
        gs_cpu = state_dict[gk]        # scalar or [...] float32

        # Determine output key and transpose requirement
        base_parts = base.split('.')
        last_part = base_parts[-1] if base_parts else ''

        if last_part in ('W1', 'W3', 'W2'):
            out_key = base
            need_transpose = True
        else:
            out_key = base + '.weight'
            need_transpose = False

        # Get target device from model parameter
        target_param = param_map.get(out_key)
        if target_param is None:
            # Parameter not found in model — skip (might be pruned or renamed)
            print_fn(f"[nvfp4] WARNING: No model parameter for {out_key}, skipping GPU dequant")
            result[pk] = state_dict[pk]
            result[sk] = state_dict[sk]
            result[gk] = state_dict[gk]
            continue

        device = target_param.device
        if not device.type == 'cuda':
            raise RuntimeError(
                f"[nvfp4] Model parameter {out_key} is on {device}, not CUDA. "
                f"Model must be on GPU before loading NVFP4 checkpoint. "
                f"CPU dequant fallback has been removed."
            )

        # Move triplet to GPU (small: NVFP4 is 4x compressed)
        packed_gpu = packed_cpu.to(device=device, non_blocking=True)
        scale_gpu = scale_cpu.to(device=device, non_blocking=True)
        gs_gpu = gs_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

        # Determine dimensions
        # packed shape: [..., M, K/2] where M = out_dim, K = in_dim
        M = packed_gpu.shape[-2] if packed_gpu.ndim >= 2 else packed_gpu.shape[0]
        K = packed_gpu.shape[-1] * 2  # Each byte holds 2 elements

        # For batched tensors (e.g. expert weights [E, M, K/2])
        if packed_gpu.ndim == 3:
            E = packed_gpu.shape[0]
            total_M = E * M
            # Flatten to [E*M, K/2] for the kernel
            packed_flat = packed_gpu.reshape(total_M, K // 2).contiguous()
            n_groups = K // group_size
            scale_flat = scale_gpu.view(torch.uint8).reshape(total_M, n_groups).contiguous()
            # Per-expert global scale
            gs_flat = gs_gpu.reshape(E).contiguous()
            gs_stride = 1
            expert_rows = M
        elif packed_gpu.ndim == 2:
            total_M = M
            packed_flat = packed_gpu.contiguous()
            n_groups = K // group_size
            scale_flat = scale_gpu.view(torch.uint8).reshape(total_M, n_groups).contiguous()
            gs_flat = gs_gpu.reshape(1).contiguous()
            gs_stride = 0
            expert_rows = 0
        else:
            raise RuntimeError(
                f"[nvfp4] Unexpected tensor shape for {out_key}: packed.ndim={packed_gpu.ndim}. "
                f"Expected 2D [M, K/2] or 3D [E, M, K/2]. CPU dequant fallback has been removed."
            )

        max_transpose_mode = int(getattr(_rdep_ext, "ct_nvfp4_to_bf16_max_transpose_mode", 1))
        if need_transpose and packed_gpu.ndim == 3 and max_transpose_mode < 2:
            raise RuntimeError(
                f"[nvfp4] {out_key}: ct_nvfp4_to_bf16 transpose mode=2 is required for production "
                "no-fallback path. Rebuild nmoe/csrc extension."
            )

        # Allocate BF16 output on GPU.
        # For 3D experts + transpose, prefer direct [E, K, M] when the extension supports mode=2.
        if need_transpose and packed_gpu.ndim == 3 and max_transpose_mode >= 2:
            out_bf16 = torch.empty(E, K, M, dtype=torch.bfloat16, device=device)
            transpose_mode = 2
        elif need_transpose:
            out_bf16 = torch.empty(K, total_M, dtype=torch.bfloat16, device=device)
            transpose_mode = 1
        else:
            out_bf16 = torch.empty(total_M, K, dtype=torch.bfloat16, device=device)
            transpose_mode = 0

        # Run GPU kernel
        stream = torch.cuda.current_stream(device)
        _rdep_ext.ct_nvfp4_to_bf16(
            packed_flat.data_ptr(), scale_flat.data_ptr(), gs_flat.data_ptr(),
            out_bf16.data_ptr(),
            total_M, K, group_size,
            gs_stride, expert_rows, transpose_mode,
            stream,
        )

        # Reshape output to match model parameter shape
        if packed_gpu.ndim == 3 and need_transpose:
            if transpose_mode == 2:
                # Already materialized as [E, K, M].
                pass
            else:
                out_bf16 = out_bf16.reshape(K, E, M).permute(1, 0, 2).contiguous()
        elif packed_gpu.ndim == 3 and not need_transpose:
            out_bf16 = out_bf16.reshape(E, M, K)
        # For 2D: already [M, K] or [K, M]

        # Write directly to model parameter
        if target_param.shape == out_bf16.shape:
            target_param.data.copy_(out_bf16)
        elif out_bf16.ndim == 2 and target_param.ndim == 2 and target_param.shape == (out_bf16.shape[1], out_bf16.shape[0]):
            target_param.data.copy_(out_bf16.transpose(0, 1).contiguous())
        elif out_bf16.ndim == 3 and target_param.ndim == 3 and target_param.shape == (out_bf16.shape[0], out_bf16.shape[2], out_bf16.shape[1]):
            target_param.data.copy_(out_bf16.transpose(1, 2).contiguous())
        else:
            # Do not reshape blindly: reshape may silently corrupt layout.
            print_fn(f"[nvfp4] WARNING: Shape mismatch for {out_key}: "
                     f"kernel output {out_bf16.shape} vs param {target_param.shape}")
            result[out_key] = out_bf16.cpu()

        # Free GPU temporaries
        del packed_gpu, scale_gpu, gs_gpu, packed_flat, scale_flat, gs_flat, out_bf16

        consumed.update([pk, sk, gk])
        gpu_count += 1

    # Copy non-NVFP4 keys for standard load_state_dict
    for k, v in state_dict.items():
        if k not in consumed and k not in result:
            result[k] = v

    if gpu_count > 0:
        print_fn(f"[nvfp4] GPU dequantized {gpu_count} NVFP4 triplets → BF16 (CUDA kernel, zero CPU intermediates)")

    return result


def _is_nvfp4_checkpoint(state: dict) -> bool:
    """Check if a checkpoint contains NVFP4 compressed_tensors data."""
    return state.get('nvfp4_format') == 'compressed_tensors'


def _get_nvfp4_group_size(state: dict) -> int:
    """Get NVFP4 group size from checkpoint metadata."""
    return int(state.get('nvfp4_group_size', 16))


def _mark_nvfp4_expert_load_counter(model: torch.nn.Module, *, direct: bool) -> None:
    """Record whether expert NVFP4 load used direct mode vs forbidden fallback."""
    if direct:
        key = "_runtime_nvfp4_checkpoint_direct_expert_loads"
    else:
        key = "_runtime_nvfp4_checkpoint_non_direct_expert_loads"
    current = int(getattr(model, key, 0))
    setattr(model, key, current + 1)


def load_nvfp4_expert_buffers(
    expert_sd: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    group_size: int = 16,
    print_fn=print,
) -> bool:
    """Load NVFP4 triplets directly into MoE module buffers (no dequant).

    This is the Option C path: NVFP4 primary weights with no BF16 master.
    Expert NVFP4 triplets are loaded to GPU and assigned as registered buffers.
    The blockscaled weight cache is built from these buffers directly.

    Args:
        expert_sd: State dict containing NVFP4 triplets (weight_packed/scale/global_scale)
        model: Model with MoE modules
        group_size: NVFP4 group size
        print_fn: Logging function

    Returns:
        True if NVFP4 buffers were loaded, False if no NVFP4 data found
    """
    from nmoe.model import MoE

    # Find NVFP4 triplets for expert weights
    packed_keys = [k for k in expert_sd if k.endswith('.weight_packed')]
    if not packed_keys:
        return False

    # Build mapping: base_name -> (packed, scale, global_scale)
    triplets = {}
    for pk in packed_keys:
        base = pk[:-len('.weight_packed')]
        sk = base + '.weight_scale'
        gk = base + '.weight_global_scale'
        if sk in expert_sd and gk in expert_sd:
            triplets[base] = (expert_sd[pk], expert_sd[sk], expert_sd[gk])

    if not triplets:
        return False

    # Map base names to MoE modules
    # Key format: blocks.{layer_id}.ffn.{W1|W3|W2}
    moe_modules = {}
    for block in getattr(model, 'blocks', []):
        ffn = getattr(block, 'ffn', None)
        if ffn is not None and isinstance(ffn, MoE):
            moe_modules[block.layer_id] = ffn

    loaded_count = 0
    for base, (packed, scale, gs) in triplets.items():
        parts = base.split('.')
        # Expected: blocks.{N}.ffn.{W1|W3|W2}
        try:
            layer_idx = int(parts[1])
            param_name = parts[3]  # W1, W3, or W2
        except (IndexError, ValueError):
            continue

        if param_name not in ('W1', 'W3', 'W2'):
            continue

        moe = moe_modules.get(layer_idx)
        if moe is None:
            continue

        # Move triplet to GPU
        device = moe.W1.device
        packed_gpu = packed.to(device=device, non_blocking=True)
        scale_gpu = scale.to(device=device, non_blocking=True)
        gs_gpu = gs.to(device=device, dtype=torch.float32, non_blocking=True)

        # Set buffer on MoE module
        setattr(moe, f'_{param_name}_packed', packed_gpu)
        setattr(moe, f'_{param_name}_scale', scale_gpu)
        setattr(moe, f'_{param_name}_gs', gs_gpu)
        loaded_count += 1

    if loaded_count > 0:
        # Enable NVFP4 primary mode, free BF16 expert params, defer cache build.
        #
        # During checkpoint load, GPU holds ~76 GiB of BF16 expert params + ~32 GiB
        # dense + ~5 GiB RDEP = ~113 GiB.  Building the blockscaled weight cache
        # (refresh_weight_cache) requires ~1.4 GiB transient per layer for the
        # NVFP4 → BF16 → MMA conversion, which can cause GPU OOM/Xid 43 crashes.
        #
        # Instead: free BF16 expert params immediately (reclaiming ~76 GiB), and
        # let the cache be built lazily on the first forward pass via model.py's
        # `if self._W_cache is None: self.refresh_weight_cache()`.  By forward time,
        # the freed BF16 memory provides ample headroom for cache construction.
        freed_bytes = 0
        nvfp4_set_count = 0
        nvfp4_missing_count = 0
        for layer_id, moe in moe_modules.items():
            if moe._W1_packed is not None:
                moe._nvfp4_primary = True
                moe._nvfp4_group_size = group_size
                nvfp4_set_count += 1
                # Do NOT call refresh_weight_cache() here — defer to first forward
                # pass when GPU memory pressure is much lower.
                moe._W_cache = None

                # Free BF16 expert parameters — they're redundant with NVFP4 buffers.
                # Forward uses _W_cache (blockscaled MMA). Backward dequants from NVFP4
                # buffers on-the-fly. Gradients are consumed by fused_eco directly.
                # Replace parameter data with empty placeholder to preserve the
                # nn.Parameter object (needed for optimizer param groups) while freeing
                # the ~76 GiB of BF16 expert weight data.
                dev = moe.W1.device
                for param_name in ('W1', 'W3', 'W2'):
                    param = getattr(moe, param_name)
                    freed_bytes += param.data.nelement() * param.data.element_size()
                    param.data = torch.empty(0, dtype=torch.bfloat16, device=dev)
                    param.requires_grad_(False)
            else:
                nvfp4_missing_count += 1
                print_fn(f"[nvfp4] WARNING: MoE layer {layer_id} has _W1_packed=None after loading!")

        freed_gib = freed_bytes / (1024 ** 3)
        print_fn(f"[nvfp4] Loaded {loaded_count} NVFP4 triplets directly to GPU buffers "
                 f"(Option C: NVFP4 primary, no BF16 master)")
        print_fn(f"[nvfp4] Set _nvfp4_primary=True for {nvfp4_set_count}/{len(moe_modules)} MoE modules")
        if nvfp4_missing_count > 0:
            print_fn(f"[nvfp4] WARNING: {nvfp4_missing_count} MoE modules have no NVFP4 buffers!")
        if freed_bytes > 0:
            print_fn(f"[nvfp4] Freed {freed_gib:.1f} GiB of redundant BF16 expert parameters")
        print_fn(f"[nvfp4] Weight cache deferred to first forward pass (lazy build)")
        return True

    print_fn(f"[nvfp4] WARNING: No NVFP4 triplets loaded (loaded_count=0). "
             f"Total MoE modules found: {len(moe_modules)}")
    return False


def extract_nvfp4_expert_buffers(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract NVFP4 buffer triplets from MoE modules for checkpointing.

    When MoE modules are in NVFP4 primary mode, the NVFP4 buffers are the
    authoritative weights. This function extracts them in compressed_tensors
    format for saving.

    Args:
        model: Model with MoE modules in NVFP4 primary mode

    Returns:
        State dict with NVFP4 triplets, or empty dict if not in NVFP4 primary mode
    """
    from nmoe.model import MoE

    result = {}
    for block in getattr(model, 'blocks', []):
        ffn = getattr(block, 'ffn', None)
        if ffn is None or not isinstance(ffn, MoE):
            continue
        if not getattr(ffn, '_nvfp4_primary', False):
            continue

        layer_id = block.layer_id
        for param_name in ('W1', 'W3', 'W2'):
            packed = getattr(ffn, f'_{param_name}_packed', None)
            scale = getattr(ffn, f'_{param_name}_scale', None)
            gs = getattr(ffn, f'_{param_name}_gs', None)
            if packed is None:
                continue
            prefix = f'blocks.{layer_id}.ffn.{param_name}'
            result[f'{prefix}.weight_packed'] = packed
            result[f'{prefix}.weight_scale'] = scale
            result[f'{prefix}.weight_global_scale'] = gs

    return result


def _split_param_names(model: torch.nn.Module) -> Tuple[Set[str], Set[str]]:
    """Return (dense_names, expert_names) using model.param_sets(), if available."""
    dense_names: Set[str] = set()
    expert_names: Set[str] = set()
    name_to_param: Dict[int, str] = {id(p): n for n, p in model.named_parameters()}
    if hasattr(model, 'param_sets'):
        expert_params, dense_params = model.param_sets()  # type: ignore[attr-defined]
        expert_names = {name_to_param.get(id(p)) for p in expert_params}
        dense_names = {name_to_param.get(id(p)) for p in dense_params}
        expert_names = {n for n in expert_names if n is not None}
        dense_names = {n for n in dense_names if n is not None}
    else:
        # Fallback: everything is dense
        dense_names = {n for n, _ in model.named_parameters()}
        expert_names = set()
    return dense_names, expert_names


def build_states(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: int,
    loader,
    config_fingerprint: str = "",
    zero2_state: Optional[dict[str, Any]] = None,
    plan: Optional[MixturePlan] = None,
    ep_shard_info: Optional[EPShardInfo] = None,
) -> Tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Build (replicated_dense_state, rank_local_state).

    - replicated_dense_state (rank 0 writes rd.pt): only replicated parameters (dense/router).
    - rank_local_state (every rank writes dp_rank_XXX.pt): expert (sharded) params + local optimizer state.

    Args:
        step: Training step number.
        model: Model to checkpoint.
        optimizer: Optimizer to checkpoint.
        tokens: Number of tokens processed.
        loader: Data loader (for state_dict).
        config_fingerprint: Config hash for validation.
        zero2_state: Optional ZeRO-2 optimizer state.
        plan: Optional mixture plan.
        ep_shard_info: Optional EP sharding info for resharding support.
    """
    full_sd = model.state_dict()
    dense_names, expert_names = _split_param_names(model)

    # Check if model is in NVFP4 primary mode
    # If so, save NVFP4 buffers instead of BF16 expert parameters
    nvfp4_expert_sd = extract_nvfp4_expert_buffers(model)
    if nvfp4_expert_sd:
        # Option C: Save NVFP4 buffers as expert state (compressed_tensors format)
        expert_sd = nvfp4_expert_sd
    else:
        # Standard: Save BF16 expert parameters
        expert_sd = {k: v for k, v in full_sd.items() if k in expert_names}

    # Keep all non-expert state (including buffers) in rd.pt for correctness.
    # Only expert weights are sharded across ranks.
    dense_sd = {k: v for k, v in full_sd.items() if k not in expert_names}

    # Run metadata (immutable) for rd.pt
    world = _world_size()
    checkpoint_dp_count = _checkpoint_shard_count()
    cfg = getattr(model, 'config', None)
    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.getcwd(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        _log.debug("Could not determine git SHA for checkpoint metadata", exc_info=True)
        git_sha = 'unknown'

    rd_state: Optional[dict[str, Any]] = {
        'checkpoint_version': CHECKPOINT_FORMAT_VERSION,
        'step': step,
        'model_dense': dense_sd,
        'tokens': tokens,
        'config_fingerprint': config_fingerprint,
        'run_info': {
            'git_sha': git_sha,
            'preset': getattr(cfg, 'preset', None),
            'dtype': getattr(cfg, 'dtype', None),
            'E': getattr(cfg, 'n_routed_experts', None),
            'K': getattr(cfg, 'n_activated_experts', None),
            'H': getattr(cfg, 'dim', None),
            'L': getattr(cfg, 'n_layers', None),
            'world': world,
            'checkpoint_dp_count': checkpoint_dp_count,
            'checkpoint_layout': (
                'local_ep' if _use_local_ep_checkpoint_layout() else 'global_rank'
            ),
            'dataset_version': getattr(loader, 'dataset_version', None),
            'tokenizer_id': getattr(loader, 'tokenizer_id', None),
        }
    }

    if plan is not None:
        # Save plan bundle (rank 0 only)
        rd_state['plan_bundle'] = {
            'mixture_id': plan.mixture_id,
            'flow_mode': plan.flow_mode,
            'plan_hash': plan.plan_hash,
            'plan_json': plan.to_json(),
        }

    dp_state: dict[str, Any] = {
        'checkpoint_version': CHECKPOINT_FORMAT_VERSION,
        'step': step,
        'model_expert': expert_sd,
        'optimizer': optimizer.state_dict(),
        'loader': loader.state_dict() if hasattr(loader, 'state_dict') else None,
        'rng': {
            'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        'config_fingerprint': config_fingerprint,
    }
    # Mark as NVFP4 compressed_tensors format for round-trip load
    if nvfp4_expert_sd:
        dp_state['nvfp4_format'] = 'compressed_tensors'
        dp_state['nvfp4_group_size'] = 16
    if zero2_state is not None:
        dp_state['zero2'] = zero2_state

    # Save FusedBackwardECO state if active (fused backward-optimizer FP8 m/v).
    # All MoE layers share the same FusedBackwardECO instance (set by attach()),
    # so we save it once from the first MoE layer found.
    _fused_eco_ref = None
    for blk in getattr(model, 'blocks', []):
        ffn = getattr(blk, 'ffn', None)
        feco = getattr(ffn, '_fused_eco', None) if ffn is not None else None
        if feco is not None:
            if _fused_eco_ref is None:
                _fused_eco_ref = feco
            else:
                assert feco is _fused_eco_ref, (
                    "Multiple FusedBackwardECO instances found — checkpoint save assumes "
                    "all MoE layers share one instance. This is a bug."
                )
    if _fused_eco_ref is not None and hasattr(_fused_eco_ref, 'state_dict'):
        dp_state['fused_eco'] = _fused_eco_ref.state_dict()

    # Add EP sharding info for resharding support (version 3+)
    if ep_shard_info is not None:
        dp_state['ep_shard_info'] = ep_shard_info.to_dict()

    return rd_state, dp_state


def build_states_with_ep(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: int,
    loader,
    ep_size: int,
    ep_rank: int,
    n_total_experts: int,
    config_fingerprint: str = "",
    zero2_state: Optional[dict[str, Any]] = None,
    plan: Optional[MixturePlan] = None,
) -> Tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Build states with EP sharding info for multi-EP checkpoint.

    Convenience wrapper around build_states that creates EPShardInfo automatically.

    Args:
        step: Training step number.
        model: Model to checkpoint.
        optimizer: Optimizer to checkpoint.
        tokens: Number of tokens processed.
        loader: Data loader.
        ep_size: Total number of EP ranks.
        ep_rank: This rank's EP index.
        n_total_experts: Total number of experts.
        config_fingerprint: Config hash.
        zero2_state: Optional ZeRO-2 state.
        plan: Optional mixture plan.

    Returns:
        (rd_state, dp_state) tuple with EP sharding info.
    """
    ep_shard_info = EPShardInfo.create(ep_size, ep_rank, n_total_experts)
    return build_states(
        step=step,
        model=model,
        optimizer=optimizer,
        tokens=tokens,
        loader=loader,
        config_fingerprint=config_fingerprint,
        zero2_state=zero2_state,
        plan=plan,
        ep_shard_info=ep_shard_info,
    )


def load_state(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader=None,
    print_fn=print,
    nvfp4_direct: bool = False,
) -> tuple[int, int, Optional[dict[str, Any]]]:
    """Load split checkpoint and restore state. Returns (step, tokens, zero2_state).

    Expects rd.pt (replicated dense/router) in the same iteration directory,
    and dp_rank_XXX.pt for the calling rank.

    Args:
        nvfp4_direct: If True, load NVFP4 expert triplets directly to GPU buffers
            (Option C: NVFP4 primary, no BF16 master weights). NVFP4 non-direct
            expert load paths are forbidden and will raise.
    """
    it_dir = os.path.dirname(path)
    rd_path = os.path.join(it_dir, 'rd.pt')

    # Load rd.pt to CPU. For NVFP4 checkpoints, dequantize on GPU via CUDA
    # kernel (no CPU float32 intermediates). Non-NVFP4 keys go through standard
    # load_state_dict path.
    rd = torch.load(rd_path, map_location='cpu', weights_only=False)
    validate_checkpoint_version(rd)  # Fail early if checkpoint is too new
    dense_sd = rd['model_dense']

    # Detect NVFP4 compressed_tensors format and dequantize on GPU
    is_nvfp4 = _is_nvfp4_checkpoint(rd)
    if is_nvfp4:
        group_size = _get_nvfp4_group_size(rd)
        print_fn(f"[nvfp4] Detected NVFP4 compressed_tensors checkpoint (group_size={group_size})")
        # GPU-native dequant: moves NVFP4 triplets to GPU, dequants via kernel,
        # writes BF16 directly to model parameters. Returns only non-NVFP4 keys.
        dense_sd = dequantize_nvfp4_to_model_gpu(dense_sd, model, group_size, print_fn)

    # Load remaining (non-NVFP4) keys via standard path
    model.load_state_dict(dense_sd, strict=False)
    del dense_sd  # Free memory before loading expert shard
    tokens = int(rd.get('tokens', 0))
    step = int(rd.get('step', 0))
    # Validate config hash against current model config (best-effort)
    # Skip fingerprint validation for imported checkpoints (empty fingerprint)
    saved_fp = str(rd.get('config_fingerprint', ''))
    if saved_fp:
        try:
            from nmoe.config import fingerprint as _fingerprint
            current_fp = _fingerprint(getattr(model, 'config', None))
            if current_fp and saved_fp != current_fp:
                raise RuntimeError(
                    f"config_fingerprint mismatch on resume (saved={saved_fp[:8]} current={current_fp[:8]}). "
                    "Refusing to resume with a different config."
                )
        except RuntimeError:
            raise
        except Exception:
            _log.warning(
                "Config fingerprint validation skipped due to error", exc_info=True
            )
    else:
        print_fn("[checkpoint] Skipping fingerprint validation for imported checkpoint (no stored fingerprint)")

    # Check plan bundle if loader expects one
    if loader is not None and hasattr(loader, 'plan') and 'plan_bundle' in rd:
        saved_plan = rd['plan_bundle']
        current_plan = loader.plan
        if saved_plan['plan_hash'] != current_plan.plan_hash:
             # Mismatch policy: strictly fail unless user explicitly allows (not implemented yet)
            raise RuntimeError(
                f"Plan mismatch on resume! Saved hash={saved_plan['plan_hash'][:8]}, "
                f"Current hash={current_plan.plan_hash[:8]}. "
                "Cannot resume deterministically with different mixture plan."
            )
        print_fn(f"[resume] Plan hash match: {current_plan.plan_hash[:8]}")

    del rd  # Free ~12 GB CPU RAM before loading expert shard

    # Load rank-local shard — always to CPU for NVFP4 (same OOM reason as rd.pt)
    print_fn(f'Loading checkpoint: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_version(ckpt)  # Validate dp_rank checkpoint version too

    # Load expert weights: NVFP4 checkpoints must use direct buffer load.
    expert_sd = ckpt['model_expert']
    ckpt_is_nvfp4 = is_nvfp4 or _is_nvfp4_checkpoint(ckpt)
    # Auto-detect Option C: if checkpoint was saved with nvfp4_format metadata, use direct path
    nvfp4_direct_load = nvfp4_direct or ckpt.get('nvfp4_format') == 'compressed_tensors'

    if ckpt_is_nvfp4 and nvfp4_direct_load:
        gs = _get_nvfp4_group_size(ckpt) if _is_nvfp4_checkpoint(ckpt) else group_size
        print_fn(f"[nvfp4] Loading expert NVFP4 triplets directly to GPU buffers (Option C)...")
        loaded = load_nvfp4_expert_buffers(expert_sd, model, gs, print_fn)
        if not loaded:
            _mark_nvfp4_expert_load_counter(model, direct=False)
            raise RuntimeError(
                "[nvfp4] No NVFP4 triplets found in expert state dict but checkpoint "
                "is marked as nvfp4_format=compressed_tensors. The checkpoint may be "
                "corrupted or incompatible. CPU dequant fallback has been removed."
            )
        _mark_nvfp4_expert_load_counter(model, direct=True)
    elif ckpt_is_nvfp4:
        _mark_nvfp4_expert_load_counter(model, direct=False)
        raise RuntimeError(
            "[nvfp4] Expert non-direct load path is forbidden for NVFP4 production runs. "
            "Set nvfp4_direct=True and provide compressed_tensors expert triplets."
        )
    else:
        model.load_state_dict(expert_sd, strict=False)
    del expert_sd

    # Restore optimizer state if available (imported checkpoints may have empty optimizer)
    opt_state = ckpt.get('optimizer')
    if opt_state and isinstance(opt_state, dict) and 'param_groups' in opt_state:
        optimizer.load_state_dict(opt_state)
    elif step == 0:
        print_fn('[resume] Imported checkpoint (step=0): skipping optimizer state restore')
    else:
        print_fn(f'[resume] WARNING: No valid optimizer state in checkpoint at step={step}')

    if loader is not None and ckpt.get('loader'):
        loader.load_state_dict(ckpt['loader'])

    if ckpt.get('rng'):
        torch_state = ckpt['rng'].get('torch')
        if torch_state is not None:
            if not torch.is_tensor(torch_state):
                torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
            torch.random.set_rng_state(torch_state.cpu())
        cuda_states = ckpt['rng'].get('cuda')
        if cuda_states is not None and torch.cuda.is_available():
            for dev, state in enumerate(cuda_states):
                if state is not None:
                    state = torch.as_tensor(state, dtype=torch.uint8).cpu()
                    torch.cuda.set_rng_state(state, dev)

    # Stash FusedBackwardECO state on model for later restoration by training loop
    fused_eco_sd = ckpt.get('fused_eco')
    if fused_eco_sd is not None:
        model._pending_fused_eco_state = fused_eco_sd
        print_fn(f'[resume] Found FusedBackwardECO state (step={fused_eco_sd.get("step_count", "?")})')

    return int(step), tokens, ckpt.get('zero2')


def load_state_with_ep_check(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ep_size: int,
    ep_rank: int,
    n_total_experts: int,
    loader=None,
    print_fn=print,
    strict_ep: bool = True,
    nvfp4_direct: bool = False,
) -> tuple[int, int, Optional[dict[str, Any]], Optional[EPShardInfo]]:
    """Load checkpoint with EP sharding compatibility check.

    This is an enhanced version of load_state that validates EP configuration
    and returns the saved EP sharding info for debugging/logging.

    Args:
        path: Path to dp_rank_*.pt file.
        model: Model to load into.
        optimizer: Optimizer to load into.
        ep_size: Current EP size.
        ep_rank: Current EP rank.
        n_total_experts: Current total experts.
        loader: Optional data loader.
        print_fn: Print function for logging.
        strict_ep: If True, raise error on EP mismatch. If False, warn only.
        nvfp4_direct: If True, load NVFP4 expert triplets directly to GPU buffers
            (Option C: NVFP4 primary, no BF16 master weights). NVFP4 non-direct
            expert load paths are forbidden and will raise.

    Returns:
        (step, tokens, zero2_state, saved_ep_info) tuple.

    Raises:
        RuntimeError: If strict_ep=True and EP configuration doesn't match.
    """
    it_dir = os.path.dirname(path)
    rd_path = os.path.join(it_dir, 'rd.pt')

    # Always torch.load to CPU first (standard practice). NVFP4 dequantization
    # is then performed on GPU via CUDA kernel, avoiding CPU float32 intermediates.

    # Load dp_rank checkpoint first to check EP sharding
    print_fn(f'Loading checkpoint: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_version(ckpt)

    # Check EP sharding compatibility
    saved_ep_info = get_ep_shard_info_from_checkpoint(ckpt)
    compatible, message = validate_ep_shard_compatibility(
        saved_ep_info, ep_size, ep_rank, n_total_experts
    )

    if not compatible:
        if strict_ep:
            raise RuntimeError(f"EP sharding incompatible: {message}")
        else:
            print_fn(f"[WARNING] EP sharding mismatch (non-strict mode): {message}")

    if saved_ep_info is not None:
        print_fn(
            f"[resume] EP sharding: saved EP={saved_ep_info.ep_size} rank={saved_ep_info.ep_rank} "
            f"experts={saved_ep_info.expert_start}-{saved_ep_info.expert_end}, "
            f"current EP={ep_size} rank={ep_rank}"
        )

    # Load replicated dense/router weights
    rd = torch.load(rd_path, map_location='cpu', weights_only=False)
    validate_checkpoint_version(rd)
    dense_sd = rd['model_dense']

    # Detect NVFP4 compressed_tensors and dequantize on GPU
    is_nvfp4 = _is_nvfp4_checkpoint(rd) or _is_nvfp4_checkpoint(ckpt)
    if is_nvfp4:
        group_size = _get_nvfp4_group_size(rd) if _is_nvfp4_checkpoint(rd) else _get_nvfp4_group_size(ckpt)
        print_fn(f"[nvfp4] Detected NVFP4 compressed_tensors checkpoint (group_size={group_size})")
        dense_sd = dequantize_nvfp4_to_model_gpu(dense_sd, model, group_size, print_fn)

    model.load_state_dict(dense_sd, strict=False)
    del dense_sd
    tokens = int(rd.get('tokens', 0))
    step = int(rd.get('step', 0))

    # Validate config hash (skip for imported checkpoints with empty fingerprint)
    saved_fp = str(rd.get('config_fingerprint', ''))
    if saved_fp:
        try:
            from nmoe.config import fingerprint as _fingerprint
            current_fp = _fingerprint(getattr(model, 'config', None))
            if current_fp and saved_fp != current_fp:
                raise RuntimeError(
                    f"config_fingerprint mismatch on resume (saved={saved_fp[:8]} current={current_fp[:8]}). "
                    "Refusing to resume with a different config."
                )
        except RuntimeError:
            raise
        except Exception:
            _log.warning(
                "Config fingerprint validation skipped due to error", exc_info=True
            )
    else:
        print_fn("[checkpoint] Skipping fingerprint validation for imported checkpoint (no stored fingerprint)")

    # Load expert weights: NVFP4 checkpoints must use direct buffer load.
    expert_sd = ckpt['model_expert']
    ckpt_is_nvfp4 = is_nvfp4 or _is_nvfp4_checkpoint(ckpt)
    nvfp4_direct_load = nvfp4_direct or ckpt.get('nvfp4_format') == 'compressed_tensors'

    if ckpt_is_nvfp4 and nvfp4_direct_load:
        gs = _get_nvfp4_group_size(ckpt) if _is_nvfp4_checkpoint(ckpt) else group_size
        print_fn(f"[nvfp4] Loading expert NVFP4 triplets directly to GPU buffers (Option C)...")
        loaded = load_nvfp4_expert_buffers(expert_sd, model, gs, print_fn)
        if not loaded:
            _mark_nvfp4_expert_load_counter(model, direct=False)
            raise RuntimeError(
                "[nvfp4] No NVFP4 triplets found in expert state dict but direct NVFP4 "
                "load is required for production."
            )
        _mark_nvfp4_expert_load_counter(model, direct=True)
    elif ckpt_is_nvfp4:
        _mark_nvfp4_expert_load_counter(model, direct=False)
        raise RuntimeError(
            "[nvfp4] Expert non-direct load path is forbidden for NVFP4 production runs. "
            "Set nvfp4_direct=True and provide compressed_tensors expert triplets."
        )
    else:
        model.load_state_dict(expert_sd, strict=False)
    del expert_sd

    # Restore optimizer state if available (imported checkpoints may have empty optimizer)
    opt_state = ckpt.get('optimizer')
    if opt_state and isinstance(opt_state, dict) and 'param_groups' in opt_state:
        optimizer.load_state_dict(opt_state)
    elif step == 0:
        print_fn('[resume] Imported checkpoint (step=0): skipping optimizer state restore')
    else:
        print_fn(f'[resume] WARNING: No valid optimizer state in checkpoint at step={step}')

    if loader is not None and ckpt.get('loader'):
        loader.load_state_dict(ckpt['loader'])

    if ckpt.get('rng'):
        torch_state = ckpt['rng'].get('torch')
        if torch_state is not None:
            if not torch.is_tensor(torch_state):
                torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
            torch.random.set_rng_state(torch_state.cpu())
        cuda_states = ckpt['rng'].get('cuda')
        if cuda_states is not None and torch.cuda.is_available():
            for dev, state in enumerate(cuda_states):
                if state is not None:
                    state = torch.as_tensor(state, dtype=torch.uint8).cpu()
                    torch.cuda.set_rng_state(state, dev)

    return int(step), tokens, ckpt.get('zero2'), saved_ep_info


# =============================================================================
# EP Resharding Utilities (for resuming with different EP configurations)
# =============================================================================

def reshard_expert_weights(
    expert_state_dict: Dict[str, torch.Tensor],
    saved_ep_info: EPShardInfo,
    target_ep_size: int,
    target_ep_rank: int,
    print_fn=print,
) -> Dict[str, torch.Tensor]:
    """Reshard expert weights from one EP configuration to another.

    This function takes expert weights from a checkpoint saved with one EP
    configuration and reshards them for loading with a different EP configuration.

    Args:
        expert_state_dict: State dict with expert weights from checkpoint.
        saved_ep_info: EP sharding info from the checkpoint.
        target_ep_size: Target EP size to reshard for.
        target_ep_rank: Target EP rank to extract.
        print_fn: Print function for logging.

    Returns:
        New state dict with resharded expert weights for target EP rank.

    Note:
        This function requires all shards to be available for full resharding.
        For partial resharding (when not all shards are loaded), use
        reshard_expert_weights_partial.
    """
    # Calculate target expert range
    n_total = saved_ep_info.n_total_experts
    if n_total % target_ep_size != 0:
        raise ValueError(
            f"Cannot reshard: n_total_experts ({n_total}) not divisible by "
            f"target_ep_size ({target_ep_size})"
        )

    n_local_target = n_total // target_ep_size
    target_start = target_ep_rank * n_local_target
    target_end = target_start + n_local_target

    # Calculate source expert range
    src_start = saved_ep_info.expert_start
    src_end = saved_ep_info.expert_end

    print_fn(
        f"[reshard] source experts {src_start}-{src_end} -> "
        f"target experts {target_start}-{target_end}"
    )

    # Check overlap between source and target
    overlap_start = max(src_start, target_start)
    overlap_end = min(src_end, target_end)

    if overlap_start >= overlap_end:
        # No overlap - this shard doesn't contain any experts for target
        print_fn(f"[reshard] No overlap, returning empty state dict")
        return {}

    # Calculate which experts to extract
    # relative_start: offset within the source shard
    # relative_end: end offset within the source shard
    relative_start = overlap_start - src_start
    relative_end = overlap_end - src_start

    resharded = {}
    for key, tensor in expert_state_dict.items():
        # Assume expert weights have expert dimension as first dim
        # e.g., [n_local_experts, hidden_size, intermediate_size]
        if tensor.ndim >= 1:
            # Extract the overlapping experts
            sliced = tensor[relative_start:relative_end].contiguous()
            resharded[key] = sliced
            print_fn(f"[reshard] {key}: {tensor.shape} -> {sliced.shape}")
        else:
            # Scalar or 0-dim tensor, keep as is
            resharded[key] = tensor

    return resharded


def gather_all_expert_shards(
    base_path: str,
    step: int,
    saved_ep_size: int,
) -> Dict[str, torch.Tensor]:
    """Gather expert weights from all EP shards into a single state dict.

    This function loads all dp_rank_*.pt files and concatenates the expert
    weights to reconstruct the full model.

    Args:
        base_path: Base checkpoint directory.
        step: Step number to load.
        saved_ep_size: EP size used when saving (number of shards).

    Returns:
        Combined state dict with all expert weights concatenated.

    Raises:
        FileNotFoundError: If any shard is missing.
    """
    it_dir = iteration_dir(base_path, step)
    all_experts: Dict[str, list] = {}

    for ep_rank in range(saved_ep_size):
        dp_path = os.path.join(it_dir, f'dp_rank_{ep_rank:03d}.pt')
        if not os.path.exists(dp_path):
            raise FileNotFoundError(f"Missing shard: {dp_path}")

        ckpt = torch.load(dp_path, map_location='cpu', weights_only=False)
        expert_sd = ckpt.get('model_expert', {})

        for key, tensor in expert_sd.items():
            if key not in all_experts:
                all_experts[key] = []
            all_experts[key].append(tensor)

    # Concatenate along expert dimension (assumed to be dim 0)
    combined = {}
    for key, tensors in all_experts.items():
        if tensors[0].ndim >= 1:
            combined[key] = torch.cat(tensors, dim=0)
        else:
            # Scalar - should be the same across shards
            combined[key] = tensors[0]

    return combined


def load_with_resharding(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target_ep_size: int,
    target_ep_rank: int,
    n_total_experts: int,
    loader=None,
    print_fn=print,
) -> tuple[int, int, Optional[dict[str, Any]]]:
    """Load checkpoint with automatic EP resharding.

    This function detects EP configuration mismatch and automatically reshards
    expert weights to match the target configuration.

    Args:
        path: Path to dp_rank_*.pt file.
        model: Model to load into.
        optimizer: Optimizer to load into.
        target_ep_size: Target EP size.
        target_ep_rank: Target EP rank.
        n_total_experts: Total number of experts.
        loader: Optional data loader.
        print_fn: Print function for logging.

    Returns:
        (step, tokens, zero2_state) tuple.

    Note:
        For resharding, this function may need to load multiple checkpoint
        shards and will be slower than direct loading.
    """
    it_dir = os.path.dirname(path)
    rd_path = os.path.join(it_dir, 'rd.pt')

    # Always torch.load to CPU first (standard practice for checkpoint loading).
    # NVFP4 dequantization is then performed on GPU via CUDA kernel, avoiding
    # CPU float32 intermediates entirely.

    # Load rd.pt (dense weights)
    rd = torch.load(rd_path, map_location='cpu', weights_only=False)
    validate_checkpoint_version(rd)
    dense_sd = rd['model_dense']

    # Detect NVFP4. Precision-invariant production path does not support
    # dequantized expert resharding.
    is_nvfp4 = _is_nvfp4_checkpoint(rd)
    if is_nvfp4:
        raise RuntimeError(
            "[reshard/nvfp4] NVFP4 resharding via dequantized expert tensors is "
            "disabled for production precision-invariant runs. Use an EP-matched "
            "NVFP4 checkpoint layout (pre-sharded) and direct expert buffer loading."
        )

    model.load_state_dict(dense_sd, strict=False)
    del dense_sd
    tokens = int(rd.get('tokens', 0))
    step = int(rd.get('step', 0))

    # Validate config hash (skip for imported checkpoints with empty fingerprint)
    saved_fp = str(rd.get('config_fingerprint', ''))
    if saved_fp:
        try:
            from nmoe.config import fingerprint as _fingerprint
            current_fp = _fingerprint(getattr(model, 'config', None))
            if current_fp and saved_fp != current_fp:
                raise RuntimeError(
                    f"config_fingerprint mismatch on resume (saved={saved_fp[:8]} current={current_fp[:8]}). "
                    "Refusing to resume with a different config."
                )
        except RuntimeError:
            raise
        except Exception:
            _log.warning(
                "Config fingerprint validation skipped due to error", exc_info=True
            )
    else:
        print_fn("[reshard] Skipping fingerprint validation for imported checkpoint (no stored fingerprint)")

    # Load the specified dp_rank checkpoint to check EP info
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_version(ckpt)

    # Also check dp checkpoint for NVFP4
    if not is_nvfp4:
        is_nvfp4 = _is_nvfp4_checkpoint(ckpt)
    if is_nvfp4:
        raise RuntimeError(
            "[reshard/nvfp4] NVFP4 resharding via dequantized expert tensors is "
            "disabled for production precision-invariant runs. Use an EP-matched "
            "NVFP4 checkpoint layout (pre-sharded) and direct expert buffer loading."
        )
    saved_ep_info = get_ep_shard_info_from_checkpoint(ckpt)

    if saved_ep_info is None:
        # Legacy checkpoint - assume world=1
        print_fn("[reshard] Legacy checkpoint, assuming EP=1")
        saved_ep_info = EPShardInfo.create(ep_size=1, ep_rank=0, n_total_experts=n_total_experts)

    # Check if resharding is needed
    compatible, msg = validate_ep_shard_compatibility(
        saved_ep_info, target_ep_size, target_ep_rank, n_total_experts
    )

    if compatible:
        # No resharding needed - load directly
        print_fn(f"[reshard] EP config matches, loading directly")
        expert_sd = ckpt['model_expert']
        model.load_state_dict(expert_sd, strict=False)
        del expert_sd
        # Restore optimizer state if available
        opt_state = ckpt.get('optimizer')
        if opt_state and isinstance(opt_state, dict) and 'param_groups' in opt_state:
            optimizer.load_state_dict(opt_state)
    else:
        # Resharding needed
        print_fn(f"[reshard] EP config mismatch: {msg}")
        print_fn(f"[reshard] Resharding from EP={saved_ep_info.ep_size} to EP={target_ep_size}")

        if saved_ep_info.ep_size == 1:
            # Single shard contains all experts - can reshard directly
            expert_sd = ckpt['model_expert']
            resharded = reshard_expert_weights(
                expert_sd, saved_ep_info, target_ep_size, target_ep_rank, print_fn
            )
            model.load_state_dict(resharded, strict=False)
        else:
            # Multiple shards - need to figure out which shards to load
            # For simplicity, load all shards that overlap with target range
            base_path = os.path.dirname(it_dir)

            # Calculate which source shards overlap with target
            n_local_target = n_total_experts // target_ep_size
            target_start = target_ep_rank * n_local_target
            target_end = target_start + n_local_target

            merged_experts = {}
            for src_rank in range(saved_ep_info.ep_size):
                src_info = EPShardInfo.create(
                    saved_ep_info.ep_size, src_rank, n_total_experts
                )
                # Check if this source shard overlaps with target
                if src_info.expert_end <= target_start or src_info.expert_start >= target_end:
                    continue  # No overlap

                src_path = os.path.join(it_dir, f'dp_rank_{src_rank:03d}.pt')
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"Missing shard for resharding: {src_path}")

                src_ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
                src_expert_sd = src_ckpt['model_expert']
                resharded = reshard_expert_weights(
                    src_expert_sd, src_info, target_ep_size, target_ep_rank, print_fn
                )

                # Merge into final dict
                for key, tensor in resharded.items():
                    if key not in merged_experts:
                        merged_experts[key] = []
                    merged_experts[key].append(tensor)

            # Concatenate tensors from multiple source shards
            final_experts = {}
            for key, tensors in merged_experts.items():
                if len(tensors) == 1:
                    final_experts[key] = tensors[0]
                else:
                    final_experts[key] = torch.cat(tensors, dim=0)

            model.load_state_dict(final_experts, strict=False)

        # Optimizer state cannot be resharded easily - reinitialize
        print_fn("[reshard] WARNING: Optimizer state not resharded, reinitializing")

    # Load data loader state if available
    if loader is not None and ckpt.get('loader'):
        loader.load_state_dict(ckpt['loader'])

    # Restore RNG state
    if ckpt.get('rng'):
        torch_state = ckpt['rng'].get('torch')
        if torch_state is not None:
            if not torch.is_tensor(torch_state):
                torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
            torch.random.set_rng_state(torch_state.cpu())
        cuda_states = ckpt['rng'].get('cuda')
        if cuda_states is not None and torch.cuda.is_available():
            for dev, state in enumerate(cuda_states):
                if state is not None:
                    state = torch.as_tensor(state, dtype=torch.uint8).cpu()
                    torch.cuda.set_rng_state(state, dev)

    return int(step), tokens, ckpt.get('zero2')


def load_checkpoint(
    checkpointer: Checkpointer,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    plan,
    cfg,
    rank: int,
    print_fn=print,
) -> tuple[int, int, dict]:
    """Wrapper for checkpoint resume. Returns (start_step, tokens_seen, zero2_state)."""
    start_step = 0
    tokens_seen = 0
    zero2_state = {}

    def _consensus_resume_step(local_step: int) -> int:
        if not _is_dist():
            return local_step
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        step_tensor = torch.tensor([int(local_step)], device=device, dtype=torch.int64)
        min_tensor = step_tensor.clone()
        max_tensor = step_tensor.clone()
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        min_step = int(min_tensor.item())
        max_step = int(max_tensor.item())
        if min_step != max_step:
            if rank == 0:
                print_fn(
                    "[checkpoint] WARNING: step skew detected across ranks "
                    f"(min={min_step}, max={max_step}); searching for last common step"
                )

            candidate = min_step
            while candidate >= 0:
                has_local = 1 if checkpointer.path_for_step(candidate) is not None else 0
                has_tensor = torch.tensor([has_local], device=device, dtype=torch.int64)
                dist.all_reduce(has_tensor, op=dist.ReduceOp.MIN)
                if int(has_tensor.item()) == 1:
                    if rank == 0 and candidate != max_step:
                        print_fn(f"[checkpoint] Falling back to common step {candidate}")
                    return candidate
                candidate -= 1
            return -1
        return min_step

    def _consensus_checkpoint_signature(step: int) -> None:
        if not _is_dist():
            return

        rd = _safe_load(_rd_path(checkpointer.base, step))
        run_info = rd.get('run_info', {}) if isinstance(rd, dict) else {}
        dp_count, saved_world = _read_checkpoint_layout_from_rd(checkpointer.base, step)
        local_sig_payload = {
            "step": int(step),
            "checkpoint_version": int(rd.get('checkpoint_version', 0)) if isinstance(rd, dict) else 0,
            "config_fingerprint": str(rd.get('config_fingerprint', '')) if isinstance(rd, dict) else "",
            "world": int(saved_world),
            "dp_count": int(dp_count),
            "layout": str(run_info.get('checkpoint_layout', '')),
            "git_sha": str(run_info.get('git_sha', '')),
        }
        local_sig_json = json.dumps(local_sig_payload, sort_keys=True, separators=(",", ":"))
        local_sig = hashlib.sha256(local_sig_json.encode("utf-8")).hexdigest()

        signatures: list[str] = ["" for _ in range(_world_size())]
        dist.all_gather_object(signatures, local_sig)
        if len(set(signatures)) != 1:
            raise RuntimeError(
                "Checkpoint signature mismatch across ranks at resume step "
                f"{step}: {signatures}"
            )

    eco_fused = getattr(cfg, 'eco_fused_backward', False)
    if getattr(cfg, 'resume', True):
        local_step, _ = checkpointer.find_latest()
        step = _consensus_resume_step(local_step)
        path = checkpointer.path_for_step(step) if step >= 0 else None
        if step >= 0 and path is None:
            raise FileNotFoundError(
                f"Checkpoint step {step} selected by consensus but local shard file is missing"
            )

        if path is not None:
            saved_dp_count, saved_world = _read_checkpoint_layout_from_rd(checkpointer.base, step)
            current_world = _world_size()
            current_dp_count = _checkpoint_shard_count()
            if saved_world > 0 and saved_world != current_world:
                raise RuntimeError(
                    "Checkpoint world-size mismatch: "
                    f"saved={saved_world}, current={current_world}"
                )
            if saved_dp_count > 0 and saved_dp_count != current_dp_count:
                raise RuntimeError(
                    "Checkpoint shard-count mismatch: "
                    f"saved={saved_dp_count}, current={current_dp_count}"
                )

            checkpoint_local_layout = _checkpoint_uses_local_layout(checkpointer.base, step)
            env_local_layout = _use_local_ep_checkpoint_layout()
            if (
                checkpoint_local_layout is not None
                and checkpoint_local_layout != env_local_layout
            ):
                mode = "local-ep" if checkpoint_local_layout else "global-rank"
                expected_env = "1" if checkpoint_local_layout else "0"
                raise RuntimeError(
                    "Checkpoint layout mismatch: on-disk layout is "
                    f"{mode} for step {step}, but NMOE_CHECKPOINT_LOCAL_EP={int(env_local_layout)}. "
                    f"Set NMOE_CHECKPOINT_LOCAL_EP={expected_env} and retry."
                )

            _consensus_checkpoint_signature(step)

            run_dtype = str(getattr(cfg, 'dtype', 'bf16') or 'bf16').lower()
            nvfp4_direct = run_dtype == 'nvfp4' or getattr(cfg, 'eco_enabled', False) or eco_fused
            if rank == 0 and nvfp4_direct:
                print_fn(f"[checkpoint] Loading with nvfp4_direct=True (eco_fused_backward={eco_fused})")
            start_step, tokens_seen, z2 = load_state(
                path, model, optimizer, loader, print_fn,
                nvfp4_direct=nvfp4_direct,
            )
            if z2 is not None:
                zero2_state = z2
            if rank == 0:
                print_fn(f"[nmoe] Resumed from step {start_step}, {tokens_seen:,} tokens")
        elif rank == 0:
            print_fn(f"[checkpoint] No checkpoint found at {checkpointer.base}")
            if eco_fused:
                print_fn(f"[checkpoint] WARNING: eco_fused_backward=True requires a valid NVFP4 checkpoint!")
    elif rank == 0:
        print_fn(f"[checkpoint] resume=False, starting fresh")

    return start_step, tokens_seen, zero2_state


def save_checkpoint(
    checkpointer: Checkpointer,
    step: int,
    tokens_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    plan,
    zero2_state: dict,
    cfg,
    rank: int,
    config_fingerprint: str,
    checkpoint_every: int,
    print_fn=print,
) -> None:
    """Wrapper for checkpoint save. Handles RD/DP split and finalization."""
    if (step % checkpoint_every == 0) or (step == cfg.steps):
        rd_state, dp_state = build_states(
            step, model, optimizer, tokens_seen, loader, config_fingerprint,
            zero2_state=zero2_state, plan=plan,
        )
        if _is_checkpoint_writer():
            checkpointer.save_dense(step, rd_state)
        checkpointer.save_rank_local(step, dp_state)

        if _is_checkpoint_writer() and checkpointer.last_ms > 0:
            size_mb = checkpointer.last_bytes / (1024 * 1024)
            print_fn(f"[ckpt] saved step={step} size={size_mb:.1f}MB time={checkpointer.last_ms:.0f}ms")

        # Try to finalize: writer rank writes manifest.json and flips tracker.
        if _is_checkpoint_writer():
            try:
                done = checkpointer.try_finalize(step)
                if done:
                    print_fn(f"[ckpt] complete step={step}")
            except Exception:
                _log.error("Failed to finalize checkpoint step=%d", step, exc_info=True)
