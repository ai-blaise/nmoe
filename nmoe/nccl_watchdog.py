"""NCCL operation watchdog — detects hung collectives and provides diagnostics."""

import os
import logging
import threading
import time
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Default timeout from env, fallback 600s (10 min)
_DEFAULT_TIMEOUT_S = int(os.getenv("NMOE_NCCL_TIMEOUT_S", "600"))


class NCCLWatchdog:
    """Monitors async NCCL operations and logs warnings if they exceed a timeout.

    Usage:
        watchdog = NCCLWatchdog(timeout_s=300)

        # For async ops:
        work = dist.all_gather_into_tensor(..., async_op=True)
        watchdog.watch(work, "all_gather params")
        work.wait()  # If this hangs, watchdog will log after timeout
        watchdog.clear()

        # Context manager for sync ops:
        with watchdog.guard("reduce_scatter chunk 3"):
            dist.reduce_scatter_tensor(...)
    """

    def __init__(self, timeout_s: int | None = None):
        self.timeout_s = timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S
        self._timer: threading.Timer | None = None
        self._op_name: str = ""
        self._start_time: float = 0.0
        self._lock = threading.Lock()

    def _on_timeout(self) -> None:
        elapsed = time.monotonic() - self._start_time
        rank = dist.get_rank() if dist.is_initialized() else -1
        msg = (
            f"[Rank {rank}] NCCL watchdog: operation '{self._op_name}' "
            f"has not completed after {elapsed:.0f}s (timeout={self.timeout_s}s). "
            f"Possible hung collective — a remote rank may have crashed."
        )
        logger.error(msg)
        # Also print to stderr for visibility in torchrun logs
        import sys
        print(msg, file=sys.stderr, flush=True)
        os._exit(1)

    def watch(self, work: dist.Work | None, op_name: str = "nccl_op") -> None:
        """Start monitoring an async NCCL operation."""
        with self._lock:
            self.clear_unlocked()
            if work is None:
                return
            self._op_name = op_name
            self._start_time = time.monotonic()
            self._timer = threading.Timer(self.timeout_s, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()

    def clear(self) -> None:
        """Cancel the current watchdog timer (call after work.wait() succeeds)."""
        with self._lock:
            self.clear_unlocked()

    def clear_unlocked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def guard(self, op_name: str = "nccl_op"):
        """Context manager for synchronous NCCL operations."""
        return _WatchdogGuard(self, op_name)


class _WatchdogGuard:
    def __init__(self, wd: NCCLWatchdog, op_name: str):
        self._wd = wd
        self._op_name = op_name
        self._dummy_event = None

    def __enter__(self):
        self._wd._lock.acquire()
        self._wd.clear_unlocked()
        self._wd._op_name = self._op_name
        self._wd._start_time = time.monotonic()
        self._wd._timer = threading.Timer(self._wd.timeout_s, self._wd._on_timeout)
        self._wd._timer.daemon = True
        self._wd._timer.start()
        self._wd._lock.release()
        return self

    def __exit__(self, *exc):
        self._wd.clear()
        return False


# Module-level singleton for convenience
_global_watchdog: NCCLWatchdog | None = None


def get_watchdog(timeout_s: int | None = None) -> NCCLWatchdog:
    """Get or create the global NCCL watchdog singleton."""
    global _global_watchdog
    if _global_watchdog is None:
        _global_watchdog = NCCLWatchdog(timeout_s)
    return _global_watchdog
