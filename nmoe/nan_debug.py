"""Process-local NaN debug scope shared across training hot-path modules."""

from __future__ import annotations

import os


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in ("1", "true", "True")


_NAN_DEBUG_ACTIVE = _env_flag("NMOE_NAN_DEBUG_ACTIVE", "0")


def set_nan_debug_active(active: bool, *, mirror_env: bool = False) -> None:
    """Set process-local NaN debug scope.

    mirror_env exists for compatibility with external tooling, but training
    hot paths should rely on this in-process flag to avoid repeated getenv/setenv.
    """
    global _NAN_DEBUG_ACTIVE
    _NAN_DEBUG_ACTIVE = bool(active)
    if mirror_env:
        os.environ["NMOE_NAN_DEBUG_ACTIVE"] = "1" if active else "0"


def nan_debug_active() -> bool:
    """Return process-local NaN debug scope."""
    return _NAN_DEBUG_ACTIVE

