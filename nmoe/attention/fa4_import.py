import importlib
import logging
import sys
import types
from functools import lru_cache
from pathlib import Path


logger = logging.getLogger(__name__)


def _flash_attn_pkg_dir() -> Path:
  # nmoe/attention/fa4_import.py -> repo root
  return Path(__file__).resolve().parents[2] / "third_party" / "flash_attn" / "flash_attn"


def _is_namespace_fallback_error(exc: Exception) -> bool:
  if isinstance(exc, ModuleNotFoundError):
    if exc.name in {"flash_attn", "flash_attn_2_cuda"}:
      return True
  msg = str(exc)
  return ("flash_attn_2_cuda" in msg) or ("No module named 'flash_attn'" in msg)


def _install_flash_attn_namespace() -> None:
  pkg_dir = _flash_attn_pkg_dir()
  if not pkg_dir.is_dir():
    raise ModuleNotFoundError(
      f"Missing flash-attention source tree at {pkg_dir}. "
      "Provision third_party/flash_attn before launch."
    )

  # Clear partial imports before rebuilding namespace package.
  sys.modules.pop("flash_attn.cute.interface", None)
  sys.modules.pop("flash_attn.cute", None)
  sys.modules.pop("flash_attn", None)

  pkg = types.ModuleType("flash_attn")
  pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
  pkg.__package__ = "flash_attn"
  sys.modules["flash_attn"] = pkg

  parent = str(pkg_dir.parent)
  if parent not in sys.path:
    sys.path.insert(0, parent)


@lru_cache(maxsize=1)
def _fa4_interface_module():
  try:
    return importlib.import_module("flash_attn.cute.interface")
  except Exception as first_err:
    if not _is_namespace_fallback_error(first_err):
      raise
    _install_flash_attn_namespace()
    try:
      return importlib.import_module("flash_attn.cute.interface")
    except Exception as second_err:
      logger.debug(
        "flash_attn namespace fallback import failed after initial error: %s",
        first_err,
        exc_info=True,
      )
      raise second_err


def get_fa4_fwd():
  mod = _fa4_interface_module()
  fn = getattr(mod, "_flash_attn_fwd", None)
  if fn is None:
    raise RuntimeError("flash_attn.cute.interface is importable but _flash_attn_fwd is missing.")
  return fn


@lru_cache(maxsize=1)
def get_flashmla_sm100_module():
  from nmoe.csrc import flashmla_sm100
  return flashmla_sm100


def get_fa4_flashmla_modules():
  return get_fa4_fwd(), get_flashmla_sm100_module()


def probe_fa4_flashmla_bindings(
    *,
    require_fa4_fwd: bool = True,
    require_flashmla_fwd: bool = False,
) -> list[str]:
  missing: list[str] = []
  if require_fa4_fwd:
    try:
      _ = get_fa4_fwd()
    except Exception as e:
      missing.append(f"flash_attn.cute.interface._flash_attn_fwd ({e})")

  try:
    flashmla = get_flashmla_sm100_module()
    if not hasattr(flashmla, "dense_prefill_bwd"):
      missing.append("flashmla_sm100.dense_prefill_bwd")
    if require_flashmla_fwd and not hasattr(flashmla, "dense_prefill_fwd"):
      missing.append("flashmla_sm100.dense_prefill_fwd")
  except Exception as e:
    missing.append(f"nmoe.csrc.flashmla_sm100 ({e})")

  return missing
