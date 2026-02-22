"""Fused MoE auxiliary loss computation.

Replaces ~12 separate PyTorch kernel launches with a single fused CUDA kernel
for computing the GShard-style load balancing auxiliary loss.

The auxiliary loss is:
    aux_loss = alpha * E * sum(f_e * P_e)
where:
    f_e = fraction of tokens routed to expert e = count(expert_ids == e) / (T * K)
    P_e = mean gate probability for expert e = sum(gates where expert_ids == e) / (T * K)

This module provides:
    fused_aux_loss(expert_ids, gates, E, alpha) -> scalar tensor

Performance:
    - Original: ~12 PyTorch ops per call (696 kernel launches @ 58 calls/step)
    - Fused: 1-2 kernel launches per call (58-116 launches/step)
    - Reduction: 6-12x fewer kernel launches
"""

import os
import ctypes
import stat
from contextlib import nullcontext
from typing import Optional

import torch


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in ("1", "true", "True")


_NVTX_ENABLED = (
    _env_flag("NMOE_NVTX", "0")
    and torch.cuda.is_available()
    and hasattr(torch.cuda, "nvtx")
    and hasattr(torch.cuda.nvtx, "range_push")
)
_AUX_LOSS_ALLOW_STANDALONE = _env_flag("NMOE_AUX_LOSS_ALLOW_STANDALONE", "0")
_AUX_LOSS_STREAM_CACHE_LIMIT = max(1, int(os.getenv("NMOE_AUX_LOSS_STREAM_CACHE_LIMIT", "8")))
_EXPECTED_RDEP_ABI = 1
_AUX_LOSS_REQUIRED_CAPS = (1 << 0) | (1 << 1)  # internal reset + no two-kernel fallback
_NULL_NVTX_CTX = nullcontext()


def _nvtx(name: str):
    """Emit an NVTX range visible in Nsight Systems / nvprof."""
    if _NVTX_ENABLED:
        return torch.cuda.nvtx.range(name)
    return _NULL_NVTX_CTX

# ---------------------------------------------------------------------------
# Load the fused aux loss CUDA extension.
# The .so can be produced by any build path that compiles aux_loss.cu.
# ---------------------------------------------------------------------------

_aux_loss_lib: Optional[ctypes.CDLL] = None
_aux_loss_sig_initialized: bool = False


def _init_aux_loss_signatures(lib: ctypes.CDLL) -> None:
    """Initialize ctypes signatures once to avoid per-call overhead."""
    global _aux_loss_sig_initialized
    if _aux_loss_sig_initialized:
        return

    # Keep full aux-loss symbol coverage required by runtime paths.
    _ = lib.fused_aux_loss_f32
    _ = lib.fused_aux_loss_bf16
    _ = lib.fused_aux_loss_f16
    _ = lib.fused_aux_loss_single_f32
    _ = lib.fused_aux_loss_single_bf16
    _ = lib.fused_aux_loss_single_f16
    _ = lib.fused_aux_loss_single_caps

    sig = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p,
    ]

    lib.fused_aux_loss_single_f32.restype = ctypes.c_int
    lib.fused_aux_loss_single_f32.argtypes = sig
    lib.fused_aux_loss_single_bf16.restype = ctypes.c_int
    lib.fused_aux_loss_single_bf16.argtypes = sig
    lib.fused_aux_loss_single_f16.restype = ctypes.c_int
    lib.fused_aux_loss_single_f16.argtypes = sig
    lib.fused_aux_loss_single_caps.restype = ctypes.c_int
    lib.fused_aux_loss_single_caps.argtypes = []

    caps = int(lib.fused_aux_loss_single_caps())
    missing = _AUX_LOSS_REQUIRED_CAPS & (~caps)
    if missing != 0:
        raise RuntimeError(
            "Loaded aux-loss kernel does not satisfy required no-fallback capabilities "
            f"(required=0x{_AUX_LOSS_REQUIRED_CAPS:x}, got=0x{caps:x}). Rebuild nmoe/csrc."
        )

    _aux_loss_sig_initialized = True


def _resolve_primary_rdep_path() -> str:
    from nmoe.csrc import rdep as _rdep_mod  # type: ignore

    path = getattr(_rdep_mod, "__file__", None)
    if not (isinstance(path, str) and path):
        raise RuntimeError("nmoe.csrc.rdep did not expose a valid extension path.")
    abi_fn = getattr(_rdep_mod, "abi_version", None)
    if not callable(abi_fn):
        raise RuntimeError("nmoe.csrc.rdep missing abi_version(); rebuild extension.")
    abi = int(abi_fn())
    if abi != _EXPECTED_RDEP_ABI:
        raise RuntimeError(
            f"nmoe.csrc.rdep ABI mismatch: expected {_EXPECTED_RDEP_ABI}, got {abi}. Rebuild extension."
        )
    real = os.path.realpath(path)
    mode = os.stat(real).st_mode
    if (mode & stat.S_IWGRP) or (mode & stat.S_IWOTH):
        raise RuntimeError(f"Refusing writable extension path: {real}")
    return real


def _load_aux_loss_lib() -> ctypes.CDLL:
    """Load and return the shared library containing fused_aux_loss_*.

    Raises RuntimeError if the library cannot be found or loaded.
    The result is cached so the dlopen only happens once.
    """
    global _aux_loss_lib

    if _aux_loss_lib is not None:
        return _aux_loss_lib

    _candidates: list[str] = []
    try:
        _candidates.append(_resolve_primary_rdep_path())
    except Exception as e:
        raise RuntimeError(
            "Failed to resolve primary nmoe.csrc.rdep extension for fused aux loss."
        ) from e
    # Standalone aux-loss libs are forbidden in production to avoid stale-binary drift.
    if _AUX_LOSS_ALLOW_STANDALONE:
        raise RuntimeError(
            "NMOE_AUX_LOSS_ALLOW_STANDALONE=1 is forbidden in production. "
            "Only the loaded nmoe.csrc.rdep extension may provide fused aux-loss kernels."
        )

    errors: list[str] = []
    for path in _candidates:
        if os.path.isfile(path):
            try:
                mode = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_LOCAL", 0)
                lib = ctypes.CDLL(path, mode=mode) if mode else ctypes.CDLL(path)
                _init_aux_loss_signatures(lib)
                _aux_loss_lib = lib
                return lib
            except (OSError, AttributeError) as exc:
                errors.append(f"  {path}: {exc}")

    msg = (
        "Failed to load fused aux loss CUDA library. "
        "The fused CUDA kernel is required; there is no fallback.\n"
        "Searched paths:\n"
        + "\n".join(f"  {p}" for p in _candidates)
    )
    if errors:
        msg += "\nErrors:\n" + "\n".join(errors)
    raise RuntimeError(msg)


# Persistent buffers to avoid allocation overhead on every call.
# Keyed by (device, E, stream_id) to avoid cross-stream races.
_buffer_cache: dict[tuple, dict] = {}


def _prune_buffer_cache(device: torch.device, limit: int) -> None:
    dev_keys = [k for k in _buffer_cache.keys() if isinstance(k, tuple) and len(k) >= 1 and k[0] == device]
    while len(dev_keys) > limit:
        _buffer_cache.pop(dev_keys[0], None)
        dev_keys.pop(0)


def _get_buffers(device: torch.device, E: int, stream_id: int) -> dict:
    """Get or create cached buffers for the given device and expert count."""
    key = (device, E, int(stream_id))
    if key not in _buffer_cache:
        _buffer_cache[key] = {
            "f_accum": torch.zeros(E, dtype=torch.float32, device=device),
            "P_accum": torch.zeros(E, dtype=torch.float32, device=device),
            "loss_out": torch.zeros(1, dtype=torch.float32, device=device),
            "block_done": torch.zeros(1, dtype=torch.int32, device=device),
        }
        _prune_buffer_cache(device, _AUX_LOSS_STREAM_CACHE_LIMIT)
    # Buffers are stream-affine by key, so we avoid per-call record_stream overhead.
    return _buffer_cache[key]


def fused_aux_loss(
    expert_ids: torch.Tensor,
    gates: torch.Tensor,
    E: int,
    alpha: float,
) -> torch.Tensor:
    """Compute fused MoE auxiliary loss using a single CUDA kernel.

    This function computes the GShard-style load balancing loss:
        aux_loss = alpha * E * sum(f_e * P_e)

    where f_e is the fraction of tokens routed to expert e, and P_e is
    the mean gate probability for expert e.

    Args:
        expert_ids: [T, K] int32 tensor of selected expert indices
        gates: [T, K] tensor of gating weights (float32, float16, or bfloat16)
        E: Number of experts
        alpha: Auxiliary loss coefficient

    Returns:
        Scalar tensor containing the auxiliary loss (float32)

    Note:
        This function requires the fused CUDA kernel to be compiled and
        available. There is no PyTorch fallback.
    """
    with _nvtx("fused_aux_loss/compute"):
        if alpha == 0.0:
            return gates.new_zeros((), dtype=torch.float32)

        if E <= 0:
            raise RuntimeError(f"E must be > 0 for fused_aux_loss, got E={E}")
        if not expert_ids.is_cuda or not gates.is_cuda:
            raise RuntimeError("fused_aux_loss requires CUDA tensors.")
        if expert_ids.device != gates.device:
            raise RuntimeError(
                f"expert_ids and gates must be on the same device (got {expert_ids.device} vs {gates.device})"
            )
        if expert_ids.dtype != torch.int32:
            raise RuntimeError(f"expert_ids must be int32 on production no-fallback path, got {expert_ids.dtype}")
        if expert_ids.numel() != gates.numel():
            raise RuntimeError(
                f"expert_ids/gates element count mismatch: {expert_ids.numel()} vs {gates.numel()}"
            )
        if not expert_ids.is_contiguous():
            raise RuntimeError("expert_ids must be contiguous on production no-fallback path.")
        if not gates.is_contiguous():
            raise RuntimeError("gates must be contiguous on production no-fallback path.")

        lib = _load_aux_loss_lib()

        device = expert_ids.device
        TK = expert_ids.numel()
        stream_obj = torch.cuda.current_stream(device)
        stream = int(stream_obj.cuda_stream)

        # Get cached buffers. The CUDA entrypoint resets accumulators internally.
        buffers = _get_buffers(device, E, stream)
        f_accum = buffers["f_accum"]
        P_accum = buffers["P_accum"]
        loss_out = buffers["loss_out"]
        block_done = buffers["block_done"]

        # Flatten contiguous inputs without materializing copies.
        expert_ids_flat = expert_ids.view(-1)
        gates_flat = gates.view(-1)

        # Determine which kernel to use based on gates dtype
        # We use the single-kernel variant for better performance when E is small
        if gates.dtype == torch.float32:
            fn = lib.fused_aux_loss_single_f32
        elif gates.dtype == torch.bfloat16:
            fn = lib.fused_aux_loss_single_bf16
        elif gates.dtype == torch.float16:
            fn = lib.fused_aux_loss_single_f16
        else:
            raise ValueError(f"Unsupported gates dtype: {gates.dtype}")

        err = fn(
            expert_ids_flat.data_ptr(),
            gates_flat.data_ptr(),
            f_accum.data_ptr(),
            P_accum.data_ptr(),
            loss_out.data_ptr(),
            block_done.data_ptr(),
            E,
            TK,
            alpha,
            stream,
        )

        if err != 0:
            raise RuntimeError(f"fused_aux_loss CUDA kernel returned error {err}")

        # Return a scalar tensor (clone to avoid returning a view into the buffer)
        return loss_out[0].clone()


class FusedAuxLoss(torch.autograd.Function):
    """Autograd-compatible fused auxiliary loss.

    This is a custom autograd function that wraps fused_aux_loss().
    The auxiliary loss does not propagate gradients (it's a regularization term
    that doesn't backprop through the gates in the standard formulation).

    However, if gradient flow is needed, this class can be extended to compute
    d(aux_loss)/d(gates) = alpha * E * f_e / (T*K) for each gate position.
    """

    @staticmethod
    def forward(
        ctx,
        expert_ids: torch.Tensor,
        gates: torch.Tensor,
        E: int,
        alpha: float,
    ) -> torch.Tensor:
        """Forward pass computing the auxiliary loss."""
        with _nvtx("fused_aux_loss/autograd_forward"):
            # The aux loss is a regularization term that typically does not
            # backprop through gates (the gradient would encourage imbalanced routing).
            # If backprop is needed, implement it in backward().
            return fused_aux_loss(expert_ids, gates, E, alpha)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - no gradients for aux loss by default."""
        # Return None for all inputs (no gradient flow)
        return None, None, None, None


def fused_aux_loss_autograd(
    expert_ids: torch.Tensor,
    gates: torch.Tensor,
    E: int,
    alpha: float,
) -> torch.Tensor:
    """Autograd-compatible wrapper for fused_aux_loss.

    This version can be used in training when the loss needs to be part of
    a computation graph, though gradients do not flow through it.
    """
    with _nvtx("fused_aux_loss/autograd_wrapper"):
        return FusedAuxLoss.apply(expert_ids, gates, E, alpha)
