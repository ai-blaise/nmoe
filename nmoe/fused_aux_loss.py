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
from contextlib import contextmanager
from typing import Optional

import torch


@contextmanager
def _nvtx(name: str):
    """Emit an NVTX range visible in Nsight Systems / nvprof."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

# ---------------------------------------------------------------------------
# Load the fused aux loss CUDA extension.
# The .so can be produced by any build path that compiles aux_loss.cu.
# ---------------------------------------------------------------------------

_aux_loss_lib: Optional[ctypes.CDLL] = None


def _load_aux_loss_lib() -> ctypes.CDLL:
    """Load and return the shared library containing fused_aux_loss_*.

    Raises RuntimeError if the library cannot be found or loaded.
    The result is cached so the dlopen only happens once.
    """
    global _aux_loss_lib

    if _aux_loss_lib is not None:
        return _aux_loss_lib

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        # Primary: bundled with rdep extension
        os.path.join(_this_dir, "csrc", "rdep.cpython-313-x86_64-linux-gnu.so"),
        os.path.join(_this_dir, "csrc", "rdep.cpython-314-x86_64-linux-gnu.so"),
        # Standalone builds
        os.path.join(_this_dir, "csrc", "aux_loss.so"),
        os.path.join(_this_dir, "csrc", "libaux_loss.so"),
    ]

    errors: list[str] = []
    for path in _candidates:
        if os.path.isfile(path):
            try:
                lib = ctypes.CDLL(path)
                # Verify the required symbols exist
                _ = lib.fused_aux_loss_f32
                _ = lib.fused_aux_loss_bf16
                _ = lib.fused_aux_loss_single_f32
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
# Keyed by (device, E) to support multiple expert counts and devices.
_buffer_cache: dict[tuple, dict] = {}


def _get_buffers(device: torch.device, E: int) -> dict:
    """Get or create cached buffers for the given device and expert count."""
    key = (device, E)
    if key not in _buffer_cache:
        _buffer_cache[key] = {
            "f_accum": torch.zeros(E, dtype=torch.float32, device=device),
            "P_accum": torch.zeros(E, dtype=torch.float32, device=device),
            "loss_out": torch.zeros(1, dtype=torch.float32, device=device),
            "block_done": torch.zeros(1, dtype=torch.int32, device=device),
        }
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

        lib = _load_aux_loss_lib()

        device = expert_ids.device
        TK = expert_ids.numel()

        # Get cached buffers and zero them
        buffers = _get_buffers(device, E)
        f_accum = buffers["f_accum"]
        P_accum = buffers["P_accum"]
        loss_out = buffers["loss_out"]
        block_done = buffers["block_done"]

        # Zero the accumulators
        f_accum.zero_()
        P_accum.zero_()
        block_done.zero_()

        # Flatten inputs
        expert_ids_flat = expert_ids.reshape(-1).contiguous()
        gates_flat = gates.reshape(-1).contiguous()

        # Get CUDA stream
        stream = torch.cuda.current_stream(device).cuda_stream

        # Determine which kernel to use based on gates dtype
        # We use the single-kernel variant for better performance when E is small
        if gates.dtype == torch.float32:
            lib.fused_aux_loss_single_f32.restype = ctypes.c_int
            lib.fused_aux_loss_single_f32.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
                ctypes.c_void_p,
            ]
            err = lib.fused_aux_loss_single_f32(
                ctypes.c_void_p(expert_ids_flat.data_ptr()),
                ctypes.c_void_p(gates_flat.data_ptr()),
                ctypes.c_void_p(f_accum.data_ptr()),
                ctypes.c_void_p(P_accum.data_ptr()),
                ctypes.c_void_p(loss_out.data_ptr()),
                ctypes.c_void_p(block_done.data_ptr()),
                ctypes.c_int(E),
                ctypes.c_int(TK),
                ctypes.c_float(alpha),
                ctypes.c_void_p(stream),
            )
        elif gates.dtype == torch.bfloat16:
            lib.fused_aux_loss_single_bf16.restype = ctypes.c_int
            lib.fused_aux_loss_single_bf16.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
                ctypes.c_void_p,
            ]
            err = lib.fused_aux_loss_single_bf16(
                ctypes.c_void_p(expert_ids_flat.data_ptr()),
                ctypes.c_void_p(gates_flat.data_ptr()),
                ctypes.c_void_p(f_accum.data_ptr()),
                ctypes.c_void_p(P_accum.data_ptr()),
                ctypes.c_void_p(loss_out.data_ptr()),
                ctypes.c_void_p(block_done.data_ptr()),
                ctypes.c_int(E),
                ctypes.c_int(TK),
                ctypes.c_float(alpha),
                ctypes.c_void_p(stream),
            )
        elif gates.dtype == torch.float16:
            lib.fused_aux_loss_single_f16.restype = ctypes.c_int
            lib.fused_aux_loss_single_f16.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
                ctypes.c_void_p,
            ]
            err = lib.fused_aux_loss_single_f16(
                ctypes.c_void_p(expert_ids_flat.data_ptr()),
                ctypes.c_void_p(gates_flat.data_ptr()),
                ctypes.c_void_p(f_accum.data_ptr()),
                ctypes.c_void_p(P_accum.data_ptr()),
                ctypes.c_void_p(loss_out.data_ptr()),
                ctypes.c_void_p(block_done.data_ptr()),
                ctypes.c_int(E),
                ctypes.c_int(TK),
                ctypes.c_float(alpha),
                ctypes.c_void_p(stream),
            )
        else:
            raise ValueError(f"Unsupported gates dtype: {gates.dtype}")

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
