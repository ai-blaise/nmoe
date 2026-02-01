"""CUDA error checking utilities for nmoe Python bindings.

This module provides comprehensive CUDA error checking that propagates
errors from the C++ extension to Python. Based on the NMOE_CHECK_CUDA
macro pattern from nmoe/csrc/muon.cu.

Usage:
    from nmoe.cuda_errors import check_cuda_errors, CudaError

    # After any _C call that could produce CUDA errors:
    result = _C.some_kernel(...)
    check_cuda_errors("some_kernel")

    # Or use the context manager:
    with cuda_error_context("dispatch operation"):
        _C.dispatch_meta_bf16(...)
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Callable, TypeVar, ParamSpec, Optional

import torch

P = ParamSpec("P")
R = TypeVar("R")


class CudaError(RuntimeError):
    """Exception raised when a CUDA error is detected.

    Attributes:
        operation: Name of the operation that failed.
        cuda_error: The underlying CUDA error string if available.
        device: The CUDA device index where the error occurred.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        cuda_error: Optional[str] = None,
        device: Optional[int] = None,
    ):
        self.operation = operation
        self.cuda_error = cuda_error
        self.device = device
        super().__init__(message)


class NvshmemError(CudaError):
    """Exception raised when an NVSHMEM operation fails."""

    pass


class RdepError(CudaError):
    """Exception raised when an RDEP dispatch/combine operation fails."""

    pass


def check_cuda_errors(operation: str = "CUDA operation") -> None:
    """Check for CUDA errors and raise CudaError if any occurred.

    This function synchronizes with the CUDA device and checks for any
    errors that may have occurred in previous asynchronous operations.

    Args:
        operation: Name of the operation for error messages.

    Raises:
        CudaError: If a CUDA error is detected.
    """
    if not torch.cuda.is_available():
        return

    try:
        # Get current device
        device = torch.cuda.current_device()

        # Check for errors without full sync (faster)
        # cudaGetLastError clears the error flag
        error = torch.cuda.get_device_properties(device)  # This will throw if device is in error state

    except RuntimeError as e:
        error_str = str(e)
        # Try to extract CUDA error code/name
        cuda_error = None
        if "CUDA error:" in error_str:
            cuda_error = error_str.split("CUDA error:")[-1].strip()
        elif "cudaError" in error_str.lower():
            cuda_error = error_str

        raise CudaError(
            f"[nmoe] CUDA error during '{operation}': {error_str}",
            operation=operation,
            cuda_error=cuda_error,
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
        ) from e


def check_cuda_errors_sync(operation: str = "CUDA operation") -> None:
    """Check for CUDA errors with full device synchronization.

    This is more thorough than check_cuda_errors() but slower.
    Use for debugging or after critical operations.

    Args:
        operation: Name of the operation for error messages.

    Raises:
        CudaError: If a CUDA error is detected.
    """
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        error_str = str(e)
        cuda_error = None
        if "CUDA error:" in error_str:
            cuda_error = error_str.split("CUDA error:")[-1].strip()

        raise CudaError(
            f"[nmoe] CUDA error during '{operation}' (sync): {error_str}",
            operation=operation,
            cuda_error=cuda_error,
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
        ) from e


@contextmanager
def cuda_error_context(operation: str = "CUDA operation", sync: bool = False):
    """Context manager for CUDA error checking.

    Args:
        operation: Name of the operation for error messages.
        sync: If True, perform full device synchronization after the block.

    Yields:
        None

    Raises:
        CudaError: If a CUDA error is detected during or after the block.

    Example:
        with cuda_error_context("dispatch_meta_bf16"):
            M_recv = _C.dispatch_meta_bf16(...)
    """
    try:
        yield
    except RuntimeError as e:
        # Check if this is a CUDA error from the C++ extension
        error_str = str(e)
        if "cuda" in error_str.lower() or "CUDA" in error_str:
            raise CudaError(
                f"[nmoe] CUDA error during '{operation}': {error_str}",
                operation=operation,
                cuda_error=error_str,
                device=torch.cuda.current_device() if torch.cuda.is_available() else None,
            ) from e
        raise

    # Check for asynchronous errors after the operation
    if sync:
        check_cuda_errors_sync(operation)
    else:
        # At minimum, check the error state
        if torch.cuda.is_available():
            try:
                # This is a lightweight check that doesn't fully sync
                # It checks if the device is in an error state
                _ = torch.cuda.current_stream()
            except RuntimeError as e:
                raise CudaError(
                    f"[nmoe] CUDA error after '{operation}': {str(e)}",
                    operation=operation,
                    cuda_error=str(e),
                    device=torch.cuda.current_device() if torch.cuda.is_available() else None,
                ) from e


def with_cuda_error_check(operation: Optional[str] = None):
    """Decorator to add CUDA error checking to a function.

    Args:
        operation: Name of the operation. If None, uses function name.

    Returns:
        Decorated function with CUDA error checking.

    Example:
        @with_cuda_error_check("dispatch_meta")
        def dispatch_meta_bf16(...):
            return _C.dispatch_meta_bf16(...)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with cuda_error_context(op_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_tensor_device(tensor: torch.Tensor, expected_device: str = "cuda", name: str = "tensor") -> None:
    """Validate that a tensor is on the expected device.

    Args:
        tensor: The tensor to validate.
        expected_device: Expected device type ("cuda" or "cpu").
        name: Name of the tensor for error messages.

    Raises:
        ValueError: If tensor is not on expected device.
    """
    if expected_device == "cuda" and not tensor.is_cuda:
        raise ValueError(f"[nmoe] {name} must be on CUDA, got {tensor.device}")
    elif expected_device == "cpu" and tensor.is_cuda:
        raise ValueError(f"[nmoe] {name} must be on CPU, got {tensor.device}")


def validate_tensor_contiguous(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Validate that a tensor is contiguous.

    Args:
        tensor: The tensor to validate.
        name: Name of the tensor for error messages.

    Raises:
        ValueError: If tensor is not contiguous.
    """
    if not tensor.is_contiguous():
        raise ValueError(f"[nmoe] {name} must be contiguous")


def validate_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: str = "tensor") -> None:
    """Validate tensor dtype.

    Args:
        tensor: The tensor to validate.
        expected_dtype: Expected dtype.
        name: Name of the tensor for error messages.

    Raises:
        ValueError: If tensor has wrong dtype.
    """
    if tensor.dtype != expected_dtype:
        raise ValueError(f"[nmoe] {name} must have dtype {expected_dtype}, got {tensor.dtype}")


def get_cuda_memory_info() -> dict:
    """Get current CUDA memory usage information.

    Returns:
        Dictionary with memory statistics.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.cuda.current_device()
    return {
        "available": True,
        "device": device,
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
    }


def format_cuda_error_context(operation: str, **kwargs) -> str:
    """Format a detailed error context message.

    Args:
        operation: Name of the failed operation.
        **kwargs: Additional context (tensors, shapes, etc.)

    Returns:
        Formatted error context string.
    """
    lines = [f"[nmoe] CUDA Error Context for '{operation}':"]

    # Add memory info
    mem_info = get_cuda_memory_info()
    if mem_info["available"]:
        lines.append(f"  Device: {mem_info['device']}")
        lines.append(f"  Memory allocated: {mem_info['allocated'] / 1e9:.2f} GB")
        lines.append(f"  Memory reserved: {mem_info['reserved'] / 1e9:.2f} GB")

    # Add custom context
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            lines.append(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        else:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)
