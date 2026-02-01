"""Tests for CUDA error checking utilities."""

import pytest
import torch


class TestCudaErrorClasses:
    """Test CUDA error exception classes."""

    def test_cuda_error_basic(self):
        """Test CudaError exception."""
        from nmoe.cuda_errors import CudaError

        err = CudaError("Test error")
        assert str(err) == "Test error"
        assert err.operation is None
        assert err.cuda_error is None
        assert err.device is None

    def test_cuda_error_with_context(self):
        """Test CudaError with context."""
        from nmoe.cuda_errors import CudaError

        err = CudaError(
            "Test error",
            operation="test_op",
            cuda_error="out of memory",
            device=0,
        )
        assert err.operation == "test_op"
        assert err.cuda_error == "out of memory"
        assert err.device == 0

    def test_nvshmem_error(self):
        """Test NvshmemError is a CudaError subclass."""
        from nmoe.cuda_errors import NvshmemError, CudaError

        err = NvshmemError("NVSHMEM failed")
        assert isinstance(err, CudaError)
        assert isinstance(err, RuntimeError)

    def test_rdep_error(self):
        """Test RdepError is a CudaError subclass."""
        from nmoe.cuda_errors import RdepError, CudaError

        err = RdepError("RDEP dispatch failed", operation="dispatch")
        assert isinstance(err, CudaError)
        assert err.operation == "dispatch"


class TestCudaErrorContext:
    """Test cuda_error_context context manager."""

    def test_context_no_error(self):
        """Test context manager with no errors."""
        from nmoe.cuda_errors import cuda_error_context

        # Should not raise
        with cuda_error_context("test_operation"):
            x = 1 + 1
            assert x == 2

    def test_context_catches_runtime_error(self):
        """Test context manager catches RuntimeError with CUDA in message."""
        from nmoe.cuda_errors import cuda_error_context, CudaError

        with pytest.raises(CudaError) as exc_info:
            with cuda_error_context("test_op"):
                raise RuntimeError("CUDA error: out of memory")

        assert "test_op" in str(exc_info.value)
        assert exc_info.value.operation == "test_op"

    def test_context_passes_non_cuda_errors(self):
        """Test context manager passes through non-CUDA errors."""
        from nmoe.cuda_errors import cuda_error_context

        with pytest.raises(ValueError):
            with cuda_error_context("test_op"):
                raise ValueError("Not a CUDA error")


class TestValidationHelpers:
    """Test tensor validation helpers."""

    def test_validate_tensor_device_cuda(self):
        """Test validate_tensor_device for CUDA tensors."""
        from nmoe.cuda_errors import validate_tensor_device

        if torch.cuda.is_available():
            cuda_tensor = torch.randn(10, device="cuda")
            # Should not raise
            validate_tensor_device(cuda_tensor, "cuda", "test_tensor")

            # Should raise for CPU tensor
            cpu_tensor = torch.randn(10, device="cpu")
            with pytest.raises(ValueError, match="must be on CUDA"):
                validate_tensor_device(cpu_tensor, "cuda", "test_tensor")

    def test_validate_tensor_device_cpu(self):
        """Test validate_tensor_device for CPU tensors."""
        from nmoe.cuda_errors import validate_tensor_device

        cpu_tensor = torch.randn(10, device="cpu")
        # Should not raise
        validate_tensor_device(cpu_tensor, "cpu", "test_tensor")

        if torch.cuda.is_available():
            cuda_tensor = torch.randn(10, device="cuda")
            with pytest.raises(ValueError, match="must be on CPU"):
                validate_tensor_device(cuda_tensor, "cpu", "test_tensor")

    def test_validate_tensor_contiguous(self):
        """Test validate_tensor_contiguous."""
        from nmoe.cuda_errors import validate_tensor_contiguous

        # Contiguous tensor should pass
        tensor = torch.randn(10, 10)
        validate_tensor_contiguous(tensor, "test_tensor")

        # Non-contiguous tensor should fail
        non_contig = tensor.t()
        assert not non_contig.is_contiguous()
        with pytest.raises(ValueError, match="must be contiguous"):
            validate_tensor_contiguous(non_contig, "test_tensor")

    def test_validate_tensor_dtype(self):
        """Test validate_tensor_dtype."""
        from nmoe.cuda_errors import validate_tensor_dtype

        tensor = torch.randn(10, dtype=torch.float32)
        # Should not raise
        validate_tensor_dtype(tensor, torch.float32, "test_tensor")

        # Wrong dtype should fail
        with pytest.raises(ValueError, match="must have dtype"):
            validate_tensor_dtype(tensor, torch.bfloat16, "test_tensor")


class TestMemoryInfo:
    """Test CUDA memory info utilities."""

    def test_get_cuda_memory_info_no_cuda(self):
        """Test get_cuda_memory_info when CUDA unavailable."""
        from nmoe.cuda_errors import get_cuda_memory_info

        info = get_cuda_memory_info()
        # Should return dict even without CUDA
        assert isinstance(info, dict)
        if not torch.cuda.is_available():
            assert info["available"] is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_cuda_memory_info_with_cuda(self):
        """Test get_cuda_memory_info with CUDA."""
        from nmoe.cuda_errors import get_cuda_memory_info

        info = get_cuda_memory_info()
        assert info["available"] is True
        assert "allocated" in info
        assert "reserved" in info
        assert info["allocated"] >= 0
        assert info["reserved"] >= 0


class TestErrorContextFormatting:
    """Test error context formatting."""

    def test_format_cuda_error_context_basic(self):
        """Test format_cuda_error_context basic usage."""
        from nmoe.cuda_errors import format_cuda_error_context

        context = format_cuda_error_context("test_op")
        assert "test_op" in context
        assert "CUDA Error Context" in context

    def test_format_cuda_error_context_with_kwargs(self):
        """Test format_cuda_error_context with additional context."""
        from nmoe.cuda_errors import format_cuda_error_context

        context = format_cuda_error_context(
            "test_op",
            batch_size=32,
            hidden_dim=512,
        )
        assert "batch_size: 32" in context
        assert "hidden_dim: 512" in context

    def test_format_cuda_error_context_with_tensor(self):
        """Test format_cuda_error_context with tensor."""
        from nmoe.cuda_errors import format_cuda_error_context

        tensor = torch.randn(10, 20)
        context = format_cuda_error_context(
            "test_op",
            input_tensor=tensor,
        )
        assert "input_tensor" in context
        assert "shape=" in context
        assert "dtype=" in context


class TestDecorator:
    """Test with_cuda_error_check decorator."""

    def test_decorator_no_error(self):
        """Test decorator with function that doesn't error."""
        from nmoe.cuda_errors import with_cuda_error_check

        @with_cuda_error_check("test_func")
        def my_func(x, y):
            return x + y

        result = my_func(1, 2)
        assert result == 3

    def test_decorator_uses_function_name(self):
        """Test decorator uses function name when operation not specified."""
        from nmoe.cuda_errors import with_cuda_error_check, CudaError

        @with_cuda_error_check()
        def my_cuda_operation():
            raise RuntimeError("CUDA error: test")

        with pytest.raises(CudaError) as exc_info:
            my_cuda_operation()

        assert "my_cuda_operation" in str(exc_info.value)


class TestCheckCudaErrors:
    """Test check_cuda_errors functions."""

    def test_check_cuda_errors_no_cuda(self):
        """Test check_cuda_errors when CUDA unavailable."""
        from nmoe.cuda_errors import check_cuda_errors

        # Should not raise when CUDA is not available
        if not torch.cuda.is_available():
            check_cuda_errors("test")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_check_cuda_errors_with_cuda(self):
        """Test check_cuda_errors with CUDA available."""
        from nmoe.cuda_errors import check_cuda_errors

        # Create a tensor to ensure CUDA is initialized
        _ = torch.randn(10, device="cuda")

        # Should not raise when no error
        check_cuda_errors("test")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_check_cuda_errors_sync(self):
        """Test check_cuda_errors_sync."""
        from nmoe.cuda_errors import check_cuda_errors_sync

        # Create a tensor to ensure CUDA is initialized
        _ = torch.randn(10, device="cuda")

        # Should not raise when no error
        check_cuda_errors_sync("test")
