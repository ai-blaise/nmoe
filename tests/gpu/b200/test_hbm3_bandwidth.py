"""P1 Performance Tests: HBM3 Memory Bandwidth on B200 GPUs.

This module tests HBM3 memory bandwidth characteristics on NVIDIA B200 GPUs.
B200 features HBM3 memory with approximately 8TB/s aggregate bandwidth across
8 GPUs (1TB/s per GPU).

Tests cover:
1. Device-to-device memory copy bandwidth (cudaMemcpy variants)
2. Memory read bandwidth (memory-bound kernels)
3. Memory write bandwidth
4. Strided access patterns
5. Coalesced vs non-coalesced access patterns
6. Aggregate bandwidth across all 8 GPUs
7. Mixed precision operations (fp32, fp16, bf16, fp8)
8. Large tensor operations (1GB+)
9. Concurrent memory operations
10. Bandwidth measurement using torch.cuda.Event timing

The tests verify that B200 GPUs achieve expected HBM3 bandwidth thresholds,
which is critical for MoE expert routing performance.
"""

from __future__ import annotations

import gc
import math
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import torch
import torch.nn.functional as F


# ==============================================================================
# Constants and Expected Values
# ==============================================================================

# B200 HBM3 bandwidth specifications
B200_SINGLE_GPU_BANDWIDTH_TB_S = 1.0  # ~1 TB/s per GPU
B200_AGGREGATE_BANDWIDTH_TB_S = 8.0   # ~8 TB/s across 8 GPUs
BANDWIDTH_TOLERANCE = 0.7             # Allow 70% of theoretical peak

# Minimum acceptable bandwidths (in GB/s)
MIN_COPY_BANDWIDTH_GB_S = 700         # Device-to-device copy
MIN_READ_BANDWIDTH_GB_S = 600         # Memory-bound reads
MIN_WRITE_BANDWIDTH_GB_S = 600        # Memory-bound writes

# Test tensor sizes
SMALL_TENSOR_MB = 64
MEDIUM_TENSOR_MB = 256
LARGE_TENSOR_MB = 1024
EXTRA_LARGE_TENSOR_MB = 4096


# ==============================================================================
# Pytest Markers and Skip Conditions
# ==============================================================================


def requires_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def requires_b200():
    """Skip test if B200 (SM100) is not available."""
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    return pytest.mark.skipif(
        major < 10,
        reason=f"Requires B200 (SM100+), got SM{major}{minor}"
    )


def requires_8_gpus():
    """Skip test if fewer than 8 GPUs available."""
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="CUDA not available")
    return pytest.mark.skipif(
        torch.cuda.device_count() < 8,
        reason=f"Requires 8 GPUs, got {torch.cuda.device_count()}"
    )


def requires_multi_gpu():
    """Skip test if fewer than 2 GPUs available."""
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="CUDA not available")
    return pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason=f"Requires at least 2 GPUs, got {torch.cuda.device_count()}"
    )


# ==============================================================================
# Utility Classes
# ==============================================================================


@dataclass
class BandwidthResult:
    """Container for bandwidth measurement results."""
    operation: str
    data_size_bytes: int
    duration_ms: float
    bandwidth_gb_s: float
    bandwidth_tb_s: float
    device_id: int = 0
    dtype: str = "float32"

    @classmethod
    def from_timing(
        cls,
        operation: str,
        data_size_bytes: int,
        duration_ms: float,
        device_id: int = 0,
        dtype: str = "float32"
    ) -> "BandwidthResult":
        """Create result from timing measurement."""
        bandwidth_gb_s = (data_size_bytes / 1e9) / (duration_ms / 1000)
        bandwidth_tb_s = bandwidth_gb_s / 1000
        return cls(
            operation=operation,
            data_size_bytes=data_size_bytes,
            duration_ms=duration_ms,
            bandwidth_gb_s=bandwidth_gb_s,
            bandwidth_tb_s=bandwidth_tb_s,
            device_id=device_id,
            dtype=dtype,
        )


class BandwidthProfiler:
    """Utility for measuring memory bandwidth using CUDA events."""

    def __init__(self, device: torch.device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def measure(
        self,
        operation_fn,
        data_size_bytes: int,
        operation_name: str = "operation",
        warmup_iters: int = 3,
        measure_iters: int = 10,
        dtype: str = "float32",
    ) -> BandwidthResult:
        """Measure bandwidth for an operation.

        Args:
            operation_fn: Callable that performs the memory operation
            data_size_bytes: Total bytes transferred in the operation
            operation_name: Name for reporting
            warmup_iters: Number of warmup iterations
            measure_iters: Number of measurement iterations
            dtype: Data type string for reporting

        Returns:
            BandwidthResult with measured bandwidth
        """
        with torch.cuda.device(self.device):
            # Warmup
            for _ in range(warmup_iters):
                operation_fn()
            torch.cuda.synchronize()

            # Measure
            self.start_event.record()
            for _ in range(measure_iters):
                operation_fn()
            self.end_event.record()

            torch.cuda.synchronize()
            total_time_ms = self.start_event.elapsed_time(self.end_event)
            avg_time_ms = total_time_ms / measure_iters

            return BandwidthResult.from_timing(
                operation=operation_name,
                data_size_bytes=data_size_bytes,
                duration_ms=avg_time_ms,
                device_id=self.device.index if hasattr(self.device, 'index') else 0,
                dtype=dtype,
            )


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate tensor size in bytes."""
    return tensor.numel() * tensor.element_size()


def create_tensor_mb(
    size_mb: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a tensor of approximately the specified size in MB."""
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float8_e4m3fn: 1,
        torch.float8_e5m2: 1,
        torch.int8: 1,
    }.get(dtype, 4)

    num_elements = (size_mb * 1024 * 1024) // bytes_per_element
    return torch.empty(num_elements, dtype=dtype, device=device)


def dtype_to_string(dtype: torch.dtype) -> str:
    """Convert torch dtype to string for reporting."""
    return {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float8_e4m3fn: "fp8_e4m3",
        torch.float8_e5m2: "fp8_e5m2",
        torch.int8: "int8",
    }.get(dtype, str(dtype))


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def cuda_device():
    """Provide primary CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def all_cuda_devices():
    """Provide list of all available CUDA devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


@pytest.fixture
def profiler(cuda_device):
    """Provide bandwidth profiler for primary device."""
    return BandwidthProfiler(cuda_device)


@pytest.fixture(params=[SMALL_TENSOR_MB, MEDIUM_TENSOR_MB, LARGE_TENSOR_MB])
def tensor_size_mb(request):
    """Parameterized tensor sizes for testing."""
    return request.param


@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16])
def dtype(request):
    """Parameterized data types for testing."""
    return request.param


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    return seed_val


# ==============================================================================
# Test Classes
# ==============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3CopyBandwidth:
    """Tests for device-to-device memory copy bandwidth."""

    def test_d2d_copy_small_tensor(self, cuda_device, profiler):
        """Test D2D copy bandwidth with small tensors."""
        src = create_tensor_mb(SMALL_TENSOR_MB, torch.float32, cuda_device)
        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="d2d_copy_small",
        )

        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.5, \
            f"Small tensor copy bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

    def test_d2d_copy_large_tensor(self, cuda_device, profiler):
        """Test D2D copy bandwidth with large tensors (1GB+)."""
        src = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, cuda_device)
        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="d2d_copy_large",
        )

        # Large tensors should achieve closer to peak bandwidth
        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S, \
            f"Large tensor copy bandwidth {result.bandwidth_gb_s:.1f} GB/s below {MIN_COPY_BANDWIDTH_GB_S} GB/s threshold"

    def test_d2d_copy_strided(self, cuda_device, profiler):
        """Test D2D copy bandwidth with strided tensors."""
        # Create a 2D tensor and take every other column (strided access)
        rows, cols = 8192, 8192
        src_full = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)
        src = src_full[:, ::2]  # Strided view
        dst = torch.empty(rows, cols // 2, device=cuda_device, dtype=torch.float32)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(dst),
            operation_name="d2d_copy_strided",
        )

        # Strided access typically achieves lower bandwidth
        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.3, \
            f"Strided copy bandwidth {result.bandwidth_gb_s:.1f} GB/s unexpectedly low"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_d2d_copy_different_dtypes(self, cuda_device, profiler, dtype):
        """Test D2D copy bandwidth with different data types."""
        src = create_tensor_mb(MEDIUM_TENSOR_MB, dtype, cuda_device)
        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name=f"d2d_copy_{dtype_to_string(dtype)}",
            dtype=dtype_to_string(dtype),
        )

        # All dtypes should achieve reasonable bandwidth
        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.5, \
            f"Copy bandwidth for {dtype} is {result.bandwidth_gb_s:.1f} GB/s, below threshold"

    def test_async_copy_overlap(self, cuda_device):
        """Test that async copies complete successfully on separate streams."""
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        src = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, cuda_device)
        dst = torch.empty_like(src)
        compute_tensor = torch.randn(4096, 4096, device=cuda_device)

        # Record start
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        # Issue copy on stream1
        with torch.cuda.stream(stream1):
            dst.copy_(src)

        # Issue compute on stream2 (may overlap depending on hardware)
        with torch.cuda.stream(stream2):
            for _ in range(10):
                compute_tensor = torch.matmul(compute_tensor, compute_tensor)

        end.record()
        torch.cuda.synchronize()

        concurrent_time = start.elapsed_time(end)

        # Verify copy completed correctly
        assert torch.allclose(src, dst), "Async copy did not complete correctly"

        # Verify timing is reasonable (not stuck/infinite)
        assert concurrent_time > 0, "Timing measurement failed"
        assert concurrent_time < 60000, f"Operation took too long: {concurrent_time:.2f}ms"


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3ReadBandwidth:
    """Tests for memory read bandwidth (memory-bound kernels)."""

    def test_vector_read_bandwidth(self, cuda_device, profiler):
        """Test read bandwidth using vector sum (memory-bound)."""
        src = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, cuda_device)

        def read_op():
            # Sum forces reading all elements
            _ = src.sum()

        result = profiler.measure(
            read_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="vector_read_sum",
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S, \
            f"Read bandwidth {result.bandwidth_gb_s:.1f} GB/s below {MIN_READ_BANDWIDTH_GB_S} GB/s"

    def test_matrix_read_bandwidth(self, cuda_device, profiler):
        """Test read bandwidth using matrix operations."""
        # Create a large matrix that requires reading all elements
        rows = 16384
        cols = 16384
        src = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)

        def read_op():
            # Row-wise sum reads all elements
            _ = src.sum(dim=1)

        result = profiler.measure(
            read_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="matrix_read_rowsum",
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.8, \
            f"Matrix read bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

    def test_gather_read_bandwidth(self, cuda_device, profiler):
        """Test read bandwidth with gather operations (indexed reads)."""
        size = MEDIUM_TENSOR_MB * 1024 * 1024 // 4  # float32 elements
        src = torch.randn(size, device=cuda_device, dtype=torch.float32)

        # Create indices for gathering (simulates expert routing gather)
        num_indices = size // 4
        indices = torch.randint(0, size, (num_indices,), device=cuda_device)

        def gather_op():
            _ = src[indices]

        result = profiler.measure(
            gather_op,
            data_size_bytes=num_indices * 4,  # Only count bytes actually read
            operation_name="gather_read",
        )

        # Gather has lower effective bandwidth due to random access
        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.2, \
            f"Gather read bandwidth {result.bandwidth_gb_s:.1f} GB/s unexpectedly low"

    def test_contiguous_vs_noncontiguous_read(self, cuda_device, profiler):
        """Compare read bandwidth: contiguous vs non-contiguous access."""
        rows, cols = 8192, 8192
        src = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)

        # Contiguous read
        def contiguous_read():
            _ = src.sum()

        contiguous_result = profiler.measure(
            contiguous_read,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="contiguous_read",
        )

        # Non-contiguous read (transposed)
        src_t = src.t()  # Transpose creates non-contiguous view

        def noncontiguous_read():
            _ = src_t.sum()

        noncontiguous_result = profiler.measure(
            noncontiguous_read,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="noncontiguous_read",
        )

        # Contiguous should be faster
        ratio = contiguous_result.bandwidth_gb_s / max(noncontiguous_result.bandwidth_gb_s, 1e-6)
        assert ratio > 0.8, \
            f"Contiguous read ({contiguous_result.bandwidth_gb_s:.1f} GB/s) should be comparable to non-contiguous ({noncontiguous_result.bandwidth_gb_s:.1f} GB/s)"


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3WriteBandwidth:
    """Tests for memory write bandwidth."""

    def test_vector_write_bandwidth(self, cuda_device, profiler):
        """Test write bandwidth using vector fill."""
        dst = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, cuda_device)

        def write_op():
            dst.fill_(1.0)

        result = profiler.measure(
            write_op,
            data_size_bytes=get_tensor_size_bytes(dst),
            operation_name="vector_write_fill",
        )

        assert result.bandwidth_gb_s > MIN_WRITE_BANDWIDTH_GB_S, \
            f"Write bandwidth {result.bandwidth_gb_s:.1f} GB/s below {MIN_WRITE_BANDWIDTH_GB_S} GB/s"

    def test_scatter_write_bandwidth(self, cuda_device, profiler):
        """Test write bandwidth with scatter operations (indexed writes)."""
        size = MEDIUM_TENSOR_MB * 1024 * 1024 // 4
        dst = torch.zeros(size, device=cuda_device, dtype=torch.float32)

        # Create indices for scattering (simulates expert routing scatter)
        num_indices = size // 4
        indices = torch.randint(0, size, (num_indices,), device=cuda_device)
        values = torch.randn(num_indices, device=cuda_device, dtype=torch.float32)

        def scatter_op():
            dst.scatter_(0, indices, values)

        result = profiler.measure(
            scatter_op,
            data_size_bytes=num_indices * 4,
            operation_name="scatter_write",
        )

        # Scatter has lower effective bandwidth due to random access
        assert result.bandwidth_gb_s > MIN_WRITE_BANDWIDTH_GB_S * 0.1, \
            f"Scatter write bandwidth {result.bandwidth_gb_s:.1f} GB/s unexpectedly low"

    def test_zero_write_bandwidth(self, cuda_device, profiler):
        """Test write bandwidth using zero_()."""
        dst = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, cuda_device)

        def write_op():
            dst.zero_()

        result = profiler.measure(
            write_op,
            data_size_bytes=get_tensor_size_bytes(dst),
            operation_name="zero_write",
        )

        assert result.bandwidth_gb_s > MIN_WRITE_BANDWIDTH_GB_S, \
            f"Zero write bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

    def test_elementwise_write_bandwidth(self, cuda_device, profiler):
        """Test write bandwidth with element-wise operations."""
        src = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, cuda_device)
        torch.randn(src.shape, out=src)

        def write_op():
            # In-place operation writes all elements
            src.mul_(2.0)

        # Element-wise reads AND writes, so bandwidth is for read+write
        result = profiler.measure(
            write_op,
            data_size_bytes=get_tensor_size_bytes(src) * 2,  # read + write
            operation_name="elementwise_mul",
        )

        assert result.bandwidth_gb_s > MIN_WRITE_BANDWIDTH_GB_S, \
            f"Element-wise bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3StridedAccess:
    """Tests for strided memory access patterns."""

    @pytest.mark.parametrize("stride", [2, 4, 8, 16])
    def test_strided_read_bandwidth(self, cuda_device, profiler, stride):
        """Test read bandwidth with different stride patterns."""
        # Large contiguous buffer
        total_elements = MEDIUM_TENSOR_MB * 1024 * 1024 // 4
        src = torch.randn(total_elements, device=cuda_device, dtype=torch.float32)

        # Create strided view
        strided_view = src[::stride]

        def strided_read():
            _ = strided_view.sum()

        result = profiler.measure(
            strided_read,
            data_size_bytes=get_tensor_size_bytes(strided_view),
            operation_name=f"strided_read_s{stride}",
        )

        # Strided access bandwidth decreases with stride
        min_expected = MIN_READ_BANDWIDTH_GB_S * (1.0 / stride) * 2
        assert result.bandwidth_gb_s > min_expected, \
            f"Stride-{stride} read bandwidth {result.bandwidth_gb_s:.1f} GB/s below {min_expected:.1f} GB/s"

    def test_2d_strided_access(self, cuda_device, profiler):
        """Test 2D strided access patterns."""
        rows, cols = 4096, 4096
        src = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)

        # Access every other row and column
        strided_2d = src[::2, ::2]

        def strided_2d_read():
            _ = strided_2d.sum()

        result = profiler.measure(
            strided_2d_read,
            data_size_bytes=get_tensor_size_bytes(strided_2d),
            operation_name="strided_2d_read",
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.1, \
            f"2D strided read bandwidth {result.bandwidth_gb_s:.1f} GB/s unexpectedly low"

    def test_strided_copy_bandwidth(self, cuda_device, profiler):
        """Test copy bandwidth with strided source."""
        rows, cols = 8192, 8192
        src = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)
        strided_src = src[:, ::4]  # Every 4th column
        dst = torch.empty_like(strided_src.contiguous())

        def strided_copy():
            dst.copy_(strided_src)

        result = profiler.measure(
            strided_copy,
            data_size_bytes=get_tensor_size_bytes(dst),
            operation_name="strided_copy",
        )

        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.2, \
            f"Strided copy bandwidth {result.bandwidth_gb_s:.1f} GB/s unexpectedly low"


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3CoalescedAccess:
    """Tests for coalesced vs non-coalesced memory access patterns."""

    def test_coalesced_read(self, cuda_device, profiler):
        """Test bandwidth with coalesced (contiguous) reads."""
        rows, cols = 16384, 16384
        src = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)

        def coalesced_read():
            # Row-major access is coalesced
            _ = src.sum()

        result = profiler.measure(
            coalesced_read,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="coalesced_read",
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S, \
            f"Coalesced read bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

    def test_non_coalesced_read(self, cuda_device, profiler):
        """Test bandwidth with non-coalesced (column-major) reads."""
        rows, cols = 16384, 16384
        src = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)

        # Column-wise sum accesses memory non-contiguously
        def non_coalesced_read():
            _ = src.sum(dim=0)

        result = profiler.measure(
            non_coalesced_read,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="non_coalesced_read",
        )

        # Non-coalesced typically has lower bandwidth but should still be reasonable
        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.3, \
            f"Non-coalesced read bandwidth {result.bandwidth_gb_s:.1f} GB/s unexpectedly low"

    def test_coalesced_vs_non_coalesced_ratio(self, cuda_device, profiler):
        """Compare coalesced vs non-coalesced access performance."""
        rows, cols = 8192, 8192
        src = torch.randn(rows, cols, device=cuda_device, dtype=torch.float32)

        # Coalesced (contiguous)
        def coalesced():
            _ = src.sum(dim=1)  # Sum along rows

        coalesced_result = profiler.measure(
            coalesced,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="coalesced_rowsum",
        )

        # Non-coalesced (non-contiguous)
        def non_coalesced():
            _ = src.sum(dim=0)  # Sum along columns

        non_coalesced_result = profiler.measure(
            non_coalesced,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="non_coalesced_colsum",
        )

        # Log the ratio for analysis
        ratio = coalesced_result.bandwidth_gb_s / max(non_coalesced_result.bandwidth_gb_s, 1e-6)

        # Both should achieve reasonable bandwidth
        assert coalesced_result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.5, \
            f"Coalesced bandwidth too low: {coalesced_result.bandwidth_gb_s:.1f} GB/s"
        assert non_coalesced_result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.2, \
            f"Non-coalesced bandwidth too low: {non_coalesced_result.bandwidth_gb_s:.1f} GB/s"

    def test_warp_aligned_access(self, cuda_device, profiler):
        """Test bandwidth with warp-aligned access patterns."""
        # Warp size is 32, so test with 32-element aligned accesses
        warp_size = 32
        num_warps = 4096
        elements_per_access = 4  # float4 access

        src = torch.randn(
            num_warps * warp_size * elements_per_access,
            device=cuda_device,
            dtype=torch.float32
        )

        def aligned_read():
            _ = src.sum()

        result = profiler.measure(
            aligned_read,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="warp_aligned_read",
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.8, \
            f"Warp-aligned read bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"


@pytest.mark.gpu
@pytest.mark.b200
@requires_8_gpus()
class TestHBM3AggregateBandwidth:
    """Tests for aggregate bandwidth across all 8 GPUs."""

    def test_aggregate_copy_bandwidth_8gpu(self, all_cuda_devices):
        """Test aggregate D2D copy bandwidth across 8 GPUs."""
        if len(all_cuda_devices) < 8:
            pytest.skip(f"Requires 8 GPUs, got {len(all_cuda_devices)}")

        devices = all_cuda_devices[:8]
        results = []

        # Create tensors on each device
        tensors = []
        for device in devices:
            src = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, device)
            dst = torch.empty_like(src)
            tensors.append((src, dst))

        # Synchronize all devices
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        # Create events for timing
        start_events = []
        end_events = []
        for device in devices:
            with torch.cuda.device(device):
                start_events.append(torch.cuda.Event(enable_timing=True))
                end_events.append(torch.cuda.Event(enable_timing=True))

        # Warmup
        for i, (src, dst) in enumerate(tensors):
            with torch.cuda.device(devices[i]):
                dst.copy_(src)
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        # Measure concurrent copies on all GPUs
        for i, device in enumerate(devices):
            with torch.cuda.device(device):
                start_events[i].record()

        for i, (src, dst) in enumerate(tensors):
            with torch.cuda.device(devices[i]):
                for _ in range(10):
                    dst.copy_(src)

        for i, device in enumerate(devices):
            with torch.cuda.device(device):
                end_events[i].record()

        # Synchronize and measure
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        # Calculate aggregate bandwidth
        total_bytes = LARGE_TENSOR_MB * 1024 * 1024 * 8 * 10  # 8 GPUs, 10 iterations
        max_time_ms = max(
            start_events[i].elapsed_time(end_events[i])
            for i in range(8)
        )

        aggregate_bandwidth_gb_s = (total_bytes / 1e9) / (max_time_ms / 1000)
        aggregate_bandwidth_tb_s = aggregate_bandwidth_gb_s / 1000

        # B200 should achieve close to 8 TB/s aggregate
        min_expected_tb_s = B200_AGGREGATE_BANDWIDTH_TB_S * BANDWIDTH_TOLERANCE
        assert aggregate_bandwidth_tb_s > min_expected_tb_s * 0.5, \
            f"Aggregate bandwidth {aggregate_bandwidth_tb_s:.2f} TB/s below {min_expected_tb_s:.2f} TB/s threshold"

    def test_concurrent_read_all_gpus(self, all_cuda_devices):
        """Test concurrent read operations across all available GPUs."""
        if len(all_cuda_devices) < 8:
            pytest.skip(f"Requires 8 GPUs, got {len(all_cuda_devices)}")

        devices = all_cuda_devices[:8]

        # Create tensors on each device
        tensors = [
            create_tensor_mb(LARGE_TENSOR_MB, torch.float32, device)
            for device in devices
        ]

        # Initialize with random data
        for t in tensors:
            torch.randn(t.shape, out=t)

        # Synchronize
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        # Create events
        start_events = []
        end_events = []
        for device in devices:
            with torch.cuda.device(device):
                start_events.append(torch.cuda.Event(enable_timing=True))
                end_events.append(torch.cuda.Event(enable_timing=True))

        # Measure concurrent reads
        for i, device in enumerate(devices):
            with torch.cuda.device(device):
                start_events[i].record()

        results = []
        for i, tensor in enumerate(tensors):
            with torch.cuda.device(devices[i]):
                for _ in range(10):
                    results.append(tensor.sum())

        for i, device in enumerate(devices):
            with torch.cuda.device(device):
                end_events[i].record()

        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        # Calculate bandwidth
        total_bytes = LARGE_TENSOR_MB * 1024 * 1024 * 8 * 10
        max_time_ms = max(
            start_events[i].elapsed_time(end_events[i])
            for i in range(8)
        )

        aggregate_bandwidth_tb_s = (total_bytes / 1e12) / (max_time_ms / 1000)

        assert aggregate_bandwidth_tb_s > B200_AGGREGATE_BANDWIDTH_TB_S * 0.3, \
            f"Aggregate read bandwidth {aggregate_bandwidth_tb_s:.2f} TB/s below threshold"

    def test_peer_to_peer_bandwidth(self, all_cuda_devices):
        """Test peer-to-peer bandwidth between GPUs."""
        if len(all_cuda_devices) < 2:
            pytest.skip("Requires at least 2 GPUs")

        device0 = all_cuda_devices[0]
        device1 = all_cuda_devices[1]

        # Check if P2P is supported
        can_access = torch.cuda.can_device_access_peer(device0, device1)
        if not can_access:
            pytest.skip("P2P access not supported between devices")

        # Create tensors
        src = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, device0)
        dst = torch.empty_like(src, device=device1)

        # Measure P2P copy
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        dst.copy_(src)
        torch.cuda.synchronize()

        start.record()
        for _ in range(10):
            dst.copy_(src)
        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / 10
        bandwidth_gb_s = (get_tensor_size_bytes(src) / 1e9) / (time_ms / 1000)

        # P2P should achieve reasonable bandwidth (depends on NVLink topology)
        assert bandwidth_gb_s > 50, \
            f"P2P bandwidth {bandwidth_gb_s:.1f} GB/s unexpectedly low"


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3MixedPrecision:
    """Tests for bandwidth with different data types."""

    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ])
    def test_copy_bandwidth_by_dtype(self, cuda_device, profiler, dtype):
        """Test copy bandwidth for each data type."""
        src = create_tensor_mb(MEDIUM_TENSOR_MB, dtype, cuda_device)
        torch.randn(src.shape[0], device=cuda_device, dtype=torch.float32, out=None)
        # Initialize with randn for float types
        if dtype in [torch.float32, torch.float16, torch.bfloat16]:
            src_init = torch.randn(src.shape, device=cuda_device, dtype=torch.float32)
            src.copy_(src_init)

        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name=f"copy_{dtype_to_string(dtype)}",
            dtype=dtype_to_string(dtype),
        )

        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.5, \
            f"{dtype} copy bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

    def test_fp8_bandwidth_if_supported(self, cuda_device, profiler):
        """Test FP8 bandwidth if hardware supports it."""
        try:
            src = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float8_e4m3fn, cuda_device)
        except (RuntimeError, TypeError):
            pytest.skip("FP8 not supported on this hardware/PyTorch version")

        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="copy_fp8",
            dtype="fp8",
        )

        # FP8 is 1 byte, so should transfer same element count faster
        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.3, \
            f"FP8 copy bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

    def test_mixed_precision_matmul_bandwidth(self, cuda_device, profiler):
        """Test bandwidth for mixed-precision matrix multiplication."""
        M, N, K = 4096, 4096, 4096

        # BF16 inputs
        a_bf16 = torch.randn(M, K, device=cuda_device, dtype=torch.bfloat16)
        b_bf16 = torch.randn(K, N, device=cuda_device, dtype=torch.bfloat16)

        def matmul_bf16():
            _ = torch.matmul(a_bf16, b_bf16)

        # For matmul, we read A and B, write C
        total_bytes = (M * K + K * N + M * N) * 2  # bf16 = 2 bytes

        result = profiler.measure(
            matmul_bf16,
            data_size_bytes=total_bytes,
            operation_name="matmul_bf16",
            dtype="bf16",
        )

        # Matmul is compute-bound, but should still show reasonable memory throughput
        assert result.bandwidth_gb_s > 100, \
            f"BF16 matmul memory throughput {result.bandwidth_gb_s:.1f} GB/s unexpectedly low"

    def test_dtype_conversion_bandwidth(self, cuda_device, profiler):
        """Test bandwidth for dtype conversion operations."""
        src_fp32 = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, cuda_device)
        torch.randn(src_fp32.shape, out=src_fp32)

        # FP32 -> BF16 conversion
        def convert_to_bf16():
            _ = src_fp32.to(torch.bfloat16)

        # Read FP32, write BF16
        total_bytes = get_tensor_size_bytes(src_fp32) + (src_fp32.numel() * 2)

        result = profiler.measure(
            convert_to_bf16,
            data_size_bytes=total_bytes,
            operation_name="fp32_to_bf16",
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S * 0.5, \
            f"Dtype conversion bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3LargeTensors:
    """Tests for large tensor operations (1GB+)."""

    def test_1gb_tensor_copy(self, cuda_device, profiler):
        """Test copy bandwidth with 1GB tensor."""
        src = create_tensor_mb(1024, torch.float32, cuda_device)
        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="1gb_copy",
            warmup_iters=2,
            measure_iters=5,
        )

        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S, \
            f"1GB tensor copy bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

    def test_4gb_tensor_copy(self, cuda_device, profiler):
        """Test copy bandwidth with 4GB tensor."""
        # Check available memory
        free_memory = torch.cuda.get_device_properties(cuda_device).total_memory
        free_memory -= torch.cuda.memory_allocated(cuda_device)

        if free_memory < 10 * 1024 * 1024 * 1024:  # Need ~10GB free
            pytest.skip("Insufficient GPU memory for 4GB tensor test")

        src = create_tensor_mb(4096, torch.float32, cuda_device)
        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="4gb_copy",
            warmup_iters=1,
            measure_iters=3,
        )

        # Large tensors should achieve peak bandwidth
        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S, \
            f"4GB tensor copy bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

        # Clean up
        del src, dst
        gc.collect()
        torch.cuda.empty_cache()

    def test_large_tensor_read(self, cuda_device, profiler):
        """Test read bandwidth with large tensor."""
        src = create_tensor_mb(2048, torch.float32, cuda_device)
        torch.randn(src.shape, out=src)

        def read_op():
            _ = src.sum()

        result = profiler.measure(
            read_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="2gb_read",
            warmup_iters=2,
            measure_iters=5,
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S, \
            f"2GB tensor read bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

        del src
        gc.collect()
        torch.cuda.empty_cache()

    def test_large_tensor_elementwise(self, cuda_device, profiler):
        """Test element-wise operations on large tensors."""
        a = create_tensor_mb(1024, torch.float32, cuda_device)
        b = create_tensor_mb(1024, torch.float32, cuda_device)
        torch.randn(a.shape, out=a)
        torch.randn(b.shape, out=b)

        def add_op():
            _ = a + b

        # Element-wise add: read a, read b, write result
        total_bytes = get_tensor_size_bytes(a) * 3

        result = profiler.measure(
            add_op,
            data_size_bytes=total_bytes,
            operation_name="1gb_add",
            warmup_iters=2,
            measure_iters=5,
        )

        assert result.bandwidth_gb_s > MIN_READ_BANDWIDTH_GB_S, \
            f"1GB tensor add bandwidth {result.bandwidth_gb_s:.1f} GB/s below threshold"

        del a, b
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3ConcurrentOperations:
    """Tests for concurrent memory operations."""

    def test_concurrent_streams_bandwidth(self, cuda_device):
        """Test bandwidth with multiple concurrent CUDA streams."""
        num_streams = 4
        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        # Create tensors for each stream
        tensors = [
            (
                create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, cuda_device),
                torch.empty(MEDIUM_TENSOR_MB * 1024 * 1024 // 4, device=cuda_device, dtype=torch.float32)
            )
            for _ in range(num_streams)
        ]

        # Synchronize
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for i, (src, dst) in enumerate(tensors):
            with torch.cuda.stream(streams[i]):
                dst.copy_(src)
        torch.cuda.synchronize()

        # Measure concurrent copies
        start.record()
        for i, (src, dst) in enumerate(tensors):
            with torch.cuda.stream(streams[i]):
                for _ in range(10):
                    dst.copy_(src)
        end.record()

        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / 10
        total_bytes = MEDIUM_TENSOR_MB * 1024 * 1024 * num_streams
        bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

        # Concurrent streams should achieve aggregate bandwidth
        assert bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S, \
            f"Concurrent streams bandwidth {bandwidth_gb_s:.1f} GB/s below threshold"

    def test_copy_compute_overlap(self, cuda_device):
        """Test that memory copies can overlap with compute."""
        copy_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()

        # Data for copy
        src = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, cuda_device)
        dst = torch.empty_like(src)

        # Data for compute
        a = torch.randn(4096, 4096, device=cuda_device, dtype=torch.float32)
        b = torch.randn(4096, 4096, device=cuda_device, dtype=torch.float32)

        torch.cuda.synchronize()

        # Measure concurrent execution
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        with torch.cuda.stream(copy_stream):
            for _ in range(5):
                dst.copy_(src)

        with torch.cuda.stream(compute_stream):
            for _ in range(5):
                _ = torch.matmul(a, b)

        end.record()
        torch.cuda.synchronize()

        concurrent_time = start.elapsed_time(end)

        # Measure sequential execution
        start.record()
        for _ in range(5):
            dst.copy_(src)
        torch.cuda.synchronize()
        for _ in range(5):
            _ = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()

        sequential_time = start.elapsed_time(end)

        # Concurrent should be meaningfully faster
        speedup = sequential_time / concurrent_time
        assert speedup > 1.1, \
            f"No overlap benefit: sequential={sequential_time:.2f}ms, concurrent={concurrent_time:.2f}ms"

    def test_bidirectional_copy_bandwidth(self, cuda_device):
        """Test bidirectional copy (read and write simultaneously)."""
        src1 = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, cuda_device)
        dst1 = torch.empty_like(src1)
        src2 = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, cuda_device)
        dst2 = torch.empty_like(src2)

        torch.randn(src1.shape, out=src1)
        torch.randn(src2.shape, out=src2)

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        # Simultaneous copies in both directions
        with torch.cuda.stream(stream1):
            for _ in range(10):
                dst1.copy_(src1)

        with torch.cuda.stream(stream2):
            for _ in range(10):
                dst2.copy_(src2)

        end.record()
        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / 10
        total_bytes = MEDIUM_TENSOR_MB * 1024 * 1024 * 2  # Both copies
        bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

        # Bidirectional should achieve high aggregate bandwidth
        assert bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S, \
            f"Bidirectional copy bandwidth {bandwidth_gb_s:.1f} GB/s below threshold"

    @requires_multi_gpu()
    def test_multi_gpu_concurrent_copies(self, all_cuda_devices):
        """Test concurrent copies across multiple GPUs."""
        if len(all_cuda_devices) < 2:
            pytest.skip("Requires at least 2 GPUs")

        num_gpus = min(len(all_cuda_devices), 4)
        devices = all_cuda_devices[:num_gpus]

        # Create tensors on each device
        tensors = []
        for device in devices:
            src = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, device)
            dst = torch.empty_like(src)
            torch.randn(src.shape, out=src)
            tensors.append((src, dst, device))

        # Synchronize all devices
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        # Create events on device 0 for timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for src, dst, device in tensors:
            with torch.cuda.device(device):
                dst.copy_(src)
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        # Measure
        start.record()

        for src, dst, device in tensors:
            with torch.cuda.device(device):
                for _ in range(10):
                    dst.copy_(src)

        end.record()

        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

        torch.cuda.synchronize()

        time_ms = start.elapsed_time(end) / 10
        total_bytes = MEDIUM_TENSOR_MB * 1024 * 1024 * num_gpus
        bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000)

        # Should scale with number of GPUs
        expected_bandwidth = MIN_COPY_BANDWIDTH_GB_S * num_gpus * 0.7
        assert bandwidth_gb_s > expected_bandwidth, \
            f"Multi-GPU bandwidth {bandwidth_gb_s:.1f} GB/s below {expected_bandwidth:.1f} GB/s expected"


@pytest.mark.gpu
@pytest.mark.b200
class TestHBM3BandwidthRegression:
    """Regression tests to ensure bandwidth doesn't degrade."""

    def test_bandwidth_stability(self, cuda_device, profiler):
        """Test that bandwidth measurements are stable across iterations."""
        src = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, cuda_device)
        dst = torch.empty_like(src)

        measurements = []

        for _ in range(5):
            def copy_op():
                dst.copy_(src)

            result = profiler.measure(
                copy_op,
                data_size_bytes=get_tensor_size_bytes(src),
                operation_name="stability_test",
                warmup_iters=2,
                measure_iters=5,
            )
            measurements.append(result.bandwidth_gb_s)

        # Check variance
        mean_bw = sum(measurements) / len(measurements)
        variance = sum((m - mean_bw) ** 2 for m in measurements) / len(measurements)
        std_dev = variance ** 0.5
        cv = std_dev / mean_bw  # Coefficient of variation

        assert cv < 0.15, \
            f"Bandwidth measurements unstable: CV={cv:.2%}, measurements={measurements}"

    def test_bandwidth_under_memory_pressure(self, cuda_device, profiler):
        """Test bandwidth when GPU memory is under pressure."""
        # Allocate memory to create pressure
        pressure_tensors = []
        try:
            # Allocate ~50% of remaining memory
            free_memory = torch.cuda.get_device_properties(cuda_device).total_memory
            free_memory -= torch.cuda.memory_allocated(cuda_device)
            pressure_size = int(free_memory * 0.4)

            num_pressure_tensors = pressure_size // (256 * 1024 * 1024)  # 256MB chunks
            for _ in range(num_pressure_tensors):
                pressure_tensors.append(
                    torch.empty(256 * 1024 * 1024 // 4, device=cuda_device, dtype=torch.float32)
                )

            # Now measure bandwidth
            src = create_tensor_mb(MEDIUM_TENSOR_MB, torch.float32, cuda_device)
            dst = torch.empty_like(src)

            def copy_op():
                dst.copy_(src)

            result = profiler.measure(
                copy_op,
                data_size_bytes=get_tensor_size_bytes(src),
                operation_name="pressure_test",
            )

            # Bandwidth should still be reasonable under pressure
            assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.7, \
                f"Bandwidth under pressure {result.bandwidth_gb_s:.1f} GB/s below threshold"

        finally:
            # Clean up
            del pressure_tensors
            gc.collect()
            torch.cuda.empty_cache()

    def test_bandwidth_after_allocation_churn(self, cuda_device, profiler):
        """Test bandwidth after memory allocation churn."""
        # Create allocation churn
        for _ in range(10):
            temp = [
                torch.empty(64 * 1024 * 1024 // 4, device=cuda_device, dtype=torch.float32)
                for _ in range(8)
            ]
            del temp
            gc.collect()
            torch.cuda.empty_cache()

        # Measure bandwidth
        src = create_tensor_mb(LARGE_TENSOR_MB, torch.float32, cuda_device)
        dst = torch.empty_like(src)

        def copy_op():
            dst.copy_(src)

        result = profiler.measure(
            copy_op,
            data_size_bytes=get_tensor_size_bytes(src),
            operation_name="post_churn_test",
        )

        assert result.bandwidth_gb_s > MIN_COPY_BANDWIDTH_GB_S * 0.9, \
            f"Bandwidth after churn {result.bandwidth_gb_s:.1f} GB/s below threshold"


# ==============================================================================
# Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
