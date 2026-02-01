"""Comprehensive NVLink Bandwidth Tests for B200 8-GPU Node.

This module tests NVLink bandwidth saturation and communication patterns
on NVIDIA B200 GPUs with NVLink 5.0 connectivity. B200 8-GPU nodes provide
~900 GB/s aggregate bidirectional bandwidth.

Run with:
    torchrun --nproc_per_node=8 -m pytest tests/gpu/b200/test_nvlink_bandwidth.py -v

Test coverage:
1. Point-to-point transfers between adjacent and non-adjacent GPUs
2. All-to-all collective bandwidth
3. Ring topology allreduce performance
4. Bidirectional transfer saturation
5. Aggregate system bandwidth verification
6. Small message latency measurements

Requirements:
    - 8x NVIDIA B200 GPUs with NVLink 5.0
    - PyTorch with NCCL backend
    - torch.cuda.Event for accurate timing
"""

from __future__ import annotations

import functools
import itertools
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import pytest
import torch
import torch.distributed as dist


# ==============================================================================
# Constants and Configuration
# ==============================================================================

# B200 NVLink 5.0 specifications
NVLINK_5_UNIDIRECTIONAL_BW_GBPS = 112.5  # GB/s per link unidirectional
NVLINK_5_BIDIRECTIONAL_BW_GBPS = 225.0   # GB/s per link bidirectional
NUM_NVLINKS_PER_GPU_PAIR = 4             # 4 NVLinks between adjacent GPUs
B200_AGGREGATE_BW_TARGET_GBPS = 900.0    # Total aggregate target

# Expected bandwidth per GPU pair (4 links x 112.5 GB/s unidirectional)
EXPECTED_P2P_BW_GBPS = NVLINK_5_UNIDIRECTIONAL_BW_GBPS * NUM_NVLINKS_PER_GPU_PAIR

# Efficiency thresholds (fraction of theoretical peak)
P2P_EFFICIENCY_THRESHOLD = 0.70          # 70% of peak for P2P
ALLTOALL_EFFICIENCY_THRESHOLD = 0.60     # 60% of peak for all-to-all
ALLREDUCE_EFFICIENCY_THRESHOLD = 0.65    # 65% of peak for allreduce
BIDIRECTIONAL_EFFICIENCY_THRESHOLD = 0.65

# Test configuration
DEFAULT_WARMUP_ITERS = 10
DEFAULT_MEASURE_ITERS = 50
DEFAULT_NUM_STREAMS = 4

# Tensor sizes for bandwidth testing (in bytes)
TENSOR_SIZES_BYTES = [
    1 * 1024 * 1024,        # 1 MB
    4 * 1024 * 1024,        # 4 MB
    16 * 1024 * 1024,       # 16 MB
    64 * 1024 * 1024,       # 64 MB
    256 * 1024 * 1024,      # 256 MB
    512 * 1024 * 1024,      # 512 MB
    1024 * 1024 * 1024,     # 1 GB
]

# Latency test sizes (small messages)
LATENCY_SIZES_BYTES = [
    4,                      # 4 bytes (single float)
    64,                     # 64 bytes (cache line)
    512,                    # 512 bytes
    4096,                   # 4 KB
    65536,                  # 64 KB
]


# ==============================================================================
# Pytest Markers
# ==============================================================================

pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.b200,
]


# ==============================================================================
# Data Classes for Results
# ==============================================================================

@dataclass
class BandwidthResult:
    """Result from a bandwidth measurement."""
    size_bytes: int
    duration_ms: float
    bandwidth_gbps: float
    efficiency: float = 0.0

    def __str__(self) -> str:
        size_mb = self.size_bytes / (1024 * 1024)
        return (f"Size: {size_mb:.1f} MB, "
                f"Duration: {self.duration_ms:.3f} ms, "
                f"Bandwidth: {self.bandwidth_gbps:.2f} GB/s, "
                f"Efficiency: {self.efficiency * 100:.1f}%")


@dataclass
class LatencyResult:
    """Result from a latency measurement."""
    size_bytes: int
    latency_us: float
    min_latency_us: float
    max_latency_us: float
    std_latency_us: float

    def __str__(self) -> str:
        return (f"Size: {self.size_bytes} B, "
                f"Latency: {self.latency_us:.2f} us "
                f"(min: {self.min_latency_us:.2f}, max: {self.max_latency_us:.2f}, "
                f"std: {self.std_latency_us:.2f})")


@dataclass
class AggregateResult:
    """Aggregate bandwidth result across multiple GPU pairs."""
    num_pairs: int
    total_bandwidth_gbps: float
    per_pair_bandwidth_gbps: float
    efficiency: float
    individual_results: List[BandwidthResult] = field(default_factory=list)


# ==============================================================================
# Distributed Utilities
# ==============================================================================

def get_world_size() -> int:
    """Get world size, handling both distributed and non-distributed cases."""
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    """Get rank, handling both distributed and non-distributed cases."""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """Get local rank within the node."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def init_distributed() -> bool:
    """Initialize distributed if not already done. Returns True if initialized."""
    if not dist.is_initialized():
        world_size = get_world_size()
        if world_size > 1:
            dist.init_process_group(backend="nccl")
            local_rank = get_local_rank()
            torch.cuda.set_device(local_rank)
            return True
    return dist.is_initialized()


def cleanup_distributed():
    """Cleanup distributed resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    """Synchronize all ranks."""
    if dist.is_initialized():
        dist.barrier()


def skip_if_not_multi_gpu(min_gpus: int = 8):
    """Skip test if not enough GPUs available."""
    world_size = get_world_size()
    if world_size < min_gpus:
        pytest.skip(f"Requires at least {min_gpus} GPUs, have {world_size}")


def skip_if_not_b200():
    """Skip if not running on B200 (SM100) GPUs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 10:
        pytest.skip(f"Requires B200 (SM100), have SM{capability[0]}{capability[1]}")


def requires_b200():
    """Decorator to require B200 GPUs."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            skip_if_not_b200()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_multi_gpu(min_gpus: int = 8):
    """Decorator to require minimum number of GPUs."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            skip_if_not_multi_gpu(min_gpus)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==============================================================================
# Timing Utilities with CUDA Events
# ==============================================================================

class CudaTimer:
    """High-precision timer using CUDA events."""

    def __init__(self, stream: Optional[torch.cuda.Stream] = None):
        self.stream = stream
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        """Record start time."""
        if self.stream:
            self.start_event.record(self.stream)
        else:
            self.start_event.record()

    def stop(self):
        """Record end time."""
        if self.stream:
            self.end_event.record(self.stream)
        else:
            self.end_event.record()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds. Synchronizes first."""
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def measure_bandwidth(
    transfer_fn: Callable[[], None],
    size_bytes: int,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    measure_iters: int = DEFAULT_MEASURE_ITERS,
    stream: Optional[torch.cuda.Stream] = None,
) -> BandwidthResult:
    """Measure bandwidth of a transfer operation.

    Args:
        transfer_fn: Function that performs the transfer
        size_bytes: Size of data transferred in bytes
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
        stream: CUDA stream to use for timing

    Returns:
        BandwidthResult with timing and bandwidth measurements
    """
    timer = CudaTimer(stream)

    # Warmup
    for _ in range(warmup_iters):
        transfer_fn()
    torch.cuda.synchronize()

    # Measurement
    timer.start()
    for _ in range(measure_iters):
        transfer_fn()
    timer.stop()

    total_time_ms = timer.elapsed_ms()
    avg_time_ms = total_time_ms / measure_iters

    # Calculate bandwidth: bytes / time = bytes/ms = KB/s
    # Convert to GB/s: (bytes / ms) * (1000 ms/s) / (1e9 bytes/GB)
    bandwidth_gbps = (size_bytes / avg_time_ms) * 1000 / 1e9

    return BandwidthResult(
        size_bytes=size_bytes,
        duration_ms=avg_time_ms,
        bandwidth_gbps=bandwidth_gbps,
    )


def measure_latency(
    transfer_fn: Callable[[], None],
    size_bytes: int,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    measure_iters: int = 100,
) -> LatencyResult:
    """Measure latency of a transfer operation.

    Args:
        transfer_fn: Function that performs the transfer
        size_bytes: Size of data transferred in bytes
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations

    Returns:
        LatencyResult with latency statistics
    """
    # Warmup
    for _ in range(warmup_iters):
        transfer_fn()
    torch.cuda.synchronize()

    # Measure individual iterations
    latencies_us = []
    for _ in range(measure_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        transfer_fn()
        end_event.record()
        end_event.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latencies_us.append(latency_ms * 1000)  # Convert to microseconds

    return LatencyResult(
        size_bytes=size_bytes,
        latency_us=statistics.mean(latencies_us),
        min_latency_us=min(latencies_us),
        max_latency_us=max(latencies_us),
        std_latency_us=statistics.stdev(latencies_us) if len(latencies_us) > 1 else 0.0,
    )


# ==============================================================================
# Tensor Allocation Utilities
# ==============================================================================

def create_tensor_on_device(
    size_bytes: int,
    device: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a tensor of specified size on a given device.

    Args:
        size_bytes: Size in bytes
        device: CUDA device index
        dtype: Data type for the tensor

    Returns:
        Tensor allocated on the specified device
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = size_bytes // element_size

    with torch.cuda.device(device):
        tensor = torch.randn(num_elements, dtype=dtype, device=f"cuda:{device}")

    return tensor


def get_adjacent_gpu_pairs(num_gpus: int = 8) -> List[Tuple[int, int]]:
    """Get list of adjacent GPU pairs (ring topology).

    For 8 GPUs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)
    """
    pairs = []
    for i in range(num_gpus):
        pairs.append((i, (i + 1) % num_gpus))
    return pairs


def get_all_gpu_pairs(num_gpus: int = 8) -> List[Tuple[int, int]]:
    """Get all unique GPU pairs (excluding self-pairs)."""
    pairs = []
    for i in range(num_gpus):
        for j in range(i + 1, num_gpus):
            pairs.append((i, j))
    return pairs


def check_p2p_access(src: int, dst: int) -> bool:
    """Check if P2P access is enabled between two GPUs."""
    with torch.cuda.device(src):
        return torch.cuda.can_device_access_peer(src, dst)


def enable_p2p_access_all():
    """Enable P2P access between all GPU pairs if possible."""
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                with torch.cuda.device(i):
                    if torch.cuda.can_device_access_peer(i, j):
                        try:
                            torch.cuda.enable_peer_access(j)
                        except RuntimeError:
                            # Already enabled
                            pass


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def distributed_setup():
    """Setup distributed environment for the test module."""
    init_distributed()
    enable_p2p_access_all()
    yield
    cleanup_distributed()


@pytest.fixture
def sync_barrier():
    """Fixture that provides a barrier sync function."""
    def _barrier():
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
    return _barrier


# ==============================================================================
# Test Class: Point-to-Point Bandwidth
# ==============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestNVLinkP2PBandwidth:
    """Test point-to-point NVLink bandwidth between GPU pairs."""

    @requires_multi_gpu(2)
    def test_p2p_adjacent_gpus_small(self, distributed_setup, sync_barrier):
        """Test P2P bandwidth between adjacent GPUs with small tensors."""
        rank = get_rank()
        world_size = get_world_size()

        if rank >= 2:
            sync_barrier()
            return

        # Only ranks 0 and 1 participate
        size_bytes = 1 * 1024 * 1024  # 1 MB

        if rank == 0:
            src_tensor = create_tensor_on_device(size_bytes, 0)
            dst_tensor = torch.empty_like(src_tensor)

            def transfer():
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()

            result = measure_bandwidth(transfer, size_bytes)
            result.efficiency = result.bandwidth_gbps / EXPECTED_P2P_BW_GBPS

            print(f"\nP2P Adjacent (0->1, 1MB): {result}")
            assert result.bandwidth_gbps > 0, "Bandwidth should be positive"

        sync_barrier()

    @requires_multi_gpu(2)
    @pytest.mark.parametrize("size_bytes", TENSOR_SIZES_BYTES)
    def test_p2p_adjacent_varying_sizes(self, distributed_setup, sync_barrier, size_bytes):
        """Test P2P bandwidth between adjacent GPUs with varying tensor sizes."""
        rank = get_rank()

        if rank != 0:
            sync_barrier()
            return

        # Test GPU 0 -> GPU 1 transfer
        src_device = 0
        dst_device = 1

        src_tensor = create_tensor_on_device(size_bytes, src_device)
        with torch.cuda.device(dst_device):
            dst_tensor = torch.empty_like(src_tensor, device=f"cuda:{dst_device}")

        def transfer():
            dst_tensor.copy_(src_tensor)
            torch.cuda.synchronize()

        result = measure_bandwidth(transfer, size_bytes)
        result.efficiency = result.bandwidth_gbps / EXPECTED_P2P_BW_GBPS

        size_mb = size_bytes / (1024 * 1024)
        print(f"\nP2P Adjacent (0->1, {size_mb:.0f}MB): {result}")

        # For larger transfers, expect higher efficiency
        if size_bytes >= 64 * 1024 * 1024:  # 64 MB+
            min_expected_bw = EXPECTED_P2P_BW_GBPS * P2P_EFFICIENCY_THRESHOLD
            assert result.bandwidth_gbps >= min_expected_bw * 0.5, (
                f"P2P bandwidth {result.bandwidth_gbps:.2f} GB/s below "
                f"expected {min_expected_bw:.2f} GB/s for {size_mb:.0f} MB"
            )

        sync_barrier()

    @requires_multi_gpu(8)
    def test_p2p_all_adjacent_pairs(self, distributed_setup, sync_barrier):
        """Test P2P bandwidth across all adjacent GPU pairs."""
        rank = get_rank()
        world_size = get_world_size()

        if rank != 0:
            sync_barrier()
            return

        size_bytes = 256 * 1024 * 1024  # 256 MB for meaningful measurement
        pairs = get_adjacent_gpu_pairs(min(world_size, 8))
        results = []

        for src, dst in pairs:
            if not check_p2p_access(src, dst):
                print(f"\nWarning: No P2P access between GPU {src} and {dst}")
                continue

            src_tensor = create_tensor_on_device(size_bytes, src)
            with torch.cuda.device(dst):
                dst_tensor = torch.empty_like(src_tensor, device=f"cuda:{dst}")

            def transfer():
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()

            result = measure_bandwidth(transfer, size_bytes)
            result.efficiency = result.bandwidth_gbps / EXPECTED_P2P_BW_GBPS
            results.append((src, dst, result))

            print(f"\nP2P ({src}->{dst}): {result}")

        # Check that all pairs achieve reasonable bandwidth
        for src, dst, result in results:
            min_bw = EXPECTED_P2P_BW_GBPS * P2P_EFFICIENCY_THRESHOLD * 0.5
            assert result.bandwidth_gbps >= min_bw, (
                f"P2P {src}->{dst} bandwidth {result.bandwidth_gbps:.2f} GB/s "
                f"below minimum {min_bw:.2f} GB/s"
            )

        sync_barrier()

    @requires_multi_gpu(8)
    def test_p2p_non_adjacent_gpus(self, distributed_setup, sync_barrier):
        """Test P2P bandwidth between non-adjacent GPUs (multi-hop)."""
        rank = get_rank()

        if rank != 0:
            sync_barrier()
            return

        size_bytes = 256 * 1024 * 1024

        # Test non-adjacent pairs: (0,2), (0,4), (1,5), (2,6)
        non_adjacent_pairs = [(0, 2), (0, 4), (1, 5), (2, 6)]
        results = []

        for src, dst in non_adjacent_pairs:
            if dst >= torch.cuda.device_count():
                continue
            if not check_p2p_access(src, dst):
                continue

            src_tensor = create_tensor_on_device(size_bytes, src)
            with torch.cuda.device(dst):
                dst_tensor = torch.empty_like(src_tensor, device=f"cuda:{dst}")

            def transfer():
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()

            result = measure_bandwidth(transfer, size_bytes)
            # Non-adjacent may have lower efficiency due to routing
            result.efficiency = result.bandwidth_gbps / EXPECTED_P2P_BW_GBPS
            results.append((src, dst, result))

            print(f"\nP2P Non-Adjacent ({src}->{dst}): {result}")

        # Verify we got some bandwidth (multi-hop may be slower)
        for src, dst, result in results:
            assert result.bandwidth_gbps > 10, (
                f"Non-adjacent P2P {src}->{dst} bandwidth unexpectedly low: "
                f"{result.bandwidth_gbps:.2f} GB/s"
            )

        sync_barrier()

    @requires_multi_gpu(2)
    def test_p2p_stream_isolation(self, distributed_setup, sync_barrier):
        """Test P2P transfers on separate CUDA streams."""
        rank = get_rank()

        if rank != 0:
            sync_barrier()
            return

        size_bytes = 128 * 1024 * 1024
        num_streams = 4

        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        tensors_src = [create_tensor_on_device(size_bytes, 0) for _ in range(num_streams)]

        with torch.cuda.device(1):
            tensors_dst = [
                torch.empty_like(t, device="cuda:1") for t in tensors_src
            ]

        # Warmup
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                tensors_dst[i].copy_(tensors_src[i])
        torch.cuda.synchronize()

        # Measure concurrent transfers
        start_events = [torch.cuda.Event(enable_timing=True) for _ in streams]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in streams]

        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                start_events[i].record()
                tensors_dst[i].copy_(tensors_src[i])
                end_events[i].record()

        torch.cuda.synchronize()

        # Calculate individual and aggregate bandwidth
        total_bytes = size_bytes * num_streams
        times_ms = [
            start_events[i].elapsed_time(end_events[i])
            for i in range(num_streams)
        ]
        max_time_ms = max(times_ms)

        # Aggregate bandwidth is total bytes / max time (parallel transfers)
        aggregate_bw = (total_bytes / max_time_ms) * 1000 / 1e9

        print(f"\nP2P Stream Isolation ({num_streams} streams):")
        print(f"  Individual times (ms): {[f'{t:.3f}' for t in times_ms]}")
        print(f"  Max time: {max_time_ms:.3f} ms")
        print(f"  Aggregate bandwidth: {aggregate_bw:.2f} GB/s")

        # Streams should not significantly slow each other down
        avg_time = sum(times_ms) / len(times_ms)
        for t in times_ms:
            assert t < avg_time * 2, "Stream times should be relatively consistent"

        sync_barrier()


# ==============================================================================
# Test Class: All-to-All Collective Bandwidth
# ==============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestNVLinkAllToAll:
    """Test all-to-all collective NVLink bandwidth."""

    @requires_multi_gpu(8)
    def test_alltoall_basic(self, distributed_setup, sync_barrier):
        """Test basic all-to-all collective bandwidth."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        # Each rank sends size_per_rank to every other rank
        size_per_rank = 32 * 1024 * 1024  # 32 MB per destination
        total_size = size_per_rank * world_size

        input_tensor = torch.randn(total_size // 4, dtype=torch.float32, device=device)
        output_tensor = torch.empty_like(input_tensor)

        # Split into chunks for each destination
        input_list = list(input_tensor.chunk(world_size))
        output_list = list(output_tensor.chunk(world_size))

        def all_to_all():
            dist.all_to_all(output_list, input_list)

        result = measure_bandwidth(
            all_to_all,
            total_size,  # Total bytes moved per rank
            warmup_iters=10,
            measure_iters=30,
        )

        # For all-to-all, effective bandwidth is harder to calculate
        # Each rank sends (world_size-1) * size_per_rank bytes
        effective_bytes = (world_size - 1) * size_per_rank
        effective_bw = (effective_bytes / result.duration_ms) * 1000 / 1e9

        if rank == 0:
            print(f"\nAll-to-All Basic (8 GPUs, {size_per_rank // (1024*1024)} MB/dest):")
            print(f"  Duration: {result.duration_ms:.3f} ms")
            print(f"  Effective bandwidth per rank: {effective_bw:.2f} GB/s")
            print(f"  Aggregate bandwidth: {effective_bw * world_size:.2f} GB/s")

        sync_barrier()

    @requires_multi_gpu(8)
    @pytest.mark.parametrize("size_mb", [1, 16, 64, 256])
    def test_alltoall_varying_sizes(self, distributed_setup, sync_barrier, size_mb):
        """Test all-to-all with varying message sizes."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_per_rank = size_mb * 1024 * 1024
        total_size = size_per_rank * world_size

        input_tensor = torch.randn(total_size // 4, dtype=torch.float32, device=device)
        output_tensor = torch.empty_like(input_tensor)

        input_list = list(input_tensor.chunk(world_size))
        output_list = list(output_tensor.chunk(world_size))

        def all_to_all():
            dist.all_to_all(output_list, input_list)

        result = measure_bandwidth(all_to_all, total_size)

        if rank == 0:
            effective_bytes = (world_size - 1) * size_per_rank
            effective_bw = (effective_bytes / result.duration_ms) * 1000 / 1e9
            print(f"\nAll-to-All ({size_mb} MB/dest): {effective_bw:.2f} GB/s per rank")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_alltoall_single_tensor(self, distributed_setup, sync_barrier):
        """Test all_to_all_single for contiguous tensor communication."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_per_rank = 64 * 1024 * 1024  # 64 MB per destination
        total_elements = (size_per_rank * world_size) // 4  # float32 = 4 bytes

        input_tensor = torch.randn(total_elements, dtype=torch.float32, device=device)
        output_tensor = torch.empty_like(input_tensor)

        def all_to_all_single():
            dist.all_to_all_single(output_tensor, input_tensor)

        result = measure_bandwidth(
            all_to_all_single,
            size_per_rank * world_size,
        )

        if rank == 0:
            effective_bytes = (world_size - 1) * size_per_rank
            effective_bw = (effective_bytes / result.duration_ms) * 1000 / 1e9
            print(f"\nAll-to-All Single: {effective_bw:.2f} GB/s per rank")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_alltoall_uneven_splits(self, distributed_setup, sync_barrier):
        """Test all-to-all with uneven split sizes."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        # Each rank sends different amounts to different destinations
        # Simulate MoE-like routing where some experts get more tokens
        base_size = 16 * 1024 * 1024  # 16 MB base

        # Create varying sizes: more data to adjacent ranks
        input_splits = []
        for i in range(world_size):
            if abs(i - rank) <= 1 or abs(i - rank) >= world_size - 1:
                input_splits.append(base_size * 2)  # 2x to adjacent
            else:
                input_splits.append(base_size // 2)  # 0.5x to distant

        output_splits = [input_splits[i] for i in range(world_size)]

        total_input = sum(input_splits)
        input_tensor = torch.randn(total_input // 4, dtype=torch.float32, device=device)

        # Gather output sizes from all ranks
        all_output_sizes = [None] * world_size
        dist.all_gather_object(all_output_sizes, input_splits)

        total_output = sum(all_output_sizes[i][rank] for i in range(world_size))
        output_tensor = torch.randn(total_output // 4, dtype=torch.float32, device=device)

        # Create split tensors
        input_split_sizes = [s // 4 for s in input_splits]
        output_split_sizes = [all_output_sizes[i][rank] // 4 for i in range(world_size)]

        input_tensors = list(input_tensor.split(input_split_sizes))
        output_tensors = list(output_tensor.split(output_split_sizes))

        def all_to_all_uneven():
            dist.all_to_all(output_tensors, input_tensors)

        result = measure_bandwidth(all_to_all_uneven, total_input)

        if rank == 0:
            print(f"\nAll-to-All Uneven: {result.bandwidth_gbps:.2f} GB/s")

        sync_barrier()


# ==============================================================================
# Test Class: Ring Allreduce Bandwidth
# ==============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestNVLinkRingAllreduce:
    """Test ring topology allreduce NVLink bandwidth."""

    @requires_multi_gpu(8)
    def test_allreduce_basic(self, distributed_setup, sync_barrier):
        """Test basic allreduce bandwidth."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_bytes = 256 * 1024 * 1024  # 256 MB
        tensor = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)

        def allreduce():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        result = measure_bandwidth(allreduce, size_bytes)

        # Ring allreduce theoretical bandwidth utilization:
        # 2 * (n-1)/n * size bytes transferred per GPU
        # Effective algorithm bandwidth = size / time
        # Bus bandwidth = algo_bw * 2 * (n-1) / n
        algo_bw = result.bandwidth_gbps
        bus_bw = algo_bw * 2 * (world_size - 1) / world_size

        if rank == 0:
            print(f"\nAllreduce Basic (256 MB):")
            print(f"  Algorithm bandwidth: {algo_bw:.2f} GB/s")
            print(f"  Bus bandwidth: {bus_bw:.2f} GB/s")
            print(f"  Duration: {result.duration_ms:.3f} ms")

        # Verify minimum performance
        min_algo_bw = EXPECTED_P2P_BW_GBPS * ALLREDUCE_EFFICIENCY_THRESHOLD * 0.5
        assert algo_bw >= min_algo_bw, (
            f"Allreduce bandwidth {algo_bw:.2f} GB/s below minimum {min_algo_bw:.2f} GB/s"
        )

        sync_barrier()

    @requires_multi_gpu(8)
    @pytest.mark.parametrize("size_bytes", TENSOR_SIZES_BYTES)
    def test_allreduce_varying_sizes(self, distributed_setup, sync_barrier, size_bytes):
        """Test allreduce with varying tensor sizes."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        tensor = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)

        def allreduce():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        result = measure_bandwidth(allreduce, size_bytes)

        if rank == 0:
            size_mb = size_bytes / (1024 * 1024)
            print(f"\nAllreduce {size_mb:.0f} MB: {result.bandwidth_gbps:.2f} GB/s, "
                  f"{result.duration_ms:.3f} ms")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_allreduce_in_place(self, distributed_setup, sync_barrier):
        """Test in-place allreduce performance."""
        rank = get_rank()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_bytes = 512 * 1024 * 1024
        tensor = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)

        def allreduce_inplace():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        result = measure_bandwidth(allreduce_inplace, size_bytes)

        if rank == 0:
            print(f"\nAllreduce In-Place (512 MB): {result.bandwidth_gbps:.2f} GB/s")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_allreduce_multiple_tensors(self, distributed_setup, sync_barrier):
        """Test allreduce with multiple small tensors (gradient bucketing scenario)."""
        rank = get_rank()
        device = torch.device(f"cuda:{get_local_rank()}")

        # Simulate gradient bucketing: many small tensors
        num_tensors = 100
        tensor_size = 4 * 1024 * 1024  # 4 MB each
        total_size = num_tensors * tensor_size

        tensors = [
            torch.randn(tensor_size // 4, dtype=torch.float32, device=device)
            for _ in range(num_tensors)
        ]

        def allreduce_multiple():
            for t in tensors:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        result = measure_bandwidth(
            allreduce_multiple,
            total_size,
            warmup_iters=5,
            measure_iters=20,
        )

        if rank == 0:
            print(f"\nAllreduce Multiple ({num_tensors} x 4MB):")
            print(f"  Total bandwidth: {result.bandwidth_gbps:.2f} GB/s")
            print(f"  Per-tensor avg: {result.duration_ms / num_tensors:.3f} ms")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_allreduce_bfloat16(self, distributed_setup, sync_barrier):
        """Test allreduce with bfloat16 (common in training)."""
        rank = get_rank()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_bytes = 256 * 1024 * 1024
        num_elements = size_bytes // 2  # bfloat16 = 2 bytes
        tensor = torch.randn(num_elements, dtype=torch.bfloat16, device=device)

        def allreduce_bf16():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        result = measure_bandwidth(allreduce_bf16, size_bytes)

        if rank == 0:
            print(f"\nAllreduce BFloat16 (256 MB): {result.bandwidth_gbps:.2f} GB/s")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_reduce_scatter(self, distributed_setup, sync_barrier):
        """Test reduce-scatter bandwidth (ZeRO-style gradient sharding)."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        shard_size = 64 * 1024 * 1024  # 64 MB per shard
        total_size = shard_size * world_size

        input_tensor = torch.randn(total_size // 4, dtype=torch.float32, device=device)
        output_tensor = torch.empty(shard_size // 4, dtype=torch.float32, device=device)

        input_list = list(input_tensor.chunk(world_size))

        def reduce_scatter():
            dist.reduce_scatter(output_tensor, input_list)

        result = measure_bandwidth(reduce_scatter, total_size)

        if rank == 0:
            print(f"\nReduce-Scatter ({world_size} x {shard_size // (1024*1024)} MB):")
            print(f"  Bandwidth: {result.bandwidth_gbps:.2f} GB/s")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_all_gather(self, distributed_setup, sync_barrier):
        """Test all-gather bandwidth (parameter reconstruction)."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        shard_size = 64 * 1024 * 1024  # 64 MB per shard
        total_size = shard_size * world_size

        input_tensor = torch.randn(shard_size // 4, dtype=torch.float32, device=device)
        output_tensor = torch.empty(total_size // 4, dtype=torch.float32, device=device)

        output_list = list(output_tensor.chunk(world_size))

        def all_gather():
            dist.all_gather(output_list, input_tensor)

        result = measure_bandwidth(all_gather, total_size)

        if rank == 0:
            print(f"\nAll-Gather ({world_size} x {shard_size // (1024*1024)} MB):")
            print(f"  Bandwidth: {result.bandwidth_gbps:.2f} GB/s")

        sync_barrier()


# ==============================================================================
# Test Class: Bidirectional Transfer Bandwidth
# ==============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestNVLinkBidirectional:
    """Test bidirectional NVLink transfers (simultaneous send/receive)."""

    @requires_multi_gpu(2)
    def test_bidirectional_adjacent(self, distributed_setup, sync_barrier):
        """Test bidirectional transfers between adjacent GPUs."""
        rank = get_rank()

        if rank >= 2:
            sync_barrier()
            return

        size_bytes = 256 * 1024 * 1024

        if rank == 0:
            # GPU 0: send to GPU 1, receive from GPU 1
            send_tensor = create_tensor_on_device(size_bytes, 0)
            recv_tensor = torch.empty_like(send_tensor)

            with torch.cuda.device(1):
                remote_send = create_tensor_on_device(size_bytes, 1)

            stream_send = torch.cuda.Stream()
            stream_recv = torch.cuda.Stream()

            # Warmup
            for _ in range(10):
                with torch.cuda.stream(stream_send):
                    with torch.cuda.device(1):
                        dst = torch.empty_like(send_tensor, device="cuda:1")
                    dst.copy_(send_tensor)
                with torch.cuda.stream(stream_recv):
                    recv_tensor.copy_(remote_send)
                torch.cuda.synchronize()

            # Measure bidirectional
            start_event = torch.cuda.Event(enable_timing=True)
            end_event_send = torch.cuda.Event(enable_timing=True)
            end_event_recv = torch.cuda.Event(enable_timing=True)

            measure_iters = 30

            start_event.record()
            for _ in range(measure_iters):
                with torch.cuda.stream(stream_send):
                    with torch.cuda.device(1):
                        dst = torch.empty_like(send_tensor, device="cuda:1")
                    dst.copy_(send_tensor)
                    end_event_send.record()
                with torch.cuda.stream(stream_recv):
                    recv_tensor.copy_(remote_send)
                    end_event_recv.record()

            torch.cuda.synchronize()

            send_time = start_event.elapsed_time(end_event_send)
            recv_time = start_event.elapsed_time(end_event_recv)
            max_time = max(send_time, recv_time)

            # Bidirectional bandwidth = 2 * size / max_time
            total_bytes = 2 * size_bytes * measure_iters
            bidir_bw = (total_bytes / max_time) * 1000 / 1e9

            # Compare to unidirectional
            unidir_bw = (size_bytes * measure_iters / send_time) * 1000 / 1e9

            print(f"\nBidirectional Adjacent (0<->1, 256 MB each direction):")
            print(f"  Unidirectional bandwidth: {unidir_bw:.2f} GB/s")
            print(f"  Bidirectional bandwidth: {bidir_bw:.2f} GB/s")
            print(f"  Bidirectional speedup: {bidir_bw / unidir_bw:.2f}x")

            # Bidirectional should be close to 2x unidirectional
            expected_bidir = NVLINK_5_BIDIRECTIONAL_BW_GBPS * NUM_NVLINKS_PER_GPU_PAIR
            efficiency = bidir_bw / expected_bidir
            print(f"  Efficiency vs theoretical: {efficiency * 100:.1f}%")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_bidirectional_ring(self, distributed_setup, sync_barrier):
        """Test bidirectional transfers in ring pattern."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_bytes = 128 * 1024 * 1024

        # Each rank sends to next and receives from previous
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size

        send_tensor = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)
        recv_tensor = torch.empty_like(send_tensor)

        def ring_exchange():
            send_req = dist.isend(send_tensor, dst=send_rank)
            recv_req = dist.irecv(recv_tensor, src=recv_rank)
            send_req.wait()
            recv_req.wait()

        result = measure_bandwidth(ring_exchange, size_bytes * 2)  # Both directions

        if rank == 0:
            print(f"\nBidirectional Ring (128 MB each way):")
            print(f"  Per-rank bandwidth: {result.bandwidth_gbps:.2f} GB/s")
            print(f"  Aggregate: {result.bandwidth_gbps * world_size:.2f} GB/s")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_sendrecv_pairs(self, distributed_setup, sync_barrier):
        """Test pairwise sendrecv for bidirectional communication."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_bytes = 128 * 1024 * 1024

        # Pair GPUs: (0,1), (2,3), (4,5), (6,7)
        pair_rank = rank ^ 1  # XOR to find pair

        send_tensor = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)
        recv_tensor = torch.empty_like(send_tensor)

        def sendrecv():
            dist.sendrecv(send_tensor, dst=pair_rank, recvbuf=recv_tensor, src=pair_rank)

        result = measure_bandwidth(sendrecv, size_bytes * 2)

        if rank == 0:
            print(f"\nSendrecv Pairs (128 MB bidirectional):")
            print(f"  Per-pair bandwidth: {result.bandwidth_gbps:.2f} GB/s")

        sync_barrier()


# ==============================================================================
# Test Class: Aggregate Bandwidth
# ==============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestNVLinkAggregateBandwidth:
    """Test aggregate NVLink bandwidth across all GPU pairs."""

    @requires_multi_gpu(8)
    def test_aggregate_concurrent_p2p(self, distributed_setup, sync_barrier):
        """Test aggregate bandwidth with concurrent P2P transfers on all pairs."""
        rank = get_rank()
        world_size = get_world_size()

        if rank != 0:
            sync_barrier()
            return

        size_bytes = 256 * 1024 * 1024
        pairs = get_adjacent_gpu_pairs(min(world_size, 8))

        # Create tensors and streams for all pairs
        streams = []
        src_tensors = []
        dst_tensors = []

        for src, dst in pairs:
            stream = torch.cuda.Stream()
            streams.append(stream)
            src_tensor = create_tensor_on_device(size_bytes, src)
            src_tensors.append(src_tensor)
            with torch.cuda.device(dst):
                dst_tensor = torch.empty_like(src_tensor, device=f"cuda:{dst}")
            dst_tensors.append(dst_tensor)

        # Warmup
        for i, (src, dst) in enumerate(pairs):
            with torch.cuda.stream(streams[i]):
                dst_tensors[i].copy_(src_tensors[i])
        torch.cuda.synchronize()

        # Measure concurrent transfers
        start_events = [torch.cuda.Event(enable_timing=True) for _ in pairs]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in pairs]

        measure_iters = 20

        # Start all transfers concurrently
        for i, (src, dst) in enumerate(pairs):
            with torch.cuda.stream(streams[i]):
                start_events[i].record()
                for _ in range(measure_iters):
                    dst_tensors[i].copy_(src_tensors[i])
                end_events[i].record()

        torch.cuda.synchronize()

        # Calculate aggregate bandwidth
        times_ms = [
            start_events[i].elapsed_time(end_events[i])
            for i in range(len(pairs))
        ]

        max_time_ms = max(times_ms)
        total_bytes = size_bytes * len(pairs) * measure_iters
        aggregate_bw = (total_bytes / max_time_ms) * 1000 / 1e9

        per_pair_bw = [
            (size_bytes * measure_iters / t) * 1000 / 1e9 for t in times_ms
        ]

        print(f"\nAggregate Concurrent P2P ({len(pairs)} pairs):")
        print(f"  Aggregate bandwidth: {aggregate_bw:.2f} GB/s")
        print(f"  Per-pair bandwidth: {[f'{bw:.1f}' for bw in per_pair_bw]} GB/s")
        print(f"  Target: {B200_AGGREGATE_BW_TARGET_GBPS:.0f} GB/s")
        print(f"  Efficiency: {aggregate_bw / B200_AGGREGATE_BW_TARGET_GBPS * 100:.1f}%")

        # Check against target (with some tolerance)
        min_aggregate = B200_AGGREGATE_BW_TARGET_GBPS * 0.4  # 40% of target minimum
        assert aggregate_bw >= min_aggregate, (
            f"Aggregate bandwidth {aggregate_bw:.2f} GB/s below minimum "
            f"{min_aggregate:.2f} GB/s (40% of {B200_AGGREGATE_BW_TARGET_GBPS} GB/s target)"
        )

        sync_barrier()

    @requires_multi_gpu(8)
    def test_aggregate_all_pairs(self, distributed_setup, sync_barrier):
        """Test aggregate bandwidth using all unique GPU pairs."""
        rank = get_rank()

        if rank != 0:
            sync_barrier()
            return

        num_gpus = min(torch.cuda.device_count(), 8)
        all_pairs = get_all_gpu_pairs(num_gpus)
        size_bytes = 64 * 1024 * 1024  # Smaller size due to memory constraints

        # Create streams and tensors for all pairs
        streams = []
        transfers = []

        for src, dst in all_pairs:
            stream = torch.cuda.Stream()
            streams.append(stream)
            src_tensor = create_tensor_on_device(size_bytes, src)
            with torch.cuda.device(dst):
                dst_tensor = torch.empty_like(src_tensor, device=f"cuda:{dst}")
            transfers.append((src_tensor, dst_tensor))

        # Warmup
        for i, _ in enumerate(all_pairs):
            with torch.cuda.stream(streams[i]):
                transfers[i][1].copy_(transfers[i][0])
        torch.cuda.synchronize()

        # Measure
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        measure_iters = 10

        start_event.record()
        for _ in range(measure_iters):
            for i, _ in enumerate(all_pairs):
                with torch.cuda.stream(streams[i]):
                    transfers[i][1].copy_(transfers[i][0])
        torch.cuda.synchronize()
        end_event.record()
        end_event.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        total_bytes = size_bytes * len(all_pairs) * measure_iters
        aggregate_bw = (total_bytes / total_time_ms) * 1000 / 1e9

        print(f"\nAggregate All Pairs ({len(all_pairs)} pairs, 64 MB each):")
        print(f"  Total bytes transferred: {total_bytes / 1e9:.2f} GB")
        print(f"  Total time: {total_time_ms:.2f} ms")
        print(f"  Aggregate bandwidth: {aggregate_bw:.2f} GB/s")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_bandwidth_scaling(self, distributed_setup, sync_barrier):
        """Test how bandwidth scales with number of concurrent transfers."""
        rank = get_rank()

        if rank != 0:
            sync_barrier()
            return

        size_bytes = 128 * 1024 * 1024
        num_gpus = min(torch.cuda.device_count(), 8)

        results = []

        for num_pairs in [1, 2, 4, 8]:
            if num_pairs > num_gpus:
                continue

            pairs = [(i, (i + 1) % num_gpus) for i in range(num_pairs)]

            streams = []
            transfers = []

            for src, dst in pairs:
                stream = torch.cuda.Stream()
                streams.append(stream)
                src_tensor = create_tensor_on_device(size_bytes, src)
                with torch.cuda.device(dst):
                    dst_tensor = torch.empty_like(src_tensor, device=f"cuda:{dst}")
                transfers.append((src_tensor, dst_tensor))

            # Warmup
            for i in range(num_pairs):
                with torch.cuda.stream(streams[i]):
                    transfers[i][1].copy_(transfers[i][0])
            torch.cuda.synchronize()

            # Measure
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]

            measure_iters = 20

            for i in range(num_pairs):
                with torch.cuda.stream(streams[i]):
                    start_events[i].record()
                    for _ in range(measure_iters):
                        transfers[i][1].copy_(transfers[i][0])
                    end_events[i].record()

            torch.cuda.synchronize()

            max_time = max(
                start_events[i].elapsed_time(end_events[i])
                for i in range(num_pairs)
            )

            total_bytes = size_bytes * num_pairs * measure_iters
            aggregate_bw = (total_bytes / max_time) * 1000 / 1e9

            results.append((num_pairs, aggregate_bw))

        print(f"\nBandwidth Scaling (128 MB transfers):")
        for num_pairs, bw in results:
            print(f"  {num_pairs} pairs: {bw:.2f} GB/s")

        # Verify scaling (should be roughly linear)
        if len(results) >= 2:
            single_pair_bw = results[0][1]
            for num_pairs, bw in results[1:]:
                expected_linear = single_pair_bw * num_pairs
                scaling_efficiency = bw / expected_linear
                print(f"  {num_pairs} pairs scaling efficiency: {scaling_efficiency * 100:.1f}%")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_aggregate_allreduce_bandwidth(self, distributed_setup, sync_barrier):
        """Test aggregate bandwidth achieved during allreduce."""
        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{get_local_rank()}")

        size_bytes = 512 * 1024 * 1024  # 512 MB
        tensor = torch.randn(size_bytes // 4, dtype=torch.float32, device=device)

        def allreduce():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        result = measure_bandwidth(allreduce, size_bytes)

        # Calculate effective bus bandwidth for ring allreduce
        # Each GPU sends and receives 2*(n-1)/n * size bytes
        ring_factor = 2 * (world_size - 1) / world_size
        bus_bw = result.bandwidth_gbps * ring_factor

        # Total aggregate = bus_bw per GPU * num_gpus / 2 (since pairs share links)
        aggregate_estimate = bus_bw * world_size / 2

        if rank == 0:
            print(f"\nAggregate Allreduce Bandwidth (512 MB):")
            print(f"  Algorithm bandwidth: {result.bandwidth_gbps:.2f} GB/s")
            print(f"  Per-GPU bus bandwidth: {bus_bw:.2f} GB/s")
            print(f"  Estimated aggregate: {aggregate_estimate:.2f} GB/s")
            print(f"  Target: {B200_AGGREGATE_BW_TARGET_GBPS:.0f} GB/s")

        sync_barrier()


# ==============================================================================
# Test Class: Latency Measurements
# ==============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestNVLinkLatency:
    """Test small message latency over NVLink."""

    @requires_multi_gpu(2)
    @pytest.mark.parametrize("size_bytes", LATENCY_SIZES_BYTES)
    def test_p2p_latency_varying_sizes(self, distributed_setup, sync_barrier, size_bytes):
        """Test P2P latency with varying small message sizes."""
        rank = get_rank()

        if rank != 0:
            sync_barrier()
            return

        src_tensor = create_tensor_on_device(size_bytes, 0)
        with torch.cuda.device(1):
            dst_tensor = torch.empty_like(src_tensor, device="cuda:1")

        def transfer():
            dst_tensor.copy_(src_tensor)
            torch.cuda.synchronize()

        result = measure_latency(transfer, size_bytes)

        print(f"\nP2P Latency ({size_bytes} bytes): {result}")

        # Latency should be reasonable (< 100 us for small messages)
        if size_bytes <= 4096:
            assert result.latency_us < 100, (
                f"Latency {result.latency_us:.2f} us too high for {size_bytes} bytes"
            )

        sync_barrier()

    @requires_multi_gpu(8)
    def test_allreduce_latency(self, distributed_setup, sync_barrier):
        """Test allreduce latency for small tensors."""
        rank = get_rank()
        device = torch.device(f"cuda:{get_local_rank()}")

        for size_bytes in [64, 4096, 65536]:
            num_elements = max(1, size_bytes // 4)
            tensor = torch.randn(num_elements, dtype=torch.float32, device=device)

            def allreduce():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            result = measure_latency(allreduce, size_bytes, measure_iters=100)

            if rank == 0:
                print(f"\nAllreduce Latency ({size_bytes} bytes): {result}")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_barrier_latency(self, distributed_setup, sync_barrier):
        """Test barrier synchronization latency."""
        rank = get_rank()

        latencies_us = []

        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            dist.barrier()
            end.record()
            end.synchronize()

            latencies_us.append(start.elapsed_time(end) * 1000)

        if rank == 0:
            avg_latency = statistics.mean(latencies_us)
            min_latency = min(latencies_us)
            max_latency = max(latencies_us)
            std_latency = statistics.stdev(latencies_us)

            print(f"\nBarrier Latency:")
            print(f"  Average: {avg_latency:.2f} us")
            print(f"  Min: {min_latency:.2f} us, Max: {max_latency:.2f} us")
            print(f"  Std: {std_latency:.2f} us")

        sync_barrier()

    @requires_multi_gpu(2)
    def test_cuda_ipc_latency(self, distributed_setup, sync_barrier):
        """Test CUDA IPC handle exchange latency."""
        rank = get_rank()
        device = torch.device(f"cuda:{get_local_rank()}")

        # Create a small tensor
        tensor = torch.randn(256, dtype=torch.float32, device=device)

        latencies_us = []

        for _ in range(50):
            # Simulate IPC-like pattern: broadcast tensor handle info
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            dist.broadcast(tensor, src=0)
            end.record()
            end.synchronize()

            latencies_us.append(start.elapsed_time(end) * 1000)

        if rank == 0:
            avg_latency = statistics.mean(latencies_us)
            print(f"\nBroadcast Latency (1KB tensor): {avg_latency:.2f} us")

        sync_barrier()

    @requires_multi_gpu(8)
    def test_latency_vs_bandwidth_tradeoff(self, distributed_setup, sync_barrier):
        """Analyze latency vs bandwidth tradeoff across message sizes."""
        rank = get_rank()

        if rank != 0:
            sync_barrier()
            return

        # Test sizes from 4 bytes to 256 MB
        test_sizes = [
            4, 64, 512, 4096, 65536,
            1024 * 1024,  # 1 MB
            16 * 1024 * 1024,  # 16 MB
            256 * 1024 * 1024,  # 256 MB
        ]

        results = []

        for size_bytes in test_sizes:
            src_tensor = create_tensor_on_device(size_bytes, 0)
            with torch.cuda.device(1):
                dst_tensor = torch.empty_like(src_tensor, device="cuda:1")

            def transfer():
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()

            if size_bytes < 1024 * 1024:
                # Latency measurement for small sizes
                lat_result = measure_latency(transfer, size_bytes)
                # Convert to bandwidth
                bw_gbps = (size_bytes / lat_result.latency_us) * 1e6 / 1e9
                results.append((size_bytes, lat_result.latency_us, bw_gbps))
            else:
                # Bandwidth measurement for large sizes
                bw_result = measure_bandwidth(transfer, size_bytes)
                latency_us = bw_result.duration_ms * 1000
                results.append((size_bytes, latency_us, bw_result.bandwidth_gbps))

        print(f"\nLatency vs Bandwidth Tradeoff:")
        print(f"  {'Size':>12} {'Latency (us)':>15} {'Bandwidth (GB/s)':>18}")
        print(f"  {'-'*12} {'-'*15} {'-'*18}")

        for size, latency, bw in results:
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size // 1024} KB"
            else:
                size_str = f"{size // (1024 * 1024)} MB"
            print(f"  {size_str:>12} {latency:>15.2f} {bw:>18.2f}")

        sync_barrier()


# ==============================================================================
# Standalone Execution Support
# ==============================================================================

if __name__ == "__main__":
    # Allow running with torchrun
    init_distributed()
    enable_p2p_access_all()

    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        print(f"Running NVLink bandwidth tests on {world_size} GPUs")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

    # Run pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
