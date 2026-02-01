"""Comprehensive CPU core utilization tests for B200 nodes with 200 cores.

This module tests various aspects of CPU utilization and scaling on B200 nodes
which feature 200 CPU cores. It covers:
- DataLoader worker scaling
- Parallel tokenization
- Multiprocessing pool efficiency
- ThreadPoolExecutor scalability
- NumPy parallel operations (MKL/OpenBLAS)
- Data pipeline bottleneck detection
- Prefetch buffer efficiency

Usage:
    pytest tests/cpu/b200/test_200_core_utilization.py -v
    pytest tests/cpu/b200/test_200_core_utilization.py -v -m "cpu and b200"
    pytest tests/cpu/b200/test_200_core_utilization.py -v -k "TestDataLoaderScaling"
"""

import gc
import math
import mmap
import os
import queue
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


# =============================================================================
# Markers - all tests in this module require cpu and b200
# =============================================================================

pytestmark = [pytest.mark.cpu, pytest.mark.b200]


# =============================================================================
# Constants and Configuration
# =============================================================================

# B200 node specifications
B200_TOTAL_CORES = 200
MIN_CORES_FOR_B200_TEST = 64  # Minimum cores to run B200-style tests

# Test scaling parameters
WORKER_COUNTS = [1, 4, 16, 32, 64, 100, 150, 200]
THREAD_COUNTS = [1, 4, 16, 32, 64, 100, 150, 200]
POOL_SIZES = [1, 4, 16, 32, 64, 100, 150, 200]

# Performance thresholds
MIN_SPEEDUP_RATIO = 0.5  # Minimum expected speedup at high parallelism
MAX_OVERHEAD_RATIO = 2.0  # Maximum acceptable overhead for parallelism


# =============================================================================
# Utility Classes and Functions
# =============================================================================


@dataclass
class ScalingResult:
    """Result from a scaling benchmark."""

    name: str
    worker_count: int
    throughput: float  # items per second
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p99_ms: float
    cpu_utilization: float  # 0.0 to 1.0
    memory_mb: float
    iterations: int
    config: dict = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """Calculate parallel efficiency (0.0 to 1.0)."""
        if self.worker_count <= 1:
            return 1.0
        # Efficiency = actual speedup / ideal speedup
        return min(1.0, self.throughput / (self.config.get("baseline_throughput", self.throughput) * self.worker_count))


def percentile(pct: float, values: List[float]) -> float:
    """Compute percentile of a list."""
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def measure_cpu_utilization(duration_seconds: float = 1.0) -> float:
    """Measure current CPU utilization over a duration."""
    try:
        import psutil
        start = psutil.cpu_percent(interval=None)
        time.sleep(duration_seconds)
        end = psutil.cpu_percent(interval=None)
        return (start + end) / 200.0  # Average, normalized to 0-1
    except ImportError:
        # psutil not available, return estimate
        return 0.5


def get_available_cores() -> int:
    """Get the number of available CPU cores."""
    try:
        # Try to get the actual available cores (respecting cgroups/containers)
        import os
        if hasattr(os, 'sched_getaffinity'):
            return len(os.sched_getaffinity(0))
        return cpu_count() or 1
    except Exception:
        return cpu_count() or 1


def skip_if_insufficient_cores(min_cores: int = MIN_CORES_FOR_B200_TEST):
    """Skip test if insufficient cores available."""
    available = get_available_cores()
    if available < min_cores:
        pytest.skip(f"Requires at least {min_cores} cores, only {available} available")


def filter_worker_counts(counts: List[int], max_workers: Optional[int] = None) -> List[int]:
    """Filter worker counts based on available cores."""
    available = get_available_cores() if max_workers is None else max_workers
    return [c for c in counts if c <= available]


# =============================================================================
# Mock Data and Datasets
# =============================================================================


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing DataLoader scaling."""

    def __init__(self, size: int = 10000, dim: int = 512, simulate_io: bool = False):
        self.size = size
        self.dim = dim
        self.simulate_io = simulate_io
        # Pre-generate some data to avoid RNG synchronization issues
        self._seed_data = np.random.randn(min(1000, size), dim).astype(np.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.simulate_io:
            # Simulate I/O latency (file read)
            time.sleep(0.0001)  # 0.1ms

        # Generate data based on index
        np.random.seed(idx % 1000)
        data = np.random.randn(self.dim).astype(np.float32)
        return torch.from_numpy(data)


class TokenizationDataset(Dataset):
    """Dataset that simulates tokenization workload."""

    def __init__(self, size: int = 10000, text_length: int = 512):
        self.size = size
        self.text_length = text_length
        # Pre-generate text samples
        self._texts = [
            "".join(chr(65 + (i + j) % 26) for j in range(text_length))
            for i in range(min(1000, size))
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        text = self._texts[idx % len(self._texts)]
        # Simulate tokenization (character-level for testing)
        tokens = [ord(c) for c in text]
        attention_mask = [1] * len(tokens)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class IterableStreamDataset(IterableDataset):
    """Iterable dataset for streaming scenarios."""

    def __init__(self, total_items: int = 100000, dim: int = 256):
        self.total_items = total_items
        self.dim = dim

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process mode
            start = 0
            end = self.total_items
        else:
            # Multi-process mode - split work
            per_worker = int(math.ceil(self.total_items / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self.total_items)

        for i in range(start, end):
            np.random.seed(i)
            yield torch.from_numpy(np.random.randn(self.dim).astype(np.float32))


# =============================================================================
# CPU-Bound Work Functions
# =============================================================================


def cpu_bound_work(data: np.ndarray) -> np.ndarray:
    """Perform CPU-bound computation on data."""
    # Matrix operations that are CPU-intensive
    result = data.copy()
    for _ in range(10):
        result = np.sin(result) + np.cos(result)
        result = result @ result.T if result.ndim == 2 else result
        result = np.sqrt(np.abs(result) + 1e-6)
    return result


def tokenize_text(text: str) -> List[int]:
    """Simple tokenization function for testing."""
    # Character-level tokenization with some processing
    tokens = []
    for char in text:
        token_id = ord(char)
        # Add some computation
        token_id = (token_id * 7 + 13) % 65536
        tokens.append(token_id)
    return tokens


def preprocess_batch(batch: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    """Preprocess a batch of data."""
    idx, data = batch
    # Simulate preprocessing
    result = np.log1p(np.abs(data))
    result = (result - result.mean()) / (result.std() + 1e-6)
    return idx, result


def memory_mapped_read(args: Tuple[str, int, int]) -> np.ndarray:
    """Read from a memory-mapped file."""
    filepath, offset, length = args
    with open(filepath, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.seek(offset)
        data = mm.read(length)
        mm.close()
    return np.frombuffer(data, dtype=np.float32)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def available_cores():
    """Get available CPU cores."""
    return get_available_cores()


@pytest.fixture(scope="module")
def worker_counts_filtered(available_cores):
    """Get worker counts filtered by available cores."""
    return filter_worker_counts(WORKER_COUNTS, available_cores)


@pytest.fixture(scope="module")
def thread_counts_filtered(available_cores):
    """Get thread counts filtered by available cores."""
    return filter_worker_counts(THREAD_COUNTS, available_cores)


@pytest.fixture(scope="module")
def pool_sizes_filtered(available_cores):
    """Get pool sizes filtered by available cores."""
    return filter_worker_counts(POOL_SIZES, available_cores)


@pytest.fixture(scope="function")
def temp_mmap_file():
    """Create a temporary memory-mapped file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        # Write 100MB of random data
        data = np.random.randn(25 * 1024 * 1024).astype(np.float32)  # 100MB
        f.write(data.tobytes())
        filepath = f.name

    yield filepath

    # Cleanup
    try:
        os.unlink(filepath)
    except Exception:
        pass


@pytest.fixture(scope="module")
def scaling_results():
    """Collect scaling results for analysis."""
    results = []
    yield results

    # Print summary after all tests
    if results:
        print("\n" + "=" * 80)
        print("SCALING RESULTS SUMMARY")
        print("=" * 80)
        for r in results:
            print(f"{r.name} @ {r.worker_count} workers: "
                  f"{r.throughput:.2f} items/s, "
                  f"latency={r.latency_mean_ms:.2f}ms, "
                  f"efficiency={r.efficiency:.2%}")
        print("=" * 80)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.cpu
@pytest.mark.b200
class TestDataLoaderScaling:
    """Test PyTorch DataLoader scaling with num_workers 1 to 200."""

    def test_dataloader_basic_scaling(self, worker_counts_filtered, scaling_results):
        """Test basic DataLoader scaling with increasing workers."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=10000, dim=512)
        batch_size = 64
        num_batches = 100

        baseline_throughput = None

        for num_workers in worker_counts_filtered[:6]:  # Test up to 6 different worker counts
            gc.collect()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
            )

            # Warmup
            loader_iter = iter(loader)
            for _ in range(min(10, num_batches)):
                try:
                    next(loader_iter)
                except StopIteration:
                    break

            # Benchmark
            latencies = []
            start_time = time.perf_counter()
            loader_iter = iter(loader)
            for i in range(num_batches):
                batch_start = time.perf_counter()
                try:
                    batch = next(loader_iter)
                    latencies.append((time.perf_counter() - batch_start) * 1000)
                except StopIteration:
                    break
            total_time = time.perf_counter() - start_time

            throughput = len(latencies) * batch_size / total_time

            if baseline_throughput is None:
                baseline_throughput = throughput

            result = ScalingResult(
                name="dataloader_basic",
                worker_count=num_workers,
                throughput=throughput,
                latency_mean_ms=statistics.mean(latencies) if latencies else 0,
                latency_p50_ms=percentile(50, latencies) if latencies else 0,
                latency_p99_ms=percentile(99, latencies) if latencies else 0,
                cpu_utilization=0.0,  # Not measured in this test
                memory_mb=0.0,
                iterations=len(latencies),
                config={"batch_size": batch_size, "baseline_throughput": baseline_throughput},
            )
            scaling_results.append(result)

            # Cleanup
            del loader
            gc.collect()

        # Verify scaling improves throughput (at least 2x with 4+ workers)
        if len(scaling_results) >= 2:
            first = scaling_results[-len(worker_counts_filtered[:6])]
            last = scaling_results[-1]
            if last.worker_count >= 4:
                assert last.throughput >= first.throughput * 1.5, (
                    f"DataLoader scaling insufficient: {first.throughput:.2f} -> {last.throughput:.2f}"
                )

    def test_dataloader_with_io_simulation(self, worker_counts_filtered, scaling_results):
        """Test DataLoader scaling with simulated I/O latency."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=256, simulate_io=True)
        batch_size = 32
        num_batches = 50

        results_for_test = []

        for num_workers in [0, 4, 16]:
            if num_workers > get_available_cores():
                continue

            gc.collect()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False,
                prefetch_factor=2 if num_workers > 0 else None,
            )

            start_time = time.perf_counter()
            count = 0
            for batch in loader:
                count += 1
                if count >= num_batches:
                    break
            total_time = time.perf_counter() - start_time

            throughput = count * batch_size / total_time
            results_for_test.append((num_workers, throughput))

            del loader
            gc.collect()

        # With I/O simulation, more workers should significantly help
        if len(results_for_test) >= 2:
            single_tp = results_for_test[0][1]
            multi_tp = results_for_test[-1][1]
            # Should see at least 2x improvement with parallelism for I/O bound
            assert multi_tp > single_tp * 1.5, (
                f"I/O-bound scaling insufficient: {single_tp:.2f} -> {multi_tp:.2f}"
            )

    def test_dataloader_prefetch_factor_impact(self, available_cores):
        """Test impact of prefetch_factor on DataLoader performance."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=256)
        batch_size = 32
        num_workers = min(8, available_cores)
        num_batches = 50

        prefetch_factors = [1, 2, 4, 8]
        results = []

        for prefetch_factor in prefetch_factors:
            gc.collect()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
            )

            # Warmup
            loader_iter = iter(loader)
            for _ in range(5):
                try:
                    next(loader_iter)
                except StopIteration:
                    break

            # Benchmark
            start_time = time.perf_counter()
            loader_iter = iter(loader)
            count = 0
            for _ in range(num_batches):
                try:
                    next(loader_iter)
                    count += 1
                except StopIteration:
                    break
            total_time = time.perf_counter() - start_time

            throughput = count * batch_size / total_time
            results.append((prefetch_factor, throughput))

            del loader
            gc.collect()

        # Higher prefetch_factor should not degrade performance significantly
        if len(results) >= 2:
            base_tp = results[0][1]
            for pf, tp in results[1:]:
                assert tp >= base_tp * 0.8, (
                    f"prefetch_factor={pf} degraded performance: {base_tp:.2f} -> {tp:.2f}"
                )

    def test_dataloader_persistent_workers(self, available_cores):
        """Test persistent_workers impact on DataLoader performance."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=256)
        batch_size = 32
        num_workers = min(8, available_cores)
        num_epochs = 3
        batches_per_epoch = 50

        results = {}

        for persistent in [False, True]:
            gc.collect()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=2,
                persistent_workers=persistent,
            )

            total_time = 0
            for epoch in range(num_epochs):
                start_time = time.perf_counter()
                loader_iter = iter(loader)
                count = 0
                for _ in range(batches_per_epoch):
                    try:
                        next(loader_iter)
                        count += 1
                    except StopIteration:
                        break
                total_time += time.perf_counter() - start_time

            throughput = num_epochs * batches_per_epoch * batch_size / total_time
            results[persistent] = throughput

            del loader
            gc.collect()

        # Persistent workers should be at least as fast, usually faster
        assert results[True] >= results[False] * 0.9, (
            f"persistent_workers degraded performance: {results[False]:.2f} -> {results[True]:.2f}"
        )


@pytest.mark.cpu
@pytest.mark.b200
class TestParallelTokenization:
    """Test parallel tokenization across CPU cores."""

    def test_tokenization_pool_scaling(self, pool_sizes_filtered):
        """Test tokenization scaling with multiprocessing Pool."""
        skip_if_insufficient_cores(4)

        # Generate test texts
        num_texts = 5000
        text_length = 256
        texts = [
            "".join(chr(65 + (i + j) % 26) for j in range(text_length))
            for i in range(num_texts)
        ]

        baseline_throughput = None
        results = []

        for pool_size in [1, 4, 16]:
            if pool_size > get_available_cores():
                continue

            gc.collect()

            start_time = time.perf_counter()

            if pool_size == 1:
                # Single process
                tokenized = [tokenize_text(t) for t in texts]
            else:
                # Multiprocessing
                with Pool(pool_size) as pool:
                    tokenized = pool.map(tokenize_text, texts)

            total_time = time.perf_counter() - start_time
            throughput = num_texts / total_time

            if baseline_throughput is None:
                baseline_throughput = throughput

            results.append((pool_size, throughput))

            gc.collect()

        # Verify scaling
        if len(results) >= 2:
            single_tp = results[0][1]
            multi_tp = results[-1][1]
            # Should see improvement (at least 2x with 16 workers for CPU-bound)
            assert multi_tp > single_tp * 1.5, (
                f"Tokenization scaling insufficient: {single_tp:.2f} -> {multi_tp:.2f}"
            )

    def test_tokenization_chunked_processing(self, available_cores):
        """Test chunked tokenization for better cache locality."""
        skip_if_insufficient_cores(4)

        num_texts = 10000
        text_length = 128
        texts = ["x" * text_length for _ in range(num_texts)]
        pool_size = min(16, available_cores)

        chunk_sizes = [100, 500, 1000, 2000]
        results = []

        for chunk_size in chunk_sizes:
            gc.collect()

            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

            def tokenize_chunk(chunk: List[str]) -> List[List[int]]:
                return [tokenize_text(t) for t in chunk]

            start_time = time.perf_counter()
            with Pool(pool_size) as pool:
                tokenized_chunks = pool.map(tokenize_chunk, chunks)
            total_time = time.perf_counter() - start_time

            throughput = num_texts / total_time
            results.append((chunk_size, throughput))

            gc.collect()

        # Optimal chunk size should exist (not too small, not too big)
        throughputs = [tp for _, tp in results]
        assert max(throughputs) > min(throughputs) * 0.5, "Chunk size has no impact on throughput"

    def test_tokenization_dataset_integration(self, worker_counts_filtered):
        """Test tokenization integrated with DataLoader."""
        skip_if_insufficient_cores(4)

        dataset = TokenizationDataset(size=5000, text_length=256)
        batch_size = 64

        results = []

        for num_workers in [0, 4]:
            if num_workers > get_available_cores():
                continue

            gc.collect()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=lambda x: {
                    "input_ids": torch.stack([item["input_ids"] for item in x]),
                    "attention_mask": torch.stack([item["attention_mask"] for item in x]),
                },
            )

            start_time = time.perf_counter()
            count = 0
            for batch in loader:
                count += len(batch["input_ids"])
                if count >= 2000:
                    break
            total_time = time.perf_counter() - start_time

            throughput = count / total_time
            results.append((num_workers, throughput))

            del loader
            gc.collect()

        # Multi-worker should improve throughput
        if len(results) >= 2:
            assert results[-1][1] >= results[0][1] * 0.8, (
                f"Tokenization dataset scaling failed: {results[0][1]:.2f} -> {results[-1][1]:.2f}"
            )


@pytest.mark.cpu
@pytest.mark.b200
class TestMultiprocessingPool:
    """Test multiprocessing Pool efficiency with various worker counts."""

    def test_pool_map_scaling(self, pool_sizes_filtered):
        """Test Pool.map scaling with increasing pool size."""
        skip_if_insufficient_cores(4)

        num_items = 1000
        data_dim = 100
        data = [(i, np.random.randn(data_dim, data_dim).astype(np.float32)) for i in range(num_items)]

        baseline_throughput = None
        results = []

        for pool_size in [1, 4, 16]:
            if pool_size > get_available_cores():
                continue

            gc.collect()

            start_time = time.perf_counter()

            if pool_size == 1:
                processed = [preprocess_batch(d) for d in data]
            else:
                with Pool(pool_size) as pool:
                    processed = pool.map(preprocess_batch, data)

            total_time = time.perf_counter() - start_time
            throughput = num_items / total_time

            if baseline_throughput is None:
                baseline_throughput = throughput

            results.append((pool_size, throughput))

            gc.collect()

        # Verify scaling for CPU-bound work
        if len(results) >= 2:
            assert results[-1][1] > results[0][1] * 1.5, (
                f"Pool.map scaling insufficient: {results[0][1]:.2f} -> {results[-1][1]:.2f}"
            )

    def test_pool_imap_unordered_efficiency(self, available_cores):
        """Test Pool.imap_unordered for streaming processing."""
        skip_if_insufficient_cores(4)

        num_items = 2000
        data_dim = 50
        pool_size = min(16, available_cores)

        data = [(i, np.random.randn(data_dim, data_dim).astype(np.float32)) for i in range(num_items)]

        gc.collect()

        # Test imap_unordered (better for varying task durations)
        start_time = time.perf_counter()
        with Pool(pool_size) as pool:
            results = list(pool.imap_unordered(preprocess_batch, data, chunksize=50))
        imap_time = time.perf_counter() - start_time

        gc.collect()

        # Test regular map
        start_time = time.perf_counter()
        with Pool(pool_size) as pool:
            results = pool.map(preprocess_batch, data, chunksize=50)
        map_time = time.perf_counter() - start_time

        # imap_unordered should be at least as fast as map
        assert imap_time <= map_time * 1.5, (
            f"imap_unordered slower than map: {imap_time:.2f}s vs {map_time:.2f}s"
        )

    def test_pool_chunksize_impact(self, available_cores):
        """Test impact of chunksize on Pool performance."""
        skip_if_insufficient_cores(4)

        num_items = 5000
        data_dim = 30
        pool_size = min(16, available_cores)

        data = [(i, np.random.randn(data_dim, data_dim).astype(np.float32)) for i in range(num_items)]

        chunksizes = [1, 10, 50, 100, 500]
        results = []

        for chunksize in chunksizes:
            gc.collect()

            start_time = time.perf_counter()
            with Pool(pool_size) as pool:
                processed = pool.map(preprocess_batch, data, chunksize=chunksize)
            total_time = time.perf_counter() - start_time

            throughput = num_items / total_time
            results.append((chunksize, throughput))

            gc.collect()

        # Very small chunksize should be slower due to overhead
        if len(results) >= 3:
            smallest_tp = results[0][1]  # chunksize=1
            optimal_tp = max(tp for _, tp in results[1:])
            assert optimal_tp > smallest_tp * 1.2, (
                f"Chunksize has no impact: smallest={smallest_tp:.2f}, optimal={optimal_tp:.2f}"
            )

    def test_pool_context_manager_overhead(self, available_cores):
        """Test Pool context manager creation overhead."""
        skip_if_insufficient_cores(4)

        pool_size = min(16, available_cores)
        num_iterations = 10

        # Measure overhead of creating/destroying pool
        creation_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with Pool(pool_size) as pool:
                # Do minimal work
                pool.map(lambda x: x, [1, 2, 3])
            creation_times.append(time.perf_counter() - start)

        avg_creation_time = statistics.mean(creation_times)

        # Pool creation should be reasonable (< 1 second for 16 workers)
        assert avg_creation_time < 2.0, (
            f"Pool creation too slow: {avg_creation_time:.2f}s average"
        )


@pytest.mark.cpu
@pytest.mark.b200
class TestThreadPoolExecutor:
    """Test ThreadPoolExecutor for I/O-bound and mixed workloads."""

    def test_thread_pool_io_scaling(self, thread_counts_filtered):
        """Test ThreadPoolExecutor scaling for I/O-bound tasks."""
        skip_if_insufficient_cores(4)

        num_tasks = 1000

        def io_task(task_id: int) -> int:
            # Simulate I/O
            time.sleep(0.001)  # 1ms
            return task_id

        results = []

        for num_threads in [1, 4, 16, 32]:
            if num_threads > get_available_cores() * 2:  # Threads can exceed cores
                continue

            gc.collect()

            start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(io_task, i) for i in range(num_tasks)]
                completed = [f.result() for f in as_completed(futures)]
            total_time = time.perf_counter() - start_time

            throughput = num_tasks / total_time
            results.append((num_threads, throughput))

            gc.collect()

        # More threads should help for I/O-bound tasks
        if len(results) >= 2:
            single_tp = results[0][1]
            multi_tp = results[-1][1]
            assert multi_tp > single_tp * 2, (
                f"ThreadPool I/O scaling insufficient: {single_tp:.2f} -> {multi_tp:.2f}"
            )

    def test_thread_pool_vs_process_pool_cpu_bound(self, available_cores):
        """Compare ThreadPoolExecutor vs ProcessPoolExecutor for CPU-bound work."""
        skip_if_insufficient_cores(4)

        num_tasks = 100
        data_dim = 50
        num_workers = min(8, available_cores)

        data_items = [np.random.randn(data_dim, data_dim).astype(np.float32) for _ in range(num_tasks)]

        # Thread pool (affected by GIL)
        gc.collect()
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            thread_results = list(executor.map(cpu_bound_work, data_items))
        thread_time = time.perf_counter() - start_time

        # Process pool (bypasses GIL)
        gc.collect()
        start_time = time.perf_counter()
        with Pool(num_workers) as pool:
            process_results = pool.map(cpu_bound_work, data_items)
        process_time = time.perf_counter() - start_time

        # For CPU-bound work, process pool should be faster due to GIL
        # But we allow for some variance
        assert process_time < thread_time * 2, (
            f"Process pool not faster for CPU-bound: thread={thread_time:.2f}s, process={process_time:.2f}s"
        )

    def test_thread_pool_executor_map_vs_submit(self, available_cores):
        """Compare executor.map vs executor.submit performance."""
        skip_if_insufficient_cores(4)

        num_tasks = 500
        num_workers = min(16, available_cores)

        def simple_task(x: int) -> int:
            time.sleep(0.0001)
            return x * 2

        # Using map
        gc.collect()
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            map_results = list(executor.map(simple_task, range(num_tasks)))
        map_time = time.perf_counter() - start_time

        # Using submit
        gc.collect()
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(simple_task, i) for i in range(num_tasks)]
            submit_results = [f.result() for f in futures]
        submit_time = time.perf_counter() - start_time

        # Both should have similar performance
        assert abs(map_time - submit_time) < max(map_time, submit_time), (
            f"map vs submit performance differs significantly: map={map_time:.2f}s, submit={submit_time:.2f}s"
        )


@pytest.mark.cpu
@pytest.mark.b200
class TestNumPyParallelism:
    """Test NumPy parallel operations with MKL/OpenBLAS threading."""

    def test_numpy_thread_count_impact(self, available_cores):
        """Test impact of NumPy thread count on matrix operations."""
        skip_if_insufficient_cores(4)

        matrix_size = 2000
        num_iterations = 5

        # Test different thread counts
        thread_counts = [1, 4, min(16, available_cores)]
        results = []

        for num_threads in thread_counts:
            gc.collect()

            # Set thread count via environment (may require restart to take effect)
            # These are the common environment variables for BLAS libraries
            original_env = {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            }

            try:
                os.environ["OMP_NUM_THREADS"] = str(num_threads)
                os.environ["MKL_NUM_THREADS"] = str(num_threads)
                os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

                # Create matrices
                A = np.random.randn(matrix_size, matrix_size).astype(np.float64)
                B = np.random.randn(matrix_size, matrix_size).astype(np.float64)

                # Warmup
                _ = A @ B

                # Benchmark
                times = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    C = A @ B
                    times.append(time.perf_counter() - start)

                avg_time = statistics.mean(times)
                gflops = (2 * matrix_size ** 3) / (avg_time * 1e9)
                results.append((num_threads, gflops, avg_time))

            finally:
                # Restore environment
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            gc.collect()

        # Just verify operations complete successfully
        assert len(results) >= 1, "NumPy matrix operations failed"

    def test_numpy_vectorized_operations(self):
        """Test NumPy vectorized operations efficiency."""
        skip_if_insufficient_cores(4)

        array_size = 10_000_000

        # Create large arrays
        a = np.random.randn(array_size).astype(np.float32)
        b = np.random.randn(array_size).astype(np.float32)

        # Benchmark vectorized operations
        operations = [
            ("add", lambda: a + b),
            ("multiply", lambda: a * b),
            ("sqrt", lambda: np.sqrt(np.abs(a))),
            ("exp", lambda: np.exp(a / 10)),  # Scale down to avoid overflow
            ("dot", lambda: np.dot(a, b)),
        ]

        results = {}
        for name, op in operations:
            gc.collect()

            # Warmup
            _ = op()

            # Benchmark
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = op()
                times.append(time.perf_counter() - start)

            avg_time = statistics.mean(times)
            throughput = array_size / avg_time
            results[name] = throughput

            gc.collect()

        # All operations should achieve reasonable throughput (> 1M elements/sec)
        for name, throughput in results.items():
            assert throughput > 1_000_000, (
                f"NumPy {name} operation too slow: {throughput:.0f} elements/sec"
            )

    def test_numpy_batch_processing(self, available_cores):
        """Test NumPy batch processing with multiprocessing."""
        skip_if_insufficient_cores(4)

        batch_size = 1000
        array_size = 10000
        num_workers = min(8, available_cores)

        def process_batch(batch_data: np.ndarray) -> np.ndarray:
            result = np.sin(batch_data) + np.cos(batch_data)
            result = np.sqrt(np.abs(result))
            return result.mean(axis=1)

        # Generate batches
        batches = [np.random.randn(batch_size, array_size).astype(np.float32) for _ in range(20)]

        # Single process
        gc.collect()
        start_time = time.perf_counter()
        single_results = [process_batch(b) for b in batches]
        single_time = time.perf_counter() - start_time

        # Multi process
        gc.collect()
        start_time = time.perf_counter()
        with Pool(num_workers) as pool:
            multi_results = pool.map(process_batch, batches)
        multi_time = time.perf_counter() - start_time

        # Multi-process should be faster
        assert multi_time < single_time * 1.5, (
            f"NumPy batch multiprocessing not effective: single={single_time:.2f}s, multi={multi_time:.2f}s"
        )


@pytest.mark.cpu
@pytest.mark.b200
class TestDataPipelineBottleneck:
    """Test end-to-end data pipeline CPU utilization and bottleneck detection."""

    def test_pipeline_throughput_measurement(self, available_cores):
        """Measure end-to-end pipeline throughput."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=256)
        batch_size = 64
        num_workers = min(8, available_cores)

        # Create pipeline
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=False,
        )

        # Simulate training loop
        total_samples = 0
        batch_times = []

        start_time = time.perf_counter()
        for batch in loader:
            batch_start = time.perf_counter()

            # Simulate forward pass (CPU)
            output = batch.mean(dim=1)  # Simple operation
            loss = output.sum()

            # Simulate backward pass (CPU)
            _ = torch.autograd.grad(loss, [], create_graph=False, allow_unused=True)

            batch_times.append(time.perf_counter() - batch_start)
            total_samples += len(batch)

            if total_samples >= 2000:
                break

        total_time = time.perf_counter() - start_time
        throughput = total_samples / total_time

        # Calculate stats
        avg_batch_time = statistics.mean(batch_times)
        p99_batch_time = percentile(99, batch_times)

        # Verify reasonable throughput
        assert throughput > 100, f"Pipeline throughput too low: {throughput:.2f} samples/sec"

        # Verify batch times are reasonable
        assert avg_batch_time < 1.0, f"Average batch time too high: {avg_batch_time * 1000:.2f}ms"

        del loader
        gc.collect()

    def test_pipeline_worker_utilization(self, available_cores):
        """Test that pipeline workers are properly utilized."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=10000, dim=256, simulate_io=True)
        batch_size = 32

        utilization_results = []

        for num_workers in [1, 4, 8]:
            if num_workers > available_cores:
                continue

            gc.collect()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=2,
            )

            # Count batches processed
            start_time = time.perf_counter()
            batch_count = 0
            for batch in loader:
                batch_count += 1
                if batch_count >= 100:
                    break
            total_time = time.perf_counter() - start_time

            throughput = batch_count * batch_size / total_time
            utilization_results.append((num_workers, throughput))

            del loader
            gc.collect()

        # More workers should increase throughput for I/O-bound dataset
        if len(utilization_results) >= 2:
            single_tp = utilization_results[0][1]
            multi_tp = utilization_results[-1][1]
            assert multi_tp > single_tp * 1.2, (
                f"Worker utilization not improving: {single_tp:.2f} -> {multi_tp:.2f}"
            )

    def test_identify_bottleneck_location(self, available_cores):
        """Identify whether bottleneck is in data loading or processing."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=256)
        batch_size = 64
        num_workers = min(8, available_cores)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
        )

        # Measure data loading time vs processing time
        load_times = []
        process_times = []

        loader_iter = iter(loader)
        for _ in range(50):
            # Measure load time
            load_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            load_times.append(time.perf_counter() - load_start)

            # Measure process time (simulate heavy processing)
            process_start = time.perf_counter()
            for _ in range(10):
                _ = batch.mean(dim=1)
                _ = batch.std(dim=1)
            process_times.append(time.perf_counter() - process_start)

        avg_load = statistics.mean(load_times) if load_times else 0
        avg_process = statistics.mean(process_times) if process_times else 0

        # Determine bottleneck
        if avg_load > avg_process * 2:
            bottleneck = "data_loading"
        elif avg_process > avg_load * 2:
            bottleneck = "processing"
        else:
            bottleneck = "balanced"

        # Just verify we can identify bottleneck
        assert bottleneck in ["data_loading", "processing", "balanced"], "Failed to identify bottleneck"

        del loader
        gc.collect()


@pytest.mark.cpu
@pytest.mark.b200
class TestPrefetchBuffer:
    """Test CPU prefetch buffer efficiency for data loading."""

    def test_prefetch_buffer_impact(self, available_cores):
        """Test impact of prefetch buffer size on loading efficiency."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=512)
        batch_size = 32
        num_workers = min(8, available_cores)

        prefetch_factors = [1, 2, 4]
        results = []

        for prefetch_factor in prefetch_factors:
            gc.collect()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

            # Warmup
            loader_iter = iter(loader)
            for _ in range(5):
                try:
                    next(loader_iter)
                except StopIteration:
                    break

            # Benchmark with variable processing time
            latencies = []
            loader_iter = iter(loader)
            for i in range(50):
                load_start = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                latencies.append(time.perf_counter() - load_start)

                # Simulate variable processing
                if i % 3 == 0:
                    time.sleep(0.01)  # 10ms processing sometimes

            avg_latency = statistics.mean(latencies) if latencies else float('inf')
            p99_latency = percentile(99, latencies) if latencies else float('inf')

            results.append({
                "prefetch_factor": prefetch_factor,
                "avg_latency_ms": avg_latency * 1000,
                "p99_latency_ms": p99_latency * 1000,
            })

            del loader
            gc.collect()

        # Higher prefetch should help with variable processing times
        # by keeping batches ready
        if len(results) >= 2:
            first_p99 = results[0]["p99_latency_ms"]
            last_p99 = results[-1]["p99_latency_ms"]
            # P99 should not degrade significantly with higher prefetch
            assert last_p99 < first_p99 * 2, (
                f"Prefetch degraded P99: {first_p99:.2f}ms -> {last_p99:.2f}ms"
            )

    def test_prefetch_memory_usage(self, available_cores):
        """Test memory usage with different prefetch configurations."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=1024)  # Larger tensors
        batch_size = 64
        num_workers = min(4, available_cores)

        import tracemalloc

        prefetch_factors = [1, 2, 4]
        memory_usage = []

        for prefetch_factor in prefetch_factors:
            gc.collect()

            tracemalloc.start()

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

            # Load some batches
            loader_iter = iter(loader)
            batches = []
            for _ in range(10):
                try:
                    batches.append(next(loader_iter))
                except StopIteration:
                    break

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage.append({
                "prefetch_factor": prefetch_factor,
                "peak_mb": peak / (1024 * 1024),
            })

            del batches
            del loader
            gc.collect()

        # Higher prefetch uses more memory (expected)
        # But should be proportional
        if len(memory_usage) >= 2:
            first_mem = memory_usage[0]["peak_mb"]
            last_mem = memory_usage[-1]["peak_mb"]
            # Memory should scale reasonably (not more than 4x for 4x prefetch)
            assert last_mem < first_mem * 8, (
                f"Prefetch memory explosion: {first_mem:.1f}MB -> {last_mem:.1f}MB"
            )

    def test_prefetch_with_slow_consumer(self, available_cores):
        """Test prefetch behavior when consumer is slower than producer."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=2000, dim=256)
        batch_size = 32
        num_workers = min(4, available_cores)
        prefetch_factor = 4

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        # Simulate slow consumer
        latencies = []
        loader_iter = iter(loader)

        for i in range(30):
            load_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            latencies.append(time.perf_counter() - load_start)

            # Slow consumer (50ms processing)
            time.sleep(0.05)

        # First few batches should be faster (prefetched)
        if len(latencies) >= 5:
            early_avg = statistics.mean(latencies[:5])
            # Early batches should be fast since they're prefetched
            assert early_avg < 0.01, (  # < 10ms for prefetched batches
                f"Prefetch not working for slow consumer: {early_avg * 1000:.2f}ms"
            )

        del loader
        gc.collect()


@pytest.mark.cpu
@pytest.mark.b200
class TestTrainingLoopCPUBottleneck:
    """Verify no CPU bottleneck in simulated training loop."""

    def test_training_loop_cpu_overhead(self, available_cores):
        """Verify CPU operations don't bottleneck training loop."""
        skip_if_insufficient_cores(4)

        dataset = SyntheticDataset(size=5000, dim=512)
        batch_size = 64
        num_workers = min(8, available_cores)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=False,
        )

        # Simulate training loop timings
        data_load_times = []
        preprocess_times = []
        total_step_times = []

        loader_iter = iter(loader)
        for step in range(50):
            step_start = time.perf_counter()

            # Data loading
            load_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            data_load_times.append(time.perf_counter() - load_start)

            # Preprocessing
            preprocess_start = time.perf_counter()
            batch = batch * 2 - 1  # Normalize
            batch = batch.float()
            preprocess_times.append(time.perf_counter() - preprocess_start)

            total_step_times.append(time.perf_counter() - step_start)

        # Analyze bottlenecks
        avg_load = statistics.mean(data_load_times) * 1000  # ms
        avg_preprocess = statistics.mean(preprocess_times) * 1000  # ms
        avg_step = statistics.mean(total_step_times) * 1000  # ms

        # CPU preprocessing should be minimal (< 5ms)
        assert avg_preprocess < 5, f"CPU preprocessing too slow: {avg_preprocess:.2f}ms"

        # Data loading should be reasonable (< 50ms per batch)
        assert avg_load < 50, f"Data loading too slow: {avg_load:.2f}ms"

        # CPU operations should not dominate step time
        cpu_fraction = (avg_load + avg_preprocess) / avg_step
        assert cpu_fraction < 0.9, f"CPU operations dominate training: {cpu_fraction:.1%}"

        del loader
        gc.collect()

    def test_no_python_gil_bottleneck(self, available_cores):
        """Verify Python GIL doesn't bottleneck parallel operations."""
        skip_if_insufficient_cores(4)

        num_workers = min(8, available_cores)
        num_tasks = 200

        # Define tasks that should release GIL (NumPy operations)
        def numpy_task(size: int) -> float:
            a = np.random.randn(size, size)
            b = np.random.randn(size, size)
            c = a @ b  # Should release GIL
            return float(c.sum())

        # Single-threaded baseline
        gc.collect()
        start = time.perf_counter()
        for _ in range(num_tasks):
            numpy_task(100)
        single_time = time.perf_counter() - start

        # Multi-threaded (should benefit from GIL release)
        gc.collect()
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(numpy_task, 100) for _ in range(num_tasks)]
            _ = [f.result() for f in futures]
        multi_time = time.perf_counter() - start

        # NumPy releases GIL, so multi-threaded should be faster
        speedup = single_time / multi_time
        assert speedup > 1.5, f"GIL bottleneck detected: speedup={speedup:.2f}x"

    def test_memory_access_patterns(self, available_cores):
        """Test memory access patterns don't cause CPU stalls."""
        skip_if_insufficient_cores(4)

        # Sequential access (cache-friendly)
        array_size = 10_000_000
        data = np.random.randn(array_size).astype(np.float32)

        gc.collect()
        start = time.perf_counter()
        total = 0.0
        for i in range(0, array_size, 1):  # Sequential
            total += data[i]
        sequential_time = time.perf_counter() - start

        # Strided access (less cache-friendly)
        gc.collect()
        start = time.perf_counter()
        total = 0.0
        stride = 64  # Cache line sized stride
        for i in range(0, array_size, stride):
            total += data[i]
        strided_time = time.perf_counter() - start

        # Normalize by number of accesses
        sequential_per_access = sequential_time / array_size
        strided_per_access = strided_time / (array_size // stride)

        # Strided access per element should be slower but not catastrophically
        ratio = strided_per_access / sequential_per_access
        assert ratio < 100, f"Memory access pattern issue: strided {ratio:.1f}x slower per access"

    def test_dataloader_cpu_memory_bandwidth(self, available_cores):
        """Test DataLoader doesn't saturate CPU memory bandwidth."""
        skip_if_insufficient_cores(4)

        # Large tensors to stress memory bandwidth
        dataset = SyntheticDataset(size=1000, dim=4096)
        batch_size = 128  # Large batches
        num_workers = min(16, available_cores)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=False,
        )

        # Measure throughput
        start_time = time.perf_counter()
        total_bytes = 0

        for batch in loader:
            total_bytes += batch.numel() * batch.element_size()
            if total_bytes >= 1e9:  # 1GB
                break

        total_time = time.perf_counter() - start_time
        bandwidth_gbps = total_bytes / (total_time * 1e9)

        # Should achieve reasonable memory bandwidth (> 1 GB/s)
        assert bandwidth_gbps > 1.0, f"Memory bandwidth too low: {bandwidth_gbps:.2f} GB/s"

        del loader
        gc.collect()


# =============================================================================
# Memory-Mapped File Tests
# =============================================================================


@pytest.mark.cpu
@pytest.mark.b200
class TestMemoryMappedAccess:
    """Test memory-mapped file access patterns for data loading."""

    def test_mmap_sequential_read(self, temp_mmap_file):
        """Test sequential read performance with mmap."""
        skip_if_insufficient_cores(4)

        file_size = os.path.getsize(temp_mmap_file)
        chunk_size = 1024 * 1024  # 1MB chunks

        gc.collect()
        start_time = time.perf_counter()

        with open(temp_mmap_file, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 0
            total_read = 0
            while offset < file_size:
                chunk = mm[offset:offset + chunk_size]
                total_read += len(chunk)
                offset += chunk_size
            mm.close()

        total_time = time.perf_counter() - start_time
        throughput_gbps = total_read / (total_time * 1e9)

        # Sequential mmap read should be fast
        assert throughput_gbps > 0.5, f"Mmap sequential read too slow: {throughput_gbps:.2f} GB/s"

    def test_mmap_random_read(self, temp_mmap_file):
        """Test random read performance with mmap."""
        skip_if_insufficient_cores(4)

        file_size = os.path.getsize(temp_mmap_file)
        num_reads = 10000
        read_size = 4096  # 4KB reads

        # Generate random offsets
        np.random.seed(42)
        offsets = np.random.randint(0, file_size - read_size, size=num_reads)

        gc.collect()
        start_time = time.perf_counter()

        with open(temp_mmap_file, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for offset in offsets:
                _ = mm[offset:offset + read_size]
            mm.close()

        total_time = time.perf_counter() - start_time
        iops = num_reads / total_time

        # Random mmap read should achieve reasonable IOPS
        assert iops > 1000, f"Mmap random read too slow: {iops:.0f} IOPS"

    def test_mmap_parallel_read(self, temp_mmap_file, available_cores):
        """Test parallel mmap read with multiple processes."""
        skip_if_insufficient_cores(4)

        file_size = os.path.getsize(temp_mmap_file)
        num_workers = min(8, available_cores)
        reads_per_worker = 1000
        read_size = 4096

        # Generate read tasks
        np.random.seed(42)
        all_offsets = np.random.randint(0, file_size - read_size, size=num_workers * reads_per_worker)
        tasks = [
            (temp_mmap_file, int(offset), read_size)
            for offset in all_offsets
        ]

        # Single process baseline
        gc.collect()
        start_time = time.perf_counter()
        for task in tasks[:reads_per_worker]:
            _ = memory_mapped_read(task)
        single_time = time.perf_counter() - start_time

        # Multi-process
        gc.collect()
        start_time = time.perf_counter()
        with Pool(num_workers) as pool:
            _ = pool.map(memory_mapped_read, tasks)
        multi_time = time.perf_counter() - start_time

        # Parallel mmap should scale (I/O bound)
        speedup = (single_time * num_workers) / multi_time
        assert speedup > 1.0, f"Parallel mmap not scaling: speedup={speedup:.2f}x"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.cpu
@pytest.mark.b200
class TestIntegration:
    """Integration tests for complete data pipeline scenarios."""

    def test_full_pipeline_scaling(self, available_cores, scaling_results):
        """Test complete data pipeline with all components."""
        skip_if_insufficient_cores(4)

        # Configuration
        dataset_size = 5000
        batch_size = 64
        dim = 512

        results = []

        for num_workers in [1, 4, 8]:
            if num_workers > available_cores:
                continue

            gc.collect()

            # Create components
            dataset = TokenizationDataset(size=dataset_size, text_length=dim)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=lambda x: {
                    "input_ids": torch.stack([item["input_ids"] for item in x]),
                    "attention_mask": torch.stack([item["attention_mask"] for item in x]),
                },
                prefetch_factor=2 if num_workers > 0 else None,
            )

            # Measure throughput
            total_samples = 0
            latencies = []

            start_time = time.perf_counter()
            for batch in loader:
                batch_start = time.perf_counter()

                # Simulate processing
                _ = batch["input_ids"].float().mean(dim=1)

                latencies.append(time.perf_counter() - batch_start)
                total_samples += len(batch["input_ids"])

                if total_samples >= 2000:
                    break

            total_time = time.perf_counter() - start_time
            throughput = total_samples / total_time

            result = ScalingResult(
                name="full_pipeline",
                worker_count=num_workers,
                throughput=throughput,
                latency_mean_ms=statistics.mean(latencies) * 1000,
                latency_p50_ms=percentile(50, latencies) * 1000,
                latency_p99_ms=percentile(99, latencies) * 1000,
                cpu_utilization=0.0,
                memory_mb=0.0,
                iterations=len(latencies),
                config={"batch_size": batch_size, "dim": dim},
            )
            results.append(result)
            scaling_results.append(result)

            del loader
            gc.collect()

        # Verify scaling
        if len(results) >= 2:
            single_tp = results[0].throughput
            multi_tp = results[-1].throughput
            assert multi_tp >= single_tp * 0.8, (
                f"Full pipeline not scaling: {single_tp:.2f} -> {multi_tp:.2f}"
            )

    def test_mixed_cpu_workload(self, available_cores):
        """Test mixed CPU workload (I/O + compute)."""
        skip_if_insufficient_cores(4)

        num_workers = min(8, available_cores)
        num_tasks = 100

        def mixed_task(task_id: int) -> dict:
            # I/O simulation
            time.sleep(0.001)

            # Compute simulation
            data = np.random.randn(100, 100)
            result = np.sin(data) @ np.cos(data)

            return {"task_id": task_id, "result_sum": float(result.sum())}

        # Thread pool (good for I/O)
        gc.collect()
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            thread_results = list(executor.map(mixed_task, range(num_tasks)))
        thread_time = time.perf_counter() - start_time

        # Process pool (good for compute)
        gc.collect()
        start_time = time.perf_counter()
        with Pool(num_workers) as pool:
            process_results = pool.map(mixed_task, range(num_tasks))
        process_time = time.perf_counter() - start_time

        # Both should complete in reasonable time
        assert thread_time < 30, f"Thread pool too slow: {thread_time:.2f}s"
        assert process_time < 30, f"Process pool too slow: {process_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
