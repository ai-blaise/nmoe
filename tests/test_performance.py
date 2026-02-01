"""Performance regression tests for nmoe.

These tests ensure that key operations maintain acceptable performance levels.
They run on GPU and compare against baseline thresholds.

Usage:
    pytest tests/test_performance.py -v --tb=short
    pytest tests/test_performance.py -v -m "not slow"  # Skip slow benchmarks
"""

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch


# Skip entire module if no CUDA
pytestmark = pytest.mark.gpu


@dataclass
class PerfResult:
    """Performance measurement result."""

    name: str
    latency_p50_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    throughput: float  # tokens/sec or ops/sec
    memory_peak_gb: float
    iterations: int
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_mean_ms": self.latency_mean_ms,
            "throughput": self.throughput,
            "memory_peak_gb": self.memory_peak_gb,
            "iterations": self.iterations,
            "config": self.config,
        }


@dataclass
class PerfBaseline:
    """Performance baseline for regression testing."""

    name: str
    max_latency_p50_ms: float
    max_latency_p99_ms: float
    min_throughput: float
    tolerance: float = 0.15  # 15% tolerance by default

    def check(self, result: PerfResult) -> tuple[bool, list[str]]:
        """Check if result meets baseline within tolerance.

        Returns:
            Tuple of (passed, list of failure messages)
        """
        failures = []
        threshold_p50 = self.max_latency_p50_ms * (1 + self.tolerance)
        threshold_p99 = self.max_latency_p99_ms * (1 + self.tolerance)
        threshold_throughput = self.min_throughput * (1 - self.tolerance)

        if result.latency_p50_ms > threshold_p50:
            failures.append(
                f"P50 latency {result.latency_p50_ms:.3f}ms exceeds threshold "
                f"{threshold_p50:.3f}ms (baseline: {self.max_latency_p50_ms:.3f}ms)"
            )

        if result.latency_p99_ms > threshold_p99:
            failures.append(
                f"P99 latency {result.latency_p99_ms:.3f}ms exceeds threshold "
                f"{threshold_p99:.3f}ms (baseline: {self.max_latency_p99_ms:.3f}ms)"
            )

        if result.throughput < threshold_throughput:
            failures.append(
                f"Throughput {result.throughput:.1f} below threshold "
                f"{threshold_throughput:.1f} (baseline: {self.min_throughput:.1f})"
            )

        return len(failures) == 0, failures


def _percentile(pct: float, xs: list[float]) -> float:
    """Compute percentile of a list."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = int(round((pct / 100.0) * (len(xs) - 1)))
    return xs[idx]


def _measure_events(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> list[float]:
    """Convert CUDA events to milliseconds."""
    return [start.elapsed_time(end) for start, end in events]


class PerfBenchmark:
    """Base class for performance benchmarks."""

    def __init__(self, warmup: int = 20, iterations: int = 100):
        self.warmup = warmup
        self.iterations = iterations

    def run_benchmark(
        self,
        name: str,
        fn: callable,
        tokens_per_iter: int = 0,
        config: dict | None = None,
    ) -> PerfResult:
        """Run a benchmark and return results.

        Args:
            name: Name of the benchmark
            fn: Function to benchmark (should be CUDA operation)
            tokens_per_iter: Number of tokens processed per iteration (for throughput)
            config: Configuration dict for logging
        """
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        # Timed iterations
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))

        torch.cuda.synchronize()

        # Compute stats
        ms = _measure_events(events)
        p50 = _percentile(50, ms)
        p99 = _percentile(99, ms)
        mean = statistics.mean(ms)

        # Throughput
        if tokens_per_iter > 0:
            throughput = tokens_per_iter / (mean / 1000)  # tokens/sec
        else:
            throughput = 1000 / mean  # ops/sec

        # Memory
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        return PerfResult(
            name=name,
            latency_p50_ms=p50,
            latency_p99_ms=p99,
            latency_mean_ms=mean,
            throughput=throughput,
            memory_peak_gb=peak_memory,
            iterations=self.iterations,
            config=config or {},
        )


# =============================================================================
# Baselines - Adjust these based on hardware (B200 targets)
# =============================================================================

# Dispatch operation baselines (T=4096, H=2048, E=8, K=2)
BASELINE_DISPATCH_BF16 = PerfBaseline(
    name="dispatch_bf16",
    max_latency_p50_ms=0.5,
    max_latency_p99_ms=1.0,
    min_throughput=8_000_000,  # 8M tokens/sec
)

BASELINE_DISPATCH_FP8 = PerfBaseline(
    name="dispatch_fp8",
    max_latency_p50_ms=0.4,
    max_latency_p99_ms=0.8,
    min_throughput=10_000_000,  # 10M tokens/sec
)

BASELINE_DISPATCH_NVFP4 = PerfBaseline(
    name="dispatch_nvfp4",
    max_latency_p50_ms=0.3,
    max_latency_p99_ms=0.6,
    min_throughput=12_000_000,  # 12M tokens/sec
)

# MoE forward pass baselines (T=4096, H=2048, E=8, K=2, Dff=1408)
BASELINE_MOE_FWD_FP8 = PerfBaseline(
    name="moe_forward_fp8",
    max_latency_p50_ms=2.0,
    max_latency_p99_ms=4.0,
    min_throughput=2_000_000,  # 2M tokens/sec
)

BASELINE_MOE_FWD_NVFP4 = PerfBaseline(
    name="moe_forward_nvfp4",
    max_latency_p50_ms=1.5,
    max_latency_p99_ms=3.0,
    min_throughput=2_500_000,  # 2.5M tokens/sec
)

# MoE forward+backward baselines
BASELINE_MOE_FWDBWD_FP8 = PerfBaseline(
    name="moe_fwdbwd_fp8",
    max_latency_p50_ms=8.0,
    max_latency_p99_ms=15.0,
    min_throughput=500_000,  # 500K tokens/sec
)

BASELINE_MOE_FWDBWD_NVFP4 = PerfBaseline(
    name="moe_fwdbwd_nvfp4",
    max_latency_p50_ms=6.0,
    max_latency_p99_ms=12.0,
    min_throughput=650_000,  # 650K tokens/sec
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def perf_benchmark():
    """Create a benchmark runner."""
    return PerfBenchmark(warmup=20, iterations=100)


@pytest.fixture(scope="module")
def perf_results():
    """Collect performance results for reporting."""
    results = []
    yield results

    # After all tests, optionally save results
    if results:
        results_path = Path(os.environ.get("NMOE_PERF_RESULTS", "/tmp/nmoe_perf_results.json"))
        with open(results_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)


@pytest.fixture(scope="module")
def rdep_bf16():
    """Create RDEP for BF16 profile."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from nmoe.rdep import Rdep

    return Rdep(dim=2048, n_local=8, topk=2, profile="bf16", capacity=32768)


@pytest.fixture(scope="module")
def rdep_fp8():
    """Create RDEP for FP8 profile."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from nmoe.rdep import Rdep

    # FP8 profile automatically allocates blockscaled buffers
    return Rdep(dim=2048, n_local=8, topk=2, profile="fp8", capacity=32768)


@pytest.fixture(scope="module")
def rdep_nvfp4():
    """Create RDEP for NVFP4 profile."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from nmoe.rdep import Rdep

    # NVFP4 profile automatically allocates blockscaled buffers
    return Rdep(dim=2048, n_local=8, topk=2, profile="nvfp4", capacity=32768)


# =============================================================================
# Dispatch Kernel Benchmarks
# =============================================================================


@pytest.mark.gpu
class TestDispatchPerformance:
    """Test dispatch kernel performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        self.device = torch.device("cuda")
        self.T = 4096
        self.H = 2048
        self.E = 8
        self.K = 2

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.x = torch.randn((self.T, self.H), device=self.device, dtype=torch.bfloat16)
        self.eids = torch.randint(0, self.E, (self.T, self.K), device=self.device, dtype=torch.int32)
        self.gates = torch.softmax(torch.randn((self.T, self.K), device=self.device), dim=-1).float()

    def test_dispatch_bf16_performance(self, perf_benchmark, perf_results, rdep_bf16):
        """Test BF16 dispatch meets performance baseline."""
        from nmoe.csrc import rdep as _C

        stream = torch.cuda.current_stream(self.device)
        offs_pad = torch.empty(self.E, device=self.device, dtype=torch.int32)
        M_host = torch.zeros(1, device="cpu", dtype=torch.int32).pin_memory()

        def run_dispatch():
            return _C.dispatch_meta_bf16(
                self.x.data_ptr(),
                self.eids.data_ptr(),
                self.gates.data_ptr(),
                int(self.T),
                int(self.K),
                128,  # block size
                offs_pad.data_ptr(),
                M_host.data_ptr(),
                stream,
            )

        result = perf_benchmark.run_benchmark(
            name="dispatch_bf16",
            fn=run_dispatch,
            tokens_per_iter=self.T,
            config={"T": self.T, "H": self.H, "E": self.E, "K": self.K, "profile": "bf16"},
        )
        perf_results.append(result)

        passed, failures = BASELINE_DISPATCH_BF16.check(result)
        if not passed:
            pytest.fail("\n".join(failures))

    def test_dispatch_blockscaled_performance(self, perf_benchmark, perf_results, rdep_fp8):
        """Test blockscaled dispatch meets performance baseline."""
        from nmoe.csrc import rdep as _C

        stream = torch.cuda.current_stream(self.device)
        offs_pad = torch.empty(self.E, device=self.device, dtype=torch.int32)
        M_host = torch.zeros(1, device="cpu", dtype=torch.int32).pin_memory()

        def run_dispatch():
            return _C.dispatch_meta_blockscaled(
                self.x.data_ptr(),
                self.eids.data_ptr(),
                self.gates.data_ptr(),
                int(self.T),
                int(self.K),
                offs_pad.data_ptr(),
                M_host.data_ptr(),
                stream,
            )

        result = perf_benchmark.run_benchmark(
            name="dispatch_blockscaled",
            fn=run_dispatch,
            tokens_per_iter=self.T,
            config={"T": self.T, "H": self.H, "E": self.E, "K": self.K, "profile": "fp8"},
        )
        perf_results.append(result)

        # Use FP8 baseline (blockscaled is used for both FP8 and NVFP4)
        passed, failures = BASELINE_DISPATCH_FP8.check(result)
        if not passed:
            pytest.fail("\n".join(failures))


# =============================================================================
# MoE Forward Pass Benchmarks
# =============================================================================


@pytest.mark.gpu
@pytest.mark.slow
class TestMoEForwardPerformance:
    """Test MoE forward pass performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        self.device = torch.device("cuda")
        self.T = 4096
        self.H = 2048
        self.Dff = 1408
        self.E = 8
        self.K = 2

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.x = torch.randn((self.T, self.H), device=self.device, dtype=torch.bfloat16) * 0.1
        self.eids = torch.randint(0, self.E, (self.T, self.K), device=self.device, dtype=torch.int32)
        self.gates = torch.softmax(torch.randn((self.T, self.K), device=self.device), dim=-1).to(torch.bfloat16)

        # Weights
        self.W1 = torch.randn((self.E, self.H, self.Dff), device=self.device, dtype=torch.bfloat16) * 0.02
        self.W3 = torch.randn((self.E, self.H, self.Dff), device=self.device, dtype=torch.bfloat16) * 0.02
        self.W2 = torch.randn((self.E, self.Dff, self.H), device=self.device, dtype=torch.bfloat16) * 0.02

    def test_moe_forward_fp8_performance(self, perf_benchmark, perf_results, rdep_fp8):
        """Test FP8 MoE forward pass meets performance baseline."""
        from nmoe.blockscaled.grouped import quantize_weights
        from nmoe.moe import _MoEBlockscaledFused

        W_cache = quantize_weights(self.W1, self.W3, self.W2, profile="fp8")

        def run_forward():
            return _MoEBlockscaledFused.apply(
                rdep_fp8, self.x, self.eids, self.gates, self.W1, self.W3, self.W2, W_cache
            )

        result = perf_benchmark.run_benchmark(
            name="moe_forward_fp8",
            fn=run_forward,
            tokens_per_iter=self.T,
            config={"T": self.T, "H": self.H, "Dff": self.Dff, "E": self.E, "K": self.K, "profile": "fp8"},
        )
        perf_results.append(result)

        passed, failures = BASELINE_MOE_FWD_FP8.check(result)
        if not passed:
            pytest.fail("\n".join(failures))

    def test_moe_forward_nvfp4_performance(self, perf_benchmark, perf_results, rdep_nvfp4):
        """Test NVFP4 MoE forward pass meets performance baseline."""
        from nmoe.blockscaled.grouped import quantize_weights
        from nmoe.moe import _MoEBlockscaledFused

        W_cache = quantize_weights(self.W1, self.W3, self.W2, profile="nvfp4")

        def run_forward():
            return _MoEBlockscaledFused.apply(
                rdep_nvfp4, self.x, self.eids, self.gates, self.W1, self.W3, self.W2, W_cache
            )

        result = perf_benchmark.run_benchmark(
            name="moe_forward_nvfp4",
            fn=run_forward,
            tokens_per_iter=self.T,
            config={"T": self.T, "H": self.H, "Dff": self.Dff, "E": self.E, "K": self.K, "profile": "nvfp4"},
        )
        perf_results.append(result)

        passed, failures = BASELINE_MOE_FWD_NVFP4.check(result)
        if not passed:
            pytest.fail("\n".join(failures))


# =============================================================================
# MoE Forward+Backward Benchmarks
# =============================================================================


@pytest.mark.gpu
@pytest.mark.slow
class TestMoEBackwardPerformance:
    """Test MoE forward+backward pass performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        self.device = torch.device("cuda")
        self.T = 4096
        self.H = 2048
        self.Dff = 1408
        self.E = 8
        self.K = 2

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.x = torch.randn((self.T, self.H), device=self.device, dtype=torch.bfloat16) * 0.1
        self.eids = torch.randint(0, self.E, (self.T, self.K), device=self.device, dtype=torch.int32)
        self.gates = torch.softmax(torch.randn((self.T, self.K), device=self.device), dim=-1).to(torch.bfloat16)

        # Weights
        self.W1 = torch.randn((self.E, self.H, self.Dff), device=self.device, dtype=torch.bfloat16) * 0.02
        self.W3 = torch.randn((self.E, self.H, self.Dff), device=self.device, dtype=torch.bfloat16) * 0.02
        self.W2 = torch.randn((self.E, self.Dff, self.H), device=self.device, dtype=torch.bfloat16) * 0.02

    def test_moe_fwdbwd_fp8_performance(self, perf_benchmark, perf_results, rdep_fp8):
        """Test FP8 MoE forward+backward pass meets performance baseline."""
        from nmoe.blockscaled.grouped import quantize_weights
        from nmoe.moe import _MoEBlockscaledFused

        W_cache = quantize_weights(self.W1, self.W3, self.W2, profile="fp8")

        def run_fwdbwd():
            x_grad = self.x.clone().requires_grad_(True)
            out = _MoEBlockscaledFused.apply(
                rdep_fp8, x_grad, self.eids, self.gates, self.W1, self.W3, self.W2, W_cache
            )
            loss = out.sum()
            loss.backward()
            return out

        result = perf_benchmark.run_benchmark(
            name="moe_fwdbwd_fp8",
            fn=run_fwdbwd,
            tokens_per_iter=self.T,
            config={"T": self.T, "H": self.H, "Dff": self.Dff, "E": self.E, "K": self.K, "profile": "fp8"},
        )
        perf_results.append(result)

        passed, failures = BASELINE_MOE_FWDBWD_FP8.check(result)
        if not passed:
            pytest.fail("\n".join(failures))

    def test_moe_fwdbwd_nvfp4_performance(self, perf_benchmark, perf_results, rdep_nvfp4):
        """Test NVFP4 MoE forward+backward pass meets performance baseline."""
        from nmoe.blockscaled.grouped import quantize_weights
        from nmoe.moe import _MoEBlockscaledFused

        W_cache = quantize_weights(self.W1, self.W3, self.W2, profile="nvfp4")

        def run_fwdbwd():
            x_grad = self.x.clone().requires_grad_(True)
            out = _MoEBlockscaledFused.apply(
                rdep_nvfp4, x_grad, self.eids, self.gates, self.W1, self.W3, self.W2, W_cache
            )
            loss = out.sum()
            loss.backward()
            return out

        result = perf_benchmark.run_benchmark(
            name="moe_fwdbwd_nvfp4",
            fn=run_fwdbwd,
            tokens_per_iter=self.T,
            config={"T": self.T, "H": self.H, "Dff": self.Dff, "E": self.E, "K": self.K, "profile": "nvfp4"},
        )
        perf_results.append(result)

        passed, failures = BASELINE_MOE_FWDBWD_NVFP4.check(result)
        if not passed:
            pytest.fail("\n".join(failures))


# =============================================================================
# Memory Benchmarks
# =============================================================================


@pytest.mark.gpu
class TestMemoryPerformance:
    """Test memory usage patterns."""

    def test_rdep_memory_usage(self):
        """Test RDEP memory allocation is within expected bounds."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nmoe.rdep import Rdep

        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1e9

        # Create RDEP with typical configuration
        # NVFP4 profile automatically allocates blockscaled buffers
        rdep = Rdep(dim=2048, n_local=8, topk=2, profile="nvfp4", capacity=32768)

        rdep_memory = (torch.cuda.memory_allocated() / 1e9) - initial_memory

        # RDEP should use less than 2GB for this configuration
        assert rdep_memory < 2.0, f"RDEP memory usage {rdep_memory:.2f}GB exceeds 2GB threshold"

    def test_weight_cache_memory(self):
        """Test weight cache memory usage is reasonable."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nmoe.blockscaled.grouped import quantize_weights

        device = torch.device("cuda")
        E, H, Dff = 8, 2048, 1408

        torch.cuda.reset_peak_memory_stats()

        # BF16 weights
        W1 = torch.randn((E, H, Dff), device=device, dtype=torch.bfloat16)
        W3 = torch.randn((E, H, Dff), device=device, dtype=torch.bfloat16)
        W2 = torch.randn((E, Dff, H), device=device, dtype=torch.bfloat16)

        bf16_memory = torch.cuda.memory_allocated() / 1e9

        # Create quantized cache
        W_cache = quantize_weights(W1, W3, W2, profile="nvfp4")

        total_memory = torch.cuda.memory_allocated() / 1e9
        cache_memory = total_memory - bf16_memory

        # NVFP4 cache should be smaller than BF16 weights (4x compression)
        assert cache_memory < bf16_memory * 0.4, (
            f"Cache memory {cache_memory:.2f}GB should be < 40% of BF16 memory {bf16_memory:.2f}GB"
        )


# =============================================================================
# Throughput Scaling Tests
# =============================================================================


@pytest.mark.gpu
@pytest.mark.slow
class TestThroughputScaling:
    """Test that throughput scales with batch size."""

    def test_throughput_increases_with_batch(self):
        """Test that throughput increases with larger batch sizes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nmoe.blockscaled.grouped import quantize_weights
        from nmoe.moe import _MoEBlockscaledFused
        from nmoe.rdep import Rdep

        device = torch.device("cuda")
        H, Dff, E, K = 2048, 1408, 8, 2

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Weights
        W1 = torch.randn((E, H, Dff), device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn((E, H, Dff), device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn((E, Dff, H), device=device, dtype=torch.bfloat16) * 0.02
        W_cache = quantize_weights(W1, W3, W2, profile="nvfp4")

        batch_sizes = [256, 1024, 4096]
        throughputs = []

        for T in batch_sizes:
            # NVFP4 profile automatically allocates blockscaled buffers
            rdep = Rdep(dim=H, n_local=E, topk=K, profile="nvfp4", capacity=T * K * 2)

            x = torch.randn((T, H), device=device, dtype=torch.bfloat16)
            eids = torch.randint(0, E, (T, K), device=device, dtype=torch.int32)
            gates = torch.softmax(torch.randn((T, K), device=device), dim=-1).to(torch.bfloat16)

            # Warmup
            for _ in range(10):
                _MoEBlockscaledFused.apply(rdep, x, eids, gates, W1, W3, W2, W_cache)
            torch.cuda.synchronize()

            # Time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(20):
                _MoEBlockscaledFused.apply(rdep, x, eids, gates, W1, W3, W2, W_cache)
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end) / 20
            throughput = T / (elapsed_ms / 1000)
            throughputs.append(throughput)

        # Throughput should increase with batch size (at least 2x from 256 to 4096)
        assert throughputs[-1] > throughputs[0] * 2, (
            f"Throughput did not scale: {throughputs[0]:.0f} -> {throughputs[-1]:.0f}"
        )
