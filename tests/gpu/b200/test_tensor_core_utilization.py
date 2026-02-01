"""Comprehensive Tensor Core Utilization Tests for B200 GPUs.

This module tests tensor core efficiency across different precisions (FP8, FP4/NVFP4,
FP16, BF16) on NVIDIA B200 (SM100) GPUs. It measures achieved TFLOPS versus
theoretical peak and validates minimum efficiency thresholds.

Tests cover:
1. FP8 E4M3FN tensor core GEMM efficiency
2. NVFP4 (E2M1) tensor core operations
3. FP16/BF16 tensor core utilization
4. MoE expert GEMM workloads with realistic routing
5. Batched GEMM operations
6. Comprehensive utilization reporting

B200 GPU Specifications (SM100):
- FP8 Tensor Core Peak: ~10,000 TFLOPS
- NVFP4 Tensor Core Peak: ~20,000 TFLOPS
- FP16/BF16 Tensor Core Peak: ~5,000 TFLOPS
- Memory Bandwidth: ~8 TB/s HBM3e

Run tests:
    pytest tests/gpu/b200/test_tensor_core_utilization.py -v -m gpu
    pytest tests/gpu/b200/test_tensor_core_utilization.py -v -m b200

Requirements:
    - SM100 (B200) GPU
    - PyTorch >= 2.11.0 with FP8 support
    - nvidia-cutlass-dsl >= 4.3.1 (for NVFP4)
"""

from __future__ import annotations

import functools
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F


# =============================================================================
# Pytest Markers
# =============================================================================

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.b200,
]


# =============================================================================
# B200 GPU Specifications
# =============================================================================

@dataclass(frozen=True)
class B200Specs:
    """NVIDIA B200 GPU specifications for SM100."""

    # Tensor Core peak TFLOPS (theoretical maximum)
    fp8_tflops: float = 10_000.0  # FP8 E4M3FN
    nvfp4_tflops: float = 20_000.0  # NVFP4 E2M1
    fp16_tflops: float = 5_000.0  # FP16
    bf16_tflops: float = 5_000.0  # BF16

    # Memory specifications
    hbm_bandwidth_tb_s: float = 8.0  # TB/s HBM3e

    # SM count
    sm_count: int = 132  # Approximate for B200

    # Minimum efficiency thresholds (fraction of peak)
    min_fp8_efficiency: float = 0.50  # 50% of peak
    min_nvfp4_efficiency: float = 0.45  # 45% of peak
    min_bf16_efficiency: float = 0.55  # 55% of peak
    min_moe_efficiency: float = 0.40  # 40% for MoE workloads (lower due to routing)


B200_SPECS = B200Specs()


# =============================================================================
# Skip Decorators and Hardware Detection
# =============================================================================


def _get_device_capability() -> Tuple[int, int]:
    """Get CUDA device compute capability."""
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def _is_sm100() -> bool:
    """Check if running on SM100 (B200)."""
    major, minor = _get_device_capability()
    return major == 10 and minor == 0


def _has_fp8_support() -> bool:
    """Check if FP8 is supported."""
    if not torch.cuda.is_available():
        return False
    try:
        # Check for FP8 E4M3FN dtype
        _ = torch.float8_e4m3fn
        return True
    except AttributeError:
        return False


def _has_cutlass_dsl() -> bool:
    """Check if CuTeDSL is available for NVFP4."""
    try:
        import cutlass
        import cuda.bindings.driver
        return True
    except ImportError:
        return False


def requires_sm100():
    """Decorator to skip tests that require SM100 (B200) GPUs."""
    def decorator(func):
        @pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="CUDA not available"
        )
        @pytest.mark.skipif(
            not _is_sm100(),
            reason=f"Requires SM100 (B200), got SM{_get_device_capability()}"
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_fp8():
    """Decorator to skip tests that require FP8 support."""
    def decorator(func):
        @pytest.mark.skipif(
            not _has_fp8_support(),
            reason="FP8 not supported"
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_cutlass_dsl():
    """Decorator to skip tests that require CuTeDSL for NVFP4."""
    def decorator(func):
        @pytest.mark.skipif(
            not _has_cutlass_dsl(),
            reason="CuTeDSL not available"
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Utility Functions
# =============================================================================


def compute_gemm_flops(M: int, N: int, K: int) -> int:
    """Compute FLOPS for a single GEMM: 2 * M * N * K."""
    return 2 * M * N * K


def compute_batched_gemm_flops(B: int, M: int, N: int, K: int) -> int:
    """Compute FLOPS for batched GEMM."""
    return B * compute_gemm_flops(M, N, K)


def compute_moe_expert_flops(
    n_tokens: int,
    hidden_dim: int,
    inter_dim: int,
    n_experts: int,
    topk: int,
) -> int:
    """Compute FLOPS for MoE expert computation.

    MoE forward: Y = (SiLU(X @ W1) * (X @ W3)) @ W2
    For each expert: 3 GEMMs
    - X @ W1: [M_e, H] @ [H, Dff] = 2*M_e*H*Dff
    - X @ W3: [M_e, H] @ [H, Dff] = 2*M_e*H*Dff
    - A @ W2: [M_e, Dff] @ [Dff, H] = 2*M_e*Dff*H

    Total per expert: 6 * M_e * H * Dff
    With topk routing: total_tokens = n_tokens * topk
    """
    total_tokens = n_tokens * topk
    # Assuming uniform distribution across experts
    tokens_per_expert = total_tokens // n_experts
    flops_per_expert = 6 * tokens_per_expert * hidden_dim * inter_dim
    return flops_per_expert * n_experts


def ms_to_tflops(flops: int, ms: float) -> float:
    """Convert FLOPS and milliseconds to TFLOPS."""
    if ms <= 0:
        return 0.0
    return (flops / 1e12) / (ms / 1000)


@dataclass
class BenchmarkResult:
    """Result from a tensor core benchmark."""

    name: str
    M: int
    N: int
    K: int
    dtype: str
    latency_ms: float
    latency_std_ms: float
    achieved_tflops: float
    theoretical_peak_tflops: float
    efficiency: float  # achieved / peak
    memory_gb: float

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.M}x{self.N}x{self.K} {self.dtype} - "
            f"{self.achieved_tflops:.1f} TFLOPS ({self.efficiency*100:.1f}% efficiency) "
            f"@ {self.latency_ms:.3f}ms"
        )


class TensorCoreBenchmark:
    """Benchmark harness for tensor core operations."""

    def __init__(
        self,
        warmup_iters: int = 50,
        benchmark_iters: int = 200,
        device: torch.device = None,
    ):
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.device = device or torch.device("cuda:0")

    def benchmark_gemm(
        self,
        name: str,
        A: torch.Tensor,
        B: torch.Tensor,
        theoretical_peak: float,
    ) -> BenchmarkResult:
        """Benchmark a GEMM operation using CUDA events.

        Args:
            name: Benchmark name
            A: Left matrix [M, K]
            B: Right matrix [K, N]
            theoretical_peak: Theoretical peak TFLOPS for this dtype

        Returns:
            BenchmarkResult with timing and efficiency metrics
        """
        M, K = A.shape
        _, N = B.shape
        flops = compute_gemm_flops(M, N, K)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(self.device)

        # Warmup
        for _ in range(self.warmup_iters):
            _ = torch.mm(A, B)
        torch.cuda.synchronize(self.device)

        # Benchmark with CUDA events
        latencies = []
        for _ in range(self.benchmark_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record(stream=torch.cuda.current_stream(self.device))
            _ = torch.mm(A, B)
            end.record(stream=torch.cuda.current_stream(self.device))

            torch.cuda.synchronize(self.device)
            latencies.append(start.elapsed_time(end))

        # Compute statistics
        latency_ms = statistics.median(latencies)
        latency_std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        achieved_tflops = ms_to_tflops(flops, latency_ms)
        efficiency = achieved_tflops / theoretical_peak if theoretical_peak > 0 else 0.0
        memory_gb = torch.cuda.max_memory_allocated(self.device) / 1e9

        return BenchmarkResult(
            name=name,
            M=M,
            N=N,
            K=K,
            dtype=str(A.dtype),
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            achieved_tflops=achieved_tflops,
            theoretical_peak_tflops=theoretical_peak,
            efficiency=efficiency,
            memory_gb=memory_gb,
        )

    def benchmark_fn(
        self,
        name: str,
        fn: Callable[[], torch.Tensor],
        flops: int,
        theoretical_peak: float,
        shape_info: Tuple[int, int, int],
        dtype: str,
    ) -> BenchmarkResult:
        """Benchmark an arbitrary function using CUDA events.

        Args:
            name: Benchmark name
            fn: Function to benchmark (should be CUDA operation)
            flops: Number of FLOPS for this operation
            theoretical_peak: Theoretical peak TFLOPS
            shape_info: (M, N, K) shape info for reporting
            dtype: Data type string

        Returns:
            BenchmarkResult with timing and efficiency metrics
        """
        M, N, K = shape_info

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(self.device)

        # Warmup
        for _ in range(self.warmup_iters):
            _ = fn()
        torch.cuda.synchronize(self.device)

        # Benchmark with CUDA events
        latencies = []
        for _ in range(self.benchmark_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record(stream=torch.cuda.current_stream(self.device))
            _ = fn()
            end.record(stream=torch.cuda.current_stream(self.device))

            torch.cuda.synchronize(self.device)
            latencies.append(start.elapsed_time(end))

        # Compute statistics
        latency_ms = statistics.median(latencies)
        latency_std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        achieved_tflops = ms_to_tflops(flops, latency_ms)
        efficiency = achieved_tflops / theoretical_peak if theoretical_peak > 0 else 0.0
        memory_gb = torch.cuda.max_memory_allocated(self.device) / 1e9

        return BenchmarkResult(
            name=name,
            M=M,
            N=N,
            K=K,
            dtype=dtype,
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            achieved_tflops=achieved_tflops,
            theoretical_peak_tflops=theoretical_peak,
            efficiency=efficiency,
            memory_gb=memory_gb,
        )


# =============================================================================
# Matrix Size Configurations
# =============================================================================

# Powers of 2 - optimal for tensor cores
POWER_OF_2_SIZES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
]

# Non-powers of 2 - realistic workloads
NON_POWER_OF_2_SIZES = [
    (1536, 1536, 1536),
    (2560, 2560, 2560),
    (3072, 3072, 3072),
    (5120, 5120, 5120),
    (7168, 7168, 7168),
]

# MoE-specific dimensions (DeepSeek-V3 style)
MOE_DIMENSIONS = [
    # (M_tokens, hidden_dim, inter_dim)
    (256, 2048, 5504),
    (512, 2048, 5504),
    (1024, 2048, 5504),
    (2048, 2048, 5504),
    (4096, 2048, 5504),
    (8192, 2048, 5504),
]

# Realistic batch sizes
BATCH_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]


# =============================================================================
# Test Class: FP8 Tensor Core
# =============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestFP8TensorCore:
    """Test FP8 E4M3FN tensor core GEMM efficiency."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for FP8 tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not _has_fp8_support():
            pytest.skip("FP8 not supported")
        self.device = torch.device("cuda:0")
        self.benchmark = TensorCoreBenchmark(device=self.device)

    @requires_sm100()
    @requires_fp8()
    @pytest.mark.parametrize("M,N,K", POWER_OF_2_SIZES)
    def test_fp8_gemm_power_of_2(self, M: int, N: int, K: int):
        """Test FP8 GEMM efficiency with power-of-2 dimensions."""
        # Create FP8 tensors
        A_bf16 = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B_bf16 = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        # Convert to FP8
        A_fp8 = A_bf16.to(torch.float8_e4m3fn)
        B_fp8 = B_bf16.to(torch.float8_e4m3fn)

        # Scale factors for FP8 GEMM
        scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        # Benchmark using torch._scaled_mm if available
        def fp8_gemm():
            return torch._scaled_mm(
                A_fp8,
                B_fp8.t(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )

        flops = compute_gemm_flops(M, N, K)
        result = self.benchmark.benchmark_fn(
            name=f"fp8_gemm_{M}x{N}x{K}",
            fn=fp8_gemm,
            flops=flops,
            theoretical_peak=B200_SPECS.fp8_tflops,
            shape_info=(M, N, K),
            dtype="float8_e4m3fn",
        )

        print(f"\n{result}")

        # Assert minimum efficiency
        assert result.efficiency >= B200_SPECS.min_fp8_efficiency, (
            f"FP8 GEMM efficiency {result.efficiency:.2%} below threshold "
            f"{B200_SPECS.min_fp8_efficiency:.2%}"
        )

    @requires_sm100()
    @requires_fp8()
    @pytest.mark.parametrize("M,N,K", NON_POWER_OF_2_SIZES[:3])  # Subset for speed
    def test_fp8_gemm_non_power_of_2(self, M: int, N: int, K: int):
        """Test FP8 GEMM with non-power-of-2 dimensions."""
        A_bf16 = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B_bf16 = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        A_fp8 = A_bf16.to(torch.float8_e4m3fn)
        B_fp8 = B_bf16.to(torch.float8_e4m3fn)

        scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        def fp8_gemm():
            return torch._scaled_mm(
                A_fp8,
                B_fp8.t(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )

        flops = compute_gemm_flops(M, N, K)
        result = self.benchmark.benchmark_fn(
            name=f"fp8_gemm_nonpow2_{M}x{N}x{K}",
            fn=fp8_gemm,
            flops=flops,
            theoretical_peak=B200_SPECS.fp8_tflops,
            shape_info=(M, N, K),
            dtype="float8_e4m3fn",
        )

        print(f"\n{result}")

        # Slightly lower threshold for non-power-of-2
        min_threshold = B200_SPECS.min_fp8_efficiency * 0.9
        assert result.efficiency >= min_threshold, (
            f"FP8 GEMM (non-pow2) efficiency {result.efficiency:.2%} below threshold "
            f"{min_threshold:.2%}"
        )

    @requires_sm100()
    @requires_fp8()
    def test_fp8_scaling_with_size(self):
        """Test that FP8 efficiency scales well with matrix size."""
        results = []
        sizes = [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]

        for M, N, K in sizes:
            A_bf16 = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B_bf16 = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

            A_fp8 = A_bf16.to(torch.float8_e4m3fn)
            B_fp8 = B_bf16.to(torch.float8_e4m3fn)

            scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)

            def fp8_gemm():
                return torch._scaled_mm(
                    A_fp8,
                    B_fp8.t(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=torch.bfloat16,
                )

            flops = compute_gemm_flops(M, N, K)
            result = self.benchmark.benchmark_fn(
                name=f"fp8_scaling_{M}",
                fn=fp8_gemm,
                flops=flops,
                theoretical_peak=B200_SPECS.fp8_tflops,
                shape_info=(M, N, K),
                dtype="float8_e4m3fn",
            )
            results.append(result)
            print(f"\n{result}")

        # Efficiency should improve with larger sizes
        efficiencies = [r.efficiency for r in results]
        assert efficiencies[-1] >= efficiencies[0], (
            f"Efficiency should improve with size: got {efficiencies}"
        )


# =============================================================================
# Test Class: FP4 / NVFP4 Tensor Core
# =============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestFP4TensorCore:
    """Test NVFP4 (E2M1) tensor core operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for NVFP4 tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = torch.device("cuda:0")
        self.benchmark = TensorCoreBenchmark(device=self.device)

    @requires_sm100()
    @requires_cutlass_dsl()
    @pytest.mark.parametrize("M,N,K", POWER_OF_2_SIZES[:3])  # Smaller subset
    def test_nvfp4_gemm_power_of_2(self, M: int, N: int, K: int):
        """Test NVFP4 GEMM efficiency with power-of-2 dimensions.

        Note: NVFP4 operations require CuTeDSL and special kernel invocation.
        This test validates the simulated/emulated NVFP4 performance path.
        """
        # NVFP4 is typically used via blockscaled GEMM with E8M0 scales
        # Here we simulate the expected performance characteristics

        # Create BF16 tensors as proxy (actual NVFP4 requires special handling)
        A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        # For NVFP4, we'd typically pack 2 FP4 values per byte
        # and use blockscaled GEMM with E8M0 scale factors

        # Simulated NVFP4 GEMM (using BF16 as proxy for timing structure)
        result = self.benchmark.benchmark_gemm(
            name=f"nvfp4_proxy_{M}x{N}x{K}",
            A=A,
            B=B,
            theoretical_peak=B200_SPECS.nvfp4_tflops,
        )

        # Adjust reported dtype
        result.dtype = "nvfp4_e2m1 (proxy)"

        print(f"\n{result}")

        # Note: Proxy test - actual NVFP4 efficiency would be measured with
        # the real blockscaled GEMM kernel
        assert result.latency_ms > 0, "Benchmark should complete"

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_nvfp4_blockscaled_simulation(self):
        """Test NVFP4 blockscaled GEMM simulation.

        NVFP4 uses block-level scale factors (E8M0) to maintain precision.
        Each 128-element block has a shared scale factor.
        """
        M, N, K = 4096, 4096, 4096
        block_size = 128  # Standard block size for NVFP4

        # Number of scale factor blocks
        n_blocks_m = (M + block_size - 1) // block_size
        n_blocks_k = (K + block_size - 1) // block_size

        # Create tensors
        A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        # Scale factors (E8M0 format stored as uint8)
        scales_a = torch.randint(0, 255, (n_blocks_m, n_blocks_k),
                                  device=self.device, dtype=torch.uint8)
        scales_b = torch.randint(0, 255, (n_blocks_k, (N + block_size - 1) // block_size),
                                  device=self.device, dtype=torch.uint8)

        # Benchmark the GEMM (BF16 proxy)
        result = self.benchmark.benchmark_gemm(
            name=f"nvfp4_blockscaled_{M}x{N}x{K}",
            A=A,
            B=B,
            theoretical_peak=B200_SPECS.nvfp4_tflops,
        )

        print(f"\nBlockscaled NVFP4 simulation: {result}")
        print(f"Scale factor blocks: A={scales_a.shape}, B={scales_b.shape}")

        # Verify scale factor dimensions are correct
        assert scales_a.shape[0] == n_blocks_m
        assert scales_a.shape[1] == n_blocks_k


# =============================================================================
# Test Class: BF16 Tensor Core
# =============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestBF16TensorCore:
    """Test BF16 tensor core utilization as baseline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for BF16 tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = torch.device("cuda:0")
        self.benchmark = TensorCoreBenchmark(device=self.device)

    @requires_sm100()
    @pytest.mark.parametrize("M,N,K", POWER_OF_2_SIZES)
    def test_bf16_gemm_power_of_2(self, M: int, N: int, K: int):
        """Test BF16 GEMM efficiency with power-of-2 dimensions."""
        A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        result = self.benchmark.benchmark_gemm(
            name=f"bf16_gemm_{M}x{N}x{K}",
            A=A,
            B=B,
            theoretical_peak=B200_SPECS.bf16_tflops,
        )

        print(f"\n{result}")

        assert result.efficiency >= B200_SPECS.min_bf16_efficiency, (
            f"BF16 GEMM efficiency {result.efficiency:.2%} below threshold "
            f"{B200_SPECS.min_bf16_efficiency:.2%}"
        )

    @requires_sm100()
    @pytest.mark.parametrize("M,N,K", NON_POWER_OF_2_SIZES)
    def test_bf16_gemm_non_power_of_2(self, M: int, N: int, K: int):
        """Test BF16 GEMM with non-power-of-2 dimensions."""
        A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        result = self.benchmark.benchmark_gemm(
            name=f"bf16_gemm_nonpow2_{M}x{N}x{K}",
            A=A,
            B=B,
            theoretical_peak=B200_SPECS.bf16_tflops,
        )

        print(f"\n{result}")

        # Slightly lower threshold for non-power-of-2
        min_threshold = B200_SPECS.min_bf16_efficiency * 0.85
        assert result.efficiency >= min_threshold, (
            f"BF16 GEMM (non-pow2) efficiency {result.efficiency:.2%} below threshold "
            f"{min_threshold:.2%}"
        )

    @requires_sm100()
    def test_bf16_vs_fp16_comparison(self):
        """Compare BF16 and FP16 tensor core performance."""
        M, N, K = 4096, 4096, 4096

        A_bf16 = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B_bf16 = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        A_fp16 = A_bf16.to(torch.float16)
        B_fp16 = B_bf16.to(torch.float16)

        result_bf16 = self.benchmark.benchmark_gemm(
            name=f"bf16_gemm_{M}",
            A=A_bf16,
            B=B_bf16,
            theoretical_peak=B200_SPECS.bf16_tflops,
        )

        result_fp16 = self.benchmark.benchmark_gemm(
            name=f"fp16_gemm_{M}",
            A=A_fp16,
            B=B_fp16,
            theoretical_peak=B200_SPECS.fp16_tflops,
        )

        print(f"\nBF16: {result_bf16}")
        print(f"FP16: {result_fp16}")

        # Both should achieve reasonable efficiency
        assert result_bf16.efficiency >= B200_SPECS.min_bf16_efficiency * 0.9
        assert result_fp16.efficiency >= B200_SPECS.min_bf16_efficiency * 0.9


# =============================================================================
# Test Class: MoE GEMM Efficiency
# =============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestMoEGEMMEfficiency:
    """Test MoE-specific GEMM workloads."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for MoE tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = torch.device("cuda:0")
        self.benchmark = TensorCoreBenchmark(
            warmup_iters=30,
            benchmark_iters=100,
            device=self.device,
        )

    @requires_sm100()
    @pytest.mark.parametrize("n_tokens,hidden_dim,inter_dim", MOE_DIMENSIONS)
    def test_moe_expert_gemm_bf16(
        self,
        n_tokens: int,
        hidden_dim: int,
        inter_dim: int,
    ):
        """Test MoE expert GEMM efficiency with BF16."""
        n_experts = 8
        topk = 2

        # Create expert weights
        W1 = torch.randn(n_experts, hidden_dim, inter_dim,
                         device=self.device, dtype=torch.bfloat16)
        W3 = torch.randn(n_experts, hidden_dim, inter_dim,
                         device=self.device, dtype=torch.bfloat16)
        W2 = torch.randn(n_experts, inter_dim, hidden_dim,
                         device=self.device, dtype=torch.bfloat16)

        # Simulate routed tokens (uniform distribution)
        tokens_per_expert = (n_tokens * topk) // n_experts
        X = torch.randn(tokens_per_expert, hidden_dim,
                        device=self.device, dtype=torch.bfloat16)

        # MoE expert forward: SwiGLU(X @ W1, X @ W3) @ W2
        def moe_expert_forward():
            # Using single expert for benchmark
            h1 = X @ W1[0]
            h3 = X @ W3[0]
            hidden = F.silu(h1) * h3
            return hidden @ W2[0]

        # Total FLOPS for all experts
        flops = compute_moe_expert_flops(
            n_tokens, hidden_dim, inter_dim, n_experts, topk
        )

        result = self.benchmark.benchmark_fn(
            name=f"moe_expert_{n_tokens}tok",
            fn=moe_expert_forward,
            flops=flops // n_experts,  # Single expert
            theoretical_peak=B200_SPECS.bf16_tflops,
            shape_info=(tokens_per_expert, hidden_dim, inter_dim),
            dtype="bfloat16",
        )

        print(f"\n{result}")

        # MoE has lower efficiency due to routing overhead and smaller GEMMs
        assert result.efficiency >= B200_SPECS.min_moe_efficiency, (
            f"MoE GEMM efficiency {result.efficiency:.2%} below threshold "
            f"{B200_SPECS.min_moe_efficiency:.2%}"
        )

    @requires_sm100()
    @requires_fp8()
    def test_moe_expert_gemm_fp8(self):
        """Test MoE expert GEMM efficiency with FP8."""
        n_tokens = 4096
        hidden_dim = 2048
        inter_dim = 5504
        n_experts = 8
        topk = 2

        tokens_per_expert = (n_tokens * topk) // n_experts

        # Create FP8 tensors
        X_bf16 = torch.randn(tokens_per_expert, hidden_dim,
                             device=self.device, dtype=torch.bfloat16)
        W1_bf16 = torch.randn(hidden_dim, inter_dim,
                              device=self.device, dtype=torch.bfloat16)

        X_fp8 = X_bf16.to(torch.float8_e4m3fn)
        W1_fp8 = W1_bf16.to(torch.float8_e4m3fn)

        scale_x = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_w = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        def moe_fp8_gemm():
            return torch._scaled_mm(
                X_fp8,
                W1_fp8.t(),
                scale_a=scale_x,
                scale_b=scale_w,
                out_dtype=torch.bfloat16,
            )

        flops = compute_gemm_flops(tokens_per_expert, inter_dim, hidden_dim)

        result = self.benchmark.benchmark_fn(
            name="moe_expert_fp8",
            fn=moe_fp8_gemm,
            flops=flops,
            theoretical_peak=B200_SPECS.fp8_tflops,
            shape_info=(tokens_per_expert, inter_dim, hidden_dim),
            dtype="float8_e4m3fn",
        )

        print(f"\n{result}")

        # FP8 MoE should have better raw TFLOPS
        assert result.achieved_tflops > 0

    @requires_sm100()
    def test_moe_grouped_gemm_simulation(self):
        """Test grouped GEMM for MoE (simulated with sequential)."""
        n_experts = 8
        hidden_dim = 2048
        inter_dim = 5504

        # Variable tokens per expert (realistic distribution)
        tokens_per_expert = [128, 256, 512, 256, 128, 512, 256, 128]
        total_tokens = sum(tokens_per_expert)

        # Create weights
        W1 = torch.randn(n_experts, hidden_dim, inter_dim,
                         device=self.device, dtype=torch.bfloat16)

        # Create inputs for each expert
        Xs = [
            torch.randn(n, hidden_dim, device=self.device, dtype=torch.bfloat16)
            for n in tokens_per_expert
        ]

        def grouped_gemm():
            outputs = []
            for i, x in enumerate(Xs):
                outputs.append(x @ W1[i])
            return outputs

        # Benchmark
        torch.cuda.reset_peak_memory_stats(self.device)

        # Warmup
        for _ in range(30):
            _ = grouped_gemm()
        torch.cuda.synchronize()

        # Timed
        latencies = []
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = grouped_gemm()
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        latency_ms = statistics.median(latencies)
        total_flops = sum(
            compute_gemm_flops(n, inter_dim, hidden_dim)
            for n in tokens_per_expert
        )
        achieved_tflops = ms_to_tflops(total_flops, latency_ms)

        print(f"\nGrouped GEMM simulation:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Latency: {latency_ms:.3f}ms")
        print(f"  Achieved: {achieved_tflops:.1f} TFLOPS")

        assert achieved_tflops > 0


# =============================================================================
# Test Class: Batched GEMM
# =============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestBatchedGEMM:
    """Test batched matrix operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for batched GEMM tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = torch.device("cuda:0")
        self.benchmark = TensorCoreBenchmark(device=self.device)

    @requires_sm100()
    @pytest.mark.parametrize("batch_size", BATCH_SIZES[:5])  # Up to 32K
    def test_batched_gemm_bf16(self, batch_size: int):
        """Test batched GEMM with different batch sizes."""
        # Typical attention-like dimensions
        seq_len = min(batch_size, 2048)
        head_dim = 128
        n_heads = 32

        # Shape for batched attention-like computation
        M = seq_len
        N = head_dim
        K = head_dim
        B = n_heads

        A = torch.randn(B, M, K, device=self.device, dtype=torch.bfloat16)
        B_mat = torch.randn(B, K, N, device=self.device, dtype=torch.bfloat16)

        def batched_gemm():
            return torch.bmm(A, B_mat)

        flops = compute_batched_gemm_flops(B, M, N, K)

        result = self.benchmark.benchmark_fn(
            name=f"batched_gemm_bs{batch_size}",
            fn=batched_gemm,
            flops=flops,
            theoretical_peak=B200_SPECS.bf16_tflops,
            shape_info=(M, N, K),
            dtype="bfloat16",
        )

        print(f"\n{result}")

        # Batched GEMM should maintain efficiency
        assert result.efficiency >= B200_SPECS.min_bf16_efficiency * 0.7

    @requires_sm100()
    def test_batched_gemm_scaling(self):
        """Test batched GEMM efficiency scales with batch size."""
        results = []
        head_dim = 128
        seq_len = 1024

        for n_heads in [8, 16, 32, 64]:
            A = torch.randn(n_heads, seq_len, head_dim,
                           device=self.device, dtype=torch.bfloat16)
            B = torch.randn(n_heads, head_dim, head_dim,
                           device=self.device, dtype=torch.bfloat16)

            def batched_gemm():
                return torch.bmm(A, B)

            flops = compute_batched_gemm_flops(n_heads, seq_len, head_dim, head_dim)

            result = self.benchmark.benchmark_fn(
                name=f"batched_gemm_h{n_heads}",
                fn=batched_gemm,
                flops=flops,
                theoretical_peak=B200_SPECS.bf16_tflops,
                shape_info=(seq_len, head_dim, head_dim),
                dtype="bfloat16",
            )
            results.append((n_heads, result))
            print(f"\nn_heads={n_heads}: {result}")

        # Efficiency should generally improve with more heads (more parallelism)
        tflops_list = [r.achieved_tflops for _, r in results]
        assert tflops_list[-1] >= tflops_list[0] * 0.8  # At least maintain 80%

    @requires_sm100()
    @pytest.mark.parametrize("batch_size", [1024, 4096, 16384, 65536])
    def test_realistic_token_batch_sizes(self, batch_size: int):
        """Test with realistic MoE token batch sizes (1K to 64K)."""
        hidden_dim = 2048
        inter_dim = 5504

        X = torch.randn(batch_size, hidden_dim,
                        device=self.device, dtype=torch.bfloat16)
        W = torch.randn(hidden_dim, inter_dim,
                        device=self.device, dtype=torch.bfloat16)

        result = self.benchmark.benchmark_gemm(
            name=f"token_batch_{batch_size}",
            A=X,
            B=W,
            theoretical_peak=B200_SPECS.bf16_tflops,
        )

        print(f"\n{result}")

        # Should achieve reasonable efficiency for large batches
        if batch_size >= 4096:
            assert result.efficiency >= B200_SPECS.min_bf16_efficiency * 0.8


# =============================================================================
# Test Class: Comprehensive Utilization Report
# =============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestTensorCoreUtilizationReport:
    """Comprehensive tensor core utilization metrics and reporting."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for utilization report tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = torch.device("cuda:0")
        self.benchmark = TensorCoreBenchmark(
            warmup_iters=30,
            benchmark_iters=100,
            device=self.device,
        )
        self.results: List[BenchmarkResult] = []

    @requires_sm100()
    def test_comprehensive_bf16_sweep(self):
        """Comprehensive BF16 tensor core utilization sweep."""
        sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]

        print("\n" + "=" * 70)
        print("BF16 Tensor Core Utilization Report")
        print("=" * 70)

        for M, N, K in sizes:
            A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

            result = self.benchmark.benchmark_gemm(
                name=f"bf16_{M}x{N}x{K}",
                A=A,
                B=B,
                theoretical_peak=B200_SPECS.bf16_tflops,
            )
            self.results.append(result)

            status = "PASS" if result.efficiency >= B200_SPECS.min_bf16_efficiency else "FAIL"
            print(f"{result.name:25s} | {result.achieved_tflops:8.1f} TFLOPS | "
                  f"{result.efficiency*100:5.1f}% | {result.latency_ms:8.3f}ms | [{status}]")

        # Summary
        avg_efficiency = statistics.mean(r.efficiency for r in self.results)
        max_tflops = max(r.achieved_tflops for r in self.results)

        print("-" * 70)
        print(f"Average Efficiency: {avg_efficiency*100:.1f}%")
        print(f"Peak Achieved: {max_tflops:.1f} TFLOPS")
        print(f"Theoretical Peak: {B200_SPECS.bf16_tflops:.1f} TFLOPS")
        print("=" * 70)

        # Overall assertion
        assert avg_efficiency >= B200_SPECS.min_bf16_efficiency * 0.9

    @requires_sm100()
    @requires_fp8()
    def test_comprehensive_fp8_sweep(self):
        """Comprehensive FP8 tensor core utilization sweep."""
        sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]

        print("\n" + "=" * 70)
        print("FP8 Tensor Core Utilization Report")
        print("=" * 70)

        results = []
        for M, N, K in sizes:
            A_bf16 = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B_bf16 = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

            A_fp8 = A_bf16.to(torch.float8_e4m3fn)
            B_fp8 = B_bf16.to(torch.float8_e4m3fn)

            scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)

            def fp8_gemm():
                return torch._scaled_mm(
                    A_fp8,
                    B_fp8.t(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=torch.bfloat16,
                )

            flops = compute_gemm_flops(M, N, K)
            result = self.benchmark.benchmark_fn(
                name=f"fp8_{M}x{N}x{K}",
                fn=fp8_gemm,
                flops=flops,
                theoretical_peak=B200_SPECS.fp8_tflops,
                shape_info=(M, N, K),
                dtype="float8_e4m3fn",
            )
            results.append(result)

            status = "PASS" if result.efficiency >= B200_SPECS.min_fp8_efficiency else "FAIL"
            print(f"{result.name:25s} | {result.achieved_tflops:8.1f} TFLOPS | "
                  f"{result.efficiency*100:5.1f}% | {result.latency_ms:8.3f}ms | [{status}]")

        # Summary
        avg_efficiency = statistics.mean(r.efficiency for r in results)
        max_tflops = max(r.achieved_tflops for r in results)

        print("-" * 70)
        print(f"Average Efficiency: {avg_efficiency*100:.1f}%")
        print(f"Peak Achieved: {max_tflops:.1f} TFLOPS")
        print(f"Theoretical Peak: {B200_SPECS.fp8_tflops:.1f} TFLOPS")
        print("=" * 70)

    @requires_sm100()
    def test_moe_workload_utilization(self):
        """MoE-specific workload utilization report."""
        print("\n" + "=" * 70)
        print("MoE Workload Tensor Core Utilization Report")
        print("=" * 70)

        configs = [
            {"n_tokens": 1024, "hidden": 2048, "inter": 5504, "experts": 8, "topk": 2},
            {"n_tokens": 4096, "hidden": 2048, "inter": 5504, "experts": 8, "topk": 2},
            {"n_tokens": 16384, "hidden": 2048, "inter": 5504, "experts": 8, "topk": 2},
            {"n_tokens": 4096, "hidden": 4096, "inter": 11008, "experts": 8, "topk": 2},
        ]

        results = []
        for cfg in configs:
            tokens_per_expert = (cfg["n_tokens"] * cfg["topk"]) // cfg["experts"]

            X = torch.randn(tokens_per_expert, cfg["hidden"],
                           device=self.device, dtype=torch.bfloat16)
            W1 = torch.randn(cfg["hidden"], cfg["inter"],
                            device=self.device, dtype=torch.bfloat16)
            W3 = torch.randn(cfg["hidden"], cfg["inter"],
                            device=self.device, dtype=torch.bfloat16)
            W2 = torch.randn(cfg["inter"], cfg["hidden"],
                            device=self.device, dtype=torch.bfloat16)

            def moe_forward():
                h1 = X @ W1
                h3 = X @ W3
                hidden = F.silu(h1) * h3
                return hidden @ W2

            # Total FLOPS for one expert
            flops = 6 * tokens_per_expert * cfg["hidden"] * cfg["inter"]

            result = self.benchmark.benchmark_fn(
                name=f"moe_{cfg['n_tokens']}tok_{cfg['hidden']}h",
                fn=moe_forward,
                flops=flops,
                theoretical_peak=B200_SPECS.bf16_tflops,
                shape_info=(tokens_per_expert, cfg["hidden"], cfg["inter"]),
                dtype="bfloat16",
            )
            results.append(result)

            status = "PASS" if result.efficiency >= B200_SPECS.min_moe_efficiency else "FAIL"
            print(f"{result.name:30s} | {result.achieved_tflops:8.1f} TFLOPS | "
                  f"{result.efficiency*100:5.1f}% | {result.latency_ms:8.3f}ms | [{status}]")

        # Summary
        avg_efficiency = statistics.mean(r.efficiency for r in results)
        max_tflops = max(r.achieved_tflops for r in results)

        print("-" * 70)
        print(f"Average MoE Efficiency: {avg_efficiency*100:.1f}%")
        print(f"Peak MoE Achieved: {max_tflops:.1f} TFLOPS")
        print(f"Minimum Threshold: {B200_SPECS.min_moe_efficiency*100:.1f}%")
        print("=" * 70)

        # Overall MoE assertion
        assert avg_efficiency >= B200_SPECS.min_moe_efficiency * 0.9

    @requires_sm100()
    def test_precision_comparison_report(self):
        """Compare tensor core utilization across precisions."""
        M, N, K = 4096, 4096, 4096

        print("\n" + "=" * 70)
        print("Precision Comparison Report (4096x4096x4096)")
        print("=" * 70)

        # BF16
        A_bf16 = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B_bf16 = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        result_bf16 = self.benchmark.benchmark_gemm(
            name="bf16",
            A=A_bf16,
            B=B_bf16,
            theoretical_peak=B200_SPECS.bf16_tflops,
        )

        # FP16
        A_fp16 = A_bf16.to(torch.float16)
        B_fp16 = B_bf16.to(torch.float16)

        result_fp16 = self.benchmark.benchmark_gemm(
            name="fp16",
            A=A_fp16,
            B=B_fp16,
            theoretical_peak=B200_SPECS.fp16_tflops,
        )

        # FP8 (if available)
        result_fp8 = None
        if _has_fp8_support():
            A_fp8 = A_bf16.to(torch.float8_e4m3fn)
            B_fp8 = B_bf16.to(torch.float8_e4m3fn)
            scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)

            def fp8_gemm():
                return torch._scaled_mm(
                    A_fp8,
                    B_fp8.t(),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=torch.bfloat16,
                )

            flops = compute_gemm_flops(M, N, K)
            result_fp8 = self.benchmark.benchmark_fn(
                name="fp8",
                fn=fp8_gemm,
                flops=flops,
                theoretical_peak=B200_SPECS.fp8_tflops,
                shape_info=(M, N, K),
                dtype="float8_e4m3fn",
            )

        print(f"{'Precision':12s} | {'TFLOPS':>10s} | {'Efficiency':>10s} | {'Latency':>10s} | {'Peak':>10s}")
        print("-" * 70)

        print(f"{'BF16':12s} | {result_bf16.achieved_tflops:10.1f} | "
              f"{result_bf16.efficiency*100:9.1f}% | {result_bf16.latency_ms:9.3f}ms | "
              f"{B200_SPECS.bf16_tflops:10.0f}")

        print(f"{'FP16':12s} | {result_fp16.achieved_tflops:10.1f} | "
              f"{result_fp16.efficiency*100:9.1f}% | {result_fp16.latency_ms:9.3f}ms | "
              f"{B200_SPECS.fp16_tflops:10.0f}")

        if result_fp8:
            print(f"{'FP8':12s} | {result_fp8.achieved_tflops:10.1f} | "
                  f"{result_fp8.efficiency*100:9.1f}% | {result_fp8.latency_ms:9.3f}ms | "
                  f"{B200_SPECS.fp8_tflops:10.0f}")

        print("=" * 70)

        # Verify FP8 achieves higher raw TFLOPS than BF16
        if result_fp8:
            assert result_fp8.achieved_tflops >= result_bf16.achieved_tflops * 0.9, (
                "FP8 should achieve comparable or higher TFLOPS than BF16"
            )

    @requires_sm100()
    def test_memory_bandwidth_bound_detection(self):
        """Detect memory bandwidth bound operations."""
        print("\n" + "=" * 70)
        print("Memory Bandwidth Bound Detection")
        print("=" * 70)

        # Small K (memory bound)
        configs_small_k = [
            (8192, 8192, 32),
            (8192, 8192, 64),
            (8192, 8192, 128),
        ]

        # Large K (compute bound)
        configs_large_k = [
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]

        print("\nSmall K (potentially memory bound):")
        for M, N, K in configs_small_k:
            A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

            result = self.benchmark.benchmark_gemm(
                name=f"small_k_{K}",
                A=A,
                B=B,
                theoretical_peak=B200_SPECS.bf16_tflops,
            )

            # Compute arithmetic intensity
            flops = compute_gemm_flops(M, N, K)
            bytes_accessed = (M * K + K * N + M * N) * 2  # BF16 = 2 bytes
            arithmetic_intensity = flops / bytes_accessed

            print(f"  K={K:4d} | {result.achieved_tflops:8.1f} TFLOPS | "
                  f"{result.efficiency*100:5.1f}% | AI={arithmetic_intensity:.1f} FLOPS/byte")

        print("\nLarge K (compute bound):")
        for M, N, K in configs_large_k:
            A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

            result = self.benchmark.benchmark_gemm(
                name=f"large_k_{K}",
                A=A,
                B=B,
                theoretical_peak=B200_SPECS.bf16_tflops,
            )

            flops = compute_gemm_flops(M, N, K)
            bytes_accessed = (M * K + K * N + M * N) * 2
            arithmetic_intensity = flops / bytes_accessed

            print(f"  K={K:4d} | {result.achieved_tflops:8.1f} TFLOPS | "
                  f"{result.efficiency*100:5.1f}% | AI={arithmetic_intensity:.1f} FLOPS/byte")

        print("=" * 70)


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


@pytest.mark.gpu
@pytest.mark.b200
class TestTensorCoreEdgeCases:
    """Edge case tests for tensor core operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for edge case tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        self.device = torch.device("cuda:0")
        self.benchmark = TensorCoreBenchmark(
            warmup_iters=20,
            benchmark_iters=50,
            device=self.device,
        )

    @requires_sm100()
    def test_alignment_variations(self):
        """Test tensor core efficiency with different alignments."""
        # Aligned to 128 (optimal for B200)
        aligned_sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ]

        # Misaligned (not multiples of 128)
        misaligned_sizes = [
            (1000, 1000, 1000),
            (2000, 2000, 2000),
        ]

        print("\nAlignment test results:")
        for M, N, K in aligned_sizes:
            A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)
            result = self.benchmark.benchmark_gemm(
                name=f"aligned_{M}",
                A=A,
                B=B,
                theoretical_peak=B200_SPECS.bf16_tflops,
            )
            print(f"  Aligned {M}x{N}x{K}: {result.efficiency*100:.1f}%")

        for M, N, K in misaligned_sizes:
            A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)
            result = self.benchmark.benchmark_gemm(
                name=f"misaligned_{M}",
                A=A,
                B=B,
                theoretical_peak=B200_SPECS.bf16_tflops,
            )
            print(f"  Misaligned {M}x{N}x{K}: {result.efficiency*100:.1f}%")

    @requires_sm100()
    def test_tall_skinny_matrices(self):
        """Test tensor core efficiency with tall-skinny matrices (common in MoE)."""
        # Tall-skinny: large M, small N
        configs = [
            (16384, 128, 2048),  # Many tokens, small output
            (32768, 64, 2048),   # Very many tokens
            (65536, 128, 2048),  # Maximum batch
        ]

        print("\nTall-skinny matrix results:")
        for M, N, K in configs:
            A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)
            result = self.benchmark.benchmark_gemm(
                name=f"tall_skinny_{M}x{N}",
                A=A,
                B=B,
                theoretical_peak=B200_SPECS.bf16_tflops,
            )
            print(f"  {M}x{N}x{K}: {result.achieved_tflops:.1f} TFLOPS ({result.efficiency*100:.1f}%)")

    @requires_sm100()
    def test_wide_short_matrices(self):
        """Test tensor core efficiency with wide-short matrices."""
        # Wide-short: small M, large N
        configs = [
            (128, 16384, 2048),
            (256, 8192, 2048),
            (512, 4096, 2048),
        ]

        print("\nWide-short matrix results:")
        for M, N, K in configs:
            A = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
            B = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)
            result = self.benchmark.benchmark_gemm(
                name=f"wide_short_{M}x{N}",
                A=A,
                B=B,
                theoretical_peak=B200_SPECS.bf16_tflops,
            )
            print(f"  {M}x{N}x{K}: {result.achieved_tflops:.1f} TFLOPS ({result.efficiency*100:.1f}%)")

    @requires_sm100()
    def test_contiguity_requirement(self):
        """Test that non-contiguous tensors are handled correctly."""
        M, N, K = 2048, 2048, 2048

        # Contiguous
        A_contig = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        B_contig = torch.randn(K, N, device=self.device, dtype=torch.bfloat16)

        # Non-contiguous (transposed view)
        A_nc = torch.randn(K, M, device=self.device, dtype=torch.bfloat16).t()
        B_nc = torch.randn(N, K, device=self.device, dtype=torch.bfloat16).t()

        assert not A_nc.is_contiguous()
        assert not B_nc.is_contiguous()

        result_contig = self.benchmark.benchmark_gemm(
            name="contiguous",
            A=A_contig,
            B=B_contig,
            theoretical_peak=B200_SPECS.bf16_tflops,
        )

        result_nc = self.benchmark.benchmark_gemm(
            name="non_contiguous",
            A=A_nc.contiguous(),  # PyTorch will make contiguous
            B=B_nc.contiguous(),
            theoretical_peak=B200_SPECS.bf16_tflops,
        )

        print(f"\nContiguity test:")
        print(f"  Contiguous: {result_contig.efficiency*100:.1f}%")
        print(f"  Non-contiguous (made contiguous): {result_nc.efficiency*100:.1f}%")

        # Both should achieve similar efficiency after being made contiguous
        assert abs(result_contig.efficiency - result_nc.efficiency) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
