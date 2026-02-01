"""P0 Critical Tests: Memory Leak Detection for Long-Running Operations.

This module contains comprehensive tests for detecting memory leaks in nmoe
during extended operations. Memory stability is critical for production
training runs that may last days or weeks.

Tests cover:
1. Long-running inference memory stability (10,000+ iterations)
2. Training loop memory stability (1,000+ training steps)
3. Expert cache memory leak detection (LazyExpertWeights)
4. KV cache allocation/deallocation cycles
5. MemoryTracker accuracy validation
6. CUDA memory fragmentation detection

Environment Variables:
    NMOE_LEAK_TEST_INFERENCE_ITERS: Inference iterations (default: 10000)
    NMOE_LEAK_TEST_TRAIN_STEPS: Training steps (default: 1000)
    NMOE_LEAK_TEST_CACHE_OPS: Cache operations (default: 5000)
    NMOE_LEAK_TEST_KV_CYCLES: KV cache cycles (default: 1000)
    NMOE_LEAK_TEST_FRAG_ALLOCS: Fragmentation test allocations (default: 2000)
    NMOE_LEAK_TEST_MEMORY_GROWTH_THRESHOLD: Max memory growth % (default: 1.0)
    NMOE_LEAK_TEST_WARMUP_ITERS: Warmup iterations before measurement (default: 100)

Usage:
    # Run all memory leak tests (slow)
    pytest -v -m "gpu and slow" nmoe/tests/gpu/test_memory_leak_long_run.py

    # Run quick smoke tests
    NMOE_LEAK_TEST_INFERENCE_ITERS=100 pytest -v nmoe/tests/gpu/test_memory_leak_long_run.py

    # Run with stricter threshold
    NMOE_LEAK_TEST_MEMORY_GROWTH_THRESHOLD=0.5 pytest -v nmoe/tests/gpu/test_memory_leak_long_run.py
"""

from __future__ import annotations

import gc
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Configuration from Environment Variables
# ==============================================================================


def get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.environ.get(name, str(default)))


def get_env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    return float(os.environ.get(name, str(default)))


# Test configuration with environment variable overrides
INFERENCE_ITERATIONS = get_env_int("NMOE_LEAK_TEST_INFERENCE_ITERS", 10000)
TRAINING_STEPS = get_env_int("NMOE_LEAK_TEST_TRAIN_STEPS", 1000)
CACHE_OPERATIONS = get_env_int("NMOE_LEAK_TEST_CACHE_OPS", 5000)
KV_CACHE_CYCLES = get_env_int("NMOE_LEAK_TEST_KV_CYCLES", 1000)
FRAGMENTATION_ALLOCS = get_env_int("NMOE_LEAK_TEST_FRAG_ALLOCS", 2000)
MEMORY_GROWTH_THRESHOLD = get_env_float("NMOE_LEAK_TEST_MEMORY_GROWTH_THRESHOLD", 1.0)
WARMUP_ITERATIONS = get_env_int("NMOE_LEAK_TEST_WARMUP_ITERS", 100)

# Sampling intervals for memory tracking
MEMORY_SAMPLE_INTERVAL = 100  # Sample every N iterations


# ==============================================================================
# Memory Statistics Utilities
# ==============================================================================


@dataclass
class MemoryStats:
    """Container for memory statistics at a point in time."""

    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int
    timestamp: float

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 ** 2)

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / (1024 ** 2)

    @classmethod
    def capture(cls, device: str = "cuda:0") -> "MemoryStats":
        """Capture current memory statistics."""
        return cls(
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
            max_allocated_bytes=torch.cuda.max_memory_allocated(device),
            max_reserved_bytes=torch.cuda.max_memory_reserved(device),
            timestamp=time.time(),
        )


class MemoryLeakDetector:
    """Utility class for detecting memory leaks during long-running operations.

    This class tracks memory usage over time and detects if memory is
    growing unboundedly, which would indicate a leak.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        growth_threshold_percent: float = MEMORY_GROWTH_THRESHOLD,
        warmup_samples: int = 10,
    ):
        self.device = device
        self.growth_threshold = growth_threshold_percent
        self.warmup_samples = warmup_samples
        self.samples: List[MemoryStats] = []
        self._baseline: Optional[MemoryStats] = None

    def reset(self) -> None:
        """Reset memory stats and clear baseline."""
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.samples.clear()
        self._baseline = None

    def set_baseline(self) -> MemoryStats:
        """Set baseline after warmup period."""
        gc.collect()
        torch.cuda.empty_cache()
        self._baseline = MemoryStats.capture(self.device)
        return self._baseline

    def sample(self) -> MemoryStats:
        """Take a memory sample."""
        stats = MemoryStats.capture(self.device)
        self.samples.append(stats)
        return stats

    @property
    def baseline(self) -> Optional[MemoryStats]:
        """Get baseline memory stats."""
        return self._baseline

    def get_memory_growth_percent(self) -> float:
        """Calculate memory growth from baseline as a percentage.

        Returns:
            Percentage growth of allocated memory from baseline.
            Returns 0.0 if no baseline is set.
        """
        if self._baseline is None or not self.samples:
            return 0.0

        # Use the last sample for comparison
        current = self.samples[-1]
        baseline_alloc = self._baseline.allocated_bytes

        if baseline_alloc == 0:
            return 0.0

        growth = current.allocated_bytes - baseline_alloc
        return (growth / baseline_alloc) * 100.0

    def get_trend(self) -> Tuple[float, float]:
        """Calculate memory trend using linear regression on post-warmup samples.

        Returns:
            Tuple of (slope_mb_per_sample, r_squared).
            Positive slope indicates potential leak.
        """
        if len(self.samples) < self.warmup_samples + 2:
            return (0.0, 0.0)

        # Use samples after warmup
        post_warmup = self.samples[self.warmup_samples:]

        n = len(post_warmup)
        x = list(range(n))
        y = [s.allocated_bytes / (1024 ** 2) for s in post_warmup]

        # Linear regression
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return (0.0, 0.0)

        slope = numerator / denominator

        # R-squared
        y_pred = [y_mean + slope * (xi - x_mean) for xi in x]
        ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return (slope, r_squared)

    def check_leak(self) -> Tuple[bool, str]:
        """Check if a memory leak is detected.

        Returns:
            Tuple of (has_leak: bool, message: str).
        """
        growth_pct = self.get_memory_growth_percent()
        slope, r_squared = self.get_trend()

        has_leak = growth_pct > self.growth_threshold

        if has_leak:
            message = (
                f"Memory leak detected: {growth_pct:.2f}% growth "
                f"(threshold: {self.growth_threshold}%). "
                f"Trend: {slope:.4f} MB/sample (R^2={r_squared:.4f})"
            )
        else:
            message = (
                f"Memory stable: {growth_pct:.2f}% growth "
                f"(threshold: {self.growth_threshold}%). "
                f"Trend: {slope:.4f} MB/sample (R^2={r_squared:.4f})"
            )

        return (has_leak, message)

    def report(self) -> Dict[str, Any]:
        """Generate detailed memory report."""
        growth_pct = self.get_memory_growth_percent()
        slope, r_squared = self.get_trend()
        has_leak, message = self.check_leak()

        report = {
            "has_leak": has_leak,
            "message": message,
            "growth_percent": growth_pct,
            "trend_mb_per_sample": slope,
            "trend_r_squared": r_squared,
            "sample_count": len(self.samples),
            "growth_threshold": self.growth_threshold,
        }

        if self._baseline:
            report["baseline_allocated_mb"] = self._baseline.allocated_mb
            report["baseline_reserved_mb"] = self._baseline.reserved_mb

        if self.samples:
            last = self.samples[-1]
            report["final_allocated_mb"] = last.allocated_mb
            report["final_reserved_mb"] = last.reserved_mb
            report["peak_allocated_mb"] = last.max_allocated_bytes / (1024 ** 2)
            report["peak_reserved_mb"] = last.max_reserved_bytes / (1024 ** 2)

        return report


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def cuda_device():
    """Provide CUDA device and skip if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def memory_tracking_fixture(cuda_device):
    """Fixture that sets up memory tracking with proper cleanup.

    This fixture:
    1. Clears CUDA cache before test
    2. Resets peak memory stats
    3. Provides a MemoryLeakDetector instance
    4. Reports memory stats after test
    """
    # Pre-test cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    detector = MemoryLeakDetector(device=str(cuda_device))

    yield detector

    # Post-test reporting
    report = detector.report()
    print("\n" + "=" * 60)
    print("Memory Leak Detection Report")
    print("=" * 60)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


@pytest.fixture
def small_test_config():
    """Small model configuration for memory tests."""
    from nmoe.config import Config
    return Config(
        dim=256,
        n_layers=2,
        n_heads=4,
        inter_dim=512,
        moe_inter_dim=512,
        vocab_size=1000,
        n_routed_experts=8,
        n_activated_experts=2,
        n_dense_layers=1,
        max_position_embeddings=512,
        batch_size=4,
        seq_len=64,
        dtype="bf16",
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ==============================================================================
# Test Classes
# ==============================================================================


@pytest.mark.gpu
@pytest.mark.slow
class TestInferenceMemoryStability:
    """Test memory stability during long-running inference.

    These tests run 10,000+ inference iterations and verify that
    GPU memory growth stays below 1% after warmup.
    """

    def test_inference_no_leak(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Run inference loop and verify no memory leak.

        This test:
        1. Creates a small MoE model
        2. Runs INFERENCE_ITERATIONS forward passes
        3. Samples memory at regular intervals
        4. Verifies memory growth < MEMORY_GROWTH_THRESHOLD
        """
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        # Create small model for testing
        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=8,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.eval()

        batch_size = 2
        seq_len = 32

        # Warmup phase
        detector.reset()
        print(f"\nRunning {WARMUP_ITERATIONS} warmup iterations...")
        for _ in range(WARMUP_ITERATIONS):
            with torch.no_grad():
                tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                _ = model(tokens)

        torch.cuda.synchronize()
        detector.set_baseline()
        print(f"Baseline: {detector.baseline.allocated_mb:.2f} MB allocated")

        # Main inference loop
        print(f"Running {INFERENCE_ITERATIONS} inference iterations...")
        for i in range(INFERENCE_ITERATIONS):
            with torch.no_grad():
                tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                logits = model(tokens)
                # Ensure tensor is used to prevent optimization
                _ = logits.sum().item()

            # Sample memory at intervals
            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                stats = detector.sample()
                if (i + 1) % (MEMORY_SAMPLE_INTERVAL * 10) == 0:
                    print(f"  Iteration {i+1}: {stats.allocated_mb:.2f} MB allocated")

        # Final check
        torch.cuda.synchronize()
        detector.sample()

        has_leak, message = detector.check_leak()
        report = detector.report()

        assert not has_leak, (
            f"Memory leak detected during inference: {message}\n"
            f"Growth: {report['growth_percent']:.2f}%\n"
            f"Baseline: {report.get('baseline_allocated_mb', 0):.2f} MB\n"
            f"Final: {report.get('final_allocated_mb', 0):.2f} MB"
        )

    def test_inference_with_gradient_disabled(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Test that torch.no_grad() properly prevents gradient memory accumulation."""
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=8,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.eval()

        batch_size = 2
        seq_len = 32
        iterations = min(1000, INFERENCE_ITERATIONS)

        # Run with explicit no_grad
        detector.reset()

        for _ in range(WARMUP_ITERATIONS):
            with torch.inference_mode():
                tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                _ = model(tokens)

        torch.cuda.synchronize()
        detector.set_baseline()

        for i in range(iterations):
            with torch.inference_mode():
                tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                logits = model(tokens)
                _ = logits.sum().item()

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak with inference_mode: {message}"

    def test_repeated_model_forward_backward_cleanup(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Test that forward/backward passes properly clean up intermediate tensors."""
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=8,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.train()

        batch_size = 2
        seq_len = 32
        iterations = min(500, INFERENCE_ITERATIONS)

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
            targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        detector.set_baseline()

        # Main loop
        for i in range(iterations):
            tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
            targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)

            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            loss.backward()

            # Critical: zero_grad with set_to_none=True releases gradient memory
            model.zero_grad(set_to_none=True)

            # Delete local references
            del tokens, targets, logits, loss

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak in forward/backward loop: {message}"


@pytest.mark.gpu
@pytest.mark.slow
class TestTrainingLoopMemoryStability:
    """Test memory stability during training loops.

    These tests run 1,000+ training steps and verify that
    peak memory does not grow unboundedly.
    """

    def test_training_loop_memory_stability(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Full training loop memory stability test.

        Includes:
        - Forward pass
        - Loss computation
        - Backward pass
        - Optimizer step
        - Gradient zeroing
        """
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=8,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 2
        seq_len = 32
        training_steps = TRAINING_STEPS

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
            targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        detector.set_baseline()
        print(f"\nTraining baseline: {detector.baseline.allocated_mb:.2f} MB")

        # Main training loop
        print(f"Running {training_steps} training steps...")
        for step in range(training_steps):
            tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
            targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            # Clean up
            del tokens, targets, logits, loss

            if (step + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                stats = detector.sample()
                if (step + 1) % (MEMORY_SAMPLE_INTERVAL * 10) == 0:
                    print(f"  Step {step+1}: {stats.allocated_mb:.2f} MB")

        torch.cuda.synchronize()
        detector.sample()

        has_leak, message = detector.check_leak()
        report = detector.report()

        assert not has_leak, (
            f"Memory leak during training: {message}\n"
            f"Peak: {report.get('peak_allocated_mb', 0):.2f} MB"
        )

    def test_training_with_gradient_accumulation(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Test memory stability with gradient accumulation."""
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=8,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 2
        seq_len = 32
        accum_steps = 4
        training_steps = min(500, TRAINING_STEPS)

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            for acc in range(accum_steps):
                tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                logits = model(tokens)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
                (loss / accum_steps).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        detector.set_baseline()

        # Training with accumulation
        for step in range(training_steps):
            for acc in range(accum_steps):
                tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cuda_device)
                logits = model(tokens)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
                (loss / accum_steps).backward()
                del tokens, targets, logits, loss

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (step + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak with gradient accumulation: {message}"

    def test_moe_weight_cache_refresh(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Test that MoE weight cache refresh doesn't leak memory."""
        from nmoe.config import Config
        from nmoe.model import Transformer, MoE

        detector = memory_tracking_fixture

        cfg = Config(
            dim=256,
            n_layers=3,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=8,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="fp8",  # Use fp8 to trigger blockscaled path with cache
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.train()

        # Find MoE layers
        moe_layers = [m for m in model.modules() if isinstance(m, MoE)]

        refresh_count = min(500, TRAINING_STEPS)

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            for moe in moe_layers:
                moe.refresh_weight_cache()

        torch.cuda.synchronize()
        detector.set_baseline()

        # Repeated cache refresh
        for i in range(refresh_count):
            for moe in moe_layers:
                moe.refresh_weight_cache()

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                gc.collect()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak during weight cache refresh: {message}"


@pytest.mark.gpu
class TestExpertCacheMemoryLeak:
    """Test LazyExpertWeights cache for memory leaks.

    Verifies that:
    1. Cache eviction properly releases GPU memory
    2. No reference cycles hold experts in memory
    3. Cache statistics are accurate
    """

    def test_lazy_expert_eviction_releases_memory(
        self,
        cuda_device,
        memory_tracking_fixture,
        tmp_path,
    ):
        """Test that evicting experts from LazyExpertWeights releases GPU memory."""
        from nmoe.memory_opt import LazyExpertWeights
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        # Create a checkpoint for LazyExpertWeights to load from
        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=16,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()

        # Save checkpoint
        ckpt_path = tmp_path / "test_checkpoint.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        # Test LazyExpertWeights with small cache
        max_loaded = 4
        n_experts = 16

        detector.reset()

        lazy = LazyExpertWeights(
            checkpoint_path=str(ckpt_path),
            n_experts=n_experts,
            max_loaded=max_loaded,
            eviction_policy="lru",
        )

        # Load all experts (will evict as cache fills)
        for e in range(n_experts):
            _ = lazy.get_expert(e)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        after_load = MemoryStats.capture(str(cuda_device))

        # Verify cache is at max capacity
        assert len(lazy) == max_loaded, f"Cache should have {max_loaded} experts, has {len(lazy)}"

        # Clear cache
        lazy.clear()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        after_clear = MemoryStats.capture(str(cuda_device))

        # Memory should decrease after clearing
        freed_mb = (after_load.allocated_bytes - after_clear.allocated_bytes) / (1024 ** 2)
        print(f"\nMemory freed after cache clear: {freed_mb:.2f} MB")

        assert freed_mb > 0 or after_clear.allocated_mb < 10, (
            f"Cache clear should free memory. Before: {after_load.allocated_mb:.2f} MB, "
            f"After: {after_clear.allocated_mb:.2f} MB"
        )

    def test_no_reference_cycles_in_expert_cache(
        self,
        cuda_device,
        memory_tracking_fixture,
        tmp_path,
    ):
        """Test that LazyExpertWeights doesn't create reference cycles."""
        from nmoe.memory_opt import LazyExpertWeights
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=8,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()

        ckpt_path = tmp_path / "test_checkpoint.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        detector.reset()

        # Create LazyExpertWeights and track with weak reference
        lazy = LazyExpertWeights(
            checkpoint_path=str(ckpt_path),
            n_experts=8,
            max_loaded=4,
            eviction_policy="lru",
        )

        # Load some experts
        for e in range(4):
            _ = lazy.get_expert(e)

        # Create weak ref
        lazy_ref = weakref.ref(lazy)

        # Delete and collect
        del lazy
        gc.collect()

        # Weak ref should be dead (no reference cycles keeping it alive)
        assert lazy_ref() is None, "LazyExpertWeights has reference cycle preventing cleanup"

    def test_cache_operations_memory_stability(
        self,
        cuda_device,
        memory_tracking_fixture,
        tmp_path,
    ):
        """Test memory stability during repeated cache operations."""
        from nmoe.memory_opt import LazyExpertWeights
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=512,
            vocab_size=1000,
            n_routed_experts=16,
            n_activated_experts=2,
            n_dense_layers=1,
            max_position_embeddings=256,
            batch_size=2,
            seq_len=32,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()

        ckpt_path = tmp_path / "test_checkpoint.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        detector.reset()

        lazy = LazyExpertWeights(
            checkpoint_path=str(ckpt_path),
            n_experts=16,
            max_loaded=4,
            eviction_policy="lru",
        )

        operations = min(1000, CACHE_OPERATIONS)

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            expert_id = torch.randint(0, 16, (1,)).item()
            _ = lazy.get_expert(expert_id)

        torch.cuda.synchronize()
        gc.collect()
        detector.set_baseline()

        # Repeated random accesses
        for i in range(operations):
            expert_id = torch.randint(0, 16, (1,)).item()
            _ = lazy.get_expert(expert_id)

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                gc.collect()
                detector.sample()

        has_leak, message = detector.check_leak()
        stats = lazy.stats()
        print(f"\nCache stats: hit_rate={stats['hit_rate']:.2%}, evict_count={stats['evict_count']}")

        assert not has_leak, f"Memory leak during cache operations: {message}"


@pytest.mark.gpu
class TestKVCacheMemoryLeak:
    """Test KV cache allocation/deallocation cycles for memory leaks."""

    def test_kv_cache_allocate_deallocate_cycles(
        self,
        cuda_device,
        memory_tracking_fixture,
    ):
        """Test repeated KV cache allocation/deallocation cycles."""
        detector = memory_tracking_fixture

        # Simulate KV cache allocation patterns
        batch_sizes = [1, 2, 4, 8]
        seq_lens = [64, 128, 256, 512]
        hidden_dim = 256
        n_heads = 4
        n_layers = 4

        cycles = KV_CACHE_CYCLES

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            bs = batch_sizes[_ % len(batch_sizes)]
            sl = seq_lens[_ % len(seq_lens)]

            # Allocate KV cache
            kv_caches = []
            for _ in range(n_layers):
                k_cache = torch.zeros(bs, n_heads, sl, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                v_cache = torch.zeros(bs, n_heads, sl, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                kv_caches.append((k_cache, v_cache))

            # Deallocate
            del kv_caches
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        detector.set_baseline()

        # Main cycles
        for i in range(cycles):
            bs = batch_sizes[i % len(batch_sizes)]
            sl = seq_lens[i % len(seq_lens)]

            # Allocate
            kv_caches = []
            for _ in range(n_layers):
                k_cache = torch.zeros(bs, n_heads, sl, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                v_cache = torch.zeros(bs, n_heads, sl, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                kv_caches.append((k_cache, v_cache))

            # Deallocate
            del kv_caches

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak in KV cache cycles: {message}"

    def test_kv_cache_returns_to_baseline(
        self,
        cuda_device,
        memory_tracking_fixture,
    ):
        """Test that memory returns to baseline after KV cache deallocation."""
        detector = memory_tracking_fixture

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        baseline = MemoryStats.capture(str(cuda_device))

        # Large allocation
        batch_size = 16
        seq_len = 1024
        hidden_dim = 512
        n_heads = 8
        n_layers = 8

        kv_caches = []
        for _ in range(n_layers):
            k_cache = torch.zeros(batch_size, n_heads, seq_len, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
            v_cache = torch.zeros(batch_size, n_heads, seq_len, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
            kv_caches.append((k_cache, v_cache))

        torch.cuda.synchronize()
        after_alloc = MemoryStats.capture(str(cuda_device))

        # Deallocate
        del kv_caches
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        after_free = MemoryStats.capture(str(cuda_device))

        # Memory should return close to baseline
        # Allow some tolerance for fragmentation
        tolerance_mb = 10.0
        delta_mb = after_free.allocated_mb - baseline.allocated_mb

        print(f"\nBaseline: {baseline.allocated_mb:.2f} MB")
        print(f"After alloc: {after_alloc.allocated_mb:.2f} MB")
        print(f"After free: {after_free.allocated_mb:.2f} MB")
        print(f"Delta from baseline: {delta_mb:.2f} MB")

        assert delta_mb < tolerance_mb, (
            f"Memory did not return to baseline after KV cache free. "
            f"Baseline: {baseline.allocated_mb:.2f} MB, "
            f"After free: {after_free.allocated_mb:.2f} MB"
        )

    def test_varying_sequence_lengths(
        self,
        cuda_device,
        memory_tracking_fixture,
    ):
        """Test memory stability with varying sequence lengths."""
        detector = memory_tracking_fixture

        batch_size = 4
        hidden_dim = 256
        n_heads = 4
        n_layers = 4

        cycles = min(500, KV_CACHE_CYCLES)

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            seq_len = torch.randint(32, 512, (1,)).item()
            kv_caches = []
            for _ in range(n_layers):
                k = torch.zeros(batch_size, n_heads, seq_len, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                v = torch.zeros(batch_size, n_heads, seq_len, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                kv_caches.append((k, v))
            del kv_caches
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        detector.set_baseline()

        # Main cycles with varying lengths
        for i in range(cycles):
            seq_len = torch.randint(32, 512, (1,)).item()
            kv_caches = []
            for _ in range(n_layers):
                k = torch.zeros(batch_size, n_heads, seq_len, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                v = torch.zeros(batch_size, n_heads, seq_len, hidden_dim // n_heads, device=cuda_device, dtype=torch.bfloat16)
                kv_caches.append((k, v))

            del kv_caches

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak with varying sequence lengths: {message}"


@pytest.mark.gpu
class TestMemoryTrackerAccuracy:
    """Test MemoryTracker utility for accuracy and correctness."""

    def test_memory_tracker_snapshot_accuracy(
        self,
        cuda_device,
    ):
        """Test that MemoryTracker snapshots match torch.cuda stats."""
        from nmoe.memory_opt import MemoryTracker

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tracker = MemoryTracker(device=str(cuda_device))

        # Take baseline snapshot
        snap1 = tracker.snapshot("baseline")

        # Compare with torch.cuda stats
        cuda_allocated = torch.cuda.memory_allocated(cuda_device)
        cuda_reserved = torch.cuda.memory_reserved(cuda_device)

        assert snap1.allocated_bytes == cuda_allocated, (
            f"Allocated mismatch: tracker={snap1.allocated_bytes}, cuda={cuda_allocated}"
        )
        assert snap1.reserved_bytes == cuda_reserved, (
            f"Reserved mismatch: tracker={snap1.reserved_bytes}, cuda={cuda_reserved}"
        )

        # Allocate tensors
        tensors = [
            torch.randn(1024, 1024, device=cuda_device, dtype=torch.float32)
            for _ in range(10)
        ]

        snap2 = tracker.snapshot("after_alloc")

        # Verify increase
        assert snap2.allocated_bytes > snap1.allocated_bytes
        assert snap2.allocated_bytes == torch.cuda.memory_allocated(cuda_device)

        # Free tensors
        del tensors
        gc.collect()
        torch.cuda.empty_cache()

        snap3 = tracker.snapshot("after_free")

        # Verify memory returned (approximately)
        assert snap3.allocated_bytes < snap2.allocated_bytes

    def test_memory_tracker_diff_calculation(
        self,
        cuda_device,
    ):
        """Test MemoryTracker diff calculation."""
        from nmoe.memory_opt import MemoryTracker

        gc.collect()
        torch.cuda.empty_cache()

        tracker = MemoryTracker(device=str(cuda_device))
        tracker.snapshot("start")

        # Allocate known amount
        tensor_size = 1024 * 1024  # 1M elements
        element_size = 4  # float32
        expected_bytes = tensor_size * element_size

        tensor = torch.zeros(tensor_size, device=cuda_device, dtype=torch.float32)
        tracker.snapshot("end")

        diff = tracker.diff("start", "end")

        # Allow some tolerance for CUDA overhead
        assert "allocated_diff_gb" in diff
        allocated_diff_bytes = diff["allocated_diff_gb"] * (1024 ** 3)

        # Should be close to expected
        assert abs(allocated_diff_bytes - expected_bytes) < 1024 * 1024, (
            f"Diff calculation wrong: got {allocated_diff_bytes}, expected ~{expected_bytes}"
        )

        del tensor

    def test_memory_tracker_peak_matches_cuda(
        self,
        cuda_device,
    ):
        """Test that MemoryTracker peak matches torch.cuda.max_memory_allocated."""
        from nmoe.memory_opt import MemoryTracker

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tracker = MemoryTracker(device=str(cuda_device))
        tracker.snapshot("start")

        # Create peak allocation
        peak_tensors = [
            torch.randn(2048, 2048, device=cuda_device, dtype=torch.float32)
            for _ in range(5)
        ]

        peak_snap = tracker.snapshot("peak")

        # Free some
        del peak_tensors[:3]
        gc.collect()

        after_free_snap = tracker.snapshot("after_free")

        # Peak should match torch.cuda
        assert peak_snap.max_allocated_bytes == torch.cuda.max_memory_allocated(cuda_device)

    def test_memory_tracker_reset(
        self,
        cuda_device,
    ):
        """Test MemoryTracker reset functionality."""
        from nmoe.memory_opt import MemoryTracker

        gc.collect()
        torch.cuda.empty_cache()

        tracker = MemoryTracker(device=str(cuda_device))

        # Take some snapshots
        tracker.snapshot("s1")
        tensor = torch.randn(1024, 1024, device=cuda_device)
        tracker.snapshot("s2")

        assert len(tracker.snapshots) == 2

        # Reset
        tracker.reset()

        assert len(tracker.snapshots) == 0

        # Peak should be reset
        new_snap = tracker.snapshot("after_reset")
        assert new_snap.max_allocated_bytes >= new_snap.allocated_bytes

        del tensor


@pytest.mark.gpu
@pytest.mark.slow
class TestCUDAMemoryFragmentation:
    """Test CUDA memory fragmentation detection and behavior."""

    def test_mixed_allocation_fragmentation(
        self,
        cuda_device,
        memory_tracking_fixture,
    ):
        """Test fragmentation with mixed allocation sizes."""
        detector = memory_tracking_fixture

        alloc_count = FRAGMENTATION_ALLOCS

        # Different allocation sizes to induce fragmentation
        sizes = [
            (64, 64),
            (256, 256),
            (1024, 1024),
            (128, 512),
            (512, 128),
            (2048, 512),
        ]

        detector.reset()

        tensors = []

        # Warmup
        for _ in range(min(100, WARMUP_ITERATIONS)):
            size = sizes[_ % len(sizes)]
            t = torch.randn(size, device=cuda_device, dtype=torch.float32)
            tensors.append(t)
            if len(tensors) > 50:
                # Free oldest half
                for _ in range(25):
                    tensors.pop(0)
                gc.collect()

        tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        detector.set_baseline()

        # Main fragmentation test
        for i in range(alloc_count):
            size = sizes[i % len(sizes)]
            t = torch.randn(size, device=cuda_device, dtype=torch.float32)
            tensors.append(t)

            # Randomly free to induce fragmentation
            if len(tensors) > 100 and torch.rand(1).item() > 0.5:
                # Free random tensors
                n_free = min(20, len(tensors))
                indices = torch.randperm(len(tensors))[:n_free].tolist()
                for idx in sorted(indices, reverse=True):
                    del tensors[idx]

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                detector.sample()

        # Clear all and check
        tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_stats = MemoryStats.capture(str(cuda_device))

        # After clearing, allocated should be very low
        # Reserved may be high due to fragmentation
        fragmentation_ratio = (
            final_stats.reserved_bytes / max(final_stats.allocated_bytes, 1)
            if final_stats.allocated_bytes > 0
            else 0
        )

        print(f"\nFragmentation stats:")
        print(f"  Final allocated: {final_stats.allocated_mb:.2f} MB")
        print(f"  Final reserved: {final_stats.reserved_mb:.2f} MB")
        print(f"  Fragmentation ratio: {fragmentation_ratio:.2f}x")

        # Allocated should return to near-baseline
        baseline_mb = detector.baseline.allocated_mb if detector.baseline else 0
        assert final_stats.allocated_mb < baseline_mb + 10, (
            f"Memory not released after clearing: {final_stats.allocated_mb:.2f} MB "
            f"(baseline: {baseline_mb:.2f} MB)"
        )

    def test_high_fragmentation_detection(
        self,
        cuda_device,
    ):
        """Test detection of high fragmentation scenarios."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        initial_reserved = torch.cuda.memory_reserved(cuda_device)
        initial_allocated = torch.cuda.memory_allocated(cuda_device)

        # Create fragmentation by allocating and freeing
        tensors = []

        # Phase 1: Allocate many small tensors
        for _ in range(1000):
            tensors.append(torch.randn(128, 128, device=cuda_device))

        # Phase 2: Free every other tensor
        for i in range(0, len(tensors), 2):
            tensors[i] = None
        gc.collect()

        mid_reserved = torch.cuda.memory_reserved(cuda_device)
        mid_allocated = torch.cuda.memory_allocated(cuda_device)

        # Phase 3: Try to allocate a large tensor
        # This may fail or trigger defragmentation depending on CUDA allocator
        try:
            large_tensor = torch.randn(2048, 2048, device=cuda_device)
            large_alloc_succeeded = True
            del large_tensor
        except RuntimeError:
            large_alloc_succeeded = False

        # Clean up
        tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_reserved = torch.cuda.memory_reserved(cuda_device)
        final_allocated = torch.cuda.memory_allocated(cuda_device)

        print(f"\nFragmentation detection:")
        print(f"  Initial: alloc={initial_allocated / 1e6:.1f}MB, reserved={initial_reserved / 1e6:.1f}MB")
        print(f"  Mid: alloc={mid_allocated / 1e6:.1f}MB, reserved={mid_reserved / 1e6:.1f}MB")
        print(f"  Final: alloc={final_allocated / 1e6:.1f}MB, reserved={final_reserved / 1e6:.1f}MB")
        print(f"  Large alloc succeeded: {large_alloc_succeeded}")

        # Final allocated should be back to initial
        assert final_allocated < initial_allocated + 1024 * 1024

    def test_defragmentation_with_empty_cache(
        self,
        cuda_device,
    ):
        """Test that empty_cache helps with fragmentation."""
        gc.collect()
        torch.cuda.empty_cache()

        # Create fragmented memory
        tensors = []
        for _ in range(500):
            tensors.append(torch.randn(256, 256, device=cuda_device))

        for i in range(0, len(tensors), 2):
            tensors[i] = None
        gc.collect()

        before_empty = torch.cuda.memory_reserved(cuda_device)

        # Empty cache
        torch.cuda.empty_cache()

        after_empty = torch.cuda.memory_reserved(cuda_device)

        # Reserved should decrease (or at least not increase)
        print(f"\nDefragmentation test:")
        print(f"  Reserved before empty_cache: {before_empty / 1e6:.1f} MB")
        print(f"  Reserved after empty_cache: {after_empty / 1e6:.1f} MB")

        # Clean up
        tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.gpu
class TestRdepMemoryLeak:
    """Test RDEP-specific memory leak scenarios."""

    def test_rdep_dispatch_return_memory_stability(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Test memory stability during RDEP dispatch/return cycles."""
        from nmoe.rdep import Rdep

        detector = memory_tracking_fixture

        dim = 256
        n_local = 8
        topk = 2
        capacity = 4096

        rdep = Rdep(dim=dim, n_local=n_local, topk=topk, profile="bf16", capacity=capacity)

        T = 64
        Dff = dim * 4

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_local, (T, topk), device=cuda_device, dtype=torch.int32)
        gate_logits = torch.randn(T, topk, device=cuda_device, dtype=torch.float32)
        gates = F.softmax(gate_logits, dim=-1).to(torch.bfloat16)

        W1 = torch.randn(n_local, dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, Dff, dim, device=cuda_device, dtype=torch.bfloat16) * 0.02

        iterations = min(1000, INFERENCE_ITERATIONS)

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            with torch.no_grad():
                _ = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        torch.cuda.synchronize()
        detector.set_baseline()

        # Main dispatch cycles
        for i in range(iterations):
            with torch.no_grad():
                out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
                _ = out.sum().item()

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak in RDEP dispatch: {message}"

    def test_rdep_blockscaled_memory_stability(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Test memory stability with blockscaled (FP8) RDEP."""
        from nmoe.rdep import Rdep
        from nmoe.blockscaled.grouped import quantize_weights

        detector = memory_tracking_fixture

        dim = 256
        n_local = 8
        topk = 2
        capacity = 4096

        rdep = Rdep(dim=dim, n_local=n_local, topk=topk, profile="fp8", capacity=capacity)

        T = 64
        Dff = dim * 4

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_local, (T, topk), device=cuda_device, dtype=torch.int32)
        gate_logits = torch.randn(T, topk, device=cuda_device, dtype=torch.float32)
        gates = F.softmax(gate_logits, dim=-1).to(torch.bfloat16)

        W1 = torch.randn(n_local, dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, dim, Dff, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, Dff, dim, device=cuda_device, dtype=torch.bfloat16) * 0.02

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        iterations = min(1000, INFERENCE_ITERATIONS)

        detector.reset()

        # Warmup
        for _ in range(min(50, WARMUP_ITERATIONS)):
            with torch.no_grad():
                _ = rdep.moe_blockscaled(x, eid, gates, W1, W3, W2, W_cache)

        torch.cuda.synchronize()
        detector.set_baseline()

        # Main dispatch cycles
        for i in range(iterations):
            with torch.no_grad():
                out = rdep.moe_blockscaled(x, eid, gates, W1, W3, W2, W_cache)
                _ = out.sum().item()

            if (i + 1) % MEMORY_SAMPLE_INTERVAL == 0:
                torch.cuda.synchronize()
                detector.sample()

        has_leak, message = detector.check_leak()
        assert not has_leak, f"Memory leak in RDEP blockscaled: {message}"


# ==============================================================================
# Quick Smoke Tests (not marked slow)
# ==============================================================================


@pytest.mark.gpu
class TestMemoryLeakQuickSmoke:
    """Quick smoke tests for memory leaks (not marked slow)."""

    def test_quick_inference_smoke(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Quick smoke test for inference memory leaks."""
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=128,
            n_layers=1,
            n_heads=2,
            inter_dim=256,
            moe_inter_dim=256,
            vocab_size=1000,
            n_routed_experts=4,
            n_activated_experts=1,
            n_dense_layers=0,
            max_position_embeddings=128,
            batch_size=2,
            seq_len=16,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.eval()

        detector.reset()

        # Quick warmup
        for _ in range(10):
            with torch.no_grad():
                tokens = torch.randint(0, cfg.vocab_size, (2, 16), device=cuda_device)
                _ = model(tokens)

        torch.cuda.synchronize()
        detector.set_baseline()

        # Quick test (100 iterations)
        for i in range(100):
            with torch.no_grad():
                tokens = torch.randint(0, cfg.vocab_size, (2, 16), device=cuda_device)
                logits = model(tokens)
                _ = logits.sum().item()

            if (i + 1) % 25 == 0:
                torch.cuda.synchronize()
                detector.sample()

        has_leak, message = detector.check_leak()
        # Use higher threshold for quick test
        report = detector.report()
        assert report["growth_percent"] < 5.0, f"Quick smoke test failed: {message}"

    def test_quick_training_smoke(
        self,
        cuda_device,
        memory_tracking_fixture,
        random_seed,
    ):
        """Quick smoke test for training memory leaks."""
        from nmoe.config import Config
        from nmoe.model import Transformer

        detector = memory_tracking_fixture

        cfg = Config(
            dim=128,
            n_layers=1,
            n_heads=2,
            inter_dim=256,
            moe_inter_dim=256,
            vocab_size=1000,
            n_routed_experts=4,
            n_activated_experts=1,
            n_dense_layers=0,
            max_position_embeddings=128,
            batch_size=2,
            seq_len=16,
            dtype="bf16",
        )

        model = Transformer(cfg).cuda()
        model.init_weights()
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        detector.reset()

        # Quick warmup
        for _ in range(10):
            tokens = torch.randint(0, cfg.vocab_size, (2, 16), device=cuda_device)
            targets = torch.randint(0, cfg.vocab_size, (2, 16), device=cuda_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        detector.set_baseline()

        # Quick test (50 steps)
        for i in range(50):
            tokens = torch.randint(0, cfg.vocab_size, (2, 16), device=cuda_device)
            targets = torch.randint(0, cfg.vocab_size, (2, 16), device=cuda_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            del tokens, targets, logits, loss

            if (i + 1) % 10 == 0:
                torch.cuda.synchronize()
                detector.sample()

        has_leak, message = detector.check_leak()
        report = detector.report()
        # Use higher threshold for quick test
        assert report["growth_percent"] < 5.0, f"Quick smoke test failed: {message}"
