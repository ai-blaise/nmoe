"""Comprehensive unit tests for LazyExpertWeights class.

Tests cover:
- LazyExpertWeights instantiation with defaults and custom parameters
- Expert loading and caching behavior
- Cache eviction with different policies (LRU, LFU, FIFO)
- Prefetch functionality
- Statistics tracking (hits, misses, evictions)
- Edge cases and error handling

These are P0 critical tests as LazyExpertWeights is essential for memory-efficient
MoE model serving and training.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator

import pytest
import torch

from nmoe.memory_opt import (
    LazyExpertWeights,
    ExpertCachePolicy,
    LRUPolicy,
    LFUPolicy,
    FIFOPolicy,
    ARCPolicy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def n_experts() -> int:
    """Number of experts for test checkpoints."""
    return 8


@pytest.fixture
def hidden_dim() -> int:
    """Hidden dimension for test weights."""
    return 64


@pytest.fixture
def intermediate_dim() -> int:
    """Intermediate dimension for test weights."""
    return 128


@pytest.fixture
def test_checkpoint_file(
    tmp_path: Path,
    n_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> Generator[str, None, None]:
    """Create a temporary checkpoint file with mock expert weights.

    The checkpoint format matches the nmoe convention with W1, W3, W2 tensors
    for each layer.
    """
    checkpoint_path = tmp_path / "test_checkpoint.pt"

    # Create mock expert weights
    # Format: blocks.{layer_id}.ffn.{W1|W3|W2} with shape [n_experts, dim1, dim2]
    model_state = {
        "blocks.0.ffn.W1": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "blocks.0.ffn.W3": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "blocks.0.ffn.W2": torch.randn(n_experts, intermediate_dim, hidden_dim),
        "blocks.1.ffn.W1": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "blocks.1.ffn.W3": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "blocks.1.ffn.W2": torch.randn(n_experts, intermediate_dim, hidden_dim),
    }

    checkpoint = {"model": model_state}
    torch.save(checkpoint, str(checkpoint_path))

    yield str(checkpoint_path)


@pytest.fixture
def test_checkpoint_dir(
    tmp_path: Path,
    n_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> Generator[str, None, None]:
    """Create a temporary checkpoint directory with model.pt file."""
    checkpoint_dir = tmp_path / "checkpoint_dir"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.pt"

    model_state = {
        "blocks.0.ffn.W1": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "blocks.0.ffn.W3": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "blocks.0.ffn.W2": torch.randn(n_experts, intermediate_dim, hidden_dim),
    }

    checkpoint = {"model": model_state}
    torch.save(checkpoint, str(checkpoint_path))

    yield str(checkpoint_dir)


@pytest.fixture
def test_checkpoint_with_model_expert(
    tmp_path: Path,
    n_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> Generator[str, None, None]:
    """Create checkpoint using model_expert key format."""
    checkpoint_path = tmp_path / "expert_checkpoint.pt"

    model_expert_state = {
        "layer.0.W1": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "layer.0.W3": torch.randn(n_experts, hidden_dim, intermediate_dim),
        "layer.0.W2": torch.randn(n_experts, intermediate_dim, hidden_dim),
    }

    checkpoint = {"model_expert": model_expert_state}
    torch.save(checkpoint, str(checkpoint_path))

    yield str(checkpoint_path)


@pytest.fixture
def empty_checkpoint(tmp_path: Path) -> Generator[str, None, None]:
    """Create a checkpoint with no expert weights."""
    checkpoint_path = tmp_path / "empty_checkpoint.pt"

    checkpoint = {"model": {"some_other_tensor": torch.randn(10, 10)}}
    torch.save(checkpoint, str(checkpoint_path))

    yield str(checkpoint_path)


# =============================================================================
# LazyExpertWeights Creation Tests
# =============================================================================


class TestLazyExpertWeightsCreation:
    """Tests for LazyExpertWeights instantiation."""

    def test_default_creation(self, test_checkpoint_file: str, n_experts: int):
        """LazyExpertWeights can be created with minimal arguments."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
        )

        assert lazy.n_experts == n_experts
        assert lazy.max_loaded == n_experts  # default is n_experts
        assert lazy.dtype == torch.bfloat16  # default dtype
        assert lazy.device == "cuda"  # default device

    def test_creation_with_all_parameters(self, test_checkpoint_file: str, n_experts: int):
        """LazyExpertWeights accepts all parameters."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            dtype=torch.float32,
            device="cpu",
            max_loaded=4,
            eviction_policy="lfu",
        )

        assert lazy.n_experts == n_experts
        assert lazy.max_loaded == 4
        assert lazy.dtype == torch.float32
        assert lazy.device == "cpu"

    def test_creation_with_different_policies(self, test_checkpoint_file: str, n_experts: int):
        """All eviction policies can be used."""
        for policy in ["lru", "lfu", "fifo", "arc"]:
            lazy = LazyExpertWeights(
                checkpoint_path=test_checkpoint_file,
                n_experts=n_experts,
                eviction_policy=policy,
                max_loaded=4,
            )
            assert lazy.max_loaded == 4

    def test_max_loaded_clamped_to_n_experts(self, test_checkpoint_file: str, n_experts: int):
        """max_loaded is clamped to n_experts if larger."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=100,  # larger than n_experts
        )

        assert lazy.max_loaded == n_experts

    def test_invalid_max_loaded_raises(self, test_checkpoint_file: str, n_experts: int):
        """max_loaded < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_loaded must be >= 1"):
            LazyExpertWeights(
                checkpoint_path=test_checkpoint_file,
                n_experts=n_experts,
                max_loaded=0,
            )

    def test_invalid_eviction_policy_raises(self, test_checkpoint_file: str, n_experts: int):
        """Unknown eviction policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy"):
            LazyExpertWeights(
                checkpoint_path=test_checkpoint_file,
                n_experts=n_experts,
                eviction_policy="invalid_policy",
            )


# =============================================================================
# Expert Loading Tests
# =============================================================================


class TestExpertLoading:
    """Tests for loading experts from checkpoint."""

    def test_get_expert_loads_from_disk(self, test_checkpoint_file: str, n_experts: int):
        """get_expert loads expert weights from checkpoint."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",  # Use CPU to avoid CUDA dependency
        )

        expert_weights = lazy.get_expert(0)

        assert isinstance(expert_weights, dict)
        assert len(expert_weights) > 0
        # Check that tensors are on correct device
        # Note: expert_weights may contain both tensors and nested dicts
        tensor_count = 0
        for key, value in expert_weights.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
                tensor_count += 1
        assert tensor_count > 0  # Should have at least one tensor

    def test_get_expert_caches_result(self, test_checkpoint_file: str, n_experts: int):
        """get_expert caches loaded expert."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        # First load
        weights1 = lazy.get_expert(0)
        # Second access should hit cache
        weights2 = lazy.get_expert(0)

        # Same object should be returned
        assert weights1 is weights2
        assert 0 in lazy

    def test_invalid_expert_id_raises(self, test_checkpoint_file: str, n_experts: int):
        """get_expert with invalid expert_id raises ValueError."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
        )

        with pytest.raises(ValueError, match="expert_id must be in"):
            lazy.get_expert(-1)

        with pytest.raises(ValueError, match="expert_id must be in"):
            lazy.get_expert(n_experts)  # Out of range

        with pytest.raises(ValueError, match="expert_id must be in"):
            lazy.get_expert(n_experts + 10)

    def test_load_from_directory(self, test_checkpoint_dir: str, n_experts: int):
        """LazyExpertWeights can load from checkpoint directory."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_dir,
            n_experts=n_experts,
            device="cpu",
        )

        weights = lazy.get_expert(0)
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_load_from_model_expert_format(
        self, test_checkpoint_with_model_expert: str, n_experts: int
    ):
        """LazyExpertWeights loads from model_expert checkpoint format."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_with_model_expert,
            n_experts=n_experts,
            device="cpu",
        )

        weights = lazy.get_expert(0)
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_nonexistent_checkpoint_raises(self, n_experts: int):
        """Loading from nonexistent checkpoint raises RuntimeError."""
        lazy = LazyExpertWeights(
            checkpoint_path="/nonexistent/path/checkpoint.pt",
            n_experts=n_experts,
        )

        with pytest.raises(RuntimeError, match="Checkpoint not found"):
            lazy.get_expert(0)

    def test_empty_checkpoint_raises(self, empty_checkpoint: str, n_experts: int):
        """Checkpoint without expert weights raises RuntimeError."""
        lazy = LazyExpertWeights(
            checkpoint_path=empty_checkpoint,
            n_experts=n_experts,
            device="cpu",
        )

        with pytest.raises(RuntimeError, match="No expert weights found"):
            lazy.get_expert(0)

    def test_dtype_conversion(self, test_checkpoint_file: str, n_experts: int):
        """Expert weights are converted to specified dtype."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            lazy = LazyExpertWeights(
                checkpoint_path=test_checkpoint_file,
                n_experts=n_experts,
                dtype=dtype,
                device="cpu",
            )

            weights = lazy.get_expert(0)
            # Note: weights may contain both tensors and nested dicts
            tensor_count = 0
            for value in weights.values():
                if isinstance(value, torch.Tensor):
                    assert value.dtype == dtype
                    tensor_count += 1
            assert tensor_count > 0  # Should have at least one tensor

            # Clear for next iteration
            lazy.clear()


# =============================================================================
# Cache Eviction Tests
# =============================================================================


class TestCacheEviction:
    """Tests for cache eviction behavior."""

    def test_eviction_when_exceeding_max_loaded(self, test_checkpoint_file: str, n_experts: int):
        """Experts are evicted when cache exceeds max_loaded."""
        max_loaded = 3
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=max_loaded,
            device="cpu",
        )

        # Load more experts than max_loaded
        for i in range(max_loaded + 2):
            lazy.get_expert(i)

        # Should have exactly max_loaded experts
        assert len(lazy) == max_loaded

    def test_lru_eviction_policy(self, test_checkpoint_file: str, n_experts: int):
        """LRU policy evicts least recently used expert."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=2,
            eviction_policy="lru",
            device="cpu",
        )

        # Load experts 0, 1
        lazy.get_expert(0)
        lazy.get_expert(1)

        # Access expert 0 to make it most recently used
        lazy.get_expert(0)

        # Load expert 2, should evict expert 1 (least recently used)
        lazy.get_expert(2)

        assert 0 in lazy
        assert 1 not in lazy
        assert 2 in lazy

    def test_lfu_eviction_policy(self, test_checkpoint_file: str, n_experts: int):
        """LFU policy evicts least frequently used expert."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=2,
            eviction_policy="lfu",
            device="cpu",
        )

        # Load experts 0, 1
        lazy.get_expert(0)
        lazy.get_expert(1)

        # Access expert 0 multiple times to increase frequency
        lazy.get_expert(0)
        lazy.get_expert(0)

        # Load expert 2, should evict expert 1 (least frequently used)
        lazy.get_expert(2)

        assert 0 in lazy
        assert 1 not in lazy
        assert 2 in lazy

    def test_fifo_eviction_policy(self, test_checkpoint_file: str, n_experts: int):
        """FIFO policy evicts first-in expert regardless of access pattern."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=2,
            eviction_policy="fifo",
            device="cpu",
        )

        # Load experts 0, 1
        lazy.get_expert(0)
        lazy.get_expert(1)

        # Access expert 0 (should not affect FIFO order)
        lazy.get_expert(0)

        # Load expert 2, should evict expert 0 (first in)
        lazy.get_expert(2)

        assert 0 not in lazy
        assert 1 in lazy
        assert 2 in lazy

    def test_arc_eviction_policy(self, test_checkpoint_file: str, n_experts: int):
        """ARC policy can be used for eviction."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=3,
            eviction_policy="arc",
            device="cpu",
        )

        # Load several experts
        for i in range(5):
            lazy.get_expert(i % n_experts)

        # Should respect max_loaded
        assert len(lazy) <= 3


# =============================================================================
# set_max_loaded Tests
# =============================================================================


class TestSetMaxLoaded:
    """Tests for set_max_loaded functionality."""

    def test_set_max_loaded_increases(self, test_checkpoint_file: str, n_experts: int):
        """set_max_loaded can increase cache size."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=2,
            device="cpu",
        )

        lazy.set_max_loaded(4)
        assert lazy.max_loaded == 4

        # Load 4 experts
        for i in range(4):
            lazy.get_expert(i)

        assert len(lazy) == 4

    def test_set_max_loaded_decreases_triggers_eviction(
        self, test_checkpoint_file: str, n_experts: int
    ):
        """set_max_loaded triggers eviction when reduced."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
            device="cpu",
        )

        # Load 4 experts
        for i in range(4):
            lazy.get_expert(i)

        assert len(lazy) == 4

        # Reduce max_loaded
        lazy.set_max_loaded(2)

        assert lazy.max_loaded == 2
        assert len(lazy) == 2

    def test_set_max_loaded_clamped(self, test_checkpoint_file: str, n_experts: int):
        """set_max_loaded is clamped to n_experts."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=2,
            device="cpu",
        )

        lazy.set_max_loaded(100)
        assert lazy.max_loaded == n_experts

    def test_set_max_loaded_invalid_raises(self, test_checkpoint_file: str, n_experts: int):
        """set_max_loaded with invalid value raises ValueError."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
        )

        with pytest.raises(ValueError, match="max_loaded must be >= 1"):
            lazy.set_max_loaded(0)

        with pytest.raises(ValueError, match="max_loaded must be >= 1"):
            lazy.set_max_loaded(-1)


# =============================================================================
# Prefetch Tests
# =============================================================================


class TestPrefetch:
    """Tests for prefetch functionality."""

    def test_prefetch_loads_experts(self, test_checkpoint_file: str, n_experts: int):
        """prefetch loads specified experts into cache."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
            device="cpu",
        )

        lazy.prefetch([0, 1, 2])

        assert 0 in lazy
        assert 1 in lazy
        assert 2 in lazy
        assert len(lazy) == 3

    def test_prefetch_respects_max_loaded(self, test_checkpoint_file: str, n_experts: int):
        """prefetch evicts to stay within max_loaded."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=2,
            device="cpu",
        )

        lazy.prefetch([0, 1, 2, 3])

        # Only 2 should be loaded
        assert len(lazy) == 2

    def test_prefetch_skips_already_loaded(self, test_checkpoint_file: str, n_experts: int):
        """prefetch doesn't reload already cached experts."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
            device="cpu",
        )

        # Load expert 0
        lazy.get_expert(0)
        initial_stats = lazy.stats()

        # Prefetch including already loaded expert
        lazy.prefetch([0, 1, 2])

        final_stats = lazy.stats()

        # Expert 0 should not have been reloaded
        assert final_stats["load_count"] == initial_stats["load_count"] + 2

    def test_prefetch_ignores_invalid_ids(self, test_checkpoint_file: str, n_experts: int):
        """prefetch ignores invalid expert IDs."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
            device="cpu",
        )

        # Include invalid IDs
        lazy.prefetch([-1, 0, 1, n_experts + 10])

        # Only valid ones should be loaded
        assert len(lazy) == 2
        assert 0 in lazy
        assert 1 in lazy

    def test_prefetch_empty_list(self, test_checkpoint_file: str, n_experts: int):
        """prefetch with empty list does nothing."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
            device="cpu",
        )

        lazy.prefetch([])
        assert len(lazy) == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    def test_stats_initial_values(self, test_checkpoint_file: str, n_experts: int):
        """stats returns zero counts initially."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
        )

        stats = lazy.stats()

        assert stats["load_count"] == 0
        assert stats["evict_count"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["currently_loaded"] == 0
        assert stats["max_loaded"] == n_experts

    def test_stats_tracks_loads(self, test_checkpoint_file: str, n_experts: int):
        """stats tracks load count."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        lazy.get_expert(0)
        lazy.get_expert(1)
        lazy.get_expert(2)

        stats = lazy.stats()
        assert stats["load_count"] == 3

    def test_stats_tracks_hits_and_misses(self, test_checkpoint_file: str, n_experts: int):
        """stats tracks cache hits and misses."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        # Miss (first access)
        lazy.get_expert(0)
        # Hit (cached)
        lazy.get_expert(0)
        # Miss (first access)
        lazy.get_expert(1)
        # Hit (cached)
        lazy.get_expert(0)

        stats = lazy.stats()
        assert stats["miss_count"] == 2
        assert stats["hit_count"] == 2
        assert stats["hit_rate"] == 0.5

    def test_stats_tracks_evictions(self, test_checkpoint_file: str, n_experts: int):
        """stats tracks eviction count."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=2,
            device="cpu",
        )

        # Load 4 experts, should cause 2 evictions
        for i in range(4):
            lazy.get_expert(i)

        stats = lazy.stats()
        assert stats["evict_count"] == 2

    def test_stats_currently_loaded(self, test_checkpoint_file: str, n_experts: int):
        """stats reports currently loaded count."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=3,
            device="cpu",
        )

        lazy.get_expert(0)
        assert lazy.stats()["currently_loaded"] == 1

        lazy.get_expert(1)
        lazy.get_expert(2)
        assert lazy.stats()["currently_loaded"] == 3

    def test_hit_rate_calculation(self, test_checkpoint_file: str, n_experts: int):
        """hit_rate is calculated correctly."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        # Pattern: miss, hit, hit, miss, hit, hit, hit, hit
        lazy.get_expert(0)  # miss
        lazy.get_expert(0)  # hit
        lazy.get_expert(0)  # hit
        lazy.get_expert(1)  # miss
        for _ in range(4):
            lazy.get_expert(1)  # 4 hits

        stats = lazy.stats()
        # 2 misses, 6 hits = 6/8 = 0.75
        assert stats["hit_rate"] == pytest.approx(0.75)


# =============================================================================
# Clear Tests
# =============================================================================


class TestClear:
    """Tests for clear functionality."""

    def test_clear_removes_all_experts(self, test_checkpoint_file: str, n_experts: int):
        """clear removes all loaded experts."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        # Load some experts
        for i in range(4):
            lazy.get_expert(i)

        assert len(lazy) == 4

        lazy.clear()

        assert len(lazy) == 0
        for i in range(4):
            assert i not in lazy

    def test_clear_allows_reloading(self, test_checkpoint_file: str, n_experts: int):
        """Experts can be reloaded after clear."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        lazy.get_expert(0)
        lazy.clear()

        # Should be able to reload
        weights = lazy.get_expert(0)
        assert isinstance(weights, dict)
        assert 0 in lazy


# =============================================================================
# Container Protocol Tests
# =============================================================================


class TestContainerProtocol:
    """Tests for __len__ and __contains__ methods."""

    def test_len_returns_loaded_count(self, test_checkpoint_file: str, n_experts: int):
        """__len__ returns number of loaded experts."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        assert len(lazy) == 0

        lazy.get_expert(0)
        assert len(lazy) == 1

        lazy.get_expert(1)
        lazy.get_expert(2)
        assert len(lazy) == 3

    def test_contains_checks_loaded(self, test_checkpoint_file: str, n_experts: int):
        """__contains__ checks if expert is loaded."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        assert 0 not in lazy
        assert 1 not in lazy

        lazy.get_expert(0)

        assert 0 in lazy
        assert 1 not in lazy


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_single_expert_cache(self, test_checkpoint_file: str, n_experts: int):
        """Cache works with max_loaded=1."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=1,
            device="cpu",
        )

        lazy.get_expert(0)
        assert 0 in lazy

        lazy.get_expert(1)
        assert 0 not in lazy
        assert 1 in lazy

        lazy.get_expert(0)
        assert 1 not in lazy
        assert 0 in lazy

    def test_load_all_experts(self, test_checkpoint_file: str, n_experts: int):
        """Can load all experts when max_loaded equals n_experts."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=n_experts,
            device="cpu",
        )

        for i in range(n_experts):
            lazy.get_expert(i)

        assert len(lazy) == n_experts
        for i in range(n_experts):
            assert i in lazy

        # No evictions should have happened
        assert lazy.stats()["evict_count"] == 0

    def test_repeated_access_same_expert(self, test_checkpoint_file: str, n_experts: int):
        """Repeated access to same expert hits cache."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        for _ in range(100):
            lazy.get_expert(0)

        stats = lazy.stats()
        assert stats["load_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_count"] == 99

    def test_boundary_expert_ids(self, test_checkpoint_file: str, n_experts: int):
        """Boundary expert IDs work correctly."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            device="cpu",
        )

        # First expert
        weights = lazy.get_expert(0)
        assert isinstance(weights, dict)

        # Last expert
        weights = lazy.get_expert(n_experts - 1)
        assert isinstance(weights, dict)


# =============================================================================
# Cache Policy Unit Tests
# =============================================================================


class TestCachePolicies:
    """Unit tests for individual cache eviction policies."""

    def test_lru_policy_basic(self):
        """LRU policy tracks access order correctly."""
        policy = LRUPolicy(max_size=3)

        policy.add(1)
        policy.add(2)
        policy.add(3)

        assert len(policy) == 3
        assert 1 in policy
        assert 2 in policy
        assert 3 in policy

        # Evict should remove first added (LRU)
        evicted = policy.evict()
        assert evicted == 1
        assert 1 not in policy

    def test_lru_policy_access_updates_order(self):
        """LRU policy updates order on access."""
        policy = LRUPolicy(max_size=3)

        policy.add(1)
        policy.add(2)
        policy.add(3)

        # Access 1 to make it most recently used
        policy.access(1)

        # Evict should now remove 2 (least recently used)
        evicted = policy.evict()
        assert evicted == 2

    def test_lfu_policy_basic(self):
        """LFU policy tracks frequency correctly."""
        policy = LFUPolicy(max_size=3)

        policy.add(1)
        policy.add(2)
        policy.add(3)

        # Access 1 multiple times
        policy.access(1)
        policy.access(1)
        # Access 2 once
        policy.access(2)

        # Evict should remove 3 (lowest frequency)
        evicted = policy.evict()
        assert evicted == 3

    def test_fifo_policy_ignores_access(self):
        """FIFO policy ignores access pattern."""
        policy = FIFOPolicy(max_size=3)

        policy.add(1)
        policy.add(2)
        policy.add(3)

        # Access 1 multiple times (should not affect order)
        policy.access(1)
        policy.access(1)
        policy.access(1)

        # Evict should still remove 1 (first in)
        evicted = policy.evict()
        assert evicted == 1

    def test_arc_policy_basic(self):
        """ARC policy works for basic operations."""
        policy = ARCPolicy(max_size=3)

        policy.add(1)
        policy.add(2)
        policy.add(3)

        assert len(policy) == 3

        # Should be able to evict
        evicted = policy.evict()
        assert evicted in [1, 2, 3]
        assert len(policy) == 2

    def test_policy_remove(self):
        """All policies support remove operation."""
        for PolicyClass in [LRUPolicy, LFUPolicy, FIFOPolicy, ARCPolicy]:
            policy = PolicyClass(max_size=3)

            policy.add(1)
            policy.add(2)

            assert 1 in policy
            policy.remove(1)
            assert 1 not in policy
            assert 2 in policy

    def test_policy_factory(self):
        """ExpertCachePolicy factory creates correct policies."""
        lru = ExpertCachePolicy.create("lru", 10)
        assert isinstance(lru, LRUPolicy)

        lfu = ExpertCachePolicy.create("lfu", 10)
        assert isinstance(lfu, LFUPolicy)

        fifo = ExpertCachePolicy.create("fifo", 10)
        assert isinstance(fifo, FIFOPolicy)

        arc = ExpertCachePolicy.create("arc", 10)
        assert isinstance(arc, ARCPolicy)

    def test_policy_invalid_max_size(self):
        """Policies reject invalid max_size."""
        for PolicyClass in [LRUPolicy, LFUPolicy, FIFOPolicy, ARCPolicy]:
            with pytest.raises(ValueError, match="max_size must be >= 1"):
                PolicyClass(max_size=0)

            with pytest.raises(ValueError, match="max_size must be >= 1"):
                PolicyClass(max_size=-1)

    def test_policy_evict_from_empty(self):
        """Evicting from empty policy raises RuntimeError."""
        for PolicyClass in [LRUPolicy, LFUPolicy, FIFOPolicy, ARCPolicy]:
            policy = PolicyClass(max_size=3)

            with pytest.raises(RuntimeError, match="Cannot evict from empty"):
                policy.evict()


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestIntegrationScenarios:
    """Tests for realistic usage scenarios."""

    def test_typical_usage_pattern(self, test_checkpoint_file: str, n_experts: int):
        """Simulates typical MoE inference pattern with expert reuse."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=3,
            eviction_policy="lru",
            device="cpu",
        )

        # Simulate token routing pattern where some experts are more popular
        access_pattern = [0, 1, 0, 2, 0, 1, 3, 0, 1, 4, 0, 5]

        for expert_id in access_pattern:
            if expert_id < n_experts:
                lazy.get_expert(expert_id)

        stats = lazy.stats()

        # Should have hits due to reuse
        assert stats["hit_count"] > 0
        # Should have evictions due to limited cache
        assert stats["evict_count"] > 0
        # Popular expert 0 should still be cached
        assert 0 in lazy

    def test_prefetch_then_use(self, test_checkpoint_file: str, n_experts: int):
        """Prefetching improves hit rate."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
            device="cpu",
        )

        # Prefetch expected experts
        lazy.prefetch([0, 1, 2, 3])

        # Prefetch loads but doesn't increment miss count
        prefetch_stats = lazy.stats()
        assert prefetch_stats["load_count"] == 4
        assert prefetch_stats["miss_count"] == 0  # prefetch doesn't count as misses
        assert prefetch_stats["hit_count"] == 0

        # Access prefetched experts - should all be hits
        for i in range(4):
            lazy.get_expert(i)

        stats = lazy.stats()
        # All accesses after prefetch should be hits
        assert stats["hit_count"] == 4
        assert stats["miss_count"] == 0  # No misses since we prefetched

    def test_dynamic_cache_resize(self, test_checkpoint_file: str, n_experts: int):
        """Dynamic cache resizing works correctly."""
        lazy = LazyExpertWeights(
            checkpoint_path=test_checkpoint_file,
            n_experts=n_experts,
            max_loaded=4,
            device="cpu",
        )

        # Load 4 experts
        for i in range(4):
            lazy.get_expert(i)

        assert len(lazy) == 4

        # Reduce cache under memory pressure
        lazy.set_max_loaded(2)
        assert len(lazy) == 2

        # Increase cache when memory available
        lazy.set_max_loaded(4)
        for i in range(4):
            lazy.get_expert(i)

        assert len(lazy) == 4
