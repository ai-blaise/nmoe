"""Comprehensive unit tests for DeterministicLoader and build_loader.

Tests cover:
- DeterministicLoader class with SWRR logic
- Stage transitions between data sources
- Cursor state management
- state_dict() / load_state_dict() for checkpointing
- build_loader() function paths
- Multi-rank scenarios with determinism
- State persistence and resume
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nmoe.data.loader import (
    DeterministicLoader,
    _tokens_available,
    _preflight_check,
    build_loader,
)
from nmoe.data.mixture import MixturePlan, StagePlan, SourcePlan


# =============================================================================
# Fixtures for creating mock data
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock .npy shard files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def single_shard(temp_data_dir):
    """Create a single shard with sequential tokens."""
    shard_path = temp_data_dir / "shard_00000.npy"
    # 10000 tokens should be enough for most tests
    tokens = np.arange(10000, dtype=np.uint32)
    np.save(shard_path, tokens)
    return str(shard_path)


@pytest.fixture
def multiple_shards(temp_data_dir):
    """Create multiple shards with different token ranges."""
    shard_paths = []
    for i in range(3):
        shard_path = temp_data_dir / f"shard_{i:05d}.npy"
        # Each shard has 5000 tokens with distinct ranges
        start = i * 5000
        tokens = np.arange(start, start + 5000, dtype=np.uint32)
        np.save(shard_path, tokens)
        shard_paths.append(str(shard_path))
    return shard_paths


@pytest.fixture
def small_shards(temp_data_dir):
    """Create small shards for testing cross-shard window reads."""
    shard_paths = []
    for i in range(5):
        shard_path = temp_data_dir / f"small_{i:05d}.npy"
        # Each shard has only 100 tokens
        start = i * 100
        tokens = np.arange(start, start + 100, dtype=np.uint32)
        np.save(shard_path, tokens)
        shard_paths.append(str(shard_path))
    return shard_paths


def create_simple_plan(
    paths: List[str],
    seq_len: int = 128,
    quota_sequences: int = 100,
    source_id: str = "test_source",
) -> MixturePlan:
    """Helper to create a simple single-source MixturePlan."""
    source = SourcePlan(
        id=source_id,
        weight_fp=1000000,  # 1.0 in fixed-point
        quota_sequences=quota_sequences,
        target_tokens=quota_sequences * seq_len,
        paths=paths,
    )
    stage = StagePlan(
        stage_id="pretrain",
        total_tokens_b=quota_sequences * seq_len / 1_000_000_000,
        sources=[source],
    )
    return MixturePlan(
        plan_id="test_plan",
        plan_hash="abc123",
        mixture_id="test_mixture",
        flow_mode="test",
        sample_temperature=1.0,
        seq_len=seq_len,
        stages=[stage],
    )


def create_multi_source_plan(
    source_configs: List[Dict],
    seq_len: int = 128,
) -> MixturePlan:
    """Helper to create a multi-source MixturePlan for SWRR testing."""
    sources = []
    for cfg in source_configs:
        source = SourcePlan(
            id=cfg["id"],
            weight_fp=cfg["weight_fp"],
            quota_sequences=cfg["quota_sequences"],
            target_tokens=cfg["quota_sequences"] * seq_len,
            paths=cfg["paths"],
        )
        sources.append(source)

    stage = StagePlan(
        stage_id="pretrain",
        total_tokens_b=sum(s.quota_sequences for s in sources) * seq_len / 1_000_000_000,
        sources=sources,
    )
    return MixturePlan(
        plan_id="test_plan",
        plan_hash="multi123",
        mixture_id="test_mixture",
        flow_mode="test",
        sample_temperature=1.0,
        seq_len=seq_len,
        stages=[stage],
    )


def create_multi_stage_plan(
    stage_configs: List[Dict],
    seq_len: int = 128,
) -> MixturePlan:
    """Helper to create a multi-stage MixturePlan for stage transition testing."""
    stages = []
    for stage_cfg in stage_configs:
        sources = []
        for src_cfg in stage_cfg["sources"]:
            source = SourcePlan(
                id=src_cfg["id"],
                weight_fp=src_cfg["weight_fp"],
                quota_sequences=src_cfg["quota_sequences"],
                target_tokens=src_cfg["quota_sequences"] * seq_len,
                paths=src_cfg["paths"],
            )
            sources.append(source)

        stage = StagePlan(
            stage_id=stage_cfg["stage_id"],
            total_tokens_b=sum(s.quota_sequences for s in sources) * seq_len / 1_000_000_000,
            sources=sources,
        )
        stages.append(stage)

    return MixturePlan(
        plan_id="test_plan",
        plan_hash="multistage123",
        mixture_id="test_mixture",
        flow_mode="test",
        sample_temperature=1.0,
        seq_len=seq_len,
        stages=stages,
    )


# =============================================================================
# DeterministicLoader Basic Tests
# =============================================================================


class TestDeterministicLoaderCreation:
    """Tests for DeterministicLoader instantiation."""

    def test_loader_creation_single_source(self, single_shard):
        """Loader can be created with a single source."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=256,  # 4 sequences per step
            device="cpu",
            prefetch_depth=0,
        )

        assert loader.dp_rank == 0
        assert loader.dp_world_size == 1
        assert loader.seq_len == 64
        assert loader.tpu == 256
        loader.close()

    def test_loader_creation_multi_rank(self, single_shard):
        """Loader can be created for multiple ranks."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        loaders = []
        for rank in range(4):
            loader = DeterministicLoader(
                plan=plan,
                dp_world_size=4,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=256,
                device="cpu",
                prefetch_depth=0,
            )
            loaders.append(loader)

        for i, loader in enumerate(loaders):
            assert loader.dp_rank == i
            assert loader.dp_world_size == 4
            loader.close()

    def test_loader_internal_state_initialized(self, single_shard):
        """Loader initializes internal state correctly."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=256,
            device="cpu",
            prefetch_depth=0,
        )

        assert loader._stage_index == 0
        assert loader._global_seq_idx == 0
        assert len(loader._src_state) == 1
        assert "test_source" in loader._src_state
        loader.close()


# =============================================================================
# SWRR (Smooth Weighted Round Robin) Tests
# =============================================================================


class TestSWRRLogic:
    """Tests for Smooth Weighted Round Robin source selection."""

    def test_swrr_single_source_always_selects_same(self, single_shard):
        """With single source, SWRR always selects that source."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,  # 1 sequence per step
            device="cpu",
            prefetch_depth=0,
        )

        # All selections should be index 0
        for _ in range(10):
            idx = loader._swr_next_source_idx()
            assert idx == 0

        loader.close()

    def test_swrr_equal_weights_alternates(self, temp_data_dir):
        """With equal weights, SWRR should alternate evenly."""
        # Create two sources with same weights
        shard1 = temp_data_dir / "source1_shard.npy"
        shard2 = temp_data_dir / "source2_shard.npy"
        np.save(shard1, np.arange(5000, dtype=np.uint32))
        np.save(shard2, np.arange(5000, 10000, dtype=np.uint32))

        plan = create_multi_source_plan(
            [
                {"id": "source1", "weight_fp": 500000, "quota_sequences": 100, "paths": [str(shard1)]},
                {"id": "source2", "weight_fp": 500000, "quota_sequences": 100, "paths": [str(shard2)]},
            ],
            seq_len=64,
        )

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Count selections over many iterations
        counts = {0: 0, 1: 0}
        for _ in range(100):
            idx = loader._swr_next_source_idx()
            counts[idx] += 1

        # Should be roughly equal (within 10% difference)
        assert abs(counts[0] - counts[1]) <= 10

        loader.close()

    def test_swrr_unequal_weights_proportional(self, temp_data_dir):
        """With unequal weights, SWRR selects proportionally."""
        shard1 = temp_data_dir / "source1_shard.npy"
        shard2 = temp_data_dir / "source2_shard.npy"
        np.save(shard1, np.arange(10000, dtype=np.uint32))
        np.save(shard2, np.arange(10000, 20000, dtype=np.uint32))

        # source1 has 3x weight of source2
        plan = create_multi_source_plan(
            [
                {"id": "source1", "weight_fp": 750000, "quota_sequences": 300, "paths": [str(shard1)]},
                {"id": "source2", "weight_fp": 250000, "quota_sequences": 100, "paths": [str(shard2)]},
            ],
            seq_len=64,
        )

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        counts = {0: 0, 1: 0}
        for _ in range(400):
            idx = loader._swr_next_source_idx()
            counts[idx] += 1

        # source1 should be selected ~3x as often as source2
        ratio = counts[0] / max(1, counts[1])
        assert 2.5 < ratio < 3.5

        loader.close()

    def test_swrr_skips_exhausted_sources(self, temp_data_dir):
        """SWRR skips sources that have reached their quota."""
        shard1 = temp_data_dir / "source1_shard.npy"
        shard2 = temp_data_dir / "source2_shard.npy"
        np.save(shard1, np.arange(5000, dtype=np.uint32))
        np.save(shard2, np.arange(5000, 10000, dtype=np.uint32))

        # source1 has very low quota
        plan = create_multi_source_plan(
            [
                {"id": "source1", "weight_fp": 500000, "quota_sequences": 2, "paths": [str(shard1)]},
                {"id": "source2", "weight_fp": 500000, "quota_sequences": 100, "paths": [str(shard2)]},
            ],
            seq_len=64,
        )

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Exhaust source1's quota
        loader._src_state["source1"].emitted_sequences = 2

        # Now all selections should be source2 (index 1)
        for _ in range(10):
            idx = loader._select_next_non_exhausted()
            assert idx == 1

        loader.close()


# =============================================================================
# Stage Transition Tests
# =============================================================================


class TestStageTransitions:
    """Tests for stage transition logic."""

    def test_stage_transition_when_sources_exhausted(self, temp_data_dir):
        """Loader transitions to next stage when all sources are exhausted."""
        shard_pretrain = temp_data_dir / "pretrain_shard.npy"
        shard_mid = temp_data_dir / "mid_shard.npy"
        np.save(shard_pretrain, np.arange(1000, dtype=np.uint32))
        np.save(shard_mid, np.arange(1000, 2000, dtype=np.uint32))

        plan = create_multi_stage_plan(
            [
                {
                    "stage_id": "pretrain",
                    "sources": [
                        {"id": "pretrain_src", "weight_fp": 1000000, "quota_sequences": 5, "paths": [str(shard_pretrain)]},
                    ],
                },
                {
                    "stage_id": "mid",
                    "sources": [
                        {"id": "mid_src", "weight_fp": 1000000, "quota_sequences": 10, "paths": [str(shard_mid)]},
                    ],
                },
            ],
            seq_len=64,
        )

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        assert loader._stage_index == 0

        # Exhaust pretrain stage
        loader._src_state["pretrain_src"].emitted_sequences = 5

        # Trigger stage transition check
        loader._advance_stage_if_done()

        assert loader._stage_index == 1
        assert "mid_src" in loader._src_state

        loader.close()

    def test_stage_transition_preserves_global_seq_idx(self, temp_data_dir):
        """Stage transition preserves global sequence index."""
        shard_pretrain = temp_data_dir / "pretrain_shard.npy"
        shard_mid = temp_data_dir / "mid_shard.npy"
        np.save(shard_pretrain, np.arange(1000, dtype=np.uint32))
        np.save(shard_mid, np.arange(1000, 2000, dtype=np.uint32))

        plan = create_multi_stage_plan(
            [
                {
                    "stage_id": "pretrain",
                    "sources": [
                        {"id": "pretrain_src", "weight_fp": 1000000, "quota_sequences": 3, "paths": [str(shard_pretrain)]},
                    ],
                },
                {
                    "stage_id": "mid",
                    "sources": [
                        {"id": "mid_src", "weight_fp": 1000000, "quota_sequences": 5, "paths": [str(shard_mid)]},
                    ],
                },
            ],
            seq_len=64,
        )

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Emit all sequences from pretrain (quota is 3)
        for _ in range(3):
            loader._emit_one_sequence()

        assert loader._global_seq_idx == 3
        # Stage should still be 0 since _emit_one_sequence doesn't auto-transition
        # The transition happens on the NEXT emit when it checks stage completion
        loader._emit_one_sequence()  # This triggers the transition check
        assert loader._stage_index == 1  # Now should have transitioned
        assert loader._global_seq_idx == 4

        loader.close()

    def test_no_transition_when_stage_not_done(self, single_shard):
        """No stage transition when current stage is not exhausted."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Emit some sequences but not all
        for _ in range(5):
            loader._emit_one_sequence()

        loader._advance_stage_if_done()
        assert loader._stage_index == 0

        loader.close()


# =============================================================================
# Cursor State Management Tests
# =============================================================================


class TestCursorStateManagement:
    """Tests for cursor state tracking."""

    def test_cursor_advances_correctly(self, single_shard):
        """Cursor position advances correctly after emitting sequences."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        initial_pos = loader._src_state["test_source"].cursor.pos_in_file

        # Emit one sequence
        loader._emit_one_sequence()

        new_pos = loader._src_state["test_source"].cursor.pos_in_file
        # Position should advance by seq_len + 1 (for targets)
        assert new_pos == initial_pos + 65

        loader.close()

    def test_cursor_wraps_at_end_of_data(self, temp_data_dir):
        """Cursor wraps around when reaching end of data."""
        small_shard = temp_data_dir / "small_shard.npy"
        # Create a small shard with only 100 tokens
        np.save(small_shard, np.arange(100, dtype=np.uint32))

        # Each sequence needs 65 tokens (seq_len + 1 for targets)
        # With only 100 tokens, we need 2 sequences to force a wrap:
        # - 1st seq: tokens 0-64, cursor at 65
        # - 2nd seq: needs tokens 65-129, but only 100 tokens, so must wrap
        plan = create_simple_plan([str(small_shard)], seq_len=64, quota_sequences=5)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Emit 2 sequences to force a wrap (100 tokens < 2 * 65 = 130 tokens needed)
        for _ in range(2):
            loader._emit_one_sequence()

        cursor = loader._src_state["test_source"].cursor
        assert cursor.wrap_count >= 1

        loader.close()

    def test_cursor_crosses_shard_boundary(self, multiple_shards):
        """Cursor correctly crosses shard boundaries."""
        plan = create_simple_plan(multiple_shards, seq_len=64, quota_sequences=200)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # First shard has 5000 tokens, emit sequences until we cross to shard 2
        sequences_to_exhaust_first_shard = 5000 // 65  # ~76
        for _ in range(80):
            loader._emit_one_sequence()

        cursor = loader._src_state["test_source"].cursor
        assert cursor.file_idx >= 1

        loader.close()


# =============================================================================
# state_dict / load_state_dict Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state serialization and restoration."""

    def test_state_dict_contains_required_keys(self, single_shard):
        """state_dict contains all required keys."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        state = loader.state_dict()

        assert "version" in state
        assert "current_stage" in state
        assert "stage_index" in state
        assert "global_sequence_index" in state
        assert "accumulators" in state
        assert "emitted_sequences" in state
        assert "cursors" in state

        loader.close()

    def test_state_dict_captures_progress(self, single_shard):
        """state_dict captures emission progress."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Emit some sequences
        for _ in range(5):
            loader._emit_one_sequence()

        state = loader.state_dict()

        assert state["global_sequence_index"] == 5
        assert state["emitted_sequences"]["test_source"] == 5

        loader.close()

    def test_load_state_dict_restores_position(self, single_shard):
        """load_state_dict restores exact position."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader1 = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Emit some sequences and save state
        for _ in range(10):
            loader1._emit_one_sequence()

        state = loader1.state_dict()
        loader1.close()

        # Create new loader and restore state
        loader2 = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )
        loader2.load_state_dict(state)

        assert loader2._global_seq_idx == 10
        assert loader2._src_state["test_source"].emitted_sequences == 10
        assert loader2._src_state["test_source"].cursor.pos_in_file == 10 * 65

        loader2.close()

    def test_state_dict_is_json_serializable(self, single_shard):
        """state_dict can be serialized to JSON."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        for _ in range(5):
            loader._emit_one_sequence()

        state = loader.state_dict()

        # Should not raise
        json_str = json.dumps(state)
        restored = json.loads(json_str)

        assert restored == state

        loader.close()

    def test_resume_produces_same_sequence(self, single_shard):
        """Resuming from state produces same sequence as continuous run."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        # Run 1: continuous sequence
        loader1 = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )
        all_sequences = []
        for _ in range(20):
            seq = loader1._emit_one_sequence()
            all_sequences.append(seq.numpy().copy())
        loader1.close()

        # Run 2: save after 10, resume, continue
        loader2 = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )
        for _ in range(10):
            loader2._emit_one_sequence()
        state = loader2.state_dict()
        loader2.close()

        loader3 = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )
        loader3.load_state_dict(state)
        resumed_sequences = []
        for _ in range(10):
            seq = loader3._emit_one_sequence()
            resumed_sequences.append(seq.numpy().copy())
        loader3.close()

        # Sequences 10-19 should match
        for i in range(10):
            np.testing.assert_array_equal(all_sequences[10 + i], resumed_sequences[i])


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_loader_determinism_same_rank(self, single_shard):
        """Same config produces identical sequences."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader1 = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )
        batch1 = [loader1._emit_one_sequence().numpy().copy() for _ in range(10)]
        loader1.close()

        loader2 = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )
        batch2 = [loader2._emit_one_sequence().numpy().copy() for _ in range(10)]
        loader2.close()

        for i in range(10):
            np.testing.assert_array_equal(batch1[i], batch2[i])

    def test_determinism_across_restarts(self, single_shard):
        """Loader produces same data across process restarts (simulated)."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        sequences = []
        for run in range(3):
            loader = DeterministicLoader(
                plan=plan,
                dp_world_size=1,
                dp_rank=0,
                seq_len=64,
                tokens_per_update=64,
                device="cpu",
                prefetch_depth=0,
            )
            run_sequences = [loader._emit_one_sequence().numpy().copy() for _ in range(5)]
            sequences.append(run_sequences)
            loader.close()

        # All runs should produce identical sequences
        for run in range(1, 3):
            for i in range(5):
                np.testing.assert_array_equal(sequences[0][i], sequences[run][i])


# =============================================================================
# Multi-Rank Tests
# =============================================================================


class TestMultiRank:
    """Tests for multi-rank scenarios."""

    def test_different_ranks_get_different_data(self, single_shard):
        """Different ranks receive different slices of data."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        rank_data = {}
        world_size = 4
        tokens_per_update = 256  # 4 sequences per step

        for rank in range(world_size):
            loader = DeterministicLoader(
                plan=plan,
                dp_world_size=world_size,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=tokens_per_update,
                device="cpu",
                prefetch_depth=0,
            )
            inputs, targets = loader.next()
            rank_data[rank] = inputs.numpy().copy()
            loader.close()

        # Each rank should have different data
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert not np.array_equal(rank_data[i], rank_data[j])

    def test_all_ranks_cover_all_data(self, single_shard):
        """Combined data from all ranks covers the global sequence."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        world_size = 4
        tokens_per_update = 256  # 4 sequences per step

        all_first_tokens = set()

        for rank in range(world_size):
            loader = DeterministicLoader(
                plan=plan,
                dp_world_size=world_size,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=tokens_per_update,
                device="cpu",
                prefetch_depth=0,
            )
            inputs, _ = loader.next()
            for seq in inputs:
                all_first_tokens.add(int(seq[0].item()))
            loader.close()

        # Should have 4 unique starting positions
        assert len(all_first_tokens) == 4

    def test_multi_rank_determinism(self, single_shard):
        """Each rank produces deterministic sequences across runs."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        world_size = 4
        tokens_per_update = 256

        for rank in range(world_size):
            loader1 = DeterministicLoader(
                plan=plan,
                dp_world_size=world_size,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=tokens_per_update,
                device="cpu",
                prefetch_depth=0,
            )
            inputs1, _ = loader1.next()
            loader1.close()

            loader2 = DeterministicLoader(
                plan=plan,
                dp_world_size=world_size,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=tokens_per_update,
                device="cpu",
                prefetch_depth=0,
            )
            inputs2, _ = loader2.next()
            loader2.close()

            np.testing.assert_array_equal(inputs1.numpy(), inputs2.numpy())

    def test_no_data_overlap_between_ranks(self, single_shard):
        """Verify no token sequence overlap between ranks in same step."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        world_size = 4
        tokens_per_update = 256

        sequences_by_rank = []

        for rank in range(world_size):
            loader = DeterministicLoader(
                plan=plan,
                dp_world_size=world_size,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=tokens_per_update,
                device="cpu",
                prefetch_depth=0,
            )
            inputs, _ = loader.next()
            sequences_by_rank.append([tuple(seq.tolist()) for seq in inputs])
            loader.close()

        # Check no overlap
        all_sequences = []
        for rank_seqs in sequences_by_rank:
            all_sequences.extend(rank_seqs)

        # Convert to set should preserve count if no duplicates
        assert len(all_sequences) == len(set(all_sequences))


# =============================================================================
# build_loader Tests
# =============================================================================


class TestBuildLoader:
    """Tests for build_loader function."""

    def test_build_loader_data_path_fast_path(self, temp_data_dir):
        """build_loader works with data_path fast path."""
        # Create shard files
        shard_dir = temp_data_dir / "shards"
        shard_dir.mkdir()
        for i in range(3):
            shard_path = shard_dir / f"shard_{i:05d}.npy"
            np.save(shard_path, np.arange(i * 1000, (i + 1) * 1000, dtype=np.uint32))

        @dataclass
        class MockConfig:
            data_path: str = str(shard_dir)
            flow_mode: Optional[str] = None
            batch_size: int = 4
            seq_len: int = 64
            steps: int = 10

        cfg = MockConfig()
        loader, plan = build_loader(cfg, rank=0, world_size=1, print_fn=lambda x: None)

        assert loader is not None
        assert plan is not None
        assert plan.plan_id == "data_path"
        assert len(plan.stages) == 1
        assert plan.stages[0].sources[0].id == "data_path"

        loader.close()

    def test_build_loader_no_shards_raises(self, temp_data_dir):
        """build_loader raises when no shards found."""
        empty_dir = temp_data_dir / "empty"
        empty_dir.mkdir()

        @dataclass
        class MockConfig:
            data_path: str = str(empty_dir)
            flow_mode: Optional[str] = None
            batch_size: int = 4
            seq_len: int = 64
            steps: int = 10

        cfg = MockConfig()

        with pytest.raises(RuntimeError, match="No .npy shards found"):
            build_loader(cfg, rank=0, world_size=1, print_fn=lambda x: None)

    def test_build_loader_flow_mode_requires_tomls(self, temp_data_dir):
        """build_loader with flow_mode requires mixture and flow profile TOMLs."""

        @dataclass
        class MockConfig:
            data_path: Optional[str] = None
            flow_mode: str = "dev"
            mixture_toml: Optional[str] = None
            flow_profiles_toml: Optional[str] = None
            batch_size: int = 4
            seq_len: int = 64
            steps: int = 10

        cfg = MockConfig()

        with pytest.raises(ValueError, match="mixture_toml/flow_profiles_toml are missing"):
            build_loader(cfg, rank=0, world_size=1, print_fn=lambda x: None)

    def test_build_loader_no_data_path_no_flow_mode_raises(self):
        """build_loader raises when neither data_path nor flow_mode set."""

        @dataclass
        class MockConfig:
            data_path: Optional[str] = None
            flow_mode: Optional[str] = None
            batch_size: int = 4
            seq_len: int = 64
            steps: int = 10

        cfg = MockConfig()

        with pytest.raises(ValueError, match="flow_mode is required"):
            build_loader(cfg, rank=0, world_size=1, print_fn=lambda x: None)


# =============================================================================
# _tokens_available Tests
# =============================================================================


class TestTokensAvailable:
    """Tests for _tokens_available function."""

    def test_tokens_available_npy_files(self, temp_data_dir):
        """_tokens_available correctly counts tokens in .npy files."""
        shard = temp_data_dir / "test.npy"
        tokens = np.arange(5000, dtype=np.uint32)
        np.save(shard, tokens)

        count = _tokens_available([str(shard)], print_fn=lambda x: None)
        assert count == 5000

    def test_tokens_available_multiple_files(self, multiple_shards):
        """_tokens_available sums tokens across multiple files."""
        count = _tokens_available(multiple_shards, print_fn=lambda x: None)
        assert count == 15000  # 3 shards * 5000 tokens each

    def test_tokens_available_missing_file(self, temp_data_dir):
        """_tokens_available handles missing files gracefully."""
        missing = str(temp_data_dir / "missing.npy")
        count = _tokens_available([missing], print_fn=lambda x: None)
        assert count == 0

    def test_tokens_available_binary_files(self, temp_data_dir):
        """_tokens_available handles binary files (non-.npy)."""
        binary_file = temp_data_dir / "tokens.bin"
        tokens = np.arange(1000, dtype=np.uint32)
        tokens.tofile(binary_file)

        count = _tokens_available([str(binary_file)], print_fn=lambda x: None)
        assert count == 1000  # 4000 bytes / 4 bytes per uint32


# =============================================================================
# _preflight_check Tests
# =============================================================================


class TestPreflightCheck:
    """Tests for _preflight_check function."""

    def test_preflight_check_passes_with_enough_tokens(self, temp_data_dir):
        """_preflight_check passes when enough tokens available."""
        shard = temp_data_dir / "test.npy"
        np.save(shard, np.arange(10000, dtype=np.uint32))

        plan = create_simple_plan([str(shard)], seq_len=64, quota_sequences=100)

        # Should not raise
        _preflight_check(plan, seq_len=64, print_fn=lambda x: None)

    def test_preflight_check_fails_with_insufficient_tokens(self, temp_data_dir):
        """_preflight_check fails when not enough tokens available."""
        shard = temp_data_dir / "small.npy"
        np.save(shard, np.arange(100, dtype=np.uint32))  # Only 100 tokens

        # Request many more tokens than available
        plan = create_simple_plan([str(shard)], seq_len=64, quota_sequences=1000)

        with pytest.raises(RuntimeError, match="Preflight failed"):
            _preflight_check(plan, seq_len=64, print_fn=lambda x: None)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_loader_handles_empty_dataset_paths(self):
        """Loader raises when source has no paths."""
        plan = create_simple_plan([], seq_len=64, quota_sequences=10)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        with pytest.raises(RuntimeError, match="dataset paths.*not set"):
            loader.next()

        loader.close()

    def test_loader_handles_seq_len_larger_than_data(self, temp_data_dir):
        """Loader handles seq_len larger than available tokens (wraps)."""
        small_shard = temp_data_dir / "tiny.npy"
        np.save(small_shard, np.arange(50, dtype=np.uint32))

        plan = create_simple_plan([str(small_shard)], seq_len=100, quota_sequences=5)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=100,
            tokens_per_update=101,
            device="cpu",
            prefetch_depth=0,
        )

        # Should work by wrapping
        seq = loader._emit_one_sequence()
        assert len(seq) == 101

        cursor = loader._src_state["test_source"].cursor
        assert cursor.wrap_count >= 2  # Had to wrap multiple times

        loader.close()

    def test_loader_next_returns_correct_shapes(self, single_shard):
        """loader.next() returns correctly shaped tensors."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=256,  # 4 sequences
            device="cpu",
            prefetch_depth=0,
        )

        inputs, targets = loader.next()

        assert inputs.shape == (4, 64)
        assert targets.shape == (4, 64)

        loader.close()

    def test_loader_close_is_idempotent(self, single_shard):
        """Calling close() multiple times is safe."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=10)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Multiple closes should not raise
        loader.close()
        loader.close()
        loader.close()

    def test_loader_handles_single_token_per_shard(self, temp_data_dir):
        """Loader handles edge case of very small shards."""
        shards = []
        for i in range(100):
            shard_path = temp_data_dir / f"tiny_{i:03d}.npy"
            np.save(shard_path, np.array([i], dtype=np.uint32))
            shards.append(str(shard_path))

        plan = create_simple_plan(shards, seq_len=10, quota_sequences=5)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=10,
            tokens_per_update=11,
            device="cpu",
            prefetch_depth=0,
        )

        # Should work by spanning many shards
        seq = loader._emit_one_sequence()
        assert len(seq) == 11

        loader.close()


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests combining multiple features."""

    def test_full_training_loop_simulation(self, single_shard):
        """Simulate a full training loop with checkpointing."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=50)

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=256,
            device="cpu",
            prefetch_depth=0,
        )

        # Simulate training steps with periodic checkpoints
        checkpoints = []
        all_batches = []

        for step in range(10):
            inputs, targets = loader.next()
            all_batches.append((inputs.numpy().copy(), targets.numpy().copy()))

            if step % 3 == 2:
                checkpoints.append((step, loader.state_dict()))

        loader.close()

        # Verify we can resume from any checkpoint
        for ckpt_step, state in checkpoints:
            loader2 = DeterministicLoader(
                plan=plan,
                dp_world_size=1,
                dp_rank=0,
                seq_len=64,
                tokens_per_update=256,
                device="cpu",
                prefetch_depth=0,
            )
            loader2.load_state_dict(state)

            # Next batch after checkpoint should match original
            inputs, targets = loader2.next()
            orig_inputs, orig_targets = all_batches[ckpt_step + 1]

            np.testing.assert_array_equal(inputs.numpy(), orig_inputs)
            np.testing.assert_array_equal(targets.numpy(), orig_targets)

            loader2.close()

    def test_multi_rank_with_checkpointing(self, single_shard):
        """Multi-rank training with synchronized checkpointing."""
        plan = create_simple_plan([single_shard], seq_len=64, quota_sequences=100)

        world_size = 4
        tokens_per_update = 256

        # Create loaders for all ranks
        loaders = []
        for rank in range(world_size):
            loader = DeterministicLoader(
                plan=plan,
                dp_world_size=world_size,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=tokens_per_update,
                device="cpu",
                prefetch_depth=0,
            )
            loaders.append(loader)

        # Run some steps
        for _ in range(5):
            for loader in loaders:
                loader.next()

        # Save states
        states = [loader.state_dict() for loader in loaders]
        for loader in loaders:
            loader.close()

        # Resume all ranks and verify they continue correctly
        new_loaders = []
        for rank, state in enumerate(states):
            loader = DeterministicLoader(
                plan=plan,
                dp_world_size=world_size,
                dp_rank=rank,
                seq_len=64,
                tokens_per_update=tokens_per_update,
                device="cpu",
                prefetch_depth=0,
            )
            loader.load_state_dict(state)
            new_loaders.append(loader)

        # All should be at same global sequence index
        global_indices = [loader._global_seq_idx for loader in new_loaders]
        assert len(set(global_indices)) == 1

        for loader in new_loaders:
            loader.close()

    def test_multi_source_with_stage_transitions(self, temp_data_dir):
        """Test complex scenario with multiple sources and stages."""
        # Create shards for each source in each stage
        paths = {}
        for stage in ["pretrain", "mid"]:
            for src in ["code", "text"]:
                shard = temp_data_dir / f"{stage}_{src}.npy"
                np.save(shard, np.arange(5000, dtype=np.uint32))
                paths[(stage, src)] = str(shard)

        plan = create_multi_stage_plan(
            [
                {
                    "stage_id": "pretrain",
                    "sources": [
                        {"id": "code", "weight_fp": 700000, "quota_sequences": 30, "paths": [paths[("pretrain", "code")]]},
                        {"id": "text", "weight_fp": 300000, "quota_sequences": 20, "paths": [paths[("pretrain", "text")]]},
                    ],
                },
                {
                    "stage_id": "mid",
                    "sources": [
                        {"id": "code", "weight_fp": 500000, "quota_sequences": 25, "paths": [paths[("mid", "code")]]},
                        {"id": "text", "weight_fp": 500000, "quota_sequences": 25, "paths": [paths[("mid", "text")]]},
                    ],
                },
            ],
            seq_len=64,
        )

        loader = DeterministicLoader(
            plan=plan,
            dp_world_size=1,
            dp_rank=0,
            seq_len=64,
            tokens_per_update=64,
            device="cpu",
            prefetch_depth=0,
        )

        # Run through pretrain stage
        stage_sequences = {"pretrain": 0, "mid": 0}
        current_stage = "pretrain"

        for _ in range(100):
            stage_before = plan.stages[loader._stage_index].stage_id
            try:
                loader.next()
            except StopIteration:
                break
            stage_after = plan.stages[loader._stage_index].stage_id
            stage_sequences[stage_after] += 1

            if stage_before != stage_after:
                current_stage = stage_after

        # Both stages should have been used
        assert stage_sequences["pretrain"] > 0
        assert stage_sequences["mid"] > 0

        loader.close()
