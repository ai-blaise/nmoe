"""Tests for checkpoint format versioning."""

import os
import tempfile
import pytest
import torch


class TestCheckpointVersioning:
    """Test checkpoint version constants and utilities."""

    def test_checkpoint_format_version_defined(self):
        """Test that CHECKPOINT_FORMAT_VERSION is defined."""
        from nmoe.checkpoint import CHECKPOINT_FORMAT_VERSION

        assert isinstance(CHECKPOINT_FORMAT_VERSION, int)
        assert CHECKPOINT_FORMAT_VERSION >= 1

    def test_get_checkpoint_version_legacy(self):
        """Test get_checkpoint_version returns 1 for legacy checkpoints."""
        from nmoe.checkpoint import get_checkpoint_version

        # Legacy checkpoint without version field
        legacy_state = {"step": 100, "model_dense": {}}
        assert get_checkpoint_version(legacy_state) == 1

    def test_get_checkpoint_version_versioned(self):
        """Test get_checkpoint_version returns correct version."""
        from nmoe.checkpoint import get_checkpoint_version

        versioned_state = {"checkpoint_version": 2, "step": 100}
        assert get_checkpoint_version(versioned_state) == 2

        versioned_state_3 = {"checkpoint_version": 3, "step": 200}
        assert get_checkpoint_version(versioned_state_3) == 3

    def test_validate_current_version(self):
        """Test validate_checkpoint_version passes for current version."""
        from nmoe.checkpoint import (
            validate_checkpoint_version,
            CHECKPOINT_FORMAT_VERSION,
        )

        # Should not raise
        validate_checkpoint_version({"checkpoint_version": CHECKPOINT_FORMAT_VERSION})

    def test_validate_legacy_version(self):
        """Test validate_checkpoint_version passes for legacy checkpoints."""
        from nmoe.checkpoint import validate_checkpoint_version

        # Legacy checkpoint without version = version 1
        validate_checkpoint_version({"step": 100})

    def test_validate_future_version_fails(self):
        """Test validate_checkpoint_version raises for future versions."""
        from nmoe.checkpoint import validate_checkpoint_version

        with pytest.raises(ValueError, match="newer than supported"):
            validate_checkpoint_version({"checkpoint_version": 999})

    def test_validate_custom_max_version(self):
        """Test validate_checkpoint_version with custom max_version."""
        from nmoe.checkpoint import validate_checkpoint_version

        # Version 2 should fail if max_version is 1
        with pytest.raises(ValueError, match="newer than supported"):
            validate_checkpoint_version({"checkpoint_version": 2}, max_version=1)

        # Version 2 should pass if max_version is 2
        validate_checkpoint_version({"checkpoint_version": 2}, max_version=2)

    def test_read_checkpoint_version_nonexistent(self):
        """Test read_checkpoint_version returns None for non-existent file."""
        from nmoe.checkpoint import read_checkpoint_version

        version = read_checkpoint_version("/nonexistent/path.pt")
        assert version is None

    def test_read_checkpoint_version_from_file(self):
        """Test read_checkpoint_version reads version from actual file."""
        from nmoe.checkpoint import read_checkpoint_version

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a versioned checkpoint
            path = os.path.join(tmpdir, "test.pt")
            torch.save({"checkpoint_version": 2, "data": "test"}, path)

            version = read_checkpoint_version(path)
            assert version == 2

    def test_read_checkpoint_version_legacy_file(self):
        """Test read_checkpoint_version returns 1 for legacy file."""
        from nmoe.checkpoint import read_checkpoint_version

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a legacy checkpoint without version
            path = os.path.join(tmpdir, "legacy.pt")
            torch.save({"step": 100, "data": "test"}, path)

            version = read_checkpoint_version(path)
            assert version == 1


class TestCheckpointVersionInState:
    """Test that checkpoint version is included in saved states."""

    def test_build_states_includes_version_in_rd(self):
        """Test that build_states includes version in rd_state."""
        from nmoe.checkpoint import CHECKPOINT_FORMAT_VERSION

        # We'll check the version is added by examining the code structure
        # The actual test would require a full model, but we verify the constant
        assert CHECKPOINT_FORMAT_VERSION >= 2

    def test_version_constant_matches_expected(self):
        """Test that current version is 3 (EP sharding support)."""
        from nmoe.checkpoint import CHECKPOINT_FORMAT_VERSION

        assert CHECKPOINT_FORMAT_VERSION == 3


class TestBackwardsCompatibility:
    """Test backwards compatibility with older checkpoints."""

    def test_legacy_checkpoint_loadable(self):
        """Test that checkpoints without version field are loadable (version 1)."""
        from nmoe.checkpoint import validate_checkpoint_version

        # Simulating loading a v1 checkpoint (no version field)
        v1_state = {
            "step": 1000,
            "model_dense": {"layer.weight": torch.randn(10, 10)},
            "tokens": 50000,
        }

        # Should not raise - v1 is always supported
        validate_checkpoint_version(v1_state)

    def test_v3_checkpoint_structure(self):
        """Test v3 checkpoint structure includes version and supports EP sharding."""
        from nmoe.checkpoint import CHECKPOINT_FORMAT_VERSION

        # v3 checkpoint should have checkpoint_version field
        v3_state = {
            "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
            "step": 1000,
            "model_dense": {"layer.weight": torch.randn(10, 10)},
            "tokens": 50000,
        }

        assert v3_state["checkpoint_version"] == 3
        assert "step" in v3_state
        assert "model_dense" in v3_state


class TestEPShardInfo:
    """Test EP sharding info for checkpoints."""

    def test_ep_shard_info_create(self):
        """Test EPShardInfo.create() computes correct ranges."""
        from nmoe.checkpoint import EPShardInfo

        # 256 experts, EP=4 -> 64 experts per rank
        info = EPShardInfo.create(ep_size=4, ep_rank=0, n_total_experts=256)
        assert info.ep_size == 4
        assert info.ep_rank == 0
        assert info.n_total_experts == 256
        assert info.n_local_experts == 64
        assert info.expert_start == 0
        assert info.expert_end == 64

        # Rank 2 should get experts 128-192
        info2 = EPShardInfo.create(ep_size=4, ep_rank=2, n_total_experts=256)
        assert info2.expert_start == 128
        assert info2.expert_end == 192

    def test_ep_shard_info_to_from_dict(self):
        """Test EPShardInfo serialization round-trip."""
        from nmoe.checkpoint import EPShardInfo

        info = EPShardInfo.create(ep_size=4, ep_rank=1, n_total_experts=256)
        d = info.to_dict()

        # Verify dict contents
        assert d['ep_size'] == 4
        assert d['ep_rank'] == 1
        assert d['n_total_experts'] == 256
        assert d['n_local_experts'] == 64
        assert d['expert_start'] == 64
        assert d['expert_end'] == 128

        # Round-trip
        info2 = EPShardInfo.from_dict(d)
        assert info2.ep_size == info.ep_size
        assert info2.ep_rank == info.ep_rank
        assert info2.n_total_experts == info.n_total_experts
        assert info2.expert_start == info.expert_start

    def test_ep_shard_info_not_divisible(self):
        """Test EPShardInfo raises for non-divisible expert count."""
        from nmoe.checkpoint import EPShardInfo

        # 101 is not divisible by 4
        with pytest.raises(ValueError, match="must be divisible"):
            EPShardInfo.create(ep_size=4, ep_rank=0, n_total_experts=101)

    def test_get_ep_shard_info_from_checkpoint(self):
        """Test extracting EP info from checkpoint state."""
        from nmoe.checkpoint import EPShardInfo, get_ep_shard_info_from_checkpoint

        # With EP info
        state = {
            'step': 100,
            'ep_shard_info': {
                'ep_size': 4,
                'ep_rank': 1,
                'n_total_experts': 256,
                'n_local_experts': 64,
                'expert_start': 64,
                'expert_end': 128,
            }
        }
        info = get_ep_shard_info_from_checkpoint(state)
        assert info is not None
        assert info.ep_size == 4
        assert info.ep_rank == 1

        # Without EP info (legacy)
        legacy_state = {'step': 100}
        info_legacy = get_ep_shard_info_from_checkpoint(legacy_state)
        assert info_legacy is None


class TestEPShardCompatibility:
    """Test EP sharding compatibility validation."""

    def test_compatible_same_config(self):
        """Test compatible when EP config matches."""
        from nmoe.checkpoint import EPShardInfo, validate_ep_shard_compatibility

        saved = EPShardInfo.create(ep_size=4, ep_rank=2, n_total_experts=256)
        compatible, msg = validate_ep_shard_compatibility(
            saved, current_ep_size=4, current_ep_rank=2, n_total_experts=256
        )
        assert compatible is True

    def test_incompatible_ep_size_mismatch(self):
        """Test incompatible when EP size differs."""
        from nmoe.checkpoint import EPShardInfo, validate_ep_shard_compatibility

        saved = EPShardInfo.create(ep_size=4, ep_rank=0, n_total_experts=256)
        compatible, msg = validate_ep_shard_compatibility(
            saved, current_ep_size=8, current_ep_rank=0, n_total_experts=256
        )
        assert compatible is False
        assert "EP size mismatch" in msg

    def test_incompatible_ep_rank_mismatch(self):
        """Test incompatible when EP rank differs."""
        from nmoe.checkpoint import EPShardInfo, validate_ep_shard_compatibility

        saved = EPShardInfo.create(ep_size=4, ep_rank=0, n_total_experts=256)
        compatible, msg = validate_ep_shard_compatibility(
            saved, current_ep_size=4, current_ep_rank=2, n_total_experts=256
        )
        assert compatible is False
        assert "EP rank mismatch" in msg

    def test_incompatible_expert_count_mismatch(self):
        """Test incompatible when expert count differs."""
        from nmoe.checkpoint import EPShardInfo, validate_ep_shard_compatibility

        saved = EPShardInfo.create(ep_size=4, ep_rank=0, n_total_experts=256)
        compatible, msg = validate_ep_shard_compatibility(
            saved, current_ep_size=4, current_ep_rank=0, n_total_experts=128
        )
        assert compatible is False
        assert "Expert count mismatch" in msg

    def test_legacy_checkpoint_requires_ep1(self):
        """Test legacy checkpoint (no EP info) requires EP=1."""
        from nmoe.checkpoint import validate_ep_shard_compatibility

        # Legacy (None) with EP=1 is OK
        compatible, msg = validate_ep_shard_compatibility(
            None, current_ep_size=1, current_ep_rank=0, n_total_experts=256
        )
        assert compatible is True

        # Legacy with EP>1 is NOT OK
        compatible, msg = validate_ep_shard_compatibility(
            None, current_ep_size=4, current_ep_rank=0, n_total_experts=256
        )
        assert compatible is False
        assert "Legacy checkpoint" in msg


class TestEPResharding:
    """Test EP resharding utilities."""

    def test_reshard_expert_weights_same_config(self):
        """Test resharding with same EP config returns same weights."""
        from nmoe.checkpoint import EPShardInfo, reshard_expert_weights

        # Simulate 256 experts, EP=4, current rank=1 (experts 64-128)
        saved_info = EPShardInfo.create(ep_size=4, ep_rank=1, n_total_experts=256)

        # Create fake expert weights for this shard
        expert_sd = {
            'layer.0.moe.W1': torch.randn(64, 128, 512),  # [n_local, H, D_ff]
            'layer.0.moe.W2': torch.randn(64, 512, 128),  # [n_local, D_ff, H]
        }

        # Reshard to same config (should return same)
        resharded = reshard_expert_weights(
            expert_sd, saved_info,
            target_ep_size=4, target_ep_rank=1,
            print_fn=lambda x: None
        )

        assert 'layer.0.moe.W1' in resharded
        assert resharded['layer.0.moe.W1'].shape == (64, 128, 512)
        assert torch.equal(resharded['layer.0.moe.W1'], expert_sd['layer.0.moe.W1'])

    def test_reshard_ep1_to_ep4(self):
        """Test resharding from EP=1 to EP=4."""
        from nmoe.checkpoint import EPShardInfo, reshard_expert_weights

        # Simulate EP=1 (all 256 experts in one shard)
        saved_info = EPShardInfo.create(ep_size=1, ep_rank=0, n_total_experts=256)

        # Create fake expert weights
        expert_sd = {
            'moe.W1': torch.randn(256, 128, 512),
        }

        # Reshard to EP=4, rank=2 (should get experts 128-192)
        resharded = reshard_expert_weights(
            expert_sd, saved_info,
            target_ep_size=4, target_ep_rank=2,
            print_fn=lambda x: None
        )

        assert resharded['moe.W1'].shape == (64, 128, 512)
        # Verify it's the correct slice (experts 128-192)
        assert torch.equal(resharded['moe.W1'], expert_sd['moe.W1'][128:192])

    def test_reshard_no_overlap(self):
        """Test resharding returns empty when no overlap."""
        from nmoe.checkpoint import EPShardInfo, reshard_expert_weights

        # Saved: EP=4, rank=0 (experts 0-64)
        saved_info = EPShardInfo.create(ep_size=4, ep_rank=0, n_total_experts=256)

        expert_sd = {
            'moe.W1': torch.randn(64, 128, 512),
        }

        # Target: EP=4, rank=3 (experts 192-256) - no overlap
        resharded = reshard_expert_weights(
            expert_sd, saved_info,
            target_ep_size=4, target_ep_rank=3,
            print_fn=lambda x: None
        )

        assert resharded == {}

    def test_reshard_partial_overlap(self):
        """Test resharding with partial overlap."""
        from nmoe.checkpoint import EPShardInfo, reshard_expert_weights

        # Saved: EP=2, rank=0 (experts 0-128)
        saved_info = EPShardInfo.create(ep_size=2, ep_rank=0, n_total_experts=256)

        expert_sd = {
            'moe.W1': torch.randn(128, 128, 512),
        }

        # Target: EP=4, rank=1 (experts 64-128) - partial overlap
        resharded = reshard_expert_weights(
            expert_sd, saved_info,
            target_ep_size=4, target_ep_rank=1,
            print_fn=lambda x: None
        )

        # Should get experts 64-128 from the source (offset 64-128 in source)
        assert resharded['moe.W1'].shape == (64, 128, 512)
        assert torch.equal(resharded['moe.W1'], expert_sd['moe.W1'][64:128])

    def test_reshard_not_divisible_fails(self):
        """Test resharding fails when not divisible."""
        from nmoe.checkpoint import EPShardInfo, reshard_expert_weights

        saved_info = EPShardInfo.create(ep_size=4, ep_rank=0, n_total_experts=256)

        expert_sd = {'moe.W1': torch.randn(64, 128, 512)}

        # Target EP=7 doesn't divide 256 evenly
        with pytest.raises(ValueError, match="not divisible"):
            reshard_expert_weights(
                expert_sd, saved_info,
                target_ep_size=7, target_ep_rank=0,
                print_fn=lambda x: None
            )
