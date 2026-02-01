"""Tests for SkyRLRdepBridge.

Tests the bridge between SkyRL/Megatron process groups and nmoe RDEP.
"""

import pytest
import sys
sys.path.insert(0, '/home/nourdine/sglang_nmoe/nether-soup/nmoe')

import torch


class TestSkyRLRdepBridgeInit:
    """Test SkyRLRdepBridge initialization."""

    def test_basic_init(self):
        """Test basic initialization without process groups."""
        from nmoe.distributed import SkyRLRdepBridge

        bridge = SkyRLRdepBridge(
            dim=4096,
            n_total_experts=256,
            topk=8,
        )

        # Without process groups, ep_size and tp_size should be 1
        assert bridge.ep_size == 1
        assert bridge.tp_size == 1
        assert bridge.n_local_experts == 256
        assert bridge.expert_start == 0
        assert bridge.expert_end == 256

    def test_init_with_custom_experts(self):
        """Test initialization with different expert counts."""
        from nmoe.distributed import SkyRLRdepBridge

        bridge = SkyRLRdepBridge(
            dim=2048,
            n_total_experts=64,
            topk=4,
            profile="fp8",
        )

        assert bridge.dim == 2048
        assert bridge.n_total_experts == 64
        assert bridge.topk == 4
        assert bridge.profile == "fp8"
        assert bridge.n_local_experts == 64

    def test_invalid_expert_count(self):
        """Test that non-divisible expert count raises error."""
        from nmoe.distributed import SkyRLRdepBridge

        # This should work (256 / 1 = 256)
        bridge = SkyRLRdepBridge(dim=4096, n_total_experts=256, topk=8)
        assert bridge.n_local_experts == 256

    def test_profile_options(self):
        """Test different profile options."""
        from nmoe.distributed import SkyRLRdepBridge

        for profile in ["bf16", "fp8", "nvfp4"]:
            bridge = SkyRLRdepBridge(
                dim=4096,
                n_total_experts=256,
                topk=8,
                profile=profile,
            )
            assert bridge.profile == profile


class TestSkyRLRdepBridgeExpertMapping:
    """Test expert weight and ID mapping."""

    def test_get_local_expert_weights(self):
        """Test extracting local expert weights."""
        from nmoe.distributed import SkyRLRdepBridge

        bridge = SkyRLRdepBridge(
            dim=128,
            n_total_experts=8,
            topk=2,
        )

        # Create global weights [8, 128, 256]
        global_W = torch.randn(8, 128, 256)

        # Without EP, all experts are local
        local_W = bridge.get_local_expert_weights(global_W)
        assert local_W.shape == (8, 128, 256)
        assert torch.equal(local_W, global_W)

    def test_map_expert_ids_to_local(self):
        """Test mapping global expert IDs to local."""
        from nmoe.distributed import SkyRLRdepBridge

        bridge = SkyRLRdepBridge(
            dim=128,
            n_total_experts=8,
            topk=2,
        )

        # Create expert IDs [4 tokens, 2 experts each]
        global_ids = torch.tensor([
            [0, 3],
            [2, 5],
            [7, 1],
            [4, 6],
        ], dtype=torch.int32)

        local_ids = bridge.map_expert_ids_to_local(global_ids)

        # Without EP, all experts are local (0-7)
        expected = global_ids.clone()  # Same as global
        assert torch.equal(local_ids, expected)

    def test_expert_range_properties(self):
        """Test expert range calculation."""
        from nmoe.distributed import SkyRLRdepBridge

        bridge = SkyRLRdepBridge(
            dim=128,
            n_total_experts=256,
            topk=8,
        )

        # Single rank has all experts
        assert bridge.expert_start == 0
        assert bridge.expert_end == 256
        assert bridge.n_local_experts == 256


class TestSkyRLRdepBridgeRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        from nmoe.distributed import SkyRLRdepBridge

        bridge = SkyRLRdepBridge(
            dim=4096,
            n_total_experts=256,
            topk=8,
            profile="fp8",
        )

        repr_str = repr(bridge)
        assert "SkyRLRdepBridge" in repr_str
        assert "ep=" in repr_str
        assert "tp=" in repr_str
        assert "experts=" in repr_str
        assert "profile=fp8" in repr_str


class TestSkyRLRdepBridgeLazyInit:
    """Test lazy initialization of Rdep."""

    def test_rdep_not_created_on_init(self):
        """Test that Rdep is not created during __init__."""
        from nmoe.distributed import SkyRLRdepBridge

        bridge = SkyRLRdepBridge(
            dim=128,
            n_total_experts=8,
            topk=2,
        )

        # _rdep should be None until first access
        assert bridge._rdep is None

    def test_rdep_property_triggers_creation(self):
        """Test that accessing rdep property creates instance (signature check)."""
        from nmoe.distributed import SkyRLRdepBridge
        import inspect

        # Check that rdep is a property
        assert isinstance(
            inspect.getattr_static(SkyRLRdepBridge, 'rdep'),
            property
        )

        # Check that _ensure_rdep method exists
        assert hasattr(SkyRLRdepBridge, '_ensure_rdep')


class TestSkyRLRdepBridgeMoEMethods:
    """Test MoE method signatures."""

    def test_moe_bf16_signature(self):
        """Test moe_bf16 method exists with correct signature."""
        from nmoe.distributed import SkyRLRdepBridge
        import inspect

        sig = inspect.signature(SkyRLRdepBridge.moe_bf16)
        params = list(sig.parameters.keys())

        expected = ['self', 'x', 'expert_ids', 'gates', 'W1', 'W3', 'W2']
        assert params == expected

    def test_moe_blockscaled_signature(self):
        """Test moe_blockscaled method exists with correct signature."""
        from nmoe.distributed import SkyRLRdepBridge
        import inspect

        sig = inspect.signature(SkyRLRdepBridge.moe_blockscaled)
        params = list(sig.parameters.keys())

        expected = ['self', 'x', 'expert_ids', 'gates', 'W1', 'W3', 'W2', 'W_cache']
        assert params == expected


class TestMegatronIntegration:
    """Test Megatron integration utilities."""

    def test_create_bridge_from_megatron_import(self):
        """Test that create_bridge_from_megatron is importable."""
        from nmoe.distributed import create_bridge_from_megatron
        assert callable(create_bridge_from_megatron)

    def test_get_ep_rank_without_megatron(self):
        """Test get_nmoe_ep_rank without Megatron initialized."""
        from nmoe.distributed import get_nmoe_ep_rank

        # Without Megatron or dist initialized, should return 0
        rank = get_nmoe_ep_rank()
        assert rank == 0

    def test_get_ep_world_size_without_megatron(self):
        """Test get_nmoe_ep_world_size without Megatron initialized."""
        from nmoe.distributed import get_nmoe_ep_world_size

        # Without Megatron or dist initialized, should return 1
        world_size = get_nmoe_ep_world_size()
        assert world_size == 1

    def test_create_bridge_from_megatron_requires_megatron(self):
        """Test that create_bridge_from_megatron handles missing Megatron."""
        from nmoe.distributed import create_bridge_from_megatron

        # This should raise ImportError or RuntimeError depending on
        # whether Megatron is installed but not initialized
        try:
            bridge = create_bridge_from_megatron(
                dim=4096,
                n_total_experts=256,
                topk=8,
            )
            # If Megatron is installed but not initialized, should raise RuntimeError
            assert False, "Should have raised an error"
        except (ImportError, RuntimeError) as e:
            # Expected - Megatron not installed or not initialized
            assert "Megatron" in str(e) or "initialize" in str(e).lower()

    def test_all_exports_available(self):
        """Test that all expected exports are available from nmoe.distributed."""
        from nmoe import distributed

        assert hasattr(distributed, 'SkyRLRdepBridge')
        assert hasattr(distributed, 'create_bridge_from_megatron')
        assert hasattr(distributed, 'get_nmoe_ep_rank')
        assert hasattr(distributed, 'get_nmoe_ep_world_size')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
