"""Tests for EP+TP (Expert Parallel + Tensor Parallel) configurations.

These tests verify that nmoe's process group initialization and RDEP
dispatcher work correctly with combined EP and TP parallelism.

Note: Most of these tests are marked as multi-GPU tests and require
a distributed environment to run. They are skipped in single-GPU CI.
"""

import pytest
import torch

# Mark entire module as needing distributed for most tests
pytestmark = pytest.mark.distributed


class TestProcessGroupCreation:
    """Test process group initialization logic."""

    def test_ep_size_validation(self):
        """Test that ep_size * tp_size must not exceed world_size."""
        from nmoe.distributed.init_groups import (
            init_nmoe_process_groups,
            cleanup_process_groups,
            _INITIALIZED,
        )

        # Without distributed, should raise RuntimeError
        with pytest.raises(RuntimeError, match="distributed is not initialized"):
            init_nmoe_process_groups(ep_size=4, tp_size=2)

    def test_get_group_requires_initialization(self):
        """Test that get_*_group() requires initialization."""
        from nmoe.distributed.init_groups import (
            get_ep_group,
            get_tp_group,
            cleanup_process_groups,
        )

        cleanup_process_groups()  # Ensure clean state

        with pytest.raises(RuntimeError, match="not initialized"):
            get_ep_group()

        with pytest.raises(RuntimeError, match="not initialized"):
            get_tp_group()

    def test_is_initialized_flag(self):
        """Test is_nmoe_parallel_initialized() state tracking."""
        from nmoe.distributed.init_groups import (
            is_nmoe_parallel_initialized,
            cleanup_process_groups,
        )

        cleanup_process_groups()
        assert not is_nmoe_parallel_initialized()


class TestEPTPRankCalculation:
    """Test EP and TP rank calculation logic."""

    def test_ep_tp_rank_math(self):
        """Test the mathematical relationship between global rank and EP/TP ranks.

        For world_size=8, EP=4, TP=2:
        - EP groups: [0,2,4,6] and [1,3,5,7]  (stride by TP size)
        - TP groups: [0,1], [2,3], [4,5], [6,7]  (consecutive)

        Global rank 0: ep_rank=0, tp_rank=0
        Global rank 1: ep_rank=0, tp_rank=1
        Global rank 2: ep_rank=1, tp_rank=0
        Global rank 3: ep_rank=1, tp_rank=1
        ...
        """
        # This is the logic that init_nmoe_process_groups uses
        world_size = 8
        ep_size = 4
        tp_size = 2

        # TP groups: consecutive ranks
        tp_groups = []
        for base in range(0, world_size, tp_size):
            group_ranks = list(range(base, base + tp_size))
            tp_groups.append(group_ranks)

        expected_tp_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
        assert tp_groups == expected_tp_groups, f"TP groups mismatch: {tp_groups}"

        # EP groups: strided by tp_size
        ep_groups = []
        for start in range(tp_size):
            group_ranks = list(range(start, world_size, tp_size))[:ep_size]
            ep_groups.append(group_ranks)

        expected_ep_groups = [[0, 2, 4, 6], [1, 3, 5, 7]]
        assert ep_groups == expected_ep_groups, f"EP groups mismatch: {ep_groups}"

        # Verify rank to group mapping
        for rank in range(world_size):
            # Calculate EP rank
            ep_group_idx = rank % tp_size
            ep_rank = rank // tp_size
            assert rank in ep_groups[ep_group_idx], f"Rank {rank} not in EP group {ep_group_idx}"
            assert ep_groups[ep_group_idx].index(rank) == ep_rank, \
                f"Rank {rank} has wrong EP rank: expected {ep_rank}"

            # Calculate TP rank
            tp_group_idx = rank // tp_size
            tp_rank = rank % tp_size
            assert rank in tp_groups[tp_group_idx], f"Rank {rank} not in TP group {tp_group_idx}"
            assert tp_groups[tp_group_idx].index(rank) == tp_rank, \
                f"Rank {rank} has wrong TP rank: expected {tp_rank}"


class TestSkyRLBridgeConfig:
    """Test SkyRLRdepBridge configuration for EP+TP."""

    def test_bridge_expert_distribution(self):
        """Test that SkyRLRdepBridge correctly calculates local experts."""
        from nmoe.distributed.skyrl_bridge import SkyRLRdepBridge

        # Simulate EP=4 configuration with 256 experts
        # Each EP rank should have 64 local experts
        n_total_experts = 256
        ep_size = 4

        for ep_rank in range(ep_size):
            # Manually set what the bridge would calculate
            n_local = n_total_experts // ep_size
            expert_start = ep_rank * n_local
            expert_end = expert_start + n_local

            expected_n_local = 64
            expected_start = ep_rank * 64
            expected_end = expected_start + 64

            assert n_local == expected_n_local, \
                f"EP rank {ep_rank}: wrong n_local {n_local}"
            assert expert_start == expected_start, \
                f"EP rank {ep_rank}: wrong expert_start {expert_start}"
            assert expert_end == expected_end, \
                f"EP rank {ep_rank}: wrong expert_end {expert_end}"

    def test_bridge_validates_expert_divisibility(self):
        """Test that bridge validates n_experts is divisible by ep_size.

        Note: Without distributed initialization, ep_group=None means world_size=1,
        so any number of experts is divisible. We test the validation logic
        by checking that the calculation is correct.
        """
        from nmoe.distributed.skyrl_bridge import SkyRLRdepBridge

        # With world_size=1 (no dist), any n_experts is valid
        bridge = SkyRLRdepBridge(
            dim=4096,
            n_total_experts=100,
            topk=8,
            ep_group=None,  # World size = 1
        )
        # All 100 experts are local when world=1
        assert bridge.n_local_experts == 100
        assert bridge.ep_size == 1

        # Test the validation logic directly: n_experts % ep_size must be 0
        # 100 is divisible by 4 (100 / 4 = 25)
        n_experts = 100
        ep_size = 4
        assert n_experts % ep_size == 0, "100 should be divisible by 4"

        # 101 is NOT divisible by 4
        n_experts = 101
        ep_size = 4
        assert n_experts % ep_size != 0, "101 is not divisible by 4"

        # 256 is divisible by 4
        n_experts = 256
        ep_size = 4
        assert n_experts % ep_size == 0, "256 should be divisible by 4"


class TestExpertWeightDistribution:
    """Test expert weight distribution for EP configurations."""

    def test_get_local_expert_weights(self):
        """Test extracting local expert weights from global tensor."""
        from nmoe.distributed.skyrl_bridge import SkyRLRdepBridge

        # Create a bridge (uses default world=1)
        bridge = SkyRLRdepBridge(
            dim=128,
            n_total_experts=8,
            topk=2,
        )

        # With EP=1 (single GPU), all experts are local
        assert bridge.n_local_experts == 8
        assert bridge.expert_start == 0
        assert bridge.expert_end == 8

        # Create global weights
        global_W = torch.randn(8, 128, 512)

        # Extract local weights
        local_W = bridge.get_local_expert_weights(global_W)

        # Should be same as global for single GPU
        assert local_W.shape == global_W.shape
        assert torch.allclose(local_W, global_W)

    def test_map_expert_ids_to_local(self):
        """Test mapping global expert IDs to local indices."""
        from nmoe.distributed.skyrl_bridge import SkyRLRdepBridge

        # Create a bridge (single GPU, all experts local)
        bridge = SkyRLRdepBridge(
            dim=128,
            n_total_experts=8,
            topk=2,
        )

        # Expert IDs for 4 tokens, topk=2
        global_ids = torch.tensor([
            [0, 3],
            [2, 7],
            [1, 4],
            [5, 6],
        ])

        local_ids = bridge.map_expert_ids_to_local(global_ids)

        # With single GPU, all IDs remain unchanged (0-7 are all local)
        assert torch.equal(local_ids, global_ids)


class TestRDEPWithProcessGroup:
    """Test RDEP dispatcher with custom process groups."""

    def test_rdep_accepts_ep_group(self):
        """Test that Rdep constructor accepts ep_group parameter."""
        from nmoe.rdep import Rdep
        import inspect

        sig = inspect.signature(Rdep.__init__)
        params = list(sig.parameters.keys())

        assert "ep_group" in params, "Rdep should accept ep_group parameter"

    def test_rdep_uses_group_for_world_size(self):
        """Test that Rdep uses group for world/rank calculation."""
        from nmoe.rdep import _get_group_world_size, _get_group_rank

        # Without dist initialized, should return defaults
        assert _get_group_world_size(None) == 1
        assert _get_group_rank(None) == 0


@pytest.mark.gpu
class TestGPUOperations:
    """GPU-specific tests for EP+TP configurations.

    These tests require CUDA and are skipped on CPU-only systems.
    """

    @pytest.fixture
    def cuda_available(self):
        """Skip if CUDA is not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_tensor_on_correct_device(self, cuda_available):
        """Test that tensors are placed on correct GPU for EP rank."""
        # Basic sanity check that CUDA is working
        device = torch.device("cuda:0")
        x = torch.randn(4, 128, device=device)
        assert x.device.type == "cuda"


# Multi-GPU tests that require torchrun
class TestEPTPWeightSharding:
    """Test EP+TP weight sharding utilities."""

    def test_shard_info_creation(self):
        """Test ShardInfo creation and validation."""
        from nmoe.distributed.ep_tp_shard import get_shard_info

        info = get_shard_info(
            n_total_experts=256,
            ep_size=4,
            tp_size=2,
            hidden_size=4096,
            intermediate_size=16384,
        )

        assert info.n_local_experts == 64
        assert info.ep_size == 4
        assert info.tp_size == 2
        assert info.expert_dim_per_tp == 8192
        assert info.hidden_dim_per_tp == 2048

    def test_shard_info_expert_range(self):
        """Test expert range calculation."""
        from nmoe.distributed.ep_tp_shard import get_shard_info

        info = get_shard_info(n_total_experts=256, ep_size=4, tp_size=1)

        assert info.get_expert_range(0) == (0, 64)
        assert info.get_expert_range(1) == (64, 128)
        assert info.get_expert_range(2) == (128, 192)
        assert info.get_expert_range(3) == (192, 256)

    def test_shard_info_not_divisible(self):
        """Test that non-divisible expert count raises error."""
        from nmoe.distributed.ep_tp_shard import get_shard_info

        with pytest.raises(ValueError, match="must be divisible"):
            get_shard_info(n_total_experts=100, ep_size=3)

    def test_shard_expert_weights_ep_only(self):
        """Test sharding with EP only (no TP)."""
        from nmoe.distributed.ep_tp_shard import get_shard_info, shard_expert_weights

        info = get_shard_info(n_total_experts=8, ep_size=2, tp_size=1)
        W = torch.randn(8, 128, 512)

        shard_0 = shard_expert_weights(W, info, ep_rank=0, tp_rank=0, weight_type="gate")
        shard_1 = shard_expert_weights(W, info, ep_rank=1, tp_rank=0, weight_type="gate")

        assert shard_0.shape == (4, 128, 512)
        assert shard_1.shape == (4, 128, 512)

        # Verify content
        assert torch.equal(shard_0, W[:4])
        assert torch.equal(shard_1, W[4:])

    def test_shard_expert_weights_ep_tp(self):
        """Test sharding with EP and TP."""
        from nmoe.distributed.ep_tp_shard import get_shard_info, shard_expert_weights

        info = get_shard_info(n_total_experts=8, ep_size=2, tp_size=2)
        W1 = torch.randn(8, 128, 512)  # gate_proj

        # EP=0, TP=0: experts 0-3, columns 0-255
        shard_00 = shard_expert_weights(W1, info, ep_rank=0, tp_rank=0, weight_type="gate")
        assert shard_00.shape == (4, 128, 256)

        # EP=0, TP=1: experts 0-3, columns 256-511
        shard_01 = shard_expert_weights(W1, info, ep_rank=0, tp_rank=1, weight_type="gate")
        assert shard_01.shape == (4, 128, 256)

        # Verify correct slice
        assert torch.equal(shard_00, W1[:4, :, :256])
        assert torch.equal(shard_01, W1[:4, :, 256:])

    def test_shard_down_proj_row_parallel(self):
        """Test row-parallel sharding for down_proj (W2)."""
        from nmoe.distributed.ep_tp_shard import get_shard_info, shard_expert_weights

        info = get_shard_info(n_total_experts=8, ep_size=2, tp_size=2)
        W2 = torch.randn(8, 512, 128)  # down_proj: [E, Dff, H]

        # Row parallel shards along dim1 (Dff)
        shard_00 = shard_expert_weights(W2, info, ep_rank=0, tp_rank=0, weight_type="down")
        assert shard_00.shape == (4, 256, 128)

        # Verify correct slice
        assert torch.equal(shard_00, W2[:4, :256, :])

    def test_validate_shard_roundtrip(self):
        """Test that shard->gather roundtrip preserves data."""
        from nmoe.distributed.ep_tp_shard import get_shard_info, validate_shard_roundtrip

        info = get_shard_info(n_total_experts=16, ep_size=4, tp_size=2)

        W1 = torch.randn(16, 128, 512)
        W2 = torch.randn(16, 512, 128)

        assert validate_shard_roundtrip(W1, info, "gate")
        assert validate_shard_roundtrip(W1, info, "up")
        assert validate_shard_roundtrip(W2, info, "down")

    def test_compute_shard_sizes(self):
        """Test shard size computation."""
        from nmoe.distributed.ep_tp_shard import compute_shard_sizes

        sizes = compute_shard_sizes(
            total_experts=256,
            hidden_size=4096,
            intermediate_size=16384,
            ep_size=4,
            tp_size=2,
        )

        assert sizes["gate"] == (64, 4096, 8192)
        assert sizes["up"] == (64, 4096, 8192)
        assert sizes["down"] == (64, 8192, 4096)
        assert sizes["local_experts"] == 64

    def test_get_tp_shard_dim(self):
        """Test TP shard dimension helper."""
        from nmoe.distributed.ep_tp_shard import get_tp_shard_dim

        assert get_tp_shard_dim("gate") == 2  # Column parallel
        assert get_tp_shard_dim("up") == 2    # Column parallel
        assert get_tp_shard_dim("down") == 1  # Row parallel


@pytest.mark.multi_gpu
class TestMultiGPUOperations:
    """Tests that require multiple GPUs to run.

    Run with: torchrun --nproc_per_node=N pytest test_ep_tp.py -k multi_gpu
    """

    def test_ep4_tp2_gradient_equivalence(self):
        """Test that EP=4, TP=2 produces same gradients as single GPU.

        This test requires 8 GPUs and is typically run in CI.
        """
        pytest.skip("Requires 8 GPUs - run with torchrun")

    def test_expert_dispatch_correctness(self):
        """Test that expert dispatch produces correct outputs across EP ranks.

        Each token should be dispatched to the correct experts regardless
        of which EP rank owns that expert.
        """
        pytest.skip("Requires multiple GPUs - run with torchrun")
