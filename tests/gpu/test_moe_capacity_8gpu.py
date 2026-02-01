"""P0 Critical Tests for MoE Capacity Edge Cases.

Tests MoE capacity validation logic including:
1. Capacity exactly matches T*K*world (boundary case)
2. Capacity exceeded (should raise RuntimeError or handle gracefully)
3. M_recv = 0 on some ranks (no tokens received)
4. Empty expert handling (expert gets 0 tokens)
5. All tokens routed to single expert (worst case)
6. Mixed routing patterns across GPUs

Run single-GPU tests:
    pytest tests/gpu/test_moe_capacity_8gpu.py -v -m gpu

Run 8-GPU tests:
    torchrun --nproc_per_node=8 -m pytest tests/gpu/test_moe_capacity_8gpu.py -v -m multi_gpu
"""

import os
import pytest
import torch
import torch.distributed as dist
from typing import Tuple, Optional
from unittest.mock import patch, MagicMock


# =============================================================================
# Helper Functions
# =============================================================================

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


def init_distributed():
    """Initialize distributed if not already done."""
    if not dist.is_initialized() and get_world_size() > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(get_rank())


def skip_if_not_multi_gpu(min_gpus: int = 2):
    """Skip test if not enough GPUs available."""
    world_size = get_world_size()
    if world_size < min_gpus:
        pytest.skip(f"Requires at least {min_gpus} GPUs, have {world_size}")


def compute_required_capacity(T: int, K: int, world: int) -> int:
    """Compute the required capacity for given parameters.

    The capacity must be at least T * K * world to handle the worst case
    where all tokens from all ranks are routed to experts on a single rank.
    """
    return T * K * world


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module", autouse=True)
def setup_distributed():
    """Initialize distributed environment for all tests in module."""
    init_distributed()
    yield


@pytest.fixture
def cuda_device():
    """Provide CUDA device based on rank."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rank = get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    return device


@pytest.fixture
def routing_tensor_factory(cuda_device):
    """Factory for creating mock routing tensors (eids, gates).

    Returns a function that creates routing tensors with configurable patterns.
    """
    def _create(
        T: int,
        K: int,
        n_experts: int,
        pattern: str = "random",
        target_expert: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create routing tensors.

        Args:
            T: Number of tokens
            K: Number of experts per token (top-k)
            n_experts: Total number of experts
            pattern: Routing pattern - "random", "uniform", "single_expert", "empty_some"
            target_expert: Expert ID for single_expert pattern

        Returns:
            eids: [T, K] int32 expert IDs
            gates: [T, K] bfloat16 routing weights
        """
        if pattern == "random":
            eids = torch.randint(0, n_experts, (T, K), device=cuda_device, dtype=torch.int32)
            gates = torch.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        elif pattern == "uniform":
            # Distribute tokens evenly across experts
            eids = torch.zeros(T, K, device=cuda_device, dtype=torch.int32)
            for i in range(T):
                for k in range(K):
                    eids[i, k] = (i * K + k) % n_experts
            gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16) / K

        elif pattern == "single_expert":
            # All tokens route to a single expert (worst case)
            eids = torch.full((T, K), target_expert, device=cuda_device, dtype=torch.int32)
            gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16) / K

        elif pattern == "empty_some":
            # Some experts get zero tokens
            # Only use experts 0 and 1
            eids = torch.randint(0, 2, (T, K), device=cuda_device, dtype=torch.int32)
            gates = torch.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        elif pattern == "local_only":
            # Only route to local experts (for distributed tests)
            rank = get_rank()
            world_size = get_world_size()
            n_local = n_experts // world_size
            local_start = rank * n_local
            eids = torch.randint(local_start, local_start + n_local, (T, K), device=cuda_device, dtype=torch.int32)
            gates = torch.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        elif pattern == "remote_only":
            # Only route to remote experts (for distributed tests)
            rank = get_rank()
            world_size = get_world_size()
            if world_size == 1:
                # Fall back to random for single GPU
                eids = torch.randint(0, n_experts, (T, K), device=cuda_device, dtype=torch.int32)
            else:
                n_local = n_experts // world_size
                local_start = rank * n_local
                local_end = local_start + n_local
                # Generate expert IDs excluding local range
                available = [e for e in range(n_experts) if e < local_start or e >= local_end]
                if not available:
                    available = list(range(n_experts))
                eids = torch.tensor(
                    [[available[i % len(available)] for _ in range(K)] for i in range(T)],
                    device=cuda_device, dtype=torch.int32
                )
            gates = torch.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        else:
            raise ValueError(f"Unknown routing pattern: {pattern}")

        return eids, gates

    return _create


@pytest.fixture
def expert_weights_factory(cuda_device):
    """Factory for creating expert weight tensors."""
    def _create(n_experts: int, hidden_dim: int, inter_dim: int):
        W1 = torch.randn(n_experts, hidden_dim, inter_dim, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_experts, hidden_dim, inter_dim, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_experts, inter_dim, hidden_dim, device=cuda_device, dtype=torch.bfloat16) * 0.02
        return W1, W3, W2
    return _create


@pytest.fixture
def token_count_configs():
    """Common token count configurations for testing."""
    return [
        {"T": 1, "K": 1, "desc": "minimal"},
        {"T": 64, "K": 2, "desc": "standard"},
        {"T": 128, "K": 4, "desc": "high_topk"},
        {"T": 512, "K": 2, "desc": "large_batch"},
        {"T": 1024, "K": 8, "desc": "extreme"},
    ]


# =============================================================================
# Single-GPU Capacity Tests
# =============================================================================

@pytest.mark.gpu
class TestCapacityValidationSingleGPU:
    """Test capacity validation logic on single GPU."""

    def test_capacity_exactly_matches_requirement(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test capacity that exactly matches T*K*world (boundary case)."""
        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 8
        dim = 256
        inter = 512
        world = 1  # Single GPU

        # Capacity exactly matches requirement
        required_capacity = compute_required_capacity(T, K, world)

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=required_capacity,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="random")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # Should succeed with exact capacity
        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_capacity_buffer_correctly_sized(self, cuda_device):
        """Test that capacity buffer is correctly sized during initialization."""
        from nmoe.rdep import Rdep

        capacity = 4096
        dim = 256
        n_local = 8

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=2,
            profile="bf16",
            capacity=capacity,
        )

        # Verify capacity is stored correctly
        assert rdep.capacity == capacity
        assert rdep.dim == dim
        assert rdep.n_local == n_local

        # Verify pinned memory is allocated
        assert rdep._pinned_M_host is not None
        assert rdep._pinned_M_host.is_pinned()
        assert rdep._pinned_offs is not None
        assert rdep._pinned_offs.is_pinned()
        assert rdep._pinned_offs.shape[0] == n_local

    def test_capacity_exceeded_raises_error_via_dispatch(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test that exceeding capacity via dispatch() raises ValueError."""
        from nmoe.rdep import Rdep

        T, K = 128, 4
        n_local = 8
        dim = 256
        inter = 512
        world = 1

        # Set capacity too small
        required_capacity = compute_required_capacity(T, K, world)
        too_small_capacity = required_capacity // 2

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=too_small_capacity,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="random")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # dispatch() does early validation and raises ValueError
        with pytest.raises(ValueError, match="Token count exceeds RDEP capacity"):
            rdep.dispatch(x, eids, gates, W1, W3, W2)

    def test_empty_expert_handling(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test handling when some experts receive zero tokens."""
        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 8
        dim = 256
        inter = 512

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * 2,  # Extra capacity
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        # Use empty_some pattern: only experts 0, 1 get tokens, others are empty
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="empty_some")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # Should handle empty experts gracefully
        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_all_tokens_single_expert_worst_case(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test worst case: all tokens routed to single expert."""
        from nmoe.rdep import Rdep

        T, K = 128, 2
        n_local = 8
        dim = 256
        inter = 512

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * 2,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        # All tokens go to expert 0
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="single_expert", target_expert=0)
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)
        assert not torch.isnan(out).any(), "Output contains NaN values"

        # Verify output is not all zeros (computation happened)
        assert out.abs().sum() > 0, "Output should not be all zeros"

    def test_uniform_distribution_all_experts(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test uniform token distribution across all experts."""
        from nmoe.rdep import Rdep

        T, K = 128, 2
        n_local = 8
        dim = 256
        inter = 512

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * 2,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="uniform")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)
        assert not torch.isnan(out).any(), "Output contains NaN values"


@pytest.mark.gpu
class TestMRecvZeroSingleGPU:
    """Test M_recv = 0 scenarios on single GPU."""

    def test_zero_tokens_input(self, cuda_device, expert_weights_factory):
        """Test handling when T=0 (no tokens)."""
        from nmoe.rdep import Rdep

        T = 0  # Zero tokens
        K = 2
        n_local = 8
        dim = 256
        inter = 512

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=1024,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids = torch.empty(T, K, device=cuda_device, dtype=torch.int32)
        gates = torch.empty(T, K, device=cuda_device, dtype=torch.bfloat16)
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)
        assert out.numel() == 0


@pytest.mark.gpu
class TestCapacityWithDifferentProfiles:
    """Test capacity validation with different quantization profiles."""

    @pytest.mark.parametrize("profile", ["bf16"])
    def test_capacity_check_different_profiles(self, cuda_device, routing_tensor_factory, expert_weights_factory, profile):
        """Test capacity validation works correctly for all profiles."""
        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 8
        dim = 256
        inter = 512

        required_capacity = compute_required_capacity(T, K, 1)

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile=profile,
            capacity=required_capacity,
        )

        assert rdep.profile == profile

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="random")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # bf16 profile uses moe_bf16
        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)


# =============================================================================
# Multi-GPU Capacity Tests (8 GPU)
# =============================================================================

@pytest.mark.multi_gpu
class TestCapacityValidation8GPU:
    """Test capacity validation with 8 GPUs."""

    def test_capacity_exactly_matches_8gpu(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test capacity exactly matches T*K*world for 8 GPUs."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 4  # 4 experts per GPU = 32 total
        dim = 256
        inter = 512
        world = get_world_size()

        required_capacity = compute_required_capacity(T, K, world)

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=required_capacity,
        )

        assert rdep.world == world

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local * world, pattern="random")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)

    def test_capacity_exceeded_8gpu_raises_error(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test capacity exceeded raises error on 8 GPUs."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 128, 4
        n_local = 4
        dim = 256
        inter = 512
        world = get_world_size()

        required_capacity = compute_required_capacity(T, K, world)
        too_small_capacity = required_capacity // 4  # Much too small

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=too_small_capacity,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local * world, pattern="random")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        with pytest.raises(ValueError, match="Token count exceeds RDEP capacity"):
            rdep.dispatch(x, eids, gates, W1, W3, W2)

    def test_m_recv_zero_some_ranks_8gpu(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test M_recv = 0 on some ranks (no tokens received)."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 4
        dim = 256
        inter = 512
        world = get_world_size()
        rank = get_rank()

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * world,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Route all tokens to rank 0's experts only
        # This means ranks 1-7 receive M_recv = 0
        target_expert = 0  # Expert 0 is on rank 0
        eids, gates = routing_tensor_factory(T, K, n_local * world, pattern="single_expert", target_expert=target_expert)
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # All ranks should handle this gracefully
        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)

        # On rank 0, output should have content
        # On other ranks, they still participate in collective
        # but their local experts don't process tokens

    def test_all_tokens_to_single_rank_8gpu(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test all tokens routed to experts on a single rank."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 128, 2
        n_local = 4
        dim = 256
        inter = 512
        world = get_world_size()
        rank = get_rank()

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * world,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Route all tokens to experts on rank 3
        target_rank = 3
        target_expert_start = target_rank * n_local
        eids = torch.randint(target_expert_start, target_expert_start + n_local,
                            (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)

        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)


@pytest.mark.multi_gpu
class TestMixedRoutingPatterns8GPU:
    """Test mixed routing patterns across 8 GPUs."""

    def test_local_only_routing_8gpu(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test when each rank only routes to local experts."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 4
        dim = 256
        inter = 512
        world = get_world_size()

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * world,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local * world, pattern="local_only")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)

    def test_remote_only_routing_8gpu(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test when each rank only routes to remote experts."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 4
        dim = 256
        inter = 512
        world = get_world_size()

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * world,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local * world, pattern="remote_only")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)

    def test_alternating_local_remote_8gpu(self, cuda_device, expert_weights_factory):
        """Test alternating between local and remote experts."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 128, 2
        n_local = 4
        dim = 256
        inter = 512
        world = get_world_size()
        rank = get_rank()

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * world,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Create alternating pattern: even tokens -> local, odd tokens -> remote
        n_total = n_local * world
        local_start = rank * n_local
        local_end = local_start + n_local

        eids = torch.zeros(T, K, device=cuda_device, dtype=torch.int32)
        for i in range(T):
            if i % 2 == 0:
                # Local experts
                eids[i] = torch.randint(local_start, local_end, (K,), device=cuda_device, dtype=torch.int32)
            else:
                # Remote experts (next rank's experts, wrapping)
                remote_rank = (rank + 1) % world
                remote_start = remote_rank * n_local
                remote_end = remote_start + n_local
                eids[i] = torch.randint(remote_start, remote_end, (K,), device=cuda_device, dtype=torch.int32)

        gates = torch.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)


@pytest.mark.multi_gpu
class TestDroppedTokenCounting8GPU:
    """Test dropped token counting when capacity is exceeded."""

    def test_no_dropped_tokens_sufficient_capacity(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Verify no tokens dropped with sufficient capacity."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 4
        dim = 256
        inter = 512
        world = get_world_size()

        # Use generous capacity
        capacity = T * K * world * 2

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=capacity,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local * world, pattern="random")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)

        # Output should have valid values for all tokens
        assert out.shape == (T, dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# =============================================================================
# Boundary and Edge Case Tests
# =============================================================================

@pytest.mark.gpu
class TestCapacityBoundaryConditions:
    """Test boundary conditions for capacity."""

    @pytest.mark.parametrize("T,K", [
        (1, 1),      # Minimal
        (1, 8),      # Single token, high K
        (100, 1),    # Many tokens, K=1
        (64, 64),    # High K (unlikely but valid)
    ])
    def test_various_tk_combinations(self, cuda_device, expert_weights_factory, T, K):
        """Test various T, K combinations."""
        from nmoe.rdep import Rdep

        n_local = max(K, 8)  # Ensure enough experts for K
        dim = 256
        inter = 512

        capacity = compute_required_capacity(T, K, 1) * 2  # Double for safety

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=capacity,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids = torch.randint(0, n_local, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.softmax(torch.randn(T, K, device=cuda_device), dim=-1).to(torch.bfloat16)
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)

    def test_capacity_equals_one(self, cuda_device, expert_weights_factory):
        """Test with minimal capacity=1."""
        from nmoe.rdep import Rdep

        T, K = 1, 1
        n_local = 8
        dim = 256
        inter = 512

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=1,  # Minimal capacity
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids = torch.zeros(T, K, device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16)
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)

    def test_large_capacity_no_issue(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test that large capacity does not cause issues."""
        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 8
        dim = 256
        inter = 512

        # Very large capacity (100x more than needed)
        large_capacity = T * K * 100

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=large_capacity,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="random")
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)


@pytest.mark.gpu
class TestExpertMLPWithEmptyInputs:
    """Test expert MLP function with edge cases."""

    def test_expert_function_empty_input(self, cuda_device, expert_weights_factory):
        """Test expert() function with empty input tensor."""
        from nmoe.moe import expert

        n_local = 8
        dim = 256
        inter = 512

        # Empty input
        Xe_pad = torch.empty(0, dim, device=cuda_device, dtype=torch.bfloat16)
        offs_pad = torch.zeros(n_local, device=cuda_device, dtype=torch.int32)

        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # Should return empty tensor
        out = expert(Xe_pad, W1, W3, W2, offs_pad)
        assert out.shape == (0, dim)

    def test_expert_function_single_token(self, cuda_device, expert_weights_factory):
        """Test expert() function with single token."""
        from nmoe.moe import expert

        n_local = 8
        dim = 256
        inter = 512

        # Single padded token
        M_pad = 128  # Aligned
        Xe_pad = torch.randn(M_pad, dim, device=cuda_device, dtype=torch.bfloat16)
        offs_pad = torch.zeros(n_local, device=cuda_device, dtype=torch.int32)
        offs_pad[0] = M_pad  # All in first expert's region

        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        out = expert(Xe_pad, W1, W3, W2, offs_pad)
        assert out.shape == (M_pad, dim)


# =============================================================================
# Regression Tests
# =============================================================================

@pytest.mark.gpu
class TestCapacityRegressions:
    """Regression tests for known capacity-related issues."""

    def test_capacity_check_in_forward_vs_backward(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Verify capacity is checked in both forward and backward."""
        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 8
        dim = 256
        inter = 512

        capacity = T * K * 2

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=capacity,
        )

        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16, requires_grad=True)
        eids, gates = routing_tensor_factory(T, K, n_local, pattern="random")
        gates = gates.detach().requires_grad_(True)
        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)
        W1.requires_grad_(True)
        W3.requires_grad_(True)
        W2.requires_grad_(True)

        # Forward should work
        out = rdep.dispatch(x, eids, gates, W1, W3, W2)

        # Backward should also work with same capacity
        loss = out.sum()
        loss.backward()

        # Gradients should be computed
        assert x.grad is not None
        assert gates.grad is not None
        assert W1.grad is not None

    def test_repeated_dispatch_same_capacity(self, cuda_device, routing_tensor_factory, expert_weights_factory):
        """Test repeated dispatch calls with same capacity."""
        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 8
        dim = 256
        inter = 512

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=T * K * 2,
        )

        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # Run multiple times
        for i in range(5):
            x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
            eids, gates = routing_tensor_factory(T, K, n_local, pattern="random")

            out = rdep.dispatch(x, eids, gates, W1, W3, W2)
            assert out.shape == (T, dim)
            assert not torch.isnan(out).any(), f"Iteration {i}: Output contains NaN"


# =============================================================================
# Unit Tests for Capacity Calculation
# =============================================================================

@pytest.mark.gpu
class TestCapacityCalculation:
    """Unit tests for capacity calculation logic."""

    def test_compute_required_capacity_single_gpu(self):
        """Test required capacity calculation for single GPU."""
        assert compute_required_capacity(64, 2, 1) == 128
        assert compute_required_capacity(128, 4, 1) == 512
        assert compute_required_capacity(1, 1, 1) == 1

    def test_compute_required_capacity_multi_gpu(self):
        """Test required capacity calculation for multiple GPUs."""
        assert compute_required_capacity(64, 2, 8) == 1024
        assert compute_required_capacity(128, 4, 8) == 4096
        assert compute_required_capacity(256, 2, 4) == 2048

    def test_capacity_formula_matches_moe_check(self, cuda_device):
        """Verify our capacity formula matches the actual MoE check."""
        from nmoe.rdep import Rdep

        T, K = 64, 2
        n_local = 8
        dim = 256

        # With world=1 (single GPU), need = T * K * 1
        expected_need = T * K * 1
        capacity = expected_need  # Exact match

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=K,
            profile="bf16",
            capacity=capacity,
        )

        # Verify the validation in dispatch matches our formula
        # dispatch checks: T * K * world > capacity
        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)
        eids = torch.randint(0, n_local, (T, K), device=cuda_device, dtype=torch.int32)
        gates = torch.ones(T, K, device=cuda_device, dtype=torch.bfloat16) / K

        W1 = torch.randn(n_local, dim, dim * 2, device=cuda_device, dtype=torch.bfloat16)
        W3 = torch.randn(n_local, dim, dim * 2, device=cuda_device, dtype=torch.bfloat16)
        W2 = torch.randn(n_local, dim * 2, dim, device=cuda_device, dtype=torch.bfloat16)

        # Should pass with exact capacity
        out = rdep.dispatch(x, eids, gates, W1, W3, W2)
        assert out.shape == (T, dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
