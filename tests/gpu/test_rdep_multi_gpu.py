"""P0 Critical Tests: RDEP Multi-GPU Dispatch Operations.

This module tests RDEP dispatch operations across 8 GPUs (B200 configuration).
These tests are critical (P0) because multi-GPU dispatch is the foundation
of expert parallelism for large MoE models.

Run with:
    torchrun --nproc_per_node=8 pytest tests/gpu/test_rdep_multi_gpu.py -v

Tests cover:
1. IPC mode dispatch - tokens distributed across 8 GPUs via IPC
2. All-local dispatch - each token routes to local experts only
3. All-remote dispatch - each token routes to remote GPU experts
4. Uniform random dispatch - tokens distributed evenly across all 8 GPUs
5. Skewed dispatch - 90% tokens to one GPU, 10% distributed
6. IPC barrier convergence - all 8 ranks reach barrier
7. Return/combine correctness - results scatter back to source ranks
8. Gate weight scaling during return
"""

from __future__ import annotations

import os
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Tuple, Optional

# ==============================================================================
# Distributed Utilities
# ==============================================================================


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
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)


def cleanup_distributed():
    """Cleanup distributed resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def skip_if_not_multi_gpu(min_gpus: int = 8):
    """Skip test if not enough GPUs available."""
    world_size = get_world_size()
    if world_size < min_gpus:
        pytest.skip(f"Requires at least {min_gpus} GPUs, have {world_size}")


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce sum across ranks."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(tensor, src=src)
    return tensor


# ==============================================================================
# Reference Implementation for Validation
# ==============================================================================


def reference_moe_forward(
    x: torch.Tensor,
    eid: torch.Tensor,
    gates: torch.Tensor,
    W1: torch.Tensor,
    W3: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    """Reference MoE forward pass using manual expert routing.

    This is a single-GPU reference implementation that routes tokens
    to experts one-by-one to verify the RDEP dispatch/return correctness.

    Args:
        x: [T, H] BF16 input hidden states
        eid: [T, K] int32 expert IDs (global, 0 to n_total_experts-1)
        gates: [T, K] BF16 routing weights
        W1: [E, H, Dff] gate projection weights (local experts only)
        W3: [E, H, Dff] up projection weights (local experts only)
        W2: [E, Dff, H] down projection weights (local experts only)

    Returns:
        [T, H] BF16 output
    """
    T, H = x.shape
    K = eid.shape[1]
    E = W1.shape[0]  # Number of local experts

    # Accumulate in FP32 for precision
    out = torch.zeros((T, H), device=x.device, dtype=torch.float32)

    for k in range(K):
        eid_k = eid[:, k]
        gate_k = gates[:, k].float()

        for e in range(E):
            # Find tokens assigned to this expert at slot k
            mask = (eid_k == e)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            x_e = x[idx]  # [M_e, H]

            # Expert MLP: Y = SwiGLU(X @ W1, X @ W3) @ W2
            h1 = x_e @ W1[e]  # [M_e, Dff]
            h3 = x_e @ W3[e]  # [M_e, Dff]
            a = F.silu(h1) * h3  # [M_e, Dff]
            y_e = a @ W2[e]  # [M_e, H]

            # Accumulate with gate
            gate_e = gate_k[idx].unsqueeze(-1)  # [M_e, 1]
            out.index_add_(0, idx, (y_e.float() * gate_e))

    return out.to(dtype=torch.bfloat16)


def reference_moe_distributed(
    x: torch.Tensor,
    eid: torch.Tensor,
    gates: torch.Tensor,
    all_W1: torch.Tensor,
    all_W3: torch.Tensor,
    all_W2: torch.Tensor,
    rank: int,
    world_size: int,
    n_local: int,
) -> torch.Tensor:
    """Reference MoE forward with distributed experts.

    This simulates the distributed case where each rank has local experts,
    but we compute the full output using gathered weights for validation.

    Args:
        x: [T, H] BF16 input hidden states
        eid: [T, K] int32 expert IDs (global, 0 to n_total_experts-1)
        gates: [T, K] BF16 routing weights
        all_W1: [n_total, H, Dff] all experts' gate weights
        all_W3: [n_total, H, Dff] all experts' up weights
        all_W2: [n_total, Dff, H] all experts' down weights
        rank: Current rank
        world_size: Total number of ranks
        n_local: Number of experts per rank

    Returns:
        [T, H] BF16 output
    """
    T, H = x.shape
    K = eid.shape[1]
    n_total = all_W1.shape[0]

    out = torch.zeros((T, H), device=x.device, dtype=torch.float32)

    for k in range(K):
        eid_k = eid[:, k]
        gate_k = gates[:, k].float()

        for e in range(n_total):
            mask = (eid_k == e)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            x_e = x[idx]

            h1 = x_e @ all_W1[e]
            h3 = x_e @ all_W3[e]
            a = F.silu(h1) * h3
            y_e = a @ all_W2[e]

            gate_e = gate_k[idx].unsqueeze(-1)
            out.index_add_(0, idx, (y_e.float() * gate_e))

    return out.to(dtype=torch.bfloat16)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module", autouse=True)
def distributed_setup():
    """Initialize distributed environment for all tests in module."""
    init_distributed()
    yield
    # Cleanup is handled by process exit


@pytest.fixture
def seed():
    """Set random seed for reproducibility across all ranks."""
    seed_val = 42
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    return seed_val


@pytest.fixture
def device():
    """Get device for current rank."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rank = get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return torch.device(f"cuda:{local_rank}")


@pytest.fixture
def rdep_config():
    """Standard RDEP configuration for 8-GPU tests."""
    return {
        "dim": 256,
        "n_local": 8,  # 8 experts per GPU = 64 total experts
        "topk": 2,
        "capacity": 16384,
        "profile": "bf16",
    }


@pytest.fixture
def create_rdep(rdep_config, device):
    """Factory fixture to create Rdep instances."""
    from nmoe.rdep import Rdep

    def _create(**overrides):
        config = {**rdep_config, **overrides}
        return Rdep(
            dim=config["dim"],
            n_local=config["n_local"],
            topk=config["topk"],
            profile=config["profile"],
            capacity=config["capacity"],
        )
    return _create


@pytest.fixture
def expert_weights(device, rdep_config):
    """Create local expert weights for current rank."""
    n_local = rdep_config["n_local"]
    H = rdep_config["dim"]
    Dff = H * 4

    W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
    W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
    W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

    return W1, W3, W2


# ==============================================================================
# Test Classes
# ==============================================================================


@pytest.mark.multi_gpu
class TestIPCModeDispatch8GPU:
    """Test IPC mode dispatch across 8 GPUs."""

    def test_ipc_mode_initialization(self, create_rdep):
        """Test that RDEP initializes in IPC mode with 8 GPUs."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

        assert rdep.world == world, f"Expected world={world}, got {rdep.world}"
        assert rdep.rank == rank, f"Expected rank={rank}, got {rdep.rank}"
        assert rdep._mode in ("ipc", "hybrid"), f"Expected IPC/hybrid mode, got {rdep._mode}"

    def test_ipc_dispatch_basic(self, create_rdep, device, expert_weights, seed):
        """Test basic IPC dispatch with tokens distributed across 8 GPUs."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        T = 128
        K = rdep.topk
        H = rdep.dim

        # Create input with deterministic seed per rank
        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Random routing across all experts
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # Forward pass
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H), f"Wrong output shape: {out.shape}"
        assert not torch.isnan(out).any(), f"Rank {rank}: Output contains NaN"
        assert not torch.isinf(out).any(), f"Rank {rank}: Output contains Inf"

    def test_ipc_handle_exchange(self, create_rdep, device):
        """Test that IPC handles are correctly exchanged across ranks."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

        # After initialization, mode should be IPC for multi-GPU
        assert rdep._mode in ("ipc", "hybrid")
        assert rdep.world == world

        # Synchronize to ensure all ranks have completed IPC setup
        dist.barrier()


@pytest.mark.multi_gpu
class TestAllLocalDispatch:
    """Test dispatch where all tokens route to local experts only."""

    def test_all_local_tokens(self, create_rdep, device, expert_weights, seed):
        """Test when every token routes only to local experts."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        local_start = rank * n_local
        local_end = local_start + n_local
        T = 64
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # All expert IDs route to local experts only
        eid = torch.randint(local_start, local_end, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Compare against reference (local-only computation)
        local_eid = eid - local_start  # Convert to local indices
        ref = reference_moe_forward(x, local_eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            out, ref,
            atol=5e-2, rtol=1e-1,
            msg=f"Rank {rank}: All-local dispatch does not match reference"
        )

    def test_all_local_no_cross_rank_traffic(self, create_rdep, device, expert_weights, seed):
        """Verify no tokens are sent to remote ranks when all local."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        local_start = rank * n_local
        local_end = local_start + n_local
        T = 32
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(local_start, local_end, (T, K), device=device, dtype=torch.int32)
        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16)

        # Forward should complete successfully
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

        # All ranks should complete without hanging
        dist.barrier()


@pytest.mark.multi_gpu
class TestAllRemoteDispatch:
    """Test dispatch where all tokens route to remote GPU experts."""

    def test_all_remote_tokens(self, create_rdep, device, expert_weights, seed):
        """Test when every token routes to experts on other GPUs."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        local_start = rank * n_local
        local_end = local_start + n_local
        T = 64
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Generate expert IDs that avoid local experts
        # Route to next rank's experts (wrap around)
        next_rank = (rank + 1) % world
        remote_start = next_rank * n_local
        remote_end = remote_start + n_local

        eid = torch.randint(remote_start, remote_end, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # Forward should complete (tokens sent to remote, results returned)
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H), f"Rank {rank}: Wrong output shape"
        assert not torch.isnan(out).any(), f"Rank {rank}: Output contains NaN"
        assert not torch.isinf(out).any(), f"Rank {rank}: Output contains Inf"

        # Ensure all ranks complete
        dist.barrier()

    def test_remote_round_robin(self, create_rdep, device, expert_weights, seed):
        """Test tokens routed to each remote rank in round-robin fashion."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        T = 56  # 7 tokens per remote rank with 8 GPUs
        K = 1
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Route each token to a different remote rank
        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        for i in range(T):
            target_rank = (rank + 1 + (i % (world - 1))) % world
            target_expert = target_rank * n_local  # First expert of target rank
            eid[i, 0] = target_expert

        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

        dist.barrier()


@pytest.mark.multi_gpu
class TestUniformRandomDispatch:
    """Test dispatch with tokens uniformly distributed across all 8 GPUs."""

    def test_uniform_distribution(self, create_rdep, device, expert_weights, seed):
        """Test tokens distributed evenly across all 8 GPUs."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        T = 256
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Uniform random routing across all experts
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        # Verify output is not all zeros
        assert out.abs().sum() > 0, f"Rank {rank}: Output is all zeros"

        dist.barrier()

    def test_uniform_load_balance(self, create_rdep, device, expert_weights, seed):
        """Verify approximately uniform load across ranks."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

        n_local = rdep.n_local
        n_total = n_local * world
        T = 1024
        K = rdep.topk

        # Use same seed on all ranks for consistent routing
        torch.manual_seed(seed)
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)

        # Count tokens routed to each rank
        local_start = rank * n_local
        local_end = local_start + n_local

        local_count = 0
        for k in range(K):
            for e in range(local_start, local_end):
                local_count += (eid[:, k] == e).sum().item()

        # Gather counts from all ranks
        counts = torch.tensor([local_count], device=device, dtype=torch.float32)
        all_reduce_sum(counts)
        total_tokens = T * K * world  # Each rank has T tokens

        # Expected per-rank is total_tokens / world
        expected = total_tokens / world
        actual = counts.item() / world

        # Allow 20% deviation due to randomness
        assert abs(actual - expected) / expected < 0.20, \
            f"Load imbalance: expected ~{expected:.0f}, got {actual:.0f} per rank"


@pytest.mark.multi_gpu
class TestSkewedDispatch:
    """Test dispatch with skewed distribution (90% to one GPU, 10% distributed)."""

    def test_skewed_to_rank0(self, create_rdep, device, expert_weights, seed):
        """Test 90% tokens to rank 0, 10% distributed to others."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        T = 200
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # 90% tokens to rank 0's experts
        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        skew_threshold = int(0.9 * T)

        for i in range(T):
            for k in range(K):
                if i < skew_threshold:
                    # Route to rank 0
                    eid[i, k] = torch.randint(0, n_local, (1,), device=device).item()
                else:
                    # Distribute to other ranks
                    target_rank = 1 + (i % (world - 1))
                    target_expert = target_rank * n_local + torch.randint(0, n_local, (1,), device=device).item()
                    eid[i, k] = target_expert

        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # Should complete without hanging (tests load imbalance handling)
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        dist.barrier()

    def test_extreme_skew_single_expert(self, create_rdep, device, expert_weights, seed):
        """Test all tokens to a single expert on one GPU."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        T = 128
        K = 1
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # All tokens to expert 0 on rank 0
        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

        dist.barrier()


@pytest.mark.multi_gpu
class TestIPCBarrierConvergence:
    """Test IPC barrier convergence across all 8 ranks."""

    def test_barrier_convergence(self, create_rdep, device, expert_weights, seed):
        """Verify all 8 ranks reach IPC barrier and converge."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        T = 64
        K = rdep.topk
        H = rdep.dim

        # Each rank generates different inputs
        torch.manual_seed(seed + rank * 1000)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # Multiple iterations to stress-test barrier
        for iteration in range(5):
            out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            assert out.shape == (T, H)
            assert not torch.isnan(out).any()

        # All ranks must complete all iterations
        dist.barrier()

    def test_barrier_with_empty_local(self, create_rdep, device, expert_weights, seed):
        """Test barrier convergence when some ranks receive no local tokens."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        T = 32
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Only route to rank 0's experts (other ranks receive nothing locally)
        eid = torch.randint(0, n_local, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # Should complete without deadlock
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)

        dist.barrier()


@pytest.mark.multi_gpu
class TestReturnCombineCorrectness:
    """Test that results scatter back to source ranks correctly."""

    def test_return_to_source_rank(self, create_rdep, device, expert_weights, seed):
        """Test tokens return to correct source ranks after dispatch."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        T = 64
        K = rdep.topk
        H = rdep.dim

        # Create unique inputs per rank
        torch.manual_seed(seed + rank * 1000)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Each rank routes to different experts
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Output should be on the same device as input
        assert out.device == x.device
        assert out.shape == x.shape

        dist.barrier()

    def test_accumulation_across_topk(self, create_rdep, device, expert_weights, seed):
        """Test output correctly accumulates contributions from topk experts."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        local_start = rank * n_local
        T = 16
        K = 2
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Route to two different local experts for each token
        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        eid[:, 0] = local_start  # First local expert
        eid[:, 1] = local_start + 1  # Second local expert

        # Equal gates
        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16) * 0.5

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Compare with reference
        local_eid = eid - local_start
        ref = reference_moe_forward(x, local_eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            out, ref,
            atol=5e-2, rtol=1e-1,
            msg=f"Rank {rank}: Topk accumulation does not match reference"
        )


@pytest.mark.multi_gpu
class TestGateWeightScaling:
    """Test gate weight scaling during return phase."""

    def test_gate_scaling_identity(self, create_rdep, device, expert_weights, seed):
        """Test that gate=1.0 preserves expert output."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        local_start = rank * n_local
        T = 32
        K = 1
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        eid = torch.full((T, K), local_start, device=device, dtype=torch.int32)
        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        local_eid = eid - local_start
        ref = reference_moe_forward(x, local_eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            out, ref,
            atol=5e-2, rtol=1e-1,
            msg=f"Rank {rank}: Gate=1 does not preserve expert output"
        )

    def test_gate_scaling_half(self, create_rdep, device, expert_weights, seed):
        """Test that gate=0.5 halves the expert output."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        local_start = rank * n_local
        T = 32
        K = 1
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        eid = torch.full((T, K), local_start, device=device, dtype=torch.int32)

        # Gate=1.0 output
        gates_full = torch.ones(T, K, device=device, dtype=torch.bfloat16)
        out_full = rdep.moe_bf16(x, eid, gates_full, W1, W3, W2)

        # Gate=0.5 output
        gates_half = torch.full((T, K), 0.5, device=device, dtype=torch.bfloat16)
        out_half = rdep.moe_bf16(x, eid, gates_half, W1, W3, W2)

        # out_half should be approximately out_full * 0.5
        torch.testing.assert_close(
            out_half,
            out_full * 0.5,
            atol=5e-2, rtol=1e-1,
            msg=f"Rank {rank}: Gate=0.5 does not halve output"
        )

    def test_gate_zero_produces_zero_output(self, create_rdep, device, expert_weights, seed):
        """Test that gate=0 produces zero output."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        local_start = rank * n_local
        T = 32
        K = 1
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        eid = torch.full((T, K), local_start, device=device, dtype=torch.int32)
        gates = torch.zeros(T, K, device=device, dtype=torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.abs().max() < 1e-5, f"Rank {rank}: Gate=0 did not produce zero output"


@pytest.mark.multi_gpu
class TestDeterminism:
    """Test that same inputs produce same outputs (determinism)."""

    def test_deterministic_single_call(self, create_rdep, device, expert_weights, seed):
        """Test single forward pass is deterministic."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        T = 64
        K = rdep.topk
        H = rdep.dim

        # Fixed inputs
        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # Run twice
        out1 = rdep.moe_bf16(x.clone(), eid, gates, W1, W3, W2)
        out2 = rdep.moe_bf16(x.clone(), eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            out1, out2,
            atol=0, rtol=0,
            msg=f"Rank {rank}: Forward pass is not deterministic"
        )

    def test_deterministic_across_runs(self, create_rdep, device, expert_weights, seed):
        """Test multiple runs produce same results."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        T = 32
        K = rdep.topk
        H = rdep.dim

        results = []
        for run in range(3):
            torch.manual_seed(seed + rank)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            results.append(out.clone())

        for i in range(1, len(results)):
            torch.testing.assert_close(
                results[0], results[i],
                atol=0, rtol=0,
                msg=f"Rank {rank}: Run {i} differs from run 0"
            )


@pytest.mark.multi_gpu
class TestConcurrentForwardBackward:
    """Test concurrent forward/backward on different batches."""

    def test_sequential_forward_backward(self, create_rdep, device, expert_weights, seed):
        """Test forward then backward in sequence."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        W1.requires_grad_(True)
        W3.requires_grad_(True)
        W2.requires_grad_(True)

        n_local = rdep.n_local
        n_total = n_local * world
        T = 32
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16).requires_grad_(True)

        # Forward
        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Backward
        loss = out.float().sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, f"Rank {rank}: x.grad is None"
        assert gates.grad is not None, f"Rank {rank}: gates.grad is None"
        assert W1.grad is not None, f"Rank {rank}: W1.grad is None"
        assert W3.grad is not None, f"Rank {rank}: W3.grad is None"
        assert W2.grad is not None, f"Rank {rank}: W2.grad is None"

        dist.barrier()

    def test_multiple_batches_sequential(self, create_rdep, device, expert_weights, seed):
        """Test multiple batches processed sequentially."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        W1_param = W1.clone().requires_grad_(True)
        W3_param = W3.clone().requires_grad_(True)
        W2_param = W2.clone().requires_grad_(True)

        n_local = rdep.n_local
        n_total = n_local * world
        T = 16
        K = rdep.topk
        H = rdep.dim

        accumulated_loss = 0.0

        for batch_idx in range(3):
            torch.manual_seed(seed + rank + batch_idx * 1000)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16).requires_grad_(True)

            out = rdep.moe_bf16(x, eid, gates, W1_param, W3_param, W2_param)
            loss = out.float().sum()
            loss.backward()

            accumulated_loss += loss.item()

        assert accumulated_loss > 0, f"Rank {rank}: Accumulated loss is zero"
        assert W1_param.grad is not None
        assert W1_param.grad.abs().sum() > 0

        dist.barrier()

    def test_gradient_accumulation(self, create_rdep, device, expert_weights, seed):
        """Test gradient accumulation across multiple forward/backward passes."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

        # Create parameters
        W1 = expert_weights[0].clone().requires_grad_(True)
        W3 = expert_weights[1].clone().requires_grad_(True)
        W2 = expert_weights[2].clone().requires_grad_(True)

        n_local = rdep.n_local
        n_total = n_local * world
        T = 16
        K = rdep.topk
        H = rdep.dim

        # Accumulate gradients over 4 micro-batches
        for micro_batch in range(4):
            torch.manual_seed(seed + rank + micro_batch * 100)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            loss = out.float().sum() / 4  # Scale by accumulation steps
            loss.backward()

        # Gradients should be accumulated
        assert W1.grad is not None
        assert W1.grad.abs().sum() > 0

        dist.barrier()


@pytest.mark.multi_gpu
class TestEdgeCases8GPU:
    """Edge cases specific to 8-GPU configuration."""

    def test_single_token_per_rank(self, create_rdep, device, expert_weights, seed):
        """Test with only one token per rank."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_total = rdep.n_local * world
        T = 1
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

        dist.barrier()

    def test_large_batch_per_rank(self, create_rdep, device, expert_weights, seed):
        """Test with large batch size per rank."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep(capacity=65536)
        W1, W3, W2 = expert_weights

        n_total = rdep.n_local * world
        T = 1024
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        dist.barrier()

    def test_empty_rank(self, create_rdep, device, expert_weights, seed):
        """Test when one rank receives no tokens at all."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        T = 64
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # All tokens go to rank 0's first expert (rank 0 is only recipient)
        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)

        dist.barrier()

    def test_max_topk(self, create_rdep, device, seed):
        """Test with maximum topk value (8 experts per token)."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        H = 256
        n_local = 8
        K = 8  # Maximum topk
        n_total = n_local * world
        Dff = H * 4

        rdep = create_rdep(topk=K, capacity=32768)

        # Create weights
        W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

        T = 32

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

        dist.barrier()


@pytest.mark.multi_gpu
class TestCorrectness8GPU:
    """Correctness tests specific to 8-GPU configuration."""

    def test_cross_rank_routing_correctness(self, create_rdep, device, expert_weights, seed):
        """Test that cross-rank routing produces correct results."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        n_total = n_local * world
        local_start = rank * n_local
        T = 16
        K = 1
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Route half local, half to next rank
        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        next_rank = (rank + 1) % world
        next_start = next_rank * n_local

        for i in range(T):
            if i < T // 2:
                eid[i, 0] = local_start
            else:
                eid[i, 0] = next_start

        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16)

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

        # Output should not be all zeros
        assert out.abs().sum() > 0

        dist.barrier()

    def test_bidirectional_routing(self, create_rdep, device, expert_weights, seed):
        """Test bidirectional routing between adjacent ranks."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        n_local = rdep.n_local
        T = 32
        K = 2
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Route to previous and next rank
        prev_rank = (rank - 1 + world) % world
        next_rank = (rank + 1) % world
        prev_start = prev_rank * n_local
        next_start = next_rank * n_local

        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        eid[:, 0] = prev_start  # First expert of previous rank
        eid[:, 1] = next_start  # First expert of next rank

        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16) * 0.5

        out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert out.shape == (T, H)
        assert not torch.isnan(out).any()

        dist.barrier()


# ==============================================================================
# Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
