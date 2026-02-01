"""P2 Tests: Cross-GPU Expert Routing.

This module tests cross-GPU expert routing for MoE models across multiple GPUs.
These tests validate:

1. Token dispatch to remote experts
2. Load balancing across GPUs
3. Cross-GPU gradient flow
4. EP group handling
5. Determinism

Run with:
    torchrun --nproc_per_node=8 pytest tests/gpu/test_cross_gpu_routing.py -v

Tests require at least 8 GPUs for full EP=8 configuration testing.
"""

from __future__ import annotations

import os
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

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


def all_gather_object(obj: Any) -> list:
    """All-gather Python objects from all ranks."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [obj]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


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

    Args:
        x: [T, H] BF16 input hidden states
        eid: [T, K] int32 expert IDs (0 to n_local-1)
        gates: [T, K] BF16 routing weights
        W1: [E, H, Dff] gate projection weights
        W3: [E, H, Dff] up projection weights
        W2: [E, Dff, H] down projection weights

    Returns:
        [T, H] BF16 output
    """
    T, H = x.shape
    K = eid.shape[1]
    E = W1.shape[0]

    out = torch.zeros((T, H), device=x.device, dtype=torch.float32)

    for k in range(K):
        eid_k = eid[:, k]
        gate_k = gates[:, k].float()

        for e in range(E):
            mask = (eid_k == e)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            x_e = x[idx]

            h1 = x_e @ W1[e]
            h3 = x_e @ W3[e]
            a = F.silu(h1) * h3
            y_e = a @ W2[e]

            gate_e = gate_k[idx].unsqueeze(-1)
            out.index_add_(0, idx, (y_e.float() * gate_e))

    return out.to(dtype=torch.bfloat16)


def compute_load_stats(
    eid: torch.Tensor,
    n_experts: int,
) -> Tuple[torch.Tensor, float, float]:
    """Compute expert load statistics.

    Args:
        eid: [T, K] expert IDs
        n_experts: Total number of experts

    Returns:
        Tuple of (loads, mean, cv) where:
        - loads: [n_experts] token count per expert
        - mean: Mean load across experts
        - cv: Coefficient of variation (std/mean)
    """
    loads = torch.bincount(
        eid.reshape(-1).long(),
        minlength=n_experts
    ).float()
    mean = loads.mean()
    std = loads.std()
    cv = (std / mean).item() if mean > 0 else 0.0
    return loads, mean.item(), cv


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
    """Standard RDEP configuration for cross-GPU routing tests."""
    return {
        "dim": 256,
        "n_local": 8,  # 8 experts per GPU = 64 total experts with 8 GPUs
        "topk": 2,
        "capacity": 32768,
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
            ep_group=config.get("ep_group"),
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
# Test Classes: Token Dispatch to Remote Experts
# ==============================================================================


@pytest.mark.multi_gpu
class TestTokenDispatchRemote:
    """Test token dispatch to remote experts on other GPUs."""

    def test_tokens_reach_remote_experts(self, create_rdep, device, expert_weights, seed):
        """Test that tokens correctly reach experts on other GPUs."""
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

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Force routing to remote experts (next rank's experts)
        next_rank = (rank + 1) % world
        remote_start = next_rank * n_local
        eid = torch.full(
            (T, K),
            remote_start,
            device=device,
            dtype=torch.int32
        )
        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16) / K

        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Verify output shape
        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

        # Verify no NaN
        assert not torch.isnan(output).any(), f"Rank {rank}: Output contains NaN"

        # Verify no Inf
        assert not torch.isinf(output).any(), f"Rank {rank}: Output contains Inf"

        dist.barrier()

    def test_output_shape_preserved(self, create_rdep, device, expert_weights, seed):
        """Test that output shape is preserved for various input shapes."""
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
        H = rdep.dim

        test_shapes = [(32,), (64,), (128,), (256,), (512,)]

        for (T,) in test_shapes:
            torch.manual_seed(seed + rank + T)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
            eid = torch.randint(0, n_total, (T, rdep.topk), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, rdep.topk, device=device), dim=-1).to(torch.bfloat16)

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            assert output.shape == x.shape, f"T={T}: Expected {x.shape}, got {output.shape}"

        dist.barrier()

    def test_gate_weights_applied_correctly(self, create_rdep, device, expert_weights, seed):
        """Test that gate weights are applied correctly during dispatch."""
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

        # Route to first local expert
        eid = torch.full((T, K), local_start, device=device, dtype=torch.int32)

        # Test gate=1.0
        gates_full = torch.ones(T, K, device=device, dtype=torch.bfloat16)
        out_full = rdep.moe_bf16(x, eid, gates_full, W1, W3, W2)

        # Test gate=0.5
        gates_half = torch.full((T, K), 0.5, device=device, dtype=torch.bfloat16)
        out_half = rdep.moe_bf16(x, eid, gates_half, W1, W3, W2)

        # out_half should be approximately out_full * 0.5
        torch.testing.assert_close(
            out_half,
            out_full * 0.5,
            atol=5e-2, rtol=1e-1,
            msg=f"Rank {rank}: Gate weight not applied correctly"
        )

        dist.barrier()

    def test_mixed_local_remote_routing(self, create_rdep, device, expert_weights, seed):
        """Test routing with mix of local and remote experts."""
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
        next_rank = (rank + 1) % world
        next_start = next_rank * n_local
        T = 64
        K = 2
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # K=0: local expert, K=1: remote expert
        eid = torch.zeros(T, K, device=device, dtype=torch.int32)
        eid[:, 0] = local_start  # First local expert
        eid[:, 1] = next_start   # First expert of next rank

        gates = torch.ones(T, K, device=device, dtype=torch.bfloat16) * 0.5

        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        dist.barrier()


# ==============================================================================
# Test Classes: Load Balancing Across GPUs
# ==============================================================================


@pytest.mark.multi_gpu
class TestLoadBalancing:
    """Test load balancing across GPUs."""

    def test_balanced_expert_loads(self, create_rdep, device, expert_weights, seed):
        """Test that expert loads are balanced with uniform routing."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        n_local = 8
        n_total = n_local * world
        T = 1024
        K = 2

        # Use same seed on all ranks for consistent analysis
        torch.manual_seed(seed)
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)

        # Compute load stats
        loads, mean, cv = compute_load_stats(eid, n_total)

        # With uniform random routing, CV should be relatively low
        # Allow up to 30% CV for statistical variance
        assert cv < 0.30, f"Load imbalance too high: CV={cv:.3f}"

        dist.barrier()

    def test_get_expert_load_stats_accuracy(self, create_rdep, device, expert_weights, seed):
        """Test that get_expert_load_stats returns accurate values."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        from nmoe.model import MoE
        from nmoe.config import Config
        from nmoe.rdep import Rdep

        n_local = 8
        n_total = n_local * world

        # Create config
        cfg = Config(
            dim=256,
            inter_dim=1024,
            moe_inter_dim=1024,
            n_routed_experts=n_total,
            n_activated_experts=2,
            n_layers=1,
            n_heads=4,
            batch_size=8,
            seq_len=128,
            dtype='bf16',
        )

        # Create RDEP and MoE
        rdep = Rdep(
            dim=cfg.dim,
            n_local=n_local,
            topk=cfg.n_activated_experts,
            profile='bf16',
            capacity=32768,
        )

        moe = MoE(cfg, layer_id=0, rdep=rdep).cuda()
        moe.init_weights()

        T = 128
        H = cfg.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Forward pass
        _ = moe(x)

        # Check last_loads is populated
        assert moe.last_loads is not None, "last_loads should be populated after forward"
        assert moe.last_loads.shape[0] == n_total, f"Expected {n_total} loads, got {moe.last_loads.shape[0]}"

        # Verify loads sum to expected total
        total_assignments = T * cfg.n_activated_experts
        assert moe.last_loads.sum().item() == total_assignments, \
            f"Load sum {moe.last_loads.sum().item()} != expected {total_assignments}"

        dist.barrier()

    def test_load_imbalance_cv_metric(self, create_rdep, device, seed):
        """Test coefficient of variation (CV) metric for load imbalance."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        n_local = 8
        n_total = n_local * world

        # Test 1: Uniform distribution (low CV)
        torch.manual_seed(seed)
        eid_uniform = torch.randint(0, n_total, (1024, 2), device=device, dtype=torch.int32)
        _, _, cv_uniform = compute_load_stats(eid_uniform, n_total)

        # Test 2: Skewed distribution (high CV)
        # 90% to first expert, 10% distributed
        eid_skewed = torch.zeros(1024, 2, device=device, dtype=torch.int32)
        skew_threshold = int(0.9 * 1024)
        eid_skewed[:skew_threshold] = 0  # All to expert 0
        eid_skewed[skew_threshold:] = torch.randint(1, n_total, (1024 - skew_threshold, 2), device=device)
        _, _, cv_skewed = compute_load_stats(eid_skewed, n_total)

        # CV for skewed should be much higher than uniform
        assert cv_skewed > cv_uniform * 2, \
            f"Skewed CV ({cv_skewed:.3f}) should be >> uniform CV ({cv_uniform:.3f})"

        dist.barrier()

    def test_per_rank_load_distribution(self, create_rdep, device, expert_weights, seed):
        """Test load distribution per rank."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        n_local = 8
        n_total = n_local * world
        T = 512
        K = 2

        # Uniform random routing
        torch.manual_seed(seed)
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)

        # Count tokens per rank
        per_rank_loads = torch.zeros(world, device=device, dtype=torch.float32)
        for r in range(world):
            rank_start = r * n_local
            rank_end = rank_start + n_local
            for e in range(rank_start, rank_end):
                per_rank_loads[r] += (eid == e).sum().float()

        # All ranks should have similar loads
        mean_load = per_rank_loads.mean()
        max_deviation = (per_rank_loads - mean_load).abs().max()
        relative_deviation = max_deviation / mean_load

        # Allow up to 30% deviation
        assert relative_deviation < 0.30, \
            f"Per-rank load deviation too high: {relative_deviation:.3f}"

        dist.barrier()


# ==============================================================================
# Test Classes: Cross-GPU Gradient Flow
# ==============================================================================


@pytest.mark.multi_gpu
class TestCrossGPUGradientFlow:
    """Test gradient flow across GPUs."""

    def test_gradients_flow_to_source_tokens(self, create_rdep, device, expert_weights, seed):
        """Test that gradients flow back to source tokens."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        W1, W3, W2 = expert_weights

        W1 = W1.clone().requires_grad_(True)
        W3 = W3.clone().requires_grad_(True)
        W2 = W2.clone().requires_grad_(True)

        n_local = rdep.n_local
        n_total = n_local * world
        T = 64
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.1

        # Route to various experts across GPUs
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)
        gates = gates.requires_grad_(True)

        # Forward
        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Backward
        loss = output.float().sum()
        loss.backward()

        # Verify gradients exist
        assert x.grad is not None, f"Rank {rank}: x.grad is None"
        assert gates.grad is not None, f"Rank {rank}: gates.grad is None"
        assert W1.grad is not None, f"Rank {rank}: W1.grad is None"
        assert W3.grad is not None, f"Rank {rank}: W3.grad is None"
        assert W2.grad is not None, f"Rank {rank}: W2.grad is None"

        # Verify gradients are non-zero
        assert x.grad.abs().sum() > 0, f"Rank {rank}: x.grad is all zeros"
        assert gates.grad.abs().sum() > 0, f"Rank {rank}: gates.grad is all zeros"

        dist.barrier()

    def test_no_gradient_corruption_across_ranks(self, create_rdep, device, expert_weights, seed):
        """Test that gradients are not corrupted across ranks."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

        W1 = expert_weights[0].clone().requires_grad_(True)
        W3 = expert_weights[1].clone().requires_grad_(True)
        W2 = expert_weights[2].clone().requires_grad_(True)

        n_local = rdep.n_local
        n_total = n_local * world
        T = 32
        K = rdep.topk
        H = rdep.dim

        # Run multiple backward passes and check consistency
        gradient_norms = []

        for run in range(3):
            if W1.grad is not None:
                W1.grad.zero_()
            if W3.grad is not None:
                W3.grad.zero_()
            if W2.grad is not None:
                W2.grad.zero_()

            torch.manual_seed(seed + rank)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            loss = output.float().sum()
            loss.backward()

            grad_norm = W1.grad.norm().item()
            gradient_norms.append(grad_norm)

        # Gradient norms should be identical across runs (deterministic)
        for i in range(1, len(gradient_norms)):
            assert abs(gradient_norms[i] - gradient_norms[0]) < 1e-5, \
                f"Rank {rank}: Gradient norms differ: {gradient_norms}"

        dist.barrier()

    def test_gradient_scaling_by_gate_weights(self, create_rdep, device, expert_weights, seed):
        """Test that gradients are correctly scaled by gate weights."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

        n_local = rdep.n_local
        local_start = rank * n_local
        T = 32
        K = 1
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Route to first local expert
        eid = torch.full((T, K), local_start, device=device, dtype=torch.int32)

        # Test with gate=1.0
        W1_full = expert_weights[0].clone().requires_grad_(True)
        W3_full = expert_weights[1].clone().requires_grad_(True)
        W2_full = expert_weights[2].clone().requires_grad_(True)
        gates_full = torch.ones(T, K, device=device, dtype=torch.bfloat16)

        out_full = rdep.moe_bf16(x.clone(), eid, gates_full, W1_full, W3_full, W2_full)
        out_full.float().sum().backward()
        grad_norm_full = W1_full.grad.norm().item()

        # Test with gate=0.5
        W1_half = expert_weights[0].clone().requires_grad_(True)
        W3_half = expert_weights[1].clone().requires_grad_(True)
        W2_half = expert_weights[2].clone().requires_grad_(True)
        gates_half = torch.full((T, K), 0.5, device=device, dtype=torch.bfloat16)

        out_half = rdep.moe_bf16(x.clone(), eid, gates_half, W1_half, W3_half, W2_half)
        out_half.float().sum().backward()
        grad_norm_half = W1_half.grad.norm().item()

        # Gradient with gate=0.5 should be approximately half
        ratio = grad_norm_half / grad_norm_full if grad_norm_full > 0 else 0
        assert abs(ratio - 0.5) < 0.15, \
            f"Rank {rank}: Gradient scaling incorrect: ratio={ratio:.3f}, expected ~0.5"

        dist.barrier()

    def test_gradient_accumulation_multi_batch(self, create_rdep, device, expert_weights, seed):
        """Test gradient accumulation across multiple micro-batches."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

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

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            loss = output.float().sum() / 4  # Scale by accumulation steps
            loss.backward()

        # Gradients should be accumulated
        assert W1.grad is not None
        assert W1.grad.abs().sum() > 0

        dist.barrier()


# ==============================================================================
# Test Classes: EP Group Handling
# ==============================================================================


@pytest.mark.multi_gpu
class TestEPGroupHandling:
    """Test Expert Parallelism group handling."""

    def test_custom_ep_group(self, device, seed):
        """Test that custom EP groups work correctly."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 4:
            pytest.skip("Requires at least 4 GPUs")

        from nmoe.rdep import Rdep

        # Create EP group for first 4 ranks
        if world >= 8:
            # Split into two EP groups of 4
            if rank < 4:
                ep_ranks = list(range(4))
            else:
                ep_ranks = list(range(4, 8))
        else:
            ep_ranks = list(range(min(4, world)))

        ep_group = dist.new_group(ep_ranks)

        # Only participate if in this EP group
        if rank in ep_ranks:
            H = 256
            n_local = 4
            K = 2

            rdep = Rdep(
                dim=H,
                n_local=n_local,
                topk=K,
                profile='bf16',
                capacity=16384,
                ep_group=ep_group,
            )

            # Verify EP group is used
            assert rdep.ep_group is ep_group
            assert rdep.world == len(ep_ranks)

            # Create weights and run forward
            Dff = H * 4
            W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
            W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
            W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

            n_total = n_local * len(ep_ranks)
            T = 32

            torch.manual_seed(seed + rank)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()

        dist.barrier()

    def test_ep4_tp2_configuration(self, device, seed):
        """Test EP=4, TP=2 configuration."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs for EP=4, TP=2")

        from nmoe.rdep import Rdep

        # EP=4, TP=2 configuration
        # Ranks 0,1,2,3 form EP group 0
        # Ranks 4,5,6,7 form EP group 1
        # Within each EP group, TP=2 pairs: (0,1), (2,3), (4,5), (6,7)

        ep_size = 4
        tp_size = 2

        # Determine EP group
        ep_group_id = rank // ep_size
        ep_ranks = list(range(ep_group_id * ep_size, (ep_group_id + 1) * ep_size))
        ep_group = dist.new_group(ep_ranks)

        H = 256
        n_local = 2  # 2 experts per rank, 8 total per EP group
        K = 2

        rdep = Rdep(
            dim=H,
            n_local=n_local,
            topk=K,
            profile='bf16',
            capacity=16384,
            ep_group=ep_group,
        )

        # Verify configuration
        assert rdep.world == ep_size

        # Create weights and test
        Dff = H * 4
        W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

        n_total = n_local * ep_size
        T = 32

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        dist.barrier()

    def test_rank_to_expert_mapping(self, create_rdep, device, seed):
        """Test rank-to-expert mapping is correct."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()
        n_local = rdep.n_local

        # Each rank owns experts [rank * n_local, (rank + 1) * n_local)
        expected_start = rank * n_local
        expected_end = expected_start + n_local

        # Verify by routing only to "own" experts
        W1, W3, W2 = [], [], []
        H = rdep.dim
        Dff = H * 4

        W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

        T = 32
        K = 2

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Route only to this rank's experts
        eid = torch.randint(expected_start, expected_end, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # This should work without cross-rank communication
        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # Compare with reference (local-only)
        local_eid = eid - expected_start
        ref = reference_moe_forward(x, local_eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            output, ref,
            atol=5e-2, rtol=1e-1,
            msg=f"Rank {rank}: Rank-to-expert mapping incorrect"
        )

        dist.barrier()

    def test_multiple_ep_groups_isolation(self, device, seed):
        """Test that multiple EP groups are properly isolated."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        from nmoe.rdep import Rdep

        # Create two EP groups of 4 ranks each
        if rank < 4:
            ep_ranks = list(range(4))
            ep_group_id = 0
        else:
            ep_ranks = list(range(4, 8))
            ep_group_id = 1

        ep_group = dist.new_group(ep_ranks)

        H = 256
        n_local = 4
        K = 2
        n_total = n_local * 4  # 4 ranks per EP group

        rdep = Rdep(
            dim=H,
            n_local=n_local,
            topk=K,
            profile='bf16',
            capacity=16384,
            ep_group=ep_group,
        )

        # Create unique weights per EP group
        Dff = H * 4
        torch.manual_seed(seed + ep_group_id * 1000)
        W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

        T = 32

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # Gather outputs to verify isolation
        output_sum = output.sum().item()
        all_sums = all_gather_object(output_sum)

        # Ranks in same EP group should have similar statistics
        # (due to same weights), different EP groups should differ
        ep0_sums = [all_sums[r] for r in range(4)]
        ep1_sums = [all_sums[r] for r in range(4, 8)]

        # EP groups use different weight seeds, so sums should be meaningfully different
        ep0_mean = sum(ep0_sums) / len(ep0_sums)
        ep1_mean = sum(ep1_sums) / len(ep1_sums)

        # Not a strict test - just verify no obvious cross-contamination
        assert ep0_mean != ep1_mean or abs(ep0_mean) < 1e-6, \
            "EP groups should produce different outputs with different weights"

        dist.barrier()


# ==============================================================================
# Test Classes: Determinism
# ==============================================================================


@pytest.mark.multi_gpu
class TestDeterminism:
    """Test determinism of cross-GPU routing."""

    def test_same_routing_with_same_seed(self, create_rdep, device, expert_weights, seed):
        """Test that same routing produces same results with same seed."""
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

        # Run twice with same seed
        results = []
        for _ in range(2):
            torch.manual_seed(seed + rank)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            results.append(output.clone())

        torch.testing.assert_close(
            results[0], results[1],
            atol=0, rtol=0,
            msg=f"Rank {rank}: Results differ with same seed"
        )

        dist.barrier()

    def test_reproducible_across_restarts(self, create_rdep, device, expert_weights, seed):
        """Test reproducibility across simulated restarts."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        n_local = 8
        n_total = n_local * world
        H = 256
        T = 32
        K = 2

        results = []

        # Simulate multiple "restarts" with fresh RDEP instances
        for restart in range(3):
            from nmoe.rdep import Rdep

            rdep = Rdep(
                dim=H,
                n_local=n_local,
                topk=K,
                profile='bf16',
                capacity=16384,
            )

            Dff = H * 4
            torch.manual_seed(seed)
            W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
            W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
            W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

            torch.manual_seed(seed + rank)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            results.append(output.clone())

        # All restarts should produce identical results
        for i in range(1, len(results)):
            torch.testing.assert_close(
                results[0], results[i],
                atol=0, rtol=0,
                msg=f"Rank {rank}: Restart {i} differs from restart 0"
            )

        dist.barrier()

    def test_deterministic_backward(self, create_rdep, device, expert_weights, seed):
        """Test that backward pass is deterministic."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        rdep = create_rdep()

        n_local = rdep.n_local
        n_total = n_local * world
        T = 32
        K = rdep.topk
        H = rdep.dim

        gradient_results = []

        for _ in range(2):
            W1 = expert_weights[0].clone().requires_grad_(True)
            W3 = expert_weights[1].clone().requires_grad_(True)
            W2 = expert_weights[2].clone().requires_grad_(True)

            torch.manual_seed(seed + rank)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
            loss = output.float().sum()
            loss.backward()

            gradient_results.append({
                'x_grad': x.grad.clone(),
                'W1_grad': W1.grad.clone(),
                'W3_grad': W3.grad.clone(),
                'W2_grad': W2.grad.clone(),
            })

        # Compare gradients
        for key in ['x_grad', 'W1_grad', 'W3_grad', 'W2_grad']:
            torch.testing.assert_close(
                gradient_results[0][key],
                gradient_results[1][key],
                atol=0, rtol=0,
                msg=f"Rank {rank}: {key} differs between runs"
            )

        dist.barrier()

    def test_deterministic_with_varying_routing(self, create_rdep, device, expert_weights, seed):
        """Test determinism with varying routing patterns."""
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
        H = rdep.dim
        K = rdep.topk

        # Test with different routing patterns
        patterns = [
            ("all_local", lambda r, n, t, k: torch.randint(r * n_local, (r + 1) * n_local, (t, k))),
            ("all_remote", lambda r, n, t, k: torch.randint(((r + 1) % world) * n_local, ((r + 2) % world) * n_local, (t, k))),
            ("uniform", lambda r, n, t, k: torch.randint(0, n, (t, k))),
        ]

        for pattern_name, pattern_fn in patterns:
            results = []
            for _ in range(2):
                torch.manual_seed(seed + rank)
                x = torch.randn(32, H, device=device, dtype=torch.bfloat16) * 0.1

                torch.manual_seed(seed + rank + 1000)  # Different seed for routing
                eid = pattern_fn(rank, n_total, 32, K).to(device=device, dtype=torch.int32)
                gates = F.softmax(torch.randn(32, K, device=device), dim=-1).to(torch.bfloat16)

                output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
                results.append(output.clone())

            torch.testing.assert_close(
                results[0], results[1],
                atol=0, rtol=0,
                msg=f"Rank {rank}: Pattern '{pattern_name}' not deterministic"
            )

        dist.barrier()


# ==============================================================================
# Test Classes: Single-GPU Reference Comparison
# ==============================================================================


@pytest.mark.multi_gpu
class TestSingleGPUReference:
    """Compare distributed results against single-GPU reference."""

    def test_local_routing_matches_reference(self, create_rdep, device, expert_weights, seed):
        """Test that local-only routing matches single-GPU reference."""
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
        T = 64
        K = rdep.topk
        H = rdep.dim

        torch.manual_seed(seed + rank)
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

        # Route only to local experts
        eid = torch.randint(local_start, local_start + n_local, (T, K), device=device, dtype=torch.int32)
        gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

        # Distributed result
        dist_output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        # Reference (local indices)
        local_eid = eid - local_start
        ref_output = reference_moe_forward(x, local_eid, gates, W1, W3, W2)

        torch.testing.assert_close(
            dist_output, ref_output,
            atol=5e-2, rtol=1e-1,
            msg=f"Rank {rank}: Local routing does not match reference"
        )

        dist.barrier()

    def test_various_ep_configurations(self, device, seed):
        """Test various EP configurations."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        from nmoe.rdep import Rdep

        H = 256
        Dff = H * 4

        # Test configurations: (ep_size, n_local)
        configs = [
            (8, 8),   # EP=8, 8 local experts
            (4, 4),   # EP=4, 4 local experts (if rank < 4)
            (2, 16),  # EP=2, 16 local experts (if rank < 2)
        ]

        for ep_size, n_local in configs:
            if rank >= ep_size:
                continue

            ep_ranks = list(range(ep_size))
            ep_group = dist.new_group(ep_ranks)

            rdep = Rdep(
                dim=H,
                n_local=n_local,
                topk=2,
                profile='bf16',
                capacity=32768,
                ep_group=ep_group,
            )

            assert rdep.world == ep_size

            # Create weights
            torch.manual_seed(seed)
            W1 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
            W3 = torch.randn(n_local, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
            W2 = torch.randn(n_local, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

            n_total = n_local * ep_size
            T = 32
            K = 2

            torch.manual_seed(seed + rank)
            x = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
            eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)
            gates = F.softmax(torch.randn(T, K, device=device), dim=-1).to(torch.bfloat16)

            output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

            assert output.shape == x.shape, f"Config EP={ep_size}, n_local={n_local}: wrong shape"
            assert not torch.isnan(output).any(), f"Config EP={ep_size}, n_local={n_local}: NaN"

        dist.barrier()


# ==============================================================================
# Test Classes: Load Stats Correctness
# ==============================================================================


@pytest.mark.multi_gpu
class TestLoadStatsCorrectness:
    """Test that load statistics are correct."""

    def test_load_stats_sum_to_total(self, create_rdep, device, seed):
        """Test that load stats sum to total token-expert assignments."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        from nmoe.model import MoE
        from nmoe.config import Config
        from nmoe.rdep import Rdep

        n_local = 8
        n_total = n_local * world
        T = 128
        K = 2

        cfg = Config(
            dim=256,
            inter_dim=1024,
            moe_inter_dim=1024,
            n_routed_experts=n_total,
            n_activated_experts=K,
            n_layers=1,
            n_heads=4,
            batch_size=8,
            seq_len=128,
            dtype='bf16',
        )

        rdep = Rdep(
            dim=cfg.dim,
            n_local=n_local,
            topk=K,
            profile='bf16',
            capacity=32768,
        )

        moe = MoE(cfg, layer_id=0, rdep=rdep).cuda()
        moe.init_weights()

        torch.manual_seed(seed + rank)
        x = torch.randn(T, cfg.dim, device=device, dtype=torch.bfloat16) * 0.1

        _ = moe(x)

        # Verify load sum
        expected_total = T * K
        actual_total = moe.last_loads.sum().item()
        assert actual_total == expected_total, \
            f"Load sum {actual_total} != expected {expected_total}"

        dist.barrier()

    def test_load_stats_per_expert(self, create_rdep, device, seed):
        """Test per-expert load statistics."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        n_local = 8
        n_total = n_local * world
        T = 256
        K = 2

        # Generate routing with known distribution
        torch.manual_seed(seed)
        eid = torch.randint(0, n_total, (T, K), device=device, dtype=torch.int32)

        # Compute expected loads
        expected_loads = torch.bincount(
            eid.reshape(-1).long(),
            minlength=n_total
        ).float()

        # Verify using compute_load_stats
        computed_loads, mean, cv = compute_load_stats(eid, n_total)

        torch.testing.assert_close(
            computed_loads, expected_loads,
            atol=0, rtol=0,
            msg="Per-expert loads do not match expected"
        )

        dist.barrier()

    def test_load_stats_with_skewed_routing(self, create_rdep, device, seed):
        """Test load stats accuracy with skewed routing."""
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")
        rank = get_rank()
        world = get_world_size()
        if world < 8:
            pytest.skip("Requires 8 GPUs")

        from nmoe.model import MoE
        from nmoe.config import Config
        from nmoe.rdep import Rdep

        n_local = 8
        n_total = n_local * world
        T = 100
        K = 2

        cfg = Config(
            dim=256,
            inter_dim=1024,
            moe_inter_dim=1024,
            n_routed_experts=n_total,
            n_activated_experts=K,
            n_layers=1,
            n_heads=4,
            batch_size=8,
            seq_len=128,
            dtype='bf16',
        )

        rdep = Rdep(
            dim=cfg.dim,
            n_local=n_local,
            topk=K,
            profile='bf16',
            capacity=32768,
        )

        moe = MoE(cfg, layer_id=0, rdep=rdep).cuda()
        moe.init_weights()

        # Create input that will route to specific experts
        # Use router directly to get actual routing
        torch.manual_seed(seed + rank)
        x = torch.randn(T, cfg.dim, device=device, dtype=torch.bfloat16) * 0.1

        _ = moe(x)

        # Check that load stats are valid
        loads = moe.last_loads
        assert loads is not None
        assert loads.shape[0] == n_total
        assert loads.sum().item() == T * K
        assert (loads >= 0).all(), "Loads should be non-negative"

        dist.barrier()


# ==============================================================================
# Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
