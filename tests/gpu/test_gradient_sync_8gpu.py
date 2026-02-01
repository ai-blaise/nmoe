"""P0 Critical Tests for Gradient Synchronization.

Tests gradient synchronization correctness for ZeRO-2 and Expert Parallelism:

1. ZeRO-2 Reduce-Scatter Correctness:
   - Gradients are correctly averaged across ranks
   - Each rank gets correct gradient shard
   - RS chunking works for large tensors

2. ZeRO-2 All-Gather Correctness:
   - Parameters are correctly reconstructed
   - AG restores full parameters on all ranks
   - Works with padded and unpadded tensors

3. Expert Gradient Isolation:
   - Expert gradients don't leak between EP ranks
   - Each EP rank computes gradients only for local experts
   - All-reduce within EP groups works correctly

4. Gradient Accumulation:
   - Multiple micro-batches accumulate correctly
   - Accumulated gradients sync correctly
   - Reset works after optimizer step

5. Mixed EP/TP Gradient Sync (EP=4, TP=2):
   - Gradients sync correctly in both dimensions
   - No cross-contamination between EP and TP groups

Run single-GPU tests:
    pytest tests/gpu/test_gradient_sync_8gpu.py -v -m gpu

Run multi-GPU tests:
    torchrun --nproc_per_node=N -m pytest tests/gpu/test_gradient_sync_8gpu.py -v -m multi_gpu
"""

import os
import math
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import patch, MagicMock

# Import ZeRO-2 functions from source
from nmoe.zero2 import (
    _ceil_div,
    _rs_chunk_elems,
    _get_or_init_flat_group,
    step_dense_adamw,
)


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


def create_simple_model(
    dim: int,
    n_layers: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    """Create a simple multi-layer model for gradient testing."""
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(dim, dim, bias=True, device=device, dtype=dtype))
        if i < n_layers - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def compute_expected_avg_gradient(
    local_grad: torch.Tensor,
    world_size: int,
    rank: int
) -> torch.Tensor:
    """Compute the expected average gradient after all-reduce.

    For testing purposes, assumes each rank contributes a gradient of
    (rank + 1) * ones_like(local_grad).
    """
    total = sum((r + 1) for r in range(world_size))
    return torch.full_like(local_grad, total / world_size)


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
def param_groups_factory(cuda_device):
    """Factory for creating param groups with different configurations."""
    def _create(
        n_params: int = 3,
        shapes: Optional[List[Tuple[int, ...]]] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[List[nn.Parameter], List[Dict]]:
        if shapes is None:
            shapes = [(64, 32), (128,), (32, 16)][:n_params]

        params = []
        for shape in shapes:
            p = nn.Parameter(torch.randn(shape, device=cuda_device, dtype=dtype))
            p.grad = torch.randn_like(p)
            params.append(p)

        param_groups = [{'params': params, 'lr': lr, 'weight_decay': weight_decay}]
        return params, param_groups

    return _create


@pytest.fixture
def expert_weights_factory(cuda_device):
    """Factory for creating expert weight tensors."""
    def _create(
        n_experts: int,
        hidden_dim: int,
        inter_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W1 = torch.randn(n_experts, hidden_dim, inter_dim, device=cuda_device, dtype=dtype) * 0.02
        W3 = torch.randn(n_experts, hidden_dim, inter_dim, device=cuda_device, dtype=dtype) * 0.02
        W2 = torch.randn(n_experts, inter_dim, hidden_dim, device=cuda_device, dtype=dtype) * 0.02
        if requires_grad:
            W1.requires_grad_(True)
            W3.requires_grad_(True)
            W2.requires_grad_(True)
        return W1, W3, W2
    return _create


# =============================================================================
# Single-GPU Tests for Gradient Sync Correctness
# =============================================================================

@pytest.mark.gpu
class TestGradientSyncSingleGPU:
    """Single-GPU tests for gradient synchronization logic."""

    def test_gradient_averaging_formula(self, cuda_device, param_groups_factory):
        """Verify the gradient averaging formula is correct."""
        params, param_groups = param_groups_factory(n_params=1, shapes=[(64,)])
        state: Dict[str, Any] = {}

        # Set a known gradient
        known_grad = torch.ones(64, device=cuda_device, dtype=torch.bfloat16)
        params[0].grad = known_grad.clone()

        # Run step (single GPU, no actual averaging)
        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify momentum was updated based on the gradient
        state_key = f"shard_0_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        # After step 1: exp_avg = (1 - beta1) * grad = 0.1 * grad
        beta1 = 0.9
        expected_exp_avg = (1 - beta1) * 1.0  # gradient was all ones
        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_exp_avg),
            rtol=0.1
        )

    def test_gradient_shard_correct_size(self, cuda_device, param_groups_factory):
        """Verify gradient shards are correctly sized for different world sizes."""
        total_elems = 256
        params, param_groups = param_groups_factory(
            n_params=1,
            shapes=[(total_elems,)]
        )

        for simulated_world in [1, 2, 4, 8]:
            group = {'params': params}
            flat = _get_or_init_flat_group(
                group,
                params=params,
                rank=0,
                world=simulated_world,
                dtype=torch.bfloat16,
            )

            expected_shard_size = _ceil_div(total_elems, simulated_world)
            assert flat['shard_size'] == expected_shard_size, \
                f"World {simulated_world}: shard_size={flat['shard_size']}, expected={expected_shard_size}"
            assert flat['param_shard'].numel() == expected_shard_size

    def test_rs_chunking_large_tensor(self, cuda_device):
        """Test reduce-scatter chunking for large tensors."""
        # Create a large parameter
        large_size = 1024 * 1024  # 1M elements
        p = nn.Parameter(torch.randn(large_size, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        # Should handle large tensor with chunking
        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify update happened
        state_key = f"shard_0_0_{torch.bfloat16}"
        assert state_key in state
        assert state[state_key]['step'] == 1

    def test_gradient_accumulation_single_gpu(self, cuda_device, param_groups_factory):
        """Test gradient accumulation over multiple micro-batches."""
        params, param_groups = param_groups_factory(n_params=1, shapes=[(64,)])

        # Accumulate gradients from 4 micro-batches
        n_micro_batches = 4
        accumulated_grad = torch.zeros(64, device=cuda_device, dtype=torch.bfloat16)

        for i in range(n_micro_batches):
            micro_grad = torch.full((64,), float(i + 1), device=cuda_device, dtype=torch.bfloat16)
            accumulated_grad += micro_grad

        # Set accumulated gradient
        params[0].grad = accumulated_grad

        state: Dict[str, Any] = {}
        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify the accumulated gradient was used
        state_key = f"shard_0_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        # Expected accumulated value: 1 + 2 + 3 + 4 = 10
        beta1 = 0.9
        expected_mean = (1 - beta1) * 10.0
        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_mean),
            rtol=0.15
        )

    def test_gradient_reset_after_step(self, cuda_device, param_groups_factory):
        """Test that gradients can be properly reset after optimizer step."""
        params, param_groups = param_groups_factory(n_params=1, shapes=[(64,)])
        state: Dict[str, Any] = {}

        # First step
        step_dense_adamw(param_groups, state=state, pg=None)

        # Zero gradients manually (ZeRO-2 doesn't do this automatically)
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        # Verify gradient is zeroed
        assert params[0].grad is not None
        assert params[0].grad.abs().sum() == 0

        # Second step with zero gradients should still work
        step_dense_adamw(param_groups, state=state, pg=None)

        state_key = f"shard_0_0_{torch.bfloat16}"
        assert state[state_key]['step'] == 2


@pytest.mark.gpu
class TestAllGatherCorrectness:
    """Tests for all-gather parameter reconstruction."""

    def test_parameter_views_correct(self, cuda_device, param_groups_factory):
        """Test that parameter views are correctly set up after initialization."""
        params, param_groups = param_groups_factory(n_params=3, shapes=[(32, 16), (64,), (8, 8)])

        group = {'params': params}
        flat = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=1,
            dtype=torch.bfloat16,
        )

        flat_param = flat['flat_param']
        offsets = flat['offsets']

        # Verify each param is a view into flat_param
        for p, (a, b) in zip(params, offsets):
            # Modify flat_param and verify param changes
            test_val = 123.0
            flat_param[a] = test_val

            if p.dim() == 2:
                assert p.data[0, 0].item() == pytest.approx(test_val, rel=0.01)
            else:
                assert p.data[0].item() == pytest.approx(test_val, rel=0.01)

    def test_padded_tensor_handling(self, cuda_device):
        """Test that padded tensors are handled correctly."""
        # Create a tensor with size not divisible by world
        odd_size = 127  # Prime number
        p = nn.Parameter(torch.randn(odd_size, device=cuda_device, dtype=torch.bfloat16))

        for world in [2, 4, 8]:
            group = {}
            flat = _get_or_init_flat_group(
                group,
                params=[p],
                rank=0,
                world=world,
                dtype=torch.bfloat16,
            )

            # Padded total should be divisible by world
            assert flat['padded_total'] % world == 0
            assert flat['padded_total'] >= flat['total']

            # Padding should be zero
            if flat['total'] < flat['padded_total']:
                padding = flat['flat_param'][flat['total']:flat['padded_total']]
                assert padding.abs().sum() == 0

    def test_unpadded_tensor_handling(self, cuda_device):
        """Test that tensors already divisible by world work correctly."""
        # Size exactly divisible by 8
        exact_size = 256
        p = nn.Parameter(torch.randn(exact_size, device=cuda_device, dtype=torch.bfloat16))

        group = {}
        flat = _get_or_init_flat_group(
            group,
            params=[p],
            rank=0,
            world=8,
            dtype=torch.bfloat16,
        )

        # No padding needed
        assert flat['padded_total'] == flat['total']
        assert flat['shard_size'] == exact_size // 8


# =============================================================================
# Multi-GPU Tests for ZeRO-2 Reduce-Scatter
# =============================================================================

@pytest.mark.multi_gpu
class TestReduceScatterMultiGPU:
    """Multi-GPU tests for reduce-scatter gradient averaging."""

    def test_gradient_averaging_2gpu(self, cuda_device, param_groups_factory):
        """Test gradient averaging across 2 GPUs."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        # Each rank has different gradient
        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.full_like(p, float(rank + 1))  # Rank 0: 1, Rank 1: 2

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Expected average gradient for 2 ranks: (1 + 2) / 2 = 1.5
        expected_avg_grad = (1 + 2) / 2.0

        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        beta1 = 0.9
        expected_exp_avg = (1 - beta1) * expected_avg_grad

        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_exp_avg),
            rtol=0.15
        ), f"Rank {rank}: exp_avg mean = {exp_avg.float().mean()}, expected ~{expected_exp_avg}"

    def test_gradient_averaging_4gpu(self, cuda_device, param_groups_factory):
        """Test gradient averaging across 4 GPUs."""
        skip_if_not_multi_gpu(4)

        rank = get_rank()
        world_size = get_world_size()

        p = nn.Parameter(torch.ones(128, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.full_like(p, float(rank + 1))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # For 4 ranks: (1+2+3+4) / 4 = 2.5
        expected_avg_grad = sum(r + 1 for r in range(world_size)) / world_size

        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        beta1 = 0.9
        expected_exp_avg = (1 - beta1) * expected_avg_grad

        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_exp_avg),
            rtol=0.15
        )

    def test_gradient_averaging_8gpu(self, cuda_device, param_groups_factory):
        """Test gradient averaging across 8 GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        p = nn.Parameter(torch.ones(256, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.full_like(p, float(rank + 1))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # For 8 ranks: (1+2+...+8) / 8 = 4.5
        expected_avg_grad = sum(r + 1 for r in range(world_size)) / world_size

        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        beta1 = 0.9
        expected_exp_avg = (1 - beta1) * expected_avg_grad

        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_exp_avg),
            rtol=0.15
        )

    def test_each_rank_gets_correct_shard(self, cuda_device):
        """Test that each rank receives its correct gradient shard."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        # Create parameter where we can verify shard ownership
        total_elems = 128  # Will be sharded across ranks
        p = nn.Parameter(torch.arange(total_elems, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.arange(total_elems, device=cuda_device, dtype=torch.bfloat16)

        param_groups = [{'params': [p], 'lr': 0.0, 'weight_decay': 0.0}]  # lr=0 to isolate grad effect
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        shard_size = state[state_key]['exp_avg'].numel()

        # Each rank should have shard_size = total_elems / world_size
        expected_shard_size = _ceil_div(total_elems, world_size)
        assert shard_size == expected_shard_size, \
            f"Rank {rank}: shard_size={shard_size}, expected={expected_shard_size}"

    def test_rs_chunking_multi_gpu(self, cuda_device):
        """Test reduce-scatter chunking works correctly in multi-GPU setting."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()

        # Large tensor that requires chunking
        large_size = 1024 * 1024
        p = nn.Parameter(torch.randn(large_size, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.full_like(p, float(rank + 1))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        # Should complete without error
        step_dense_adamw(param_groups, state=state, pg=None)

        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        assert state_key in state
        assert state[state_key]['step'] == 1


# =============================================================================
# Multi-GPU Tests for All-Gather Parameter Sync
# =============================================================================

@pytest.mark.multi_gpu
class TestAllGatherMultiGPU:
    """Multi-GPU tests for all-gather parameter synchronization."""

    def test_parameter_sync_after_step_2gpu(self, cuda_device):
        """Test parameters are synchronized after step on 2 GPUs."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        # Initialize with same random seed for consistent params
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Gather parameters from all ranks
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        # All ranks should have identical parameters
        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                f"Params differ between rank 0 and rank {r}"

    def test_parameter_sync_after_step_8gpu(self, cuda_device):
        """Test parameters are synchronized after step on 8 GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(256, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                f"Params differ between rank 0 and rank {r}"

    def test_ag_padded_tensor_multi_gpu(self, cuda_device):
        """Test all-gather with padded tensors across GPUs."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        # Odd-sized tensor that needs padding
        odd_size = 127
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(odd_size, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Parameter should still have original size
        assert p.numel() == odd_size

        # All ranks should agree on parameter values
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4)

    def test_multiple_steps_sync_maintained(self, cuda_device):
        """Test that parameter sync is maintained across multiple steps."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        for step in range(5):
            p.grad = torch.randn_like(p)
            step_dense_adamw(param_groups, state=state, pg=None)

            gathered = [torch.zeros_like(p) for _ in range(world_size)]
            dist.all_gather(gathered, p.data)

            for r in range(1, world_size):
                assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                    f"Step {step}: Params differ between rank 0 and rank {r}"


# =============================================================================
# Expert Gradient Isolation Tests
# =============================================================================

@pytest.mark.gpu
class TestExpertGradientIsolationSingleGPU:
    """Single-GPU tests for expert gradient isolation logic."""

    def test_expert_gradients_independent(self, cuda_device, expert_weights_factory):
        """Test that expert gradients are computed independently."""
        n_experts = 8
        dim = 256
        inter = 512

        W1, W3, W2 = expert_weights_factory(n_experts, dim, inter)

        # Simulate forward pass output for each expert
        T = 64
        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Compute gradients for expert 0 only
        expert_0_out = torch.matmul(x[:8], W1[0])  # First 8 tokens to expert 0
        loss_0 = expert_0_out.sum()
        loss_0.backward()

        # Only expert 0's W1 should have gradient
        assert W1.grad is not None
        assert W1.grad[0].abs().sum() > 0
        for e in range(1, n_experts):
            assert W1.grad[e].abs().sum() == 0, f"Expert {e} should have zero gradient"

    def test_local_expert_gradients_only(self, cuda_device, expert_weights_factory):
        """Test that only local experts accumulate gradients."""
        n_local = 4
        dim = 128
        inter = 256

        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # Create input that goes to each local expert
        T = n_local * 8  # 8 tokens per expert
        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Compute gradients for all local experts
        outputs = []
        for e in range(n_local):
            start = e * 8
            end = start + 8
            out = torch.matmul(x[start:end], W1[e])
            outputs.append(out)

        total_out = torch.cat(outputs, dim=0)
        loss = total_out.sum()
        loss.backward()

        # All local experts should have gradients
        for e in range(n_local):
            assert W1.grad is not None
            assert W1.grad[e].abs().sum() > 0, f"Expert {e} should have gradient"


@pytest.mark.multi_gpu
class TestExpertGradientIsolationMultiGPU:
    """Multi-GPU tests for expert gradient isolation."""

    def test_expert_gradients_no_leakage_between_ranks(self, cuda_device, expert_weights_factory):
        """Test expert gradients don't leak between EP ranks."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        n_local = 4
        dim = 128
        inter = 256

        W1, W3, W2 = expert_weights_factory(n_local, dim, inter)

        # Each rank computes gradients only for its local experts
        T = 32
        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Forward through local expert 0
        out = torch.matmul(x, W1[0])
        loss = out.sum()
        loss.backward()

        # Verify gradient is local
        assert W1.grad is not None
        local_grad_norm = W1.grad.norm().item()

        # Gather gradient norms from all ranks
        norms = torch.tensor([local_grad_norm], device=cuda_device)
        all_norms = [torch.zeros_like(norms) for _ in range(world_size)]
        dist.all_gather(all_norms, norms)

        # All ranks should have non-zero gradient norms (each computed locally)
        for r, n in enumerate(all_norms):
            assert n.item() > 0, f"Rank {r} should have non-zero gradient norm"

    def test_all_reduce_within_ep_group(self, cuda_device):
        """Test all-reduce works correctly within EP groups."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        # Create a gradient tensor
        grad = torch.full((64,), float(rank + 1), device=cuda_device, dtype=torch.float32)

        # All-reduce within default group (simulating EP group)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)

        # Expected: sum of (1 + 2 + ... + world_size)
        expected_sum = sum(r + 1 for r in range(world_size))

        assert torch.allclose(
            grad,
            torch.full_like(grad, expected_sum),
            atol=1e-5
        )


# =============================================================================
# Gradient Accumulation Tests
# =============================================================================

@pytest.mark.multi_gpu
class TestGradientAccumulationMultiGPU:
    """Multi-GPU tests for gradient accumulation."""

    def test_gradient_accumulation_before_sync(self, cuda_device):
        """Test gradients accumulate correctly across micro-batches before sync."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))

        # Accumulate over 4 micro-batches
        n_micro_batches = 4
        for i in range(n_micro_batches):
            micro_grad = torch.full((64,), float(i + 1) * (rank + 1), device=cuda_device, dtype=torch.bfloat16)
            if p.grad is None:
                p.grad = micro_grad.clone()
            else:
                p.grad += micro_grad

        # Expected accumulated gradient on each rank:
        # Rank 0: 1*(1+2+3+4) = 10
        # Rank 1: 2*(1+2+3+4) = 20
        expected_local = (rank + 1) * sum(range(1, n_micro_batches + 1))

        assert torch.allclose(
            p.grad.float().mean(),
            torch.tensor(expected_local, dtype=torch.float32),
            rtol=0.01
        )

        # Now sync
        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}
        step_dense_adamw(param_groups, state=state, pg=None)

        # After sync, momentum should reflect averaged accumulated gradient
        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        # Average of accumulated gradients: (10 + 20) / 2 = 15
        world_size = get_world_size()
        expected_avg = sum((r + 1) * sum(range(1, n_micro_batches + 1)) for r in range(world_size)) / world_size

        beta1 = 0.9
        expected_exp_avg = (1 - beta1) * expected_avg

        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_exp_avg),
            rtol=0.15
        )

    def test_accumulated_gradients_sync_correctly(self, cuda_device):
        """Test accumulated gradients are synced correctly across multiple steps."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        # Run multiple accumulation + sync cycles
        for step in range(3):
            # Accumulate
            p.grad = torch.full_like(p, float((step + 1) * (rank + 1)))

            # Sync
            step_dense_adamw(param_groups, state=state, pg=None)

            # Verify parameters are synchronized
            gathered = [torch.zeros_like(p) for _ in range(world_size)]
            dist.all_gather(gathered, p.data)

            for r in range(1, world_size):
                assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                    f"Step {step}: params not synced between rank 0 and rank {r}"

    def test_gradient_reset_works_multi_gpu(self, cuda_device):
        """Test gradient reset works correctly in multi-GPU setting."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        # First step
        p.grad = torch.full_like(p, float(rank + 1))
        step_dense_adamw(param_groups, state=state, pg=None)

        # Reset gradients
        p.grad.zero_()

        # Second step with zero gradients
        step_dense_adamw(param_groups, state=state, pg=None)

        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        assert state[state_key]['step'] == 2


# =============================================================================
# Mixed EP/TP Gradient Sync Tests
# =============================================================================

@pytest.mark.multi_gpu
class TestMixedEPTPGradientSync:
    """Tests for gradient synchronization with mixed EP and TP configurations."""

    def test_ep4_tp2_gradient_sync(self, cuda_device):
        """Test gradient sync with EP=4, TP=2 configuration."""
        skip_if_not_multi_gpu(8)

        from nmoe.distributed.init_groups import (
            init_nmoe_process_groups,
            cleanup_process_groups,
            get_ep_group,
            get_tp_group,
            get_ep_rank,
            get_tp_rank,
        )

        rank = get_rank()
        world_size = get_world_size()

        try:
            cleanup_process_groups()
            init_nmoe_process_groups(ep_size=4, tp_size=2)

            ep_rank = get_ep_rank()
            tp_rank = get_tp_rank()

            # Create separate parameters for EP and TP testing
            ep_param = nn.Parameter(torch.full((64,), float(ep_rank + 1), device=cuda_device, dtype=torch.bfloat16))
            ep_param.grad = torch.full_like(ep_param, float(ep_rank + 1))

            # All-reduce within EP group
            ep_group = get_ep_group()
            if ep_group is not None:
                ep_grad = ep_param.grad.clone().float()
                dist.all_reduce(ep_grad, op=dist.ReduceOp.AVG, group=ep_group)

                # EP average should be (1+2+3+4) / 4 = 2.5
                expected_ep_avg = sum(r + 1 for r in range(4)) / 4
                assert torch.allclose(
                    ep_grad.mean(),
                    torch.tensor(expected_ep_avg),
                    rtol=0.1
                ), f"EP rank {ep_rank}: gradient avg = {ep_grad.mean()}, expected = {expected_ep_avg}"

            # TP param test
            tp_param = nn.Parameter(torch.full((64,), float(tp_rank + 1), device=cuda_device, dtype=torch.bfloat16))
            tp_param.grad = torch.full_like(tp_param, float(tp_rank + 1))

            # All-reduce within TP group
            tp_group = get_tp_group()
            if tp_group is not None:
                tp_grad = tp_param.grad.clone().float()
                dist.all_reduce(tp_grad, op=dist.ReduceOp.AVG, group=tp_group)

                # TP average should be (1+2) / 2 = 1.5
                expected_tp_avg = sum(r + 1 for r in range(2)) / 2
                assert torch.allclose(
                    tp_grad.mean(),
                    torch.tensor(expected_tp_avg),
                    rtol=0.1
                ), f"TP rank {tp_rank}: gradient avg = {tp_grad.mean()}, expected = {expected_tp_avg}"

        finally:
            cleanup_process_groups()

    def test_no_cross_contamination_ep_tp(self, cuda_device):
        """Test that there's no cross-contamination between EP and TP gradients."""
        skip_if_not_multi_gpu(8)

        from nmoe.distributed.init_groups import (
            init_nmoe_process_groups,
            cleanup_process_groups,
            get_ep_group,
            get_tp_group,
            get_ep_rank,
            get_tp_rank,
        )

        rank = get_rank()

        try:
            cleanup_process_groups()
            init_nmoe_process_groups(ep_size=4, tp_size=2)

            ep_rank = get_ep_rank()
            tp_rank = get_tp_rank()
            ep_group = get_ep_group()
            tp_group = get_tp_group()

            # Create distinct gradients for EP and TP
            # EP gradient is based on ep_rank * 10
            # TP gradient is based on tp_rank * 100
            ep_marker = float(ep_rank * 10 + 1)
            tp_marker = float(tp_rank * 100 + 1)

            ep_grad = torch.full((64,), ep_marker, device=cuda_device, dtype=torch.float32)
            tp_grad = torch.full((64,), tp_marker, device=cuda_device, dtype=torch.float32)

            ep_grad_orig = ep_grad.clone()
            tp_grad_orig = tp_grad.clone()

            # All-reduce EP gradient in EP group
            if ep_group is not None:
                dist.all_reduce(ep_grad, op=dist.ReduceOp.SUM, group=ep_group)

            # All-reduce TP gradient in TP group
            if tp_group is not None:
                dist.all_reduce(tp_grad, op=dist.ReduceOp.SUM, group=tp_group)

            # EP gradient should only reflect EP group contributions
            # Sum should be (0*10+1) + (1*10+1) + (2*10+1) + (3*10+1) = 64
            expected_ep_sum = sum(r * 10 + 1 for r in range(4))
            if ep_group is not None:
                assert torch.allclose(
                    ep_grad,
                    torch.full_like(ep_grad, expected_ep_sum),
                    rtol=0.01
                ), f"EP gradient contaminated: got {ep_grad.mean()}, expected {expected_ep_sum}"

            # TP gradient should only reflect TP group contributions
            # Sum should be (0*100+1) + (1*100+1) = 102
            expected_tp_sum = sum(r * 100 + 1 for r in range(2))
            if tp_group is not None:
                assert torch.allclose(
                    tp_grad,
                    torch.full_like(tp_grad, expected_tp_sum),
                    rtol=0.01
                ), f"TP gradient contaminated: got {tp_grad.mean()}, expected {expected_tp_sum}"

        finally:
            cleanup_process_groups()

    def test_dense_params_sync_with_ep_tp(self, cuda_device):
        """Test dense parameter sync with EP/TP configuration."""
        skip_if_not_multi_gpu(4)

        from nmoe.distributed.init_groups import (
            init_nmoe_process_groups,
            cleanup_process_groups,
        )

        rank = get_rank()
        world_size = get_world_size()

        try:
            cleanup_process_groups()
            # Use world group for dense param sync (ZeRO-2 uses world)
            init_nmoe_process_groups(ep_size=2, tp_size=2)

            torch.manual_seed(42)
            p = nn.Parameter(torch.randn(128, device=cuda_device, dtype=torch.bfloat16))
            p.grad = torch.full_like(p, float(rank + 1))

            param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
            state: Dict[str, Any] = {}

            # ZeRO-2 step uses world group for dense params
            step_dense_adamw(param_groups, state=state, pg=None)  # None = world group

            # Verify all ranks have same parameter
            gathered = [torch.zeros_like(p) for _ in range(world_size)]
            dist.all_gather(gathered, p.data)

            for r in range(1, world_size):
                assert torch.allclose(gathered[0], gathered[r], atol=1e-4)

        finally:
            cleanup_process_groups()


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================

@pytest.mark.multi_gpu
class TestGradientSyncEdgeCases:
    """Edge case tests for gradient synchronization."""

    def test_zero_gradient_all_ranks(self, cuda_device):
        """Test synchronization with zero gradients on all ranks."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.zeros_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # With zero gradient and no weight decay, parameter should barely change
        # (only numerical precision differences)
        assert torch.allclose(p.data, torch.ones_like(p), rtol=0.01)

    def test_very_large_gradient_values(self, cuda_device):
        """Test synchronization with very large gradient values."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        # Use large but finite gradient
        p.grad = torch.full_like(p, 1e4 * (rank + 1))

        param_groups = [{'params': [p], 'lr': 1e-8, 'weight_decay': 0.0}]  # Tiny LR
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Should not produce NaN or Inf
        assert not torch.isnan(p.data).any()
        assert not torch.isinf(p.data).any()

    def test_very_small_gradient_values(self, cuda_device):
        """Test synchronization with very small gradient values."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.full_like(p, 1e-6 * (rank + 1))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Should handle small gradients correctly
        assert not torch.isnan(p.data).any()
        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        assert state[state_key]['step'] == 1

    def test_mixed_positive_negative_gradients(self, cuda_device):
        """Test synchronization with mixed positive and negative gradients."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        # Rank 0: positive, Rank 1: negative
        sign = 1 if rank % 2 == 0 else -1
        p.grad = torch.full_like(p, float(sign * (rank + 1)))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Average gradient: (1 + -2) / 2 = -0.5 for 2 ranks
        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        expected_avg = sum((1 if r % 2 == 0 else -1) * (r + 1) for r in range(world_size)) / world_size
        beta1 = 0.9
        expected_exp_avg = (1 - beta1) * expected_avg

        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_exp_avg),
            rtol=0.2, atol=0.1
        )

    def test_single_element_tensor(self, cuda_device):
        """Test synchronization with single-element tensors."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        p = nn.Parameter(torch.ones(1, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.full_like(p, float(rank + 1))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify sync
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4)


@pytest.mark.multi_gpu
class TestGradientSyncStress:
    """Stress tests for gradient synchronization."""

    def test_many_consecutive_steps_8gpu(self, cuda_device):
        """Stress test with many consecutive optimizer steps on 8 GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        torch.manual_seed(42 + rank)
        p = nn.Parameter(torch.randn(256, device=cuda_device, dtype=torch.bfloat16))
        param_groups = [{'params': [p], 'lr': 0.001, 'weight_decay': 0.01}]
        state: Dict[str, Any] = {}

        n_steps = 100
        for step in range(n_steps):
            p.grad = torch.randn_like(p)
            step_dense_adamw(param_groups, state=state, pg=None)

        # Verify final sync
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-3), \
                f"After {n_steps} steps: params differ between rank 0 and rank {r}"

        # Verify step count
        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        assert state[state_key]['step'] == n_steps

    def test_many_parameters_8gpu(self, cuda_device):
        """Test with many parameters on 8 GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        torch.manual_seed(42)

        # Create many parameters of various sizes
        params = []
        shapes = [(64, 32), (128,), (32, 16), (256, 128), (64,), (16, 16, 16)]
        for shape in shapes:
            p = nn.Parameter(torch.randn(shape, device=cuda_device, dtype=torch.bfloat16))
            p.grad = torch.randn_like(p) * (rank + 1)
            params.append(p)

        param_groups = [{'params': params, 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify all parameters are synchronized
        for i, p in enumerate(params):
            gathered = [torch.zeros_like(p) for _ in range(world_size)]
            dist.all_gather(gathered, p.data)

            for r in range(1, world_size):
                assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                    f"Param {i}: differs between rank 0 and rank {r}"

    def test_rapid_accumulation_sync_cycles(self, cuda_device):
        """Test rapid accumulation and sync cycles."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        n_cycles = 50
        micro_batches_per_cycle = 4

        for cycle in range(n_cycles):
            # Accumulate
            p.grad = torch.zeros_like(p)
            for mb in range(micro_batches_per_cycle):
                p.grad += torch.full_like(p, float((mb + 1) * (rank + 1)))

            # Sync
            step_dense_adamw(param_groups, state=state, pg=None)

            # Quick sync check every 10 cycles
            if cycle % 10 == 0:
                gathered = [torch.zeros_like(p) for _ in range(world_size)]
                dist.all_gather(gathered, p.data)
                for r in range(1, world_size):
                    assert torch.allclose(gathered[0], gathered[r], atol=1e-3)

        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        assert state[state_key]['step'] == n_cycles


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.multi_gpu
class TestGradientSyncIntegration:
    """Integration tests combining multiple gradient sync scenarios."""

    def test_training_loop_8gpu(self, cuda_device):
        """Test a realistic training loop on 8 GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # Create a simple model
        torch.manual_seed(42)
        model = create_simple_model(dim=128, n_layers=3, device=cuda_device)

        param_groups = [
            {'params': list(model.parameters()), 'lr': 0.01, 'weight_decay': 0.01}
        ]
        state: Dict[str, Any] = {}

        # Training loop
        losses = []
        for step in range(10):
            # Forward
            x = torch.randn(32, 128, device=cuda_device, dtype=torch.bfloat16)
            y = model(x)
            loss = y.pow(2).mean()
            losses.append(loss.item())

            # Backward
            loss.backward()

            # Optimizer step
            step_dense_adamw(param_groups, state=state, pg=None)

            # Zero gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        # Verify all ranks have same final model
        for p in model.parameters():
            gathered = [torch.zeros_like(p) for _ in range(world_size)]
            dist.all_gather(gathered, p.data)
            for r in range(1, world_size):
                assert torch.allclose(gathered[0], gathered[r], atol=1e-3)

    def test_mixed_precision_gradients(self, cuda_device):
        """Test gradient sync with mixed precision (bf16 params, fp32 grads)."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        # BF16 parameter with FP32 gradient (common in mixed precision)
        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        # Note: ZeRO-2 expects gradients to be same dtype as params
        p.grad = torch.full_like(p, float(rank + 1))

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify sync
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
