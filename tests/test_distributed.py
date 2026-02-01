"""Distributed tests for nmoe with multi-GPU configurations.

These tests are designed to run with torchrun:
    torchrun --nproc_per_node=N pytest tests/test_distributed.py -v

The tests automatically detect the world size and test appropriate
EP/TP configurations.
"""

import os
import pytest
import torch
import torch.distributed as dist


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


# Mark entire module as distributed
pytestmark = pytest.mark.distributed


@pytest.fixture(scope="module", autouse=True)
def setup_distributed():
    """Initialize distributed environment for all tests in module."""
    init_distributed()
    yield
    # Cleanup is handled by process exit


class TestDistributedBasics:
    """Basic distributed operation tests."""

    def test_world_size_detection(self):
        """Test that world size is correctly detected."""
        world_size = get_world_size()
        rank = get_rank()
        print(f"Rank {rank}/{world_size}")

        if dist.is_initialized():
            assert world_size == dist.get_world_size()
            assert rank == dist.get_rank()

    def test_cuda_device_assignment(self):
        """Test that each rank gets correct CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rank = get_rank()
        expected_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        # Allocate tensor on expected device
        x = torch.randn(10, device=expected_device)
        assert x.device == expected_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_reduce_basic(self):
        """Test basic all-reduce operation."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()

        # Each rank contributes its rank value
        x = torch.tensor([float(rank)], device=f"cuda:{rank}")

        dist.all_reduce(x, op=dist.ReduceOp.SUM)

        # Sum of 0 + 1 + ... + (world_size - 1) = world_size * (world_size - 1) / 2
        expected = world_size * (world_size - 1) / 2
        assert x.item() == expected, f"Expected {expected}, got {x.item()}"


class TestRDEPDistributed:
    """Test RDEP dispatcher in distributed setting."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rdep_initialization_distributed(self):
        """Test RDEP initialization with multiple ranks."""
        skip_if_not_multi_gpu(2)

        from nmoe.rdep import Rdep

        rank = get_rank()
        world_size = get_world_size()

        # Each rank creates its own RDEP
        # With ep_group=None and dist initialized, uses default group
        rdep = Rdep(
            dim=2048,
            n_local=8,  # 8 local experts per rank
            topk=2,
            profile="bf16",
            capacity=4096,
        )

        assert rdep.world == world_size
        assert rdep.rank == rank
        assert rdep.n_local == 8

        # Mode should be IPC for multi-GPU
        assert rdep._mode in ("ipc", "hybrid"), f"Unexpected mode: {rdep._mode}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dispatch_distributed(self):
        """Test token dispatch across multiple GPUs."""
        skip_if_not_multi_gpu(2)

        from nmoe.rdep import Rdep

        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{rank}")

        # Create RDEP - 4 experts per rank
        n_local = 4
        n_total = n_local * world_size
        topk = 2

        rdep = Rdep(
            dim=256,
            n_local=n_local,
            topk=topk,
            profile="bf16",
            capacity=1024,
        )

        # Create input tokens
        T = 64
        x = torch.randn(T, 256, device=device, dtype=torch.bfloat16)

        # Create expert IDs - mix of local and remote experts
        # Each rank sees expert IDs from 0 to n_total-1
        # Local experts are: rank * n_local to (rank + 1) * n_local
        eids = torch.randint(0, n_total, (T, topk), device=device, dtype=torch.int32)
        gates = torch.softmax(torch.randn(T, topk, device=device), dim=-1).to(torch.bfloat16)

        # Dispatch should not error
        try:
            out, _ = rdep.dispatch(x, eids, gates)
            print(f"Rank {rank}: Dispatch succeeded, output shape: {out.shape}")
        except Exception as e:
            pytest.fail(f"Dispatch failed on rank {rank}: {e}")


class TestProcessGroupInitialization:
    """Test nmoe process group initialization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ep_group_creation(self):
        """Test EP group creation with different configurations."""
        skip_if_not_multi_gpu(2)

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

        # EP = world_size (no TP)
        cleanup_process_groups()
        init_nmoe_process_groups(ep_size=world_size, tp_size=1)

        ep_group = get_ep_group()
        assert ep_group is not None

        ep_rank = get_ep_rank()
        assert ep_rank == rank  # EP rank equals global rank when no TP

        cleanup_process_groups()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ep_tp_group_creation(self):
        """Test combined EP+TP group creation."""
        skip_if_not_multi_gpu(4)

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

        if world_size < 4:
            pytest.skip("Need at least 4 GPUs for EP=2, TP=2")

        # EP=2, TP=2
        cleanup_process_groups()
        init_nmoe_process_groups(ep_size=2, tp_size=2)

        ep_rank = get_ep_rank()
        tp_rank = get_tp_rank()

        # Verify ranks are valid
        assert 0 <= ep_rank < 2
        assert 0 <= tp_rank < 2

        print(f"Rank {rank}: EP rank = {ep_rank}, TP rank = {tp_rank}")

        cleanup_process_groups()


class TestWeightSyncDistributed:
    """Test weight synchronization across distributed ranks."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_gather_expert_weights(self):
        """Test all-gathering expert weights across EP ranks."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{rank}")

        # Each rank has local weights for its experts
        n_local = 4
        hidden = 128
        inter = 256

        local_W1 = torch.randn(n_local, hidden, inter, device=device, dtype=torch.bfloat16)

        # Tag weights with rank for verification
        local_W1.fill_(float(rank))

        # Gather all weights
        gathered = [torch.zeros_like(local_W1) for _ in range(world_size)]
        dist.all_gather(gathered, local_W1)

        # Verify each rank's contribution
        for r, w in enumerate(gathered):
            expected_val = float(r)
            assert torch.allclose(w, torch.full_like(w, expected_val)), \
                f"Rank {rank}: Gathered weights from rank {r} incorrect"

        # Stack into global weights
        global_W1 = torch.cat(gathered, dim=0)
        assert global_W1.shape == (n_local * world_size, hidden, inter)


class TestGradientSyncDistributed:
    """Test gradient synchronization for RL training."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gradient_all_reduce(self):
        """Test gradient all-reduce across ranks."""
        skip_if_not_multi_gpu(2)

        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{rank}")

        # Simulate a parameter with gradient
        param = torch.randn(100, 100, device=device, requires_grad=True, dtype=torch.float32)
        loss = param.sum() * (rank + 1)  # Different loss on each rank
        loss.backward()

        # Before sync
        local_grad = param.grad.clone()

        # Sync gradients
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)  # Average

        # After sync, all ranks should have same gradient
        expected_grad = sum(torch.ones_like(local_grad) * (r + 1) for r in range(world_size)) / world_size

        assert torch.allclose(param.grad, expected_grad, atol=1e-5), \
            f"Rank {rank}: Gradient sync failed"


class TestMoEDistributed:
    """Test MoE forward/backward in distributed setting."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_moe_forward_distributed(self):
        """Test MoE forward pass with expert parallelism."""
        skip_if_not_multi_gpu(2)

        from nmoe.rdep import Rdep
        from nmoe.moe import _MoEBf16Fused

        rank = get_rank()
        world_size = get_world_size()
        device = torch.device(f"cuda:{rank}")

        # Model config
        n_local = 4
        n_total = n_local * world_size
        dim = 256
        inter = 512
        topk = 2

        # Create RDEP
        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=topk,
            profile="bf16",
            capacity=2048,
        )

        # Input
        T = 32
        x = torch.randn(T, dim, device=device, dtype=torch.bfloat16)
        eids = torch.randint(0, n_total, (T, topk), device=device, dtype=torch.int32)
        gates = torch.softmax(torch.randn(T, topk, device=device), dim=-1).to(torch.bfloat16)

        # Local expert weights (each rank has its own)
        W1 = torch.randn(n_local, dim, inter, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, dim, inter, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, inter, dim, device=device, dtype=torch.bfloat16) * 0.02

        # Forward pass
        try:
            out = _MoEBf16Fused.apply(rdep, x, eids, gates, W1, W3, W2)
            assert out.shape == (T, dim), f"Wrong output shape: {out.shape}"
            print(f"Rank {rank}: MoE forward succeeded")
        except Exception as e:
            pytest.fail(f"MoE forward failed on rank {rank}: {e}")


# EP configuration tests parametrized by world size
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEPConfigurations:
    """Test various EP configurations."""

    def test_ep2(self):
        """Test EP=2 configuration."""
        skip_if_not_multi_gpu(2)

        from nmoe.rdep import Rdep

        rank = get_rank()
        device = torch.device(f"cuda:{rank}")

        rdep = Rdep(dim=256, n_local=4, topk=2, profile="bf16", capacity=1024)
        assert rdep.world == 2 or rdep.world >= 2

    def test_ep4(self):
        """Test EP=4 configuration."""
        skip_if_not_multi_gpu(4)

        from nmoe.rdep import Rdep

        rank = get_rank()
        device = torch.device(f"cuda:{rank}")

        rdep = Rdep(dim=256, n_local=4, topk=2, profile="bf16", capacity=1024)
        assert rdep.world >= 4

    def test_ep8(self):
        """Test EP=8 configuration."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        rank = get_rank()
        device = torch.device(f"cuda:{rank}")

        rdep = Rdep(dim=256, n_local=4, topk=2, profile="bf16", capacity=1024)
        assert rdep.world == 8
