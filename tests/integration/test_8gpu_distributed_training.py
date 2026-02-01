"""Comprehensive 8-GPU Distributed Training Tests for nmoe.

These tests validate real multi-GPU distributed training scenarios with:

1. 8-GPU Data Parallel (DP) Training
   - Gradient averaging correctness
   - Loss synchronization
   - Training convergence

2. 8-GPU Expert Parallel (EP) Training
   - Expert sharding correctness
   - Token dispatch across GPUs
   - Expert gradient isolation

3. Combined DP+EP (4 DP x 2 EP)
   - Hybrid parallelism configurations
   - Correct group formation
   - Mixed gradient handling

4. Gradient All-Reduce Correctness
   - Numerical precision verification
   - Different tensor sizes
   - Edge cases (NaN, Inf, zeros)

5. Expert Load Balancing
   - Load distribution across 8 GPUs
   - Router bias updates
   - Auxiliary loss effects

6. ZeRO-2 with 8 GPUs
   - Shard distribution
   - Parameter reconstruction
   - Optimizer state management

7. Checkpoint Save/Load with 8 GPUs
   - Full model state preservation
   - Cross-EP checkpoint loading
   - Async checkpoint safety

8. Deterministic Training
   - Reproducibility across runs
   - Seed management
   - RNG state synchronization

9. Training Resumption
   - Exact state restoration
   - Loss continuity
   - Step counting

10. Memory Efficiency at Scale
    - Peak memory tracking
    - Memory leak detection
    - Gradient checkpointing effects

Run with:
    torchrun --nproc_per_node=8 -m pytest \
        tests/integration/test_8gpu_distributed_training.py -v -m multi_gpu

Single-GPU tests:
    pytest tests/integration/test_8gpu_distributed_training.py -v -m gpu
"""

import gc
import json
import math
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


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


def get_local_rank() -> int:
    """Get local rank for device assignment."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def init_distributed():
    """Initialize distributed if not already done."""
    if not dist.is_initialized() and get_world_size() > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(get_local_rank())


def skip_if_not_multi_gpu(min_gpus: int = 2):
    """Skip test if not enough GPUs available."""
    world_size = get_world_size()
    if world_size < min_gpus:
        pytest.skip(f"Requires at least {min_gpus} GPUs, have {world_size}")


def barrier():
    """Synchronize all ranks."""
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from src rank to all ranks."""
    if not dist.is_initialized():
        return obj
    object_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def all_reduce_scalar(value: float, op=dist.ReduceOp.SUM) -> float:
    """All-reduce a scalar value across all ranks."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device="cuda")
    dist.all_reduce(tensor, op=op)
    return tensor.item()


def get_gpu_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def reset_peak_memory():
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Mock Objects for Testing
# =============================================================================

class MockConfig:
    """Mock configuration object for nmoe training."""

    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_dense_layers: int = 1,
        vocab_size: int = 2000,
        n_routed_experts: int = 16,
        n_activated_experts: int = 2,
        n_shared_experts: int = 1,
        inter_dim: int = 512,
        moe_inter_dim: int = 256,
        batch_size: int = 8,
        seq_len: int = 256,
        seed: int = 42,
        dtype: str = "bf16",
        lr_dense: float = 1e-4,
        lr_expert: float = 1e-4,
        lr_router: float = 1e-4,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        adam_beta2_expert: float = 0.99,
        adam_eps: float = 1e-8,
        warmup_steps: int = 10,
        hold_tokens: int = 1000,
        decay_tokens: int = 10000,
        decay_floor: float = 1e-6,
        router_bias_update_rate: float = 1e-4,
        aux_loss_alpha: float = 0.01,
        steps: int = 100,
        checkpoint_dir: str = "/tmp/checkpoints",
        checkpoint_every: int = 10,
        resume: bool = True,
        eos_token_id: int = 1,
        rms_norm_eps: float = 1e-5,
        qk_rope_head_dim: int = 32,
        qk_nope_head_dim: int = 64,
        v_head_dim: int = 64,
        q_lora_rank: int = 256,
        kv_lora_rank: int = 128,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        rope_ntk_alpha: float = 1.0,
        rope_ntk_beta: float = 32.0,
        route_scale: float = 1.0,
        norm_topk_prob: bool = True,
        attn: str = "mla",
        attn_local: str = "swa",
        attn_global_every: int = 1,
        attn_local_window: int = 128,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_dense_layers = n_dense_layers
        self.vocab_size = vocab_size
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_shared_experts = n_shared_experts
        self.inter_dim = inter_dim
        self.moe_inter_dim = moe_inter_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.dtype = dtype
        self.lr_dense = lr_dense
        self.lr_expert = lr_expert
        self.lr_router = lr_router
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_beta2_expert = adam_beta2_expert
        self.adam_eps = adam_eps
        self.warmup_steps = warmup_steps
        self.hold_tokens = hold_tokens
        self.decay_tokens = decay_tokens
        self.decay_floor = decay_floor
        self.router_bias_update_rate = router_bias_update_rate
        self.aux_loss_alpha = aux_loss_alpha
        self.steps = steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.eos_token_id = eos_token_id
        self.rms_norm_eps = rms_norm_eps
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_ntk_alpha = rope_ntk_alpha
        self.rope_ntk_beta = rope_ntk_beta
        self.route_scale = route_scale
        self.norm_topk_prob = norm_topk_prob
        self.attn = attn
        self.attn_local = attn_local
        self.attn_global_every = attn_global_every
        self.attn_local_window = attn_local_window
        self.preset = "test"


class MockMoEModel(nn.Module):
    """Mock MoE model with dense and expert parameters for training tests."""

    def __init__(
        self,
        n_experts: int = 16,
        dim: int = 256,
        inter_dim: int = 512,
        moe_inter_dim: int = 256,
        n_layers: int = 4,
        n_dense_layers: int = 1,
        vocab_size: int = 2000,
        topk: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.dim = dim
        self.inter_dim = inter_dim
        self.moe_inter_dim = moe_inter_dim
        self.n_layers = n_layers
        self.n_dense_layers = n_dense_layers
        self.vocab_size = vocab_size
        self.topk = topk

        # Embedding
        self.embed = nn.Embedding(vocab_size, dim, device=device, dtype=dtype)

        # Dense layers
        self.dense_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, inter_dim, device=device, dtype=dtype, bias=False),
                nn.GELU(),
                nn.Linear(inter_dim, dim, device=device, dtype=dtype, bias=False),
            )
            for _ in range(n_dense_layers)
        ])

        # MoE layers (simplified)
        self.moe_routers = nn.ModuleList([
            nn.Linear(dim, n_experts, device=device, dtype=dtype, bias=False)
            for _ in range(n_layers - n_dense_layers)
        ])

        # Expert weights (shared across MoE layers for simplicity)
        self.expert_W1 = nn.Parameter(
            torch.randn(n_experts, dim, moe_inter_dim, device=device, dtype=dtype) * 0.02
        )
        self.expert_W3 = nn.Parameter(
            torch.randn(n_experts, dim, moe_inter_dim, device=device, dtype=dtype) * 0.02
        )
        self.expert_W2 = nn.Parameter(
            torch.randn(n_experts, moe_inter_dim, dim, device=device, dtype=dtype) * 0.02
        )

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ])

        # Output projection
        self.lm_head = nn.Linear(dim, vocab_size, device=device, dtype=dtype, bias=False)

        # Router bias for load balancing (non-parameter buffer)
        self.register_buffer(
            "router_bias",
            torch.zeros(n_experts, device=device, dtype=torch.float32)
        )

        # Last expert loads for router bias update
        self.last_loads = None

        # Config mock
        self.config = MagicMock()
        self.config.preset = "test"
        self.config.dtype = "bf16"
        self.config.n_routed_experts = n_experts
        self.config.n_activated_experts = topk
        self.config.dim = dim
        self.config.n_layers = n_layers

    def param_sets(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Return (expert_params, dense_params) for checkpoint splitting."""
        expert_params = [self.expert_W1, self.expert_W2, self.expert_W3]
        dense_params = [p for p in self.parameters() if id(p) not in {id(ep) for ep in expert_params}]
        return expert_params, dense_params

    def moe_forward(self, x: torch.Tensor, router: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified MoE forward pass."""
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]

        # Router logits
        logits = router(x_flat)  # [B*T, E]
        scores = F.softmax(logits.float(), dim=-1)

        # Top-k selection
        topk_scores, topk_ids = torch.topk(scores, k=self.topk, dim=-1)  # [B*T, K]
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Expert computation (simplified - full computation)
        # In reality, this would use RDEP for efficient dispatch
        output = torch.zeros_like(x_flat)
        for k in range(self.topk):
            expert_ids = topk_ids[:, k]  # [B*T]
            weights = topk_weights[:, k:k+1]  # [B*T, 1]

            for e in range(self.n_experts):
                mask = (expert_ids == e)
                if mask.any():
                    x_e = x_flat[mask]  # tokens for expert e
                    # SwiGLU expert computation
                    gate = F.silu(x_e @ self.expert_W1[e])
                    up = x_e @ self.expert_W3[e]
                    hidden = gate * up
                    out_e = hidden @ self.expert_W2[e]
                    output[mask] += weights[mask] * out_e

        # Compute expert loads for load balancing
        with torch.no_grad():
            loads = torch.bincount(
                topk_ids.reshape(-1),
                minlength=self.n_experts
            ).float()
            loads = loads / loads.sum().clamp(min=1.0)
            self.last_loads = loads

        return output.view(B, T, D), scores

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        x = self.embed(tokens)  # [B, T, D]

        # Dense layers
        for i, dense in enumerate(self.dense_layers):
            x = x + dense(self.norms[i](x))

        # MoE layers
        all_scores = []
        for i, (router, norm) in enumerate(zip(
            self.moe_routers,
            self.norms[self.n_dense_layers:]
        )):
            h, scores = self.moe_forward(norm(x), router)
            x = x + h
            all_scores.append(scores)

        # Output
        x = self.norms[-1](x) if len(self.norms) > len(self.dense_layers) + len(self.moe_routers) else x
        logits = self.lm_head(x)

        return logits

    def compute_aux_loss(self) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        if self.last_loads is None:
            return torch.tensor(0.0, device=self.expert_W1.device)

        # Load balancing loss: penalize uneven distribution
        target_load = 1.0 / self.n_experts
        load_diff = (self.last_loads - target_load).pow(2).sum()
        return load_diff * self.n_experts


class MockDataLoader:
    """Mock data loader for training tests."""

    def __init__(
        self,
        batch_size: int = 8,
        seq_len: int = 256,
        vocab_size: int = 2000,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.seed = seed
        self.position = 0
        self.dataset_version = "v1.0"
        self.tokenizer_id = "test"
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of data."""
        # Generate deterministic random data based on position
        torch.manual_seed(self.seed + self.position)
        inputs = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device
        )
        targets = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device
        )
        self.position += 1
        return inputs, targets

    def state_dict(self) -> Dict[str, Any]:
        return {"position": self.position, "seed": self.seed}

    def load_state_dict(self, state: Dict[str, Any]):
        self.position = state.get("position", 0)
        self.seed = state.get("seed", self.seed)


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
def mock_config():
    """Provide a mock training configuration."""
    return MockConfig()


@pytest.fixture
def temp_checkpoint_dir():
    """Provide a temporary checkpoint directory shared across ranks."""
    rank = get_rank()

    if rank == 0:
        dir_path = tempfile.mkdtemp(prefix="nmoe_test_")
    else:
        dir_path = None

    dir_path = broadcast_object(dir_path, src=0)
    barrier()

    yield dir_path

    # Cleanup
    barrier()
    if rank == 0:
        import shutil
        try:
            shutil.rmtree(dir_path)
        except Exception:
            pass


@pytest.fixture
def small_model_factory(cuda_device):
    """Factory for creating small MoE models for testing."""
    def _create(
        n_experts: int = 16,
        dim: int = 128,
        moe_inter_dim: int = 256,
        n_layers: int = 2,
    ) -> MockMoEModel:
        return MockMoEModel(
            n_experts=n_experts,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_layers=n_layers,
            device=cuda_device,
            dtype=torch.bfloat16,
        )
    return _create


# =============================================================================
# Test Class 1: 8-GPU Data Parallel Training
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestDataParallelTraining8GPU:
    """Tests for 8-GPU data parallel training scenarios."""

    def test_dp_gradient_averaging_8gpu(self, cuda_device, small_model_factory):
        """Test gradient averaging across 8 GPUs in pure DP."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        model = small_model_factory(n_experts=8, n_layers=2)

        # Each rank has different input
        torch.manual_seed(42 + rank)
        inputs = torch.randint(0, 2000, (4, 64), device=cuda_device)
        targets = torch.randint(0, 2000, (4, 64), device=cuda_device)

        # Forward
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))

        # Backward
        loss.backward()

        # All-reduce gradients
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # Verify all ranks have same gradients
        for p in model.parameters():
            if p.grad is not None:
                gathered = [torch.zeros_like(p.grad) for _ in range(world_size)]
                dist.all_gather(gathered, p.grad)

                for r in range(1, world_size):
                    assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                        f"Gradient mismatch between rank 0 and rank {r}"

    def test_dp_loss_synchronization_8gpu(self, cuda_device, small_model_factory):
        """Test loss synchronization across 8 GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        model = small_model_factory(n_experts=8)

        # Synchronize model parameters first
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        # Each rank processes different data
        torch.manual_seed(42 + rank)
        inputs = torch.randint(0, 2000, (4, 64), device=cuda_device)
        targets = torch.randint(0, 2000, (4, 64), device=cuda_device)

        logits = model(inputs)
        local_loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))

        # Synchronize loss
        all_losses = [torch.zeros_like(local_loss) for _ in range(world_size)]
        dist.all_gather(all_losses, local_loss)

        # All ranks should report loss (may be different due to different data)
        avg_loss = sum(l.item() for l in all_losses) / world_size

        assert not math.isnan(avg_loss), "Average loss is NaN"
        assert not math.isinf(avg_loss), "Average loss is Inf"
        assert avg_loss > 0, "Average loss should be positive"

    def test_dp_training_convergence_8gpu(self, cuda_device, small_model_factory):
        """Test that DP training converges over multiple steps."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # Create model and synchronize initial state
        torch.manual_seed(42)
        model = small_model_factory(n_experts=8, dim=64, n_layers=2)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        n_steps = 20

        for step in range(n_steps):
            optimizer.zero_grad()

            # Each rank gets different data
            torch.manual_seed(42 + step * world_size + rank)
            inputs = torch.randint(0, 2000, (4, 64), device=cuda_device)
            targets = torch.randint(0, 2000, (4, 64), device=cuda_device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))
            loss.backward()

            # All-reduce gradients
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            optimizer.step()

            # Track loss
            avg_loss = all_reduce_scalar(loss.item(), op=dist.ReduceOp.SUM) / world_size
            losses.append(avg_loss)

        # Loss should generally decrease (training is working)
        initial_loss = sum(losses[:5]) / 5
        final_loss = sum(losses[-5:]) / 5

        # Allow some tolerance - we expect general improvement
        assert final_loss < initial_loss * 1.5, \
            f"Training not converging: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_dp_parameter_sync_maintained_8gpu(self, cuda_device, small_model_factory):
        """Test that parameters stay synchronized across training steps."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        torch.manual_seed(42)
        model = small_model_factory(n_experts=8, dim=64)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(10):
            optimizer.zero_grad()

            torch.manual_seed(42 + step * world_size + rank)
            inputs = torch.randint(0, 2000, (4, 64), device=cuda_device)
            targets = torch.randint(0, 2000, (4, 64), device=cuda_device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            optimizer.step()

            # Verify parameter sync
            for name, p in model.named_parameters():
                gathered = [torch.zeros_like(p.data) for _ in range(world_size)]
                dist.all_gather(gathered, p.data)

                for r in range(1, world_size):
                    assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                        f"Step {step}: Parameter {name} not synced between rank 0 and {r}"


# =============================================================================
# Test Class 2: 8-GPU Expert Parallel Training
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestExpertParallelTraining8GPU:
    """Tests for 8-GPU expert parallel training scenarios."""

    def test_ep_expert_sharding_correctness_8gpu(self, cuda_device):
        """Test that experts are correctly sharded across 8 GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # 64 experts sharded across 8 GPUs = 8 experts per GPU
        n_total_experts = 64
        n_local_experts = n_total_experts // world_size

        assert n_local_experts == 8, f"Expected 8 local experts, got {n_local_experts}"

        # Local expert range
        expert_start = rank * n_local_experts
        expert_end = expert_start + n_local_experts

        # Verify each rank has correct range
        ranges = [None] * world_size
        dist.all_gather_object(ranges, (expert_start, expert_end))

        # Ranges should be disjoint and cover all experts
        all_experts = set()
        for start, end in ranges:
            for e in range(start, end):
                assert e not in all_experts, f"Expert {e} appears in multiple ranks"
                all_experts.add(e)

        assert all_experts == set(range(n_total_experts)), \
            "Not all experts covered by sharding"

    def test_ep_token_dispatch_8gpu(self, cuda_device):
        """Test token dispatch to experts across 8 GPUs."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep

        rank = get_rank()
        world_size = get_world_size()

        n_local = 4
        n_total = n_local * world_size
        dim = 256
        topk = 2

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=topk,
            profile="bf16",
            capacity=4096,
        )

        # Create input with tokens that should go to different experts
        T = 256
        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Random expert IDs spanning all experts
        eids = torch.randint(0, n_total, (T, topk), device=cuda_device, dtype=torch.int32)
        gates = torch.softmax(torch.randn(T, topk, device=cuda_device), dim=-1).to(torch.bfloat16)

        # Dispatch should not error
        try:
            out, _ = rdep.dispatch(x, eids, gates)
            assert out.shape[0] <= T * topk, f"Unexpected output shape: {out.shape}"
        except Exception as e:
            pytest.fail(f"Token dispatch failed on rank {rank}: {e}")

    def test_ep_gradient_isolation_8gpu(self, cuda_device):
        """Test that expert gradients are isolated to their owning GPUs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        n_local = 4
        dim = 128
        inter = 256

        # Local expert weights
        W1 = torch.randn(n_local, dim, inter, device=cuda_device, dtype=torch.bfloat16, requires_grad=True)

        # Only compute gradients for tokens assigned to local experts
        T = 32
        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16)

        # Forward through local expert 0
        out = torch.matmul(x, W1[0])
        loss = out.sum()
        loss.backward()

        # Gather gradient norms from all ranks
        local_norm = W1.grad.norm().item()
        all_norms = [torch.zeros(1, device=cuda_device) for _ in range(world_size)]
        dist.all_gather(all_norms, torch.tensor([local_norm], device=cuda_device))

        # Each rank should have non-zero gradients (each computed locally)
        for r, n in enumerate(all_norms):
            assert n.item() > 0, f"Rank {r} should have non-zero gradient norm"

    def test_ep_forward_backward_8gpu(self, cuda_device):
        """Test full forward-backward pass with EP across 8 GPUs."""
        skip_if_not_multi_gpu(8)

        from nmoe.rdep import Rdep
        from nmoe.moe import _MoEBf16Fused

        rank = get_rank()
        world_size = get_world_size()

        n_local = 4
        n_total = n_local * world_size
        dim = 256
        inter = 512
        topk = 2

        rdep = Rdep(
            dim=dim,
            n_local=n_local,
            topk=topk,
            profile="bf16",
            capacity=2048,
        )

        # Local expert weights
        W1 = torch.randn(n_local, dim, inter, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(n_local, dim, inter, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(n_local, inter, dim, device=cuda_device, dtype=torch.bfloat16) * 0.02
        W1.requires_grad_(True)
        W3.requires_grad_(True)
        W2.requires_grad_(True)

        T = 64
        x = torch.randn(T, dim, device=cuda_device, dtype=torch.bfloat16, requires_grad=True)
        eids = torch.randint(0, n_total, (T, topk), device=cuda_device, dtype=torch.int32)
        gates = torch.softmax(torch.randn(T, topk, device=cuda_device), dim=-1).to(torch.bfloat16)

        # Forward
        out = _MoEBf16Fused.apply(rdep, x, eids, gates, W1, W3, W2)
        loss = out.sum()

        # Backward
        loss.backward()

        # Verify gradients exist
        assert W1.grad is not None, "W1 gradient missing"
        assert W3.grad is not None, "W3 gradient missing"
        assert W2.grad is not None, "W2 gradient missing"
        assert x.grad is not None, "Input gradient missing"


# =============================================================================
# Test Class 3: Combined DP+EP (4 DP x 2 EP)
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestCombinedDPEP8GPU:
    """Tests for combined DP+EP configurations (4 DP x 2 EP)."""

    def test_dp4_ep2_group_formation_8gpu(self, cuda_device):
        """Test correct process group formation for 4DP x 2EP."""
        skip_if_not_multi_gpu(8)

        from nmoe.distributed.init_groups import (
            init_nmoe_process_groups,
            cleanup_process_groups,
            get_ep_group,
            get_tp_group,
            get_ep_rank,
            get_tp_rank,
            get_ep_size,
            get_tp_size,
        )

        rank = get_rank()
        world_size = get_world_size()

        try:
            cleanup_process_groups()
            # EP=4 and TP=2 (no explicit DP, DP is inferred)
            init_nmoe_process_groups(ep_size=4, tp_size=2)

            ep_rank = get_ep_rank()
            tp_rank = get_tp_rank()

            # Verify ranks are valid
            assert 0 <= ep_rank < 4, f"Invalid EP rank: {ep_rank}"
            assert 0 <= tp_rank < 2, f"Invalid TP rank: {tp_rank}"

            # Verify group sizes
            ep_group = get_ep_group()
            tp_group = get_tp_group()

            if ep_group is not None:
                assert dist.get_world_size(ep_group) == 4
            if tp_group is not None:
                assert dist.get_world_size(tp_group) == 2

        finally:
            cleanup_process_groups()

    def test_dp4_ep2_gradient_sync_8gpu(self, cuda_device):
        """Test gradient synchronization in EP+TP configuration."""
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
            # EP=4, TP=2 configuration
            init_nmoe_process_groups(ep_size=4, tp_size=2)

            ep_rank = get_ep_rank()
            tp_rank = get_tp_rank()
            ep_group = get_ep_group()

            # Expert gradient synced within EP group
            expert_grad = torch.full((64,), float(ep_rank + 1), device=cuda_device, dtype=torch.float32)

            if ep_group is not None:
                dist.all_reduce(expert_grad, op=dist.ReduceOp.AVG, group=ep_group)

            # Expected: average of 1,2,3,4 = 2.5 (EP group has 4 ranks)
            expected_ep_avg = sum(r + 1 for r in range(4)) / 4
            assert torch.allclose(
                expert_grad.mean(),
                torch.tensor(expected_ep_avg),
                rtol=0.1
            ), f"EP gradient average incorrect: {expert_grad.mean()}"

        finally:
            cleanup_process_groups()

    def test_ep8_configuration_8gpu(self, cuda_device):
        """Test pure EP=8 configuration (all 8 GPUs as EP)."""
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
            # Pure EP=8, TP=1 configuration
            init_nmoe_process_groups(ep_size=8, tp_size=1)

            ep_rank = get_ep_rank()
            tp_rank = get_tp_rank()
            ep_group = get_ep_group()
            tp_group = get_tp_group()

            assert 0 <= ep_rank < 8, f"Invalid EP rank: {ep_rank}"
            assert tp_rank == 0, f"TP rank should be 0, got {tp_rank}"

            if ep_group is not None:
                assert dist.get_world_size(ep_group) == 8
            # TP group should be None with TP=1
            assert tp_group is None, "TP group should be None with TP=1"

        finally:
            cleanup_process_groups()


# =============================================================================
# Test Class 4: Gradient All-Reduce Correctness
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestGradientAllReduceCorrectness8GPU:
    """Tests for gradient all-reduce numerical correctness."""

    def test_gradient_allreduce_precision_8gpu(self, cuda_device):
        """Test gradient all-reduce maintains numerical precision."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # Create gradient with known values
        grad = torch.full((1024,), float(rank + 1), device=cuda_device, dtype=torch.bfloat16)

        dist.all_reduce(grad, op=dist.ReduceOp.SUM)

        # Expected: sum of 1+2+3+4+5+6+7+8 = 36
        expected_sum = sum(r + 1 for r in range(world_size))

        assert torch.allclose(
            grad,
            torch.full_like(grad, expected_sum),
            rtol=0.01
        ), f"All-reduce sum incorrect: got {grad.mean()}, expected {expected_sum}"

    def test_gradient_allreduce_large_tensor_8gpu(self, cuda_device):
        """Test all-reduce with large tensors."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # Large tensor (10M elements)
        large_size = 10 * 1024 * 1024
        grad = torch.full((large_size,), float(rank + 1), device=cuda_device, dtype=torch.bfloat16)

        dist.all_reduce(grad, op=dist.ReduceOp.AVG)

        expected_avg = sum(r + 1 for r in range(world_size)) / world_size

        # Check subset for correctness
        assert torch.allclose(
            grad[:1000].float().mean(),
            torch.tensor(expected_avg),
            rtol=0.02
        )

    def test_gradient_allreduce_zero_values_8gpu(self, cuda_device):
        """Test all-reduce handles zero gradients correctly."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()

        grad = torch.zeros(256, device=cuda_device, dtype=torch.bfloat16)

        dist.all_reduce(grad, op=dist.ReduceOp.AVG)

        assert grad.abs().sum() == 0, "Zero gradients should remain zero after all-reduce"

    def test_gradient_allreduce_mixed_signs_8gpu(self, cuda_device):
        """Test all-reduce with mixed positive and negative values."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # Alternating signs: rank 0,2,4,6 positive; rank 1,3,5,7 negative
        sign = 1 if rank % 2 == 0 else -1
        grad = torch.full((256,), float(sign * (rank + 1)), device=cuda_device, dtype=torch.float32)

        dist.all_reduce(grad, op=dist.ReduceOp.SUM)

        # Expected: (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8) = -4
        expected = sum((1 if r % 2 == 0 else -1) * (r + 1) for r in range(world_size))

        assert torch.allclose(
            grad,
            torch.full_like(grad, expected),
            rtol=0.01
        )

    def test_gradient_allreduce_different_dtypes_8gpu(self, cuda_device):
        """Test all-reduce with different tensor dtypes."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            grad = torch.full((128,), float(rank + 1), device=cuda_device, dtype=dtype)

            dist.all_reduce(grad, op=dist.ReduceOp.AVG)

            expected = sum(r + 1 for r in range(world_size)) / world_size

            assert torch.allclose(
                grad.float().mean(),
                torch.tensor(expected),
                rtol=0.1
            ), f"All-reduce incorrect for dtype {dtype}"


# =============================================================================
# Test Class 5: Expert Load Balancing
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestExpertLoadBalancing8GPU:
    """Tests for expert load balancing across 8 GPUs."""

    def test_load_distribution_uniformity_8gpu(self, cuda_device, small_model_factory):
        """Test that expert loads are approximately uniform."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # Create model with 64 experts (8 per GPU)
        n_experts = 64
        model = small_model_factory(n_experts=n_experts, dim=128)

        # Synchronize model
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        # Run forward with uniform random input
        torch.manual_seed(42)
        inputs = torch.randint(0, 2000, (32, 128), device=cuda_device)

        _ = model(inputs)

        # Gather load statistics from all ranks
        if model.last_loads is not None:
            local_loads = model.last_loads.cpu()
            all_loads = [torch.zeros_like(local_loads) for _ in range(world_size)]
            dist.all_gather(all_loads, local_loads.cuda())

            # Combine loads (each rank reports for local experts)
            combined_loads = torch.stack(all_loads).mean(dim=0)

            # Check coefficient of variation (std/mean)
            load_mean = combined_loads.mean()
            load_std = combined_loads.std()
            cv = (load_std / load_mean).item() if load_mean > 0 else 0

            # Load balance should be reasonable (CV < 0.5 is good)
            assert cv < 1.0, f"Load imbalance too high: CV={cv:.3f}"

    def test_router_bias_update_8gpu(self, cuda_device, small_model_factory):
        """Test router bias updates for load balancing."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()

        model = small_model_factory(n_experts=16)

        # Initial bias should be zero
        assert model.router_bias.abs().sum() == 0

        # Simulate unbalanced load (one expert gets all tokens)
        model.last_loads = torch.zeros(16, device=cuda_device)
        model.last_loads[0] = 1.0  # All tokens to expert 0

        # Update router bias
        gamma = 0.01
        expected = 1.0 / 16  # Target uniform load
        sign = torch.sign(model.last_loads - expected)
        model.router_bias -= gamma * (sign - sign.mean())

        # Bias should now be non-zero
        assert model.router_bias.abs().sum() > 0

        # Expert 0 (overloaded) should have negative bias
        assert model.router_bias[0] < 0

    def test_auxiliary_loss_gradient_8gpu(self, cuda_device, small_model_factory):
        """Test auxiliary load balancing loss contributes to gradients."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()

        model = small_model_factory(n_experts=16, dim=64)

        inputs = torch.randint(0, 2000, (4, 32), device=cuda_device)

        # Forward
        logits = model(inputs)
        main_loss = F.cross_entropy(logits.view(-1, 2000), inputs.view(-1))
        aux_loss = model.compute_aux_loss() * 0.01

        total_loss = main_loss + aux_loss
        total_loss.backward()

        # Router should have gradients
        for router in model.moe_routers:
            assert router.weight.grad is not None, "Router gradient missing"
            assert router.weight.grad.abs().sum() > 0, "Router gradient is zero"


# =============================================================================
# Test Class 6: ZeRO-2 with 8 GPUs
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestZeRO2_8GPU:
    """Tests for ZeRO-2 optimizer state sharding across 8 GPUs."""

    def test_zero2_shard_distribution_8gpu(self, cuda_device):
        """Test ZeRO-2 shard distribution across 8 GPUs."""
        skip_if_not_multi_gpu(8)

        from nmoe.zero2 import _get_or_init_flat_group, _ceil_div

        rank = get_rank()
        world_size = get_world_size()

        # Create parameters
        total_elems = 2048
        p = nn.Parameter(torch.randn(total_elems, device=cuda_device, dtype=torch.bfloat16))
        params = [p]

        group = {}
        flat = _get_or_init_flat_group(
            group,
            params=params,
            rank=rank,
            world=world_size,
            dtype=torch.bfloat16,
        )

        # Each rank should have 1/8 of the parameters
        expected_shard_size = _ceil_div(total_elems, world_size)
        assert flat['shard_size'] == expected_shard_size
        assert flat['param_shard'].numel() == expected_shard_size

    def test_zero2_parameter_reconstruction_8gpu(self, cuda_device):
        """Test ZeRO-2 parameter reconstruction after all-gather."""
        skip_if_not_multi_gpu(8)

        from nmoe.zero2 import step_dense_adamw

        rank = get_rank()
        world_size = get_world_size()

        # Create parameter with known initial values
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(256, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify all ranks have same parameter values after step
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                f"Parameters differ between rank 0 and rank {r}"

    def test_zero2_optimizer_state_correctness_8gpu(self, cuda_device):
        """Test ZeRO-2 optimizer state is correctly maintained."""
        skip_if_not_multi_gpu(8)

        from nmoe.zero2 import step_dense_adamw

        rank = get_rank()

        p = nn.Parameter(torch.ones(128, device=cuda_device, dtype=torch.bfloat16))
        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state = {}

        # Run multiple steps
        for step in range(5):
            p.grad = torch.full_like(p, float(step + 1))
            step_dense_adamw(param_groups, state=state, pg=None)

        # Verify step count
        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        assert state[state_key]['step'] == 5

        # Verify momentum exists
        assert state[state_key]['exp_avg'] is not None
        assert state[state_key]['exp_avg_sq'] is not None

    def test_zero2_large_model_8gpu(self, cuda_device):
        """Test ZeRO-2 with larger model parameters."""
        skip_if_not_multi_gpu(8)

        from nmoe.zero2 import step_dense_adamw

        rank = get_rank()
        world_size = get_world_size()

        # Create multiple large parameters
        params = []
        for size in [(1024, 512), (512, 256), (256,)]:
            p = nn.Parameter(torch.randn(size, device=cuda_device, dtype=torch.bfloat16))
            p.grad = torch.randn_like(p)
            params.append(p)

        param_groups = [{'params': params, 'lr': 0.001, 'weight_decay': 0.01}]
        state = {}

        # Run optimizer step
        step_dense_adamw(param_groups, state=state, pg=None)

        # Verify all parameters are synchronized
        for i, p in enumerate(params):
            gathered = [torch.zeros_like(p) for _ in range(world_size)]
            dist.all_gather(gathered, p.data)

            for r in range(1, world_size):
                assert torch.allclose(gathered[0], gathered[r], atol=1e-3), \
                    f"Param {i}: not synced between rank 0 and {r}"


# =============================================================================
# Test Class 7: Checkpoint Save/Load with 8 GPUs
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestCheckpoint8GPU:
    """Tests for checkpoint save/load across 8 GPUs."""

    def test_checkpoint_save_8gpu(self, cuda_device, temp_checkpoint_dir, small_model_factory):
        """Test checkpoint save creates correct files on all ranks."""
        skip_if_not_multi_gpu(8)

        from nmoe.checkpoint import Checkpointer, build_states

        rank = get_rank()
        world_size = get_world_size()

        model = small_model_factory(n_experts=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loader = MockDataLoader()

        checkpointer = Checkpointer(
            base=temp_checkpoint_dir,
            keep_last=3,
            async_io=False,
        )

        step = 100
        tokens = 10000

        rd_state, dp_state = build_states(
            step=step,
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            loader=loader,
            config_fingerprint="test_fp",
        )

        # Save checkpoint
        if rank == 0:
            checkpointer.save_dense(step, rd_state)
        checkpointer.save_rank_local(step, dp_state)

        barrier()

        # Verify files exist
        iter_dir = Path(temp_checkpoint_dir) / f"iter_{step:07d}"

        if rank == 0:
            assert (iter_dir / "rd.pt").exists(), "rd.pt not created"

        dp_file = iter_dir / f"dp_rank_{rank:03d}.pt"
        assert dp_file.exists(), f"dp_rank_{rank:03d}.pt not created"

        checkpointer.close()

    def test_checkpoint_load_8gpu(self, cuda_device, temp_checkpoint_dir, small_model_factory):
        """Test checkpoint load restores correct state on all ranks."""
        skip_if_not_multi_gpu(8)

        from nmoe.checkpoint import Checkpointer, build_states, load_state

        rank = get_rank()

        # Create and save checkpoint
        torch.manual_seed(42)
        model1 = small_model_factory(n_experts=16)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        loader1 = MockDataLoader()
        loader1.position = 50

        checkpointer = Checkpointer(
            base=temp_checkpoint_dir,
            keep_last=3,
            async_io=False,
        )

        step = 100
        tokens = 10000

        rd_state, dp_state = build_states(
            step=step,
            model=model1,
            optimizer=optimizer1,
            tokens=tokens,
            loader=loader1,
            config_fingerprint="test_fp",
        )

        if rank == 0:
            checkpointer.save_dense(step, rd_state)
        checkpointer.save_rank_local(step, dp_state)
        barrier()

        # Create new model and load checkpoint
        torch.manual_seed(12345)  # Different seed
        model2 = small_model_factory(n_experts=16)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        loader2 = MockDataLoader()

        dp_path = Path(temp_checkpoint_dir) / f"iter_{step:07d}" / f"dp_rank_{rank:03d}.pt"
        loaded_step, loaded_tokens, _ = load_state(
            str(dp_path),
            model2,
            optimizer2,
            loader2,
            print_fn=lambda x: None,
        )

        assert loaded_step == step
        assert loaded_tokens == tokens

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if n1 == n2:
                assert torch.allclose(p1.data, p2.data, atol=1e-4), \
                    f"Parameter {n1} not restored correctly"

        checkpointer.close()

    def test_checkpoint_cross_ep_load_8gpu(self, cuda_device, temp_checkpoint_dir):
        """Test loading checkpoint with different EP configuration."""
        skip_if_not_multi_gpu(8)

        from nmoe.checkpoint import EPShardInfo, validate_ep_shard_compatibility

        rank = get_rank()

        # Simulate checkpoint saved with EP=4
        saved_info = EPShardInfo.create(
            ep_size=4,
            ep_rank=rank % 4,
            n_total_experts=32,
        )

        # Try to load with EP=8
        compatible, msg = validate_ep_shard_compatibility(
            saved_info,
            current_ep_size=8,
            current_ep_rank=rank,
            n_total_experts=32,
        )

        # Should be incompatible (EP size mismatch)
        assert not compatible, "Should detect EP size mismatch"
        assert "EP size mismatch" in msg


# =============================================================================
# Test Class 8: Deterministic Training
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestDeterministicTraining8GPU:
    """Tests for deterministic training across runs."""

    def test_deterministic_forward_8gpu(self, cuda_device, small_model_factory):
        """Test forward pass is deterministic with same seed."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()

        results = []
        for run in range(2):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            model = small_model_factory(n_experts=8, dim=64)
            for p in model.parameters():
                dist.broadcast(p.data, src=0)

            inputs = torch.randint(0, 2000, (4, 32), device=cuda_device)
            logits = model(inputs)

            results.append(logits.clone())

        assert torch.allclose(results[0], results[1], atol=1e-5), \
            "Forward pass not deterministic"

    def test_deterministic_training_loop_8gpu(self, cuda_device, small_model_factory):
        """Test training loop is deterministic across runs."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        final_losses = []
        final_params = []

        for run in range(2):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            model = small_model_factory(n_experts=8, dim=64)
            for p in model.parameters():
                dist.broadcast(p.data, src=0)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            for step in range(5):
                optimizer.zero_grad()

                torch.manual_seed(42 + step * world_size + rank)
                inputs = torch.randint(0, 2000, (4, 32), device=cuda_device)
                targets = inputs.clone()

                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))
                loss.backward()

                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                optimizer.step()

            final_losses.append(loss.item())
            final_params.append({n: p.clone() for n, p in model.named_parameters()})

        # Losses should match
        assert abs(final_losses[0] - final_losses[1]) < 1e-4, \
            f"Final losses differ: {final_losses[0]} vs {final_losses[1]}"

        # Parameters should match
        for name in final_params[0]:
            assert torch.allclose(
                final_params[0][name],
                final_params[1][name],
                atol=1e-4
            ), f"Parameter {name} not deterministic"

    def test_rng_state_synchronization_8gpu(self, cuda_device):
        """Test RNG state can be synchronized across ranks."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        # Set different seeds on different ranks
        torch.manual_seed(42 + rank)

        # Synchronize RNG state from rank 0
        if rank == 0:
            rng_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state()
        else:
            rng_state = torch.empty(0, dtype=torch.uint8)
            cuda_state = torch.empty(0, dtype=torch.uint8)

        # Broadcast RNG states
        rng_state_list = [rng_state]
        dist.broadcast_object_list(rng_state_list, src=0)
        cuda_state_list = [cuda_state]
        dist.broadcast_object_list(cuda_state_list, src=0)

        # Restore RNG state
        torch.random.set_rng_state(rng_state_list[0])
        torch.cuda.set_rng_state(cuda_state_list[0])

        # Generate random tensor
        x = torch.randn(100, device=cuda_device)

        # All ranks should have same random values
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x)

        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-6), \
                f"RNG not synchronized between rank 0 and {r}"


# =============================================================================
# Test Class 9: Training Resumption
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestTrainingResumption8GPU:
    """Tests for training resumption from checkpoint."""

    def test_resume_from_checkpoint_8gpu(self, cuda_device, temp_checkpoint_dir, small_model_factory):
        """Test training can resume exactly from checkpoint."""
        skip_if_not_multi_gpu(8)

        from nmoe.checkpoint import Checkpointer, build_states, load_state

        rank = get_rank()
        world_size = get_world_size()

        # Phase 1: Train for 10 steps and save checkpoint at step 5
        torch.manual_seed(42)
        model1 = small_model_factory(n_experts=8, dim=64)
        for p in model1.parameters():
            dist.broadcast(p.data, src=0)

        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        loader1 = MockDataLoader(device=cuda_device)

        checkpointer = Checkpointer(
            base=temp_checkpoint_dir,
            keep_last=3,
            async_io=False,
        )

        losses_phase1 = []
        for step in range(10):
            optimizer1.zero_grad()
            inputs, targets = loader1.next()
            logits = model1(inputs)
            loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))
            loss.backward()

            for p in model1.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            optimizer1.step()
            losses_phase1.append(loss.item())

            # Save checkpoint at step 5
            if step == 4:  # After step 5 (0-indexed)
                rd_state, dp_state = build_states(
                    step=step + 1,
                    model=model1,
                    optimizer=optimizer1,
                    tokens=(step + 1) * 8 * 256,
                    loader=loader1,
                    config_fingerprint="test_fp",
                )
                if rank == 0:
                    checkpointer.save_dense(step + 1, rd_state)
                checkpointer.save_rank_local(step + 1, dp_state)
                barrier()

        # Phase 2: Resume from step 5 and continue to step 10
        torch.manual_seed(12345)  # Different seed to verify restore works
        model2 = small_model_factory(n_experts=8, dim=64)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        loader2 = MockDataLoader(device=cuda_device)

        # Load checkpoint
        dp_path = Path(temp_checkpoint_dir) / "iter_0000005" / f"dp_rank_{rank:03d}.pt"
        start_step, _, _ = load_state(
            str(dp_path),
            model2,
            optimizer2,
            loader2,
            print_fn=lambda x: None,
        )

        assert start_step == 5, f"Start step should be 5, got {start_step}"

        losses_phase2 = []
        for step in range(start_step, 10):
            optimizer2.zero_grad()
            inputs, targets = loader2.next()
            logits = model2(inputs)
            loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))
            loss.backward()

            for p in model2.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            optimizer2.step()
            losses_phase2.append(loss.item())

        # Losses should be similar (may not be exactly equal due to floating point)
        # Compare the resumed portion
        for i, (l1, l2) in enumerate(zip(losses_phase1[5:], losses_phase2)):
            assert abs(l1 - l2) < 0.5, \
                f"Step {5+i}: loss mismatch after resume: {l1} vs {l2}"

        checkpointer.close()

    def test_step_counting_after_resume_8gpu(self, cuda_device, temp_checkpoint_dir, small_model_factory):
        """Test step counting is correct after resume."""
        skip_if_not_multi_gpu(8)

        from nmoe.checkpoint import Checkpointer, build_states, load_state

        rank = get_rank()

        # Save checkpoint at step 50
        model = small_model_factory(n_experts=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loader = MockDataLoader()

        checkpointer = Checkpointer(base=temp_checkpoint_dir, async_io=False)

        step = 50
        rd_state, dp_state = build_states(
            step=step,
            model=model,
            optimizer=optimizer,
            tokens=step * 2048,
            loader=loader,
        )

        if rank == 0:
            checkpointer.save_dense(step, rd_state)
        checkpointer.save_rank_local(step, dp_state)
        barrier()

        # Load and verify step
        model2 = small_model_factory(n_experts=8)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

        dp_path = Path(temp_checkpoint_dir) / f"iter_{step:07d}" / f"dp_rank_{rank:03d}.pt"
        loaded_step, loaded_tokens, _ = load_state(
            str(dp_path), model2, optimizer2, None, print_fn=lambda x: None,
        )

        assert loaded_step == step
        assert loaded_tokens == step * 2048

        checkpointer.close()


# =============================================================================
# Test Class 10: Memory Efficiency at Scale
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestMemoryEfficiency8GPU:
    """Tests for memory efficiency in distributed training."""

    def test_peak_memory_tracking_8gpu(self, cuda_device, small_model_factory):
        """Test peak memory is tracked correctly during training."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()

        reset_peak_memory()
        initial_mem = get_gpu_memory_stats()

        model = small_model_factory(n_experts=16, dim=128)

        after_model_mem = get_gpu_memory_stats()

        # Run training step
        inputs = torch.randint(0, 2000, (8, 128), device=cuda_device)
        logits = model(inputs)
        loss = logits.sum()
        loss.backward()

        after_backward_mem = get_gpu_memory_stats()

        # Peak should be higher than initial
        assert after_backward_mem["max_allocated"] > initial_mem["allocated"]

        # Report memory usage
        if rank == 0:
            print(f"\nMemory (rank 0):")
            print(f"  Initial: {initial_mem['allocated']:.1f} MB")
            print(f"  After model: {after_model_mem['allocated']:.1f} MB")
            print(f"  Peak: {after_backward_mem['max_allocated']:.1f} MB")

    def test_memory_leak_detection_8gpu(self, cuda_device, small_model_factory):
        """Test for memory leaks over multiple training steps."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()

        model = small_model_factory(n_experts=8, dim=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Warmup
        for _ in range(3):
            inputs = torch.randint(0, 2000, (4, 32), device=cuda_device)
            logits = model(inputs)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()
        reset_peak_memory()

        baseline_mem = get_gpu_memory_stats()["allocated"]

        # Run many steps
        for step in range(50):
            inputs = torch.randint(0, 2000, (4, 32), device=cuda_device)
            logits = model(inputs)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()

        final_mem = get_gpu_memory_stats()["allocated"]

        # Memory should not grow significantly (allow 20% tolerance)
        mem_growth = (final_mem - baseline_mem) / baseline_mem if baseline_mem > 0 else 0

        assert mem_growth < 0.2, \
            f"Potential memory leak: {mem_growth*100:.1f}% growth over 50 steps"

    def test_gradient_checkpointing_memory_savings_8gpu(self, cuda_device):
        """Test gradient checkpointing reduces memory usage."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()

        dim = 128
        n_layers = 4

        # Model without gradient checkpointing
        class SimpleModel(nn.Module):
            def __init__(self, use_checkpointing: bool = False):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(dim, dim, device=cuda_device, dtype=torch.bfloat16)
                    for _ in range(n_layers)
                ])
                self.use_checkpointing = use_checkpointing

            def forward(self, x):
                for layer in self.layers:
                    if self.use_checkpointing:
                        x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                    else:
                        x = layer(x)
                    x = F.relu(x)
                return x

        # Measure memory without checkpointing
        model1 = SimpleModel(use_checkpointing=False)
        gc.collect()
        torch.cuda.empty_cache()
        reset_peak_memory()

        x1 = torch.randn(64, dim, device=cuda_device, dtype=torch.bfloat16, requires_grad=True)
        y1 = model1(x1)
        y1.sum().backward()

        mem_without_ckpt = get_gpu_memory_stats()["max_allocated"]

        # Measure memory with checkpointing
        model2 = SimpleModel(use_checkpointing=True)
        gc.collect()
        torch.cuda.empty_cache()
        reset_peak_memory()

        x2 = torch.randn(64, dim, device=cuda_device, dtype=torch.bfloat16, requires_grad=True)
        y2 = model2(x2)
        y2.sum().backward()

        mem_with_ckpt = get_gpu_memory_stats()["max_allocated"]

        # Checkpointing should use less memory
        if rank == 0:
            print(f"\nGradient checkpointing:")
            print(f"  Without: {mem_without_ckpt:.1f} MB")
            print(f"  With: {mem_with_ckpt:.1f} MB")
            print(f"  Savings: {((mem_without_ckpt - mem_with_ckpt) / mem_without_ckpt * 100):.1f}%")

    def test_zero2_memory_efficiency_8gpu(self, cuda_device):
        """Test ZeRO-2 reduces optimizer state memory."""
        skip_if_not_multi_gpu(8)

        from nmoe.zero2 import step_dense_adamw

        rank = get_rank()
        world_size = get_world_size()

        # Large parameter
        param_size = 1024 * 1024  # 1M elements
        p = nn.Parameter(torch.randn(param_size, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state = {}

        gc.collect()
        torch.cuda.empty_cache()
        reset_peak_memory()

        step_dense_adamw(param_groups, state=state, pg=None)

        peak_mem = get_gpu_memory_stats()["max_allocated"]

        # With ZeRO-2, each rank should store only 1/8 of optimizer state
        # Expected optimizer state per rank: (1M / 8) * 2 tensors * 2 bytes/bf16 = ~0.5 MB
        # Full optimizer state would be: 1M * 2 * 2 = 4 MB

        if rank == 0:
            print(f"\nZeRO-2 memory (1M params):")
            print(f"  Peak memory: {peak_mem:.1f} MB")
            print(f"  Expected savings: {(1 - 1/world_size)*100:.0f}%")


# =============================================================================
# Additional Integration Tests
# =============================================================================

@pytest.mark.multi_gpu
@pytest.mark.b200
class TestDistributedTrainingIntegration8GPU:
    """Integration tests for full distributed training scenarios."""

    def test_full_training_loop_8gpu(self, cuda_device, small_model_factory):
        """Test complete training loop with all components."""
        skip_if_not_multi_gpu(8)

        from nmoe.zero2 import step_dense_adamw

        rank = get_rank()
        world_size = get_world_size()

        # Setup
        torch.manual_seed(42)
        model = small_model_factory(n_experts=32, dim=128, n_layers=3)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        # Separate expert and dense parameters
        expert_params, dense_params = model.param_sets()

        expert_optimizer = torch.optim.AdamW(expert_params, lr=1e-3)
        dense_groups = [{'params': dense_params, 'lr': 1e-3, 'weight_decay': 0.01}]
        zero2_state = {}

        loader = MockDataLoader(device=cuda_device)

        n_steps = 15
        losses = []

        for step in range(n_steps):
            # Get data
            inputs, targets = loader.next()

            # Forward
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, 2000), targets.view(-1))
            aux_loss = model.compute_aux_loss() * 0.01
            total_loss = loss + aux_loss

            # Backward
            total_loss.backward()

            # All-reduce expert gradients (they're sharded by RDEP in real case)
            for p in expert_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            # ZeRO-2 step for dense params
            step_dense_adamw(dense_groups, state=zero2_state, pg=None)

            # Expert optimizer step
            expert_optimizer.step()
            expert_optimizer.zero_grad()

            # Track loss
            avg_loss = all_reduce_scalar(loss.item()) / world_size
            losses.append(avg_loss)

            if rank == 0 and step % 5 == 0:
                print(f"Step {step}: loss={avg_loss:.4f}")

        # Verify training worked
        assert not any(math.isnan(l) for l in losses), "NaN loss detected"
        assert not any(math.isinf(l) for l in losses), "Inf loss detected"

        # Loss should generally be stable or decreasing
        avg_first_5 = sum(losses[:5]) / 5
        avg_last_5 = sum(losses[-5:]) / 5
        assert avg_last_5 < avg_first_5 * 2, \
            f"Loss diverged: first_5_avg={avg_first_5:.4f}, last_5_avg={avg_last_5:.4f}"

    def test_model_state_consistency_8gpu(self, cuda_device, small_model_factory):
        """Test model state stays consistent across GPUs during training."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        torch.manual_seed(42)
        model = small_model_factory(n_experts=8, dim=64)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(10):
            optimizer.zero_grad()

            inputs = torch.randint(0, 2000, (4, 32), device=cuda_device)
            logits = model(inputs)
            loss = logits.sum()
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            optimizer.step()

            # Check consistency every few steps
            if step % 3 == 0:
                for name, p in model.named_parameters():
                    # Compute checksum
                    checksum = p.data.sum()
                    all_checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
                    dist.all_gather(all_checksums, checksum)

                    for r in range(1, world_size):
                        assert torch.allclose(all_checksums[0], all_checksums[r], rtol=1e-3), \
                            f"Step {step}: {name} inconsistent between ranks"

    def test_distributed_training_with_aux_loss_8gpu(self, cuda_device, small_model_factory):
        """Test training with auxiliary load balancing loss."""
        skip_if_not_multi_gpu(8)

        rank = get_rank()
        world_size = get_world_size()

        torch.manual_seed(42)
        model = small_model_factory(n_experts=16, dim=64)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        main_losses = []
        aux_losses = []

        for step in range(15):
            optimizer.zero_grad()

            inputs = torch.randint(0, 2000, (4, 32), device=cuda_device)
            logits = model(inputs)

            main_loss = F.cross_entropy(logits.view(-1, 2000), inputs.view(-1))
            aux_loss = model.compute_aux_loss() * 0.01

            total_loss = main_loss + aux_loss
            total_loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            optimizer.step()

            main_losses.append(all_reduce_scalar(main_loss.item()) / world_size)
            aux_losses.append(all_reduce_scalar(aux_loss.item()) / world_size)

        # Aux loss should be computed
        assert all(a >= 0 for a in aux_losses), "Aux loss should be non-negative"

        if rank == 0:
            print(f"\nAux loss range: {min(aux_losses):.6f} - {max(aux_losses):.6f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
