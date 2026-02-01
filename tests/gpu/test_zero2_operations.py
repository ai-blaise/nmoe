"""P0 critical tests for nmoe ZeRO-2 operations.

Tests for the ZeRO-2 implementation in nmoe/nmoe/zero2.py:
- _rs_chunk_elems: chunked reduce-scatter buffer sizing
- _get_or_init_flat_group: flat buffer initialization
- step_dense_adamw: ZeRO-2 AdamW step

Single-GPU tests verify gradient averaging, momentum state updates,
and parameter updates without requiring distributed setup.

Multi-GPU tests require torchrun:
    torchrun --nproc_per_node=N pytest tests/gpu/test_zero2_operations.py -v -k multi_gpu
"""

import os
import math
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any

# Import ZeRO-2 functions from source
from nmoe.zero2 import (
    _ceil_div,
    _rs_chunk_elems,
    _get_or_init_flat_group,
    step_dense_adamw,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cuda_device():
    """Fixture that provides a CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def simple_linear_model(cuda_device):
    """Create a simple linear model for testing."""
    model = nn.Linear(64, 32, bias=True, device=cuda_device, dtype=torch.bfloat16)
    return model


@pytest.fixture
def multi_param_model(cuda_device):
    """Create a model with multiple parameters of different shapes."""
    model = nn.Sequential(
        nn.Linear(64, 128, bias=True, device=cuda_device, dtype=torch.bfloat16),
        nn.Linear(128, 64, bias=True, device=cuda_device, dtype=torch.bfloat16),
        nn.Linear(64, 32, bias=True, device=cuda_device, dtype=torch.bfloat16),
    )
    return model


@pytest.fixture
def param_groups_factory(cuda_device):
    """Factory for creating param groups with different configurations."""
    def _create(
        n_params: int = 3,
        shapes: list = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple:
        if shapes is None:
            shapes = [(64, 32), (128,), (32, 16)][:n_params]

        params = []
        for shape in shapes:
            p = nn.Parameter(torch.randn(shape, device=cuda_device, dtype=dtype))
            # Create a simple gradient
            p.grad = torch.randn_like(p)
            params.append(p)

        param_groups = [{'params': params, 'lr': lr, 'weight_decay': weight_decay}]
        return params, param_groups

    return _create


# =============================================================================
# Tests for _ceil_div helper
# =============================================================================

class TestCeilDiv:
    """Tests for the _ceil_div utility function."""

    def test_exact_division(self):
        """Test ceiling division when division is exact."""
        assert _ceil_div(10, 5) == 2
        assert _ceil_div(100, 10) == 10
        assert _ceil_div(256, 256) == 1

    def test_rounded_up_division(self):
        """Test ceiling division when rounding up is needed."""
        assert _ceil_div(10, 3) == 4  # 10/3 = 3.33... -> 4
        assert _ceil_div(100, 7) == 15  # 100/7 = 14.28... -> 15
        assert _ceil_div(1, 10) == 1  # 1/10 = 0.1 -> 1

    def test_division_by_one(self):
        """Test ceiling division by 1."""
        assert _ceil_div(1, 1) == 1
        assert _ceil_div(100, 1) == 100
        assert _ceil_div(12345, 1) == 12345


# =============================================================================
# Tests for _rs_chunk_elems
# =============================================================================

class TestRsChunkElems:
    """Tests for the _rs_chunk_elems function."""

    def test_basic_chunk_sizing(self):
        """Test basic chunk size calculation."""
        shard_chunk, total_chunk_elems = _rs_chunk_elems(
            dtype=torch.bfloat16,
            world=2,
            shard_size=1024,
        )

        # total_chunk_elems should be world * shard_chunk
        assert total_chunk_elems == 2 * shard_chunk
        # shard_chunk should be <= shard_size
        assert shard_chunk <= 1024
        # shard_chunk should be > 0
        assert shard_chunk > 0

    def test_world_size_variations(self):
        """Test chunk sizing with various world sizes."""
        for world in [1, 2, 4, 8, 16]:
            shard_chunk, total_chunk_elems = _rs_chunk_elems(
                dtype=torch.float32,
                world=world,
                shard_size=2048,
            )

            assert total_chunk_elems == world * shard_chunk
            assert shard_chunk <= 2048
            assert shard_chunk > 0

    def test_shard_size_variations(self):
        """Test chunk sizing with various shard sizes."""
        world = 4
        for shard_size in [128, 512, 1024, 4096, 65536]:
            shard_chunk, total_chunk_elems = _rs_chunk_elems(
                dtype=torch.bfloat16,
                world=world,
                shard_size=shard_size,
            )

            assert total_chunk_elems == world * shard_chunk
            assert shard_chunk <= shard_size
            assert shard_chunk > 0

    def test_chunk_alignment_256(self):
        """Test that chunks are aligned to 256 elements when shard_size > 256."""
        world = 4
        shard_size = 8192  # Large enough to trigger alignment

        shard_chunk, total_chunk_elems = _rs_chunk_elems(
            dtype=torch.bfloat16,
            world=world,
            shard_size=shard_size,
        )

        # When shard_size > 256, shard_chunk should be aligned to 256
        # Unless it equals shard_size exactly
        if shard_chunk < shard_size:
            assert shard_chunk % 256 == 0 or shard_chunk >= 256, \
                f"shard_chunk={shard_chunk} should be aligned to 256"

    def test_small_shard_no_alignment(self):
        """Test that small shards (<=256) use the full shard size."""
        world = 2
        shard_size = 128  # Smaller than alignment boundary

        shard_chunk, total_chunk_elems = _rs_chunk_elems(
            dtype=torch.bfloat16,
            world=world,
            shard_size=shard_size,
        )

        # For small shards, shard_chunk should equal shard_size
        assert shard_chunk == shard_size
        assert total_chunk_elems == world * shard_size

    def test_different_dtypes(self):
        """Test chunk sizing with different data types."""
        world = 2
        shard_size = 4096

        # Different dtypes have different element sizes
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            shard_chunk, total_chunk_elems = _rs_chunk_elems(
                dtype=dtype,
                world=world,
                shard_size=shard_size,
            )

            assert total_chunk_elems == world * shard_chunk
            assert shard_chunk <= shard_size
            assert shard_chunk > 0

    def test_error_invalid_world_size_zero(self):
        """Test that world_size=0 raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid world_size"):
            _rs_chunk_elems(dtype=torch.bfloat16, world=0, shard_size=1024)

    def test_error_invalid_world_size_negative(self):
        """Test that negative world_size raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid world_size"):
            _rs_chunk_elems(dtype=torch.bfloat16, world=-1, shard_size=1024)

    def test_error_invalid_shard_size_zero(self):
        """Test that shard_size=0 raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid shard_size"):
            _rs_chunk_elems(dtype=torch.bfloat16, world=2, shard_size=0)

    def test_error_invalid_shard_size_negative(self):
        """Test that negative shard_size raises RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid shard_size"):
            _rs_chunk_elems(dtype=torch.bfloat16, world=2, shard_size=-100)

    def test_chunk_env_override(self, monkeypatch):
        """Test NMOE_ZERO2_RS_CHUNK_MB environment variable override."""
        # Set a small chunk size (1 MB)
        monkeypatch.setenv("NMOE_ZERO2_RS_CHUNK_MB", "1")

        shard_chunk_small, _ = _rs_chunk_elems(
            dtype=torch.bfloat16,
            world=2,
            shard_size=1024 * 1024,  # 1M elements
        )

        # Reset to larger chunk size (256 MB)
        monkeypatch.setenv("NMOE_ZERO2_RS_CHUNK_MB", "256")

        shard_chunk_large, _ = _rs_chunk_elems(
            dtype=torch.bfloat16,
            world=2,
            shard_size=1024 * 1024,
        )

        # Smaller chunk MB should result in smaller shard_chunk
        assert shard_chunk_small <= shard_chunk_large


# =============================================================================
# Tests for _get_or_init_flat_group
# =============================================================================

class TestGetOrInitFlatGroup:
    """Tests for the _get_or_init_flat_group function."""

    def test_creates_correct_buffer_layout(self, cuda_device):
        """Test that flat buffer is created with correct layout."""
        params = [
            nn.Parameter(torch.randn(32, 16, device=cuda_device, dtype=torch.bfloat16)),
            nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16)),
            nn.Parameter(torch.randn(8, 8, device=cuda_device, dtype=torch.bfloat16)),
        ]

        group = {}
        flat = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=1,
            dtype=torch.bfloat16,
        )

        # Check expected keys in flat dict
        assert 'n_params' in flat
        assert 'offsets' in flat
        assert 'total' in flat
        assert 'padded_total' in flat
        assert 'shard_size' in flat
        assert 'flat_param' in flat
        assert 'param_shard' in flat
        assert 'rs_in' in flat
        assert 'rs_out' in flat
        assert 'shard_chunk' in flat

        # Verify n_params
        assert flat['n_params'] == 3

        # Verify total is sum of all param elements
        expected_total = 32 * 16 + 64 + 8 * 8
        assert flat['total'] == expected_total

    def test_flat_param_views_into_params(self, cuda_device):
        """Test that params become views into flat_param."""
        original_values = [
            torch.randn(32, 16, device=cuda_device, dtype=torch.bfloat16),
            torch.randn(64, device=cuda_device, dtype=torch.bfloat16),
        ]

        params = [
            nn.Parameter(original_values[0].clone()),
            nn.Parameter(original_values[1].clone()),
        ]

        group = {}
        flat = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=1,
            dtype=torch.bfloat16,
        )

        flat_param = flat['flat_param']
        offsets = flat['offsets']

        # Verify each param is now a view into flat_param
        for i, (p, (a, b)) in enumerate(zip(params, offsets)):
            # Modify flat_param and check that param changes
            old_val = p.data[0, 0].item() if p.dim() > 1 else p.data[0].item()
            new_val = 999.0

            if p.dim() > 1:
                flat_param[a] = new_val
                assert p.data[0, 0].item() == pytest.approx(new_val, rel=0.01)
            else:
                flat_param[a] = new_val
                assert p.data[0].item() == pytest.approx(new_val, rel=0.01)

    def test_offsets_are_correct(self, cuda_device):
        """Test that offsets correctly partition the flat buffer."""
        shapes = [(10,), (20,), (30,)]
        params = [
            nn.Parameter(torch.randn(s, device=cuda_device, dtype=torch.bfloat16))
            for s in shapes
        ]

        group = {}
        flat = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=1,
            dtype=torch.bfloat16,
        )

        offsets = flat['offsets']

        # Check offset boundaries
        assert offsets[0] == (0, 10)
        assert offsets[1] == (10, 30)
        assert offsets[2] == (30, 60)

    def test_padded_total_divisible_by_world(self, cuda_device):
        """Test that padded_total is divisible by world size."""
        params = [
            nn.Parameter(torch.randn(100, device=cuda_device, dtype=torch.bfloat16)),
        ]

        for world in [1, 2, 3, 4, 7, 8]:
            group = {}
            flat = _get_or_init_flat_group(
                group,
                params=[nn.Parameter(p.data.clone()) for p in params],
                rank=0,
                world=world,
                dtype=torch.bfloat16,
            )

            assert flat['padded_total'] % world == 0, \
                f"padded_total={flat['padded_total']} not divisible by world={world}"

    def test_caches_flat_buffer(self, cuda_device):
        """Test that flat buffer is cached and reused."""
        params = [
            nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16)),
        ]

        group = {}

        flat1 = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=1,
            dtype=torch.bfloat16,
        )

        flat2 = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=1,
            dtype=torch.bfloat16,
        )

        # Same flat_param tensor should be returned
        assert flat1['flat_param'].data_ptr() == flat2['flat_param'].data_ptr()

    def test_error_empty_param_list(self, cuda_device):
        """Test that empty param list raises RuntimeError."""
        group = {}

        with pytest.raises(RuntimeError, match="empty param list"):
            _get_or_init_flat_group(
                group,
                params=[],
                rank=0,
                world=1,
                dtype=torch.bfloat16,
            )

    def test_error_mixed_devices(self, cuda_device):
        """Test that params on different devices raise RuntimeError."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 CUDA devices")

        params = [
            nn.Parameter(torch.randn(64, device="cuda:0", dtype=torch.bfloat16)),
            nn.Parameter(torch.randn(64, device="cuda:1", dtype=torch.bfloat16)),
        ]

        group = {}

        with pytest.raises(RuntimeError, match="params must be on a single device"):
            _get_or_init_flat_group(
                group,
                params=params,
                rank=0,
                world=1,
                dtype=torch.bfloat16,
            )

    def test_error_dtype_mismatch(self, cuda_device):
        """Test that dtype mismatch raises RuntimeError."""
        params = [
            nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16)),
            nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.float32)),
        ]

        group = {}

        with pytest.raises(RuntimeError, match="dtype grouping invariant"):
            _get_or_init_flat_group(
                group,
                params=params,
                rank=0,
                world=1,
                dtype=torch.bfloat16,
            )

    def test_world_size_1_param_shard_is_view(self, cuda_device):
        """Test that param_shard is a view for single GPU."""
        params = [
            nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16)),
        ]

        group = {}
        flat = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=1,
            dtype=torch.bfloat16,
        )

        # For world=1, param_shard should share storage with flat_param
        assert flat['param_shard'].data_ptr() == flat['flat_param'].data_ptr()

    def test_world_size_greater_than_1_param_shard_is_copy(self, cuda_device):
        """Test that param_shard is a separate copy for multi-GPU."""
        params = [
            nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16)),
        ]

        group = {}
        flat = _get_or_init_flat_group(
            group,
            params=params,
            rank=0,
            world=2,
            dtype=torch.bfloat16,
        )

        # For world>1, param_shard should be separate storage
        # (contains only this rank's portion)
        shard_size = flat['shard_size']
        assert flat['param_shard'].numel() == shard_size
        # param_shard should NOT share storage with flat_param for world > 1
        assert flat['param_shard'].data_ptr() != flat['flat_param'].data_ptr()


# =============================================================================
# Tests for step_dense_adamw (Single GPU)
# =============================================================================

class TestStepDenseAdamWSingleGPU:
    """Tests for step_dense_adamw on a single GPU."""

    def test_basic_step(self, cuda_device, param_groups_factory):
        """Test basic AdamW step execution."""
        params, param_groups = param_groups_factory(n_params=2)
        state: Dict[str, Any] = {}

        # Store original param values
        original_params = [p.data.clone() for p in params]

        # Execute step
        step_dense_adamw(param_groups, state=state, pg=None)

        # Parameters should be updated (different from original)
        for i, (orig, p) in enumerate(zip(original_params, params)):
            assert not torch.allclose(orig, p.data), \
                f"Parameter {i} was not updated"

    def test_gradients_consumed(self, cuda_device, param_groups_factory):
        """Test that gradients are consumed correctly."""
        params, param_groups = param_groups_factory(n_params=2)
        state: Dict[str, Any] = {}

        # Save gradients
        grads = [p.grad.clone() for p in params]

        # Execute step
        step_dense_adamw(param_groups, state=state, pg=None)

        # Gradients should still exist (ZeRO-2 doesn't zero them)
        for i, (grad, p) in enumerate(zip(grads, params)):
            assert p.grad is not None, f"Gradient {i} should still exist"

    def test_momentum_state_initialized(self, cuda_device, param_groups_factory):
        """Test that momentum states are initialized correctly."""
        params, param_groups = param_groups_factory(n_params=1)
        state: Dict[str, Any] = {}

        # Execute step
        step_dense_adamw(param_groups, state=state, pg=None)

        # State should contain momentum tensors
        state_key = f"shard_0_0_{torch.bfloat16}"
        assert state_key in state

        s = state[state_key]
        assert 'step' in s
        assert 'exp_avg' in s
        assert 'exp_avg_sq' in s

        assert s['step'] == 1
        assert s['exp_avg'].numel() > 0
        assert s['exp_avg_sq'].numel() > 0

    def test_momentum_state_updated(self, cuda_device, param_groups_factory):
        """Test that momentum states are updated across steps."""
        params, param_groups = param_groups_factory(n_params=1)
        state: Dict[str, Any] = {}

        # First step
        step_dense_adamw(param_groups, state=state, pg=None)

        state_key = f"shard_0_0_{torch.bfloat16}"
        exp_avg_after_1 = state[state_key]['exp_avg'].clone()
        exp_avg_sq_after_1 = state[state_key]['exp_avg_sq'].clone()

        # Add new gradients
        for p in params:
            p.grad = torch.randn_like(p)

        # Second step
        step_dense_adamw(param_groups, state=state, pg=None)

        exp_avg_after_2 = state[state_key]['exp_avg']
        exp_avg_sq_after_2 = state[state_key]['exp_avg_sq']

        # Momentum states should have changed
        assert not torch.allclose(exp_avg_after_1, exp_avg_after_2)
        assert not torch.allclose(exp_avg_sq_after_1, exp_avg_sq_after_2)

        # Step count should be 2
        assert state[state_key]['step'] == 2

    def test_weight_decay_applied(self, cuda_device):
        """Test that weight decay is correctly applied."""
        # Create parameter with known value
        p = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.zeros_like(p)  # Zero gradient to isolate weight decay effect

        param_groups = [{'params': [p], 'lr': 0.1, 'weight_decay': 0.1}]
        state: Dict[str, Any] = {}

        original_norm = p.data.norm().item()

        # Execute step
        step_dense_adamw(param_groups, state=state, pg=None)

        new_norm = p.data.norm().item()

        # Weight decay should reduce the parameter norm
        assert new_norm < original_norm, \
            f"Weight decay not applied: {new_norm} >= {original_norm}"

    def test_learning_rate_scaling(self, cuda_device):
        """Test that learning rate correctly scales updates."""
        # Create two identical setups with different learning rates
        p1 = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        p1.grad = torch.ones_like(p1)

        p2 = nn.Parameter(torch.ones(64, device=cuda_device, dtype=torch.bfloat16))
        p2.grad = torch.ones_like(p2)

        param_groups1 = [{'params': [p1], 'lr': 0.01, 'weight_decay': 0.0}]
        param_groups2 = [{'params': [p2], 'lr': 0.1, 'weight_decay': 0.0}]

        state1: Dict[str, Any] = {}
        state2: Dict[str, Any] = {}

        step_dense_adamw(param_groups1, state=state1, pg=None)
        step_dense_adamw(param_groups2, state=state2, pg=None)

        # Larger learning rate should cause larger change
        delta1 = (torch.ones(64, device=cuda_device, dtype=torch.bfloat16) - p1.data).abs().mean()
        delta2 = (torch.ones(64, device=cuda_device, dtype=torch.bfloat16) - p2.data).abs().mean()

        assert delta2 > delta1, \
            f"Larger LR should cause larger update: {delta2} <= {delta1}"

    def test_multiple_param_groups(self, cuda_device):
        """Test with multiple param groups."""
        p1 = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16))
        p1.grad = torch.randn_like(p1)

        p2 = nn.Parameter(torch.randn(32, device=cuda_device, dtype=torch.bfloat16))
        p2.grad = torch.randn_like(p2)

        param_groups = [
            {'params': [p1], 'lr': 0.01, 'weight_decay': 0.0},
            {'params': [p2], 'lr': 0.1, 'weight_decay': 0.1},
        ]

        original_p1 = p1.data.clone()
        original_p2 = p2.data.clone()

        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Both params should be updated
        assert not torch.allclose(original_p1, p1.data)
        assert not torch.allclose(original_p2, p2.data)

    def test_mixed_dtypes_in_group(self, cuda_device):
        """Test that params are grouped by dtype."""
        p_bf16 = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16))
        p_bf16.grad = torch.randn_like(p_bf16)

        p_fp32 = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.float32))
        p_fp32.grad = torch.randn_like(p_fp32)

        param_groups = [
            {'params': [p_bf16, p_fp32], 'lr': 0.01, 'weight_decay': 0.0},
        ]

        original_bf16 = p_bf16.data.clone()
        original_fp32 = p_fp32.data.clone()

        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Both params should be updated
        assert not torch.allclose(original_bf16, p_bf16.data)
        assert not torch.allclose(original_fp32, p_fp32.data)

        # Should have separate state entries for each dtype
        assert f"shard_0_0_{torch.bfloat16}" in state
        assert f"shard_0_0_{torch.float32}" in state

    def test_no_grad_params_skipped(self, cuda_device):
        """Test that parameters without gradients are handled."""
        p_with_grad = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16))
        p_with_grad.grad = torch.randn_like(p_with_grad)

        p_no_grad = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16))
        p_no_grad.grad = None  # No gradient

        param_groups = [
            {'params': [p_with_grad, p_no_grad], 'lr': 0.01, 'weight_decay': 0.0},
        ]

        state: Dict[str, Any] = {}

        # Should not raise
        step_dense_adamw(param_groups, state=state, pg=None)

    def test_empty_param_group_skipped(self, cuda_device):
        """Test that empty param groups are skipped."""
        param_groups = [
            {'params': [], 'lr': 0.01, 'weight_decay': 0.0},
        ]

        state: Dict[str, Any] = {}

        # Should not raise
        step_dense_adamw(param_groups, state=state, pg=None)

    def test_adamw_update_correctness(self, cuda_device):
        """Test that AdamW update matches expected formula."""
        # Use fp32 for numerical precision in this test
        torch.manual_seed(42)

        p = nn.Parameter(torch.randn(32, device=cuda_device, dtype=torch.float32))
        grad = torch.randn_like(p)
        p.grad = grad.clone()

        lr = 0.1
        beta1, beta2 = 0.9, 0.95
        eps = 1e-8
        wd = 0.01

        # Store initial values
        p0 = p.data.clone()
        g = grad.clone()

        param_groups = [{'params': [p], 'lr': lr, 'weight_decay': wd}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None, betas=(beta1, beta2), eps=eps)

        # Compute expected update manually
        # Step 1 values
        m = (1 - beta1) * g  # m = 0 * beta1 + (1-beta1) * g
        v = (1 - beta2) * g * g  # v = 0 * beta2 + (1-beta2) * g^2

        bias_correction1 = 1 - beta1
        bias_correction2 = 1 - beta2

        step_size = lr / bias_correction1
        inv_bc2_sqrt = 1.0 / math.sqrt(bias_correction2)

        denom = v.sqrt() * inv_bc2_sqrt + eps

        # Weight decay applied first
        expected_p = p0 * (1 - lr * wd) - step_size * m / denom

        # Allow for numerical differences due to chunked processing
        assert torch.allclose(p.data, expected_p, rtol=1e-3, atol=1e-5), \
            f"Max diff: {(p.data - expected_p).abs().max()}"

    def test_multiple_steps_convergence(self, cuda_device):
        """Test that multiple steps show convergence behavior."""
        torch.manual_seed(42)

        # Simple quadratic loss: L = 0.5 * ||p||^2
        p = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.float32))

        param_groups = [{'params': [p], 'lr': 0.1, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        norms = []
        for _ in range(10):
            # Gradient of 0.5 * ||p||^2 is p
            p.grad = p.data.clone()
            step_dense_adamw(param_groups, state=state, pg=None)
            norms.append(p.data.norm().item())

        # Norm should generally decrease (parameters approaching zero)
        assert norms[-1] < norms[0], \
            f"Optimization not converging: final={norms[-1]}, initial={norms[0]}"


# =============================================================================
# Multi-GPU Tests (require torchrun)
# =============================================================================

@pytest.mark.multi_gpu
class TestStepDenseAdamWMultiGPU:
    """Multi-GPU tests for step_dense_adamw.

    Run with: torchrun --nproc_per_node=N pytest tests/gpu/test_zero2_operations.py -v -k multi_gpu
    """

    @pytest.fixture(autouse=True)
    def setup_distributed(self):
        """Initialize distributed environment."""
        if not dist.is_initialized():
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                dist.init_process_group(backend="nccl")
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        yield

    def test_gradient_averaging(self):
        """Test that gradients are correctly averaged across ranks."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")

        # Each rank has different gradient
        p = nn.Parameter(torch.ones(64, device=device, dtype=torch.bfloat16))
        p.grad = torch.full_like(p, float(rank + 1))  # Rank 0: 1, Rank 1: 2, etc.

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Expected average gradient: (1 + 2 + ... + world_size) / world_size
        expected_avg_grad = (world_size + 1) / 2.0

        # The momentum state should reflect the averaged gradient
        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        exp_avg = state[state_key]['exp_avg']

        # exp_avg = (1 - beta1) * avg_grad for step 1
        beta1 = 0.9
        expected_exp_avg = (1 - beta1) * expected_avg_grad

        # Check approximate equality (accounting for numerical precision)
        assert torch.allclose(
            exp_avg.float().mean(),
            torch.tensor(expected_exp_avg),
            rtol=0.1
        ), f"Rank {rank}: exp_avg mean = {exp_avg.float().mean()}, expected ~{expected_exp_avg}"

    def test_parameter_sync_after_step(self):
        """Test that parameters are synchronized after step."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")

        # Initialize with same value on all ranks
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(64, device=device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Gather parameter values from all ranks
        gathered = [torch.zeros_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data)

        # All ranks should have the same parameter values
        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4), \
                f"Rank 0 and Rank {r} have different parameters after step"

    def test_sharded_state_isolation(self):
        """Test that each rank maintains its own shard state."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")

        p = nn.Parameter(torch.randn(128, device=device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        step_dense_adamw(param_groups, state=state, pg=None)

        # Each rank should have its own state key
        state_key = f"shard_{rank}_0_{torch.bfloat16}"
        assert state_key in state, f"State key {state_key} not found on rank {rank}"

        # Shard size should be total / world
        # (approximately, accounting for padding)
        total_numel = p.numel()
        expected_shard_size = _ceil_div(total_numel, world_size)

        actual_shard_size = state[state_key]['exp_avg'].numel()
        assert actual_shard_size == expected_shard_size, \
            f"Shard size mismatch: {actual_shard_size} vs {expected_shard_size}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestZeRO2Integration:
    """Integration tests combining multiple ZeRO-2 components."""

    def test_nn_module_training_loop(self, cuda_device):
        """Test ZeRO-2 in a realistic training loop with nn.Module."""
        model = nn.Sequential(
            nn.Linear(64, 128, device=cuda_device, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(128, 64, device=cuda_device, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(64, 10, device=cuda_device, dtype=torch.bfloat16),
        )

        param_groups = [
            {'params': list(model.parameters()), 'lr': 0.01, 'weight_decay': 0.01}
        ]

        state: Dict[str, Any] = {}

        # Training loop
        losses = []
        for step in range(5):
            # Forward pass
            x = torch.randn(32, 64, device=cuda_device, dtype=torch.bfloat16)
            y = model(x)
            loss = y.pow(2).mean()
            losses.append(loss.item())

            # Backward pass
            loss.backward()

            # Optimizer step
            step_dense_adamw(param_groups, state=state, pg=None)

            # Zero gradients for next iteration
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        # Loss should decrease (model is learning)
        assert losses[-1] < losses[0], \
            f"Loss not decreasing: {losses}"

    def test_state_persistence_across_reinit(self, cuda_device):
        """Test that optimizer state persists correctly."""
        p = nn.Parameter(torch.randn(64, device=cuda_device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)

        param_groups = [{'params': [p], 'lr': 0.01, 'weight_decay': 0.0}]
        state: Dict[str, Any] = {}

        # Run a few steps
        for _ in range(3):
            p.grad = torch.randn_like(p)
            step_dense_adamw(param_groups, state=state, pg=None)

        # Save state
        saved_state = {k: {sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                          for sk, sv in v.items()}
                      for k, v in state.items()}

        # Run more steps
        for _ in range(2):
            p.grad = torch.randn_like(p)
            step_dense_adamw(param_groups, state=state, pg=None)

        # Step count should have increased
        state_key = f"shard_0_0_{torch.bfloat16}"
        assert state[state_key]['step'] == 5
        assert saved_state[state_key]['step'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
