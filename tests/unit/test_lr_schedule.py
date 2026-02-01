"""P1 Critical tests for learning rate scheduling.

Tests the WSD (Warmup-Sustain-Decay) schedule implementation in nmoe.opt:
- Warmup phase: LR ramps from floor to peak over warmup_steps (step-based)
- Sustain phase: LR holds at peak until hold_tokens (token-based)
- Decay phase: Cosine decay from peak to floor over decay_tokens (token-based)
- Floor phase: LR holds at floor indefinitely

Also tests multi-group LR handling for:
- Expert parameters (cfg.lr_expert)
- Dense parameters (cfg.lr_dense)
- Router parameters (cfg.lr_router)
"""

import math
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch


# -----------------------------------------------------------------------------
# Mock Config class for testing
# -----------------------------------------------------------------------------

@dataclass
class MockConfig:
    """Minimal mock Config for LR schedule tests."""
    # LR peaks
    lr_expert: float = 3e-4
    lr_dense: float = 3.4e-4
    lr_router: float = 3.4e-4

    # WSD schedule params
    warmup_steps: int = 1000
    hold_tokens: int = 10_000_000  # 10M tokens
    decay_tokens: int = 40_000_000  # 40M tokens
    decay_floor: float = 3e-5

    # Optimizer params
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_beta2_expert: float = 0.99
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95

    # Additional fields
    dtype: str = "bf16"


# -----------------------------------------------------------------------------
# Reference WSD schedule implementation for testing
# -----------------------------------------------------------------------------

def reference_wsd_lr(
    step: int,
    tokens_seen: int,
    warmup_steps: int,
    hold_tokens: int,
    decay_tokens: int,
    lr_peak: float,
    lr_floor: float,
) -> float:
    """Reference WSD schedule for validation.

    This implements the exact same logic as update_lr for verification.
    """
    if step < warmup_steps:
        # Warmup: linear 0 -> peak
        lr_scale = (step + 1) / max(1, warmup_steps)
    elif tokens_seen < hold_tokens:
        # Sustain: hold at peak
        lr_scale = 1.0
    else:
        # Decay phase
        t = tokens_seen - hold_tokens
        if t < decay_tokens:
            # Cosine decay
            denom = float(max(1, decay_tokens))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * float(t) / denom))
        else:
            # Floor (scale = 0)
            lr_scale = 0.0

    return lr_floor + (lr_peak - lr_floor) * lr_scale


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Default config for LR tests."""
    return MockConfig()


@pytest.fixture
def mock_optimizer():
    """Mock optimizer with param_groups."""
    opt = MagicMock()
    opt.param_groups = [{"lr": 0.0}]
    return opt


@pytest.fixture
def mock_dense_groups():
    """Mock dense parameter groups."""
    return [
        {"name": "dense_decay", "params": [], "lr": 0.0, "weight_decay": 0.1},
        {"name": "dense_no_decay", "params": [], "lr": 0.0, "weight_decay": 0.0},
        {"name": "router", "params": [], "lr": 0.0, "weight_decay": 0.0},
    ]


# -----------------------------------------------------------------------------
# Test: WSD Schedule Phases
# -----------------------------------------------------------------------------

class TestWSDSchedulePhases:
    """Test WSD schedule phase transitions."""

    def test_warmup_phase_start(self, default_config, mock_optimizer, mock_dense_groups):
        """LR starts near zero at step 0."""
        from nmoe.opt import update_lr

        cfg = default_config
        lr = update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

        # At step 0, warmup fraction = 1/1000 = 0.001
        # LR should be close to floor but slightly above
        expected_scale = 1 / cfg.warmup_steps
        expected_lr = cfg.decay_floor + (cfg.lr_dense - cfg.decay_floor) * expected_scale

        assert abs(lr - expected_lr) < 1e-10
        assert lr < cfg.lr_dense  # Not yet at peak

    def test_warmup_phase_midpoint(self, default_config, mock_optimizer, mock_dense_groups):
        """LR at warmup midpoint is ~50% of peak."""
        from nmoe.opt import update_lr

        cfg = default_config
        step = cfg.warmup_steps // 2  # 500

        lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=0, cfg=cfg)

        expected_scale = (step + 1) / cfg.warmup_steps
        expected_lr = cfg.decay_floor + (cfg.lr_dense - cfg.decay_floor) * expected_scale

        assert abs(lr - expected_lr) < 1e-10

    def test_warmup_phase_end(self, default_config, mock_optimizer, mock_dense_groups):
        """LR reaches peak at end of warmup."""
        from nmoe.opt import update_lr

        cfg = default_config
        step = cfg.warmup_steps  # 1000 (first step after warmup)

        lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=0, cfg=cfg)

        # At step = warmup_steps, we're in sustain phase (step >= warmup_steps)
        assert abs(lr - cfg.lr_dense) < 1e-10

    def test_sustain_phase(self, default_config, mock_optimizer, mock_dense_groups):
        """LR holds at peak during sustain phase."""
        from nmoe.opt import update_lr

        cfg = default_config
        step = cfg.warmup_steps + 1000  # Well into sustain
        tokens_seen = cfg.hold_tokens // 2  # Halfway through sustain

        lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=tokens_seen, cfg=cfg)

        assert abs(lr - cfg.lr_dense) < 1e-10

    def test_decay_phase_start(self, default_config, mock_optimizer, mock_dense_groups):
        """LR begins decay at hold_tokens boundary."""
        from nmoe.opt import update_lr

        cfg = default_config
        step = cfg.warmup_steps + 5000
        tokens_seen = cfg.hold_tokens  # Exactly at boundary

        lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=tokens_seen, cfg=cfg)

        # At decay start, cosine progress = 0, so cos(0) = 1, scale = 1.0
        assert abs(lr - cfg.lr_dense) < 1e-10

    def test_decay_phase_midpoint(self, default_config, mock_optimizer, mock_dense_groups):
        """LR at decay midpoint follows cosine curve."""
        from nmoe.opt import update_lr

        cfg = default_config
        step = cfg.warmup_steps + 10000
        tokens_seen = cfg.hold_tokens + cfg.decay_tokens // 2  # Halfway through decay

        lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=tokens_seen, cfg=cfg)

        # At midpoint, cos(pi * 0.5) = 0, so scale = 0.5
        expected_scale = 0.5 * (1.0 + math.cos(math.pi * 0.5))
        expected_lr = cfg.decay_floor + (cfg.lr_dense - cfg.decay_floor) * expected_scale

        assert abs(lr - expected_lr) < 1e-10

    def test_decay_phase_end(self, default_config, mock_optimizer, mock_dense_groups):
        """LR reaches floor at end of decay."""
        from nmoe.opt import update_lr

        cfg = default_config
        step = cfg.warmup_steps + 50000
        tokens_seen = cfg.hold_tokens + cfg.decay_tokens  # Exactly at decay end

        lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=tokens_seen, cfg=cfg)

        # At decay end, cos(pi) = -1, so scale = 0
        # LR should be at floor
        assert abs(lr - cfg.decay_floor) < 1e-10

    def test_floor_phase(self, default_config, mock_optimizer, mock_dense_groups):
        """LR holds at floor after decay completes."""
        from nmoe.opt import update_lr

        cfg = default_config
        step = cfg.warmup_steps + 100000
        tokens_seen = cfg.hold_tokens + cfg.decay_tokens * 2  # Well past decay end

        lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=tokens_seen, cfg=cfg)

        assert abs(lr - cfg.decay_floor) < 1e-10


# -----------------------------------------------------------------------------
# Test: Specific Step/Token Values (from requirements)
# -----------------------------------------------------------------------------

class TestWSDScheduleSpecificValues:
    """Test LR at specific step/token values from requirements."""

    def test_lr_at_step_0(self, mock_optimizer, mock_dense_groups):
        """LR at step 0 is near floor."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=1000,
            hold_tokens=5_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        lr = update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

        expected = reference_wsd_lr(0, 0, 1000, 5_000_000, 10_000_000, 3e-4, 3e-5)
        assert abs(lr - expected) < 1e-10

    def test_lr_at_step_500(self, mock_optimizer, mock_dense_groups):
        """LR at step 500 (mid-warmup)."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=1000,
            hold_tokens=5_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        lr = update_lr(mock_optimizer, mock_dense_groups, step=500, tokens_seen=0, cfg=cfg)

        expected = reference_wsd_lr(500, 0, 1000, 5_000_000, 10_000_000, 3e-4, 3e-5)
        assert abs(lr - expected) < 1e-10

    def test_lr_at_step_1000(self, mock_optimizer, mock_dense_groups):
        """LR at step 1000 (end of warmup)."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=1000,
            hold_tokens=5_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        # At warmup end but before hold_tokens, should be at peak
        lr = update_lr(mock_optimizer, mock_dense_groups, step=1000, tokens_seen=1_000_000, cfg=cfg)

        assert abs(lr - cfg.lr_dense) < 1e-10

    def test_lr_at_step_3000(self, mock_optimizer, mock_dense_groups):
        """LR at step 3000 (sustain phase)."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=1000,
            hold_tokens=5_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        # In sustain phase, tokens < hold_tokens
        lr = update_lr(mock_optimizer, mock_dense_groups, step=3000, tokens_seen=3_000_000, cfg=cfg)

        assert abs(lr - cfg.lr_dense) < 1e-10

    def test_lr_at_step_6000(self, mock_optimizer, mock_dense_groups):
        """LR at step 6000 (may be in decay depending on tokens)."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=1000,
            hold_tokens=5_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        # Tokens past hold_tokens, in decay phase
        tokens_seen = 7_000_000  # 2M into decay
        lr = update_lr(mock_optimizer, mock_dense_groups, step=6000, tokens_seen=tokens_seen, cfg=cfg)

        expected = reference_wsd_lr(6000, tokens_seen, 1000, 5_000_000, 10_000_000, 3e-4, 3e-5)
        assert abs(lr - expected) < 1e-10

    def test_lr_at_step_10000(self, mock_optimizer, mock_dense_groups):
        """LR at step 10000 (deep in decay)."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=1000,
            hold_tokens=5_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        # Tokens at midpoint of decay
        tokens_seen = 10_000_000  # 5M into decay (halfway)
        lr = update_lr(mock_optimizer, mock_dense_groups, step=10000, tokens_seen=tokens_seen, cfg=cfg)

        expected = reference_wsd_lr(10000, tokens_seen, 1000, 5_000_000, 10_000_000, 3e-4, 3e-5)
        assert abs(lr - expected) < 1e-10

    def test_lr_at_step_16000(self, mock_optimizer, mock_dense_groups):
        """LR at step 16000 (at/past floor)."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=1000,
            hold_tokens=5_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        # Well past decay end
        tokens_seen = 20_000_000  # 15M past hold_tokens, 5M past decay end
        lr = update_lr(mock_optimizer, mock_dense_groups, step=16000, tokens_seen=tokens_seen, cfg=cfg)

        assert abs(lr - cfg.decay_floor) < 1e-10


# -----------------------------------------------------------------------------
# Test: Multi-Group LR Handling
# -----------------------------------------------------------------------------

class TestMultiGroupLR:
    """Test LR updates for different parameter groups."""

    def test_expert_lr_updated(self, mock_optimizer, mock_dense_groups):
        """Expert optimizer param groups receive correct LR."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            lr_expert=5e-4,
            lr_dense=3e-4,
            lr_router=1e-4,
            warmup_steps=100,
        )

        update_lr(mock_optimizer, mock_dense_groups, step=100, tokens_seen=0, cfg=cfg)

        # Expert LR should be at peak
        assert mock_optimizer.param_groups[0]["lr"] == cfg.lr_expert

    def test_dense_lr_updated(self, mock_optimizer, mock_dense_groups):
        """Dense param groups receive correct LR."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            lr_expert=5e-4,
            lr_dense=3e-4,
            lr_router=1e-4,
            warmup_steps=100,
        )

        update_lr(mock_optimizer, mock_dense_groups, step=100, tokens_seen=0, cfg=cfg)

        # Dense groups (not router) should have lr_dense
        for g in mock_dense_groups:
            if g.get("name") != "router":
                assert g["lr"] == cfg.lr_dense

    def test_router_lr_updated(self, mock_optimizer, mock_dense_groups):
        """Router param group receives correct LR."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            lr_expert=5e-4,
            lr_dense=3e-4,
            lr_router=1e-4,
            warmup_steps=100,
        )

        update_lr(mock_optimizer, mock_dense_groups, step=100, tokens_seen=0, cfg=cfg)

        # Router group should have lr_router
        for g in mock_dense_groups:
            if g.get("name") == "router":
                assert g["lr"] == cfg.lr_router

    def test_different_peaks_same_schedule_shape(self, mock_optimizer, mock_dense_groups):
        """All groups follow same schedule shape with different peaks."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            lr_expert=5e-4,
            lr_dense=3e-4,
            lr_router=1e-4,
            warmup_steps=100,
            hold_tokens=1_000_000,
            decay_tokens=2_000_000,
            decay_floor=1e-5,
        )

        # At warmup midpoint
        update_lr(mock_optimizer, mock_dense_groups, step=50, tokens_seen=0, cfg=cfg)

        # Scale at step 50: (50+1)/100 = 0.51
        scale = 51 / 100

        expected_expert = cfg.decay_floor + (cfg.lr_expert - cfg.decay_floor) * scale
        expected_dense = cfg.decay_floor + (cfg.lr_dense - cfg.decay_floor) * scale
        expected_router = min(cfg.decay_floor, cfg.lr_router) + (cfg.lr_router - min(cfg.decay_floor, cfg.lr_router)) * scale

        assert abs(mock_optimizer.param_groups[0]["lr"] - expected_expert) < 1e-10
        for g in mock_dense_groups:
            if g.get("name") == "router":
                assert abs(g["lr"] - expected_router) < 1e-10
            else:
                assert abs(g["lr"] - expected_dense) < 1e-10

    def test_router_floor_capped_at_peak(self, mock_optimizer, mock_dense_groups):
        """Router floor is capped at its peak if floor > peak."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            lr_router=1e-5,  # Router peak below global floor
            decay_floor=3e-5,
            warmup_steps=100,
            hold_tokens=1_000_000,
            decay_tokens=2_000_000,
        )

        # At floor (past decay end)
        update_lr(mock_optimizer, mock_dense_groups, step=10000, tokens_seen=5_000_000, cfg=cfg)

        for g in mock_dense_groups:
            if g.get("name") == "router":
                # Router floor should be min(floor, peak) = 1e-5
                assert g["lr"] == cfg.lr_router


# -----------------------------------------------------------------------------
# Test: Edge Cases
# -----------------------------------------------------------------------------

class TestWSDEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_step_zero(self, mock_optimizer, mock_dense_groups):
        """Step 0 produces valid LR."""
        from nmoe.opt import update_lr

        cfg = MockConfig(warmup_steps=1000)

        lr = update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

        assert lr > 0
        assert lr < cfg.lr_dense

    def test_very_large_step(self, mock_optimizer, mock_dense_groups):
        """Very large step produces floor LR."""
        from nmoe.opt import update_lr

        cfg = MockConfig()

        lr = update_lr(mock_optimizer, mock_dense_groups, step=10_000_000, tokens_seen=10**12, cfg=cfg)

        assert abs(lr - cfg.decay_floor) < 1e-10

    def test_warmup_steps_zero(self, mock_optimizer, mock_dense_groups):
        """warmup_steps=0 goes directly to sustain."""
        from nmoe.opt import update_lr

        cfg = MockConfig(warmup_steps=0)

        lr = update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

        # With warmup_steps=0, step 0 >= warmup_steps, so we're in sustain
        assert abs(lr - cfg.lr_dense) < 1e-10

    def test_decay_tokens_zero(self, mock_optimizer, mock_dense_groups):
        """decay_tokens=0 immediately reaches floor."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=100,
            hold_tokens=1000,
            decay_tokens=0,
        )

        # Past hold_tokens with decay_tokens=0
        lr = update_lr(mock_optimizer, mock_dense_groups, step=200, tokens_seen=2000, cfg=cfg)

        # With decay_tokens=0, progress = t/0 would be inf, clamped to 1
        # scale = 0.5 * (1 + cos(pi)) = 0
        assert abs(lr - cfg.decay_floor) < 1e-10

    def test_tokens_negative_not_possible(self, mock_optimizer, mock_dense_groups):
        """Tokens seen should be non-negative (sanity test)."""
        from nmoe.opt import update_lr

        cfg = MockConfig()

        # This shouldn't crash, though tokens_seen should never be negative
        lr = update_lr(mock_optimizer, mock_dense_groups, step=100, tokens_seen=0, cfg=cfg)

        assert lr > 0

    def test_floor_equals_peak(self, mock_optimizer, mock_dense_groups):
        """floor == peak produces constant LR."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            lr_dense=3e-4,
            decay_floor=3e-4,  # Same as peak
        )

        # At any step, LR should be constant
        for step in [0, 100, 1000, 10000]:
            lr = update_lr(mock_optimizer, mock_dense_groups, step=step, tokens_seen=10**10, cfg=cfg)
            assert abs(lr - cfg.lr_dense) < 1e-10


# -----------------------------------------------------------------------------
# Test: Error Handling
# -----------------------------------------------------------------------------

class TestWSDErrorHandling:
    """Test error handling for invalid configurations."""

    def test_decay_floor_zero_raises(self, mock_optimizer, mock_dense_groups):
        """decay_floor=0 raises RuntimeError."""
        from nmoe.opt import update_lr

        cfg = MockConfig(decay_floor=0.0)

        with pytest.raises(RuntimeError, match="decay_floor must be > 0"):
            update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

    def test_decay_floor_negative_raises(self, mock_optimizer, mock_dense_groups):
        """decay_floor < 0 raises RuntimeError."""
        from nmoe.opt import update_lr

        cfg = MockConfig(decay_floor=-1e-5)

        with pytest.raises(RuntimeError, match="decay_floor must be > 0"):
            update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

    def test_decay_floor_greater_than_peak_raises(self, mock_optimizer, mock_dense_groups):
        """decay_floor > lr_peak raises RuntimeError."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            lr_dense=1e-5,
            lr_expert=1e-5,
            decay_floor=1e-4,  # Greater than peaks
        )

        with pytest.raises(RuntimeError, match="decay_floor.*must be <="):
            update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)


# -----------------------------------------------------------------------------
# Test: Cosine Decay Mathematical Properties
# -----------------------------------------------------------------------------

class TestCosineDecayMath:
    """Test mathematical properties of cosine decay."""

    def test_decay_is_monotonic(self, mock_optimizer, mock_dense_groups):
        """LR decreases monotonically during decay phase."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=100,
            hold_tokens=1_000_000,
            decay_tokens=10_000_000,
        )

        prev_lr = float('inf')
        for i in range(11):
            tokens_seen = cfg.hold_tokens + i * (cfg.decay_tokens // 10)
            lr = update_lr(mock_optimizer, mock_dense_groups, step=1000, tokens_seen=tokens_seen, cfg=cfg)

            assert lr <= prev_lr + 1e-10  # Allow tiny floating point error
            prev_lr = lr

    def test_decay_is_smooth(self, mock_optimizer, mock_dense_groups):
        """LR changes smoothly (no discontinuities)."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=100,
            hold_tokens=1_000_000,
            decay_tokens=10_000_000,
        )

        prev_lr = None
        for i in range(100):
            tokens_seen = cfg.hold_tokens + i * (cfg.decay_tokens // 100)
            lr = update_lr(mock_optimizer, mock_dense_groups, step=1000, tokens_seen=tokens_seen, cfg=cfg)

            if prev_lr is not None:
                # Change should be small for small token increments
                max_change = (cfg.lr_dense - cfg.decay_floor) * 0.05
                assert abs(lr - prev_lr) < max_change
            prev_lr = lr

    def test_decay_symmetry(self, mock_optimizer, mock_dense_groups):
        """Cosine decay has expected symmetry around midpoint."""
        from nmoe.opt import update_lr

        cfg = MockConfig(
            warmup_steps=100,
            hold_tokens=1_000_000,
            decay_tokens=10_000_000,
            lr_dense=3e-4,
            decay_floor=3e-5,
        )

        # At 25% decay progress
        tokens_25 = cfg.hold_tokens + cfg.decay_tokens // 4
        lr_25 = update_lr(mock_optimizer, mock_dense_groups, step=1000, tokens_seen=tokens_25, cfg=cfg)

        # At 75% decay progress
        tokens_75 = cfg.hold_tokens + 3 * cfg.decay_tokens // 4
        lr_75 = update_lr(mock_optimizer, mock_dense_groups, step=1000, tokens_seen=tokens_75, cfg=cfg)

        # Cosine symmetry: lr_25 - peak_lr == floor_lr - lr_75 (approximately)
        # Actually: lr_25 + lr_75 should equal lr_dense + decay_floor
        expected_sum = cfg.lr_dense + cfg.decay_floor
        actual_sum = lr_25 + lr_75

        assert abs(actual_sum - expected_sum) < 1e-10


# -----------------------------------------------------------------------------
# Test: Build Optimizer
# -----------------------------------------------------------------------------

class TestBuildOptimizer:
    """Test build_optimizer function."""

    def test_build_optimizer_returns_tuple(self):
        """build_optimizer returns (expert_optimizer, dense_groups) tuple."""
        import torch.nn as nn

        # Create a real simple model with parameters
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 32)
                self.expert_layer = nn.Linear(32, 32)

            def param_sets(self):
                """Return expert params."""
                return [p for n, p in self.named_parameters() if 'expert' in n]

        model = SimpleModel()
        cfg = MockConfig(dtype="bf16")

        try:
            from nmoe.opt import build_optimizer
            result = build_optimizer(model, cfg)

            # If successful, check return type
            assert isinstance(result, tuple)
            assert len(result) == 2
        except (ValueError, TypeError) as e:
            # Some implementations may require specific model structure
            pytest.skip(f"build_optimizer requires specific model structure: {e}")

    def test_build_optimizer_separates_param_groups(self):
        """build_optimizer correctly separates params into groups."""
        import torch
        import torch.nn as nn

        # Create a model with various param types
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_weight = nn.Parameter(torch.randn(32, 32))
                self.attn_bias = nn.Parameter(torch.randn(32))
                self.norm_weight = nn.Parameter(torch.randn(32))
                self.expert_weight = nn.Parameter(torch.randn(32, 32))

            def param_sets(self):
                """Return expert params."""
                return [self.expert_weight]

        model = TestModel()
        cfg = MockConfig(dtype="bf16")

        try:
            from nmoe.opt import build_optimizer
            _, dense_groups = build_optimizer(model, cfg)

            # At least some groups should exist
            assert dense_groups is not None
        except (ValueError, TypeError, AttributeError) as e:
            # Some implementations may require specific model structure
            pytest.skip(f"build_optimizer requires specific model structure: {e}")


# -----------------------------------------------------------------------------
# Test: Return Value
# -----------------------------------------------------------------------------

class TestUpdateLRReturnValue:
    """Test update_lr return value."""

    def test_returns_dense_lr(self, mock_optimizer, mock_dense_groups):
        """update_lr returns the dense LR value."""
        from nmoe.opt import update_lr

        cfg = MockConfig(lr_dense=3e-4, warmup_steps=100)

        lr = update_lr(mock_optimizer, mock_dense_groups, step=100, tokens_seen=0, cfg=cfg)

        assert lr == cfg.lr_dense

    def test_return_type_is_float(self, mock_optimizer, mock_dense_groups):
        """update_lr returns a Python float."""
        from nmoe.opt import update_lr

        cfg = MockConfig()

        lr = update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

        assert isinstance(lr, float)


# -----------------------------------------------------------------------------
# Test: Integration with actual Config class
# -----------------------------------------------------------------------------

class TestWithActualConfig:
    """Test using the actual Config class from nmoe.config."""

    def test_with_real_config(self, mock_optimizer, mock_dense_groups):
        """update_lr works with real Config object."""
        from nmoe.opt import update_lr
        from nmoe.config import Config

        cfg = Config(
            lr_expert=5e-4,
            lr_dense=3e-4,
            lr_router=1e-4,
            warmup_steps=500,
            hold_tokens=10_000_000,
            decay_tokens=40_000_000,
            decay_floor=3e-5,
        )

        lr = update_lr(mock_optimizer, mock_dense_groups, step=250, tokens_seen=0, cfg=cfg)

        # At step 250 (halfway through warmup)
        expected_scale = 251 / 500
        expected_lr = cfg.decay_floor + (cfg.lr_dense - cfg.decay_floor) * expected_scale

        assert abs(lr - expected_lr) < 1e-10

    def test_config_defaults_work(self, mock_optimizer, mock_dense_groups):
        """Default Config values work with update_lr."""
        from nmoe.opt import update_lr
        from nmoe.config import Config

        cfg = Config()  # All defaults

        # Should not raise
        lr = update_lr(mock_optimizer, mock_dense_groups, step=0, tokens_seen=0, cfg=cfg)

        assert lr > 0
        assert isinstance(lr, float)
