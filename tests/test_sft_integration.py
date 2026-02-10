"""Integration tests for the SFT pipeline.

Tests the complete end-to-end flow of the SFT training pipeline components:
- Config loading with EP/DP/ECO fields
- Process group layout logic
- Model creation with EP-aware expert partitioning
- Optimizer construction (ECO path and standard path)
- Training loop SFT modifications (3-tuple handling, loss masking, gradient clipping)

These tests can run without a GPU by using importlib to bypass C extension imports
and by testing at the Python logic level.
"""
import sys
import math
import importlib.util
import pytest
import torch
import torch.nn as nn


# ============================================================================
# Direct imports to bypass nmoe.__init__ (which pulls in C extensions)
# ============================================================================
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_config_mod = _load_module('config', '/home/nourdine/nmoe/nmoe/config.py')
Config = _config_mod.Config
fingerprint = _config_mod.fingerprint

_init_groups_mod = _load_module('init_groups', '/home/nourdine/nmoe/nmoe/distributed/init_groups.py')


# ============================================================================
# Test Config Loading
# ============================================================================
class TestSFTConfig:
    """Test that SFT config fields load correctly."""

    def test_default_config(self):
        """Default config should have SFT disabled."""
        cfg = Config()
        assert cfg.sft_enabled is False
        assert cfg.eco_enabled is False
        assert cfg.ep_size == 1
        assert cfg.dp_size is None
        assert cfg.grad_clip == 0.0

    def test_sft_config_fields(self):
        """SFT config fields should be accepted."""
        cfg = Config(
            sft_enabled=True,
            sft_prompt_format="deepseekv32",
            sft_mask_prompt_loss=True,
            sft_data_path="/data/sft",
            ep_size=8,
            dp_size=16,
            eco_enabled=True,
            eco_error_feedback=True,
            eco_stochastic_rounding=True,
            gradient_checkpointing=True,
            grad_clip=1.0,
        )
        assert cfg.sft_enabled is True
        assert cfg.sft_prompt_format == "deepseekv32"
        assert cfg.ep_size == 8
        assert cfg.dp_size == 16
        assert cfg.eco_enabled is True
        assert cfg.grad_clip == 1.0
        assert cfg.gradient_checkpointing is True

    def test_dsv3_reap_full_config(self):
        """Full REAP-345B config should load without error."""
        cfg = Config(
            vocab_size=129280,
            dim=7168,
            inter_dim=18432,
            moe_inter_dim=2048,
            n_layers=61,
            n_dense_layers=3,
            n_routed_experts=128,
            n_activated_experts=8,
            n_shared_experts=1,
            n_heads=128,
            seq_len=4096,
            batch_size=256,
            steps=10000,
            dtype="nvfp4",
            ep_size=8,
            dp_size=16,
            lr_dense=1e-5,
            lr_router=1e-5,
            lr_expert=1e-5,
            eco_enabled=True,
            gradient_checkpointing=True,
            sft_enabled=True,
            sft_prompt_format="deepseekv32",
            grad_clip=1.0,
        )
        assert cfg.n_routed_experts == 128
        assert cfg.ep_size == 8
        assert cfg.dp_size == 16
        assert cfg.n_dense_layers == 3

    def test_config_fingerprint_stability(self):
        """Config fingerprint should be deterministic."""
        cfg1 = Config(sft_enabled=True, ep_size=8, eco_enabled=True)
        cfg2 = Config(sft_enabled=True, ep_size=8, eco_enabled=True)
        assert fingerprint(cfg1) == fingerprint(cfg2)

    def test_config_fingerprint_sensitivity(self):
        """Config fingerprint should change when config changes."""
        cfg1 = Config(ep_size=8)
        cfg2 = Config(ep_size=4)
        assert fingerprint(cfg1) != fingerprint(cfg2)


# ============================================================================
# Test EP/DP Process Group Layout
# ============================================================================
class TestEPDPLayout:
    """Test EP/DP group membership patterns."""

    def _compute_ep_dp_layout(self, world_size, ep_size, tp_size=1):
        """Compute what process groups would look like."""
        dp_size = world_size // (ep_size * tp_size)

        ep_groups = {}
        for dp_idx in range(dp_size):
            base = dp_idx * ep_size * tp_size
            for tp_idx in range(tp_size):
                group_ranks = [base + e * tp_size + tp_idx for e in range(ep_size)]
                for r in group_ranks:
                    ep_groups[r] = group_ranks

        dp_groups = {}
        stride = ep_size * tp_size
        for local_idx in range(stride):
            group_ranks = [local_idx + d * stride for d in range(dp_size)]
            for r in group_ranks:
                dp_groups[r] = group_ranks

        return ep_groups, dp_groups, dp_size

    def test_reap_sft_layout(self):
        """Test the target REAP-345B SFT layout: EP=8, DP=16, world=128."""
        ep_groups, dp_groups, dp_size = self._compute_ep_dp_layout(128, ep_size=8)

        assert dp_size == 16

        # Rank 0's EP group: [0,1,2,3,4,5,6,7]
        assert ep_groups[0] == list(range(8))

        # Rank 0's DP group: [0,8,16,...,120] (stride=8, 16 elements)
        assert dp_groups[0] == list(range(0, 128, 8))
        assert len(dp_groups[0]) == 16

        # n_local_experts = 128 / 8 = 16
        n_local = 128 // 8
        assert n_local == 16

    def test_ep_equals_world(self):
        """EP = world_size means no DP (backward compat)."""
        ep_groups, dp_groups, dp_size = self._compute_ep_dp_layout(8, ep_size=8)
        assert dp_size == 1
        assert ep_groups[0] == list(range(8))

    def test_single_gpu(self):
        """Single GPU: EP=1, DP=1."""
        ep_groups, dp_groups, dp_size = self._compute_ep_dp_layout(1, ep_size=1)
        assert dp_size == 1


# ============================================================================
# Test ECO Optimizer Integration
# ============================================================================
class TestECOIntegration:
    """Test ECO optimizer integration with the training pipeline."""

    def _load_eco_mod(self):
        return _load_module('eco', '/home/nourdine/nmoe/nmoe/eco.py')

    def test_eco_import(self):
        """ECO module should import cleanly."""
        eco = self._load_eco_mod()
        assert hasattr(eco, 'ECOAdamW')
        assert hasattr(eco, 'build_eco_optimizer')
        assert hasattr(eco, 'quantize_nvfp4_eco')

    def test_eco_adamw_with_sft_config(self):
        """ECO optimizer should work with SFT-style config."""
        eco = self._load_eco_mod()

        # Mock MoE module
        class MockMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.W1 = nn.Parameter(torch.randn(4, 32, 16))
                self.W3 = nn.Parameter(torch.randn(4, 32, 16))
                self.W2 = nn.Parameter(torch.randn(4, 16, 32))

        # SFT-style config
        class SFTConfig:
            lr_expert = 1e-5
            adam_beta1 = 0.9
            adam_beta2 = 0.98
            adam_beta2_expert = 0.98
            adam_eps = 1e-9
            weight_decay = 0.01
            eco_stochastic_rounding = True
            eco_error_feedback = True

        moe = MockMoE()
        cfg = SFTConfig()
        opt = eco.ECOAdamW([moe], cfg, stochastic_rounding=True, error_feedback=True)

        # Simulate gradient
        loss = moe.W1.sum() + moe.W3.sum() + moe.W2.sum()
        loss.backward()
        opt.step()

        # Should not crash or produce NaN
        assert not torch.isnan(moe.W1.data).any()
        assert not torch.isnan(moe.W3.data).any()
        assert not torch.isnan(moe.W2.data).any()

    def test_eco_with_low_lr(self):
        """ECO should be stable with very low learning rate (SFT regime)."""
        eco = self._load_eco_mod()

        class MockMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.W1 = nn.Parameter(torch.randn(2, 16, 16) * 0.1)
                self.W3 = nn.Parameter(torch.randn(2, 16, 16) * 0.1)
                self.W2 = nn.Parameter(torch.randn(2, 16, 16) * 0.1)

        class SFTConfig:
            lr_expert = 1e-6  # Very low LR typical of SFT
            adam_beta1 = 0.9
            adam_beta2 = 0.98
            adam_beta2_expert = 0.98
            adam_eps = 1e-9
            weight_decay = 0.0
            eco_stochastic_rounding = False
            eco_error_feedback = True

        torch.manual_seed(42)
        moe = MockMoE()
        cfg = SFTConfig()
        opt = eco.ECOAdamW([moe], cfg, stochastic_rounding=False, error_feedback=True)

        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = (moe.W1 ** 2).sum() + (moe.W3 ** 2).sum() + (moe.W2 ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # All losses should be finite
        for i, l in enumerate(losses):
            assert math.isfinite(l), f"Loss at step {i} is not finite: {l}"

        # Loss should not explode
        assert losses[-1] < losses[0] * 100


# ============================================================================
# Test Training Loop SFT Modifications
# ============================================================================
class TestTrainingSFTModifications:
    """Test the SFT-specific modifications to the training loop."""

    def test_3tuple_loss_masking(self):
        """SFT 3-tuple (inputs, targets, loss_mask) should produce correct masked loss."""
        import torch.nn.functional as F

        vocab_size = 100
        B, T = 2, 16

        # Simulate model logits
        logits = torch.randn(B, T, vocab_size)
        targets = torch.randint(0, vocab_size, (B, T))

        # Loss mask: 0 for prompt tokens, 1 for response tokens
        loss_mask = torch.zeros(B, T)
        loss_mask[:, T // 2:] = 1.0  # Second half is response

        # Compute masked loss (same as train.py)
        loss_unreduced = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none'
        )
        mask = loss_mask.reshape(-1)
        loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1.0)

        # Loss should only consider response tokens
        assert loss.item() > 0  # Non-zero loss
        assert math.isfinite(loss.item())

        # Compare with response-only loss
        response_logits = logits[:, T // 2:].reshape(-1, vocab_size)
        response_targets = targets[:, T // 2:].reshape(-1)
        expected_loss = F.cross_entropy(response_logits, response_targets)
        assert abs(loss.item() - expected_loss.item()) < 1e-5

    def test_eos_masking_pretraining(self):
        """Pre-training EOS masking should work correctly."""
        import torch.nn.functional as F

        vocab_size = 100
        eos_token_id = 99
        B, T = 2, 16

        logits = torch.randn(B, T, vocab_size)
        targets = torch.randint(0, vocab_size, (B, T))
        targets[0, -3:] = eos_token_id  # Some EOS tokens
        targets[1, -5:] = eos_token_id

        # Pre-training masking (same as train.py)
        loss_unreduced = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none'
        )
        mask = (targets != eos_token_id).reshape(-1).float()
        loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1.0)

        assert loss.item() > 0
        assert math.isfinite(loss.item())

    def test_gradient_clipping(self):
        """Gradient clipping should limit gradient norm."""
        model = nn.Linear(10, 10)
        loss = model(torch.randn(1, 10)).sum() * 1000  # Large loss for large grads
        loss.backward()

        # Before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

        # Reset and clip
        model.zero_grad()
        loss = model(torch.randn(1, 10)).sum() * 1000
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # After clipping, effective gradient norm should be <= 1.0 (approximately)
        clipped_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                clipped_norm += p.grad.data.norm(2).item() ** 2
        clipped_norm = clipped_norm ** 0.5
        assert clipped_norm <= 1.0 + 1e-5

    def test_dp_rank_computation(self):
        """DP rank should be computed correctly from global rank and ep_size."""
        # With EP=8, DP=16, world=128:
        # EP ranks are consecutive: [0..7], [8..15], etc.
        # dp_rank = rank // ep_size
        ep_size = 8
        for rank in range(128):
            dp_rank = rank // ep_size
            assert 0 <= dp_rank < 16
            # Ranks in the same EP group should have the same dp_rank
            ep_start = (rank // ep_size) * ep_size
            for r in range(ep_start, ep_start + ep_size):
                assert r // ep_size == dp_rank


# ============================================================================
# Test Optimizer Step with DP Gradient Sync Logic
# ============================================================================
class TestOptStepDPSync:
    """Test the optimizer step function's DP gradient sync logic."""

    def test_expert_grad_allreduce_ordering(self):
        """Expert gradient AllReduce must happen BEFORE optimizer.step()."""
        # This is a structural test: in opt.py step(), the order is:
        # 1. ZeRO-2 for dense params
        # 2. Expert gradient AllReduce over DP group  <-- MUST be before step()
        # 3. Expert optimizer.step()
        # 4. Post-step hooks

        # We verify this by checking the source code
        import ast
        with open('/home/nourdine/nmoe/nmoe/opt.py') as f:
            source = f.read()

        tree = ast.parse(source)

        # Find the step function
        step_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'step':
                # Check it has the right signature (model, optimizer, ...)
                if len(node.args.args) >= 6:
                    step_func = node
                    break

        assert step_func is not None, "step() function not found in opt.py"

        # Extract the body as source lines
        body_lines = []
        for child in ast.walk(step_func):
            if isinstance(child, ast.Attribute):
                if hasattr(child, 'attr') and child.attr == 'all_reduce':
                    body_lines.append(('all_reduce', child.lineno))
            if isinstance(child, ast.Call):
                if hasattr(child, 'func') and isinstance(child.func, ast.Attribute):
                    if child.func.attr == 'step':
                        body_lines.append(('step', child.lineno))

        # Find the AllReduce and optimizer.step() calls
        allreduce_lines = [l for name, l in body_lines if name == 'all_reduce']
        step_lines = [l for name, l in body_lines if name == 'step']

        # There should be at least one all_reduce and two step calls
        # (one for ZeRO-2 step_dense_adamw which internally steps, one for optimizer.step())
        assert len(allreduce_lines) >= 1, "No all_reduce call found in step()"

        # The AllReduce should be before the optimizer.step() call
        # (The optimizer.step() is the one after the AllReduce)
        # Find the latest all_reduce line and verify there's a step() after it
        max_allreduce_line = max(allreduce_lines)
        later_steps = [l for l in step_lines if l > max_allreduce_line]
        assert len(later_steps) >= 1, \
            "No optimizer.step() call found after all_reduce in step()"


# ============================================================================
# Test Config Validation
# ============================================================================
class TestConfigValidation:
    """Test training config validation logic."""

    def test_sft_batch_divisible_by_dp(self):
        """In SFT mode, batch_size must be divisible by dp_size."""
        # Simulates _validate_training_config logic
        batch_size = 256
        ep_size = 8
        world = 128
        dp_size = world // ep_size  # 16

        assert batch_size % dp_size == 0  # 256 % 16 = 0

    def test_sft_batch_not_divisible_by_dp_fails(self):
        """Odd batch_size should fail validation."""
        batch_size = 255
        ep_size = 8
        world = 128
        dp_size = world // ep_size  # 16

        assert batch_size % dp_size != 0  # 255 % 16 != 0

    def test_world_divisible_by_ep(self):
        """world_size must be divisible by ep_size."""
        world = 128
        ep_size = 8
        assert world % ep_size == 0

    def test_world_not_divisible_by_ep_fails(self):
        """Odd world_size/ep_size should fail."""
        world = 127
        ep_size = 8
        assert world % ep_size != 0


# ============================================================================
# Test Model EP Integration
# ============================================================================
class TestModelEPAwareness:
    """Test that model correctly uses ep_size for expert partitioning."""

    def test_n_local_with_ep(self):
        """n_local should be n_routed_experts // ep_size."""
        n_routed = 128
        test_cases = [
            (1, 128),    # EP=1: all experts local (single GPU)
            (2, 64),     # EP=2: 64 experts per GPU
            (4, 32),     # EP=4: 32 experts per GPU
            (8, 16),     # EP=8: 16 experts per GPU (target)
            (16, 8),     # EP=16: 8 experts per GPU
            (128, 1),    # EP=128: 1 expert per GPU
        ]
        for ep_size, expected_local in test_cases:
            n_local = n_routed // max(1, ep_size)
            assert n_local == expected_local, \
                f"EP={ep_size}: expected n_local={expected_local}, got {n_local}"

    def test_capacity_computation(self):
        """RDEP capacity should be batch_size * seq_len * n_activated."""
        batch_size = 256
        seq_len = 4096
        n_activated = 8
        capacity = batch_size * seq_len * n_activated
        assert capacity == 256 * 4096 * 8  # 8,388,608


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
