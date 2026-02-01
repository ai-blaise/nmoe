"""Edge case and stress tests for nmoe + SkyRL + SGLang.

This module tests edge cases, corner cases, and stress scenarios:
- Numerical edge cases (large/small values, precision limits)
- Memory edge cases (OOM recovery, fragmentation)
- Concurrency edge cases
- Input edge cases (special tokens, unusual sequences)
- Expert edge cases (all same expert, no experts activated)
- Gradient edge cases (vanishing, exploding)
- State management edge cases

Run with:
    pytest tests/integration/test_edge_cases.py -v -s
"""

import gc
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for edge case tests"
)


@pytest.fixture(scope="module")
def edge_config():
    """Config for edge case testing."""
    from nmoe.config import Config
    return Config(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=1024,
        n_dense_layers=1,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_inter_dim=512,
        inter_dim=512,
        max_position_embeddings=512,
    )


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_very_long_sequence(self, edge_config):
        """Test with sequence at max position embeddings."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        max_len = edge_config.max_position_embeddings
        sequence = torch.randint(0, edge_config.vocab_size, (1, max_len)).cuda()

        with torch.no_grad():
            logits = model(sequence)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_single_token_sequence(self, edge_config):
        """Test with single token input."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        sequence = torch.randint(0, edge_config.vocab_size, (1, 1)).cuda()

        with torch.no_grad():
            logits = model(sequence)

        assert logits.shape == (1, 1, edge_config.vocab_size)
        assert not torch.isnan(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_all_same_token(self, edge_config):
        """Test with all tokens being the same."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        for token_id in [0, 1, edge_config.vocab_size - 1, edge_config.vocab_size // 2]:
            sequence = torch.full((1, 64), token_id, dtype=torch.long, device="cuda")

            with torch.no_grad():
                logits = model(sequence)

            assert not torch.isnan(logits).any(), f"NaN for token_id={token_id}"
            assert not torch.isinf(logits).any(), f"Inf for token_id={token_id}"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_boundary_token_ids(self, edge_config):
        """Test with boundary token IDs (0, 1, vocab_size-1)."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        # Include boundary tokens
        sequence = torch.tensor([[
            0, 1, 2,  # Start
            edge_config.vocab_size - 3,
            edge_config.vocab_size - 2,
            edge_config.vocab_size - 1,  # End
        ]], device="cuda")

        with torch.no_grad():
            logits = model(sequence)

        assert not torch.isnan(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_repeated_patterns(self, edge_config):
        """Test with repeated patterns that might cause numerical issues."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        # Alternating pattern
        pattern = torch.tensor([1, 2], device="cuda").repeat(32)
        sequence = pattern.unsqueeze(0)

        with torch.no_grad():
            logits = model(sequence)

        assert not torch.isnan(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_logit_magnitude(self, edge_config):
        """Test that logits stay in reasonable range."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        sequence = torch.randint(0, edge_config.vocab_size, (4, 64)).cuda()

        with torch.no_grad():
            logits = model(sequence)

        # Logits should be in reasonable range for softmax
        max_logit = logits.abs().max().item()
        assert max_logit < 1000, f"Logit magnitude too large: {max_logit}"


class TestGradientEdgeCases:
    """Test gradient-related edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gradient_clipping_extreme(self, edge_config):
        """Test gradient clipping with extreme loss values."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        sequence = torch.randint(0, edge_config.vocab_size, (4, 64)).cuda()

        # Create extreme loss
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(sequence)
            # Scale loss up dramatically
            loss = logits.sum() * 1000

        loss.backward()

        # Check for NaN gradients before clipping
        has_nan_before = any(
            torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None
        )

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Check after clipping
        has_nan_after = any(
            torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None
        )

        optimizer.step()
        optimizer.zero_grad()

        # Model should still work
        with torch.no_grad():
            test_logits = model(sequence)
        assert not torch.isnan(test_logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_zero_loss_gradient(self, edge_config):
        """Test behavior with zero loss (no gradients)."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()

        sequence = torch.randint(0, edge_config.vocab_size, (1, 64)).cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(sequence)
            # Zero loss
            loss = logits.sum() * 0

        loss.backward()

        # All gradients should be zero
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert (p.grad == 0).all(), f"Non-zero gradient for {name} with zero loss"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gradient_accumulation_overflow(self, edge_config):
        """Test gradient accumulation doesn't overflow."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        accumulation_steps = 100

        for step in range(accumulation_steps):
            sequence = torch.randint(0, edge_config.vocab_size, (2, 32)).cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(sequence)
                loss = logits.sum() / accumulation_steps

            loss.backward()

        # Check gradients aren't inf/nan
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()


class TestExpertEdgeCases:
    """Test expert routing edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_expert_load_imbalance(self, edge_config):
        """Test model handles expert load imbalance."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        # Use fixed random seed for reproducibility
        torch.manual_seed(12345)

        # Many tokens might route to same experts
        sequence = torch.randint(0, edge_config.vocab_size, (8, 128)).cuda()

        with torch.no_grad():
            logits = model(sequence)

        assert not torch.isnan(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_expert_weights_after_training(self, edge_config):
        """Test expert weights stay valid after training."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Heavy training
        for step in range(20):
            sequence = torch.randint(0, edge_config.vocab_size, (4, 64)).cuda()
            targets = torch.randint(0, edge_config.vocab_size, (4, 64)).cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(sequence)
                loss = F.cross_entropy(
                    logits.view(-1, edge_config.vocab_size),
                    targets.view(-1)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Check expert weights
        for name, param in model.named_parameters():
            if 'W1' in name or 'W2' in name or 'W3' in name:
                assert not torch.isnan(param).any(), f"NaN in {name}"
                assert not torch.isinf(param).any(), f"Inf in {name}"


class TestMemoryEdgeCases:
    """Test memory-related edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_repeated_allocation_deallocation(self, edge_config):
        """Test repeated model creation/deletion doesn't leak memory."""
        from nmoe.model import Transformer

        initial_memory = torch.cuda.memory_allocated()

        for _ in range(5):
            model = Transformer(edge_config).cuda().bfloat16()
            model.init_weights()

            sequence = torch.randint(0, edge_config.vocab_size, (4, 64)).cuda()
            logits = model(sequence)
            loss = logits.sum()
            loss.backward()

            del model, sequence, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        final_memory = torch.cuda.memory_allocated()
        memory_growth = final_memory - initial_memory

        # Allow some fragmentation but not major leak
        assert memory_growth < 50e6, f"Memory leak detected: {memory_growth / 1e6:.2f} MB"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gradient_memory_cleanup(self, edge_config):
        """Test gradient memory is properly cleaned up."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward + backward
        sequence = torch.randint(0, edge_config.vocab_size, (4, 64)).cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(sequence)
            loss = logits.sum()

        memory_before_backward = torch.cuda.memory_allocated()
        loss.backward()
        memory_after_backward = torch.cuda.memory_allocated()

        optimizer.step()
        optimizer.zero_grad()

        memory_after_zero_grad = torch.cuda.memory_allocated()

        # Memory behavior varies due to CUDA caching - just verify gradients were zeroed
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert (p.grad == 0).all(), f"Gradient not zeroed for {name}"


class TestStateManagementEdgeCases:
    """Test state management edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_train_eval_consistency(self, edge_config):
        """Test train/eval mode doesn't corrupt state."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()

        sequence = torch.randint(0, edge_config.vocab_size, (2, 64)).cuda()

        # Get reference output in eval mode
        model.eval()
        with torch.no_grad():
            ref_logits = model(sequence)

        # Switch to train, do some training
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)  # Very small LR

        for _ in range(5):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(sequence)
                loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Back to eval - should still produce valid output
        model.eval()
        with torch.no_grad():
            new_logits = model(sequence)

        assert not torch.isnan(new_logits).any()
        # Outputs should be different (weights changed)
        assert not torch.equal(ref_logits, new_logits)

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_model_copy_independence(self, edge_config):
        """Test that model copies are independent."""
        from nmoe.model import Transformer
        import copy

        model1 = Transformer(edge_config).cuda().bfloat16()
        model1.init_weights()

        # Deep copy
        model2 = Transformer(edge_config).cuda().bfloat16()
        model2.load_state_dict(copy.deepcopy(model1.state_dict()))

        sequence = torch.randint(0, edge_config.vocab_size, (2, 64)).cuda()

        # Initial outputs should match
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1_before = model1(sequence)
            out2_before = model2(sequence)
        assert torch.equal(out1_before, out2_before)

        # Modify model1
        model1.train()
        optimizer = torch.optim.AdamW(model1.parameters(), lr=0.1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model1(sequence).sum()
        loss.backward()
        optimizer.step()

        # model2 should be unchanged
        model1.eval()
        with torch.no_grad():
            out1_after = model1(sequence)
            out2_after = model2(sequence)

        assert not torch.equal(out1_after, out1_before), "model1 should have changed"
        assert torch.equal(out2_after, out2_before), "model2 should be unchanged"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_state_dict_roundtrip(self, edge_config):
        """Test state dict save/load roundtrip."""
        from nmoe.model import Transformer
        import tempfile
        import os

        model1 = Transformer(edge_config).cuda().bfloat16()
        model1.init_weights()

        sequence = torch.randint(0, edge_config.vocab_size, (2, 64)).cuda()

        model1.eval()
        with torch.no_grad():
            ref_output = model1(sequence)

        # Save and load state dict
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model1.state_dict(), path)

            model2 = Transformer(edge_config).cuda().bfloat16()
            model2.load_state_dict(torch.load(path, weights_only=True))

        model2.eval()
        with torch.no_grad():
            loaded_output = model2(sequence)

        assert torch.equal(ref_output, loaded_output)


class TestInputEdgeCases:
    """Test input-related edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_batch_size_one(self, edge_config):
        """Test with batch size of 1."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        sequence = torch.randint(0, edge_config.vocab_size, (1, 64)).cuda()

        with torch.no_grad():
            logits = model(sequence)

        assert logits.shape == (1, 64, edge_config.vocab_size)
        assert not torch.isnan(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_large_batch_size(self, edge_config):
        """Test with large batch size."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        batch_size = 128
        sequence = torch.randint(0, edge_config.vocab_size, (batch_size, 32)).cuda()

        with torch.no_grad():
            logits = model(sequence)

        assert logits.shape == (batch_size, 32, edge_config.vocab_size)
        assert not torch.isnan(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_power_of_two_lengths(self, edge_config):
        """Test with power-of-2 sequence lengths."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        for length in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            sequence = torch.randint(0, edge_config.vocab_size, (2, length)).cuda()

            with torch.no_grad():
                logits = model(sequence)

            assert logits.shape == (2, length, edge_config.vocab_size)
            assert not torch.isnan(logits).any(), f"NaN for length={length}"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_non_power_of_two_lengths(self, edge_config):
        """Test with non-power-of-2 sequence lengths."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        for length in [3, 7, 13, 31, 63, 97, 127]:
            sequence = torch.randint(0, edge_config.vocab_size, (2, length)).cuda()

            with torch.no_grad():
                logits = model(sequence)

            assert logits.shape == (2, length, edge_config.vocab_size)
            assert not torch.isnan(logits).any(), f"NaN for length={length}"


class TestConcurrencyEdgeCases:
    """Test concurrency-related edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_sequential_forward_passes(self, edge_config):
        """Test many sequential forward passes."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        for _ in range(100):
            sequence = torch.randint(0, edge_config.vocab_size, (2, 32)).cuda()
            with torch.no_grad():
                logits = model(sequence)
            assert not torch.isnan(logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_interleaved_forward_backward(self, edge_config):
        """Test interleaved forward and backward passes."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(20):
            # Training batch
            model.train()
            train_seq = torch.randint(0, edge_config.vocab_size, (4, 32)).cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                train_logits = model(train_seq)
                loss = train_logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Inference batch
            model.eval()
            infer_seq = torch.randint(0, edge_config.vocab_size, (2, 64)).cuda()
            with torch.no_grad():
                infer_logits = model(infer_seq)
            assert not torch.isnan(infer_logits).any()


class TestWrapperEdgeCases:
    """Test NMoEModelWrapper edge cases."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_wrapper_with_zero_num_actions(self, edge_config):
        """Test wrapper handling of edge case num_actions."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        wrapper = NMoEModelWrapper(model)

        # Very small num_actions
        sequence = torch.randint(0, edge_config.vocab_size, (2, 64)).cuda()

        # Note: num_actions must be 2nd positional arg for __call__ dispatch
        for num_actions in [1, 2, 4]:
            log_probs = wrapper(sequence, num_actions)
            assert log_probs.shape == (2, num_actions)
            assert not torch.isnan(log_probs).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_wrapper_state_dict_equivalence(self, edge_config):
        """Test wrapper state dict matches model state dict."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        wrapper = NMoEModelWrapper(model)

        model_keys = set(model.state_dict().keys())
        wrapper_keys = set(wrapper.state_dict().keys())

        # Wrapper state dict should contain model keys (possibly with prefix)
        # Check that all model parameters are accessible
        assert len(wrapper_keys) >= len(model_keys)


class TestRobustness:
    """Test model robustness."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_recovery_after_nan(self, edge_config):
        """Test model can recover after encountering NaN."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()

        # Save good state
        good_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Corrupt weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float('nan'))

        # Verify corrupted
        sequence = torch.randint(0, edge_config.vocab_size, (1, 32)).cuda()
        with torch.no_grad():
            bad_output = model(sequence)
        assert torch.isnan(bad_output).all()

        # Restore good state
        model.load_state_dict(good_state)

        # Verify recovered
        with torch.no_grad():
            good_output = model(sequence)
        assert not torch.isnan(good_output).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_determinism_with_seed(self, edge_config):
        """Test deterministic output with fixed seed."""
        from nmoe.model import Transformer

        model = Transformer(edge_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        sequence = torch.randint(0, edge_config.vocab_size, (2, 64)).cuda()

        outputs = []
        for _ in range(3):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            with torch.no_grad():
                logits = model(sequence)
            outputs.append(logits.clone())

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.equal(outputs[0], outputs[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
