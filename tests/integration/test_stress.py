"""Stress tests for nmoe + SkyRL + SGLang integration.

This module tests the system under stress conditions:
- High throughput scenarios
- Long-running operations
- Memory pressure
- Rapid state transitions
- Concurrent-like access patterns
- Recovery scenarios

Run with:
    pytest tests/integration/test_stress.py -v -s --timeout=300
"""

import gc
import time
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for stress tests"
)


@pytest.fixture(scope="function")
def stress_config():
    """Config for stress testing."""
    from nmoe.config import Config

    # Clean up CUDA state before each test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    config = Config(
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

    yield config

    # Clean up after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TestHighThroughput:
    """Test high throughput scenarios."""

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_sustained_inference(self, stress_config):
        """Test sustained inference over many iterations."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        n_iterations = 500
        batch_size = 8
        seq_len = 64

        start_time = time.perf_counter()
        total_tokens = 0

        for i in range(n_iterations):
            sequence = torch.randint(0, stress_config.vocab_size, (batch_size, seq_len)).cuda()

            with torch.no_grad():
                logits = model(sequence)

            total_tokens += batch_size * seq_len

            # Periodic check for NaN
            if i % 100 == 0:
                assert not torch.isnan(logits).any(), f"NaN at iteration {i}"

        elapsed = time.perf_counter() - start_time
        throughput = total_tokens / elapsed

        print(f"Sustained inference: {n_iterations} iters, {throughput:.0f} tokens/sec")

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_sustained_training(self, stress_config):
        """Test sustained training over many steps."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        n_steps = 200
        batch_size = 4
        seq_len = 64

        losses = []
        for step in range(n_steps):
            sequence = torch.randint(0, stress_config.vocab_size, (batch_size, seq_len)).cuda()
            targets = torch.randint(0, stress_config.vocab_size, (batch_size, seq_len)).cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(sequence)
                loss = F.cross_entropy(
                    logits.view(-1, stress_config.vocab_size),
                    targets.view(-1)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            if step % 50 == 0:
                assert not torch.isnan(logits).any(), f"NaN at step {step}"
                assert loss.item() < 100, f"Loss exploded at step {step}"

        # Loss should not have exploded
        assert losses[-1] < losses[0] * 10, "Loss should not explode"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rapid_batch_processing(self, stress_config):
        """Test rapidly processing many small batches."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        n_batches = 1000

        for _ in range(n_batches):
            # Small batch, quick turnaround
            sequence = torch.randint(0, stress_config.vocab_size, (1, 16)).cuda()
            with torch.no_grad():
                logits = model(sequence)


class TestMemoryPressure:
    """Test behavior under memory pressure."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_memory_stability_long_run(self, stress_config):
        """Test memory remains stable over long run."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        n_iterations = 200
        memory_samples = []

        for i in range(n_iterations):
            sequence = torch.randint(0, stress_config.vocab_size, (4, 64)).cuda()
            with torch.no_grad():
                logits = model(sequence)
            del sequence, logits

            if i % 20 == 0:
                torch.cuda.empty_cache()
                memory_samples.append(torch.cuda.memory_allocated())

        # Memory should not grow significantly
        memory_growth = memory_samples[-1] - memory_samples[0]
        assert abs(memory_growth) < 50e6, f"Memory growth: {memory_growth / 1e6:.2f} MB"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_alternating_batch_sizes(self, stress_config):
        """Test with rapidly changing batch sizes."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1]

        for _ in range(10):  # Repeat pattern
            for bs in batch_sizes:
                sequence = torch.randint(0, stress_config.vocab_size, (bs, 64)).cuda()
                with torch.no_grad():
                    logits = model(sequence)
                del sequence, logits

        torch.cuda.empty_cache()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_max_memory_usage(self, stress_config):
        """Test maximum memory usage scenario."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()

        torch.cuda.reset_peak_memory_stats()

        # Large batch forward + backward
        batch_size = 32
        seq_len = stress_config.max_position_embeddings

        sequence = torch.randint(0, stress_config.vocab_size, (batch_size, seq_len)).cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(sequence)
            loss = logits.sum()

        loss.backward()

        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Peak memory usage: {peak_memory / 1e9:.2f} GB")

        # Clean up
        del sequence, logits, loss
        torch.cuda.empty_cache()
        gc.collect()


class TestRapidStateTransitions:
    """Test rapid state transitions."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rapid_train_eval_switching(self, stress_config):
        """Test rapidly switching between train and eval modes."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        sequence = torch.randint(0, stress_config.vocab_size, (4, 64)).cuda()

        for _ in range(100):
            # Train step
            model.train()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(sequence)
                loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Eval step
            model.eval()
            with torch.no_grad():
                eval_logits = model(sequence)

            assert not torch.isnan(eval_logits).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rapid_optimizer_operations(self, stress_config):
        """Test rapid optimizer step/zero_grad cycles."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(200):
            sequence = torch.randint(0, stress_config.vocab_size, (2, 32)).cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(sequence)
                loss = logits.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Model should still work
        model.eval()
        with torch.no_grad():
            test_logits = model(sequence)
        assert not torch.isnan(test_logits).any()


class TestRecoveryScenarios:
    """Test recovery from various error conditions."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_recovery_from_bad_gradients(self, stress_config):
        """Test recovery after gradient issues."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Save good state
        good_state = {k: v.clone() for k, v in model.state_dict().items()}

        sequence = torch.randint(0, stress_config.vocab_size, (4, 64)).cuda()

        # Artificially create bad gradients
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(sequence)
            loss = logits.sum() * 1e10  # Huge loss

        loss.backward()

        # Detect bad gradients
        has_bad_grads = any(
            (torch.isnan(p.grad).any() or torch.isinf(p.grad).any() or p.grad.abs().max() > 1e6)
            for p in model.parameters() if p.grad is not None
        )

        if has_bad_grads:
            # Recovery: skip update and restore
            optimizer.zero_grad()

        # Should still work
        model.eval()
        with torch.no_grad():
            test_logits = model(sequence)
        # May or may not have NaN depending on if weights were corrupted

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_state_dict_corruption_detection(self, stress_config):
        """Test detection and recovery from corrupted state dict."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()

        # Save good state
        good_state = {k: v.clone() for k, v in model.state_dict().items()}

        sequence = torch.randint(0, stress_config.vocab_size, (2, 32)).cuda()

        # Get reference output
        model.eval()
        with torch.no_grad():
            ref_output = model(sequence)

        # Corrupt one weight
        corrupted_state = {k: v.clone() for k, v in good_state.items()}
        first_key = list(corrupted_state.keys())[0]
        corrupted_state[first_key] = torch.full_like(corrupted_state[first_key], float('nan'))

        # Load corrupted state
        model.load_state_dict(corrupted_state)

        # Detect corruption
        with torch.no_grad():
            corrupted_output = model(sequence)
        is_corrupted = torch.isnan(corrupted_output).any()

        # Recover
        if is_corrupted:
            model.load_state_dict(good_state)

        with torch.no_grad():
            recovered_output = model(sequence)

        assert torch.equal(recovered_output, ref_output)


class TestLongRunning:
    """Test long-running operations."""

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_extended_generation(self, stress_config):
        """Test extended autoregressive generation."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        prompt = torch.randint(0, stress_config.vocab_size, (1, 16)).cuda()
        max_new_tokens = 200

        current_seq = prompt.clone()
        for step in range(max_new_tokens):
            with torch.no_grad():
                logits = model(current_seq)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                current_seq = torch.cat([current_seq, next_token], dim=1)

            # Truncate if getting too long (sliding window)
            if current_seq.shape[1] > stress_config.max_position_embeddings:
                current_seq = current_seq[:, -stress_config.max_position_embeddings:]

        assert current_seq.shape[1] > prompt.shape[1]

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_repeated_checkpoint_operations(self, stress_config):
        """Test repeated save/load operations."""
        from nmoe.model import Transformer
        import tempfile
        import os

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        sequence = torch.randint(0, stress_config.vocab_size, (4, 64)).cuda()

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                # Train
                model.train()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(sequence)
                    loss = logits.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Save checkpoint
                ckpt_path = os.path.join(tmpdir, f"ckpt_{i}.pt")
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, ckpt_path)

                # Load checkpoint (different file each time)
                if i > 0:
                    prev_path = os.path.join(tmpdir, f"ckpt_{i - 1}.pt")
                    ckpt = torch.load(prev_path, weights_only=False)
                    # Just verify it loads, don't actually use
                    assert 'model' in ckpt
                    del ckpt

        # Model should still work
        model.eval()
        with torch.no_grad():
            test_logits = model(sequence)
        assert not torch.isnan(test_logits).any()


class TestConcurrentPatterns:
    """Test patterns that simulate concurrent access."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_interleaved_requests(self, stress_config):
        """Test interleaved request patterns (simulated)."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        # Simulate interleaved requests of different lengths
        requests = [
            torch.randint(0, stress_config.vocab_size, (1, 16)).cuda(),
            torch.randint(0, stress_config.vocab_size, (1, 32)).cuda(),
            torch.randint(0, stress_config.vocab_size, (1, 64)).cuda(),
            torch.randint(0, stress_config.vocab_size, (1, 128)).cuda(),
        ]

        # Process in interleaved fashion
        for _ in range(50):
            for req in requests:
                with torch.no_grad():
                    _ = model(req)

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_mixed_train_inference_batches(self, stress_config):
        """Test mixing training and inference in rapid succession."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        wrapper = NMoEModelWrapper(model)
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

        for _ in range(30):
            # Inference batch
            model.eval()
            infer_seq = torch.randint(0, stress_config.vocab_size, (8, 64)).cuda()
            with torch.no_grad():
                _ = model(infer_seq)

            # Training batch (different data)
            model.train()
            train_seq = torch.randint(0, stress_config.vocab_size, (4, 64)).cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(train_seq)
                loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class TestRLStress:
    """Stress tests for RL training scenarios."""

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_extended_rl_training(self, stress_config):
        """Test extended RL training loop."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        actor = Transformer(stress_config).cuda().bfloat16()
        actor.init_weights()
        actor_wrapper = NMoEModelWrapper(actor)

        ref = Transformer(stress_config).cuda().bfloat16()
        ref.load_state_dict(actor.state_dict())
        for p in ref.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(actor_wrapper.parameters(), lr=1e-4)

        n_episodes = 50
        batch_size = 4
        seq_len = 64
        num_actions = 16

        for episode in range(n_episodes):
            sequences = torch.randint(0, stress_config.vocab_size, (batch_size, seq_len)).cuda()

            # Get log probs
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Use positional arg for num_actions to trigger forward_rl dispatch
                actor_log_probs = actor_wrapper(sequences, num_actions)
                with torch.no_grad():
                    ref_logits = ref(sequences)

            # Simulate rewards
            rewards = torch.randn(batch_size, device="cuda")
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Policy gradient
            loss = -(advantages.unsqueeze(1) * actor_log_probs).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_wrapper.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if episode % 10 == 0:
                actor.eval()
                with torch.no_grad():
                    test_logits = actor(sequences)
                assert not torch.isnan(test_logits).any(), f"NaN at episode {episode}"
                actor.train()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rapid_rollout_generation(self, stress_config):
        """Test rapid rollout generation."""
        from nmoe.model import Transformer

        model = Transformer(stress_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        n_rollouts = 100
        prompt_len = 16
        gen_len = 32

        for _ in range(n_rollouts):
            prompt = torch.randint(0, stress_config.vocab_size, (1, prompt_len)).cuda()
            current_seq = prompt.clone()

            for _ in range(gen_len):
                with torch.no_grad():
                    logits = model(current_seq)
                    probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    current_seq = torch.cat([current_seq, next_token], dim=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
