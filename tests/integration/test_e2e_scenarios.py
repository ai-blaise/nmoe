"""End-to-end integration scenarios for nmoe + SGLang + SkyRL.

This module tests complete workflows that span multiple components:
- Full training → serving pipeline
- RL training loop with inference backend
- Model export and loading
- Weight synchronization across components

Run with:
    pytest tests/integration/test_e2e_scenarios.py -v -s
"""

import gc
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for E2E tests"
)


@pytest.fixture(scope="module")
def e2e_config():
    """Configuration for E2E tests."""
    from nmoe.config import Config
    return Config(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=1024,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_inter_dim=512,
        inter_dim=512,
        max_position_embeddings=256,
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory(prefix="nmoe_e2e_") as tmpdir:
        yield Path(tmpdir)


class TestTrainServeLoop:
    """Test training → serving loop scenarios."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_train_then_infer(self, e2e_config):
        """Test training a model then running inference."""
        from nmoe.model import Transformer

        # Create and train model
        model = Transformer(e2e_config).cuda().bfloat16()
        model.init_weights()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Training loop
        for step in range(5):
            sequences = torch.randint(0, e2e_config.vocab_size, (4, 64)).cuda()
            targets = torch.randint(0, e2e_config.vocab_size, (4, 64)).cuda()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(sequences)
                loss = F.cross_entropy(
                    logits.view(-1, e2e_config.vocab_size),
                    targets.view(-1)
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Switch to inference mode
        model.eval()

        # Inference
        with torch.no_grad():
            prompt = torch.randint(0, e2e_config.vocab_size, (1, 32)).cuda()
            logits = model(prompt)

            # Generate a few tokens
            for _ in range(16):
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token], dim=-1)
                logits = model(prompt)

        assert prompt.shape[1] == 32 + 16

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_checkpoint_roundtrip(self, e2e_config, temp_dir):
        """Test saving and loading checkpoints."""
        from nmoe.model import Transformer

        # Create model
        model1 = Transformer(e2e_config).cuda().bfloat16()
        model1.init_weights()

        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.pt"
        torch.save({
            "model_state_dict": model1.state_dict(),
            "config": e2e_config,
        }, checkpoint_path)

        # Create new model and load checkpoint
        model2 = Transformer(e2e_config).cuda().bfloat16()
        model2.init_weights()
        model2.init_weights()
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Verify models produce same output
        torch.manual_seed(42)
        sequences = torch.randint(0, e2e_config.vocab_size, (2, 32)).cuda()

        with torch.no_grad():
            output1 = model1(sequences)
            output2 = model2(sequences)

        assert torch.equal(output1, output2)


class TestRLTrainingScenarios:
    """Test RL training scenarios."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rl_training_loop(self, e2e_config):
        """Test a complete RL training loop."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        # Create actor and reference
        actor = Transformer(e2e_config).cuda().bfloat16()
        actor.init_weights()
        actor_wrapper = NMoEModelWrapper(actor)

        ref_model = Transformer(e2e_config).cuda().bfloat16()
        ref_model.init_weights()
        ref_model.load_state_dict(actor.state_dict())
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_wrapper = NMoEModelWrapper(ref_model)

        optimizer = torch.optim.AdamW(actor_wrapper.parameters(), lr=1e-4)

        # Simulated RL loop
        for episode in range(3):
            # Generate prompts
            prompts = torch.randint(0, e2e_config.vocab_size, (4, 32)).cuda()

            # Generate completions (simplified - in practice would sample)
            with torch.no_grad():
                actor.eval()
                sequences = prompts.clone()
                for _ in range(32):
                    logits = actor(sequences)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    sequences = torch.cat([sequences, next_token], dim=-1)
                actor.train()

            # Compute log probs
            # Note: num_actions must be 2nd positional arg for __call__ dispatch
            num_actions = 32
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                actor_log_probs = actor_wrapper(sequences, num_actions)
                with torch.no_grad():
                    ref_log_probs = ref_wrapper(sequences, num_actions)

            # Simulated rewards
            rewards = torch.randn(4, device="cuda")
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Policy gradient loss
            ratio = (actor_log_probs - ref_log_probs).exp().mean(dim=1)
            pg_loss = -(advantages * ratio).mean()

            # KL penalty
            kl = (ref_log_probs.exp() * (ref_log_probs - actor_log_probs)).mean()
            total_loss = pg_loss + 0.01 * kl

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Verify actor weights changed
        actor_state = actor_wrapper.state_dict()
        ref_state = ref_wrapper.state_dict()

        weights_changed = False
        for key in actor_state:
            if key in ref_state and not torch.equal(actor_state[key], ref_state[key]):
                weights_changed = True
                break

        assert weights_changed, "Actor weights should have changed"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rollout_generation(self, e2e_config):
        """Test rollout generation for RL."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(e2e_config).cuda().bfloat16()
        model.init_weights()
        wrapper = NMoEModelWrapper(model)

        # Generate rollouts
        batch_size = 8
        prompt_len = 32
        max_new_tokens = 64

        prompts = torch.randint(0, e2e_config.vocab_size, (batch_size, prompt_len)).cuda()

        # Autoregressive generation
        model.eval()
        with torch.no_grad():
            sequences = prompts.clone()

            for step in range(max_new_tokens):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(sequences)

                # Sample from distribution
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                sequences = torch.cat([sequences, next_token], dim=-1)

        assert sequences.shape == (batch_size, prompt_len + max_new_tokens)

        # Compute log probs for entire rollout
        # Note: num_actions must be 2nd positional arg for __call__ dispatch
        model.train()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            log_probs = wrapper(sequences, max_new_tokens)

        assert log_probs.shape == (batch_size, max_new_tokens)


class TestWeightSyncScenarios:
    """Test weight synchronization scenarios."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_actor_to_inference_sync(self, e2e_config):
        """Test syncing actor weights to inference engine."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        # Create actor
        actor = Transformer(e2e_config).cuda().bfloat16()
        actor.init_weights()
        actor_wrapper = NMoEModelWrapper(actor)

        # Create "inference engine" (separate model instance)
        inference = Transformer(e2e_config).cuda().bfloat16()
        inference.init_weights()

        # Train actor for a few steps
        optimizer = torch.optim.AdamW(actor_wrapper.parameters(), lr=1e-3)

        for _ in range(5):
            sequences = torch.randint(0, e2e_config.vocab_size, (4, 64)).cuda()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = actor_wrapper.model(sequences)
                loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Sync weights to inference
        inference.load_state_dict(actor.state_dict())

        # Verify outputs match
        test_input = torch.randint(0, e2e_config.vocab_size, (2, 32)).cuda()

        actor.eval()
        inference.eval()

        with torch.no_grad():
            actor_output = actor(test_input)
            inference_output = inference(test_input)

        assert torch.equal(actor_output, inference_output)

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_incremental_weight_updates(self, e2e_config):
        """Test incremental weight updates and delta computation."""
        from nmoe.model import Transformer

        model = Transformer(e2e_config).cuda().bfloat16()
        model.init_weights()

        # Get initial state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Simulate incremental updates with proper loss and gradient clipping
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(5):
            sequences = torch.randint(0, e2e_config.vocab_size, (2, 32)).cuda()
            targets = torch.randint(0, e2e_config.vocab_size, (2, 32)).cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(sequences)
                loss = F.cross_entropy(
                    logits.view(-1, e2e_config.vocab_size),
                    targets.view(-1)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Get final state
        final_state = model.state_dict()

        # Verify no NaN in final state
        for key, val in final_state.items():
            assert not torch.isnan(val).any(), f"NaN in {key} after training"

        # Compute deltas
        changed_keys = []
        for key in initial_state:
            if key in final_state:
                if not torch.equal(initial_state[key], final_state[key]):
                    changed_keys.append(key)

        # Verify some weights changed
        assert len(changed_keys) > 0, "No weights changed during training"

        # Verify loading final_state into new model works correctly
        model2 = Transformer(e2e_config).cuda().bfloat16()
        model2.load_state_dict(final_state)

        # Verify models produce identical output
        test_input = torch.randint(0, e2e_config.vocab_size, (2, 32)).cuda()
        with torch.no_grad():
            out1 = model(test_input)
            out2 = model2(test_input)

        # Both should be non-NaN and equal
        assert not torch.isnan(out1).any(), "NaN in model output"
        assert torch.equal(out1, out2), "Loaded model should produce same output"


class TestExportImportScenarios:
    """Test export and import scenarios."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_export_to_hf_format(self, e2e_config, temp_dir):
        """Test exporting to HuggingFace format."""
        from nmoe.model import Transformer
        from nmoe.unified.config import NMoEModelConfig

        model = Transformer(e2e_config).cuda().bfloat16()
        model.init_weights()

        # Create checkpoint
        checkpoint_dir = temp_dir / "checkpoint"
        iteration_dir = checkpoint_dir / "iteration_00001"
        iteration_dir.mkdir(parents=True)

        # Save model state
        torch.save(model.state_dict(), iteration_dir / "rd.pt")

        # Save config
        unified_config = NMoEModelConfig.from_nmoe_config(e2e_config)
        import json
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(unified_config.to_dict(), f)

        # Export
        export_dir = temp_dir / "hf_export"
        export_dir.mkdir()

        try:
            from nmoe.tools.export_to_hf import export_nmoe_to_hf
            export_nmoe_to_hf(
                checkpoint_path=str(checkpoint_dir),
                output_path=str(export_dir),
                config=unified_config,
            )

            # Verify export files exist
            assert (export_dir / "config.json").exists()
            weight_files = list(export_dir.glob("*.safetensors")) + list(export_dir.glob("*.bin"))
            assert len(weight_files) > 0
        except ImportError:
            pytest.skip("safetensors not available")

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_config_roundtrip(self, e2e_config):
        """Test config roundtrip through various formats."""
        from nmoe.unified.config import NMoEModelConfig

        # nmoe -> unified
        unified = NMoEModelConfig.from_nmoe_config(e2e_config)

        # unified -> HF dict
        hf_dict = unified.to_hf_config()

        # HF dict -> unified (use from_hf_config for proper field mapping)
        unified2 = NMoEModelConfig.from_hf_config(hf_dict)

        # Verify key fields preserved
        assert unified.hidden_size == unified2.hidden_size
        assert unified.num_hidden_layers == unified2.num_hidden_layers
        assert unified.num_attention_heads == unified2.num_attention_heads
        assert unified.num_experts == unified2.num_experts


class TestMixedWorkloads:
    """Test mixed training and inference workloads."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_interleaved_train_infer(self, e2e_config):
        """Test interleaved training and inference."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(e2e_config).cuda().bfloat16()
        model.init_weights()
        wrapper = NMoEModelWrapper(model)
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

        for iteration in range(10):
            if iteration % 2 == 0:
                # Training step
                model.train()
                sequences = torch.randint(0, e2e_config.vocab_size, (4, 64)).cuda()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = wrapper.model(sequences)
                    loss = logits.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                # Inference step
                model.eval()
                with torch.no_grad():
                    prompts = torch.randint(0, e2e_config.vocab_size, (2, 32)).cuda()
                    logits = wrapper.model(prompts)
                    # Sample tokens
                    _ = torch.argmax(logits[:, -1, :], dim=-1)

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_batch_size_variation(self, e2e_config):
        """Test handling of varying batch sizes."""
        from nmoe.model import Transformer

        model = Transformer(e2e_config).cuda().bfloat16()
        model.init_weights()

        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_lengths = [16, 32, 64, 128]

        model.eval()
        with torch.no_grad():
            for bs in batch_sizes:
                for seq_len in seq_lengths:
                    sequences = torch.randint(0, e2e_config.vocab_size, (bs, seq_len)).cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits = model(sequences)

                    assert logits.shape == (bs, seq_len, e2e_config.vocab_size)
                    assert not torch.isnan(logits).any()


class TestMemoryManagement:
    """Test memory management scenarios."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_memory_cleanup(self, e2e_config):
        """Test proper memory cleanup between operations."""
        from nmoe.model import Transformer

        initial_memory = torch.cuda.memory_allocated()

        for _ in range(5):
            model = Transformer(e2e_config).cuda().bfloat16()
            model.init_weights()
            sequences = torch.randint(0, e2e_config.vocab_size, (8, 64)).cuda()
            logits = model(sequences)
            loss = logits.sum()
            loss.backward()

            del model, sequences, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        final_memory = torch.cuda.memory_allocated()

        # Memory should return close to initial (some fragmentation OK)
        memory_growth = final_memory - initial_memory
        print(f"Memory growth: {memory_growth / 1e6:.2f} MB")

        # Allow some growth but not excessive
        assert memory_growth < 100e6, f"Memory leak detected: {memory_growth / 1e6:.2f} MB"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_large_batch_memory(self, e2e_config):
        """Test memory usage with large batches."""
        from nmoe.model import Transformer

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = Transformer(e2e_config).cuda().bfloat16()
        model.init_weights()

        # Start with small batch
        small_batch = torch.randint(0, e2e_config.vocab_size, (4, 64)).cuda()
        _ = model(small_batch)
        small_batch_memory = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()

        # Now large batch
        large_batch = torch.randint(0, e2e_config.vocab_size, (32, 128)).cuda()
        _ = model(large_batch)
        large_batch_memory = torch.cuda.max_memory_allocated()

        print(f"Small batch (4x64) peak memory: {small_batch_memory / 1e9:.3f} GB")
        print(f"Large batch (32x128) peak memory: {large_batch_memory / 1e9:.3f} GB")

        # Large batch should use more memory but not excessively more
        ratio = large_batch_memory / small_batch_memory
        assert ratio < 50, f"Memory scaling ratio too high: {ratio:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
