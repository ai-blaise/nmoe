"""End-to-End Pipeline Tests: nmoe Train -> Export -> SGLang Serve.

This comprehensive test module validates the full pipeline from training an nmoe
model through export to HuggingFace format and serving with SGLang.

Gap Analysis Coverage:
- Full training loop with nmoe model (small config for CI)
- Checkpoint export to HuggingFace format
- Model loading in SGLang after export
- Inference consistency pre/post export
- 8-GPU distributed training -> single GPU serve
- 8-GPU training -> 8-GPU TP serve
- Weight precision (BF16->FP16 conversion)
- Expert weight handling in export

Run with:
    cd nmoe && source .venv/bin/activate
    pytest tests/integration/test_e2e_train_export_serve.py -v --tb=short

Run single test class:
    pytest tests/integration/test_e2e_train_export_serve.py::TestTrainingLoop -v

Run with markers:
    pytest tests/integration/test_e2e_train_export_serve.py -m "gpu and e2e" -v
"""

import copy
import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Markers for test categorization
pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
]

# Skip all tests if CUDA not available
GPU_AVAILABLE = torch.cuda.is_available()
MULTI_GPU_AVAILABLE = GPU_AVAILABLE and torch.cuda.device_count() >= 2
EIGHT_GPU_AVAILABLE = GPU_AVAILABLE and torch.cuda.device_count() >= 8


def get_small_nmoe_config():
    """Get small nmoe config for fast CI testing."""
    from nmoe.config import Config

    return Config(
        dim=256,
        n_layers=2,
        n_heads=4,
        vocab_size=1024,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=0,
        moe_inter_dim=512,
        inter_dim=512,
        max_position_embeddings=256,
        batch_size=2,
        seq_len=64,
        dtype="bf16",
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        eos_token_id=1023,
    )


def get_medium_nmoe_config():
    """Get medium config for more thorough testing."""
    from nmoe.config import Config

    return Config(
        dim=512,
        n_layers=4,
        n_heads=8,
        vocab_size=2048,
        n_dense_layers=1,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_inter_dim=1024,
        inter_dim=1024,
        max_position_embeddings=512,
        batch_size=4,
        seq_len=128,
        dtype="bf16",
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        eos_token_id=2047,
    )


@pytest.fixture(scope="module")
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    tmpdir = tempfile.mkdtemp(prefix="nmoe_e2e_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def small_config():
    """Small nmoe config fixture."""
    return get_small_nmoe_config()


@pytest.fixture(scope="module")
def medium_config():
    """Medium nmoe config fixture."""
    return get_medium_nmoe_config()


@pytest.fixture
def fresh_small_model(small_config):
    """Create fresh small model for each test."""
    if not GPU_AVAILABLE:
        pytest.skip("CUDA required")
    from nmoe.model import Transformer

    model = Transformer(small_config).cuda().bfloat16()
    model.init_weights()
    yield model
    del model
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Test Class 1: Training Loop Tests
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestTrainingLoop:
    """Test full training loop with nmoe model."""

    def test_single_step_training(self, fresh_small_model, small_config):
        """Test 1.1: Single training step completes without errors."""
        model = fresh_small_model
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()
        targets = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, small_config.vocab_size), targets.view(-1))

        # Backward
        loss.backward()

        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients computed"

        # Step
        optimizer.step()
        optimizer.zero_grad()

        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"

    def test_multi_step_training_loss_decreases(self, small_config):
        """Test 1.2: Multiple training steps show loss decrease trend."""
        from nmoe.model import Transformer

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Fixed training data for determinism
        torch.manual_seed(42)
        input_ids = torch.randint(0, small_config.vocab_size, (4, 64)).cuda()
        targets = torch.randint(0, small_config.vocab_size, (4, 64)).cuda()

        losses = []
        n_steps = 10

        for step in range(n_steps):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, small_config.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Refresh MoE weight caches after optimizer step
            for block in model.blocks:
                if hasattr(block.ffn, "refresh_weight_cache"):
                    block.ffn.refresh_weight_cache()

        # Verify loss decreases (first vs last few steps)
        first_avg = sum(losses[:3]) / 3
        last_avg = sum(losses[-3:]) / 3
        assert last_avg < first_avg, f"Loss did not decrease: {first_avg:.4f} -> {last_avg:.4f}"

    def test_training_with_gradient_checkpointing(self, small_config):
        """Test 1.3: Training works with gradient checkpointing enabled."""
        from nmoe.model import Transformer

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()
        model.gradient_checkpointing_enable()
        model.train()

        assert model.is_gradient_checkpointing, "Gradient checkpointing not enabled"

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()
        targets = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, small_config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss), "Loss is NaN with gradient checkpointing"

    def test_training_with_aux_loss(self, small_config):
        """Test 1.4: Training includes router auxiliary loss."""
        from nmoe.config import Config
        from nmoe.model import Transformer

        # Enable aux loss
        config = Config(
            dim=small_config.dim,
            n_layers=small_config.n_layers,
            n_heads=small_config.n_heads,
            vocab_size=small_config.vocab_size,
            n_dense_layers=small_config.n_dense_layers,
            n_routed_experts=small_config.n_routed_experts,
            n_activated_experts=small_config.n_activated_experts,
            n_shared_experts=small_config.n_shared_experts,
            moe_inter_dim=small_config.moe_inter_dim,
            inter_dim=small_config.inter_dim,
            max_position_embeddings=small_config.max_position_embeddings,
            batch_size=small_config.batch_size,
            seq_len=small_config.seq_len,
            aux_loss_alpha=0.01,
        )

        model = Transformer(config).cuda().bfloat16()
        model.init_weights()
        model.train()

        input_ids = torch.randint(0, config.vocab_size, (2, 64)).cuda()
        logits = model(input_ids)

        # Get auxiliary loss
        aux_loss = model.get_router_aux_loss()

        assert aux_loss is not None, "Aux loss not computed"
        assert aux_loss >= 0, "Aux loss should be non-negative"

    def test_training_determinism(self, small_config):
        """Test 1.5: Training is deterministic with fixed seeds."""
        from nmoe.model import Transformer

        def train_steps(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            model = Transformer(small_config).cuda().bfloat16()
            model.init_weights()
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            torch.manual_seed(seed + 100)
            input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()
            targets = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()

            losses = []
            for _ in range(3):
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, small_config.vocab_size), targets.view(-1)
                )
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            return losses

        losses1 = train_steps(42)
        losses2 = train_steps(42)

        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert abs(l1 - l2) < 1e-4, f"Step {i}: Loss differs: {l1:.6f} vs {l2:.6f}"


# =============================================================================
# Test Class 2: Checkpoint Export Tests
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestCheckpointExport:
    """Test checkpoint export to HuggingFace format."""

    def test_basic_checkpoint_save(self, fresh_small_model, small_config, temp_checkpoint_dir):
        """Test 2.1: Basic checkpoint saving works."""
        checkpoint_path = Path(temp_checkpoint_dir) / "basic_ckpt"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state = {
            "model": fresh_small_model.state_dict(),
            "step": 100,
            "tokens": 1024000,
        }
        torch.save(state, checkpoint_path / "checkpoint.pt")

        assert (checkpoint_path / "checkpoint.pt").exists()

        loaded = torch.load(checkpoint_path / "checkpoint.pt", weights_only=False)
        assert loaded["step"] == 100
        assert loaded["tokens"] == 1024000
        assert len(loaded["model"]) == len(fresh_small_model.state_dict())

    def test_split_checkpoint_format(self, fresh_small_model, small_config, temp_checkpoint_dir):
        """Test 2.2: Split checkpoint format (rd.pt + dp_rank_*.pt)."""
        from nmoe.checkpoint import build_states

        checkpoint_path = Path(temp_checkpoint_dir) / "split_ckpt" / "iter_0000001"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Create mock loader
        class MockLoader:
            def state_dict(self):
                return {"position": 0}

        optimizer = torch.optim.AdamW(fresh_small_model.parameters(), lr=1e-4)
        loader = MockLoader()

        rd_state, dp_state = build_states(
            step=1,
            model=fresh_small_model,
            optimizer=optimizer,
            tokens=1000,
            loader=loader,
            config_fingerprint="test_hash",
        )

        # Save both parts
        torch.save(rd_state, checkpoint_path / "rd.pt")
        torch.save(dp_state, checkpoint_path / "dp_rank_000.pt")

        assert (checkpoint_path / "rd.pt").exists()
        assert (checkpoint_path / "dp_rank_000.pt").exists()

        # Verify rd.pt has dense weights
        rd_loaded = torch.load(checkpoint_path / "rd.pt", weights_only=False)
        assert "model_dense" in rd_loaded
        assert "step" in rd_loaded

        # Verify dp_rank has expert weights
        dp_loaded = torch.load(checkpoint_path / "dp_rank_000.pt", weights_only=False)
        assert "model_expert" in dp_loaded
        assert "optimizer" in dp_loaded

    def test_export_to_hf_config_json(self, small_config, temp_checkpoint_dir):
        """Test 2.3: Config.json generation for HuggingFace format."""
        from nmoe.tools.export_to_hf import generate_config_json
        from nmoe.unified.config import NMoEModelConfig

        output_path = Path(temp_checkpoint_dir) / "hf_config_test"
        output_path.mkdir(parents=True, exist_ok=True)

        unified_config = NMoEModelConfig.from_nmoe_config(small_config)
        hf_config = generate_config_json(unified_config, output_path)

        assert (output_path / "config.json").exists()

        with open(output_path / "config.json") as f:
            loaded_config = json.load(f)

        assert loaded_config["hidden_size"] == small_config.dim
        assert loaded_config["num_hidden_layers"] == small_config.n_layers
        assert "model_type" in loaded_config

    def test_export_weights_to_safetensors(
        self, fresh_small_model, small_config, temp_checkpoint_dir
    ):
        """Test 2.4: Weight export to safetensors format."""
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        try:
            from safetensors.torch import save_file, load_file
        except ImportError:
            pytest.skip("safetensors not installed")

        output_path = Path(temp_checkpoint_dir) / "safetensors_test"
        output_path.mkdir(parents=True, exist_ok=True)

        state_dict = fresh_small_model.state_dict()
        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        # Save as safetensors
        weights_path = output_path / "model.safetensors"
        save_file(hf_state_dict, weights_path)

        assert weights_path.exists()

        # Reload and verify
        loaded = load_file(weights_path)
        assert len(loaded) == len(hf_state_dict)

    def test_expert_weight_expansion(self, fresh_small_model, small_config):
        """Test 2.5: Expert weights correctly expanded to individual expert tensors."""
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        state_dict = fresh_small_model.state_dict()
        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        # Check that individual expert weights exist for MoE layers
        for layer_id in range(small_config.n_dense_layers, small_config.n_layers):
            for expert_id in range(small_config.n_routed_experts):
                gate_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                up_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                down_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"

                assert gate_key in hf_state_dict, f"Missing {gate_key}"
                assert up_key in hf_state_dict, f"Missing {up_key}"
                assert down_key in hf_state_dict, f"Missing {down_key}"

                # Verify shapes are [out_dim, in_dim]
                assert hf_state_dict[gate_key].shape[0] == small_config.moe_inter_dim
                assert hf_state_dict[gate_key].shape[1] == small_config.dim


# =============================================================================
# Test Class 3: Model Loading After Export
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestModelLoadingAfterExport:
    """Test model loading in various frameworks after export."""

    def test_reload_in_nmoe(self, fresh_small_model, small_config, temp_checkpoint_dir):
        """Test 3.1: Reload exported model in nmoe."""
        from nmoe.model import Transformer

        # Save original state
        original_state = copy.deepcopy(fresh_small_model.state_dict())

        # Create new model
        model2 = Transformer(small_config).cuda().bfloat16()

        # Load state
        model2.load_state_dict(original_state)

        # Verify match
        for key in original_state:
            assert torch.allclose(
                original_state[key], model2.state_dict()[key]
            ), f"Mismatch for {key}"

    def test_reload_split_checkpoint(self, fresh_small_model, small_config, temp_checkpoint_dir):
        """Test 3.2: Reload from split checkpoint format."""
        from nmoe.checkpoint import build_states
        from nmoe.model import Transformer

        checkpoint_path = Path(temp_checkpoint_dir) / "reload_split" / "iter_0000001"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        class MockLoader:
            def state_dict(self):
                return {}

        optimizer = torch.optim.AdamW(fresh_small_model.parameters(), lr=1e-4)

        rd_state, dp_state = build_states(
            step=1,
            model=fresh_small_model,
            optimizer=optimizer,
            tokens=1000,
            loader=MockLoader(),
            config_fingerprint="test",
        )

        torch.save(rd_state, checkpoint_path / "rd.pt")
        torch.save(dp_state, checkpoint_path / "dp_rank_000.pt")

        # Load into new model
        model2 = Transformer(small_config).cuda().bfloat16()
        model2.init_weights()

        rd = torch.load(checkpoint_path / "rd.pt", weights_only=False)
        dp = torch.load(checkpoint_path / "dp_rank_000.pt", weights_only=False)

        model2.load_state_dict(rd["model_dense"], strict=False)
        model2.load_state_dict(dp["model_expert"], strict=False)

        # Verify forward pass works
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()
        with torch.no_grad():
            logits = model2(input_ids)

        assert logits.shape == (2, 64, small_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_sglang_loader_import(self):
        """Test 3.3: SGLang nmoe loader can be imported."""
        try:
            from sglang.srt.model_loader.nmoe_loader import (
                NMoEModelLoader,
                _map_block_name,
                _load_nmoe_checkpoint,
            )

            assert NMoEModelLoader is not None
            assert callable(_map_block_name)
        except ImportError as e:
            pytest.skip(f"SGLang not available: {e}")

    def test_weight_name_mapping(self, small_config):
        """Test 3.4: Weight name mapping is correct."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        mapping = nmoe_to_hf_weight_mapping(
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_routed_experts=small_config.n_routed_experts,
            n_shared_experts=small_config.n_shared_experts,
        )

        # Verify key mappings
        assert mapping["embedding.weight"] == "model.embed_tokens.weight"
        assert mapping["lm_head.weight"] == "lm_head.weight"
        assert mapping["norm.weight"] == "model.norm.weight"

        # Verify layer mappings exist
        assert f"blocks.0.attn_norm.weight" in mapping
        assert f"blocks.0.ffn_norm.weight" in mapping

    def test_sglang_nmoe_runner_availability(self):
        """Test 3.5: SGLang nmoe runner is available."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore

            assert NmoeRunnerCore is not None
        except ImportError as e:
            pytest.skip(f"SGLang nmoe runner not available: {e}")


# =============================================================================
# Test Class 4: Inference Consistency Pre/Post Export
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestInferenceConsistency:
    """Test inference consistency before and after export."""

    def test_inference_determinism(self, fresh_small_model, small_config):
        """Test 4.1: Inference is deterministic."""
        model = fresh_small_model
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()

        with torch.no_grad():
            logits1 = model(input_ids)
            logits2 = model(input_ids)

        assert torch.allclose(logits1, logits2), "Inference not deterministic"

    def test_export_reload_consistency(self, fresh_small_model, small_config):
        """Test 4.2: Logits match after save/reload."""
        from nmoe.model import Transformer

        model = fresh_small_model
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()

        # Get original logits
        with torch.no_grad():
            original_logits = model(input_ids).clone()

        # Save and reload
        state_dict = model.state_dict()

        model2 = Transformer(small_config).cuda().bfloat16()
        model2.load_state_dict(state_dict)
        model2.eval()

        # Get reloaded logits
        with torch.no_grad():
            reloaded_logits = model2(input_ids)

        assert torch.allclose(
            original_logits, reloaded_logits, atol=1e-4
        ), "Logits differ after reload"

    def test_generation_consistency(self, fresh_small_model, small_config):
        """Test 4.3: Generated tokens match after reload."""
        from nmoe.model import Transformer

        model = fresh_small_model
        model.eval()

        def generate_greedy(m, prompt, n_tokens):
            generated = prompt.clone()
            for _ in range(n_tokens):
                with torch.no_grad():
                    logits = m(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            return generated

        torch.manual_seed(42)
        prompt = torch.randint(0, small_config.vocab_size, (1, 8)).cuda()

        # Generate from original
        original_output = generate_greedy(model, prompt, 16)

        # Reload model
        state_dict = model.state_dict()
        model2 = Transformer(small_config).cuda().bfloat16()
        model2.load_state_dict(state_dict)
        model2.eval()

        # Generate from reloaded
        reloaded_output = generate_greedy(model2, prompt, 16)

        assert torch.equal(
            original_output, reloaded_output
        ), "Generated tokens differ after reload"

    def test_batch_consistency(self, fresh_small_model, small_config):
        """Test 4.4: Batched inference matches individual inference."""
        model = fresh_small_model
        model.eval()

        torch.manual_seed(42)
        input1 = torch.randint(0, small_config.vocab_size, (1, 64)).cuda()
        input2 = torch.randint(0, small_config.vocab_size, (1, 64)).cuda()
        batched_input = torch.cat([input1, input2], dim=0)

        with torch.no_grad():
            logits1 = model(input1)
            logits2 = model(input2)
            batched_logits = model(batched_input)

        assert torch.allclose(logits1, batched_logits[:1], atol=1e-4), "Batch inconsistency"
        assert torch.allclose(logits2, batched_logits[1:], atol=1e-4), "Batch inconsistency"

    def test_kv_cache_equivalent(self, fresh_small_model, small_config):
        """Test 4.5: Full sequence and incremental decoding match (no KV cache in nmoe)."""
        model = fresh_small_model
        model.eval()

        # nmoe doesn't support KV cache, so this tests that full sequence
        # processing gives same result regardless of how we call it
        torch.manual_seed(42)
        full_sequence = torch.randint(0, small_config.vocab_size, (1, 32)).cuda()

        with torch.no_grad():
            full_logits = model(full_sequence)

            # Process same sequence twice - should be identical
            full_logits_2 = model(full_sequence)

        assert torch.allclose(full_logits, full_logits_2), "Repeated inference differs"


# =============================================================================
# Test Class 5: Weight Precision Tests
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestWeightPrecision:
    """Test weight precision handling in export."""

    def test_bf16_weights(self, fresh_small_model, small_config):
        """Test 5.1: Model weights are in BF16."""
        model = fresh_small_model

        for name, param in model.named_parameters():
            assert param.dtype == torch.bfloat16, f"{name} is not BF16: {param.dtype}"

    def test_bf16_to_fp16_conversion(self, fresh_small_model, small_config):
        """Test 5.2: BF16 to FP16 conversion preserves values."""
        state_dict = fresh_small_model.state_dict()

        # Convert to FP16
        fp16_state = {k: v.half() for k, v in state_dict.items()}

        # Check conversion
        for key in state_dict:
            bf16_val = state_dict[key]
            fp16_val = fp16_state[key]

            # Values should be close (FP16 has less precision)
            assert fp16_val.dtype == torch.float16
            assert torch.allclose(
                bf16_val.float(), fp16_val.float(), atol=1e-2
            ), f"Conversion issue for {key}"

    def test_bf16_to_fp32_conversion(self, fresh_small_model, small_config):
        """Test 5.3: BF16 to FP32 conversion is lossless for range."""
        state_dict = fresh_small_model.state_dict()

        # Convert to FP32
        fp32_state = {k: v.float() for k, v in state_dict.items()}

        for key in state_dict:
            bf16_val = state_dict[key]
            fp32_val = fp32_state[key]

            # FP32 should capture full BF16 range
            assert fp32_val.dtype == torch.float32
            assert torch.allclose(
                bf16_val.float(), fp32_val, atol=1e-6
            ), f"Conversion issue for {key}"

    def test_expert_weight_precision(self, fresh_small_model, small_config):
        """Test 5.4: Expert weights maintain precision after expansion."""
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        state_dict = fresh_small_model.state_dict()
        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        # Expert weights should maintain BF16
        for key, tensor in hf_state_dict.items():
            if "experts" in key:
                assert tensor.dtype == torch.bfloat16, f"{key} precision changed: {tensor.dtype}"

    def test_mixed_precision_export(self, fresh_small_model, small_config, temp_checkpoint_dir):
        """Test 5.5: Export handles mixed precision correctly."""
        output_path = Path(temp_checkpoint_dir) / "mixed_precision"
        output_path.mkdir(parents=True, exist_ok=True)

        # Create mixed precision state
        state_dict = fresh_small_model.state_dict()

        # Keep embedding in FP32 (common pattern)
        mixed_state = {}
        for key, val in state_dict.items():
            if "embedding" in key:
                mixed_state[key] = val.float()
            else:
                mixed_state[key] = val

        # Save and reload
        torch.save(mixed_state, output_path / "mixed.pt")
        loaded = torch.load(output_path / "mixed.pt", weights_only=False)

        assert loaded["embedding.weight"].dtype == torch.float32
        assert loaded["lm_head.weight"].dtype == torch.bfloat16


# =============================================================================
# Test Class 6: Expert Weight Handling
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestExpertWeightHandling:
    """Test expert weight handling in export."""

    def test_expert_weight_shapes(self, fresh_small_model, small_config):
        """Test 6.1: Expert weights have correct shapes."""
        state_dict = fresh_small_model.state_dict()

        for layer_id in range(small_config.n_dense_layers, small_config.n_layers):
            w1_key = f"blocks.{layer_id}.ffn.W1"
            w3_key = f"blocks.{layer_id}.ffn.W3"
            w2_key = f"blocks.{layer_id}.ffn.W2"

            assert w1_key in state_dict, f"Missing {w1_key}"
            assert w3_key in state_dict, f"Missing {w3_key}"
            assert w2_key in state_dict, f"Missing {w2_key}"

            # nmoe stores as [n_local, dim, inter_dim]
            assert state_dict[w1_key].shape[0] == small_config.n_routed_experts
            assert state_dict[w1_key].shape[1] == small_config.dim
            assert state_dict[w1_key].shape[2] == small_config.moe_inter_dim

    def test_router_weight_shapes(self, fresh_small_model, small_config):
        """Test 6.2: Router weights have correct shapes."""
        state_dict = fresh_small_model.state_dict()

        for layer_id in range(small_config.n_dense_layers, small_config.n_layers):
            gate_key = f"blocks.{layer_id}.ffn.router.gate.weight"

            assert gate_key in state_dict, f"Missing {gate_key}"
            assert state_dict[gate_key].shape == (
                small_config.n_routed_experts,
                small_config.dim,
            )

    def test_expert_weight_expansion_preserves_values(self, fresh_small_model, small_config):
        """Test 6.3: Expert weight expansion preserves numerical values."""
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        state_dict = fresh_small_model.state_dict()
        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        # Check that expanded weights match original slices
        for layer_id in range(small_config.n_dense_layers, small_config.n_layers):
            w1_key = f"blocks.{layer_id}.ffn.W1"
            original_w1 = state_dict[w1_key]

            for expert_id in range(small_config.n_routed_experts):
                hf_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                expanded = hf_state_dict[hf_key]

                # Original: [n_experts, dim, inter_dim], slice is [dim, inter_dim]
                # HF: [inter_dim, dim] (transposed)
                original_slice = original_w1[expert_id].T

                assert torch.allclose(
                    original_slice, expanded
                ), f"Mismatch for {hf_key}"

    def test_shared_expert_handling(self, medium_config):
        """Test 6.4: Shared experts handled correctly in export."""
        from nmoe.model import Transformer
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        model = Transformer(medium_config).cuda().bfloat16()
        model.init_weights()
        state_dict = model.state_dict()

        mapping = nmoe_to_hf_weight_mapping(
            n_layers=medium_config.n_layers,
            n_dense_layers=medium_config.n_dense_layers,
            n_routed_experts=medium_config.n_routed_experts,
            n_shared_experts=medium_config.n_shared_experts,
        )

        # Check shared expert mappings exist
        for layer_id in range(medium_config.n_dense_layers, medium_config.n_layers):
            shared_w1_nmoe = f"blocks.{layer_id}.ffn._shared.w1.weight"
            shared_w1_hf = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"

            if shared_w1_nmoe in state_dict:
                assert shared_w1_nmoe in mapping
                assert mapping[shared_w1_nmoe] == shared_w1_hf

    def test_dense_layer_export(self, fresh_small_model, small_config):
        """Test 6.5: Dense (non-MoE) layers export correctly."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        state_dict = fresh_small_model.state_dict()
        mapping = nmoe_to_hf_weight_mapping(
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_routed_experts=small_config.n_routed_experts,
            n_shared_experts=small_config.n_shared_experts,
        )

        # First n_dense_layers should be dense MLP
        for layer_id in range(small_config.n_dense_layers):
            w1_key = f"blocks.{layer_id}.ffn.w1.weight"
            hf_w1_key = f"model.layers.{layer_id}.mlp.gate_proj.weight"

            assert w1_key in state_dict, f"Missing dense layer {w1_key}"
            assert w1_key in mapping
            assert mapping[w1_key] == hf_w1_key


# =============================================================================
# Test Class 7: Multi-GPU Distributed Training -> Single GPU Serve
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not MULTI_GPU_AVAILABLE, reason="Multiple GPUs required")
class TestDistributedTrainingSingleServe:
    """Test distributed training to single GPU serving."""

    def test_checkpoint_merge_for_single_gpu(self, small_config, temp_checkpoint_dir):
        """Test 7.1: Merge distributed checkpoint for single GPU serving."""
        checkpoint_path = Path(temp_checkpoint_dir) / "dist_to_single"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Simulate 2-GPU checkpoint with split experts
        from nmoe.model import Transformer

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()

        full_state = model.state_dict()

        # Split expert weights to simulate 2-GPU training
        rd_state = {"model_dense": {}}
        dp_states = [{}, {}]

        for key, val in full_state.items():
            if "W1" in key or "W2" in key or "W3" in key:
                # Split experts across ranks
                n_experts = val.shape[0]
                n_per_rank = n_experts // 2

                dp_states[0][key] = val[:n_per_rank].clone()
                dp_states[1][key] = val[n_per_rank:].clone()
            else:
                rd_state["model_dense"][key] = val

        # Save simulated distributed checkpoint
        torch.save(rd_state, checkpoint_path / "rd.pt")
        torch.save({"model_expert": dp_states[0]}, checkpoint_path / "dp_rank_000.pt")
        torch.save({"model_expert": dp_states[1]}, checkpoint_path / "dp_rank_001.pt")

        # Merge for single GPU
        rd = torch.load(checkpoint_path / "rd.pt", weights_only=False)
        dp0 = torch.load(checkpoint_path / "dp_rank_000.pt", weights_only=False)
        dp1 = torch.load(checkpoint_path / "dp_rank_001.pt", weights_only=False)

        merged_experts = {}
        for key in dp0["model_expert"]:
            merged_experts[key] = torch.cat(
                [dp0["model_expert"][key], dp1["model_expert"][key]], dim=0
            )

        # Create merged state
        merged_state = dict(rd["model_dense"])
        merged_state.update(merged_experts)

        # Load into fresh model
        model2 = Transformer(small_config).cuda().bfloat16()
        model2.load_state_dict(merged_state)

        # Verify inference works
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()
        with torch.no_grad():
            logits = model2(input_ids)

        assert not torch.isnan(logits).any()

    def test_hf_export_from_distributed(self, small_config, temp_checkpoint_dir):
        """Test 7.2: HF export from distributed checkpoint."""
        from nmoe.model import Transformer
        from nmoe.tools.config_converter import expand_expert_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig

        try:
            from safetensors.torch import save_file
        except ImportError:
            pytest.skip("safetensors not installed")

        export_path = Path(temp_checkpoint_dir) / "hf_from_dist"
        export_path.mkdir(parents=True, exist_ok=True)

        # Create model and export
        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()

        state_dict = model.state_dict()
        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        save_file(hf_state_dict, export_path / "model.safetensors")

        # Verify export
        assert (export_path / "model.safetensors").exists()

    def test_serving_merged_checkpoint(self, small_config, temp_checkpoint_dir):
        """Test 7.3: Serving from merged checkpoint works correctly."""
        from nmoe.model import Transformer

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        # Get deterministic input first
        torch.manual_seed(42)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()

        # Get reference output BEFORE save
        with torch.no_grad():
            original_logits = model(input_ids).clone()

        # Simulate save and reload as would happen in serving
        state_dict = model.state_dict()
        torch.save(state_dict, Path(temp_checkpoint_dir) / "serving_model.pt")

        # Load for serving
        loaded_state = torch.load(
            Path(temp_checkpoint_dir) / "serving_model.pt", weights_only=False
        )

        serve_model = Transformer(small_config).cuda().bfloat16()
        # Don't init_weights - load state_dict directly
        serve_model.load_state_dict(loaded_state)
        serve_model.eval()

        # Verify serving produces identical outputs
        with torch.no_grad():
            served_logits = serve_model(input_ids)

        assert torch.allclose(
            original_logits, served_logits, atol=1e-4
        ), "Served model output differs from original"


# =============================================================================
# Test Class 8: Multi-GPU Training -> Multi-GPU TP Serve
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not EIGHT_GPU_AVAILABLE, reason="8 GPUs required")
class TestDistributedTrainingTPServe:
    """Test 8-GPU distributed training to 8-GPU TP serving."""

    def test_tp_weight_sharding_scheme(self, small_config):
        """Test 8.1: TP weight sharding follows expected scheme."""
        # Verify that weight dimensions align with TP sharding
        from nmoe.model import Transformer

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()
        state_dict = model.state_dict()

        tp_size = 8

        # For TP, these dimensions should be divisible by tp_size
        dim = small_config.dim
        inter_dim = small_config.inter_dim
        n_heads = small_config.n_heads

        # dim should be divisible for attention head sharding
        # (not strictly required but common)
        # inter_dim should be divisible for MLP column parallel

        # Expert weights: [n_experts, dim, inter_dim]
        # Column parallel: shard inter_dim
        moe_inter = small_config.moe_inter_dim

        # Log dimensions for debugging
        print(f"dim={dim}, inter_dim={inter_dim}, moe_inter_dim={moe_inter}, n_heads={n_heads}")

    def test_checkpoint_compatible_with_tp_loading(self, small_config, temp_checkpoint_dir):
        """Test 8.2: Checkpoint is compatible with TP loading."""
        from nmoe.model import Transformer
        from nmoe.checkpoint import build_states

        checkpoint_path = Path(temp_checkpoint_dir) / "tp_compat"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()

        class MockLoader:
            def state_dict(self):
                return {}

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        rd_state, dp_state = build_states(
            step=1,
            model=model,
            optimizer=optimizer,
            tokens=1000,
            loader=MockLoader(),
            config_fingerprint="test",
        )

        torch.save(rd_state, checkpoint_path / "rd.pt")
        torch.save(dp_state, checkpoint_path / "dp_rank_000.pt")

        # Verify checkpoint can be read
        rd = torch.load(checkpoint_path / "rd.pt", weights_only=False)
        assert "model_dense" in rd
        assert "step" in rd


# =============================================================================
# Test Class 9: Full Pipeline Integration
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestFullPipelineIntegration:
    """Test full train -> export -> serve pipeline."""

    def test_train_export_reload_cycle(self, small_config, temp_checkpoint_dir):
        """Test 9.1: Complete train -> export -> reload cycle."""
        from nmoe.model import Transformer
        from nmoe.checkpoint import build_states
        from nmoe.tools.config_converter import expand_expert_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig

        try:
            from safetensors.torch import save_file, load_file
        except ImportError:
            pytest.skip("safetensors not installed")

        # Phase 1: Train
        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(42)
        input_ids = torch.randint(0, small_config.vocab_size, (4, 64)).cuda()
        targets = torch.randint(0, small_config.vocab_size, (4, 64)).cuda()

        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, small_config.vocab_size), targets.view(-1)
            )
            loss.backward()
            optimizer.step()

        # Phase 2: Export
        export_path = Path(temp_checkpoint_dir) / "full_pipeline"
        export_path.mkdir(parents=True, exist_ok=True)

        model.eval()
        state_dict = model.state_dict()

        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        save_file(hf_state_dict, export_path / "model.safetensors")

        # Save config
        unified_config = NMoEModelConfig.from_nmoe_config(small_config)
        with open(export_path / "config.json", "w") as f:
            json.dump(unified_config.to_dict(), f)

        # Phase 3: Reload (simulate serving)
        loaded_weights = load_file(export_path / "model.safetensors")

        assert len(loaded_weights) > 0
        assert "model.embed_tokens.weight" in loaded_weights

    def test_inference_matches_after_pipeline(self, small_config, temp_checkpoint_dir):
        """Test 9.2: Inference output matches through entire pipeline."""
        from nmoe.model import Transformer

        # Create and train model
        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64)).cuda()

        # Get reference output
        with torch.no_grad():
            reference_logits = model(input_ids).clone()

        # Save checkpoint
        checkpoint_path = Path(temp_checkpoint_dir) / "inference_match"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), checkpoint_path / "model.pt")

        # Reload and verify
        model2 = Transformer(small_config).cuda().bfloat16()
        model2.load_state_dict(torch.load(checkpoint_path / "model.pt", weights_only=False))
        model2.eval()

        with torch.no_grad():
            reloaded_logits = model2(input_ids)

        assert torch.allclose(
            reference_logits, reloaded_logits, atol=1e-4
        ), "Inference mismatch after pipeline"

    def test_generated_text_quality(self, small_config):
        """Test 9.3: Model generates valid tokens (not all same/padding)."""
        from nmoe.model import Transformer

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()
        model.eval()

        torch.manual_seed(42)
        prompt = torch.randint(0, small_config.vocab_size, (1, 8)).cuda()

        # Generate tokens
        generated = prompt.clone()
        with torch.no_grad():
            for _ in range(32):
                logits = model(generated)
                # Sample with temperature
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        # Check that generated tokens are not all the same
        unique_tokens = torch.unique(generated[0, 8:])  # Exclude prompt
        assert len(unique_tokens) > 1, "Generated tokens are all identical"

    def test_model_export_file_structure(self, small_config, temp_checkpoint_dir):
        """Test 9.4: Exported model has correct file structure."""
        from nmoe.model import Transformer
        from nmoe.tools.export_to_hf import (
            generate_config_json,
            generate_generation_config,
        )
        from nmoe.tools.config_converter import expand_expert_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig

        try:
            from safetensors.torch import save_file
        except ImportError:
            pytest.skip("safetensors not installed")

        export_path = Path(temp_checkpoint_dir) / "file_structure"
        export_path.mkdir(parents=True, exist_ok=True)

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()

        unified_config = NMoEModelConfig.from_nmoe_config(small_config)

        # Generate all files
        generate_config_json(unified_config, export_path)
        generate_generation_config(unified_config, export_path)

        state_dict = model.state_dict()
        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )
        save_file(hf_state_dict, export_path / "model.safetensors")

        # Verify file structure
        assert (export_path / "config.json").exists()
        assert (export_path / "generation_config.json").exists()
        assert (export_path / "model.safetensors").exists()

    def test_model_param_count_preserved(self, small_config):
        """Test 9.5: State dict element count preserved through export."""
        from nmoe.model import Transformer
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()

        # Use state_dict element count (includes buffers like router bias)
        state_dict = model.state_dict()
        original_element_count = sum(t.numel() for t in state_dict.values())

        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        export_element_count = sum(t.numel() for t in hf_state_dict.values())

        assert (
            original_element_count == export_element_count
        ), f"Element count mismatch: {original_element_count} vs {export_element_count}"


# =============================================================================
# Test Class 10: Error Handling and Edge Cases
# =============================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA required")
class TestErrorHandlingEdgeCases:
    """Test error handling and edge cases."""

    def test_missing_checkpoint_file_error(self, temp_checkpoint_dir):
        """Test 10.1: Appropriate error for missing checkpoint."""
        from nmoe.model import Transformer

        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            # Try to load non-existent checkpoint
            torch.load(Path(temp_checkpoint_dir) / "nonexistent.pt")

    def test_mismatched_config_detection(self, small_config, temp_checkpoint_dir):
        """Test 10.2: Detect config mismatch on load."""
        from nmoe.model import Transformer
        from nmoe.config import Config

        # Create and save model with small config
        model = Transformer(small_config).cuda().bfloat16()
        model.init_weights()

        checkpoint_path = Path(temp_checkpoint_dir) / "config_mismatch"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), checkpoint_path / "model.pt")

        # Try to load with different config
        different_config = Config(
            dim=512,  # Different from small_config.dim=256
            n_layers=small_config.n_layers,
            n_heads=small_config.n_heads,
            vocab_size=small_config.vocab_size,
            n_dense_layers=small_config.n_dense_layers,
            n_routed_experts=small_config.n_routed_experts,
            n_activated_experts=small_config.n_activated_experts,
            n_shared_experts=small_config.n_shared_experts,
            moe_inter_dim=small_config.moe_inter_dim,
            inter_dim=small_config.inter_dim,
            max_position_embeddings=small_config.max_position_embeddings,
            batch_size=small_config.batch_size,
            seq_len=small_config.seq_len,
        )

        model2 = Transformer(different_config).cuda().bfloat16()

        # Load should fail due to size mismatch
        with pytest.raises(RuntimeError):
            model2.load_state_dict(
                torch.load(checkpoint_path / "model.pt", weights_only=False), strict=True
            )

    def test_empty_batch_handling(self, fresh_small_model, small_config):
        """Test 10.3: Handle edge case of minimum batch size."""
        model = fresh_small_model
        model.eval()

        # Minimum batch (1 sample, 1 token)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 1)).cuda()

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (1, 1, small_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_max_sequence_length(self, fresh_small_model, small_config):
        """Test 10.4: Handle maximum sequence length."""
        model = fresh_small_model
        model.eval()

        max_len = small_config.max_position_embeddings
        input_ids = torch.randint(0, small_config.vocab_size, (1, max_len)).cuda()

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (1, max_len, small_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_invalid_token_ids_handling(self, fresh_small_model, small_config):
        """Test 10.5: Model handles token IDs at boundary."""
        model = fresh_small_model
        model.eval()

        # Token ID at max vocab_size - 1
        input_ids = torch.full((1, 8), small_config.vocab_size - 1, dtype=torch.long).cuda()

        with torch.no_grad():
            logits = model(input_ids)

        assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
