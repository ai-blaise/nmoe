"""End-to-end test for model export.

Tests export on small model (dim=512, n_layers=4).
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/nourdine/sglang_nmoe/nether-soup/nmoe')

import torch
import torch.distributed as dist

from nmoe.config import Config
from nmoe.model import Transformer
from nmoe.unified.config import NMoEModelConfig
from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping, expand_expert_weights_to_hf
from nmoe.tools.export_to_hf import generate_config_json, generate_generation_config


class TestSmallModelExport:
    """Test export on small model."""

    @pytest.fixture
    def small_config(self):
        """Create small test config (dim=512, n_layers=4)."""
        return Config(
            dim=512,
            n_layers=4,
            n_heads=8,
            inter_dim=1024,
            moe_inter_dim=256,
            n_routed_experts=8,
            n_activated_experts=2,
            n_shared_experts=1,
            n_dense_layers=1,
            vocab_size=1000,
            batch_size=2,
            seq_len=64,
            max_position_embeddings=128,
            dtype='bf16',
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

    @pytest.fixture
    def init_distributed(self):
        """Initialize distributed if needed."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29503',
                world_size=1,
                rank=0,
            )
        yield
        # Don't destroy - other tests may need it

    def test_config_json_generation(self, small_config):
        """Test config.json generation."""
        unified = NMoEModelConfig.from_nmoe_config(small_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            hf_config = generate_config_json(unified, output_path)

            # Verify config.json exists and is valid
            config_path = output_path / "config.json"
            assert config_path.exists()

            with open(config_path) as f:
                loaded = json.load(f)

            assert loaded['hidden_size'] == small_config.dim
            assert loaded['num_hidden_layers'] == small_config.n_layers
            assert loaded['n_routed_experts'] == small_config.n_routed_experts
            assert loaded['model_type'] == 'nmoe'
            assert 'NMoEForCausalLM' in loaded['architectures']

    def test_generation_config_generation(self, small_config):
        """Test generation_config.json generation."""
        unified = NMoEModelConfig.from_nmoe_config(small_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            gen_config = generate_generation_config(unified, output_path)

            config_path = output_path / "generation_config.json"
            assert config_path.exists()

            with open(config_path) as f:
                loaded = json.load(f)

            assert loaded['eos_token_id'] == small_config.eos_token_id
            assert loaded['max_length'] == small_config.max_position_embeddings

    def test_weight_export_shapes(self, small_config, init_distributed):
        """Test that exported weights have correct shapes."""
        model = Transformer(small_config).cuda()
        model.init_weights()

        state_dict = model.state_dict()
        unified = NMoEModelConfig.from_nmoe_config(small_config)

        # Expand expert weights
        hf_state_dict = expand_expert_weights_to_hf(
            state_dict,
            n_layers=small_config.n_layers,
            n_dense_layers=small_config.n_dense_layers,
            n_experts=small_config.n_routed_experts,
        )

        # Verify embedding
        assert 'model.embed_tokens.weight' in hf_state_dict
        assert hf_state_dict['model.embed_tokens.weight'].shape == (
            small_config.vocab_size, small_config.dim
        )

        # Verify expanded expert weights exist
        # Layer 1 is first MoE layer (layer 0 is dense)
        for expert_id in range(small_config.n_routed_experts):
            gate_key = f'model.layers.1.mlp.experts.{expert_id}.gate_proj.weight'
            assert gate_key in hf_state_dict, f"Missing {gate_key}"
            # HF format: [inter_dim, dim]
            assert hf_state_dict[gate_key].shape == (
                small_config.moe_inter_dim, small_config.dim
            )

    def test_full_export_pipeline(self, small_config, init_distributed):
        """Test complete export pipeline."""
        model = Transformer(small_config).cuda()
        model.init_weights()

        state_dict = model.state_dict()
        unified = NMoEModelConfig.from_nmoe_config(small_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Generate configs
            generate_config_json(unified, output_path)
            generate_generation_config(unified, output_path)

            # Export weights
            hf_state_dict = expand_expert_weights_to_hf(
                state_dict,
                n_layers=small_config.n_layers,
                n_dense_layers=small_config.n_dense_layers,
                n_experts=small_config.n_routed_experts,
            )

            # Save with safetensors
            from safetensors.torch import save_file
            weights_path = output_path / "model.safetensors"
            save_file(hf_state_dict, weights_path)

            # Verify all files exist
            assert (output_path / "config.json").exists()
            assert (output_path / "generation_config.json").exists()
            assert weights_path.exists()

            # Verify safetensors can be loaded
            from safetensors.torch import load_file
            loaded = load_file(weights_path)
            assert len(loaded) == len(hf_state_dict)

    def test_model_output_after_export_reload(self, small_config, init_distributed):
        """Test that model produces same output after export/reload."""
        model = Transformer(small_config).cuda()
        model.init_weights()
        model.eval()

        # Get original output
        tokens = torch.randint(0, small_config.vocab_size, (1, 32), device='cuda')
        with torch.no_grad():
            original_logits = model(tokens)

        # Export state dict
        state_dict = model.state_dict()

        # Create new model and load state
        model2 = Transformer(small_config).cuda()
        model2.load_state_dict(state_dict)
        model2.eval()

        # Get output from reloaded model
        with torch.no_grad():
            reloaded_logits = model2(tokens)

        # Outputs should be identical
        assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


class TestExportedModelInference:
    """Test inference with exported model."""

    @pytest.fixture
    def small_config(self):
        return Config(
            dim=512,
            n_layers=4,
            n_heads=8,
            inter_dim=1024,
            moe_inter_dim=256,
            n_routed_experts=8,
            n_activated_experts=2,
            n_shared_experts=1,
            n_dense_layers=1,
            vocab_size=1000,
            batch_size=2,
            seq_len=64,
            max_position_embeddings=128,
            dtype='bf16',
        )

    @pytest.fixture
    def init_distributed(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29504',
                world_size=1,
                rank=0,
            )
        yield

    def test_greedy_generation(self, small_config, init_distributed):
        """Test greedy generation produces valid tokens."""
        model = Transformer(small_config).cuda()
        model.init_weights()
        model.eval()

        # Simple greedy generation
        prompt = torch.randint(0, small_config.vocab_size, (1, 8), device='cuda')

        generated = prompt.clone()
        with torch.no_grad():
            for _ in range(16):
                logits = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        assert generated.shape == (1, 24)  # 8 prompt + 16 generated
        assert (generated >= 0).all()
        assert (generated < small_config.vocab_size).all()

    def test_batch_generation_consistency(self, small_config, init_distributed):
        """Test that batched generation is consistent with single-sample."""
        model = Transformer(small_config).cuda()
        model.init_weights()
        model.eval()

        prompt = torch.randint(0, small_config.vocab_size, (1, 8), device='cuda')
        batched_prompt = prompt.repeat(2, 1)

        with torch.no_grad():
            single_logits = model(prompt)
            batched_logits = model(batched_prompt)

        # Both samples in batch should have same output
        assert torch.allclose(
            batched_logits[0], batched_logits[1], atol=1e-5
        )
        # Should match single sample output
        assert torch.allclose(
            single_logits[0], batched_logits[0], atol=1e-5
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
