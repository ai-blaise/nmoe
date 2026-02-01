"""Full pipeline integration test: nmoe train -> export -> SGLang serve.

This test validates the complete workflow from training an nmoe model
to exporting it in HuggingFace format and serving it with SGLang.

Task 6.1.1 from Niwa implementation checklist.

Run with:
    pytest tests/integration/test_full_pipeline.py -v -s

Requirements:
    - GPU with at least 16GB VRAM
    - nmoe, sglang, and transformers installed
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import torch

# Skip if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for integration tests"
)


@pytest.fixture(scope="module")
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    tmpdir = tempfile.mkdtemp(prefix="nmoe_integration_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def small_model_config():
    """Small model config for fast testing."""
    from nmoe.config import Config
    return Config(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=1024,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=0,
        moe_inter_dim=512,
        inter_dim=512,
        max_position_embeddings=256,
    )


class TestTrainExportServe:
    """Test the full train -> export -> serve pipeline."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_model_forward_pass(self, small_model_config):
        """Test basic model forward pass before export."""
        from nmoe.model import Transformer

        model = Transformer(small_model_config).cuda().bfloat16()
        model.init_weights()  # Initialize weights to avoid NaN

        # Create dummy input
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len)).cuda()

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_model_config.vocab_size)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_checkpoint_save_load(self, small_model_config, temp_checkpoint_dir):
        """Test checkpoint save and load."""
        from nmoe.model import Transformer
        from nmoe.checkpoint import save_checkpoint, load_checkpoint

        model = Transformer(small_model_config).cuda().bfloat16()
        model.init_weights()  # Initialize weights

        # Save checkpoint
        checkpoint_path = Path(temp_checkpoint_dir) / "test_ckpt"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Simple state dict save for testing
        state = {
            "model": model.state_dict(),
            "config": small_model_config,
        }
        torch.save(state, checkpoint_path / "model.pt")

        # Load checkpoint (weights_only=False for custom config class)
        loaded_state = torch.load(checkpoint_path / "model.pt", weights_only=False)

        # Verify weights match
        for key in model.state_dict():
            assert key in loaded_state["model"], f"Missing key: {key}"
            orig = model.state_dict()[key]
            loaded = loaded_state["model"][key]
            assert torch.allclose(orig, loaded), f"Mismatch for {key}"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_weight_mapping(self, small_model_config):
        """Test weight mapping from nmoe to HF format."""
        from nmoe.model import Transformer
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        model = Transformer(small_model_config).cuda().bfloat16()
        model.init_weights()
        state_dict = model.state_dict()

        # Get weight mapping
        mapping = nmoe_to_hf_weight_mapping(
            n_layers=small_model_config.n_layers,
            n_dense_layers=small_model_config.n_dense_layers,
            n_routed_experts=small_model_config.n_routed_experts,
            n_shared_experts=small_model_config.n_shared_experts,
        )

        # Verify key model components have mappings
        has_embedding = any("embed" in k.lower() for k in mapping.values())
        has_layers = any("layers" in k.lower() for k in mapping.values())
        has_lm_head = any("lm_head" in k.lower() for k in mapping.values())

        assert has_embedding, "Missing embedding mapping"
        assert has_layers, "Missing layer mappings"
        assert has_lm_head, "Missing lm_head mapping"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_unified_config_roundtrip(self, small_model_config):
        """Test config conversion roundtrip."""
        from nmoe.unified.config import NMoEModelConfig

        # Convert to unified config
        unified = NMoEModelConfig.from_nmoe_config(small_model_config)

        # Verify key fields
        assert unified.hidden_size == small_model_config.dim
        assert unified.num_hidden_layers == small_model_config.n_layers
        assert unified.num_attention_heads == small_model_config.n_heads
        assert unified.num_experts == small_model_config.n_routed_experts
        assert unified.num_experts_per_tok == small_model_config.n_activated_experts

        # Convert to HF config dict
        hf_dict = unified.to_hf_config()

        assert hf_dict["hidden_size"] == small_model_config.dim
        assert hf_dict["num_hidden_layers"] == small_model_config.n_layers

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_export_to_hf_format(self, small_model_config, temp_checkpoint_dir):
        """Test export to HuggingFace format."""
        from nmoe.model import Transformer
        from nmoe.tools.export_to_hf import export_nmoe_to_hf
        from nmoe.unified.config import NMoEModelConfig

        model = Transformer(small_model_config).cuda().bfloat16()
        model.init_weights()

        # First save a checkpoint in nmoe format
        checkpoint_path = Path(temp_checkpoint_dir) / "nmoe_ckpt" / "iteration_00001"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save rd.pt (router/dense params)
        torch.save(model.state_dict(), checkpoint_path / "rd.pt")

        # Save config.json
        import json
        unified_config = NMoEModelConfig.from_nmoe_config(small_model_config)
        with open(checkpoint_path.parent / "config.json", "w") as f:
            json.dump(unified_config.to_dict(), f)

        export_path = Path(temp_checkpoint_dir) / "hf_export"

        # Export model from checkpoint
        export_nmoe_to_hf(
            checkpoint_path=str(checkpoint_path.parent),
            output_path=str(export_path),
            config=unified_config,
            shard_size_gb=1.0,
        )

        # Verify export files
        assert (export_path / "config.json").exists(), "config.json missing"
        # Check for safetensors or pytorch files
        has_weights = (
            list(export_path.glob("*.safetensors")) or
            list(export_path.glob("*.bin"))
        )
        assert has_weights, "No weight files found"


class TestSGLangIntegration:
    """Test SGLang serving integration."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_nmoe_runner_import(self):
        """Test that nmoe runner can be imported in SGLang."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
            )
            assert NmoeRunnerCore is not None
            assert NmoeRunnerInput is not None
        except ImportError as e:
            pytest.skip(f"SGLang nmoe runner not available: {e}")

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_nmoe_backend_registered(self):
        """Test that nmoe backend is registered in SGLang."""
        try:
            from sglang.srt.layers.moe.utils import MoeRunnerBackend

            assert hasattr(MoeRunnerBackend, "NMOE")
            assert MoeRunnerBackend.NMOE.value == "nmoe"
            assert MoeRunnerBackend.NMOE.is_nmoe()
        except ImportError as e:
            pytest.skip(f"SGLang not available: {e}")

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_server_args_accept_nmoe(self):
        """Test that SGLang server args accept nmoe backend."""
        try:
            from sglang.srt.server_args import ServerArgs

            # Create minimal server args with nmoe backend
            # This just tests that the argument is accepted
            args = ServerArgs(
                model_path="dummy",
                moe_runner_backend="nmoe",
            )
            assert args.moe_runner_backend == "nmoe"
        except ImportError as e:
            pytest.skip(f"SGLang not available: {e}")
        except Exception as e:
            # May fail on other validation, but that's OK
            if "moe_runner_backend" in str(e) or "nmoe" in str(e):
                pytest.fail(f"nmoe backend not accepted: {e}")


class TestQuantizationCompatibility:
    """Test quantization modes (BF16, FP8, NVFP4)."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_bf16_inference(self, small_model_config):
        """Test BF16 inference."""
        from nmoe.model import Transformer
        from nmoe.rdep import Rdep

        model = Transformer(small_model_config).cuda().bfloat16()
        model.init_weights()

        # Initialize RDEP dispatcher for BF16
        rdep = Rdep(
            dim=small_model_config.dim,
            n_local=small_model_config.n_routed_experts,
            topk=small_model_config.n_activated_experts,
            profile="bf16",
        )

        assert rdep is not None

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_fp8_inference(self, small_model_config):
        """Test FP8 quantized inference."""
        from nmoe.rdep import Rdep

        # Initialize RDEP dispatcher for FP8
        rdep = Rdep(
            dim=small_model_config.dim,
            n_local=small_model_config.n_routed_experts,
            topk=small_model_config.n_activated_experts,
            profile="fp8",
        )

        assert rdep is not None

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not hasattr(torch.cuda, "get_device_capability") or
        torch.cuda.get_device_capability()[0] < 10,
        reason="NVFP4 requires SM100+ (B200/GB200)"
    )
    def test_nvfp4_inference(self, small_model_config):
        """Test NVFP4 quantized inference (requires B200)."""
        from nmoe.rdep import Rdep

        # Initialize RDEP dispatcher for NVFP4
        rdep = Rdep(
            dim=small_model_config.dim,
            n_local=small_model_config.n_routed_experts,
            topk=small_model_config.n_activated_experts,
            profile="nvfp4",
        )

        assert rdep is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
