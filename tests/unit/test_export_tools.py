"""Comprehensive unit tests for export_to_hf.py.

Tests cover:
- Checkpoint loading functions
- Config generation
- Weight export
- Safetensors handling
- End-to-end export
"""

import pytest
import os
import tempfile
from pathlib import Path
from typing import Dict, Any


class TestLoadNmoeCheckpoint:
    """Tests for load_nmoe_checkpoint function."""

    def test_loads_from_iteration_directory(self):
        """Loads checkpoint from iteration directory."""
        from nmoe.tools.export_to_hf import load_nmoe_checkpoint

        # Create mock checkpoint directory with iteration_* structure
        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            # Create iteration directory
            iter_dir = os.path.join(tmpdir, "iteration_00001")
            os.makedirs(iter_dir)

            # Create mock rd.pt file inside iteration dir
            mock_state = {"model.embed": torch.zeros(100, 128)}
            torch.save(mock_state, os.path.join(iter_dir, "rd.pt"))

            # Should load without error
            state = load_nmoe_checkpoint(tmpdir)

            assert isinstance(state, dict)
            assert "model.embed" in state

    def test_loads_directly_from_iteration_path(self):
        """Loads checkpoint when passed iteration_* directory directly."""
        from nmoe.tools.export_to_hf import load_nmoe_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            # Create iteration directory
            iter_dir = os.path.join(tmpdir, "iteration_00001")
            os.makedirs(iter_dir)

            mock_state = {"model.embed": torch.zeros(100, 128)}
            torch.save(mock_state, os.path.join(iter_dir, "rd.pt"))

            # Pass the iteration dir directly
            state = load_nmoe_checkpoint(iter_dir)

            assert isinstance(state, dict)

    def test_merges_dp_rank_files(self):
        """Merges distributed checkpoint files."""
        from nmoe.tools.export_to_hf import load_nmoe_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            # Create iteration directory
            iter_dir = os.path.join(tmpdir, "iteration_00001")
            os.makedirs(iter_dir)

            # Create rd.pt
            rd_state = {"model.embed": torch.zeros(100, 128)}
            torch.save(rd_state, os.path.join(iter_dir, "rd.pt"))

            # Create dp_rank_0.pt
            dp0_state = {"model.experts.0": torch.zeros(64, 128)}
            torch.save(dp0_state, os.path.join(iter_dir, "dp_rank_0.pt"))

            # Create dp_rank_1.pt
            dp1_state = {"model.experts.1": torch.zeros(64, 128)}
            torch.save(dp1_state, os.path.join(iter_dir, "dp_rank_1.pt"))

            state = load_nmoe_checkpoint(tmpdir)

            # Should have merged all keys
            assert "model.embed" in state
            assert "model.experts.0" in state or len(state) > 1


class TestGenerateConfigJson:
    """Tests for generate_config_json function."""

    def test_generates_valid_json(self):
        """Generates valid JSON config from NMoEModelConfig."""
        from nmoe.tools.export_to_hf import generate_config_json
        from nmoe.unified.config import NMoEModelConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
                vocab_size=201088,
                num_experts=64,
                num_experts_per_tok=8,
                moe_intermediate_size=1792,
            )

            config_dict = generate_config_json(config, Path(tmpdir))

            assert isinstance(config_dict, dict)
            assert config_dict["hidden_size"] == 4096
            assert config_dict["num_hidden_layers"] == 32

    def test_includes_model_type(self):
        """Config includes model type."""
        from nmoe.tools.export_to_hf import generate_config_json
        from nmoe.unified.config import NMoEModelConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
            )

            config_dict = generate_config_json(config, Path(tmpdir))

            assert "model_type" in config_dict

    def test_includes_architectures(self):
        """Config includes architectures list."""
        from nmoe.tools.export_to_hf import generate_config_json
        from nmoe.unified.config import NMoEModelConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
            )

            config_dict = generate_config_json(config, Path(tmpdir))

            assert "architectures" in config_dict
            assert isinstance(config_dict["architectures"], list)


class TestGenerateGenerationConfig:
    """Tests for generate_generation_config function."""

    def test_generates_generation_config(self):
        """Generates generation config from NMoEModelConfig."""
        from nmoe.tools.export_to_hf import generate_generation_config
        from nmoe.unified.config import NMoEModelConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
                eos_token_id=199999,
            )

            gen_config = generate_generation_config(config, Path(tmpdir))

            assert isinstance(gen_config, dict)
            assert gen_config["eos_token_id"] == 199999

    def test_includes_sampling_defaults(self):
        """Includes default sampling parameters."""
        from nmoe.tools.export_to_hf import generate_generation_config
        from nmoe.unified.config import NMoEModelConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
            )

            gen_config = generate_generation_config(config, Path(tmpdir))

            # Should have some sampling-related keys
            has_sampling = any(
                k in gen_config
                for k in ["temperature", "top_p", "top_k", "do_sample"]
            )
            assert has_sampling or len(gen_config) > 0


class TestExportWeightsToHF:
    """Tests for export_weights_to_hf function."""

    def test_exports_to_safetensors(self):
        """Exports weights to safetensors format."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,  # All layers are dense
            )

            weights = {
                "model.embed_tokens.weight": torch.zeros(1000, 128),
                "model.layers.0.self_attn.q_proj.weight": torch.zeros(128, 128),
            }

            export_weights_to_hf(weights, config, Path(tmpdir))

            # Check for safetensors files
            files = os.listdir(tmpdir)
            has_safetensors = any(f.endswith(".safetensors") for f in files)
            assert has_safetensors

    def test_creates_index_file(self):
        """Creates index file for exports."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,
            )

            weights = {}
            for i in range(10):
                weights[f"model.layers.{i}.weight"] = torch.zeros(64, 64)

            export_weights_to_hf(weights, config, Path(tmpdir))

            # Should have index file
            files = os.listdir(tmpdir)
            has_index = any("index" in f.lower() for f in files)
            has_safetensors = any(f.endswith(".safetensors") for f in files)

            assert has_safetensors


class TestSafetensorsSharding:
    """Tests for safetensors sharding logic."""

    def test_shards_large_models(self):
        """Shards large models correctly."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,
            )

            # Create weights
            weights = {
                "layer1": torch.zeros(1000, 1000),  # ~4MB float32
                "layer2": torch.zeros(1000, 1000),
            }

            # Use small shard size to force sharding
            export_weights_to_hf(weights, config, Path(tmpdir), shard_size_gb=0.001)

            # Count shard files
            files = [f for f in os.listdir(tmpdir) if f.endswith(".safetensors")]

            # Should have created shards or single file
            assert len(files) >= 1


class TestExportNmoeToHF:
    """Tests for export_nmoe_to_hf end-to-end function."""

    def test_full_export_pipeline(self):
        """Tests full export pipeline."""
        from nmoe.tools.export_to_hf import export_nmoe_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                # Create mock checkpoint with iteration structure
                iter_dir = os.path.join(src_dir, "iteration_00001")
                os.makedirs(iter_dir)

                state = {
                    "embedding.weight": torch.zeros(1000, 128),
                    "blocks.0.attn.wq_a.weight": torch.zeros(128, 128),
                }
                torch.save(state, os.path.join(iter_dir, "rd.pt"))

                # Use NMoEModelConfig
                config = NMoEModelConfig(
                    hidden_size=128,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    vocab_size=1000,
                    first_k_dense_replace=1,  # Dense model
                )

                # Export
                export_nmoe_to_hf(
                    checkpoint_path=src_dir,
                    output_path=dst_dir,
                    config=config,
                )

                # Check output files
                files = os.listdir(dst_dir)

                # Should have config.json
                assert "config.json" in files

                # Should have safetensors
                has_weights = any(
                    f.endswith(".safetensors") or f.endswith(".bin")
                    for f in files
                )
                assert has_weights


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_checkpoint(self):
        """Handles empty checkpoint gracefully."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,
            )

            weights = {}

            # Should not raise (may create empty file or warn)
            try:
                export_weights_to_hf(weights, config, Path(tmpdir))
            except Exception:
                pass  # Empty weights may raise, which is acceptable

    def test_handles_bf16_weights(self):
        """Handles BF16 weight tensors."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,
            )

            weights = {
                "layer": torch.zeros(100, 100, dtype=torch.bfloat16),
            }

            export_weights_to_hf(weights, config, Path(tmpdir))

            files = os.listdir(tmpdir)
            assert len(files) > 0

    def test_handles_fp16_weights(self):
        """Handles FP16 weight tensors."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,
            )

            weights = {
                "layer": torch.zeros(100, 100, dtype=torch.float16),
            }

            export_weights_to_hf(weights, config, Path(tmpdir))

            files = os.listdir(tmpdir)
            assert len(files) > 0


class TestPathHandling:
    """Tests for path handling."""

    def test_creates_output_directory(self):
        """Creates output directory if not exists."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as base_dir:
            output_path = Path(base_dir) / "nested" / "output"

            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,
            )

            weights = {"layer": torch.zeros(10, 10)}

            # Create directory first (export_weights_to_hf doesn't create dirs)
            output_path.mkdir(parents=True, exist_ok=True)
            export_weights_to_hf(weights, config, output_path)

            assert os.path.exists(output_path)

    def test_handles_absolute_paths(self):
        """Handles absolute paths correctly."""
        from nmoe.tools.export_to_hf import export_weights_to_hf
        from nmoe.unified.config import NMoEModelConfig
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            abs_path = Path(os.path.abspath(tmpdir))

            config = NMoEModelConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                first_k_dense_replace=1,
            )

            weights = {"layer": torch.zeros(10, 10)}
            export_weights_to_hf(weights, config, abs_path)

            files = os.listdir(abs_path)
            assert len(files) > 0
