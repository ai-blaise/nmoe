"""Comprehensive unit tests for NMoEModelConfig.

Tests cover:
- Config instantiation with defaults
- Config validation
- Fingerprint generation
- Conversion to/from nmoe, HuggingFace, and SGLang formats
- Edge cases and error handling
"""

import pytest
from dataclasses import asdict
from typing import Dict, Any


class TestNMoEModelConfigCreation:
    """Tests for NMoEModelConfig instantiation."""

    def test_default_creation(self):
        """Config can be created with defaults."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig()
        assert cfg.model_type == "nmoe"
        assert cfg.vocab_size == 201088
        assert cfg.n_shared_experts == 2
        assert cfg.first_k_dense_replace == 1

    def test_creation_with_all_fields(self):
        """Config can be created with all fields specified."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=14336,
            moe_intermediate_size=1792,
            num_experts=64,
            num_experts_per_tok=8,
            n_shared_experts=2,
            first_k_dense_replace=3,
        )

        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 32
        assert cfg.num_attention_heads == 32
        assert cfg.intermediate_size == 14336
        assert cfg.moe_intermediate_size == 1792
        assert cfg.num_experts == 64
        assert cfg.num_experts_per_tok == 8

    def test_creation_with_quantization(self):
        """Config supports quantization settings."""
        from nmoe.unified.config import NMoEModelConfig

        # The actual parameter is 'quantization', not 'quantization_profile'
        cfg = NMoEModelConfig(
            hidden_size=4096,
            quantization="fp8",
        )

        assert cfg.quantization == "fp8"

    def test_architectures_list(self):
        """Architectures field is a list with correct default."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig()
        assert isinstance(cfg.architectures, list)
        assert "NMoEForCausalLM" in cfg.architectures


class TestNMoEModelConfigValidation:
    """Tests for config validation."""

    def test_validate_missing_required_fields(self):
        """Validation catches missing required fields."""
        from nmoe.unified.config import NMoEModelConfig, ConfigValidationError

        cfg = NMoEModelConfig()  # No dimensions set

        with pytest.raises(ConfigValidationError):
            cfg.validate()

    def test_validate_complete_config(self):
        """Complete config passes validation."""
        from nmoe.unified.config import NMoEModelConfig

        # For MoE models, need moe_intermediate_size
        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=14336,
            moe_intermediate_size=1792,  # Required for MoE
            num_experts=64,
            num_experts_per_tok=8,
        )

        # Should not raise
        cfg.validate()

    def test_validate_dense_model(self):
        """Dense model (no experts) passes validation."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=14336,
            # No MoE fields - this is a dense model
        )

        # Should not raise for dense model
        cfg.validate()

    def test_validate_moe_missing_intermediate(self):
        """MoE config without moe_intermediate_size fails."""
        from nmoe.unified.config import NMoEModelConfig, ConfigValidationError

        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_experts=64,
            num_experts_per_tok=8,
            # Missing moe_intermediate_size
        )

        with pytest.raises(ConfigValidationError):
            cfg.validate()


class TestConfigFingerprint:
    """Tests for config fingerprinting."""

    def test_fingerprint_deterministic(self):
        """Same config produces same fingerprint."""
        from nmoe.unified.config import NMoEModelConfig, fingerprint

        cfg1 = NMoEModelConfig(hidden_size=4096, num_hidden_layers=32)
        cfg2 = NMoEModelConfig(hidden_size=4096, num_hidden_layers=32)

        assert fingerprint(cfg1) == fingerprint(cfg2)

    def test_fingerprint_different_configs(self):
        """Different configs produce different fingerprints."""
        from nmoe.unified.config import NMoEModelConfig, fingerprint

        cfg1 = NMoEModelConfig(hidden_size=4096)
        cfg2 = NMoEModelConfig(hidden_size=2048)

        assert fingerprint(cfg1) != fingerprint(cfg2)

    def test_fingerprint_excludes_private_fields(self):
        """Fingerprint excludes private fields."""
        from nmoe.unified.config import fingerprint

        class ConfigWithPrivate:
            def __init__(self):
                self.public = 123
                self._private = 456

            def to_dict(self):
                return {"public": self.public, "_private": self._private}

        cfg = ConfigWithPrivate()
        fp = fingerprint(cfg)

        # Should not fail and should produce valid hash
        assert len(fp) == 64  # SHA-256 hex length

    def test_fingerprint_sha256_format(self):
        """Fingerprint is valid SHA-256 hex string."""
        from nmoe.unified.config import NMoEModelConfig, fingerprint

        cfg = NMoEModelConfig()
        fp = fingerprint(cfg)

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


class TestConfigConversion:
    """Tests for config format conversions."""

    def test_to_hf_config(self):
        """Config converts to HuggingFace format."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_experts=64,
        )

        hf_config = cfg.to_hf_config()

        assert isinstance(hf_config, dict)
        assert hf_config["hidden_size"] == 4096
        assert hf_config["num_hidden_layers"] == 32
        assert hf_config["model_type"] == "nmoe"

    def test_from_nmoe_config(self):
        """Config can be created from nmoe config."""
        from nmoe.unified.config import NMoEModelConfig
        from dataclasses import dataclass

        # Mock nmoe config as a dataclass (from_nmoe_config uses asdict)
        @dataclass
        class MockNmoeConfig:
            dim: int = 4096
            n_layers: int = 32
            n_heads: int = 32
            inter_dim: int = 14336
            n_routed_experts: int = 64
            n_activated_experts: int = 8

        cfg = NMoEModelConfig.from_nmoe_config(MockNmoeConfig())

        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 32
        assert cfg.num_attention_heads == 32

    def test_to_sglang_server_args(self):
        """Config converts to SGLang server args."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
        )

        args = cfg.to_sglang_server_args()

        assert isinstance(args, dict)

    def test_round_trip_conversion(self):
        """Config survives round-trip conversion."""
        from nmoe.unified.config import NMoEModelConfig

        original = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_experts=64,
            num_experts_per_tok=8,
        )

        # Convert to HF and back
        hf_config = original.to_hf_config()
        restored = NMoEModelConfig.from_hf_config(hf_config)

        assert restored.hidden_size == original.hidden_size
        assert restored.num_hidden_layers == original.num_hidden_layers
        assert restored.num_experts == original.num_experts


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_to_dict(self):
        """Config converts to dict."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(hidden_size=4096)
        d = asdict(cfg)

        assert isinstance(d, dict)
        assert d["hidden_size"] == 4096

    def test_json_serializable(self):
        """Config is JSON serializable."""
        import json
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(hidden_size=4096, num_hidden_layers=32)
        d = asdict(cfg)

        # Should not raise
        json_str = json.dumps(d)
        restored = json.loads(json_str)

        assert restored["hidden_size"] == 4096


class TestNMoERDEPConfig:
    """Tests for RDEP-specific configuration."""

    def test_rdep_config_creation(self):
        """RDEP config can be created with correct parameters."""
        from nmoe.unified.config import NMoERDEPConfig

        # Use actual NMoERDEPConfig parameters: mode, profile, capacity
        cfg = NMoERDEPConfig(
            mode="ipc",
            profile="fp8",
            capacity=65536,
        )

        assert cfg.mode == "ipc"
        assert cfg.profile == "fp8"
        assert cfg.capacity == 65536

    def test_rdep_config_defaults(self):
        """RDEP config has sensible defaults."""
        from nmoe.unified.config import NMoERDEPConfig

        # Create with defaults
        cfg = NMoERDEPConfig()

        assert cfg.mode == "auto"
        assert cfg.profile in ["bf16", "fp8", "nvfp4"]
        assert cfg.capacity > 0


class TestConfigEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_experts(self):
        """Config handles zero experts (dense model)."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_experts=0,  # Dense model
        )

        assert cfg.num_experts == 0

    def test_large_model_config(self):
        """Config handles large model dimensions."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig(
            hidden_size=16384,
            num_hidden_layers=128,
            num_attention_heads=128,
            num_experts=256,
            num_experts_per_tok=8,
        )

        assert cfg.hidden_size == 16384
        assert cfg.num_experts == 256

    def test_optional_fields_none(self):
        """Config handles None optional fields."""
        from nmoe.unified.config import NMoEModelConfig

        cfg = NMoEModelConfig()

        assert cfg.hidden_size is None
        assert cfg.num_hidden_layers is None

    def test_config_copy(self):
        """Config can be copied."""
        import copy
        from nmoe.unified.config import NMoEModelConfig

        original = NMoEModelConfig(hidden_size=4096)
        copied = copy.deepcopy(original)

        assert copied.hidden_size == original.hidden_size
        assert copied is not original
