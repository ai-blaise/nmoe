"""Extended unit tests for nmoe unified interface.

This module provides comprehensive P3 tests covering:
1. Multi-backend switching (SGLang/SkyRL wrappers)
2. Config conversion edge cases
3. Weight format conversion
4. Interface compliance
5. Error handling

These tests extend the basic tests in test_interface.py with more complex
scenarios and edge cases.
"""

import pytest
import dataclasses
from abc import ABC
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_nmoe_config_dict():
    """Sample nmoe.config.Config-style dictionary."""
    return {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "inter_dim": 14336,
        "moe_inter_dim": 2048,
        "n_routed_experts": 64,
        "n_activated_experts": 8,
        "n_shared_experts": 2,
        "n_dense_layers": 1,
        "vocab_size": 201088,
        "max_position_embeddings": 8192,
        "rope_theta": 50000.0,
        "rope_scaling_factor": 1.0,
        "rope_ntk_alpha": 1.0,
        "rope_ntk_beta": 32.0,
        "rms_norm_eps": 1e-5,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "attn": "mla",
        "attn_local": "swa",
        "attn_global_every": 1,
        "attn_local_window": 128,
        "aux_loss_alpha": 0.001,
        "norm_topk_prob": True,
        "route_scale": 1.0,
        "router_bias_update_rate": 1e-4,
        "dtype": "bf16",
        "tokenizer": "o200k_harmony",
        "eos_token_id": 199999,
    }


@pytest.fixture
def minimal_config_dict():
    """Minimal valid config dict."""
    return {
        "dim": 512,
        "n_layers": 4,
        "n_heads": 8,
    }


@pytest.fixture
def moe_config_dict():
    """MoE-specific config dict."""
    return {
        "dim": 1024,
        "n_layers": 8,
        "n_heads": 16,
        "inter_dim": 4096,
        "moe_inter_dim": 1024,
        "n_routed_experts": 16,
        "n_activated_experts": 4,
        "n_shared_experts": 2,
        "n_dense_layers": 1,
    }


# =============================================================================
# Multi-Backend Switching Tests
# =============================================================================

class TestMultiBackendSwitching:
    """Tests for switching between SGLang and SkyRL wrappers."""

    def test_config_preserved_on_dict_roundtrip(self, sample_nmoe_config_dict):
        """Config values preserved through dict conversion."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        d = config.to_dict()
        restored = NMoEModelConfig.from_dict(d)

        assert restored.hidden_size == sample_nmoe_config_dict["dim"]
        assert restored.num_hidden_layers == sample_nmoe_config_dict["n_layers"]
        assert restored.num_attention_heads == sample_nmoe_config_dict["n_heads"]
        assert restored.num_experts == sample_nmoe_config_dict["n_routed_experts"]
        assert restored.num_experts_per_tok == sample_nmoe_config_dict["n_activated_experts"]

    def test_fingerprint_stable_across_conversions(self, sample_nmoe_config_dict):
        """Fingerprint remains stable through conversions."""
        from nmoe.unified.config import NMoEModelConfig

        config1 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        fp1 = config1.fingerprint()

        # Convert to dict and back
        d = config1.to_dict()
        config2 = NMoEModelConfig.from_dict(d)
        fp2 = config2.fingerprint()

        assert fp1 == fp2

    def test_hf_config_roundtrip_preserves_fields(self, sample_nmoe_config_dict):
        """HF config roundtrip preserves fields."""
        from nmoe.unified.config import NMoEModelConfig

        original = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        hf_dict = original.to_hf_config()
        restored = NMoEModelConfig.from_hf_config(hf_dict)

        assert original.hidden_size == restored.hidden_size
        assert original.num_hidden_layers == restored.num_hidden_layers
        assert original.num_attention_heads == restored.num_attention_heads
        assert original.vocab_size == restored.vocab_size

    def test_sglang_args_contain_moe_backend(self, moe_config_dict):
        """SGLang server args include MoE backend hint."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(moe_config_dict)
        args = config.to_sglang_server_args()

        assert "moe_runner_backend" in args
        assert args["moe_runner_backend"] == "nmoe"

    def test_config_copy_with_updates(self, sample_nmoe_config_dict):
        """Config copy allows field updates."""
        from nmoe.unified.config import NMoEModelConfig

        original = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        modified = original.copy(hidden_size=8192, num_hidden_layers=64)

        # Original unchanged
        assert original.hidden_size == sample_nmoe_config_dict["dim"]
        assert original.num_hidden_layers == sample_nmoe_config_dict["n_layers"]

        # Modified has new values
        assert modified.hidden_size == 8192
        assert modified.num_hidden_layers == 64

        # Other fields preserved
        assert modified.num_attention_heads == original.num_attention_heads
        assert modified.vocab_size == original.vocab_size


# =============================================================================
# Config Conversion Edge Cases
# =============================================================================

class TestConfigConversionEdgeCases:
    """Tests for config conversion edge cases."""

    def test_to_hf_config_with_all_fields(self, sample_nmoe_config_dict):
        """to_hf_config includes all required fields."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        hf_config = config.to_hf_config()

        # Core fields must be present
        assert "hidden_size" in hf_config
        assert "num_hidden_layers" in hf_config
        assert "num_attention_heads" in hf_config
        assert "vocab_size" in hf_config

        # MoE fields
        assert "n_routed_experts" in hf_config
        assert "num_experts_per_tok" in hf_config
        assert "n_shared_experts" in hf_config
        assert "first_k_dense_replace" in hf_config

        # MLA fields
        assert "q_lora_rank" in hf_config
        assert "kv_lora_rank" in hf_config
        assert "qk_nope_head_dim" in hf_config
        assert "qk_rope_head_dim" in hf_config
        assert "v_head_dim" in hf_config

        # RoPE fields
        assert "max_position_embeddings" in hf_config
        assert "rope_theta" in hf_config

        # Model identity
        assert "model_type" in hf_config
        assert "architectures" in hf_config

    def test_from_nmoe_config_with_optional_fields(self, minimal_config_dict):
        """from_nmoe_config handles missing optional fields."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(minimal_config_dict)

        # Required fields set
        assert config.hidden_size == minimal_config_dict["dim"]
        assert config.num_hidden_layers == minimal_config_dict["n_layers"]
        assert config.num_attention_heads == minimal_config_dict["n_heads"]

        # Optional fields have defaults
        assert config.vocab_size == 201088  # Default
        assert config.num_experts is None  # No MoE
        assert config.attention_type == "mla"  # Default
        assert config.torch_dtype == "bfloat16"  # Default

    def test_from_nmoe_config_fp8_quantization(self, sample_nmoe_config_dict):
        """FP8 dtype maps to quantization correctly."""
        from nmoe.unified.config import NMoEModelConfig

        sample_nmoe_config_dict["dtype"] = "fp8"
        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        assert config.quantization == "fp8"
        assert config.torch_dtype == "bfloat16"  # Base dtype
        assert config.dtype == "fp8"  # nmoe-style accessor

    def test_from_nmoe_config_nvfp4_quantization(self, sample_nmoe_config_dict):
        """NVFP4 dtype maps to quantization correctly."""
        from nmoe.unified.config import NMoEModelConfig

        sample_nmoe_config_dict["dtype"] = "nvfp4"
        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        assert config.quantization == "modelopt_fp4"
        assert config.torch_dtype == "bfloat16"  # Base dtype
        assert config.dtype == "nvfp4"  # nmoe-style accessor

    def test_to_sglang_server_args_with_ep_tp(self, moe_config_dict):
        """SGLang args include context length and dtype."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(moe_config_dict)
        args = config.to_sglang_server_args()

        assert "context_length" in args
        assert args["context_length"] == config.max_position_embeddings
        assert "dtype" in args

    def test_to_sglang_server_args_with_quantization(self, sample_nmoe_config_dict):
        """SGLang args include quantization when set."""
        from nmoe.unified.config import NMoEModelConfig

        sample_nmoe_config_dict["dtype"] = "fp8"
        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        args = config.to_sglang_server_args()

        assert "quantization" in args
        assert args["quantization"] == "fp8"

    def test_rope_scaling_conversion(self, sample_nmoe_config_dict):
        """Non-default rope_scaling_factor creates rope_scaling dict."""
        from nmoe.unified.config import NMoEModelConfig

        sample_nmoe_config_dict["rope_scaling_factor"] = 2.0
        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        assert config.rope_scaling is not None
        assert config.rope_scaling["type"] == "yarn"
        assert config.rope_scaling["factor"] == 2.0

    def test_rope_scaling_not_set_when_default(self, sample_nmoe_config_dict):
        """Default rope_scaling_factor doesn't create rope_scaling."""
        from nmoe.unified.config import NMoEModelConfig

        sample_nmoe_config_dict["rope_scaling_factor"] = 1.0
        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        assert config.rope_scaling is None

    def test_nmoe_aliases_work(self, sample_nmoe_config_dict):
        """nmoe-style property aliases work correctly."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        # Aliases should return same values as canonical names
        assert config.dim == config.hidden_size
        assert config.n_layers == config.num_hidden_layers
        assert config.n_heads == config.num_attention_heads
        assert config.inter_dim == config.intermediate_size
        assert config.moe_inter_dim == config.moe_intermediate_size
        assert config.n_routed_experts == config.num_experts
        assert config.n_activated_experts == config.num_experts_per_tok
        assert config.n_dense_layers == config.first_k_dense_replace
        assert config.attn == config.attention_type
        assert config.aux_loss_alpha == config.router_aux_loss_coef
        assert config.route_scale == config.routed_scaling_factor


# =============================================================================
# Weight Format Conversion Tests
# =============================================================================

class TestWeightFormatConversion:
    """Tests for weight format conversion functions."""

    def test_nmoe_to_hf_weight_mapping_basic(self):
        """Basic weight mapping returns valid dictionary."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1, n_routed_experts=8)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Check key patterns
        assert "embedding.weight" in mapping
        assert "lm_head.weight" in mapping
        assert "norm.weight" in mapping

    def test_nmoe_to_hf_weight_mapping_layer_coverage(self):
        """Mapping covers all layers."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        n_layers = 8
        mapping = nmoe_to_hf_weight_mapping(n_layers=n_layers, n_dense_layers=1, n_routed_experts=16)

        for layer_id in range(n_layers):
            # Each layer should have attention norm
            assert f"blocks.{layer_id}.attn_norm.weight" in mapping
            # Each layer should have ffn norm
            assert f"blocks.{layer_id}.ffn_norm.weight" in mapping

    def test_hf_to_nmoe_weight_mapping_inverse(self):
        """Inverse mapping is consistent with forward mapping."""
        from nmoe.tools.config_converter import (
            nmoe_to_hf_weight_mapping,
            hf_to_nmoe_weight_mapping,
        )

        n_layers = 4
        forward = nmoe_to_hf_weight_mapping(n_layers=n_layers, n_dense_layers=1, n_routed_experts=8)
        inverse = hf_to_nmoe_weight_mapping(n_layers=n_layers, n_dense_layers=1, n_routed_experts=8)

        # For non-expert weights (those without pattern expansion)
        for nmoe_key, hf_key in forward.items():
            if ".W1" not in nmoe_key and ".W2" not in nmoe_key and ".W3" not in nmoe_key:
                assert inverse.get(hf_key) == nmoe_key

    def test_expert_weight_expansion(self):
        """Expert weights expand correctly to HF format."""
        import torch
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        n_experts = 4
        dim = 128
        inter_dim = 256
        n_layers = 2
        n_dense_layers = 0

        # Create nmoe-style stacked expert weights
        nmoe_state_dict = {}
        for layer_id in range(n_layers):
            nmoe_state_dict[f"blocks.{layer_id}.ffn.W1"] = torch.randn(n_experts, dim, inter_dim)
            nmoe_state_dict[f"blocks.{layer_id}.ffn.W2"] = torch.randn(n_experts, inter_dim, dim)
            nmoe_state_dict[f"blocks.{layer_id}.ffn.W3"] = torch.randn(n_experts, dim, inter_dim)

        expanded = expand_expert_weights_to_hf(
            nmoe_state_dict,
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_experts=n_experts,
        )

        # Should have n_experts * 3 weight types * n_layers expanded keys
        expert_keys = [k for k in expanded.keys() if "experts" in k]
        expected_count = n_experts * 3 * n_layers
        assert len(expert_keys) == expected_count

        # Check specific expert key pattern
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in expanded
        assert "model.layers.0.mlp.experts.0.up_proj.weight" in expanded
        assert "model.layers.0.mlp.experts.0.down_proj.weight" in expanded

    def test_expert_weight_shapes_after_expansion(self):
        """Expanded expert weights have correct transposed shapes."""
        import torch
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        n_experts = 2
        dim = 64
        inter_dim = 128
        n_layers = 1
        n_dense_layers = 0

        nmoe_state_dict = {
            "blocks.0.ffn.W1": torch.randn(n_experts, dim, inter_dim),  # [E, dim, inter]
            "blocks.0.ffn.W2": torch.randn(n_experts, inter_dim, dim),  # [E, inter, dim]
            "blocks.0.ffn.W3": torch.randn(n_experts, dim, inter_dim),  # [E, dim, inter]
        }

        expanded = expand_expert_weights_to_hf(
            nmoe_state_dict,
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_experts=n_experts,
        )

        # W1 (gate_proj) and W3 (up_proj): [inter_dim, dim] after transpose
        assert expanded["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (inter_dim, dim)
        assert expanded["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (inter_dim, dim)

        # W2 (down_proj): [dim, inter_dim] after transpose
        assert expanded["model.layers.0.mlp.experts.0.down_proj.weight"].shape == (dim, inter_dim)

    def test_get_expert_weight_info_nmoe_format(self):
        """Parses nmoe expert weight names correctly."""
        from nmoe.tools.config_converter import get_expert_weight_info

        info = get_expert_weight_info("blocks.5.ffn.W1")

        assert info is not None
        assert info["layer_id"] == 5
        assert info["weight_type"] == "W1"
        assert info["format"] == "nmoe"

    def test_get_expert_weight_info_hf_format(self):
        """Parses HF expert weight names correctly."""
        from nmoe.tools.config_converter import get_expert_weight_info

        info = get_expert_weight_info("model.layers.3.mlp.experts.7.gate_proj.weight")

        assert info is not None
        assert info["layer_id"] == 3
        assert info["expert_id"] == 7
        assert info["weight_type"] == "W1"  # gate_proj -> W1
        assert info["format"] == "hf"

    def test_get_expert_weight_info_non_expert(self):
        """Returns None for non-expert weights."""
        from nmoe.tools.config_converter import get_expert_weight_info

        assert get_expert_weight_info("model.embed_tokens.weight") is None
        assert get_expert_weight_info("model.norm.weight") is None
        assert get_expert_weight_info("lm_head.weight") is None

    def test_weight_mapping_for_various_layer_counts(self):
        """Weight mapping works for various layer counts."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        for n_layers in [1, 4, 16, 32, 64]:
            mapping = nmoe_to_hf_weight_mapping(
                n_layers=n_layers,
                n_dense_layers=1,
                n_routed_experts=64,
            )

            # Should have per-layer keys
            assert f"blocks.0.attn_norm.weight" in mapping
            assert f"blocks.{n_layers - 1}.attn_norm.weight" in mapping


# =============================================================================
# Interface Compliance Tests
# =============================================================================

class TestInterfaceCompliance:
    """Tests for interface compliance verification."""

    def test_all_abstract_methods_defined(self):
        """NMoEModelInterface defines all required abstract methods."""
        from nmoe.unified.interface import NMoEModelInterface

        # Core methods
        assert hasattr(NMoEModelInterface, "forward")
        assert hasattr(NMoEModelInterface, "generate")
        assert hasattr(NMoEModelInterface, "forward_with_log_probs")

        # Expert cache
        assert hasattr(NMoEModelInterface, "refresh_expert_caches")
        assert hasattr(NMoEModelInterface, "uses_quantized_experts")

        # Load balancing
        assert hasattr(NMoEModelInterface, "get_router_aux_loss")
        assert hasattr(NMoEModelInterface, "get_expert_load_stats")
        assert hasattr(NMoEModelInterface, "update_router_biases")

        # Gradient checkpointing
        assert hasattr(NMoEModelInterface, "gradient_checkpointing_enable")
        assert hasattr(NMoEModelInterface, "gradient_checkpointing_disable")
        assert hasattr(NMoEModelInterface, "is_gradient_checkpointing")

        # Model properties
        assert hasattr(NMoEModelInterface, "config")
        assert hasattr(NMoEModelInterface, "device")
        assert hasattr(NMoEModelInterface, "dtype")

        # Parameter access
        assert hasattr(NMoEModelInterface, "param_sets")
        assert hasattr(NMoEModelInterface, "named_parameters_by_type")

        # State dict
        assert hasattr(NMoEModelInterface, "state_dict_for_save")
        assert hasattr(NMoEModelInterface, "load_state_dict_from_checkpoint")

    def test_forward_return_type_signature(self):
        """Forward method has correct return type annotation."""
        import inspect
        from nmoe.unified.interface import NMoEModelInterface

        sig = inspect.signature(NMoEModelInterface.forward)
        return_annotation = sig.return_annotation

        # Should return Dict[str, torch.Tensor]
        assert "Dict" in str(return_annotation)

    def test_generate_return_type_signature(self):
        """Generate method has correct return type annotation."""
        import inspect
        from nmoe.unified.interface import NMoEModelInterface

        sig = inspect.signature(NMoEModelInterface.generate)
        return_annotation = sig.return_annotation

        # Should return torch.Tensor
        assert "Tensor" in str(return_annotation)

    def test_forward_with_log_probs_return_type_signature(self):
        """forward_with_log_probs has correct return type annotation."""
        import inspect
        from nmoe.unified.interface import NMoEModelInterface

        sig = inspect.signature(NMoEModelInterface.forward_with_log_probs)
        return_annotation = sig.return_annotation

        # Should return Tuple[torch.Tensor, torch.Tensor]
        assert "Tuple" in str(return_annotation)

    def test_concrete_implementation_validates_interface(self):
        """Concrete implementation must implement all abstract methods."""
        from nmoe.unified.interface import NMoEModelInterface

        class IncompleteImpl(NMoEModelInterface):
            def forward(self, input_ids, **kwargs):
                return {}

        # Should raise TypeError on instantiation
        with pytest.raises(TypeError):
            IncompleteImpl()

    def test_complete_implementation_instantiates(self):
        """Complete implementation can be instantiated."""
        from nmoe.unified.interface import NMoEModelInterface

        class CompleteImpl(NMoEModelInterface):
            def __init__(self):
                self._gc = False

            def forward(self, input_ids, attention_mask=None, position_ids=None,
                       past_key_values=None, use_cache=False):
                return {"logits": None}

            def generate(self, input_ids, max_new_tokens=128, temperature=1.0,
                        top_p=1.0, top_k=0, do_sample=True, **kwargs):
                return input_ids

            def forward_with_log_probs(self, input_ids, attention_mask=None, action_ids=None):
                return (None, None)

            def refresh_expert_caches(self):
                pass

            @property
            def uses_quantized_experts(self):
                return False

            def get_router_aux_loss(self):
                return 0.0

            def get_expert_load_stats(self):
                return {}

            def update_router_biases(self, gamma=0.001):
                pass

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gc = True

            def gradient_checkpointing_disable(self):
                self._gc = False

            @property
            def is_gradient_checkpointing(self):
                return self._gc

            @property
            def config(self):
                return {}

            @property
            def device(self):
                return "cpu"

            @property
            def dtype(self):
                return "float32"

            def get_input_embeddings(self):
                return None

            def get_output_embeddings(self):
                return None

            def param_sets(self):
                return ([], [])

            def named_parameters_by_type(self):
                return {}

            def state_dict_for_save(self):
                return {}

            def load_state_dict_from_checkpoint(self, state_dict, strict=True):
                pass

        # Should not raise
        impl = CompleteImpl()
        assert impl is not None
        assert impl.forward([[1, 2, 3]]) == {"logits": None}

    def test_optional_methods_handle_none(self):
        """Optional method parameters handle None correctly."""
        from nmoe.unified.interface import NMoEModelInterface

        # Create complete implementation for testing
        class TestImpl(NMoEModelInterface):
            def __init__(self):
                self._gc = False

            def forward(self, input_ids, attention_mask=None, position_ids=None,
                       past_key_values=None, use_cache=False):
                return {
                    "logits": None,
                    "attention_mask_was": attention_mask,
                    "position_ids_was": position_ids,
                    "past_key_values_was": past_key_values,
                }

            def generate(self, input_ids, max_new_tokens=128, temperature=1.0,
                        top_p=1.0, top_k=0, do_sample=True, **kwargs):
                return input_ids

            def forward_with_log_probs(self, input_ids, attention_mask=None, action_ids=None):
                return (None, None)

            def refresh_expert_caches(self):
                pass

            @property
            def uses_quantized_experts(self):
                return False

            def get_router_aux_loss(self):
                return 0.0

            def get_expert_load_stats(self):
                return {}

            def update_router_biases(self, gamma=0.001):
                pass

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gc = True

            def gradient_checkpointing_disable(self):
                self._gc = False

            @property
            def is_gradient_checkpointing(self):
                return self._gc

            @property
            def config(self):
                return {}

            @property
            def device(self):
                return "cpu"

            @property
            def dtype(self):
                return "float32"

            def get_input_embeddings(self):
                return None

            def get_output_embeddings(self):
                return None

            def param_sets(self):
                return ([], [])

            def named_parameters_by_type(self):
                return {}

            def state_dict_for_save(self):
                return {}

            def load_state_dict_from_checkpoint(self, state_dict, strict=True):
                pass

        impl = TestImpl()

        # Call with all None optional params
        result = impl.forward([[1, 2, 3]])
        assert result["attention_mask_was"] is None
        assert result["position_ids_was"] is None
        assert result["past_key_values_was"] is None


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_config_raises_validation_error(self):
        """Invalid config raises ConfigValidationError."""
        from nmoe.unified.config import NMoEModelConfig, ConfigValidationError

        # Create config with missing required fields
        config = NMoEModelConfig()

        with pytest.raises(ConfigValidationError):
            config.validate()

    def test_moe_config_missing_fields_raises_error(self):
        """MoE config missing required fields raises error."""
        from nmoe.unified.config import NMoEModelConfig, ConfigValidationError

        # Create config with num_experts but missing num_experts_per_tok
        config = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_experts=64,  # Set MoE
            num_experts_per_tok=None,  # Missing!
            moe_intermediate_size=None,  # Missing!
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            config.validate()

        assert "num_experts_per_tok" in str(exc_info.value) or "moe_intermediate_size" in str(exc_info.value)

    def test_from_nmoe_config_invalid_type_raises_error(self):
        """from_nmoe_config raises TypeError for invalid input."""
        from nmoe.unified.config import NMoEModelConfig

        with pytest.raises(TypeError):
            NMoEModelConfig.from_nmoe_config(12345)  # Invalid type

    def test_from_dict_ignores_unknown_fields(self):
        """from_dict ignores unknown fields gracefully."""
        from nmoe.unified.config import NMoEModelConfig

        d = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "unknown_field_xyz": "should be ignored",
            "another_unknown": 12345,
        }

        # Should not raise
        config = NMoEModelConfig.from_dict(d)
        assert config.hidden_size == 4096
        assert not hasattr(config, "unknown_field_xyz")

    def test_validate_weight_mapping_detects_mismatches(self):
        """validate_weight_mapping detects mismatched keys."""
        from nmoe.tools.config_converter import validate_weight_mapping

        nmoe_keys = ["embedding.weight", "lm_head.weight", "blocks.0.attn_norm.weight"]
        hf_keys = []  # Empty - all missing

        matched, missing, extra = validate_weight_mapping(
            nmoe_keys=nmoe_keys,
            hf_keys=hf_keys,
            n_layers=1,
            n_dense_layers=0,
            n_experts=8,
            n_shared_experts=2,
        )

        # Should detect missing mappings
        assert len(missing) > 0 or len(matched) < len(nmoe_keys)

    def test_config_validation_valid_config_passes(self, sample_nmoe_config_dict):
        """Valid config passes validation."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        # Should not raise
        config.validate()


# =============================================================================
# Config Roundtrip Tests
# =============================================================================

class TestConfigRoundtrip:
    """Tests for config roundtrip conversions."""

    def test_nmoe_to_hf_to_nmoe_roundtrip(self, sample_nmoe_config_dict):
        """Config survives nmoe -> HF -> nmoe roundtrip."""
        from nmoe.unified.config import NMoEModelConfig

        # nmoe -> unified
        unified1 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        # unified -> HF
        hf_dict = unified1.to_hf_config()

        # HF -> unified
        unified2 = NMoEModelConfig.from_hf_config(hf_dict)

        # Check core fields preserved
        assert unified1.hidden_size == unified2.hidden_size
        assert unified1.num_hidden_layers == unified2.num_hidden_layers
        assert unified1.num_attention_heads == unified2.num_attention_heads
        assert unified1.vocab_size == unified2.vocab_size

    def test_dict_roundtrip_preserves_all_fields(self, sample_nmoe_config_dict):
        """dict -> config -> dict roundtrip preserves all fields."""
        from nmoe.unified.config import NMoEModelConfig

        config1 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        d1 = config1.to_dict()
        config2 = NMoEModelConfig.from_dict(d1)
        d2 = config2.to_dict()

        # All keys should match
        assert set(d1.keys()) == set(d2.keys())

        # All values should match
        for key in d1:
            assert d1[key] == d2[key], f"Mismatch for {key}: {d1[key]} != {d2[key]}"

    def test_fingerprint_deterministic(self, sample_nmoe_config_dict):
        """Fingerprint is deterministic for same config."""
        from nmoe.unified.config import NMoEModelConfig

        config1 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        config2 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        assert config1.fingerprint() == config2.fingerprint()

    def test_fingerprint_changes_with_config(self, sample_nmoe_config_dict):
        """Fingerprint changes when config changes."""
        from nmoe.unified.config import NMoEModelConfig

        config1 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        sample_nmoe_config_dict["dim"] = 8192  # Change a field
        config2 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        assert config1.fingerprint() != config2.fingerprint()


# =============================================================================
# RDEP Config Tests
# =============================================================================

class TestRDEPConfig:
    """Tests for NMoERDEPConfig."""

    def test_rdep_config_defaults(self):
        """RDEP config has sensible defaults."""
        from nmoe.unified.config import NMoERDEPConfig

        config = NMoERDEPConfig()

        assert config.mode == "auto"
        assert config.profile == "bf16"
        assert config.capacity == 65536
        assert config.nvshmem_enabled is False

    def test_rdep_profile_id_mapping(self):
        """Profile ID maps correctly."""
        from nmoe.unified.config import NMoERDEPConfig

        bf16_config = NMoERDEPConfig(profile="bf16")
        assert bf16_config.get_profile_id() == -1

        fp8_config = NMoERDEPConfig(profile="fp8")
        assert fp8_config.get_profile_id() == 0

        nvfp4_config = NMoERDEPConfig(profile="nvfp4")
        assert nvfp4_config.get_profile_id() == 1

    def test_rdep_mode_detection_single(self):
        """Auto mode detects single GPU."""
        from nmoe.unified.config import NMoERDEPConfig

        config = NMoERDEPConfig(mode="auto")
        detected = config.detect_mode(world_size=1, local_world_size=1)

        assert detected == "single"

    def test_rdep_mode_detection_ipc(self):
        """Auto mode detects IPC (single node multi-GPU)."""
        from nmoe.unified.config import NMoERDEPConfig

        config = NMoERDEPConfig(mode="auto")
        detected = config.detect_mode(world_size=8, local_world_size=8)

        assert detected == "ipc"

    def test_rdep_mode_detection_hybrid(self):
        """Auto mode detects hybrid (multi-node)."""
        from nmoe.unified.config import NMoERDEPConfig

        config = NMoERDEPConfig(mode="auto")
        detected = config.detect_mode(world_size=16, local_world_size=8)

        assert detected == "hybrid"

    def test_rdep_mode_override(self):
        """Explicit mode overrides auto detection."""
        from nmoe.unified.config import NMoERDEPConfig

        config = NMoERDEPConfig(mode="single")
        detected = config.detect_mode(world_size=16, local_world_size=8)

        assert detected == "single"  # Overridden, not hybrid

    def test_rdep_config_to_dict(self):
        """RDEP config serializes to dict."""
        from nmoe.unified.config import NMoERDEPConfig

        config = NMoERDEPConfig(mode="ipc", profile="fp8", capacity=32768)
        d = config.to_dict()

        assert d["mode"] == "ipc"
        assert d["profile"] == "fp8"
        assert d["capacity"] == 32768

    def test_rdep_config_from_dict(self):
        """RDEP config deserializes from dict."""
        from nmoe.unified.config import NMoERDEPConfig

        d = {"mode": "hybrid", "profile": "nvfp4", "nvshmem_enabled": True}
        config = NMoERDEPConfig.from_dict(d)

        assert config.mode == "hybrid"
        assert config.profile == "nvfp4"
        assert config.nvshmem_enabled is True

    def test_rdep_config_fingerprint(self):
        """RDEP config has deterministic fingerprint."""
        from nmoe.unified.config import NMoERDEPConfig

        config1 = NMoERDEPConfig(mode="ipc", profile="fp8")
        config2 = NMoERDEPConfig(mode="ipc", profile="fp8")

        assert config1.fingerprint() == config2.fingerprint()


# =============================================================================
# Properties and Computed Fields Tests
# =============================================================================

class TestPropertiesAndComputedFields:
    """Tests for config properties and computed fields."""

    def test_is_moe_property(self, moe_config_dict, minimal_config_dict):
        """is_moe property correctly identifies MoE models."""
        from nmoe.unified.config import NMoEModelConfig

        moe_config = NMoEModelConfig.from_nmoe_config(moe_config_dict)
        assert moe_config.is_moe is True

        dense_config = NMoEModelConfig.from_nmoe_config(minimal_config_dict)
        assert dense_config.is_moe is False

    def test_total_experts_property(self, moe_config_dict):
        """total_experts includes shared experts."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(moe_config_dict)

        expected_total = moe_config_dict["n_routed_experts"] + moe_config_dict["n_shared_experts"]
        assert config.total_experts == expected_total

    def test_total_experts_zero_for_dense(self, minimal_config_dict):
        """total_experts is 0 for dense models."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(minimal_config_dict)
        assert config.total_experts == 0

    def test_head_dim_property(self, sample_nmoe_config_dict):
        """head_dim is sum of nope and rope dims."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        expected = sample_nmoe_config_dict["qk_nope_head_dim"] + sample_nmoe_config_dict["qk_rope_head_dim"]
        assert config.head_dim == expected

    def test_pad_token_id_defaults_to_eos(self, sample_nmoe_config_dict):
        """pad_token_id defaults to eos_token_id."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        assert config.pad_token_id == config.eos_token_id


# =============================================================================
# Integration with nmoe.config Tests
# =============================================================================

class TestIntegrationWithNMoEConfig:
    """Tests for integration with nmoe.config module."""

    def test_validate_config_mapping_passes_for_valid(self, sample_nmoe_config_dict):
        """validate_config_mapping passes for valid mapping."""
        from nmoe.config import validate_config_mapping, Config

        # Create dataclass-style config
        @dataclass
        class MockConfig:
            dim: int
            n_layers: int
            n_heads: int
            vocab_size: int
            max_position_embeddings: int
            rms_norm_eps: float
            rope_theta: float
            n_routed_experts: int
            n_activated_experts: int
            inter_dim: int
            moe_inter_dim: int

        @dataclass
        class MockHFConfig:
            hidden_size: int
            num_hidden_layers: int
            num_attention_heads: int
            vocab_size: int
            max_position_embeddings: int
            rms_norm_eps: float
            rope_theta: float
            num_local_experts: int
            num_experts_per_tok: int
            intermediate_size: int
            moe_intermediate_size: int

        nmoe_cfg = MockConfig(
            dim=4096,
            n_layers=32,
            n_heads=32,
            vocab_size=201088,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            rope_theta=50000.0,
            n_routed_experts=64,
            n_activated_experts=8,
            inter_dim=14336,
            moe_inter_dim=2048,
        )

        hf_cfg = MockHFConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            vocab_size=201088,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            rope_theta=50000.0,
            num_local_experts=64,
            num_experts_per_tok=8,
            intermediate_size=14336,
            moe_intermediate_size=2048,
        )

        # Should not raise
        validate_config_mapping(nmoe_cfg, hf_cfg)

    def test_validate_config_mapping_fails_for_mismatch(self):
        """validate_config_mapping raises for mismatched values."""
        from nmoe.config import validate_config_mapping

        @dataclass
        class MockNMoE:
            dim: int = 4096

        @dataclass
        class MockHF:
            hidden_size: int = 8192  # Mismatch!

        nmoe_cfg = MockNMoE()
        hf_cfg = MockHF()

        with pytest.raises(ValueError) as exc_info:
            validate_config_mapping(nmoe_cfg, hf_cfg)

        assert "mismatch" in str(exc_info.value).lower()

    def test_nmoe_to_hf_config_function(self, sample_nmoe_config_dict):
        """nmoe_to_hf_config function converts correctly."""
        from nmoe.config import nmoe_to_hf_config, Config

        # Create actual Config instance
        cfg = Config(**{k: v for k, v in sample_nmoe_config_dict.items()
                       if hasattr(Config, k)})

        hf_dict = nmoe_to_hf_config(cfg)

        # Check key mappings
        assert hf_dict.get("hidden_size") == cfg.dim
        assert hf_dict.get("num_hidden_layers") == cfg.n_layers


# =============================================================================
# State Preservation Tests
# =============================================================================

class TestStatePreservation:
    """Tests for state preservation across operations."""

    def test_config_immutable_after_creation(self, sample_nmoe_config_dict):
        """Config fields can be modified but don't affect original dict."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        original_dim = config.hidden_size

        # Modify config
        config.hidden_size = 8192

        # Original dict unchanged
        assert sample_nmoe_config_dict["dim"] == original_dim
        assert sample_nmoe_config_dict["dim"] != 8192

    def test_copy_creates_independent_instance(self, sample_nmoe_config_dict):
        """copy() creates fully independent instance."""
        from nmoe.unified.config import NMoEModelConfig

        config1 = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)
        config2 = config1.copy()

        # Modify copy
        config2.hidden_size = 8192

        # Original unchanged
        assert config1.hidden_size == sample_nmoe_config_dict["dim"]
        assert config1.hidden_size != config2.hidden_size

    def test_to_dict_creates_new_dict(self, sample_nmoe_config_dict):
        """to_dict() creates new dictionary each time."""
        from nmoe.unified.config import NMoEModelConfig

        config = NMoEModelConfig.from_nmoe_config(sample_nmoe_config_dict)

        d1 = config.to_dict()
        d2 = config.to_dict()

        # Different objects
        assert d1 is not d2

        # Modifying one doesn't affect other
        d1["hidden_size"] = 8192
        assert d2["hidden_size"] == sample_nmoe_config_dict["dim"]
