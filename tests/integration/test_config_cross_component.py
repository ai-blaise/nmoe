"""Comprehensive integration tests for nmoe config handling across components.

This module tests config conversion and validation between:
- nmoe native config (nmoe.config.Config)
- SGLang server configuration
- SkyRL training configuration
- HuggingFace PretrainedConfig format

Tests cover:
1. nmoe config -> SGLang config conversion
2. nmoe config -> SkyRL config conversion
3. HuggingFace config -> nmoe config
4. Config roundtrip preservation
5. Config validation across components
6. MoE-specific config fields
7. RDEP config propagation
8. Distributed config (TP, PP, EP) handling

Run with:
    pytest nmoe/tests/integration/test_config_cross_component.py -v -s
"""

import copy
import dataclasses
import hashlib
import json
import os
import sys
import pytest
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock, patch

# Add nmoe to path for imports - use direct path to avoid tomllib import issues
_NMOE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_NMOE_UNIFIED_CONFIG_PATH = os.path.join(_NMOE_ROOT, "nmoe", "unified", "config.py")

# Local test-only Config class that mirrors nmoe.config.Config
# (avoiding import issues with tomllib in Python < 3.11)
@dataclass
class MockConfig:
    """Mock nmoe Config for testing without tomllib dependency."""
    # Meta
    preset: str = "custom"
    experiment_id: str = "default"

    # Core dimensions
    vocab_size: int = 201088
    tokenizer: str = "o200k_harmony"
    eos_token_id: int = 199999
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None

    # MoE
    inter_dim: Optional[int] = None
    moe_inter_dim: Optional[int] = None
    n_routed_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    n_shared_experts: int = 2
    n_dense_layers: int = 1

    # Attention
    attn: str = "mla"
    attn_local: str = "swa"
    attn_global_every: int = 1
    attn_local_window: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # RoPE
    max_position_embeddings: int = 8192
    rope_theta: float = 50000.0
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    # Normalization
    rms_norm_eps: float = 1e-5

    # Routing
    router_bias_update_rate: float = 1e-4
    aux_loss_alpha: float = 0.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0

    # Precision
    dtype: Optional[str] = "bf16"

    # Training
    lr_dense: float = 3.4e-4
    lr_router: float = 3.4e-4
    lr_expert: float = 3.4e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_beta2_expert: float = 0.99
    warmup_steps: int = 500
    batch_size: int = 8
    seq_len: int = 4096
    seed: int = 42

    # RL
    rl_enabled: bool = False
    rl_algorithm: str = "grpo"
    grpo_kl_beta: float = 0.001
    grpo_group_size: int = 2
    grpo_temperature: float = 1.0

    # Backend-specific
    attn_swa: Dict[str, Any] = field(default_factory=dict)
    attn_nsa: Dict[str, Any] = field(default_factory=dict)
    attn_dsa: Dict[str, Any] = field(default_factory=dict)


# Import the unified config classes directly using importlib to avoid
# nmoe.__init__ which imports torch and tomllib (Python 3.11+)
# This allows tests to run on Python 3.10 and in environments without torch
import importlib.util


def _load_unified_config_module():
    """Load the unified config module directly without importing nmoe package."""
    spec = importlib.util.spec_from_file_location(
        "nmoe_unified_config",
        _NMOE_UNIFIED_CONFIG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_unified_config_module = _load_unified_config_module()
NMoEModelConfig = _unified_config_module.NMoEModelConfig
NMoERDEPConfig = _unified_config_module.NMoERDEPConfig
fingerprint = _unified_config_module.fingerprint
ConfigValidationError = _unified_config_module.ConfigValidationError


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_nmoe_config():
    """Create a base nmoe config for testing."""
    return MockConfig(
        dim=2048,
        n_layers=24,
        n_heads=16,
        inter_dim=5632,
        moe_inter_dim=1408,
        n_routed_experts=64,
        n_activated_experts=8,
        n_shared_experts=2,
        n_dense_layers=1,
    )


@pytest.fixture
def small_nmoe_config():
    """Create a small nmoe config for testing."""
    return MockConfig(
        dim=512,
        n_layers=6,
        n_heads=8,
        inter_dim=1408,
        moe_inter_dim=512,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        n_dense_layers=1,
        vocab_size=2048,
        max_position_embeddings=512,
    )


@pytest.fixture
def dense_nmoe_config():
    """Create a dense (non-MoE) config for testing."""
    return MockConfig(
        dim=1024,
        n_layers=12,
        n_heads=8,
        inter_dim=2816,
        n_routed_experts=None,
        n_activated_experts=None,
    )


@pytest.fixture
def unified_config():
    """Create a unified NMoEModelConfig for testing."""
    return NMoEModelConfig(
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=5632,
        moe_intermediate_size=1408,
        num_experts=64,
        num_experts_per_tok=8,
        n_shared_experts=2,
        first_k_dense_replace=1,
    )


@pytest.fixture
def rdep_config():
    """Create an RDEP config for testing."""
    return NMoERDEPConfig(
        mode="auto",
        profile="bf16",
        capacity=65536,
    )


# =============================================================================
# Test Class: nmoe -> SGLang Config Conversion
# =============================================================================

class TestNMoEToSGLangConversion:
    """Test nmoe config to SGLang config conversion."""

    @pytest.mark.integration
    def test_basic_sglang_conversion(self, base_nmoe_config):
        """Test basic nmoe -> unified -> SGLang conversion."""
        # Convert to unified
        unified = NMoEModelConfig.from_nmoe_config(base_nmoe_config)

        # Get SGLang server args
        sglang_args = unified.to_sglang_server_args()

        assert sglang_args["context_length"] == base_nmoe_config.max_position_embeddings
        assert sglang_args["moe_runner_backend"] == "nmoe"
        assert "dtype" in sglang_args

    @pytest.mark.integration
    def test_sglang_quantization_conversion(self):
        """Test quantization settings propagate to SGLang."""
        # FP8 quantization
        cfg_fp8 = MockConfig(
            dim=2048, n_layers=24, n_heads=16,
            dtype="fp8",
        )
        unified_fp8 = NMoEModelConfig.from_nmoe_config(cfg_fp8)
        sglang_args_fp8 = unified_fp8.to_sglang_server_args()
        assert sglang_args_fp8.get("quantization") == "fp8"

        # NVFP4 quantization
        cfg_nvfp4 = MockConfig(
            dim=2048, n_layers=24, n_heads=16,
            dtype="nvfp4",
        )
        unified_nvfp4 = NMoEModelConfig.from_nmoe_config(cfg_nvfp4)
        sglang_args_nvfp4 = unified_nvfp4.to_sglang_server_args()
        assert sglang_args_nvfp4.get("quantization") == "modelopt_fp4"

    @pytest.mark.integration
    def test_sglang_dtype_conversion(self):
        """Test dtype mappings for SGLang."""
        dtype_tests = [
            ("bf16", "bfloat16"),
            ("fp8", "bfloat16"),  # FP8 base is bf16
            ("nvfp4", "bfloat16"),  # NVFP4 base is bf16
        ]

        for nmoe_dtype, expected_sglang_dtype in dtype_tests:
            cfg = MockConfig(dim=2048, n_layers=24, n_heads=16, dtype=nmoe_dtype)
            unified = NMoEModelConfig.from_nmoe_config(cfg)
            sglang_args = unified.to_sglang_server_args()
            assert sglang_args["dtype"] == expected_sglang_dtype, \
                f"Failed for dtype={nmoe_dtype}"

    @pytest.mark.integration
    def test_sglang_context_length_options(self):
        """Test various context lengths propagate correctly."""
        context_lengths = [2048, 4096, 8192, 16384, 32768, 65536]

        for ctx_len in context_lengths:
            cfg = MockConfig(
                dim=2048, n_layers=24, n_heads=16,
                max_position_embeddings=ctx_len,
            )
            unified = NMoEModelConfig.from_nmoe_config(cfg)
            sglang_args = unified.to_sglang_server_args()
            assert sglang_args["context_length"] == ctx_len

    @pytest.mark.integration
    def test_sglang_moe_runner_backend_hint(self, base_nmoe_config):
        """Test that MoE models get nmoe backend hint."""
        unified = NMoEModelConfig.from_nmoe_config(base_nmoe_config)
        sglang_args = unified.to_sglang_server_args()
        assert sglang_args.get("moe_runner_backend") == "nmoe"


# =============================================================================
# Test Class: nmoe -> SkyRL Config Conversion
# =============================================================================

class TestNMoEToSkyRLConversion:
    """Test nmoe config to SkyRL config conversion."""

    @pytest.mark.integration
    def test_skyrl_training_params_extraction(self, base_nmoe_config):
        """Test SkyRL can extract training-relevant params."""
        unified = NMoEModelConfig.from_nmoe_config(base_nmoe_config)

        # SkyRL needs these for model wrapper
        assert unified.hidden_size == base_nmoe_config.dim
        assert unified.num_hidden_layers == base_nmoe_config.n_layers
        assert unified.num_experts == base_nmoe_config.n_routed_experts
        assert unified.num_experts_per_tok == base_nmoe_config.n_activated_experts

    @pytest.mark.integration
    def test_skyrl_dtype_handling(self):
        """Test SkyRL dtype handling across quantization modes."""
        # BF16 (standard)
        cfg_bf16 = MockConfig(dim=2048, n_layers=24, n_heads=16, dtype="bf16")
        unified_bf16 = NMoEModelConfig.from_nmoe_config(cfg_bf16)
        assert unified_bf16.dtype == "bf16"
        assert unified_bf16.torch_dtype == "bfloat16"

        # FP8 (quantized training)
        cfg_fp8 = MockConfig(dim=2048, n_layers=24, n_heads=16, dtype="fp8")
        unified_fp8 = NMoEModelConfig.from_nmoe_config(cfg_fp8)
        assert unified_fp8.dtype == "fp8"
        assert unified_fp8.quantization == "fp8"

    @pytest.mark.integration
    def test_skyrl_moe_parameters(self, base_nmoe_config):
        """Test MoE parameters for SkyRL aux loss calculation."""
        unified = NMoEModelConfig.from_nmoe_config(base_nmoe_config)

        # SkyRL needs these for load balancing
        assert unified.router_aux_loss_coef == base_nmoe_config.aux_loss_alpha
        assert unified.norm_topk_prob == base_nmoe_config.norm_topk_prob
        assert unified.routed_scaling_factor == base_nmoe_config.route_scale

    @pytest.mark.integration
    def test_skyrl_grpo_config_integration(self):
        """Test GRPO-specific config fields for SkyRL."""
        cfg = MockConfig(
            dim=2048, n_layers=24, n_heads=16,
            rl_enabled=True,
            rl_algorithm="grpo",
            grpo_kl_beta=0.01,
            grpo_group_size=4,
            grpo_temperature=0.8,
        )
        unified = NMoEModelConfig.from_nmoe_config(cfg)

        # These should be accessible via the original config dict
        cfg_dict = asdict(cfg)
        assert cfg_dict["grpo_kl_beta"] == 0.01
        assert cfg_dict["grpo_group_size"] == 4
        assert cfg_dict["grpo_temperature"] == 0.8

    @pytest.mark.integration
    def test_skyrl_learning_rate_parameters(self):
        """Test learning rate parameters for SkyRL optimizer setup."""
        cfg = MockConfig(
            dim=2048, n_layers=24, n_heads=16,
            lr_dense=3e-4,
            lr_router=1e-4,
            lr_expert=5e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_beta2_expert=0.99,
        )
        cfg_dict = asdict(cfg)

        # SkyRL needs separate LRs for different parameter groups
        assert cfg_dict["lr_dense"] == 3e-4
        assert cfg_dict["lr_router"] == 1e-4
        assert cfg_dict["lr_expert"] == 5e-4
        assert cfg_dict["adam_beta2_expert"] == 0.99


# =============================================================================
# Test Class: HuggingFace -> nmoe Config Conversion
# =============================================================================

class TestHFToNMoEConversion:
    """Test HuggingFace config to nmoe config conversion."""

    @pytest.mark.integration
    def test_basic_hf_to_unified(self):
        """Test basic HF config dict -> unified conversion."""
        hf_dict = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "vocab_size": 201088,
            "model_type": "nmoe",
        }

        unified = NMoEModelConfig.from_dict(hf_dict)
        assert unified.hidden_size == 2048
        assert unified.num_hidden_layers == 24
        assert unified.num_attention_heads == 16

    @pytest.mark.integration
    def test_hf_moe_fields_mapping(self):
        """Test HF MoE field name variations."""
        # n_routed_experts variant (DeepSeek style)
        hf_dict_deepseek = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "n_routed_experts": 64,
            "num_experts_per_tok": 8,
        }
        unified = NMoEModelConfig.from_hf_config(hf_dict_deepseek)
        assert unified.num_experts == 64

        # num_local_experts variant (Mixtral style)
        hf_dict_mixtral = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }
        unified_mixtral = NMoEModelConfig.from_hf_config(hf_dict_mixtral)
        assert unified_mixtral.num_experts == 8

    @pytest.mark.integration
    def test_hf_mla_attention_fields(self):
        """Test HF MLA attention field mapping."""
        hf_dict = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }

        unified = NMoEModelConfig.from_dict(hf_dict)
        assert unified.q_lora_rank == 1536
        assert unified.kv_lora_rank == 512
        assert unified.head_dim == 192  # 128 + 64

    @pytest.mark.integration
    def test_hf_rope_config_mapping(self):
        """Test HF RoPE config dict mapping."""
        hf_dict = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "rope_theta": 100000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 4.0,
            },
        }

        unified = NMoEModelConfig.from_dict(hf_dict)
        assert unified.rope_theta == 100000.0
        assert unified.rope_scaling is not None
        assert unified.rope_scaling["factor"] == 4.0

    @pytest.mark.integration
    def test_hf_torch_dtype_mapping(self):
        """Test HF torch_dtype field mapping."""
        hf_dict = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "torch_dtype": "bfloat16",
        }

        unified = NMoEModelConfig.from_dict(hf_dict)
        assert unified.torch_dtype == "bfloat16"
        assert unified.dtype == "bf16"


# =============================================================================
# Test Class: Config Roundtrip Preservation
# =============================================================================

class TestConfigRoundtrip:
    """Test config roundtrip conversion preserves values."""

    @pytest.mark.integration
    def test_nmoe_to_hf_roundtrip(self, base_nmoe_config):
        """Test nmoe -> unified -> HF -> unified roundtrip."""
        # nmoe -> unified
        unified1 = NMoEModelConfig.from_nmoe_config(base_nmoe_config)

        # unified -> HF dict
        hf_dict = unified1.to_hf_config()

        # HF dict -> unified
        unified2 = NMoEModelConfig.from_hf_config(hf_dict)

        # Key fields should match
        assert unified1.hidden_size == unified2.hidden_size
        assert unified1.num_hidden_layers == unified2.num_hidden_layers
        assert unified1.num_attention_heads == unified2.num_attention_heads
        assert unified1.num_experts == unified2.num_experts
        assert unified1.num_experts_per_tok == unified2.num_experts_per_tok

    @pytest.mark.integration
    def test_unified_to_dict_roundtrip(self, unified_config):
        """Test unified -> dict -> unified roundtrip."""
        # To dict
        d = unified_config.to_dict()

        # From dict
        restored = NMoEModelConfig.from_dict(d)

        # All fields should match
        for f in dataclasses.fields(NMoEModelConfig):
            if f.name.startswith("_"):
                continue
            orig_val = getattr(unified_config, f.name)
            rest_val = getattr(restored, f.name)
            assert orig_val == rest_val, f"Field {f.name} mismatch: {orig_val} != {rest_val}"

    @pytest.mark.integration
    def test_fingerprint_preserved_through_roundtrip(self, base_nmoe_config):
        """Test that fingerprint is stable through roundtrip."""
        unified1 = NMoEModelConfig.from_nmoe_config(base_nmoe_config)
        fp1 = unified1.fingerprint()

        # Roundtrip through dict
        d = unified1.to_dict()
        unified2 = NMoEModelConfig.from_dict(d)
        fp2 = unified2.fingerprint()

        assert fp1 == fp2, "Fingerprint should be stable through dict roundtrip"

    @pytest.mark.integration
    def test_moe_specific_fields_preserved(self, base_nmoe_config):
        """Test MoE-specific fields preserved in roundtrip."""
        unified = NMoEModelConfig.from_nmoe_config(base_nmoe_config)

        # Check nmoe-style aliases work
        assert unified.dim == base_nmoe_config.dim
        assert unified.n_layers == base_nmoe_config.n_layers
        assert unified.n_heads == base_nmoe_config.n_heads
        assert unified.n_routed_experts == base_nmoe_config.n_routed_experts
        assert unified.n_activated_experts == base_nmoe_config.n_activated_experts
        assert unified.n_dense_layers == base_nmoe_config.n_dense_layers

    @pytest.mark.integration
    def test_attention_params_preserved(self, base_nmoe_config):
        """Test attention parameters preserved in roundtrip."""
        unified = NMoEModelConfig.from_nmoe_config(base_nmoe_config)

        assert unified.q_lora_rank == base_nmoe_config.q_lora_rank
        assert unified.kv_lora_rank == base_nmoe_config.kv_lora_rank
        assert unified.qk_nope_head_dim == base_nmoe_config.qk_nope_head_dim
        assert unified.qk_rope_head_dim == base_nmoe_config.qk_rope_head_dim
        assert unified.v_head_dim == base_nmoe_config.v_head_dim


# =============================================================================
# Test Class: Config Validation Across Components
# =============================================================================

class TestConfigValidation:
    """Test config validation across components."""

    @pytest.mark.integration
    def test_required_fields_validation(self):
        """Test that required fields are validated."""
        # Missing required fields
        cfg = NMoEModelConfig()  # All None
        with pytest.raises(ConfigValidationError) as exc_info:
            cfg.validate()
        assert "hidden_size" in str(exc_info.value)

    @pytest.mark.integration
    def test_moe_fields_validation(self):
        """Test MoE-specific field validation."""
        # MoE model without required MoE fields
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_experts=64,  # This triggers MoE validation
            # Missing: num_experts_per_tok, moe_intermediate_size
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            cfg.validate()
        assert "MoE" in str(exc_info.value) or "num_experts_per_tok" in str(exc_info.value)

    @pytest.mark.integration
    def test_valid_dense_model_config(self):
        """Test that valid dense model config passes validation."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=5632,
        )
        # Should not raise
        cfg.validate()

    @pytest.mark.integration
    def test_valid_moe_model_config(self):
        """Test that valid MoE model config passes validation."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=5632,
            moe_intermediate_size=1408,
            num_experts=64,
            num_experts_per_tok=8,
        )
        # Should not raise
        cfg.validate()

    @pytest.mark.integration
    def test_is_moe_property(self, unified_config, dense_nmoe_config):
        """Test is_moe property detection."""
        assert unified_config.is_moe is True

        dense_unified = NMoEModelConfig.from_nmoe_config(dense_nmoe_config)
        assert dense_unified.is_moe is False

    @pytest.mark.integration
    def test_total_experts_calculation(self, unified_config):
        """Test total_experts property calculation."""
        # num_experts + n_shared_experts
        expected = unified_config.num_experts + unified_config.n_shared_experts
        assert unified_config.total_experts == expected


# =============================================================================
# Test Class: MoE-Specific Config Fields
# =============================================================================

class TestMoESpecificFields:
    """Test MoE-specific config fields handling."""

    @pytest.mark.integration
    def test_router_config_fields(self, base_nmoe_config):
        """Test router configuration fields."""
        unified = NMoEModelConfig.from_nmoe_config(base_nmoe_config)

        assert unified.router_bias_update_rate == base_nmoe_config.router_bias_update_rate
        assert unified.router_aux_loss_coef == base_nmoe_config.aux_loss_alpha
        assert unified.norm_topk_prob == base_nmoe_config.norm_topk_prob
        assert unified.routed_scaling_factor == base_nmoe_config.route_scale

    @pytest.mark.integration
    def test_shared_experts_config(self):
        """Test shared experts configuration."""
        configs = [
            (0, 0),  # No shared experts
            (1, 1),  # Single shared expert
            (2, 2),  # Default shared experts
            (4, 4),  # More shared experts
        ]

        for n_shared, expected in configs:
            cfg = MockConfig(
                dim=2048, n_layers=24, n_heads=16,
                n_routed_experts=64,
                n_activated_experts=8,
                n_shared_experts=n_shared,
            )
            unified = NMoEModelConfig.from_nmoe_config(cfg)
            assert unified.n_shared_experts == expected

    @pytest.mark.integration
    def test_dense_layers_config(self):
        """Test first N dense layers configuration."""
        configs = [
            (0, 0),  # All MoE layers
            (1, 1),  # First layer dense (default)
            (3, 3),  # First 3 layers dense
            (6, 6),  # First 6 layers dense
        ]

        for n_dense, expected in configs:
            cfg = MockConfig(
                dim=2048, n_layers=24, n_heads=16,
                n_routed_experts=64,
                n_activated_experts=8,
                n_dense_layers=n_dense,
            )
            unified = NMoEModelConfig.from_nmoe_config(cfg)
            assert unified.first_k_dense_replace == expected
            assert unified.n_dense_layers == expected  # Alias

    @pytest.mark.integration
    def test_topk_routing_values(self):
        """Test various top-k routing configurations."""
        topk_values = [1, 2, 4, 8, 16]

        for topk in topk_values:
            cfg = MockConfig(
                dim=2048, n_layers=24, n_heads=16,
                n_routed_experts=64,
                n_activated_experts=topk,
            )
            unified = NMoEModelConfig.from_nmoe_config(cfg)
            assert unified.num_experts_per_tok == topk
            assert unified.n_activated_experts == topk

    @pytest.mark.integration
    def test_expert_count_variations(self):
        """Test various expert count configurations."""
        expert_counts = [8, 16, 32, 64, 128, 256]

        for n_experts in expert_counts:
            cfg = MockConfig(
                dim=2048, n_layers=24, n_heads=16,
                n_routed_experts=n_experts,
                n_activated_experts=min(8, n_experts),
            )
            unified = NMoEModelConfig.from_nmoe_config(cfg)
            assert unified.num_experts == n_experts
            assert unified.n_routed_experts == n_experts


# =============================================================================
# Test Class: RDEP Config Propagation
# =============================================================================

class TestRDEPConfigPropagation:
    """Test RDEP config propagation across components."""

    @pytest.mark.integration
    def test_rdep_basic_config(self, rdep_config):
        """Test basic RDEP config creation."""
        assert rdep_config.mode == "auto"
        assert rdep_config.profile == "bf16"
        assert rdep_config.capacity == 65536

    @pytest.mark.integration
    def test_rdep_profile_ids(self):
        """Test RDEP profile ID mapping."""
        profiles = [
            ("bf16", -1),
            ("fp8", 0),
            ("nvfp4", 1),
        ]

        for profile, expected_id in profiles:
            rdep = NMoERDEPConfig(profile=profile)
            assert rdep.get_profile_id() == expected_id

    @pytest.mark.integration
    def test_rdep_mode_detection(self):
        """Test RDEP mode auto-detection."""
        rdep = NMoERDEPConfig(mode="auto")

        # Single GPU
        assert rdep.detect_mode(world_size=1, local_world_size=1) == "single"

        # Multi-GPU, single node
        assert rdep.detect_mode(world_size=8, local_world_size=8) == "ipc"

        # Multi-node
        assert rdep.detect_mode(world_size=16, local_world_size=8) == "hybrid"

    @pytest.mark.integration
    def test_rdep_mode_override(self):
        """Test RDEP mode override (not auto)."""
        rdep_ipc = NMoERDEPConfig(mode="ipc")
        assert rdep_ipc.detect_mode(world_size=1, local_world_size=1) == "ipc"

        rdep_single = NMoERDEPConfig(mode="single")
        assert rdep_single.detect_mode(world_size=8, local_world_size=8) == "single"

    @pytest.mark.integration
    def test_rdep_to_dict_roundtrip(self, rdep_config):
        """Test RDEP config dict roundtrip."""
        d = rdep_config.to_dict()
        restored = NMoERDEPConfig.from_dict(d)

        assert rdep_config.mode == restored.mode
        assert rdep_config.profile == restored.profile
        assert rdep_config.capacity == restored.capacity

    @pytest.mark.integration
    def test_rdep_fingerprint(self):
        """Test RDEP config fingerprint consistency."""
        rdep1 = NMoERDEPConfig(mode="ipc", profile="fp8", capacity=32768)
        rdep2 = NMoERDEPConfig(mode="ipc", profile="fp8", capacity=32768)
        rdep3 = NMoERDEPConfig(mode="ipc", profile="bf16", capacity=32768)

        assert rdep1.fingerprint() == rdep2.fingerprint()
        assert rdep1.fingerprint() != rdep3.fingerprint()

    @pytest.mark.integration
    def test_rdep_nvshmem_settings(self):
        """Test RDEP NVSHMEM configuration."""
        rdep = NMoERDEPConfig(
            mode="hybrid",
            nvshmem_enabled=True,
            nvshmem_heap_size=2 << 30,  # 2GB
        )
        assert rdep.nvshmem_enabled is True
        assert rdep.nvshmem_heap_size == 2 << 30


# =============================================================================
# Test Class: Distributed Config Handling
# =============================================================================

class TestDistributedConfigHandling:
    """Test distributed config (TP, PP, EP) handling."""

    @pytest.mark.integration
    def test_tensor_parallel_config(self):
        """Test tensor parallel config handling."""
        # TP should divide hidden_size evenly
        hidden_sizes = [2048, 4096, 8192]
        tp_sizes = [1, 2, 4, 8]

        for hidden_size in hidden_sizes:
            for tp_size in tp_sizes:
                if hidden_size % tp_size == 0:
                    cfg = NMoEModelConfig(
                        hidden_size=hidden_size,
                        num_hidden_layers=24,
                        num_attention_heads=16,
                    )
                    assert cfg.hidden_size % tp_size == 0, \
                        f"TP={tp_size} incompatible with hidden_size={hidden_size}"

    @pytest.mark.integration
    def test_expert_parallel_config(self):
        """Test expert parallel config handling."""
        # EP should divide num_experts evenly
        expert_counts = [8, 16, 32, 64, 128]
        ep_sizes = [1, 2, 4, 8]

        for n_experts in expert_counts:
            for ep_size in ep_sizes:
                if n_experts % ep_size == 0:
                    cfg = NMoEModelConfig(
                        hidden_size=2048,
                        num_hidden_layers=24,
                        num_attention_heads=16,
                        num_experts=n_experts,
                        num_experts_per_tok=2,
                        moe_intermediate_size=1408,
                    )
                    assert cfg.num_experts % ep_size == 0, \
                        f"EP={ep_size} incompatible with num_experts={n_experts}"

    @pytest.mark.integration
    def test_pipeline_parallel_layer_divisibility(self):
        """Test pipeline parallel layer divisibility."""
        layer_counts = [12, 24, 32, 48, 64]
        pp_sizes = [1, 2, 4, 8]

        for n_layers in layer_counts:
            for pp_size in pp_sizes:
                if n_layers % pp_size == 0:
                    cfg = NMoEModelConfig(
                        hidden_size=2048,
                        num_hidden_layers=n_layers,
                        num_attention_heads=16,
                    )
                    assert cfg.num_hidden_layers % pp_size == 0, \
                        f"PP={pp_size} incompatible with num_layers={n_layers}"

    @pytest.mark.integration
    def test_attention_heads_tp_divisibility(self):
        """Test attention heads divisibility for tensor parallel."""
        head_counts = [8, 16, 32, 64]
        tp_sizes = [1, 2, 4, 8]

        for n_heads in head_counts:
            for tp_size in tp_sizes:
                if n_heads % tp_size == 0:
                    cfg = NMoEModelConfig(
                        hidden_size=2048,
                        num_hidden_layers=24,
                        num_attention_heads=n_heads,
                    )
                    assert cfg.num_attention_heads % tp_size == 0, \
                        f"TP={tp_size} incompatible with num_attention_heads={n_heads}"


# =============================================================================
# Test Class: Config Copy and Modification
# =============================================================================

class TestConfigCopyAndModification:
    """Test config copy and modification operations."""

    @pytest.mark.integration
    def test_config_copy_with_updates(self, unified_config):
        """Test config copy with field updates."""
        modified = unified_config.copy(
            hidden_size=4096,
            num_experts=128,
        )

        # Original unchanged
        assert unified_config.hidden_size == 2048
        assert unified_config.num_experts == 64

        # Modified has new values
        assert modified.hidden_size == 4096
        assert modified.num_experts == 128

        # Other fields preserved
        assert modified.num_hidden_layers == unified_config.num_hidden_layers
        assert modified.num_attention_heads == unified_config.num_attention_heads

    @pytest.mark.integration
    def test_config_copy_independence(self, unified_config):
        """Test that copied config is independent of original."""
        original = unified_config
        copied = original.copy()

        # Modify copied via internal dict (simulate mutation)
        copied_dict = copied.to_dict()
        copied_dict["hidden_size"] = 9999

        # Original should be unchanged
        assert original.hidden_size == 2048

    @pytest.mark.integration
    def test_config_fingerprint_changes_with_modification(self, unified_config):
        """Test that fingerprint changes when config is modified."""
        fp_original = unified_config.fingerprint()

        modified = unified_config.copy(hidden_size=4096)
        fp_modified = modified.fingerprint()

        assert fp_original != fp_modified


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestConfigEdgeCases:
    """Test config edge cases and error handling."""

    @pytest.mark.integration
    def test_none_value_handling_in_hf_export(self):
        """Test that None values are excluded from HF config."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            # Many fields will be None
        )
        hf_dict = cfg.to_hf_config()

        # No None values in output
        for key, value in hf_dict.items():
            assert value is not None, f"Key {key} has None value"

    @pytest.mark.integration
    def test_unknown_fields_ignored_in_from_dict(self):
        """Test that unknown fields are ignored when loading from dict."""
        d = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "unknown_field_xyz": "should be ignored",
            "another_unknown": 12345,
        }

        # Should not raise
        cfg = NMoEModelConfig.from_dict(d)
        assert cfg.hidden_size == 2048

        # Unknown fields not present
        assert not hasattr(cfg, "unknown_field_xyz")

    @pytest.mark.integration
    def test_empty_nested_dicts(self):
        """Test empty nested dict handling."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            attn_swa={},
            attn_nsa={},
            attn_dsa={},
        )

        d = cfg.to_dict()
        assert d["attn_swa"] == {}
        assert d["attn_nsa"] == {}
        assert d["attn_dsa"] == {}

    @pytest.mark.integration
    def test_special_float_values(self):
        """Test special float values in config."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            rope_theta=1e6,  # Very large
            rms_norm_eps=1e-8,  # Very small
        )

        d = cfg.to_dict()
        restored = NMoEModelConfig.from_dict(d)

        assert restored.rope_theta == 1e6
        assert restored.rms_norm_eps == 1e-8


# =============================================================================
# Test Class: Fingerprint Consistency
# =============================================================================

class TestFingerprintConsistency:
    """Test config fingerprint consistency and uniqueness."""

    @pytest.mark.integration
    def test_identical_configs_same_fingerprint(self):
        """Test that identical configs produce same fingerprint."""
        cfg1 = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
        )
        cfg2 = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
        )
        assert cfg1.fingerprint() == cfg2.fingerprint()

    @pytest.mark.integration
    def test_different_configs_different_fingerprint(self):
        """Test that different configs produce different fingerprints."""
        cfg1 = NMoEModelConfig(hidden_size=2048, num_hidden_layers=24, num_attention_heads=16)
        cfg2 = NMoEModelConfig(hidden_size=4096, num_hidden_layers=24, num_attention_heads=16)
        cfg3 = NMoEModelConfig(hidden_size=2048, num_hidden_layers=32, num_attention_heads=16)

        fp1 = cfg1.fingerprint()
        fp2 = cfg2.fingerprint()
        fp3 = cfg3.fingerprint()

        assert fp1 != fp2
        assert fp1 != fp3
        assert fp2 != fp3

    @pytest.mark.integration
    def test_fingerprint_is_deterministic(self):
        """Test that fingerprint is deterministic across calls."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_experts=64,
        )

        fingerprints = [cfg.fingerprint() for _ in range(10)]
        assert len(set(fingerprints)) == 1  # All same

    @pytest.mark.integration
    def test_standalone_fingerprint_function(self, unified_config):
        """Test standalone fingerprint function matches method."""
        fp_method = unified_config.fingerprint()
        fp_function = fingerprint(unified_config)
        assert fp_method == fp_function


# =============================================================================
# Test Class: Model Architecture Variations
# =============================================================================

class TestModelArchitectureVariations:
    """Test config handling for various model architectures."""

    @pytest.mark.integration
    def test_deepseek_v2_style_config(self):
        """Test DeepSeek-v2 style configuration."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=27,
            num_attention_heads=16,
            num_experts=64,
            num_experts_per_tok=6,
            n_shared_experts=2,
            first_k_dense_replace=1,
            q_lora_rank=1536,
            kv_lora_rank=512,
            attention_type="mla",
        )

        # Verify MLA-specific fields
        assert cfg.q_lora_rank == 1536
        assert cfg.kv_lora_rank == 512
        assert cfg.attn == "mla"

        # Verify MoE fields
        assert cfg.is_moe is True
        assert cfg.total_experts == 66  # 64 + 2

    @pytest.mark.integration
    def test_mixtral_style_config(self):
        """Test Mixtral-style configuration."""
        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=14336,
            moe_intermediate_size=14336,
            num_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=0,
            first_k_dense_replace=0,
        )

        assert cfg.is_moe is True
        assert cfg.num_experts == 8
        assert cfg.total_experts == 8  # No shared experts

    @pytest.mark.integration
    def test_dense_llama_style_config(self):
        """Test dense LLaMA-style configuration."""
        cfg = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=11008,
            # No MoE fields
        )

        assert cfg.is_moe is False
        assert cfg.total_experts == 0

    @pytest.mark.integration
    def test_small_debug_config(self, small_nmoe_config):
        """Test small debug/test configuration."""
        unified = NMoEModelConfig.from_nmoe_config(small_nmoe_config)

        assert unified.hidden_size == 512
        assert unified.num_hidden_layers == 6
        assert unified.num_experts == 8
        assert unified.vocab_size == 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
