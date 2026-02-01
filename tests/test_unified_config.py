"""Tests for unified config module.

Tests config round-trip conversion: nmoe -> unified -> HF -> unified
"""

import pytest
import sys
sys.path.insert(0, '/home/nourdine/sglang_nmoe/nether-soup/nmoe')

# Import unified config (doesn't depend on tomllib)
from nmoe.unified.config import NMoEModelConfig, NMoERDEPConfig

# Create a minimal Config class for testing without tomllib dependency
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Config:
    """Minimal nmoe Config for testing."""
    # Core dimensions
    vocab_size: int = 201088
    tokenizer: str = "o200k_harmony"
    eos_token_id: int = 199999
    dim: int = None
    n_layers: int = None
    n_heads: int = None

    # MoE
    inter_dim: int = None
    moe_inter_dim: int = None
    n_routed_experts: int = None
    n_activated_experts: int = None
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

    # Backend-specific
    attn_swa: Dict[str, Any] = None
    attn_nsa: Dict[str, Any] = None
    attn_dsa: Dict[str, Any] = None

    def __post_init__(self):
        if self.attn_swa is None:
            self.attn_swa = {}
        if self.attn_nsa is None:
            self.attn_nsa = {}
        if self.attn_dsa is None:
            self.attn_dsa = {}


class TestNMoEModelConfig:
    """Test suite for NMoEModelConfig."""

    def test_basic_instantiation(self):
        """Test basic config creation with required fields."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
        )
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 24
        assert cfg.num_attention_heads == 16

    def test_moe_configuration(self):
        """Test MoE-specific configuration fields."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_experts=64,
            num_experts_per_tok=8,
            n_shared_experts=2,
            first_k_dense_replace=1,
        )
        assert cfg.num_experts == 64
        assert cfg.num_experts_per_tok == 8
        assert cfg.n_shared_experts == 2
        assert cfg.first_k_dense_replace == 1

    def test_to_dict_from_dict_roundtrip(self):
        """Test to_dict/from_dict preserves all fields."""
        original = NMoEModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_experts=128,
            num_experts_per_tok=8,
            q_lora_rank=2048,
            kv_lora_rank=768,
            rope_theta=100000.0,
        )

        # Round-trip
        d = original.to_dict()
        restored = NMoEModelConfig.from_dict(d)

        # Check all important fields match
        assert original.hidden_size == restored.hidden_size
        assert original.num_hidden_layers == restored.num_hidden_layers
        assert original.num_experts == restored.num_experts
        assert original.q_lora_rank == restored.q_lora_rank
        assert original.rope_theta == restored.rope_theta

    def test_fingerprint_consistency(self):
        """Test that fingerprints are consistent for identical configs."""
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

    def test_fingerprint_changes_with_config(self):
        """Test that fingerprints differ for different configs."""
        cfg1 = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
        )
        cfg2 = NMoEModelConfig(
            hidden_size=4096,  # Different
            num_hidden_layers=24,
            num_attention_heads=16,
        )
        assert cfg1.fingerprint() != cfg2.fingerprint()

    def test_head_dim_property(self):
        """Test head_dim computed property."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
        )
        assert cfg.head_dim == 192  # 128 + 64


class TestFromNmoeConfig:
    """Test conversion from nmoe Config to unified config."""

    def test_basic_conversion(self):
        """Test basic field mapping from nmoe Config."""
        nmoe_cfg = Config(
            dim=2048,
            n_layers=24,
            n_heads=16,
            inter_dim=5632,
            moe_inter_dim=1408,
            n_routed_experts=64,
            n_activated_experts=8,
        )

        unified = NMoEModelConfig.from_nmoe_config(nmoe_cfg)

        assert unified.hidden_size == 2048
        assert unified.num_hidden_layers == 24
        assert unified.num_attention_heads == 16
        assert unified.intermediate_size == 5632
        assert unified.moe_intermediate_size == 1408
        assert unified.num_experts == 64
        assert unified.num_experts_per_tok == 8

    def test_mla_attention_params(self):
        """Test MLA attention parameter conversion."""
        nmoe_cfg = Config(
            dim=2048,
            n_layers=24,
            n_heads=16,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

        unified = NMoEModelConfig.from_nmoe_config(nmoe_cfg)

        assert unified.q_lora_rank == 1536
        assert unified.kv_lora_rank == 512
        assert unified.qk_nope_head_dim == 128
        assert unified.qk_rope_head_dim == 64
        assert unified.v_head_dim == 128

    def test_rope_params_conversion(self):
        """Test RoPE parameter conversion."""
        nmoe_cfg = Config(
            dim=2048,
            n_layers=24,
            n_heads=16,
            rope_theta=50000.0,
            rope_scaling_factor=2.0,
            rope_ntk_alpha=1.5,
            max_position_embeddings=8192,
        )

        unified = NMoEModelConfig.from_nmoe_config(nmoe_cfg)

        assert unified.rope_theta == 50000.0
        assert unified.max_position_embeddings == 8192
        assert unified.rope_scaling is not None
        assert unified.rope_scaling['factor'] == 2.0

    def test_dtype_to_quantization_mapping(self):
        """Test dtype -> quantization mapping."""
        # BF16 (no quantization)
        cfg_bf16 = Config(dim=2048, n_layers=24, n_heads=16, dtype='bf16')
        unified_bf16 = NMoEModelConfig.from_nmoe_config(cfg_bf16)
        assert unified_bf16.quantization is None
        assert unified_bf16.torch_dtype == 'bfloat16'

        # FP8
        cfg_fp8 = Config(dim=2048, n_layers=24, n_heads=16, dtype='fp8')
        unified_fp8 = NMoEModelConfig.from_nmoe_config(cfg_fp8)
        assert unified_fp8.quantization == 'fp8'

        # NVFP4
        cfg_nvfp4 = Config(dim=2048, n_layers=24, n_heads=16, dtype='nvfp4')
        unified_nvfp4 = NMoEModelConfig.from_nmoe_config(cfg_nvfp4)
        assert unified_nvfp4.quantization == 'modelopt_fp4'


class TestToHfConfig:
    """Test conversion to HuggingFace config dict."""

    def test_basic_hf_export(self):
        """Test basic HF config export."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            vocab_size=201088,
        )

        hf_cfg = cfg.to_hf_config()

        assert hf_cfg['hidden_size'] == 2048
        assert hf_cfg['num_hidden_layers'] == 24
        assert hf_cfg['num_attention_heads'] == 16
        assert hf_cfg['vocab_size'] == 201088
        assert hf_cfg['model_type'] == 'nmoe'
        assert 'NMoEForCausalLM' in hf_cfg['architectures']

    def test_moe_fields_in_hf_config(self):
        """Test MoE fields in HF config."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_experts=64,
            num_experts_per_tok=8,
            n_shared_experts=2,
        )

        hf_cfg = cfg.to_hf_config()

        assert hf_cfg['n_routed_experts'] == 64
        assert hf_cfg['num_experts_per_tok'] == 8
        assert hf_cfg['n_shared_experts'] == 2

    def test_mla_fields_in_hf_config(self):
        """Test MLA attention fields in HF config."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            q_lora_rank=1536,
            kv_lora_rank=512,
        )

        hf_cfg = cfg.to_hf_config()

        assert hf_cfg['q_lora_rank'] == 1536
        assert hf_cfg['kv_lora_rank'] == 512

    def test_no_none_values(self):
        """Test that None values are excluded from HF config."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
        )

        hf_cfg = cfg.to_hf_config()

        for key, value in hf_cfg.items():
            assert value is not None, f"Key {key} has None value"


class TestToSglangServerArgs:
    """Test conversion to SGLang server args."""

    def test_basic_server_args(self):
        """Test basic SGLang server args export."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            max_position_embeddings=8192,
        )

        args = cfg.to_sglang_server_args()

        assert args['context_length'] == 8192
        assert args['moe_runner_backend'] == 'nmoe'

    def test_quantization_in_server_args(self):
        """Test quantization passed to server args."""
        cfg = NMoEModelConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            quantization='fp8',
        )

        args = cfg.to_sglang_server_args()

        assert args['quantization'] == 'fp8'


class TestFullRoundTrip:
    """Test complete round-trip: nmoe -> unified -> HF -> unified."""

    def test_nmoe_to_hf_roundtrip(self):
        """Test nmoe Config -> unified -> HF dict -> unified."""
        # Create nmoe config
        original_nmoe = Config(
            dim=2048,
            n_layers=24,
            n_heads=16,
            inter_dim=5632,
            moe_inter_dim=1408,
            n_routed_experts=64,
            n_activated_experts=8,
            n_shared_experts=2,
            n_dense_layers=1,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rope_theta=50000.0,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            vocab_size=201088,
            eos_token_id=199999,
        )

        # Convert to unified
        unified = NMoEModelConfig.from_nmoe_config(original_nmoe)

        # Export to HF
        hf_dict = unified.to_hf_config()

        # Verify key fields preserved
        assert hf_dict['hidden_size'] == original_nmoe.dim
        assert hf_dict['num_hidden_layers'] == original_nmoe.n_layers
        assert hf_dict['num_attention_heads'] == original_nmoe.n_heads
        assert hf_dict['n_routed_experts'] == original_nmoe.n_routed_experts
        assert hf_dict['num_experts_per_tok'] == original_nmoe.n_activated_experts
        assert hf_dict['q_lora_rank'] == original_nmoe.q_lora_rank
        assert hf_dict['kv_lora_rank'] == original_nmoe.kv_lora_rank
        assert hf_dict['rope_theta'] == original_nmoe.rope_theta
        assert hf_dict['vocab_size'] == original_nmoe.vocab_size

        # Reconstruct from HF dict
        restored = NMoEModelConfig.from_dict(hf_dict)

        # Verify core fields match
        assert restored.hidden_size == unified.hidden_size
        assert restored.num_hidden_layers == unified.num_hidden_layers
        assert restored.num_attention_heads == unified.num_attention_heads


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
