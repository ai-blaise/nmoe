"""Tests for weight mapping functions.

Tests correctness of nmoe <-> HuggingFace weight name translation.
"""

import pytest
import sys
sys.path.insert(0, '/home/nourdine/sglang_nmoe/nether-soup/nmoe')

from nmoe.tools.config_converter import (
    nmoe_to_hf_weight_mapping,
    hf_to_nmoe_weight_mapping,
    get_expert_weight_info,
    expand_expert_weights_to_hf,
    validate_weight_mapping,
)


class TestNmoeToHfMapping:
    """Test nmoe to HuggingFace weight name mapping."""

    def test_embedding_mapping(self):
        """Test embedding weight mapping."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)
        assert mapping['embedding.weight'] == 'model.embed_tokens.weight'

    def test_lm_head_mapping(self):
        """Test lm_head weight mapping."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)
        assert mapping['lm_head.weight'] == 'lm_head.weight'

    def test_final_norm_mapping(self):
        """Test final norm weight mapping."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)
        assert mapping['norm.weight'] == 'model.norm.weight'

    def test_attention_norm_mapping(self):
        """Test attention norm mapping per layer."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)

        for layer_id in range(4):
            nmoe_key = f'blocks.{layer_id}.attn_norm.weight'
            hf_key = f'model.layers.{layer_id}.input_layernorm.weight'
            assert mapping[nmoe_key] == hf_key

    def test_ffn_norm_mapping(self):
        """Test FFN norm mapping per layer."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)

        for layer_id in range(4):
            nmoe_key = f'blocks.{layer_id}.ffn_norm.weight'
            hf_key = f'model.layers.{layer_id}.post_attention_layernorm.weight'
            assert mapping[nmoe_key] == hf_key

    def test_dense_mlp_mapping(self):
        """Test dense layer MLP weight mapping."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)

        # Layer 0 is dense
        assert mapping['blocks.0.ffn.w1.weight'] == 'model.layers.0.mlp.gate_proj.weight'
        assert mapping['blocks.0.ffn.w3.weight'] == 'model.layers.0.mlp.up_proj.weight'
        assert mapping['blocks.0.ffn.w2.weight'] == 'model.layers.0.mlp.down_proj.weight'

    def test_moe_router_mapping(self):
        """Test MoE router weight mapping."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)

        # Layers 1-3 are MoE
        for layer_id in range(1, 4):
            gate_key = f'blocks.{layer_id}.ffn.router.gate.weight'
            assert mapping[gate_key] == f'model.layers.{layer_id}.mlp.gate.weight'

    def test_router_bias_mapping(self):
        """Test router bias -> e_score_correction_bias mapping."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)

        for layer_id in range(1, 4):
            bias_key = f'blocks.{layer_id}.ffn.router.bias'
            expected = f'model.layers.{layer_id}.mlp.gate.e_score_correction_bias'
            assert mapping[bias_key] == expected

    def test_expert_weight_mapping(self):
        """Test expert weight mapping (stacked format)."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)

        for layer_id in range(1, 4):
            # These are stacked tensors, mapped to special keys
            assert f'blocks.{layer_id}.ffn.W1' in mapping
            assert f'blocks.{layer_id}.ffn.W3' in mapping
            assert f'blocks.{layer_id}.ffn.W2' in mapping

    def test_shared_expert_mapping(self):
        """Test shared expert MLP weight mapping."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1, n_shared_experts=2)

        for layer_id in range(1, 4):
            w1_key = f'blocks.{layer_id}.ffn._shared.w1.weight'
            w3_key = f'blocks.{layer_id}.ffn._shared.w3.weight'
            w2_key = f'blocks.{layer_id}.ffn._shared.w2.weight'

            assert mapping[w1_key] == f'model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight'
            assert mapping[w3_key] == f'model.layers.{layer_id}.mlp.shared_experts.up_proj.weight'
            assert mapping[w2_key] == f'model.layers.{layer_id}.mlp.shared_experts.down_proj.weight'

    def test_no_shared_experts(self):
        """Test mapping without shared experts."""
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1, n_shared_experts=0)

        # Should not have shared expert keys
        for layer_id in range(1, 4):
            w1_key = f'blocks.{layer_id}.ffn._shared.w1.weight'
            assert w1_key not in mapping


class TestHfToNmoeMapping:
    """Test inverse mapping from HF to nmoe."""

    def test_inverse_mapping_consistency(self):
        """Test that inverse mapping is consistent with forward mapping."""
        forward = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=1)
        inverse = hf_to_nmoe_weight_mapping(n_layers=4, n_dense_layers=1)

        # Every forward mapping should have an inverse
        for nmoe_key, hf_key in forward.items():
            assert inverse[hf_key] == nmoe_key


class TestGetExpertWeightInfo:
    """Test expert weight info extraction."""

    def test_nmoe_format_w1(self):
        """Test parsing nmoe W1 weight names."""
        info = get_expert_weight_info('blocks.5.ffn.W1')
        assert info is not None
        assert info['layer_id'] == 5
        assert info['expert_id'] is None
        assert info['weight_type'] == 'W1'
        assert info['format'] == 'nmoe'

    def test_nmoe_format_w2(self):
        """Test parsing nmoe W2 weight names."""
        info = get_expert_weight_info('blocks.10.ffn.W2')
        assert info is not None
        assert info['layer_id'] == 10
        assert info['weight_type'] == 'W2'

    def test_nmoe_format_w3(self):
        """Test parsing nmoe W3 weight names."""
        info = get_expert_weight_info('blocks.0.ffn.W3')
        assert info is not None
        assert info['layer_id'] == 0
        assert info['weight_type'] == 'W3'

    def test_hf_format_gate_proj(self):
        """Test parsing HF gate_proj weight names."""
        info = get_expert_weight_info('model.layers.5.mlp.experts.3.gate_proj.weight')
        assert info is not None
        assert info['layer_id'] == 5
        assert info['expert_id'] == 3
        assert info['weight_type'] == 'W1'
        assert info['format'] == 'hf'

    def test_hf_format_up_proj(self):
        """Test parsing HF up_proj weight names."""
        info = get_expert_weight_info('model.layers.7.mlp.experts.15.up_proj.weight')
        assert info is not None
        assert info['layer_id'] == 7
        assert info['expert_id'] == 15
        assert info['weight_type'] == 'W3'

    def test_hf_format_down_proj(self):
        """Test parsing HF down_proj weight names."""
        info = get_expert_weight_info('model.layers.2.mlp.experts.0.down_proj.weight')
        assert info is not None
        assert info['layer_id'] == 2
        assert info['expert_id'] == 0
        assert info['weight_type'] == 'W2'

    def test_non_expert_weight(self):
        """Test non-expert weights return None."""
        assert get_expert_weight_info('embedding.weight') is None
        assert get_expert_weight_info('blocks.0.attn_norm.weight') is None
        assert get_expert_weight_info('blocks.0.ffn.w1.weight') is None


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestExpandExpertWeights:
    """Test expert weight expansion from stacked to individual format."""

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_expand_single_layer(self):
        """Test expanding expert weights for a single layer."""
        import torch

        # Create mock nmoe state dict with stacked expert weights
        n_experts = 4
        dim = 64
        inter_dim = 128

        nmoe_state_dict = {
            'blocks.1.ffn.W1': torch.randn(n_experts, dim, inter_dim),
            'blocks.1.ffn.W3': torch.randn(n_experts, dim, inter_dim),
            'blocks.1.ffn.W2': torch.randn(n_experts, inter_dim, dim),
        }

        hf_state_dict = expand_expert_weights_to_hf(
            nmoe_state_dict,
            n_layers=2,
            n_dense_layers=1,
            n_experts=n_experts,
        )

        # Should have expanded to individual expert keys
        for expert_id in range(n_experts):
            assert f'model.layers.1.mlp.experts.{expert_id}.gate_proj.weight' in hf_state_dict
            assert f'model.layers.1.mlp.experts.{expert_id}.up_proj.weight' in hf_state_dict
            assert f'model.layers.1.mlp.experts.{expert_id}.down_proj.weight' in hf_state_dict

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_weight_shapes_after_expansion(self):
        """Test that expanded weights have correct shapes (transposed)."""
        import torch

        n_experts = 2
        dim = 32
        inter_dim = 64

        nmoe_state_dict = {
            'blocks.1.ffn.W1': torch.randn(n_experts, dim, inter_dim),
            'blocks.1.ffn.W3': torch.randn(n_experts, dim, inter_dim),
            'blocks.1.ffn.W2': torch.randn(n_experts, inter_dim, dim),
        }

        hf_state_dict = expand_expert_weights_to_hf(
            nmoe_state_dict,
            n_layers=2,
            n_dense_layers=1,
            n_experts=n_experts,
        )

        # W1, W3: [dim, inter_dim] -> [inter_dim, dim] in HF
        gate_proj = hf_state_dict['model.layers.1.mlp.experts.0.gate_proj.weight']
        assert gate_proj.shape == (inter_dim, dim)

        up_proj = hf_state_dict['model.layers.1.mlp.experts.0.up_proj.weight']
        assert up_proj.shape == (inter_dim, dim)

        # W2: [inter_dim, dim] -> [dim, inter_dim] in HF
        down_proj = hf_state_dict['model.layers.1.mlp.experts.0.down_proj.weight']
        assert down_proj.shape == (dim, inter_dim)


class TestMappingCoverage:
    """Test that mapping covers all expected weights."""

    def test_32_layer_model_coverage(self):
        """Test mapping coverage for a 32-layer model."""
        n_layers = 32
        n_dense_layers = 1
        n_experts = 64

        mapping = nmoe_to_hf_weight_mapping(
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_routed_experts=n_experts,
            n_shared_experts=2,
        )

        # Count expected keys
        # Embedding: 1, LM head: 1, Final norm: 1
        # Per layer: attn_norm, ffn_norm, attn weights, ffn weights
        # Dense layers: w1, w2, w3
        # MoE layers: router.gate, router.bias, W1, W3, W2, shared w1, w2, w3

        # At minimum we should have:
        # - 3 global keys (embedding, lm_head, norm)
        # - 2 norm keys per layer (attn_norm, ffn_norm)
        min_expected = 3 + (n_layers * 2)
        assert len(mapping) >= min_expected

        print(f"Total mapping entries for 32-layer model: {len(mapping)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
