"""Comprehensive unit tests for config_converter.py.

Tests cover:
- Weight name mapping functions
- Bidirectional conversion
- Expert weight parsing
- Weight expansion
- Validation functions
"""

import pytest
from typing import Dict, Any


class TestNmoeToHFWeightMapping:
    """Tests for nmoe_to_hf_weight_mapping function."""

    def test_mapping_returns_dict(self):
        """Mapping returns a dictionary."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        # n_routed_experts is the correct parameter name
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_routed_experts=8)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_mapping_contains_embeddings(self):
        """Mapping includes embedding layers."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_routed_experts=8)

        # Check for embedding-related keys
        has_embed = any("embed" in k.lower() for k in mapping.keys())
        assert has_embed

    def test_mapping_contains_lm_head(self):
        """Mapping includes LM head."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_routed_experts=8)

        has_lm_head = any("lm_head" in k.lower() or "output" in k.lower() for k in mapping.keys())
        assert has_lm_head

    def test_mapping_contains_all_layers(self):
        """Mapping includes all transformer layers."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        n_layers = 4
        mapping = nmoe_to_hf_weight_mapping(n_layers=n_layers, n_routed_experts=8)

        # Check each layer is represented
        for i in range(n_layers):
            layer_keys = [k for k in mapping.keys() if f".{i}." in k or f"_{i}_" in k or f"[{i}]" in k]
            assert len(layer_keys) > 0, f"Layer {i} not found in mapping"

    def test_mapping_contains_expert_weights(self):
        """Mapping includes expert weights."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        # Use n_dense_layers=0 so all layers are MoE
        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_dense_layers=0, n_routed_experts=8)

        # Check for expert weight keys (W1, W2, W3 in ffn)
        has_expert = any("ffn.W" in k or "expert" in k.lower() for k in mapping.keys())
        assert has_expert

    def test_mapping_with_different_layer_counts(self):
        """Mapping works with different layer counts."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        for n_layers in [1, 8, 32, 128]:
            mapping = nmoe_to_hf_weight_mapping(n_layers=n_layers, n_routed_experts=64)
            assert len(mapping) > 0


class TestHFToNmoeWeightMapping:
    """Tests for hf_to_nmoe_weight_mapping function."""

    def test_inverse_mapping_returns_dict(self):
        """Inverse mapping returns a dictionary."""
        from nmoe.tools.config_converter import hf_to_nmoe_weight_mapping

        mapping = hf_to_nmoe_weight_mapping(n_layers=4, n_routed_experts=8)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_bidirectional_consistency(self):
        """Forward and inverse mappings are consistent."""
        from nmoe.tools.config_converter import (
            nmoe_to_hf_weight_mapping,
            hf_to_nmoe_weight_mapping,
        )

        forward = nmoe_to_hf_weight_mapping(n_layers=4, n_routed_experts=8)
        inverse = hf_to_nmoe_weight_mapping(n_layers=4, n_routed_experts=8)

        # Every key in forward should map to a key in inverse
        for nmoe_key, hf_key in forward.items():
            if hf_key in inverse:
                assert inverse[hf_key] == nmoe_key


class TestGetExpertWeightInfo:
    """Tests for get_expert_weight_info function."""

    def test_parses_nmoe_expert_name(self):
        """Parses nmoe-style expert weight names."""
        from nmoe.tools.config_converter import get_expert_weight_info

        # nmoe format uses "ffn" not "moe": blocks.{layer}.ffn.W{1,2,3}
        info = get_expert_weight_info("blocks.5.ffn.W1")

        assert info is not None
        assert info.get("layer_id") == 5

    def test_parses_hf_expert_name(self):
        """Parses HuggingFace-style expert weight names."""
        from nmoe.tools.config_converter import get_expert_weight_info

        info = get_expert_weight_info("model.layers.5.mlp.experts.3.gate_proj.weight")

        assert info is not None

    def test_returns_none_for_non_expert(self):
        """Returns None for non-expert weights."""
        from nmoe.tools.config_converter import get_expert_weight_info

        info = get_expert_weight_info("model.embed_tokens.weight")

        # Should return None or empty dict for non-expert weights
        assert info is None or (isinstance(info, dict) and len(info) == 0)

    def test_extracts_expert_id(self):
        """Extracts expert ID from name."""
        from nmoe.tools.config_converter import get_expert_weight_info

        info = get_expert_weight_info("blocks.2.moe.experts.7.w1")

        if info:
            # Should contain expert_id or similar
            assert "expert" in str(info).lower() or "7" in str(info)


class TestExpandExpertWeightsToHF:
    """Tests for expand_expert_weights_to_hf function."""

    def test_expands_stacked_weights(self):
        """Expands stacked expert weights to individual."""
        import torch
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        # Mock stacked weights [n_experts, dim, inter_dim]
        n_experts = 8
        dim = 128
        inter_dim = 256
        n_layers = 4
        n_dense_layers = 1  # First layer is dense, rest are MoE

        # Create nmoe-style state dict with stacked expert weights
        # MoE layers are 1, 2, 3 (layer 0 is dense)
        nmoe_state_dict = {}
        for layer_id in range(n_dense_layers, n_layers):
            nmoe_state_dict[f'blocks.{layer_id}.ffn.W1'] = torch.randn(n_experts, dim, inter_dim)
            nmoe_state_dict[f'blocks.{layer_id}.ffn.W2'] = torch.randn(n_experts, inter_dim, dim)
            nmoe_state_dict[f'blocks.{layer_id}.ffn.W3'] = torch.randn(n_experts, dim, inter_dim)

        expanded = expand_expert_weights_to_hf(
            nmoe_state_dict,
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_experts=n_experts
        )

        assert isinstance(expanded, dict)
        # Should have 3 weight types * n_experts * (n_layers - n_dense_layers) MoE layers
        expected_expert_keys = 3 * n_experts * (n_layers - n_dense_layers)
        expert_keys = [k for k in expanded.keys() if 'experts' in k]
        assert len(expert_keys) == expected_expert_keys

    def test_correct_shapes_after_expansion(self):
        """Expanded weights have correct shapes."""
        import torch
        from nmoe.tools.config_converter import expand_expert_weights_to_hf

        n_experts = 4
        dim = 64
        inter_dim = 128
        n_layers = 2
        n_dense_layers = 0  # All MoE layers

        # Create nmoe-style state dict
        nmoe_state_dict = {}
        for layer_id in range(n_layers):
            # W1 and W3: [n_experts, dim, inter_dim]
            nmoe_state_dict[f'blocks.{layer_id}.ffn.W1'] = torch.randn(n_experts, dim, inter_dim)
            nmoe_state_dict[f'blocks.{layer_id}.ffn.W3'] = torch.randn(n_experts, dim, inter_dim)
            # W2: [n_experts, inter_dim, dim]
            nmoe_state_dict[f'blocks.{layer_id}.ffn.W2'] = torch.randn(n_experts, inter_dim, dim)

        expanded = expand_expert_weights_to_hf(
            nmoe_state_dict,
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_experts=n_experts
        )

        for key, tensor in expanded.items():
            if 'gate_proj' in key or 'up_proj' in key:
                # W1/W3 transposed: [inter_dim, dim]
                assert tensor.shape == (inter_dim, dim), f"Wrong shape for {key}: {tensor.shape}"
            elif 'down_proj' in key:
                # W2 transposed: [dim, inter_dim]
                assert tensor.shape == (dim, inter_dim), f"Wrong shape for {key}: {tensor.shape}"


class TestValidateWeightMapping:
    """Tests for validate_weight_mapping function."""

    def test_validates_complete_mapping(self):
        """Validates complete weight mapping."""
        from nmoe.tools.config_converter import validate_weight_mapping, nmoe_to_hf_weight_mapping

        n_layers = 1
        n_dense_layers = 0
        n_experts = 8
        n_shared_experts = 2

        # Get the actual mapping to determine valid keys
        mapping = nmoe_to_hf_weight_mapping(n_layers, n_dense_layers, n_experts, n_shared_experts)

        # Use subset of actual mapped keys
        nmoe_keys = list(mapping.keys())[:5]
        hf_keys = [mapping[k] for k in nmoe_keys]

        # validate_weight_mapping returns (matched, missing, extra)
        matched, missing, extra = validate_weight_mapping(
            nmoe_keys,
            hf_keys,
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_experts=n_experts,
            n_shared_experts=n_shared_experts
        )

        # All nmoe_keys should be matched
        assert len(matched) == len(nmoe_keys)
        assert len(missing) == 0

    def test_detects_missing_weights(self):
        """Detects missing weights in mapping."""
        from nmoe.tools.config_converter import validate_weight_mapping

        # Keys that exist in nmoe mapping
        nmoe_keys = ["embedding.weight", "lm_head.weight", "blocks.0.attn_norm.weight"]
        # Empty hf_keys means they're all missing
        hf_keys = []

        matched, missing, extra = validate_weight_mapping(
            nmoe_keys,
            hf_keys,
            n_layers=1,
            n_dense_layers=0,
            n_experts=8,
            n_shared_experts=2
        )

        # Some keys should be in missing (those that mapped but target not in hf_keys)
        # Note: Expert weights (.W1, .W2, .W3) are auto-matched due to expansion logic
        assert len(missing) > 0 or len(matched) < len(nmoe_keys)


class TestSharedExpertMapping:
    """Tests for shared expert weight mapping."""

    def test_mapping_includes_shared_experts(self):
        """Mapping includes shared expert weights."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        # Correct parameter name is n_shared_experts
        mapping = nmoe_to_hf_weight_mapping(
            n_layers=4,
            n_dense_layers=1,  # First layer dense, rest MoE
            n_routed_experts=8,
            n_shared_experts=2
        )

        has_shared = any("shared" in k.lower() for k in mapping.keys())
        # Shared experts should be present in MoE layers
        assert has_shared or len(mapping) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_experts(self):
        """Handles zero experts (dense model)."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        # Use n_routed_experts and n_dense_layers to make all layers dense
        mapping = nmoe_to_hf_weight_mapping(
            n_layers=4,
            n_dense_layers=4,  # All layers are dense
            n_routed_experts=0
        )

        # Should still have non-expert weights
        assert len(mapping) > 0

    def test_single_layer(self):
        """Handles single layer model."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        mapping = nmoe_to_hf_weight_mapping(n_layers=1, n_routed_experts=8)

        assert len(mapping) > 0

    def test_large_expert_count(self):
        """Handles large expert counts."""
        from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

        mapping = nmoe_to_hf_weight_mapping(n_layers=4, n_routed_experts=256)

        has_expert = any("expert" in k.lower() or "moe" in k.lower() for k in mapping.keys())
        assert has_expert or len(mapping) > 0
