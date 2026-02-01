"""Config and weight mapping utilities for nmoe model export.

This module provides functions for converting between nmoe weight names
and HuggingFace/SGLang weight names for model export and import.
"""

from typing import Dict, List, Tuple, Optional
import re


def nmoe_to_hf_weight_mapping(
    n_layers: int,
    n_dense_layers: int = 1,
    n_routed_experts: int = 64,
    n_shared_experts: int = 2,
) -> Dict[str, str]:
    """Create mapping from nmoe weight names to HuggingFace weight names.

    This function generates a complete mapping of all weight keys from nmoe's
    naming convention to HuggingFace's DeepSeek-v2 style naming convention.

    Args:
        n_layers: Total number of transformer layers
        n_dense_layers: Number of initial dense (non-MoE) layers
        n_routed_experts: Number of routed experts per MoE layer
        n_shared_experts: Number of shared experts

    Returns:
        Dict mapping nmoe weight names to HF weight names

    Example:
        >>> mapping = nmoe_to_hf_weight_mapping(n_layers=24, n_dense_layers=1)
        >>> mapping['blocks.1.ffn.W1']
        'model.layers.1.mlp.experts.0.gate_proj.weight'
    """
    mapping = {}

    # Embedding
    mapping['embedding.weight'] = 'model.embed_tokens.weight'

    # LM Head
    mapping['lm_head.weight'] = 'lm_head.weight'

    # Final norm
    mapping['norm.weight'] = 'model.norm.weight'

    # Per-layer mappings
    for layer_id in range(n_layers):
        nmoe_prefix = f'blocks.{layer_id}'
        hf_prefix = f'model.layers.{layer_id}'

        # Attention norms
        mapping[f'{nmoe_prefix}.attn_norm.weight'] = f'{hf_prefix}.input_layernorm.weight'
        mapping[f'{nmoe_prefix}.ffn_norm.weight'] = f'{hf_prefix}.post_attention_layernorm.weight'

        # Attention weights (MLA style)
        # nmoe MLA has: wq_a, q_norm, wq_b, wkv_a, kv_norm, wkv_b, wo
        # HF DeepSeek-v2 style: q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj

        # Q-path: wq_a -> q_a_proj, q_norm -> q_a_layernorm, wq_b -> q_b_proj
        mapping[f'{nmoe_prefix}.attn.wq_a.weight'] = f'{hf_prefix}.self_attn.q_a_proj.weight'
        mapping[f'{nmoe_prefix}.attn.q_norm.weight'] = f'{hf_prefix}.self_attn.q_a_layernorm.weight'
        mapping[f'{nmoe_prefix}.attn.wq_b.weight'] = f'{hf_prefix}.self_attn.q_b_proj.weight'

        # KV-path: wkv_a -> kv_a_proj_with_mqa, kv_norm -> kv_a_layernorm, wkv_b -> kv_b_proj
        mapping[f'{nmoe_prefix}.attn.wkv_a.weight'] = f'{hf_prefix}.self_attn.kv_a_proj_with_mqa.weight'
        mapping[f'{nmoe_prefix}.attn.kv_norm.weight'] = f'{hf_prefix}.self_attn.kv_a_layernorm.weight'
        mapping[f'{nmoe_prefix}.attn.wkv_b.weight'] = f'{hf_prefix}.self_attn.kv_b_proj.weight'

        # Output: wo -> o_proj
        mapping[f'{nmoe_prefix}.attn.wo.weight'] = f'{hf_prefix}.self_attn.o_proj.weight'

        # FFN/MoE mappings
        is_moe_layer = layer_id >= n_dense_layers

        if is_moe_layer:
            # Router gate
            mapping[f'{nmoe_prefix}.ffn.router.gate.weight'] = f'{hf_prefix}.mlp.gate.weight'

            # Router bias -> e_score_correction_bias (for aux-free load balancing)
            mapping[f'{nmoe_prefix}.ffn.router.bias'] = f'{hf_prefix}.mlp.gate.e_score_correction_bias'

            # Expert weights: W1 -> gate_proj, W3 -> up_proj, W2 -> down_proj
            # nmoe stores as [n_local, dim, inter_dim] stacked tensors
            # HF stores as individual expert modules
            mapping[f'{nmoe_prefix}.ffn.W1'] = f'{hf_prefix}.mlp.experts.W1'  # Needs reshape
            mapping[f'{nmoe_prefix}.ffn.W3'] = f'{hf_prefix}.mlp.experts.W3'  # Needs reshape
            mapping[f'{nmoe_prefix}.ffn.W2'] = f'{hf_prefix}.mlp.experts.W2'  # Needs reshape

            # Shared experts (if present)
            if n_shared_experts > 0:
                mapping[f'{nmoe_prefix}.ffn._shared.w1.weight'] = f'{hf_prefix}.mlp.shared_experts.gate_proj.weight'
                mapping[f'{nmoe_prefix}.ffn._shared.w3.weight'] = f'{hf_prefix}.mlp.shared_experts.up_proj.weight'
                mapping[f'{nmoe_prefix}.ffn._shared.w2.weight'] = f'{hf_prefix}.mlp.shared_experts.down_proj.weight'
        else:
            # Dense MLP: w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj
            mapping[f'{nmoe_prefix}.ffn.w1.weight'] = f'{hf_prefix}.mlp.gate_proj.weight'
            mapping[f'{nmoe_prefix}.ffn.w3.weight'] = f'{hf_prefix}.mlp.up_proj.weight'
            mapping[f'{nmoe_prefix}.ffn.w2.weight'] = f'{hf_prefix}.mlp.down_proj.weight'

    return mapping


def hf_to_nmoe_weight_mapping(
    n_layers: int,
    n_dense_layers: int = 1,
    n_routed_experts: int = 64,
    n_shared_experts: int = 2,
) -> Dict[str, str]:
    """Create inverse mapping from HuggingFace to nmoe weight names.

    Args:
        n_layers: Total number of transformer layers
        n_dense_layers: Number of initial dense layers
        n_routed_experts: Number of routed experts
        n_shared_experts: Number of shared experts

    Returns:
        Dict mapping HF weight names to nmoe weight names
    """
    forward_mapping = nmoe_to_hf_weight_mapping(
        n_layers, n_dense_layers, n_routed_experts, n_shared_experts
    )
    return {v: k for k, v in forward_mapping.items()}


def get_expert_weight_info(weight_name: str) -> Optional[Dict[str, any]]:
    """Parse expert weight name to extract layer and expert info.

    Args:
        weight_name: Weight name like 'blocks.5.ffn.W1' or 'model.layers.5.mlp.experts.3.gate_proj.weight'

    Returns:
        Dict with 'layer_id', 'expert_id' (if applicable), 'weight_type' (W1/W3/W2)
        or None if not an expert weight
    """
    # nmoe format: blocks.{layer}.ffn.W{1,2,3}
    nmoe_match = re.match(r'blocks\.(\d+)\.ffn\.W([123])', weight_name)
    if nmoe_match:
        return {
            'layer_id': int(nmoe_match.group(1)),
            'expert_id': None,  # nmoe stacks all experts
            'weight_type': f'W{nmoe_match.group(2)}',
            'format': 'nmoe',
        }

    # HF format: model.layers.{layer}.mlp.experts.{expert}.{proj}.weight
    hf_match = re.match(
        r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight',
        weight_name
    )
    if hf_match:
        proj_to_w = {'gate': 'W1', 'up': 'W3', 'down': 'W2'}
        return {
            'layer_id': int(hf_match.group(1)),
            'expert_id': int(hf_match.group(2)),
            'weight_type': proj_to_w[hf_match.group(3)],
            'format': 'hf',
        }

    return None


def expand_expert_weights_to_hf(
    nmoe_state_dict: Dict[str, "torch.Tensor"],
    n_layers: int,
    n_dense_layers: int,
    n_experts: int,
) -> Dict[str, "torch.Tensor"]:
    """Expand nmoe stacked expert weights to individual HF expert weights.

    nmoe stores expert weights as [n_local_experts, dim, inter_dim] tensors.
    HF stores them as individual expert.{i}.{proj}.weight tensors.

    Args:
        nmoe_state_dict: State dict with nmoe weight names
        n_layers: Total layers
        n_dense_layers: Dense layers count
        n_experts: Total experts (n_routed_experts)

    Returns:
        State dict with expanded HF-style expert weights
    """
    import torch

    hf_state_dict = {}

    for key, tensor in nmoe_state_dict.items():
        info = get_expert_weight_info(key)

        if info and info['format'] == 'nmoe':
            # This is a stacked expert weight, expand it
            layer_id = info['layer_id']
            weight_type = info['weight_type']

            # Map weight type to HF projection name
            w_to_proj = {'W1': 'gate_proj', 'W3': 'up_proj', 'W2': 'down_proj'}
            proj_name = w_to_proj[weight_type]

            # Expand: [n_experts, dim, inter_dim] -> n_experts individual weights
            n_local = tensor.shape[0]
            for expert_id in range(n_local):
                hf_key = f'model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_name}.weight'
                # nmoe: [n_local, in_dim, out_dim], HF: [out_dim, in_dim]
                # W1, W3: [n_local, dim, inter_dim] -> [inter_dim, dim]
                # W2: [n_local, inter_dim, dim] -> [dim, inter_dim]
                if weight_type in ('W1', 'W3'):
                    hf_state_dict[hf_key] = tensor[expert_id].T.contiguous()
                else:  # W2
                    hf_state_dict[hf_key] = tensor[expert_id].T.contiguous()
        else:
            # Not an expert weight, use standard mapping
            mapping = nmoe_to_hf_weight_mapping(n_layers, n_dense_layers, n_experts)
            if key in mapping:
                hf_key = mapping[key]
                # Handle router bias specially
                if 'router.bias' in key:
                    # e_score_correction_bias is the HF name
                    hf_state_dict[hf_key] = tensor
                else:
                    hf_state_dict[hf_key] = tensor
            else:
                # Keep original key if no mapping found
                hf_state_dict[key] = tensor

    return hf_state_dict


def validate_weight_mapping(
    nmoe_keys: List[str],
    hf_keys: List[str],
    n_layers: int,
    n_dense_layers: int,
    n_experts: int,
    n_shared_experts: int = 2,
) -> Tuple[List[str], List[str], List[str]]:
    """Validate weight mapping between nmoe and HF state dicts.

    Args:
        nmoe_keys: List of nmoe weight keys
        hf_keys: List of expected HF weight keys
        n_layers: Total layers
        n_dense_layers: Dense layers
        n_experts: Routed experts
        n_shared_experts: Shared experts

    Returns:
        Tuple of (matched_keys, missing_in_hf, extra_in_hf)
    """
    mapping = nmoe_to_hf_weight_mapping(n_layers, n_dense_layers, n_experts, n_shared_experts)

    nmoe_set = set(nmoe_keys)
    hf_set = set(hf_keys)

    matched = []
    missing = []

    for nmoe_key in nmoe_keys:
        if nmoe_key in mapping:
            hf_key = mapping[nmoe_key]
            # Handle expert weight expansion
            if '.W1' in nmoe_key or '.W2' in nmoe_key or '.W3' in nmoe_key:
                # These get expanded to multiple keys
                matched.append(nmoe_key)
            elif hf_key in hf_set:
                matched.append(nmoe_key)
            else:
                missing.append(nmoe_key)
        else:
            missing.append(nmoe_key)

    # Find extra keys in HF that weren't mapped
    mapped_hf_keys = set(mapping.values())
    extra = [k for k in hf_keys if k not in mapped_hf_keys]

    return matched, missing, extra
