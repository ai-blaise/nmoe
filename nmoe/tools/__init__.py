"""nmoe tools for model export, conversion, and checkpoint management."""

from .config_converter import (
    nmoe_to_hf_weight_mapping,
    hf_to_nmoe_weight_mapping,
    get_expert_weight_info,
    expand_expert_weights_to_hf,
    validate_weight_mapping,
)

from .reshard_checkpoint import (
    reshard_checkpoint,
    load_and_gather_experts,
    shard_experts_for_rank,
    detect_ep_size,
    detect_n_experts,
)

from .import_from_hf import (
    import_nvfp4_to_nmoe,
    validate_import,
    build_hf_to_nmoe_base_mapping,
    parse_expert_nvfp4_key,
    stack_expert_nvfp4_triplets,
    shard_expert_tensor,
)

__all__ = [
    # Config converter
    "nmoe_to_hf_weight_mapping",
    "hf_to_nmoe_weight_mapping",
    "get_expert_weight_info",
    "expand_expert_weights_to_hf",
    "validate_weight_mapping",
    # Checkpoint resharding
    "reshard_checkpoint",
    "load_and_gather_experts",
    "shard_experts_for_rank",
    "detect_ep_size",
    "detect_n_experts",
    # HF import
    "import_nvfp4_to_nmoe",
    "validate_import",
    "build_hf_to_nmoe_base_mapping",
    "parse_expert_nvfp4_key",
    "stack_expert_nvfp4_triplets",
    "shard_expert_tensor",
]
