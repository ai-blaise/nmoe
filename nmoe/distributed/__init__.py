"""Distributed utilities for nmoe integration with external frameworks.

This package provides bridges between nmoe's RDEP dispatcher and
distributed training frameworks like SkyRL and Megatron.
"""

from nmoe.distributed.skyrl_bridge import (
    SkyRLRdepBridge,
    create_bridge_from_megatron,
    get_nmoe_ep_rank,
    get_nmoe_ep_world_size,
)

from nmoe.distributed.init_groups import (
    init_nmoe_process_groups,
    init_nmoe_from_megatron,
    cleanup_process_groups,
    is_nmoe_parallel_initialized,
    get_ep_group,
    get_tp_group,
    get_ep_size,
    get_tp_size,
    get_ep_rank,
    get_tp_rank,
    get_data_parallel_group,
)

from nmoe.distributed.ep_tp_shard import (
    ShardInfo,
    get_shard_info,
    shard_expert_weights,
    gather_expert_weights,
    validate_shard_roundtrip,
    get_tp_shard_dim,
    compute_shard_sizes,
)

__all__ = [
    # SkyRL Bridge
    "SkyRLRdepBridge",
    "create_bridge_from_megatron",
    "get_nmoe_ep_rank",
    "get_nmoe_ep_world_size",
    # Process Group Initialization
    "init_nmoe_process_groups",
    "init_nmoe_from_megatron",
    "cleanup_process_groups",
    "is_nmoe_parallel_initialized",
    "get_ep_group",
    "get_tp_group",
    "get_ep_size",
    "get_tp_size",
    "get_ep_rank",
    "get_tp_rank",
    "get_data_parallel_group",
    # EP+TP Weight Sharding
    "ShardInfo",
    "get_shard_info",
    "shard_expert_weights",
    "gather_expert_weights",
    "validate_shard_roundtrip",
    "get_tp_shard_dim",
    "compute_shard_sizes",
]
