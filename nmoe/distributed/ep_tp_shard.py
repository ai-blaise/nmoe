"""EP+TP weight sharding utilities for nmoe.

This module provides utilities for sharding and gathering expert weights
across combined Expert Parallelism (EP) and Tensor Parallelism (TP) configurations.

Weight Sharding Strategy:
- EP: Experts are distributed across EP ranks (each rank owns a subset of experts)
- TP: Each expert's weights are further sharded across TP ranks (column/row parallel)

For a model with 256 experts, EP=4, TP=2:
- EP rank 0: experts 0-63, TP-sharded across 2 ranks
- EP rank 1: experts 64-127, TP-sharded across 2 ranks
- etc.

Usage:
    from nmoe.distributed import (
        shard_expert_weights,
        gather_expert_weights,
        get_shard_info,
    )

    # Shard weights for distributed training
    info = get_shard_info(n_experts=256, ep_size=4, tp_size=2)
    sharded_W1 = shard_expert_weights(
        W1, info, ep_rank=0, tp_rank=0, weight_type="gate"
    )

    # Gather weights back to full tensor
    full_W1 = gather_expert_weights(sharded_W1, info, weight_type="gate")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


@dataclass
class ShardInfo:
    """Information about weight sharding configuration.

    Attributes:
        n_total_experts: Total number of experts in the model.
        ep_size: Number of Expert Parallelism ranks.
        tp_size: Number of Tensor Parallelism ranks.
        n_local_experts: Number of experts per EP rank.
        expert_dim_per_tp: Expert hidden dim per TP rank (for TP-sharded dims).
    """
    n_total_experts: int
    ep_size: int
    tp_size: int
    n_local_experts: int

    # Optional dimension info (set when sharding specific weights)
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None

    @property
    def expert_dim_per_tp(self) -> int:
        """Intermediate dimension per TP rank."""
        if self.intermediate_size is None:
            raise ValueError("intermediate_size not set in ShardInfo")
        assert self.intermediate_size % self.tp_size == 0, \
            f"intermediate_size ({self.intermediate_size}) must be divisible by tp_size ({self.tp_size})"
        return self.intermediate_size // self.tp_size

    @property
    def hidden_dim_per_tp(self) -> int:
        """Hidden dimension per TP rank (for row-parallel weights)."""
        if self.hidden_size is None:
            raise ValueError("hidden_size not set in ShardInfo")
        assert self.hidden_size % self.tp_size == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by tp_size ({self.tp_size})"
        return self.hidden_size // self.tp_size

    def get_expert_range(self, ep_rank: int) -> Tuple[int, int]:
        """Get the expert index range for a given EP rank.

        Args:
            ep_rank: EP rank (0 to ep_size-1).

        Returns:
            (start_idx, end_idx) tuple for experts owned by this EP rank.
        """
        start = ep_rank * self.n_local_experts
        end = start + self.n_local_experts
        return start, end

    def __repr__(self) -> str:
        return (
            f"ShardInfo(experts={self.n_total_experts}, "
            f"ep={self.ep_size}, tp={self.tp_size}, "
            f"local_experts={self.n_local_experts})"
        )


def get_shard_info(
    n_total_experts: int,
    ep_size: int,
    tp_size: int = 1,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
) -> ShardInfo:
    """Create ShardInfo for a given parallelism configuration.

    Args:
        n_total_experts: Total number of experts.
        ep_size: Number of EP ranks.
        tp_size: Number of TP ranks (default 1 = no TP).
        hidden_size: Model hidden dimension (optional, for TP sharding).
        intermediate_size: MLP intermediate dimension (optional, for TP sharding).

    Returns:
        ShardInfo with computed sharding parameters.

    Raises:
        ValueError: If experts not evenly divisible by ep_size.
    """
    if n_total_experts % ep_size != 0:
        raise ValueError(
            f"n_total_experts ({n_total_experts}) must be divisible by "
            f"ep_size ({ep_size})"
        )

    n_local_experts = n_total_experts // ep_size

    return ShardInfo(
        n_total_experts=n_total_experts,
        ep_size=ep_size,
        tp_size=tp_size,
        n_local_experts=n_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def shard_expert_weights(
    weights: torch.Tensor,
    info: ShardInfo,
    ep_rank: int,
    tp_rank: int = 0,
    weight_type: str = "gate",
) -> torch.Tensor:
    """Shard expert weights for EP+TP parallelism.

    For MoE layers, there are three weight matrices per expert:
    - W1 (gate_proj): [n_experts, hidden_size, intermediate_size] - column parallel
    - W3 (up_proj): [n_experts, hidden_size, intermediate_size] - column parallel
    - W2 (down_proj): [n_experts, intermediate_size, hidden_size] - row parallel

    TP sharding:
    - Column parallel (W1, W3): shard along intermediate_size dimension
    - Row parallel (W2): shard along intermediate_size dimension (input dim)

    Args:
        weights: Full weight tensor [n_experts, dim1, dim2].
        info: ShardInfo with parallelism configuration.
        ep_rank: This rank's EP index.
        tp_rank: This rank's TP index.
        weight_type: "gate" (W1), "up" (W3), or "down" (W2).

    Returns:
        Sharded weight tensor for this EP+TP rank.
    """
    assert weight_type in ("gate", "up", "down"), f"Unknown weight_type: {weight_type}"
    n_experts, dim1, dim2 = weights.shape

    # Step 1: EP sharding - select local experts
    expert_start, expert_end = info.get_expert_range(ep_rank)
    local_weights = weights[expert_start:expert_end].contiguous()

    # Step 2: TP sharding (if tp_size > 1)
    if info.tp_size == 1:
        return local_weights

    n_local = local_weights.shape[0]

    if weight_type in ("gate", "up"):
        # Column parallel: shard along dim2 (intermediate_size)
        assert dim2 % info.tp_size == 0, \
            f"dim2 ({dim2}) must be divisible by tp_size ({info.tp_size})"
        shard_size = dim2 // info.tp_size
        shard_start = tp_rank * shard_size
        shard_end = shard_start + shard_size
        sharded = local_weights[:, :, shard_start:shard_end].contiguous()
        logger.debug(
            f"Column parallel shard: [{n_local}, {dim1}, {dim2}] -> "
            f"[{n_local}, {dim1}, {shard_size}] (tp_rank={tp_rank})"
        )
    else:  # "down"
        # Row parallel: shard along dim1 (input dimension = intermediate_size)
        assert dim1 % info.tp_size == 0, \
            f"dim1 ({dim1}) must be divisible by tp_size ({info.tp_size})"
        shard_size = dim1 // info.tp_size
        shard_start = tp_rank * shard_size
        shard_end = shard_start + shard_size
        sharded = local_weights[:, shard_start:shard_end, :].contiguous()
        logger.debug(
            f"Row parallel shard: [{n_local}, {dim1}, {dim2}] -> "
            f"[{n_local}, {shard_size}, {dim2}] (tp_rank={tp_rank})"
        )

    return sharded


def gather_expert_weights(
    sharded_weights: torch.Tensor,
    info: ShardInfo,
    weight_type: str = "gate",
    ep_group: Optional[ProcessGroup] = None,
    tp_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """Gather sharded expert weights back to full tensor.

    This is the inverse of shard_expert_weights. Used for:
    - Checkpoint saving (gather to full then save)
    - Inference with different parallelism config
    - Debugging and validation

    Args:
        sharded_weights: Sharded weight tensor from this rank.
        info: ShardInfo with parallelism configuration.
        weight_type: "gate", "up", or "down".
        ep_group: EP process group for all_gather (optional).
        tp_group: TP process group for all_gather (optional).

    Returns:
        Full weight tensor [n_total_experts, dim1, dim2].

    Note:
        If process groups are not provided, returns the local shard
        (no distributed communication).
    """
    import torch.distributed as dist

    assert weight_type in ("gate", "up", "down"), f"Unknown weight_type: {weight_type}"

    # Get local shape
    n_local, dim1, dim2 = sharded_weights.shape

    # Step 1: Gather across TP (if tp_size > 1 and tp_group provided)
    if info.tp_size > 1 and tp_group is not None:
        # Allocate buffer for all TP shards
        tp_gathered = [torch.empty_like(sharded_weights) for _ in range(info.tp_size)]
        dist.all_gather(tp_gathered, sharded_weights, group=tp_group)

        # Concatenate along the sharded dimension
        if weight_type in ("gate", "up"):
            # Column parallel: concat along dim2
            local_full = torch.cat(tp_gathered, dim=2)
        else:  # "down"
            # Row parallel: concat along dim1
            local_full = torch.cat(tp_gathered, dim=1)

        logger.debug(f"TP gather: {sharded_weights.shape} x {info.tp_size} -> {local_full.shape}")
    else:
        local_full = sharded_weights

    # Step 2: Gather across EP (if ep_size > 1 and ep_group provided)
    if info.ep_size > 1 and ep_group is not None:
        # Allocate buffer for all EP shards
        ep_gathered = [torch.empty_like(local_full) for _ in range(info.ep_size)]
        dist.all_gather(ep_gathered, local_full, group=ep_group)

        # Concatenate along expert dimension
        full_weights = torch.cat(ep_gathered, dim=0)
        logger.debug(f"EP gather: {local_full.shape} x {info.ep_size} -> {full_weights.shape}")
    else:
        full_weights = local_full

    return full_weights.contiguous()


def validate_shard_roundtrip(
    original: torch.Tensor,
    info: ShardInfo,
    weight_type: str = "gate",
) -> bool:
    """Validate that shard->gather produces the original tensor.

    This is a local validation that doesn't require distributed.
    It simulates the sharding by manually concatenating shards.

    Args:
        original: Original full weight tensor.
        info: ShardInfo configuration.
        weight_type: Weight type to test.

    Returns:
        True if roundtrip is successful, False otherwise.
    """
    # Shard for each EP+TP rank
    all_shards: List[List[torch.Tensor]] = []
    for ep_rank in range(info.ep_size):
        tp_shards = []
        for tp_rank in range(info.tp_size):
            shard = shard_expert_weights(
                original, info, ep_rank, tp_rank, weight_type
            )
            tp_shards.append(shard)
        all_shards.append(tp_shards)

    # Reconstruct: first gather TP, then EP
    ep_reconstructed = []
    for ep_rank in range(info.ep_size):
        if info.tp_size > 1:
            if weight_type in ("gate", "up"):
                tp_gathered = torch.cat(all_shards[ep_rank], dim=2)
            else:
                tp_gathered = torch.cat(all_shards[ep_rank], dim=1)
        else:
            tp_gathered = all_shards[ep_rank][0]
        ep_reconstructed.append(tp_gathered)

    reconstructed = torch.cat(ep_reconstructed, dim=0)

    # Compare
    if not torch.allclose(original, reconstructed):
        logger.error(
            f"Shard roundtrip failed for {weight_type}: "
            f"original shape {original.shape}, reconstructed shape {reconstructed.shape}"
        )
        return False

    logger.debug(f"Shard roundtrip validated for {weight_type}")
    return True


def get_tp_shard_dim(weight_type: str) -> int:
    """Get the dimension index that is TP-sharded for a given weight type.

    Args:
        weight_type: "gate", "up", or "down".

    Returns:
        Dimension index (1 or 2) that is sharded across TP.
    """
    if weight_type in ("gate", "up"):
        return 2  # Column parallel: shard output dim
    else:  # "down"
        return 1  # Row parallel: shard input dim


def compute_shard_sizes(
    total_experts: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    tp_size: int,
) -> dict:
    """Compute the sizes of sharded weight tensors.

    Useful for pre-allocating buffers or verifying tensor shapes.

    Args:
        total_experts: Total number of experts.
        hidden_size: Model hidden dimension.
        intermediate_size: MLP intermediate dimension.
        ep_size: Number of EP ranks.
        tp_size: Number of TP ranks.

    Returns:
        Dict with shapes for each weight type per rank.
    """
    n_local = total_experts // ep_size
    inter_per_tp = intermediate_size // tp_size

    return {
        "gate": (n_local, hidden_size, inter_per_tp),      # W1: [E/ep, H, Dff/tp]
        "up": (n_local, hidden_size, inter_per_tp),        # W3: [E/ep, H, Dff/tp]
        "down": (n_local, inter_per_tp, hidden_size),      # W2: [E/ep, Dff/tp, H]
        "total_experts": total_experts,
        "local_experts": n_local,
        "ep_size": ep_size,
        "tp_size": tp_size,
    }
