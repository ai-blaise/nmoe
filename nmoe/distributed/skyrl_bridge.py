"""SkyRL/Megatron bridge for nmoe RDEP dispatcher.

This module provides a bridge between SkyRL/Megatron's expert-parallel
process groups and nmoe's Rdep dispatcher. This enables using nmoe's
high-performance MoE dispatch with Megatron's expert parallelism.

Usage:
    from nmoe.distributed import SkyRLRdepBridge

    # With Megatron process groups
    bridge = SkyRLRdepBridge(
        dim=4096,
        n_total_experts=256,
        topk=8,
        ep_group=mpu.get_expert_model_parallel_group(),
        tp_group=mpu.get_tensor_model_parallel_group(),
    )

    # Use bridge for MoE forward
    output = bridge.moe_forward(hidden_states, expert_ids, gates, W1, W3, W2)
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

from nmoe.rdep import Rdep, _get_group_world_size, _get_group_rank

logger = logging.getLogger(__name__)


class SkyRLRdepBridge:
    """Bridge between SkyRL/Megatron process groups and nmoe RDEP.

    This class handles the mapping between Megatron-style process groups
    (expert_model_parallel_group, tensor_model_parallel_group) and nmoe's
    Rdep dispatcher. It ensures correct rank mapping and weight sharding
    for expert parallelism.

    Key responsibilities:
    - Map Megatron EP ranks to nmoe RDEP ranks
    - Create Rdep instance with proper ep_group
    - Handle expert weight distribution across EP ranks
    - Support combined EP+TP configurations

    Attributes:
        rdep: The underlying nmoe Rdep instance.
        ep_size: Number of expert-parallel ranks.
        ep_rank: This rank's position within the EP group.
        tp_size: Number of tensor-parallel ranks.
        tp_rank: This rank's position within the TP group.
        n_local_experts: Number of experts on this rank.
    """

    def __init__(
        self,
        dim: int,
        n_total_experts: int,
        topk: int,
        ep_group: Optional[ProcessGroup] = None,
        tp_group: Optional[ProcessGroup] = None,
        profile: str = "bf16",
        capacity: int = 65536,
    ):
        """Initialize the SkyRL/Megatron RDEP bridge.

        Args:
            dim: Hidden dimension size.
            n_total_experts: Total number of experts in the model.
            topk: Number of experts activated per token.
            ep_group: Expert-parallel process group. If None, uses default.
            tp_group: Tensor-parallel process group. If None, no TP.
            profile: Quantization profile - 'bf16', 'fp8', or 'nvfp4'.
            capacity: Maximum tokens per RDEP buffer.

        Raises:
            ValueError: If n_total_experts is not divisible by ep_size.
        """
        self.dim = dim
        self.n_total_experts = n_total_experts
        self.topk = topk
        self.profile = profile
        self.capacity = capacity

        # Get EP group info
        self.ep_group = ep_group
        self.ep_size = _get_group_world_size(ep_group)
        self.ep_rank = _get_group_rank(ep_group)

        # Get TP group info
        self.tp_group = tp_group
        self.tp_size = _get_group_world_size(tp_group) if tp_group else 1
        self.tp_rank = _get_group_rank(tp_group) if tp_group else 0

        # Calculate local experts for this EP rank
        if n_total_experts % self.ep_size != 0:
            raise ValueError(
                f"n_total_experts ({n_total_experts}) must be divisible by "
                f"ep_size ({self.ep_size})"
            )
        self.n_local_experts = n_total_experts // self.ep_size

        # Expert range for this rank
        self.expert_start = self.ep_rank * self.n_local_experts
        self.expert_end = self.expert_start + self.n_local_experts

        logger.info(
            f"SkyRLRdepBridge initialized: ep_rank={self.ep_rank}/{self.ep_size}, "
            f"tp_rank={self.tp_rank}/{self.tp_size}, "
            f"experts={self.expert_start}-{self.expert_end} "
            f"({self.n_local_experts} local)"
        )

        # Create Rdep with EP group
        self._rdep: Optional[Rdep] = None

    def _ensure_rdep(self) -> Rdep:
        """Lazily initialize Rdep on first use.

        Deferred initialization allows process groups to be fully set up
        before creating the RDEP buffers and IPC handles.
        """
        if self._rdep is None:
            self._rdep = Rdep(
                dim=self.dim,
                n_local=self.n_local_experts,
                topk=self.topk,
                profile=self.profile,
                capacity=self.capacity,
                ep_group=self.ep_group,
            )
            logger.info(
                f"Created Rdep instance for SkyRLRdepBridge: "
                f"mode={self._rdep._mode}, world={self._rdep.world}, rank={self._rdep.rank}"
            )
        return self._rdep

    @property
    def rdep(self) -> Rdep:
        """Get the underlying Rdep instance (creates if needed)."""
        return self._ensure_rdep()

    @property
    def mode(self) -> str:
        """Get the RDEP mode (single, ipc, or hybrid)."""
        return self.rdep._mode

    def get_local_expert_weights(
        self,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """Extract local expert weights from global weight tensor.

        Given a global expert weight tensor [E, ...], extracts the slice
        corresponding to this EP rank's local experts.

        Args:
            W: Global expert weights [n_total_experts, ...].

        Returns:
            Local expert weights [n_local_experts, ...].

        Example:
            # With EP=4 and 256 total experts
            # Rank 0 gets experts 0-63
            # Rank 1 gets experts 64-127
            # etc.
            local_W1 = bridge.get_local_expert_weights(global_W1)
        """
        return W[self.expert_start:self.expert_end].contiguous()

    def map_expert_ids_to_local(
        self,
        global_expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map global expert IDs to local indices.

        Converts global expert indices to local indices for this EP rank.
        Experts not on this rank are masked to -1.

        Args:
            global_expert_ids: [T, K] global expert indices.

        Returns:
            [T, K] local expert indices (-1 for non-local experts).

        Note:
            This is mainly for debugging/validation. The Rdep dispatcher
            handles global->local mapping internally based on ep_rank.
        """
        local_ids = global_expert_ids.clone()

        # Mask experts not on this rank
        mask_below = local_ids < self.expert_start
        mask_above = local_ids >= self.expert_end
        non_local = mask_below | mask_above

        # Convert to local indices
        local_ids = local_ids - self.expert_start
        local_ids[non_local] = -1

        return local_ids

    def moe_bf16(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        gates: torch.Tensor,
        W1: torch.Tensor,
        W3: torch.Tensor,
        W2: torch.Tensor,
    ) -> torch.Tensor:
        """Execute MoE forward with BF16 precision.

        Wrapper around Rdep.moe_bf16 that ensures proper initialization.

        Args:
            x: Hidden states [T, H] in BF16.
            expert_ids: Expert indices [T, K] (global indices).
            gates: Routing weights [T, K] in BF16.
            W1: Gate projection weights [n_local_experts, H, Dff].
            W3: Up projection weights [n_local_experts, H, Dff].
            W2: Down projection weights [n_local_experts, Dff, H].

        Returns:
            Output hidden states [T, H] in BF16.
        """
        return self.rdep.moe_bf16(x, expert_ids, gates, W1, W3, W2)

    def moe_blockscaled(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        gates: torch.Tensor,
        W1: torch.Tensor,
        W3: torch.Tensor,
        W2: torch.Tensor,
        W_cache,
    ) -> torch.Tensor:
        """Execute MoE forward with blockscaled (FP8/NVFP4) precision.

        Wrapper around Rdep.moe_blockscaled that ensures proper initialization.

        Args:
            x: Hidden states [T, H] in BF16.
            expert_ids: Expert indices [T, K] (global indices).
            gates: Routing weights [T, K] in BF16.
            W1: Gate projection weights [n_local_experts, H, Dff].
            W3: Up projection weights [n_local_experts, H, Dff].
            W2: Down projection weights [n_local_experts, Dff, H].
            W_cache: Pre-computed quantized weight cache.

        Returns:
            Output hidden states [T, H] in BF16.
        """
        return self.rdep.moe_blockscaled(x, expert_ids, gates, W1, W3, W2, W_cache)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SkyRLRdepBridge("
            f"ep={self.ep_rank}/{self.ep_size}, "
            f"tp={self.tp_rank}/{self.tp_size}, "
            f"experts={self.expert_start}:{self.expert_end}, "
            f"profile={self.profile}"
            f")"
        )


def create_bridge_from_megatron(
    dim: int,
    n_total_experts: int,
    topk: int,
    profile: str = "bf16",
    capacity: int = 65536,
) -> SkyRLRdepBridge:
    """Create SkyRLRdepBridge from Megatron process groups.

    This factory function automatically extracts the expert-parallel and
    tensor-parallel process groups from Megatron's parallel state, creating
    a bridge that integrates nmoe's RDEP dispatcher with Megatron's
    distributed training infrastructure.

    Args:
        dim: Hidden dimension size.
        n_total_experts: Total number of experts in the model.
        topk: Number of experts activated per token.
        profile: Quantization profile - 'bf16', 'fp8', or 'nvfp4'.
        capacity: Maximum tokens per RDEP buffer.

    Returns:
        SkyRLRdepBridge configured with Megatron's process groups.

    Raises:
        ImportError: If Megatron is not installed.
        RuntimeError: If Megatron parallel state is not initialized.

    Example:
        # After Megatron parallel state is initialized:
        bridge = create_bridge_from_megatron(
            dim=4096,
            n_total_experts=256,
            topk=8,
        )

        # Use in MoE layer:
        output = bridge.moe_bf16(hidden, expert_ids, gates, W1, W3, W2)
    """
    try:
        import megatron.core.parallel_state as mpu
    except ImportError as e:
        raise ImportError(
            "Megatron is required for create_bridge_from_megatron. "
            "Install it with: pip install megatron-core"
        ) from e

    # Check that Megatron is initialized
    if not mpu.is_initialized():
        raise RuntimeError(
            "Megatron parallel state is not initialized. "
            "Call mpu.initialize_model_parallel() first."
        )

    # Get Megatron's process groups
    ep_group = mpu.get_expert_model_parallel_group()
    tp_group = mpu.get_tensor_model_parallel_group()

    logger.info(
        f"Creating SkyRLRdepBridge from Megatron: "
        f"ep_size={mpu.get_expert_model_parallel_world_size()}, "
        f"tp_size={mpu.get_tensor_model_parallel_world_size()}"
    )

    return SkyRLRdepBridge(
        dim=dim,
        n_total_experts=n_total_experts,
        topk=topk,
        ep_group=ep_group,
        tp_group=tp_group,
        profile=profile,
        capacity=capacity,
    )


def get_nmoe_ep_rank() -> int:
    """Get the current rank's position in the expert-parallel group.

    This function works with or without Megatron. If Megatron is initialized,
    it uses Megatron's EP rank. Otherwise, it falls back to torch.distributed.

    Returns:
        EP rank, or 0 if distributed is not initialized.
    """
    try:
        import megatron.core.parallel_state as mpu
        if mpu.is_initialized():
            return mpu.get_expert_model_parallel_rank()
    except ImportError:
        pass

    # Fallback to default process group
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_nmoe_ep_world_size() -> int:
    """Get the world size of the expert-parallel group.

    This function works with or without Megatron. If Megatron is initialized,
    it uses Megatron's EP world size. Otherwise, it falls back to torch.distributed.

    Returns:
        EP world size, or 1 if distributed is not initialized.
    """
    try:
        import megatron.core.parallel_state as mpu
        if mpu.is_initialized():
            return mpu.get_expert_model_parallel_world_size()
    except ImportError:
        pass

    # Fallback to default process group
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
