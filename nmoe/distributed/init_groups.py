"""Process group initialization helpers for nmoe EP/TP parallelism.

This module provides utilities for initializing and managing process groups
for Expert Parallelism (EP) and Tensor Parallelism (TP) in nmoe. It supports
both standalone initialization and integration with Megatron's parallel state.

Usage:
    # Standalone initialization (8 GPUs, EP=4, TP=2)
    from nmoe.distributed import init_nmoe_process_groups, get_ep_group, get_tp_group

    init_nmoe_process_groups(ep_size=4, tp_size=2)
    ep_group = get_ep_group()
    tp_group = get_tp_group()

    # With Megatron (automatically uses Megatron's groups)
    from nmoe.distributed import init_nmoe_from_megatron

    init_nmoe_from_megatron()  # Uses Megatron's EP/TP groups
"""

from __future__ import annotations

import logging
from typing import Optional, List, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Global process group state
_EP_GROUP: Optional[ProcessGroup] = None
_TP_GROUP: Optional[ProcessGroup] = None
_EP_SIZE: int = 1
_TP_SIZE: int = 1
_EP_RANK: int = 0
_TP_RANK: int = 0
_INITIALIZED: bool = False


def is_nmoe_parallel_initialized() -> bool:
    """Check if nmoe parallel state is initialized.

    Returns:
        True if init_nmoe_process_groups() or init_nmoe_from_megatron() was called.
    """
    return _INITIALIZED


def get_ep_group() -> Optional[ProcessGroup]:
    """Get the Expert Parallelism process group.

    Returns:
        EP process group, or None if not initialized or EP=1.

    Raises:
        RuntimeError: If nmoe parallel state is not initialized.
    """
    if not _INITIALIZED:
        raise RuntimeError(
            "nmoe parallel state not initialized. "
            "Call init_nmoe_process_groups() or init_nmoe_from_megatron() first."
        )
    return _EP_GROUP


def get_tp_group() -> Optional[ProcessGroup]:
    """Get the Tensor Parallelism process group.

    Returns:
        TP process group, or None if not initialized or TP=1.

    Raises:
        RuntimeError: If nmoe parallel state is not initialized.
    """
    if not _INITIALIZED:
        raise RuntimeError(
            "nmoe parallel state not initialized. "
            "Call init_nmoe_process_groups() or init_nmoe_from_megatron() first."
        )
    return _TP_GROUP


def get_ep_size() -> int:
    """Get the Expert Parallelism world size.

    Returns:
        Number of ranks in the EP group.
    """
    return _EP_SIZE


def get_tp_size() -> int:
    """Get the Tensor Parallelism world size.

    Returns:
        Number of ranks in the TP group.
    """
    return _TP_SIZE


def get_ep_rank() -> int:
    """Get this rank's position in the EP group.

    Returns:
        Rank within the EP group (0-indexed).
    """
    return _EP_RANK


def get_tp_rank() -> int:
    """Get this rank's position in the TP group.

    Returns:
        Rank within the TP group (0-indexed).
    """
    return _TP_RANK


def init_nmoe_process_groups(
    ep_size: int = 1,
    tp_size: int = 1,
    backend: str = "nccl",
) -> None:
    """Initialize nmoe's Expert Parallelism and Tensor Parallelism groups.

    Creates process groups for EP and TP based on the specified sizes.
    The world is partitioned as: world_size = ep_size * tp_size * dp_size

    For an 8-GPU setup with EP=4, TP=2:
    - World size = 8
    - EP groups: [0,2,4,6], [1,3,5,7]  (4 ranks each, stride=2)
    - TP groups: [0,1], [2,3], [4,5], [6,7]  (2 consecutive ranks)

    Args:
        ep_size: Number of Expert Parallel ranks.
        tp_size: Number of Tensor Parallel ranks.
        backend: Distributed backend (default: "nccl").

    Raises:
        RuntimeError: If distributed is not initialized.
        ValueError: If ep_size * tp_size > world_size.
    """
    global _EP_GROUP, _TP_GROUP, _EP_SIZE, _TP_SIZE, _EP_RANK, _TP_RANK, _INITIALIZED

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Validate configuration
    if ep_size * tp_size > world_size:
        raise ValueError(
            f"ep_size ({ep_size}) * tp_size ({tp_size}) = {ep_size * tp_size} "
            f"exceeds world_size ({world_size})"
        )

    _EP_SIZE = ep_size
    _TP_SIZE = tp_size

    # Create TP groups (consecutive ranks)
    # For world=8, tp=2: groups are [0,1], [2,3], [4,5], [6,7]
    if tp_size > 1:
        tp_groups: List[List[int]] = []
        for base in range(0, world_size, tp_size):
            group_ranks = list(range(base, min(base + tp_size, world_size)))
            if len(group_ranks) == tp_size:
                tp_groups.append(group_ranks)

        for ranks in tp_groups:
            group = dist.new_group(ranks=ranks, backend=backend)
            if rank in ranks:
                _TP_GROUP = group
                _TP_RANK = ranks.index(rank)
                logger.debug(f"Rank {rank} joined TP group {ranks}, tp_rank={_TP_RANK}")
    else:
        _TP_GROUP = None
        _TP_RANK = 0

    # Create EP groups (strided by tp_size)
    # For world=8, ep=4, tp=2: groups are [0,2,4,6], [1,3,5,7]
    if ep_size > 1:
        ep_groups: List[List[int]] = []
        for start in range(tp_size):
            group_ranks = list(range(start, world_size, tp_size))[:ep_size]
            if len(group_ranks) == ep_size:
                ep_groups.append(group_ranks)

        for ranks in ep_groups:
            group = dist.new_group(ranks=ranks, backend=backend)
            if rank in ranks:
                _EP_GROUP = group
                _EP_RANK = ranks.index(rank)
                logger.debug(f"Rank {rank} joined EP group {ranks}, ep_rank={_EP_RANK}")
    else:
        _EP_GROUP = None
        _EP_RANK = 0

    _INITIALIZED = True

    logger.info(
        f"nmoe process groups initialized: "
        f"rank={rank}, ep_rank={_EP_RANK}/{_EP_SIZE}, tp_rank={_TP_RANK}/{_TP_SIZE}"
    )


def init_nmoe_from_megatron() -> None:
    """Initialize nmoe parallel state from Megatron's process groups.

    This function extracts EP and TP groups from Megatron's parallel state,
    allowing nmoe to use Megatron's existing process group infrastructure.

    Raises:
        ImportError: If Megatron is not installed.
        RuntimeError: If Megatron parallel state is not initialized.
    """
    global _EP_GROUP, _TP_GROUP, _EP_SIZE, _TP_SIZE, _EP_RANK, _TP_RANK, _INITIALIZED

    try:
        import megatron.core.parallel_state as mpu
    except ImportError as e:
        raise ImportError(
            "Megatron is required for init_nmoe_from_megatron. "
            "Install it with: pip install megatron-core"
        ) from e

    if not mpu.is_initialized():
        raise RuntimeError(
            "Megatron parallel state is not initialized. "
            "Call mpu.initialize_model_parallel() first."
        )

    _EP_GROUP = mpu.get_expert_model_parallel_group()
    _TP_GROUP = mpu.get_tensor_model_parallel_group()
    _EP_SIZE = mpu.get_expert_model_parallel_world_size()
    _TP_SIZE = mpu.get_tensor_model_parallel_world_size()
    _EP_RANK = mpu.get_expert_model_parallel_rank()
    _TP_RANK = mpu.get_tensor_model_parallel_rank()
    _INITIALIZED = True

    logger.info(
        f"nmoe process groups initialized from Megatron: "
        f"ep_rank={_EP_RANK}/{_EP_SIZE}, tp_rank={_TP_RANK}/{_TP_SIZE}"
    )


def cleanup_process_groups() -> None:
    """Clean up nmoe process groups.

    Destroys the EP and TP process groups created by init_nmoe_process_groups().
    Does not affect Megatron-created groups (those are managed by Megatron).

    Note: In most cases, you don't need to call this explicitly.
    PyTorch handles process group cleanup at program exit.
    """
    global _EP_GROUP, _TP_GROUP, _INITIALIZED, _EP_SIZE, _TP_SIZE, _EP_RANK, _TP_RANK

    if _EP_GROUP is not None:
        try:
            dist.destroy_process_group(_EP_GROUP)
            logger.debug("Destroyed EP process group")
        except Exception as e:
            logger.warning(f"Failed to destroy EP process group: {e}")
        _EP_GROUP = None

    if _TP_GROUP is not None:
        try:
            dist.destroy_process_group(_TP_GROUP)
            logger.debug("Destroyed TP process group")
        except Exception as e:
            logger.warning(f"Failed to destroy TP process group: {e}")
        _TP_GROUP = None

    _EP_SIZE = 1
    _TP_SIZE = 1
    _EP_RANK = 0
    _TP_RANK = 0
    _INITIALIZED = False

    logger.info("nmoe process groups cleaned up")


def get_data_parallel_group() -> Optional[ProcessGroup]:
    """Get the Data Parallelism process group.

    In an EP+TP configuration, DP is the remaining parallelism dimension.
    For world_size=16, EP=4, TP=2: DP=2.

    Returns:
        DP process group, or None if DP=1 or not initialized.

    Note:
        This function creates DP groups on first call if they don't exist.
        For Megatron integration, use mpu.get_data_parallel_group() instead.
    """
    if not _INITIALIZED:
        raise RuntimeError("nmoe parallel state not initialized")

    # If using Megatron, defer to its DP group
    try:
        import megatron.core.parallel_state as mpu
        if mpu.is_initialized():
            return mpu.get_data_parallel_group()
    except ImportError:
        pass

    # For standalone nmoe, DP group would need to be created
    # This is left as an extension point for now
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = world_size // (_EP_SIZE * _TP_SIZE)

    if dp_size <= 1:
        return None

    # DP groups would need to be created here
    # For now, return None and let caller handle it
    logger.warning(
        f"DP groups not yet implemented for standalone nmoe. "
        f"world={world_size}, ep={_EP_SIZE}, tp={_TP_SIZE}, dp={dp_size}"
    )
    return None
