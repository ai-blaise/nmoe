"""Process group initialization helpers for nmoe EP/TP/DP parallelism.

This module provides utilities for initializing and managing process groups
for Expert Parallelism (EP), Tensor Parallelism (TP), and Data Parallelism (DP)
in nmoe. It supports both standalone initialization and integration with
Megatron's parallel state.

Layout for EP=8, DP=16, world=128 (no TP):
  EP groups (consecutive):  [0..7], [8..15], ..., [120..127]
  DP groups (strided by EP): [0,8,16,...,120], [1,9,17,...,121], ...

Usage:
    # Standalone initialization (128 GPUs, EP=8)
    from nmoe.distributed import init_nmoe_process_groups, get_ep_group, get_data_parallel_group

    init_nmoe_process_groups(ep_size=8)
    ep_group = get_ep_group()   # 8 ranks for expert AlltoAll
    dp_group = get_data_parallel_group()  # 16 ranks for gradient sync

    # With Megatron (automatically uses Megatron's groups)
    from nmoe.distributed import init_nmoe_from_megatron

    init_nmoe_from_megatron()  # Uses Megatron's EP/TP/DP groups
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional, List, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Global process group state
_EP_GROUP: Optional[ProcessGroup] = None
_TP_GROUP: Optional[ProcessGroup] = None
_DP_GROUP: Optional[ProcessGroup] = None
_EP_SIZE: int = 1
_TP_SIZE: int = 1
_DP_SIZE: int = 1
_EP_RANK: int = 0
_TP_RANK: int = 0
_DP_RANK: int = 0
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


def get_dp_size() -> int:
    """Get the Data Parallelism world size.

    Returns:
        Number of ranks in the DP group.
    """
    return _DP_SIZE


def get_dp_rank() -> int:
    """Get this rank's position in the DP group.

    Returns:
        Rank within the DP group (0-indexed).
    """
    return _DP_RANK


def init_nmoe_process_groups(
    ep_size: int = 1,
    tp_size: int = 1,
    backend: str = "nccl",
) -> None:
    """Initialize nmoe's Expert Parallelism, Tensor Parallelism, and Data Parallelism groups.

    Creates process groups for EP, TP, and DP based on the specified sizes.
    The world is partitioned as: world_size = ep_size * tp_size * dp_size,
    where dp_size = world_size / (ep_size * tp_size).

    Layout (EP-first, consecutive EP ranks):
      For world=128, EP=8, TP=1:
        EP groups (consecutive):  [0..7], [8..15], ..., [120..127]
        DP groups (strided by EP): [0,8,16,...,120], [1,9,17,...,121], ...

      For world=8, EP=4, TP=2:
        TP groups (consecutive):  [0,1], [2,3], [4,5], [6,7]
        EP groups (strided by TP): [0,2,4,6], [1,3,5,7]
        DP groups: dp_size=1, no group needed

    Args:
        ep_size: Number of Expert Parallel ranks.
        tp_size: Number of Tensor Parallel ranks.
        backend: Distributed backend (default: "nccl").

    Raises:
        RuntimeError: If distributed is not initialized.
        ValueError: If ep_size * tp_size > world_size or world_size not divisible.
    """
    global _EP_GROUP, _TP_GROUP, _DP_GROUP
    global _EP_SIZE, _TP_SIZE, _DP_SIZE
    global _EP_RANK, _TP_RANK, _DP_RANK
    global _INITIALIZED

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
    if world_size % (ep_size * tp_size) != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"ep_size ({ep_size}) * tp_size ({tp_size}) = {ep_size * tp_size}"
        )

    dp_size = world_size // (ep_size * tp_size)

    _EP_SIZE = ep_size
    _TP_SIZE = tp_size
    _DP_SIZE = dp_size

    # -------------------------------------------------------------------------
    # Create TP groups (consecutive ranks within each EP rank)
    # For world=8, tp=2: groups are [0,1], [2,3], [4,5], [6,7]
    # -------------------------------------------------------------------------
    if tp_size > 1:
        tp_groups: List[List[int]] = []
        for base in range(0, world_size, tp_size):
            group_ranks = list(range(base, min(base + tp_size, world_size)))
            if len(group_ranks) == tp_size:
                tp_groups.append(group_ranks)

        for ranks in tp_groups:
            group = dist.new_group(ranks=ranks, backend=backend, timeout=timedelta(seconds=1800))
            if rank in ranks:
                _TP_GROUP = group
                _TP_RANK = ranks.index(rank)
                logger.debug(f"Rank {rank} joined TP group {ranks}, tp_rank={_TP_RANK}")
    else:
        _TP_GROUP = None
        _TP_RANK = 0

    # -------------------------------------------------------------------------
    # Create EP groups (consecutive EP ranks, strided by TP)
    #
    # With TP=1 (common SFT case): EP groups are simply consecutive.
    #   world=128, EP=8: [0..7], [8..15], ..., [120..127]
    #
    # With TP>1: EP ranks are interleaved with TP.
    #   world=8, EP=4, TP=2: [0,2,4,6], [1,3,5,7]
    # -------------------------------------------------------------------------
    if ep_size > 1:
        ep_groups: List[List[int]] = []
        # Each DP replica (dp_rank) has ep_size*tp_size consecutive global ranks.
        # Within each DP replica, EP groups are strided by tp_size.
        for dp_idx in range(dp_size):
            base = dp_idx * ep_size * tp_size
            for tp_idx in range(tp_size):
                group_ranks = [base + e * tp_size + tp_idx for e in range(ep_size)]
                ep_groups.append(group_ranks)

        for ranks in ep_groups:
            group = dist.new_group(ranks=ranks, backend=backend, timeout=timedelta(seconds=1800))
            if rank in ranks:
                _EP_GROUP = group
                _EP_RANK = ranks.index(rank)
                logger.debug(f"Rank {rank} joined EP group {ranks}, ep_rank={_EP_RANK}")
    else:
        _EP_GROUP = None
        _EP_RANK = 0

    # -------------------------------------------------------------------------
    # Create DP groups (strided by ep_size * tp_size)
    #
    # DP ranks are the "same position" across EP replicas.
    #   world=128, EP=8, TP=1: DP groups are [0,8,16,...,120], [1,9,17,...,121], etc.
    #   Each DP group has dp_size=16 ranks.
    #
    # With TP>1: stride = ep_size * tp_size.
    #   world=16, EP=4, TP=2: stride=8, dp_size=2. DP groups: [0,8],[1,9],...,[7,15]
    # -------------------------------------------------------------------------
    if dp_size > 1:
        stride = ep_size * tp_size
        dp_groups: List[List[int]] = []
        for local_idx in range(stride):
            group_ranks = [local_idx + d * stride for d in range(dp_size)]
            dp_groups.append(group_ranks)

        for ranks in dp_groups:
            group = dist.new_group(ranks=ranks, backend=backend, timeout=timedelta(seconds=1800))
            if rank in ranks:
                _DP_GROUP = group
                _DP_RANK = ranks.index(rank)
                logger.debug(f"Rank {rank} joined DP group {ranks}, dp_rank={_DP_RANK}")
    else:
        _DP_GROUP = None
        _DP_RANK = 0

    _INITIALIZED = True

    logger.info(
        f"nmoe process groups initialized: "
        f"rank={rank}, ep_rank={_EP_RANK}/{_EP_SIZE}, "
        f"tp_rank={_TP_RANK}/{_TP_SIZE}, dp_rank={_DP_RANK}/{_DP_SIZE}"
    )


def init_nmoe_from_megatron() -> None:
    """Initialize nmoe parallel state from Megatron's process groups.

    This function extracts EP, TP, and DP groups from Megatron's parallel state,
    allowing nmoe to use Megatron's existing process group infrastructure.

    Raises:
        ImportError: If Megatron is not installed.
        RuntimeError: If Megatron parallel state is not initialized.
    """
    global _EP_GROUP, _TP_GROUP, _DP_GROUP
    global _EP_SIZE, _TP_SIZE, _DP_SIZE
    global _EP_RANK, _TP_RANK, _DP_RANK
    global _INITIALIZED

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

    # Megatron DP group
    try:
        _DP_GROUP = mpu.get_data_parallel_group()
        _DP_SIZE = mpu.get_data_parallel_world_size()
        _DP_RANK = mpu.get_data_parallel_rank()
    except (AttributeError, RuntimeError):
        # Older Megatron versions may not have DP accessors
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        _DP_SIZE = max(1, world_size // (_EP_SIZE * _TP_SIZE))
        _DP_RANK = 0
        _DP_GROUP = None

    _INITIALIZED = True

    logger.info(
        f"nmoe process groups initialized from Megatron: "
        f"ep_rank={_EP_RANK}/{_EP_SIZE}, tp_rank={_TP_RANK}/{_TP_SIZE}, "
        f"dp_rank={_DP_RANK}/{_DP_SIZE}"
    )


def cleanup_process_groups() -> None:
    """Clean up nmoe process groups.

    Destroys the EP, TP, and DP process groups created by init_nmoe_process_groups().
    Does not affect Megatron-created groups (those are managed by Megatron).

    Note: In most cases, you don't need to call this explicitly.
    PyTorch handles process group cleanup at program exit.
    """
    global _EP_GROUP, _TP_GROUP, _DP_GROUP
    global _INITIALIZED, _EP_SIZE, _TP_SIZE, _DP_SIZE, _EP_RANK, _TP_RANK, _DP_RANK

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

    if _DP_GROUP is not None:
        try:
            dist.destroy_process_group(_DP_GROUP)
            logger.debug("Destroyed DP process group")
        except Exception as e:
            logger.warning(f"Failed to destroy DP process group: {e}")
        _DP_GROUP = None

    _EP_SIZE = 1
    _TP_SIZE = 1
    _DP_SIZE = 1
    _EP_RANK = 0
    _TP_RANK = 0
    _DP_RANK = 0
    _INITIALIZED = False

    logger.info("nmoe process groups cleaned up")


def get_data_parallel_group() -> Optional[ProcessGroup]:
    """Get the Data Parallelism process group.

    In an EP+TP configuration, DP is the remaining parallelism dimension.
    For world_size=128, EP=8, TP=1: DP=16.

    Returns:
        DP process group, or None if DP=1 or not initialized.

    Raises:
        RuntimeError: If nmoe parallel state is not initialized.
    """
    if not _INITIALIZED:
        raise RuntimeError(
            "nmoe parallel state not initialized. "
            "Call init_nmoe_process_groups() or init_nmoe_from_megatron() first."
        )
    return _DP_GROUP
