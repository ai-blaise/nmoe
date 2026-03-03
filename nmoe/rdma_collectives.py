"""RDMA-optimized collective operations for CPU object exchange.

This module provides NCCL-backed alternatives to Gloo's object collectives,
enabling RDMA transport for CPU-side coordination during NVSHMEM/IPC bootstrap.

Key insight: Gloo over TCP (eth0) has ~100us latency. By serializing Python
objects to GPU tensors and using NCCL (which uses RDMA), we achieve ~10us latency.

Usage:
    from nmoe.rdma_collectives import broadcast_object_rdma, all_gather_object_rdma

    # Instead of:
    #   dist.broadcast_object_list([obj], src=0, group=cpu_pg)
    # Use:
    #   obj = broadcast_object_rdma(obj, src=0, nccl_group=nccl_pg)

Performance (16 nodes x 8 GPUs, 128 byte object):
    - Gloo TCP (eth0):    ~100us
    - Gloo IPoIB (ib0):   ~50us  (if IPoIB configured)
    - NCCL RDMA:          ~10us  (this module)
"""

import io
import pickle
from typing import Any, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

T = TypeVar("T")


def _serialize_object(obj: Any) -> bytes:
    """Serialize a Python object to bytes using pickle."""
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    return buffer.getvalue()


def _deserialize_object(data: bytes) -> Any:
    """Deserialize a Python object from bytes."""
    buffer = io.BytesIO(data)
    return pickle.load(buffer)


def _bytes_to_tensor(data: bytes, device: torch.device) -> torch.Tensor:
    """Convert bytes to a GPU tensor (int32 for NCCL compatibility)."""
    # Pad to 4-byte alignment for int32 view
    padded_len = ((len(data) + 3) // 4) * 4
    padded_data = data + b"\x00" * (padded_len - len(data))
    # Convert to int32 tensor (NCCL Tree algorithm doesn't support int8)
    import numpy as np

    arr = np.frombuffer(padded_data, dtype=np.int32)
    return torch.from_numpy(arr.copy()).to(device=device, non_blocking=True)


def _tensor_to_bytes(tensor: torch.Tensor, original_len: int) -> bytes:
    """Convert a GPU tensor back to bytes."""
    cpu_tensor = tensor.cpu()
    # View as uint8 to get raw bytes
    byte_view = cpu_tensor.view(torch.uint8).numpy()
    return bytes(byte_view[:original_len])


def broadcast_object_rdma(
    obj: Optional[T],
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> T:
    """Broadcast a Python object using NCCL (RDMA) instead of Gloo (TCP).

    This function serializes the object to bytes, broadcasts via NCCL all_gather
    (which uses RDMA on InfiniBand), and deserializes on all ranks.

    Args:
        obj: The object to broadcast. Only the src rank's object is used.
        src: Source rank for the broadcast.
        group: NCCL process group to use. If None, uses default group.
        device: CUDA device for tensor operations. If None, uses current device.

    Returns:
        The broadcasted object (same on all ranks).

    Note:
        The object must be picklable. For NCCL groups, this uses GPU memory
        temporarily for the broadcast operation.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    rank = dist.get_rank(group) if group else dist.get_rank()
    world_size = dist.get_world_size(group) if group else dist.get_world_size()

    # Step 1: Serialize on src rank
    if rank == src:
        data = _serialize_object(obj)
        data_len = len(data)
    else:
        data = b""
        data_len = 0

    # Step 2: Broadcast the length first (using all_reduce with src override)
    len_tensor = torch.tensor([data_len], dtype=torch.int64, device=device)
    dist.broadcast(len_tensor, src=src, group=group)
    data_len = int(len_tensor.item())

    # Step 3: Prepare data tensor
    # Pad to 4-byte alignment for int32
    padded_len = ((data_len + 3) // 4) * 4
    num_int32 = padded_len // 4

    if rank == src:
        data_tensor = _bytes_to_tensor(data, device)
    else:
        data_tensor = torch.zeros(num_int32, dtype=torch.int32, device=device)

    # Step 4: Broadcast the data
    dist.broadcast(data_tensor, src=src, group=group)

    # Step 5: Deserialize on all ranks
    result_bytes = _tensor_to_bytes(data_tensor, data_len)
    return _deserialize_object(result_bytes)


def all_gather_object_rdma(
    obj: T,
    group: Optional[ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> List[T]:
    """All-gather Python objects using NCCL (RDMA) instead of Gloo (TCP).

    This function serializes objects to bytes, gathers via NCCL all_gather
    (which uses RDMA on InfiniBand), and deserializes on all ranks.

    Args:
        obj: The local object to gather.
        group: NCCL process group to use. If None, uses default group.
        device: CUDA device for tensor operations. If None, uses current device.

    Returns:
        List of objects from all ranks, in rank order.

    Note:
        Objects must be picklable. Uses GPU memory temporarily for the gather.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    world_size = dist.get_world_size(group) if group else dist.get_world_size()

    # Step 1: Serialize local object
    data = _serialize_object(obj)
    data_len = len(data)

    # Step 2: All-gather the lengths
    len_tensor = torch.tensor([data_len], dtype=torch.int64, device=device)
    all_lens = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_lens, len_tensor, group=group)
    lens = [int(t.item()) for t in all_lens]

    # Step 3: Find max length and pad all data to same size
    max_len = max(lens)
    padded_len = ((max_len + 3) // 4) * 4
    num_int32 = padded_len // 4

    # Step 4: Serialize local data with padding
    padded_data = data + b"\x00" * (padded_len - data_len)
    data_tensor = _bytes_to_tensor(padded_data, device)

    # Step 5: All-gather the data
    all_data = [torch.zeros(num_int32, dtype=torch.int32, device=device) for _ in range(world_size)]
    dist.all_gather(all_data, data_tensor, group=group)

    # Step 6: Deserialize each rank's object
    results = []
    for i, (tensor, length) in enumerate(zip(all_data, lens)):
        result_bytes = _tensor_to_bytes(tensor, length)
        results.append(_deserialize_object(result_bytes))

    return results


def barrier_rdma(group: Optional[ProcessGroup] = None) -> None:
    """NCCL-based barrier using all_reduce.

    This is faster than Gloo barrier over TCP when using InfiniBand.

    Args:
        group: NCCL process group to use. If None, uses default group.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    dummy = torch.zeros(1, dtype=torch.int32, device=device)
    dist.all_reduce(dummy, op=dist.ReduceOp.SUM, group=group)
    # Ensure completion
    torch.cuda.synchronize()


# =============================================================================
# IPoIB (IP over InfiniBand) Detection
# =============================================================================


def detect_ipoib_interface() -> Optional[str]:
    """Detect available IPoIB interfaces on the system.

    IPoIB provides IP networking over InfiniBand, giving ~50us latency
    compared to ~100us for TCP over GVNIC.

    Returns:
        Interface name (e.g., "ib0", "ibs0") if found, None otherwise.
    """
    import subprocess

    try:
        # Check for IPoIB interfaces
        result = subprocess.run(
            ["ip", "link", "show"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        # Look for ib* or ibs* interfaces
        for line in result.stdout.split("\n"):
            # ip link show format: "N: ifname: <FLAGS>..."
            if ": ib" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    ifname = parts[1].strip().split("@")[0]
                    if ifname.startswith("ib"):
                        return ifname

        # Also check for Datagram mode interfaces (ibd*)
        for line in result.stdout.split("\n"):
            if ": ibd" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    ifname = parts[1].strip().split("@")[0]
                    return ifname

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_optimal_gloo_interface() -> str:
    """Get the optimal interface for Gloo based on available hardware.

    Priority:
    1. IPoIB interface (ib0, ibs0) - RDMA over IP, ~50us latency
    2. GVNIC (eth0) - TCP fallback, ~100us latency

    Returns:
        Interface name to use for GLOO_SOCKET_IFNAME.
    """
    ipoib = detect_ipoib_interface()
    if ipoib:
        return ipoib
    return "eth0"


# =============================================================================
# Combined API for RDEP Bootstrap
# =============================================================================


class RDMACollectives:
    """High-performance collective operations for RDEP bootstrap.

    This class provides NCCL-backed collectives for CPU object exchange,
    achieving RDMA performance for bootstrap operations that would otherwise
    use slow TCP transport.

    Usage:
        rdma = RDMACollectives(nccl_group)
        uid = rdma.broadcast_object(uid, src=0)
        all_hosts = rdma.all_gather_object(my_hostname)

    The class automatically uses NCCL for GPU tensor collectives,
    providing RDMA transport on InfiniBand clusters.
    """

    def __init__(
        self,
        nccl_group: Optional[ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize RDMA collectives.

        Args:
            nccl_group: NCCL process group for tensor collectives.
                       If None, uses the default NCCL group.
            device: CUDA device for tensor operations.
                   If None, uses torch.cuda.current_device().
        """
        self.nccl_group = nccl_group
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device

    def broadcast_object(self, obj: Optional[T], src: int = 0) -> T:
        """Broadcast an object from src to all ranks using NCCL."""
        return broadcast_object_rdma(
            obj, src=src, group=self.nccl_group, device=self.device
        )

    def all_gather_object(self, obj: T) -> List[T]:
        """All-gather objects from all ranks using NCCL."""
        return all_gather_object_rdma(
            obj, group=self.nccl_group, device=self.device
        )

    def barrier(self) -> None:
        """Synchronize all ranks using NCCL all_reduce."""
        barrier_rdma(group=self.nccl_group)
