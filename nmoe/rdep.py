import logging
import os
import socket
from contextlib import nullcontext
from datetime import timedelta
from typing import Dict, List, Optional

import torch

_rdep_logger = logging.getLogger(__name__)
import torch.distributed as dist
from torch.distributed import ProcessGroup
import numpy as np

# Import C extension (built in csrc/)
from .csrc import rdep as _C
from .moe import _MoEBlockscaledFused
from .cuda_errors import (
    CudaError,
    NvshmemError,
    RdepError,
    cuda_error_context,
)
from .nccl_watchdog import get_watchdog
from .rdma_collectives import RDMACollectives


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in ("1", "true", "True")


_NVTX_ENABLED = (
    _env_flag("NMOE_NVTX", "0")
    and torch.cuda.is_available()
    and hasattr(torch.cuda, "nvtx")
    and hasattr(torch.cuda.nvtx, "range")
)


def _nvtx(tag: str):
    if _NVTX_ENABLED:
        return torch.cuda.nvtx.range(tag)
    return nullcontext()


def _get_local_world_size() -> int:
    """Get local world size (GPUs on this node)."""
    local = os.environ.get("LOCAL_WORLD_SIZE")
    if local is not None:
        return int(local)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        raise RuntimeError("LOCAL_WORLD_SIZE must be set when WORLD_SIZE > 1.")
    return 1


def _get_group_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Get world size for the given process group.

    Args:
        group: Process group to query. If None, uses default group.

    Returns:
        World size of the group, or 1 if dist is not initialized.
    """
    if not dist.is_initialized():
        return 1
    if group is None:
        return dist.get_world_size()
    return dist.get_world_size(group)


def _get_group_rank(group: Optional[ProcessGroup] = None) -> int:
    """Get rank within the given process group.

    Args:
        group: Process group to query. If None, uses default group.

    Returns:
        Rank within the group, or 0 if dist is not initialized.
    """
    if not dist.is_initialized():
        return 0
    if group is None:
        return dist.get_rank()
    return dist.get_rank(group)

_CPU_PG = None
_CPU_PG_WORLD: int | None = None
_USE_RDMA_COLLECTIVES: bool = _env_flag("NMOE_RDMA_COLLECTIVES", "1")


def _cpu_pg():
    """CPU-only process group for bootstrap collectives (Gloo backend).

    NOTE: Gloo is intentionally used here (not NCCL) because this group handles
    CPU tensor/object exchange during NVSHMEM/IPC bootstrap. NCCL requires GPU
    buffers and cannot be used for CPU-side coordination. This is the correct
    pattern for distributed systems that need CPU-side collective before GPU
    collectives are available.

    B200 cluster (RoCE):
    - Uses Gloo over TCP/eth0 for control plane (~100us latency)
    - RoCE clusters have NO IPoIB interfaces (Ethernet-based RDMA)
    - Actual RDMA data path uses NCCL via mlx5 driver

    For fast bootstrap, NMOE_RDMA_COLLECTIVES=1 (default) uses NCCL-based
    tensor exchange instead of Gloo object exchange (~10us via RoCE).
    """
    global _CPU_PG, _CPU_PG_WORLD
    if not dist.is_initialized():
        return None
    world = int(dist.get_world_size())
    if world <= 1:
        return None
    if _CPU_PG is None or _CPU_PG_WORLD != world:
        # Use eth0 for Gloo control plane (RoCE clusters have no IPoIB)
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")
        _rdep_logger.debug(
            "Using eth0 for Gloo control plane (RoCE data path uses NCCL)"
        )
        _CPU_PG = dist.new_group(backend="gloo", timeout=timedelta(seconds=1800))
        _CPU_PG_WORLD = world
    return _CPU_PG


def _get_rdma_collectives(device: torch.device) -> Optional[RDMACollectives]:
    """Get RDMA collectives instance for NCCL-based object exchange.

    When NMOE_RDMA_COLLECTIVES=1 (default), this returns an RDMACollectives
    instance that uses NCCL (RDMA) for object broadcast/gather operations.

    Returns:
        RDMACollectives instance if enabled, None otherwise.
    """
    if not _USE_RDMA_COLLECTIVES:
        return None
    if not dist.is_initialized():
        return None
    # Use the default NCCL group for RDMA collectives
    return RDMACollectives(nccl_group=None, device=device)


class Rdep:
    # Profile mapping: string name -> C extension profile ID
    # bf16=-1 (no quantization), fp8=0, nvfp4=1 (blockscaled quantization)
    PROFILES = {'bf16': -1, 'fp8': 0, 'nvfp4': 1}

    def __init__(
        self,
        dim: int,
        n_local: int,
        topk: int,
        profile: str = 'nvfp4',
        capacity: int = 65536,
        ep_group: Optional[ProcessGroup] = None,
    ):
        """Initialize RDEP dispatcher for expert parallelism.

        Args:
            dim: Hidden dimension size.
            n_local: Number of local experts per rank.
            topk: Number of experts activated per token.
            profile: Quantization profile - 'bf16', 'fp8', or 'nvfp4'.
                    Defaults to 'nvfp4' for production B200 training.
            capacity: Maximum tokens per RDEP buffer.
            ep_group: Optional expert-parallel process group. If None, uses the
                default process group. This allows integration with custom EP
                groups from frameworks like Megatron or SkyRL.
        """
        # P3.12: Use proper exception instead of assert for validation
        if profile not in self.PROFILES:
            raise TypeError(f"profile must be one of {list(self.PROFILES.keys())}, got {profile!r}")
        if profile == "bf16":
            raise ValueError(
                "RDEP bf16 profile is not supported in production. Use profile='fp8' or 'nvfp4'."
            )
        if n_local <= 0:
            raise ValueError(f"n_local must be > 0, got {n_local}")

        self.dim = dim
        self.n_local = n_local
        self.topk = topk
        self.profile = profile
        self.capacity = capacity
        self.ep_group = ep_group
        self.world = _get_group_world_size(ep_group)
        self.rank = _get_group_rank(ep_group)
        self.local_world = _get_local_world_size()
        self._device = torch.device("cuda", torch.cuda.current_device())
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            if local_rank != self._device.index:
                raise RuntimeError(
                    f"[RDEP] LOCAL_RANK={local_rank} does not match current CUDA device {self._device.index}. "
                    "Set CUDA device by local rank before constructing Rdep."
                )

        # P3.14: Pre-allocate pinned memory for D2H transfers
        # These are reused across forward/backward calls to avoid allocation overhead
        self._pinned_M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
        self._pinned_offs = torch.empty(n_local, dtype=torch.int32, device='cpu', pin_memory=True)

        # Initialize RDEP with CUDA error checking
        try:
            with cuda_error_context("rdep.init"):
                _C.init(self.rank, self.world, self.local_world)
                mode_int = _C.get_mode()
        except CudaError as e:
            raise RdepError(
                f"[RDEP] Failed to initialize dispatcher: {e}",
                operation="init",
            ) from e

        mode_map = {0: 'single', 1: 'ipc', 2: 'hybrid'}
        self._mode = mode_map.get(mode_int)
        if self._mode is None:
            raise RdepError(
                f"[RDEP] unsupported mode from extension: {mode_int}",
                operation="init",
            )
        if self._mode == "hybrid" and dist.is_initialized():
            global_world = dist.get_world_size()
            global_rank = dist.get_rank()
            if self.world != global_world or self.rank != global_rank:
                raise RuntimeError(
                    "[RDEP] Hybrid mode requires EP group == global WORLD group "
                    "(inter-node dispatch path)."
                )
        if self._mode == 'hybrid' and not _C.has_nvshmem():
            raise RuntimeError(
                f"Multi-node configuration (world={self.world} > local_world={self.local_world}) "
                "requires NVSHMEM support. Rebuild rdep with NVSHMEM or use single-node."
            )
        try:
            if self._mode == 'hybrid':
                with _nvtx("rdep/init_hybrid"):
                    self._setup_hybrid()
            elif self._mode == 'ipc':
                with _nvtx("rdep/init_buffers"):
                    if profile != 'bf16':
                        with cuda_error_context("rdep.alloc_blockscaled"):
                            _C.alloc_blockscaled(capacity, dim, n_local, self.PROFILES[profile])
                    with cuda_error_context("rdep.alloc_bf16"):
                        _C.alloc_bf16(capacity, dim, n_local)
                with _nvtx("rdep/init_ipc"):
                    self._setup_ipc()
            elif self._mode == 'single':
                with _nvtx("rdep/init_buffers"):
                    with cuda_error_context("rdep.alloc_bf16"):
                        _C.alloc_bf16(capacity, dim, n_local)
                        _C.sync_buffer_ptrs_bf16()
                    if profile != 'bf16':
                        with cuda_error_context("rdep.alloc_blockscaled"):
                            _C.alloc_blockscaled(capacity, dim, n_local, self.PROFILES[profile])
                            _C.sync_buffer_ptrs_blockscaled()
        except CudaError as e:
            raise RdepError(
                f"[RDEP] Failed to allocate buffers (mode={self._mode}, capacity={capacity}, dim={dim}): {e}",
                operation="buffer_allocation",
            ) from e

        self._dispatch_impl = (
            self._dispatch_bf16_impl
            if self.profile == 'bf16'
            else self._dispatch_blockscaled_impl
        )

    def _setup_hybrid(self):
        """Set up NVSHMEM for multi-node hybrid mode with CUDA error checking.

        RDMA Optimization (NMOE_RDMA_COLLECTIVES=1, default):
        Instead of using Gloo over TCP (~100us latency), this method uses
        NCCL-based tensor exchange for object collectives (~10us latency).
        The NCCL backend already uses RDMA over InfiniBand.

        Fallback (NMOE_RDMA_COLLECTIVES=0):
        Uses Gloo backend with IPoIB detection for ~50us latency if IPoIB
        interfaces are available, otherwise ~100us over GVNIC/eth0.
        """
        global_world = dist.get_world_size() if dist.is_initialized() else self.world
        global_rank = dist.get_rank() if dist.is_initialized() else self.rank
        if self.world != global_world or self.rank != global_rank:
            raise RuntimeError(
                "[RDEP] Hybrid mode currently requires EP group == global WORLD group."
            )

        # Get RDMA collectives (NCCL-based) if enabled, else use Gloo
        rdma = _get_rdma_collectives(self._device)
        use_rdma = rdma is not None

        # Gloo fallback for when RDMA collectives are disabled
        cpu_pg = None
        if not use_rdma:
            cpu_pg = _cpu_pg()
            if cpu_pg is None:
                raise RuntimeError("[RDEP] internal error: expected dist to be initialized for hybrid bootstrap")

        if self.rank == 0 and use_rdma:
            _rdep_logger.info("Using NCCL-based RDMA collectives for hybrid bootstrap (~10us latency)")
        elif self.rank == 0 and not use_rdma:
            _rdep_logger.info("Using Gloo for hybrid bootstrap (set NMOE_RDMA_COLLECTIVES=1 for RDMA)")

        try:
            uid_size = _C.nvshmem_get_uid_size()
            use_bf16_hybrid = (self.profile == "bf16")
            with _nvtx("rdep/hybrid_bootstrap"):
                if self.rank == 0:
                    _rdep_logger.info("rank=%d: Getting UID (size=%d)...", self.rank, uid_size)
                    with cuda_error_context("nvshmem_get_uid"):
                        uid = _C.nvshmem_get_uid()
                    _rdep_logger.info("rank=%d: Got UID", self.rank)
                else:
                    uid = None

                if self.rank == 0:
                    _rdep_logger.info("rank=%d: Broadcasting UID via %s...", self.rank, "NCCL/RDMA" if use_rdma else "Gloo")

                # Broadcast UID from rank 0 to all ranks
                if use_rdma:
                    uid = rdma.broadcast_object(uid, src=0)
                else:
                    uid_list = [uid]
                    dist.broadcast_object_list(uid_list, src=0, group=cpu_pg)
                    uid = uid_list[0]

                if self.rank == 0:
                    _rdep_logger.info("rank=%d: UID broadcast complete", self.rank)
                if self.rank == 0:
                    _rdep_logger.info("rank=%d: Initializing NVSHMEM...", self.rank)

                with cuda_error_context("nvshmem_init"):
                    _C.nvshmem_init(uid, self.rank, self.world, self.local_world)
                if self.rank == 0:
                    _rdep_logger.info("rank=%d: NVSHMEM initialized!", self.rank)

                if use_bf16_hybrid:
                    with cuda_error_context("nvshmem_alloc_bf16"):
                        _C.nvshmem_alloc_bf16(self.capacity, self.dim, self.n_local)
                else:
                    with cuda_error_context("nvshmem_alloc_blockscaled"):
                        _C.nvshmem_alloc_blockscaled(
                            self.capacity, self.dim, self.n_local, self.PROFILES[self.profile]
                        )

            my_host = socket.gethostname()

            with _nvtx("rdep/hybrid_allgather"):
                # Gather hostnames and local ranks from all ranks
                if use_rdma:
                    all_hosts = rdma.all_gather_object(my_host)
                else:
                    all_hosts = [None] * self.world
                    dist.all_gather_object(all_hosts, my_host, group=cpu_pg)

                local_rank_env = os.environ.get("LOCAL_RANK")
                if local_rank_env is None:
                    raise RuntimeError("[RDEP] LOCAL_RANK must be set for hybrid IPC-handle mapping.")

                if use_rdma:
                    all_local_ranks = rdma.all_gather_object(int(local_rank_env))
                else:
                    all_local_ranks = [None] * self.world
                    dist.all_gather_object(all_local_ranks, int(local_rank_env), group=cpu_pg)

                local_entries = []
                for r in range(self.world):
                    if all_hosts[r] == my_host:
                        local_entries.append((int(all_local_ranks[r]), r))
                if len(local_entries) != self.local_world:
                    raise RuntimeError(
                        f"[RDEP] expected {self.local_world} local IPC handles on host {my_host}, "
                        f"got {len(local_entries)}."
                    )
                seen_local_ranks = set()
                for lr, _ in local_entries:
                    if lr < 0 or lr >= self.local_world or lr in seen_local_ranks:
                        raise RuntimeError(
                            f"[RDEP] invalid/duplicate LOCAL_RANK mapping for host {my_host}: {lr}."
                        )
                    seen_local_ranks.add(lr)
                local_entries.sort(key=lambda x: x[0])
                local_peer_ranks = [r for _, r in local_entries]

            with _nvtx("rdep/hybrid_open"):
                if use_bf16_hybrid:
                    with cuda_error_context("nvshmem_get_ipc_handle_bf16"):
                        local_handle_bf16 = _C.nvshmem_get_ipc_handle_bf16()

                    # All-gather IPC handles using RDMA or Gloo
                    if use_rdma:
                        all_handles_bf16 = rdma.all_gather_object(local_handle_bf16)
                    else:
                        all_handles_bf16 = [None] * self.world
                        dist.all_gather_object(all_handles_bf16, local_handle_bf16, group=cpu_pg)

                    local_handles_bf16_np = np.concatenate([all_handles_bf16[r] for r in local_peer_ranks])
                    with cuda_error_context("nvshmem_open_ipc_handles_bf16"):
                        _C.nvshmem_open_ipc_handles_bf16(local_handles_bf16_np, self.local_world)
                        _C.nvshmem_sync_ipc_buffer_ptrs_bf16()
                else:
                    with cuda_error_context("nvshmem_get_ipc_handle_blockscaled"):
                        local_handle_block = _C.nvshmem_get_ipc_handle_blockscaled()

                    # All-gather IPC handles using RDMA or Gloo
                    if use_rdma:
                        all_handles_block = rdma.all_gather_object(local_handle_block)
                    else:
                        all_handles_block = [None] * self.world
                        dist.all_gather_object(all_handles_block, local_handle_block, group=cpu_pg)

                    local_handles_block_np = np.concatenate([all_handles_block[r] for r in local_peer_ranks])
                    with cuda_error_context("nvshmem_open_ipc_handles_blockscaled"):
                        _C.nvshmem_open_ipc_handles_blockscaled(local_handles_block_np, self.local_world)
                        _C.nvshmem_sync_ipc_buffer_ptrs_blockscaled()

            # Final barrier
            if use_rdma:
                rdma.barrier()
            else:
                dist.barrier(group=cpu_pg)

        except CudaError as e:
            raise NvshmemError(
                f"[RDEP] NVSHMEM setup failed (rank={self.rank}, world={self.world}): {e}",
                operation="setup_hybrid",
            ) from e

    @staticmethod
    def _ipc_handle_to_int32(
        handle_np: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert an int8 IPC handle (numpy) to a CUDA int32 tensor for NCCL all_gather.

        NCCL's Tree algorithm does not support ncclInt8, but it does support
        ncclInt32.  We reinterpret the raw bytes as int32 (same underlying data,
        just 4x fewer elements).  The byte count must be divisible by 4; IPC
        handles are 64 bytes so this is always satisfied.

        Args:
            handle_np: 1-D int8 numpy array from the C extension.

        Returns:
            1-D int32 CUDA tensor (length = handle_np.nbytes / 4).
        """
        if handle_np.dtype not in (np.int8, np.uint8):
            raise ValueError(f"expected int8/uint8 IPC handle, got {handle_np.dtype}")
        n = handle_np.nbytes
        if n % 4 != 0:
            raise ValueError(f"IPC handle byte count ({n}) must be divisible by 4")
        # reinterpret bytes as int32 on the numpy side, then move to CUDA
        # Use .view(np.uint8) first to normalise, then reinterpret as int32
        handle_i32_np = handle_np.view(np.uint8).view(np.int32)
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return torch.from_numpy(handle_i32_np.copy()).to(device=device, non_blocking=True)

    @staticmethod
    def _int32_handles_to_numpy(handle_tensors: list) -> np.ndarray:
        """Convert gathered int32 CUDA tensors back to a contiguous int8 numpy array.

        Args:
            handle_tensors: List of 1-D int32 CUDA tensors (one per EP rank).

        Returns:
            Contiguous 1-D int8 numpy array containing all handles concatenated.
        """
        # View each int32 tensor back as int8, move to CPU, convert to numpy
        parts = []
        for h in handle_tensors:
            # h is int32 on CUDA; view as int8 to get original bytes
            h_cpu = h.cpu()
            h_bytes = h_cpu.view(torch.uint8).numpy()
            parts.append(h_bytes)
        return np.concatenate(parts)

    def _setup_ipc(self):
        """Exchange IPC handles via NCCL all_gather on the EP group (one-time at init).

        IPC handles are raw byte arrays (int8).  NCCL's Tree algorithm does not
        support AllGather with ncclInt8, so we reinterpret the bytes as int32
        (which Tree does support) before the collective and view them back to
        int8 afterwards.  This keeps us on the NCCL EP group (correct rank
        count) and avoids creating a separate Gloo process group.

        The handles are small (typically 64 bytes each), so the dtype
        reinterpretation has negligible overhead.

        Raises:
            RdepError: If IPC handle exchange fails with CUDA error.
        """
        wd = get_watchdog()
        ipc_group = self.ep_group
        try:
            # --- bf16 handles ---
            with _nvtx("rdep/ipc_get_handle_bf16"):
                with cuda_error_context("get_ipc_handle_bf16"):
                    local_handle_bf16 = _C.get_ipc_handle_bf16()
            with _nvtx("rdep/ipc_to_int32_bf16"):
                handle_tensor_bf16 = self._ipc_handle_to_int32(local_handle_bf16, device=self._device)

            with _nvtx("rdep/ipc_allgather_bf16"):
                all_handles_bf16 = [torch.zeros_like(handle_tensor_bf16) for _ in range(self.world)]
                with wd.guard("all_gather IPC handles bf16"):
                    dist.all_gather(all_handles_bf16, handle_tensor_bf16, group=ipc_group)

            all_handles_bf16_np = self._int32_handles_to_numpy(all_handles_bf16)
            with _nvtx("rdep/ipc_open_bf16"):
                with cuda_error_context("open_ipc_handles_bf16"):
                    _C.open_ipc_handles_bf16(all_handles_bf16_np, self.world)
                    _C.sync_buffer_ptrs_bf16()

            # --- blockscaled handles (fp8 / nvfp4) ---
            if self.profile != 'bf16':
                with _nvtx("rdep/ipc_get_handle_block"):
                    with cuda_error_context("get_ipc_handle_blockscaled"):
                        local_handle_block = _C.get_ipc_handle_blockscaled()
                handle_tensor_block = self._ipc_handle_to_int32(local_handle_block, device=self._device)

                with _nvtx("rdep/ipc_allgather_block"):
                    all_handles_block = [torch.zeros_like(handle_tensor_block) for _ in range(self.world)]
                    with wd.guard("all_gather IPC handles blockscaled"):
                        dist.all_gather(all_handles_block, handle_tensor_block, group=ipc_group)

                all_handles_block_np = self._int32_handles_to_numpy(all_handles_block)
                with _nvtx("rdep/ipc_open_block"):
                    with cuda_error_context("open_ipc_handles_blockscaled"):
                        _C.open_ipc_handles_blockscaled(all_handles_block_np, self.world)
                        _C.sync_buffer_ptrs_blockscaled()

        except CudaError as e:
            raise RdepError(
                f"[RDEP] IPC handle exchange failed (rank={self.rank}, world={self.world}): {e}",
                operation="setup_ipc",
            ) from e

    def moe_bf16(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        with _nvtx("rdep/moe_bf16"):
            raise RuntimeError(
                "BF16 MoE path removed — use blockscaled (dtype=nvfp4). "
                "The _MoEBf16Fused autograd Function has been deleted; "
                "set profile='fp8' or profile='nvfp4' and call moe_blockscaled()."
            )

    def moe_blockscaled(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                        W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
                        W_cache, fused_eco=None, moe_ref=None) -> torch.Tensor:
        with _nvtx("rdep/moe_blockscaled"):
            if self.profile == 'bf16':
                raise RuntimeError("moe_blockscaled() requires profile in {'fp8','nvfp4'}")
            return _MoEBlockscaledFused.apply(self, x, eid, gates, W1, W3, W2, W_cache, fused_eco, moe_ref)

    def _dispatch_bf16_impl(
        self,
        x: torch.Tensor,
        eid: torch.Tensor,
        gates: torch.Tensor,
        W1: torch.Tensor,
        W3: torch.Tensor,
        W2: torch.Tensor,
        W_cache=None,
        fused_eco=None,
        moe_ref=None,
    ) -> torch.Tensor:
        return self.moe_bf16(x, eid, gates, W1, W3, W2)

    def _dispatch_blockscaled_impl(
        self,
        x: torch.Tensor,
        eid: torch.Tensor,
        gates: torch.Tensor,
        W1: torch.Tensor,
        W3: torch.Tensor,
        W2: torch.Tensor,
        W_cache=None,
        fused_eco=None,
        moe_ref=None,
    ) -> torch.Tensor:
        if W_cache is None:
            raise ValueError(
                f"Blockscaled profile '{self.profile}' requires W_cache argument"
            )
        return self.moe_blockscaled(x, eid, gates, W1, W3, W2, W_cache, fused_eco, moe_ref)

    def dispatch(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                 W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
                 W_cache=None, fused_eco=None, moe_ref=None) -> torch.Tensor:
        """Unified dispatch method that routes to moe_bf16 or moe_blockscaled.

        This method provides a single entry point for CUDA graph capture/replay,
        automatically selecting the correct MoE kernel based on the profile.

        Args:
            x: [T, H] BF16 hidden states
            eid: [T, K] int32 expert IDs
            gates: [T, K] BF16 routing weights
            W1: [E, H, Dff] gate projection weights
            W3: [E, H, Dff] up projection weights
            W2: [E, Dff, H] down projection weights
            W_cache: Pre-computed weight cache for blockscaled mode (optional)
            fused_eco: FusedBackwardECO controller (optional, for fused backward-optimizer)
            moe_ref: MoE module reference (optional, for fused backward-optimizer)

        Returns:
            [T, H] BF16 output tensor

        Raises:
            ValueError: If token count exceeds buffer capacity
        """
        # P3.8: Early capacity validation to fail fast with helpful error
        T = x.shape[0]
        K = eid.shape[1]
        world = max(1, self.world)
        required = T * K * world
        if required > self.capacity:
            raise ValueError(
                f"Token count exceeds RDEP capacity: T={T} * K={K} * world={world} = {required:,} > capacity={self.capacity:,}. "
                f"Increase capacity via compute_rdep_capacity() or reduce batch size."
            )
        expected_device = self._device
        for name, tensor in (("x", x), ("eid", eid), ("gates", gates), ("W1", W1), ("W3", W3), ("W2", W2)):
            if tensor.device != expected_device:
                raise ValueError(
                    f"{name} must be on {expected_device}, got {tensor.device}"
                )
        if W_cache is not None and W_cache.device != expected_device:
            raise ValueError(
                f"W_cache must be on {expected_device}, got {W_cache.device}"
            )

        with _nvtx("rdep/dispatch"):
            return self._dispatch_impl(
                x, eid, gates, W1, W3, W2,
                W_cache=W_cache, fused_eco=fused_eco, moe_ref=moe_ref
            )


class CudaGraphDispatch:
    """CUDA graph wrapper for RDEP dispatch.

    This class captures RDEP dispatch operations into a CUDA graph for efficient
    replay during decode phase, eliminating kernel launch overhead.

    Usage:
        rdep = Rdep(dim=256, n_local=8, topk=2, profile="nvfp4")
        graph_dispatch = CudaGraphDispatch(rdep)

        # Capture with representative inputs
        graph_dispatch.capture(x, eid, gates, W1, W3, W2)

        # Replay with new inputs (same shapes required)
        result = graph_dispatch.replay(x_new, eid_new, gates_new)

    Note:
        - Input tensor shapes must match between capture and replay
        - Static weight tensors (W1, W3, W2) are stored during capture
        - For blockscaled mode, W_cache is stored during capture
    """

    def __init__(self, rdep: Rdep, warmup_iterations: int = 3):
        """Initialize CUDA graph dispatch wrapper.

        Args:
            rdep: The Rdep instance to wrap
            warmup_iterations: Number of warmup iterations before capture (default: 3)
        """
        self.rdep = rdep
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs: Dict[str, torch.Tensor] = {}
        self.static_outputs: Dict[str, torch.Tensor] = {}
        self.static_weights: Dict[str, torch.Tensor] = {}
        self._warmup_iterations = warmup_iterations
        self._w_cache = None

    def capture(
        self,
        x: torch.Tensor,
        eid: torch.Tensor,
        gates: torch.Tensor,
        W1: Optional[torch.Tensor] = None,
        W3: Optional[torch.Tensor] = None,
        W2: Optional[torch.Tensor] = None,
        W_cache=None,
    ) -> None:
        """Capture dispatch into a CUDA graph.

        Args:
            x: [T, H] BF16 hidden states (defines static input shape)
            eid: [T, K] int32 expert IDs
            gates: [T, K] BF16 routing weights
            W1: [E, H, Dff] gate projection weights (stored for replay)
            W3: [E, H, Dff] up projection weights (stored for replay)
            W2: [E, Dff, H] down projection weights (stored for replay)
            W_cache: Pre-computed weight cache for blockscaled mode
        """
        # Allocate static input tensors (will be reused across replays)
        self.static_inputs = {
            'x': torch.empty_like(x),
            'eid': torch.empty_like(eid),
            'gates': torch.empty_like(gates),
        }

        # Copy inputs to static buffers
        self.static_inputs['x'].copy_(x)
        self.static_inputs['eid'].copy_(eid)
        self.static_inputs['gates'].copy_(gates)

        # Store weight references (these don't change between replays)
        if W1 is not None:
            self.static_weights['W1'] = W1
        if W3 is not None:
            self.static_weights['W3'] = W3
        if W2 is not None:
            self.static_weights['W2'] = W2
        self._w_cache = W_cache

        # Get weight tensors for dispatch
        w1 = self.static_weights.get('W1')
        w3 = self.static_weights.get('W3')
        w2 = self.static_weights.get('W2')

        if w1 is None or w3 is None or w2 is None:
            raise ValueError("W1, W3, W2 must be provided for CUDA graph capture")
        device = x.device
        for name, tensor in (("eid", eid), ("gates", gates), ("W1", w1), ("W3", w3), ("W2", w2)):
            if tensor.device != device:
                raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if self._w_cache is not None and self._w_cache.device != device:
            raise ValueError(f"W_cache must be on {device}, got {self._w_cache.device}")

        # Warmup: run several iterations on a side stream to ensure kernels are compiled
        with torch.cuda.device(device):
            warmup_stream = torch.cuda.Stream(device=device)
            current_stream = torch.cuda.current_stream(device=device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(self._warmup_iterations):
                    self.rdep.dispatch(
                        self.static_inputs['x'],
                        self.static_inputs['eid'],
                        self.static_inputs['gates'],
                        w1, w3, w2,
                        self._w_cache,
                    )
            # Stream dependency is sufficient here; no host-side synchronize needed.
            current_stream.wait_stream(warmup_stream)

            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs['result'] = self.rdep.dispatch(
                    self.static_inputs['x'],
                    self.static_inputs['eid'],
                    self.static_inputs['gates'],
                    w1, w3, w2,
                    self._w_cache,
                )

    def replay(
        self,
        x: torch.Tensor,
        eid: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Replay captured graph with new inputs.

        Args:
            x: [T, H] BF16 hidden states (must match captured shape)
            eid: [T, K] int32 expert IDs (must match captured shape)
            gates: [T, K] BF16 routing weights (must match captured shape)

        Returns:
            [T, H] BF16 output tensor

        Raises:
            RuntimeError: If graph has not been captured yet
        """
        if self.graph is None:
            raise RuntimeError("Graph not captured. Call capture() first.")
        if x.device != self.static_inputs['x'].device:
            raise ValueError(
                f"x must be on {self.static_inputs['x'].device}, got {x.device}"
            )
        if x.dtype != self.static_inputs['x'].dtype or x.shape != self.static_inputs['x'].shape:
            raise ValueError(
                f"x must match captured shape/dtype {tuple(self.static_inputs['x'].shape)}/{self.static_inputs['x'].dtype}, "
                f"got {tuple(x.shape)}/{x.dtype}"
            )
        if eid.device != self.static_inputs['eid'].device:
            raise ValueError(
                f"eid must be on {self.static_inputs['eid'].device}, got {eid.device}"
            )
        if eid.dtype != self.static_inputs['eid'].dtype or eid.shape != self.static_inputs['eid'].shape:
            raise ValueError(
                f"eid must match captured shape/dtype {tuple(self.static_inputs['eid'].shape)}/{self.static_inputs['eid'].dtype}, "
                f"got {tuple(eid.shape)}/{eid.dtype}"
            )
        if gates.device != self.static_inputs['gates'].device:
            raise ValueError(
                f"gates must be on {self.static_inputs['gates'].device}, got {gates.device}"
            )
        if gates.dtype != self.static_inputs['gates'].dtype or gates.shape != self.static_inputs['gates'].shape:
            raise ValueError(
                f"gates must match captured shape/dtype {tuple(self.static_inputs['gates'].shape)}/{self.static_inputs['gates'].dtype}, "
                f"got {tuple(gates.shape)}/{gates.dtype}"
            )

        # Copy new inputs to static buffers
        self.static_inputs['x'].copy_(x)
        self.static_inputs['eid'].copy_(eid)
        self.static_inputs['gates'].copy_(gates)

        # Replay the captured graph on the captured device context.
        with torch.cuda.device(self.static_inputs['x'].device):
            self.graph.replay()

        return self.static_outputs['result']

    def reset(self) -> None:
        """Reset the captured graph, freeing resources.

        Call this to release CUDA graph resources when switching to different
        batch sizes or when the graph is no longer needed.
        """
        if self.graph is not None:
            del self.graph
            self.graph = None
        self.static_inputs.clear()
        self.static_outputs.clear()
        self.static_weights.clear()
        self._w_cache = None

    @property
    def is_captured(self) -> bool:
        """Check if a graph has been captured."""
        return self.graph is not None
