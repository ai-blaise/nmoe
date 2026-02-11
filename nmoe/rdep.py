import logging
import os
from typing import Dict, Optional

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


def _get_local_world_size() -> int:
    """Get local world size (GPUs on this node)."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))


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


def _cpu_pg():
    """CPU-only process group for bootstrap collectives (Gloo)."""
    global _CPU_PG, _CPU_PG_WORLD
    if not dist.is_initialized():
        return None
    world = int(dist.get_world_size())
    if world <= 1:
        return None
    if _CPU_PG is None or _CPU_PG_WORLD != world:
        _CPU_PG = dist.new_group(backend="gloo")
        _CPU_PG_WORLD = world
    return _CPU_PG


class Rdep:
    # Profile mapping: string name -> C extension profile ID
    # bf16=-1 (no quantization), fp8=0, nvfp4=1 (blockscaled quantization)
    PROFILES = {'bf16': -1, 'fp8': 0, 'nvfp4': 1}

    def __init__(
        self,
        dim: int,
        n_local: int,
        topk: int,
        profile: str = 'bf16',  # P3.7: Default to bf16 for hardware compatibility
        capacity: int = 65536,
        ep_group: Optional[ProcessGroup] = None,
    ):
        """Initialize RDEP dispatcher for expert parallelism.

        Args:
            dim: Hidden dimension size.
            n_local: Number of local experts per rank.
            topk: Number of experts activated per token.
            profile: Quantization profile - 'bf16', 'fp8', or 'nvfp4'.
                    Defaults to 'bf16' for maximum hardware compatibility.
            capacity: Maximum tokens per RDEP buffer.
            ep_group: Optional expert-parallel process group. If None, uses the
                default process group. This allows integration with custom EP
                groups from frameworks like Megatron or SkyRL.
        """
        # P3.12: Use proper exception instead of assert for validation
        if profile not in self.PROFILES:
            raise TypeError(f"profile must be one of {list(self.PROFILES.keys())}, got {profile!r}")

        self.dim = dim
        self.n_local = n_local
        self.topk = topk
        self.profile = profile
        self.capacity = capacity
        self.ep_group = ep_group
        self.world = _get_group_world_size(ep_group)
        self.rank = _get_group_rank(ep_group)
        self.local_world = _get_local_world_size()

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

        self._mode = {0: 'single', 1: 'ipc', 2: 'hybrid'}[mode_int]
        if self._mode == 'hybrid' and not _C.has_nvshmem():
            raise RuntimeError(
                f"Multi-node configuration (world={self.world} > local_world={self.local_world}) "
                "requires NVSHMEM support. Rebuild rdep with NVSHMEM or use single-node."
            )

        try:
            if self._mode == 'hybrid':
                self._setup_hybrid()
            elif self._mode == 'ipc':
                with cuda_error_context("rdep.alloc_bf16"):
                    _C.alloc_bf16(capacity, dim, n_local)
                if profile != 'bf16':
                    with cuda_error_context("rdep.alloc_blockscaled"):
                        _C.alloc_blockscaled(capacity, dim, n_local, self.PROFILES[profile])
                self._setup_ipc()
            elif self._mode == 'single':
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

    def _setup_hybrid(self):
        """Set up NVSHMEM for multi-node hybrid mode with CUDA error checking."""
        cpu_pg = _cpu_pg()
        if cpu_pg is None:
            raise RuntimeError("[RDEP] internal error: expected dist to be initialized for hybrid bootstrap")

        try:
            uid_size = _C.nvshmem_get_uid_size()
            if self.rank == 0:
                _rdep_logger.info("rank=%d: Getting UID (size=%d)...", self.rank, uid_size)
                with cuda_error_context("nvshmem_get_uid"):
                    uid = _C.nvshmem_get_uid()
                _rdep_logger.info("rank=%d: Got UID", self.rank)
            else:
                uid = None

            if self.rank == 0:
                _rdep_logger.info("rank=%d: Broadcasting UID via CPU...", self.rank)
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

            with cuda_error_context("nvshmem_alloc_bf16"):
                _C.nvshmem_alloc_bf16(self.capacity, self.dim, self.n_local)

            node_id = self.rank // self.local_world
            with cuda_error_context("nvshmem_get_ipc_handle_bf16"):
                local_handle_bf16 = _C.nvshmem_get_ipc_handle_bf16()

            all_handles_bf16 = [None] * self.world
            dist.all_gather_object(all_handles_bf16, local_handle_bf16, group=cpu_pg)
            local_handles_bf16 = []
            for r in range(self.world):
                if r // self.local_world == node_id:
                    local_handles_bf16.append(all_handles_bf16[r])
            local_handles_bf16_np = np.concatenate(local_handles_bf16)

            with cuda_error_context("nvshmem_open_ipc_handles_bf16"):
                _C.nvshmem_open_ipc_handles_bf16(local_handles_bf16_np, self.local_world)
                _C.nvshmem_sync_ipc_buffer_ptrs_bf16()

            dist.barrier(group=cpu_pg)

        except CudaError as e:
            raise NvshmemError(
                f"[RDEP] NVSHMEM setup failed (rank={self.rank}, world={self.world}): {e}",
                operation="setup_hybrid",
            ) from e

    def _setup_ipc(self):
        """Exchange IPC handles via NCCL all_gather (one-time at init).

        Uses the ep_group for collectives if provided, allowing custom EP
        process groups from Megatron or SkyRL.

        Raises:
            RdepError: If IPC handle exchange fails with CUDA error.
        """
        try:
            with cuda_error_context("get_ipc_handle_bf16"):
                local_handle_bf16 = _C.get_ipc_handle_bf16()
            handle_tensor_bf16 = torch.from_numpy(local_handle_bf16).cuda()

            all_handles_bf16 = [torch.zeros_like(handle_tensor_bf16) for _ in range(self.world)]
            dist.all_gather(all_handles_bf16, handle_tensor_bf16, group=self.ep_group)

            all_handles_bf16_np = np.concatenate([h.cpu().numpy() for h in all_handles_bf16])
            with cuda_error_context("open_ipc_handles_bf16"):
                _C.open_ipc_handles_bf16(all_handles_bf16_np, self.world)
                _C.sync_buffer_ptrs_bf16()

            if self.profile != 'bf16':
                with cuda_error_context("get_ipc_handle_blockscaled"):
                    local_handle_block = _C.get_ipc_handle_blockscaled()
                handle_tensor_block = torch.from_numpy(local_handle_block).cuda()
                all_handles_block = [torch.zeros_like(handle_tensor_block) for _ in range(self.world)]
                dist.all_gather(all_handles_block, handle_tensor_block, group=self.ep_group)
                all_handles_block_np = np.concatenate([h.cpu().numpy() for h in all_handles_block])
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
        raise RuntimeError(
            "BF16 MoE path removed — use blockscaled (dtype=nvfp4). "
            "The _MoEBf16Fused autograd Function has been deleted; "
            "set profile='fp8' or profile='nvfp4' and call moe_blockscaled()."
        )

    def moe_blockscaled(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                        W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
                        W_cache, fused_eco=None, moe_ref=None) -> torch.Tensor:
        if self.profile == 'bf16':
            raise RuntimeError("moe_blockscaled() requires profile in {'fp8','nvfp4'}")
        return _MoEBlockscaledFused.apply(self, x, eid, gates, W1, W3, W2, W_cache, fused_eco, moe_ref)

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

        if self.profile == 'bf16':
            return self.moe_bf16(x, eid, gates, W1, W3, W2)
        else:
            if W_cache is None:
                raise ValueError(
                    f"Blockscaled profile '{self.profile}' requires W_cache argument"
                )
            return self.moe_blockscaled(x, eid, gates, W1, W3, W2, W_cache, fused_eco, moe_ref)


class CudaGraphDispatch:
    """CUDA graph wrapper for RDEP dispatch.

    This class captures RDEP dispatch operations into a CUDA graph for efficient
    replay during decode phase, eliminating kernel launch overhead.

    Usage:
        rdep = Rdep(dim=256, n_local=8, topk=2, profile="bf16")
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

        # Warmup: run several iterations on a side stream to ensure kernels are compiled
        # P3.5: Use context manager pattern to ensure stream cleanup
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(self._warmup_iterations):
                self.rdep.dispatch(
                    self.static_inputs['x'],
                    self.static_inputs['eid'],
                    self.static_inputs['gates'],
                    w1, w3, w2,
                    self._w_cache,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        # P3.5: Explicitly synchronize and delete stream to release resources
        warmup_stream.synchronize()
        del warmup_stream

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

        # Copy new inputs to static buffers
        self.static_inputs['x'].copy_(x)
        self.static_inputs['eid'].copy_(eid)
        self.static_inputs['gates'].copy_(gates)

        # Replay the captured graph
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
