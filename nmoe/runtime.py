"""Runtime initialization and cleanup for nmoe training.

Handles platform checks, distributed setup, seeds, and EP/DP process groups.
Seamlessly supports single GPU, single-node multi-GPU, and multi-node training.

B200 Cluster Configuration (16x nodes, 8x GPUs per node, 8x RDMA NICs per node):
- CUDA 13.1, Driver 590.48
- Python 3.12, PyTorch 2.9.1+cu130
- Blackwell architecture: sm_100 (compute capability 10.0)
- 8x 200GB/s NVLink per GPU (NVSwitch interconnect)
- 8x Mellanox ConnectX-7 @ 400 Gbps (RoCE transport, MTU 8896)
  Interfaces: gpu0rdma0 through gpu7rdma0
  Underlying devices: mlx5_0 through mlx5_7
"""
import logging
import os
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

# =============================================================================
# B200 CLUSTER MAXIMUM PERFORMANCE SETTINGS (HARD-CODED - NO FALLBACKS)
# Optimized for: 16 nodes x 8 B200 GPUs x 8 RDMA NICs = 128 GPUs
# Source of truth: infrastructure/install_packages.sh + cluster.yaml
# =============================================================================

# -----------------------------------------------------------------------------
# NCCL NETWORK TRANSPORT (8x mlx5 RoCE NICs per node, MTU 8896)
# NOTE: NCCL_NET=IB works for both InfiniBand and RoCE (same mlx5 driver)
# -----------------------------------------------------------------------------
os.environ["NCCL_NET"] = "IB"                    # Force RDMA (works for IB and RoCE)
os.environ["NCCL_IB_DISABLE"] = "0"              # RDMA enabled
os.environ["NCCL_SHM_DISABLE"] = "0"             # Shared memory for intra-node
os.environ["NCCL_IB_HCA"] = "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
os.environ["NCCL_IB_GID_INDEX"] = "3"            # RoCEv2 GID index

# GPUDirect RDMA: Level 2 = PHB (same PCIe root complex as GPU)
# B200 NVSwitch topology: GPU-to-NIC via PCIe switch, not NVLink
os.environ["NCCL_NET_GDR_LEVEL"] = "2"           # PHB level for B200
os.environ["NCCL_NET_GDR_READ"] = "1"            # GPUDirect RDMA read path

# -----------------------------------------------------------------------------
# NCCL MULTI-RAIL RoCE TUNING (Critical for 8x NIC saturation)
# -----------------------------------------------------------------------------
# Channel count: 8 channels = 1 per RDMA NIC = maximum multi-rail utilization
os.environ["NCCL_MIN_NCHANNELS"] = "8"           # Minimum 8 for 8-rail
os.environ["NCCL_MAX_NCHANNELS"] = "16"          # Allow up to 16 for large collectives

# Multi-rail balancing: CROSS_NIC=2 enables round-robin across all NICs
# Without this, NCCL may prefer a single NIC per connection
os.environ["NCCL_CROSS_NIC"] = "2"               # Round-robin multi-rail

# Multiple Queue Pairs per connection for higher throughput
# More QPs allow parallel RDMA operations on each NIC
os.environ["NCCL_IB_QPS_PER_CONNECTION"] = "4"   # 4 QPs per connection
os.environ["NCCL_IB_SPLIT_DATA_ON_QPS"] = "1"    # Split data across QPs

# NCCL buffer for 8-rail RDMA (128MB total, ~16MB per rail)
# Reduced from 512MB to free ~384MB/GPU for model activations during backward pass
os.environ["NCCL_BUFFSIZE"] = "134217728"         # 128MB (reduced from 512MB to fix CUDA OOM)

# -----------------------------------------------------------------------------
# NCCL RDMA/RoCE TRANSPORT TUNING (IB_* settings work for RoCE too)
# -----------------------------------------------------------------------------
# IB timeout: 20 = ~4 seconds (2^20 * 4.096us base)
# Default 14 (~67ms) is too aggressive for large-scale RDMA
os.environ["NCCL_IB_TIMEOUT"] = "20"             # ~4s timeout
os.environ["NCCL_IB_RETRY_CNT"] = "7"            # 7 retries before failure

# PCIe relaxed ordering: improves GPU-to-NIC DMA throughput
os.environ["NCCL_IB_PCI_RELAXED_ORDERING"] = "1"

# Disable adaptive routing (explicit multi-rail is better for this topology)
os.environ["NCCL_IB_AR_THRESHOLD"] = "0"         # Disable AR

# Traffic class for RDMA priority (default is fine for dedicated fabric)
# os.environ["NCCL_IB_TC"] = "0"                 # Default TC
# os.environ["NCCL_IB_SL"] = "0"                 # Default service level

# -----------------------------------------------------------------------------
# NCCL ALGORITHM AND PROTOCOL
# -----------------------------------------------------------------------------
# TREE algorithm for large 16-node all-reduce (better than RING for >8 nodes)
os.environ["NCCL_ALGO"] = "RING,TREE"
# SIMPLE protocol for large messages (>256KB), LL128 for small messages
# NCCL auto-selects, but SIMPLE is optimal for MoE large tensors
os.environ["NCCL_PROTO"] = "LL,LL128,SIMPLE"
# 512 threads per NCCL block (optimal for B200 SM count)
os.environ["NCCL_NTHREADS"] = "512"

# -----------------------------------------------------------------------------
# NCCL P2P / NVSwitch SETTINGS
# -----------------------------------------------------------------------------
# P2P via NVSwitch for intra-node GPU-to-GPU
os.environ["NCCL_P2P_DISABLE"] = "0"             # P2P enabled
os.environ["NCCL_P2P_LEVEL"] = "LOC"             # Local P2P (NVSwitch)

# NVLink SHARP (NVLS): B200/NVSwitch native collective acceleration
# Enables in-network reduction on NVSwitch for intra-node collectives
os.environ["NCCL_NVLS_ENABLE"] = "1"             # NVLink SHARP enabled

# CUDA Graph mixing support for NCCL operations
os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "1"

# -----------------------------------------------------------------------------
# GLOO TRANSPORT FOR CPU COLLECTIVES (Control Plane Only)
# -----------------------------------------------------------------------------
# Gloo is used for CPU-side object exchange during NVSHMEM/IPC bootstrap.
# Performance hierarchy (B200 cluster with 8x ConnectX-7 RoCE NICs):
#
# 1. NCCL-based RDMA collectives (NMOE_RDMA_COLLECTIVES=1, default)
#    - Serializes objects to GPU tensors, uses NCCL all_gather/broadcast
#    - ~10us latency (uses RDMA over RoCE via NCCL)
#    - Enabled by default in nmoe/rdma_collectives.py
#
# 2. Gloo over TCP (eth0, GVNIC)
#    - TCP over Google Virtual NIC, ~100us latency
#    - Used only for control plane (torchrun rendezvous) when RDMA collectives disabled
#
# NOTE: RoCE clusters do NOT have IPoIB interfaces (no ib0/ibs0).
# RoCE uses Ethernet interfaces (gpu0rdma0-gpu7rdma0) for RDMA data path.
# Gloo cannot use RoCE directly - it uses TCP for control plane only.
# Actual data transfer uses NCCL which handles RoCE natively via mlx5 driver.
os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")

# Enable RDMA collectives for CPU object exchange (uses NCCL instead of Gloo)
# Set to "0" to fall back to Gloo (useful for debugging or if NCCL init fails)
os.environ.setdefault("NMOE_RDMA_COLLECTIVES", "1")

# -----------------------------------------------------------------------------
# NCCL TIMEOUT SETTINGS
# -----------------------------------------------------------------------------
# 30-minute timeout for large-model init and long-running collectives
os.environ["NCCL_TIMEOUT"] = "1800"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"
# Non-blocking wait for async error detection
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"

# -----------------------------------------------------------------------------
# UCX SETTINGS FOR MULTI-RAIL RDMA TRANSPORT
# -----------------------------------------------------------------------------
# Transport selection: RC (Reliable Connection), UD (Unreliable Datagram),
# SM (Shared Memory for intra-node), Self (loopback)
os.environ["UCX_TLS"] = "rc,ud,sm,self"

# All 8 RDMA NICs with port 1
os.environ["UCX_NET_DEVICES"] = "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"

# Maximum 8 rails for rendezvous protocol (matches NIC count)
os.environ["UCX_MAX_RNDV_RAILS"] = "8"
os.environ["UCX_IB_GID_INDEX"] = "3"             # RoCEv2 GID

# Rendezvous threshold: 8KB (switch to zero-copy for larger messages)
os.environ["UCX_RNDV_THRESH"] = "8192"

# Direct memory registration (no caching, better for large GPU allocations)
os.environ["UCX_REG_METHODS"] = "direct"
os.environ["UCX_MEMTYPE_CACHE"] = "n"            # Disable memtype cache

# MLX5 DevX acceleration
os.environ["UCX_IB_MLX5_DEVX"] = "y"

# UCX queue depths for high-throughput RDMA
os.environ["UCX_RC_TX_QUEUE_LEN"] = "4096"       # Send queue depth
os.environ["UCX_RC_RX_QUEUE_LEN"] = "4096"       # Receive queue depth

# Inline data for small messages (reduce latency)
os.environ["UCX_IB_TX_INLINE"] = "128"           # Inline send threshold
os.environ["UCX_IB_RX_INLINE"] = "128"           # Inline receive threshold

# DC (Dynamic Connection) polling for scalability
os.environ["UCX_DC_TX_POLL_BUDGET"] = "256"      # TX polling budget

# -----------------------------------------------------------------------------
# CUDA / GPU SETTINGS
# -----------------------------------------------------------------------------
# CUDA architecture for Blackwell B200 (sm_100)
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

# Maximum CUDA device connections: higher value enables more async kernel overlap
# Default is 8, but 16 allows better overlap for MoE dispatch/combine
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "16"

# CUDA memory allocator: expandable segments + max split size
# max_split_size_mb prevents fragmentation from large tensor allocations
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512"
)

# -----------------------------------------------------------------------------
# PYTORCH DISTRIBUTED SETTINGS
# -----------------------------------------------------------------------------
# Async error handling for NCCL (detect hangs without blocking)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Avoid record_streams for NCCL to reduce memory overhead
os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

# Enable high-priority CUDA streams for NCCL
os.environ["TORCH_NCCL_HIGH_PRIORITY"] = "1"

# Coalesced reduce for better performance with many small tensors
os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "1"

# =============================================================================

import torch
import torch.distributed as dist

_log = logging.getLogger(__name__)


@contextmanager
def _nvtx(name: str):
    """Emit an NVTX range visible in Nsight Systems / nvprof."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _require_b200():
  """Hard-target NVIDIA B200 (sm_100a). Off-target is not supported."""
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device required (B200, sm_100a). Off-target is not supported.")
  major, minor = torch.cuda.get_device_capability()
  if (major, minor) != (10, 0):
    raise RuntimeError(
      f"This repo targets NVIDIA B200 (sm_100a). Detected compute capability {major}.{minor}. "
      "Off-target is not supported."
    )


def _ensure_third_party_imports() -> None:
  """Ensure vendored deps are importable (container-first contract)."""
  root = Path(__file__).resolve().parents[1]
  flash_attn = root / "third_party" / "flash_attn"
  if flash_attn.exists():
    p = str(flash_attn)
    if p not in sys.path:
      sys.path.insert(0, p)


def init(seed: int = 42, ep_size: int = 1, tp_size: int = 1) -> tuple[int, int]:
  """Initialize runtime for training. Returns (rank, world).

  Handles:
  - Platform check (B200 required)
  - Seeds and TF32
  - Device assignment (LOCAL_RANK env var)
  - Distributed init (automatic for multi-GPU)
  - EP/DP process group creation (when ep_size > 1)

  Args:
    seed: Random seed for reproducibility.
    ep_size: Expert parallelism group size. Default 1 (no EP).
    tp_size: Tensor parallelism group size. Default 1 (no TP).

  Works seamlessly for:
  - Single GPU: rank=0, world=1
  - Single-node multi-GPU: torchrun sets LOCAL_RANK, init NCCL
  - Multi-node: same as single-node, world > local_world
  """
  with _nvtx("runtime/require_b200"):
    _require_b200()
  _ensure_third_party_imports()

  # Seeds and TF32
  with _nvtx("runtime/seeds_and_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  # Device assignment (torchrun sets LOCAL_RANK; single-process defaults to 0)
  local_rank = int(os.environ.get('LOCAL_RANK', '0'))
  torch.cuda.set_device(local_rank)

  # Distributed init (only when launched under torchrun)
  world_env = int(os.environ.get('WORLD_SIZE', '1'))
  if world_env > 1 and not dist.is_initialized():
    with _nvtx("runtime/init_process_group"):
      dist.init_process_group("nccl", timeout=timedelta(seconds=1800))

  # Get rank and world (or default to single GPU)
  rank = dist.get_rank() if dist.is_initialized() else 0
  world = dist.get_world_size() if dist.is_initialized() else 1

  # Initialize EP/TP/DP process groups when parallelism is configured.
  # This must happen after dist.init_process_group() and before model creation.
  if (ep_size > 1 or tp_size > 1) and world > 1:
    from nmoe.distributed.init_groups import init_nmoe_process_groups, is_nmoe_parallel_initialized
    if not is_nmoe_parallel_initialized():
      with _nvtx("runtime/init_nmoe_groups"):
        init_nmoe_process_groups(ep_size=ep_size, tp_size=tp_size)
      if rank == 0:
        import logging as _rt_logging
        dp_size = world // (ep_size * tp_size)
        _rt_logging.getLogger(__name__).info(
            "Process groups: EP=%d, TP=%d, DP=%d, world=%d", ep_size, tp_size, dp_size, world
        )

  return rank, world


def finalize():
  """Cleanup distributed state (process groups and EP/DP groups)."""
  with _nvtx("runtime/finalize"):
    # Clean up nmoe process groups first (they are sub-groups of WORLD)
    try:
      from nmoe.distributed.init_groups import cleanup_process_groups, is_nmoe_parallel_initialized
      if is_nmoe_parallel_initialized():
        cleanup_process_groups()
    except Exception:
      _log.warning("Failed to clean up nmoe process groups during finalize", exc_info=True)

    if dist.is_initialized():
      dist.destroy_process_group()
