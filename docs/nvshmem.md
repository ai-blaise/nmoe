# NVSHMEM Build Requirements for nmoe

This document describes the requirements and process for building nmoe with NVSHMEM support, enabling multi-node Mixture-of-Experts training with RDEP hybrid mode.

## Overview

nmoe uses NVSHMEM for inter-node communication in the RDEP (Redistribution Expert Parallelism) dispatcher. For single-node training, only CUDA IPC is used. Multi-node training requires NVSHMEM for the hybrid dispatch mode.

**Communication Modes:**
- `MODE_SINGLE` - Single GPU, no communication
- `MODE_IPC` - Single node, uses CUDA IPC for intra-node GPU-to-GPU communication
- `MODE_HYBRID` - Multi-node, uses CUDA IPC intra-node + NVSHMEM inter-node

## Requirements

### Hardware Requirements

- **NVIDIA B200 GPUs** (SM100/sm_100a architecture)
- **InfiniBand network** with RDMA support for multi-node
  - Tested with: ConnectX-7/8 NICs
- **NVLink** for intra-node GPU communication

### Software Requirements

- **CUDA Toolkit 12.x+** with sm_100a support
- **NVSHMEM 3.5.7** (patched version, see below)
- **InfiniBand libraries:**
  - `libibverbs-dev`
  - `librdmacm-dev`
  - `libnuma-dev`
- **CMake 3.20+**

### NVSHMEM Version

nmoe requires a patched version of NVSHMEM 3.5.7 with:
- IBGDA (InfiniBand GPUDirect Async) support enabled
- GDRCopy disabled (stability improvement for B200)
- Bidirectional RC QPs for symmetric communication

## Build Instructions

### Using Docker (Recommended)

The easiest way to build with NVSHMEM is using the provided Dockerfile:

```bash
# Build base training image first
docker build -f docker/Dockerfile.base -t xjdr/nmoe:base .
docker build -f docker/Dockerfile.train -t xjdr/nmoe_train:latest .

# Build NVSHMEM-enabled distributed image
docker build -f docker/Dockerfile.dist -t xjdr/nmoe_dist:latest .
```

### Manual Build

If building manually, follow these steps:

#### 1. Install Dependencies

```bash
apt-get install -y \
    libibverbs-dev \
    librdmacm-dev \
    libnuma-dev \
    pkg-config
```

#### 2. Clone and Patch NVSHMEM

```bash
git clone https://github.com/NVIDIA/nvshmem.git third_party/nvshmem
cd third_party/nvshmem
git checkout 9cc869bc28e565e6944c4ddf76976ada4a1ebbf7
git apply ../nvshmem-3.5.7-ibgda.patch
```

#### 3. Build NVSHMEM

```bash
cd third_party/nvshmem
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH="100a" \
    -DCMAKE_CUDA_ARCHITECTURES="100a" \
    -DNVSHMEM_USE_GDRCOPY=OFF \
    -DNVSHMEM_IBGDA_SUPPORT=ON \
    -DNVSHMEM_IBDEVX_SUPPORT=OFF \
    -DNVSHMEM_IBRC_SUPPORT=OFF \
    -DNVSHMEM_MPI_SUPPORT=OFF \
    -DNVSHMEM_BUILD_PYTHON_LIB=OFF \
    -DNVSHMEM_BUILD_PYTHON_DEVICE_LIB=OFF \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
    -DCUDA_HOME=${CUDA_HOME}

make -j$(nproc)
```

#### 4. Build nmoe RDEP Kernels with NVSHMEM

```bash
cd nmoe/csrc
make clean
make \
    NVSHMEM_INCLUDE=/path/to/nvshmem/src/include \
    NVSHMEM_LIB=/path/to/nvshmem/build/src/lib
```

## Environment Variables

### Required Runtime Environment

```bash
# Library paths
export LD_LIBRARY_PATH=/path/to/nvshmem/build/src/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=/path/to/nvshmem/build/src/lib/libnvshmem_host.so.3.5.7

# NVSHMEM configuration (IBGDA mode)
export NVSHMEM_HOME=/path/to/nvshmem
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_DISABLE_NVLS=1
export NVSHMEM_MAX_TEAMS=7
export NVSHMEM_CUMEM_GRANULARITY=536870912
export NVSHMEM_DISABLE_MNNVL=1
export NVSHMEM_DISABLE_P2P=0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eno1  # Your IB interface
export NVSHMEM_DEBUG=WARN
export NVSHMEM_IBGDA_SUPPORT=1
```

### NCCL Configuration (for non-MoE collectives)

```bash
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1  # Match NVSHMEM interface
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
```

## Architecture

### Hybrid Mode Communication Pattern

```
Node 0                                Node 1
┌────────────────────┐                ┌────────────────────┐
│ GPU 0   GPU 1      │                │ GPU 0   GPU 1      │
│  ↕ IPC   ↕ IPC     │                │  ↕ IPC   ↕ IPC     │
│ GPU 2   GPU 3      │◄─── NVSHMEM ──►│ GPU 2   GPU 3      │
│  ↕ IPC   ↕ IPC     │                │  ↕ IPC   ↕ IPC     │
│ GPU 4   GPU 5      │                │ GPU 4   GPU 5      │
│  ...               │                │  ...               │
└────────────────────┘                └────────────────────┘
```

- **Intra-node (IPC)**: GPUs on the same node communicate via CUDA IPC shared memory
- **Inter-node (NVSHMEM)**: GPUs on different nodes communicate via NVSHMEM symmetric heap over InfiniBand

### RDEP Mode Selection

The RDEP mode is automatically selected based on the distributed topology:

```python
# Mode selection logic (simplified)
if world_size == 1:
    mode = MODE_SINGLE
elif world_size == local_world_size:
    mode = MODE_IPC  # Single node
else:
    mode = MODE_HYBRID  # Multi-node
```

## Buffer Layout

### NVSHMEM Symmetric Heap

The symmetric heap is allocated once during initialization and contains:

| Buffer | Size | Purpose |
|--------|------|---------|
| `x_buf_bf16` | `capacity * H * 2` | BF16 activations |
| `x_buf_block` | `capacity * Hp` | Blockscaled packed data |
| `sfa_buf` | `capacity * Hsf` | Scale factors |
| `y_buf` | `capacity * H * 2` | Return buffer |
| `meta` | `capacity * sizeof(Meta)` | Dispatch metadata |
| `counter` | 4 bytes | Receive counter |
| `barrier_signals` | `MAX_NODES * 4` | RDMA barriers |

### IPC Buffers (separate from NVSHMEM)

IPC buffers are allocated via `cudaMalloc` (not `nvshmem_malloc`) because NVSHMEM memory cannot be used with CUDA IPC handles.

## Troubleshooting

### NVSHMEM Initialization Fails

```
RuntimeError: RDEP MODE_HYBRID requires NVSHMEM support. Rebuild rdep with NVSHMEM or use single-node.
```

**Fix:** Ensure nmoe was built with `WITH_NVSHMEM` defined. Check that `libnvshmem_host.so` is in `LD_LIBRARY_PATH`.

### Bootstrap UID Distribution Fails

```
NVSHMEM init hangs or times out
```

**Fix:**
1. Check that `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` matches your InfiniBand interface
2. Verify all nodes can reach each other on that interface
3. Check firewall rules allow NVSHMEM bootstrap traffic

### IBGDA Not Working

```
NVSHMEM falls back to slower transport
```

**Fix:**
1. Verify IBGDA support: `NVSHMEM_IB_ENABLE_IBGDA=1`
2. Check InfiniBand configuration: `ibv_devinfo`
3. Ensure NIC firmware supports GPUDirect Async

### OOM on NVSHMEM Heap

```
nvshmem_malloc failed
```

**Fix:**
1. Reduce `capacity` in RDEP configuration
2. Increase GPU memory available to NVSHMEM
3. Check `NVSHMEM_CUMEM_GRANULARITY` setting

## Verification

### Check NVSHMEM Build

```bash
# Verify library exists
ls -lh /path/to/nvshmem/build/src/lib/libnvshmem_host.so*

# Check if nmoe was built with NVSHMEM
python -c "from nmoe import _C; print('NVSHMEM:', _C.has_nvshmem())"
```

### Test Multi-Node Setup

```bash
# Two-node test
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node0:29500 \
    -m nmoe.train configs/dsv2.toml
```

## References

- [NVSHMEM Documentation](https://docs.nvidia.com/hpc-sdk/nvshmem/)
- [GPUDirect Async (IBGDA)](https://docs.nvidia.com/networking/display/rdmacore/gpudirect+async)
- [CUDA IPC](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication)
