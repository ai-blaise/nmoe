# Multi-Node Training Deployment Plan — DeepSeek V3.2 REAP-345B NVFP4 SFT

## Cluster Topology

- **Nodes**: 32 (a4-rl class, 8x NVIDIA B200 183 GiB HBM each)
- **Total GPUs**: 256
- **Parallelism**: EP=8 (intra-node, NVLink) + DP=32 (inter-node, RDMA)
- **Network**: Dual NIC per node
  - nic0: main VPC (10.x.x.x) — SSH, management, NFS/checkpoint I/O
  - nic1: RDMA network (192.168.50.x) — GPUDirect-TCPXO/gVNIC for NCCL collectives
- **No InfiniBand**: GCP uses GPUDirect-TCPXO over gVNIC, not IB verbs

## Memory Budget (per GPU, DP=32)

| Component | GiB |
|-----------|-----|
| Dense/Attention BF16 parameters | 32.00 |
| NVFP4 expert buffers (packed + scale + gs) | 23.00 |
| FP8 ECO optimizer states (m + v, 58 layers × 3 weights × 16 experts) | 76.24 |
| ZeRO-2 dense optimizer (FP32 master + m + v, sharded /32) | 5.96 |
| **Total static** | **137.20** |
| **Headroom (of 178.35 GiB)** | **41.15** |

Headroom is used for: blockscaled weight cache (~23 GiB, rebuilt lazily per-layer in forward),
transient BF16 dequant (~1.3 GiB peak), gradients, activation checkpoints, NCCL buffers.
41 GiB is comfortable — single-node DP=1 had only ~26 GiB and OOM'd.

---

## Phase 1: Designate Head Node

Pick one node as the rendezvous/head node. All other nodes connect to it for torch.distributed init.

```
HEAD_NODE=<hostname>          # e.g. a4-rl-0ftn
HEAD_RDMA_IP=<192.168.50.x>  # RDMA IP of head node
MASTER_PORT=29500
```

The head node runs rank 0..7, second node runs rank 8..15, etc.

---

## Phase 2: Software Environment Setup (All 32 Nodes)

### 2a. Python Version

Current dev box (a4-us-002-rl9) has Python 3.13. The asia a4-rl nodes have Python 3.14.
Compiled .so extensions (rdep, eco_adam, quant kernels) are NOT portable across Python minor versions.

**Option A (recommended)**: Build everything on an a4-rl node natively.
- SSH into the head node
- Clone/copy nmoe source
- Create venv with Python 3.14
- Build all extensions from source on that node
- Distribute the built venv + compiled .so to all other nodes

**Option B**: Install Python 3.13 on all a4-rl nodes via deadsnakes/pyenv.
- More fragile, prefer Option A.

### 2b. Code + Venv Deployment

Since there's no shared filesystem (GCP VMs have local disks only), we need to push
the code and venv to each node. Two approaches:

**Approach 1 — rsync from head node (simple)**:
```bash
# On head node, after building:
NODES="a4-rl-21cs a4-rl-69k1 ..."  # all 31 other nodes by nic0 IP
for NODE_IP in $NODES; do
    rsync -az --delete \
        /home/nourdine/nmoe/ \
        $NODE_IP:/home/nourdine/nmoe/ &
done
wait
```

**Approach 2 — GCS bucket (parallel, robust)**:
```bash
# On head node after building:
gsutil -m cp -r /home/nourdine/nmoe gs://your-bucket/nmoe/

# On each node (can be done in parallel via SSH):
gsutil -m cp -r gs://your-bucket/nmoe/ /home/nourdine/nmoe/
```

### 2c. CUDA / Driver Verification

All nodes must have matching:
- NVIDIA driver version (check: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`)
- CUDA toolkit version (check: `nvcc --version`)
- NCCL version (check: `python -c "import torch; print(torch.cuda.nccl.version())"`)

Run pre-flight on all nodes:
```bash
for IP in ${ALL_IPS[@]}; do
    ssh $IP "nvidia-smi -L && python3 -c 'import torch; print(torch.cuda.nccl.version())'" &
done
wait
```

### 2d. Build Extensions on Head Node

```bash
ssh $HEAD_NODE
cd /home/nourdine/nmoe
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision  # or whatever the project needs
pip install -e .               # builds rdep, eco_adam, quant kernels from csrc/
# Verify:
python -c "from nmoe import rdep; print('rdep OK')"
python -c "from nmoe.csrc import eco_adam; print('eco_adam OK')"
```

---

## Phase 3: Checkpoint Distribution

The NVFP4 checkpoint at `/home/nourdine/nmoe_checkpoints/reap345b_nvfp4/` is ~50 GiB.
Each node needs a local copy (no shared filesystem).

**Option A — GCS (recommended for 32 nodes)**:
```bash
# Upload once:
gsutil -m cp -r /home/nourdine/nmoe_checkpoints/reap345b_nvfp4/ \
    gs://your-bucket/checkpoints/reap345b_nvfp4/

# On each node:
gsutil -m cp -r gs://your-bucket/checkpoints/reap345b_nvfp4/ \
    /home/nourdine/nmoe_checkpoints/reap345b_nvfp4/
```

**Option B — rsync from dev box** (slower, sequential bottleneck):
```bash
for IP in ${ALL_IPS[@]}; do
    rsync -az /home/nourdine/nmoe_checkpoints/reap345b_nvfp4/ \
        $IP:/home/nourdine/nmoe_checkpoints/reap345b_nvfp4/ &
done
wait
```

**Option C — NFS/Filestore**: Mount a GCP Filestore instance on all nodes.
Most robust for iterative development but requires setup.

---

## Phase 4: NCCL Configuration for GPUDirect-TCPXO

GCP A3/A4 VMs do NOT use InfiniBand. They use GPUDirect-TCPXO over gVNIC on nic1.
NCCL must be configured to use the RDMA network, not nic0.

### 4a. Environment Variables (set on every node before launch)

```bash
# Force NCCL to use the RDMA NIC (nic1 = 192.168.50.x)
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1

# GPUDirect-TCPXO plugin (should be pre-installed on GCP A4 images)
export NCCL_NET=GPUDirectTCPX
# If the above doesn't work, try:
# export NCCL_NET_GDR_LEVEL=PIX
# export NCCL_P2P_LEVEL=PXB

# NCCL tuning for 32-node
export NCCL_IB_DISABLE=1              # No InfiniBand hardware
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_BUFFSIZE=8388608           # 8 MiB buffer
export NCCL_MIN_NCHANNELS=4

# Debug (enable during first run, disable for production)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

### 4b. Verify NCCL Connectivity (pre-flight)

Run a 2-node NCCL allreduce test before the full 32-node launch:
```bash
# On head node:
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${HEAD_RDMA_IP}:${MASTER_PORT} \
    --node_rank=0 \
    -m torch.distributed.test_nccl

# On second node (simultaneously):
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${HEAD_RDMA_IP}:${MASTER_PORT} \
    --node_rank=1 \
    -m torch.distributed.test_nccl
```

If NCCL can't find the GPUDirect plugin, fall back to socket transport:
```bash
export NCCL_NET=Socket
```
This is slower but functional. Debug with `NCCL_DEBUG=INFO` to see which transport is selected.

---

## Phase 5: Multi-Node Training Config

### 5a. New Config File: `configs/dsv3_reap_sft_32node.toml`

```toml
[model]
checkpoint = "/home/nourdine/nmoe_checkpoints/reap345b_nvfp4/"
nvfp4_primary = true

[training]
steps = 100
batch_size = 8                    # per-GPU micro batch
seq_len = 512
gradient_checkpointing = true
eco_fused_backward = true
gradient_accumulation_steps = 1   # effective global batch = 8 * 256 = 2048

[optimizer]
type = "eco_adamw"
lr = 1e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
fp8_format = "e5m2_e4m3"

[distributed]
expert_parallelism = 8            # intra-node (NVLink)
data_parallelism = 32             # inter-node (RDMA)
zero_stage = 2                    # shard optimizer states across DP ranks
backend = "nccl"

[data]
dataset = "/path/to/sft_dataset"
num_workers = 4
```

### 5b. Process Group Layout

```
World size = 256 (32 nodes × 8 GPUs)

EP groups (size=8, intra-node):
  [0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15], ..., [248,...,255]

DP groups (size=32, inter-node, same local rank):
  [0,8,16,24,...,248], [1,9,17,25,...,249], ..., [7,15,23,...,255]

ZeRO-2 shards dense optimizer across each DP group of 32.
Expert optimizer (FP8 ECO) is NOT sharded — each GPU owns its 16 local experts fully.
```

---

## Phase 6: Launch Procedure

### 6a. Hostfile

Create `/home/nourdine/nmoe/hostfile.txt` with RDMA IPs:
```
192.168.50.XX slots=8
192.168.50.XX slots=8
... (32 lines, one per node)
```

### 6b. Launch Script: `launch_32node.sh`

```bash
#!/bin/bash
set -euo pipefail

HEAD_RDMA_IP="192.168.50.XX"   # RDMA IP of head node
MASTER_PORT=29500
NNODES=32
NPROC=8

# NCCL config
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
export NCCL_NET=GPUDirectTCPX
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_BUFFSIZE=8388608
export NCCL_DEBUG=INFO

# Activate venv
source /home/nourdine/nmoe/.venv/bin/activate

# Launch with torchrun (elastic)
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${HEAD_RDMA_IP}:${MASTER_PORT} \
    -m nmoe.train \
    configs/dsv3_reap_sft_32node.toml \
    --steps=5
```

### 6c. Multi-Node Orchestration

Each node must run the same launch script. Use `pdsh`, `pssh`, or a loop:

```bash
# From dev box or head node:
ALL_IPS=(192.168.50.13 192.168.50.14 ... )  # all 32 RDMA IPs

for i in "${!ALL_IPS[@]}"; do
    IP=${ALL_IPS[$i]}
    ssh -o StrictHostKeyChecking=no -i /home/nourdine/.ssh/google_compute_engine \
        ${IP} "cd /home/nourdine/nmoe && bash launch_32node.sh" \
        > /home/nourdine/nmoe/logs/node_${i}.log 2>&1 &
done
wait
```

torchrun with `rdzv_backend=c10d` handles elastic rendezvous — all 32 nodes connect
to `HEAD_RDMA_IP:29500` and are assigned ranks automatically. No manual `--node_rank` needed.

---

## Phase 7: Pre-Flight Checklist

Run these checks before the first training launch:

### 7a. Hardware Verification (all 32 nodes)
```bash
# Verify 8x B200 on every node
for IP in ${ALL_IPS[@]}; do
    echo "=== $IP ==="
    ssh $IP "nvidia-smi -L | wc -l"  # expect: 8
done
```

### 7b. Software Verification (all 32 nodes)
```bash
for IP in ${ALL_IPS[@]}; do
    echo "=== $IP ==="
    ssh $IP "
        source /home/nourdine/nmoe/.venv/bin/activate
        python -c '
import torch
print(f\"torch={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}\")
print(f\"GPUs={torch.cuda.device_count()}\")
from nmoe import rdep
print(\"rdep OK\")
'
    "
done
```

### 7c. NCCL 2-Node Smoke Test
```bash
# Head node + one other node, verify allreduce works over RDMA
# Use the NCCL connectivity test from Phase 4b
```

### 7d. Checkpoint Presence (all 32 nodes)
```bash
for IP in ${ALL_IPS[@]}; do
    ssh $IP "ls -la /home/nourdine/nmoe_checkpoints/reap345b_nvfp4/ | head -5"
done
```

---

## Phase 8: Monitoring and Debugging

### 8a. Per-Node Logs
Each node writes to `/home/nourdine/nmoe/logs/node_${i}.log`.
Tail from dev box:
```bash
ssh ${IP} "tail -f /home/nourdine/nmoe/logs/node_${i}.log"
```

### 8b. GPU Memory Monitoring
```bash
# On any node during training:
watch -n 2 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
```

### 8c. NCCL Debugging
If hangs occur, check for:
- Mismatched NCCL versions across nodes
- Firewall blocking 192.168.50.x traffic (port 29500 + ephemeral NCCL ports)
- GPUDirect-TCPXO plugin missing → fall back to `NCCL_NET=Socket`

### 8d. Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Timeout at init_process_group | Firewall on nic1 / wrong IFNAME | Check `NCCL_SOCKET_IFNAME=eth1`, open ports |
| Rank X stuck, others timeout | OOM on rank X | Check `nvidia-smi` on that node |
| NCCL error: no usable NIC | NCCL can't find RDMA device | Set `NCCL_NET=Socket` as fallback |
| Slow allreduce (>100ms for 1MB) | Using nic0 instead of nic1 | Verify NCCL_DEBUG shows 192.168.50.x |
| Python/SO version mismatch | Didn't rebuild on target Python | Rebuild .so on a4-rl node with Python 3.14 |

---

## Phase 9: Scaling Notes

### DP=32 vs DP=14 (current 14 nodes)

| | DP=14 | DP=32 |
|---|---|---|
| ZeRO-2 dense opt/GPU | 13.62 GiB | 5.96 GiB |
| Total static/GPU | 144.62 GiB | 137.20 GiB |
| Headroom/GPU | 33.73 GiB | 41.15 GiB |
| Global batch (bs=8) | 112 × 8 = 896 | 256 × 8 = 2048 |
| DP allreduce volume | 32 GiB / 14 peers | 32 GiB / 32 peers |

More nodes = more headroom + better ZeRO-2 sharding. The inter-node allreduce
cost grows logarithmically with node count (ring allreduce), not linearly.

### Gradient Accumulation

If global batch 2048 is too large for SFT convergence, reduce per-GPU batch or add
gradient accumulation steps to decouple global batch from GPU count:
```toml
batch_size = 2
gradient_accumulation_steps = 4
# effective global batch = 2 * 4 * 256 = 2048 (same, but smaller micro-batch)
```

---

## Summary of Steps

1. **Pick head node**, note its RDMA IP
2. **Build venv + extensions on head node** (Python 3.14 native)
3. **Distribute code + venv** to all 32 nodes (rsync or GCS)
4. **Distribute checkpoint** to all 32 nodes (GCS recommended)
5. **Set NCCL env vars** for GPUDirect-TCPXO on nic1
6. **Run 2-node NCCL smoke test** before full launch
7. **Run pre-flight checks** (GPUs, software, checkpoint presence)
8. **Launch on all 32 nodes** via SSH + torchrun elastic rendezvous
9. **Monitor logs + GPU memory** during first 5 steps
10. **Scale to full training** once smoke test passes
