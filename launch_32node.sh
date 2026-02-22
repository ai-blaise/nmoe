#!/bin/bash
set -euo pipefail

###############################################################################
# Multi-Node Launch Script — DeepSeek V3.2 REAP-345B NVFP4 SFT
# 32 nodes × 8 GPUs = 256 GPUs | EP=8 (NVLink) + DP=32 (RDMA)
###############################################################################

# ── Configuration ────────────────────────────────────────────────────────────
HEAD_RDMA_IP="${HEAD_RDMA_IP:?Set HEAD_RDMA_IP to the RDMA (192.168.50.x) IP of the head node}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-32}"
NPROC="${NPROC:-8}"
CONFIG="${CONFIG:-configs/dsv3_reap_sft_32node.toml}"
STEPS="${STEPS:-5}"
NMOE_DIR="${NMOE_DIR:-/home/nourdine/nmoe}"
LOG_DIR="${NMOE_DIR}/logs"

mkdir -p "${LOG_DIR}"

# ── NCCL — GPUDirect-TCPXO over gVNIC (nic1 = 192.168.50.x) ────────────────
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
export NCCL_NET="${NCCL_NET:-GPUDirectTCPX}"
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4

# Debug logging (set NCCL_DEBUG=WARN for production)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

# ── CUDA ─────────────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── nmoe no-fallback production contract ─────────────────────────────────────
export NMOE_USE_FA4=1
export NMOE_REQUIRE_FLASHMLA=1
export NMOE_PACKED_ATTN_BACKEND=flashmla
export NMOE_ROUTER_BWD_ALLOW_STANDALONE=0
export NMOE_AUX_LOSS_ALLOW_STANDALONE=0

# ── Activate venv ────────────────────────────────────────────────────────────
source "${NMOE_DIR}/.venv/bin/activate"

# ── Launch ───────────────────────────────────────────────────────────────────
HOSTNAME=$(hostname)
echo "[$(date -Is)] Launching on ${HOSTNAME} | HEAD=${HEAD_RDMA_IP}:${MASTER_PORT} | ${NNODES}×${NPROC} GPUs"

exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${HEAD_RDMA_IP}:${MASTER_PORT}" \
    --rdzv_conf="timeout=900,read_timeout=600" \
    --redirects=3 \
    --log_dir="${LOG_DIR}" \
    -m nmoe.train \
    "${CONFIG}" \
    --steps="${STEPS}"
