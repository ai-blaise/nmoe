#!/usr/bin/env bash
# launch_sft.sh — SLURM launch script for REAP-345B SFT training
#
# 16 nodes x 8 B200 GPUs = 128 GPUs total
# EP=8 (expert parallelism), DP=16 (data parallelism)
#
# Usage:
#   sbatch scripts/launch_sft.sh
#   sbatch scripts/launch_sft.sh --export=NMOE_STEPS=100  # override steps
#
# Single-node debug (no SLURM):
#   bash scripts/launch_sft.sh --local
#
#SBATCH --job-name=reap345b-sft
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
NMOE_ROOT="${NMOE_ROOT:-/home/nourdine/nmoe}"
CONFIG="${CONFIG:-${NMOE_ROOT}/configs/dsv3_reap_sft.toml}"
VENV="${VENV:-${NMOE_ROOT}/.venv}"
GPUS_PER_NODE=8

# ============================================================================
# Local mode (single-node, no SLURM)
# ============================================================================
if [[ "${1:-}" == "--local" ]]; then
    shift
    echo "[launch] Local mode: single node, ${GPUS_PER_NODE} GPUs"
    source "${VENV}/bin/activate"
    cd "${NMOE_ROOT}"

    export NCCL_P2P_LEVEL=NVL
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    exec torchrun \
        --standalone \
        --nproc_per_node="${GPUS_PER_NODE}" \
        -m nmoe.train \
        "${CONFIG}" \
        "$@"
fi

# ============================================================================
# SLURM multi-node setup
# ============================================================================
echo "========================================"
echo "REAP-345B SFT Training"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Nodes:     ${SLURM_NNODES}"
echo "GPUs/node: ${GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Config:    ${CONFIG}"
echo "========================================"

# Rendezvous
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "[launch] MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "[launch] Node rank: ${SLURM_NODEID}  Node name: $(hostname)"

# ============================================================================
# NCCL configuration
# ============================================================================

# Intra-node: NVSwitch (NV18 all-to-all on B200 DGX/HGX)
export NCCL_P2P_LEVEL=NVL

# Inter-node: auto-detect best transport (IB/RoCE/TCP)
# These are safe defaults; override per-cluster as needed.
# For InfiniBand clusters:
#   export NCCL_IB_DISABLE=0
#   export NCCL_IB_HCA=mlx5  # or specific HCA
#   export NCCL_NET_GDR_LEVEL=5  # GPU Direct RDMA
# For RoCE/Ethernet clusters:
#   export NCCL_IB_DISABLE=1
#   export NCCL_SOCKET_IFNAME=eth0

# Auto-detect: let NCCL probe available transports
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# Socket interface for OOB communication (fallback / bootstrap)
# Common: eth0, ens5, ib0 — override via NCCL_SOCKET_IFNAME env var
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    # Auto-detect: pick the first UP interface that isn't loopback or docker
    NCCL_SOCKET_IFNAME=$(ip -o link show up | awk -F': ' '{print $2}' \
        | grep -v -E '^(lo|docker|veth|br-)' | head -n 1)
    export NCCL_SOCKET_IFNAME
fi

# Debug level (INFO for first run, WARN for production)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

# Timeout for large-model init (default 30min may not be enough for 345B load)
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT=1

echo "[launch] NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "[launch] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "[launch] NCCL_DEBUG=${NCCL_DEBUG}"

# ============================================================================
# PyTorch / CUDA configuration
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Deterministic ordering for reproducibility (small perf hit)
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Activate environment
# ============================================================================
source "${VENV}/bin/activate"
cd "${NMOE_ROOT}"

# Create log directory
mkdir -p logs

# ============================================================================
# Launch with srun + torchrun
# ============================================================================
# srun launches one task per node (ntasks-per-node=1).
# torchrun spawns 8 GPU workers per node.
# Rendezvous uses c10d (TCP) with SLURM job ID as run ID.

srun --kill-on-bad-exit=1 \
    torchrun \
        --nnodes="${SLURM_NNODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m nmoe.train \
        "${CONFIG}" \
        "$@"

echo "[launch] Job ${SLURM_JOB_ID} finished at $(date)"
