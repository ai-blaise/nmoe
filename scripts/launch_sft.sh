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
# nmoe no-fallback production contract
# ============================================================================
export NMOE_USE_FA4=1
export NMOE_REQUIRE_FLASHMLA=1
export NMOE_PACKED_ATTN_BACKEND=flashmla
export NMOE_ROUTER_BWD_ALLOW_STANDALONE=0
export NMOE_AUX_LOSS_ALLOW_STANDALONE=0

# ============================================================================
# Local mode (single-node, no SLURM)
# ============================================================================
if [[ "${1:-}" == "--local" ]]; then
    shift
    echo "[launch] Local mode: single node, ${GPUS_PER_NODE} GPUs"
    source "${VENV}/bin/activate"
    cd "${NMOE_ROOT}"

    export NCCL_NET=Socket
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_P2P_LEVEL=NVL
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_NVLS_ENABLE=0
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
    export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
    export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
    export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-16777216}"
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
    export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
    export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring}"
    export NCCL_PROTO="${NCCL_PROTO:-Simple,LL128,LL}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
    export NMOE_ZERO2_RS_CHUNK_MB="${NMOE_ZERO2_RS_CHUNK_MB:-512}"
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

# TCP-only cluster defaults (single 200 Gbps gVNIC; no RDMA, no GPUDirect).
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-16777216}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring}"
export NCCL_PROTO="${NCCL_PROTO:-Simple,LL128,LL}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
export NMOE_ZERO2_RS_CHUNK_MB="${NMOE_ZERO2_RS_CHUNK_MB:-512}"

# Debug level (INFO for first run, WARN for production)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"

# Timeout for large-model init (default 30min may not be enough for 345B load)
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"

echo "[launch] NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "[launch] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "[launch] NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS}"
echo "[launch] NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD}"
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
