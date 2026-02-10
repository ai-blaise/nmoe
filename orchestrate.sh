#!/bin/bash
set -euo pipefail

###############################################################################
# Orchestrator — Launch training on all nodes from dev box or head node
# Usage: ./orchestrate.sh [preflight|deploy|launch|kill]
###############################################################################

SSH_KEY="/home/nourdine/.ssh/google_compute_engine"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ${SSH_KEY}"
NMOE_DIR="/home/nourdine/nmoe"
CKPT_DIR="/home/nourdine/nmoe_checkpoints/reap345b_nvfp4"
LOG_DIR="${NMOE_DIR}/logs/orchestrator"

mkdir -p "${LOG_DIR}"

# ── Node List (RDMA IPs on 192.168.50.x) ────────────────────────────────────
# Update this list with all 32 nodes once provisioned.
# Current 14 asia-southeast1-b nodes:
RDMA_IPS=(
    192.168.50.19   # a4-rl-0ftn  (nic0: 10.148.0.15)
    192.168.50.16   # a4-rl-21cs  (nic0: 10.148.0.19)
    192.168.50.13   # a4-rl-69k1  (nic0: 10.148.0.13)
    192.168.50.23   # a4-rl-73b0  (nic0: 10.148.0.21)
    192.168.50.25   # a4-rl-7k0w  (nic0: 10.148.0.26)
    192.168.50.28   # a4-rl-87s2  (nic0: 10.148.0.28)
    192.168.50.15   # a4-rl-bfp0  (nic0: 10.148.0.22)
    192.168.50.24   # a4-rl-bs43  (nic0: 10.148.0.24)
    192.168.50.18   # a4-rl-dkhl  (nic0: 10.148.0.20)
    192.168.50.22   # a4-rl-nhnz  (nic0: 10.148.0.14)
    192.168.50.17   # a4-rl-rwhp  (nic0: 10.148.0.18)
    192.168.50.20   # a4-rl-s7r5  (nic0: 10.148.0.17)
    192.168.50.26   # a4-rl-sr8s  (nic0: 10.148.0.27)
    192.168.50.14   # a4-rl-z4v1  (nic0: 10.148.0.16)
)

# Corresponding nic0 IPs (for SSH from dev box — dev box may not route to 192.168.50.x)
NIC0_IPS=(
    10.148.0.15   # a4-rl-0ftn
    10.148.0.19   # a4-rl-21cs
    10.148.0.13   # a4-rl-69k1
    10.148.0.21   # a4-rl-73b0
    10.148.0.26   # a4-rl-7k0w
    10.148.0.28   # a4-rl-87s2
    10.148.0.22   # a4-rl-bfp0
    10.148.0.24   # a4-rl-bs43
    10.148.0.20   # a4-rl-dkhl
    10.148.0.14   # a4-rl-nhnz
    10.148.0.18   # a4-rl-rwhp
    10.148.0.17   # a4-rl-s7r5
    10.148.0.27   # a4-rl-sr8s
    10.148.0.16   # a4-rl-z4v1
)

NODE_NAMES=(
    a4-rl-0ftn a4-rl-21cs a4-rl-69k1 a4-rl-73b0 a4-rl-7k0w
    a4-rl-87s2 a4-rl-bfp0 a4-rl-bs43 a4-rl-dkhl a4-rl-nhnz
    a4-rl-rwhp a4-rl-s7r5 a4-rl-sr8s a4-rl-z4v1
)

HEAD_RDMA_IP="${RDMA_IPS[0]}"
NNODES="${#RDMA_IPS[@]}"

_ssh() {
    local ip="$1"; shift
    ssh ${SSH_OPTS} "${ip}" "$@"
}

# ── preflight: verify all nodes are healthy ──────────────────────────────────
cmd_preflight() {
    echo "=== Pre-Flight Check (${NNODES} nodes) ==="
    local failed=0

    for i in "${!NIC0_IPS[@]}"; do
        local ip="${NIC0_IPS[$i]}"
        local name="${NODE_NAMES[$i]}"
        local rdma="${RDMA_IPS[$i]}"

        result=$(_ssh "${ip}" "
            echo -n 'hostname='; hostname
            echo -n 'gpus='; nvidia-smi -L 2>/dev/null | wc -l
            echo -n 'rdma_ip='; ip addr show eth1 2>/dev/null | grep -oP '192\.168\.50\.\d+' | head -1
            echo -n 'python='; python3 --version 2>&1 | awk '{print \$2}'
            if [ -d ${NMOE_DIR}/.venv ]; then echo 'venv=yes'; else echo 'venv=no'; fi
            if [ -d ${CKPT_DIR} ]; then echo 'ckpt=yes'; else echo 'ckpt=no'; fi
            if [ -f ${NMOE_DIR}/.venv/bin/python ]; then
                ${NMOE_DIR}/.venv/bin/python -c 'import torch; print(f\"torch={torch.__version__}\"); print(f\"nccl={torch.cuda.nccl.version()}\")' 2>/dev/null || echo 'torch=FAIL'
            fi
        " 2>&1) || { echo "[FAIL] ${name} (${ip}): SSH failed"; ((failed++)); continue; }

        echo "[${name}] ${ip} | rdma=${rdma}"
        echo "${result}" | sed 's/^/  /'

        # Validate GPU count
        gpu_count=$(echo "${result}" | grep '^gpus=' | cut -d= -f2)
        if [ "${gpu_count}" != "8" ]; then
            echo "  [WARN] Expected 8 GPUs, got ${gpu_count}"
            ((failed++))
        fi
    done

    echo ""
    if [ "${failed}" -gt 0 ]; then
        echo "=== ${failed} issue(s) found ==="
        return 1
    else
        echo "=== All ${NNODES} nodes OK ==="
    fi
}

# ── deploy: rsync code + venv from head node to all others ───────────────────
cmd_deploy() {
    local src_ip="${NIC0_IPS[0]}"
    echo "=== Deploying from head node ${NODE_NAMES[0]} (${src_ip}) to ${NNODES} nodes ==="

    # First, sync from dev box to head node
    echo "[1/${NNODES}] Syncing to head node ${NODE_NAMES[0]}..."
    rsync -az --delete \
        --exclude='.git' \
        --exclude='logs/' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "${NMOE_DIR}/" \
        "${src_ip}:${NMOE_DIR}/" 2>&1 | tail -3

    # Then fan out from head node to all others (parallel)
    local pids=()
    for i in $(seq 1 $((NNODES - 1))); do
        local dst_ip="${NIC0_IPS[$i]}"
        local name="${NODE_NAMES[$i]}"
        echo "[${i}/${NNODES}] Syncing to ${name} (${dst_ip})..."
        _ssh "${src_ip}" "
            rsync -az --delete \
                --exclude='.git' \
                --exclude='logs/' \
                --exclude='__pycache__' \
                --exclude='*.pyc' \
                ${NMOE_DIR}/ ${dst_ip}:${NMOE_DIR}/
        " > "${LOG_DIR}/deploy_${name}.log" 2>&1 &
        pids+=($!)
    done

    # Wait for all
    local failed=0
    for i in $(seq 1 $((NNODES - 1))); do
        wait "${pids[$((i-1))]}" || { echo "[FAIL] ${NODE_NAMES[$i]}"; ((failed++)); }
    done

    if [ "${failed}" -gt 0 ]; then
        echo "=== ${failed} deploy(s) failed — check ${LOG_DIR}/deploy_*.log ==="
        return 1
    else
        echo "=== Deploy complete (${NNODES} nodes) ==="
    fi
}

# ── launch: start training on all nodes ──────────────────────────────────────
cmd_launch() {
    local steps="${1:-5}"
    echo "=== Launching training: ${NNODES} nodes × 8 GPUs | steps=${steps} ==="
    echo "    HEAD_RDMA_IP=${HEAD_RDMA_IP}:29500"
    echo ""

    local pids=()
    for i in "${!NIC0_IPS[@]}"; do
        local ip="${NIC0_IPS[$i]}"
        local name="${NODE_NAMES[$i]}"
        local rdma="${RDMA_IPS[$i]}"
        local logfile="${LOG_DIR}/train_${name}.log"

        echo "  Starting ${name} (${ip}, rdma=${rdma})..."
        _ssh "${ip}" "
            export HEAD_RDMA_IP=${HEAD_RDMA_IP}
            export MASTER_PORT=29500
            export NNODES=${NNODES}
            export NPROC=8
            export STEPS=${steps}
            export NCCL_DEBUG=INFO
            cd ${NMOE_DIR} && bash launch_32node.sh
        " > "${logfile}" 2>&1 &
        pids+=($!)
    done

    echo ""
    echo "All ${NNODES} nodes launched. Logs in ${LOG_DIR}/train_*.log"
    echo "Waiting for completion (Ctrl+C to detach, training continues)..."
    echo ""

    # Wait and report
    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" 2>/dev/null || {
            echo "[FAIL] ${NODE_NAMES[$i]} exited with error — see ${LOG_DIR}/train_${NODE_NAMES[$i]}.log"
            ((failed++))
        }
    done

    if [ "${failed}" -gt 0 ]; then
        echo "=== ${failed} node(s) failed ==="
        return 1
    else
        echo "=== Training complete ==="
    fi
}

# ── kill: stop training on all nodes ─────────────────────────────────────────
cmd_kill() {
    echo "=== Killing training on all ${NNODES} nodes ==="
    for i in "${!NIC0_IPS[@]}"; do
        local ip="${NIC0_IPS[$i]}"
        local name="${NODE_NAMES[$i]}"
        _ssh "${ip}" "pkill -f 'nmoe.train' 2>/dev/null; pkill -f 'torchrun' 2>/dev/null" 2>/dev/null || true
        echo "  ${name}: killed"
    done
    echo "=== Done ==="
}

# ── logs: tail logs from a specific node ─────────────────────────────────────
cmd_logs() {
    local node="${1:-${NODE_NAMES[0]}}"
    local logfile="${LOG_DIR}/train_${node}.log"
    if [ -f "${logfile}" ]; then
        tail -f "${logfile}"
    else
        echo "No log found at ${logfile}"
        echo "Available: $(ls ${LOG_DIR}/train_*.log 2>/dev/null | xargs -I{} basename {} .log | sed 's/train_//' | tr '\n' ' ')"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    preflight) cmd_preflight ;;
    deploy)    cmd_deploy ;;
    launch)    cmd_launch "${2:-5}" ;;
    kill)      cmd_kill ;;
    logs)      cmd_logs "${2:-}" ;;
    help|*)
        echo "Usage: $0 {preflight|deploy|launch [steps]|kill|logs [node]}"
        echo ""
        echo "  preflight  — Verify GPUs, software, checkpoint on all nodes"
        echo "  deploy     — Rsync code + venv from dev box → head → all nodes"
        echo "  launch N   — Start training on all nodes (default: 5 steps)"
        echo "  kill       — Kill training on all nodes"
        echo "  logs NODE  — Tail training log for a node (default: head)"
        ;;
esac
