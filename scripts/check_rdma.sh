#!/bin/bash
# Check RDMA/RoCE configuration on B200 cluster nodes
#
# Usage: ./scripts/check_rdma.sh
#
# This script verifies:
# 1. RDMA hardware (mlx5 devices)
# 2. RoCE interfaces (gpu0rdma0-gpu7rdma0)
# 3. RDMA verbs availability
# 4. NCCL transport configuration
#
# Expected output on B200 cluster (8x ConnectX-7 RoCE NICs):
#   - 8x mlx5_N devices in ibv_devinfo
#   - gpu0rdma0-gpu7rdma0 RoCE interfaces (MTU 8896)
#   - RDMA verbs working
#   - NO IPoIB interfaces (RoCE is Ethernet-based)

set -euo pipefail

echo "=============================================="
echo "RDMA/RoCE Configuration Check (B200 Cluster)"
echo "=============================================="
echo ""

# 1. Check RDMA devices
echo "1. RDMA Devices (ibv_devinfo)"
echo "----------------------------------------------"
if command -v ibv_devinfo &> /dev/null; then
    RDMA_COUNT=$(ibv_devinfo 2>/dev/null | grep -c "mlx5_" || echo 0)
    echo "   Found $RDMA_COUNT mlx5 RDMA devices (expected: 8)"
    echo ""
    ibv_devinfo 2>/dev/null | head -60 || echo "   No RDMA devices found"
else
    echo "   ibv_devinfo not found - install rdma-core"
fi
echo ""

# 2. Check RoCE network interfaces (gpu0rdma0-gpu7rdma0)
echo "2. RoCE Network Interfaces"
echo "----------------------------------------------"
echo "   Looking for gpuXrdma0 interfaces (RoCE over Ethernet):"
ROCE_IFACES=$(ip link show 2>/dev/null | grep -E "^[0-9]+: gpu[0-7]rdma0" || true)
if [ -n "$ROCE_IFACES" ]; then
    echo "$ROCE_IFACES" | while read line; do
        IFACE=$(echo "$line" | cut -d: -f2 | tr -d ' ')
        MTU=$(ip link show "$IFACE" 2>/dev/null | grep -oP 'mtu \K[0-9]+' || echo "unknown")
        STATE=$(ip link show "$IFACE" 2>/dev/null | grep -oP 'state \K\w+' || echo "unknown")
        echo "   $IFACE: MTU=$MTU, state=$STATE"
    done
else
    echo "   No gpuXrdma0 interfaces found"
fi
echo ""

echo "   NOTE: RoCE clusters do NOT have IPoIB interfaces (no ib0/ibs0)"
echo "   Checking for IPoIB (should be empty):"
IPOIB_IFACES=$(ip link show 2>/dev/null | grep -E "^[0-9]+: (ib|ibs|ibd)" || true)
if [ -n "$IPOIB_IFACES" ]; then
    echo "   WARNING: IPoIB interfaces found (unexpected on RoCE cluster):"
    echo "$IPOIB_IFACES"
else
    echo "   None found (correct for RoCE)"
fi
echo ""

# 3. Check mlx5 kernel modules
echo "3. Mellanox Driver Modules"
echo "----------------------------------------------"
lsmod | grep -E "mlx5|rdma|ib_" | head -20 || echo "   No mlx5/RDMA modules loaded"
echo ""

# 4. Check RDMA device info for each mlx5 device
echo "4. RDMA Device Details"
echo "----------------------------------------------"
for i in {0..7}; do
    if ibv_devinfo mlx5_$i &>/dev/null 2>&1; then
        PORT_STATE=$(ibv_devinfo mlx5_$i 2>/dev/null | grep -E "^\s+state:" | head -1 | awk '{print $2}')
        LINK_LAYER=$(ibv_devinfo mlx5_$i 2>/dev/null | grep -E "^\s+link_layer:" | head -1 | awk '{print $2}')
        echo "   mlx5_$i: $PORT_STATE ($LINK_LAYER)"
    else
        echo "   mlx5_$i: NOT FOUND"
    fi
done
echo ""

# 5. Check environment variables
echo "5. Environment Variables"
echo "----------------------------------------------"
echo "   GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-not set (default: eth0)}"
echo "   NMOE_RDMA_COLLECTIVES: ${NMOE_RDMA_COLLECTIVES:-not set (default: 1)}"
echo "   NCCL_NET: ${NCCL_NET:-not set}"
echo "   NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-not set}"
echo "   NCCL_IB_HCA: ${NCCL_IB_HCA:-not set}"
echo "   NCCL_IB_GID_INDEX: ${NCCL_IB_GID_INDEX:-not set (should be 3 for RoCEv2)}"
echo ""

# 6. Check PyTorch distributed backends
echo "6. PyTorch Distributed Configuration"
echo "----------------------------------------------"
python3 -c "
import torch
import torch.distributed as dist

print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
print(f'   NCCL available: {dist.is_nccl_available()}')
print(f'   Gloo available: {dist.is_gloo_available()}')

if torch.cuda.is_available():
    print(f'   GPU count: {torch.cuda.device_count()}')
    print(f'   GPU 0: {torch.cuda.get_device_name(0)}')

print('')
print('   For RoCE clusters:')
print('   - NCCL uses mlx5 driver for RDMA (handles RoCE natively)')
print('   - Gloo uses TCP over eth0 for control plane only')
print('   - NMOE_RDMA_COLLECTIVES=1 (default) uses NCCL for fast object exchange')
" 2>/dev/null || echo "   Python check failed"
echo ""

# 7. Summary
echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""

# Check RDMA device count
RDMA_COUNT=$(ibv_devinfo 2>/dev/null | grep -c "mlx5_" || echo 0)
if [ "$RDMA_COUNT" -eq 8 ]; then
    echo "RDMA Devices: OK (8/8 mlx5 devices found)"
elif [ "$RDMA_COUNT" -gt 0 ]; then
    echo "RDMA Devices: PARTIAL ($RDMA_COUNT/8 mlx5 devices found)"
else
    echo "RDMA Devices: NOT FOUND"
fi

# Check RoCE interfaces
ROCE_COUNT=$(ip link show 2>/dev/null | grep -cE "^[0-9]+: gpu[0-7]rdma0" || echo 0)
if [ "$ROCE_COUNT" -eq 8 ]; then
    echo "RoCE Interfaces: OK (8/8 gpuXrdma0 interfaces found)"
elif [ "$ROCE_COUNT" -gt 0 ]; then
    echo "RoCE Interfaces: PARTIAL ($ROCE_COUNT/8 gpuXrdma0 interfaces found)"
else
    echo "RoCE Interfaces: NOT FOUND"
fi

echo ""
echo "CONFIGURATION:"
echo "  This cluster uses RoCE (RDMA over Converged Ethernet)"
echo "  - Data path: NCCL uses mlx5_0-mlx5_7 via RoCE"
echo "  - Control path: Gloo uses eth0 via TCP"
echo "  - NMOE_RDMA_COLLECTIVES=1 (default) for fast CPU object exchange"
echo ""
echo "  Performance:"
echo "    - NCCL RoCE (data path):      ~10us, 400 Gbps per NIC"
echo "    - NCCL RDMA collectives:      ~10us (object exchange)"
echo "    - Gloo TCP (control plane):   ~100us (fallback only)"
echo ""
