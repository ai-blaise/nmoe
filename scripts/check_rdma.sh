#!/bin/bash
# Check RDMA and IPoIB configuration on B200 cluster nodes
#
# Usage: ./scripts/check_rdma.sh
#
# This script verifies:
# 1. InfiniBand hardware (mlx5 devices)
# 2. IPoIB interfaces (ib0, ibs0, etc.)
# 3. RDMA verbs availability
# 4. PyTorch Gloo ibverbs support
#
# Expected output on B200 cluster (8x mlx5 RDMA NICs):
#   - 8x mlx5_N devices in ibstat
#   - ib0-ib7 IPoIB interfaces (if configured)
#   - RDMA verbs working (ibv_devinfo)

set -euo pipefail

echo "=============================================="
echo "RDMA/IPoIB Configuration Check"
echo "=============================================="
echo ""

# 1. Check InfiniBand hardware
echo "1. InfiniBand Hardware (ibstat)"
echo "----------------------------------------------"
if command -v ibstat &> /dev/null; then
    ibstat | head -40
else
    echo "   ibstat not found - install rdma-core/infiniband-diags"
fi
echo ""

# 2. Check network interfaces for IPoIB
echo "2. Network Interfaces (looking for IPoIB)"
echo "----------------------------------------------"
echo "   IPoIB interfaces (ib*, ibs*, ibd*):"
ip link show 2>/dev/null | grep -E "^[0-9]+: (ib|ibs|ibd)" || echo "   No IPoIB interfaces found"
echo ""
echo "   All interfaces:"
ip link show 2>/dev/null | grep -E "^[0-9]+:" | head -20
echo ""

# 3. Check RDMA devices
echo "3. RDMA Devices (ibv_devinfo)"
echo "----------------------------------------------"
if command -v ibv_devinfo &> /dev/null; then
    ibv_devinfo 2>/dev/null | head -50 || echo "   No RDMA devices found"
else
    echo "   ibv_devinfo not found - install rdma-core"
fi
echo ""

# 4. Check mlx5 kernel modules
echo "4. Mellanox Driver Modules"
echo "----------------------------------------------"
lsmod | grep -E "mlx5|rdma|ib_" | head -20 || echo "   No mlx5/RDMA modules loaded"
echo ""

# 5. Check PyTorch Gloo configuration
echo "5. PyTorch Gloo Configuration"
echo "----------------------------------------------"
python3 -c "
import torch
import torch.distributed as dist

print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
print(f'   NCCL available: {dist.is_nccl_available()}')
print(f'   Gloo available: {dist.is_gloo_available()}')

# Check if Gloo was built with ibverbs
# This is tricky - standard PyTorch pip packages do NOT include ibverbs
# You need to build from source with USE_GLOO_IBVERBS=1
try:
    import torch._C
    if hasattr(torch._C, '_distributed_c10d'):
        print('   Gloo C10d module: loaded')
except:
    print('   Gloo C10d module: not available')

print('')
print('   NOTE: Standard PyTorch packages do NOT include Gloo ibverbs.')
print('   For native RDMA with Gloo, build PyTorch from source:')
print('     USE_DISTRIBUTED=1 USE_GLOO_IBVERBS=1 python setup.py install')
print('')
print('   RECOMMENDED: Use NMOE_RDMA_COLLECTIVES=1 (default) which uses')
print('   NCCL-based object exchange for RDMA transport.')
" 2>/dev/null || echo "   Python check failed"
echo ""

# 6. Check environment variables
echo "6. Environment Variables"
echo "----------------------------------------------"
echo "   GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-not set}"
echo "   NMOE_RDMA_COLLECTIVES: ${NMOE_RDMA_COLLECTIVES:-not set (default: 1)}"
echo "   NCCL_NET: ${NCCL_NET:-not set}"
echo "   NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-not set}"
echo "   NCCL_IB_HCA: ${NCCL_IB_HCA:-not set}"
echo ""

# 7. Check NCCL transport
echo "7. NCCL Transport Test"
echo "----------------------------------------------"
python3 -c "
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'NET'

import torch
if torch.cuda.is_available():
    # This will print NCCL initialization info
    t = torch.zeros(1, device='cuda')
    print('   CUDA tensor created - NCCL will use available transports')
    print('   Check for \"NET/IB\" or \"NET/Socket\" in debug output')
else:
    print('   No CUDA device available')
" 2>&1 | head -20 || echo "   NCCL test failed"
echo ""

# 8. Summary
echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""

# Check IPoIB
IPOIB_IF=$(ip link show 2>/dev/null | grep -E "^[0-9]+: ib" | head -1 | cut -d: -f2 | tr -d ' ')
if [ -n "$IPOIB_IF" ]; then
    echo "IPoIB:    AVAILABLE ($IPOIB_IF)"
    echo "          Set GLOO_SOCKET_IFNAME=$IPOIB_IF for ~50us latency"
else
    echo "IPoIB:    NOT CONFIGURED"
    echo "          Using TCP over GVNIC (~100us latency)"
fi
echo ""

# Check RDMA collectives recommendation
echo "RECOMMENDATION:"
echo "  For fastest CPU collectives, use NMOE_RDMA_COLLECTIVES=1 (default)"
echo "  This uses NCCL-based object exchange (~10us latency over RDMA)"
echo ""
echo "  Performance hierarchy:"
echo "    1. NCCL RDMA (NMOE_RDMA_COLLECTIVES=1):  ~10us"
echo "    2. Gloo IPoIB (GLOO_SOCKET_IFNAME=ib0):  ~50us"
echo "    3. Gloo TCP (GLOO_SOCKET_IFNAME=eth0):   ~100us"
echo ""
