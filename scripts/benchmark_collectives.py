#!/usr/bin/env python3
"""Benchmark CPU collective transport options for RDEP bootstrap.

This script measures the latency of different collective transport options
on a B200 cluster with InfiniBand RDMA networking.

Usage:
    # Run on 2+ nodes with torchrun
    torchrun --nnodes=2 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$HEAD_IP:29500 \
        scripts/benchmark_collectives.py

    # Compare RDMA vs Gloo
    NMOE_RDMA_COLLECTIVES=0 torchrun ... scripts/benchmark_collectives.py  # Gloo only
    NMOE_RDMA_COLLECTIVES=1 torchrun ... scripts/benchmark_collectives.py  # NCCL RDMA

Expected Results (16 nodes x 8 GPUs = 128 ranks):
    - Gloo over TCP (eth0):   ~100us per operation
    - Gloo over IPoIB (ib0):  ~50us per operation
    - NCCL RDMA:              ~10us per operation

The NCCL RDMA path is 10x faster than Gloo TCP because it uses the same
InfiniBand transport as GPU collectives instead of TCP sockets.
"""

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def benchmark_gloo_broadcast(group, warmup: int, iterations: int, obj_size: int) -> float:
    """Benchmark Gloo broadcast_object_list."""
    rank = dist.get_rank(group)

    # Create test object
    test_obj = {"data": b"x" * obj_size, "rank": rank}

    # Warmup
    for _ in range(warmup):
        obj_list = [test_obj if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0, group=group)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        obj_list = [test_obj if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0, group=group)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1e6  # Return microseconds


def benchmark_gloo_allgather(group, warmup: int, iterations: int, obj_size: int) -> float:
    """Benchmark Gloo all_gather_object."""
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # Create test object
    test_obj = {"data": b"x" * obj_size, "rank": rank}

    # Warmup
    for _ in range(warmup):
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, test_obj, group=group)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, test_obj, group=group)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1e6  # Return microseconds


def benchmark_rdma_broadcast(warmup: int, iterations: int, obj_size: int) -> float:
    """Benchmark NCCL-based RDMA broadcast."""
    from nmoe.rdma_collectives import RDMACollectives

    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    rdma = RDMACollectives(nccl_group=None, device=device)

    # Create test object
    test_obj = {"data": b"x" * obj_size, "rank": rank}

    # Warmup
    for _ in range(warmup):
        rdma.broadcast_object(test_obj if rank == 0 else None, src=0)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        rdma.broadcast_object(test_obj if rank == 0 else None, src=0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1e6  # Return microseconds


def benchmark_rdma_allgather(warmup: int, iterations: int, obj_size: int) -> float:
    """Benchmark NCCL-based RDMA all_gather."""
    from nmoe.rdma_collectives import RDMACollectives

    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    rdma = RDMACollectives(nccl_group=None, device=device)

    # Create test object
    test_obj = {"data": b"x" * obj_size, "rank": rank}

    # Warmup
    for _ in range(warmup):
        rdma.all_gather_object(test_obj)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        rdma.all_gather_object(test_obj)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1e6  # Return microseconds


def check_ipoib_interface():
    """Check if IPoIB interface is available."""
    from nmoe.rdma_collectives import detect_ipoib_interface
    return detect_ipoib_interface()


def main():
    parser = argparse.ArgumentParser(description="Benchmark collective transports")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--obj-size", type=int, default=128, help="Object size in bytes")
    args = parser.parse_args()

    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"RDEP Bootstrap Collective Benchmark")
        print(f"{'='*70}")
        print(f"World size: {world_size}")
        print(f"Object size: {args.obj_size} bytes")
        print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")

        ipoib = check_ipoib_interface()
        if ipoib:
            print(f"IPoIB interface: {ipoib} (detected)")
        else:
            print("IPoIB interface: NOT FOUND (using TCP fallback)")

        print(f"NMOE_RDMA_COLLECTIVES: {os.environ.get('NMOE_RDMA_COLLECTIVES', '1')}")
        print(f"{'='*70}\n")

    # Create Gloo group for comparison
    gloo_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))

    # Synchronize before benchmarks
    dist.barrier()

    # Benchmark Gloo broadcast
    gloo_bcast_us = benchmark_gloo_broadcast(gloo_group, args.warmup, args.iterations, args.obj_size)
    dist.barrier()

    # Benchmark Gloo all_gather
    gloo_gather_us = benchmark_gloo_allgather(gloo_group, args.warmup, args.iterations, args.obj_size)
    dist.barrier()

    # Benchmark RDMA broadcast
    rdma_bcast_us = benchmark_rdma_broadcast(args.warmup, args.iterations, args.obj_size)
    dist.barrier()

    # Benchmark RDMA all_gather
    rdma_gather_us = benchmark_rdma_allgather(args.warmup, args.iterations, args.obj_size)
    dist.barrier()

    # Print results (rank 0 only)
    if rank == 0:
        print(f"{'Operation':<30} {'Gloo (us)':<15} {'NCCL RDMA (us)':<15} {'Speedup':<10}")
        print(f"{'-'*70}")
        print(f"{'broadcast_object':<30} {gloo_bcast_us:<15.1f} {rdma_bcast_us:<15.1f} {gloo_bcast_us/rdma_bcast_us:.1f}x")
        print(f"{'all_gather_object':<30} {gloo_gather_us:<15.1f} {rdma_gather_us:<15.1f} {gloo_gather_us/rdma_gather_us:.1f}x")
        print(f"\n{'='*70}")

        # Estimate bootstrap time savings (typical RDEP hybrid setup)
        # 1 broadcast (UID) + 3 all_gathers (hosts, ranks, handles) + 1 barrier
        gloo_bootstrap = gloo_bcast_us + 3 * gloo_gather_us
        rdma_bootstrap = rdma_bcast_us + 3 * rdma_gather_us

        print(f"\nEstimated RDEP hybrid bootstrap time:")
        print(f"  Gloo (TCP/IPoIB):  {gloo_bootstrap:.0f} us")
        print(f"  NCCL (RDMA):       {rdma_bootstrap:.0f} us")
        print(f"  Savings:           {gloo_bootstrap - rdma_bootstrap:.0f} us ({gloo_bootstrap/rdma_bootstrap:.1f}x faster)")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
