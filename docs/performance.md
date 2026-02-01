# Performance Tuning Guide

This guide covers profiling, benchmarking, and optimization techniques for nmoe.

## Overview

nmoe performance depends on:

1. **Quantization Profile** - BF16, FP8, or NVFP4
2. **Parallelism Strategy** - EP, TP, and their combination
3. **Buffer Configuration** - RDEP capacity and memory layout
4. **Batch Size** - Throughput vs latency trade-off
5. **Hardware Utilization** - NVLink, memory bandwidth

## Profiling Tools

### Built-in Benchmarks

```bash
# Benchmark RDEP dispatch/gather kernels
python -m tests.bench_rdep_kernels \
    --profile nvfp4 \
    --T 4096 --H 2048 --E 8 --K 2 \
    --iters 200

# Benchmark full MoE forward/backward
python -m nmoe.bench_moe_e2e \
    --profile nvfp4 \
    --T 4096 --H 2048 --Dff 1408 --E 8 --K 2
```

### NVIDIA Nsight Systems

Profile GPU kernels and identify bottlenecks:

```bash
# Profile training
nsys profile -o nmoe_profile \
    python -m nmoe.train --config config.yaml

# Analyze timeline
nsys stats nmoe_profile.nsys-rep
```

Key metrics to check:
- **Kernel duration** - Are MoE kernels taking expected time?
- **Memory transfers** - Any unexpected H2D/D2H copies?
- **Idle time** - Gaps between kernels?

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("moe_forward"):
        output = moe_layer(x, expert_ids, gates)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

## Quantization Selection

### BF16 (Baseline)

```yaml
quantization: bf16
```

**When to use:**
- Debugging and development
- Maximum numerical precision needed
- Memory not constrained

**Performance:**
- 1x throughput (baseline)
- 1x memory usage

### FP8

```yaml
quantization: fp8
```

**When to use:**
- Production training on Hopper/Blackwell
- Good precision/throughput balance
- 2x memory savings needed

**Performance:**
- 1.5-2x throughput vs BF16
- 0.5x memory usage

### NVFP4

```yaml
quantization: nvfp4
```

**When to use:**
- Production on Blackwell (B200)
- Maximum throughput needed
- 4x memory savings needed
- Can tolerate slight precision loss

**Performance:**
- 2-3x throughput vs BF16
- 0.25x memory usage

## Memory Optimization

### Reducing Memory Footprint

#### 1. Gradient Checkpointing

Trade compute for memory:

```yaml
training:
  gradient_checkpointing: true
  checkpoint_layers: [0, 4, 8, 12, 16, 20, 24, 28]
```

Saves ~40% activation memory at cost of ~25% more compute.

#### 2. Reduce RDEP Buffer Capacity

Default is often oversized:

```bash
# Default
export NMOE_RDEP_CAPACITY=65536

# Reduce for smaller batches
export NMOE_RDEP_CAPACITY=16384
```

Memory formula:
```
RDEP memory ≈ capacity × hidden_dim × 6 bytes
```

#### 3. Mixed Precision Optimizer

Use FP16 optimizer states:

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    fused=True,  # Fused CUDA implementation
)
```

#### 4. ZeRO-2 for Optimizer States

```yaml
distributed:
  zero_stage: 2
  zero_cpu_offload: false
```

Shards optimizer states across data parallel ranks.

### Memory Monitoring

```python
import torch

# Peak memory
print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Peak reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

# Reset stats
torch.cuda.reset_peak_memory_stats()
```

## Throughput Optimization

### 1. Increase Batch Size

Larger batches improve GPU utilization:

```yaml
training:
  batch_size: 32  # Try 64, 128 if memory allows
  gradient_accumulation_steps: 4
```

Effective batch = batch_size × gradient_accumulation × world_size

### 2. Use NVFP4 Quantization

Maximum compute throughput on Blackwell:

```yaml
training:
  quantization: nvfp4
```

### 3. Optimize Expert Parallelism

Balance EP vs TP:

| GPUs | Best for Training | Best for Inference |
|------|-------------------|-------------------|
| 2 | EP=2, TP=1 | EP=1, TP=2 |
| 4 | EP=4, TP=1 | EP=2, TP=2 |
| 8 | EP=8, TP=1 or EP=4, TP=2 | EP=4, TP=2 |

### 4. Prefetch Expert Weights

For inference, prefetch next layer's experts:

```python
# Overlap computation with next layer's expert prefetch
async def forward_with_prefetch(x, layer_idx):
    # Start prefetch for next layer
    prefetch_future = prefetch_experts(layer_idx + 1)

    # Compute current layer
    output = layers[layer_idx](x)

    # Wait for prefetch
    await prefetch_future

    return output
```

### 5. Tune CUDA Graphs

For inference with static shapes:

```python
# Capture CUDA graph
static_input = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    # Warmup
    for _ in range(3):
        _ = model(static_input)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = model(static_input)

# Replay
graph.replay()
```

## Latency Optimization

### 1. Reduce Batch Size

For low-latency serving:

```yaml
inference:
  max_batch_size: 8  # Small batches for low latency
  enable_low_latency: true
```

### 2. Use Tensor Parallelism

TP reduces per-GPU compute:

```bash
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --tensor-parallel-size 8 \
    --enable-low-latency
```

### 3. Minimize Synchronization

- Use async weight updates
- Overlap compute and communication
- Avoid unnecessary barriers

### 4. Warm Up the Model

First inference is always slow:

```python
# Warmup with representative inputs
for _ in range(10):
    _ = model.generate(warmup_prompt, max_new_tokens=1)
torch.cuda.synchronize()
```

## RDEP Buffer Tuning

### Capacity Sizing

```
capacity = batch_size × sequence_length × topk × expert_parallel_size × safety_factor
```

Example for training:
- Batch: 32
- Seq: 4096
- TopK: 8
- EP: 4
- Safety: 2x

```bash
export NMOE_RDEP_CAPACITY=$((32 * 4096 * 8 * 4 * 2))  # 8,388,608
```

### Too Small Symptoms

- `RuntimeError: RDEP buffer overflow`
- Truncated outputs
- Silent errors

### Too Large Symptoms

- OOM errors
- Wasted memory
- Longer initialization

### Recommended Values

| Workload | Capacity |
|----------|----------|
| Dev/testing | 16,384 |
| Training (small batch) | 65,536 |
| Training (large batch) | 262,144 |
| Inference (low latency) | 32,768 |
| Inference (high throughput) | 524,288 |

## Multi-GPU Optimization

### NVLink Topology

Check topology:

```bash
nvidia-smi topo -m
```

Best performance when GPUs are NVLink-connected:

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV12    NV12    NV12
GPU1    NV12     X      NV12    NV12
GPU2    NV12    NV12     X      NV12
GPU3    NV12    NV12    NV12     X
```

### Process Placement

Bind processes to NUMA nodes:

```bash
# For 8-GPU training
numactl --cpunodebind=0 --membind=0 \
    torchrun --nproc_per_node=4 ... &
numactl --cpunodebind=1 --membind=1 \
    torchrun --nproc_per_node=4 ... &
```

### NCCL Tuning

```bash
# Enable tree algorithms for large allreduce
export NCCL_ALGO=Tree

# Set buffer size
export NCCL_BUFFSIZE=8388608

# Enable network optimization
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
```

## Performance Benchmarks

### Expected Performance (B200)

| Configuration | Forward (ms) | Throughput (M tok/s) |
|---------------|--------------|---------------------|
| 7B, BF16, EP=1 | 2.0 | 2.0 |
| 7B, FP8, EP=1 | 1.2 | 3.3 |
| 7B, NVFP4, EP=1 | 0.8 | 5.0 |
| 22B, NVFP4, EP=4 | 1.5 | 2.7 |
| 72B, NVFP4, EP=8 | 3.0 | 1.3 |

### Running Benchmarks

```bash
# Quick benchmark
pytest tests/test_performance.py -v -m "not slow"

# Full benchmark suite
pytest tests/test_performance.py -v

# Save results
NMOE_PERF_RESULTS=/tmp/perf.json \
    pytest tests/test_performance.py -v
```

## Troubleshooting Performance Issues

### Low GPU Utilization

```bash
nvidia-smi dmon -s u
```

If < 80% utilization:
1. Increase batch size
2. Check for CPU bottlenecks
3. Profile for synchronization points

### Memory Bandwidth Limited

Check with:
```bash
nvidia-smi dmon -s m
```

Solutions:
1. Use lower precision (FP8/NVFP4)
2. Reduce RDEP buffer size
3. Enable gradient checkpointing

### Communication Bottleneck

Check with:
```bash
nsys profile --trace=nvtx,cuda,nccl ...
```

Solutions:
1. Reduce EP if cross-GPU communication heavy
2. Use NVLink-connected GPUs
3. Overlap compute and communication

### Kernel Launch Overhead

For small batch sizes, kernel launch overhead can dominate:

```python
# Use CUDA graphs
torch.cuda.CUDAGraph()

# Or increase batch size
batch_size = max(batch_size, 32)
```

## Configuration Templates

### Maximum Throughput (Training)

```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 4
  quantization: nvfp4
  gradient_checkpointing: true

distributed:
  expert_parallel_size: 8
  tensor_parallel_size: 1
  zero_stage: 2

rdep:
  capacity: 524288
```

### Low Latency (Inference)

```yaml
inference:
  max_batch_size: 8
  quantization: nvfp4
  enable_low_latency: true

distributed:
  tensor_parallel_size: 8
  expert_parallel_size: 1

rdep:
  capacity: 32768
```

### Balanced (Training)

```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  quantization: fp8

distributed:
  expert_parallel_size: 4
  tensor_parallel_size: 2
  zero_stage: 2

rdep:
  capacity: 131072
```

## Related Documentation

- [Hardware Requirements](hardware.md)
- [Training Guide](training.md)
- [SGLang Integration](integration/sglang.md)
- [Testing Guide](development/testing.md)
