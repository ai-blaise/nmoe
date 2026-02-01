# Hardware Requirements

This document details the hardware requirements for running nmoe models across different workflows and configurations.

## Overview

nmoe is optimized for NVIDIA Blackwell (B200) GPUs with SM100 architecture. While some configurations may work on older hardware, full feature support requires Blackwell.

## GPU Requirements by Architecture

| Feature | Hopper (H100) | Blackwell (B200) |
|---------|---------------|------------------|
| BF16 Training | Yes | Yes |
| FP8 Training | Yes | Yes (optimized) |
| NVFP4 Training | No | Yes |
| RDEP IPC | Yes | Yes |
| NVSHMEM Multi-Node | Yes | Yes |
| MLA Attention | Yes | Yes |

## Memory Requirements

### Model Size vs GPU Memory

#### Single GPU (Inference Only)

| Model Size | BF16 | FP8 | NVFP4 |
|------------|------|-----|-------|
| 7B params | 14 GB | 7 GB | 4 GB |
| 22B params | 44 GB | 22 GB | 11 GB |
| 72B params | 144 GB | 72 GB | 36 GB |
| 236B params | 472 GB | 236 GB | 118 GB |

#### Training (Per GPU with Batch Size 1)

| Model Size | BF16 + Adam | FP8 + Adam | NVFP4 + Adam |
|------------|-------------|------------|--------------|
| 7B params | 56 GB | 35 GB | 25 GB |
| 22B params | 176 GB | 110 GB | 80 GB |
| 72B params | 576 GB | 360 GB | 260 GB |

*Note: Training requires approximately 4x model size for BF16 weights + optimizer states + activations.*

### Expert Parallelism Memory Scaling

Memory per GPU decreases linearly with Expert Parallelism (EP):

```
Memory per GPU = Total Expert Memory / EP Size
```

| Experts | EP=1 | EP=4 | EP=8 | EP=16 |
|---------|------|------|------|-------|
| 256 experts, 4B params | 4 GB | 1 GB | 0.5 GB | 0.25 GB |

## Configuration Recommendations

### Training Workflows

#### Small Models (7B parameters, 64 experts)

| Quantization | GPUs | Configuration | Batch Size |
|--------------|------|---------------|------------|
| BF16 | 1x B200 | TP=1, EP=1 | 4-8 |
| FP8 | 1x B200 | TP=1, EP=1 | 8-16 |
| NVFP4 | 1x B200 | TP=1, EP=1 | 16-32 |

#### Medium Models (22B parameters, 128 experts)

| Quantization | GPUs | Configuration | Batch Size |
|--------------|------|---------------|------------|
| BF16 | 4x B200 | TP=2, EP=2 | 8-16 |
| FP8 | 4x B200 | TP=2, EP=2 | 16-32 |
| NVFP4 | 4x B200 | TP=1, EP=4 | 32-64 |

#### Large Models (72B parameters, 256 experts)

| Quantization | GPUs | Configuration | Batch Size |
|--------------|------|---------------|------------|
| BF16 | 8x B200 | TP=4, EP=2 | 8-16 |
| FP8 | 8x B200 | TP=2, EP=4 | 16-32 |
| NVFP4 | 8x B200 | TP=2, EP=4 | 32-64 |

#### Very Large Models (236B parameters, 256 experts)

| Quantization | GPUs | Configuration | Batch Size |
|--------------|------|---------------|------------|
| BF16 | 16x B200 (2 nodes) | TP=4, EP=4 | 4-8 |
| FP8 | 16x B200 (2 nodes) | TP=2, EP=8 | 8-16 |
| NVFP4 | 16x B200 (2 nodes) | TP=2, EP=8 | 16-32 |

### Inference/Serving Workflows

#### Low-Latency Serving (Batch Size 1-8)

| Model Size | GPUs | Configuration | Tokens/sec |
|------------|------|---------------|------------|
| 7B | 1x B200 | TP=1, NVFP4 | ~500 |
| 22B | 2x B200 | TP=2, NVFP4 | ~300 |
| 72B | 4x B200 | TP=4, NVFP4 | ~150 |

#### High-Throughput Serving (Batch Size 64+)

| Model Size | GPUs | Configuration | Tokens/sec |
|------------|------|---------------|------------|
| 7B | 1x B200 | TP=1, NVFP4 | ~4000 |
| 22B | 4x B200 | TP=4, NVFP4 | ~2000 |
| 72B | 8x B200 | TP=8, NVFP4 | ~1000 |

### RL Training Workflows

RL training requires additional memory for:
- Reference model (frozen copy)
- Value head / critic (if PPO)
- Replay buffers

| Model Size | Algorithm | GPUs | Configuration |
|------------|-----------|------|---------------|
| 7B | GRPO | 4x B200 | TP=2, EP=2 |
| 7B | PPO | 8x B200 | TP=2, EP=4 |
| 22B | GRPO | 8x B200 | TP=4, EP=2 |
| 22B | PPO | 16x B200 | TP=4, EP=4 |

## Network Requirements

### Single Node

- **NVLink**: Required for TP > 1 with good performance
- **PCIe 5.0**: Acceptable for EP-only configurations

### Multi-Node

| Interconnect | Performance | Use Case |
|--------------|-------------|----------|
| InfiniBand HDR | 200 Gb/s | Required for multi-node |
| InfiniBand NDR | 400 Gb/s | Recommended for multi-node |
| RDMA/RoCE | 100 Gb/s | Supported with NVSHMEM |

NVSHMEM Configuration:
- GPUDirect RDMA required
- IBGDA transport recommended
- See [NVSHMEM documentation](nvshmem.md) for setup

## RDEP Buffer Sizing

The RDEP buffer capacity determines maximum tokens that can be dispatched:

```
Required Capacity = Batch Size × Sequence Length × TopK × EP Size
```

| Configuration | Recommended Capacity |
|---------------|---------------------|
| Training (small batch) | 65,536 |
| Training (large batch) | 262,144 |
| Serving (low latency) | 32,768 |
| Serving (high throughput) | 524,288 |

Memory overhead per GPU:
```
RDEP Memory = Capacity × (Hidden Dim × 2 + Metadata) × 2 buffers
            ≈ Capacity × Hidden Dim × 6 bytes
```

| Capacity | Hidden=4096 | Hidden=8192 |
|----------|-------------|-------------|
| 65,536 | 1.5 GB | 3.0 GB |
| 262,144 | 6.0 GB | 12.0 GB |
| 524,288 | 12.0 GB | 24.0 GB |

## Storage Requirements

### Checkpoint Storage

| Model Size | Checkpoint Size (BF16) | Checkpoint Size (FP8) |
|------------|------------------------|----------------------|
| 7B | 14 GB | 9 GB |
| 22B | 44 GB | 28 GB |
| 72B | 144 GB | 90 GB |

Recommended storage:
- **NVMe SSD**: For checkpoint I/O during training
- **Parallel Filesystem**: For multi-node checkpoint coordination

### Dataset Storage

- **Tokenized data**: ~2 bytes per token
- **1T tokens**: ~2 TB storage
- **Recommended**: High-bandwidth parallel filesystem (Lustre, GPFS)

## Minimum System Requirements

### Development / Testing

```
GPU: 1x NVIDIA GPU with ≥24GB VRAM
RAM: 64 GB
Storage: 500 GB SSD
CUDA: 12.1+
```

### Production Training

```
GPU: 8x NVIDIA B200 (192 GB each)
RAM: 512 GB
Storage: 10 TB NVMe SSD
Network: InfiniBand NDR 400 Gb/s (for multi-node)
CUDA: 12.8+
```

### Production Serving

```
GPU: 1-8x NVIDIA B200 (based on model size)
RAM: 128 GB
Storage: 500 GB SSD
CUDA: 12.8+
```

## Docker Resource Configuration

### Training Container

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 8
          capabilities: [gpu]
    limits:
      memory: 512G
      cpus: '128'
```

### Serving Container

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    limits:
      memory: 64G
      cpus: '16'
```

## Performance Benchmarks

### Training Throughput (tokens/second/GPU)

| Model | BF16 | FP8 | NVFP4 |
|-------|------|-----|-------|
| 7B, EP=1 | 12,000 | 18,000 | 24,000 |
| 22B, EP=4 | 8,000 | 12,000 | 16,000 |
| 72B, EP=8 | 4,000 | 6,000 | 8,000 |

### Serving Latency (ms, batch=1)

| Model | BF16 | FP8 | NVFP4 |
|-------|------|-----|-------|
| 7B | 15 | 10 | 8 |
| 22B | 25 | 18 | 14 |
| 72B | 45 | 32 | 25 |

*Benchmarks on B200 with default configurations. Actual performance varies.*

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size**
2. **Enable gradient checkpointing**
3. **Increase EP (expert parallelism)**
4. **Use quantization (FP8 → NVFP4)**
5. **Reduce RDEP buffer capacity**

### Slow Training

1. **Check NVLink topology**: `nvidia-smi topo -m`
2. **Verify RDEP mode**: Should be IPC for multi-GPU
3. **Check ZeRO-2 settings**: Ensure proper chunk size
4. **Profile with Nsight**: Identify bottlenecks

### Multi-Node Issues

1. **Verify NVSHMEM setup**: Check bootstrap UID distribution
2. **Test InfiniBand**: `ibv_devinfo`, `ib_write_bw`
3. **Check firewall**: NVSHMEM ports must be open
4. **Verify GPUDirect**: `nvidia-smi nvlink -s`

## Related Documentation

- [NVSHMEM Build Requirements](nvshmem.md)
- [SGLang Integration](integration/sglang.md)
- [SkyRL Integration](integration/skyrl.md)
- [Checkpoint Resharding](tools/reshard.md)
