# Training Guide

This guide covers training MoE models with nmoe.

## Overview

nmoe supports training with:

- **Quantization Profiles**: BF16, FP8, NVFP4
- **Expert Parallelism**: Distribute experts across GPUs
- **Tensor Parallelism**: Shard large weights within experts
- **ZeRO-2**: Optimizer state sharding

## Basic Training

### Single GPU

```bash
python -m nmoe.train \
    --config configs/7b_bf16.yaml \
    --output-dir /checkpoints/run_001
```

### Multi-GPU (Expert Parallel)

```bash
torchrun --nproc_per_node=8 \
    -m nmoe.train \
    --config configs/7b_nvfp4.yaml \
    --expert-parallel-size 8
```

## Configuration

### Model Config

```yaml
model:
  hidden_size: 4096
  num_hidden_layers: 32
  num_attention_heads: 32
  num_key_value_heads: 8
  num_experts: 256
  num_experts_per_tok: 8
  moe_intermediate_size: 1408
  vocab_size: 128256
  max_position_embeddings: 8192
```

### Training Config

```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  warmup_steps: 2000
  max_steps: 100000
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95

  # Quantization
  quantization: nvfp4  # bf16, fp8, or nvfp4

  # Checkpointing
  save_steps: 1000
  checkpoint_dir: /checkpoints
```

### Distributed Config

```yaml
distributed:
  expert_parallel_size: 8
  tensor_parallel_size: 1
  zero_stage: 2
```

## Checkpointing

### Checkpoint Format

nmoe uses a versioned checkpoint format:

```
/checkpoints/run_001/
├── config.json           # Model config
├── tracker               # Latest step number
├── iter_0001000/
│   ├── rd.pt            # Dense weights (rank-dependent)
│   └── re.pt            # Expert weights (EP-sharded)
└── iter_0002000/
    ├── rd.pt
    └── re.pt
```

### Save Checkpoint

```python
from nmoe.checkpoint import save_checkpoint

save_checkpoint(
    base_path="/checkpoints/run_001",
    step=1000,
    model=model,
    optimizer=optimizer,
    ep_size=8,
    ep_rank=rank,
)
```

### Load Checkpoint

```python
from nmoe.checkpoint import load_checkpoint

step = load_checkpoint(
    base_path="/checkpoints/run_001",
    model=model,
    optimizer=optimizer,
    ep_size=8,
    ep_rank=rank,
)
```

### Reshard Checkpoint

To change EP size:

```bash
python -m nmoe.tools.reshard_checkpoint \
    --input /checkpoints/run_001 \
    --output /checkpoints/run_001_ep4 \
    --step latest \
    --target-ep 4
```

## Quantization

### BF16 (Baseline)

```yaml
training:
  quantization: bf16
```

Full precision, largest memory footprint.

### FP8 (2x Compression)

```yaml
training:
  quantization: fp8
```

Requires Hopper or Blackwell. 2x memory savings.

### NVFP4 (4x Compression)

```yaml
training:
  quantization: nvfp4
```

Blackwell only. 4x memory savings, highest throughput.

## Expert Cache Refresh

For FP8/NVFP4 training, refresh the quantization cache after each optimizer step:

```python
# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Refresh expert caches (important!)
    model.refresh_expert_caches()
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir /checkpoints/run_001/logs
```

### Metrics

Key metrics to monitor:

- **Loss**: Training loss
- **Throughput**: Tokens/second
- **Router balance**: Expert utilization
- **Memory**: Peak GPU memory

## Multi-Node Training

### SLURM

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun python -m nmoe.train \
    --config configs/72b_nvfp4.yaml \
    --expert-parallel-size 16
```

### Manual Launch

```bash
# Node 0
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=node0 --master_port=29500 \
    -m nmoe.train --config config.yaml

# Node 1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=node0 --master_port=29500 \
    -m nmoe.train --config config.yaml
```

## Best Practices

1. **Start with BF16** for debugging, then switch to NVFP4 for production
2. **Use gradient checkpointing** for large models
3. **Monitor router balance** to ensure experts are utilized evenly
4. **Save checkpoints frequently** during long training runs
5. **Use ZeRO-2** for optimizer state sharding
