# SkyRL Integration Guide

This guide explains how to use nmoe models with SkyRL for reinforcement learning (RL) training workflows like GRPO and PPO.

## Overview

nmoe integrates with SkyRL through:
1. **NMoEModelWrapper** - Wraps nmoe models for RL training
2. **Weight Extractor** - EP-aware weight extraction for weight sync
3. **Checkpoint Integration** - Seamless save/load during RL training

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SkyRL Training Loop                         │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   Actor     │    │  Reference   │    │   SGLang Engine      │   │
│  │  (nmoe)     │◄──►│   Model      │    │   (inference)        │   │
│  └─────────────┘    └──────────────┘    └──────────────────────┘   │
│         │                                         ▲                 │
│         │                                         │                 │
│         ▼                                         │                 │
│  ┌─────────────┐                          ┌──────┴───────┐         │
│  │  Trainer    │                          │ Weight Sync  │         │
│  │ (GRPO/PPO)  │─────────────────────────►│   Bridge     │         │
│  └─────────────┘                          └──────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements

| Configuration | GPUs | Memory |
|---------------|------|--------|
| Small (7B) | 4x B200 | 768GB total |
| Medium (22B) | 8x B200 | 1.5TB total |
| Large (72B) | 16x B200 | 3TB total |

### Software Requirements

- Python 3.12+
- PyTorch 2.11+ with CUDA 12.8
- SkyRL (from nether-soup repo)
- nmoe (from nether-soup repo)
- SGLang (for inference engine)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/nether-soup.git
cd nether-soup

# Install SkyRL
cd SkyRL/skyrl-train
pip install -e .

# Install nmoe
cd ../../nmoe
pip install -e .

# Install SGLang (for inference)
cd ../sglang
pip install -e "python[all]"
```

## Quick Start

### Basic GRPO Training

```bash
# Train with GRPO using nmoe actor
python -m skyrl_train.train \
    --config configs/nmoe_grpo.yaml \
    --model.path /path/to/nmoe/checkpoint \
    --model.type nmoe
```

### Example Configuration

```yaml
# configs/nmoe_grpo.yaml
model:
  type: nmoe
  path: /path/to/nmoe/checkpoint
  load_format: nmoe
  dtype: bfloat16

training:
  algorithm: grpo
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-6
  num_epochs: 3

  # Expert-specific settings
  expert_cache_refresh_interval: 1  # Refresh after each step

distributed:
  expert_parallel_size: 4
  tensor_parallel_size: 2
  data_parallel_size: 1

inference:
  engine: sglang
  moe_runner_backend: nmoe
  quantization: modelopt_fp8

checkpointing:
  save_steps: 500
  output_dir: /checkpoints/grpo_run
```

## Model Wrapper

### Using NMoEModelWrapper

```python
from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper

# Create wrapper
wrapper = NMoEModelWrapper(
    model_path="/path/to/checkpoint",
    load_format="nmoe",
    dtype=torch.bfloat16,
    expert_parallel_size=4,
)

# Forward pass for RL (returns log probs)
log_probs = wrapper.forward(
    sequences=input_ids,
    num_actions=num_action_tokens,
    attention_mask=attention_mask,
)

# Generate (delegates to SGLang engine)
outputs = wrapper.generate(
    prompts=prompts,
    max_new_tokens=256,
    temperature=0.7,
)
```

### Expert Cache Refresh

For FP8/NVFP4 training, refresh the quantization cache after optimizer steps:

```python
# In training loop
optimizer.step()
wrapper.refresh_expert_caches()  # Important for quantized training!
```

## Weight Synchronization

### Actor-to-Inference Weight Sync

```python
from skyrl_train.distributed.nmoe_weight_extractor import MoEWeightExtractor

# Create extractor
extractor = MoEWeightExtractor(
    model=actor_model,
    expert_parallel_size=4,
)

# Extract weights (handles EP gathering)
actor_weights = extractor.get_weights()

# Sync to inference engine
inference_engine.update_weights(actor_weights)
```

### Partial Weight Updates (LoRA)

```python
# For LoRA fine-tuning, only sync adapter weights
adapter_weights = extractor.get_lora_weights()
inference_engine.update_lora_weights(adapter_weights)
```

## Training Configurations

### GRPO Training

```yaml
training:
  algorithm: grpo

  # GRPO-specific settings
  grpo:
    num_generations: 8
    kl_coef: 0.1
    clip_range: 0.2
    value_loss_coef: 0.5

  # Batch settings
  batch_size: 32
  gradient_accumulation_steps: 4
  max_seq_len: 4096

  # Optimization
  learning_rate: 1.0e-6
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0
```

### PPO Training

```yaml
training:
  algorithm: ppo

  # PPO-specific settings
  ppo:
    num_rollouts: 128
    ppo_epochs: 4
    clip_range: 0.2
    value_loss_coef: 0.5
    entropy_coef: 0.01
    gae_lambda: 0.95

  # Batch settings
  batch_size: 64
  gradient_accumulation_steps: 2
```

### Distributed Training

```yaml
distributed:
  # Expert parallelism (for MoE experts)
  expert_parallel_size: 4

  # Tensor parallelism (for dense layers)
  tensor_parallel_size: 2

  # Data parallelism (gradient averaging)
  data_parallel_size: 2

  # Total GPUs = EP * TP * DP = 4 * 2 * 2 = 16

  # Zero optimization
  zero_stage: 2
  zero_cpu_offload: false
```

## Expert Parallel Training

### Process Group Setup

nmoe automatically creates the correct process groups for EP:

```python
from nmoe.distributed import init_groups

# Initialize process groups
init_groups(
    expert_parallel_size=4,
    tensor_parallel_size=2,
)

# Create model with EP group
from nmoe.rdep import Rdep

rdep = Rdep(
    dim=4096,
    n_local=64,  # 256 total / 4 EP
    topk=8,
    profile='nvfp4',
    ep_group=get_expert_parallel_group(),  # Uses custom EP group
)
```

### Gradient Handling

- **Dense parameters**: Synchronized via ZeRO-2 on data parallel group
- **Expert parameters**: Local only (already EP-sharded)
- **Router parameters**: Synchronized via all-reduce on DP group

## Checkpointing

### Save Checkpoint

```python
from nmoe.checkpoint import save_checkpoint

save_checkpoint(
    base_path="/checkpoints/run_001",
    step=1000,
    model=actor_model,
    optimizer=optimizer,
    # EP-aware saving
    ep_size=4,
    ep_rank=rank,
)
```

### Load Checkpoint

```python
from nmoe.checkpoint import load_checkpoint

step = load_checkpoint(
    base_path="/checkpoints/run_001",
    model=actor_model,
    optimizer=optimizer,
    # EP-aware loading
    ep_size=4,
    ep_rank=rank,
)
```

### Resume with Different EP

Use the resharding tool to convert checkpoints between EP configurations:

```bash
python -m nmoe.tools.reshard_checkpoint \
    --input /checkpoints/run_001 \
    --output /checkpoints/run_001_ep8 \
    --step 1000 \
    --target-ep 8
```

## Reward Models

### Using External Reward Model

```yaml
reward:
  type: external
  model_path: /path/to/reward_model

  # Or use API-based reward
  # type: api
  # endpoint: http://localhost:9000/reward
```

### Using Rule-Based Rewards

```yaml
reward:
  type: rule_based
  rules:
    - type: length_penalty
      target_length: 500
      penalty: -0.01
    - type: format_check
      pattern: "^\\d+\\."
      reward: 0.1
```

## Memory Optimization

### Activation Checkpointing

```yaml
memory:
  activation_checkpointing: true
  checkpoint_layers: [0, 4, 8, 12, 16, 20, 24, 28]  # Checkpoint every 4th layer
```

### Gradient Accumulation

```yaml
training:
  batch_size: 8  # Per-GPU batch
  gradient_accumulation_steps: 16  # Effective batch = 8 * 16 = 128
```

### Mixed Precision

```yaml
model:
  dtype: bfloat16

training:
  # Expert compute uses blockscaled (FP8/NVFP4)
  expert_precision: nvfp4
```

## Multi-Node Training

### Launch Script

```bash
#!/bin/bash
# launch_skyrl.sh

# Node 0 (Master)
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=node0 \
    --master_port=29500 \
    -m skyrl_train.train \
    --config configs/nmoe_grpo.yaml

# Node 1 (same command with --node_rank=1)
```

### SLURM Job

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00

srun python -m skyrl_train.train \
    --config configs/nmoe_grpo.yaml
```

## Troubleshooting

### Common Issues

**1. Weight Sync Deadlock**

Ensure all ranks participate in weight extraction:
```python
# Wrong - only rank 0 extracts
if rank == 0:
    weights = extractor.get_weights()

# Right - all ranks participate
weights = extractor.get_weights()
if rank == 0:
    inference_engine.update_weights(weights)
```

**2. Expert Cache Not Refreshing**

Call `refresh_expert_caches()` after optimizer step:
```python
optimizer.step()
optimizer.zero_grad()
wrapper.refresh_expert_caches()  # Don't forget!
```

**3. OOM During Generation**

Reduce generation batch size:
```yaml
inference:
  max_batch_size: 32  # Reduce if OOM
```

**4. Checkpoint EP Mismatch**

Reshard checkpoint to match current EP:
```bash
python -m nmoe.tools.reshard_checkpoint \
    --input /checkpoints/ep4 \
    --output /checkpoints/ep8 \
    --target-ep 8
```

### Debug Mode

```bash
# Enable verbose logging
SKYRL_DEBUG=1 \
NMOE_RDEP_DEBUG=1 \
python -m skyrl_train.train --config config.yaml
```

## Examples

### Complete Training Script

```python
#!/usr/bin/env python3
"""Example GRPO training with nmoe."""

import torch
from skyrl_train.trainers import GRPOTrainer
from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
from nmoe.distributed import init_groups

def main():
    # Initialize distributed
    torch.distributed.init_process_group(backend="nccl")
    init_groups(expert_parallel_size=4, tensor_parallel_size=2)

    # Create model wrapper
    actor = NMoEModelWrapper(
        model_path="/models/nmoe-7b",
        load_format="nmoe",
        dtype=torch.bfloat16,
    )

    # Create trainer
    trainer = GRPOTrainer(
        actor=actor,
        learning_rate=1e-6,
        batch_size=32,
        num_generations=8,
    )

    # Training loop
    for epoch in range(3):
        for batch in dataloader:
            # Generate responses
            responses = actor.generate(batch["prompts"])

            # Get rewards
            rewards = reward_model(responses)

            # GRPO update
            loss = trainer.step(batch, responses, rewards)

            # Refresh expert caches (important for FP8/NVFP4!)
            actor.refresh_expert_caches()

            # Sync weights to inference engine periodically
            if trainer.global_step % 100 == 0:
                trainer.sync_inference_weights()

        # Save checkpoint
        trainer.save_checkpoint(f"/checkpoints/epoch_{epoch}")

if __name__ == "__main__":
    main()
```

## Related Documentation

- [nmoe Training Guide](../training.md)
- [SGLang Integration](sglang.md)
- [Checkpoint Resharding](../tools/reshard.md)
- [Distributed Training](../distributed.md)
- [SkyRL Documentation](https://github.com/your-org/SkyRL)
