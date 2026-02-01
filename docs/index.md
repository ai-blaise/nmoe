# nmoe

B200-targeted Mixture-of-Experts training library with RDEP (Redistribution Expert Parallelism).

## Overview

nmoe is a high-performance library for training and serving Mixture-of-Experts (MoE) models on NVIDIA Blackwell (B200) GPUs. It provides:

- **RDEP Dispatcher**: Efficient token routing with IPC and NVSHMEM backends
- **Blockscaled Quantization**: FP8 and NVFP4 support for maximum throughput
- **Expert Parallelism**: Scale across multiple GPUs and nodes
- **Framework Integration**: Works with SGLang for serving and SkyRL for RL training

## Key Features

### High Performance

- Optimized CUDA kernels for SM100 (Blackwell) architecture
- Blockscaled quantization (FP8, NVFP4) for 2-4x memory savings
- Fused dispatch/gather/scatter operations
- Zero-copy IPC for single-node multi-GPU

### Scalability

- Expert Parallelism (EP) for distributing experts across GPUs
- Tensor Parallelism (TP) for large expert weights
- NVSHMEM support for multi-node training
- ZeRO-2 integration for optimizer state sharding

### Integration

- **SGLang**: High-throughput serving with continuous batching
- **SkyRL**: RL training with GRPO and PPO algorithms
- **HuggingFace**: Export to HF format for deployment

## Quick Start

```bash
# Install nmoe
pip install -e .

# Train a model
python -m nmoe.train --config configs/7b_nvfp4.yaml

# Serve with SGLang
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe
```

## Hardware Requirements

| Model Size | Training | Inference |
|------------|----------|-----------|
| 7B | 1x B200 | 1x B200 |
| 22B | 4x B200 | 2x B200 |
| 72B | 8x B200 | 4x B200 |
| 236B | 16x B200 | 8x B200 |

See [Hardware Requirements](hardware.md) for detailed specifications.

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Training Guide](training.md)
- [SGLang Integration](integration/sglang.md)
- [SkyRL Integration](integration/skyrl.md)
- [API Reference](api/unified.md)

## License

Apache 2.0
