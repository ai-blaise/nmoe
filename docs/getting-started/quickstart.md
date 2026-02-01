# Quick Start

Get up and running with nmoe in minutes.

## Basic Usage

### Creating an RDEP Dispatcher

```python
from nmoe.rdep import Rdep
import torch

# Create RDEP for 8 local experts, top-2 routing
rdep = Rdep(
    dim=2048,           # Hidden dimension
    n_local=8,          # Experts per GPU
    topk=2,             # Activated experts per token
    profile="nvfp4",    # Quantization: bf16, fp8, or nvfp4
    capacity=65536,     # Max tokens in buffer
)

print(f"Mode: {rdep._mode}")  # single, ipc, or hybrid
```

### Using the MoE Layer

```python
from nmoe.moe import _MoEBlockscaledFused
from nmoe.blockscaled.grouped import quantize_weights

# Input
x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
eids = torch.randint(0, 8, (1024, 2), device="cuda", dtype=torch.int32)
gates = torch.softmax(torch.randn(1024, 2, device="cuda"), dim=-1).bfloat16()

# Weights
W1 = torch.randn(8, 2048, 1408, device="cuda", dtype=torch.bfloat16)
W3 = torch.randn(8, 2048, 1408, device="cuda", dtype=torch.bfloat16)
W2 = torch.randn(8, 1408, 2048, device="cuda", dtype=torch.bfloat16)

# Quantize weights
W_cache = quantize_weights(W1, W3, W2, profile="nvfp4")

# Forward pass
output = _MoEBlockscaledFused.apply(
    rdep, x, eids, gates, W1, W3, W2, W_cache
)
```

### Using the Unified Interface

```python
from nmoe.unified import NMoEModelConfig, create_nmoe_model

# Create config
config = NMoEModelConfig(
    hidden_size=2048,
    num_hidden_layers=8,
    num_attention_heads=16,
    num_experts=64,
    num_experts_per_tok=2,
    moe_intermediate_size=1408,
    vocab_size=32000,
)

# Create model
model = create_nmoe_model(config, device="cuda")
```

## Multi-GPU Usage

### Expert Parallelism

```python
import torch.distributed as dist
from nmoe.distributed import init_nmoe_process_groups

# Initialize distributed
dist.init_process_group(backend="nccl")

# Create EP groups (4 GPUs, EP=4)
init_nmoe_process_groups(ep_size=4, tp_size=1)

# RDEP automatically uses the distributed setup
rdep = Rdep(
    dim=2048,
    n_local=16,  # 64 total / 4 EP = 16 local
    topk=2,
    profile="nvfp4",
)
```

### Combined EP + TP

```python
# 8 GPUs: EP=4, TP=2
init_nmoe_process_groups(ep_size=4, tp_size=2)

# Now each EP rank has TP=2 for dense layers
```

## Serving with SGLang

```bash
python -m sglang.launch_server \
    --model-path /path/to/nmoe/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --quantization modelopt_fp8 \
    --port 8000
```

Then query with:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="nmoe",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## RL Training with SkyRL

```bash
python -m skyrl_train.train \
    --config configs/nmoe_grpo.yaml \
    --model.path /path/to/nmoe/checkpoint
```

## Next Steps

- [Training Guide](../training.md) - Full training workflow
- [Hardware Requirements](../hardware.md) - GPU and memory requirements
- [SGLang Integration](../integration/sglang.md) - Detailed serving guide
- [API Reference](../api/unified.md) - Complete API documentation
