# SGLang Integration Guide

This guide explains how to serve nmoe models using SGLang for high-performance inference.

## Overview

nmoe integrates with SGLang through:
1. **NMoE Runner Backend** - Efficient MoE dispatch/combine using RDEP
2. **Model Loader** - Direct loading of nmoe checkpoints
3. **Process Group Bridge** - Seamless integration with SGLang's distributed runtime

## Prerequisites

### Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| Single GPU | 1x B200 (192GB) | 1x B200 |
| Multi-GPU | 2x B200 | 8x B200 |
| Multi-Node | 2 nodes, 8 GPUs each | 4+ nodes |

### Software Requirements

- Python 3.12+
- PyTorch 2.11+ with CUDA 12.8
- SGLang (from nether-soup repo)
- nmoe (from nether-soup repo)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/nether-soup.git
cd nether-soup

# Install SGLang
cd sglang
pip install -e "python[all]"

# Install nmoe
cd ../nmoe
pip install -e .
```

## Quick Start

### Basic Serving

```bash
# Serve a trained nmoe model
python -m sglang.launch_server \
    --model-path /path/to/nmoe/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --port 8000
```

### With Quantization

```bash
# FP8 quantization (recommended for B200)
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --quantization modelopt_fp8 \
    --port 8000

# NVFP4 quantization (maximum throughput)
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --quantization modelopt_fp4 \
    --port 8000
```

## Configuration

### Server Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | Path to nmoe checkpoint | Required |
| `--load-format` | Checkpoint format | `auto` (use `nmoe` for nmoe checkpoints) |
| `--moe-runner-backend` | MoE dispatch backend | `triton` (use `nmoe` for RDEP) |
| `--quantization` | Weight quantization | `None` |
| `--tensor-parallel-size` | TP degree | 1 |
| `--expert-parallel-size` | EP degree | Auto-detected |
| `--port` | Server port | 30000 |
| `--host` | Server host | `0.0.0.0` |

### Expert Parallelism

nmoe automatically detects the EP configuration from the checkpoint. To override:

```bash
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --expert-parallel-size 4 \
    --port 8000
```

### Multi-GPU Serving

```bash
# Single-node multi-GPU (8 GPUs)
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --tensor-parallel-size 8 \
    --port 8000

# With Expert Parallelism
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --tensor-parallel-size 2 \
    --expert-parallel-size 4 \
    --port 8000
```

## Checkpoint Export

Before serving, you may need to export your nmoe checkpoint to HuggingFace format:

```bash
# Export to HF format
python -m nmoe.tools.export_to_hf \
    --input /path/to/nmoe/checkpoint \
    --output /path/to/hf/export \
    --step latest

# Or serve directly (nmoe loader handles conversion)
python -m sglang.launch_server \
    --model-path /path/to/nmoe/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe
```

## API Usage

Once the server is running, use the OpenAI-compatible API:

### Chat Completions

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="nmoe",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Text Completions

```python
response = client.completions.create(
    model="nmoe",
    prompt="Once upon a time",
    max_tokens=100
)

print(response.choices[0].text)
```

### Batched Requests

```python
# SGLang efficiently batches concurrent requests
import asyncio
import aiohttp

async def send_request(session, prompt):
    async with session.post(
        "http://localhost:8000/v1/completions",
        json={"model": "nmoe", "prompt": prompt, "max_tokens": 50}
    ) as response:
        return await response.json()

async def main():
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
    return results
```

## RDEP Configuration

The RDEP dispatcher can be configured through environment variables:

```bash
# Set RDEP buffer capacity (default: 65536)
export NMOE_RDEP_CAPACITY=131072

# Enable debug logging
export NMOE_RDEP_DEBUG=1

# Run server
python -m sglang.launch_server ...
```

### Buffer Sizing

The RDEP buffer capacity should be set based on your workload:

```
capacity = max_batch_size * max_sequence_length * topk * expert_parallel_size
```

For example:
- Batch size: 64
- Sequence length: 4096
- TopK: 8
- EP: 4

```bash
export NMOE_RDEP_CAPACITY=$((64 * 4096 * 8 * 4))  # 8,388,608
```

## Multi-Node Deployment

For multi-node serving with NVSHMEM:

### Node 0 (Master)

```bash
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --expert-parallel-size 16 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr node0 \
    --master-port 29500 \
    --port 8000
```

### Node 1

```bash
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --expert-parallel-size 16 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr node0 \
    --master-port 29500
```

See [NVSHMEM documentation](../nvshmem.md) for multi-node requirements.

## Performance Tuning

### Low-Latency Mode

For real-time applications with small batch sizes:

```bash
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --enable-low-latency \
    --port 8000
```

### Throughput Mode

For batch processing with high throughput:

```bash
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --max-num-batched-tokens 32768 \
    --max-batch-size 256 \
    --port 8000
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

Reduce batch size or enable quantization:
```bash
--quantization modelopt_fp8
--max-batch-size 32
```

**2. RDEP Buffer Overflow**

Increase buffer capacity:
```bash
export NMOE_RDEP_CAPACITY=262144
```

**3. NVSHMEM Initialization Failed**

Check multi-node setup:
- Verify InfiniBand connectivity
- Check NVSHMEM environment variables
- See [NVSHMEM troubleshooting](../nvshmem.md#troubleshooting)

**4. Checkpoint Loading Failed**

Ensure checkpoint format matches:
```bash
# For nmoe native checkpoints
--load-format nmoe

# For HF-exported checkpoints
--load-format auto
```

### Debug Mode

Enable verbose logging:

```bash
SGLANG_LOG_LEVEL=debug \
NMOE_RDEP_DEBUG=1 \
python -m sglang.launch_server ...
```

## Monitoring

### Prometheus Metrics

SGLang exposes Prometheus metrics:

```bash
# Get metrics
curl http://localhost:8000/metrics
```

Key metrics:
- `sglang_running_req`: Current running requests
- `sglang_token_throughput`: Tokens per second
- `sglang_cache_hit_rate`: KV cache hit rate

### Health Check

```bash
curl http://localhost:8000/health
```

## Examples

### Complete Serving Script

```bash
#!/bin/bash
# serve_nmoe.sh

MODEL_PATH=${1:-"/models/nmoe-7b"}
PORT=${2:-8000}
TP_SIZE=${3:-8}

# Set RDEP capacity based on expected workload
export NMOE_RDEP_CAPACITY=131072

# Launch server
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --load-format nmoe \
    --moe-runner-backend nmoe \
    --tensor-parallel-size "$TP_SIZE" \
    --quantization modelopt_fp8 \
    --max-num-batched-tokens 16384 \
    --max-batch-size 128 \
    --port "$PORT" \
    --host 0.0.0.0
```

### Docker Deployment

```dockerfile
FROM xjdr/nmoe_train:latest

WORKDIR /app
COPY serve_nmoe.sh /app/

# Download model (or mount volume)
# COPY /path/to/model /models/nmoe-7b

EXPOSE 8000

CMD ["./serve_nmoe.sh", "/models/nmoe-7b", "8000", "8"]
```

## Related Documentation

- [nmoe Training Guide](../training.md)
- [NVSHMEM Build Requirements](../nvshmem.md)
- [Checkpoint Resharding](../tools/reshard.md)
- [SGLang Documentation](https://sgl-project.github.io/)
