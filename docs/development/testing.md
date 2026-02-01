# Testing Guide

This guide covers the nmoe test suite and how to run tests.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_cuda_errors.py      # CUDA error handling
├── test_distributed.py      # Multi-GPU tests
├── test_ep_tp.py           # EP+TP configuration
├── test_moe.py             # MoE layer tests
├── test_performance.py      # Performance regression
├── test_rdep.py            # RDEP dispatcher
└── ...
```

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### By Marker

```bash
# GPU tests only
pytest tests/ -v -m gpu

# Multi-GPU tests
pytest tests/ -v -m multi_gpu

# Skip slow tests
pytest tests/ -v -m "not slow"

# Performance tests
pytest tests/ -v -m benchmark
```

### Distributed Tests

```bash
# 2 GPUs
torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py -v

# 8 GPUs
torchrun --nproc_per_node=8 -m pytest tests/test_distributed.py -v
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `gpu` | Requires CUDA GPU |
| `multi_gpu` | Requires 2+ GPUs |
| `distributed` | Requires distributed setup |
| `slow` | Long-running tests |
| `nvshmem` | Requires NVSHMEM |
| `benchmark` | Performance benchmarks |

## Fixtures

### `cuda_device`

Provides CUDA device, skips if unavailable:

```python
def test_something(cuda_device):
    x = torch.randn(10, device=cuda_device)
```

### `small_model_config`

Standard small config for testing:

```python
def test_model(small_model_config):
    model = create_model(**small_model_config)
```

### `bf16_tensor_factory`

Creates BF16 tensors:

```python
def test_moe(bf16_tensor_factory):
    x = bf16_tensor_factory((100, 256))
```

## Performance Testing

Performance tests compare against baselines:

```python
from tests.test_performance import PerfBaseline, PerfBenchmark

baseline = PerfBaseline(
    name="my_op",
    max_latency_p50_ms=1.0,
    min_throughput=1000000,
)

result = benchmark.run_benchmark("my_op", my_function)
passed, failures = baseline.check(result)
```

## CI Integration

Tests run automatically on:

- Push to main
- Pull requests
- Scheduled (nightly for GPU tests)

See `.github/workflows/ci.yml` for configuration.
