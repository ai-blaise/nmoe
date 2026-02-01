# Installation

This guide covers installing nmoe for training and serving MoE models.

## Prerequisites

### Hardware

- NVIDIA GPU with Compute Capability 8.0+ (recommended: B200/H100)
- For multi-GPU: NVLink or InfiniBand
- For multi-node: InfiniBand with GPUDirect RDMA

### Software

- Python 3.12+
- CUDA 12.8+
- PyTorch 2.11+

## Installation Methods

### Docker (Recommended)

The easiest way to get started is using our Docker images:

```bash
# Training image
docker pull xjdr/nmoe_train:latest

# Run container
docker run --gpus all -it xjdr/nmoe_train:latest
```

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/nether-soup.git
cd nether-soup/nmoe

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy pytest safetensors

# Install nmoe in editable mode
pip install -e .
```

### Building CUDA Extensions

The CUDA extensions are built automatically on first import. To build manually:

```bash
cd nmoe/csrc
python setup.py build_ext --inplace
```

## NVSHMEM (Multi-Node)

For multi-node training, you need NVSHMEM:

```bash
# Set NVSHMEM root
export NVSHMEM_HOME=/path/to/nvshmem

# Rebuild with NVSHMEM support
cd nmoe/csrc
NMOE_BUILD_NVSHMEM=1 python setup.py build_ext --inplace
```

See [NVSHMEM Setup](../nvshmem.md) for detailed instructions.

## Verification

Verify your installation:

```python
import torch
from nmoe.rdep import Rdep

# Check CUDA
assert torch.cuda.is_available()
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check RDEP
rdep = Rdep(dim=256, n_local=8, topk=2, profile="bf16")
print(f"RDEP mode: {rdep._mode}")
print("Installation verified!")
```

## Troubleshooting

### CUDA Extension Build Fails

```bash
# Check CUDA toolkit
nvcc --version

# Ensure correct PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Import Errors

```bash
# Rebuild extensions
pip install -e . --no-build-isolation --force-reinstall
```

### GPU Not Found

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA visibility
echo $CUDA_VISIBLE_DEVICES
```
