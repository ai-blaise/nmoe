"""Pytest configuration and shared fixtures for nmoe tests."""

import os
import warnings
import pytest
import torch
from typing import Optional
from packaging import version

# =============================================================================
# PINNED VERSIONS - Update these when upgrading dependencies
# =============================================================================
REQUIRED_VERSIONS = {
    "torch": "2.11.0",  # Minimum torch version for B200/SM100
    "triton": "3.6.0",  # Minimum triton version
    "nvidia-cutlass-dsl": "4.3.1",  # Minimum cutlass-dsl for Int4 support
}


def _check_versions():
    """Check that critical dependencies are compatible versions."""
    issues = []

    # Check torch version
    torch_ver = version.parse(torch.__version__.split("+")[0].split(".dev")[0])
    required_torch = version.parse(REQUIRED_VERSIONS["torch"].split(".dev")[0])
    if torch_ver < required_torch:
        issues.append(f"torch {torch.__version__} < {REQUIRED_VERSIONS['torch']} (may lack B200 support)")

    # Check triton version
    try:
        import triton
        triton_ver = version.parse(triton.__version__.split("+")[0])
        required_triton = version.parse(REQUIRED_VERSIONS["triton"])
        if triton_ver < required_triton:
            issues.append(f"triton {triton.__version__} < {REQUIRED_VERSIONS['triton']}")
    except ImportError:
        issues.append("triton not installed")

    # Check cutlass-dsl version
    try:
        import nvidia_cutlass_dsl
        cutlass_ver = version.parse(nvidia_cutlass_dsl.__version__)
        required_cutlass = version.parse(REQUIRED_VERSIONS["nvidia-cutlass-dsl"])
        if cutlass_ver < required_cutlass:
            issues.append(f"nvidia-cutlass-dsl {nvidia_cutlass_dsl.__version__} < {REQUIRED_VERSIONS['nvidia-cutlass-dsl']} (missing Int4 support)")
    except (ImportError, AttributeError):
        pass  # Optional dependency

    return issues


def pytest_configure(config):
    """Configure pytest with custom markers and version checks."""
    # Check versions and warn about issues
    issues = _check_versions()
    for issue in issues:
        warnings.warn(f"Version issue: {issue}", UserWarning)

    # Markers are now defined in pyproject.toml
    pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available hardware."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_multi_gpu = pytest.mark.skip(reason="Multiple GPUs required")
    skip_nvshmem = pytest.mark.skip(reason="NVSHMEM not available")

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    # Check NVSHMEM availability
    nvshmem_available = False
    try:
        from nmoe.csrc import rdep as _C
        nvshmem_available = _C.has_nvshmem()
    except (ImportError, AttributeError):
        pass

    for item in items:
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords and not cuda_available:
            item.add_marker(skip_gpu)

        # Skip multi-GPU tests if not enough GPUs
        if "multi_gpu" in item.keywords and gpu_count < 2:
            item.add_marker(skip_multi_gpu)

        # Skip distributed tests if only 1 GPU
        if "distributed" in item.keywords and gpu_count < 2:
            item.add_marker(skip_multi_gpu)

        # Skip NVSHMEM tests if not available
        if "nvshmem" in item.keywords and not nvshmem_available:
            item.add_marker(skip_nvshmem)


@pytest.fixture
def cuda_device():
    """Fixture that provides a CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def device():
    """Fixture that provides the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def small_model_config():
    """Fixture providing a small model config for testing."""
    return {
        "dim": 256,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "vocab_size": 1000,
        "n_routed_experts": 8,
        "n_activated_experts": 2,
        "moe_inter_dim": 512,
        "max_seq_len": 512,
        "dtype": "bfloat16",
    }


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Fixture providing a temporary directory for checkpoints."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


@pytest.fixture(scope="session")
def gpu_count():
    """Session-scoped fixture returning GPU count."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


@pytest.fixture
def random_seed():
    """Fixture that sets a fixed random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def bf16_tensor_factory(device):
    """Factory fixture for creating BF16 tensors."""
    def _create(shape, requires_grad=False):
        tensor = torch.randn(shape, device=device, dtype=torch.bfloat16)
        tensor.requires_grad_(requires_grad)
        return tensor
    return _create


@pytest.fixture
def expert_weight_factory(device):
    """Factory fixture for creating expert weight tensors."""
    def _create(n_experts, hidden_dim, inter_dim):
        W1 = torch.randn(n_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
        W3 = torch.randn(n_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
        W2 = torch.randn(n_experts, inter_dim, hidden_dim, device=device, dtype=torch.bfloat16)
        return W1, W3, W2
    return _create


def pytest_report_header(config):
    """Add hardware info to pytest header."""
    lines = []
    lines.append(f"PyTorch version: {torch.__version__}")
    lines.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"CUDA version: {torch.version.cuda}")
        lines.append(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            lines.append(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    return lines
