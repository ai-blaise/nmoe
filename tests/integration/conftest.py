"""Pytest configuration for integration tests.

This module provides fixtures and hooks for test isolation, especially
for CUDA and RDEP state cleanup between tests.
"""

import gc
import sys
import importlib
import pytest
import torch


def _cleanup_cuda_state():
    """Deep cleanup of CUDA state between tests."""
    gc.collect()

    if not torch.cuda.is_available():
        return

    # Synchronize and clear all devices
    for device_id in range(torch.cuda.device_count()):
        with torch.cuda.device(device_id):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Reset CUDA generator state completely
    # This is critical to fix "Offset increment outside graph capture" errors
    # that occur when CUDA graphs corrupt the Philox RNG offset
    try:
        for device_id in range(torch.cuda.device_count()):
            gen = torch.cuda.default_generators[device_id]
            # Use graphsafe methods to properly reset state after graph capture
            # First get a clean state snapshot
            clean_gen = torch.Generator(device=f'cuda:{device_id}')
            clean_gen.manual_seed(42 + device_id)
            # Set the default generator to this clean state
            gen.set_state(clean_gen.get_state())
    except Exception as e:
        print(f"Warning: Could not reset CUDA generators: {e}", file=sys.stderr)
        # Fallback to simple seed reset
        try:
            torch.cuda.manual_seed_all(42)
        except Exception as e2:
            print(f"Warning: Fallback seed reset also failed: {e2}", file=sys.stderr)

    # Also reset via the standard API
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Force GC to release tensors
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _reset_rdep_module():
    """Reset RDEP module state by reloading it."""
    # Remove cached rdep modules to force fresh state
    rdep_modules = [m for m in sys.modules if 'rdep' in m.lower()]
    for mod_name in rdep_modules:
        try:
            del sys.modules[mod_name]
        except KeyError:
            pass

    # Also clear nmoe.csrc cache
    nmoe_csrc_modules = [m for m in sys.modules if 'nmoe.csrc' in m]
    for mod_name in nmoe_csrc_modules:
        try:
            del sys.modules[mod_name]
        except KeyError:
            pass

    gc.collect()


@pytest.fixture(autouse=True)
def cleanup_between_tests(request):
    """Autouse fixture to clean up state between every test.

    This runs before and after each test to ensure isolation.
    The 'Offset increment outside graph capture' error typically
    occurs when CUDA graph state from a previous test interferes.
    """
    # Pre-test cleanup
    _cleanup_cuda_state()

    yield

    # Post-test cleanup - more aggressive
    _cleanup_cuda_state()

    # If test failed, do extra cleanup
    if hasattr(request.node, 'rep_call') and request.node.rep_call.failed:
        _reset_rdep_module()
        _cleanup_cuda_state()


@pytest.fixture(scope="function")
def cuda_device():
    """Fixture providing a clean CUDA device context."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    _cleanup_cuda_state()
    device = torch.device("cuda:0")
    yield device
    _cleanup_cuda_state()


@pytest.fixture(scope="function")
def fresh_model_factory():
    """Factory fixture to create models with guaranteed clean state."""
    created_models = []

    def factory(config):
        from nmoe.model import Transformer

        _cleanup_cuda_state()

        model = Transformer(config).cuda().bfloat16()
        model.init_weights()
        created_models.append(model)
        return model

    yield factory

    # Cleanup all created models
    for model in created_models:
        del model
    created_models.clear()
    _cleanup_cuda_state()


@pytest.fixture(scope="function")
def test_config():
    """Standard test configuration with small dimensions for fast testing."""
    from nmoe.config import Config

    return Config(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=1024,
        n_dense_layers=1,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_inter_dim=512,
        inter_dim=512,
        max_position_embeddings=512,
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires GPU")
    config.addinivalue_line("markers", "multi_gpu: test requires multiple GPUs")
    config.addinivalue_line("markers", "slow: slow test")
    config.addinivalue_line("markers", "integration: integration test")


def pytest_runtest_setup(item):
    """Skip tests based on available resources."""
    gpu_markers = list(item.iter_markers(name="gpu"))
    if gpu_markers and not torch.cuda.is_available():
        pytest.skip("GPU not available")

    multi_gpu_markers = list(item.iter_markers(name="multi_gpu"))
    if multi_gpu_markers and torch.cuda.device_count() < 2:
        pytest.skip("Multiple GPUs required")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for cleanup decisions."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
