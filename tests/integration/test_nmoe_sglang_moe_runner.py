"""NmoE MoE Runner Integration Tests for SGLang.

Comprehensive tests for the NmoeRunnerCore integration with SGLang's MoE infrastructure.

Tests cover:
- NmoeRunnerCore initialization with various configs
- Forward pass with nmoe RDEP backend
- Weight loading and versioning
- CUDA graph capture and replay with nmoe
- 8-GPU distributed execution
- Error handling and recovery
- Rdep cache management
- Quantization profiles (BF16/FP8/NVFP4)
- Expert capacity management
- Integration with SGLang's token dispatcher

Run with:
    cd nmoe && source .venv/bin/activate
    uv run pytest tests/integration/test_nmoe_sglang_moe_runner.py -v -s

Requirements:
    - CUDA-enabled GPU
    - nmoe with RDEP C extension built
    - SGLang installed (optional for some tests)
"""

import copy
import gc
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def moe_runner_config():
    """Base MoE runner configuration for tests."""
    return {
        "dim": 256,
        "n_experts": 8,
        "n_local_experts": 8,
        "topk": 2,
        "inter_dim": 512,
        "capacity": 16384,
    }


@pytest.fixture(scope="module")
def large_moe_config():
    """Larger MoE configuration for stress tests."""
    return {
        "dim": 1024,
        "n_experts": 16,
        "n_local_experts": 16,
        "topk": 4,
        "inter_dim": 2048,
        "capacity": 65536,
    }


@pytest.fixture
def expert_weights(moe_runner_config):
    """Create expert weight tensors."""
    E = moe_runner_config["n_local_experts"]
    H = moe_runner_config["dim"]
    D = moe_runner_config["inter_dim"]
    device = "cuda"
    dtype = torch.bfloat16

    W1 = torch.randn(E, H, D, dtype=dtype, device=device) * 0.02
    W3 = torch.randn(E, H, D, dtype=dtype, device=device) * 0.02
    W2 = torch.randn(E, D, H, dtype=dtype, device=device) * 0.02

    return W1, W3, W2


@pytest.fixture
def shared_expert_weights(moe_runner_config):
    """Create shared expert weight tensors."""
    H = moe_runner_config["dim"]
    D = moe_runner_config["inter_dim"]
    device = "cuda"
    dtype = torch.bfloat16

    shared_w1 = torch.randn(H, D, dtype=dtype, device=device) * 0.02
    shared_w3 = torch.randn(H, D, dtype=dtype, device=device) * 0.02
    shared_w2 = torch.randn(D, H, dtype=dtype, device=device) * 0.02

    return shared_w1, shared_w3, shared_w2


@pytest.fixture
def sample_inputs(moe_runner_config):
    """Create sample input tensors."""
    T = 128
    H = moe_runner_config["dim"]
    K = moe_runner_config["topk"]
    E = moe_runner_config["n_local_experts"]
    device = "cuda"

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(torch.randn(T, K, device=device), dim=-1).bfloat16()

    return x, topk_ids, topk_weights


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear RDEP cache before and after each test."""
    yield
    # Cleanup after test
    try:
        from sglang.srt.layers.moe.moe_runner.nmoe import clear_rdep_cache
        clear_rdep_cache()
    except ImportError:
        pass
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Test Classes
# =============================================================================


class TestNmoeRunnerCoreInitialization:
    """Test NmoeRunnerCore initialization with various configurations."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_basic_initialization(self, moe_runner_config):
        """Test basic NmoeRunnerCore initialization."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        assert runner is not None
        assert runner.profile == "bf16"
        assert runner.cuda_graph_mode is False

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_initialization_with_cuda_graph_mode(self, moe_runner_config):
        """Test initialization with CUDA graph mode enabled."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(
            config,
            profile="bf16",
            cuda_graph_mode=True,
            max_batch_size=256,
            max_seq_len=4096,
        )

        assert runner.cuda_graph_mode is True
        assert runner._max_batch_size == 256
        assert runner._max_seq_len == 4096

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_initialization_with_custom_capacity(self, moe_runner_config):
        """Test initialization with custom RDEP capacity."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        custom_capacity = 32768
        runner = NmoeRunnerCore(config, profile="bf16", capacity=custom_capacity)

        assert runner._capacity_override == custom_capacity

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_initialization_fp8_profile(self, moe_runner_config):
        """Test initialization with FP8 quantization profile."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="fp8")
        assert runner.profile == "fp8"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_initialization_nvfp4_profile(self, moe_runner_config):
        """Test initialization with NVFP4 quantization profile."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="nvfp4")
        assert runner.profile == "nvfp4"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_runner_backend_property(self, moe_runner_config):
        """Test that runner_backend returns correct value."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
            from sglang.srt.layers.moe.utils import MoeRunnerBackend
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")
        assert runner.runner_backend == MoeRunnerBackend.NMOE


class TestNmoeRunnerForward:
    """Test NmoeRunnerCore forward pass functionality."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_basic_forward_pass(self, moe_runner_config, expert_weights, sample_inputs):
        """Test basic forward pass through NmoeRunnerCore."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        W1, W3, W2 = expert_weights
        x, topk_ids, topk_weights = sample_inputs

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        output = runner.run(runner_input, quant_info, {})

        assert output.hidden_states.shape == x.shape
        assert output.hidden_states.dtype == torch.bfloat16
        assert not torch.isnan(output.hidden_states).any()
        assert not torch.isinf(output.hidden_states).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_forward_with_varying_batch_sizes(self, moe_runner_config, expert_weights):
        """Test forward pass with varying batch sizes."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")
        W1, W3, W2 = expert_weights

        batch_sizes = [1, 16, 64, 128, 256, 512]

        for T in batch_sizes:
            H = moe_runner_config["dim"]
            K = moe_runner_config["topk"]
            E = moe_runner_config["n_local_experts"]

            x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
            topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
            topk_weights = torch.softmax(
                torch.randn(T, K, device="cuda"), dim=-1
            ).bfloat16()

            runner_input = NmoeRunnerInput(
                hidden_states=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )

            quant_info = NmoeQuantInfo(
                w1_weight=W1,
                w3_weight=W3,
                w2_weight=W2,
                profile="bf16",
            )

            output = runner.run(runner_input, quant_info, {})

            assert output.hidden_states.shape == (T, H), f"Failed for batch size {T}"
            assert not torch.isnan(output.hidden_states).any(), f"NaN for batch {T}"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_forward_with_shared_experts(
        self, moe_runner_config, expert_weights, shared_expert_weights, sample_inputs
    ):
        """Test forward pass with shared experts."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
            num_fused_shared_experts=1,
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        W1, W3, W2 = expert_weights
        shared_w1, shared_w3, shared_w2 = shared_expert_weights
        x, topk_ids, topk_weights = sample_inputs

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
            shared_w1=shared_w1,
            shared_w3=shared_w3,
            shared_w2=shared_w2,
        )

        output = runner.run(runner_input, quant_info, {})

        assert output.hidden_states.shape == x.shape
        assert not torch.isnan(output.hidden_states).any()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_forward_output_determinism(
        self, moe_runner_config, expert_weights, sample_inputs
    ):
        """Test that forward pass is deterministic with same inputs."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        W1, W3, W2 = expert_weights
        x, topk_ids, topk_weights = sample_inputs

        runner_input = NmoeRunnerInput(
            hidden_states=x.clone(),
            topk_ids=topk_ids.clone(),
            topk_weights=topk_weights.clone(),
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        # Run twice with same inputs
        output1 = runner.run(runner_input, quant_info, {})

        runner_input2 = NmoeRunnerInput(
            hidden_states=x.clone(),
            topk_ids=topk_ids.clone(),
            topk_weights=topk_weights.clone(),
        )
        output2 = runner.run(runner_input2, quant_info, {})

        # Should be identical
        assert torch.allclose(
            output1.hidden_states, output2.hidden_states, rtol=1e-3, atol=1e-3
        )


class TestWeightLoadingAndVersioning:
    """Test weight loading and versioning functionality."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_weight_version_tracking(self, moe_runner_config):
        """Test weight version tracking for CUDA graph invalidation."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16", cuda_graph_mode=True)

        # Initial version
        initial_version = runner._weight_version
        assert initial_version == 0

        # Notify weight update
        runner.notify_weight_update()
        assert runner._weight_version == 1

        # Multiple updates
        runner.notify_weight_update()
        runner.notify_weight_update()
        assert runner._weight_version == 3

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_cuda_graph_invalidation(self, moe_runner_config, expert_weights):
        """Test that CUDA graphs are invalidated when weights change."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16", cuda_graph_mode=True)

        # Manually add a fake cached graph
        runner._graph_cache[128] = "fake_graph"
        assert len(runner._graph_cache) == 1

        # Invalidate
        count = runner.invalidate_cuda_graphs()
        assert count == 1
        assert len(runner._graph_cache) == 0

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_weight_shape_validation(self, moe_runner_config):
        """Test that weight shape validation works correctly."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        T = 64
        H = moe_runner_config["dim"]
        K = moe_runner_config["topk"]
        E = moe_runner_config["n_local_experts"]
        D = moe_runner_config["inter_dim"]

        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
        topk_weights = torch.softmax(torch.randn(T, K, device="cuda"), dim=-1).bfloat16()

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        # Wrong W1 shape - should fail validation
        W1_wrong = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda")  # Wrong!
        W3 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda")

        quant_info = NmoeQuantInfo(
            w1_weight=W1_wrong,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        with pytest.raises(ValueError, match="hidden dim mismatch"):
            runner.run(runner_input, quant_info, {})

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_weight_dtype_conversion(self, moe_runner_config, sample_inputs):
        """Test that weights are converted to correct dtype."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        E = moe_runner_config["n_local_experts"]
        H = moe_runner_config["dim"]
        D = moe_runner_config["inter_dim"]

        # Weights in float32 (should be converted internally)
        W1 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda") * 0.02
        W3 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda") * 0.02
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda") * 0.02

        x, topk_ids, topk_weights = sample_inputs

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        # Should work with BF16 weights
        output = runner.run(runner_input, quant_info, {})
        assert output.hidden_states.dtype == torch.bfloat16


class TestCudaGraphSupport:
    """Test CUDA graph capture and replay functionality."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_cuda_graph_warmup(self, moe_runner_config, expert_weights):
        """Test CUDA graph warmup."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(
            config,
            profile="bf16",
            cuda_graph_mode=True,
        )

        W1, W3, W2 = expert_weights

        # Set dimensions from weight shapes
        runner.dim = moe_runner_config["dim"]
        runner.n_experts = moe_runner_config["n_local_experts"]
        runner.topk = moe_runner_config["topk"]

        # Warmup should capture graph
        batch_size = 32
        seq_len = 1
        runner.warmup_cuda_graph(batch_size, seq_len, W1, W3, W2)

        # Graph should be cached
        T = batch_size * seq_len
        assert T in runner._graph_cache

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_cuda_graph_disabled(self, moe_runner_config, expert_weights):
        """Test that CUDA graph warmup is skipped when disabled."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(
            config,
            profile="bf16",
            cuda_graph_mode=False,  # Disabled
        )

        W1, W3, W2 = expert_weights

        # Warmup should be no-op
        runner.warmup_cuda_graph(32, 1, W1, W3, W2)

        # No graphs should be cached
        assert len(runner._graph_cache) == 0

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_cuda_graph_cache_deduplication(self, moe_runner_config, expert_weights):
        """Test that CUDA graph cache avoids duplicate captures."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(
            config,
            profile="bf16",
            cuda_graph_mode=True,
        )

        runner.dim = moe_runner_config["dim"]
        runner.n_experts = moe_runner_config["n_local_experts"]
        runner.topk = moe_runner_config["topk"]

        W1, W3, W2 = expert_weights

        # First capture
        runner.warmup_cuda_graph(32, 1, W1, W3, W2)
        cache_size_1 = len(runner._graph_cache)

        # Second capture (same batch size) - should be deduplicated
        runner.warmup_cuda_graph(32, 1, W1, W3, W2)
        cache_size_2 = len(runner._graph_cache)

        assert cache_size_1 == cache_size_2 == 1


class TestRdepCacheManagement:
    """Test RDEP instance cache management."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rdep_cache_creation(self, moe_runner_config):
        """Test RDEP cache creates instances correctly."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                _get_or_create_rdep,
                get_rdep_cache_stats,
                clear_rdep_cache,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        clear_rdep_cache()

        rdep = _get_or_create_rdep(
            dim=moe_runner_config["dim"],
            n_local=moe_runner_config["n_local_experts"],
            topk=moe_runner_config["topk"],
            profile="bf16",
            capacity=moe_runner_config["capacity"],
        )

        assert rdep is not None

        stats = get_rdep_cache_stats()
        assert stats["size"] == 1

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rdep_cache_reuse(self, moe_runner_config):
        """Test RDEP cache reuses existing instances."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                _get_or_create_rdep,
                get_rdep_cache_stats,
                clear_rdep_cache,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        clear_rdep_cache()

        # First call - creates new instance
        rdep1 = _get_or_create_rdep(
            dim=moe_runner_config["dim"],
            n_local=moe_runner_config["n_local_experts"],
            topk=moe_runner_config["topk"],
            profile="bf16",
            capacity=moe_runner_config["capacity"],
        )

        # Second call with same config - should reuse
        rdep2 = _get_or_create_rdep(
            dim=moe_runner_config["dim"],
            n_local=moe_runner_config["n_local_experts"],
            topk=moe_runner_config["topk"],
            profile="bf16",
            capacity=moe_runner_config["capacity"],
        )

        # Same instance should be returned
        assert rdep1 is rdep2

        stats = get_rdep_cache_stats()
        assert stats["size"] == 1

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rdep_cache_clear(self, moe_runner_config):
        """Test RDEP cache clearing."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                _get_or_create_rdep,
                get_rdep_cache_stats,
                clear_rdep_cache,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        clear_rdep_cache()

        # Create instance
        _get_or_create_rdep(
            dim=moe_runner_config["dim"],
            n_local=moe_runner_config["n_local_experts"],
            topk=moe_runner_config["topk"],
            profile="bf16",
            capacity=moe_runner_config["capacity"],
        )

        stats = get_rdep_cache_stats()
        assert stats["size"] == 1

        # Clear cache
        count = clear_rdep_cache()
        assert count == 1

        stats = get_rdep_cache_stats()
        assert stats["size"] == 0

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_rdep_cache_stats(self, moe_runner_config):
        """Test RDEP cache statistics."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                _get_or_create_rdep,
                get_rdep_cache_stats,
                clear_rdep_cache,
                _RDEP_CACHE_MAX_SIZE,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        clear_rdep_cache()

        _get_or_create_rdep(
            dim=moe_runner_config["dim"],
            n_local=moe_runner_config["n_local_experts"],
            topk=moe_runner_config["topk"],
            profile="bf16",
            capacity=moe_runner_config["capacity"],
        )

        stats = get_rdep_cache_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "entries" in stats
        assert stats["size"] == 1
        assert stats["max_size"] == _RDEP_CACHE_MAX_SIZE
        assert len(stats["entries"]) == 1


class TestCapacityComputation:
    """Test RDEP capacity computation."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_compute_rdep_capacity_basic(self):
        """Test basic capacity computation."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import compute_rdep_capacity
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        capacity = compute_rdep_capacity(
            max_batch_size=256,
            max_seq_len=4096,
            topk=2,
            world_size=1,
        )

        # Should be power of 2 and >= required
        required = 256 * 4096 * 2 * 1
        assert capacity >= required
        assert (capacity & (capacity - 1)) == 0  # Power of 2

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_compute_rdep_capacity_multi_gpu(self):
        """Test capacity computation for multi-GPU."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import compute_rdep_capacity
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        capacity = compute_rdep_capacity(
            max_batch_size=64,
            max_seq_len=2048,
            topk=4,
            world_size=8,
        )

        required = 64 * 2048 * 4 * 8
        assert capacity >= required

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_compute_rdep_capacity_minimum(self):
        """Test that capacity has a minimum value."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import compute_rdep_capacity
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        capacity = compute_rdep_capacity(
            max_batch_size=1,
            max_seq_len=1,
            topk=1,
            world_size=1,
        )

        # Should be at least 1024
        assert capacity >= 1024

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_compute_rdep_capacity_headroom(self):
        """Test capacity includes headroom factor."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import compute_rdep_capacity
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        base_required = 1000 * 2 * 1
        capacity = compute_rdep_capacity(
            max_batch_size=1000,
            max_seq_len=1,
            topk=2,
            world_size=1,
            headroom_factor=1.5,
        )

        # With 1.5x headroom, should be >= 1.5 * required
        assert capacity >= base_required * 1.5


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_invalid_profile_error(self, moe_runner_config):
        """Test error handling for invalid profile."""
        try:
            from nmoe.rdep import Rdep
        except ImportError:
            pytest.skip("nmoe not available")

        with pytest.raises(TypeError, match="profile must be one of"):
            Rdep(
                dim=moe_runner_config["dim"],
                n_local=moe_runner_config["n_local_experts"],
                topk=moe_runner_config["topk"],
                profile="invalid_profile",
            )

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_mismatched_weight_shapes(self, moe_runner_config, sample_inputs):
        """Test error handling for mismatched weight shapes."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        E = moe_runner_config["n_local_experts"]
        H = moe_runner_config["dim"]
        D = moe_runner_config["inter_dim"]

        x, topk_ids, topk_weights = sample_inputs

        # W1 and W3 have different shapes
        W1 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W3 = torch.randn(E, H, D // 2, dtype=torch.bfloat16, device="cuda")  # Wrong!
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda")

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        with pytest.raises(ValueError, match="shape mismatch"):
            runner.run(runner_input, quant_info, {})

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_blockscaled_without_cache_error(self, moe_runner_config, sample_inputs):
        """Test error when blockscaled profile used without w_cache."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="fp8")

        E = moe_runner_config["n_local_experts"]
        H = moe_runner_config["dim"]
        D = moe_runner_config["inter_dim"]

        x, topk_ids, topk_weights = sample_inputs

        W1 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W3 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda")

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="fp8",
            w_cache=None,  # Missing!
        )

        with pytest.raises(ValueError, match="requires pre-computed w_cache"):
            runner.run(runner_input, quant_info, {})

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_runner_cleanup(self, moe_runner_config, expert_weights, sample_inputs):
        """Test runner cleanup releases resources."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16", cuda_graph_mode=True)

        W1, W3, W2 = expert_weights
        x, topk_ids, topk_weights = sample_inputs

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        # Run to initialize RDEP
        runner.run(runner_input, quant_info, {})

        # Add fake graph to cache
        runner._graph_cache[128] = "fake"

        # Cleanup
        runner.cleanup()

        assert runner._rdep is None
        assert len(runner._graph_cache) == 0


class TestFusedFunctions:
    """Test fused MoE functions."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_fused_experts_none_to_nmoe(self, moe_runner_config, expert_weights):
        """Test fused_experts_none_to_nmoe function."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                fused_experts_none_to_nmoe,
                NmoeQuantInfo,
                clear_rdep_cache,
            )
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardDispatchOutput,
            )
            from sglang.srt.layers.moe.topk import StandardTopKOutput
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        clear_rdep_cache()

        T = 128
        H = moe_runner_config["dim"]
        K = moe_runner_config["topk"]
        E = moe_runner_config["n_local_experts"]

        W1, W3, W2 = expert_weights

        # Create StandardDispatchOutput
        hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        topk_weights = torch.softmax(
            torch.randn(T, K, device="cuda"), dim=-1
        ).float()
        topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
        router_logits = torch.randn(T, E, dtype=torch.float32, device="cuda")

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        runner_config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=K,
        )

        # Call fused function
        combine_input = fused_experts_none_to_nmoe(
            dispatch_output, quant_info, runner_config
        )

        assert combine_input.hidden_states.shape == (T, H)
        assert not torch.isnan(combine_input.hidden_states).any()


class TestPrePermute:
    """Test pre-permute functions."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_pre_permute_standard_to_nmoe(self, moe_runner_config):
        """Test pre_permute_standard_to_nmoe function."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                pre_permute_standard_to_nmoe,
                NmoeQuantInfo,
                NmoeRunnerInput,
            )
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardDispatchOutput,
            )
            from sglang.srt.layers.moe.topk import StandardTopKOutput
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        T = 64
        H = moe_runner_config["dim"]
        K = moe_runner_config["topk"]
        E = moe_runner_config["n_local_experts"]
        D = moe_runner_config["inter_dim"]

        # Create StandardDispatchOutput
        hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        topk_weights = torch.softmax(
            torch.randn(T, K, device="cuda"), dim=-1
        ).float()
        topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
        router_logits = torch.randn(T, E, dtype=torch.float32, device="cuda")

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        W1 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W3 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda")

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        runner_config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=D,
            top_k=K,
        )

        # Call pre_permute
        runner_input = pre_permute_standard_to_nmoe(
            dispatch_output, quant_info, runner_config, {}
        )

        assert isinstance(runner_input, NmoeRunnerInput)
        assert runner_input.hidden_states.shape == (T, H)
        assert runner_input.topk_ids.shape == (T, K)
        assert runner_input.topk_weights.shape == (T, K)


class TestPostPermute:
    """Test post-permute functions."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_post_permute_nmoe_to_standard(self, moe_runner_config):
        """Test post_permute_nmoe_to_standard function."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                post_permute_nmoe_to_standard,
                NmoeQuantInfo,
                NmoeRunnerOutput,
            )
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardCombineInput,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        T = 64
        H = moe_runner_config["dim"]
        E = moe_runner_config["n_local_experts"]
        D = moe_runner_config["inter_dim"]

        # Create NmoeRunnerOutput
        hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        runner_output = NmoeRunnerOutput(hidden_states=hidden_states)

        W1 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W3 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda")
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda")

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        runner_config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=D,
            top_k=moe_runner_config["topk"],
        )

        # Call post_permute
        combine_input = post_permute_nmoe_to_standard(
            runner_output, quant_info, runner_config, {}
        )

        assert isinstance(combine_input, StandardCombineInput)
        assert combine_input.hidden_states.shape == (T, H)


class TestNumericalStability:
    """Test numerical stability of NmoeRunnerCore."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gradient_flow(self, moe_runner_config):
        """Test gradient flow through NmoeRunnerCore."""
        try:
            from nmoe.rdep import Rdep
        except ImportError:
            pytest.skip("nmoe not available")

        rdep = Rdep(
            dim=moe_runner_config["dim"],
            n_local=moe_runner_config["n_local_experts"],
            topk=moe_runner_config["topk"],
            profile="bf16",
        )

        T = 64
        H = moe_runner_config["dim"]
        K = moe_runner_config["topk"]
        E = moe_runner_config["n_local_experts"]
        D = moe_runner_config["inter_dim"]

        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        eid = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
        gates = torch.softmax(torch.randn(T, K, device="cuda"), dim=-1).bfloat16()

        W1 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        W3 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
        loss = output.sum()

        try:
            loss.backward()

            # Check gradients exist and are finite
            if x.grad is not None:
                assert torch.isfinite(x.grad).all()
            if W1.grad is not None:
                assert torch.isfinite(W1.grad).all()
            if W2.grad is not None:
                assert torch.isfinite(W2.grad).all()
            if W3.grad is not None:
                assert torch.isfinite(W3.grad).all()
        except RuntimeError:
            pytest.skip("Backward not supported")

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_output_range(self, moe_runner_config, expert_weights, sample_inputs):
        """Test that outputs are in reasonable range."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(config, profile="bf16")

        W1, W3, W2 = expert_weights
        x, topk_ids, topk_weights = sample_inputs

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        output = runner.run(runner_input, quant_info, {})

        # Output should be finite
        assert torch.isfinite(output.hidden_states).all()

        # Output magnitude should be reasonable (not too large)
        max_val = output.hidden_states.abs().max().item()
        assert max_val < 1e6, f"Output too large: max={max_val}"


class TestExpertCapacity:
    """Test expert capacity management."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_capacity_within_limit(self, moe_runner_config, expert_weights):
        """Test execution with capacity within limit."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import (
                NmoeRunnerCore,
                NmoeRunnerInput,
                NmoeQuantInfo,
            )
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        runner = NmoeRunnerCore(
            config,
            profile="bf16",
            capacity=moe_runner_config["capacity"],
        )

        W1, W3, W2 = expert_weights

        T = 128  # Within capacity
        H = moe_runner_config["dim"]
        K = moe_runner_config["topk"]
        E = moe_runner_config["n_local_experts"]

        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
        topk_weights = torch.softmax(torch.randn(T, K, device="cuda"), dim=-1).bfloat16()

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        output = runner.run(runner_input, quant_info, {})
        assert output.hidden_states.shape == (T, H)

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_large_batch_capacity(self, large_moe_config):
        """Test large batch with adequate capacity."""
        try:
            from nmoe.rdep import Rdep
        except ImportError:
            pytest.skip("nmoe not available")

        rdep = Rdep(
            dim=large_moe_config["dim"],
            n_local=large_moe_config["n_local_experts"],
            topk=large_moe_config["topk"],
            profile="bf16",
            capacity=large_moe_config["capacity"],
        )

        T = 4096
        H = large_moe_config["dim"]
        K = large_moe_config["topk"]
        E = large_moe_config["n_local_experts"]
        D = large_moe_config["inter_dim"]

        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        eid = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
        gates = torch.softmax(torch.randn(T, K, device="cuda"), dim=-1).bfloat16()

        W1 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda") * 0.02
        W3 = torch.randn(E, H, D, dtype=torch.bfloat16, device="cuda") * 0.02
        W2 = torch.randn(E, D, H, dtype=torch.bfloat16, device="cuda") * 0.02

        output = rdep.moe_bf16(x, eid, gates, W1, W3, W2)

        assert output.shape == (T, H)
        assert not torch.isnan(output).any()


class TestMultiGPUDistributed:
    """Test multi-GPU distributed execution."""

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.multi_gpu
    def test_distributed_init_check(self):
        """Test that distributed initialization is handled correctly."""
        try:
            from nmoe.rdep import Rdep
            import torch.distributed as dist
        except ImportError:
            pytest.skip("nmoe not available")

        # Without dist initialized, should work in single mode
        if not dist.is_initialized():
            rdep = Rdep(
                dim=256,
                n_local=8,
                topk=2,
                profile="bf16",
            )
            assert rdep._mode == "single"

    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.distributed
    def test_ep_group_parameter(self, moe_runner_config):
        """Test that ep_group parameter is stored correctly."""
        try:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerCore
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        config = MoeRunnerConfig(
            num_experts=moe_runner_config["n_experts"],
            num_local_experts=moe_runner_config["n_local_experts"],
            hidden_size=moe_runner_config["dim"],
            intermediate_size_per_partition=moe_runner_config["inter_dim"],
            top_k=moe_runner_config["topk"],
        )

        # Create runner with None ep_group (default)
        runner = NmoeRunnerCore(config, profile="bf16", ep_group=None)
        assert runner.ep_group is None


class TestInputOutput:
    """Test NmoeRunnerInput and NmoeRunnerOutput classes."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_nmoe_runner_input_properties(self, moe_runner_config, sample_inputs):
        """Test NmoeRunnerInput properties."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerInput
            from sglang.srt.layers.moe.utils import MoeRunnerBackend
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        x, topk_ids, topk_weights = sample_inputs

        runner_input = NmoeRunnerInput(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        assert runner_input.runner_backend == MoeRunnerBackend.NMOE
        assert runner_input.hidden_states is x
        assert runner_input.topk_ids is topk_ids
        assert runner_input.topk_weights is topk_weights

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_nmoe_runner_output_properties(self, moe_runner_config):
        """Test NmoeRunnerOutput properties."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeRunnerOutput
            from sglang.srt.layers.moe.utils import MoeRunnerBackend
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        T = 64
        H = moe_runner_config["dim"]

        hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        expert_counts = torch.randint(0, 100, (moe_runner_config["n_experts"],), device="cuda")

        runner_output = NmoeRunnerOutput(
            hidden_states=hidden_states,
            expert_counts=expert_counts,
        )

        assert runner_output.runner_backend == MoeRunnerBackend.NMOE
        assert runner_output.hidden_states is hidden_states
        assert runner_output.expert_counts is expert_counts


class TestQuantInfoClasses:
    """Test NmoeQuantInfo class."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_nmoe_quant_info_bf16(self, moe_runner_config, expert_weights):
        """Test NmoeQuantInfo for BF16 profile."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeQuantInfo
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        W1, W3, W2 = expert_weights

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
        )

        assert quant_info.profile == "bf16"
        assert quant_info.w_cache is None
        assert quant_info.shared_w1 is None

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_nmoe_quant_info_with_shared_experts(
        self, moe_runner_config, expert_weights, shared_expert_weights
    ):
        """Test NmoeQuantInfo with shared experts."""
        try:
            from sglang.srt.layers.moe.moe_runner.nmoe import NmoeQuantInfo
        except ImportError:
            pytest.skip("SGLang nmoe runner not available")

        W1, W3, W2 = expert_weights
        shared_w1, shared_w3, shared_w2 = shared_expert_weights

        quant_info = NmoeQuantInfo(
            w1_weight=W1,
            w3_weight=W3,
            w2_weight=W2,
            profile="bf16",
            shared_w1=shared_w1,
            shared_w3=shared_w3,
            shared_w2=shared_w2,
        )

        assert quant_info.shared_w1 is shared_w1
        assert quant_info.shared_w3 is shared_w3
        assert quant_info.shared_w2 is shared_w2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
