"""SkyRL integration tests: nmoe + SGLang + SkyRL RL training.

This test validates the integration between nmoe models and SkyRL's
reinforcement learning training framework.

Task 6.1.2 from Niwa implementation checklist.

Run with:
    pytest tests/integration/test_skyrl_integration.py -v -s

Requirements:
    - GPU with at least 24GB VRAM
    - nmoe, sglang, and skyrl installed
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Skip if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for integration tests"
)


@pytest.fixture(scope="module")
def small_model_config():
    """Small model config for fast testing."""
    from nmoe.config import Config
    return Config(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=1024,
        n_dense_layers=1,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=0,
        moe_inter_dim=512,
        inter_dim=512,
        max_position_embeddings=256,
    )


class TestNMoEModelWrapper:
    """Test NMoEModelWrapper for SkyRL compatibility."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_wrapper_import(self):
        """Test that NMoEModelWrapper can be imported."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            assert NMoEModelWrapper is not None
        except ImportError as e:
            pytest.skip(f"SkyRL nmoe wrapper not available: {e}")

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_wrapper_initialization(self, small_model_config):
        """Test wrapper initialization from config."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        # Create model
        model = Transformer(small_model_config).cuda().bfloat16()

        # Wrap for RL training
        wrapper = NMoEModelWrapper(model)

        assert wrapper is not None
        assert isinstance(wrapper, nn.Module)

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_wrapper_forward(self, small_model_config):
        """Test wrapper forward pass returns log probs."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()
        wrapper = NMoEModelWrapper(model)

        # Create dummy input
        batch_size = 2
        seq_len = 64
        num_actions = 16
        sequences = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()

        # Forward pass using forward_rl directly (SkyRL RL training pattern)
        # The __call__ method dispatches based on positional args, so use forward_rl explicitly
        with torch.no_grad():
            output = wrapper.forward_rl(
                sequences=sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
            )

        # Check output structure - forward_rl returns action_log_probs tensor
        assert isinstance(output, torch.Tensor)
        # Output should be [batch_size, num_actions]
        assert output.dim() == 2
        assert output.shape[0] == batch_size
        assert output.shape[1] == num_actions

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gradient_checkpointing(self, small_model_config):
        """Test gradient checkpointing enable/disable."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()
        wrapper = NMoEModelWrapper(model)

        # Enable gradient checkpointing
        if hasattr(wrapper, "gradient_checkpointing_enable"):
            wrapper.gradient_checkpointing_enable()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_expert_cache_refresh(self, small_model_config):
        """Test expert cache refresh for quantized models."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()
        wrapper = NMoEModelWrapper(model)

        # Refresh expert caches (for FP8/NVFP4)
        if hasattr(wrapper, "refresh_expert_caches"):
            wrapper.refresh_expert_caches()

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_load_balancing_loss(self, small_model_config):
        """Test router auxiliary loss computation."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()
        wrapper = NMoEModelWrapper(model)

        # Get router aux loss
        if hasattr(wrapper, "get_router_aux_loss"):
            aux_loss = wrapper.get_router_aux_loss()
            if aux_loss is not None:
                assert isinstance(aux_loss, torch.Tensor)


class TestWeightSync:
    """Test weight synchronization between training and inference."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_weight_extraction(self, small_model_config):
        """Test extracting weights for sync to inference engines."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()
        wrapper = NMoEModelWrapper(model)

        # Get state dict
        state_dict = wrapper.state_dict()

        assert len(state_dict) > 0
        # Check for key model components
        has_embedding = any("embed" in k.lower() or "wte" in k.lower() for k in state_dict)
        has_layers = any("layer" in k.lower() or "block" in k.lower() for k in state_dict)

        assert has_embedding or has_layers, "Missing model components in state dict"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_weight_consistency(self, small_model_config):
        """Test that weights are consistent after wrapper creation."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()

        # Get original weights
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Wrap model
        wrapper = NMoEModelWrapper(model)

        # Get wrapped weights
        if hasattr(wrapper, "model"):
            wrapped_state = wrapper.model.state_dict()
        else:
            wrapped_state = wrapper.state_dict()

        # Verify weights match
        for key in original_state:
            if key in wrapped_state:
                orig = original_state[key]
                wrapped = wrapped_state[key]
                # Use allclose for floating-point robustness, ensure same contiguity
                orig_c = orig.contiguous() if not orig.is_contiguous() else orig
                wrapped_c = wrapped.contiguous() if not wrapped.is_contiguous() else wrapped
                assert torch.allclose(orig_c, wrapped_c, rtol=0, atol=0), f"Weight mismatch for {key}"


class TestRLTrainingLoop:
    """Test RL training loop integration."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_forward_backward(self, small_model_config):
        """Test forward/backward pass for RL training."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()
        wrapper = NMoEModelWrapper(model)

        # Create dummy input
        batch_size = 2
        seq_len = 64
        sequences = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len)).cuda()

        # Forward pass with grad
        output = wrapper.model(sequences)  # Get logits

        # Simulate RL loss (dummy)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_grads = False
        for param in wrapper.parameters():
            if param.grad is not None:
                has_grads = True
                break

        assert has_grads, "No gradients computed"

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_optimizer_step(self, small_model_config):
        """Test optimizer step updates weights."""
        try:
            from skyrl_train.model_wrapper_nmoe import NMoEModelWrapper
            from nmoe.model import Transformer
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        model = Transformer(small_model_config).cuda().bfloat16()
        wrapper = NMoEModelWrapper(model)

        # Get initial weights
        initial_weights = {k: v.clone() for k, v in wrapper.state_dict().items()}

        # Setup optimizer
        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)

        # Forward/backward
        batch_size = 2
        seq_len = 32
        sequences = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len)).cuda()

        output = wrapper.model(sequences)
        loss = output.sum()
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Check weights changed
        weights_changed = False
        for key in initial_weights:
            if key in wrapper.state_dict():
                if not torch.allclose(initial_weights[key], wrapper.state_dict()[key]):
                    weights_changed = True
                    break

        assert weights_changed, "Weights did not change after optimizer step"


class TestSkyRLBridge:
    """Test SkyRL bridge for EP coordination."""

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_bridge_import(self):
        """Test that SkyRL bridge can be imported."""
        try:
            from nmoe.distributed.skyrl_bridge import SkyRLRdepBridge
            assert SkyRLRdepBridge is not None
        except ImportError as e:
            pytest.skip(f"SkyRL bridge not available: {e}")

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_bridge_initialization(self, small_model_config):
        """Test bridge initialization."""
        try:
            from nmoe.distributed.skyrl_bridge import SkyRLRdepBridge
        except ImportError as e:
            pytest.skip(f"SkyRL bridge not available: {e}")

        # Create bridge for single GPU (EP=1)
        bridge = SkyRLRdepBridge(
            dim=small_model_config.dim,
            n_total_experts=small_model_config.n_routed_experts,
            topk=small_model_config.n_activated_experts,
        )

        assert bridge is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
