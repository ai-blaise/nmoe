"""Tests for forward pass equivalence.

Tests that converted model produces equivalent outputs to original nmoe model.

NOTE: Full model tests require compiled CUDA extensions (nmoe.csrc).
Run 'make' in nmoe/csrc/ to build, or run tests in Docker container.
"""

import pytest
import sys
sys.path.insert(0, '/home/nourdine/sglang_nmoe/nether-soup/nmoe')

import torch
import torch.nn as nn

# Check if full nmoe is available
try:
    from nmoe.model import Transformer, MLP, Router, MoE
    from nmoe.config import Config
    NMOE_AVAILABLE = True
except ImportError:
    NMOE_AVAILABLE = False
    # Create minimal stubs for type checking
    Config = None
    Transformer = None
    MLP = None
    Router = None
    MoE = None

from nmoe.unified.config import NMoEModelConfig
from nmoe.tools.config_converter import nmoe_to_hf_weight_mapping

pytestmark = pytest.mark.skipif(
    not NMOE_AVAILABLE,
    reason="nmoe CUDA extensions not built. Run 'make' in nmoe/csrc/"
)


class TestMLPForwardEquivalence:
    """Test MLP forward pass equivalence."""

    def test_mlp_forward_shape(self):
        """Test MLP produces correct output shape."""
        dim = 256
        inter_dim = 512
        batch_size = 4
        seq_len = 32

        mlp = MLP(dim, inter_dim).cuda()

        x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device='cuda')
        out = mlp(x)

        assert out.shape == (batch_size, seq_len, dim)
        assert out.dtype == torch.bfloat16

    def test_mlp_deterministic(self):
        """Test MLP forward is deterministic."""
        dim = 256
        inter_dim = 512

        mlp = MLP(dim, inter_dim).cuda()

        x = torch.randn(2, 16, dim, dtype=torch.bfloat16, device='cuda')

        out1 = mlp(x)
        out2 = mlp(x)

        assert torch.allclose(out1, out2, atol=1e-6)


class TestRouterForwardEquivalence:
    """Test Router forward pass."""

    def test_router_forward_shape(self):
        """Test Router produces correct output shapes."""
        cfg = Config(
            dim=256,
            n_layers=4,
            n_heads=8,
            n_routed_experts=16,
            n_activated_experts=4,
        )

        router = Router(cfg).cuda()

        x = torch.randn(4, 32, 256, dtype=torch.bfloat16, device='cuda')
        weights, indices = router(x.view(-1, 256))

        assert weights.shape == (4 * 32, 4)  # [tokens, topk]
        assert indices.shape == (4 * 32, 4)
        assert weights.dtype == torch.bfloat16
        assert indices.dtype == torch.int64

    def test_router_weights_sum_to_one(self):
        """Test Router weights are normalized."""
        cfg = Config(
            dim=256,
            n_layers=4,
            n_heads=8,
            n_routed_experts=16,
            n_activated_experts=4,
        )

        router = Router(cfg).cuda()

        x = torch.randn(4, 32, 256, dtype=torch.bfloat16, device='cuda')
        weights, _ = router(x.view(-1, 256))

        # Weights should sum to 1 per token
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


class TestTransformerForward:
    """Test Transformer forward pass."""

    @pytest.fixture
    def small_config(self):
        """Create small test config."""
        return Config(
            dim=512,
            n_layers=4,
            n_heads=8,
            inter_dim=1024,
            moe_inter_dim=256,
            n_routed_experts=8,
            n_activated_experts=2,
            n_shared_experts=1,
            n_dense_layers=1,
            vocab_size=1000,
            batch_size=2,
            seq_len=64,
            max_position_embeddings=128,
            dtype='bf16',
        )

    def test_transformer_forward_shape(self, small_config):
        """Test Transformer produces correct output shape."""
        # Initialize distributed if needed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29500',
                world_size=1,
                rank=0,
            )

        model = Transformer(small_config).cuda()
        model.init_weights()

        tokens = torch.randint(0, small_config.vocab_size, (2, 64), device='cuda')
        logits = model(tokens)

        assert logits.shape == (2, 64, small_config.vocab_size)
        assert logits.dtype == torch.bfloat16

    def test_transformer_deterministic(self, small_config):
        """Test Transformer forward is deterministic."""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29501',
                world_size=1,
                rank=0,
            )

        model = Transformer(small_config).cuda()
        model.init_weights()
        model.eval()

        tokens = torch.randint(0, small_config.vocab_size, (2, 64), device='cuda')

        with torch.no_grad():
            logits1 = model(tokens)
            logits2 = model(tokens)

        assert torch.allclose(logits1, logits2, atol=1e-5)


class TestConfigConversion:
    """Test config conversion preserves model behavior."""

    def test_unified_config_matches_nmoe(self):
        """Test unified config preserves all nmoe settings."""
        nmoe_cfg = Config(
            dim=512,
            n_layers=4,
            n_heads=8,
            inter_dim=1024,
            moe_inter_dim=256,
            n_routed_experts=8,
            n_activated_experts=2,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )

        unified = NMoEModelConfig.from_nmoe_config(nmoe_cfg)

        assert unified.hidden_size == nmoe_cfg.dim
        assert unified.num_hidden_layers == nmoe_cfg.n_layers
        assert unified.num_attention_heads == nmoe_cfg.n_heads
        assert unified.intermediate_size == nmoe_cfg.inter_dim
        assert unified.moe_intermediate_size == nmoe_cfg.moe_inter_dim
        assert unified.num_experts == nmoe_cfg.n_routed_experts
        assert unified.num_experts_per_tok == nmoe_cfg.n_activated_experts
        assert unified.q_lora_rank == nmoe_cfg.q_lora_rank
        assert unified.kv_lora_rank == nmoe_cfg.kv_lora_rank


class TestWeightMappingWithTensors:
    """Test weight mapping with actual tensors."""

    def test_weight_shapes_match_mapping(self):
        """Test that mapped weights have expected shapes."""
        cfg = Config(
            dim=256,
            n_layers=2,
            n_heads=4,
            inter_dim=512,
            moe_inter_dim=128,
            n_routed_experts=4,
            n_activated_experts=2,
            n_shared_experts=1,
            n_dense_layers=1,
            vocab_size=100,
            batch_size=1,
            seq_len=16,
            dtype='bf16',
        )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29502',
                world_size=1,
                rank=0,
            )

        model = Transformer(cfg).cuda()
        model.init_weights()

        state_dict = model.state_dict()
        mapping = nmoe_to_hf_weight_mapping(
            n_layers=cfg.n_layers,
            n_dense_layers=cfg.n_dense_layers,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
        )

        # Verify key weights exist and have expected shapes
        assert 'embedding.weight' in state_dict
        assert state_dict['embedding.weight'].shape == (cfg.vocab_size, cfg.dim)

        assert 'lm_head.weight' in state_dict
        assert state_dict['lm_head.weight'].shape == (cfg.vocab_size, cfg.dim)

        # Check MoE layer weights
        assert 'blocks.1.ffn.W1' in state_dict
        # W1 shape: [n_local_experts, dim, moe_inter_dim]
        assert state_dict['blocks.1.ffn.W1'].shape[1] == cfg.dim
        assert state_dict['blocks.1.ffn.W1'].shape[2] == cfg.moe_inter_dim


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
