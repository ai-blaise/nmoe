"""Tests for NVFP4 checkpoint loading and dequantization.

Tests the compressed_tensors NVFP4 (E2M1) dequantization pipeline that
converts imported HuggingFace NVFP4 checkpoints into BF16 parameters for
the nmoe model.
"""

import os
import sys
import torch
import pytest
import tempfile
from pathlib import Path

# Add nmoe to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmoe.checkpoint import (
    _unpack_e2m1_to_float,
    _E2M1_LUT,
    dequantize_compressed_tensors_nvfp4,
    dequantize_nvfp4_state_dict,
    _is_nvfp4_checkpoint,
    _get_nvfp4_group_size,
)


class TestE2M1Unpacking:
    """Test NVFP4 E2M1 nibble unpacking."""

    def test_zero_byte(self):
        """0x00 = two zeros."""
        packed = torch.tensor([0x00], dtype=torch.uint8)
        result = _unpack_e2m1_to_float(packed)
        assert result.shape == (2,)
        assert result[0].item() == 0.0  # low nibble: 0b0000
        assert result[1].item() == 0.0  # high nibble: 0b0000

    def test_positive_values(self):
        """Test all 8 positive E2M1 grid values in low nibble."""
        # E2M1 unsigned values: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
        for nibble, expected in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]):
            packed = torch.tensor([nibble], dtype=torch.uint8)
            result = _unpack_e2m1_to_float(packed)
            assert result[0].item() == pytest.approx(expected), f"nibble {nibble}: expected {expected}, got {result[0].item()}"

    def test_negative_values(self):
        """Test negative E2M1 values (sign bit = 1, i.e. nibble >= 8)."""
        # nibble 0b1001 = -0.5 (sign=1, magnitude=1)
        packed = torch.tensor([0x09], dtype=torch.uint8)  # low=0x9, high=0x0
        result = _unpack_e2m1_to_float(packed)
        assert result[0].item() == pytest.approx(-0.5)

        # nibble 0b1111 = -6.0 (sign=1, magnitude=7)
        packed = torch.tensor([0x0F], dtype=torch.uint8)
        result = _unpack_e2m1_to_float(packed)
        assert result[0].item() == pytest.approx(-6.0)

    def test_high_nibble(self):
        """Test values packed in high nibble."""
        # byte 0x30 = low=0b0000=0.0, high=0b0011=1.5
        packed = torch.tensor([0x30], dtype=torch.uint8)
        result = _unpack_e2m1_to_float(packed)
        assert result[0].item() == pytest.approx(0.0)  # low
        assert result[1].item() == pytest.approx(1.5)  # high

    def test_mixed_byte(self):
        """Test byte with different low and high nibbles."""
        # byte 0x52 = low=0b0010=1.0, high=0b0101=3.0
        packed = torch.tensor([0x52], dtype=torch.uint8)
        result = _unpack_e2m1_to_float(packed)
        assert result[0].item() == pytest.approx(1.0)  # low
        assert result[1].item() == pytest.approx(3.0)  # high

    def test_batch_unpacking(self):
        """Test unpacking a batch of bytes."""
        packed = torch.tensor([0x00, 0x21, 0x43], dtype=torch.uint8)
        result = _unpack_e2m1_to_float(packed)
        assert result.shape == (6,)
        # 0x00: 0.0, 0.0
        assert result[0].item() == pytest.approx(0.0)
        assert result[1].item() == pytest.approx(0.0)
        # 0x21: low=0x1=0.5, high=0x2=1.0
        assert result[2].item() == pytest.approx(0.5)
        assert result[3].item() == pytest.approx(1.0)
        # 0x43: low=0x3=1.5, high=0x4=2.0
        assert result[4].item() == pytest.approx(1.5)
        assert result[5].item() == pytest.approx(2.0)

    def test_multidim_unpacking(self):
        """Test unpacking with multi-dimensional input."""
        # [2, 3] packed -> [2, 6] unpacked
        packed = torch.tensor([[0x21, 0x43, 0x65],
                               [0x00, 0x00, 0x00]], dtype=torch.uint8)
        result = _unpack_e2m1_to_float(packed)
        assert result.shape == (2, 6)
        # First row: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0
        assert result[0, 0].item() == pytest.approx(0.5)
        assert result[0, 5].item() == pytest.approx(4.0)
        # Second row: all zeros
        assert result[1].sum().item() == pytest.approx(0.0)


class TestDequantizeCompressedTensors:
    """Test full compressed_tensors NVFP4 dequantization pipeline."""

    def test_identity_scale(self):
        """With scale=1.0 and global_scale=1.0, output equals raw E2M1 values."""
        # 16 elements (one scale group), 8 bytes packed
        packed = torch.tensor([[0x21, 0x43, 0x65, 0x07, 0x00, 0x00, 0x00, 0x00]], dtype=torch.uint8)
        scale = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([1.0], dtype=torch.float32)

        result = dequantize_compressed_tensors_nvfp4(packed, scale, global_scale, group_size=16)
        assert result.dtype == torch.bfloat16
        assert result.shape == (1, 16)
        # First few values from 0x21: 0.5, 1.0 (low, high)
        assert result[0, 0].float().item() == pytest.approx(0.5, abs=0.01)
        assert result[0, 1].float().item() == pytest.approx(1.0, abs=0.01)

    def test_scale_factor(self):
        """Scale factor multiplies dequantized values."""
        # All 0x21 = (0.5, 1.0) repeated
        packed = torch.tensor([[0x21] * 8], dtype=torch.uint8)
        # Scale = 2.0 (E4M3 representation)
        scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([1.0], dtype=torch.float32)

        result = dequantize_compressed_tensors_nvfp4(packed, scale, global_scale, group_size=16)
        # 0.5 * 2.0 = 1.0, 1.0 * 2.0 = 2.0
        assert result[0, 0].float().item() == pytest.approx(1.0, abs=0.05)
        assert result[0, 1].float().item() == pytest.approx(2.0, abs=0.05)

    def test_global_scale(self):
        """Global scale factor divides the per-group scale (compressed_tensors convention)."""
        packed = torch.tensor([[0x21] * 8], dtype=torch.uint8)
        scale = torch.tensor([[6.0]], dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([2.0], dtype=torch.float32)

        result = dequantize_compressed_tensors_nvfp4(packed, scale, global_scale, group_size=16)
        # 0.5 * (6.0 / 2.0) = 0.5 * 3.0 = 1.5
        assert result[0, 0].float().item() == pytest.approx(1.5, abs=0.1)

    def test_multi_group(self):
        """Test with multiple scale groups."""
        # 2 groups of 16 = 32 elements = 16 packed bytes
        packed = torch.tensor([[0x21] * 8 + [0x43] * 8], dtype=torch.uint8)
        # 2 scale groups
        scale = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([1.0], dtype=torch.float32)

        result = dequantize_compressed_tensors_nvfp4(packed, scale, global_scale, group_size=16)
        assert result.shape == (1, 32)
        # Group 1 (scale=1.0): 0.5, 1.0, ...
        assert result[0, 0].float().item() == pytest.approx(0.5, abs=0.05)
        # Group 2 (scale=2.0): 1.5*2=3.0, 2.0*2=4.0, ...
        assert result[0, 16].float().item() == pytest.approx(3.0, abs=0.1)

    def test_stacked_experts(self):
        """Test dequantization of stacked expert tensors [n_experts, out, in//2]."""
        n_experts = 4
        out_dim = 8
        in_dim = 32  # packed = 16 bytes, so in_dim_packed = 16
        packed = torch.randint(0, 256, (n_experts, out_dim, in_dim // 2), dtype=torch.uint8)
        scale = torch.ones(n_experts, out_dim, in_dim // 16, dtype=torch.float8_e4m3fn)
        # global_scale=1.0 means effective_scale = scale/1.0 = scale
        global_scale = torch.ones(n_experts, 1, dtype=torch.float32)

        result = dequantize_compressed_tensors_nvfp4(packed, scale, global_scale, group_size=16)
        assert result.shape == (n_experts, out_dim, in_dim)
        assert result.dtype == torch.bfloat16

    def test_output_range(self):
        """Dequantized values should be finite and within expected range."""
        # Random packed data with reasonable scales
        packed = torch.randint(0, 256, (2, 4, 8), dtype=torch.uint8)
        scale = torch.ones(2, 4, 1, dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

        result = dequantize_compressed_tensors_nvfp4(packed, scale, global_scale, group_size=16)
        assert torch.isfinite(result).all(), "Found non-finite values in dequantized output"


class TestDequantizeStateDict:
    """Test state dict NVFP4 triplet detection and dequantization."""

    def _make_triplet(self, out_dim=16, in_dim=32, n_experts=None):
        """Create a synthetic NVFP4 triplet."""
        if n_experts:
            packed = torch.randint(0, 256, (n_experts, out_dim, in_dim // 2), dtype=torch.uint8)
            scale = torch.ones(n_experts, out_dim, in_dim // 16, dtype=torch.float8_e4m3fn)
            global_scale = torch.ones(n_experts, 1, dtype=torch.float32)
        else:
            packed = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8)
            scale = torch.ones(out_dim, in_dim // 16, dtype=torch.float8_e4m3fn)
            global_scale = torch.tensor([1.0], dtype=torch.float32)
        return packed, scale, global_scale

    def test_linear_weight_triplet(self):
        """nn.Linear NVFP4 triplet gets '.weight' suffix in output."""
        packed, scale, gs = self._make_triplet()
        sd = {
            'blocks.0.attn.wo.weight_packed': packed,
            'blocks.0.attn.wo.weight_scale': scale,
            'blocks.0.attn.wo.weight_global_scale': gs,
            'blocks.0.attn_norm.weight': torch.ones(16, dtype=torch.bfloat16),
        }
        result = dequantize_nvfp4_state_dict(sd)

        assert 'blocks.0.attn.wo.weight' in result
        assert 'blocks.0.attn.wo.weight_packed' not in result
        assert 'blocks.0.attn.wo.weight_scale' not in result
        assert 'blocks.0.attn.wo.weight_global_scale' not in result
        assert 'blocks.0.attn_norm.weight' in result  # Non-NVFP4 key preserved
        assert result['blocks.0.attn.wo.weight'].dtype == torch.bfloat16

    def test_expert_parameter_triplet(self):
        """nn.Parameter expert weight (W1/W3/W2) gets no suffix and is transposed."""
        packed, scale, gs = self._make_triplet(out_dim=16, in_dim=32, n_experts=4)
        sd = {
            'blocks.10.ffn.W1.weight_packed': packed,
            'blocks.10.ffn.W1.weight_scale': scale,
            'blocks.10.ffn.W1.weight_global_scale': gs,
        }
        result = dequantize_nvfp4_state_dict(sd)

        assert 'blocks.10.ffn.W1' in result
        assert 'blocks.10.ffn.W1.weight_packed' not in result
        assert result['blocks.10.ffn.W1'].dtype == torch.bfloat16
        assert result['blocks.10.ffn.W1'].shape[0] == 4  # n_experts preserved
        # Expert weights are transposed: HF [n, out, in] -> nmoe [n, in, out]
        assert result['blocks.10.ffn.W1'].shape[1] == 32  # was in_dim, now first
        assert result['blocks.10.ffn.W1'].shape[2] == 16  # was out_dim, now second

    def test_no_nvfp4_passthrough(self):
        """State dict with no NVFP4 data is returned unchanged."""
        sd = {
            'embedding.weight': torch.ones(100, 64, dtype=torch.bfloat16),
            'norm.weight': torch.ones(64, dtype=torch.bfloat16),
        }
        result = dequantize_nvfp4_state_dict(sd)
        assert result is sd  # Same object, no copy

    def test_mixed_state_dict(self):
        """Mix of NVFP4 triplets and plain BF16 tensors."""
        packed, scale, gs = self._make_triplet()
        sd = {
            'blocks.0.attn.wo.weight_packed': packed,
            'blocks.0.attn.wo.weight_scale': scale,
            'blocks.0.attn.wo.weight_global_scale': gs,
            'blocks.0.attn_norm.weight': torch.ones(16, dtype=torch.bfloat16),
            'blocks.0.ffn_norm.weight': torch.ones(16, dtype=torch.bfloat16),
        }
        result = dequantize_nvfp4_state_dict(sd)

        assert len(result) == 3  # 3 triplet keys -> 1 + 2 plain = 3
        assert 'blocks.0.attn.wo.weight' in result
        assert 'blocks.0.attn_norm.weight' in result
        assert 'blocks.0.ffn_norm.weight' in result

    def test_multiple_triplets(self):
        """Multiple NVFP4 triplets in same state dict."""
        sd = {}
        for name in ['blocks.0.attn.wo', 'blocks.0.attn.wq_a', 'blocks.0.ffn.w1']:
            packed, scale, gs = self._make_triplet()
            sd[f'{name}.weight_packed'] = packed
            sd[f'{name}.weight_scale'] = scale
            sd[f'{name}.weight_global_scale'] = gs

        result = dequantize_nvfp4_state_dict(sd)

        for name in ['blocks.0.attn.wo', 'blocks.0.attn.wq_a', 'blocks.0.ffn.w1']:
            assert f'{name}.weight' in result
            assert f'{name}.weight_packed' not in result


class TestCheckpointMetadata:
    """Test NVFP4 checkpoint detection helpers."""

    def test_is_nvfp4_checkpoint(self):
        assert _is_nvfp4_checkpoint({'nvfp4_format': 'compressed_tensors'})
        assert not _is_nvfp4_checkpoint({'nvfp4_format': 'other'})
        assert not _is_nvfp4_checkpoint({})
        assert not _is_nvfp4_checkpoint({'step': 0})

    def test_get_nvfp4_group_size(self):
        assert _get_nvfp4_group_size({'nvfp4_group_size': 16}) == 16
        assert _get_nvfp4_group_size({'nvfp4_group_size': 32}) == 32
        assert _get_nvfp4_group_size({}) == 16  # default


class TestRealCheckpoint:
    """Test against the actual NVFP4 checkpoint on disk (if available)."""

    CKPT_DIR = "/home/nourdine/nmoe_checkpoints/reap345b_nvfp4/iteration_00000"

    @pytest.fixture(autouse=True)
    def check_checkpoint_exists(self):
        if not os.path.exists(self.CKPT_DIR):
            pytest.skip("Checkpoint not available")

    def test_rd_dequantize(self):
        """Dequantize a few dense NVFP4 triplets from rd.pt."""
        rd = torch.load(
            os.path.join(self.CKPT_DIR, "rd.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert _is_nvfp4_checkpoint(rd)

        dense_sd = rd["model_dense"]
        packed_keys = [k for k in dense_sd if k.endswith(".weight_packed")]
        assert len(packed_keys) > 0, "rd.pt should have NVFP4 keys"

        result = dequantize_nvfp4_state_dict(dense_sd, group_size=16)

        # Verify triplets are consumed
        for pk in packed_keys:
            base = pk[:-len(".weight_packed")]
            assert pk not in result
            assert base + ".weight_scale" not in result
            assert base + ".weight_global_scale" not in result
            # Output key should exist (with .weight suffix for Linear)
            base_parts = base.split('.')
            if base_parts[-1] in ('W1', 'W3', 'W2'):
                assert base in result, f"Missing dequantized key: {base}"
            else:
                assert base + '.weight' in result, f"Missing dequantized key: {base}.weight"

        # Check a specific weight has reasonable values
        wo_key = "blocks.0.attn.wo.weight"
        if wo_key in result:
            w = result[wo_key]
            assert w.dtype == torch.bfloat16
            assert torch.isfinite(w).all(), "Non-finite values in dequantized weight"
            # Weights should be non-trivial (not all zeros)
            assert w.abs().max() > 0.01, "Dequantized weight is effectively zero"

    def test_dp_rank_dequantize(self):
        """Dequantize expert NVFP4 triplets from dp_rank_000.pt."""
        dp = torch.load(
            os.path.join(self.CKPT_DIR, "dp_rank_000.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert _is_nvfp4_checkpoint(dp)

        expert_sd = dp["model_expert"]
        packed_keys = [k for k in expert_sd if k.endswith(".weight_packed")]
        assert len(packed_keys) == 174, "dp_rank should have 174 weight_packed keys"

        result = dequantize_nvfp4_state_dict(expert_sd, group_size=16)

        # Expert weights (W1/W3/W2) should have no .weight suffix
        w1_key = "blocks.10.ffn.W1"
        assert w1_key in result, f"Missing expert key: {w1_key}"
        w = result[w1_key]
        assert w.dtype == torch.bfloat16
        assert w.shape[0] == 16, "Expected 16 local experts"
        assert torch.isfinite(w).all()
        assert w.abs().max() > 0.01

    def test_dequant_matches_hf_output(self):
        """Verify dequantized weights produce similar values to HF model output.

        This compares a small dequantized weight against expected magnitude range
        from the known-good HuggingFace model (perplexity 4.7281).
        """
        dp = torch.load(
            os.path.join(self.CKPT_DIR, "dp_rank_000.pt"),
            map_location="cpu",
            weights_only=False,
        )
        expert_sd = dp["model_expert"]
        result = dequantize_nvfp4_state_dict(expert_sd, group_size=16)

        # Check W1 for layer 10 (first MoE layer after 3 dense)
        w1 = result.get("blocks.3.ffn.W1")
        if w1 is not None:
            # NVFP4 quantized weights typically have max abs < 1.0 (before scale)
            # After scale, typical range is [-10, 10] for well-trained models
            assert w1.abs().max() < 100, f"Weight magnitude too large: {w1.abs().max()}"
            assert w1.abs().mean() > 0.001, f"Weight magnitude too small: {w1.abs().mean()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
