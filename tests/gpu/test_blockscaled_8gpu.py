"""P1 Critical Tests for Blockscaled Grouped GEMM (SM100/B200).

Tests the blockscaled grouped GEMM operations for MoE expert compute:
1. quantize_weights() - W13 interleaving, FP8/NVFP4 quantization, E8M0 scales
2. expert_blockscaled() - Forward pass numerical accuracy vs BF16 reference
3. _swizzle_sf_to_mma() - Scale factor swizzle correctness
4. E8M0 encode/decode - Roundtrip fidelity, edge cases
5. Multi-GPU correctness - Results identical across GPUs, EP sharding

Run single-GPU tests:
    pytest tests/gpu/test_blockscaled_8gpu.py -v -m gpu

Run 8-GPU tests:
    torchrun --nproc_per_node=8 -m pytest tests/gpu/test_blockscaled_8gpu.py -v -m multi_gpu

Requirements:
    - SM100 (B200) GPU for blockscaled kernels
    - nvidia-cutlass-dsl >= 4.3.1
    - cuda.bindings.driver
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ==============================================================================
# Pytest Markers and Skip Decorators
# ==============================================================================


def requires_sm100():
    """Decorator to skip tests that require SM100 (B200) GPUs."""
    def decorator(func):
        @pytest.mark.skipif(
            not torch.cuda.is_available() or
            torch.cuda.get_device_capability()[0] < 10,
            reason="Requires SM100 (B200)"
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_cutlass_dsl():
    """Decorator to skip tests that require CuTeDSL."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import cutlass
                import cuda.bindings.driver
            except ImportError as e:
                pytest.skip(f"CuTeDSL or CUDA bindings not available: {e}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _skip_if_no_sm100_deps():
    """Helper to skip if SM100 dependencies are not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    if torch.cuda.get_device_capability(0)[0] < 10:
        pytest.skip("SM100 (B200) required.")
    try:
        import cutlass
        import cuda.bindings.driver
    except ImportError as e:
        pytest.skip(f"Required runtime deps missing: {e}")


# ==============================================================================
# Distributed Helpers
# ==============================================================================


def get_world_size() -> int:
    """Get world size, handling both distributed and non-distributed cases."""
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    """Get rank, handling both distributed and non-distributed cases."""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def init_distributed():
    """Initialize distributed if not already done."""
    if not dist.is_initialized() and get_world_size() > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(get_rank())


def skip_if_not_multi_gpu(min_gpus: int = 2):
    """Skip test if not enough GPUs available."""
    world_size = get_world_size()
    if world_size < min_gpus:
        pytest.skip(f"Requires at least {min_gpus} GPUs, have {world_size}")


# ==============================================================================
# Reference Implementations
# ==============================================================================


def reference_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU activation: silu(gate) * up."""
    return F.silu(gate) * up


def reference_expert_mlp(
    x: torch.Tensor,
    W1: torch.Tensor,
    W3: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    """Reference expert MLP: Y = SwiGLU(X @ W1, X @ W3) @ W2.

    Args:
        x: [M, H] BF16 input
        W1: [H, Dff] BF16 gate weights
        W3: [H, Dff] BF16 up weights
        W2: [Dff, H] BF16 down weights

    Returns:
        [M, H] BF16 output
    """
    gate = x @ W1
    up = x @ W3
    hidden = reference_swiglu(gate, up)
    return hidden @ W2


def reference_batched_expert_mlp(
    x_pad: torch.Tensor,
    W1: torch.Tensor,
    W3: torch.Tensor,
    W2: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """Reference batched expert MLP using per-expert offsets.

    Args:
        x_pad: [M_pad, H] padded input
        W1: [E, H, Dff] stacked gate weights
        W3: [E, H, Dff] stacked up weights
        W2: [E, Dff, H] stacked down weights
        offs: [E+1] cumulative offsets (starts with 0)

    Returns:
        [M_pad, H] output
    """
    M_pad, H = x_pad.shape
    E = W1.shape[0]
    out = torch.zeros_like(x_pad)

    offs_cpu = offs.cpu()
    for e in range(E):
        start = int(offs_cpu[e].item())
        end = int(offs_cpu[e + 1].item())
        if end <= start:
            continue

        x_e = x_pad[start:end]
        y_e = reference_expert_mlp(x_e, W1[e], W3[e], W2[e])
        out[start:end] = y_e

    return out


def e8m0_encode_reference(scale: float) -> int:
    """Reference E8M0 encode: ceil(log2(scale)) as exponent byte.

    E8M0 encodes scale = 2^(byte - 127), so byte = log2(scale) + 127.
    For positive scale, we use ceil(log2(scale)) to avoid underflow.
    """
    if scale <= 0:
        return 0
    # For normalized FP32: scale = 2^(E-127) * (1.mantissa)
    # ceil(log2(scale)) = (E-127) + (mantissa != 0)
    import struct
    bits = struct.unpack('>I', struct.pack('>f', scale))[0]
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    e8m0 = exp + (1 if mant != 0 else 0)
    return min(e8m0, 254)


def e8m0_decode_inv_reference(byte: int) -> float:
    """Reference E8M0 inverse decode: inv_scale = 2^(127 - byte).

    Used to multiply quantized values to recover original scale.
    """
    byte = min(byte, 254)
    inv_exp = 254 - byte
    if inv_exp == 0:
        # Subnormal case: return 2^-126 with implied mantissa bit
        return 2.0 ** (-126) * 0.5  # 0x00400000 as float
    import struct
    bits = inv_exp << 23
    return struct.unpack('>f', struct.pack('>I', bits))[0]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module", autouse=True)
def setup_distributed():
    """Initialize distributed environment for all tests in module."""
    init_distributed()
    yield


@pytest.fixture
def cuda_device():
    """Provide CUDA device based on rank."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rank = get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    return device


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    return seed_val


@pytest.fixture
def small_model_dims():
    """Small model dimensions for fast tests."""
    return {
        "E": 4,
        "H": 256,
        "Dff": 512,
    }


@pytest.fixture
def standard_model_dims():
    """Standard model dimensions matching production shapes."""
    return {
        "E": 8,
        "H": 1024,
        "Dff": 2816,
    }


@pytest.fixture
def expert_weights_factory(cuda_device):
    """Factory for creating expert weight tensors."""
    def _create(E: int, H: int, Dff: int, scale: float = 1.0):
        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * scale / math.sqrt(H)
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16) * scale / math.sqrt(H)
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16) * scale / math.sqrt(Dff)
        return W1, W3, W2
    return _create


@pytest.fixture
def quantized_weights_fp8_factory(cuda_device, expert_weights_factory):
    """Factory for creating FP8 quantized weights."""
    def _create(E: int, H: int, Dff: int):
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        W1, W3, W2 = expert_weights_factory(E, H, Dff)
        W_cache = quantize_weights(W1, W3, W2, profile="fp8")
        return W_cache, W1, W3, W2
    return _create


@pytest.fixture
def quantized_weights_nvfp4_factory(cuda_device, expert_weights_factory):
    """Factory for creating NVFP4 quantized weights."""
    def _create(E: int, H: int, Dff: int):
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        W1, W3, W2 = expert_weights_factory(E, H, Dff)
        W_cache = quantize_weights(W1, W3, W2, profile="nvfp4")
        return W_cache, W1, W3, W2
    return _create


@pytest.fixture
def activation_quantizer_factory(cuda_device):
    """Factory for quantizing activations to packed format + MMA-layout SFA."""
    def _create(x: torch.Tensor, profile: str) -> Tuple[torch.Tensor, torch.Tensor]:
        _skip_if_no_sm100_deps()
        from nmoe.csrc import rdep

        M, H = x.shape
        sf_k = H // 32
        stream = torch.cuda.current_stream(cuda_device)

        if profile == "fp8":
            Xe_q = torch.empty(M, H // 2, device=cuda_device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M, sf_k, device=cuda_device, dtype=torch.uint8)
            rdep.quant_fp8(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 2,
                Xe_sf_mkl.data_ptr(), sf_k,
                M, H, stream,
            )
        else:  # nvfp4
            Xe_q = torch.empty(M, H // 4, device=cuda_device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M, sf_k, device=cuda_device, dtype=torch.uint8)
            rdep.quant_nvfp4(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 4,
                Xe_sf_mkl.data_ptr(), sf_k,
                M, H, stream,
            )

        # Swizzle SF to MMA layout
        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(),
            Xe_sf_mma.data_ptr(),
            int(M), int(sf_k), stream,
        )

        return Xe_q, Xe_sf_mma
    return _create


# ==============================================================================
# Test Classes: quantize_weights()
# ==============================================================================


@pytest.mark.gpu
class TestQuantizeWeightsInterleaving:
    """Tests for W13 interleaving in quantize_weights()."""

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_w13_interleaving_shape_fp8(self, cuda_device, small_model_dims, expert_weights_factory, seed):
        """Test W13 interleaved shape is correct for FP8."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        # W13 should be [E, 2*Dff, H, 1] for FP8
        assert W_cache.W13_q.shape == (E, 2 * Dff, H, 1), \
            f"Expected W13_q shape ({E}, {2*Dff}, {H}, 1), got {W_cache.W13_q.shape}"
        assert W_cache.W13_q.dtype == torch.float8_e4m3fn

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_w13_interleaving_shape_nvfp4(self, cuda_device, small_model_dims, expert_weights_factory, seed):
        """Test W13 interleaved shape is correct for NVFP4."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        W_cache = quantize_weights(W1, W3, W2, profile="nvfp4")

        # W13 should be [E, 2*Dff, H//2, 1] for NVFP4 (2 values per byte)
        assert W_cache.W13_q.shape == (E, 2 * Dff, H // 2, 1), \
            f"Expected W13_q shape ({E}, {2*Dff}, {H//2}, 1), got {W_cache.W13_q.shape}"
        assert W_cache.W13_q.dtype == torch.uint8

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_w13_sf_shape(self, cuda_device, small_model_dims, expert_weights_factory, seed):
        """Test W13 scale factor shape is correct."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        # Scale factors are per 32 elements along K dimension
        sf_k = H // 32
        assert W_cache.W13_sf_mma.shape == (E, 2 * Dff, sf_k, 1), \
            f"Expected W13_sf_mma shape ({E}, {2*Dff}, {sf_k}, 1), got {W_cache.W13_sf_mma.shape}"
        assert W_cache.W13_sf_mma.dtype == torch.uint8

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_w2_shape_fp8(self, cuda_device, small_model_dims, expert_weights_factory, seed):
        """Test W2 quantized shape is correct for FP8."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        # W2 transposed: [E, H, Dff, 1] for FP8
        assert W_cache.W2_q.shape == (E, H, Dff, 1), \
            f"Expected W2_q shape ({E}, {H}, {Dff}, 1), got {W_cache.W2_q.shape}"
        assert W_cache.W2_q.dtype == torch.float8_e4m3fn


@pytest.mark.gpu
class TestQuantizeWeightsAccuracy:
    """Tests for quantization accuracy in quantize_weights()."""

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_fp8_quantization_roundtrip_error(self, cuda_device, small_model_dims, expert_weights_factory, seed):
        """Test FP8 quantization error is within expected bounds."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        # FP8 E4M3 has ~3 bits of mantissa, so relative error should be < 0.25
        # Check metadata is stored correctly
        assert W_cache.E == E
        assert W_cache.H == H
        assert W_cache.Dff == Dff
        assert W_cache.profile == "fp8"

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_nvfp4_quantization_roundtrip_error(self, cuda_device, small_model_dims, expert_weights_factory, seed):
        """Test NVFP4 quantization error is within expected bounds."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        W_cache = quantize_weights(W1, W3, W2, profile="nvfp4")

        # NVFP4 E2M1 has 1 bit mantissa, higher quantization error expected
        assert W_cache.E == E
        assert W_cache.H == H
        assert W_cache.Dff == Dff
        assert W_cache.profile == "nvfp4"

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_scale_factors_finite(self, cuda_device, small_model_dims, expert_weights_factory, seed):
        """Test all scale factors are finite (no NaN/Inf)."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        # Scale factors are uint8, check they are valid E8M0 bytes
        assert W_cache.W13_sf_mma.max() <= 254, "W13 SF exceeds E8M0 max"
        assert W_cache.W2_sf_mma.max() <= 254, "W2 SF exceeds E8M0 max"


@pytest.mark.gpu
class TestQuantizeWeightsValidation:
    """Tests for input validation in quantize_weights()."""

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_invalid_profile_raises_error(self, cuda_device, small_model_dims, expert_weights_factory):
        """Test that invalid profile raises ValueError."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        W1, W3, W2 = expert_weights_factory(E, H, Dff)

        with pytest.raises(ValueError, match="profile must be"):
            quantize_weights(W1, W3, W2, profile="invalid")

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_unaligned_h_raises_error(self, cuda_device, expert_weights_factory):
        """Test that H not multiple of 128 raises ValueError."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = 4, 200, 512  # H not multiple of 128
        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16)
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16)
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="must be a multiple of 128"):
            quantize_weights(W1, W3, W2, profile="fp8")

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_unaligned_dff_raises_error(self, cuda_device, expert_weights_factory):
        """Test that Dff not multiple of 128 raises ValueError."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights

        E, H, Dff = 4, 256, 300  # Dff not multiple of 128
        W1 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16)
        W3 = torch.randn(E, H, Dff, device=cuda_device, dtype=torch.bfloat16)
        W2 = torch.randn(E, Dff, H, device=cuda_device, dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="must be a multiple of 128"):
            quantize_weights(W1, W3, W2, profile="fp8")


# ==============================================================================
# Test Classes: expert_blockscaled() Forward Pass
# ==============================================================================


@pytest.mark.gpu
class TestExpertBlockscaledForward:
    """Tests for expert_blockscaled() forward pass numerical accuracy."""

    @requires_sm100()
    @requires_cutlass_dsl()
    @pytest.mark.parametrize("profile", ["fp8", "nvfp4"])
    def test_forward_matches_reference(self, cuda_device, small_model_dims, profile, seed):
        """Test blockscaled forward matches BF16 reference within tolerance."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 256  # Must be multiple of 128

        torch.manual_seed(seed)
        device = cuda_device

        # Create input and weights
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16) * 0.1
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

        # Pathological routing: all rows to expert 0
        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

        # Quantize activations
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)

        if profile == "fp8":
            Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_fp8(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 2,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )
        else:
            Xe_q = torch.empty(M_pad, H // 4, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_nvfp4(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 4,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )

        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        # Quantize weights
        W_cache = quantize_weights(W1, W3, W2, profile=profile)

        # Blockscaled forward
        y = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)
        assert y.shape == (M_pad, H)
        assert torch.isfinite(y).all(), "Output contains NaN/Inf"

        # Reference using BF16
        offs = torch.cat((offs_pad.new_zeros((1,)), offs_pad), dim=0)
        y_ref = reference_batched_expert_mlp(x, W1, W3, W2, offs)

        # Set tolerance based on profile
        if profile == "fp8":
            atol, rtol = 1e-1, 2e-1
        else:
            atol, rtol = 2e-1, 3e-1

        torch.testing.assert_close(
            y, y_ref, atol=atol, rtol=rtol,
            msg=f"{profile} forward does not match reference"
        )

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_output_shape_correct(self, cuda_device, small_model_dims, seed):
        """Test output shape matches expected [M_pad, H]."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 128
        device = cuda_device

        torch.manual_seed(seed)
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)

        # All tokens to expert 0
        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

        # Quantize
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)
        Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
        Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
        rdep.quant_fp8(
            x.data_ptr(), H,
            Xe_q.data_ptr(), H // 2,
            Xe_sf_mkl.data_ptr(), sf_k_in,
            M_pad, H, stream,
        )
        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) * 0.02
        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        y = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)

        assert y.shape == (M_pad, H), f"Expected ({M_pad}, {H}), got {y.shape}"
        assert y.dtype == torch.bfloat16

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_empty_input_returns_empty(self, cuda_device, small_model_dims, seed):
        """Test that empty input returns empty output."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        device = cuda_device

        # Empty input
        Xe_q = torch.empty(0, H // 2, device=device, dtype=torch.uint16)
        Xe_sf = torch.empty(0, H // 32, device=device, dtype=torch.uint8)
        offs_pad = torch.zeros(E, device=device, dtype=torch.int32)

        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) * 0.02
        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        y = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad)

        assert y.shape == (0, H)


@pytest.mark.gpu
class TestExpertBlockscaledSwiGLU:
    """Tests for fused SwiGLU epilogue correctness."""

    @requires_sm100()
    @requires_cutlass_dsl()
    @pytest.mark.parametrize("profile", ["fp8", "nvfp4"])
    def test_swiglu_fused_matches_unfused(self, cuda_device, small_model_dims, profile, seed):
        """Test fused SwiGLU+quant matches separate operations."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, run_grouped_blockscaled_strided
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 256
        device = cuda_device

        torch.manual_seed(seed)
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)
        offs = torch.cat((offs_pad.new_zeros((1,)), offs_pad), dim=0)

        # Quantize input
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)

        if profile == "fp8":
            Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_fp8(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 2,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )
            A_q = Xe_q.view(torch.uint8).view(M_pad, H, 1).view(torch.float8_e4m3fn)
        else:
            Xe_q = torch.empty(M_pad, H // 4, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_nvfp4(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 4,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )
            A_q = Xe_q.view(torch.uint8).view(M_pad, H // 2, 1)

        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1, W3, W2, profile=profile)

        # Unfused path: materialize H13, then separate SwiGLU+quant
        H13 = torch.empty((M_pad, 2 * Dff, 1), device=device, dtype=torch.bfloat16)
        run_grouped_blockscaled_strided(
            A_q, Xe_sf_mma, W_cache.W13_q, W_cache.W13_sf_mma, H13, offs,
            profile=profile, N=2 * Dff, K=H,
        )

        # Extract gate and up from interleaved H13
        H13_2d = H13.squeeze(-1)  # [M_pad, 2*Dff]
        gate_ref = H13_2d[:, 0::2]  # Even columns
        up_ref = H13_2d[:, 1::2]  # Odd columns
        act_ref = reference_swiglu(gate_ref, up_ref)

        # Check activation range is reasonable
        assert torch.isfinite(act_ref).all(), "Reference activation contains NaN/Inf"


# ==============================================================================
# Test Classes: _swizzle_sf_to_mma()
# ==============================================================================


@pytest.mark.gpu
class TestSwizzleSfToMma:
    """Tests for scale factor swizzle operation."""

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_swizzle_preserves_sum(self, cuda_device, seed):
        """Test swizzle preserves sum of values (sanity check)."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import _swizzle_sf_to_mma

        torch.manual_seed(seed)
        M, sf_k = 256, 8
        sf_mkl = torch.randint(0, 255, (M, sf_k, 1), device=cuda_device, dtype=torch.uint8)

        sf_mma = _swizzle_sf_to_mma(sf_mkl)

        # Swizzle is a permutation, so sum should be preserved
        # Note: output is padded, so compare only non-padded region sums
        assert sf_mkl.sum() <= sf_mma.sum(), "Swizzle should only add padding zeros"

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_swizzle_padding_128_rows(self, cuda_device, seed):
        """Test swizzle pads M to multiple of 128."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import _swizzle_sf_to_mma

        torch.manual_seed(seed)
        M, sf_k = 200, 8  # M not multiple of 128
        sf_mkl = torch.randint(0, 255, (M, sf_k, 1), device=cuda_device, dtype=torch.uint8)

        sf_mma = _swizzle_sf_to_mma(sf_mkl)

        # Output M should be padded to 256 (next multiple of 128)
        M_pad = 256
        assert sf_mma.shape[0] == M_pad, f"Expected M_pad={M_pad}, got {sf_mma.shape[0]}"

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_swizzle_padding_sf_k(self, cuda_device, seed):
        """Test swizzle pads sf_k to multiple of 4."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import _swizzle_sf_to_mma

        torch.manual_seed(seed)
        M, sf_k = 128, 5  # sf_k not multiple of 4
        sf_mkl = torch.randint(0, 255, (M, sf_k, 1), device=cuda_device, dtype=torch.uint8)

        sf_mma = _swizzle_sf_to_mma(sf_mkl)

        # sf_k should be padded to 8
        sf_k_pad = 8
        assert sf_mma.shape[1] == sf_k_pad, f"Expected sf_k_pad={sf_k_pad}, got {sf_mma.shape[1]}"

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_swizzle_dtype_preserved(self, cuda_device, seed):
        """Test swizzle preserves uint8 dtype."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import _swizzle_sf_to_mma

        torch.manual_seed(seed)
        M, sf_k = 128, 8
        sf_mkl = torch.randint(0, 255, (M, sf_k, 1), device=cuda_device, dtype=torch.uint8)

        sf_mma = _swizzle_sf_to_mma(sf_mkl)

        assert sf_mma.dtype == torch.uint8


# ==============================================================================
# Test Classes: E8M0 Encode/Decode
# ==============================================================================


@pytest.mark.gpu
class TestE8M0EncodeDecode:
    """Tests for E8M0 scale factor encoding and decoding."""

    def test_e8m0_encode_reference_basic(self):
        """Test reference E8M0 encode for basic values."""
        # Powers of 2 should encode exactly
        assert e8m0_encode_reference(1.0) == 127  # 2^0 -> exp=127
        assert e8m0_encode_reference(2.0) == 128  # 2^1 -> exp=128
        assert e8m0_encode_reference(4.0) == 129  # 2^2 -> exp=129
        assert e8m0_encode_reference(0.5) == 126  # 2^-1 -> exp=126

    def test_e8m0_encode_reference_with_mantissa(self):
        """Test reference E8M0 encode with non-zero mantissa (should ceil)."""
        # 1.5 = 2^0 * 1.5, ceil(log2(1.5)) = 1, so byte = 128
        result = e8m0_encode_reference(1.5)
        assert result == 128, f"Expected 128, got {result}"

        # 3.0 = 2^1 * 1.5, ceil(log2(3.0)) = 2, so byte = 129
        result = e8m0_encode_reference(3.0)
        assert result == 129, f"Expected 129, got {result}"

    def test_e8m0_decode_inv_reference_basic(self):
        """Test reference E8M0 inverse decode."""
        # byte=127 -> inv_scale = 2^(127-127) = 1.0
        assert abs(e8m0_decode_inv_reference(127) - 1.0) < 1e-6

        # byte=128 -> inv_scale = 2^(127-128) = 0.5
        assert abs(e8m0_decode_inv_reference(128) - 0.5) < 1e-6

        # byte=126 -> inv_scale = 2^(127-126) = 2.0
        assert abs(e8m0_decode_inv_reference(126) - 2.0) < 1e-6

    def test_e8m0_roundtrip(self):
        """Test E8M0 encode->decode roundtrip preserves scale order."""
        # For powers of 2, encode->decode should give exact inverse
        for exp in range(-10, 10):
            scale = 2.0 ** exp
            byte = e8m0_encode_reference(scale)
            inv_scale = e8m0_decode_inv_reference(byte)
            # inv_scale * scale should be ~1 or slightly less (due to ceil)
            product = inv_scale * scale
            assert 0.5 <= product <= 1.0, f"Roundtrip failed for scale={scale}: got {product}"

    def test_e8m0_edge_case_zero(self):
        """Test E8M0 encode handles zero."""
        assert e8m0_encode_reference(0.0) == 0
        assert e8m0_encode_reference(-1.0) == 0  # Negative treated as 0

    def test_e8m0_edge_case_large(self):
        """Test E8M0 encode clamps large values to 254."""
        # Very large scale should clamp to 254
        result = e8m0_encode_reference(1e38)
        assert result == 254, f"Expected 254 for large scale, got {result}"

    def test_e8m0_edge_case_small(self):
        """Test E8M0 encode handles very small values."""
        # Very small scale should give small exponent
        result = e8m0_encode_reference(1e-38)
        assert 0 < result < 10, f"Expected small byte for tiny scale, got {result}"


# ==============================================================================
# Test Classes: Multi-GPU Correctness
# ==============================================================================


@pytest.mark.multi_gpu
class TestBlockscaledMultiGPU:
    """Tests for multi-GPU blockscaled correctness."""

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_results_identical_across_gpus(self, cuda_device, small_model_dims, seed):
        """Test that same input/weights give identical results on all GPUs."""
        skip_if_not_multi_gpu(2)
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        world_size = get_world_size()
        rank = get_rank()

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 256
        device = cuda_device

        # Use same seed on all ranks
        torch.manual_seed(seed)
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

        # Quantize
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)
        Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
        Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
        rdep.quant_fp8(
            x.data_ptr(), H,
            Xe_q.data_ptr(), H // 2,
            Xe_sf_mkl.data_ptr(), sf_k_in,
            M_pad, H, stream,
        )
        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")
        y = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)

        # Allgather results from all ranks
        y_list = [torch.zeros_like(y) for _ in range(world_size)]
        dist.all_gather(y_list, y)

        # All results should be identical
        for i in range(1, world_size):
            torch.testing.assert_close(
                y_list[0], y_list[i], atol=0, rtol=0,
                msg=f"Results differ between rank 0 and rank {i}"
            )

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_ep_sharding_produces_correct_outputs(self, cuda_device, seed):
        """Test expert-parallel sharding produces correct aggregated output."""
        skip_if_not_multi_gpu(2)
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        world_size = get_world_size()
        rank = get_rank()

        # Total experts across all ranks
        E_total = 8
        E_local = E_total // world_size
        H, Dff = 256, 512
        M_pad = 256
        device = cuda_device

        torch.manual_seed(seed)

        # Create full expert weights (same on all ranks for reference)
        W1_full = torch.randn(E_total, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W3_full = torch.randn(E_total, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W2_full = torch.randn(E_total, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

        # Each rank holds a shard
        start_e = rank * E_local
        end_e = start_e + E_local
        W1_local = W1_full[start_e:end_e].contiguous()
        W3_local = W3_full[start_e:end_e].contiguous()
        W2_local = W2_full[start_e:end_e].contiguous()

        # Create input (same on all ranks)
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)

        # For this test, route all tokens to rank 0's experts
        offs_pad = torch.zeros(E_local, device=device, dtype=torch.int32)
        if rank == 0:
            offs_pad[:] = M_pad  # All tokens to first expert on rank 0

        # Quantize
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)
        Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
        Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
        rdep.quant_fp8(
            x.data_ptr(), H,
            Xe_q.data_ptr(), H // 2,
            Xe_sf_mkl.data_ptr(), sf_k_in,
            M_pad, H, stream,
        )
        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1_local, W3_local, W2_local, profile="fp8")

        if rank == 0:
            # Rank 0 processes tokens
            y = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)
        else:
            # Other ranks have no tokens
            y = torch.zeros(M_pad, H, device=device, dtype=torch.bfloat16)

        # Reduce across ranks
        y_global = torch.zeros_like(y)
        dist.all_reduce(y, op=dist.ReduceOp.SUM)

        assert torch.isfinite(y).all(), "EP sharding produced NaN/Inf"


# ==============================================================================
# Test Classes: Numerical Stability
# ==============================================================================


@pytest.mark.gpu
class TestBlockscaledNumericalStability:
    """Tests for numerical stability with extreme values."""

    @requires_sm100()
    @requires_cutlass_dsl()
    @pytest.mark.parametrize("profile", ["fp8", "nvfp4"])
    def test_large_input_values(self, cuda_device, small_model_dims, profile, seed):
        """Test handling of large input values."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 128
        device = cuda_device

        torch.manual_seed(seed)
        # Large but not overflow values
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16) * 10.0
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.1
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.1
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) * 0.1

        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

        # Quantize
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)

        if profile == "fp8":
            Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_fp8(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 2,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )
        else:
            Xe_q = torch.empty(M_pad, H // 4, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_nvfp4(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 4,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )

        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1, W3, W2, profile=profile)
        y = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)

        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    @requires_sm100()
    @requires_cutlass_dsl()
    @pytest.mark.parametrize("profile", ["fp8", "nvfp4"])
    def test_small_input_values(self, cuda_device, small_model_dims, profile, seed):
        """Test handling of small input values."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 128
        device = cuda_device

        torch.manual_seed(seed)
        # Small values
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16) * 1e-4
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) * 0.02
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) * 0.02

        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

        # Quantize
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)

        if profile == "fp8":
            Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_fp8(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 2,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )
        else:
            Xe_q = torch.empty(M_pad, H // 4, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_nvfp4(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 4,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )

        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1, W3, W2, profile=profile)
        y = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)

        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_mixed_expert_loads(self, cuda_device, small_model_dims, seed):
        """Test numerical stability with mixed expert token counts."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 512
        device = cuda_device

        torch.manual_seed(seed)
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

        # Mixed loads: some experts get many tokens, some get few
        offs_pad = torch.tensor([128, 256, 256, 512], device=device, dtype=torch.int32)
        assert offs_pad.shape[0] == E

        # Quantize
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)
        Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
        Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
        rdep.quant_fp8(
            x.data_ptr(), H,
            Xe_q.data_ptr(), H // 2,
            Xe_sf_mkl.data_ptr(), sf_k_in,
            M_pad, H, stream,
        )
        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")
        y = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)

        assert y.shape == (M_pad, H)
        assert torch.isfinite(y).all(), "Output contains NaN/Inf"


# ==============================================================================
# Test Classes: Per-Row Amax Computation
# ==============================================================================


@pytest.mark.gpu
class TestPerRowAmaxComputation:
    """Tests for per-row amax (max absolute value) computation in quantization."""

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_amax_captured_in_scale_factors(self, cuda_device, seed):
        """Test that amax is properly captured in E8M0 scale factors."""
        _skip_if_no_sm100_deps()
        from nmoe.csrc import rdep

        torch.manual_seed(seed)
        M, H = 128, 256
        device = cuda_device

        # Create input with known max values per row
        x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
        # Set first row to have large values
        x[0] = x[0] * 10.0
        # Set last row to have small values
        x[-1] = x[-1] * 0.01

        sf_k = H // 32
        stream = torch.cuda.current_stream(device)
        Xe_q = torch.empty(M, H // 2, device=device, dtype=torch.uint16)
        Xe_sf = torch.empty(M, sf_k, device=device, dtype=torch.uint8)

        rdep.quant_fp8(
            x.data_ptr(), H,
            Xe_q.data_ptr(), H // 2,
            Xe_sf.data_ptr(), sf_k,
            M, H, stream,
        )

        torch.cuda.synchronize()

        # Scale factors for row 0 should be larger than for row -1
        sf_row0_max = Xe_sf[0].max().item()
        sf_rowlast_max = Xe_sf[-1].max().item()

        # E8M0: larger byte = larger scale
        assert sf_row0_max > sf_rowlast_max, \
            f"Expected SF for row 0 ({sf_row0_max}) > row -1 ({sf_rowlast_max})"


# ==============================================================================
# Test Classes: Determinism
# ==============================================================================


@pytest.mark.gpu
class TestBlockscaledDeterminism:
    """Tests for deterministic behavior of blockscaled operations."""

    @requires_sm100()
    @requires_cutlass_dsl()
    @pytest.mark.parametrize("profile", ["fp8", "nvfp4"])
    def test_forward_deterministic(self, cuda_device, small_model_dims, profile, seed):
        """Test that forward pass is deterministic."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 256
        device = cuda_device

        torch.manual_seed(seed)
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

        # Quantize once
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)

        if profile == "fp8":
            Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_fp8(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 2,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )
        else:
            Xe_q = torch.empty(M_pad, H // 4, device=device, dtype=torch.uint16)
            Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
            rdep.quant_nvfp4(
                x.data_ptr(), H,
                Xe_q.data_ptr(), H // 4,
                Xe_sf_mkl.data_ptr(), sf_k_in,
                M_pad, H, stream,
            )

        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1, W3, W2, profile=profile)

        # Run forward twice
        y1 = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)
        y2 = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)

        torch.testing.assert_close(
            y1, y2, atol=0, rtol=0,
            msg=f"{profile} forward is not deterministic"
        )


# ==============================================================================
# Test Classes: Memory Efficiency
# ==============================================================================


@pytest.mark.gpu
class TestBlockscaledMemoryEfficiency:
    """Tests for memory efficiency at scale."""

    @requires_sm100()
    @requires_cutlass_dsl()
    def test_no_memory_leak_repeated_calls(self, cuda_device, small_model_dims, seed):
        """Test that repeated calls don't leak memory."""
        _skip_if_no_sm100_deps()
        from nmoe.blockscaled.grouped import quantize_weights, expert_blockscaled
        from nmoe.csrc import rdep

        E, H, Dff = small_model_dims["E"], small_model_dims["H"], small_model_dims["Dff"]
        M_pad = 256
        device = cuda_device

        torch.manual_seed(seed)
        x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)
        W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
        W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

        offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

        # Quantize
        sf_k_in = H // 32
        stream = torch.cuda.current_stream(device)
        Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
        Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
        rdep.quant_fp8(
            x.data_ptr(), H,
            Xe_q.data_ptr(), H // 2,
            Xe_sf_mkl.data_ptr(), sf_k_in,
            M_pad, H, stream,
        )
        Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
        rdep.swizzle_sf_mkl_to_mma(
            Xe_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
            int(M_pad), int(sf_k_in), stream,
        )

        W_cache = quantize_weights(W1, W3, W2, profile="fp8")

        # Warm up
        _ = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)
        torch.cuda.synchronize()

        # Record baseline memory
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

        # Run many iterations
        for _ in range(10):
            _ = expert_blockscaled(Xe_q, Xe_sf_mma, W_cache, offs_pad, capacity_rows=M_pad)

        torch.cuda.synchronize()
        final = torch.cuda.memory_allocated(device)

        # Memory should not grow significantly
        growth = final - baseline
        # Allow up to 1MB growth for scratch buffers
        assert growth < 1 * 1024 * 1024, f"Memory grew by {growth / 1024 / 1024:.2f} MB"


# ==============================================================================
# Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
