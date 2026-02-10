"""ECO Optimizer: Error-Compensating Optimizer for NVFP4 primary weights.

Implements Algorithm 3 from "ECO: Quantized Training without Full-Precision
Master Weights" (Nikdan et al., Google Research + ISTA, arXiv:2601.22101v1).

ECO eliminates BF16 master weights by injecting quantization error into Adam's
momentum buffer. The error is computed and consumed in the same step -- no
persistent error buffer needed.

This is the Python prototype. The fused CUDA kernel version (Phase 5b) will
modify adamw.cu with integrated NVFP4 load/store + FP8 m/v + SR + ECO injection.

Key properties:
  - Zero new hyperparameters: eco_alpha = (beta1 - 1) / (beta1 * step_size)
  - Only expert W1/W3/W2 get ECO treatment (dense params stay BF16)
  - FP8 optimizer states: momentum as E5M2, variance as E4M3
  - Stochastic rounding for NVFP4 E2M1 quantization
  - Weight storage: NVFP4 primary (no BF16 master copy)

Memory savings vs BF16 master:
  Per expert parameter: BF16 master (2B) + BF16 m (2B) + BF16 v (2B) = 6 bytes
  ECO:                  NVFP4 primary (0.5B) + FP8 m (1B) + FP8 v (1B) = 2.5 bytes
  Savings: 58% reduction in optimizer memory for expert weights

Usage:
    eco = ECOAdamW(moe_modules, cfg)
    eco.step()  # Performs Adam update + NVFP4 quantization + ECO error injection
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# NVFP4 E2M1 quantization grid
# ============================================================================

# E2M1 representable values (positive side): 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
# With sign bit: 4 bits total = 16 values (8 positive + 8 negative)
NVFP4_MAX = 6.0
NVFP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# FP8 limits
FP8_E5M2_MAX = 57344.0  # Max representable in E5M2
FP8_E4M3_MAX = 448.0    # Max representable in E4M3

# Block size for NVFP4 scale factors (SF_VEC)
SF_VEC = 32


# ============================================================================
# Pure-Python NVFP4 quantization with stochastic rounding
# ============================================================================

def _compute_e8m0_scale(x_block: torch.Tensor) -> torch.Tensor:
    """Compute E8M0 block scale factor.

    Scale = max|x_block| / NVFP4_MAX, stored as E8M0 (power-of-2 only).

    Args:
        x_block: [..., SF_VEC] tensor

    Returns:
        scale: [...] E8M0 scale factor
    """
    amax = x_block.abs().amax(dim=-1)
    # E8M0: round scale to nearest power of 2
    # scale = 2^ceil(log2(amax / NVFP4_MAX))
    raw_scale = amax / NVFP4_MAX
    # Clamp to avoid log2(0) and extremely small scales
    raw_scale = raw_scale.clamp(min=1e-30)
    exponent = torch.ceil(torch.log2(raw_scale))
    scale = torch.pow(2.0, exponent)
    # If block is all zeros, scale should be smallest positive E8M0
    scale = torch.where(amax == 0, torch.ones_like(scale) * (2.0 ** -127), scale)
    return scale


def _quantize_nvfp4_sr(
    x: torch.Tensor,
    scale: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Quantize FP32 values to NVFP4 E2M1 with stochastic rounding.

    Args:
        x: FP32 values to quantize
        scale: E8M0 scale factors (one per SF_VEC block)
        generator: Optional PRNG for reproducibility

    Returns:
        x_hat: Dequantized FP32 values (quantize then dequant back)
    """
    device = x.device
    grid = NVFP4_GRID.to(device)

    # Scale down
    x_scaled = x / scale.unsqueeze(-1)

    # Separate sign and magnitude
    sign = torch.sign(x_scaled)
    ax = x_scaled.abs()

    # Clamp to representable range
    ax = ax.clamp(max=NVFP4_MAX)

    # Find floor and ceil grid indices
    # grid = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    floor_idx = torch.searchsorted(grid, ax, right=True) - 1
    floor_idx = floor_idx.clamp(min=0, max=len(grid) - 2)
    ceil_idx = floor_idx + 1

    floor_val = grid[floor_idx]
    ceil_val = grid[ceil_idx]

    # Stochastic rounding probability
    gap = ceil_val - floor_val
    frac = (ax - floor_val) / gap.clamp(min=1e-10)

    # Random comparison for stochastic rounding decision
    if generator is not None:
        rand = torch.rand(frac.shape, device=frac.device, dtype=frac.dtype, generator=generator)
    else:
        rand = torch.rand(frac.shape, device=frac.device, dtype=frac.dtype)
    use_ceil = rand < frac

    # Select quantized value
    q_val = torch.where(use_ceil, ceil_val, floor_val)

    # Restore sign and scale back up
    x_hat = sign * q_val * scale.unsqueeze(-1)

    return x_hat


def _quantize_nvfp4_rtn(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize FP32 values to NVFP4 E2M1 with round-to-nearest (RTN).

    Used when stochastic rounding is disabled.
    """
    device = x.device
    grid = NVFP4_GRID.to(device)

    x_scaled = x / scale.unsqueeze(-1)
    sign = torch.sign(x_scaled)
    ax = x_scaled.abs().clamp(max=NVFP4_MAX)

    # Find nearest grid point
    idx = torch.bucketize(ax, grid)
    idx = idx.clamp(max=len(grid) - 1)
    # Check if closer to idx or idx-1
    lower = grid[(idx - 1).clamp(min=0)]
    upper = grid[idx]
    use_lower = (ax - lower).abs() < (ax - upper).abs()
    q_val = torch.where(use_lower, lower, upper)

    x_hat = sign * q_val * scale.unsqueeze(-1)
    return x_hat


def quantize_nvfp4_eco(
    x: torch.Tensor,
    stochastic_rounding: bool = True,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to NVFP4 with ECO-compatible interface.

    Args:
        x: [*, K] FP32 tensor, K should be multiple of SF_VEC (32)
        stochastic_rounding: Use SR (True) or RTN (False)
        generator: Optional PRNG for SR reproducibility

    Returns:
        x_hat: Dequantized FP32 values (for error computation)
        scale: E8M0 scale factors
    """
    orig_shape = x.shape
    K = x.shape[-1]

    # Pad K to multiple of SF_VEC if needed
    pad_k = (SF_VEC - K % SF_VEC) % SF_VEC
    if pad_k > 0:
        x = torch.nn.functional.pad(x, (0, pad_k))
        K = x.shape[-1]

    # Reshape to blocks
    x_flat = x.reshape(-1, K)
    M = x_flat.shape[0]
    x_blocks = x_flat.reshape(M, K // SF_VEC, SF_VEC)

    # Compute block scales
    scale = _compute_e8m0_scale(x_blocks)  # [M, K // SF_VEC]

    # Quantize
    if stochastic_rounding:
        x_hat_blocks = _quantize_nvfp4_sr(x_blocks, scale, generator)
    else:
        x_hat_blocks = _quantize_nvfp4_rtn(x_blocks, scale)

    # Reshape back
    x_hat = x_hat_blocks.reshape(M, K)

    # Remove padding
    if pad_k > 0:
        x_hat = x_hat[..., :orig_shape[-1]]

    x_hat = x_hat.reshape(orig_shape)
    return x_hat, scale


# ============================================================================
# NVFP4 packing/unpacking for buffer storage
# ============================================================================

def _pack_nvfp4_e2m1(x_hat: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack dequantized NVFP4 values into compressed_tensors format.

    Args:
        x_hat: Dequantized FP32 values (already quantized to NVFP4 grid)
        scale: E8M0 block scales [*, n_blocks]

    Returns:
        packed: [*, K//2] uint8 (2 E2M1 nibbles per byte)
        scale_fp8: [*, K//group_size] float8_e4m3fn per-group scales
        global_scale: [*, 1] float32
    """
    device = x_hat.device
    grid = NVFP4_GRID.to(device)
    orig_shape = x_hat.shape
    K = orig_shape[-1]

    # Compute per-group (group_size=16) effective scale from block (32) E8M0 scale
    # For Option C, we store as compressed_tensors format:
    #   global_scale = 1.0 (identity, since we use the E8M0 scales directly)
    #   per-group scale = scale expanded to group_size=16
    # Each block of 32 has one E8M0 scale. Group_size=16 means 2 groups per block.
    # Both groups within a block share the same E8M0 scale.
    global_scale = torch.ones(*orig_shape[:-1], 1, dtype=torch.float32, device=device)

    # Convert E8M0 block scale [*, n_blocks] to per-group scale
    # n_blocks = K // 32, n_groups = K // 16 = 2 * n_blocks
    n_blocks = scale.shape[-1]
    # Repeat each block scale for 2 groups (16 elements each = 32 total)
    scale_per_group = scale.unsqueeze(-1).expand(*scale.shape, 2).reshape(*scale.shape[:-1], 2 * n_blocks)
    # Convert to float8_e4m3fn (scale * global_scale, but global_scale=1.0)
    if hasattr(torch, 'float8_e4m3fn'):
        scale_fp8 = scale_per_group.float().to(torch.float8_e4m3fn)
    else:
        scale_fp8 = scale_per_group.float().to(torch.bfloat16)

    # Pack values into uint8 nibble pairs
    # Scale down by block scale, then map to E2M1 index
    x_flat = x_hat.reshape(-1, K)
    M = x_flat.shape[0]
    x_blocks = x_flat.reshape(M, K // SF_VEC, SF_VEC)
    scale_2d = scale.reshape(M, -1)

    # Divide by scale to get normalized values in [-6, 6]
    x_scaled = x_blocks / scale_2d.unsqueeze(-1).clamp(min=1e-30)
    sign = (x_scaled < 0).to(torch.int32)
    ax = x_scaled.abs()

    # Map absolute value to E2M1 index (0-7)
    idx = torch.searchsorted(grid, ax, right=True) - 1
    idx = idx.clamp(min=0, max=7)

    # Combine sign + magnitude into 4-bit nibble
    nibbles = (sign << 3) | idx  # [M, n_blocks, 32]
    nibbles = nibbles.reshape(M, K)

    # Pack pairs of nibbles into uint8
    # Low nibble = even index, high nibble = odd index
    lo = nibbles[..., 0::2].to(torch.uint8)
    hi = nibbles[..., 1::2].to(torch.uint8)
    packed = lo | (hi << 4)
    packed = packed.reshape(*orig_shape[:-1], K // 2)

    # Reshape scale to match compressed_tensors convention
    scale_fp8 = scale_fp8.reshape(*orig_shape[:-1], K // 16)

    return packed, scale_fp8, global_scale


def _dequant_nvfp4_buffers_to_fp32(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 compressed_tensors triplet to FP32.

    Used by ECO optimizer to read current weights.

    Args:
        packed: [*, K//2] uint8
        scale: [*, K//group_size] float8_e4m3fn
        global_scale: [*, 1] float32

    Returns:
        [*, K] float32 tensor
    """
    device = packed.device
    lut = NVFP4_GRID.to(device)

    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)

    lo_sign = ((lo >> 3) & 1).float() * (-2.0) + 1.0
    lo_mag = lo & 0x07
    hi_sign = ((hi >> 3) & 1).float() * (-2.0) + 1.0
    hi_mag = hi & 0x07

    lo_val = lut[lo_mag.long()] * lo_sign
    hi_val = lut[hi_mag.long()] * hi_sign

    out_shape = list(packed.shape)
    out_shape[-1] *= 2
    unpacked = torch.empty(out_shape, dtype=torch.float32, device=device)
    unpacked[..., 0::2] = lo_val
    unpacked[..., 1::2] = hi_val

    scale_f32 = scale.float()
    gs = global_scale.float()
    while gs.ndim < scale_f32.ndim:
        gs = gs.unsqueeze(-1)
    effective_scale = scale_f32 / gs.clamp(min=1e-10)

    n_groups = effective_scale.shape[-1]
    grouped = unpacked.reshape(*unpacked.shape[:-1], n_groups, group_size)
    scaled = grouped * effective_scale.unsqueeze(-1)

    return scaled.reshape(out_shape)


# ============================================================================
# FP8 state quantization helpers
# ============================================================================

def _quantize_to_fp8_e5m2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 tensor to FP8 E5M2 with per-row scale.

    E5M2 has range [-57344, 57344] and is used for momentum (first moment).
    Per-row scaling preserves dynamic range better than per-tensor for
    tensors with varying magnitudes across rows.

    Returns:
        x_fp8: FP8 E5M2 tensor (stored as uint8 for compatibility)
        scale: [nrows] per-row scale factors (FP32)
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.unsqueeze(0)

    amax = x_2d.abs().amax(dim=-1, keepdim=True)  # [nrows, 1]
    scale = amax / FP8_E5M2_MAX
    scale = scale.clamp(min=1e-30)  # Avoid div by zero

    x_scaled = x_2d / scale
    if hasattr(torch, 'float8_e5m2'):
        x_fp8 = x_scaled.to(torch.float8_e5m2)
    else:
        x_fp8 = x_scaled.clamp(-FP8_E5M2_MAX, FP8_E5M2_MAX).to(torch.bfloat16)

    return x_fp8.reshape(orig_shape), scale.reshape(orig_shape[:-1] + (1,))


def _dequantize_fp8_e5m2(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 E5M2 to FP32."""
    return x_fp8.float() * scale


def _quantize_to_fp8_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 tensor to FP8 E4M3 with per-row scale.

    E4M3 has range [-448, 448] and is used for variance (second moment).
    Per-row scaling preserves dynamic range for variance tensors that have
    different magnitudes across different expert/dimension rows.

    Returns:
        x_fp8: FP8 E4M3 tensor
        scale: [nrows] per-row scale factors (FP32)
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.unsqueeze(0)

    amax = x_2d.abs().amax(dim=-1, keepdim=True)
    scale = amax / FP8_E4M3_MAX
    scale = scale.clamp(min=1e-30)  # Avoid div by zero

    x_scaled = x_2d / scale
    if hasattr(torch, 'float8_e4m3fn'):
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    else:
        x_fp8 = x_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.bfloat16)

    return x_fp8.reshape(orig_shape), scale.reshape(orig_shape[:-1] + (1,))


def _dequantize_fp8_e4m3(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 E4M3 to FP32."""
    return x_fp8.float() * scale


# ============================================================================
# ECO AdamW Optimizer
# ============================================================================

class ECOAdamW(torch.optim.Optimizer):
    """ECO-enhanced AdamW for expert weights with NVFP4 primary + FP8 states.

    Implements Algorithm 3 from the ECO paper:
    1. Dequant NVFP4 primary weight -> FP32
    2. Standard AdamW update in FP32 (using dequanted FP8 m/v)
    3. Quantize updated weight to NVFP4 (with stochastic rounding)
    4. Compute quantization error: e = w_updated - w_quantized
    5. Inject error into momentum: m += eco_alpha * denom * e
    6. Requantize m/v to FP8

    The injection strength eco_alpha = (beta1 - 1) / (beta1 * step_size)
    requires NO new hyperparameters -- it's derived from standard Adam params.
    """

    emits_weight_cache = True  # We handle weight cache refresh in step()

    def __init__(
        self,
        moe_modules: list[nn.Module],
        cfg,
        stochastic_rounding: bool = True,
        error_feedback: bool = True,
        factored_v: bool = False,
    ):
        """Initialize ECO optimizer.

        Args:
            moe_modules: List of MoE modules with W1/W3/W2 parameters.
            cfg: Config with lr_expert, adam_beta1, adam_beta2_expert, adam_eps,
                 weight_decay.
            stochastic_rounding: Use SR for NVFP4 quantization (recommended).
            error_feedback: Enable ECO error injection into momentum.
            factored_v: Use Adafactor-style factored second moment (v_row/v_col)
                instead of full FP8 v matrix. Saves ~38 GiB for typical MoE configs.
        """
        params = []
        for moe in moe_modules:
            for name in ("W1", "W3", "W2"):
                p = getattr(moe, name, None)
                if p is None:
                    raise ValueError(f"MoE module missing {name}")
                params.append(p)

        defaults = {
            "lr": cfg.lr_expert,
            "betas": (cfg.adam_beta1, getattr(cfg, 'adam_beta2_expert', cfg.adam_beta2)),
            "eps": cfg.adam_eps,
            "weight_decay": cfg.weight_decay,
        }
        super().__init__(params, defaults)

        self._moes = moe_modules
        self._stochastic_rounding = stochastic_rounding
        self._error_feedback = error_feedback
        self._factored_v = factored_v
        self._step_count = 0

        logger.info(
            f"ECOAdamW initialized: {len(moe_modules)} MoE modules, "
            f"SR={stochastic_rounding}, error_feedback={error_feedback}, "
            f"factored_v={factored_v}"
        )

    def _init_state(self, p: torch.Tensor) -> dict:
        """Initialize optimizer state for a parameter.

        Uses FP8 storage for momentum and variance to save memory:
          - exp_avg (momentum): FP8 E5M2 + per-tensor scale
          - exp_avg_sq (variance): FP8 E4M3 + per-tensor scale
            OR (when factored_v=True):
          - v_row: [*batch, in_dim] FP32 (Adafactor row factor)
          - v_col: [*batch, out_dim] FP32 (Adafactor column factor)
        """
        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, dtype=torch.float32)

            # Per-row scale shape: all dims except last get a scale, last dim gets 1
            scale_shape = p.shape[:-1] + (1,)

            # FP8 optimizer states with per-row scaling
            if hasattr(torch, 'float8_e5m2'):
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float8_e5m2)
                state["exp_avg_scale"] = torch.ones(scale_shape, device=p.device, dtype=torch.float32) * 1e-30
            else:
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                state["exp_avg_scale"] = torch.ones(scale_shape, device=p.device, dtype=torch.float32) * 1e-30

            if self._factored_v:
                # Adafactor-style factored second moment.
                # For p with shape [E, in_dim, out_dim]:
                #   v_row: [E, in_dim] — EMA of mean(grad^2, dim=-1)
                #   v_col: [E, out_dim] — EMA of mean(grad^2, dim=-2)
                # Stored in FP32 (tiny: ~36 KiB per weight for [7168, 2048]).
                v_row_shape = p.shape[:-1]   # [E, in_dim]
                v_col_shape = p.shape[:-2] + p.shape[-1:]  # [E, out_dim]
                state["v_row"] = torch.zeros(v_row_shape, device=p.device, dtype=torch.float32)
                state["v_col"] = torch.zeros(v_col_shape, device=p.device, dtype=torch.float32)
            else:
                if hasattr(torch, 'float8_e4m3fn'):
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float8_e4m3fn)
                    state["exp_avg_sq_scale"] = torch.ones(scale_shape, device=p.device, dtype=torch.float32) * 1e-30
                else:
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["exp_avg_sq_scale"] = torch.ones(scale_shape, device=p.device, dtype=torch.float32) * 1e-30

        return state

    def _get_nvfp4_buffer_for_param(self, p: torch.Tensor) -> tuple:
        """Find the NVFP4 buffer triplet for a given parameter.

        Returns (packed, scale, gs, moe_module, param_name) or None if not in buffer mode.
        """
        for moe in self._moes:
            if not getattr(moe, '_nvfp4_primary', False):
                continue
            for name in ("W1", "W3", "W2"):
                if getattr(moe, name, None) is p:
                    packed = getattr(moe, f'_{name}_packed', None)
                    scale = getattr(moe, f'_{name}_scale', None)
                    gs = getattr(moe, f'_{name}_gs', None)
                    if packed is not None:
                        return packed, scale, gs, moe, name
        return None

    @torch.no_grad()
    def step(self, closure=None):
        """Perform ECO AdamW update on all expert parameters.

        When MoE modules have NVFP4 primary buffers:
        1. Dequant NVFP4 buffers → FP32 (transient, per-param streaming)
        2. Standard AdamW update with FP8 m/v
        3. Quantize to NVFP4 with SR → write to NVFP4 buffers
        4. Rebuild _W_cache via direct kernel
        5. No BF16 write-back needed

        When MoE modules use BF16 params (legacy):
        1. Read BF16 params → FP32
        2-6. Same as above but writes back to p.data
        """
        if closure is not None:
            raise RuntimeError("ECOAdamW does not support closure")
        if len(self.param_groups) != 1:
            raise RuntimeError("ECOAdamW expects a single param group")

        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])

        # Initialize state and bump step
        step_t = None
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self._init_state(p)
            st["step"] += 1
            if step_t is None:
                step_t = st["step"]

        if step_t is None:
            return  # No gradients

        step = int(step_t.item())
        self._step_count = step

        # Bias corrections (standard Adam)
        bias_correction1 = 1.0 - (beta1 ** step)
        bias_correction2 = 1.0 - (beta2 ** step)
        step_size = lr / bias_correction1
        inv_bc2_sqrt = 1.0 / math.sqrt(bias_correction2)

        # ECO injection strength (Section 3.2, Eq. 7)
        # eco_alpha = (beta1 - 1) / (beta1 * step_size)
        # Note: eco_alpha < 0 because beta1 < 1
        eco_alpha = (beta1 - 1.0) / (beta1 * step_size) if self._error_feedback else 0.0

        # Determine device from first parameter
        param_device = next((p.device for p in group["params"] if p.grad is not None), torch.device('cpu'))
        # PRNG for stochastic rounding (step-seeded for reproducibility)
        gen_device = param_device if param_device.type == 'cuda' else torch.device('cpu')
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(step * 1000003 + 42)  # Deterministic per step

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.float()
            st = self.state[p]

            # Check for NVFP4 buffer mode
            buf_info = self._get_nvfp4_buffer_for_param(p)
            use_nvfp4_buffers = buf_info is not None

            # 1. Load current weight as FP32
            if use_nvfp4_buffers:
                packed, scale_buf, gs_buf, moe, param_name = buf_info
                group_size = getattr(moe, '_nvfp4_group_size', 16)
                w = _dequant_nvfp4_buffers_to_fp32(packed, scale_buf, gs_buf, group_size)
                # NVFP4 buffers are in HF layout [E, out_features, in_features].
                # Transpose to nmoe layout [E, in_dim, out_dim] to match p.grad shape.
                w = w.transpose(-1, -2).contiguous()
            else:
                w = p.data.float()

            # 2. Dequantize FP8 optimizer states to FP32
            m = _dequantize_fp8_e5m2(st["exp_avg"], st["exp_avg_scale"])

            # 3. Decoupled weight decay (AdamW: applied before gradient update)
            if weight_decay > 0:
                w.mul_(1.0 - lr * weight_decay)

            # 4. Standard Adam EMA updates (momentum always full)
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

            # 5. Variance update + adaptive denominator
            effective_eps = max(eps, 1e-6)

            if self._factored_v:
                # Adafactor-style factored second moment update.
                # v_row = beta2 * v_row + (1 - beta2) * mean(grad^2, dim=-1)
                # v_col = beta2 * v_col + (1 - beta2) * mean(grad^2, dim=-2)
                grad_sq = grad * grad
                v_row = st["v_row"]
                v_col = st["v_col"]
                v_row.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=1.0 - beta2)
                v_col.mul_(beta2).add_(grad_sq.mean(dim=-2), alpha=1.0 - beta2)
                del grad_sq

                # Reconstruct: v[i,j] = v_row[i] * v_col[j] / mean(v_row)
                # The RMS normalization prevents v from growing unboundedly.
                # v_row: [*batch, in_dim] -> [*batch, in_dim, 1]
                # v_col: [*batch, out_dim] -> [*batch, 1, out_dim]
                rms = v_row.mean(dim=-1, keepdim=True).clamp(min=1e-30)
                v_row_3d = v_row.unsqueeze(-1)
                v_col_3d = v_col.unsqueeze(v_col.ndim - 1)
                v_reconstructed = (v_row_3d * v_col_3d) / rms.unsqueeze(-1)

                denom = (v_reconstructed.sqrt() * inv_bc2_sqrt) + effective_eps
                del v_reconstructed, v_row_3d, v_col_3d
            else:
                v = _dequantize_fp8_e4m3(st["exp_avg_sq"], st["exp_avg_sq_scale"])
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # FP8 E4M3 can zero out very small variance values, making denom ~ eps.
                # With eps=1e-9, this causes update = step_size * m / 1e-9 to be huge.
                # Solution: use a variance-aware epsilon that ensures denom is reasonable.
                # In the fused CUDA kernel, this issue doesn't arise because v is computed
                # in FP32 in-register before quantization. For the Python prototype, we
                # use a larger effective epsilon = max(eps, 1e-6) to prevent this.
                denom = (v.sqrt() * inv_bc2_sqrt) + effective_eps

            # 6. Apply Adam update: w = w - step_size * m / denom
            w.addcdiv_(m, denom, value=-step_size)

            # 7. NVFP4 quantization (with stochastic rounding)
            # When using NVFP4 buffers, quantize in HF layout for lossless packing.
            # When using BF16 params, quantize in nmoe layout (no packing needed).
            if use_nvfp4_buffers:
                # Transpose w to HF layout for quantization + packing
                w_hf = w.transpose(-1, -2).contiguous()
                hf_shape = w_hf.shape
                K_hf = hf_shape[-1]

                pad_k = (SF_VEC - K_hf % SF_VEC) % SF_VEC
                if pad_k > 0:
                    w_padded = torch.nn.functional.pad(w_hf.reshape(-1, K_hf), (0, pad_k))
                else:
                    w_padded = w_hf.reshape(-1, K_hf)

                w_blocks = w_padded.reshape(-1, w_padded.shape[-1] // SF_VEC, SF_VEC)
                scale = _compute_e8m0_scale(w_blocks)

                if self._stochastic_rounding:
                    w_hat_blocks = _quantize_nvfp4_sr(w_blocks, scale, generator)
                else:
                    w_hat_blocks = _quantize_nvfp4_rtn(w_blocks, scale)

                w_hat_hf = w_hat_blocks.reshape(w_padded.shape)
                if pad_k > 0:
                    w_hat_hf = w_hat_hf[:, :K_hf]
                w_hat_hf = w_hat_hf.reshape(hf_shape)

                # Transpose back to nmoe layout for ECO error computation
                w_hat = w_hat_hf.transpose(-1, -2).contiguous()
            else:
                orig_shape = w.shape
                K = w.shape[-1]

                pad_k = (SF_VEC - K % SF_VEC) % SF_VEC
                if pad_k > 0:
                    w_padded = torch.nn.functional.pad(w.reshape(-1, K), (0, pad_k))
                else:
                    w_padded = w.reshape(-1, K)

                w_blocks = w_padded.reshape(-1, w_padded.shape[-1] // SF_VEC, SF_VEC)
                scale = _compute_e8m0_scale(w_blocks)

                if self._stochastic_rounding:
                    w_hat_blocks = _quantize_nvfp4_sr(w_blocks, scale, generator)
                else:
                    w_hat_blocks = _quantize_nvfp4_rtn(w_blocks, scale)

                w_hat = w_hat_blocks.reshape(w_padded.shape)
                if pad_k > 0:
                    w_hat = w_hat[:, :K]
                w_hat = w_hat.reshape(orig_shape)

            # 8. ECO error injection into momentum (always in nmoe layout)
            if self._error_feedback and eco_alpha != 0.0:
                # e = theta_tilde - theta_hat (quantization error)
                error = w - w_hat
                # m_corrected = m + eco_alpha * denom * e
                m.add_(denom * error, alpha=eco_alpha)

            # 9. Write back quantized weight
            if use_nvfp4_buffers:
                # Pack w_hat_hf (HF layout) directly into NVFP4 compressed_tensors format
                new_packed, new_scale, new_gs = _pack_nvfp4_e2m1(
                    w_hat_hf, scale.reshape(hf_shape[:-1] + (-1,))
                )
                # Update the MoE module's NVFP4 buffers in-place
                buf_packed = getattr(moe, f'_{param_name}_packed')
                buf_scale = getattr(moe, f'_{param_name}_scale')
                buf_gs = getattr(moe, f'_{param_name}_gs')
                buf_packed.copy_(new_packed)
                buf_scale.copy_(new_scale.to(buf_scale.dtype))
                # _pack_nvfp4_e2m1 returns global_scale=1.0 but shape may differ
                # from original buf_gs. Just fill with 1.0.
                buf_gs.fill_(1.0)
            else:
                # Legacy path: store dequantized NVFP4 value as BF16
                p.data.copy_(w_hat.to(p.dtype))

            # 10. Requantize m to FP8 for storage (always)
            st["exp_avg"], st["exp_avg_scale"] = _quantize_to_fp8_e5m2(m)
            # Requantize v to FP8 only when NOT using factored_v
            # (factored_v stores v_row/v_col as FP32, already updated in-place above)
            if not self._factored_v:
                st["exp_avg_sq"], st["exp_avg_sq_scale"] = _quantize_to_fp8_e4m3(v)

        # Refresh weight caches for MoE modules (blockscaled GEMM)
        for moe in self._moes:
            if hasattr(moe, 'refresh_weight_cache'):
                try:
                    moe.refresh_weight_cache()
                except Exception:
                    pass

        return None

    def state_dict(self):
        """Save optimizer state."""
        sd = super().state_dict()
        sd['eco_config'] = {
            'stochastic_rounding': self._stochastic_rounding,
            'error_feedback': self._error_feedback,
            'step_count': self._step_count,
        }
        return sd

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        eco_config = state_dict.get('eco_config', {})
        # Filter out eco_config before passing to parent (avoid mutating caller's dict)
        sd = {k: v for k, v in state_dict.items() if k != 'eco_config'}
        super().load_state_dict(sd)
        self._step_count = eco_config.get('step_count', 0)
        # Keep construction-time SR/EF settings (don't override from checkpoint)
        for st in self.state.values():
            step = st.get("step", None)
            if torch.is_tensor(step) and step.is_cuda:
                st["step"] = step.cpu()


# ============================================================================
# Builder function
# ============================================================================

def build_eco_optimizer(
    model: nn.Module,
    cfg,
) -> ECOAdamW:
    """Build ECO optimizer for MoE expert weights.

    Args:
        model: Transformer model with MoE blocks.
        cfg: Config with eco_* settings.

    Returns:
        ECOAdamW optimizer for expert parameters.
    """
    moes = []
    for blk in getattr(model, "blocks", []):
        ffn = getattr(blk, "ffn", None)
        if ffn is not None and hasattr(ffn, "W1") and hasattr(ffn, "W3") and hasattr(ffn, "W2"):
            moes.append(ffn)

    if not moes:
        raise ValueError("No MoE modules found in model for ECO optimizer")

    sr = getattr(cfg, 'eco_stochastic_rounding', True)
    ef = getattr(cfg, 'eco_error_feedback', True)
    fv = getattr(cfg, 'eco_factored_v', False)

    eco = ECOAdamW(
        moe_modules=moes,
        cfg=cfg,
        stochastic_rounding=sr,
        error_feedback=ef,
        factored_v=fv,
    )

    logger.info(
        f"ECO optimizer built: {len(moes)} MoE modules, "
        f"{sum(p.numel() for p in eco.param_groups[0]['params']):,} expert params, "
        f"SR={sr}, EF={ef}, factored_v={fv}"
    )

    return eco


# ============================================================================
# Fused Backward-Optimizer: ECO step inside backward pass
# ============================================================================

class FusedBackwardECO:
    """Fused backward-optimizer for NVFP4 primary mode (Option C).

    Instead of accumulating gradients for all 58 MoE layers and running
    optimizer.step() afterwards, this controller applies the ECO update
    *inside* the backward pass, per-layer. Benefits:

      1. Eliminates 76 GiB of BF16 nn.Parameter storage (W1/W3/W2 become
         requires_grad=False with freed .data).
      2. Eliminates 76 GiB of gradient accumulation (dW consumed immediately).
      3. Only 1 layer's transient BF16 + gradients live at any time (~5 GiB).

    The backward autograd Function (_MoEBlockscaledFused) calls
    fused_update() for each weight after computing dW, then returns None
    for those gradient positions.

    Usage:
        fused_eco = FusedBackwardECO(model, cfg)
        fused_eco.attach(model)

        for step in range(steps):
            fused_eco.set_lr(lr_expert)
            fused_eco.pre_backward(step)
            logits = model(inputs)
            loss = ...
            loss.backward()  # Expert optimization happens here
            fused_eco.post_backward()
            # Only dense optimizer.step() needed
    """

    def __init__(self, model: nn.Module, cfg):
        """Initialize fused backward ECO controller.

        Scans model for MoE modules, initializes FP8 optimizer states.

        Args:
            model: Transformer model with MoE blocks.
            cfg: Config with eco_* and optimizer settings.
        """
        self._moes: list[nn.Module] = []
        for blk in getattr(model, "blocks", []):
            ffn = getattr(blk, "ffn", None)
            if ffn is not None and hasattr(ffn, "W1"):
                self._moes.append(ffn)

        if not self._moes:
            raise ValueError("No MoE modules found for FusedBackwardECO")

        # Stable index mapping: id(moe) -> list position (survives within session)
        self._moe_to_idx: dict[int, int] = {id(m): i for i, m in enumerate(self._moes)}

        # Hyperparameters
        self.lr = float(cfg.lr_expert)
        self.beta1 = float(cfg.adam_beta1)
        self.beta2 = float(getattr(cfg, 'adam_beta2_expert', cfg.adam_beta2))
        self.eps = float(cfg.adam_eps)
        self.weight_decay = float(cfg.weight_decay)
        self.grad_clip = float(getattr(cfg, 'grad_clip', 0.0))
        self._stochastic_rounding = getattr(cfg, 'eco_stochastic_rounding', True)
        self._error_feedback = getattr(cfg, 'eco_error_feedback', True)
        self._factored_v = getattr(cfg, 'eco_factored_v', False)

        # DP group for gradient AllReduce
        self._dp_group = None
        self._dp_size = 1

        # Step tracking
        self._step_count = 0
        self._current_step = 0

        # Per-step bias corrections (computed in pre_backward)
        self._bias_correction1 = 1.0
        self._bias_correction2 = 1.0
        self._step_size = 0.0
        self._inv_bc2_sqrt = 1.0
        self._eco_alpha = 0.0

        # Per-step gradient norm tracking (for per-layer clipping).
        # _norm_sq_gpu accumulates on GPU to avoid 174 GPU-CPU syncs per backward.
        self._norm_sq_gpu: torch.Tensor | None = None  # Lazy-init on first fused_update
        self._prev_global_norm = 1.0

        # PRNG for stochastic rounding
        self._generator = None

        # FP8 optimizer states: keyed by (id(moe), param_name)
        self._states: dict[tuple[int, str], dict] = {}

        logger.info(
            f"FusedBackwardECO initialized: {len(self._moes)} MoE modules, "
            f"SR={self._stochastic_rounding}, EF={self._error_feedback}, "
            f"factored_v={self._factored_v}"
        )

    def set_dp_group(self, dp_group, dp_size: int):
        """Set the DP process group for gradient AllReduce."""
        self._dp_group = dp_group
        self._dp_size = dp_size

    def attach(self, model: nn.Module):
        """Attach to model: disable gradients on expert params, free BF16 storage.

        Must be called AFTER checkpoint loading (so NVFP4 buffers are populated).
        """
        for moe in self._moes:
            if not getattr(moe, '_nvfp4_primary', False):
                logger.warning(
                    f"MoE module does not have NVFP4 primary buffers set. "
                    f"Call set_nvfp4_buffers() or load checkpoint first."
                )
                continue

            # Disable gradient tracking on expert params
            for name in ("W1", "W3", "W2"):
                p = getattr(moe, name, None)
                if p is not None and isinstance(p, nn.Parameter):
                    p.requires_grad_(False)
                    # Free the BF16 storage (will be repopulated transiently in forward)
                    p.data = torch.empty(0, dtype=torch.bfloat16, device=p.device)

            # Set the fused_eco reference on the module
            moe._fused_eco = self

        logger.info(
            f"FusedBackwardECO attached: freed BF16 expert params, "
            f"disabled requires_grad on {len(self._moes)} MoE modules"
        )

    def _get_or_init_state(self, moe: nn.Module, param_name: str) -> dict:
        """Get or lazily initialize FP8 optimizer state for one weight."""
        idx = self._moe_to_idx[id(moe)]
        key = (idx, param_name)
        if key not in self._states:
            # Determine shape from NVFP4 buffer (HF layout -> nmoe layout)
            packed = getattr(moe, f'_{param_name}_packed')
            # packed is [E, out_dim, in_dim//2], full weight is [E, out_dim, in_dim]
            # in nmoe layout (transposed): [E, in_dim, out_dim]
            E = packed.shape[0]
            out_dim = packed.shape[1]
            in_dim = packed.shape[2] * 2
            # nmoe layout shape (after transpose): [E, in_dim, out_dim]
            shape = (E, in_dim, out_dim)
            device = packed.device
            scale_shape = shape[:-1] + (1,)

            state = {
                "step": torch.tensor(0.0, dtype=torch.float32),
            }
            if hasattr(torch, 'float8_e5m2'):
                state["exp_avg"] = torch.zeros(shape, dtype=torch.float8_e5m2, device=device)
                state["exp_avg_scale"] = torch.ones(scale_shape, device=device, dtype=torch.float32) * 1e-30
            else:
                state["exp_avg"] = torch.zeros(shape, dtype=torch.bfloat16, device=device)
                state["exp_avg_scale"] = torch.ones(scale_shape, device=device, dtype=torch.float32) * 1e-30

            if self._factored_v:
                # Adafactor-style factored second moment.
                # shape = (E, in_dim, out_dim) in nmoe layout.
                #   v_row: [E, in_dim] — EMA of mean(grad^2, dim=-1)
                #   v_col: [E, out_dim] — EMA of mean(grad^2, dim=-2)
                v_row_shape = shape[:-1]   # (E, in_dim)
                v_col_shape = shape[:-2] + shape[-1:]  # (E, out_dim)
                state["v_row"] = torch.zeros(v_row_shape, device=device, dtype=torch.float32)
                state["v_col"] = torch.zeros(v_col_shape, device=device, dtype=torch.float32)
            else:
                if hasattr(torch, 'float8_e4m3fn'):
                    state["exp_avg_sq"] = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
                    state["exp_avg_sq_scale"] = torch.ones(scale_shape, device=device, dtype=torch.float32) * 1e-30
                else:
                    state["exp_avg_sq"] = torch.zeros(shape, dtype=torch.bfloat16, device=device)
                    state["exp_avg_sq_scale"] = torch.ones(scale_shape, device=device, dtype=torch.float32) * 1e-30

            self._states[key] = state

        return self._states[key]

    def set_lr(self, lr: float):
        """Set learning rate for the next backward pass."""
        self.lr = lr

    def pre_backward(self, step: int):
        """Prepare for backward pass. Call before loss.backward().

        Args:
            step: Current training step number (0-indexed).
        """
        self._current_step = step
        self._step_count = step + 1  # 1-indexed for Adam bias correction

        # Bias corrections
        bc1 = 1.0 - (self.beta1 ** self._step_count)
        bc2 = 1.0 - (self.beta2 ** self._step_count)
        self._bias_correction1 = bc1
        self._bias_correction2 = bc2
        self._step_size = self.lr / bc1
        self._inv_bc2_sqrt = 1.0 / math.sqrt(bc2)

        # ECO injection strength
        if self._error_feedback and self._step_size != 0:
            self._eco_alpha = (self.beta1 - 1.0) / (self.beta1 * self._step_size)
        else:
            self._eco_alpha = 0.0

        # PRNG for stochastic rounding
        device = next((getattr(moe, '_W1_packed').device for moe in self._moes
                       if getattr(moe, '_W1_packed', None) is not None), torch.device('cpu'))
        gen_device = device if device.type == 'cuda' else torch.device('cpu')
        self._generator = torch.Generator(device=gen_device)
        self._generator.manual_seed(self._step_count * 1000003 + 42)

        # Reset GPU gradient norm accumulator
        if self._norm_sq_gpu is not None:
            self._norm_sq_gpu.zero_()

    def post_backward(self):
        """Finalize after backward pass. Call after loss.backward()."""
        # Update global norm estimate for next step's clipping.
        # Single .item() call here replaces 174 per-weight GPU-CPU syncs.
        if self._norm_sq_gpu is not None:
            norm_sq = self._norm_sq_gpu.item()
        else:
            norm_sq = 0.0
        self._prev_global_norm = math.sqrt(max(norm_sq, 1e-30))

    def _try_cuda_fused_update(
        self, moe: nn.Module, param_name: str, grad: torch.Tensor, st: dict,
    ) -> bool:
        """Attempt to run the fused CUDA kernel. Returns True on success.

        The CUDA kernel (eco_adam.cu) does the entire ECO AdamW step — dequant
        NVFP4 → AdamW → SR → ECO error → requant — in registers/shared memory,
        with zero FP32 global memory materialization (~7 bytes/element vs ~68).

        When factored_v=True, calls the factored-v variant which runs reduction
        kernels (v_row/v_col update) followed by the main kernel with on-the-fly
        v reconstruction from v_row * v_col / v_rms.

        Requirements for the CUDA path:
          - nmoe.csrc.rdep module has eco_adam_nvfp4_update binding
          - out_dim and in_dim are multiples of 32
          - FP8 m states stored as uint8 with per-row FP32 scale
          - For non-factored: FP8 v states as uint8 with per-row FP32 scale
          - For factored_v: v_row, v_col as FP32, v_rms scratch buffer

        Falls back to Python if any requirement is unmet.
        """
        try:
            from nmoe.csrc import rdep as _rdep
            if not hasattr(_rdep, 'eco_adam_nvfp4_update'):
                return False
            if self._factored_v and not hasattr(_rdep, 'eco_adam_nvfp4_fv_update'):
                return False
        except ImportError:
            return False

        packed = getattr(moe, f'_{param_name}_packed')
        scale_buf = getattr(moe, f'_{param_name}_scale')
        gs_buf = getattr(moe, f'_{param_name}_gs')
        group_size = getattr(moe, '_nvfp4_group_size', 16)

        # Dimensions from packed buffer: [E, out_dim, in_dim/2]
        E = packed.shape[0]
        out_dim = packed.shape[1]
        in_dim = packed.shape[2] * 2

        # Alignment check (kernel tiles at 32-element boundaries)
        if (out_dim & 31) != 0 or (in_dim & 31) != 0:
            return False
        if in_dim % group_size != 0:
            return False

        # FP8 m state must be uint8 with FP32 per-row scale
        m_data = st["exp_avg"]
        m_sc = st["exp_avg_scale"]
        if m_data.dtype not in (torch.uint8, torch.float8_e5m2):
            return False

        # Shape validation: grad must be [E, in_dim, out_dim] (nmoe layout)
        expected_grad_shape = (E, in_dim, out_dim)
        if grad.shape != expected_grad_shape:
            logger.warning(
                "CUDA eco_adam: grad shape %s != expected %s, falling back",
                grad.shape, expected_grad_shape,
            )
            return False

        # Contiguity: kernel reads raw pointers assuming C-contiguous layout
        if not grad.is_contiguous():
            grad = grad.contiguous()

        # Shape validation: m must be [E, in_dim, out_dim] (nmoe layout)
        expected_mv_shape = (E, in_dim, out_dim)
        if m_data.shape != expected_mv_shape:
            return False

        # View FP8 m as uint8 for the kernel (requires contiguous)
        if not m_data.is_contiguous():
            return False
        m_u8 = m_data.view(torch.uint8) if m_data.dtype != torch.uint8 else m_data

        # Flatten per-row scale to [E * in_dim] (from [E, in_dim, 1])
        m_sc_flat = m_sc.reshape(-1).contiguous()

        # PRNG seeds for stochastic rounding (baked into Philox counter[2:3])
        _param_offset = {'W1': 0, 'W2': 1, 'W3': 2}[param_name]
        idx = self._moe_to_idx[id(moe)]
        prng_seed0 = (self._step_count * 1000003 + idx * 7 + _param_offset) & 0xFFFFFFFF
        prng_seed1 = (self._step_count * 7919 + idx * 31 + _param_offset * 127) & 0xFFFFFFFF

        # Keep references alive across async kernel execution
        scale_u8 = scale_buf.view(torch.uint8)
        stream = torch.cuda.current_stream(packed.device)

        if self._factored_v:
            # Factored-v path: v_row/v_col/v_rms instead of v_data/v_scale
            v_row = st["v_row"]
            v_col = st["v_col"]

            # Shape validation
            if v_row.shape != (E, in_dim) or v_col.shape != (E, out_dim):
                return False
            if not v_row.is_contiguous() or not v_col.is_contiguous():
                return False

            # Allocate v_rms scratch buffer [E] (reused across calls via state)
            if "v_rms" not in st:
                st["v_rms"] = torch.zeros(E, device=packed.device, dtype=torch.float32)
            v_rms = st["v_rms"]

            _rdep.eco_adam_nvfp4_fv_update(
                packed.data_ptr(),
                scale_u8.data_ptr(),
                gs_buf.data_ptr(),
                m_u8.data_ptr(),
                m_sc_flat.data_ptr(),
                v_row.data_ptr(),
                v_col.data_ptr(),
                v_rms.data_ptr(),
                grad.data_ptr(),
                E, out_dim, in_dim, group_size,
                self.lr, self.beta1, self.beta2,
                self.weight_decay, self.eps,
                self._step_size, self._inv_bc2_sqrt,
                self._eco_alpha,
                1 if self._stochastic_rounding else 0,
                1 if self._error_feedback else 0,
                prng_seed0, prng_seed1,
                stream,
            )

            # Write updated per-row scales back to state (m only; v is factored)
            st["exp_avg_scale"] = m_sc_flat.reshape(m_sc.shape)

        else:
            # Standard path: full FP8 v
            v_data = st["exp_avg_sq"]
            v_sc = st["exp_avg_sq_scale"]
            if v_data.dtype not in (torch.uint8, torch.float8_e4m3fn):
                return False
            if v_data.shape != expected_mv_shape:
                return False
            if not v_data.is_contiguous():
                return False
            v_u8 = v_data.view(torch.uint8) if v_data.dtype != torch.uint8 else v_data
            v_sc_flat = v_sc.reshape(-1).contiguous()

            _rdep.eco_adam_nvfp4_update(
                packed.data_ptr(),
                scale_u8.data_ptr(),
                gs_buf.data_ptr(),
                m_u8.data_ptr(),
                m_sc_flat.data_ptr(),
                v_u8.data_ptr(),
                v_sc_flat.data_ptr(),
                grad.data_ptr(),
                E, out_dim, in_dim, group_size,
                self.lr, self.beta1, self.beta2,
                self.weight_decay, self.eps,
                self._step_size, self._inv_bc2_sqrt,
                self._eco_alpha,
                1 if self._stochastic_rounding else 0,
                1 if self._error_feedback else 0,
                prng_seed0, prng_seed1,
                stream,
            )

            # Write updated per-row scales back to state (kernel modifies them in-place
            # via the k_fp8_recompute_row_scale pass).
            st["exp_avg_scale"] = m_sc_flat.reshape(m_sc.shape)
            st["exp_avg_sq_scale"] = v_sc_flat.reshape(v_sc.shape)

        return True

    @torch.no_grad()
    def fused_update(self, moe: nn.Module, param_name: str, grad_bf16: torch.Tensor):
        """Apply one ECO AdamW step for a single expert weight, in-place.

        Called from _MoEBlockscaledFused.backward() immediately after computing
        the weight gradient. Consumes the gradient — caller should free it after.

        Tries the fused CUDA kernel first (zero FP32 materialization). Falls back
        to the Python implementation if the kernel is unavailable or dimensions
        are incompatible.

        Args:
            moe: The MoE module containing NVFP4 buffers.
            param_name: 'W1', 'W3', or 'W2'.
            grad_bf16: [E, dim1, dim2] BF16 gradient (nmoe layout).
        """
        import torch.distributed as dist

        assert id(moe) in self._moe_to_idx, (
            f"fused_update called with unknown MoE module (id={id(moe)}). "
            f"Known ids: {list(self._moe_to_idx.keys())}"
        )
        assert param_name in ('W1', 'W2', 'W3'), f"Invalid param_name: {param_name}"

        # DP AllReduce in FP32 to preserve gradient precision during averaging.
        grad = grad_bf16.float()
        del grad_bf16
        if self._dp_size > 1 and self._dp_group is not None:
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=self._dp_group)

        # Gradient clipping using previous step's global norm estimate.
        if self.grad_clip > 0:
            grad_flat = grad.reshape(-1)
            grad_norm_sq = torch.dot(grad_flat, grad_flat)
            if self._norm_sq_gpu is None:
                self._norm_sq_gpu = torch.zeros(1, device=grad.device, dtype=torch.float64)
            self._norm_sq_gpu += grad_norm_sq
            if self._step_count > 1:
                clip_coeff = self.grad_clip / (self._prev_global_norm + 1e-6)
                if clip_coeff < 1.0:
                    grad.mul_(clip_coeff)

        # Get or init optimizer state
        st = self._get_or_init_state(moe, param_name)
        st["step"] = torch.tensor(float(self._step_count), dtype=torch.float32)

        # Try CUDA fused kernel (zero FP32 materialization)
        if self._try_cuda_fused_update(moe, param_name, grad, st):
            del grad
            return


        # ===== Python fallback =====

        # 1. Dequant NVFP4 buffers -> FP32 (HF layout, then transpose to nmoe)
        packed = getattr(moe, f'_{param_name}_packed')
        scale_buf = getattr(moe, f'_{param_name}_scale')
        gs_buf = getattr(moe, f'_{param_name}_gs')
        group_size = getattr(moe, '_nvfp4_group_size', 16)

        w = _dequant_nvfp4_buffers_to_fp32(packed, scale_buf, gs_buf, group_size)
        # Transpose HF [E, out_features, in_features] -> nmoe [E, in_dim, out_dim]
        w = w.transpose(-1, -2).contiguous()

        # 2. Dequant FP8 momentum
        m = _dequantize_fp8_e5m2(st["exp_avg"], st["exp_avg_scale"])

        # 3. Decoupled weight decay
        if self.weight_decay > 0:
            w.mul_(1.0 - self.lr * self.weight_decay)

        # 4. Adam EMA updates (momentum always full)
        m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)

        # 5. Variance update + adaptive denominator
        effective_eps = max(self.eps, 1e-6)

        if self._factored_v:
            # Adafactor-style factored second moment update.
            grad_sq = grad * grad
            del grad  # Free FP32 gradient

            v_row = st["v_row"]
            v_col = st["v_col"]
            v_row.mul_(self.beta2).add_(grad_sq.mean(dim=-1), alpha=1.0 - self.beta2)
            v_col.mul_(self.beta2).add_(grad_sq.mean(dim=-2), alpha=1.0 - self.beta2)
            del grad_sq

            # Reconstruct v on-the-fly: v[i,j] = v_row[i] * v_col[j] / mean(v_row)
            # v_row: [*batch, in_dim] -> [*batch, in_dim, 1]
            # v_col: [*batch, out_dim] -> [*batch, 1, out_dim]
            rms = v_row.mean(dim=-1, keepdim=True).clamp(min=1e-30)
            v_row_3d = v_row.unsqueeze(-1)
            v_col_3d = v_col.unsqueeze(v_col.ndim - 1)
            v_reconstructed = (v_row_3d * v_col_3d) / rms.unsqueeze(-1)
            del v_row_3d, v_col_3d

            # Compute denominator from reconstructed v
            v_reconstructed.sqrt_().mul_(self._inv_bc2_sqrt).add_(effective_eps)
            denom = v_reconstructed  # reconstructed v is now the denominator
        else:
            v = _dequantize_fp8_e4m3(st["exp_avg_sq"], st["exp_avg_sq_scale"])
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            del grad  # Free FP32 gradient

            # Requantize v to FP8 early (before in-place denom computation consumes it)
            st["exp_avg_sq"], st["exp_avg_sq_scale"] = _quantize_to_fp8_e4m3(v)

            # Adaptive denominator (computed in-place over v to save ~900 MiB FP32)
            v.sqrt_().mul_(self._inv_bc2_sqrt).add_(effective_eps)
            denom = v  # v is now the denominator; original v is consumed

        # 7. Apply Adam update
        w.addcdiv_(m, denom, value=-self._step_size)

        # Precompute ECO injection coefficient and free denom early (~896 MiB).
        if self._error_feedback and self._eco_alpha != 0.0:
            denom.mul_(self._eco_alpha)
            eco_coeff = denom
        else:
            del denom

        # 8. NVFP4 quantization in HF layout for packing.
        w_hf = w.transpose(-1, -2).contiguous()
        del w
        hf_shape = w_hf.shape
        K_hf = hf_shape[-1]

        pad_k = (SF_VEC - K_hf % SF_VEC) % SF_VEC
        if pad_k > 0:
            w_padded = torch.nn.functional.pad(w_hf.reshape(-1, K_hf), (0, pad_k))
        else:
            w_padded = w_hf.reshape(-1, K_hf)

        w_blocks = w_padded.reshape(-1, w_padded.shape[-1] // SF_VEC, SF_VEC)
        scale = _compute_e8m0_scale(w_blocks)

        if self._stochastic_rounding:
            _param_offset = {'W1': 0, 'W2': 1, 'W3': 2}[param_name]
            idx = self._moe_to_idx[id(moe)]
            self._generator.manual_seed(
                self._step_count * 1000003 + idx * 7 + _param_offset
            )
            w_hat_blocks = _quantize_nvfp4_sr(w_blocks, scale, self._generator)
        else:
            w_hat_blocks = _quantize_nvfp4_rtn(w_blocks, scale)

        w_hat_hf = w_hat_blocks.reshape(w_padded.shape)
        if pad_k > 0:
            w_hat_hf = w_hat_hf[:, :K_hf]
        w_hat_hf = w_hat_hf.reshape(hf_shape)

        # 9. ECO error injection into momentum (nmoe layout).
        if self._error_feedback and self._eco_alpha != 0.0:
            error_hf = w_hf - w_hat_hf
            del w_hf
            error = error_hf.transpose(-1, -2).contiguous()
            del error_hf
            m.add_(eco_coeff * error)
            del error, eco_coeff
        else:
            del w_hf

        # 10. Pack updated weights to NVFP4 buffers in-place
        new_packed, new_scale, new_gs = _pack_nvfp4_e2m1(
            w_hat_hf, scale.reshape(hf_shape[:-1] + (-1,))
        )
        packed.copy_(new_packed)
        scale_buf.copy_(new_scale.to(scale_buf.dtype))
        gs_buf.fill_(1.0)
        del w_hat_hf, new_packed, new_scale, new_gs

        # 11. Requantize m to FP8
        st["exp_avg"], st["exp_avg_scale"] = _quantize_to_fp8_e5m2(m)
        del m

    def refresh_layer_cache(self, moe: nn.Module):
        """Invalidate blockscaled _W_cache after fused update.

        During backward, GPU memory is nearly fully occupied by activations,
        gradients, and optimizer intermediates — there isn't enough headroom to
        dequant NVFP4 → BF16 → blockscaled MMA (~1.4 GiB transient per weight).
        Instead of rebuilding here, we invalidate the cache by setting it to None
        and freeing the old tensors.  The next forward pass will lazily rebuild
        via model.py's `if self._W_cache is None: self.refresh_weight_cache()`,
        when all backward activation memory has already been freed.
        """
        if hasattr(moe, '_W_cache') and moe._W_cache is not None:
            del moe._W_cache
            moe._W_cache = None

    def state_dict(self) -> dict:
        """Serialize optimizer state for checkpointing."""
        states = {}
        for (idx, param_name), st in self._states.items():
            key = f"layer_{idx}.{param_name}"
            state_entry = {
                "step": st["step"],
                "exp_avg": st["exp_avg"],
                "exp_avg_scale": st["exp_avg_scale"],
            }
            if self._factored_v:
                state_entry["v_row"] = st["v_row"]
                state_entry["v_col"] = st["v_col"]
            else:
                state_entry["exp_avg_sq"] = st["exp_avg_sq"]
                state_entry["exp_avg_sq_scale"] = st["exp_avg_sq_scale"]
            states[key] = state_entry

        return {
            "step_count": self._step_count,
            "prev_global_norm": self._prev_global_norm,
            "eco_config": {
                "stochastic_rounding": self._stochastic_rounding,
                "error_feedback": self._error_feedback,
                "factored_v": self._factored_v,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            },
            "states": states,
        }

    def load_state_dict(self, state_dict: dict):
        """Restore optimizer state from checkpoint."""
        self._step_count = state_dict.get("step_count", 0)
        self._prev_global_norm = state_dict.get("prev_global_norm", 1.0)

        saved_states = state_dict.get("states", {})
        for i, moe in enumerate(self._moes):
            for param_name in ("W1", "W3", "W2"):
                key = f"layer_{i}.{param_name}"
                if key not in saved_states:
                    continue

                saved = saved_states[key]
                st = self._get_or_init_state(moe, param_name)

                # Copy saved tensors to initialized state (handles device/dtype)
                st["step"].copy_(saved["step"])
                st["exp_avg"].copy_(saved["exp_avg"].to(st["exp_avg"].dtype))
                st["exp_avg_scale"].copy_(saved["exp_avg_scale"])

                if self._factored_v:
                    if "v_row" in saved and "v_col" in saved:
                        st["v_row"].copy_(saved["v_row"])
                        st["v_col"].copy_(saved["v_col"])
                    else:
                        logger.warning(
                            f"factored_v=True but checkpoint has full v state for {key}. "
                            f"Initializing v_row/v_col to zero (cold-start)."
                        )
                else:
                    if "exp_avg_sq" in saved and "exp_avg_sq_scale" in saved:
                        st["exp_avg_sq"].copy_(saved["exp_avg_sq"].to(st["exp_avg_sq"].dtype))
                        st["exp_avg_sq_scale"].copy_(saved["exp_avg_sq_scale"])
                    else:
                        logger.warning(
                            f"factored_v=False but checkpoint has factored v state for {key}. "
                            f"Initializing exp_avg_sq to zero (cold-start)."
                        )

        logger.info(
            f"FusedBackwardECO state loaded: step={self._step_count}, "
            f"{len(saved_states)} parameter states restored"
        )
