"""ECO Reference Implementation: Python-only ECOAdamW optimizer.

This is the original Python prototype of the ECO optimizer, preserved as a
reference implementation for testing and validation. The production training
path uses FusedBackwardECO (in eco.py), which fuses the optimizer step into
the backward pass via CUDA kernels with zero FP32 materialization.

ECOAdamW is NOT used in production training. It exists for:
  - Unit tests: numerical comparison against the CUDA kernel
  - Debugging: a readable Python implementation of Algorithm 3
  - Documentation: understanding the ECO update math

Usage:
    from nmoe.eco_reference import ECOAdamW, build_eco_optimizer
    eco = ECOAdamW(moe_modules, cfg)
    eco.step()
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

    global_scale = torch.ones(*orig_shape[:-1], 1, dtype=torch.float32, device=device)

    # Convert E8M0 block scale [*, n_blocks] to per-group scale
    # n_blocks = K // 32, n_groups = K // 16 = 2 * n_blocks
    n_blocks = scale.shape[-1]
    # Repeat each block scale for 2 groups (16 elements each = 32 total)
    scale_per_group = scale.unsqueeze(-1).expand(*scale.shape, 2).reshape(*scale.shape[:-1], 2 * n_blocks)
    # Convert to float8_e4m3fn (scale * global_scale, but global_scale=1.0)
    scale_fp8 = scale_per_group.float().to(torch.float8_e4m3fn)

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
    x_fp8 = x_scaled.to(torch.float8_e5m2)

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
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)

    return x_fp8.reshape(orig_shape), scale.reshape(orig_shape[:-1] + (1,))


def _dequantize_fp8_e4m3(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 E4M3 to FP32."""
    return x_fp8.float() * scale


# ============================================================================
# Reference ECO optimizer
# ============================================================================

class ECOAdamW(torch.optim.Optimizer):
    """ECO-enhanced AdamW for expert weights with NVFP4 primary + FP8 states.

    **Reference implementation only — not used in production training.**
    See FusedBackwardECO in eco.py for the production CUDA-kernel path.

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
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float8_e5m2)
            state["exp_avg_scale"] = torch.ones(scale_shape, device=p.device, dtype=torch.float32) * 1e-30

            if self._factored_v:
                # Adafactor-style factored second moment.
                v_row_shape = p.shape[:-1]   # [E, in_dim]
                v_col_shape = p.shape[:-2] + p.shape[-1:]  # [E, out_dim]
                state["v_row"] = torch.zeros(v_row_shape, device=p.device, dtype=torch.float32)
                state["v_col"] = torch.zeros(v_col_shape, device=p.device, dtype=torch.float32)
            else:
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float8_e4m3fn)
                state["exp_avg_sq_scale"] = torch.ones(scale_shape, device=p.device, dtype=torch.float32) * 1e-30

        return state

    def _get_nvfp4_buffer_for_param(self, p: torch.Tensor) -> tuple | None:
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
        1. Dequant NVFP4 buffers -> FP32 (transient, per-param streaming)
        2. Standard AdamW update with FP8 m/v
        3. Quantize to NVFP4 with SR -> write to NVFP4 buffers
        4. Rebuild _W_cache via direct kernel
        5. No BF16 write-back needed

        When MoE modules use BF16 params (legacy):
        1. Read BF16 params -> FP32
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
        eco_alpha = (beta1 - 1.0) / (beta1 * step_size) if self._error_feedback else 0.0

        # Determine device from first parameter
        param_device = next((p.device for p in group["params"] if p.grad is not None), torch.device('cpu'))
        gen_device = param_device if param_device.type == 'cuda' else torch.device('cpu')
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(step * 1000003 + 42)

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
                w = w.transpose(-1, -2).contiguous()
            else:
                w = p.data.float()

            # 2. Dequantize FP8 optimizer states to FP32
            m = _dequantize_fp8_e5m2(st["exp_avg"], st["exp_avg_scale"])

            # 3. Decoupled weight decay (AdamW)
            if weight_decay > 0:
                w.mul_(1.0 - lr * weight_decay)

            # 4. Standard Adam EMA updates
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

            # 5. Variance update + adaptive denominator
            effective_eps = max(eps, 1e-6)

            if self._factored_v:
                grad_sq = grad * grad
                v_row = st["v_row"]
                v_col = st["v_col"]
                v_row.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=1.0 - beta2)
                v_col.mul_(beta2).add_(grad_sq.mean(dim=-2), alpha=1.0 - beta2)
                del grad_sq

                rms = v_row.mean(dim=-1, keepdim=True).clamp(min=1e-30)
                v_row_3d = v_row.unsqueeze(-1)
                v_col_3d = v_col.unsqueeze(v_col.ndim - 1)
                v_reconstructed = (v_row_3d * v_col_3d) / rms.unsqueeze(-1)

                denom = (v_reconstructed.sqrt() * inv_bc2_sqrt) + effective_eps
                del v_reconstructed, v_row_3d, v_col_3d
            else:
                v = _dequantize_fp8_e4m3(st["exp_avg_sq"], st["exp_avg_sq_scale"])
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = (v.sqrt() * inv_bc2_sqrt) + effective_eps

            # 6. Apply Adam update
            w.addcdiv_(m, denom, value=-step_size)

            # 7. NVFP4 quantization (with stochastic rounding)
            if use_nvfp4_buffers:
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

            # 8. ECO error injection into momentum
            if self._error_feedback and eco_alpha != 0.0:
                error = w - w_hat
                m.add_(denom * error, alpha=eco_alpha)

            # 9. Write back quantized weight
            if use_nvfp4_buffers:
                new_packed, new_scale, new_gs = _pack_nvfp4_e2m1(
                    w_hat_hf, scale.reshape(hf_shape[:-1] + (-1,))
                )
                buf_packed = getattr(moe, f'_{param_name}_packed')
                buf_scale = getattr(moe, f'_{param_name}_scale')
                buf_gs = getattr(moe, f'_{param_name}_gs')
                buf_packed.copy_(new_packed)
                buf_scale.copy_(new_scale.to(buf_scale.dtype))
                buf_gs.fill_(1.0)
            else:
                p.data.copy_(w_hat.to(p.dtype))

            # 10. Requantize m to FP8
            st["exp_avg"], st["exp_avg_scale"] = _quantize_to_fp8_e5m2(m)
            if not self._factored_v:
                st["exp_avg_sq"], st["exp_avg_sq_scale"] = _quantize_to_fp8_e4m3(v)

        # Refresh weight caches for MoE modules
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
        sd = {k: v for k, v in state_dict.items() if k != 'eco_config'}
        super().load_state_dict(sd)
        self._step_count = eco_config.get('step_count', 0)
        for st in self.state.values():
            step = st.get("step", None)
            if torch.is_tensor(step) and step.is_cuda:
                st["step"] = step.cpu()


def build_eco_optimizer(
    model: nn.Module,
    cfg,
) -> ECOAdamW:
    """Build ECO optimizer for MoE expert weights.

    **Reference implementation only — not used in production training.**

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
