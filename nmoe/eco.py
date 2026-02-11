"""ECO Optimizer — production module: FusedBackwardECO + CUDA kernel dispatch.

Implements Algorithm 3 from "ECO: Quantized Training without Full-Precision
Master Weights" (Nikdan et al., Google Research + ISTA, arXiv:2601.22101v1).

ECO eliminates BF16 master weights by injecting quantization error into Adam's
momentum buffer. The error is computed and consumed in the same step -- no
persistent error buffer needed.

FusedBackwardECO fuses the optimizer step into the backward pass via CUDA
kernels with zero FP32 global memory materialization.

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
"""

from __future__ import annotations

import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Fail fast if PyTorch lacks FP8 support — silent BF16 fallback is a 10x perf trap.
if not hasattr(torch, 'float8_e5m2') or not hasattr(torch, 'float8_e4m3fn'):
    raise ImportError(
        f"PyTorch {torch.__version__} does not support FP8 dtypes (float8_e5m2, float8_e4m3fn). "
        f"ECO requires PyTorch >= 2.1. Upgrade PyTorch or disable ECO."
    )


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

        # Gradient accumulation state
        self._accum_steps = int(getattr(cfg, 'gradient_accumulation_steps', 1))
        self._current_microstep = 0  # Set by training loop via set_microstep()

        # CUDA kernel requirement: fail fast if kernels are unavailable
        self._require_cuda = getattr(cfg, 'eco_require_cuda', True)
        if self._require_cuda:
            try:
                from nmoe.csrc import rdep as _rdep
            except ImportError as e:
                raise ImportError(
                    "eco_require_cuda=True but nmoe.csrc.rdep is not importable. "
                    "Build the CUDA extension with `python setup.py build_ext --inplace`. "
                    "Set eco_require_cuda=False to allow Python fallback (10x slower)."
                ) from e

            missing = []
            for name in ('eco_adam_nvfp4_update', 'eco_mv_accumulate'):
                if not hasattr(_rdep, name):
                    missing.append(name)
            if self._factored_v:
                for name in ('eco_adam_nvfp4_fv_update', 'eco_mv_accumulate_fv'):
                    if not hasattr(_rdep, name):
                        missing.append(name)
            if missing:
                raise RuntimeError(
                    f"eco_require_cuda=True but CUDA bindings missing: {missing}. "
                    f"Rebuild the CUDA extension or set eco_require_cuda=False."
                )
            self._rdep = _rdep
            logger.info("FusedBackwardECO: CUDA kernels validated at init")

        logger.info(
            f"FusedBackwardECO initialized: {len(self._moes)} MoE modules, "
            f"SR={self._stochastic_rounding}, EF={self._error_feedback}, "
            f"factored_v={self._factored_v}, "
            f"accum_steps={self._accum_steps}, "
            f"require_cuda={self._require_cuda}"
        )

    def set_dp_group(self, dp_group, dp_size: int):
        """Set the DP process group for gradient AllReduce."""
        self._dp_group = dp_group
        self._dp_size = dp_size

    def set_microstep(self, microstep: int, total_microsteps: int):
        """Set the current micro-step index for gradient accumulation.

        Called by the training loop before each micro-batch forward/backward pass.

        Args:
            microstep: Current micro-step index (0 to total_microsteps - 1).
            total_microsteps: Total number of micro-steps (= gradient_accumulation_steps).
        """
        self._current_microstep = microstep
        self._accum_steps = total_microsteps

    @property
    def is_accumulating(self) -> bool:
        """True if gradient accumulation is active and this is not the final micro-step."""
        return self._accum_steps > 1 and self._current_microstep < self._accum_steps - 1

    @property
    def is_final_microstep(self) -> bool:
        """True if this is the final micro-step (should run full Adam update)."""
        return self._accum_steps <= 1 or self._current_microstep == self._accum_steps - 1

    def _cuda_mv_accumulate(
        self, moe: nn.Module, param_name: str, grad: torch.Tensor, st: dict,
        beta1_frac: float, beta2_frac: float,
    ) -> None:
        """Accumulate gradient into FP8 m/v states using AdamA CUDA kernel.

        Non-final micro-steps call this instead of the full Adam update.
        Updates m and v with fractional betas β₁^(1/K) and β₂^(1/K):
          m = β₁_frac * m + (1 - β₁_frac) * g
          v = β₂_frac * v + (1 - β₂_frac) * g²

        Raises RuntimeError if CUDA kernel requirements are not met.
        """
        _rdep = getattr(self, '_rdep', None)
        if _rdep is None:
            raise RuntimeError("CUDA kernel required but _rdep not initialized")

        packed = getattr(moe, f'_{param_name}_packed')
        E = packed.shape[0]
        out_dim = packed.shape[1]
        in_dim = packed.shape[2] * 2

        def _fail(msg):
            raise RuntimeError(f"eco_mv_accumulate CUDA kernel: {msg}")

        # Alignment check
        if (out_dim & 31) != 0 or (in_dim & 31) != 0:
            return _fail(f"dims not 32-aligned: out_dim={out_dim}, in_dim={in_dim}")

        expected = (E, in_dim, out_dim)
        if grad.shape != expected:
            return _fail(f"grad shape {grad.shape} != expected {expected}")
        if not grad.is_contiguous():
            grad = grad.contiguous()

        m_data = st["exp_avg"]
        m_sc = st["exp_avg_scale"]
        if m_data.dtype not in (torch.uint8, torch.float8_e5m2):
            return _fail(f"m dtype {m_data.dtype} not FP8 E5M2")
        if m_data.shape != expected:
            return _fail(f"m shape {m_data.shape} != expected {expected}")
        if not m_data.is_contiguous():
            return _fail("m not contiguous")

        m_u8 = m_data.view(torch.uint8) if m_data.dtype != torch.uint8 else m_data
        m_sc_flat = m_sc.reshape(-1).contiguous()
        stream = torch.cuda.current_stream(packed.device)

        if self._factored_v:
            v_row = st["v_row"]
            v_col = st["v_col"]
            if v_row.shape != (E, in_dim) or v_col.shape != (E, out_dim):
                return _fail(f"v_row/v_col shape mismatch")
            if not v_row.is_contiguous() or not v_col.is_contiguous():
                return _fail("v_row/v_col not contiguous")
            if "v_rms" not in st:
                st["v_rms"] = torch.zeros(E, device=packed.device, dtype=torch.float32)
            v_rms = st["v_rms"]

            _rdep.eco_mv_accumulate_fv(
                m_u8.data_ptr(),
                m_sc_flat.data_ptr(),
                v_row.data_ptr(),
                v_col.data_ptr(),
                v_rms.data_ptr(),
                grad.data_ptr(),
                E, in_dim, out_dim,
                beta1_frac, beta2_frac,
                stream,
            )
            st["exp_avg_scale"] = m_sc_flat.reshape(m_sc.shape)
        else:
            v_data = st["exp_avg_sq"]
            v_sc = st["exp_avg_sq_scale"]
            if v_data.dtype not in (torch.uint8, torch.float8_e4m3fn):
                return _fail(f"v dtype {v_data.dtype} not FP8 E4M3")
            if v_data.shape != expected:
                return _fail(f"v shape {v_data.shape} != expected {expected}")
            if not v_data.is_contiguous():
                return _fail("v not contiguous")
            v_u8 = v_data.view(torch.uint8) if v_data.dtype != torch.uint8 else v_data
            v_sc_flat = v_sc.reshape(-1).contiguous()

            _rdep.eco_mv_accumulate(
                m_u8.data_ptr(),
                m_sc_flat.data_ptr(),
                v_u8.data_ptr(),
                v_sc_flat.data_ptr(),
                grad.data_ptr(),
                E, in_dim, out_dim,
                beta1_frac, beta2_frac,
                stream,
            )
            st["exp_avg_scale"] = m_sc_flat.reshape(m_sc.shape)
            st["exp_avg_sq_scale"] = v_sc_flat.reshape(v_sc.shape)

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
            state["exp_avg"] = torch.zeros(shape, dtype=torch.float8_e5m2, device=device)
            state["exp_avg_scale"] = torch.ones(scale_shape, device=device, dtype=torch.float32) * 1e-30

            if self._factored_v:
                v_row_shape = shape[:-1]   # (E, in_dim)
                v_col_shape = shape[:-2] + shape[-1:]  # (E, out_dim)
                state["v_row"] = torch.zeros(v_row_shape, device=device, dtype=torch.float32)
                state["v_col"] = torch.zeros(v_col_shape, device=device, dtype=torch.float32)
            else:
                state["exp_avg_sq"] = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
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

    def _cuda_fused_update(
        self, moe: nn.Module, param_name: str, grad: torch.Tensor, st: dict,
        beta1_eff: float | None = None, beta2_eff: float | None = None,
    ) -> bool:
        """Run the fused CUDA kernel for the full ECO AdamW step.

        The CUDA kernel (eco_adam.cu) does the entire ECO AdamW step — dequant
        NVFP4 → AdamW → SR → ECO error → requant — in registers/shared memory,
        with zero FP32 global memory materialization (~7 bytes/element vs ~68).

        Raises RuntimeError if requirements are not met.
        Returns False if the kernel cannot handle the given shapes.
        """
        b1 = beta1_eff if beta1_eff is not None else self.beta1
        b2 = beta2_eff if beta2_eff is not None else self.beta2

        _rdep = getattr(self, '_rdep', None)
        if _rdep is None:
            if self._require_cuda:
                raise RuntimeError("CUDA kernel required but _rdep not initialized")
            return False

        def _fail(msg):
            if self._require_cuda:
                raise RuntimeError(f"eco_adam_nvfp4_update CUDA kernel: {msg}")
            return False

        packed = getattr(moe, f'_{param_name}_packed')
        scale_buf = getattr(moe, f'_{param_name}_scale')
        gs_buf = getattr(moe, f'_{param_name}_gs')
        group_size = getattr(moe, '_nvfp4_group_size', 16)

        E = packed.shape[0]
        out_dim = packed.shape[1]
        in_dim = packed.shape[2] * 2

        if (out_dim & 31) != 0 or (in_dim & 31) != 0:
            return _fail(f"dims not 32-aligned: out_dim={out_dim}, in_dim={in_dim}")
        if in_dim % group_size != 0:
            return _fail(f"in_dim={in_dim} not divisible by group_size={group_size}")

        m_data = st["exp_avg"]
        m_sc = st["exp_avg_scale"]
        if m_data.dtype not in (torch.uint8, torch.float8_e5m2):
            return _fail(f"m dtype {m_data.dtype} not FP8 E5M2")

        expected = (E, in_dim, out_dim)
        if grad.shape != expected:
            return _fail(f"grad shape {grad.shape} != expected {expected}")
        if not grad.is_contiguous():
            grad = grad.contiguous()
        if m_data.shape != expected:
            return _fail(f"m shape {m_data.shape} != expected {expected}")
        if not m_data.is_contiguous():
            return _fail("m not contiguous")

        m_u8 = m_data.view(torch.uint8) if m_data.dtype != torch.uint8 else m_data
        m_sc_flat = m_sc.reshape(-1).contiguous()

        _param_offset = {'W1': 0, 'W2': 1, 'W3': 2}[param_name]
        idx = self._moe_to_idx[id(moe)]
        prng_seed0 = (self._step_count * 1000003 + idx * 7 + _param_offset) & 0xFFFFFFFF
        prng_seed1 = (self._step_count * 7919 + idx * 31 + _param_offset * 127) & 0xFFFFFFFF

        scale_u8 = scale_buf.view(torch.uint8)
        stream = torch.cuda.current_stream(packed.device)

        if self._factored_v:
            v_row = st["v_row"]
            v_col = st["v_col"]
            if v_row.shape != (E, in_dim) or v_col.shape != (E, out_dim):
                return _fail(f"v_row/v_col shape mismatch")
            if not v_row.is_contiguous() or not v_col.is_contiguous():
                return _fail("v_row/v_col not contiguous")
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
                self.lr, b1, b2,
                self.weight_decay, self.eps,
                self._step_size, self._inv_bc2_sqrt,
                self._eco_alpha,
                1 if self._stochastic_rounding else 0,
                1 if self._error_feedback else 0,
                prng_seed0, prng_seed1,
                stream,
            )
            st["exp_avg_scale"] = m_sc_flat.reshape(m_sc.shape)

        else:
            v_data = st["exp_avg_sq"]
            v_sc = st["exp_avg_sq_scale"]
            if v_data.dtype not in (torch.uint8, torch.float8_e4m3fn):
                return _fail(f"v dtype {v_data.dtype} not FP8 E4M3")
            if v_data.shape != expected:
                return _fail(f"v shape {v_data.shape} != expected {expected}")
            if not v_data.is_contiguous():
                return _fail("v not contiguous")
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
                self.lr, b1, b2,
                self.weight_decay, self.eps,
                self._step_size, self._inv_bc2_sqrt,
                self._eco_alpha,
                1 if self._stochastic_rounding else 0,
                1 if self._error_feedback else 0,
                prng_seed0, prng_seed1,
                stream,
            )
            st["exp_avg_scale"] = m_sc_flat.reshape(m_sc.shape)
            st["exp_avg_sq_scale"] = v_sc_flat.reshape(v_sc.shape)

        return True

    @torch.no_grad()
    def fused_update(self, moe: nn.Module, param_name: str, grad_bf16: torch.Tensor):
        """Apply one ECO AdamW step for a single expert weight, in-place.

        Called from _MoEBlockscaledFused.backward() immediately after computing
        the weight gradient. Consumes the gradient — caller should free it after.

        Uses Adam Accumulation (AdamA, arXiv:2305.19982) for gradient accumulation:
        instead of maintaining separate FP8 gradient buffers (~5.85 GiB), we
        accumulate directly into the existing FP8 m/v optimizer states using
        fractional betas β₁^(1/K) and β₂^(1/K) where K = accum_steps.

        After K micro-steps with fractional betas, the result is mathematically
        equivalent to standard Adam with the full β₁, β₂:
          m_K = β₁^(1/K) · m_{K-1} + (1 - β₁^(1/K)) · g_K  (repeated K times)
        is equivalent to:
          m = β₁ · m + (1 - β₁) · g̅  (where g̅ is the mean gradient)

        Bias correction is unchanged: 1 - (β^(1/K))^(t·K) = 1 - β^t.

        When gradient accumulation is active (accum_steps > 1):
        - Non-final micro-steps: AllReduce + update m/v only (no weight update).
          Zero additional memory — reuses existing FP8 optimizer states.
        - Final micro-step: AllReduce + full ECO Adam step with fractional betas.

        Dispatches to the fused CUDA kernel (zero FP32 materialization).
        Raises RuntimeError if the kernel fails.

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
        # AllReduce happens every micro-step (each micro-batch's gradient is averaged
        # across DP ranks before accumulation, matching the semantics of averaging
        # a larger batch).
        grad = grad_bf16.float()
        del grad_bf16
        if self._dp_size > 1 and self._dp_group is not None:
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=self._dp_group)

        # AdamA fractional betas: β^(1/K) for K micro-steps.
        # When K=1 (no accumulation), β^(1/1) = β — standard Adam.
        K = self._accum_steps
        beta1_frac = self.beta1 ** (1.0 / K)
        beta2_frac = self.beta2 ** (1.0 / K)

        # Get or init optimizer state (needed for both accumulate and full step)
        st = self._get_or_init_state(moe, param_name)

        # Non-final micro-steps: update m/v with fractional betas only, no weight update.
        if self.is_accumulating:
            self._cuda_mv_accumulate(moe, param_name, grad, st,
                                     beta1_frac, beta2_frac)
            del grad
            return

        # Final micro-step (or no accumulation): full ECO Adam step.
        # The fractional betas make the kernel's EMA update equivalent to standard
        # Adam over the accumulated K micro-steps.
        st["step"] = torch.tensor(float(self._step_count), dtype=torch.float32)

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

        # CUDA fused kernel with fractional betas (zero FP32 materialization)
        if self._cuda_fused_update(moe, param_name, grad, st,
                                    beta1_eff=beta1_frac, beta2_eff=beta2_frac):
            del grad
            return

        raise RuntimeError(
            "BUG: _cuda_fused_update returned False. "
            "CUDA kernel is required for production (eco_require_cuda=True)."
        )

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
