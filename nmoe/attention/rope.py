import os
import math

import torch
from torch import nn

from nmoe.csrc import rdep as _C


def _env_flag(name: str, default: str = "0") -> bool:
  return os.getenv(name, default) in ("1", "true", "True")


def _nan_debug_active() -> bool:
  return _env_flag("NMOE_NAN_DEBUG_ACTIVE", "0") or _env_flag("NMOE_MLA_FINITE_DEBUG", "0")


def _require_finite(tag: str, tensor: torch.Tensor) -> None:
  if not _nan_debug_active():
    return
  if not (tensor.is_floating_point() or tensor.is_complex()):
    return
  finite_mask = torch.isfinite(tensor)
  if bool(finite_mask.all().item()):
    return
  numel = int(tensor.numel())
  finite = int(finite_mask.sum().item())
  nan = int(torch.isnan(tensor).sum().item())
  inf = int(torch.isinf(tensor).sum().item())
  raise RuntimeError(
    f"[NANDBG][rope] non-finite {tag} "
    f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
    f"finite={finite}/{numel} nan={nan} inf={inf}"
  )


class _FusedRoPEFunction(torch.autograd.Function):
  """Autograd wrapper for the fused CUDA RoPE kernel."""

  @staticmethod
  def forward(ctx, x: torch.Tensor, cos_2d: torch.Tensor, sin_2d: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x:      [B, T, H, D] contiguous BF16 tensor
      cos_2d: [T, half_dim] contiguous BF16 tensor (already sliced to seq_len)
      sin_2d: [T, half_dim] contiguous BF16 tensor (already sliced to seq_len)
    """
    B, T, H, D = x.shape
    _require_finite("full.forward.x", x)

    # Do not reuse output storage across calls: aliasing can silently overwrite
    # earlier layer outputs before downstream consumers read them.
    out = torch.empty_like(x)

    total_vecs = B * T * H
    stream = torch.cuda.current_stream(x.device)

    _C.fused_rope_forward(
      x.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(), out.data_ptr(),
      total_vecs, T, H, D,
      stream,
    )
    _require_finite("full.forward.out", out)

    ctx.save_for_backward(cos_2d, sin_2d)
    ctx.T = T
    ctx.H = H
    ctx.D = D
    return out

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):
    cos_2d, sin_2d = ctx.saved_tensors
    T, H, D = ctx.T, ctx.H, ctx.D

    grad_output = grad_output.contiguous()
    _require_finite("full.backward.grad_out", grad_output)
    grad_x = torch.empty_like(grad_output)
    total_vecs = grad_output.numel() // D
    stream = torch.cuda.current_stream(grad_output.device)

    _C.fused_rope_backward(
      grad_output.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(), grad_x.data_ptr(),
      total_vecs, T, H, D,
      stream,
    )
    _require_finite("full.backward.grad_x", grad_x)
    return grad_x, None, None


class _FusedRoPEPartialFunction(torch.autograd.Function):
  """Autograd wrapper for in-place partial RoPE CUDA kernel.

  Applies RoPE only to elements [nope_dim:head_dim] within each head,
  leaving elements [0:nope_dim] unchanged. Operates IN-PLACE.

  This eliminates the need for torch.split + rotate_pe + torch.cat
  (3 kernels -> 1 kernel), saving 4 kernel launches per MLA layer
  (2 for Q, 2 for K) = 244 kernel launches per step for 61 layers.
  """

  @staticmethod
  def forward(ctx, x: torch.Tensor, cos_2d: torch.Tensor, sin_2d: torch.Tensor, nope_dim: int) -> torch.Tensor:
    """
    Args:
      x:       [B, T, H, D] contiguous BF16 tensor (modified IN-PLACE)
      cos_2d:  [T, half_dim] contiguous BF16 tensor (half_dim = (D - nope_dim) / 2)
      sin_2d:  [T, half_dim] contiguous BF16 tensor
      nope_dim: number of elements at start of head_dim to leave unchanged
    """
    B, T, H, D = x.shape
    _require_finite("partial.forward.x", x)

    total_vecs = B * T * H
    stream = torch.cuda.current_stream(x.device)

    _C.fused_rope_forward_partial(
      x.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(),
      total_vecs, T, H, D, nope_dim,
      stream,
    )
    _require_finite("partial.forward.x_out", x)

    ctx.mark_dirty(x)
    ctx.save_for_backward(cos_2d, sin_2d)
    ctx.T = T
    ctx.H = H
    ctx.D = D
    ctx.nope_dim = nope_dim
    return x

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):
    cos_2d, sin_2d = ctx.saved_tensors
    T, H, D = ctx.T, ctx.H, ctx.D
    nope_dim = ctx.nope_dim

    # grad_output is already the right shape, apply in-place backward
    grad_output = grad_output.contiguous()
    _require_finite("partial.backward.grad_out", grad_output)
    total_vecs = grad_output.numel() // D
    stream = torch.cuda.current_stream(grad_output.device)

    _C.fused_rope_backward_partial(
      grad_output.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(),
      total_vecs, T, H, D, nope_dim,
      stream,
    )
    _require_finite("partial.backward.grad_x", grad_output)
    return grad_output, None, None, None


def rotate_pe_partial(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, nope_dim: int) -> torch.Tensor:
  """Apply rotary position embedding IN-PLACE to the rope portion of tensor x.

  Only elements [nope_dim:head_dim] are rotated. Elements [0:nope_dim] are unchanged.
  This eliminates the need for torch.split + rotate_pe + torch.cat (3 kernels -> 1).

  Args:
    x:        [B, S, H, D] contiguous BF16 tensor (modified IN-PLACE)
    cos, sin: [max_seq_len, half_dim] where half_dim = (D - nope_dim) / 2
    nope_dim: number of elements at start of head_dim to leave unchanged

  Returns:
    x (modified in-place)
  """
  if x.dtype != torch.bfloat16:
    raise RuntimeError(f"fused_rope_forward_partial requires BF16 x, got {x.dtype}.")
  if not x.is_contiguous():
    raise RuntimeError("fused_rope_forward_partial requires contiguous x tensor.")
  seq_len = x.size(1)
  rope_dim = x.size(-1) - nope_dim
  half_dim = rope_dim // 2
  if (half_dim % 2) != 0:
    raise RuntimeError(
      f"fused_rope_forward_partial requires even half_dim for vectorized path; "
      f"got rope_dim={rope_dim}, half_dim={half_dim}"
    )

  if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
    raise RuntimeError("fused_rope_forward_partial requires BF16 cos/sin tensors.")
  if cos.ndim == 2 and sin.ndim == 2:
    cos_2d = cos[:seq_len, :half_dim]
    sin_2d = sin[:seq_len, :half_dim]
    if not cos_2d.is_contiguous() or not sin_2d.is_contiguous():
      raise RuntimeError("fused_rope_forward_partial requires contiguous sliced cos/sin tensors.")
    return _FusedRoPEPartialFunction.apply(x, cos_2d, sin_2d, nope_dim)

  if cos.ndim == 3 and sin.ndim == 3:
    if cos.shape != sin.shape:
      raise RuntimeError("fused_rope_forward_partial requires matching 3D cos/sin shapes.")
    if cos.shape[0] != x.shape[0] or cos.shape[1] != seq_len or cos.shape[2] != half_dim:
      raise RuntimeError(
        f"fused_rope_forward_partial 3D cos/sin must match [B,S,half_dim]={x.shape[0], seq_len, half_dim}, "
        f"got {tuple(cos.shape)}."
      )
    if not cos.is_contiguous() or not sin.is_contiguous():
      raise RuntimeError("fused_rope_forward_partial requires contiguous 3D cos/sin tensors.")
    x_flat = x.view(1, x.shape[0] * seq_len, x.shape[2], x.shape[3])
    cos_2d = cos.reshape(x.shape[0] * seq_len, half_dim)
    sin_2d = sin.reshape(x.shape[0] * seq_len, half_dim)
    out_flat = _FusedRoPEPartialFunction.apply(x_flat, cos_2d, sin_2d, nope_dim)
    return out_flat.view_as(x)

  raise RuntimeError(
    f"fused_rope_forward_partial requires 2D or 3D cos/sin tensors, got cos.ndim={cos.ndim}, sin.ndim={sin.ndim}."
  )


def rotate_pe(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """Apply rotary position embedding to tensor x using a fused CUDA kernel.

  cos/sin should be 2D [S, D] (standard sequential positions).  A head dimension
  is inserted internally for broadcasting against x: [B, S, H, D].

  The fused CUDA kernel replaces ~8 PyTorch kernel launches with a single launch
  (BF16 I/O, FP32 compute).
  """
  if x.dtype != torch.bfloat16:
    raise RuntimeError(f"fused_rope_forward requires BF16 x, got {x.dtype}.")
  if not x.is_contiguous():
    raise RuntimeError("fused_rope_forward requires contiguous x tensor.")
  seq_len = x.size(1)
  half_dim = x.size(-1) // 2
  if (half_dim % 2) != 0:
    raise RuntimeError(
      f"fused_rope_forward requires even half_dim for vectorized path; "
      f"got head_dim={x.size(-1)}, half_dim={half_dim}"
    )

  if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
    raise RuntimeError("fused_rope_forward requires BF16 cos/sin tensors.")
  if cos.ndim == 2 and sin.ndim == 2:
    cos_2d = cos[:seq_len, :]
    sin_2d = sin[:seq_len, :]
    if not cos_2d.is_contiguous() or not sin_2d.is_contiguous():
      raise RuntimeError("fused_rope_forward requires contiguous sliced cos/sin tensors.")
    return _FusedRoPEFunction.apply(x, cos_2d, sin_2d)

  if cos.ndim == 3 and sin.ndim == 3:
    if cos.shape != sin.shape:
      raise RuntimeError("fused_rope_forward requires matching 3D cos/sin shapes.")
    if cos.shape[0] != x.shape[0] or cos.shape[1] != seq_len or cos.shape[2] != half_dim:
      raise RuntimeError(
        f"fused_rope_forward 3D cos/sin must match [B,S,half_dim]={x.shape[0], seq_len, half_dim}, "
        f"got {tuple(cos.shape)}."
      )
    if not cos.is_contiguous() or not sin.is_contiguous():
      raise RuntimeError("fused_rope_forward requires contiguous 3D cos/sin tensors.")
    x_flat = x.view(1, x.shape[0] * seq_len, x.shape[2], x.shape[3])
    cos_2d = cos.reshape(x.shape[0] * seq_len, half_dim)
    sin_2d = sin.reshape(x.shape[0] * seq_len, half_dim)
    out = _FusedRoPEFunction.apply(x_flat, cos_2d, sin_2d)
    return out.view_as(x)

  raise RuntimeError(
    f"fused_rope_forward requires 2D or 3D cos/sin tensors, got cos.ndim={cos.ndim}, sin.ndim={sin.ndim}."
  )


class RotaryEmbedding(nn.Module):
  def __init__(
    self,
    head_dim: int,
    base: int,
    dtype: torch.dtype,
    initial_context_length: int = 4096,
    max_context_length: int = 131072,
    scaling_factor: float = 1.0,
    ntk_alpha: float = 1.0,
    ntk_beta: float = 32.0,
    device: torch.device | None = None,
  ) -> None:
    super().__init__()
    self.head_dim = head_dim
    self.base = base
    self.dtype = dtype
    self.initial_context_length = initial_context_length
    self.max_context_length = max_context_length
    self.scaling_factor = scaling_factor
    self.ntk_alpha = ntk_alpha
    self.ntk_beta = ntk_beta
    self.device = device
    cos, sin = self._compute_cos_sin(0, self.max_context_length)
    # Register as buffers so they move with model.cuda() and pre-cast to target dtype
    self.register_buffer('cos', cos.to(dtype), persistent=False)
    self.register_buffer('sin', sin.to(dtype), persistent=False)

  def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
    freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) / self.head_dim)
    if self.scaling_factor > 1.0:
      concentration = (0.1 * math.log(self.scaling_factor) + 1.0)
      d_half = self.head_dim // 2
      low  = (d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base))
      high = (d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base))
      if not (0 < low < high < d_half - 1):
        raise RuntimeError(
          f"Invalid NTK scaling window: low={low}, high={high}, d_half={d_half}."
        )
      interpolation = 1.0 / (self.scaling_factor * freq)
      extrapolation = 1.0 / freq
      ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
      mask = 1 - ramp.clamp(0, 1)
      inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
      concentration = 1.0
      inv_freq = 1.0 / freq
    return concentration, inv_freq

  def _compute_cos_sin(self, start: int, num_tokens: int):
    concentration, inv_freq = self._compute_concentration_and_inv_freq()
    t = torch.arange(start, start + num_tokens, dtype=torch.float32, device=self.device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin
