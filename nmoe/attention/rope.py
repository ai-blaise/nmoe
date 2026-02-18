import math

import torch
from torch import nn

from nmoe.csrc import rdep as _C


_rope_buf: dict[tuple, torch.Tensor] = {}


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

    # Reuse cached output buffer to avoid per-call allocation
    key = (x.shape, x.dtype, x.device)
    buf = _rope_buf.get(key)
    if buf is None or buf.shape != x.shape:
      buf = torch.empty_like(x)
      _rope_buf[key] = buf
    out = buf

    total_vecs = B * T * H
    stream = torch.cuda.current_stream()

    _C.fused_rope_forward(
      x.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(), out.data_ptr(),
      total_vecs, T, H, D,
      stream,
    )

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
    grad_x = torch.empty_like(grad_output)
    total_vecs = grad_output.numel() // D
    stream = torch.cuda.current_stream()

    _C.fused_rope_backward(
      grad_output.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(), grad_x.data_ptr(),
      total_vecs, T, H, D,
      stream,
    )
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

    total_vecs = B * T * H
    stream = torch.cuda.current_stream()

    _C.fused_rope_forward_partial(
      x.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(),
      total_vecs, T, H, D, nope_dim,
      stream,
    )

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
    total_vecs = grad_output.numel() // D
    stream = torch.cuda.current_stream()

    _C.fused_rope_backward_partial(
      grad_output.data_ptr(), cos_2d.data_ptr(), sin_2d.data_ptr(),
      total_vecs, T, H, D, nope_dim,
      stream,
    )
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
  x_contig = x.contiguous()
  seq_len = x.size(1)
  rope_dim = x.size(-1) - nope_dim
  half_dim = rope_dim // 2

  # Slice cos/sin to [seq_len, half_dim] and ensure contiguous BF16
  cos_2d = cos[:seq_len, :half_dim].contiguous()
  sin_2d = sin[:seq_len, :half_dim].contiguous()
  if cos_2d.dtype != torch.bfloat16:
    cos_2d = cos_2d.to(torch.bfloat16)
  if sin_2d.dtype != torch.bfloat16:
    sin_2d = sin_2d.to(torch.bfloat16)

  return _FusedRoPEPartialFunction.apply(x_contig, cos_2d, sin_2d, nope_dim)


def rotate_pe(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """Apply rotary position embedding to tensor x using a fused CUDA kernel.

  cos/sin should be 2D [S, D] (standard sequential positions).  A head dimension
  is inserted internally for broadcasting against x: [B, S, H, D].

  Uses a cached output buffer to avoid allocation on every call (called once per
  layer, ~61 layers per forward pass).  The cache is keyed on (shape, dtype, device)
  so buffer reuse is safe across layers with identical tensor geometry.

  The fused CUDA kernel replaces ~8 PyTorch kernel launches with a single launch
  (BF16 I/O, FP32 compute).
  """
  x_contig = x.contiguous()
  seq_len = x.size(1)

  # Slice cos/sin to [seq_len, half_dim] and ensure contiguous BF16
  cos_2d = cos[:seq_len, :].contiguous()
  sin_2d = sin[:seq_len, :].contiguous()
  if cos_2d.dtype != torch.bfloat16:
    cos_2d = cos_2d.to(torch.bfloat16)
  if sin_2d.dtype != torch.bfloat16:
    sin_2d = sin_2d.to(torch.bfloat16)

  return _FusedRoPEFunction.apply(x_contig, cos_2d, sin_2d)


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
      d_half = self.head_dim / 2
      low  = (d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base))
      high = (d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base))
      assert 0 < low < high < d_half - 1
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
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin
