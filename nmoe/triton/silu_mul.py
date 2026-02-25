"""Fused SiLU(x) * y Triton kernel with backward.

This replaces two elementwise kernels:
  1) hidden = silu(x)
  2) hidden = hidden * y
with one fused pass in both forward and backward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
  import triton
  import triton.language as tl
except Exception:  # pragma: no cover - runtime fallback when Triton is unavailable
  triton = None
  tl = None


if triton is not None:
  @triton.jit
  def _silu_mul_fwd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
  ):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(x)
    out = (x * sig) * y
    tl.store(out_ptr + offs, out, mask=mask)


  @triton.jit
  def _silu_mul_bwd_kernel(
    x_ptr,
    y_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
  ):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(x)
    silu = x * sig
    dsilu = sig * (1.0 + x * (1.0 - sig))

    grad_x = grad_out * y * dsilu
    grad_y = grad_out * silu

    tl.store(grad_x_ptr + offs, grad_x, mask=mask)
    tl.store(grad_y_ptr + offs, grad_y, mask=mask)


class _SiluMulFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if (
      triton is None
      or not x.is_cuda
      or not y.is_cuda
      or x.dtype != y.dtype
      or x.shape != y.shape
    ):
      ctx.save_for_backward(x, y)
      return F.silu(x) * y

    if not x.is_contiguous() or not y.is_contiguous():
      x = x.contiguous()
      y = y.contiguous()

    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _silu_mul_fwd_kernel[grid](
      x,
      y,
      out,
      n_elements,
      BLOCK=1024,
      num_warps=4,
      num_stages=2,
    )
    ctx.save_for_backward(x, y)
    return out

  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    x, y = ctx.saved_tensors
    if (
      triton is None
      or not grad_out.is_cuda
      or not x.is_cuda
      or not y.is_cuda
      or grad_out.dtype != x.dtype
    ):
      sig = torch.sigmoid(x)
      silu = x * sig
      dsilu = sig * (1.0 + x * (1.0 - sig))
      grad_x = grad_out * y * dsilu
      grad_y = grad_out * silu
      return grad_x, grad_y

    if not grad_out.is_contiguous():
      grad_out = grad_out.contiguous()

    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)
    n_elements = grad_out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _silu_mul_bwd_kernel[grid](
      x,
      y,
      grad_out,
      grad_x,
      grad_y,
      n_elements,
      BLOCK=1024,
      num_warps=4,
      num_stages=2,
    )
    return grad_x, grad_y


def silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Compute SiLU(x) * y using a fused Triton kernel when available."""
  if x.is_cuda and y.is_cuda and triton is None:
    raise RuntimeError(
      "fuse_mlp_silu_mul=true requires Triton to be installed for CUDA execution."
    )
  if x.shape != y.shape:
    return F.silu(x) * y
  return _SiluMulFunction.apply(x, y)
