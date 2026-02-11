import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from nmoe.config import Config
from nmoe.norm import RMSNorm
import nmoe.triton.kda as kda_k

try:
  import triton
  if hasattr(triton, "set_allocator"):
    def _alloc(size: int, alignment: int, stream: int):
      return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(_alloc)
except Exception as _triton_err:
  logging.getLogger(__name__).warning("Triton import failed: %s. Triton attention kernels unavailable.", _triton_err)


class KDA(nn.Module):
  """Chunked Decoupled Attention (GLA-style) using vendored Triton kernels."""

  def __init__(self, config: Config):
    super().__init__()
    self.dim = config.dim
    self.n_heads = config.n_heads
    self.q_lora_rank = config.q_lora_rank
    self.kv_lora_rank = config.kv_lora_rank
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    self.v_head_dim = config.v_head_dim
    self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
    self.q_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
    self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False, dtype=torch.bfloat16)
    self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, dtype=torch.bfloat16)
    self.kv_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
    self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, dtype=torch.bfloat16)
    self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False, dtype=torch.bfloat16)
    self.wg = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=True, dtype=torch.bfloat16)
    self.wb = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)

  def init_weights(self, init_std: float = 0.02):
    for proj in [self.wq_a, self.wq_b, self.wkv_a, self.wkv_b, self.wg, self.wb]:
      nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
    if self.wg.bias is not None:
      nn.init.zeros_(self.wg.bias)
    nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
    self.q_norm.weight.data.fill_(1.0)
    self.kv_norm.weight.data.fill_(1.0)

  @record_function("attn")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              cu_seqlens: list[torch.Tensor] | None = None) -> torch.Tensor:
    """KDA forward with optional packed sequence support.

    When cu_seqlens is provided, sequences within each batch element are treated
    as independent documents using KDA's native cu_seqlens support. The input must
    be reshaped to [1, B*T, H, D] with a single flattened cu_seqlens tensor.
    For simplicity when packing is per-batch-element (not flattened), we pass
    cu_seqlens=None and rely on the per-position gating to handle document isolation.

    Args:
      x: Input tensor [B, T, D].
      cos, sin: RoPE rotation tensors.
      cu_seqlens: Optional list of B int32 tensors for packed sequences.
    """
    B, T, _ = x.size()
    q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, T, self.n_heads, self.qk_head_dim)
    kv = self.wkv_a(x)
    kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.unsqueeze(2)
    kv = self.wkv_b(self.kv_norm(kv))
    kv = kv.view(B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
    g = F.logsigmoid(self.wg(x)).view(B, T, self.n_heads, self.qk_head_dim)
    beta = torch.sigmoid(self.wb(x)).view(B, T, self.n_heads)

    # KDA natively supports cu_seqlens for variable-length inputs.
    # When packing is enabled, flatten batch to [1, B*T, ...] and construct
    # a unified cu_seqlens tensor from the per-element boundaries.
    kda_cu_seqlens = None
    if cu_seqlens is not None and B == 1:
      # Single-element batch with packing: pass cu_seqlens directly
      kda_cu_seqlens = cu_seqlens[0].to(torch.long)
    elif cu_seqlens is not None and B > 1:
      # Multi-element batch: flatten to [1, B*T, ...] and merge cu_seqlens
      # Each batch element's cu_seqlens is offset by the element's position
      merged = [torch.tensor([0], dtype=torch.long, device=x.device)]
      offset = 0
      for b_idx in range(B):
        cu = cu_seqlens[b_idx]
        # Skip the initial 0 and add offset
        merged.append(cu[1:].to(torch.long) + offset)
        offset += T
      kda_cu_seqlens = torch.cat(merged, dim=0)
      q = q.reshape(1, B * T, self.n_heads, self.qk_head_dim)
      k = k.reshape(1, B * T, self.n_heads, self.qk_head_dim)
      v = v.reshape(1, B * T, self.n_heads, self.v_head_dim)
      g = g.reshape(1, B * T, self.n_heads, self.qk_head_dim)
      beta = beta.reshape(1, B * T, self.n_heads)

    with record_function("attn.kernel[kda]"):
      o, _ = kda_k.chunk_kda(
        q, k, v, g, beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=kda_cu_seqlens,
      )

    if cu_seqlens is not None and B > 1:
      o = o.reshape(B, T, self.n_heads, self.v_head_dim)

    o = torch.nan_to_num(o, nan=0.0, posinf=1e4, neginf=-1e4)
    o = o.contiguous().view(B, T, self.n_heads * self.v_head_dim)
    return self.wo(o)
