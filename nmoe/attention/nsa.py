from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from nmoe.attention.rope import rotate_pe
from nmoe.config import Config
from nmoe.triton.nsa import cmp_attention
import nmoe.triton.swa as swa_k

try:
  _flex_attention = torch.compile(flex_attention)
except Exception as _e:
  raise RuntimeError(
    "torch.compile(flex_attention) failed for NSA. "
    "Eager fallback is disabled on production no-fallback path."
  ) from _e


class NSA(nn.Module):
  """Native Sparse Attention (arXiv:2502.11089).

  Paper-faithful implementation with three attention paths:
    - CMP: Compression path with learnable MLP φ over overlapping blocks (Eq 7)
    - SLC: Selection path using CMP attention scores for block ranking (Eq 8-12)
    - SWA: Sliding window for local context

  Gates are sigmoid per paper (Eq 5), not softmax.

  Key equations from paper:
    Eq 5:  o*_t = Σ_{c∈C} g_t^c · Attn(q_t, K̃_t^c, Ṽ_t^c)  [gating]
    Eq 7:  K̃_t^cmp = {φ(k_{id+1:id+l}) | 0 ≤ i ≤ ⌊(t-l)/d⌋}  [compression]
    Eq 8:  p_t^cmp = Softmax(q_t^T · K̃_t^cmp)  [importance scoring]
    Eq 9:  p_t^slc[j] = Σ_m Σ_n p_t^cmp[(l'/d)j - m - n]  [score aggregation]
    Eq 10: p'_t^slc = Σ_{h=1}^H p_t^slc,(h)  [GQA head aggregation]
    Eq 11: I_t = {i | rank(p'_t^slc[i]) ≤ n}  [top-n selection]
    Eq 12: K̃_t^slc = Cat[{k_{il'+1:(i+1)l'} | i ∈ I_t}]  [block concatenation]

  Parameters: l=32 (cmp_len), d=16 (cmp_stride), l'=64 (slc_block), n=16 (topk), w=512 (window)
  """

  def __init__(self, config: Config):
    super().__init__()
    self.n_heads = config.n_heads
    self.head_dim = config.v_head_dim
    qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if qk_dim != self.head_dim:
      raise RuntimeError(f"NSA requires qk_head_dim == v_head_dim (got {qk_dim} vs {self.head_dim})")

    nsa_opts = getattr(config, "attn_nsa", {}) or {}
    swa_opts = getattr(config, "attn_swa", {}) or {}
    self.n_kv_heads = int(nsa_opts.get("kv_heads", swa_opts.get("kv_heads", self.n_heads)))
    if self.n_heads % self.n_kv_heads != 0:
      raise RuntimeError(f"NSA requires H % KV == 0 (H={self.n_heads}, KV={self.n_kv_heads})")
    self.repeat_kv = self.n_heads // self.n_kv_heads

    # Projections
    out_feats = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
    self.qkv = nn.Linear(config.dim, out_feats, bias=True, dtype=torch.bfloat16)
    self.out = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=True, dtype=torch.bfloat16)
    self.sinks = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.bfloat16))

    # Paper params: l=32 (block length), d=16 (stride), l'=64 (selection block), n=16 (topk)
    self.cmp_len = int(nsa_opts.get("cmp_len", 32))       # l: compression block length
    self.cmp_stride = int(nsa_opts.get("cmp_stride", 16)) # d: stride (d < l means overlap)
    self.slc_block = int(nsa_opts.get("slc_block", 64))   # l': selection block size
    self.topk = int(nsa_opts.get("topk_blocks", 16))      # n: number of selected blocks
    self.tile_q = int(nsa_opts.get("tile_q", 128))
    self.flex_block = int(nsa_opts.get("flex_block", 128))
    self.window = 0  # set by factory per-layer
    self.softmax_scale = self.head_dim ** -0.5

    # Compression MLP φ with intra-block position encoding (Eq 7)
    cmp_hidden = int(nsa_opts.get("cmp_hidden", self.head_dim * 2))
    self.cmp_pos = nn.Parameter(torch.zeros(self.cmp_len, self.head_dim, dtype=torch.bfloat16))
    self.cmp_mlp = nn.Sequential(
      nn.Linear(self.cmp_len * self.head_dim, cmp_hidden, bias=True, dtype=torch.bfloat16),
      nn.GELU(),
      nn.Linear(cmp_hidden, self.head_dim, bias=True, dtype=torch.bfloat16),
    )

    # Gate MLP with sigmoid (Eq 5) - produces [cmp, slc, swa] weights in [0,1]
    # Input is raw features x (dim), not query heads
    gate_hidden = int(nsa_opts.get("gate_hidden", config.dim // 4))
    self.gate_mlp = nn.Sequential(
      nn.Linear(config.dim, gate_hidden, bias=True, dtype=torch.bfloat16),
      nn.GELU(),
      nn.Linear(gate_hidden, 3, bias=True, dtype=torch.bfloat16),
    )

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.qkv.bias)
    nn.init.trunc_normal_(self.out.weight, mean=0.0, std=init_std)
    nn.init.zeros_(self.out.bias)
    self.sinks.data.zero_()
    # Compression MLP
    nn.init.trunc_normal_(self.cmp_pos, mean=0.0, std=0.02)
    for layer in self.cmp_mlp:
      if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(layer.bias)
    # Gate MLP - bias final layer towards SWA: sigmoid([−2.2, −2.2, 1.4]) ≈ [0.1, 0.1, 0.8]
    for layer in self.gate_mlp:
      if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(layer.bias)
    self.gate_mlp[-1].bias.data.copy_(torch.tensor([-2.2, -2.2, 1.4], dtype=torch.bfloat16))

  def _compress(self, x: torch.Tensor) -> torch.Tensor:
    """Apply learnable compression φ over overlapping blocks.

    Eq 7: K̃_t^cmp = {φ(k_{id+1:id+l}) | 0 ≤ i ≤ ⌊(t-l)/d⌋}

    Where:
      - l = cmp_len (32): block length
      - d = cmp_stride (16): stride between blocks (d < l means overlap)
      - φ = learnable MLP with intra-block position encoding

    Args:
      x: [B, T, H, D] - K or V tensor

    Returns:
      [B, Tc, H, D] where Tc = ⌊(T-l)/d⌋ + 1
    """
    B, T, H, D = x.shape
    l, d = self.cmp_len, self.cmp_stride

    if T < l:
      return x.new_zeros(B, 0, H, D)

    Tc = (T - l) // d + 1

    # Extract overlapping blocks via unfold: x[..., i*d : i*d+l] for i in [0, Tc)
    # x: [B, T, H, D] -> transpose for unfold on T
    x_t = x.permute(0, 2, 1, 3)  # [B, H, T, D]
    blocks = x_t.unfold(2, l, d)  # [B, H, Tc, D, l]
    blocks = blocks.permute(0, 2, 1, 4, 3)  # [B, Tc, H, l, D]

    # Add intra-block position encoding
    blocks = blocks + self.cmp_pos.view(1, 1, 1, l, D)

    # Flatten and apply MLP φ: [B*Tc*H, l*D] -> [B*Tc*H, D]
    blocks_flat = blocks.reshape(B * Tc * H, l * D)
    compressed = self.cmp_mlp(blocks_flat)

    return compressed.view(B, Tc, H, D)

  def _cmp_attention(self, q: torch.Tensor, k_cmp: torch.Tensor, v_cmp: torch.Tensor, T: int
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """CMP path: fused attention over compressed KV with block score aggregation.

    Uses Triton kernel that fuses:
      - Eq 8: p_t^cmp = Softmax(q_t^T · K̃_t^cmp) - attention computation
      - Eq 9: p_t^slc[j] = Σ aggregation to selection block scores

    Args:
      q: [B, T, H, D] - queries
      k_cmp, v_cmp: [B, Tc, H, D] - compressed K, V
      T: original sequence length (for computing Ns)

    Returns:
      output: [B, T, H, D] - attention output
      block_scores: [B, T, H, Ns] - aggregated selection block scores (Eq 9)
    """
    # Stability: the NSA CMP kernel's backward clamps all‑masked rows
    # (LSE = -inf) so exp(qk - LSE) is numerically safe (p=0), matching
    # forward's no‑valid‑keys behavior. See nmoe.triton.nsa::_cmp_attn_bwd.

    B, Tq, H, D = q.shape
    Tc = k_cmp.size(1)
    Ns = T // self.slc_block

    if Tc == 0 or Ns == 0:
      return q.new_zeros(B, Tq, H, D), q.new_zeros(B, Tq, H, max(Ns, 1))

    # Fused CMP attention + block score aggregation
    out, block_scores = cmp_attention(
      q, k_cmp, v_cmp,
      self.softmax_scale,
      self.cmp_len,
      self.cmp_stride,
      self.slc_block,
      Ns,
    )

    return out, block_scores

  @torch.no_grad()
  def _select_blocks(self, block_scores: torch.Tensor, T: int) -> torch.Tensor:
    """Select top-k blocks per query using pre-aggregated block scores.

    Implements Equations 10-12 from the paper (Eq 9 done in fused kernel):

    Eq 10: p'_t^slc = Σ_{h=1}^H p_t^slc,(h)
           Sums scores across all heads in GQA group for consistent block selection.

    Eq 11: I_t = {i | rank(p'_t^slc[i]) ≤ n}
           Selects top-n blocks by aggregated score.
           Paper: "n=16 selected blocks (including 1 fixed initial + 2 local blocks)"

    Eq 12: K̃_t^slc = Cat[{k_{il'+1:(i+1)l'} | i ∈ I_t}]
           Concatenates tokens from selected blocks (done in _slc_path).

    Args:
      block_scores: [B, T, H, Ns] - pre-aggregated selection block scores from CMP kernel
      T: sequence length

    Returns:
      block_idx: [B, T, H, topk] int32 - selected block indices at l' (slc_block) granularity
    """
    B, Tq, H, Ns = block_scores.shape
    l_prime = self.slc_block

    if Ns == 0:
      return block_scores.new_zeros(B, Tq, H, self.topk, dtype=torch.int32)

    # -------------------------------------------------------------------------
    # Eq 10: p'_t^slc = Σ_{h=1}^H p_t^slc,(h)
    # -------------------------------------------------------------------------
    # Sum scores across all heads in GQA group for consistent block selection.
    slc_scores_agg = block_scores.sum(dim=2, keepdim=True).expand_as(block_scores)  # [B, T, H, Ns]

    # Causal mask: query at position t can only select blocks j where (j+1)*l' <= t
    t_idx = torch.arange(Tq, device=block_scores.device)
    max_block = t_idx // l_prime  # [T] - exclusive upper bound (block containing t)
    j_idx = torch.arange(Ns, device=block_scores.device)
    causal_mask = j_idx[None, :] >= max_block[:, None]  # [T, Ns]
    slc_scores_agg = slc_scores_agg.masked_fill(causal_mask.view(1, Tq, 1, Ns), float('-inf'))

    # -------------------------------------------------------------------------
    # Eq 11: I_t = {i | rank(p'_t^slc[i]) ≤ n}
    # -------------------------------------------------------------------------
    # Paper: "n=16 selected blocks (including 1 fixed initial + 2 local blocks)"
    # Reserve 3 slots for fixed blocks, select top-(n-3) by score.
    n_fixed = 3  # 1 initial + 2 local
    n_dynamic = max(self.topk - n_fixed, 0)

    # Select top-k dynamic blocks by score
    k_dyn = min(n_dynamic, Ns)
    if k_dyn > 0:
      topk_dyn = slc_scores_agg.topk(k_dyn, dim=-1).indices  # [B, T, H, k_dyn]
    else:
      topk_dyn = block_scores.new_zeros(B, Tq, H, 0, dtype=torch.int64)

    # Build fixed block indices per query position:
    # - Block 0 (initial): always included
    # - Last 2 blocks before current position (local)
    # For query at position t, current block is t // l', local = [current-2, current-1]
    cur_block = t_idx // l_prime  # [T]
    local_1 = (cur_block - 1).clamp(min=0)  # [T]
    local_2 = (cur_block - 2).clamp(min=0)  # [T]
    initial = torch.zeros_like(cur_block)  # [T] - block 0

    # Per paper (n=16), reserve 3 fixed blocks per query: initial (0)
    # and the last two local blocks; the remaining (topk-3) come from scores.
    # Stack fixed blocks: [T, 3] -> expand to [B, T, H, 3]
    fixed_blocks = torch.stack([initial, local_2, local_1], dim=-1)  # [T, 3]
    fixed_blocks = fixed_blocks.view(1, Tq, 1, n_fixed).expand(B, Tq, H, n_fixed)

    # Concatenate fixed + dynamic
    all_blocks = torch.cat([fixed_blocks, topk_dyn], dim=-1)  # [B, T, H, n_fixed + k_dyn]

    # Pad to topk if needed
    current_len = all_blocks.size(-1)
    if current_len < self.topk:
      pad = all_blocks.new_zeros(B, Tq, H, self.topk - current_len)
      all_blocks = torch.cat([all_blocks, pad], dim=-1)
    elif current_len > self.topk:
      all_blocks = all_blocks[..., :self.topk]

    # Expand from KV heads to full heads for SLC path (consistent selection per Eq 10)
    # H here is n_kv_heads, expand to n_heads
    all_blocks = all_blocks.repeat_interleave(self.repeat_kv, dim=2)  # [B, T, n_heads, topk]

    return all_blocks.to(torch.int32)

  def _slc_path(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                block_idx: torch.Tensor) -> torch.Tensor:
    """SLC path: sparse block attention via FlexAttention.

    Eq 12: K̃_t^slc = Cat[{k_{il'+1:(i+1)l'} | i ∈ I_t}]

    Single FlexAttention call over full sequence. The mask encodes:
    - Own block: dense causal attention (query always attends within its block)
    - Previous blocks: sparse attention to selected blocks only

    Args:
      q: [B, T, H, D] - full queries (all heads)
      k, v: [B, T, KV, D] - K, V at KV-head granularity
      block_idx: [B, T, H, topk] - selected block indices from _select_blocks

    Returns:
      [B, T, H, D] - attention output over selected blocks
    """
    B, T, H, D = q.shape
    l_prime = self.slc_block
    num_blocks = (T + l_prime - 1) // l_prime

    # Expand K/V from KV-heads to full heads for GQA
    K = k.repeat_interleave(self.repeat_kv, dim=2)  # [B, T, H, D]
    V = v.repeat_interleave(self.repeat_kv, dim=2)  # [B, T, H, D]

    Q = q.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]
    K = K.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]
    V = V.permute(0, 2, 1, 3).contiguous()  # [B, H, T, D]

    # Build dense block selection mask: block_allowed[b, h, t, j] = True if query t selected block j
    # block_idx: [B, T, H, topk] -> block_allowed: [B, H, T, num_blocks]
    block_allowed = torch.zeros(B, T, H, num_blocks, dtype=torch.bool, device=q.device)
    block_idx_clamped = block_idx.clamp(0, num_blocks - 1).long()
    block_allowed.scatter_(3, block_idx_clamped, True)
    block_allowed = block_allowed.permute(0, 2, 1, 3).contiguous()  # [B, H, T, num_blocks]

    def mask_mod(b, h, q_idx, kv_idx, _allowed=block_allowed, _l=l_prime, _nb=num_blocks):
      q_block = q_idx // _l
      kv_block = kv_idx // _l
      kv_block_clamped = kv_block.clamp(max=_nb - 1)

      # Own block: always allow (dense causal within block)
      own_block = (kv_block == q_block)

      # Previous blocks: check if selected
      selected = _allowed[b, h, q_idx, kv_block_clamped]

      # Causal constraint
      causal_ok = kv_idx <= q_idx

      return causal_ok & (own_block | selected)

    # Use BLOCK_SIZE=128 for inductor compatibility
    bmask = create_block_mask(mask_mod, B=B, H=H, Q_LEN=T, KV_LEN=T,
                              device=q.device, BLOCK_SIZE=(128, 128))

    O = _flex_attention(
      Q, K, V,
      block_mask=bmask,
      scale=self.softmax_scale,
      enable_gqa=False,
      return_lse=False,
    )

    return O.permute(0, 2, 1, 3)  # [B, T, H, D]

  @record_function("attn")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.size()
    H, KV, D = self.n_heads, self.n_kv_heads, self.head_dim
    G = self.repeat_kv

    # Project and reshape
    qkv = self.qkv(x)
    q, k, v = qkv.split([H * D, KV * D, KV * D], dim=-1)
    q = rotate_pe(q.view(B, T, H, D), cos, sin)
    k = rotate_pe(k.view(B, T, KV, D), cos, sin)
    v = v.view(B, T, KV, D)

    # Reduce queries to KV-head granularity for CMP (GQA)
    qh = q.view(B, T, KV, G, D).mean(3)  # [B, T, KV, D]

    # Compress K and V with learnable MLP φ (Eq 7)
    with record_function("attn.kernel[nsa.cmp+slc]"):
      k_cmp = self._compress(k)  # [B, Tc, KV, D]
      v_cmp = self._compress(v)  # [B, Tc, KV, D]

      # CMP path: fused attention + block score aggregation (Eq 8-9)
      o_cmp, block_scores = self._cmp_attention(qh, k_cmp, v_cmp, T)

      # Select blocks using aggregated block scores (Eq 10-12)
      block_idx = self._select_blocks(block_scores, T)

      # SLC path: sparse attention over selected blocks via FlexAttention
      # Paper Eq 12: uses full queries (not qh) for the actual attention
      o_slc = self._slc_path(q, k, v, block_idx)

    # Expand CMP output back to full heads (SLC already has full heads)
    o_cmp = o_cmp.unsqueeze(3).expand(B, T, KV, G, D).reshape(B, T, H, D)

    # SWA path (optional)
    if self.window > 0:
      q_5d = q.view(B, T, KV, G, D)
      start_q = torch.tensor([0], dtype=torch.int32, device=x.device)
      with record_function("attn.kernel[nsa.swa]"):
        o_swa = swa_k.attention(q_5d, k, v, self.sinks, self.softmax_scale, self.window, start_q)
      o_swa = o_swa.view(B, T, H, D)
    else:
      o_swa = torch.zeros_like(o_cmp)

    # -------------------------------------------------------------------------
    # Eq 5: o*_t = Σ_{c∈C} g_t^c · Attn(q_t, K̃_t^c, Ṽ_t^c)
    # -------------------------------------------------------------------------
    # Gating with sigmoid (not softmax) per paper.
    # g_t^c ∈ [0,1] for each branch c ∈ {cmp, slc, swa}.
    # Paper: gates "derived from input features via an MLP"
    gate_input = x  # [B, T, D] - raw input features per paper
    gates = torch.sigmoid(self.gate_mlp(gate_input))  # [B, T, 3]
    g_cmp = gates[..., 0].view(B, T, 1, 1)
    g_slc = gates[..., 1].view(B, T, 1, 1)
    g_swa = gates[..., 2].view(B, T, 1, 1)

    # Weighted sum of attention outputs from all branches
    o = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
    return self.out(o.view(B, T, H * D))
