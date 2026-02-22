import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from nmoe.attention.mla import MLA
from nmoe.attention.rope import rotate_pe
from nmoe.config import Config
from nmoe.triton.dsa import lightning_indexer

try:
  _flex_attention = torch.compile(flex_attention)
except Exception as _e:
  raise RuntimeError(
    "torch.compile(flex_attention) failed for DSA. "
    "Eager fallback is disabled on production no-fallback path."
  ) from _e


class DSA(MLA):
  """DeepSeek Sparse Attention (DSA): lightning indexer with sparse token selection; dense warm‑up materializes [B,T,T] (short flows only).

  Extends MLA with a lightning indexer for sparse token selection, reducing
  attention complexity from O(L²) to O(L·k).

  Key equations:
    Eq 1: I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)  [indexer score]
    Eq 2: u_t = Attn(h_t, {c_s | I_{t,s} ∈ Top-k(I_{t,:})})  [sparse attention]

  Training (Section 2.1.1):
    - Dense warm-up: freeze main model, train indexer with KL alignment loss (Eq 3)
    - Sparse training: joint training, indexer loss detached from main model (Eq 4)

  The indexer uses few heads (H^I), small dimension (d^I), and ReLU activation
  for computational efficiency.

  Note: the dense warm‑up alignment path materializes a full [B, T, T]
  score matrix and is intended only for short dev/proxy flows; the sparse
  path is the default for training.
  """

  def __init__(self, config: Config):
    super().__init__(config)

    # DSA-specific config
    dsa_opts = getattr(config, "attn_dsa", {}) or {}

    # Lightning indexer projections (Eq 1)
    self.n_idx_heads = int(dsa_opts.get("n_idx_heads", 4))
    self.idx_dim = int(dsa_opts.get("idx_dim", 64))
    self.top_k = int(dsa_opts.get("top_k", 2048))

    # q^I: [dim] -> [H_idx * D_idx]
    self.wq_idx = nn.Linear(self.dim, self.n_idx_heads * self.idx_dim, bias=False, dtype=torch.bfloat16)
    # k^I: [dim] -> [D_idx] (shared across indexer heads, MQA-style per paper)
    self.wk_idx = nn.Linear(self.dim, self.idx_dim, bias=False, dtype=torch.bfloat16)
    # w^I: [dim] -> [H_idx] (per-head weights)
    self.w_idx = nn.Linear(self.dim, self.n_idx_heads, bias=False, dtype=torch.bfloat16)

    # Training mode: 'dense_warmup' or 'sparse'
    self.training_mode = dsa_opts.get("training_mode", "sparse")

    # Cache for indexer alignment loss computation
    self._cached_indexer_scores: torch.Tensor | None = None  # [B, T, T] dense or [B, T, k] sparse
    self._cached_attn_weights: torch.Tensor | None = None    # [B, H, T, T] for dense warmup
    # For sparse mode: cache for lazy attention weight computation in alignment loss
    self._cached_q: torch.Tensor | None = None               # [B, T, H, D]
    self._cached_k: torch.Tensor | None = None               # [B, T, H, D]
    self._cached_selected: torch.Tensor | None = None        # [B, T, k]

  def init_weights(self, init_std: float = 0.02):
    super().init_weights(init_std)
    # Indexer projections
    nn.init.trunc_normal_(self.wq_idx.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.wk_idx.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.w_idx.weight, mean=0.0, std=0.02)

  def _lightning_index(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute lightning indexer scores and select top-k indices (Eq 1-2).

    Fused Triton kernel that computes:
        I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
    and selects top-k indices per query in O(T·k) memory.

    Args:
      x: [B, T, D] - input hidden states

    Returns:
      scores: [B, T, k] - top-k indexer scores
      indices: [B, T, k] - top-k key indices (int32)
    """
    B, T, _ = x.shape

    q_idx = self.wq_idx(x).view(B, T, self.n_idx_heads, self.idx_dim)  # [B, T, H_idx, D_idx]
    k_idx = self.wk_idx(x)  # [B, T, D_idx]
    w_idx = self.w_idx(x)   # [B, T, H_idx]

    # Fused kernel: score computation + streaming top-k selection
    scores, indices = lightning_indexer(q_idx, k_idx, w_idx, self.top_k, causal=True)

    return scores, indices

  def _compute_indexer_scores_dense(self, x: torch.Tensor) -> torch.Tensor:
    """Compute full indexer score matrix for dense warm-up alignment loss.

    This is the non-fused version that materializes the full [B, T, T] score
    matrix, needed for KL divergence computation during dense warm-up.

    Args:
      x: [B, T, D] - input hidden states

    Returns:
      scores: [B, T, T] - raw indexer scores (before causal mask)
    """
    B, T, _ = x.shape

    q_idx = self.wq_idx(x).view(B, T, self.n_idx_heads, self.idx_dim)
    k_idx = self.wk_idx(x)
    w_idx = self.w_idx(x)

    # scores[b,t,h,s] = q_idx[b,t,h,:] · k_idx[b,s,:]
    scores = torch.einsum('bthd,bsd->bths', q_idx, k_idx)

    # Eq 1: weighted sum of ReLU scores across indexer heads
    scores = (w_idx.unsqueeze(-1) * F.relu(scores)).sum(dim=2)  # [B, T, T]

    return scores

  def indexer_alignment_loss(self) -> torch.Tensor:
    """Compute KL alignment loss for indexer training (Eq 3-4).

    During dense warm-up (Eq 3):
        L^I = Σ_t D_KL(p_{t,:} || Softmax(I_{t,:}))

    During sparse training (Eq 4):
        L^I = Σ_t D_KL(p_{t,S_t} || Softmax(I_{t,S_t}))

    Where p_{t,:} is the L1-normalized sum of main attention weights across heads,
    and S_t is the selected token set for query t.

    Must be called after forward() which caches the required tensors.

    Returns:
      loss: scalar tensor - KL divergence loss for indexer
    """
    scores = self._cached_indexer_scores

    # =====================================================================
    # Dense warm-up mode (Eq 3): full [B, T, T] scores and attention weights
    # =====================================================================
    if self._cached_attn_weights is not None:
      attn_weights = self._cached_attn_weights  # [B, H, T, T]
      B, T, _ = scores.shape

      # Target distribution: L1-normalized sum of attention weights across heads
      p_target = attn_weights.sum(dim=1)  # [B, T, T]
      p_target = p_target / (p_target.sum(dim=-1, keepdim=True) + 1e-8)

      # Causal mask
      causal_mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
      scores_masked = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

      # KL divergence with proper masking to avoid 0 * (-inf) = nan
      log_q = F.log_softmax(scores_masked, dim=-1)
      p_target = p_target.masked_fill(causal_mask.unsqueeze(0), 0.0)
      kl_per_element = F.kl_div(log_q, p_target, reduction='none')
      kl_per_element = kl_per_element.masked_fill(causal_mask.unsqueeze(0), 0.0)
      loss = kl_per_element.sum() / (B * T)

      return loss

    # =====================================================================
    # Sparse mode (Eq 4): compute attention weights lazily over selected tokens
    # =====================================================================
    if self._cached_q is None or scores is None:
      raise RuntimeError("indexer_alignment_loss() must be called after forward()")

    # With FlexAttention approach, we don't cache k_sel during forward.
    # For the alignment loss, we need to gather K for the selected indices.
    # This is only called during training, so the gather overhead is acceptable.
    if not hasattr(self, '_cached_k') or self._cached_k is None:
      raise RuntimeError("indexer_alignment_loss() requires _cached_k to be set during forward()")

    q = self._cached_q          # [B, T, H, D]
    k = self._cached_k          # [B, T, H, D]
    selected = self._cached_selected  # [B, T, k]
    B, T, top_k = scores.shape

    # Gather K for selected indices (only for loss computation)
    k_flat = k.view(B * T, self.n_heads, self.qk_head_dim)
    batch_offsets = torch.arange(B, device=k.device, dtype=torch.int64).view(B, 1, 1) * T
    selected_flat = (selected.to(torch.int64) + batch_offsets).view(-1)
    k_sel = k_flat.index_select(0, selected_flat).view(B, T, top_k, self.n_heads, self.qk_head_dim)

    # Compute attention weights over selected KV: [B, H, T, k]
    # scores[b,h,t,i] = q[b,t,h,:] · k_sel[b,t,i,h,:] / sqrt(D)
    attn_scores = torch.einsum('bthd,btkhd->bhtk', q, k_sel) * self.softmax_scale
    attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, k]

    # Target distribution: L1-normalized sum across heads
    p_target = attn_weights.sum(dim=1)  # [B, T, k]
    p_target = p_target / (p_target.sum(dim=-1, keepdim=True) + 1e-8)

    # Indexer scores are already [B, T, k] over selected set
    # Mask for valid positions (scores == -inf means padding from causal queries with < k valid keys)
    valid_mask = ~torch.isinf(scores)
    scores_masked = scores.masked_fill(~valid_mask, 0.0)  # avoid -inf in log_softmax
    log_q = F.log_softmax(scores_masked, dim=-1)

    # KL divergence with proper masking
    kl_per_element = F.kl_div(log_q, p_target, reduction='none')
    kl_per_element = kl_per_element.masked_fill(~valid_mask, 0.0)
    n_valid = valid_mask.sum().float()
    loss = kl_per_element.sum() / n_valid.clamp(min=1.0)

    return loss

  @record_function("attn")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, _ = x.size()

    # =====================================================================
    # MLA Projections (inherited from parent)
    # =====================================================================
    q = self.wq_b(self.q_norm(self.wq_a(x)))
    q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = rotate_pe(q_pe, cos, sin)
    q = torch.cat([q_nope, q_pe], dim=-1)  # [B, T, H, qk_head_dim]

    kv = self.wkv_a(x)
    kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = rotate_pe(k_pe.unsqueeze(2), cos, sin)
    kv = self.wkv_b(self.kv_norm(kv))
    kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)  # [B, T, H, qk_head_dim]

    if self.training_mode == "dense_warmup":
      with record_function("attn.kernel[dsa.dense]"):
        Q = q.transpose(1, 2)  # [B, H, T, D]
        K = k.transpose(1, 2)
        V = v.transpose(1, 2)

        # Compute attention weights explicitly for indexer alignment
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.softmax_scale  # [B, H, T, T]
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask.view(1, 1, seqlen, seqlen), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, T]

        output = torch.matmul(attn_weights, V)  # [B, H, T, D]
        output = output.transpose(1, 2).contiguous()  # [B, T, H, D]

      # Cache for indexer alignment loss (dense scores needed for KL divergence)
      with record_function("attn.kernel[dsa.indexer]"):
        self._cached_indexer_scores = self._compute_indexer_scores_dense(x.detach())
      self._cached_attn_weights = attn_weights.detach()
      self._cached_q = None  # clear sparse caches
      self._cached_k = None
      self._cached_selected = None

      output = output.view(bsz, seqlen, self.n_heads * self.v_head_dim)
      return self.wo(output)

    # =====================================================================
    # Sparse Mode: indexer selects top-k tokens, attention via FlexAttention
    # =====================================================================
    with record_function("attn.kernel[dsa.indexer]"):
      # Detach input for indexer (Eq 4: separate optimization)
      # Fused kernel: computes scores + top-k selection in O(T·k) memory
      indexer_scores, selected = self._lightning_index(x.detach())  # [B, T, k], [B, T, k]

    self._cached_indexer_scores = indexer_scores
    top_k = selected.size(-1)

    # =====================================================================
    # Sparse Attention via FlexAttention (no gather - O(T·k) mask instead)
    # =====================================================================
    with record_function("attn.kernel[dsa.flex_attn]"):
      # Build token selection mask: token_allowed[b, t, kv_idx] = True if selected
      # selected: [B, T, k] contains the k selected kv indices per query
      token_allowed = torch.zeros(bsz, seqlen, seqlen, dtype=torch.bool, device=x.device)
      token_allowed.scatter_(2, selected.to(torch.int64), True)

      # Prepare tensors for FlexAttention: [B, H, T, D]
      Q = q.permute(0, 2, 1, 3).contiguous()
      K = k.permute(0, 2, 1, 3).contiguous()
      V = v.permute(0, 2, 1, 3).contiguous()

      def mask_mod(b, h, q_idx, kv_idx, _allowed=token_allowed):
        # Token is allowed if selected by indexer AND causal
        selected_ok = _allowed[b, q_idx, kv_idx]
        causal_ok = kv_idx <= q_idx
        return causal_ok & selected_ok

      bmask = create_block_mask(
        mask_mod, B=bsz, H=self.n_heads, Q_LEN=seqlen, KV_LEN=seqlen,
        device=x.device, BLOCK_SIZE=(128, 128)
      )

      output = _flex_attention(
        Q, K, V,
        block_mask=bmask,
        scale=self.softmax_scale,
        enable_gqa=False,
        return_lse=False,
      )

      output = output.permute(0, 2, 1, 3).contiguous()  # [B, T, H, D]

    # Cache for indexer alignment loss (sparse mode)
    self._cached_attn_weights = None  # signals sparse mode
    self._cached_q = q.detach()
    self._cached_k = k.detach()  # needed for alignment loss gather
    self._cached_selected = selected.detach()  # indices for alignment loss

    output = output.view(bsz, seqlen, self.n_heads * self.v_head_dim)
    return self.wo(output)
