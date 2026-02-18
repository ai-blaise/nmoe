from __future__ import annotations
import os
from typing import Optional, Tuple
from contextlib import nullcontext
from importlib import import_module

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import record_function
import torch.distributed as dist

from nmoe.attention.rope import RotaryEmbedding
from nmoe.rdep import Rdep
from nmoe.blockscaled.grouped import quantize_weights, quantize_weights_from_nvfp4
from nmoe.norm import RMSNorm
from nmoe.fused_router import (
    FusedRouterTopKDispatch,
    fused_update_bias_from_expert_ids,
    fused_update_bias_from_counts,
)
from nmoe.fused_aux_loss import fused_aux_loss


def _nvtx(tag: str):
    if os.getenv('NMOE_NVTX', '0') not in ('1', 'true', 'True'):
        return nullcontext()
    if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'):
        return torch.cuda.nvtx.range(tag)
    return nullcontext()


# =============================================================================
# Fused Residual Add Helpers
# =============================================================================
# These compiled functions fuse the residual add with the surrounding computation
# (RMSNorm + layer), eliminating 2 separate residual add kernel launches per block.
# With 61 layers, this saves ~122 kernel launches per forward pass.
#
# torch.compile enables the compiler to:
# 1. Fuse the residual add with the preceding layer output
# 2. Potentially fuse RMSNorm + matmul operations
# 3. Reduce memory bandwidth by avoiding intermediate tensor allocations
# =============================================================================


def _make_fused_residual_attn():
    """Create compiled fused residual + attention function."""
    @torch.compile(fullgraph=False, dynamic=True)
    def _fused_residual_attn(x, normed_x, attn, cos, sin):
        # x: residual input, normed_x: pre-computed RMSNorm(x)
        # Compiler fuses: attn output + residual add into single kernel
        return x + attn(normed_x, cos, sin)
    return _fused_residual_attn


def _make_fused_residual_attn_cu():
    """Create compiled fused residual + attention function with cu_seqlens."""
    @torch.compile(fullgraph=False, dynamic=True)
    def _fused_residual_attn_cu(x, normed_x, attn, cos, sin, cu_seqlens):
        return x + attn(normed_x, cos, sin, cu_seqlens=cu_seqlens)
    return _fused_residual_attn_cu


def _make_fused_residual_ffn():
    """Create compiled fused residual + FFN/MoE function."""
    @torch.compile(fullgraph=False, dynamic=True)
    def _fused_residual_ffn(x, normed_x, ffn):
        # x: residual input, normed_x: pre-computed RMSNorm(x)
        # Compiler fuses: ffn output + residual add into single kernel
        return x + ffn(normed_x)
    return _fused_residual_ffn


# Lazy initialization of compiled functions to avoid import-time compilation
_fused_residual_attn = None
_fused_residual_attn_cu = None
_fused_residual_ffn = None


def _get_fused_residual_attn():
    global _fused_residual_attn
    if _fused_residual_attn is None:
        _fused_residual_attn = _make_fused_residual_attn()
    return _fused_residual_attn


def _get_fused_residual_attn_cu():
    global _fused_residual_attn_cu
    if _fused_residual_attn_cu is None:
        _fused_residual_attn_cu = _make_fused_residual_attn_cu()
    return _fused_residual_attn_cu


def _get_fused_residual_ffn():
    global _fused_residual_ffn
    if _fused_residual_ffn is None:
        _fused_residual_ffn = _make_fused_residual_ffn()
    return _fused_residual_ffn


ATTN = {
  "mla": "nmoe.attention.mla.MLA",
  "swa": "nmoe.attention.swa.SWA",
  "nsa": "nmoe.attention.nsa.NSA",
  "dsa": "nmoe.attention.dsa.DSA",
  "kda": "nmoe.attention.kda.KDA",
}


def _validate_moe_config(config, ep_size: int) -> None:
  """Validate MoE configuration parameters.

  Args:
      config: Model configuration with MoE parameters
      ep_size: Expert parallelism group size (number of GPUs sharing experts)

  Raises:
      ValueError: If MoE configuration is invalid
  """
  if config.n_routed_experts is None or config.n_activated_experts is None:
    raise ValueError("MoE requires n_routed_experts and n_activated_experts")
  if config.n_routed_experts % max(1, ep_size) != 0:
    raise ValueError(
      f"n_routed_experts ({config.n_routed_experts}) must be divisible by ep_size ({ep_size})"
    )


def _get_ep_group():
  """Get EP process group, or None if not initialized or EP=1."""
  try:
    from nmoe.distributed.init_groups import is_nmoe_parallel_initialized, get_ep_group
    if is_nmoe_parallel_initialized():
      return get_ep_group()
  except ImportError:
    pass
  return None


def _create_rdep(config, ep_size: int) -> Rdep:
  """Create RDEP dispatcher for MoE layers.

  Uses ep_size (not world_size) to determine local expert count. With EP=8 and
  128 routed experts: n_local = 128 // 8 = 16 experts per GPU.
  """
  with _nvtx("model/create_rdep"):
    import sys
    dp = dist.get_world_size() // max(1, ep_size) if dist.is_initialized() else 1
    n_local = config.n_routed_experts // max(1, ep_size)
    # Capacity = max token-expert slots per GPU in one micro-batch.
    micro_batch = max(1, config.batch_size // (dp * config.gradient_accumulation_steps))
    # Auto-compute: T * K * ep_size (worst-case dispatch)
    auto_capacity = int(micro_batch * config.seq_len * config.n_activated_experts * max(1, ep_size))
    if hasattr(config, "rdep_capacity") and config.rdep_capacity > 0:
      capacity = int(config.rdep_capacity)
    else:
      capacity = auto_capacity
    # --- Enhanced RDEP logging ---
    rank = int(dist.get_rank()) if dist.is_initialized() else 0
    if rank == 0:
      mem_est_gb = capacity * config.dim * 2 * n_local / (1024**3)  # bf16 estimate
      print(f"[RDEP] capacity={capacity:,} (auto={auto_capacity:,})", flush=True)
      print(f"[RDEP] micro_batch={micro_batch} seq_len={config.seq_len} K={config.n_activated_experts} ep={ep_size} dp={dp}", flush=True)
      print(f"[RDEP] n_local={n_local} dim={config.dim} dtype={config.dtype}", flush=True)
      print(f"[RDEP] estimated_buffer_mem={mem_est_gb:.2f} GiB (bf16, {n_local} experts)", flush=True)
      sys.stdout.flush()
    ep_group = _get_ep_group()
    return Rdep(
      config.dim,
      n_local,
      config.n_routed_experts,
      profile=config.dtype,
      capacity=capacity,
      ep_group=ep_group,
    )


def get_attention(name: str):
  if name not in ATTN:
    raise ValueError(f"Unknown attention '{name}'. Expected one of: {sorted(ATTN.keys())}")
  path = ATTN[name]
  module_path, cls_name = path.rsplit(".", 1)
  return getattr(import_module(module_path), cls_name)


class MLP(nn.Module):
  def __init__(self, dim: int, inter_dim: int):
    super().__init__()
    self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
    self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
    self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=torch.bfloat16)

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)

  @record_function("mlp")
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Router(nn.Module):
  def __init__(self, config, device: Optional[torch.device] = None):
    super().__init__()
    self.n_experts = config.n_routed_experts
    self.topk = config.n_activated_experts
    self.route_scale = getattr(config, 'route_scale', 1.0)
    # P3.1: Explicitly set device for gate and bias buffer
    self.gate = nn.Linear(config.dim, self.n_experts, bias=False, dtype=torch.bfloat16, device=device)
    self.register_buffer("bias", torch.zeros(self.n_experts, dtype=torch.float32, device=device))

  @record_function("router")
  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with _nvtx("router/gate"):
      logits = self.gate(x).float()
      if self.route_scale != 1.0:
        logits = logits * self.route_scale
      scores = torch.sigmoid(logits)
    with _nvtx("router/topk"):
      scores_for_selection = scores + self.bias
      _, indices = torch.topk(scores_for_selection, k=self.topk, dim=-1)
      weights = torch.gather(scores, 1, indices)
      weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return weights.to(x.dtype), indices

  @torch.no_grad()
  def update_bias(
    self,
    expert_loads: torch.Tensor,
    gamma: float = 0.001,
    *,
    expert_ids: Optional[torch.Tensor] = None,
  ):
    """Update bias for load balancing using fused Triton kernels.

    This method uses fused GPU kernels to eliminate CPU synchronization
    and reduce kernel launch overhead from 8 kernels + 1 sync to 2 kernels.

    Args:
        expert_loads: [E] expert counts (unnormalized int32 or normalized float32).
                     Used when expert_ids is not provided.
        gamma: Learning rate for bias update (default 0.001)
        expert_ids: Optional [T, K] int32 tensor of selected expert IDs.
                   When provided, computes counts internally via fused bincount.
                   This is the most efficient path - 2 kernel launches total.

    Note:
        - NO torch.cuda.synchronize() - runs entirely on GPU
        - When expert_ids is provided: uses fused bincount + update (2 kernels)
        - When expert_loads is provided: uses fused update only (1 kernel)
    """
    if expert_ids is not None:
      # Most efficient path: fused bincount + bias update
      fused_update_bias_from_expert_ids(expert_ids, self.bias, gamma)
    else:
      # Fallback: expert counts already computed, just do fused update
      fused_update_bias_from_counts(expert_loads, self.bias, gamma)

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
  def __init__(self, cfg, layer_id: int, *, rdep: Rdep, use_fused_router: bool = False):
    super().__init__()
    self.dim = cfg.dim
    self.moe_inter_dim = getattr(cfg, 'moe_inter_dim', cfg.inter_dim)
    self._rdep = rdep
    self.n_local = rdep.n_local
    self.n_experts = cfg.n_routed_experts
    self.K = rdep.topk
    self._use_fused_router = use_fused_router
    self.aux_loss_alpha = getattr(cfg, 'aux_loss_alpha', 0.0)

    if use_fused_router:
      # Use fused Triton kernel for router + TopK + dispatch metadata
      # This reduces kernel launch overhead from 3 to 1
      route_scale = getattr(cfg, 'route_scale', 1.0)
      self.router = FusedRouterTopKDispatch(
        hidden_dim=cfg.dim,
        n_experts=cfg.n_routed_experts,
        topk=cfg.n_activated_experts,
        route_scale=route_scale,
        dtype=torch.bfloat16,
      )
    else:
      self.router = Router(cfg)

    self.W1 = nn.Parameter(torch.empty(self.n_local, self.dim, self.moe_inter_dim, dtype=torch.bfloat16))
    self.W3 = nn.Parameter(torch.empty(self.n_local, self.dim, self.moe_inter_dim, dtype=torch.bfloat16))
    self.W2 = nn.Parameter(torch.empty(self.n_local, self.moe_inter_dim, self.dim, dtype=torch.bfloat16))
    self._dtype = getattr(cfg, 'dtype', 'nvfp4')
    self._use_blockscaled = self._dtype in ('fp8', 'nvfp4')
    self._W_cache = None  # QuantizedWeightsFused cache, refreshed after each optimizer step
    self._nvfp4_primary = False  # True when NVFP4 buffers are the primary weight storage
    self._fused_eco = None  # FusedBackwardECO controller, set by attach()

    # NVFP4 compressed_tensors buffers (Option C: NVFP4 primary, no BF16 master)
    # These are populated by checkpoint load or ECO optimizer.
    # When _nvfp4_primary is True, these buffers ARE the weights.
    # W1/W3/W2 nn.Parameters become transient (for autograd only).
    self.register_buffer('_W1_packed', None, persistent=False)
    self.register_buffer('_W1_scale', None, persistent=False)
    self.register_buffer('_W1_gs', None, persistent=False)
    self.register_buffer('_W3_packed', None, persistent=False)
    self.register_buffer('_W3_scale', None, persistent=False)
    self.register_buffer('_W3_gs', None, persistent=False)
    self.register_buffer('_W2_packed', None, persistent=False)
    self.register_buffer('_W2_scale', None, persistent=False)
    self.register_buffer('_W2_gs', None, persistent=False)
    self._nvfp4_group_size = 16

    n_shared = getattr(cfg, 'n_shared_experts', 0)
    self._shared = MLP(self.dim, n_shared * self.moe_inter_dim) if n_shared else None
    self.last_loads = None
    self.last_aux_loss = None

  def init_weights(self, init_std: float = 0.02):
    for W in (self.W1, self.W3, self.W2):
      nn.init.trunc_normal_(W, mean=0.0, std=init_std)
    if self._use_fused_router:
      # FusedRouterTopKDispatch uses kaiming init by default
      # Reinit with truncated normal to match standard router
      nn.init.trunc_normal_(self.router.router_weight, mean=0.0, std=init_std)
    else:
      self.router.init_weights(init_std)
    if self._shared:
      self._shared.init_weights(init_std)
    if self._use_blockscaled:
      self.refresh_weight_cache()

  def has_nvfp4_buffers(self) -> bool:
    """Check if NVFP4 primary buffers are populated."""
    return self._nvfp4_primary and self._W1_packed is not None

  def set_nvfp4_buffers(
    self,
    W1_packed: torch.Tensor, W1_scale: torch.Tensor, W1_gs: torch.Tensor,
    W3_packed: torch.Tensor, W3_scale: torch.Tensor, W3_gs: torch.Tensor,
    W2_packed: torch.Tensor, W2_scale: torch.Tensor, W2_gs: torch.Tensor,
    group_size: int = 16,
  ) -> None:
    """Set NVFP4 compressed_tensors buffers as primary weights.

    After calling this, refresh_weight_cache() will use these buffers
    to build the blockscaled cache (no BF16 master weights needed).
    """
    self._W1_packed = W1_packed
    self._W1_scale = W1_scale
    self._W1_gs = W1_gs
    self._W3_packed = W3_packed
    self._W3_scale = W3_scale
    self._W3_gs = W3_gs
    self._W2_packed = W2_packed
    self._W2_scale = W2_scale
    self._W2_gs = W2_gs
    self._nvfp4_group_size = group_size
    self._nvfp4_primary = True

  @torch.no_grad()
  def refresh_weight_cache(self):
    """Refresh quantized weight cache. Call after optimizer step."""
    if self._use_blockscaled:
      # P3.3: Explicitly delete old cache to free GPU memory before allocating new
      if self._W_cache is not None:
        del self._W_cache
        self._W_cache = None

      if self._nvfp4_primary and self._W1_packed is not None:
        # Option C: Build cache from NVFP4 buffers directly (no BF16 master)
        self._W_cache = quantize_weights_from_nvfp4(
          self._W1_packed, self._W1_scale, self._W1_gs,
          self._W3_packed, self._W3_scale, self._W3_gs,
          self._W2_packed, self._W2_scale, self._W2_gs,
          group_size=self._nvfp4_group_size,
          profile=self._dtype,
        )
      else:
        # Standard path: quantize from BF16 parameters
        self._W_cache = quantize_weights(self.W1, self.W3, self.W2, profile=self._dtype)

  def _compute_aux_loss(self, gates: torch.Tensor, expert_ids: torch.Tensor, T: int) -> torch.Tensor:
    """Compute auxiliary load balancing loss using fused CUDA kernel.

    Standard load balancing loss from "GShard: Scaling Giant Models with Conditional Computation":
    aux_loss = alpha * E * sum_i(f_i * P_i)

    where:
    - E = number of experts
    - f_i = fraction of tokens routed to expert i (dispatch fraction)
    - P_i = mean routing probability assigned to expert i

    This loss encourages balanced expert utilization by penalizing when
    both the dispatch fraction AND probability are high for the same expert.

    Implementation:
    Uses a fused CUDA kernel that computes f and P in a single pass over the data,
    replacing ~12 separate PyTorch operations with 1-2 kernel launches.

    Args:
        gates: [T, K] routing weights (normalized)
        expert_ids: [T, K] selected expert indices
        T: number of tokens

    Returns:
        Scalar auxiliary loss tensor
    """
    # Use fused CUDA kernel for aux loss computation.
    # This replaces ~12 PyTorch ops (scatter_add, zeros, ones_like, div, mul, sum, etc.)
    # with a single fused kernel, reducing kernel launch overhead by 6-12x.
    return fused_aux_loss(expert_ids, gates, self.n_experts, self.aux_loss_alpha)

  @record_function("moe")
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
      raise ValueError(
        f"MoE input must be on CUDA device, got {x.device}. "
        f"Please move input to GPU with x.cuda() or x.to('cuda')."
      )

    X = x.view(-1, x.size(-1))
    T = X.size(0)

    with _nvtx("moe/router"):
      if self._use_fused_router:
        # Fused router returns: expert_ids, gates, dispatch_indices, expert_counts
        # dispatch_indices and expert_counts are computed by the fused kernel
        # but the actual dispatch is still handled by rdep
        eid, g, dispatch_indices, expert_counts = self.router(X)
      else:
        g, eid = self.router(X)

    with _nvtx("moe/aux_loss"):
      # Compute auxiliary loss for load balancing
      self.last_aux_loss = self._compute_aux_loss(g, eid, T)
      if self._use_fused_router:
        with torch.no_grad():
          # Use expert_counts directly instead of bincount
          self.last_loads = expert_counts.float()
      else:
        with torch.no_grad():
          loads = torch.bincount(eid.reshape(-1), minlength=self.router.n_experts).to(torch.float32)
          self.last_loads = loads

    # Track load imbalance for monitoring (coefficient of variation)
    with torch.no_grad():
      loads = self.last_loads
      if loads is not None and loads.numel() > 0:
        _load_mean = loads.float().mean()
        _load_std = loads.float().std()
        self._last_load_cv = (_load_std / _load_mean.clamp(min=1.0)).item() if _load_mean > 0 else 0.0
      else:
        self._last_load_cv = 0.0

    with _nvtx("moe/blockscaled_dispatch"):
      if self._use_blockscaled:
        if self._W_cache is None:
          self.refresh_weight_cache()

        # NVFP4 primary mode: transiently populate W1/W3/W2 data from NVFP4 buffers
        # for backward pass (STE: forward uses blockscaled cache, backward uses BF16).
        # When fused_eco is active, skip this — backward will dequant per-layer on-the-fly
        # from moe_ref's NVFP4 buffers, avoiding 76 GiB of simultaneous BF16 allocations.
        if self._nvfp4_primary and self._W1_packed is not None and self._fused_eco is None:
          from nmoe.moe import dequant_nvfp4_to_bf16_transient
          gs = self._nvfp4_group_size
          # W1/W3: HF [E, moe_inter_dim, dim//2] → nmoe [E, dim, moe_inter_dim]
          self.W1.data = dequant_nvfp4_to_bf16_transient(
            self._W1_packed, self._W1_scale, self._W1_gs, gs, transpose=True)
          self.W3.data = dequant_nvfp4_to_bf16_transient(
            self._W3_packed, self._W3_scale, self._W3_gs, gs, transpose=True)
          # W2: HF [E, dim, moe_inter_dim//2] → nmoe [E, moe_inter_dim, dim]
          self.W2.data = dequant_nvfp4_to_bf16_transient(
            self._W2_packed, self._W2_scale, self._W2_gs, gs, transpose=True)

        out = self._rdep.moe_blockscaled(X.bfloat16(), eid, g, self.W1, self.W3, self.W2, self._W_cache,
                                          fused_eco=self._fused_eco, moe_ref=self)
      else:
        out = self._rdep.moe_bf16(X.bfloat16(), eid, g, self.W1, self.W3, self.W2)

    if self._shared:
      with _nvtx("moe/shared_expert"):
        out = out + self._shared(X)
    return out.view_as(x)


class TransformerBlock(nn.Module):
  def __init__(
    self,
    config: Config,
    layer_id: int,
    *,
    rdep: Rdep | None = None,
    n_layers: int | None = None,
    use_fused_router: bool = False,
  ):
    super().__init__()
    self.layer_id = layer_id
    self._use_gradient_checkpointing = False  # Enable via gradient_checkpointing_enable()
    self._gradient_checkpointing_kwargs = {"use_reentrant": False}  # Default kwargs
    self.attn_norm = RMSNorm(config.dim, config.rms_norm_eps)
    self.ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)

    global_every = int(getattr(config, "attn_global_every", 1))
    if global_every < 1:
      raise ValueError(f"attn_global_every must be >= 1, got {global_every}.")
    is_last = n_layers is not None and layer_id == n_layers - 1
    is_global = (global_every == 1) or (((layer_id + 1) % global_every) == 0) or is_last
    attn_name = config.attn if is_global else config.attn_local

    self.attn = get_attention(attn_name)(config)
    if not is_global:
      window = int(getattr(config, "attn_local_window", 0))
      if window <= 0:
        raise ValueError(f"attn_local_window must be > 0 when using local attention, got {window}.")
      if not hasattr(self.attn, "window"):
        raise ValueError(
          f"Local attention '{attn_name}' does not expose a 'window' attribute, "
          f"but attn_local_window={window} was requested."
        )
      self.attn.window = window
    self.is_moe = layer_id >= config.n_dense_layers
    if layer_id < config.n_dense_layers:
      self.ffn = MLP(dim=config.dim, inter_dim=config.inter_dim)
    else:
      if rdep is None:
        raise ValueError("MoE layers require an Rdep instance")
      self.ffn = MoE(config, layer_id, rdep=rdep, use_fused_router=use_fused_router)
    # Depth-dependent initialization std.
    self.init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

  def init_weights(self):
    self.attn_norm.weight.data.fill_(1.0)
    self.ffn_norm.weight.data.fill_(1.0)
    self.attn.init_weights(self.init_std)
    self.ffn.init_weights(self.init_std)

  def _attn_supports_cu_seqlens(self) -> bool:
    """Check if the attention module accepts a cu_seqlens parameter.

    Cached after first call to avoid repeated introspection overhead.
    """
    if not hasattr(self, '_attn_has_cu_seqlens'):
      import inspect
      try:
        sig = inspect.signature(self.attn.forward)
        self._attn_has_cu_seqlens = 'cu_seqlens' in sig.parameters
      except (ValueError, TypeError):
        self._attn_has_cu_seqlens = False
    return self._attn_has_cu_seqlens

  @record_function("block")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              cu_seqlens: list[torch.Tensor] | None = None) -> torch.Tensor:
    pass_cu = cu_seqlens is not None and self._attn_supports_cu_seqlens()

    if self._use_gradient_checkpointing:
      # Checkpoint both attention and FFN/MoE for memory efficiency
      # Use configured kwargs (defaults to use_reentrant=False for modern PyTorch)
      # NOTE: In checkpointed regions, we use standard residual adds because:
      # 1. In-place operations break recomputation during backward pass
      # 2. torch.compile works best outside checkpoint boundaries
      # The checkpoint itself provides memory savings, and the recomputation
      # naturally fuses operations during the backward pass.
      with _nvtx("block/attn"):
        if pass_cu:
          x = x + torch.utils.checkpoint.checkpoint(
            self.attn, self.attn_norm(x), cos, sin, cu_seqlens, **self._gradient_checkpointing_kwargs
          )
        else:
          x = x + torch.utils.checkpoint.checkpoint(
            self.attn, self.attn_norm(x), cos, sin, **self._gradient_checkpointing_kwargs
          )
      with _nvtx("block/ffn"):
        x = x + torch.utils.checkpoint.checkpoint(
          self.ffn, self.ffn_norm(x), **self._gradient_checkpointing_kwargs
        )
    else:
      # No checkpointing - use fused residual add helpers
      # torch.compile fuses: RMSNorm output consumed immediately + layer + residual add
      # This eliminates separate residual add kernels (saves ~122 kernel launches across 61 layers)
      #
      # Pre-compute normed inputs so the compiler can fuse the entire pattern:
      # x_normed = RMSNorm(x)  -> consumed by attn/ffn
      # out = layer(x_normed) -> fused with residual
      # x = x + out           -> single fused kernel
      with _nvtx("block/attn"):
        if pass_cu:
          normed_x = self.attn_norm(x)
          x = _get_fused_residual_attn_cu()(x, normed_x, self.attn, cos, sin, cu_seqlens)
        else:
          normed_x = self.attn_norm(x)
          x = _get_fused_residual_attn()(x, normed_x, self.attn, cos, sin)
      with _nvtx("block/ffn"):
        normed_x = self.ffn_norm(x)
        x = _get_fused_residual_ffn()(x, normed_x, self.ffn)
    return x


class Transformer(nn.Module):
  # HuggingFace compatibility attribute
  supports_gradient_checkpointing = True

  def __init__(self, config: Config, use_fused_router: bool = False):
    super().__init__()
    self.config = config
    self._use_fused_router = use_fused_router
    self._gradient_checkpointing_kwargs = {"use_reentrant": False}  # Default kwargs
    ep_size = getattr(config, 'ep_size', 1)
    if ep_size <= 1:
      # Backward compat: if no ep_size configured, use world_size (EP = world)
      ep_size = dist.get_world_size() if dist.is_initialized() else 1
    has_moe = config.n_layers > config.n_dense_layers
    rdep: Rdep | None = _create_rdep(config, ep_size) if has_moe else None
    self.embedding = nn.Embedding(config.vocab_size, config.dim, dtype=torch.bfloat16)
    self.rope = RotaryEmbedding(
      head_dim=config.qk_rope_head_dim,
      base=int(config.rope_theta),
      dtype=torch.bfloat16,
      initial_context_length=config.max_position_embeddings,
      max_context_length=config.max_position_embeddings * 2,  # Allow some headroom
      scaling_factor=config.rope_scaling_factor,
      ntk_alpha=config.rope_ntk_alpha,
      ntk_beta=config.rope_ntk_beta,
    )
    self.blocks = nn.ModuleList([
      TransformerBlock(
        config,
        layer_id,
        rdep=rdep,
        n_layers=config.n_layers,
        use_fused_router=use_fused_router,
      )
      for layer_id in range(config.n_layers)
    ])
    self.norm = RMSNorm(config.dim, config.rms_norm_eps)
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False, dtype=torch.bfloat16)
    # μP scaling (validated via proxy sweep - both scales needed for proper gradient flow)
    self.mup_scale_factor = 10.667
    self.logits_scale_factor = 0.125

  def init_weights(self):
    nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    for block in self.blocks:
      block.init_weights()
    self.norm.weight.data.fill_(1.0)
    final_std = self.config.dim ** -0.5
    nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=final_std)

  def param_sets(self):
    expert_params: list[torch.nn.Parameter] = []
    for m in self.modules():
      if isinstance(m, MoE):
        expert_params.extend([m.W1, m.W3, m.W2])
    expert_ids = {id(p) for p in expert_params}
    dense_params: list[torch.nn.Parameter] = []
    for p in self.parameters():
      if id(p) not in expert_ids:
        dense_params.append(p)
    return expert_params, dense_params

  def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None, **kwargs) -> None:
    """Enable gradient checkpointing for all transformer blocks.

    This checkpoints both attention and FFN/MoE layers, reducing memory
    usage by ~30-50% at the cost of ~20-30% slower backward passes.

    Args:
        gradient_checkpointing_kwargs: Dict of kwargs passed to torch.utils.checkpoint.checkpoint.
            Common options: use_reentrant (bool), preserve_rng_state (bool).
            Defaults to {"use_reentrant": False} for modern PyTorch compatibility.
    """
    # Merge provided kwargs with defaults
    if gradient_checkpointing_kwargs is not None:
      self._gradient_checkpointing_kwargs = {**self._gradient_checkpointing_kwargs, **gradient_checkpointing_kwargs}
    # Also support kwargs passed directly (for compatibility)
    if kwargs:
      self._gradient_checkpointing_kwargs = {**self._gradient_checkpointing_kwargs, **kwargs}
    for block in self.blocks:
      block._use_gradient_checkpointing = True
      block._gradient_checkpointing_kwargs = self._gradient_checkpointing_kwargs

  def gradient_checkpointing_disable(self) -> None:
    """Disable gradient checkpointing for all transformer blocks."""
    for block in self.blocks:
      block._use_gradient_checkpointing = False

  @property
  def is_gradient_checkpointing(self) -> bool:
    """Check if gradient checkpointing is enabled."""
    if not self.blocks:
      return False
    return self.blocks[0]._use_gradient_checkpointing

  @property
  def total_dropped_tokens(self) -> int:
    """Sum of dropped tokens across all MoE layers in the last forward pass."""
    total = 0
    for block in self.blocks:
      ffn = getattr(block, 'ffn', None)
      if ffn is not None and hasattr(ffn, '_last_dropped_count'):
        total += ffn._last_dropped_count
    return total

  @property
  def mean_expert_load_cv(self) -> float:
    """Mean coefficient of variation of expert loads across MoE layers.

    A CV > 0.5 indicates severe load imbalance where some experts handle
    significantly more tokens than others, wasting compute and degrading
    model quality. Values near 0 indicate perfectly balanced routing.
    """
    cvs = []
    for block in self.blocks:
      ffn = getattr(block, 'ffn', None)
      if ffn is not None and hasattr(ffn, '_last_load_cv'):
        cvs.append(ffn._last_load_cv)
    return sum(cvs) / max(len(cvs), 1)

  @record_function("transformer")
  def forward(self, tokens: torch.Tensor,
              cu_seqlens: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Forward pass through the full transformer.

    Args:
      tokens: Input token IDs [B, S].
      cu_seqlens: Optional list of B int32 tensors for packed sequence attention.
                  When provided, attention layers use document-isolated causal masking
                  so tokens from different packed documents cannot attend to each other.
    """
    with record_function("embedding"), _nvtx("model/embed"):
      x = self.embedding(tokens) * self.mup_scale_factor
    seqlen = tokens.size(1)
    with _nvtx("model/rope"):
      if cu_seqlens is not None:
        # Packed sequences: build per-document position IDs that reset to 0
        # at each document boundary so RoPE encodes intra-document positions.
        # Without this, documents packed at position 500+ would get RoPE
        # frequencies as if they were at absolute positions 500+, which is wrong.
        #
        # Algorithm: seq_positions = [0,1,2,...,S-1] per row, then subtract the
        # cumulative start of each document. E.g. cu_seqlens=[0,300,700,4096]:
        #   positions 0-299   -> 0-299   (doc 0, subtract 0)
        #   positions 300-699 -> 0-399   (doc 1, subtract 300)
        #   positions 700+    -> 0-3395  (doc 2, subtract 700)
        bsz = tokens.size(0)
        position_ids = torch.arange(seqlen, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        # Build doc_starts: for each position, the cu_seqlen boundary that starts its document.
        # Use searchsorted to find which document each position belongs to, then look up its start.
        doc_starts = torch.zeros(bsz, seqlen, dtype=torch.long, device=tokens.device)
        for b in range(bsz):
          cu = cu_seqlens[b].to(dtype=torch.long, device=tokens.device)
          # searchsorted(cu, positions, right=True) - 1 gives the document index for each position
          doc_idx = torch.searchsorted(cu, position_ids[b], right=True) - 1
          doc_idx = doc_idx.clamp(min=0, max=cu.shape[0] - 2)
          doc_starts[b] = cu[doc_idx]
        position_ids = position_ids - doc_starts
        # Index into precomputed RoPE table: [B, S] -> [B, S, head_dim//2]
        cos = self.rope.cos[position_ids]
        sin = self.rope.sin[position_ids]
      else:
        # Standard sequential positions
        cos = self.rope.cos[:seqlen]
        sin = self.rope.sin[:seqlen]
    with _nvtx("model/layers"):
      for block in self.blocks:
        x = block(x, cos, sin, cu_seqlens=cu_seqlens)
    with torch.no_grad():
      moe_layers = [blk.ffn for blk in self.blocks if isinstance(getattr(blk, 'ffn', None), MoE)]
      if moe_layers:
        loads = torch.stack([m.last_loads for m in moe_layers], dim=0)
        if dist.is_available() and dist.is_initialized():
          # Use EP group for load balancing AllReduce (not WORLD).
          # Load counts should be aggregated across EP ranks that share
          # the same expert partition, not across DP replicas.
          ep_group = _get_ep_group()
          dist.all_reduce(loads, op=dist.ReduceOp.SUM, group=ep_group)
        loads = loads / loads.sum(dim=-1, keepdim=True).clamp_min(1.0)
        for m, l in zip(moe_layers, loads):
          m.last_loads = l
    with record_function("norm_f"), _nvtx("model/norm_f"):
      x = self.norm(x)
    # Dynamic amax scaling handles range - no clamp needed (TorchTitan/Megatron pattern)
    with record_function("lm_head"), _nvtx("model/lm_head"):
      logits = self.lm_head(x) * self.logits_scale_factor

    return logits

  def get_router_aux_loss(self) -> torch.Tensor:
    """Get aggregated auxiliary loss from all MoE layers for load balancing.

    This method collects `last_aux_loss` from all MoE layers and sums them.
    Call this after forward() to get the auxiliary loss for training.

    The auxiliary loss encourages balanced expert utilization using the
    GShard-style formula: aux_loss = alpha * E * sum(f_i * P_i)
    where f_i is the fraction of tokens dispatched to expert i,
    and P_i is the mean routing probability for expert i.

    Returns:
        Scalar tensor with the sum of all MoE auxiliary losses.
        Returns 0 if no MoE layers or aux_loss_alpha is 0.

    Example:
        >>> logits = model(tokens)
        >>> loss = criterion(logits, targets)
        >>> aux_loss = model.get_router_aux_loss()
        >>> total_loss = loss + aux_loss  # aux_loss already scaled by alpha
        >>> total_loss.backward()
    """
    moe_layers = [blk.ffn for blk in self.blocks if isinstance(getattr(blk, 'ffn', None), MoE)]
    if not moe_layers:
      return torch.tensor(0.0, device=self.embedding.weight.device)

    aux_losses = []
    for moe in moe_layers:
      if hasattr(moe, 'last_aux_loss') and moe.last_aux_loss is not None:
        aux_losses.append(moe.last_aux_loss)

    if not aux_losses:
      return torch.tensor(0.0, device=self.embedding.weight.device)

    return torch.stack(aux_losses).sum()

  def get_expert_load_stats(self) -> dict:
    """Get expert load statistics from all MoE layers.

    Returns:
        dict with:
          - 'loads': List of load tensors per MoE layer
          - 'mean_load': Mean load across all experts
          - 'load_imbalance': Coefficient of variation (std/mean)
    """
    moe_layers = [blk.ffn for blk in self.blocks if isinstance(getattr(blk, 'ffn', None), MoE)]
    if not moe_layers:
      return {'loads': [], 'mean_load': 0.0, 'load_imbalance': 0.0}

    loads = [moe.last_loads for moe in moe_layers if hasattr(moe, 'last_loads')]
    if not loads:
      return {'loads': [], 'mean_load': 0.0, 'load_imbalance': 0.0}

    all_loads = torch.stack(loads)  # [n_moe_layers, n_experts]
    mean_load = all_loads.mean().item()
    std_load = all_loads.std().item()
    load_imbalance = std_load / mean_load if mean_load > 0 else 0.0

    return {
      'loads': [l.cpu().numpy().tolist() for l in loads],
      'mean_load': mean_load,
      'load_imbalance': load_imbalance,
    }
