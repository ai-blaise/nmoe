"""Memory optimization utilities for MoE training.

This module contains implementations for:
- 5.3.1: LazyExpertWeights - Lazy weight loading for experts
- 5.3.2: ExpertCachePolicy - Configurable cache eviction policies (LRU, LFU, FIFO, ARC)
- 5.3.3: Activation checkpointing helpers for MoE layers
- 5.3.4: MemoryEfficientGradAccum - Gradient accumulation with memory efficiency
- 5.3.5: Mixed-precision optimizer states (see opt.py)
- 5.3.6: DynamicBatchSizer - Adjust batch size based on available memory
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# =============================================================================
# 5.3.2: Cache Eviction Policies
# =============================================================================


class BaseEvictionPolicy(ABC):
    """Base class for cache eviction policies."""

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.max_size = max_size

    @abstractmethod
    def access(self, key: int) -> None:
        """Record an access to the given key."""
        pass

    @abstractmethod
    def evict(self) -> int:
        """Choose a key to evict and remove it from tracking. Returns the key."""
        pass

    @abstractmethod
    def add(self, key: int) -> None:
        """Add a new key to tracking."""
        pass

    @abstractmethod
    def remove(self, key: int) -> None:
        """Remove a key from tracking (e.g., when manually removed from cache)."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return current number of tracked keys."""
        pass

    @abstractmethod
    def __contains__(self, key: int) -> bool:
        """Check if key is being tracked."""
        pass


class LRUPolicy(BaseEvictionPolicy):
    """Least Recently Used eviction policy.

    Evicts the key that was accessed least recently.
    Uses OrderedDict for O(1) access/eviction.
    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._access_order: OrderedDict[int, None] = OrderedDict()

    def access(self, key: int) -> None:
        if key in self._access_order:
            self._access_order.move_to_end(key)

    def evict(self) -> int:
        if not self._access_order:
            raise RuntimeError("Cannot evict from empty cache")
        key, _ = self._access_order.popitem(last=False)
        return key

    def add(self, key: int) -> None:
        self._access_order[key] = None
        self._access_order.move_to_end(key)

    def remove(self, key: int) -> None:
        self._access_order.pop(key, None)

    def __len__(self) -> int:
        return len(self._access_order)

    def __contains__(self, key: int) -> bool:
        return key in self._access_order


class LFUPolicy(BaseEvictionPolicy):
    """Least Frequently Used eviction policy.

    Evicts the key that was accessed least frequently.
    Ties are broken by LRU (least recently used among least frequent).
    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._freq: Dict[int, int] = {}
        self._access_order: OrderedDict[int, None] = OrderedDict()

    def access(self, key: int) -> None:
        if key in self._freq:
            self._freq[key] += 1
            self._access_order.move_to_end(key)

    def evict(self) -> int:
        if not self._freq:
            raise RuntimeError("Cannot evict from empty cache")
        # Find minimum frequency
        min_freq = min(self._freq.values())
        # Among keys with min frequency, find the LRU one
        for key in self._access_order:
            if self._freq[key] == min_freq:
                self._access_order.pop(key)
                del self._freq[key]
                return key
        # Should not reach here
        raise RuntimeError("Inconsistent LFU state")

    def add(self, key: int) -> None:
        self._freq[key] = 1
        self._access_order[key] = None
        self._access_order.move_to_end(key)

    def remove(self, key: int) -> None:
        self._freq.pop(key, None)
        self._access_order.pop(key, None)

    def __len__(self) -> int:
        return len(self._freq)

    def __contains__(self, key: int) -> bool:
        return key in self._freq


class FIFOPolicy(BaseEvictionPolicy):
    """First In First Out eviction policy.

    Evicts the key that was added first, regardless of access pattern.
    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self._order: OrderedDict[int, None] = OrderedDict()

    def access(self, key: int) -> None:
        # FIFO doesn't update order on access
        pass

    def evict(self) -> int:
        if not self._order:
            raise RuntimeError("Cannot evict from empty cache")
        key, _ = self._order.popitem(last=False)
        return key

    def add(self, key: int) -> None:
        self._order[key] = None

    def remove(self, key: int) -> None:
        self._order.pop(key, None)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, key: int) -> bool:
        return key in self._order


class ARCPolicy(BaseEvictionPolicy):
    """Adaptive Replacement Cache (ARC) eviction policy.

    ARC dynamically balances between recency (LRU) and frequency (LFU) based
    on workload characteristics. It maintains two LRU lists:
    - T1: Pages accessed only once recently (recency)
    - T2: Pages accessed at least twice recently (frequency)

    And two ghost lists for evicted items:
    - B1: Ghost entries evicted from T1
    - B2: Ghost entries evicted from T2

    The algorithm adaptively adjusts the partition point 'p' between T1 and T2
    based on cache hit/miss patterns in the ghost lists.

    Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache"
    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        # T1: Recent items (LRU for items seen once)
        self._t1: OrderedDict[int, None] = OrderedDict()
        # T2: Frequent items (LRU for items seen more than once)
        self._t2: OrderedDict[int, None] = OrderedDict()
        # B1: Ghost list of recently evicted from T1
        self._b1: OrderedDict[int, None] = OrderedDict()
        # B2: Ghost list of recently evicted from T2
        self._b2: OrderedDict[int, None] = OrderedDict()
        # Target size for T1 (adaptive)
        self._p: float = 0.0

    def _replace(self, key_in_b2: bool) -> int:
        """Internal helper: evict from T1 or T2 and return evicted key."""
        t1_len = len(self._t1)
        # Decide whether to evict from T1 or T2
        if t1_len > 0 and (t1_len > self._p or (key_in_b2 and t1_len == int(self._p))):
            # Evict from T1 (move to B1)
            evicted, _ = self._t1.popitem(last=False)
            self._b1[evicted] = None
        else:
            # Evict from T2 (move to B2)
            if not self._t2:
                # T2 empty, must evict from T1
                evicted, _ = self._t1.popitem(last=False)
                self._b1[evicted] = None
            else:
                evicted, _ = self._t2.popitem(last=False)
                self._b2[evicted] = None
        return evicted

    def access(self, key: int) -> None:
        """Record an access to the given key."""
        # Case 1: Key is in T1 - move to T2 (it's now "frequent")
        if key in self._t1:
            del self._t1[key]
            self._t2[key] = None
            return

        # Case 2: Key is in T2 - move to MRU position in T2
        if key in self._t2:
            self._t2.move_to_end(key)
            return

        # Case 3: Key is in B1 (ghost) - adapt p upward (favor recency)
        if key in self._b1:
            # Increase target for T1
            delta = max(1.0, len(self._b2) / max(1, len(self._b1)))
            self._p = min(self._p + delta, float(self.max_size))
            del self._b1[key]
            # Note: actual insertion handled by add()
            return

        # Case 4: Key is in B2 (ghost) - adapt p downward (favor frequency)
        if key in self._b2:
            # Decrease target for T1
            delta = max(1.0, len(self._b1) / max(1, len(self._b2)))
            self._p = max(self._p - delta, 0.0)
            del self._b2[key]
            # Note: actual insertion handled by add()
            return

        # Case 5: Key is not in cache or ghosts - this is a new key
        # Handled by add()

    def evict(self) -> int:
        """Choose a key to evict and remove it from tracking."""
        total = len(self._t1) + len(self._t2)
        if total == 0:
            raise RuntimeError("Cannot evict from empty cache")

        # Perform replacement and get the evicted key
        # We use False for key_in_b2 since we're evicting, not inserting
        return self._replace(key_in_b2=False)

    def add(self, key: int) -> None:
        """Add a new key to tracking."""
        # Check if key was in ghost lists (ARC adaptation already done in access())
        was_in_b1 = key in self._b1
        was_in_b2 = key in self._b2

        # Clean up ghost entry if present
        self._b1.pop(key, None)
        self._b2.pop(key, None)

        # Limit ghost list sizes to max_size
        while len(self._b1) > self.max_size:
            self._b1.popitem(last=False)
        while len(self._b2) > self.max_size:
            self._b2.popitem(last=False)

        if was_in_b1 or was_in_b2:
            # Key was in ghost list - add to T2 (it's now "frequent")
            self._t2[key] = None
        else:
            # Completely new key - add to T1 (recency list)
            self._t1[key] = None

    def remove(self, key: int) -> None:
        """Remove a key from tracking."""
        self._t1.pop(key, None)
        self._t2.pop(key, None)
        self._b1.pop(key, None)
        self._b2.pop(key, None)

    def __len__(self) -> int:
        return len(self._t1) + len(self._t2)

    def __contains__(self, key: int) -> bool:
        return key in self._t1 or key in self._t2


class ExpertCachePolicy:
    """Factory for creating cache eviction policies."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ARC = "arc"

    @staticmethod
    def create(policy: str, max_size: int) -> BaseEvictionPolicy:
        """Create a cache eviction policy.

        Args:
            policy: One of "lru", "lfu", "fifo", or "arc".
            max_size: Maximum number of items to track.

        Returns:
            An instance of the requested eviction policy.

        Raises:
            ValueError: If policy is not recognized.
        """
        if policy == ExpertCachePolicy.LRU:
            return LRUPolicy(max_size)
        elif policy == ExpertCachePolicy.LFU:
            return LFUPolicy(max_size)
        elif policy == ExpertCachePolicy.FIFO:
            return FIFOPolicy(max_size)
        elif policy == ExpertCachePolicy.ARC:
            return ARCPolicy(max_size)
        raise ValueError(f"Unknown policy: {policy}. Expected one of: lru, lfu, fifo, arc")


# =============================================================================
# 5.3.1: Lazy Expert Weight Loading
# =============================================================================


class LazyExpertWeights:
    """Load expert weights on-demand to reduce memory footprint.

    This class provides lazy loading of expert weights from checkpoints,
    with configurable cache eviction policies. Experts are loaded from disk
    only when accessed and can be evicted based on usage patterns.

    Args:
        checkpoint_path: Path to checkpoint file or directory.
        n_experts: Total number of experts.
        dtype: Data type for loaded weights (default: bfloat16).
        device: Device to load weights onto (default: "cuda").
        max_loaded: Maximum number of experts to keep in memory.
            Defaults to n_experts (no eviction). Set lower for memory savings.
        eviction_policy: Cache eviction policy ("lru", "lfu", "fifo").

    Example:
        >>> lazy = LazyExpertWeights("checkpoint.pt", n_experts=64, max_loaded=8)
        >>> expert_weights = lazy.get_expert(42)  # Loads from disk if not cached
        >>> lazy.set_max_loaded(4)  # Reduce memory footprint
    """

    def __init__(
        self,
        checkpoint_path: str,
        n_experts: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_loaded: Optional[int] = None,
        eviction_policy: str = "lru",
    ):
        self.checkpoint_path = checkpoint_path
        self.n_experts = n_experts
        self.dtype = dtype
        self.device = device
        self._max_loaded = max_loaded if max_loaded is not None else n_experts

        # Validate max_loaded
        if self._max_loaded < 1:
            raise ValueError(f"max_loaded must be >= 1, got {self._max_loaded}")
        if self._max_loaded > n_experts:
            self._max_loaded = n_experts

        # Loaded expert cache
        self._loaded_experts: Dict[int, Dict[str, torch.Tensor]] = {}

        # Eviction policy
        self._policy = ExpertCachePolicy.create(eviction_policy, self._max_loaded)

        # Checkpoint state (lazy loaded)
        self._checkpoint_state: Optional[Dict[str, Any]] = None

        # Statistics
        self._load_count = 0
        self._evict_count = 0
        self._hit_count = 0
        self._miss_count = 0

    @property
    def max_loaded(self) -> int:
        """Maximum number of experts to keep loaded."""
        return self._max_loaded

    def set_max_loaded(self, max_loaded: int) -> None:
        """Set the maximum number of loaded experts.

        If current loaded count exceeds new max, experts will be evicted.

        Args:
            max_loaded: New maximum (must be >= 1 and <= n_experts).
        """
        if max_loaded < 1:
            raise ValueError(f"max_loaded must be >= 1, got {max_loaded}")
        if max_loaded > self.n_experts:
            max_loaded = self.n_experts

        self._max_loaded = max_loaded
        self._policy.max_size = max_loaded

        # Evict excess experts
        while len(self._loaded_experts) > self._max_loaded:
            self._evict()

    def get_expert(self, expert_id: int) -> Dict[str, torch.Tensor]:
        """Get expert weights, loading from disk if needed.

        Args:
            expert_id: Expert index (0 to n_experts-1).

        Returns:
            Dictionary with expert weight tensors (W1, W3, W2).

        Raises:
            ValueError: If expert_id is out of range.
            RuntimeError: If checkpoint cannot be loaded.
        """
        if not 0 <= expert_id < self.n_experts:
            raise ValueError(f"expert_id must be in [0, {self.n_experts}), got {expert_id}")

        if expert_id in self._loaded_experts:
            self._hit_count += 1
            self._policy.access(expert_id)
            return self._loaded_experts[expert_id]

        self._miss_count += 1
        self._load_expert(expert_id)
        return self._loaded_experts[expert_id]

    def _load_expert(self, expert_id: int) -> None:
        """Load expert from checkpoint."""
        # Evict if at capacity
        while len(self._loaded_experts) >= self._max_loaded:
            self._evict()

        # Lazy load checkpoint
        if self._checkpoint_state is None:
            self._load_checkpoint()

        # Extract expert weights
        expert_weights = self._extract_expert_weights(expert_id)
        self._loaded_experts[expert_id] = expert_weights
        self._policy.add(expert_id)
        self._load_count += 1

    def _load_checkpoint(self) -> None:
        """Load checkpoint from disk."""
        path = Path(self.checkpoint_path)
        if not path.exists():
            raise RuntimeError(f"Checkpoint not found: {self.checkpoint_path}")

        # Determine if it's a file or directory
        if path.is_file():
            self._checkpoint_state = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        else:
            # Try common checkpoint file names
            for name in ["dp_rank_000.pt", "model.pt", "checkpoint.pt"]:
                file_path = path / name
                if file_path.exists():
                    self._checkpoint_state = torch.load(
                        str(file_path), map_location="cpu", weights_only=False
                    )
                    break
            else:
                raise RuntimeError(
                    f"No checkpoint file found in directory: {self.checkpoint_path}"
                )

    def _extract_expert_weights(self, expert_id: int) -> Dict[str, torch.Tensor]:
        """Extract weights for a specific expert from checkpoint state.

        This handles the nmoe checkpoint format where expert weights are stored
        as [n_local_experts, hidden_dim, intermediate_dim] tensors.
        """
        if self._checkpoint_state is None:
            raise RuntimeError("Checkpoint not loaded")

        # Look for expert weights in common locations
        model_state = self._checkpoint_state.get("model_expert", {})
        if not model_state:
            model_state = self._checkpoint_state.get("model", {})
        if not model_state:
            model_state = self._checkpoint_state

        expert_weights: Dict[str, torch.Tensor] = {}

        # Find W1, W3, W2 for this expert
        # Pattern: blocks.{layer_id}.ffn.W1, etc.
        for key, tensor in model_state.items():
            if not isinstance(tensor, torch.Tensor):
                continue

            # Check if this is an expert weight tensor
            if any(w in key for w in [".W1", ".W3", ".W2"]):
                # Expert weights have shape [n_experts, hidden, intermediate]
                if tensor.ndim >= 1 and tensor.shape[0] > expert_id:
                    weight_name = "W1" if ".W1" in key else ("W3" if ".W3" in key else "W2")
                    layer_key = key.rsplit(".", 1)[0]  # Get layer path

                    # Initialize layer dict if needed
                    if layer_key not in expert_weights:
                        expert_weights[layer_key] = {}

                    # Extract this expert's weight slice
                    sliced = tensor[expert_id : expert_id + 1].to(
                        device=self.device, dtype=self.dtype
                    )
                    expert_weights[f"{layer_key}.{weight_name}"] = sliced.squeeze(0)

        if not expert_weights:
            raise RuntimeError(f"No expert weights found for expert {expert_id}")

        return expert_weights

    def _evict(self) -> None:
        """Evict an expert based on the eviction policy."""
        if not self._loaded_experts:
            return

        evict_id = self._policy.evict()
        if evict_id in self._loaded_experts:
            del self._loaded_experts[evict_id]
            self._evict_count += 1
            torch.cuda.empty_cache()

    def prefetch(self, expert_ids: List[int]) -> None:
        """Prefetch multiple experts into cache.

        Loads experts in order, evicting as needed to stay within max_loaded.

        Args:
            expert_ids: List of expert IDs to prefetch.
        """
        for eid in expert_ids:
            if 0 <= eid < self.n_experts and eid not in self._loaded_experts:
                self._load_expert(eid)

    def clear(self) -> None:
        """Clear all loaded experts and free memory."""
        self._loaded_experts.clear()
        self._policy = ExpertCachePolicy.create(
            self._policy.__class__.__name__.lower().replace("policy", ""),
            self._max_loaded,
        )
        torch.cuda.empty_cache()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with load_count, evict_count, hit_count, miss_count,
            hit_rate, and currently_loaded.
        """
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        return {
            "load_count": self._load_count,
            "evict_count": self._evict_count,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "currently_loaded": len(self._loaded_experts),
            "max_loaded": self._max_loaded,
        }

    def __len__(self) -> int:
        """Return number of currently loaded experts."""
        return len(self._loaded_experts)

    def __contains__(self, expert_id: int) -> bool:
        """Check if expert is currently loaded."""
        return expert_id in self._loaded_experts


def moe_forward_with_checkpointing(
    experts: nn.ModuleList,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gates: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """MoE forward pass with activation checkpointing for memory efficiency.

    This function applies activation checkpointing to each expert computation,
    trading compute for memory during the backward pass.

    Args:
        experts: ModuleList of expert modules.
        x: Input tensor of shape [batch * seq_len, hidden_dim].
        expert_ids: Expert assignments of shape [batch * seq_len, topk].
        gates: Gating weights of shape [batch * seq_len, topk].
        topk: Number of experts per token.

    Returns:
        Output tensor of shape [batch * seq_len, hidden_dim].
    """

    def expert_forward(expert_idx: int, expert_input: torch.Tensor) -> torch.Tensor:
        """Compute single expert forward (checkpointable)."""
        return experts[expert_idx](expert_input)

    result = torch.zeros_like(x)

    for k in range(topk):
        eid = expert_ids[:, k]
        gate = gates[:, k : k + 1]

        unique_experts = eid.unique()
        for e in unique_experts:
            mask = eid == e.item()
            if not mask.any():
                continue

            expert_input = x[mask]

            # Checkpoint the expert forward pass
            expert_out = checkpoint(
                expert_forward,
                e.item(),
                expert_input,
                use_reentrant=False,
            )

            result[mask] = result[mask] + gate[mask] * expert_out

    return result


class CheckpointedMoEWrapper(nn.Module):
    """Wrapper that applies activation checkpointing to an MoE layer.

    This wrapper can be used to add checkpointing to existing MoE layers
    without modifying the original implementation.

    Args:
        moe_layer: The MoE layer to wrap.
        checkpoint_every_n: Apply checkpointing every N experts (default: 1).
            Higher values trade memory for compute.
    """

    def __init__(self, moe_layer: nn.Module, checkpoint_every_n: int = 1):
        super().__init__()
        self.moe_layer = moe_layer
        self.checkpoint_every_n = max(1, checkpoint_every_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with selective checkpointing."""
        if not self.training:
            return self.moe_layer(x)

        # For training, use checkpointing
        return checkpoint(
            self.moe_layer,
            x,
            use_reentrant=False,
        )


# =============================================================================
# 5.3.4: Memory-Efficient Gradient Accumulation
# =============================================================================


class MemoryEfficientGradAccum:
    """Memory-efficient gradient accumulation for MoE training.

    This class handles gradient accumulation with proper loss scaling,
    allowing for larger effective batch sizes without increasing memory usage.

    Features:
    - Proper loss scaling for gradient accumulation
    - Optional gradient clipping
    - Memory-efficient gradient storage (no extra copies)
    - Sync-free accumulation (gradients accumulate in-place)

    Args:
        accumulation_steps: Number of steps to accumulate before updating.
        max_grad_norm: Optional gradient clipping threshold.
        scale_loss: Whether to scale loss by accumulation_steps (default: True).

    Example:
        >>> accum = MemoryEfficientGradAccum(accumulation_steps=4)
        >>> for micro_batch in data:
        ...     loss = model(micro_batch).loss
        ...     ready = accum.step(model, loss)
        ...     if ready:
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """

    def __init__(
        self,
        accumulation_steps: int = 4,
        max_grad_norm: Optional[float] = None,
        scale_loss: bool = True,
    ):
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scale_loss = scale_loss
        self._step = 0
        self._accumulated_loss = 0.0

    def step(self, model: nn.Module, loss: torch.Tensor) -> bool:
        """Accumulate gradients, return True when ready to update.

        Args:
            model: The model (used for gradient clipping if enabled).
            loss: The loss tensor to backward.

        Returns:
            True if accumulation is complete and ready for optimizer step.
        """
        # Scale loss by accumulation steps
        if self.scale_loss:
            scaled_loss = loss / self.accumulation_steps
        else:
            scaled_loss = loss

        # Backward pass (gradients accumulate in-place)
        scaled_loss.backward()

        # Track accumulated loss for logging
        self._accumulated_loss += loss.detach().item()
        self._step += 1

        if self._step >= self.accumulation_steps:
            # Apply gradient clipping if enabled
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.max_grad_norm
                )

            self._step = 0
            return True

        return False

    def get_accumulated_loss(self) -> float:
        """Get the accumulated loss (not scaled).

        Returns:
            Total accumulated loss value.
        """
        return self._accumulated_loss

    def reset_accumulated_loss(self) -> float:
        """Reset and return the accumulated loss.

        Returns:
            Total accumulated loss value before reset.
        """
        loss = self._accumulated_loss
        self._accumulated_loss = 0.0
        return loss

    @property
    def current_step(self) -> int:
        """Current accumulation step (0 to accumulation_steps-1)."""
        return self._step

    def is_accumulating(self) -> bool:
        """Check if currently accumulating (not ready for update)."""
        return self._step > 0 and self._step < self.accumulation_steps


# =============================================================================
# 5.3.6: Dynamic Batch Sizing
# =============================================================================


class DynamicBatchSizer:
    """Adjust batch size based on available GPU memory.

    This class monitors GPU memory usage and dynamically adjusts the batch size
    to maximize throughput while avoiding OOM errors.

    Features:
    - Automatic batch size adjustment based on memory pressure
    - Configurable memory target and safety margins
    - Exponential backoff on OOM detection
    - Gradual recovery when memory is available

    Args:
        min_batch: Minimum batch size (default: 1).
        max_batch: Maximum batch size (default: 64).
        target_memory_fraction: Target GPU memory usage fraction (default: 0.85).
        recovery_threshold: Memory fraction below which to increase batch (default: 0.6).
        growth_factor: Factor to multiply batch size when growing (default: 1.5).
        shrink_factor: Factor to divide batch size when shrinking (default: 2.0).
        device: CUDA device to monitor (default: "cuda:0").

    Example:
        >>> sizer = DynamicBatchSizer(min_batch=4, max_batch=64)
        >>> for epoch in range(num_epochs):
        ...     batch_size = sizer.get_batch_size()
        ...     try:
        ...         train_step(batch_size)
        ...     except RuntimeError as e:
        ...         if "out of memory" in str(e):
        ...             sizer.report_oom()
        ...             torch.cuda.empty_cache()
    """

    def __init__(
        self,
        min_batch: int = 1,
        max_batch: int = 64,
        target_memory_fraction: float = 0.85,
        recovery_threshold: float = 0.6,
        growth_factor: float = 1.5,
        shrink_factor: float = 2.0,
        device: str = "cuda:0",
    ):
        if min_batch < 1:
            raise ValueError(f"min_batch must be >= 1, got {min_batch}")
        if max_batch < min_batch:
            raise ValueError(f"max_batch must be >= min_batch, got max={max_batch}, min={min_batch}")
        if not 0 < target_memory_fraction < 1:
            raise ValueError(f"target_memory_fraction must be in (0, 1), got {target_memory_fraction}")
        if not 0 < recovery_threshold < target_memory_fraction:
            raise ValueError(
                f"recovery_threshold must be in (0, target_memory_fraction), "
                f"got recovery={recovery_threshold}, target={target_memory_fraction}"
            )

        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_memory_fraction = target_memory_fraction
        self.recovery_threshold = recovery_threshold
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.device = device

        self.current_batch = max_batch
        self._oom_count = 0
        self._stable_steps = 0
        self._last_adjustment_time = 0.0

    def get_batch_size(self) -> int:
        """Get recommended batch size based on current memory usage.

        Returns:
            Recommended batch size.
        """
        if not torch.cuda.is_available():
            return self.current_batch

        try:
            free_mem, total_mem = torch.cuda.mem_get_info(self.device)
        except (RuntimeError, AssertionError):
            # Device not available or error
            return self.current_batch

        used_fraction = 1 - (free_mem / total_mem)

        current_time = time.time()
        min_adjustment_interval = 1.0  # Don't adjust more than once per second

        if current_time - self._last_adjustment_time < min_adjustment_interval:
            return self.current_batch

        if used_fraction > self.target_memory_fraction:
            # Memory pressure - reduce batch size
            new_batch = int(self.current_batch / self.shrink_factor)
            self.current_batch = max(self.min_batch, new_batch)
            self._stable_steps = 0
            self._last_adjustment_time = current_time

        elif used_fraction < self.recovery_threshold:
            # Low memory usage - can try increasing batch size
            self._stable_steps += 1

            # Only grow after several stable steps
            if self._stable_steps >= 5 and self._oom_count == 0:
                new_batch = int(self.current_batch * self.growth_factor)
                self.current_batch = min(self.max_batch, new_batch)
                self._stable_steps = 0
                self._last_adjustment_time = current_time

        return self.current_batch

    def report_oom(self) -> int:
        """Report an OOM event and get new reduced batch size.

        Call this when an OOM error is caught. The batch size will be
        reduced aggressively.

        Returns:
            New reduced batch size.
        """
        self._oom_count += 1

        # Aggressive reduction on OOM
        new_batch = int(self.current_batch / self.shrink_factor)
        self.current_batch = max(self.min_batch, new_batch)
        self._stable_steps = 0
        self._last_adjustment_time = time.time()

        return self.current_batch

    def reset_oom_count(self) -> None:
        """Reset the OOM counter.

        Call this periodically (e.g., at epoch boundaries) to allow
        batch size recovery after OOM events.
        """
        self._oom_count = 0

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics.

        Returns:
            Dictionary with memory info and batch sizer state.
        """
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "current_batch": self.current_batch,
            }

        try:
            free_mem, total_mem = torch.cuda.mem_get_info(self.device)
            used_mem = total_mem - free_mem
        except (RuntimeError, AssertionError):
            return {
                "cuda_available": True,
                "device_error": True,
                "current_batch": self.current_batch,
            }

        return {
            "cuda_available": True,
            "total_memory_gb": total_mem / (1024**3),
            "used_memory_gb": used_mem / (1024**3),
            "free_memory_gb": free_mem / (1024**3),
            "used_fraction": used_mem / total_mem,
            "current_batch": self.current_batch,
            "min_batch": self.min_batch,
            "max_batch": self.max_batch,
            "oom_count": self._oom_count,
            "stable_steps": self._stable_steps,
        }

    def set_batch_size(self, batch_size: int) -> None:
        """Manually set the batch size.

        Args:
            batch_size: New batch size (will be clamped to [min_batch, max_batch]).
        """
        self.current_batch = max(self.min_batch, min(self.max_batch, batch_size))



# =============================================================================
# Convenience: Memory Profiling Utilities
# =============================================================================


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state."""

    timestamp: float
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int

    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / (1024**3)

    @property
    def reserved_gb(self) -> float:
        return self.reserved_bytes / (1024**3)


class MemoryTracker:
    """Track GPU memory usage over time.

    Useful for debugging memory issues and identifying memory leaks.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.snapshot("before_forward")
        >>> output = model(input)
        >>> tracker.snapshot("after_forward")
        >>> tracker.report()
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self._start_time = time.time()

    def snapshot(self, name: str) -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            name: Identifier for this snapshot.

        Returns:
            MemorySnapshot with current memory state.
        """
        if not torch.cuda.is_available():
            snap = MemorySnapshot(
                timestamp=time.time() - self._start_time,
                allocated_bytes=0,
                reserved_bytes=0,
                max_allocated_bytes=0,
                max_reserved_bytes=0,
            )
        else:
            snap = MemorySnapshot(
                timestamp=time.time() - self._start_time,
                allocated_bytes=torch.cuda.memory_allocated(self.device),
                reserved_bytes=torch.cuda.memory_reserved(self.device),
                max_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                max_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            )

        self.snapshots[name] = snap
        return snap

    def diff(self, name1: str, name2: str) -> Dict[str, float]:
        """Calculate memory difference between two snapshots.

        Args:
            name1: First snapshot name (earlier).
            name2: Second snapshot name (later).

        Returns:
            Dictionary with memory differences in GB.
        """
        s1 = self.snapshots.get(name1)
        s2 = self.snapshots.get(name2)

        if s1 is None or s2 is None:
            return {}

        return {
            "allocated_diff_gb": (s2.allocated_bytes - s1.allocated_bytes) / (1024**3),
            "reserved_diff_gb": (s2.reserved_bytes - s1.reserved_bytes) / (1024**3),
            "time_diff_s": s2.timestamp - s1.timestamp,
        }

    def report(self) -> str:
        """Generate a text report of all snapshots.

        Returns:
            Formatted string report.
        """
        lines = ["Memory Tracking Report", "=" * 60]

        for name, snap in self.snapshots.items():
            lines.append(
                f"{name:30s} | Alloc: {snap.allocated_gb:6.2f} GB | "
                f"Reserved: {snap.reserved_gb:6.2f} GB | "
                f"Time: {snap.timestamp:6.2f}s"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all snapshots and reset timers."""
        self.snapshots.clear()
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
