"""Abstract base class for nmoe model interface.

This module defines the common interface that both SGLang and SkyRL
wrappers must implement for nmoe model integration.

Note: This module uses string annotations for torch types to allow importing
without torch installed. The actual implementation classes will require torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, List

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class NMoEModelInterface(ABC):
    """Abstract interface for nmoe models in serving and training contexts.

    This interface defines the contract that both SGLang (serving) and SkyRL
    (RL training) wrappers must implement for nmoe models.

    Key design principles:
    - Unified forward pass semantics across serving and training
    - Explicit expert cache management for quantized weights
    - Standardized load balancing metrics access
    - Gradient checkpointing support for memory efficiency
    """

    # =========================================================================
    # Core Forward Pass
    # =========================================================================

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits and optional auxiliary outputs.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs for RoPE [batch, seq_len]
            past_key_values: KV cache for incremental decoding
            use_cache: Whether to return updated KV cache

        Returns:
            Dict containing:
                - 'logits': Output logits [batch, seq_len, vocab_size]
                - 'past_key_values': Updated KV cache (if use_cache=True)
                - 'router_logits': Router logits for aux loss (optional)
        """
        pass

    # =========================================================================
    # Generation (Inference)
    # =========================================================================

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs [batch, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            do_sample: Whether to sample (False = greedy)

        Returns:
            Generated token IDs [batch, prompt_len + generated_len]
        """
        pass

    # =========================================================================
    # RL Training Support
    # =========================================================================

    @abstractmethod
    def forward_with_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning log probabilities for RL training.

        Args:
            input_ids: Full sequence token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            action_ids: Action token IDs for log prob computation

        Returns:
            Tuple of:
                - log_probs: Log probabilities of actions [batch, action_len]
                - entropy: Entropy of action distribution [batch]
        """
        pass

    # =========================================================================
    # Expert Cache Management
    # =========================================================================

    @abstractmethod
    def refresh_expert_caches(self) -> None:
        """Refresh quantized weight caches after optimizer step.

        Must be called after each optimizer.step() when using FP8/NVFP4
        quantization to update the cached quantized expert weights.
        """
        pass

    @property
    @abstractmethod
    def uses_quantized_experts(self) -> bool:
        """Whether this model uses quantized expert weights (FP8/NVFP4)."""
        pass

    # =========================================================================
    # Load Balancing
    # =========================================================================

    @abstractmethod
    def get_router_aux_loss(self) -> torch.Tensor:
        """Get auxiliary load balancing loss from routers.

        Returns:
            Scalar tensor with auxiliary loss (0 if aux_loss_alpha=0)
        """
        pass

    @abstractmethod
    def get_expert_load_stats(self) -> Dict[str, torch.Tensor]:
        """Get expert load statistics for monitoring.

        Returns:
            Dict containing:
                - 'loads': Expert load per layer [n_layers, n_experts]
                - 'load_mean': Mean load across experts
                - 'load_std': Std dev of loads (imbalance metric)
        """
        pass

    @abstractmethod
    def update_router_biases(self, gamma: float = 0.001) -> None:
        """Update router biases for aux-free load balancing.

        Args:
            gamma: Update rate for bias adjustment
        """
        pass

    # =========================================================================
    # Gradient Checkpointing
    # =========================================================================

    @abstractmethod
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict] = None) -> None:
        """Enable gradient checkpointing for memory efficiency.

        Args:
            gradient_checkpointing_kwargs: Additional kwargs for checkpointing
        """
        pass

    @abstractmethod
    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        pass

    @property
    @abstractmethod
    def is_gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is currently enabled."""
        pass

    # =========================================================================
    # Model Properties
    # =========================================================================

    @property
    @abstractmethod
    def config(self) -> Any:
        """Get model configuration."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get model device."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        pass

    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer."""
        pass

    @abstractmethod
    def get_output_embeddings(self) -> nn.Module:
        """Get the output (lm_head) embedding layer."""
        pass

    # =========================================================================
    # Parameter Access
    # =========================================================================

    @abstractmethod
    def param_sets(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Get parameter sets for different learning rates.

        Returns:
            Tuple of:
                - expert_params: MoE expert parameters (W1, W3, W2)
                - dense_params: All other parameters
        """
        pass

    @abstractmethod
    def named_parameters_by_type(self) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        """Get named parameters organized by type.

        Returns:
            Dict with keys: 'expert', 'router', 'attention', 'dense', 'embedding'
        """
        pass

    # =========================================================================
    # State Dict
    # =========================================================================

    @abstractmethod
    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """Get state dict suitable for checkpoint saving.

        Returns:
            State dict with all model weights
        """
        pass

    @abstractmethod
    def load_state_dict_from_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Load state dict from checkpoint.

        Args:
            state_dict: State dict to load
            strict: Whether to require exact key matching
        """
        pass
