"""Unified NMoE Model Configuration.

This module provides a unified config dataclass that bridges nmoe, SGLang, and
HuggingFace configurations for seamless interoperability.

Usage:
    # From nmoe config
    unified = NMoEModelConfig.from_nmoe_config(nmoe_cfg)

    # To HuggingFace config dict
    hf_dict = unified.to_hf_config()

    # To SGLang server args
    server_args = unified.to_sglang_server_args()
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Dict, Any, List
import dataclasses
import hashlib
import json


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def fingerprint(cfg) -> str:
    """Stable config fingerprint for reproducible resume checks.

    Uses a canonical JSON encoding of the dataclass fields (sorted keys) and
    returns a SHA-256 hex digest. Private fields (starting with _) are excluded.

    Args:
        cfg: Dataclass config instance (NMoEModelConfig or nmoe.config.Config)

    Returns:
        SHA-256 hex digest string
    """
    if dataclasses.is_dataclass(cfg):
        d = asdict(cfg)
    elif hasattr(cfg, "to_dict"):
        d = cfg.to_dict()
    else:
        d = {"value": str(cfg)}
    # Exclude private fields
    d = {k: v for k, v in d.items() if not str(k).startswith("_")}
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class NMoEModelConfig:
    """Unified configuration for nmoe models.

    This dataclass contains all model architecture parameters needed for:
    - nmoe training and inference
    - SGLang serving
    - HuggingFace model export

    Attributes are organized by category with sensible defaults matching
    the nmoe codebase conventions.

    Field naming convention:
    - Uses HuggingFace-style naming (hidden_size, num_hidden_layers) for
      compatibility with AutoConfig and transformers ecosystem
    - See docstrings for mapping to nmoe-style names (dim, n_layers)
    """

    # ==========================================================================
    # Model Identity
    # ==========================================================================
    model_type: str = "nmoe"
    architectures: List[str] = field(default_factory=lambda: ["NMoEForCausalLM"])

    # ==========================================================================
    # Core Dimensions (REQUIRED)
    # ==========================================================================
    vocab_size: int = 201088  # o200k_harmony tokenizer
    hidden_size: Optional[int] = None  # Alias: dim in nmoe
    num_hidden_layers: Optional[int] = None  # Alias: n_layers in nmoe
    num_attention_heads: Optional[int] = None  # Alias: n_heads in nmoe

    # ==========================================================================
    # MLP Dimensions
    # ==========================================================================
    intermediate_size: Optional[int] = None  # Alias: inter_dim in nmoe (dense MLP)
    moe_intermediate_size: Optional[int] = None  # Alias: moe_inter_dim in nmoe (expert MLP)

    # ==========================================================================
    # MoE Configuration
    # ==========================================================================
    num_experts: Optional[int] = None  # Alias: n_routed_experts in nmoe
    num_experts_per_tok: Optional[int] = None  # Alias: n_activated_experts (topk)
    n_shared_experts: int = 2  # Shared experts (fused into single MLP)
    first_k_dense_replace: int = 1  # Alias: n_dense_layers (first N layers are dense)

    # MoE Routing
    router_bias_update_rate: float = 1e-4
    router_aux_loss_coef: float = 0.0  # Alias: aux_loss_alpha
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0  # Alias: route_scale

    # ==========================================================================
    # Attention Configuration (MLA defaults)
    # ==========================================================================
    attention_type: str = "mla"  # mla | swa | nsa | dsa | kda
    attention_local_type: str = "swa"  # For hybrid attention
    attention_global_every: int = 1  # Every Nth layer is global
    attention_local_window: int = 128  # Window size for local attention

    # MLA-specific dimensions
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # Derived head dimension (for compatibility)
    @property
    def head_dim(self) -> int:
        """Total head dimension for MLA."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    # ==========================================================================
    # RoPE Configuration
    # ==========================================================================
    max_position_embeddings: int = 8192
    rope_theta: float = 50000.0
    rope_scaling: Optional[Dict[str, Any]] = None  # HF-style rope_scaling dict

    # nmoe-specific RoPE params (converted to rope_scaling for HF)
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    # ==========================================================================
    # Normalization
    # ==========================================================================
    rms_norm_eps: float = 1e-5

    # ==========================================================================
    # Tokenizer
    # ==========================================================================
    tokenizer_name: str = "o200k_harmony"
    eos_token_id: int = 199999
    bos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None  # Defaults to eos_token_id

    # ==========================================================================
    # Precision / Quantization
    # ==========================================================================
    torch_dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    quantization: Optional[str] = None  # None | fp8 | nvfp4

    # ==========================================================================
    # Backend-specific nested configs
    # ==========================================================================
    attn_swa: Dict[str, Any] = field(default_factory=dict)
    attn_nsa: Dict[str, Any] = field(default_factory=dict)
    attn_dsa: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================================
    # Metadata
    # ==========================================================================
    _nmoe_version: str = "1.0"

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Set pad_token_id to eos_token_id if not specified
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id

    def validate(self) -> None:
        """Validate that all required fields are set.

        Raises:
            ConfigValidationError: If required fields are missing
        """
        required = ['hidden_size', 'num_hidden_layers', 'num_attention_heads']
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ConfigValidationError(
                f"Required fields missing: {', '.join(missing)}"
            )

        # Validate MoE fields if this is an MoE model
        if self.num_experts is not None:
            moe_required = ["num_experts_per_tok", "moe_intermediate_size"]
            moe_missing = [f for f in moe_required if getattr(self, f) is None]
            if moe_missing:
                raise ConfigValidationError(
                    f"MoE model requires: {', '.join(moe_missing)}"
                )

    @property
    def is_moe(self) -> bool:
        """Check if this config represents an MoE model."""
        return self.num_experts is not None and self.num_experts > 0

    @property
    def total_experts(self) -> int:
        """Total number of experts including shared."""
        if not self.is_moe:
            return 0
        return (self.num_experts or 0) + self.n_shared_experts

    # ==========================================================================
    # nmoe naming aliases (for compatibility with nmoe.config.Config style)
    # ==========================================================================
    @property
    def dim(self) -> Optional[int]:
        """Alias for hidden_size (nmoe naming)."""
        return self.hidden_size

    @property
    def n_layers(self) -> Optional[int]:
        """Alias for num_hidden_layers (nmoe naming)."""
        return self.num_hidden_layers

    @property
    def n_heads(self) -> Optional[int]:
        """Alias for num_attention_heads (nmoe naming)."""
        return self.num_attention_heads

    @property
    def inter_dim(self) -> Optional[int]:
        """Alias for intermediate_size (nmoe naming)."""
        return self.intermediate_size

    @property
    def moe_inter_dim(self) -> Optional[int]:
        """Alias for moe_intermediate_size (nmoe naming)."""
        return self.moe_intermediate_size

    @property
    def n_routed_experts(self) -> Optional[int]:
        """Alias for num_experts (nmoe naming)."""
        return self.num_experts

    @property
    def n_activated_experts(self) -> Optional[int]:
        """Alias for num_experts_per_tok (nmoe naming)."""
        return self.num_experts_per_tok

    @property
    def n_dense_layers(self) -> int:
        """Alias for first_k_dense_replace (nmoe naming)."""
        return self.first_k_dense_replace

    @property
    def attn(self) -> str:
        """Alias for attention_type (nmoe naming)."""
        return self.attention_type

    @property
    def attn_local(self) -> str:
        """Alias for attention_local_type (nmoe naming)."""
        return self.attention_local_type

    @property
    def attn_global_every(self) -> int:
        """Alias for attention_global_every (nmoe naming)."""
        return self.attention_global_every

    @property
    def attn_local_window(self) -> int:
        """Alias for attention_local_window (nmoe naming)."""
        return self.attention_local_window

    @property
    def aux_loss_alpha(self) -> float:
        """Alias for router_aux_loss_coef (nmoe naming)."""
        return self.router_aux_loss_coef

    @property
    def route_scale(self) -> float:
        """Alias for routed_scaling_factor (nmoe naming)."""
        return self.routed_scaling_factor

    @property
    def tokenizer(self) -> str:
        """Alias for tokenizer_name (nmoe naming)."""
        return self.tokenizer_name

    @property
    def dtype(self) -> str:
        """Get nmoe-style dtype string (bf16/fp8/nvfp4)."""
        if self.quantization == "fp8":
            return "fp8"
        elif self.quantization in ("nvfp4", "modelopt_fp4"):
            return "nvfp4"
        elif self.torch_dtype == "bfloat16":
            return "bf16"
        elif self.torch_dtype == "float16":
            return "fp16"
        return self.torch_dtype

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict with all config fields, excluding private fields starting with '_'.
        """
        d = asdict(self)
        # Remove private fields
        d = {k: v for k, v in d.items() if not k.startswith('_')}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NMoEModelConfig":
        """Create config from dictionary.

        Args:
            d: Dictionary with config fields

        Returns:
            NMoEModelConfig instance
        """
        # Filter to only known fields
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def fingerprint(self) -> str:
        """Compute stable fingerprint for config comparison.

        Returns:
            SHA-256 hex digest of canonical JSON representation.
        """
        return fingerprint(self)

    def copy(self, **updates) -> "NMoEModelConfig":
        """Create a copy of this config with optional field updates.

        Args:
            **updates: Field values to update in the copy

        Returns:
            New NMoEModelConfig instance
        """
        d = self.to_dict()
        d.update(updates)
        return self.from_dict(d)

    # ==========================================================================
    # nmoe Config Conversion (Task 1.1.2)
    # ==========================================================================

    @classmethod
    def from_nmoe_config(cls, cfg: Any) -> "NMoEModelConfig":
        """Create unified config from nmoe Config.

        Args:
            cfg: nmoe.config.Config instance

        Returns:
            NMoEModelConfig with mapped fields
        """
        # Handle dataclass or dict input
        if dataclasses.is_dataclass(cfg):
            cfg_dict = asdict(cfg)
        elif hasattr(cfg, "__dict__"):
            cfg_dict = vars(cfg)
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            raise TypeError(f"Cannot convert {type(cfg)} to NMoEModelConfig")

        # Map dtype
        cfg_dtype = cfg_dict.get("dtype", "bf16")
        dtype_map = {
            "bf16": "bfloat16",
            "fp8": "bfloat16",  # FP8 is quantization, base is bf16
            "nvfp4": "bfloat16",
        }
        torch_dtype = dtype_map.get(cfg_dtype, "bfloat16")

        # Map quantization
        quant_map = {
            "bf16": None,
            "fp8": "fp8",
            "nvfp4": "modelopt_fp4",
        }
        quantization = quant_map.get(cfg_dtype)

        # Build rope_scaling dict if non-default
        rope_scaling = None
        rope_scaling_factor = cfg_dict.get("rope_scaling_factor", 1.0)
        if rope_scaling_factor != 1.0:
            rope_scaling = {
                "type": "yarn",
                "factor": rope_scaling_factor,
                "mscale_all_dim": cfg_dict.get("rope_ntk_alpha", 1.0),
            }

        return cls(
            # Core dimensions (nmoe -> HF naming)
            vocab_size=cfg_dict.get("vocab_size", 201088),
            hidden_size=cfg_dict.get("dim"),
            num_hidden_layers=cfg_dict.get("n_layers"),
            num_attention_heads=cfg_dict.get("n_heads"),

            # MLP dimensions
            intermediate_size=cfg_dict.get("inter_dim"),
            moe_intermediate_size=cfg_dict.get("moe_inter_dim"),

            # MoE configuration
            num_experts=cfg_dict.get("n_routed_experts"),
            num_experts_per_tok=cfg_dict.get("n_activated_experts"),
            n_shared_experts=cfg_dict.get("n_shared_experts", 2),
            first_k_dense_replace=cfg_dict.get("n_dense_layers", 1),
            router_bias_update_rate=cfg_dict.get("router_bias_update_rate", 1e-4),
            router_aux_loss_coef=cfg_dict.get("aux_loss_alpha", 0.0),
            norm_topk_prob=cfg_dict.get("norm_topk_prob", True),
            routed_scaling_factor=cfg_dict.get("route_scale", 1.0),

            # Attention configuration
            attention_type=cfg_dict.get("attn", "mla"),
            attention_local_type=cfg_dict.get("attn_local", "swa"),
            attention_global_every=cfg_dict.get("attn_global_every", 1),
            attention_local_window=cfg_dict.get("attn_local_window", 128),
            q_lora_rank=cfg_dict.get("q_lora_rank", 1536),
            kv_lora_rank=cfg_dict.get("kv_lora_rank", 512),
            qk_nope_head_dim=cfg_dict.get("qk_nope_head_dim", 128),
            qk_rope_head_dim=cfg_dict.get("qk_rope_head_dim", 64),
            v_head_dim=cfg_dict.get("v_head_dim", 128),

            # RoPE configuration
            max_position_embeddings=cfg_dict.get("max_position_embeddings", 8192),
            rope_theta=cfg_dict.get("rope_theta", 50000.0),
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            rope_ntk_alpha=cfg_dict.get("rope_ntk_alpha", 1.0),
            rope_ntk_beta=cfg_dict.get("rope_ntk_beta", 32.0),

            # Normalization
            rms_norm_eps=cfg_dict.get("rms_norm_eps", 1e-5),

            # Tokenizer
            tokenizer_name=cfg_dict.get("tokenizer", "o200k_harmony"),
            eos_token_id=cfg_dict.get("eos_token_id", 199999),

            # Precision
            torch_dtype=torch_dtype,
            quantization=quantization,

            # Backend-specific
            attn_swa=cfg_dict.get("attn_swa", {}),
            attn_nsa=cfg_dict.get("attn_nsa", {}),
            attn_dsa=cfg_dict.get("attn_dsa", {}),
        )

    # ==========================================================================
    # HuggingFace Config Conversion (Task 1.1.3)
    # ==========================================================================

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "NMoEModelConfig":
        """Create unified config from HuggingFace PretrainedConfig.

        Args:
            hf_config: HuggingFace PretrainedConfig instance

        Returns:
            NMoEModelConfig instance
        """
        # Get config as dict
        if hasattr(hf_config, "to_dict"):
            d = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            d = hf_config
        else:
            d = {}
            for f in fields(cls):
                if hasattr(hf_config, f.name):
                    d[f.name] = getattr(hf_config, f.name)

        # Map HF-specific names to our names
        hf_mappings = {
            "n_routed_experts": "num_experts",
            "num_local_experts": "num_experts",
        }
        for hf_name, our_name in hf_mappings.items():
            if hf_name in d and our_name not in d:
                d[our_name] = d.pop(hf_name)

        return cls.from_dict(d)

    def to_hf_config(self) -> Dict[str, Any]:
        """Export to HuggingFace-compatible config dict.

        Returns:
            Dict suitable for saving as config.json or loading with AutoConfig.
        """
        config = {
            # Model identity
            "model_type": self.model_type,
            "architectures": self.architectures,

            # Core dimensions
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,

            # MLP dimensions
            "intermediate_size": self.intermediate_size or (self.hidden_size * 4 if self.hidden_size else None),
            "moe_intermediate_size": self.moe_intermediate_size,

            # MoE configuration (DeepSeek-v2 style naming)
            "n_routed_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_shared_experts": self.n_shared_experts,
            "first_k_dense_replace": self.first_k_dense_replace,
            "norm_topk_prob": self.norm_topk_prob,
            "routed_scaling_factor": self.routed_scaling_factor,

            # Attention (MLA-style)
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,

            # RoPE
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,

            # Normalization
            "rms_norm_eps": self.rms_norm_eps,

            # Tokenizer
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,

            # Precision
            "torch_dtype": self.torch_dtype,
        }

        # Remove None values for cleaner output
        config = {k: v for k, v in config.items() if v is not None}

        return config

    # Alias for backwards compatibility
    def to_hf_dict(self) -> Dict[str, Any]:
        """Alias for to_hf_config()."""
        return self.to_hf_config()

    # ==========================================================================
    # SGLang Server Args Conversion (Task 1.1.4)
    # ==========================================================================

    def to_sglang_server_args(self) -> Dict[str, Any]:
        """Export to SGLang server arguments dict.

        Returns:
            Dict of arguments suitable for SGLang server initialization.
        """
        args = {
            "context_length": self.max_position_embeddings,
            "dtype": self._torch_dtype_to_sglang(self.torch_dtype),
        }

        # Add quantization if specified
        if self.quantization:
            args["quantization"] = self.quantization

        # Add MoE runner backend hint
        args["moe_runner_backend"] = "nmoe"

        return args

    def _torch_dtype_to_sglang(self, dtype: str) -> str:
        """Convert torch dtype string to SGLang dtype."""
        dtype_map = {
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
        }
        return dtype_map.get(dtype, "auto")

    def __repr__(self) -> str:
        """Compact representation showing key fields."""
        parts = ["NMoEModelConfig("]
        if self.hidden_size:
            parts.append(f"hidden_size={self.hidden_size}")
        if self.num_hidden_layers:
            parts.append(f", num_hidden_layers={self.num_hidden_layers}")
        if self.num_attention_heads:
            parts.append(f", num_attention_heads={self.num_attention_heads}")
        if self.is_moe:
            parts.append(f", experts={self.num_experts}x{self.num_experts_per_tok}")
        parts.append(")")
        return "".join(parts)


# =============================================================================
# RDEP Mode Configuration (Task 1.1.6)
# =============================================================================

@dataclass
class NMoERDEPConfig:
    """Configuration for RDEP (Redistribution Expert Parallelism) dispatcher.

    Controls how expert dispatch operates across GPUs:
    - MODE_SINGLE: Single GPU, no communication
    - MODE_IPC: Multi-GPU on single node via IPC
    - MODE_HYBRID: Multi-node via NVSHMEM
    """

    # RDEP mode selection
    mode: str = "auto"  # auto | single | ipc | hybrid

    # Quantization profile for RDEP
    profile: str = "bf16"  # bf16 | fp8 | nvfp4

    # Buffer capacity (max tokens per expert)
    capacity: int = 65536

    # Multi-node NVSHMEM settings
    nvshmem_enabled: bool = False
    nvshmem_heap_size: int = 1 << 30  # 1GB default

    # IPC settings
    ipc_barrier_timeout_ms: int = 5000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NMoERDEPConfig":
        """Create from dictionary."""
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def fingerprint(self) -> str:
        """Compute stable fingerprint for config comparison."""
        return fingerprint(self)

    def get_profile_id(self) -> int:
        """Get numeric profile ID for RDEP CUDA kernels.

        Returns:
            Profile ID: -1 for bf16, 0 for fp8, 1 for nvfp4
        """
        profile_map = {
            "bf16": -1,
            "fp8": 0,
            "nvfp4": 1,
        }
        return profile_map.get(self.profile, -1)

    def detect_mode(self, world_size: int, local_world_size: int) -> str:
        """Auto-detect RDEP mode based on GPU topology.

        Args:
            world_size: Total number of GPUs across all nodes
            local_world_size: Number of GPUs on this node

        Returns:
            Detected mode string: 'single', 'ipc', or 'hybrid'
        """
        if self.mode != "auto":
            return self.mode

        if world_size == 1:
            return "single"
        elif world_size == local_world_size:
            return "ipc"
        else:
            return "hybrid"
