# Unified Interface API Reference

This document provides the complete API reference for the nmoe unified interface, which bridges nmoe, SGLang, and HuggingFace configurations.

## Module: `nmoe.unified`

```python
from nmoe.unified import (
    NMoEModelConfig,
    NMoEModelInterface,
    NMoERDEPConfig,
    ConfigValidationError,
    fingerprint,
)
```

---

## NMoEModelConfig

A dataclass containing all model architecture parameters needed for nmoe training, SGLang serving, and HuggingFace model export.

### Class Definition

```python
@dataclass
class NMoEModelConfig:
    """Unified configuration for nmoe models."""
```

### Attributes

#### Model Identity

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `str` | `"nmoe"` | Model type identifier |
| `architectures` | `List[str]` | `["NMoEForCausalLM"]` | Model architecture names |

#### Core Dimensions

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | `201088` | Vocabulary size (o200k_harmony tokenizer) |
| `hidden_size` | `Optional[int]` | `None` | Hidden dimension (alias: `dim` in nmoe) |
| `num_hidden_layers` | `Optional[int]` | `None` | Number of layers (alias: `n_layers`) |
| `num_attention_heads` | `Optional[int]` | `None` | Number of attention heads (alias: `n_heads`) |

#### MLP Dimensions

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `intermediate_size` | `Optional[int]` | `None` | Dense MLP intermediate dim (alias: `inter_dim`) |
| `moe_intermediate_size` | `Optional[int]` | `None` | Expert MLP intermediate dim (alias: `moe_inter_dim`) |

#### MoE Configuration

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_experts` | `Optional[int]` | `None` | Total routed experts (alias: `n_routed_experts`) |
| `num_experts_per_tok` | `Optional[int]` | `None` | TopK experts per token (alias: `n_activated_experts`) |
| `n_shared_experts` | `int` | `2` | Shared experts (fused into single MLP) |
| `first_k_dense_replace` | `int` | `1` | First N layers are dense (alias: `n_dense_layers`) |
| `router_bias_update_rate` | `float` | `1e-4` | Aux-free bias update rate |
| `router_aux_loss_coef` | `float` | `0.0` | Load balancing aux loss coefficient |
| `norm_topk_prob` | `bool` | `True` | Normalize TopK probabilities |
| `routed_scaling_factor` | `float` | `1.0` | Expert output scaling (alias: `route_scale`) |

#### Attention Configuration

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_type` | `str` | `"mla"` | Attention type: mla, swa, nsa, dsa, kda |
| `attention_local_type` | `str` | `"swa"` | Local attention for hybrid |
| `attention_global_every` | `int` | `1` | Every Nth layer is global |
| `attention_local_window` | `int` | `128` | Local attention window size |
| `q_lora_rank` | `int` | `1536` | Q LoRA rank for MLA |
| `kv_lora_rank` | `int` | `512` | KV LoRA rank for MLA |
| `qk_nope_head_dim` | `int` | `128` | QK no-PE head dimension |
| `qk_rope_head_dim` | `int` | `64` | QK RoPE head dimension |
| `v_head_dim` | `int` | `128` | Value head dimension |

#### RoPE Configuration

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_position_embeddings` | `int` | `8192` | Maximum sequence length |
| `rope_theta` | `float` | `50000.0` | RoPE base frequency |
| `rope_scaling` | `Optional[Dict]` | `None` | HF-style rope_scaling dict |
| `rope_scaling_factor` | `float` | `1.0` | Context extension factor |
| `rope_ntk_alpha` | `float` | `1.0` | NTK-aware alpha |
| `rope_ntk_beta` | `float` | `32.0` | NTK-aware beta |

#### Normalization

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `rms_norm_eps` | `float` | `1e-5` | RMSNorm epsilon |

#### Tokenizer

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer_name` | `str` | `"o200k_harmony"` | Tokenizer name |
| `eos_token_id` | `int` | `199999` | End of sequence token ID |
| `bos_token_id` | `Optional[int]` | `None` | Beginning of sequence token ID |
| `pad_token_id` | `Optional[int]` | `None` | Padding token ID (defaults to eos) |

#### Precision

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `torch_dtype` | `str` | `"bfloat16"` | PyTorch dtype string |
| `quantization` | `Optional[str]` | `None` | Quantization: None, fp8, nvfp4 |

### Properties

#### nmoe Naming Aliases

```python
# These properties provide nmoe-style naming for compatibility
config.dim           # -> hidden_size
config.n_layers      # -> num_hidden_layers
config.n_heads       # -> num_attention_heads
config.inter_dim     # -> intermediate_size
config.moe_inter_dim # -> moe_intermediate_size
config.n_routed_experts    # -> num_experts
config.n_activated_experts # -> num_experts_per_tok
config.n_dense_layers      # -> first_k_dense_replace
config.attn          # -> attention_type
config.aux_loss_alpha # -> router_aux_loss_coef
config.route_scale   # -> routed_scaling_factor
config.tokenizer     # -> tokenizer_name
config.dtype         # -> "bf16"/"fp8"/"nvfp4" based on quantization
```

#### Computed Properties

```python
config.is_moe        # bool: True if num_experts > 0
config.total_experts # int: num_experts + n_shared_experts
config.head_dim      # int: qk_nope_head_dim + qk_rope_head_dim
```

### Methods

#### Validation

```python
def validate(self) -> None:
    """Validate that all required fields are set.

    Raises:
        ConfigValidationError: If required fields are missing

    Example:
        config = NMoEModelConfig(hidden_size=4096)
        config.validate()  # Raises: missing num_hidden_layers, num_attention_heads
    """
```

#### Serialization

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary (excludes private fields)."""

@classmethod
def from_dict(cls, d: Dict[str, Any]) -> "NMoEModelConfig":
    """Create config from dictionary (filters unknown fields)."""

def copy(self, **updates) -> "NMoEModelConfig":
    """Create a copy with optional field updates."""

def fingerprint(self) -> str:
    """Compute SHA-256 fingerprint for comparison."""
```

#### nmoe Conversion

```python
@classmethod
def from_nmoe_config(cls, cfg: Any) -> "NMoEModelConfig":
    """Create from nmoe.config.Config instance.

    Args:
        cfg: nmoe Config dataclass or dict

    Returns:
        NMoEModelConfig with mapped fields

    Example:
        from nmoe.config import Config
        nmoe_cfg = Config(dim=4096, n_layers=32, ...)
        unified = NMoEModelConfig.from_nmoe_config(nmoe_cfg)
    """
```

#### HuggingFace Conversion

```python
@classmethod
def from_hf_config(cls, hf_config: Any) -> "NMoEModelConfig":
    """Create from HuggingFace PretrainedConfig.

    Args:
        hf_config: HuggingFace config or dict

    Returns:
        NMoEModelConfig instance
    """

def to_hf_config(self) -> Dict[str, Any]:
    """Export to HuggingFace config dict.

    Returns:
        Dict suitable for config.json

    Example:
        hf_dict = config.to_hf_config()
        with open("config.json", "w") as f:
            json.dump(hf_dict, f)
    """
```

#### SGLang Conversion

```python
def to_sglang_server_args(self) -> Dict[str, Any]:
    """Export to SGLang server arguments.

    Returns:
        Dict with context_length, dtype, quantization, moe_runner_backend

    Example:
        args = config.to_sglang_server_args()
        # {'context_length': 8192, 'dtype': 'bfloat16', 'moe_runner_backend': 'nmoe'}
    """
```

### Usage Examples

```python
from nmoe.unified import NMoEModelConfig

# Create a new config
config = NMoEModelConfig(
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_experts=256,
    num_experts_per_tok=8,
    moe_intermediate_size=1536,
)

# Validate
config.validate()

# Check MoE status
print(f"Is MoE: {config.is_moe}")  # True
print(f"Total experts: {config.total_experts}")  # 258 (256 + 2 shared)

# Use nmoe aliases
print(f"dim={config.dim}, n_layers={config.n_layers}")

# Export to HuggingFace
hf_config = config.to_hf_config()

# Export to SGLang
sglang_args = config.to_sglang_server_args()

# Create fingerprint for comparison
fp = config.fingerprint()
```

---

## NMoEModelInterface

Abstract base class defining the interface that SGLang and SkyRL wrappers must implement.

### Class Definition

```python
class NMoEModelInterface(ABC):
    """Abstract interface for nmoe models in serving and training contexts."""
```

### Abstract Methods

#### Core Forward Pass

```python
@abstractmethod
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    use_cache: bool = False,
) -> Dict[str, torch.Tensor]:
    """Forward pass returning logits and auxiliary outputs.

    Args:
        input_ids: Token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        position_ids: Position IDs for RoPE [batch, seq_len]
        past_key_values: KV cache for incremental decoding
        use_cache: Whether to return updated KV cache

    Returns:
        Dict containing:
            - 'logits': [batch, seq_len, vocab_size]
            - 'past_key_values': Updated KV cache (if use_cache=True)
            - 'router_logits': Router logits for aux loss (optional)
    """
```

#### Generation

```python
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
```

#### RL Training Support

```python
@abstractmethod
def forward_with_log_probs(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    action_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning log probabilities for RL.

    Args:
        input_ids: Full sequence token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        action_ids: Action token IDs for log prob computation

    Returns:
        Tuple of:
            - log_probs: Log probabilities [batch, action_len]
            - entropy: Entropy of distribution [batch]
    """
```

#### Expert Cache Management

```python
@abstractmethod
def refresh_expert_caches(self) -> None:
    """Refresh quantized weight caches after optimizer step.

    Must be called after optimizer.step() when using FP8/NVFP4.
    """

@property
@abstractmethod
def uses_quantized_experts(self) -> bool:
    """Whether model uses quantized expert weights."""
```

#### Load Balancing

```python
@abstractmethod
def get_router_aux_loss(self) -> torch.Tensor:
    """Get auxiliary load balancing loss from routers.

    Returns:
        Scalar tensor with aux loss (0 if aux_loss_alpha=0)
    """

@abstractmethod
def get_expert_load_stats(self) -> Dict[str, torch.Tensor]:
    """Get expert load statistics for monitoring.

    Returns:
        Dict containing:
            - 'loads': [n_layers, n_experts]
            - 'load_mean': Mean load
            - 'load_std': Load standard deviation
    """

@abstractmethod
def update_router_biases(self, gamma: float = 0.001) -> None:
    """Update router biases for aux-free load balancing.

    Args:
        gamma: Update rate for bias adjustment
    """
```

#### Gradient Checkpointing

```python
@abstractmethod
def gradient_checkpointing_enable(
    self,
    gradient_checkpointing_kwargs: Optional[Dict] = None
) -> None:
    """Enable gradient checkpointing."""

@abstractmethod
def gradient_checkpointing_disable(self) -> None:
    """Disable gradient checkpointing."""

@property
@abstractmethod
def is_gradient_checkpointing(self) -> bool:
    """Whether gradient checkpointing is enabled."""
```

#### Model Properties

```python
@property
@abstractmethod
def config(self) -> Any:
    """Get model configuration."""

@property
@abstractmethod
def device(self) -> torch.device:
    """Get model device."""

@property
@abstractmethod
def dtype(self) -> torch.dtype:
    """Get model dtype."""

@abstractmethod
def get_input_embeddings(self) -> nn.Module:
    """Get input embedding layer."""

@abstractmethod
def get_output_embeddings(self) -> nn.Module:
    """Get output (lm_head) embedding layer."""
```

#### Parameter Access

```python
@abstractmethod
def param_sets(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Get parameter sets for different learning rates.

    Returns:
        Tuple of (expert_params, dense_params)
    """

@abstractmethod
def named_parameters_by_type(self) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
    """Get named parameters organized by type.

    Returns:
        Dict with keys: 'expert', 'router', 'attention', 'dense', 'embedding'
    """
```

#### State Dict

```python
@abstractmethod
def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
    """Get state dict suitable for checkpoint saving."""

@abstractmethod
def load_state_dict_from_checkpoint(
    self,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load state dict from checkpoint."""
```

### Implementation Example

```python
from nmoe.unified import NMoEModelInterface
import torch
import torch.nn as nn

class MyNMoEWrapper(NMoEModelInterface):
    def __init__(self, model_path: str):
        self.model = load_nmoe_model(model_path)
        self._gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return {"logits": outputs.logits}

    def generate(self, input_ids, max_new_tokens=128, **kwargs):
        return self.model.generate(input_ids, max_length=input_ids.size(1) + max_new_tokens)

    def forward_with_log_probs(self, input_ids, attention_mask=None, action_ids=None):
        logits = self.forward(input_ids, attention_mask)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather log probs for actions
        if action_ids is not None:
            action_log_probs = log_probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
        else:
            action_log_probs = log_probs[..., -1, :]
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        return action_log_probs, entropy

    def refresh_expert_caches(self):
        for layer in self.model.layers:
            if hasattr(layer, "moe"):
                layer.moe.refresh_cache()

    @property
    def uses_quantized_experts(self) -> bool:
        return self.model.config.quantization is not None

    # ... implement remaining abstract methods
```

---

## NMoERDEPConfig

Configuration for the RDEP (Redistribution Expert Parallelism) dispatcher.

### Class Definition

```python
@dataclass
class NMoERDEPConfig:
    """Configuration for RDEP dispatcher."""
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"auto"` | Mode: auto, single, ipc, hybrid |
| `profile` | `str` | `"bf16"` | Quantization: bf16, fp8, nvfp4 |
| `capacity` | `int` | `65536` | Max tokens per expert buffer |
| `nvshmem_enabled` | `bool` | `False` | Enable NVSHMEM for multi-node |
| `nvshmem_heap_size` | `int` | `1073741824` | NVSHMEM heap size (1GB) |
| `ipc_barrier_timeout_ms` | `int` | `5000` | IPC barrier timeout |

### Methods

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary."""

@classmethod
def from_dict(cls, d: Dict[str, Any]) -> "NMoERDEPConfig":
    """Create from dictionary."""

def fingerprint(self) -> str:
    """Compute stable fingerprint."""

def get_profile_id(self) -> int:
    """Get numeric profile ID: -1=bf16, 0=fp8, 1=nvfp4."""

def detect_mode(self, world_size: int, local_world_size: int) -> str:
    """Auto-detect mode based on GPU topology."""
```

### Usage Example

```python
from nmoe.unified import NMoERDEPConfig

# Create RDEP config
rdep_config = NMoERDEPConfig(
    mode="auto",
    profile="nvfp4",
    capacity=131072,
)

# Detect mode based on topology
mode = rdep_config.detect_mode(world_size=8, local_world_size=8)
# Returns "ipc" (single node multi-GPU)

# Get profile ID for CUDA kernels
profile_id = rdep_config.get_profile_id()  # 1 for nvfp4
```

---

## Helper Functions

### fingerprint

```python
def fingerprint(cfg) -> str:
    """Compute stable config fingerprint.

    Args:
        cfg: Dataclass config or object with to_dict() method

    Returns:
        SHA-256 hex digest string

    Example:
        from nmoe.unified import fingerprint, NMoEModelConfig

        cfg1 = NMoEModelConfig(hidden_size=4096)
        cfg2 = NMoEModelConfig(hidden_size=4096)

        assert fingerprint(cfg1) == fingerprint(cfg2)
    """
```

---

## Exceptions

### ConfigValidationError

```python
class ConfigValidationError(Exception):
    """Raised when config validation fails."""
```

Raised by `NMoEModelConfig.validate()` when required fields are missing.

---

## Type Annotations

The interface uses forward references for torch types to allow importing without torch:

```python
# These are string annotations evaluated at runtime
torch.Tensor -> "torch.Tensor"
torch.device -> "torch.device"
torch.dtype -> "torch.dtype"
nn.Module -> "nn.Module"
```

---

## Related Documentation

- [SGLang Integration Guide](../integration/sglang.md)
- [SkyRL Integration Guide](../integration/skyrl.md)
- [Checkpoint Format](../checkpoints.md)
- [RDEP Dispatcher](../rdep.md)
