"""nmoe - Mixture of Experts training framework with RDEP dispatch.

This package provides:
- Config: Model configuration utilities
- Model: Transformer with MoE layers
- MoE: Mixture of Experts layer with RDEP dispatch
- Rdep: Router Dispatch Expert Protocol for efficient MoE execution
- Checkpoint: Versioned checkpoint save/load with EP sharding
- Distributed: EP/TP coordination and process group utilities
- Optimization: Fused kernels, CUDA graphs, memory optimization

Usage:
    from nmoe import Config, Transformer, Rdep
    from nmoe.unified import NMoEModelConfig, NMoEModelInterface
    from nmoe.fused_router import FusedRouterTopKDispatch
    from nmoe.persistent_dispatch import PersistentDecodeRunner
    from nmoe.memory_opt import LazyExpertWeights, DynamicBatchSizer
"""

# Core components
from nmoe.config import Config
from nmoe.model import Transformer, TransformerBlock, MoE, MLP, Router
from nmoe.rdep import Rdep, CudaGraphDispatch

# MoE autograd functions (internal use - exported for advanced users)
from nmoe.moe import _MoEBlockscaledFused

# Optimization components
from nmoe.fused_router import FusedRouterTopKDispatch
from nmoe.persistent_dispatch import (
    PersistentDispatchQueue,
    PersistentDecodeRunner,
    PersistentWorkItem,
)
from nmoe.memory_opt import (
    LazyExpertWeights,
    ExpertCachePolicy,
    LRUPolicy,
    LFUPolicy,
    FIFOPolicy,
    MemoryEfficientGradAccum,
    DynamicBatchSizer,
    MemoryTracker,
    moe_forward_with_checkpointing,
    CheckpointedMoEWrapper,
)

# Error handling
from nmoe.cuda_errors import (
    CudaError,
    NvshmemError,
    RdepError,
    cuda_error_context,
    with_cuda_error_check,
    validate_tensor_device,
    validate_tensor_contiguous,
    validate_tensor_dtype,
)

# Checkpoint utilities
from nmoe.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    EPShardInfo,
    reshard_expert_weights,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Config",
    "Transformer",
    "TransformerBlock",
    "MoE",
    "MLP",
    "Router",
    "_MoEBlockscaledFused",
    "Rdep",
    "CudaGraphDispatch",
    # Optimization
    "FusedRouterTopKDispatch",
    "PersistentDispatchQueue",
    "PersistentDecodeRunner",
    "PersistentWorkItem",
    "LazyExpertWeights",
    "ExpertCachePolicy",
    "LRUPolicy",
    "LFUPolicy",
    "FIFOPolicy",
    "MemoryEfficientGradAccum",
    "DynamicBatchSizer",
    "MemoryTracker",
    "moe_forward_with_checkpointing",
    "CheckpointedMoEWrapper",
    # Error handling
    "CudaError",
    "NvshmemError",
    "RdepError",
    "cuda_error_context",
    "with_cuda_error_check",
    "validate_tensor_device",
    "validate_tensor_contiguous",
    "validate_tensor_dtype",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "EPShardInfo",
    "reshard_expert_weights",
]
