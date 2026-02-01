"""NMoE Unified Configuration Module.

This module provides a unified configuration dataclass (NMoEModelConfig) that bridges
nmoe training configs, SGLang serving configs, and HuggingFace PretrainedConfig.

The unified config enables:
- Seamless weight transfer between training and serving
- Consistent model architecture specification
- Config fingerprinting for reproducibility
"""

from nmoe.unified.config import (
    NMoEModelConfig,
    NMoERDEPConfig,
    fingerprint,
    ConfigValidationError,
)
from nmoe.unified.interface import NMoEModelInterface

__all__ = [
    "NMoEModelConfig",
    "NMoERDEPConfig",
    "NMoEModelInterface",
    "fingerprint",
    "ConfigValidationError",
]
