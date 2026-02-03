"""FLA Model implementations.

This module provides Vision-Language-Action (VLA) models for robotics.

Main classes:
- Pi05Config: Configuration for Pi0.5 model
- Pi05Model: Pi0.5 model with frozen VLM + trainable action expert

Note: Pi05Model requires openpi for full functionality. The training modules
(LoRA, ReinFlow, Knowledge Insulation) work standalone without openpi.
"""

from fla.models.base import BaseModel, BaseModelConfig, Observation, Actions, IMAGE_RESOLUTION

# Conditionally import Pi05 classes - they may fail if openpi is not installed
try:
    from fla.models.pi05 import Pi05Config, Pi05Model
    _HAS_PI05 = True
except ImportError:
    Pi05Config = None
    Pi05Model = None
    _HAS_PI05 = False

__all__ = [
    "BaseModel",
    "BaseModelConfig",
    "Observation",
    "Actions",
    "IMAGE_RESOLUTION",
]

if _HAS_PI05:
    __all__.extend(["Pi05Config", "Pi05Model"])
