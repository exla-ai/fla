"""FLA (Finetune VLA) - Fine-tune Vision-Language-Action Models.

A minimal, research-focused library for fine-tuning VLA models on robotics data.

Key Features:
- Frozen backbone fine-tuning (memory efficient)
- Knowledge Insulation for preventing catastrophic forgetting
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- ReinFlow RL fine-tuning for flow matching policies
- LeRobot dataset integration
- Flow matching action generation
- Multi-GPU training support

Example:
    >>> from fla import Pi05Model, LeRobotDataLoader, Trainer
    >>> model = Pi05Model.from_pretrained("pi0.5-base")
    >>> data = LeRobotDataLoader("lerobot/aloha_sim_transfer_cube_human")
    >>> trainer = Trainer(model, data, freeze_backbone=True)
    >>> trainer.train(steps=10000)

Advanced - Knowledge Insulation:
    >>> from fla import KnowledgeInsulationConfig
    >>> ki_config = KnowledgeInsulationConfig(mode="full")

Advanced - LoRA:
    >>> from fla import LoRAConfig
    >>> lora_config = LoRAConfig(rank=16, alpha=16.0)

Advanced - ReinFlow:
    >>> from fla import ReinFlowConfig, ReinFlowTrainer
    >>> rf_config = ReinFlowConfig(algorithm="ppo")
"""

import logging

__version__ = "0.1.0"


def setup_logging(
    level: int = logging.INFO,
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """Configure logging for FLA.

    Args:
        level: Logging level (default: INFO)
        format: Log message format
        datefmt: Date format for timestamps
    """
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    # Set FLA loggers
    for name in ["fla", "fla.training", "fla.models", "fla.data", "fla.evaluation"]:
        logging.getLogger(name).setLevel(level)


def __getattr__(name):
    """Lazy import to avoid circular dependencies and heavy imports at startup."""
    # Models
    if name == "Pi05Config":
        from fla.models import Pi05Config
        return Pi05Config
    elif name == "Pi05Model":
        from fla.models import Pi05Model
        return Pi05Model
    # Data
    elif name == "LeRobotDataLoader":
        from fla.data import LeRobotDataLoader
        return LeRobotDataLoader
    # Core training
    elif name == "Trainer":
        from fla.training import Trainer
        return Trainer
    elif name == "TrainConfig":
        from fla.training import TrainConfig
        return TrainConfig
    elif name == "DistributedTrainer":
        from fla.training import DistributedTrainer
        return DistributedTrainer
    # Knowledge Insulation
    elif name == "KnowledgeInsulationConfig":
        from fla.training import KnowledgeInsulationConfig
        return KnowledgeInsulationConfig
    elif name == "apply_knowledge_insulation":
        from fla.training import apply_knowledge_insulation
        return apply_knowledge_insulation
    elif name == "DiscreteStateEncoder":
        from fla.training import DiscreteStateEncoder
        return DiscreteStateEncoder
    # LoRA
    elif name == "LoRAConfig":
        from fla.training import LoRAConfig
        return LoRAConfig
    elif name == "LoRALinear":
        from fla.training import LoRALinear
        return LoRALinear
    elif name == "save_lora_adapter":
        from fla.training import save_lora_adapter
        return save_lora_adapter
    elif name == "load_lora_adapter":
        from fla.training import load_lora_adapter
        return load_lora_adapter
    elif name == "count_lora_params":
        from fla.training import count_lora_params
        return count_lora_params
    # ReinFlow
    elif name == "ReinFlowConfig":
        from fla.training import ReinFlowConfig
        return ReinFlowConfig
    elif name == "ReinFlowTrainer":
        from fla.training import ReinFlowTrainer
        return ReinFlowTrainer
    elif name == "DPOTrainer":
        from fla.training import DPOTrainer
        return DPOTrainer
    elif name == "Trajectory":
        from fla.training import Trajectory
        return Trajectory
    elif name == "create_reinflow_trainer":
        from fla.training import create_reinflow_trainer
        return create_reinflow_trainer
    raise AttributeError(f"module 'fla' has no attribute {name!r}")


__all__ = [
    # Setup
    "setup_logging",
    # Models
    "Pi05Config",
    "Pi05Model",
    # Data
    "LeRobotDataLoader",
    # Core training
    "Trainer",
    "TrainConfig",
    "DistributedTrainer",
    # Knowledge Insulation
    "KnowledgeInsulationConfig",
    "apply_knowledge_insulation",
    "DiscreteStateEncoder",
    # LoRA
    "LoRAConfig",
    "LoRALinear",
    "save_lora_adapter",
    "load_lora_adapter",
    "count_lora_params",
    # ReinFlow
    "ReinFlowConfig",
    "ReinFlowTrainer",
    "DPOTrainer",
    "Trajectory",
    "create_reinflow_trainer",
]
