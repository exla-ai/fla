"""FLA Training Module.

Provides training infrastructure for VLA fine-tuning.

Main classes:
- Trainer: Main training loop
- TrainConfig: Training configuration
- LoRAConfig: Configuration for LoRA fine-tuning
- KnowledgeInsulationConfig: Configuration for knowledge insulation
- ReinFlowConfig: Configuration for RL fine-tuning
"""

# Use explicit imports for lightweight modules
from fla.training.optimizer import create_optimizer, CosineSchedule
from fla.training.checkpoints import save_checkpoint, load_checkpoint


def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == "Trainer":
        from fla.training.trainer import Trainer
        return Trainer
    elif name == "TrainConfig":
        from fla.training.trainer import TrainConfig
        return TrainConfig
    elif name == "DistributedTrainer":
        from fla.training.trainer import DistributedTrainer
        return DistributedTrainer
    # Knowledge Insulation
    elif name == "KnowledgeInsulationConfig":
        from fla.training.knowledge_insulation import KnowledgeInsulationConfig
        return KnowledgeInsulationConfig
    elif name == "apply_knowledge_insulation":
        from fla.training.knowledge_insulation import apply_knowledge_insulation
        return apply_knowledge_insulation
    elif name == "insulate_prefix_tokens":
        from fla.training.knowledge_insulation import insulate_prefix_tokens
        return insulate_prefix_tokens
    elif name == "DiscreteStateEncoder":
        from fla.training.knowledge_insulation import DiscreteStateEncoder
        return DiscreteStateEncoder
    # LoRA
    elif name == "LoRAConfig":
        from fla.training.lora import LoRAConfig
        return LoRAConfig
    elif name == "LoRALinear":
        from fla.training.lora import LoRALinear
        return LoRALinear
    elif name == "save_lora_adapter":
        from fla.training.lora import save_lora_adapter
        return save_lora_adapter
    elif name == "load_lora_adapter":
        from fla.training.lora import load_lora_adapter
        return load_lora_adapter
    elif name == "get_lora_params_filter":
        from fla.training.lora import get_lora_params_filter
        return get_lora_params_filter
    elif name == "count_lora_params":
        from fla.training.lora import count_lora_params
        return count_lora_params
    elif name == "get_lora_gemma_variant":
        from fla.training.lora import get_lora_gemma_variant
        return get_lora_gemma_variant
    elif name == "create_lora_config":
        from fla.training.lora import create_lora_config
        return create_lora_config
    # ReinFlow
    elif name == "ReinFlowConfig":
        from fla.training.reinflow import ReinFlowConfig
        return ReinFlowConfig
    elif name == "ReinFlowTrainer":
        from fla.training.reinflow import ReinFlowTrainer
        return ReinFlowTrainer
    elif name == "DPOTrainer":
        from fla.training.reinflow import DPOTrainer
        return DPOTrainer
    elif name == "Trajectory":
        from fla.training.reinflow import Trajectory
        return Trajectory
    elif name == "create_reinflow_trainer":
        from fla.training.reinflow import create_reinflow_trainer
        return create_reinflow_trainer
    raise AttributeError(f"module 'fla.training' has no attribute {name!r}")


__all__ = [
    # Core training
    "Trainer",
    "TrainConfig",
    "DistributedTrainer",
    "create_optimizer",
    "CosineSchedule",
    "save_checkpoint",
    "load_checkpoint",
    # Knowledge Insulation
    "KnowledgeInsulationConfig",
    "apply_knowledge_insulation",
    "insulate_prefix_tokens",
    "DiscreteStateEncoder",
    # LoRA
    "LoRAConfig",
    "LoRALinear",
    "save_lora_adapter",
    "load_lora_adapter",
    "get_lora_params_filter",
    "count_lora_params",
    "get_lora_gemma_variant",
    "create_lora_config",
    # ReinFlow
    "ReinFlowConfig",
    "ReinFlowTrainer",
    "DPOTrainer",
    "Trajectory",
    "create_reinflow_trainer",
]
