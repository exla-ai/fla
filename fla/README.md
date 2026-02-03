# FLA (Finetune VLA)

A minimal, research-focused library for fine-tuning Vision-Language-Action models on robotics data.

## Features

- **Frozen Backbone Fine-tuning**: Train only the action expert (~300M params) while keeping the VLM frozen (2.4B params). Reduces memory from 38GB to 15GB per GPU.
- **Knowledge Insulation**: Prevent catastrophic forgetting with discrete/continuous token separation.
- **LoRA Support**: Parameter-efficient fine-tuning with Low-Rank Adaptation.
- **ReinFlow**: RL fine-tuning for flow matching policies (REINFORCE, PPO, DPO).
- **LeRobot Integration**: Native support for LeRobot format datasets from HuggingFace.
- **Flow Matching**: Non-autoregressive action generation for 26x faster inference.
- **Action Chunking**: Predict K=50 future actions in a single forward pass.
- **Multi-GPU Training**: Data parallel training on 8xA100.

## Installation

### Install from GitHub (Recommended)

Using [uv](https://docs.astral.sh/uv/) (fast Python package installer):

```bash
# Minimal install (core training modules)
uv pip install git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=fla

# With CUDA support (for GPU training)
uv pip install "fla[cuda] @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=fla"

# With data loading support (LeRobot datasets)
uv pip install "fla[data] @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=fla"

# With training dependencies
uv pip install "fla[train] @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=fla"

# Full installation (all dependencies)
uv pip install "fla[all] @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=fla"

# For evaluation environments
uv pip install "fla[eval] @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=fla"
```

Or with standard pip:

```bash
pip install git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=fla
```

### Install from Source (Development)

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi/fla

# Using uv (recommended)
uv pip install -e .
uv pip install -e ".[all]"
uv pip install -e ".[dev]"

# Or using pip
pip install -e .
```

### Dependency Groups

| Group | Dependencies | Use Case |
|-------|-------------|----------|
| (base) | jax, flax, optax, numpy, einops, tqdm | Core training modules |
| `cuda` | jax[cuda12] | GPU acceleration |
| `train` | orbax, safetensors, sentencepiece, tyro | Full training pipeline |
| `data` | lerobot, torch, datasets | Loading LeRobot datasets |
| `augment` | augmax, pillow | Image augmentation |
| `eval` | gymnasium, gym-aloha, mujoco | Simulation evaluation |
| `dev` | pytest, ruff, mypy | Development tools |
| `all` | All of the above | Everything |

## Quick Start

### Fine-tune on ALOHA Simulation

```python
from fla import Pi05Config, Pi05Model, LeRobotDataLoader, Trainer, TrainConfig

# Create model with frozen backbone
config = Pi05Config(freeze_vision_backbone=True)
model = Pi05Model.from_pretrained("path/to/checkpoint")

# Load data
dataloader = LeRobotDataLoader("lerobot/aloha_sim_transfer_cube_human")

# Train
trainer = Trainer(
    model,
    dataloader,
    config=TrainConfig(
        max_steps=10000,
        learning_rate=2.5e-5,
        checkpoint_dir="./checkpoints/my_experiment",
    ),
)
trainer.train()
```

### Evaluate on ALOHA Tasks

```python
from fla.evaluation import AlohaEvaluator, EvalConfig

evaluator = AlohaEvaluator(
    model,
    task="gym_aloha/AlohaTransferCube-v0",
    config=EvalConfig(num_episodes=50),
)
results = evaluator.evaluate()
print(f"Success Rate: {results.success_rate:.1%}")
```

## Supported Datasets

| Dataset | Tasks | Robot | Source |
|---------|-------|-------|--------|
| ALOHA Sim | Transfer Cube, Insertion | Bimanual (14 DOF) | HuggingFace |
| LIBERO | 130+ manipulation tasks | Franka (7 DOF) | HuggingFace |
| DROID | 92k episodes | Franka (7 DOF) | HuggingFace |
| Open X-Embodiment | 60+ datasets | Various | HuggingFace |

### Download Datasets

```bash
# Minimal for testing
python -c "from fla.data import create_dataloader; create_dataloader('lerobot/aloha_sim_transfer_cube_human')"

# LIBERO benchmark
python -c "from fla.data import create_dataloader; create_dataloader('lerobot/libero_10')"
```

## Model Architecture

FLA is built on Pi0.5, a Vision-Language-Action model:

```
┌─────────────────────────────────────────────────────────────┐
│                        Pi0.5 Model                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PaliGemma VLM (2B params)              │   │
│  │  ┌──────────────────┐  ┌──────────────────────┐    │   │
│  │  │   SigLIP Vision  │  │   Gemma-2B Language  │    │   │
│  │  │   Encoder (400M) │  │      Model (2B)      │    │   │
│  │  └──────────────────┘  └──────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                    [FROZEN DURING TRAINING]                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Action Expert (300M params)              │   │
│  │  ┌──────────────────┐  ┌──────────────────────┐    │   │
│  │  │   Gemma-300M     │  │   Flow Matching      │    │   │
│  │  │   Transformer    │  │   Action Head        │    │   │
│  │  └──────────────────┘  └──────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                  [TRAINABLE - ~300M PARAMS]                 │
└─────────────────────────────────────────────────────────────┘
```

## Fine-tuning Methods

### P0: Frozen Backbone (Default)

```python
from fla import Pi05Config

config = Pi05Config(freeze_vision_backbone=True)
```

Memory: ~15GB/GPU | Speed: ~0.5s/step | Best for: Small datasets

### P1: Knowledge Insulation

Prevent catastrophic forgetting by maintaining separation between VLM's discrete tokens and action expert's continuous embeddings.

```python
from fla import KnowledgeInsulationConfig, apply_knowledge_insulation

# Full gradient isolation (default)
ki_config = KnowledgeInsulationConfig(mode="full")

# Soft mode: scale gradients instead of stopping them
ki_config = KnowledgeInsulationConfig(mode="soft", gradient_scale=0.1)

# Selective mode: only insulate specific layers
ki_config = KnowledgeInsulationConfig(
    mode="selective",
    insulated_layers=(".*llm.*", ".*img.*")
)

# Discrete state encoding (Pi0.5 style)
from fla import DiscreteStateEncoder
encoder = DiscreteStateEncoder(state_dim=7, num_bins=256, embedding_dim=1024, rngs=rngs)
```

### P2: LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning that dramatically reduces trainable parameters while maintaining performance.

```python
from fla import LoRAConfig, LoRALinear, save_lora_adapter, load_lora_adapter

# Configure LoRA
lora_config = LoRAConfig(
    rank=16,                      # LoRA rank (higher = more capacity)
    alpha=16.0,                   # Scaling factor
    target_modules="all",         # "attention", "ffn", or "all"
    apply_to_vlm=False,           # Don't train VLM
    apply_to_action_expert=True,  # Train action expert with LoRA
    rslora=True,                  # Use rank-stabilized LoRA
)

# Create LoRA layer
layer = LoRALinear(
    in_features=1024,
    out_features=256,
    config=lora_config,
    rngs=rngs,
)

# Save only adapter weights (very small file)
save_lora_adapter(model, "adapter.pkl", lora_config)

# Load adapter into new model
load_lora_adapter(model, "adapter.pkl")

# Merge LoRA into base weights for faster inference
layer.merge_lora()
```

LoRA reduces trainable parameters by ~90%+ while achieving similar performance to full fine-tuning.

### P2: ReinFlow (RL Fine-tuning)

Reinforcement learning fine-tuning for flow matching policies using reward signals.

```python
from fla import ReinFlowConfig, ReinFlowTrainer, Trajectory, create_reinflow_trainer

# REINFORCE (simple policy gradient)
config = ReinFlowConfig(
    algorithm="reinforce",
    learning_rate=1e-5,
    gamma=0.99,
    reward_baseline="mean",  # or "value" for learned baseline
)
trainer = ReinFlowTrainer(model, config)

# PPO (Proximal Policy Optimization)
config = ReinFlowConfig(
    algorithm="ppo",
    clip_ratio=0.2,
    num_updates_per_rollout=4,
    target_kl=0.01,  # Early stopping
)
trainer = create_reinflow_trainer(model, config)

# DPO (Direct Preference Optimization) - offline RL
from fla import DPOTrainer
config = ReinFlowConfig(algorithm="dpo")
trainer = DPOTrainer(model, reference_model, config)

# Train on trajectory
trajectory = Trajectory(
    observations=[...],
    actions=[...],
    rewards=[...],
    dones=[...],
)
metrics = trainer.train_on_trajectory(rng, trajectory)
```

ReinFlow supports three algorithms:
- **REINFORCE**: Simple, works well with sparse rewards
- **PPO**: More stable, better sample efficiency
- **DPO**: Offline RL from preference data (no reward model needed)

## Training on Multiple GPUs

```python
from fla.training import DistributedTrainer

trainer = DistributedTrainer(model, dataloader, config=config)
trainer.train()  # Automatically uses all available GPUs
```

## Evaluation

FLA supports evaluation on simulation environments:

```python
from fla.evaluation import run_evaluation

# Evaluate on ALOHA simulation
results = run_evaluation(
    model=model,
    task="gym_aloha/AlohaTransferCube-v0",
    num_episodes=50,
)
print(f"Success Rate: {results['success_rate']:.1%}")
```

See `examples/` for complete evaluation scripts.

## Project Structure

```
fla/
├── fla/
│   ├── models/          # Pi0.5 model implementation
│   │   ├── base.py      # Base classes
│   │   └── pi05.py      # Pi0.5 model
│   ├── data/            # Data loading
│   │   ├── lerobot_loader.py
│   │   └── transforms.py
│   ├── training/        # Training infrastructure
│   │   ├── trainer.py              # Main trainer
│   │   ├── optimizer.py            # Optimizers and schedules
│   │   ├── checkpoints.py          # Checkpoint management
│   │   ├── knowledge_insulation.py # Knowledge Insulation (P1)
│   │   ├── lora.py                 # LoRA support (P2)
│   │   └── reinflow.py             # ReinFlow RL training (P2)
│   ├── evaluation/      # Evaluation
│   │   ├── evaluator.py
│   │   └── aloha.py
│   └── shared/          # Utilities
├── tests/               # Test suite
│   ├── test_knowledge_insulation.py
│   ├── test_lora.py
│   └── test_reinflow.py
├── configs/             # Training configs
└── scripts/             # CLI scripts
```

## Running Tests

```bash
# Run all tests (95 pass, 14 skipped for expensive operations)
pytest tests/ -v

# Run specific test module
pytest tests/test_lora.py -v
pytest tests/test_reinflow.py -v
pytest tests/test_knowledge_insulation.py -v

# Run with coverage
pytest tests/ --cov=fla
```

Tests are designed to run without the full openpi dependency. Tests requiring openpi or expensive model initialization are automatically skipped.

## License

Apache 2.0

## Citation

If you use FLA in your research, please cite:

```bibtex
@software{fla2025,
  title={FLA: Fine-tune Vision-Language-Action Models},
  author={Arizona},
  year={2025},
  url={https://github.com/Physical-Intelligence/openpi}
}
```

## Acknowledgments

FLA builds on:
- [OpenPI](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence
- [Pi0](https://www.physicalintelligence.company/blog/pi0) by Physical Intelligence
- [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace
- [Flax](https://github.com/google/flax) by Google
- [JAX](https://github.com/google/jax) by Google
