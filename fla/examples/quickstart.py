#!/usr/bin/env python3
"""FLA Quickstart Example.

This example shows how to fine-tune Pi0.5 on ALOHA simulation.

Prerequisites:
    pip install fla[eval]  # Installs gym-aloha for evaluation

Usage:
    python examples/quickstart.py
"""

import logging
import jax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"JAX devices: {jax.devices()}")

    # ==========================================================================
    # Step 1: Load Dataset
    # ==========================================================================
    logger.info("Step 1: Loading ALOHA dataset...")

    from fla.data import create_dataloader
    from fla.data.transforms import create_transform_pipeline

    # Create transforms for data preprocessing
    transforms = create_transform_pipeline(
        target_action_dim=14,  # ALOHA bimanual
        max_token_len=200,
    )

    # Load ALOHA transfer cube dataset
    dataloader = create_dataloader(
        "lerobot/aloha_sim_transfer_cube_human",
        batch_size=4,  # Small batch for demo
        action_horizon=50,
        prompt="Pick up the cube and transfer it",
        transforms=[transforms],
    )

    logger.info(f"Dataset loaded: {len(dataloader.dataset)} samples")

    # Get a sample batch
    batch = next(iter(dataloader))
    logger.info(f"Batch shapes:")
    for key in ["state", "actions"]:
        if key in batch:
            logger.info(f"  {key}: {batch[key].shape}")

    # ==========================================================================
    # Step 2: Create Model
    # ==========================================================================
    logger.info("\nStep 2: Creating Pi0.5 model...")

    from fla.models import Pi05Config

    config = Pi05Config(
        action_dim=14,
        action_horizon=50,
        freeze_vision_backbone=True,  # Memory efficient
    )

    logger.info(f"Model config:")
    logger.info(f"  Action dim: {config.action_dim}")
    logger.info(f"  Action horizon: {config.action_horizon}")
    logger.info(f"  Freeze backbone: {config.freeze_vision_backbone}")

    # Note: Creating the actual model requires openpi in path
    # model = config.create(jax.random.key(0))

    # ==========================================================================
    # Step 3: Training Setup
    # ==========================================================================
    logger.info("\nStep 3: Setting up training...")

    from fla.training import TrainConfig
    from fla.training.optimizer import CosineSchedule, create_optimizer

    train_config = TrainConfig(
        max_steps=10000,
        batch_size=32,
        learning_rate=2.5e-5,
        warmup_steps=1000,
        save_interval=5000,
        checkpoint_dir="./checkpoints/quickstart",
    )

    schedule = CosineSchedule(
        peak_lr=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        decay_steps=train_config.max_steps,
    )

    optimizer = create_optimizer(schedule, gradient_clip=1.0)
    logger.info(f"Optimizer created with cosine schedule")
    logger.info(f"  Peak LR: {train_config.learning_rate}")
    logger.info(f"  Warmup: {train_config.warmup_steps} steps")

    # ==========================================================================
    # Step 4: Evaluation Setup
    # ==========================================================================
    logger.info("\nStep 4: Setting up evaluation...")

    from fla.evaluation.metrics import success_rate, episode_return

    # Demo metrics computation
    demo_rewards = [2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    sr = success_rate(demo_rewards, threshold=1.9)
    mean, std = episode_return(demo_rewards)

    logger.info(f"Demo metrics (10 episodes):")
    logger.info(f"  Success rate: {sr:.1%}")
    logger.info(f"  Mean reward: {mean:.2f} +/- {std:.2f}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Quickstart Complete!")
    print("=" * 60)
    print("""
Next steps:

1. Full training (requires GPU):
   fla-train --dataset lerobot/aloha_sim_transfer_cube_human --exp-name my_exp

2. Evaluation (requires gym-aloha):
   fla-eval --checkpoint ./checkpoints/my_exp/30000 --task transfer_cube

3. Multi-robot training:
   fla-train --config configs/cross_embodiment.yaml --exp-name cross_v1

Documentation: fla/README.md
""")


if __name__ == "__main__":
    main()
