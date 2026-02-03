#!/usr/bin/env python3
"""FLA Training Script.

Fine-tune Pi0.5 model on LeRobot datasets.

Usage:
    # Train on ALOHA simulation
    fla-train --dataset lerobot/aloha_sim_transfer_cube_human --exp-name cube_v1

    # Train on LIBERO
    fla-train --dataset lerobot/libero_10 --exp-name libero_v1

    # Resume training
    fla-train --resume ./checkpoints/cube_v1/latest
"""

import dataclasses
import logging
import sys

import tyro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Training arguments."""

    # Data
    dataset: str = "lerobot/aloha_sim_transfer_cube_human"
    """LeRobot dataset repo ID."""

    prompt: str = "Complete the manipulation task"
    """Default language prompt."""

    # Model
    checkpoint: str | None = None
    """Path to pretrained checkpoint. If None, trains from scratch."""

    freeze_backbone: bool = True
    """Freeze VLM backbone (reduces memory from 38GB to 15GB)."""

    action_dim: int = 14
    """Action dimension for the robot."""

    # Training
    max_steps: int = 30000
    """Maximum training steps."""

    batch_size: int = 32
    """Training batch size."""

    learning_rate: float = 2.5e-5
    """Peak learning rate."""

    warmup_steps: int = 1000
    """Learning rate warmup steps."""

    gradient_clip: float = 1.0
    """Gradient clipping threshold."""

    # Logging
    exp_name: str = "fla_experiment"
    """Experiment name."""

    checkpoint_dir: str = "./checkpoints"
    """Checkpoint directory."""

    log_interval: int = 100
    """Steps between logging."""

    save_interval: int = 5000
    """Steps between checkpoints."""

    # System
    seed: int = 42
    """Random seed."""

    num_workers: int = 4
    """Data loading workers."""

    resume: str | None = None
    """Resume from checkpoint path."""


def main(args: Args | None = None) -> None:
    """Main training entry point."""
    if args is None:
        args = tyro.cli(Args)

    import jax

    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"Training on {len(jax.devices())} GPUs")

    # Import FLA modules
    from fla.models import Pi05Config, Pi05Model
    from fla.data import LeRobotDataLoader, DataConfig
    from fla.data.transforms import create_transform_pipeline
    from fla.training import Trainer, TrainConfig

    # Create model config
    model_config = Pi05Config(
        action_dim=args.action_dim,
        freeze_vision_backbone=args.freeze_backbone,
    )

    # Load or create model
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model = Pi05Model.from_pretrained(
            args.checkpoint,
            freeze_vision_backbone=args.freeze_backbone,
            action_dim=args.action_dim,
        )
    else:
        logger.info("Creating new model (random initialization)")
        model = model_config.create(jax.random.key(args.seed))

    # Create data loader
    logger.info(f"Loading dataset: {args.dataset}")
    data_config = DataConfig(
        repo_id=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prompt=args.prompt,
    )

    transforms = create_transform_pipeline(
        target_action_dim=args.action_dim,
        max_token_len=model_config.max_token_len,
    )

    dataloader = LeRobotDataLoader(data_config, transforms=[transforms])
    logger.info(f"Dataset loaded: {len(dataloader.dataset)} samples")

    # Create trainer
    train_config = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=f"{args.checkpoint_dir}/{args.exp_name}",
        seed=args.seed,
    )

    if len(jax.devices()) > 1:
        from fla.training.trainer import DistributedTrainer
        trainer = DistributedTrainer(model, dataloader, config=train_config)
    else:
        trainer = Trainer(model, dataloader, config=train_config)

    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load(args.resume)

    # Train
    logger.info(f"Starting training: {args.exp_name}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Freeze backbone: {args.freeze_backbone}")

    metrics = trainer.train()

    logger.info("Training complete!")
    logger.info(f"Final loss: {metrics['loss'][-1]:.4f}")
    logger.info(f"Checkpoints saved to: {train_config.checkpoint_dir}")


if __name__ == "__main__":
    main()
