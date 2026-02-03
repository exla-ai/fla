"""Main training loop for VLA fine-tuning.

Provides a clean, research-friendly training interface.
"""

import dataclasses
import logging
import time
from collections.abc import Iterator
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from fla.models.base import BaseModel, Observation, Actions
from fla.data.lerobot_loader import LeRobotDataLoader
from fla.training.optimizer import CosineSchedule, create_optimizer
from fla.training.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        max_steps: Maximum training steps.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        warmup_steps: LR warmup steps.
        weight_decay: Weight decay coefficient.
        gradient_clip: Gradient clipping threshold.
        log_interval: Steps between logging.
        save_interval: Steps between checkpoints.
        eval_interval: Steps between evaluations.
        checkpoint_dir: Directory for checkpoints.
        seed: Random seed.
    """

    max_steps: int = 30000
    batch_size: int = 32
    learning_rate: float = 2.5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 5000
    eval_interval: int = 5000
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42


class Trainer:
    """Main trainer for VLA models.

    Example:
        >>> model = Pi05Model.from_pretrained("pi0.5-base")
        >>> data = LeRobotDataLoader("lerobot/aloha_sim_transfer_cube_human")
        >>> trainer = Trainer(model, data, config=TrainConfig(max_steps=10000))
        >>> trainer.train()
    """

    def __init__(
        self,
        model: BaseModel,
        dataloader: LeRobotDataLoader,
        *,
        config: TrainConfig | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ):
        """Initialize trainer.

        Args:
            model: VLA model to train.
            dataloader: Training data loader.
            config: Training configuration.
            optimizer: Custom optimizer (uses default if None).
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config or TrainConfig()

        # Setup optimizer
        if optimizer is None:
            schedule = CosineSchedule(
                peak_lr=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.max_steps,
                end_lr=self.config.learning_rate / 10,
            )
            optimizer = create_optimizer(
                schedule,
                weight_decay=self.config.weight_decay,
                gradient_clip=self.config.gradient_clip,
            )

        # Initialize optimizer state
        _, state = nnx.split(model)
        self.opt_state = optimizer.init(state)
        self.optimizer = optimizer

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir,
            keep_last_n=3,
            save_interval=self.config.save_interval,
        )

        # Training state
        self.step = 0
        self.rng = jax.random.PRNGKey(self.config.seed)

        # Metrics
        self._metrics = {
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
        }

    def train(
        self,
        *,
        max_steps: int | None = None,
        eval_fn: Any | None = None,
    ) -> dict[str, list]:
        """Run training loop.

        Args:
            max_steps: Override max steps from config.
            eval_fn: Optional evaluation function called at eval_interval.

        Returns:
            Dictionary of training metrics.
        """
        max_steps = max_steps or self.config.max_steps
        logger.info(f"Starting training for {max_steps} steps")

        # Create JIT-compiled train step
        train_step_fn = self._create_train_step()

        # Training loop
        data_iter = iter(self.dataloader)
        start_time = time.time()

        while self.step < max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Train step
            self.rng, step_rng = jax.random.split(self.rng)
            loss, grad_norm = train_step_fn(step_rng, batch)

            # Record metrics
            self._metrics["loss"].append(float(loss))
            self._metrics["grad_norm"].append(float(grad_norm))

            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (self.step + 1) / elapsed
                avg_loss = sum(self._metrics["loss"][-100:]) / min(
                    100, len(self._metrics["loss"])
                )
                logger.info(
                    f"Step {self.step}/{max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Grad: {grad_norm:.4f} | "
                    f"Speed: {steps_per_sec:.2f} steps/s"
                )

            # Checkpointing
            if self.step % self.config.save_interval == 0 and self.step > 0:
                self.checkpoint_manager.save(
                    self.step, self.model, self.opt_state
                )

            # Evaluation
            if (
                eval_fn is not None
                and self.step % self.config.eval_interval == 0
                and self.step > 0
            ):
                eval_metrics = eval_fn(self.model, self.step)
                logger.info(f"Eval at step {self.step}: {eval_metrics}")

            self.step += 1

        # Final save
        self.checkpoint_manager.save(self.step, self.model, self.opt_state)
        self.checkpoint_manager.wait()

        total_time = time.time() - start_time
        logger.info(
            f"Training complete: {self.step} steps in {total_time:.1f}s "
            f"({self.step / total_time:.2f} steps/s)"
        )

        return self._metrics

    def _create_train_step(self):
        """Create JIT-compiled training step function."""

        def loss_fn(state, rng, observation, actions):
            """Compute loss with model state."""
            graphdef, _ = nnx.split(self.model)
            model = nnx.merge(graphdef, state)
            loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(loss), loss

        @jax.jit
        def train_step(state, opt_state, rng, batch):
            """Single training step."""
            # Prepare observation
            observation = Observation.from_dict(batch)
            actions = batch["actions"]

            # Compute gradients
            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state, rng, observation, actions
            )

            # Compute gradient norm
            grad_norm = optax.global_norm(grads)

            # Update parameters
            updates, new_opt_state = self.optimizer.update(
                grads, opt_state, state
            )
            new_state = optax.apply_updates(state, updates)

            return new_state, new_opt_state, loss, grad_norm

        def wrapped_train_step(rng, batch):
            """Wrapper that updates trainer state."""
            graphdef, state = nnx.split(self.model)
            state, self.opt_state, loss, grad_norm = train_step(
                state, self.opt_state, rng, batch
            )
            self.model = nnx.merge(graphdef, state)
            return loss, grad_norm

        return wrapped_train_step

    def save(self, path: str | None = None) -> None:
        """Save current model checkpoint."""
        path = path or self.config.checkpoint_dir
        self.checkpoint_manager.save(self.step, self.model, self.opt_state)
        self.checkpoint_manager.wait()
        logger.info(f"Saved checkpoint at step {self.step}")

    def load(self, path: str, step: int | None = None) -> None:
        """Load model from checkpoint."""
        from fla.training.checkpoints import load_checkpoint

        ckpt = load_checkpoint(path, step=step)
        graphdef, state = nnx.split(self.model)
        state.replace_by_pure_dict(ckpt["params"])
        self.model = nnx.merge(graphdef, state)

        if "optimizer" in ckpt:
            self.opt_state = ckpt["optimizer"]
        if "step" in ckpt:
            self.step = ckpt["step"]

        logger.info(f"Loaded checkpoint from {path} at step {self.step}")


class DistributedTrainer(Trainer):
    """Trainer with multi-GPU support via data parallelism."""

    def __init__(
        self,
        model: BaseModel,
        dataloader: LeRobotDataLoader,
        *,
        config: TrainConfig | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ):
        super().__init__(model, dataloader, config=config, optimizer=optimizer)

        # Setup mesh for data parallelism
        self.mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
        self.data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec("batch")
        )
        self.replicated_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec()
        )

        # Log device info
        logger.info(
            f"Distributed training on {len(jax.devices())} devices: "
            f"{[str(d) for d in jax.devices()]}"
        )

    def _create_train_step(self):
        """Create sharded training step."""

        def loss_fn(state, rng, observation, actions):
            graphdef, _ = nnx.split(self.model)
            model = nnx.merge(graphdef, state)
            loss = model.compute_loss(rng, observation, actions, train=True)
            # Average loss across batch
            return jnp.mean(loss), loss

        @jax.jit
        def train_step(state, opt_state, rng, batch):
            observation = Observation.from_dict(batch)
            actions = batch["actions"]

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state, rng, observation, actions
            )

            # Average gradients across devices
            grads = jax.lax.pmean(grads, axis_name="batch")
            loss = jax.lax.pmean(loss, axis_name="batch")

            grad_norm = optax.global_norm(grads)

            updates, new_opt_state = self.optimizer.update(
                grads, opt_state, state
            )
            new_state = optax.apply_updates(state, updates)

            return new_state, new_opt_state, loss, grad_norm

        # Shard train step
        train_step = jax.pmap(train_step, axis_name="batch")

        def wrapped_train_step(rng, batch):
            graphdef, state = nnx.split(self.model)

            # Replicate state and optimizer across devices
            state = jax.device_put(state, self.replicated_sharding)
            opt_state = jax.device_put(self.opt_state, self.replicated_sharding)

            # Shard batch across devices
            batch = jax.device_put(batch, self.data_sharding)

            # Replicate rng
            rngs = jax.random.split(rng, len(jax.devices()))

            state, self.opt_state, loss, grad_norm = train_step(
                state, opt_state, rngs, batch
            )

            # Take first device's state (they're all the same after pmean)
            state = jax.tree_map(lambda x: x[0], state)
            self.model = nnx.merge(graphdef, state)

            return float(loss[0]), float(grad_norm[0])

        return wrapped_train_step
