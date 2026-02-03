"""Optimizer configuration for VLA training.

Provides learning rate schedules and optimizer creation.
"""

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import optax


@dataclasses.dataclass
class CosineSchedule:
    """Cosine decay learning rate schedule with warmup.

    Attributes:
        peak_lr: Peak learning rate after warmup.
        warmup_steps: Number of warmup steps.
        decay_steps: Total steps for cosine decay.
        end_lr: Final learning rate.
    """

    peak_lr: float = 2.5e-5
    warmup_steps: int = 1000
    decay_steps: int = 30000
    end_lr: float = 2.5e-6

    def __call__(self, step: int) -> float:
        """Compute learning rate at given step."""
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps

        progress = (step - self.warmup_steps) / max(
            1, self.decay_steps - self.warmup_steps
        )
        progress = min(1.0, progress)
        return self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (
            1 + jnp.cos(jnp.pi * progress)
        )

    def to_optax(self) -> optax.Schedule:
        """Convert to optax schedule."""
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(0, self.peak_lr, self.warmup_steps),
                optax.cosine_decay_schedule(
                    self.peak_lr,
                    self.decay_steps - self.warmup_steps,
                    alpha=self.end_lr / self.peak_lr,
                ),
            ],
            boundaries=[self.warmup_steps],
        )


@dataclasses.dataclass
class RsqrtSchedule:
    """Inverse square root learning rate schedule.

    lr = peak_lr * sqrt(warmup_steps) / sqrt(step)
    """

    peak_lr: float = 1e-4
    warmup_steps: int = 1000

    def __call__(self, step: int) -> float:
        """Compute learning rate at given step."""
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        return self.peak_lr * jnp.sqrt(self.warmup_steps / step)

    def to_optax(self) -> optax.Schedule:
        """Convert to optax schedule."""
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(0, self.peak_lr, self.warmup_steps),
                lambda count: self.peak_lr
                * jnp.sqrt(self.warmup_steps / (count + self.warmup_steps)),
            ],
            boundaries=[self.warmup_steps],
        )


def create_optimizer(
    schedule: CosineSchedule | RsqrtSchedule | None = None,
    *,
    learning_rate: float | None = None,
    weight_decay: float = 0.0,
    gradient_clip: float = 1.0,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with gradient clipping.

    Args:
        schedule: Learning rate schedule.
        learning_rate: Constant learning rate (if schedule not provided).
        weight_decay: Weight decay coefficient.
        gradient_clip: Gradient clipping threshold.
        b1: Adam beta1.
        b2: Adam beta2.
        eps: Adam epsilon.

    Returns:
        Optax optimizer.
    """
    if schedule is not None:
        lr = schedule.to_optax()
    elif learning_rate is not None:
        lr = learning_rate
    else:
        lr = CosineSchedule().to_optax()

    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adamw(
            learning_rate=lr,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
        ),
    )

    return optimizer


def create_frozen_backbone_optimizer(
    freeze_filter,
    *,
    schedule: CosineSchedule | RsqrtSchedule | None = None,
    learning_rate: float | None = None,
    weight_decay: float = 0.0,
    gradient_clip: float = 1.0,
) -> optax.GradientTransformation:
    """Create optimizer that only updates non-frozen parameters.

    This is used for frozen backbone fine-tuning where the VLM
    parameters are frozen and only the action expert is trained.

    Args:
        freeze_filter: NNX filter for frozen parameters.
        schedule: Learning rate schedule.
        learning_rate: Constant learning rate.
        weight_decay: Weight decay.
        gradient_clip: Gradient clipping.

    Returns:
        Optax optimizer that masks frozen parameters.
    """
    base_optimizer = create_optimizer(
        schedule=schedule,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
    )

    # Note: The actual masking is handled by JAX's stop_gradient
    # in the model's compute_loss method when freeze_vision_backbone=True.
    # The optimizer still sees the full parameter tree, but frozen params
    # have zero gradients due to stop_gradient.

    return base_optimizer
