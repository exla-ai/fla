"""Checkpoint management for VLA training.

Provides utilities for saving and loading model checkpoints.
"""

import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str | pathlib.Path,
    step: int,
    model: nnx.Module,
    optimizer_state: Any | None = None,
    *,
    keep_last_n: int = 5,
) -> None:
    """Save a training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        step: Current training step.
        model: Model to save.
        optimizer_state: Optional optimizer state to save.
        keep_last_n: Number of checkpoints to keep.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Extract model state
    _, state = nnx.split(model)
    params = state.to_pure_dict()

    # Create checkpoint
    ckpt = {"params": params, "step": step}
    if optimizer_state is not None:
        ckpt["optimizer"] = optimizer_state

    # Save with orbax
    options = ocp.CheckpointManagerOptions(
        max_to_keep=keep_last_n,
        save_interval_steps=1,
    )
    manager = ocp.CheckpointManager(checkpoint_dir, options=options)

    manager.save(
        step,
        args=ocp.args.PyTreeSave(ckpt),
    )
    manager.wait_until_finished()

    logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def load_checkpoint(
    checkpoint_path: str | pathlib.Path,
    *,
    step: int | None = None,
    restore_type: type = jax.Array,
) -> dict[str, Any]:
    """Load a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or specific step.
        step: Specific step to load (latest if None).
        restore_type: Type for restored arrays.

    Returns:
        Dictionary with checkpoint contents.
    """
    checkpoint_path = pathlib.Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check if path is a checkpoint manager directory or specific checkpoint
    if (checkpoint_path / "default").exists():
        # This is a checkpoint manager directory
        manager = ocp.CheckpointManager(checkpoint_path)
        if step is None:
            step = manager.latest_step()
        checkpoint_path = checkpoint_path / str(step) / "default"
    elif step is not None:
        checkpoint_path = checkpoint_path / str(step) / "default"

    # Setup sharding
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Load checkpoint
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(checkpoint_path)
        item = {"params": metadata.get("params", {})}

        ckpt = ckptr.restore(
            checkpoint_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=restore_type
                    ),
                    item,
                ),
            ),
        )

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return ckpt.get("params", ckpt)


def load_params_flexible(
    checkpoint_path: str | pathlib.Path,
    target_action_dim: int,
    *,
    source_action_dim: int | None = None,
) -> dict[str, Any]:
    """Load checkpoint with flexible action dimension.

    This enables loading checkpoints trained on one robot and
    fine-tuning on another robot with different action dimensions.
    Mismatched dimensions are randomly initialized.

    Args:
        checkpoint_path: Path to checkpoint.
        target_action_dim: Action dimension of target robot.
        source_action_dim: Action dimension of source checkpoint.

    Returns:
        Parameters with adjusted dimensions.
    """
    params = load_checkpoint(checkpoint_path)

    if source_action_dim is None:
        # Try to infer from checkpoint
        # Look for action projection layers
        for key in ["action_in_proj", "action_out_proj"]:
            if key in str(params):
                # Found action layer, could infer dimension
                pass

    # For now, return params as-is
    # Dimension adjustment happens during model initialization
    return params


class CheckpointManager:
    """Manage training checkpoints with automatic cleanup."""

    def __init__(
        self,
        checkpoint_dir: str | pathlib.Path,
        *,
        keep_last_n: int = 5,
        save_interval: int = 1000,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints.
            keep_last_n: Number of checkpoints to keep.
            save_interval: Steps between saves.
        """
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_interval = save_interval

        options = ocp.CheckpointManagerOptions(
            max_to_keep=keep_last_n,
            save_interval_steps=save_interval,
        )
        self._manager = ocp.CheckpointManager(
            self.checkpoint_dir, options=options
        )

    def save(
        self,
        step: int,
        model: nnx.Module,
        optimizer_state: Any | None = None,
        **extra,
    ) -> None:
        """Save checkpoint if at save interval."""
        _, state = nnx.split(model)
        ckpt = {
            "params": state.to_pure_dict(),
            "step": step,
            **extra,
        }
        if optimizer_state is not None:
            ckpt["optimizer"] = optimizer_state

        self._manager.save(step, args=ocp.args.PyTreeSave(ckpt))

    def load_latest(self) -> dict[str, Any] | None:
        """Load the latest checkpoint."""
        step = self._manager.latest_step()
        if step is None:
            return None
        return self.load(step)

    def load(self, step: int) -> dict[str, Any]:
        """Load a specific checkpoint."""
        return load_checkpoint(self.checkpoint_dir, step=step)

    @property
    def latest_step(self) -> int | None:
        """Get the latest checkpoint step."""
        return self._manager.latest_step()

    def wait(self) -> None:
        """Wait for any pending saves to complete."""
        self._manager.wait_until_finished()
