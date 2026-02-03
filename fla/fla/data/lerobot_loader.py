"""LeRobot Dataset Loading.

This module provides data loading for LeRobot format datasets,
which is the standard format for robotics datasets on HuggingFace.

LeRobot Format:
- Parquet files for metadata and state/actions
- MP4 videos for camera images
- Standardized column names (observation.*, action.*)

Supported datasets:
- ALOHA: Bimanual manipulation (sim + real)
- LIBERO: 130+ manipulation tasks
- DROID: 92k episodes across diverse environments
- Open X-Embodiment: 60+ datasets, 22 robot types
"""

import dataclasses
import logging
from collections.abc import Iterator, Sequence
from typing import Any, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.data

from fla.shared.normalize import NormStats

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DataConfig:
    """Configuration for LeRobot dataset loading.

    Attributes:
        repo_id: HuggingFace repo ID (e.g., "lerobot/aloha_sim_transfer_cube_human")
        action_horizon: Number of future actions to load (for action chunking)
        batch_size: Training batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        prompt: Default language prompt for the task
        action_keys: Keys for action data (default: ["action"])
        state_keys: Keys for state data
        image_keys: Keys for image data
        norm_stats: Pre-computed normalization statistics
    """

    repo_id: str
    action_horizon: int = 50
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    prompt: str = "Complete the manipulation task"
    action_keys: tuple[str, ...] = ("action",)
    state_keys: tuple[str, ...] = ("observation.state",)
    image_keys: tuple[str, ...] = (
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    norm_stats: dict[str, NormStats] | None = None


class Dataset(Protocol):
    """Protocol for indexable datasets."""

    def __getitem__(self, index: int) -> dict[str, Any]: ...
    def __len__(self) -> int: ...


class LeRobotDataset:
    """Wrapper around LeRobot dataset for FLA.

    Handles:
    - Loading from HuggingFace
    - Action chunking (loading K future actions)
    - Image resizing and normalization
    - Prompt injection
    """

    def __init__(
        self,
        repo_id: str,
        action_horizon: int = 50,
        prompt: str | None = None,
    ):
        """Initialize LeRobot dataset.

        Args:
            repo_id: HuggingFace repo ID.
            action_horizon: Number of future actions to load.
            prompt: Default language prompt.
        """
        try:
            from lerobot.common.datasets.lerobot_dataset import (
                LeRobotDataset as _LeRobotDataset,
                LeRobotDatasetMetadata,
            )
        except ImportError:
            raise ImportError(
                "lerobot package not found. Install with: pip install lerobot"
            )

        self.repo_id = repo_id
        self.action_horizon = action_horizon
        self.prompt = prompt

        # Load dataset metadata
        self._meta = LeRobotDatasetMetadata(repo_id)
        fps = self._meta.fps

        # Load dataset with action chunking
        # delta_timestamps specifies which future timesteps to load
        delta_timestamps = {
            "action": [t / fps for t in range(action_horizon)]
        }

        self._dataset = _LeRobotDataset(
            repo_id,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

        logger.info(
            f"Loaded {repo_id}: {len(self._dataset)} samples, "
            f"{self._meta.total_episodes} episodes, {fps} fps"
        )

    @property
    def tasks(self) -> dict[int, str]:
        """Get task descriptions from dataset."""
        return self._meta.tasks or {}

    @property
    def fps(self) -> float:
        """Get dataset frames per second."""
        return self._meta.fps

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample.

        Returns dictionary with:
        - image: dict of camera images
        - state: robot proprioceptive state
        - actions: future actions [action_horizon, action_dim]
        - prompt: language instruction
        """
        sample = self._dataset[index]

        # Extract images
        images = {}
        for key, value in sample.items():
            if "image" in key.lower() or "rgb" in key.lower():
                # Convert to numpy, handle various formats
                if isinstance(value, torch.Tensor):
                    images[key] = value.numpy()
                else:
                    images[key] = np.asarray(value)

        # Extract state
        state = None
        for key in ["observation.state", "state"]:
            if key in sample:
                state = np.asarray(sample[key])
                break

        if state is None:
            # Fallback: use zeros if no state
            state = np.zeros(14, dtype=np.float32)

        # Extract actions
        actions = np.asarray(sample["action"])

        # Add prompt
        prompt = self.prompt
        if prompt is None and "task_index" in sample:
            task_idx = int(sample["task_index"])
            prompt = self.tasks.get(task_idx, "Complete the manipulation task")

        return {
            "image": images,
            "state": state,
            "actions": actions,
            "prompt": prompt or "Complete the manipulation task",
        }


class LeRobotDataLoader:
    """DataLoader for LeRobot datasets.

    Handles batching, shuffling, and conversion to JAX arrays.
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        transforms: Sequence[Any] | None = None,
        sharding: jax.sharding.Sharding | None = None,
    ):
        """Initialize data loader.

        Args:
            config: Data configuration.
            transforms: Optional list of transforms to apply.
            sharding: JAX sharding for distributed training.
        """
        self.config = config

        # Create dataset
        self._dataset = LeRobotDataset(
            config.repo_id,
            action_horizon=config.action_horizon,
            prompt=config.prompt,
        )

        # Apply transforms
        if transforms:
            from fla.data.transforms import TransformedDataset
            self._dataset = TransformedDataset(self._dataset, transforms)

        # Setup sharding
        if sharding is None:
            # Default: replicate across devices
            mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
            sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("batch")
            )
        self._sharding = sharding

        # Create PyTorch DataLoader for efficient batching
        self._torch_loader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            collate_fn=_collate_fn,
            drop_last=True,
            persistent_workers=config.num_workers > 0,
        )

    @property
    def dataset(self) -> LeRobotDataset:
        """Get underlying dataset."""
        return self._dataset

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self._torch_loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over batches.

        Yields dictionaries with JAX arrays.
        """
        for batch in self._torch_loader:
            # Convert to JAX arrays with sharding
            yield jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(
                    self._sharding, np.asarray(x)
                ),
                batch,
            )


def _collate_fn(samples: list[dict]) -> dict[str, Any]:
    """Collate samples into a batch."""
    batch = {}
    for key in samples[0].keys():
        if key == "image":
            # Handle nested image dict
            batch["image"] = {}
            for img_key in samples[0]["image"].keys():
                batch["image"][img_key] = np.stack(
                    [s["image"][img_key] for s in samples], axis=0
                )
        elif key == "prompt":
            # Keep prompts as list
            batch["prompt"] = [s["prompt"] for s in samples]
        else:
            batch[key] = np.stack([s[key] for s in samples], axis=0)
    return batch


def create_dataloader(
    repo_id: str,
    *,
    action_horizon: int = 50,
    batch_size: int = 32,
    prompt: str | None = None,
    transforms: Sequence[Any] | None = None,
    **kwargs,
) -> LeRobotDataLoader:
    """Convenience function to create a LeRobot data loader.

    Args:
        repo_id: HuggingFace repo ID.
        action_horizon: Number of future actions to predict.
        batch_size: Training batch size.
        prompt: Default language prompt.
        transforms: Optional transforms to apply.
        **kwargs: Additional DataConfig arguments.

    Returns:
        Configured LeRobotDataLoader.
    """
    config = DataConfig(
        repo_id=repo_id,
        action_horizon=action_horizon,
        batch_size=batch_size,
        prompt=prompt or "Complete the manipulation task",
        **kwargs,
    )
    return LeRobotDataLoader(config, transforms=transforms)


class ConcatDataLoader:
    """Concatenate multiple datasets for multi-task training."""

    def __init__(
        self,
        loaders: Sequence[LeRobotDataLoader],
        *,
        sampling_weights: Sequence[float] | None = None,
    ):
        """Initialize concatenated loader.

        Args:
            loaders: List of data loaders.
            sampling_weights: Optional sampling weights (uniform if None).
        """
        self._loaders = list(loaders)
        if sampling_weights is None:
            # Weight by dataset size
            sizes = [len(l.dataset) for l in loaders]
            total = sum(sizes)
            sampling_weights = [s / total for s in sizes]
        self._weights = np.array(sampling_weights)
        self._weights /= self._weights.sum()

    def __len__(self) -> int:
        return sum(len(l) for l in self._loaders)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate, sampling from loaders according to weights."""
        iterators = [iter(l) for l in self._loaders]
        exhausted = [False] * len(self._loaders)

        while not all(exhausted):
            # Sample a loader
            idx = np.random.choice(len(self._loaders), p=self._weights)
            if exhausted[idx]:
                continue

            try:
                yield next(iterators[idx])
            except StopIteration:
                exhausted[idx] = True
                # Renormalize weights
                remaining = [i for i, e in enumerate(exhausted) if not e]
                if remaining:
                    new_weights = np.zeros_like(self._weights)
                    for i in remaining:
                        new_weights[i] = self._weights[i]
                    new_weights /= new_weights.sum()
                    self._weights = new_weights
