"""Base model classes for FLA.

Defines the core abstractions for VLA models including:
- Observation: Structured input format (images, state, prompt)
- Actions: Output action format
- BaseModelConfig: Configuration interface
- BaseModel: Model interface
"""

import abc
import dataclasses
from typing import Any, Generic, TypeVar

import flax.nnx as nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from fla.shared import array_typing as at

# Type variable for array types
ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray)

# Standard image resolution for VLA models
IMAGE_RESOLUTION = (224, 224)

# Standard image keys
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


@struct.dataclass
class Observation(Generic[ArrayT]):
    """Structured observation input for VLA models.

    Attributes:
        images: Dictionary of camera images, shape [batch, height, width, 3], values in [-1, 1]
        image_masks: Dictionary of masks indicating valid images
        state: Robot proprioceptive state, shape [batch, state_dim]
        tokenized_prompt: Optional tokenized language prompt
        tokenized_prompt_mask: Mask for tokenized prompt
    """

    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        """Create Observation from a dictionary.

        Expected keys:
            - image: dict of images
            - image_mask: dict of masks
            - state: robot state
            - tokenized_prompt (optional): tokenized text
            - tokenized_prompt_mask (optional): mask for tokens
        """
        # Convert uint8 images to float32 in [-1, 1]
        images = {}
        for key, img in data["image"].items():
            if img.dtype == np.uint8:
                images[key] = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            else:
                images[key] = img

        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = {
            "image": self.images,
            "image_mask": self.image_masks,
            "state": self.state,
        }
        if self.tokenized_prompt is not None:
            result["tokenized_prompt"] = self.tokenized_prompt
            result["tokenized_prompt_mask"] = self.tokenized_prompt_mask
        return result


# Actions type alias: [batch, action_horizon, action_dim]
Actions = at.Float[ArrayT, "*b ah ad"]


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Base configuration for VLA models.

    Attributes:
        action_dim: Dimension of the action space
        action_horizon: Number of future actions to predict (chunking)
        max_token_len: Maximum length of tokenized prompt
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model with randomly initialized parameters."""

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Return input specifications as jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """Create fake observation for testing."""
        obs_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)

    def fake_actions(self, batch_size: int = 1) -> Actions:
        """Create fake actions for testing."""
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


class BaseModel(nnx.Module, abc.ABC):
    """Base class for VLA models.

    All VLA models should inherit from this class and implement:
    - compute_loss: Compute training loss
    - sample_actions: Generate actions from observation
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute training loss.

        Args:
            rng: Random key for dropout/noise.
            observation: Model inputs.
            actions: Ground truth actions.
            train: Whether in training mode (enables augmentation).

        Returns:
            Loss tensor of shape [batch, action_horizon].
        """

    @abc.abstractmethod
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        **kwargs,
    ) -> Actions:
        """Sample actions from the model.

        Args:
            rng: Random key for sampling.
            observation: Model inputs.
            **kwargs: Additional sampling parameters.

        Returns:
            Sampled actions of shape [batch, action_horizon, action_dim].
        """
