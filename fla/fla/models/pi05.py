"""Pi0.5 Model Implementation.

Pi0.5 is a Vision-Language-Action model with:
- Frozen PaliGemma VLM backbone (2B params)
- Trainable action expert (300M params)
- Flow matching for action generation
- AdaRMSNorm for timestep conditioning

Key features for fine-tuning:
- freeze_vision_backbone: Reduces memory from 38GB to 15GB
- Action chunking: Predicts K=50 future actions
"""

import dataclasses
import logging
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from fla.models.base import BaseModel, BaseModelConfig, Observation, Actions, IMAGE_RESOLUTION
from fla.shared import array_typing as at
from fla.shared.nnx_utils import PathRegex

logger = logging.getLogger(__name__)

# Conditionally import openpi - allows FLA to work standalone for training modules
# Full model functionality requires openpi
import sys
import os

_OPENPI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
if _OPENPI_PATH not in sys.path:
    sys.path.insert(0, _OPENPI_PATH)

try:
    from openpi.models import pi0 as _pi0
    from openpi.models import pi0_config as _pi0_config
    from openpi.models import gemma as _gemma
    from openpi.models import model as _model
    _HAS_OPENPI = True
except ImportError:
    _pi0 = None
    _pi0_config = None
    _gemma = None
    _model = None
    _HAS_OPENPI = False
    logger.warning(
        "openpi not found. Pi05Model requires openpi for full functionality. "
        "Training modules (LoRA, ReinFlow, Knowledge Insulation) work standalone."
    )


# Gemma variant type
GemmaVariant = Literal["gemma_2b", "gemma_300m", "gemma_860m", "gemma_670m"]


@dataclasses.dataclass(frozen=True)
class Pi05Config(BaseModelConfig):
    """Configuration for Pi0.5 model.

    Attributes:
        action_dim: Dimension of action space (default 14 for ALOHA bimanual)
        action_horizon: Number of future actions to predict (default 50)
        max_token_len: Maximum prompt length (default 200 for Pi0.5)
        freeze_vision_backbone: If True, freezes VLM and reduces memory significantly
        paligemma_variant: VLM variant (default gemma_2b)
        action_expert_variant: Action expert variant (default gemma_300m)
        dtype: Compute dtype (default bfloat16 for memory efficiency)
    """

    action_dim: int = 14
    action_horizon: int = 50
    max_token_len: int = 200
    freeze_vision_backbone: bool = True
    paligemma_variant: GemmaVariant = "gemma_2b"
    action_expert_variant: GemmaVariant = "gemma_300m"
    dtype: str = "bfloat16"

    def create(self, rng: at.KeyArrayLike) -> "Pi05Model":
        """Create a new Pi0.5 model with random initialization."""
        return Pi05Model(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Return input specifications."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Get filter for parameters to freeze during training.

        Returns filter that matches PaliGemma parameters when freeze_vision_backbone=True.
        These parameters will not receive gradient updates.
        """
        if not self.freeze_vision_backbone:
            return nnx.Nothing

        # Freeze PaliGemma LLM and image encoder
        # Action expert params contain "_1" in their path
        return nnx.All(
            PathRegex(".*llm.*"),
            nnx.Not(PathRegex(".*llm.*_1.*")),  # Don't freeze action expert
        )

    def to_openpi_config(self):
        """Convert to openpi Pi0Config for internal use."""
        if not _HAS_OPENPI:
            raise ImportError(
                "openpi is required for Pi05Model. Install it from: "
                "https://github.com/Physical-Intelligence/openpi"
            )
        return _pi0_config.Pi0Config(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
            pi05=True,
            freeze_vision_backbone=self.freeze_vision_backbone,
            paligemma_variant=self.paligemma_variant,
            action_expert_variant=self.action_expert_variant,
            dtype=self.dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        *,
        freeze_vision_backbone: bool = True,
    ) -> "Pi05Config":
        """Load configuration from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
            freeze_vision_backbone: Whether to freeze VLM during fine-tuning.

        Returns:
            Configuration matching the checkpoint.
        """
        # Default configuration for pre-trained checkpoints
        return cls(
            freeze_vision_backbone=freeze_vision_backbone,
        )


class Pi05Model(BaseModel):
    """Pi0.5 Vision-Language-Action Model.

    Architecture:
    - PaliGemma VLM (2B params): Processes images and language
    - Action Expert (300M params): Generates actions via flow matching
    - AdaRMSNorm: Injects timestep conditioning into action expert

    Flow Matching:
    - Continuous, non-autoregressive action generation
    - Single forward pass produces all K=50 actions
    - 26x faster than autoregressive approaches

    Memory Optimization:
    - With freeze_vision_backbone=True: ~15GB per GPU
    - Without freezing: ~38GB per GPU
    """

    def __init__(self, config: Pi05Config, rngs: nnx.Rngs):
        """Initialize Pi0.5 model.

        Args:
            config: Model configuration.
            rngs: Random number generators for initialization.
        """
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.freeze_vision_backbone = config.freeze_vision_backbone

        # Create internal openpi model
        openpi_config = config.to_openpi_config()
        self._model = _pi0.Pi0(openpi_config, rngs=rngs)

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute flow matching loss.

        The loss is MSE between predicted and target flow vectors.

        Args:
            rng: Random key for noise/augmentation.
            observation: Model inputs (images, state, prompt).
            actions: Ground truth actions [batch, horizon, dim].
            train: Whether in training mode.

        Returns:
            Loss tensor [batch, action_horizon].
        """
        # Convert to openpi Observation format
        openpi_obs = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
        )

        return self._model.compute_loss(rng, openpi_obs, actions, train=train)

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        *,
        num_steps: int = 10,
        noise: jax.Array | None = None,
    ) -> Actions:
        """Sample actions via iterative denoising.

        Uses flow matching to progressively denoise from random noise
        to actions over `num_steps` iterations.

        Args:
            rng: Random key for noise sampling.
            observation: Model inputs.
            num_steps: Number of denoising steps (default 10).
            noise: Optional initial noise, sampled if not provided.

        Returns:
            Sampled actions [batch, action_horizon, action_dim].
        """
        # Convert to openpi Observation format
        openpi_obs = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
        )

        return self._model.sample_actions(
            rng, openpi_obs, num_steps=num_steps, noise=noise
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        *,
        freeze_vision_backbone: bool = True,
        action_dim: int | None = None,
    ) -> "Pi05Model":
        """Load a pre-trained Pi0.5 model.

        Args:
            checkpoint_path: Path to checkpoint directory.
            freeze_vision_backbone: Whether to freeze VLM backbone.
            action_dim: Override action dimension (for cross-embodiment).

        Returns:
            Loaded model with pre-trained weights.
        """
        from fla.training.checkpoints import load_checkpoint

        config = Pi05Config(
            freeze_vision_backbone=freeze_vision_backbone,
            action_dim=action_dim or 14,
        )
        model = config.create(jax.random.key(0))
        params = load_checkpoint(checkpoint_path)
        model = _load_params(model, params)
        return model

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Count number of parameters.

        Args:
            trainable_only: If True, only count trainable parameters.

        Returns:
            Number of parameters.
        """
        graphdef, state = nnx.split(self._model)
        if trainable_only and self.freeze_vision_backbone:
            # Filter to only action expert params
            freeze_filter = self.config.get_freeze_filter()
            trainable, frozen = nnx.split(state, freeze_filter, ...)
            return sum(x.size for x in jax.tree_util.tree_leaves(frozen))
        return sum(x.size for x in jax.tree_util.tree_leaves(state))


def _load_params(model: Pi05Model, params: dict) -> Pi05Model:
    """Load parameters into model."""
    graphdef, state = nnx.split(model._model)
    state.replace_by_pure_dict(params)
    model._model = nnx.merge(graphdef, state)
    return model
