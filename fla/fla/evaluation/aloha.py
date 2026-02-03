"""ALOHA simulation evaluation.

Provides evaluation on ALOHA sim tasks:
- gym_aloha/AlohaTransferCube-v0: Transfer cube between hands
- gym_aloha/AlohaInsertion-v0: Insert peg into socket
"""

import dataclasses
import logging
from typing import Any

import jax.numpy as jnp
import numpy as np

from fla.models.base import BaseModel, Observation
from fla.evaluation.evaluator import Evaluator, EvalConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AlohaEvalConfig(EvalConfig):
    """ALOHA-specific evaluation config.

    Attributes:
        task: ALOHA task name.
        prompt: Language instruction for the task.
    """

    task: str = "gym_aloha/AlohaTransferCube-v0"
    prompt: str = "Pick up the cube and transfer it to a new location"


# Default prompts for ALOHA tasks
TASK_PROMPTS = {
    "gym_aloha/AlohaTransferCube-v0": "Pick up the cube and transfer it to a new location",
    "gym_aloha/AlohaInsertion-v0": "Insert the peg into the socket",
    "AlohaTransferCube-v0": "Pick up the cube and transfer it to a new location",
    "AlohaInsertion-v0": "Insert the peg into the socket",
}


class AlohaEvaluator(Evaluator):
    """Evaluator for ALOHA simulation tasks."""

    def __init__(
        self,
        model: BaseModel,
        task: str = "gym_aloha/AlohaTransferCube-v0",
        config: EvalConfig | None = None,
    ):
        """Initialize ALOHA evaluator.

        Args:
            model: VLA model to evaluate.
            task: ALOHA task environment ID.
            config: Evaluation configuration.
        """
        super().__init__(model, config)
        self.task = task
        self.prompt = TASK_PROMPTS.get(task, "Complete the manipulation task")
        self._env = None

    def create_env(self) -> Any:
        """Create ALOHA gym environment."""
        try:
            import gymnasium as gym
            import gym_aloha  # noqa: F401 - registers environments
        except ImportError:
            raise ImportError(
                "gym-aloha required for ALOHA evaluation. "
                "Install with: pip install gymnasium gym-aloha"
            )

        if self._env is None:
            self._env = gym.make(self.task, obs_type="pixels_agent_pos")
            logger.info(f"Created environment: {self.task}")

        return self._env

    def reset_env(self, env: Any) -> tuple[dict, dict]:
        """Reset ALOHA environment."""
        obs, info = env.reset(seed=self.config.seed + len(getattr(self, "_episode_count", [])))
        return obs, info

    def step_env(
        self, env: Any, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """Step ALOHA environment."""
        # ALOHA expects action in [-1, 1] range for 14 DOF
        action = np.clip(action[:14], -1.0, 1.0)
        return env.step(action)

    def obs_to_model_input(self, obs: dict, prompt: str) -> Observation:
        """Convert ALOHA observation to model input.

        ALOHA obs has:
        - pixels/top: [480, 640, 3] uint8
        - pixels/left_wrist: [480, 640, 3] uint8
        - pixels/right_wrist: [480, 640, 3] uint8
        - agent_pos: [14] float32 (joint positions)
        """
        from PIL import Image

        # Resize images to 224x224
        def resize_image(img: np.ndarray) -> np.ndarray:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((224, 224), Image.BILINEAR)
            return np.array(pil_img)

        # Process images
        images = {}
        image_masks = {}

        # Map ALOHA keys to model keys
        key_mapping = {
            "pixels/top": "base_0_rgb",
            "top": "base_0_rgb",
            "pixels/left_wrist": "left_wrist_0_rgb",
            "left_wrist": "left_wrist_0_rgb",
            "pixels/right_wrist": "right_wrist_0_rgb",
            "right_wrist": "right_wrist_0_rgb",
        }

        for obs_key, model_key in key_mapping.items():
            if obs_key in obs:
                img = resize_image(obs[obs_key])
                # Normalize to [-1, 1]
                images[model_key] = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                image_masks[model_key] = True

        # Ensure all required keys exist
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            if key not in images:
                images[key] = np.zeros((224, 224, 3), dtype=np.float32)
                image_masks[key] = False

        # Add batch dimension
        images = {k: v[np.newaxis] for k, v in images.items()}
        image_masks = {k: np.array([v]) for k, v in image_masks.items()}

        # Get state (agent_pos or qpos)
        state = obs.get("agent_pos", obs.get("qpos", np.zeros(14)))
        state = state.astype(np.float32)

        # Pad to action_dim if needed
        if len(state) < self.model.action_dim:
            state = np.pad(state, (0, self.model.action_dim - len(state)))
        state = state[np.newaxis]  # Add batch dimension

        # Tokenize prompt
        # For now, create a simple tokenized prompt placeholder
        # In production, use the proper tokenizer
        max_len = self.model.max_token_len
        tokenized_prompt = np.zeros((1, max_len), dtype=np.int32)
        tokenized_prompt_mask = np.ones((1, max_len), dtype=bool)

        return Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    def get_success_threshold(self) -> float:
        """ALOHA success is reward >= 2.0 for full task completion."""
        return 1.9  # Slightly below 2.0 to handle floating point

    def evaluate(self, prompt: str | None = None) -> Any:
        """Run ALOHA evaluation."""
        prompt = prompt or self.prompt
        logger.info(f"Evaluating on {self.task} with prompt: '{prompt}'")
        return super().evaluate(prompt)


def evaluate_aloha(
    model: BaseModel,
    task: str = "gym_aloha/AlohaTransferCube-v0",
    num_episodes: int = 50,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function for ALOHA evaluation.

    Args:
        model: VLA model to evaluate.
        task: ALOHA task environment ID.
        num_episodes: Number of evaluation episodes.
        **kwargs: Additional EvalConfig arguments.

    Returns:
        Evaluation results dictionary.
    """
    config = EvalConfig(num_episodes=num_episodes, **kwargs)
    evaluator = AlohaEvaluator(model, task=task, config=config)
    result = evaluator.evaluate()

    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Task: {task}")
    logger.info(f"Episodes: {result.num_episodes}")
    logger.info(f"Success Rate: {result.success_rate:.1%}")
    logger.info(f"Mean Reward: {result.mean_reward:.3f} +/- {result.std_reward:.3f}")
    logger.info(f"Mean Steps: {result.mean_steps:.1f}")
    logger.info(f"{'='*60}")

    return result.to_dict()
