"""Base evaluator for VLA models."""

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any

import jax
import numpy as np

from fla.models.base import BaseModel, Observation

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        num_episodes: Number of episodes to evaluate.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.
        render: Whether to render episodes.
        save_videos: Whether to save episode videos.
        output_dir: Directory for outputs.
    """

    num_episodes: int = 50
    max_steps: int = 400
    seed: int = 0
    render: bool = False
    save_videos: bool = False
    output_dir: str = "./eval_results"


@dataclasses.dataclass
class EpisodeResult:
    """Result from a single episode."""

    success: bool
    reward: float
    steps: int
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class EvalResult:
    """Aggregate evaluation results."""

    success_rate: float
    mean_reward: float
    std_reward: float
    mean_steps: float
    num_episodes: int
    episodes: list[EpisodeResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_steps": self.mean_steps,
            "num_episodes": self.num_episodes,
        }


class Evaluator(ABC):
    """Base class for VLA evaluators."""

    def __init__(
        self,
        model: BaseModel,
        config: EvalConfig | None = None,
    ):
        """Initialize evaluator.

        Args:
            model: VLA model to evaluate.
            config: Evaluation configuration.
        """
        self.model = model
        self.config = config or EvalConfig()
        self.rng = jax.random.PRNGKey(self.config.seed)

    @abstractmethod
    def create_env(self) -> Any:
        """Create evaluation environment."""

    @abstractmethod
    def reset_env(self, env: Any) -> tuple[dict, dict]:
        """Reset environment and return initial observation."""

    @abstractmethod
    def step_env(
        self, env: Any, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """Step environment with action."""

    @abstractmethod
    def obs_to_model_input(self, obs: dict, prompt: str) -> Observation:
        """Convert environment observation to model input."""

    def evaluate(self, prompt: str = "Complete the task") -> EvalResult:
        """Run full evaluation.

        Args:
            prompt: Language instruction for the task.

        Returns:
            Evaluation results.
        """
        env = self.create_env()
        episodes = []

        for ep in range(self.config.num_episodes):
            result = self._run_episode(env, prompt)
            episodes.append(result)

            status = "success" if result.success else "fail"
            logger.info(
                f"Episode {ep + 1}/{self.config.num_episodes}: "
                f"{status} | reward={result.reward:.3f} | steps={result.steps}"
            )

            # Log running success rate every 10 episodes
            if (ep + 1) % 10 == 0:
                running_sr = sum(e.success for e in episodes) / len(episodes)
                logger.info(f"  Running success rate: {running_sr:.1%}")

        # Compute aggregate metrics
        rewards = [e.reward for e in episodes]
        return EvalResult(
            success_rate=sum(e.success for e in episodes) / len(episodes),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_steps=float(np.mean([e.steps for e in episodes])),
            num_episodes=len(episodes),
            episodes=episodes,
        )

    def _run_episode(self, env: Any, prompt: str) -> EpisodeResult:
        """Run a single episode."""
        obs, info = self.reset_env(env)
        total_reward = 0.0
        steps = 0

        for step in range(self.config.max_steps):
            # Convert observation to model input
            model_input = self.obs_to_model_input(obs, prompt)

            # Sample actions from model
            self.rng, action_rng = jax.random.split(self.rng)
            actions = self.model.sample_actions(action_rng, model_input)

            # Execute first action (action chunking)
            action = np.array(actions[0, 0])  # [batch, horizon, dim] -> first action

            # Step environment
            obs, reward, terminated, truncated, info = self.step_env(env, action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # Determine success (task-specific threshold)
        success = total_reward >= self.get_success_threshold()

        return EpisodeResult(
            success=success,
            reward=total_reward,
            steps=steps,
            info=info,
        )

    def get_success_threshold(self) -> float:
        """Get reward threshold for success. Override in subclass."""
        return 0.95
