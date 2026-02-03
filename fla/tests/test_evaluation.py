"""Tests for FLA evaluation components."""

import pytest
import numpy as np


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_success_rate(self):
        """Test success rate computation."""
        from fla.evaluation.metrics import success_rate

        # 3 out of 5 successful
        rewards = [1.0, 0.5, 1.0, 0.3, 1.0]
        sr = success_rate(rewards, threshold=0.95)
        assert sr == pytest.approx(0.6, rel=0.01)

    def test_success_rate_empty(self):
        """Test success rate with empty list."""
        from fla.evaluation.metrics import success_rate

        sr = success_rate([])
        assert sr == 0.0

    def test_episode_return(self):
        """Test episode return computation."""
        from fla.evaluation.metrics import episode_return

        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, std = episode_return(rewards)
        assert mean == pytest.approx(3.0, rel=0.01)
        assert std == pytest.approx(np.std(rewards), rel=0.01)

    def test_episode_return_empty(self):
        """Test episode return with empty list."""
        from fla.evaluation.metrics import episode_return

        mean, std = episode_return([])
        assert mean == 0.0
        assert std == 0.0

    def test_normalized_score(self):
        """Test normalized score computation."""
        from fla.evaluation.metrics import normalized_score

        # Score exactly at expert level
        score = normalized_score(reward=100, random_score=0, expert_score=100)
        assert score == pytest.approx(100.0, rel=0.01)

        # Score exactly at random level
        score = normalized_score(reward=0, random_score=0, expert_score=100)
        assert score == pytest.approx(0.0, rel=0.01)

        # Score in between
        score = normalized_score(reward=50, random_score=0, expert_score=100)
        assert score == pytest.approx(50.0, rel=0.01)

    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        from fla.evaluation.metrics import aggregate_metrics

        results = [
            {"success_rate": 0.6, "mean_reward": 1.5},
            {"success_rate": 0.8, "mean_reward": 2.0},
            {"success_rate": 0.7, "mean_reward": 1.8},
        ]

        agg = aggregate_metrics(results)

        assert "success_rate_mean" in agg
        assert "success_rate_std" in agg
        assert agg["success_rate_mean"] == pytest.approx(0.7, rel=0.01)


class TestEvalConfig:
    """Tests for EvalConfig."""

    def test_default_config(self):
        """Test default evaluation config."""
        from fla.evaluation.evaluator import EvalConfig

        config = EvalConfig()
        assert config.num_episodes == 50
        assert config.max_steps == 400
        assert config.seed == 0

    def test_custom_config(self):
        """Test custom evaluation config."""
        from fla.evaluation.evaluator import EvalConfig

        config = EvalConfig(
            num_episodes=100,
            max_steps=200,
            save_videos=True,
        )
        assert config.num_episodes == 100
        assert config.max_steps == 200
        assert config.save_videos == True


class TestEpisodeResult:
    """Tests for EpisodeResult."""

    def test_episode_result_creation(self):
        """Test EpisodeResult creation."""
        from fla.evaluation.evaluator import EpisodeResult

        result = EpisodeResult(
            success=True,
            reward=2.0,
            steps=150,
        )
        assert result.success == True
        assert result.reward == 2.0
        assert result.steps == 150

    def test_episode_result_with_info(self):
        """Test EpisodeResult with info dict."""
        from fla.evaluation.evaluator import EpisodeResult

        result = EpisodeResult(
            success=False,
            reward=0.5,
            steps=400,
            info={"reason": "timeout"},
        )
        assert result.info["reason"] == "timeout"


class TestEvalResult:
    """Tests for EvalResult."""

    def test_eval_result_to_dict(self):
        """Test EvalResult serialization."""
        from fla.evaluation.evaluator import EvalResult, EpisodeResult

        episodes = [
            EpisodeResult(success=True, reward=2.0, steps=100),
            EpisodeResult(success=False, reward=0.0, steps=400),
        ]

        result = EvalResult(
            success_rate=0.5,
            mean_reward=1.0,
            std_reward=1.0,
            mean_steps=250,
            num_episodes=2,
            episodes=episodes,
        )

        d = result.to_dict()
        assert d["success_rate"] == 0.5
        assert d["mean_reward"] == 1.0
        assert d["num_episodes"] == 2
        # episodes list not included in to_dict


class TestAlohaEvalConfig:
    """Tests for ALOHA-specific config."""

    def test_task_prompts(self):
        """Test ALOHA task prompts mapping."""
        from fla.evaluation.aloha import TASK_PROMPTS

        assert "gym_aloha/AlohaTransferCube-v0" in TASK_PROMPTS
        assert "gym_aloha/AlohaInsertion-v0" in TASK_PROMPTS

    def test_aloha_eval_config(self):
        """Test ALOHA eval config."""
        from fla.evaluation.aloha import AlohaEvalConfig

        config = AlohaEvalConfig(
            task="gym_aloha/AlohaInsertion-v0",
            num_episodes=10,
        )
        assert config.task == "gym_aloha/AlohaInsertion-v0"
        assert config.num_episodes == 10


# Skip tests that require gym environment
@pytest.mark.skip(reason="Requires gym-aloha installation")
class TestAlohaEvaluator:
    """Tests for ALOHA evaluator."""

    def test_env_creation(self):
        """Test ALOHA environment creation."""
        from fla.evaluation.aloha import AlohaEvaluator
        from fla.models import Pi05Config

        config = Pi05Config()
        model = config.create(jax.random.key(0))

        evaluator = AlohaEvaluator(model)
        env = evaluator.create_env()
        assert env is not None

    def test_obs_conversion(self):
        """Test observation conversion."""
        pass
