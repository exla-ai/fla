"""Tests for ReinFlow module."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import flax.nnx as nnx

from fla.training.reinflow import (
    ReinFlowConfig,
    Trajectory,
    FlowPolicyLogProb,
    ValueFunction,
    compute_gae,
    compute_returns,
    ReinFlowTrainer,
    DPOTrainer,
    create_reinflow_trainer,
)
from fla.models.base import Observation, BaseModel


class MockModel(nnx.Module):
    """Mock model for testing."""

    def __init__(self, action_dim: int = 7, action_horizon: int = 50):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = 200
        self.weight = nnx.Param(jnp.ones((action_dim, action_dim)))

    def compute_loss(self, rng, observation, actions, *, train=False):
        """Mock loss computation."""
        # Simple MSE-like loss
        batch_size = actions.shape[0]
        return jnp.ones((batch_size, self.action_horizon)) * 0.5

    def sample_actions(self, rng, observation, **kwargs):
        """Mock action sampling."""
        batch_size = observation.state.shape[0]
        return jax.random.normal(
            rng, (batch_size, self.action_horizon, self.action_dim)
        )


def create_mock_observation(batch_size: int = 2) -> Observation:
    """Create mock observation for testing."""
    return Observation(
        images={
            "base_0_rgb": jnp.zeros((batch_size, 224, 224, 3)),
        },
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        },
        state=jnp.zeros((batch_size, 7)),
    )


class TestReinFlowConfig:
    """Test ReinFlowConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReinFlowConfig()
        assert config.algorithm == "reinforce"
        assert config.learning_rate == 1e-5
        assert config.entropy_coef == 0.01
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_ratio == 0.2
        assert config.flow_steps == 10
        assert config.use_advantage_normalization is True

    def test_ppo_config(self):
        """Test PPO configuration."""
        config = ReinFlowConfig(
            algorithm="ppo",
            clip_ratio=0.1,
            num_updates_per_rollout=8,
        )
        assert config.algorithm == "ppo"
        assert config.clip_ratio == 0.1
        assert config.num_updates_per_rollout == 8

    def test_dpo_config(self):
        """Test DPO configuration."""
        config = ReinFlowConfig(algorithm="dpo")
        assert config.algorithm == "dpo"


class TestTrajectory:
    """Test Trajectory dataclass."""

    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        obs = [create_mock_observation(2) for _ in range(10)]
        actions = [jnp.zeros((2, 50, 7)) for _ in range(10)]
        rewards = [jnp.ones((2,)) for _ in range(10)]
        dones = [jnp.zeros((2,), dtype=jnp.bool_) for _ in range(10)]

        traj = Trajectory(
            observations=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        assert len(traj.observations) == 10
        assert len(traj.actions) == 10
        assert len(traj.rewards) == 10
        assert len(traj.dones) == 10
        assert traj.log_probs is None
        assert traj.advantages is None


class TestFlowPolicyLogProb:
    """Test FlowPolicyLogProb module."""

    def test_log_prob_computation(self):
        """Test log probability computation."""
        model = MockModel()
        log_prob_fn = FlowPolicyLogProb(model)

        obs = create_mock_observation(2)
        actions = jnp.zeros((2, 50, 7))
        rng = jax.random.key(0)

        log_probs = log_prob_fn(rng, obs, actions)

        assert log_probs.shape == (2,)
        # Mock model returns loss of 0.5, so log_prob should be -0.5
        np.testing.assert_allclose(log_probs, jnp.full((2,), -0.5), rtol=1e-5)


class TestValueFunction:
    """Test ValueFunction module."""

    def test_value_prediction(self):
        """Test value function prediction."""
        rngs = nnx.Rngs(0)
        value_fn = ValueFunction(feature_dim=64, hidden_dim=128, rngs=rngs)

        features = jax.random.normal(jax.random.key(0), (2, 64))
        values = value_fn(features)

        assert values.shape == (2,)

    def test_value_gradient(self):
        """Test value function is differentiable."""
        rngs = nnx.Rngs(0)
        value_fn = ValueFunction(feature_dim=64, hidden_dim=128, rngs=rngs)

        def loss_fn(vf):
            graphdef, state = nnx.split(vf)
            model = nnx.merge(graphdef, state)
            features = jax.random.normal(jax.random.key(0), (2, 64))
            return jnp.mean(model(features))

        grad = jax.grad(loss_fn)(value_fn)
        graphdef, grad_state = nnx.split(grad)

        # Should have non-zero gradients
        grad_leaves = jax.tree_util.tree_leaves(grad_state)
        has_nonzero = any(
            jnp.any(leaf.value != 0) if hasattr(leaf, 'value') else False
            for leaf in grad_leaves
        )
        assert has_nonzero


class TestComputeGAE:
    """Test compute_gae function."""

    def test_gae_computation(self):
        """Test GAE computation."""
        # Simple trajectory: constant rewards, no termination
        rewards = [jnp.ones((2,)) for _ in range(5)]
        values = [jnp.ones((2,)) * 0.5 for _ in range(6)]  # T+1 values
        dones = [jnp.zeros((2,), dtype=jnp.bool_) for _ in range(5)]

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95
        )

        assert len(advantages) == 5
        assert len(returns) == 5

        # Advantages should be non-zero
        assert all(jnp.any(adv != 0) for adv in advantages)

    def test_gae_with_termination(self):
        """Test GAE with episode termination."""
        rewards = [jnp.ones((2,)) for _ in range(5)]
        values = [jnp.ones((2,)) * 0.5 for _ in range(6)]
        dones = [jnp.zeros((2,), dtype=jnp.bool_) for _ in range(5)]
        dones[2] = jnp.ones((2,), dtype=jnp.bool_)  # Terminate at step 2

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95
        )

        assert len(advantages) == 5


class TestComputeReturns:
    """Test compute_returns function."""

    def test_returns_computation(self):
        """Test returns computation."""
        rewards = [jnp.ones((2,)) for _ in range(5)]
        dones = [jnp.zeros((2,), dtype=jnp.bool_) for _ in range(5)]

        returns = compute_returns(rewards, dones, gamma=0.99)

        assert len(returns) == 5

        # First return should be highest (sum of discounted rewards)
        assert jnp.all(returns[0] >= returns[-1])

    def test_returns_with_termination(self):
        """Test returns with episode termination."""
        rewards = [jnp.ones((2,)) for _ in range(5)]
        dones = [jnp.zeros((2,), dtype=jnp.bool_) for _ in range(5)]
        dones[2] = jnp.ones((2,), dtype=jnp.bool_)

        returns = compute_returns(rewards, dones, gamma=0.99)

        # Return at step 3 should start fresh (after termination)
        # This is handled by masking with (1 - done)
        assert len(returns) == 5


class TestReinFlowTrainer:
    """Test ReinFlowTrainer."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = MockModel()
        config = ReinFlowConfig()
        trainer = ReinFlowTrainer(model, config)

        assert trainer.model is model
        assert trainer.config is config
        assert trainer.opt_state is not None

    def test_process_trajectory(self):
        """Test trajectory processing."""
        model = MockModel()
        config = ReinFlowConfig(reward_baseline="mean")
        trainer = ReinFlowTrainer(model, config)

        obs = [create_mock_observation(2) for _ in range(10)]
        actions = [jnp.zeros((2, 50, 7)) for _ in range(10)]
        rewards = [jnp.ones((2,)) * (i + 1) for i in range(10)]  # Increasing rewards
        dones = [jnp.zeros((2,), dtype=jnp.bool_) for _ in range(10)]

        traj = Trajectory(
            observations=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        processed = trainer.process_trajectory(traj)

        assert processed.advantages is not None
        assert processed.returns is not None
        assert len(processed.advantages) == 10
        assert len(processed.returns) == 10

    def test_compute_loss_reinforce(self):
        """Test REINFORCE loss computation."""
        model = MockModel()
        config = ReinFlowConfig(algorithm="reinforce")
        trainer = ReinFlowTrainer(model, config)

        obs = [create_mock_observation(2)]
        actions = [jnp.zeros((2, 50, 7))]
        advantages = [jnp.ones((2,))]

        rng = jax.random.key(0)
        loss, metrics = trainer.compute_loss(rng, obs, actions, advantages)

        assert loss.shape == ()
        assert "policy_loss" in metrics


class TestDPOTrainer:
    """Test DPOTrainer."""

    def test_dpo_trainer_initialization(self):
        """Test DPO trainer initialization."""
        model = MockModel()
        reference = MockModel()
        config = ReinFlowConfig(algorithm="dpo")

        trainer = DPOTrainer(model, reference, config)

        assert trainer.model is model
        assert trainer.reference_model is reference
        assert trainer.beta == 0.1

    def test_dpo_loss_computation(self):
        """Test DPO loss computation."""
        model = MockModel()
        reference = MockModel()
        config = ReinFlowConfig(algorithm="dpo")

        trainer = DPOTrainer(model, reference, config)

        obs = create_mock_observation(2)
        preferred = jnp.zeros((2, 50, 7))
        rejected = jnp.ones((2, 50, 7))

        rng = jax.random.key(0)
        loss, metrics = trainer.compute_dpo_loss(rng, obs, preferred, rejected)

        assert loss.shape == ()
        assert "dpo_loss" in metrics
        assert "preferred_log_prob" in metrics
        assert "rejected_log_prob" in metrics
        assert "margin" in metrics


class TestCreateReinFlowTrainer:
    """Test create_reinflow_trainer factory function."""

    def test_create_reinforce_trainer(self):
        """Test creating REINFORCE trainer."""
        model = MockModel()
        config = ReinFlowConfig(algorithm="reinforce")

        trainer = create_reinflow_trainer(model, config)

        assert isinstance(trainer, ReinFlowTrainer)

    def test_create_ppo_trainer(self):
        """Test creating PPO trainer."""
        model = MockModel()
        config = ReinFlowConfig(algorithm="ppo")

        trainer = create_reinflow_trainer(model, config)

        assert isinstance(trainer, ReinFlowTrainer)

    def test_create_dpo_trainer(self):
        """Test creating DPO trainer."""
        model = MockModel()
        reference = MockModel()
        config = ReinFlowConfig(algorithm="dpo")

        trainer = create_reinflow_trainer(model, config, reference_model=reference)

        assert isinstance(trainer, DPOTrainer)

    def test_dpo_requires_reference(self):
        """Test that DPO requires reference model."""
        model = MockModel()
        config = ReinFlowConfig(algorithm="dpo")

        with pytest.raises(ValueError, match="reference model"):
            create_reinflow_trainer(model, config)


class TestIntegration:
    """Integration tests for ReinFlow."""

    def test_full_training_loop(self):
        """Test a minimal training loop."""
        model = MockModel()
        config = ReinFlowConfig(
            algorithm="reinforce",
            num_updates_per_rollout=1,
            use_advantage_normalization=True,
        )
        trainer = ReinFlowTrainer(model, config)

        # Create trajectory
        obs = [create_mock_observation(2) for _ in range(5)]
        actions = [jnp.zeros((2, 50, 7)) for _ in range(5)]
        rewards = [jnp.ones((2,)) for _ in range(5)]
        dones = [jnp.zeros((2,), dtype=jnp.bool_) for _ in range(5)]

        traj = Trajectory(
            observations=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        rng = jax.random.key(0)
        metrics = trainer.train_on_trajectory(rng, traj)

        assert "policy_loss" in metrics
        assert "grad_norm" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
