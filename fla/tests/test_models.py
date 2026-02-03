"""Tests for FLA model implementations."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestPi05Config:
    """Tests for Pi05Config."""

    def test_default_config(self):
        """Test default configuration values."""
        from fla.models import Pi05Config

        config = Pi05Config()
        assert config.action_dim == 14
        assert config.action_horizon == 50
        assert config.max_token_len == 200
        assert config.freeze_vision_backbone == True
        assert config.paligemma_variant == "gemma_2b"
        assert config.action_expert_variant == "gemma_300m"

    def test_custom_config(self):
        """Test custom configuration."""
        from fla.models import Pi05Config

        config = Pi05Config(
            action_dim=7,
            action_horizon=20,
            freeze_vision_backbone=False,
        )
        assert config.action_dim == 7
        assert config.action_horizon == 20
        assert config.freeze_vision_backbone == False

    def test_inputs_spec(self):
        """Test input specification generation."""
        from fla.models import Pi05Config

        config = Pi05Config(action_dim=14, action_horizon=50)
        obs_spec, action_spec = config.inputs_spec(batch_size=4)

        # Check observation spec
        assert obs_spec.state.shape == (4, 14)
        assert obs_spec.tokenized_prompt.shape == (4, 200)

        # Check image specs
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            assert key in obs_spec.images
            assert obs_spec.images[key].shape == (4, 224, 224, 3)

        # Check action spec
        assert action_spec.shape == (4, 50, 14)

    def test_freeze_filter(self):
        """Test freeze filter generation."""
        from fla.models import Pi05Config
        import flax.nnx as nnx

        # With freezing enabled
        config = Pi05Config(freeze_vision_backbone=True)
        filter_fn = config.get_freeze_filter()
        assert filter_fn != nnx.Nothing

        # Without freezing
        config_no_freeze = Pi05Config(freeze_vision_backbone=False)
        filter_fn = config_no_freeze.get_freeze_filter()
        assert filter_fn == nnx.Nothing


class TestObservation:
    """Tests for Observation class."""

    def test_from_dict(self):
        """Test creating Observation from dictionary."""
        from fla.models.base import Observation

        # Create test data
        data = {
            "image": {
                "base_0_rgb": np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8),
                "left_wrist_0_rgb": np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8),
                "right_wrist_0_rgb": np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.ones(4, dtype=bool),
                "left_wrist_0_rgb": np.ones(4, dtype=bool),
                "right_wrist_0_rgb": np.ones(4, dtype=bool),
            },
            "state": np.random.randn(4, 14).astype(np.float32),
        }

        obs = Observation.from_dict(data)

        # Check images are normalized to [-1, 1]
        for key in data["image"]:
            assert obs.images[key].min() >= -1.0
            assert obs.images[key].max() <= 1.0
            assert obs.images[key].dtype == np.float32

    def test_to_dict(self):
        """Test converting Observation to dictionary."""
        from fla.models.base import Observation

        obs = Observation(
            images={"base_0_rgb": np.zeros((4, 224, 224, 3))},
            image_masks={"base_0_rgb": np.ones(4, dtype=bool)},
            state=np.zeros((4, 14)),
        )

        result = obs.to_dict()
        assert "image" in result
        assert "image_mask" in result
        assert "state" in result


class TestBaseModelConfig:
    """Tests for BaseModelConfig abstract class."""

    def test_fake_obs(self):
        """Test fake observation generation."""
        from fla.models import Pi05Config

        config = Pi05Config()
        fake_obs = config.fake_obs(batch_size=2)

        assert fake_obs.state.shape == (2, 14)
        assert fake_obs.images["base_0_rgb"].shape == (2, 224, 224, 3)

    def test_fake_actions(self):
        """Test fake actions generation."""
        from fla.models import Pi05Config

        config = Pi05Config()
        fake_actions = config.fake_actions(batch_size=2)

        assert fake_actions.shape == (2, 50, 14)


# Skip model creation tests that require full model initialization
# These are expensive and should be run separately
@pytest.mark.skip(reason="Requires full model initialization - expensive")
class TestPi05Model:
    """Tests for Pi0.5 model."""

    def test_model_creation(self):
        """Test model creation."""
        from fla.models import Pi05Config, Pi05Model

        config = Pi05Config()
        model = config.create(jax.random.key(0))
        assert isinstance(model, Pi05Model)

    def test_compute_loss(self):
        """Test loss computation."""
        from fla.models import Pi05Config

        config = Pi05Config()
        model = config.create(jax.random.key(0))

        obs = config.fake_obs(batch_size=2)
        actions = config.fake_actions(batch_size=2)

        rng = jax.random.key(42)
        loss = model.compute_loss(rng, obs, actions, train=True)

        assert loss.shape == (2, 50)  # [batch, horizon]
        assert jnp.isfinite(loss).all()

    def test_sample_actions(self):
        """Test action sampling."""
        from fla.models import Pi05Config

        config = Pi05Config()
        model = config.create(jax.random.key(0))

        obs = config.fake_obs(batch_size=2)

        rng = jax.random.key(42)
        actions = model.sample_actions(rng, obs, num_steps=5)

        assert actions.shape == (2, 50, 14)
        assert jnp.isfinite(actions).all()
