"""Tests for FLA training components."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax


class TestOptimizer:
    """Tests for optimizer creation."""

    def test_cosine_schedule(self):
        """Test cosine decay schedule."""
        from fla.training.optimizer import CosineSchedule

        schedule = CosineSchedule(
            peak_lr=1e-4,
            warmup_steps=100,
            decay_steps=1000,
            end_lr=1e-6,
        )

        # During warmup
        assert schedule(0) == 0.0
        assert schedule(50) == pytest.approx(5e-5, rel=0.01)
        assert schedule(100) == pytest.approx(1e-4, rel=0.01)

        # After warmup
        assert schedule(1000) == pytest.approx(1e-6, rel=0.01)

    def test_rsqrt_schedule(self):
        """Test inverse square root schedule."""
        from fla.training.optimizer import RsqrtSchedule

        schedule = RsqrtSchedule(
            peak_lr=1e-4,
            warmup_steps=100,
        )

        # During warmup
        assert schedule(0) == 0.0
        assert schedule(100) == pytest.approx(1e-4, rel=0.01)

        # After warmup: lr = peak * sqrt(warmup / step)
        assert schedule(400) == pytest.approx(5e-5, rel=0.01)

    def test_create_optimizer(self):
        """Test optimizer creation."""
        from fla.training.optimizer import create_optimizer, CosineSchedule

        schedule = CosineSchedule()
        optimizer = create_optimizer(schedule, gradient_clip=1.0)

        # Test that optimizer works
        params = {"w": jnp.ones((10, 10))}
        opt_state = optimizer.init(params)

        grads = {"w": jnp.ones((10, 10)) * 0.1}
        updates, new_state = optimizer.update(grads, opt_state, params)

        assert "w" in updates
        assert updates["w"].shape == (10, 10)

    def test_optimizer_with_learning_rate(self):
        """Test optimizer with constant learning rate."""
        from fla.training.optimizer import create_optimizer

        optimizer = create_optimizer(learning_rate=1e-4)

        params = {"w": jnp.ones((5,))}
        opt_state = optimizer.init(params)

        assert opt_state is not None


class TestTrainConfig:
    """Tests for TrainConfig."""

    def test_default_config(self):
        """Test default training config."""
        from fla.training.trainer import TrainConfig

        config = TrainConfig()
        assert config.max_steps == 30000
        assert config.batch_size == 32
        assert config.learning_rate == 2.5e-5
        assert config.warmup_steps == 1000

    def test_custom_config(self):
        """Test custom training config."""
        from fla.training.trainer import TrainConfig

        config = TrainConfig(
            max_steps=10000,
            batch_size=64,
            learning_rate=1e-4,
            checkpoint_dir="./my_checkpoints",
        )
        assert config.max_steps == 10000
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.checkpoint_dir == "./my_checkpoints"


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_global_norm_clipping(self):
        """Test that gradients are clipped by global norm."""
        from fla.training.optimizer import create_optimizer

        optimizer = create_optimizer(learning_rate=1e-3, gradient_clip=1.0)

        params = {"w": jnp.ones((10,))}
        opt_state = optimizer.init(params)

        # Large gradients
        grads = {"w": jnp.ones((10,)) * 10.0}
        updates, _ = optimizer.update(grads, opt_state, params)

        # Check that updates are bounded
        update_norm = jnp.sqrt(jnp.sum(updates["w"] ** 2))
        # After clipping and Adam, the norm should be reasonable
        assert update_norm < 1.0


class TestCheckpoints:
    """Tests for checkpoint management."""

    def test_checkpoint_manager_creation(self):
        """Test checkpoint manager creation."""
        import tempfile
        from fla.training.checkpoints import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                tmpdir,
                keep_last_n=3,
                save_interval=100,
            )
            assert manager.latest_step is None


# Skip tests that require full training setup
@pytest.mark.skip(reason="Requires full training setup - expensive")
class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_creation(self):
        """Test trainer creation."""
        from fla.training.trainer import Trainer, TrainConfig
        from fla.models import Pi05Config
        from fla.data.lerobot_loader import LeRobotDataLoader, DataConfig

        model_config = Pi05Config()
        model = model_config.create(jax.random.key(0))

        data_config = DataConfig(repo_id="fake")
        dataloader = LeRobotDataLoader(data_config)

        config = TrainConfig(max_steps=10)
        trainer = Trainer(model, dataloader, config=config)

        assert trainer.step == 0
        assert trainer.model is not None

    def test_train_step(self):
        """Test single training step."""
        # This would test the actual training loop
        pass
