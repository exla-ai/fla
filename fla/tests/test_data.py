"""Tests for FLA data loading."""

import pytest
import numpy as np


class TestNormStats:
    """Tests for normalization statistics."""

    def test_norm_stats_creation(self):
        """Test NormStats creation."""
        from fla.shared.normalize import NormStats

        stats = NormStats(
            mean=np.array([0.0, 1.0, 2.0]),
            std=np.array([1.0, 2.0, 3.0]),
        )
        assert stats.mean.shape == (3,)
        assert stats.std.shape == (3,)
        assert stats.q01 is None
        assert stats.q99 is None

    def test_norm_stats_with_quantiles(self):
        """Test NormStats with quantile values."""
        from fla.shared.normalize import NormStats

        stats = NormStats(
            mean=np.array([0.0]),
            std=np.array([1.0]),
            q01=np.array([-2.5]),
            q99=np.array([2.5]),
        )
        assert stats.q01 is not None
        assert stats.q99 is not None

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        from fla.shared.normalize import NormStats

        original = NormStats(
            mean=np.array([1.0, 2.0]),
            std=np.array([0.5, 0.5]),
            q01=np.array([-1.0, 0.0]),
            q99=np.array([3.0, 4.0]),
        )

        d = original.to_dict()
        restored = NormStats.from_dict(d)

        np.testing.assert_array_almost_equal(original.mean, restored.mean)
        np.testing.assert_array_almost_equal(original.std, restored.std)
        np.testing.assert_array_almost_equal(original.q01, restored.q01)
        np.testing.assert_array_almost_equal(original.q99, restored.q99)


class TestNormalization:
    """Tests for normalization functions."""

    def test_normalize_zscore(self):
        """Test z-score normalization."""
        from fla.shared.normalize import NormStats, normalize

        stats = NormStats(
            mean=np.array([0.0]),
            std=np.array([2.0]),
        )

        x = np.array([2.0])
        normalized = normalize(x, stats, use_quantiles=False)
        expected = np.array([1.0])  # (2 - 0) / 2 = 1

        np.testing.assert_array_almost_equal(normalized, expected)

    def test_unnormalize_zscore(self):
        """Test z-score unnormalization."""
        from fla.shared.normalize import NormStats, unnormalize

        stats = NormStats(
            mean=np.array([0.0]),
            std=np.array([2.0]),
        )

        x = np.array([1.0])
        unnormalized = unnormalize(x, stats, use_quantiles=False)
        expected = np.array([2.0])  # 1 * 2 + 0 = 2

        np.testing.assert_array_almost_equal(unnormalized, expected)

    def test_normalize_quantile(self):
        """Test quantile normalization."""
        from fla.shared.normalize import NormStats, normalize

        stats = NormStats(
            mean=np.array([0.0]),
            std=np.array([1.0]),
            q01=np.array([-1.0]),
            q99=np.array([1.0]),
        )

        # Value at q01 should map to -1, at q99 should map to 1
        x = np.array([0.0])  # Middle value
        normalized = normalize(x, stats, use_quantiles=True)
        expected = np.array([0.0])  # (0 - (-1)) / (1 - (-1)) * 2 - 1 = 0

        np.testing.assert_array_almost_equal(normalized, expected)

    def test_running_stats(self):
        """Test online statistics computation."""
        from fla.shared.normalize import RunningStats

        stats = RunningStats(shape=(2,))

        # Add samples
        samples = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ]
        for s in samples:
            stats.update(s)

        expected_mean = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(stats.mean, expected_mean)

        assert stats.n == 3


class TestTransforms:
    """Tests for data transforms."""

    def test_pad_actions(self):
        """Test action padding."""
        from fla.data.transforms import PadActions

        transform = PadActions(target_dim=14)

        # 7-DOF robot action
        data = {
            "state": np.random.randn(7).astype(np.float32),
            "actions": np.random.randn(50, 7).astype(np.float32),
        }

        result = transform(data)

        assert result["state"].shape == (14,)
        assert result["actions"].shape == (50, 14)
        # Check that original values are preserved
        np.testing.assert_array_equal(result["state"][:7], data["state"][:7])

    def test_normalize_images(self):
        """Test image normalization."""
        from fla.data.transforms import NormalizeImages

        transform = NormalizeImages()

        data = {
            "image": {
                "base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            }
        }

        result = transform(data)

        # Check range is [-1, 1]
        assert result["image"]["base_0_rgb"].min() >= -1.0
        assert result["image"]["base_0_rgb"].max() <= 1.0
        assert result["image"]["base_0_rgb"].dtype == np.float32

    def test_composite_transform(self):
        """Test transform composition."""
        from fla.data.transforms import CompositeTransform, NormalizeImages, PadActions

        transform = CompositeTransform([
            PadActions(target_dim=14),
        ])

        data = {
            "state": np.random.randn(7).astype(np.float32),
            "actions": np.random.randn(50, 7).astype(np.float32),
        }

        result = transform(data)
        assert result["state"].shape == (14,)

    def test_repack_images(self):
        """Test image key repacking."""
        from fla.data.transforms import RepackImages

        transform = RepackImages()

        data = {
            "image": {
                "observation.images.top": np.zeros((224, 224, 3), dtype=np.uint8),
            }
        }

        result = transform(data)

        # Check standard keys are present
        assert "base_0_rgb" in result["image"]
        assert "left_wrist_0_rgb" in result["image"]
        assert "right_wrist_0_rgb" in result["image"]

        # Check masks
        assert result["image_mask"]["base_0_rgb"] == True
        assert result["image_mask"]["left_wrist_0_rgb"] == False  # Missing -> dummy


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_config(self):
        """Test default data config."""
        from fla.data.lerobot_loader import DataConfig

        config = DataConfig(repo_id="lerobot/aloha_sim_transfer_cube_human")
        assert config.action_horizon == 50
        assert config.batch_size == 32
        assert config.shuffle == True

    def test_custom_config(self):
        """Test custom data config."""
        from fla.data.lerobot_loader import DataConfig

        config = DataConfig(
            repo_id="lerobot/libero_10",
            action_horizon=20,
            batch_size=16,
            prompt="Move the object",
        )
        assert config.action_horizon == 20
        assert config.batch_size == 16
        assert config.prompt == "Move the object"


# Skip tests that require actual dataset download
@pytest.mark.skip(reason="Requires dataset download")
class TestLeRobotDataset:
    """Tests for LeRobot dataset loading."""

    def test_load_dataset(self):
        """Test loading a LeRobot dataset."""
        from fla.data.lerobot_loader import LeRobotDataset

        dataset = LeRobotDataset(
            "lerobot/aloha_sim_transfer_cube_human",
            action_horizon=50,
        )

        assert len(dataset) > 0

        # Check sample format
        sample = dataset[0]
        assert "image" in sample
        assert "state" in sample
        assert "actions" in sample
        assert sample["actions"].shape[0] == 50  # action_horizon
