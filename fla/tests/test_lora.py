"""Tests for LoRA module."""

import os
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import flax.nnx as nnx

from fla.training.lora import (
    LoRAConfig,
    LoRALinear,
    create_lora_config,
    get_lora_gemma_variant,
    get_lora_params_filter,
    get_frozen_params_filter,
    count_lora_params,
    save_lora_adapter,
    load_lora_adapter,
)


class TestLoRAConfig:
    """Test LoRAConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        assert config.target_modules == "all"
        assert config.apply_to_vlm is False
        assert config.apply_to_action_expert is True
        assert config.rslora is True
        assert config.init_scale == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = LoRAConfig(
            rank=32,
            alpha=64.0,
            dropout=0.1,
            target_modules="attention",
            apply_to_vlm=True,
        )
        assert config.rank == 32
        assert config.alpha == 64.0
        assert config.dropout == 0.1
        assert config.target_modules == "attention"
        assert config.apply_to_vlm is True


class TestCreateLoRAConfig:
    """Test create_lora_config function."""

    def test_attention_only(self):
        """Test attention-only LoRA config."""
        config = LoRAConfig(target_modules="attention")
        lora_configs = create_lora_config(config)

        assert "attn" in lora_configs
        assert "ffn" not in lora_configs

    def test_ffn_only(self):
        """Test FFN-only LoRA config."""
        config = LoRAConfig(target_modules="ffn")
        lora_configs = create_lora_config(config)

        assert "ffn" in lora_configs
        assert "attn" not in lora_configs

    def test_all_modules(self):
        """Test all modules LoRA config."""
        config = LoRAConfig(target_modules="all")
        lora_configs = create_lora_config(config)

        assert "attn" in lora_configs
        assert "ffn" in lora_configs

    def test_rslora_scaling(self):
        """Test rsLoRA scaling is applied."""
        config = LoRAConfig(rank=16, alpha=16.0, rslora=True)
        lora_configs = create_lora_config(config)

        # rsLoRA scaling = alpha / sqrt(rank)
        expected_scaling = 16.0 / jnp.sqrt(16)
        assert lora_configs["attn"].rslora is True


class TestGetLoRAGemmaVariant:
    """Test get_lora_gemma_variant function."""

    def test_default_variants(self):
        """Test default variants without LoRA."""
        config = LoRAConfig(apply_to_vlm=False, apply_to_action_expert=False)
        vlm, expert = get_lora_gemma_variant("gemma_2b", config)

        assert vlm == "gemma_2b"
        assert expert == "gemma_300m"

    def test_vlm_lora(self):
        """Test VLM with LoRA."""
        config = LoRAConfig(apply_to_vlm=True, apply_to_action_expert=False)
        vlm, expert = get_lora_gemma_variant("gemma_2b", config)

        assert vlm == "gemma_2b_lora"
        assert expert == "gemma_300m"

    def test_expert_lora(self):
        """Test action expert with LoRA."""
        config = LoRAConfig(apply_to_vlm=False, apply_to_action_expert=True)
        vlm, expert = get_lora_gemma_variant("gemma_2b", config)

        assert vlm == "gemma_2b"
        assert expert == "gemma_300m_lora"

    def test_both_lora(self):
        """Test both VLM and expert with LoRA."""
        config = LoRAConfig(apply_to_vlm=True, apply_to_action_expert=True)
        vlm, expert = get_lora_gemma_variant("gemma_2b", config)

        assert vlm == "gemma_2b_lora"
        assert expert == "gemma_300m_lora"


class TestLoRALinear:
    """Test LoRALinear module."""

    def test_initialization(self):
        """Test LoRA linear initialization."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        assert layer.weight.value.shape == (64, 128)
        assert layer.lora_a.value.shape == (64, 8)
        assert layer.lora_b.value.shape == (8, 128)

    def test_lora_b_zero_init(self):
        """Test that lora_b is initialized to zero."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        np.testing.assert_array_equal(
            layer.lora_b.value, jnp.zeros((8, 128))
        )

    def test_forward_pass(self):
        """Test forward pass."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.key(0), (2, 10, 64))
        output = layer(x)

        assert output.shape == (2, 10, 128)

    def test_initial_output_unchanged(self):
        """Test that initial output equals base linear."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.key(0), (2, 10, 64))

        # Initial output should equal base linear (since lora_b is zero)
        lora_output = layer(x)
        base_output = jnp.dot(x, layer.weight.value)

        np.testing.assert_allclose(lora_output, base_output, rtol=1e-5)

    def test_lora_adaptation(self):
        """Test that LoRA changes output when trained."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.key(0), (2, 10, 64))

        # Modify lora_b
        layer.lora_b.value = jax.random.normal(jax.random.key(1), (8, 128))

        # Output should now differ from base
        lora_output = layer(x)
        base_output = jnp.dot(x, layer.weight.value)

        assert not jnp.allclose(lora_output, base_output)

    def test_merge_lora(self):
        """Test LoRA weight merging."""
        config = LoRAConfig(rank=8, alpha=8.0, rslora=False)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        # Set non-zero LoRA weights
        layer.lora_a.value = jax.random.normal(jax.random.key(1), (64, 8))
        layer.lora_b.value = jax.random.normal(jax.random.key(2), (8, 128))

        x = jax.random.normal(jax.random.key(0), (2, 10, 64))
        output_before = layer(x)

        # Merge LoRA into base weights
        layer.merge_lora()

        # Output should be the same
        output_after = layer(x)
        np.testing.assert_allclose(output_before, output_after, rtol=1e-4)

        # LoRA matrices should be zero
        np.testing.assert_array_equal(
            layer.lora_a.value, jnp.zeros((64, 8))
        )
        np.testing.assert_array_equal(
            layer.lora_b.value, jnp.zeros((8, 128))
        )

    def test_with_bias(self):
        """Test LoRA linear with bias."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
            use_bias=True,
        )

        assert layer.bias is not None
        assert layer.bias.value.shape == (128,)

        x = jax.random.normal(jax.random.key(0), (2, 10, 64))
        output = layer(x)
        assert output.shape == (2, 10, 128)


class TestLoRAParamsFilter:
    """Test LoRA parameter filtering functions."""

    def test_lora_params_filter(self):
        """Test get_lora_params_filter returns correct filter."""
        config = LoRAConfig()
        filter = get_lora_params_filter(config)

        # Filter should match "lora" in path
        assert filter is not None

    def test_frozen_params_filter_no_vlm(self):
        """Test frozen params filter when VLM is not trained."""
        config = LoRAConfig(apply_to_vlm=False, apply_to_action_expert=True)
        filter = get_frozen_params_filter(config)

        # Should freeze VLM but not LoRA
        assert filter is not None
        assert filter != nnx.Nothing

    def test_frozen_params_filter_no_lora(self):
        """Test frozen params filter when no LoRA is used."""
        config = LoRAConfig(apply_to_vlm=False, apply_to_action_expert=False)
        filter = get_frozen_params_filter(config)

        # With no LoRA, should return Nothing (train everything)
        assert filter == nnx.Nothing


class TestCountLoRAParams:
    """Test count_lora_params function."""

    def test_count_params(self):
        """Test parameter counting."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        graphdef, state = nnx.split(layer)
        lora_count, total_count = count_lora_params(state)

        # LoRA params: lora_a (64*8) + lora_b (8*128) = 512 + 1024 = 1536
        expected_lora = 64 * 8 + 8 * 128
        assert lora_count == expected_lora

        # Total includes base weight (64*128) + LoRA
        expected_total = 64 * 128 + expected_lora
        assert total_count == expected_total


class TestLoRASaveLoad:
    """Test LoRA adapter save/load functions."""

    def test_save_load_adapter(self):
        """Test saving and loading LoRA adapter."""
        config = LoRAConfig(rank=8)
        rngs = nnx.Rngs(0)
        layer = LoRALinear(
            in_features=64,
            out_features=128,
            config=config,
            rngs=rngs,
        )

        # Set non-zero LoRA weights
        layer.lora_a.value = jax.random.normal(jax.random.key(1), (64, 8))
        layer.lora_b.value = jax.random.normal(jax.random.key(2), (8, 128))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "adapter.pkl")

            # Save adapter
            save_lora_adapter(layer, path, config)

            # Create new layer
            rngs2 = nnx.Rngs(99)
            new_layer = LoRALinear(
                in_features=64,
                out_features=128,
                config=config,
                rngs=rngs2,
            )

            # Verify they're different initially
            assert not jnp.allclose(
                layer.lora_a.value, new_layer.lora_a.value
            )

            # Load adapter
            loaded_config = load_lora_adapter(new_layer, path)

            # Config should match
            assert loaded_config.rank == config.rank
            assert loaded_config.alpha == config.alpha


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
