"""Tests for Knowledge Insulation module."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import flax.nnx as nnx

from fla.training.knowledge_insulation import (
    KnowledgeInsulationConfig,
    apply_knowledge_insulation,
    insulate_prefix_tokens,
    InsulatedEmbedding,
    InsulatedLayerNorm,
    DiscreteStateEncoder,
    create_insulated_model_filter,
)


class TestKnowledgeInsulationConfig:
    """Test KnowledgeInsulationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KnowledgeInsulationConfig()
        assert config.mode == "full"
        assert config.gradient_scale == 0.0
        assert config.preserve_vlm_statistics is True
        assert config.use_discrete_state is True

    def test_soft_mode_config(self):
        """Test soft mode configuration."""
        config = KnowledgeInsulationConfig(mode="soft", gradient_scale=0.1)
        assert config.mode == "soft"
        assert config.gradient_scale == 0.1

    def test_selective_mode_config(self):
        """Test selective mode configuration."""
        config = KnowledgeInsulationConfig(
            mode="selective",
            insulated_layers=(".*llm.*",),
        )
        assert config.mode == "selective"
        assert ".*llm.*" in config.insulated_layers


class TestApplyKnowledgeInsulation:
    """Test apply_knowledge_insulation function."""

    def test_full_insulation_stops_gradient(self):
        """Test that full mode stops gradients."""
        config = KnowledgeInsulationConfig(mode="full")
        tokens = jnp.ones((2, 10, 64))

        # Forward pass
        insulated = apply_knowledge_insulation(tokens, config)
        assert insulated.shape == tokens.shape
        np.testing.assert_array_equal(insulated, tokens)

        # Check gradient is zero
        def loss_fn(x):
            return jnp.sum(apply_knowledge_insulation(x, config))

        grad = jax.grad(loss_fn)(tokens)
        np.testing.assert_array_equal(grad, jnp.zeros_like(tokens))

    def test_soft_insulation_scales_gradient(self):
        """Test that soft mode scales gradients."""
        scale = 0.1
        config = KnowledgeInsulationConfig(mode="soft", gradient_scale=scale)
        tokens = jnp.ones((2, 10, 64))

        def loss_fn(x):
            return jnp.sum(apply_knowledge_insulation(x, config))

        grad = jax.grad(loss_fn)(tokens)
        expected = jnp.ones_like(tokens) * scale
        np.testing.assert_allclose(grad, expected, rtol=1e-5)

    def test_selective_passes_through(self):
        """Test that selective mode passes tokens through."""
        config = KnowledgeInsulationConfig(mode="selective")
        tokens = jnp.ones((2, 10, 64))

        insulated = apply_knowledge_insulation(tokens, config)
        np.testing.assert_array_equal(insulated, tokens)


class TestInsulatedPrefixTokens:
    """Test insulate_prefix_tokens function."""

    def test_insulates_prefix_only(self):
        """Test that only prefix tokens are insulated."""
        config = KnowledgeInsulationConfig(mode="full")
        prefix = jnp.ones((2, 10, 64))
        suffix = jnp.ones((2, 5, 64))

        insulated_prefix, returned_suffix = insulate_prefix_tokens(
            prefix, suffix, config
        )

        # Suffix should be unchanged
        np.testing.assert_array_equal(returned_suffix, suffix)

        # Prefix should have stopped gradients
        def loss_fn(p, s):
            ip, rs = insulate_prefix_tokens(p, s, config)
            return jnp.sum(ip) + jnp.sum(rs)

        grad_prefix, grad_suffix = jax.grad(loss_fn, argnums=(0, 1))(prefix, suffix)

        # Prefix gradient should be zero
        np.testing.assert_array_equal(grad_prefix, jnp.zeros_like(prefix))
        # Suffix gradient should be ones
        np.testing.assert_array_equal(grad_suffix, jnp.ones_like(suffix))


class TestInsulatedEmbedding:
    """Test InsulatedEmbedding module."""

    def test_embedding_lookup(self):
        """Test basic embedding lookup."""
        config = KnowledgeInsulationConfig(mode="full")
        rngs = nnx.Rngs(0)
        embedding = InsulatedEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            config=config,
            rngs=rngs,
        )

        indices = jnp.array([[0, 1, 2], [3, 4, 5]])
        output = embedding(indices)

        assert output.shape == (2, 3, 64)

    def test_embedding_gradient_stopped(self):
        """Test that gradients are stopped for full mode."""
        config = KnowledgeInsulationConfig(mode="full")
        rngs = nnx.Rngs(0)
        embedding = InsulatedEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            config=config,
            rngs=rngs,
        )

        indices = jnp.array([[0, 1, 2]])

        # Split module first to get pytree state
        graphdef, state = nnx.split(embedding)

        def loss_fn(params):
            model = nnx.merge(graphdef, params)
            return jnp.sum(model(indices))

        # Gradient should be zero for embedding weights
        grad = jax.grad(loss_fn)(state)
        grad_leaves = jax.tree_util.tree_leaves(grad)
        for leaf in grad_leaves:
            if hasattr(leaf, 'value'):
                np.testing.assert_array_equal(leaf.value, jnp.zeros_like(leaf.value))


class TestInsulatedLayerNorm:
    """Test InsulatedLayerNorm module."""

    def test_layer_norm_output(self):
        """Test layer normalization output."""
        config = KnowledgeInsulationConfig(preserve_vlm_statistics=False)
        rngs = nnx.Rngs(0)
        norm = InsulatedLayerNorm(dim=64, config=config, rngs=rngs)

        x = jax.random.normal(jax.random.key(0), (2, 10, 64))
        output = norm(x)

        assert output.shape == x.shape
        # Check roughly normalized
        assert jnp.abs(jnp.mean(output)).item() < 0.1
        assert jnp.abs(jnp.std(output) - 1.0).item() < 0.5

    def test_running_stats_mode(self):
        """Test using running statistics."""
        config = KnowledgeInsulationConfig(preserve_vlm_statistics=True)
        rngs = nnx.Rngs(0)
        norm = InsulatedLayerNorm(dim=64, config=config, rngs=rngs)

        x = jax.random.normal(jax.random.key(0), (2, 10, 64))
        output_running = norm(x, use_running_stats=True)
        output_batch = norm(x, use_running_stats=False)

        # With different stats, outputs should differ
        assert not jnp.allclose(output_running, output_batch)


class TestDiscreteStateEncoder:
    """Test DiscreteStateEncoder module."""

    def test_discretization(self):
        """Test state discretization."""
        rngs = nnx.Rngs(0)
        encoder = DiscreteStateEncoder(
            state_dim=7,
            num_bins=256,
            embedding_dim=64,
            rngs=rngs,
        )

        state = jnp.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25]])
        indices = encoder.discretize(state)

        assert indices.shape == (1, 7)
        # Check indices are in valid range
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < 7 * 256)

    def test_encoding(self):
        """Test full state encoding."""
        rngs = nnx.Rngs(0)
        encoder = DiscreteStateEncoder(
            state_dim=7,
            num_bins=256,
            embedding_dim=64,
            rngs=rngs,
        )

        state = jnp.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25]])
        embeddings = encoder(state)

        assert embeddings.shape == (1, 7, 64)

    def test_statistics_update(self):
        """Test running statistics update."""
        rngs = nnx.Rngs(0)
        encoder = DiscreteStateEncoder(
            state_dim=7,
            num_bins=256,
            embedding_dim=64,
            rngs=rngs,
        )

        initial_min = encoder.state_min.value.copy()
        initial_max = encoder.state_max.value.copy()

        # Update with new data
        state = jax.random.uniform(jax.random.key(1), (32, 7), minval=-2, maxval=2)
        encoder.update_statistics(state, momentum=0.99)

        # Statistics should have changed
        assert not jnp.allclose(encoder.state_min.value, initial_min)
        assert not jnp.allclose(encoder.state_max.value, initial_max)


class TestModelFilter:
    """Test create_insulated_model_filter function."""

    def test_full_mode_returns_nothing(self):
        """Test that full mode returns Nothing filter."""
        config = KnowledgeInsulationConfig(mode="full")
        filter = create_insulated_model_filter(config)
        assert filter == nnx.Nothing

    def test_selective_mode_returns_filter(self):
        """Test that selective mode returns proper filter."""
        config = KnowledgeInsulationConfig(
            mode="selective",
            insulated_layers=(".*llm.*", ".*img.*"),
        )
        filter = create_insulated_model_filter(config)
        assert filter is not None
        assert filter != nnx.Nothing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
