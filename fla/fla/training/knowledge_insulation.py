"""Knowledge Insulation for VLA Fine-tuning.

Logging:
    This module uses Python's logging module. Configure with:
    >>> import logging
    >>> logging.getLogger("fla.training.knowledge_insulation").setLevel(logging.DEBUG)

Knowledge Insulation (from Pi0.5 paper) prevents catastrophic forgetting during
fine-tuning by maintaining separation between:
- Discrete tokens: VLM processes language/images using discrete embeddings
- Continuous tokens: Action expert uses continuous embeddings for actions

Key insight: By stopping gradients at the VLM output and using separate
embedding spaces, the pre-trained VLM knowledge is preserved while the
action expert adapts to new tasks.

Reference: Physical Intelligence, "pi0.5: A Vision-Language-Action Model"
"""

import dataclasses
import logging
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from fla.shared import array_typing as at

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class KnowledgeInsulationConfig:
    """Configuration for Knowledge Insulation.

    Attributes:
        mode: Insulation mode:
            - "full": Complete gradient isolation (stop_gradient on VLM outputs)
            - "soft": Scaled gradients (reduce gradient magnitude by factor)
            - "selective": Only insulate specific layers
        gradient_scale: For soft mode, scale factor for VLM gradients (0.0-1.0)
        insulated_layers: For selective mode, regex patterns for layers to insulate
        preserve_vlm_statistics: If True, use running statistics from pre-training
        use_discrete_state: If True, encode state as discrete tokens (Pi0.5 style)
    """

    mode: Literal["full", "soft", "selective"] = "full"
    gradient_scale: float = 0.0
    insulated_layers: tuple[str, ...] = (".*llm.*", ".*img.*")
    preserve_vlm_statistics: bool = True
    use_discrete_state: bool = True


def apply_knowledge_insulation(
    tokens: at.Float[at.Array, "b s d"],
    config: KnowledgeInsulationConfig,
) -> at.Float[at.Array, "b s d"]:
    """Apply knowledge insulation to token embeddings.

    This function controls gradient flow through the VLM backbone.

    Args:
        tokens: Token embeddings from VLM [batch, seq, dim]
        config: Knowledge insulation configuration

    Returns:
        Tokens with appropriate gradient handling
    """
    if config.mode == "full":
        # Complete gradient isolation
        return jax.lax.stop_gradient(tokens)

    elif config.mode == "soft":
        # Scale gradients - allows some learning signal through
        # Uses custom_vjp to scale gradients during backward pass
        return _soft_insulation(tokens, config.gradient_scale)

    else:  # selective - handled at model level
        return tokens


@jax.custom_vjp
def _soft_insulation(
    tokens: at.Float[at.Array, "b s d"],
    scale: float,
) -> at.Float[at.Array, "b s d"]:
    """Soft insulation with scaled gradients."""
    return tokens


def _soft_insulation_fwd(tokens, scale):
    return tokens, scale


def _soft_insulation_bwd(scale, g):
    # Scale the gradients during backward pass
    return (g * scale, None)


_soft_insulation.defvjp(_soft_insulation_fwd, _soft_insulation_bwd)


class InsulatedEmbedding(nnx.Module):
    """Embedding layer with knowledge insulation support.

    Wraps an embedding table with optional gradient isolation.
    Used for VLM token embeddings during fine-tuning.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: KnowledgeInsulationConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize insulated embedding.

        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embeddings
            config: Knowledge insulation configuration
            rngs: Random number generators
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config

        # Initialize embedding table
        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (num_embeddings, embedding_dim))
            * 0.02
        )

    def __call__(self, indices: at.Int[at.Array, "..."]) -> at.Float[at.Array, "... d"]:
        """Look up embeddings with insulation.

        Args:
            indices: Token indices

        Returns:
            Embedded tokens with knowledge insulation applied
        """
        tokens = self.embedding.value[indices]
        return apply_knowledge_insulation(tokens, self.config)


class InsulatedLayerNorm(nnx.Module):
    """Layer normalization with optional statistics preservation.

    When preserve_vlm_statistics is True, uses fixed statistics from
    pre-training instead of batch statistics, preserving VLM behavior.
    """

    def __init__(
        self,
        dim: int,
        config: KnowledgeInsulationConfig,
        *,
        rngs: nnx.Rngs,
        eps: float = 1e-6,
    ):
        """Initialize insulated layer norm.

        Args:
            dim: Feature dimension
            config: Knowledge insulation configuration
            rngs: Random number generators
            eps: Epsilon for numerical stability
        """
        self.dim = dim
        self.config = config
        self.eps = eps

        self.scale = nnx.Param(jnp.ones(dim))
        self.bias = nnx.Param(jnp.zeros(dim))

        # Running statistics from pre-training (if preserved)
        if config.preserve_vlm_statistics:
            self.running_mean = nnx.Variable(jnp.zeros(dim))
            self.running_var = nnx.Variable(jnp.ones(dim))
        else:
            self.running_mean = None
            self.running_var = None

    def __call__(
        self,
        x: at.Float[at.Array, "... d"],
        *,
        use_running_stats: bool = False,
    ) -> at.Float[at.Array, "... d"]:
        """Apply layer normalization.

        Args:
            x: Input tensor
            use_running_stats: If True and preserve_vlm_statistics, use running stats

        Returns:
            Normalized tensor
        """
        if use_running_stats and self.running_mean is not None:
            mean = self.running_mean.value
            var = self.running_var.value
        else:
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        return x_norm * self.scale.value + self.bias.value


def create_insulated_model_filter(
    config: KnowledgeInsulationConfig,
) -> nnx.filterlib.Filter:
    """Create parameter filter for selective knowledge insulation.

    Returns a filter that matches parameters that should be insulated
    (i.e., not receive gradient updates during fine-tuning).

    Args:
        config: Knowledge insulation configuration

    Returns:
        NNX filter for insulated parameters
    """
    from fla.shared.nnx_utils import PathRegex

    if config.mode != "selective":
        return nnx.Nothing

    filters = [PathRegex(pattern) for pattern in config.insulated_layers]
    if len(filters) == 1:
        return filters[0]
    return nnx.Any(*filters)


def insulate_prefix_tokens(
    prefix_tokens: at.Float[at.Array, "b s d"],
    suffix_tokens: at.Float[at.Array, "b t d"],
    config: KnowledgeInsulationConfig,
) -> tuple[at.Float[at.Array, "b s d"], at.Float[at.Array, "b t d"]]:
    """Apply knowledge insulation to prefix (VLM) tokens.

    This is the main entry point for knowledge insulation during training.
    Prefix tokens (from VLM processing images/language) are insulated,
    while suffix tokens (action expert) receive full gradients.

    Args:
        prefix_tokens: VLM output tokens [batch, prefix_len, dim]
        suffix_tokens: Action expert tokens [batch, suffix_len, dim]
        config: Knowledge insulation configuration

    Returns:
        Tuple of (insulated_prefix, suffix) tokens
    """
    insulated_prefix = apply_knowledge_insulation(prefix_tokens, config)
    return insulated_prefix, suffix_tokens


class DiscreteStateEncoder(nnx.Module):
    """Encode continuous robot state as discrete tokens.

    Pi0.5 style: Instead of projecting state to continuous embeddings,
    discretize state values and use token embeddings. This maintains
    the discrete/continuous separation for knowledge insulation.
    """

    def __init__(
        self,
        state_dim: int,
        num_bins: int = 256,
        embedding_dim: int = 1024,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize discrete state encoder.

        Args:
            state_dim: Dimension of robot state
            num_bins: Number of discretization bins per dimension
            embedding_dim: Dimension of token embeddings
            rngs: Random number generators
        """
        self.state_dim = state_dim
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim

        # Embedding table for discretized state values
        # Each state dimension gets its own embedding range
        total_embeddings = state_dim * num_bins
        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (total_embeddings, embedding_dim))
            * 0.02
        )

        # Statistics for normalization (updated during training)
        self.state_min = nnx.Variable(jnp.full(state_dim, -1.0))
        self.state_max = nnx.Variable(jnp.full(state_dim, 1.0))

    def discretize(
        self,
        state: at.Float[at.Array, "b d"],
    ) -> at.Int[at.Array, "b d"]:
        """Discretize continuous state values.

        Args:
            state: Continuous state [batch, state_dim]

        Returns:
            Discretized state indices [batch, state_dim]
        """
        # Normalize to [0, 1]
        state_norm = (state - self.state_min.value) / (
            self.state_max.value - self.state_min.value + 1e-8
        )
        state_norm = jnp.clip(state_norm, 0.0, 1.0)

        # Convert to bin indices
        bin_indices = (state_norm * (self.num_bins - 1)).astype(jnp.int32)

        # Offset by dimension to use correct embedding range
        dim_offsets = jnp.arange(self.state_dim) * self.num_bins
        return bin_indices + dim_offsets

    def __call__(
        self,
        state: at.Float[at.Array, "b d"],
    ) -> at.Float[at.Array, "b d emb"]:
        """Encode state as discrete token embeddings.

        Args:
            state: Continuous state [batch, state_dim]

        Returns:
            Token embeddings [batch, state_dim, embedding_dim]
        """
        indices = self.discretize(state)
        return self.embedding.value[indices]

    def update_statistics(
        self,
        state: at.Float[at.Array, "b d"],
        momentum: float = 0.99,
    ) -> None:
        """Update running statistics for normalization.

        Args:
            state: Batch of states
            momentum: EMA momentum for statistics update
        """
        batch_min = jnp.min(state, axis=0)
        batch_max = jnp.max(state, axis=0)

        self.state_min.value = momentum * self.state_min.value + (1 - momentum) * batch_min
        self.state_max.value = momentum * self.state_max.value + (1 - momentum) * batch_max
