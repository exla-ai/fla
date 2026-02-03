#!/usr/bin/env python3
"""Demo: Train a LoRA model on real robotics data.

This example shows FLA training modules working with real ALOHA data from HuggingFace.
Uses a small transformer model (not the full Pi0.5) to demonstrate the pipeline.

Usage:
    python examples/train_lora_demo.py

Requirements:
    - datasets (for HuggingFace data loading)
    - GPU with 4GB+ memory
"""

import logging
import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_aloha_data(num_samples: int = 100):
    """Load real ALOHA simulation data from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading ALOHA Transfer Cube dataset from HuggingFace...")
    dataset = load_dataset(
        "lerobot/aloha_sim_transfer_cube_human",
        split=f"train[:{num_samples}]"
    )

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Features: {list(dataset.features.keys())}")

    # Extract states and actions
    states = jnp.array([d['observation.state'] for d in dataset])
    actions = jnp.array([d['action'] for d in dataset])

    return states, actions, dataset


class SimpleActionPredictor(nnx.Module):
    """Simple MLP for action prediction - demonstrates LoRA integration."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        from fla.training import LoRAConfig, LoRALinear

        # LoRA configuration
        lora_config = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)

        # Network with LoRA layers
        self.layer1 = LoRALinear(state_dim, hidden_dim, lora_config, rngs=rngs)
        self.layer2 = LoRALinear(hidden_dim, hidden_dim, lora_config, rngs=rngs)
        self.layer3 = LoRALinear(hidden_dim, action_dim, lora_config, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        x = nnx.relu(self.layer1(state))
        x = nnx.relu(self.layer2(x))
        return self.layer3(x)


def compute_loss(model, states, actions):
    """MSE loss for action prediction."""
    predicted = model(states)
    return jnp.mean((predicted - actions) ** 2)


def train_step(model, optimizer, opt_state, states, actions):
    """Single training step."""
    graphdef, state = nnx.split(model)

    def loss_fn(params):
        model_with_params = nnx.merge(graphdef, params)
        return compute_loss(model_with_params, states, actions)

    loss, grads = jax.value_and_grad(loss_fn)(state)
    updates, opt_state = optimizer.update(grads, opt_state, state)
    state = optax.apply_updates(state, updates)
    model = nnx.merge(graphdef, state)

    return model, opt_state, loss


def main():
    logger.info("=" * 60)
    logger.info("FLA Demo: LoRA Training on ALOHA Data")
    logger.info("=" * 60)

    # Check device
    devices = jax.devices()
    logger.info(f"JAX devices: {devices}")

    # Load real data
    try:
        states, actions, dataset = load_aloha_data(num_samples=200)
        logger.info(f"States shape: {states.shape}")
        logger.info(f"Actions shape: {actions.shape}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Make sure 'datasets' is installed: pip install datasets")
        return

    # Model dimensions
    state_dim = states.shape[-1]  # 14 for ALOHA
    action_dim = actions.shape[-1]  # 14 for ALOHA
    hidden_dim = 256

    # Create model
    logger.info(f"\nCreating LoRA model (state_dim={state_dim}, action_dim={action_dim})...")
    rng = jax.random.key(42)
    model = SimpleActionPredictor(state_dim, action_dim, hidden_dim, rngs=nnx.Rngs(rng))

    # Count parameters
    from fla.training import count_lora_params
    _, state = nnx.split(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    lora_params, _ = count_lora_params(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"LoRA parameters: {lora_params:,} ({100*lora_params/total_params:.1f}%)")

    # Setup optimizer
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    _, state = nnx.split(model)
    opt_state = optimizer.init(state)

    logger.info(f"Optimizer: Adam, lr={learning_rate}")

    # Training loop
    num_epochs = 10
    batch_size = 32
    num_batches = len(states) // batch_size

    logger.info(f"\nTraining for {num_epochs} epochs...")
    logger.info("-" * 40)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.perf_counter()

        # Shuffle data
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, len(states))
        states_shuffled = states[perm]
        actions_shuffled = actions[perm]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_states = states_shuffled[start_idx:end_idx]
            batch_actions = actions_shuffled[start_idx:end_idx]

            model, opt_state, loss = train_step(
                model, optimizer, opt_state, batch_states, batch_actions
            )
            epoch_loss += float(loss)

        avg_loss = epoch_loss / num_batches
        epoch_time = time.perf_counter() - epoch_start
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    logger.info("-" * 40)
    logger.info("Training complete!")

    # Test inference
    logger.info("\nTesting inference...")
    test_states = states[:5]
    predicted_actions = model(test_states)

    logger.info(f"Input states shape: {test_states.shape}")
    logger.info(f"Predicted actions shape: {predicted_actions.shape}")
    logger.info(f"Sample prediction: {predicted_actions[0][:5]}...")
    logger.info(f"Actual action: {actions[0][:5]}...")

    # Save LoRA adapter
    from fla.training import LoRAConfig, save_lora_adapter
    logger.info("\nSaving LoRA adapter...")
    lora_config = LoRAConfig(rank=8, alpha=16.0)
    save_lora_adapter(model, "/tmp/aloha_lora_adapter.pkl", lora_config)
    logger.info("Saved to /tmp/aloha_lora_adapter.pkl")

    logger.info("\n" + "=" * 60)
    logger.info("FLA Demo Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
