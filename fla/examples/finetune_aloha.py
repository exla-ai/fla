#!/usr/bin/env python3
"""Fine-tune Pi0.5 on ALOHA simulation data.

This example demonstrates real fine-tuning on the ALOHA Transfer Cube task
using the FLA library with frozen backbone training.

Usage:
    PYTHONPATH=/lambda/nfs/arizona/pi-openpi/src:$PYTHONPATH python examples/finetune_aloha.py

Requirements:
    - openpi (parent repo)
    - lerobot dataset
    - GPU with 80GB+ memory (A100-80GB or H100)
      - Model: ~7GB (3.3B params in bf16)
      - Gradients: ~7GB
      - Optimizer state: ~3GB (for 430M trainable params)
      - Activations: ~20GB+ (depends on batch size)

    For GPUs with less memory, consider:
    - Using LoRA instead (see train_lora_demo.py)
    - Using gradient checkpointing
    - Using model parallelism across multiple GPUs
"""

import logging
import os
import sys
import time

# Add openpi to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

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
    """Load ALOHA simulation data from HuggingFace datasets.

    Returns real training data from the ALOHA Transfer Cube dataset.
    """
    from datasets import load_dataset

    logger.info("Loading ALOHA Transfer Cube dataset from HuggingFace...")
    dataset = load_dataset(
        "lerobot/aloha_sim_transfer_cube_human",
        split=f"train[:{num_samples}]"
    )

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Features: {list(dataset.features.keys())}")

    # Convert to list of samples
    samples = [dataset[i] for i in range(len(dataset))]

    return samples, dataset


def prepare_batch(samples, batch_size: int = 4, action_dim: int = 14, action_horizon: int = 50):
    """Prepare a batch of data for training.

    Note: This dataset has state and actions but no images.
    We create placeholder images for the model.
    """
    from fla.models.base import Observation
    import numpy as np

    batch_samples = samples[:batch_size]

    # Create placeholder images (model expects images but dataset doesn't have them)
    # In real use, you would load actual camera images
    images = {
        "base_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
        "left_wrist_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
        "right_wrist_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
    }
    image_masks = {
        "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
    }

    # Stack states from real data
    states = []
    for s in batch_samples:
        state = s.get('observation.state', [0.0] * action_dim)
        state = jnp.array(state, dtype=jnp.float32)
        # Pad/truncate to action_dim
        if len(state) < action_dim:
            state = jnp.pad(state, (0, action_dim - len(state)))
        elif len(state) > action_dim:
            state = state[:action_dim]
        states.append(state)
    state = jnp.stack(states)

    # Stack actions from real data
    actions = []
    for s in batch_samples:
        action = s.get('action', [0.0] * action_dim)
        action = jnp.array(action, dtype=jnp.float32)
        # Pad/truncate to action_dim
        if len(action) < action_dim:
            action = jnp.pad(action, (0, action_dim - len(action)))
        elif len(action) > action_dim:
            action = action[:action_dim]
        # Repeat to create action sequence (action_horizon, action_dim)
        action_seq = jnp.tile(action[None, :], (action_horizon, 1))
        actions.append(action_seq)
    actions = jnp.stack(actions)

    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )

    return obs, actions


def is_action_expert_param(path: tuple) -> bool:
    """Check if parameter path belongs to action expert (trainable)."""
    path_str = "/".join(str(k) for k in path)
    # Action expert params have "_1" in path (llm_1)
    return "llm" in path_str and "_1" in path_str


def train_step(model, optimizer, opt_state, rng, obs, actions, trainable_mask):
    """Single training step with frozen backbone.

    Only updates action expert parameters (~300M), not VLM (~2.7B).
    Uses stop_gradient to avoid computing gradients for frozen params.
    """
    graphdef, state = nnx.split(model)

    def loss_fn(params):
        # Apply stop_gradient to frozen params to save memory
        def maybe_stop_grad(mask, param):
            if mask:
                return param  # Trainable - keep gradients
            else:
                return jax.lax.stop_gradient(param)  # Frozen - no gradients

        stopped_params = jax.tree_util.tree_map(maybe_stop_grad, trainable_mask, params)
        model_with_params = nnx.merge(graphdef, stopped_params)
        loss = model_with_params.compute_loss(rng, obs, actions, train=True)
        return jnp.mean(loss)

    # Compute gradients (only for non-stopped params)
    loss, grads = jax.value_and_grad(loss_fn)(state)

    # Apply optimizer with masked gradients
    updates, opt_state = optimizer.update(grads, opt_state, state)
    state = optax.apply_updates(state, updates)

    # Update model
    model = nnx.merge(graphdef, state)

    return model, opt_state, loss


def main():
    logger.info("=" * 60)
    logger.info("FLA Fine-tuning: ALOHA Transfer Cube")
    logger.info("=" * 60)

    # Check GPU
    devices = jax.devices()
    logger.info(f"JAX devices: {devices}")

    if not any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        logger.warning("No GPU detected! Training will be slow.")

    # Load model
    from fla.models import Pi05Config, Pi05Model

    logger.info("Creating Pi0.5 model with frozen backbone...")
    config = Pi05Config(
        action_dim=14,  # ALOHA bimanual
        action_horizon=50,
        freeze_vision_backbone=True,  # Key for memory efficiency
    )

    rng = jax.random.key(42)
    model = config.create(rng)

    # Count parameters
    _, state = nnx.split(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    logger.info(f"Total parameters: {total_params:,}")

    # Load real data
    try:
        samples, dataset = load_aloha_data(num_samples=32)
        logger.info(f"Loaded {len(samples)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Make sure lerobot is installed: pip install lerobot")
        return

    # Prepare batch (small batch to fit in memory)
    batch_size = 1
    try:
        obs, actions = prepare_batch(samples, batch_size=batch_size)
        logger.info(f"Batch prepared: {batch_size} samples")
        logger.info(f"  Images: {list(obs.images.keys())}")
        logger.info(f"  State shape: {obs.state.shape}")
        logger.info(f"  Actions shape: {actions.shape}")
    except Exception as e:
        logger.error(f"Failed to prepare batch: {e}")
        import traceback
        traceback.print_exc()
        return

    # Setup optimizer with masking for memory efficiency
    learning_rate = 2.5e-5

    # Create mask: True for trainable params, False for frozen
    _, state = nnx.split(model)

    def create_mask(path, leaf):
        return is_action_expert_param(path)

    trainable_mask = jax.tree_util.tree_map_with_path(create_mask, state)

    # Count trainable vs frozen params
    flat_state = jax.tree_util.tree_leaves_with_path(state)
    flat_mask = jax.tree_util.tree_leaves(trainable_mask)
    trainable_params = sum(
        leaf.size if hasattr(leaf, 'size') else 0
        for (path, leaf), m in zip(flat_state, flat_mask)
        if m
    )
    frozen_params = total_params - trainable_params
    logger.info(f"Trainable parameters: {trainable_params:,} (action expert)")
    logger.info(f"Frozen parameters: {frozen_params:,} (VLM backbone)")

    # Use SGD (no optimizer state) with masking for memory efficiency
    # SGD doesn't store momentum so it uses much less memory than Adam
    base_optimizer = optax.sgd(learning_rate)
    optimizer = optax.masked(base_optimizer, trainable_mask)

    opt_state = optimizer.init(state)

    logger.info(f"Optimizer: SGD (masked), lr={learning_rate}")

    # Training loop
    num_steps = 5  # Reduced for demo
    logger.info(f"Starting training for {num_steps} steps...")
    logger.info("-" * 40)

    for step in range(num_steps):
        rng, step_rng = jax.random.split(rng)

        start = time.perf_counter()
        model, opt_state, loss = train_step(
            model, optimizer, opt_state, step_rng, obs, actions, trainable_mask
        )
        jax.block_until_ready(loss)
        step_time = time.perf_counter() - start

        logger.info(f"Step {step+1}/{num_steps} | Loss: {float(loss):.4f} | Time: {step_time:.2f}s")

    logger.info("-" * 40)
    logger.info("Training complete!")
    logger.info(f"Final loss: {float(loss):.4f}")

    # Test inference
    logger.info("\nTesting inference...")
    rng, infer_rng = jax.random.split(rng)

    start = time.perf_counter()
    predicted_actions = model.sample_actions(infer_rng, obs, num_steps=10)
    jax.block_until_ready(predicted_actions)
    infer_time = time.perf_counter() - start

    logger.info(f"Predicted actions shape: {predicted_actions.shape}")
    logger.info(f"Inference time: {infer_time:.3f}s")

    logger.info("\n" + "=" * 60)
    logger.info("Fine-tuning example complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
