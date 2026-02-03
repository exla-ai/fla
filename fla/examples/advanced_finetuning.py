"""Advanced Fine-tuning Examples for FLA.

This file demonstrates the advanced fine-tuning features in FLA:
1. Knowledge Insulation - Prevent catastrophic forgetting
2. LoRA - Parameter-efficient fine-tuning
3. ReinFlow - RL fine-tuning for flow policies

Each example is self-contained and can be run independently.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx


# =============================================================================
# Example 1: Knowledge Insulation
# =============================================================================

def example_knowledge_insulation():
    """Demonstrate Knowledge Insulation for preventing catastrophic forgetting.

    Knowledge Insulation maintains separation between:
    - VLM discrete tokens (frozen, preserve pre-training knowledge)
    - Action expert continuous tokens (trainable, adapt to new tasks)
    """
    from fla.training import (
        KnowledgeInsulationConfig,
        apply_knowledge_insulation,
        insulate_prefix_tokens,
        DiscreteStateEncoder,
    )

    print("=" * 60)
    print("Example 1: Knowledge Insulation")
    print("=" * 60)

    # 1. Full gradient isolation (default, recommended)
    print("\n1. Full Mode - Complete gradient isolation:")
    config_full = KnowledgeInsulationConfig(mode="full")

    tokens = jnp.ones((2, 10, 64))
    insulated = apply_knowledge_insulation(tokens, config_full)

    # Verify gradients are stopped
    def loss_fn(x):
        return jnp.sum(apply_knowledge_insulation(x, config_full))

    grad = jax.grad(loss_fn)(tokens)
    print(f"   Gradient norm: {jnp.linalg.norm(grad):.4f} (should be 0.0)")

    # 2. Soft mode - scaled gradients
    print("\n2. Soft Mode - Scaled gradients:")
    config_soft = KnowledgeInsulationConfig(mode="soft", gradient_scale=0.1)

    def loss_fn_soft(x):
        return jnp.sum(apply_knowledge_insulation(x, config_soft))

    grad_soft = jax.grad(loss_fn_soft)(tokens)
    expected_norm = jnp.linalg.norm(jnp.ones_like(tokens) * 0.1)
    print(f"   Gradient norm: {jnp.linalg.norm(grad_soft):.4f}")
    print(f"   Expected norm: {expected_norm:.4f}")

    # 3. Prefix/suffix separation
    print("\n3. Prefix/Suffix Separation:")
    prefix = jnp.ones((2, 100, 64))  # VLM output
    suffix = jnp.ones((2, 50, 64))   # Action expert input

    insulated_prefix, suffix_out = insulate_prefix_tokens(
        prefix, suffix, config_full
    )

    def combined_loss(p, s):
        ip, so = insulate_prefix_tokens(p, s, config_full)
        return jnp.sum(ip) + jnp.sum(so)

    grad_p, grad_s = jax.grad(combined_loss, argnums=(0, 1))(prefix, suffix)
    print(f"   Prefix gradient norm: {jnp.linalg.norm(grad_p):.4f} (should be 0)")
    print(f"   Suffix gradient norm: {jnp.linalg.norm(grad_s):.4f} (should be non-zero)")

    # 4. Discrete state encoding
    print("\n4. Discrete State Encoding (Pi0.5 style):")
    rngs = nnx.Rngs(42)
    encoder = DiscreteStateEncoder(
        state_dim=7,
        num_bins=256,
        embedding_dim=64,
        rngs=rngs,
    )

    state = jax.random.uniform(jax.random.key(0), (4, 7), minval=-1, maxval=1)
    embeddings = encoder(state)
    print(f"   Input state shape: {state.shape}")
    print(f"   Output embedding shape: {embeddings.shape}")
    print(f"   Embedding per dimension: {embeddings.shape[1]}")

    print("\n✓ Knowledge Insulation example complete")


# =============================================================================
# Example 2: LoRA (Low-Rank Adaptation)
# =============================================================================

def example_lora():
    """Demonstrate LoRA for parameter-efficient fine-tuning.

    LoRA adds trainable low-rank matrices to frozen weights:
    W' = W + BA where B, A are low-rank matrices
    """
    from fla.training import (
        LoRAConfig,
        LoRALinear,
        count_lora_params,
        get_lora_gemma_variant,
    )

    print("\n" + "=" * 60)
    print("Example 2: LoRA (Low-Rank Adaptation)")
    print("=" * 60)

    # 1. Basic LoRA configuration
    print("\n1. LoRA Configuration:")
    config = LoRAConfig(
        rank=16,
        alpha=16.0,
        target_modules="all",
        apply_to_vlm=False,
        apply_to_action_expert=True,
        rslora=True,
    )
    print(f"   Rank: {config.rank}")
    print(f"   Alpha: {config.alpha}")
    print(f"   Target: {config.target_modules}")
    print(f"   rsLoRA: {config.rslora}")

    # 2. Create LoRA layer
    print("\n2. LoRA Linear Layer:")
    rngs = nnx.Rngs(42)
    layer = LoRALinear(
        in_features=1024,
        out_features=256,
        config=config,
        rngs=rngs,
    )

    print(f"   Base weight shape: {layer.weight.value.shape}")
    print(f"   LoRA A shape: {layer.lora_a.value.shape}")
    print(f"   LoRA B shape: {layer.lora_b.value.shape}")

    # 3. Parameter count comparison
    print("\n3. Parameter Efficiency:")
    base_params = 1024 * 256
    lora_params = 1024 * 16 + 16 * 256
    print(f"   Base layer params: {base_params:,}")
    print(f"   LoRA params: {lora_params:,}")
    print(f"   Reduction: {(1 - lora_params/base_params)*100:.1f}%")

    # 4. Forward pass
    print("\n4. Forward Pass:")
    x = jax.random.normal(jax.random.key(0), (2, 10, 1024))
    output = layer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    # 5. Initial output equals base (since lora_b is zero)
    print("\n5. Initial Output Test:")
    base_output = jnp.dot(x, layer.weight.value)
    diff = jnp.abs(output - base_output).max()
    print(f"   Max diff from base: {diff:.6f} (should be ~0)")

    # 6. After training (simulate)
    print("\n6. After Training (simulated):")
    layer.lora_b.value = jax.random.normal(jax.random.key(1), (16, 256)) * 0.01
    output_trained = layer(x)
    diff_trained = jnp.abs(output_trained - base_output).max()
    print(f"   Max diff from base: {diff_trained:.6f} (should be non-zero)")

    # 7. Merge LoRA weights
    print("\n7. Merge LoRA into Base:")
    output_before_merge = layer(x)
    layer.merge_lora()
    output_after_merge = layer(x)
    merge_diff = jnp.abs(output_before_merge - output_after_merge).max()
    print(f"   Max diff after merge: {merge_diff:.6f} (should be ~0)")
    print(f"   LoRA A max after merge: {jnp.abs(layer.lora_a.value).max():.6f}")
    print(f"   LoRA B max after merge: {jnp.abs(layer.lora_b.value).max():.6f}")

    # 8. Gemma variant selection
    print("\n8. Gemma Variant Selection:")
    vlm, expert = get_lora_gemma_variant("gemma_2b", config)
    print(f"   VLM variant: {vlm}")
    print(f"   Action expert variant: {expert}")

    print("\n✓ LoRA example complete")


# =============================================================================
# Example 3: ReinFlow (RL Fine-tuning)
# =============================================================================

def example_reinflow():
    """Demonstrate ReinFlow for RL fine-tuning of flow policies.

    ReinFlow supports:
    - REINFORCE: Basic policy gradient
    - PPO: Proximal Policy Optimization
    - DPO: Direct Preference Optimization
    """
    from fla.training import (
        ReinFlowConfig,
        ReinFlowTrainer,
        DPOTrainer,
        Trajectory,
        create_reinflow_trainer,
    )
    from fla.training.reinflow import (
        compute_gae,
        compute_returns,
        FlowPolicyLogProb,
        ValueFunction,
    )
    from fla.models.base import Observation

    print("\n" + "=" * 60)
    print("Example 3: ReinFlow (RL Fine-tuning)")
    print("=" * 60)

    # Mock model for demonstration
    class MockFlowModel(nnx.Module):
        def __init__(self):
            self.action_dim = 7
            self.action_horizon = 50
            self.max_token_len = 200
            self.param = nnx.Param(jnp.ones((7, 7)))

        def compute_loss(self, rng, obs, actions, train=False):
            return jnp.ones((actions.shape[0], self.action_horizon)) * 0.5

        def sample_actions(self, rng, obs, **kwargs):
            return jax.random.normal(rng, (obs.state.shape[0], self.action_horizon, self.action_dim))

    def create_mock_obs(batch_size=2):
        return Observation(
            images={"base": jnp.zeros((batch_size, 224, 224, 3))},
            image_masks={"base": jnp.ones((batch_size,), dtype=jnp.bool_)},
            state=jnp.zeros((batch_size, 7)),
        )

    # 1. REINFORCE configuration
    print("\n1. REINFORCE Configuration:")
    config_reinforce = ReinFlowConfig(
        algorithm="reinforce",
        learning_rate=1e-5,
        gamma=0.99,
        reward_baseline="mean",
    )
    print(f"   Algorithm: {config_reinforce.algorithm}")
    print(f"   Learning rate: {config_reinforce.learning_rate}")
    print(f"   Gamma: {config_reinforce.gamma}")
    print(f"   Baseline: {config_reinforce.reward_baseline}")

    # 2. PPO configuration
    print("\n2. PPO Configuration:")
    config_ppo = ReinFlowConfig(
        algorithm="ppo",
        clip_ratio=0.2,
        num_updates_per_rollout=4,
        target_kl=0.01,
        gae_lambda=0.95,
    )
    print(f"   Algorithm: {config_ppo.algorithm}")
    print(f"   Clip ratio: {config_ppo.clip_ratio}")
    print(f"   Updates per rollout: {config_ppo.num_updates_per_rollout}")
    print(f"   GAE lambda: {config_ppo.gae_lambda}")

    # 3. DPO configuration
    print("\n3. DPO Configuration:")
    config_dpo = ReinFlowConfig(algorithm="dpo")
    print(f"   Algorithm: {config_dpo.algorithm}")
    print(f"   (Offline RL from preference data)")

    # 4. Compute returns
    print("\n4. Return Computation:")
    rewards = [jnp.ones((4,)) for _ in range(5)]
    dones = [jnp.zeros((4,), dtype=jnp.bool_) for _ in range(5)]
    returns = compute_returns(rewards, dones, gamma=0.99)
    print(f"   Steps: {len(returns)}")
    print(f"   Return at t=0: {returns[0][0]:.4f}")
    print(f"   Return at t=4: {returns[4][0]:.4f}")

    # 5. GAE computation
    print("\n5. GAE Computation:")
    values = [jnp.ones((4,)) * 0.5 for _ in range(6)]  # T+1 values
    advantages, gae_returns = compute_gae(
        rewards, values, dones, gamma=0.99, gae_lambda=0.95
    )
    print(f"   Advantage at t=0: {advantages[0][0]:.4f}")
    print(f"   Advantage at t=4: {advantages[4][0]:.4f}")

    # 6. Create trainer
    print("\n6. Create Trainer:")
    model = MockFlowModel()
    trainer = ReinFlowTrainer(model, config_reinforce)
    print(f"   Trainer created for {config_reinforce.algorithm}")

    # 7. Process trajectory
    print("\n7. Process Trajectory:")
    trajectory = Trajectory(
        observations=[create_mock_obs(4) for _ in range(10)],
        actions=[jnp.zeros((4, 50, 7)) for _ in range(10)],
        rewards=[jnp.ones((4,)) * (i + 1) for i in range(10)],
        dones=[jnp.zeros((4,), dtype=jnp.bool_) for _ in range(10)],
    )
    processed = trainer.process_trajectory(trajectory)
    print(f"   Original rewards: {len(trajectory.rewards)}")
    print(f"   Computed advantages: {len(processed.advantages)}")
    print(f"   Computed returns: {len(processed.returns)}")

    # 8. Log probability computation
    print("\n8. Flow Policy Log Probability:")
    log_prob_fn = FlowPolicyLogProb(model)
    obs = create_mock_obs(2)
    actions = jnp.zeros((2, 50, 7))
    rng = jax.random.key(0)
    log_probs = log_prob_fn(rng, obs, actions)
    print(f"   Log prob shape: {log_probs.shape}")
    print(f"   Log prob values: {log_probs}")

    # 9. Value function
    print("\n9. Value Function (for baseline):")
    rngs = nnx.Rngs(42)
    value_fn = ValueFunction(feature_dim=64, hidden_dim=128, rngs=rngs)
    features = jax.random.normal(jax.random.key(0), (4, 64))
    values = value_fn(features)
    print(f"   Input shape: {features.shape}")
    print(f"   Value shape: {values.shape}")

    # 10. Factory function
    print("\n10. Factory Function:")
    trainer_reinforce = create_reinflow_trainer(model, config_reinforce)
    print(f"   REINFORCE: {type(trainer_reinforce).__name__}")

    trainer_ppo = create_reinflow_trainer(model, config_ppo)
    print(f"   PPO: {type(trainer_ppo).__name__}")

    reference = MockFlowModel()
    trainer_dpo = create_reinflow_trainer(model, config_dpo, reference_model=reference)
    print(f"   DPO: {type(trainer_dpo).__name__}")

    print("\n✓ ReinFlow example complete")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("FLA Advanced Fine-tuning Examples")
    print("=" * 60)
    print()

    # Run all examples
    example_knowledge_insulation()
    example_lora()
    example_reinflow()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
