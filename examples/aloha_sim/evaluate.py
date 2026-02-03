#!/usr/bin/env python3
"""Evaluate ALOHA sim policy with multiple episodes and compute success rate."""

import dataclasses
import logging
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import tyro

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/eval_results")
    task: str = "gym_aloha/AlohaTransferCube-v0"
    prompt: str = "Pick up the cube and transfer it to a new location"
    num_episodes: int = 50
    seed: int = 0
    action_horizon: int = 10
    host: str = "0.0.0.0"
    port: int = 8000
    max_steps: int = 400


class EvaluationSubscriber:
    """Subscriber to track episode results."""

    def __init__(self):
        self.results = []
        self.current_episode = {"rewards": [], "steps": 0}

    def on_step(self, observation, action, reward, done, info):
        self.current_episode["rewards"].append(reward)
        self.current_episode["steps"] += 1

    def on_episode_end(self):
        max_reward = max(self.current_episode["rewards"]) if self.current_episode["rewards"] else 0
        success = max_reward >= 0.95
        self.results.append({
            "success": success,
            "max_reward": max_reward,
            "steps": self.current_episode["steps"],
        })
        self.current_episode = {"rewards": [], "steps": 0}


def run_episode(env: _env.AlohaSimEnvironment, policy, prompt: str, max_steps: int = 400) -> dict:
    """Run a single episode and return results."""
    env.reset()

    max_reward = 0.0
    step = 0

    while step < max_steps and not env.is_episode_complete():
        obs = env.get_observation()
        obs["prompt"] = prompt  # Add prompt to observation

        # Get action from policy
        try:
            result = policy.infer(obs)
            actions = np.asarray(result["actions"])
            # Actions shape is (action_horizon, action_dim) = (50, 14) for ALOHA
            # Take first action
            if actions.ndim == 2:
                action = actions[0]  # Shape: (14,)
            else:
                action = actions  # Already 1D
            action = {"actions": action}
        except Exception as e:
            logging.error(f"Policy inference failed: {e}")
            import traceback
            traceback.print_exc()
            break

        env.apply_action(action)
        max_reward = max(max_reward, env._episode_reward)
        step += 1

    success = max_reward >= 0.95
    return {
        "success": success,
        "max_reward": float(max_reward),
        "steps": step,
    }


def main(args: Args) -> None:
    logging.info(f"Starting evaluation: {args.num_episodes} episodes on {args.task}")
    logging.info(f"Connecting to policy server at {args.host}:{args.port}")

    # Create environment
    env = _env.AlohaSimEnvironment(
        task=args.task,
        seed=args.seed,
    )

    # Create policy client
    policy = action_chunk_broker.ActionChunkBroker(
        policy=_websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        ),
        action_horizon=args.action_horizon,
    )

    # Run episodes
    results = []
    for ep in range(args.num_episodes):
        env._rng = np.random.default_rng(args.seed + ep)
        result = run_episode(env, policy, args.prompt, args.max_steps)
        results.append(result)

        status = "✓" if result["success"] else "✗"
        logging.info(
            f"Episode {ep+1:3d}/{args.num_episodes}: {status} "
            f"reward={result['max_reward']:.3f} steps={result['steps']}"
        )

        # Print running stats every 10 episodes
        if (ep + 1) % 10 == 0:
            successes = [r["success"] for r in results]
            running_sr = np.mean(successes) * 100
            logging.info(f"  Running success rate: {running_sr:.1f}%")

    # Compute final statistics
    successes = [r["success"] for r in results]
    rewards = [r["max_reward"] for r in results]

    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Print summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Task: {args.task}")
    logging.info(f"Episodes: {args.num_episodes}")
    logging.info(f"Success Rate: {success_rate:.1f}%")
    logging.info(f"Avg Reward: {avg_reward:.3f} ± {std_reward:.3f}")
    logging.info("=" * 60)

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": args.task,
        "num_episodes": args.num_episodes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "timestamp": datetime.now().isoformat(),
        "episodes": results,
    }

    output_path = args.out_dir / f"eval_{args.task.replace('/', '_')}_{args.num_episodes}ep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Results saved to {output_path}")

    # Return success rate for CI/CD
    return success_rate


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    tyro.cli(main)
