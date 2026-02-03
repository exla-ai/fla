#!/usr/bin/env python3
"""FLA Evaluation Script.

Evaluate trained models on benchmark tasks.

Usage:
    # Evaluate on ALOHA Transfer Cube
    fla-eval --checkpoint ./checkpoints/my_model/30000 --task transfer_cube

    # Evaluate on ALOHA Insertion
    fla-eval --checkpoint ./checkpoints/my_model/30000 --task insertion

    # Run multiple episodes
    fla-eval --checkpoint ./checkpoints/my_model/30000 --num-episodes 100
"""

import dataclasses
import json
import logging
import pathlib

import tyro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Evaluation arguments."""

    checkpoint: str
    """Path to model checkpoint."""

    task: str = "transfer_cube"
    """Task to evaluate: transfer_cube, insertion"""

    num_episodes: int = 50
    """Number of evaluation episodes."""

    max_steps: int = 400
    """Maximum steps per episode."""

    prompt: str | None = None
    """Override task prompt."""

    output_dir: str = "./eval_results"
    """Output directory for results."""

    seed: int = 0
    """Random seed."""

    render: bool = False
    """Render episodes (slower)."""

    save_videos: bool = False
    """Save episode videos."""


# Task configurations
TASKS = {
    "transfer_cube": {
        "env_id": "gym_aloha/AlohaTransferCube-v0",
        "prompt": "Pick up the cube and transfer it to a new location",
    },
    "insertion": {
        "env_id": "gym_aloha/AlohaInsertion-v0",
        "prompt": "Insert the peg into the socket",
    },
}


def main(args: Args | None = None) -> None:
    """Main evaluation entry point."""
    if args is None:
        args = tyro.cli(Args)

    import jax

    # Get task configuration
    if args.task not in TASKS:
        raise ValueError(f"Unknown task: {args.task}. Available: {list(TASKS.keys())}")

    task_config = TASKS[args.task]
    env_id = task_config["env_id"]
    prompt = args.prompt or task_config["prompt"]

    logger.info(f"Task: {args.task}")
    logger.info(f"Environment: {env_id}")
    logger.info(f"Prompt: {prompt}")

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")

    from fla.models import Pi05Config, Pi05Model

    model = Pi05Model.from_pretrained(
        args.checkpoint,
        freeze_vision_backbone=True,
    )

    # Create evaluator
    from fla.evaluation import AlohaEvaluator, EvalConfig

    eval_config = EvalConfig(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render,
        save_videos=args.save_videos,
        output_dir=args.output_dir,
    )

    evaluator = AlohaEvaluator(
        model,
        task=env_id,
        config=eval_config,
    )

    # Run evaluation
    logger.info(f"Running {args.num_episodes} episodes...")
    result = evaluator.evaluate(prompt)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {result.num_episodes}")
    print(f"Success Rate: {result.success_rate:.1%}")
    print(f"Mean Reward: {result.mean_reward:.3f} +/- {result.std_reward:.3f}")
    print(f"Mean Steps: {result.mean_steps:.1f}")
    print("=" * 60)

    # Save results
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"eval_{args.task}_{args.num_episodes}ep.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "task": args.task,
                "checkpoint": args.checkpoint,
                "prompt": prompt,
                **result.to_dict(),
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
