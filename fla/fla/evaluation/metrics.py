"""Evaluation metrics for VLA models."""

import numpy as np
from typing import Sequence


def success_rate(rewards: Sequence[float], threshold: float = 0.95) -> float:
    """Compute success rate from episode rewards.

    Args:
        rewards: List of episode rewards.
        threshold: Minimum reward to count as success.

    Returns:
        Success rate in [0, 1].
    """
    if not rewards:
        return 0.0
    return sum(r >= threshold for r in rewards) / len(rewards)


def episode_return(rewards: Sequence[float]) -> tuple[float, float]:
    """Compute mean and std of episode returns.

    Args:
        rewards: List of episode rewards.

    Returns:
        Tuple of (mean, std).
    """
    if not rewards:
        return 0.0, 0.0
    return float(np.mean(rewards)), float(np.std(rewards))


def normalized_score(
    reward: float,
    random_score: float,
    expert_score: float,
) -> float:
    """Compute normalized score relative to random and expert.

    Args:
        reward: Achieved reward.
        random_score: Score achieved by random policy.
        expert_score: Score achieved by expert policy.

    Returns:
        Normalized score in [0, 100].
    """
    if expert_score == random_score:
        return 0.0
    return 100.0 * (reward - random_score) / (expert_score - random_score)


def aggregate_metrics(
    results: Sequence[dict],
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Aggregate metrics across multiple evaluation runs.

    Args:
        results: List of result dictionaries.
        metric_names: Metrics to aggregate (all if None).

    Returns:
        Dictionary with aggregated metrics.
    """
    if not results:
        return {}

    if metric_names is None:
        metric_names = list(results[0].keys())

    aggregated = {}
    for name in metric_names:
        values = [r[name] for r in results if name in r]
        if values:
            aggregated[f"{name}_mean"] = float(np.mean(values))
            aggregated[f"{name}_std"] = float(np.std(values))

    return aggregated
