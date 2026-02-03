"""FLA Evaluation Module.

Provides evaluation infrastructure for VLA models on standard benchmarks.

Supported benchmarks:
- ALOHA Sim: Transfer cube, insertion tasks
- LIBERO: 130+ manipulation tasks
"""

# Import lightweight metrics module directly
from fla.evaluation.metrics import success_rate, episode_return


def __getattr__(name):
    """Lazy import for heavy modules that depend on models."""
    if name == "Evaluator":
        from fla.evaluation.evaluator import Evaluator
        return Evaluator
    elif name == "EvalConfig":
        from fla.evaluation.evaluator import EvalConfig
        return EvalConfig
    elif name == "AlohaEvaluator":
        from fla.evaluation.aloha import AlohaEvaluator
        return AlohaEvaluator
    raise AttributeError(f"module 'fla.evaluation' has no attribute {name!r}")


__all__ = [
    "Evaluator",
    "EvalConfig",
    "AlohaEvaluator",
    "success_rate",
    "episode_return",
]
