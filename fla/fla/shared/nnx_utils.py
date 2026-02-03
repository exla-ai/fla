"""NNX utilities for parameter filtering and module operations."""

import re
from typing import Any, Callable

import flax.nnx as nnx
import jax


class PathRegex:
    """Filter NNX state by regex pattern on parameter paths.

    This enables selective freezing of parameters during training.
    For example, to freeze all PaliGemma parameters:
        filter = PathRegex(".*llm.*")

    Args:
        pattern: Regex pattern to match against parameter paths.
    """

    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def __call__(self, path: tuple[str, ...], value: Any) -> bool:
        """Check if the path matches the pattern."""
        path_str = "/".join(str(p) for p in path)
        return bool(self.pattern.match(path_str))


def module_jit(
    module: nnx.Module,
    method: str = "__call__",
    **jit_kwargs,
) -> Callable:
    """JIT compile a module method while preserving state.

    This is useful for inference where you want to compile the forward pass
    but keep the module state accessible.

    Args:
        module: The NNX module.
        method: The method name to JIT compile.
        **jit_kwargs: Additional arguments passed to jax.jit.

    Returns:
        A JIT-compiled function that takes inputs and returns outputs.
    """
    graphdef, state = nnx.split(module)

    @jax.jit(**jit_kwargs)
    def jitted_fn(state, *args, **kwargs):
        model = nnx.merge(graphdef, state)
        method_fn = getattr(model, method)
        return method_fn(*args, **kwargs)

    def wrapper(*args, **kwargs):
        nonlocal state
        result = jitted_fn(state, *args, **kwargs)
        return result

    return wrapper


def state_map(
    state: nnx.State,
    fn: Callable[[Any], Any],
    filter_fn: nnx.filterlib.Filter | None = None,
) -> nnx.State:
    """Apply a function to filtered state leaves.

    Args:
        state: The NNX state.
        fn: Function to apply to each leaf.
        filter_fn: Optional filter to select which leaves to transform.

    Returns:
        New state with transformed leaves.
    """
    if filter_fn is None:
        return jax.tree_util.tree_map(fn, state)

    def conditional_fn(path, value):
        if filter_fn(path, value):
            return fn(value)
        return value

    flat_state = state.to_pure_dict()
    transformed = jax.tree_util.tree_map_with_path(conditional_fn, flat_state)
    state.replace_by_pure_dict(transformed)
    return state


def count_params(state: nnx.State) -> int:
    """Count total number of parameters in state."""
    return sum(x.size for x in jax.tree_util.tree_leaves(state))


def freeze_params(
    state: nnx.State,
    filter_fn: nnx.filterlib.Filter,
) -> tuple[nnx.State, nnx.State]:
    """Split state into frozen and trainable parameters.

    Args:
        state: The full model state.
        filter_fn: Filter that returns True for parameters to freeze.

    Returns:
        Tuple of (trainable_state, frozen_state).
    """
    frozen, trainable = nnx.split(state, filter_fn, ...)
    return trainable, frozen
