"""Type annotations for JAX arrays.

Provides type hints for JAX arrays with shape and dtype information.
Uses jaxtyping for runtime type checking when enabled.
"""

from collections.abc import Mapping
from typing import Any, TypeAlias, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray, jaxtyped
    from beartype import beartype

    def typecheck(fn):
        """Decorator to enable runtime type checking."""
        return jaxtyped(typechecker=beartype)(fn)

    _TYPECHECKING_ENABLED = True
except ImportError:
    # Fallback when jaxtyping is not available
    # Create subscriptable dummy types that just return Any
    class _SubscriptableAny:
        """A type that can be subscripted but returns Any."""
        def __class_getitem__(cls, item):
            return Any
        def __getitem__(self, item):
            return Any

    Array = jax.Array
    Float = _SubscriptableAny
    Int = _SubscriptableAny
    Bool = _SubscriptableAny
    PRNGKeyArray = Any

    def typecheck(fn):
        return fn

    _TYPECHECKING_ENABLED = False


# Type aliases
KeyArrayLike: TypeAlias = Union[jax.Array, "PRNGKeyArray"]
Params: TypeAlias = dict[str, Any]
PyTree: TypeAlias = Union[dict, list, tuple, Any]

T = TypeVar("T")


def disable_typechecking():
    """Context manager to disable runtime type checking."""
    class NoTypecheck:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoTypecheck()


def check_pytree_equality(
    expected: PyTree,
    got: PyTree,
    *,
    check_shapes: bool = True,
    check_dtypes: bool = True,
) -> None:
    """Check that two PyTrees have the same structure and optionally shapes/dtypes."""
    expected_flat = jax.tree_util.tree_leaves_with_path(expected)
    got_flat = jax.tree_util.tree_leaves_with_path(got)

    if len(expected_flat) != len(got_flat):
        raise ValueError(
            f"PyTree structure mismatch: expected {len(expected_flat)} leaves, "
            f"got {len(got_flat)} leaves"
        )

    for (exp_path, exp_val), (got_path, got_val) in zip(expected_flat, got_flat):
        if check_shapes:
            exp_shape = getattr(exp_val, "shape", None)
            got_shape = getattr(got_val, "shape", None)
            if exp_shape != got_shape:
                raise ValueError(
                    f"Shape mismatch at {exp_path}: expected {exp_shape}, got {got_shape}"
                )
        if check_dtypes:
            exp_dtype = getattr(exp_val, "dtype", None)
            got_dtype = getattr(got_val, "dtype", None)
            if exp_dtype != got_dtype:
                raise ValueError(
                    f"Dtype mismatch at {exp_path}: expected {exp_dtype}, got {got_dtype}"
                )
