"""Normalization statistics computation and application."""

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class NormStats:
    """Normalization statistics for a single array."""

    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None  # 1st percentile for quantile norm
    q99: Optional[np.ndarray] = None  # 99th percentile for quantile norm

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        if self.q01 is not None:
            result["q01"] = self.q01.tolist()
        if self.q99 is not None:
            result["q99"] = self.q99.tolist()
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        """Create from dictionary."""
        return cls(
            mean=np.array(d["mean"]),
            std=np.array(d["std"]),
            q01=np.array(d["q01"]) if "q01" in d else None,
            q99=np.array(d["q99"]) if "q99" in d else None,
        )


class RunningStats:
    """Online computation of mean, std, and quantiles."""

    def __init__(self, shape: tuple[int, ...]):
        self.n = 0
        self.mean = np.zeros(shape)
        self.M2 = np.zeros(shape)
        self.values = []  # For quantile computation

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new sample (Welford's algorithm)."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.values.append(x.copy())

    @property
    def variance(self) -> np.ndarray:
        """Compute variance."""
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        """Compute standard deviation."""
        return np.sqrt(self.variance)

    def get_quantiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1st and 99th percentiles."""
        if not self.values:
            return self.mean, self.mean
        all_values = np.stack(self.values, axis=0)
        q01 = np.percentile(all_values, 1, axis=0)
        q99 = np.percentile(all_values, 99, axis=0)
        return q01, q99

    def get_stats(self, *, include_quantiles: bool = True) -> NormStats:
        """Get normalization statistics."""
        q01, q99 = None, None
        if include_quantiles:
            q01, q99 = self.get_quantiles()
        return NormStats(
            mean=self.mean.copy(),
            std=self.std.copy(),
            q01=q01,
            q99=q99,
        )


def compute_norm_stats(
    data_iterator,
    keys: list[str],
    *,
    max_samples: int = 10000,
    include_quantiles: bool = True,
) -> dict[str, NormStats]:
    """Compute normalization statistics from a data iterator.

    Args:
        data_iterator: Iterator yielding dictionaries with arrays.
        keys: Keys to compute statistics for (e.g., ["state", "actions"]).
        max_samples: Maximum number of samples to use.
        include_quantiles: Whether to compute quantile statistics.

    Returns:
        Dictionary mapping keys to NormStats.
    """
    running_stats = {}

    for i, sample in enumerate(data_iterator):
        if i >= max_samples:
            break

        for key in keys:
            if key not in sample:
                continue
            value = np.asarray(sample[key])
            if key not in running_stats:
                running_stats[key] = RunningStats(value.shape)
            running_stats[key].update(value)

    return {
        key: stats.get_stats(include_quantiles=include_quantiles)
        for key, stats in running_stats.items()
    }


def normalize(x: np.ndarray, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    """Normalize an array using precomputed statistics."""
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile stats required but not provided")
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
    return (x - stats.mean) / (stats.std + 1e-6)


def unnormalize(x: np.ndarray, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    """Unnormalize an array using precomputed statistics."""
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile stats required but not provided")
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
    return x * (stats.std + 1e-6) + stats.mean
