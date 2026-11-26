"""Paired bootstrap over items: how large an accuracy gap between two settings could be chance."""
from collections.abc import Sequence

import numpy as np


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """(mean of a - b, 2.5th, 97.5th percentile of its bootstrap means). a and b are per item, same order."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0:
        raise ValueError("no items to bootstrap")
    picks = np.random.default_rng(seed).integers(0, diff.size, size=(n_boot, diff.size))  # (n_boot, n)
    low, high = np.percentile(diff[picks].mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def diff_of_means_ci(a: Sequence[float], b: Sequence[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """(mean a - mean b, 2.5th, 97.5th percentile) when a and b come from different item sets: each is resampled alone."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("no items to bootstrap")
    rng = np.random.default_rng(seed)
    means_a = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    means_b = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    low, high = np.percentile(means_a - means_b, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(low), float(high)
