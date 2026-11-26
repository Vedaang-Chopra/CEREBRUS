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
