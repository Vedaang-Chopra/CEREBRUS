"""Reading an iso-cost curve at a target cost (DEC3 S14): linear interpolation, never extrapolation."""
import numpy as np


def interpolation_weights(costs: np.ndarray, target: float) -> np.ndarray | None:
    """Weights over the curve points (sum 1) whose weighted mean cost is `target`; None if the curve does not reach it.

    The two points next to `target` in cost order get the weights of a straight line between them. Points with the same
    cost share their weight equally. The same weights applied to per-item scores give per-item interpolated scores.
    """
    costs = np.round(np.asarray(costs, dtype=float), 12)
    unique, group, size = np.unique(costs, return_inverse=True, return_counts=True)
    if target < unique[0] - 1e-9 or target > unique[-1] + 1e-9:
        return None
    target = float(np.clip(target, unique[0], unique[-1]))
    hi = int(np.searchsorted(unique, target))
    at_group = np.zeros(len(unique))
    if unique[hi] == target or hi == 0:
        at_group[hi] = 1.0
    else:
        share = (target - unique[hi - 1]) / (unique[hi] - unique[hi - 1])
        at_group[hi - 1], at_group[hi] = 1.0 - share, share
    return at_group[group] / size[group]
