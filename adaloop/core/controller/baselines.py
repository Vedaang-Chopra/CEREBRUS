"""Fixed-setting baselines over grid cells (items x cells arrays)."""
import numpy as np

from adaloop.core.controller.curves import interpolation_weights


def best_fixed_cell(select_values: np.ndarray, select_costs: np.ndarray, report_costs: np.ndarray,
                    cap: float) -> int | None:
    """The one cell for every item: most accurate on the selection items among the cells that fit the cap.

    "Fits" means mean FLOPs on the REPORT items <= `cap` (cost uses no labels, and the cap then binds the baseline on
    the same items as the oracle). Accuracy comes from the selection items only. Ties: cheaper, then first cell.
    None if no cell fits.
    """
    assert select_values.shape == select_costs.shape and select_values.shape[1] == report_costs.shape[1]
    fits = np.flatnonzero(report_costs.mean(axis=0) <= cap * (1 + 1e-12))
    if fits.size == 0:
        return None
    accuracy, cost = select_values.mean(axis=0), select_costs.mean(axis=0)
    return int(min(fits, key=lambda j: (-round(float(accuracy[j]), 12), float(cost[j]), j)))


def upper_hull(accuracy: np.ndarray, costs: np.ndarray) -> list[int]:
    """Cells on the rising part of the upper concave hull of (cost, accuracy), cheapest first.

    It runs from the best cell at the lowest cost to the (cheapest) most accurate cell. Past that cell the hull would
    only fall, so it is cut there. Equal costs: the more accurate cell, then the first.
    """
    order = sorted(range(len(costs)), key=lambda j: (float(costs[j]), -float(accuracy[j]), j))
    hull: list[int] = []
    for j in order:
        if hull and costs[j] == costs[hull[-1]]:
            continue
        # drop the last point while it lies on or under the line from the one before it to the new point
        while len(hull) > 1:
            a, b = hull[-2], hull[-1]
            if (accuracy[b] - accuracy[a]) * (costs[j] - costs[a]) > (accuracy[j] - accuracy[a]) * (costs[b] - costs[a]):
                break
            hull.pop()
        hull.append(j)
    top = max(range(len(hull)), key=lambda i: (round(float(accuracy[hull[i]]), 12), -i))
    return hull[: top + 1]


def hull_mix(select_values: np.ndarray, report_costs: np.ndarray, target: float) -> np.ndarray | None:
    """Weights over the cells (sum 1) of the best fixed-setting mix at mean cost `target`; None outside the cost range.

    The mix gives each item one of two neighbouring hull cells at random; the weights are the probabilities, so
    `values @ weights` is each item's expected score (nothing is drawn). The hull is built like `best_fixed_cell`
    chooses: accuracy from the selection items, cost from the REPORT items. No report label is read.
    Range: cheapest to most expensive cell. Above the most accurate (selection) cell the weights stay on that cell, so
    the mix costs LESS than `target`: spending more would only buy a worse or equal cell.
    """
    assert select_values.shape[1] == report_costs.shape[1]
    cost = report_costs.mean(axis=0)
    if target < cost.min() * (1 - 1e-12) or target > cost.max() * (1 + 1e-12):
        return None
    hull = upper_hull(select_values.mean(axis=0), cost)
    if target < cost[hull[0]] * (1 - 1e-12):
        return None
    scale = cost[hull[-1]]  # interpolate on costs near 1: FLOPs are too large for its rounding
    at_hull = interpolation_weights(cost[hull] / scale, min(target, cost[hull[-1]]) / scale)
    weights = np.zeros(len(cost))
    weights[hull] = at_hull
    return weights
