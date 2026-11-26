"""Fixed-setting baselines over grid cells (items x cells arrays)."""
import numpy as np


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
