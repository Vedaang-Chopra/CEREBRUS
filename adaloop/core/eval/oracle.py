"""Per-item oracle over grid cells under a MEAN cost cap, and its noise floor.

Arrays are items x cells: `values` is the score of each item in each cell (0..1), `costs` its FLOPs.
"""
import numpy as np

SCORE_UNITS = (1, 10, 100)  # GQA scores are 0/1, VQA accuracy is one of 0, 0.3, 0.6, 0.9, 1


def _score_units(values: np.ndarray) -> np.ndarray:
    """`values` as small integers on a common scale, so that total score can index a table."""
    for scale in SCORE_UNITS:
        units = np.rint(values * scale)
        if np.allclose(units, values * scale, rtol=0, atol=1e-6):
            return units.astype(np.int64)
    raise ValueError(f"scores are not multiples of 1/{SCORE_UNITS[-1]}; the exact oracle needs discrete scores")


def oracle_choices(values: np.ndarray, costs: np.ndarray, cap: float) -> np.ndarray | None:
    """One cell per item with the highest total score such that the MEAN cost of the chosen cells is <= `cap`.

    Exact (not a Lagrangian or greedy approximation): a multiple-choice knapsack solved by dynamic programming over
    the total score, keeping the cheapest way to reach each total. Among equal totals the cheapest is returned.
    None if even the cheapest cell of every item exceeds the cap.
    """
    assert values.shape == costs.shape and values.ndim == 2 and values.size, (values.shape, costs.shape)
    units = _score_units(values)  # (n, C)
    assert units.min() >= 0, "scores must be >= 0"
    n, n_cells = units.shape
    size = int(units.max(axis=1).sum()) + 1
    cheapest = np.full(size, np.inf)  # cheapest total cost of each total score over the items so far
    cheapest[0] = 0.0
    taken = np.zeros((n, size), dtype=np.int32)  # (item, total score) -> cell
    for i in range(n):
        new = np.full(size, np.inf)
        for j in range(n_cells):
            u = int(units[i, j])
            if u >= size:
                continue
            cost = cheapest[: size - u] + costs[i, j]
            better = cost < new[u:]
            new[u:][better] = cost[better]
            taken[i, u:][better] = j
        cheapest = new
    feasible = np.flatnonzero(cheapest <= cap * n * (1 + 1e-12))
    if feasible.size == 0:
        return None
    total = int(feasible[-1])
    choices = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        choices[i] = taken[i, total]
        total -= int(units[i, choices[i]])
    return choices


def noise_oracle_gain(values: np.ndarray, costs: np.ndarray, values_again: np.ndarray, costs_again: np.ndarray,
                      cap: float) -> np.ndarray | None:
    """Per item, the part of the oracle's score that does not survive a re-run (the "noise oracle", in score units).

    Our reading of "noise oracle" (DEC3 names it without a formula; it is subtracted from oracle - best_fixed, so it
    must be a gain, not an accuracy): the oracle picks its cells on one run and is scored on that run and, with the
    same cells, on the other run; the difference is what it won by picking lucky outcomes. Averaged over both
    directions (pick on run 0, pick on run 1). Zero when the two runs agree everywhere. It measures run-to-run noise
    only, not how much a per-item choice among cells would carry over to other data. None if the cap is infeasible.
    """
    assert values.shape == values_again.shape == costs.shape == costs_again.shape, "the two runs must align"
    first = oracle_choices(values, costs, cap)
    second = oracle_choices(values_again, costs_again, cap)
    if first is None or second is None:
        return None
    item = np.arange(len(values))
    return 0.5 * ((values[item, first] - values_again[item, first])
                  + (values_again[item, second] - values[item, second]))
