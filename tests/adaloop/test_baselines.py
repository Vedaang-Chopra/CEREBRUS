"""S13 / S14: baselines and headline arithmetic on a made-up grid. CPU only.

Run: `python tests/adaloop/test_baselines.py` (or pytest) from the repo root.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaloop.core.controller.curves import interpolation_weights
from adaloop.core.controller.synthetic import synthetic_grid
from adaloop.core.eval.bootstrap import holm_adjust, paired_bootstrap_p
from adaloop.orchestration.cell_policies import best_fixed_rows, cell_arrays, grid_cells, oracle_rows

CTRL = {"splits": {"train": "ctrl_train", "dev": "dev", "eval": ["dev"]}, "cap_fracs": [0.5, 1.0]}


def test_interpolation() -> None:
    costs = np.array([1.0, 0.2, 0.6, 0.6])
    weights = interpolation_weights(costs, 0.4)
    assert np.allclose(weights, [0, 0.5, 0.25, 0.25]) and np.isclose(weights @ costs, 0.4)
    assert np.allclose(interpolation_weights(costs, 1.0), [1, 0, 0, 0])
    assert interpolation_weights(costs, 0.1) is None and interpolation_weights(costs, 1.1) is None
    print("✓ PASS curves: linear between neighbours, no extrapolation")


def test_holm_and_p() -> None:
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    adjusted = holm_adjust([0.01, float("nan")])
    assert np.isclose(adjusted[0], 0.02) and np.isnan(adjusted[1])
    a = np.ones(50)
    assert paired_bootstrap_p(a, a, 200, 0) == 1.0 and paired_bootstrap_p(a, 0 * a, 200, 0) < 0.02
    print("✓ PASS statistics: Holm (a missing claim still counts) and bootstrap p")


def test_cell_policies() -> None:
    grid, _ = synthetic_grid(n_items=20)
    cells = grid_cells(grid)
    keys, values, costs, cost_ref = cell_arrays(grid, "dev", cells)
    assert values.shape == (40, 15) and np.allclose(cost_ref, costs[:, cells.index((1.0, 3))])
    best, oracle = best_fixed_rows(grid, CTRL), oracle_rows(grid, CTRL)
    for frac in CTRL["cap_fracs"]:
        fixed, free = best[best["cap_frac"] == frac], oracle[oracle["cap_frac"] == frac]
        assert fixed["path"].nunique() <= 1
        for part in (fixed, free):
            if len(part):
                assert part["flops_total"].mean() <= frac * part["cost_ref"].mean() * (1 + 1e-9)
        if len(fixed):
            assert free["correct"].mean() >= fixed["correct"].mean()
    print("✓ PASS cell policies: caps respected, one fixed cell, oracle >= best fixed")


if __name__ == "__main__":
    test_interpolation()
    test_holm_and_p()
    test_cell_policies()
