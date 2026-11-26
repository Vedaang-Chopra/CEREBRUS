"""S13 / S14: baselines and headline arithmetic on a made-up grid. CPU only.

Run: `python tests/adaloop/test_baselines.py` (or pytest) from the repo root.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaloop.core.controller.baselines import hull_mix, upper_hull
from adaloop.core.controller.curves import interpolation_weights
from adaloop.core.controller.synthetic import synthetic_grid
from adaloop.core.eval.bootstrap import holm_adjust, paired_bootstrap_p
from adaloop.orchestration.cell_policies import (
    best_fixed_cell_rows,
    best_fixed_rows,
    cell_arrays,
    grid_cells,
    oracle_rows,
)
from adaloop.orchestration.grid import grid_dir
from adaloop.orchestration.headroom_gate import headroom_gate

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


def test_hull_by_hand() -> None:
    # cells (cost, selection accuracy): (1, .2) (2, .3) (3, .8) (4, .9) (5, .7)
    # (2, .3) is under the line (1, .2)-(3, .8); (5, .7) is past the most accurate cell: hull = cells 0, 2, 3
    select = np.tile([0.2, 0.3, 0.8, 0.9, 0.7], (10, 1))
    costs = np.tile([1e12, 2e12, 3e12, 4e12, 5e12], (4, 1))  # report items (FLOPs-sized numbers)
    assert upper_hull(select.mean(axis=0), costs.mean(axis=0)) == [0, 2, 3]
    for target, expected in [(1, [1, 0, 0, 0, 0]), (2, [0.5, 0, 0.5, 0, 0]), (3.5, [0, 0, 0.5, 0.5, 0]),
                             (4.5, [0, 0, 0, 1, 0]), (5, [0, 0, 0, 1, 0])]:  # above the top: stay on it
        assert np.allclose(hull_mix(select, costs, target * 1e12), expected), target
    assert np.isclose(select.mean(axis=0) @ hull_mix(select, costs, 2e12), 0.5)
    assert hull_mix(select, costs, 0.5e12) is None and hull_mix(select, costs, 5.1e12) is None
    print("✓ PASS best_fixed hull: hand example, cost matched, flat above the top, no extrapolation")


def test_cell_policies() -> None:
    grid, _ = synthetic_grid(n_items=20)
    cells = grid_cells(grid)
    keys, values, costs, cost_ref = cell_arrays(grid, "dev", cells)
    assert values.shape == (40, 15) and np.allclose(cost_ref, costs[:, cells.index((1.0, 3))])
    mix, single, oracle = best_fixed_rows(grid, CTRL), best_fixed_cell_rows(grid, CTRL), oracle_rows(grid, CTRL)
    assert set(mix["method"]) == {"best_fixed"} and set(single["method"]) == {"best_fixed_cell"}
    for frac in CTRL["cap_fracs"]:
        mixed, fixed, free = (part[part["cap_frac"] == frac] for part in (mix, single, oracle))
        assert fixed["path"].nunique() <= 1
        for part in (mixed, fixed, free):
            if len(part):
                assert part["flops_total"].mean() <= frac * part["cost_ref"].mean() * (1 + 1e-9) + 1  # whole FLOPs
        top = mixed["path"].str.count("~").eq(0).all()  # on one cell: it may cost less than the cap
        if len(mixed) and not top:
            assert abs(mixed["flops_total"].mean() - frac * mixed["cost_ref"].mean()) <= 1
        if len(fixed):
            assert free["correct"].mean() >= fixed["correct"].mean()
        if len(mixed):  # a mix is a fractional choice: it can beat the integer oracle by less than one item
            assert free["correct"].mean() >= mixed["correct"].mean() - 1 / len(mixed)
    # labels of the report split change no choice
    shuffled = grid.assign(correct=np.where(grid["split"] == "dev", 1 - grid["correct"], grid["correct"]))
    assert best_fixed_rows(shuffled, CTRL)["path"].equals(mix["path"])
    print("✓ PASS cell policies: caps respected, one fixed cell, oracle bounds, no report label in the hull")


def test_headroom_gate_by_hand() -> None:
    # R = 3 cells A (k=.25, cost 1), B (k=.5, cost 3), C (k=1, cost 4 = cost_ref); cap 0.5 -> 2. 4 items per split.
    # ctrl_train accuracy A .25, B .75, C .75: hull A-B, at cost 2 half A half B. dev scores below.
    # hull: mean(A, B) = (25 + 75) / 2 = 50. single cell <= 2: A = 25. oracle: items 2 and 3 get B = 75.
    # the re-run repeats every score: noise 0. headroom: hull 75 - 50 = 25, cell 75 - 25 = 50.
    cells = {0.25: (1, [1, 0, 0, 0], [1, 0, 0, 0]), 0.5: (3, [1, 1, 1, 0], [1, 1, 1, 0]),
             1.0: (4, [1, 1, 1, 0], [1, 1, 1, 1])}  # k -> (cost, ctrl_train scores, dev scores)
    rows = [{"benchmark": "gqa", "item_id": f"{split}{i}", "split": split, "model_variant": "base", "rerun_id": rerun,
             "k": k, "R": 3, "correct": float(scores[i]), "flops_total": cost * 10**9}
            for k, (cost, train, dev) in cells.items()
            for split, scores in (("ctrl_train", train), ("dev", dev))
            for i in range(4) for rerun in ((0, 1) if split == "dev" else (0,))]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"out_dir": tmp, "grid_batch": {"size": 8}, "batch_grouping": "equal_length", "splits": {"seed": 0},
               "headroom": {"select_split": "ctrl_train", "report_split": "dev", "cap_fracs": [0.5],
                            "headline_frac": 0.5, "min_pts": 30.0, "n_boot": 200}}
        folder = grid_dir(cfg, tmp)
        folder.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(folder / "grid.parquet", index=False)
        gate = headroom_gate(cfg, tmp).iloc[0]  # no Step-2b rows here: re-run noise only
    got = [gate[name] for name in ("oracle", "best_fixed", "best_fixed_cell", "noise_used", "headroom", "headroom_cell")]
    assert np.allclose(got, [75, 50, 25, 0, 25, 50]), got
    assert gate["noise_source"] == "re-run" and not gate["met"] and gate["met_cell"]
    print("✓ PASS headroom gate: hand example over the hull mix and over the single cell")


if __name__ == "__main__":
    test_interpolation()
    test_holm_and_p()
    test_hull_by_hand()
    test_cell_policies()
    test_headroom_gate_by_hand()
