"""S11: tree outcomes and costs against the grid rows they come from, on a made-up grid. CPU only.

Run: `python tests/adaloop/test_tree.py` (or pytest) from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaloop.core.controller.diagnostics import path_actions, path_stats
from adaloop.core.controller.fit_split import carve_val
from adaloop.core.controller.inputs import tree_data
from adaloop.core.controller.rows import policy_rows
from adaloop.core.controller.synthetic import HIDDEN, synthetic_grid
from adaloop.core.controller.synthetic import SYNTHETIC_PARAMS as PARAMS
from adaloop.core.controller.tree import EXIT, LOOK, LOOP, action_mask, walk
from adaloop.core.cost.flops import flops_partial, flops_prefill, flops_readout
from adaloop.orchestration.tree import build_tree, check_test_allowed

CTRL = {"k0": [0.25, 0.5], "look": {"mode": "restart", "k": 1.0}, "charge_features": True,
        "features": ["delta_norm", "kl_prev", "entropy", "top1_prob", "vis_attn_mass"]}


def test_leaves_match_grid() -> None:
    grid, features = synthetic_grid()
    tree = build_tree(grid, features, PARAMS, CTRL)
    n_items = len(features)
    assert len(tree) == n_items * len(CTRL["k0"]) * (5 + 4 * 5)
    cells = grid.set_index(["benchmark", "item_id", "k", "R"])
    for leaf in tree.itertuples():
        key = (leaf.benchmark, leaf.item_id)
        p, r, k0 = leaf.look_pass, leaf.exit_pass, leaf.k0
        k = 1.0 if p else k0
        row, feats = cells.loc[(*key, k, r)], features[key]
        assert leaf.correct == row["correct"] and leaf.flops_cell == row["flops_total"]
        assert leaf.cost_ref == cells.loc[(*key, 1.0, 3)]["flops_total"]
        low = cells.loc[(*key, k0, 1)]
        n_low = int(low["n_txt"] + low["n_vis"])
        discarded = flops_partial(PARAMS, n_low, p) + low["flops_compression"] if p else 0
        readout = sum(feats[k0, q]["flops_readout"] for q in range(1, p + 1))
        readout += sum(feats[k, q]["flops_readout"] for q in range(1, r))
        state = sum(feats[k0, q]["flops_state"] for q in range(1, p + 1))
        state += sum(feats[k, q]["flops_state"] for q in range(1, min(r, 4) + 1))
        assert (leaf.flops_discarded, leaf.flops_readout, leaf.flops_state) == (discarded, readout, state)
        assert leaf.flops_total == row["flops_total"] + discarded + readout + state
    first = tree[(tree["path"] == "X@1")]
    assert (first["flops_readout"] == 0).all()  # exit at pass 1 reads out as part of the cell anyway
    print(f"✓ PASS tree: {len(tree)} leaves match their grid cells and costs")


def test_cost_arithmetic() -> None:
    n = 77
    for p in range(1, 6):
        coda = flops_readout(PARAMS, n) - 2 * PARAMS.lm_head
        assert flops_partial(PARAMS, n, p) + coda == flops_prefill(PARAMS, n, p)
    grid, features = synthetic_grid(n_items=1)
    free = build_tree(grid, features, PARAMS, {**CTRL, "charge_features": False})
    assert (free["flops_total"] == free["flops_cell"] + free["flops_discarded"]).all()
    scalar_only = build_tree(grid, features, PARAMS, {**CTRL, "features": ["vis_attn_mass"]})
    assert (scalar_only["flops_total"] ==
            scalar_only["flops_cell"] + scalar_only["flops_discarded"] + scalar_only["flops_state"]).all()
    bad = grid.assign(flops_prefill=grid["flops_prefill"] + 1)
    try:
        build_tree(bad, features, PARAMS, CTRL)
    except ValueError:
        pass
    else:
        raise AssertionError("a grid that the cost parameters do not reproduce must be refused")
    print("✓ PASS tree: partial + coda = prefill; charging flags; wrong cost parameters refused")


def test_walk_and_arrays() -> None:
    grid, features = synthetic_grid()
    tree = build_tree(grid, features, PARAMS, CTRL)
    data = tree_data(tree, features, 0.25, 1.0, "dev")
    n, B, R = data.correct.shape
    assert (B, R) == (5, 5) and data.h.shape == (n, 10, HIDDEN)
    mask = action_mask(B, R)
    assert not mask[:, 4, LOOP].any() and not mask[:, 4, LOOK].any() and not mask[1:, :, LOOK].any()
    actions = np.full((n, B, R), EXIT)
    actions[:, 0, 0] = LOOP
    actions[:, 0, 1] = LOOK  # look after pass 2
    actions[:, 2, 0] = LOOP  # then two passes with all tokens
    rows = policy_rows(data, walk(actions), "m", 0.0, 0, True, True)
    assert (rows["path"] == "L@2,X@2").all()
    leaf = tree[(tree["k0"] == 0.25) & (tree["split"] == "dev") & (tree["path"] == "L@2,X@2")]
    both = rows.merge(leaf, on=["benchmark", "item_id"], suffixes=("", "_leaf"))
    assert len(both) == n and (both["flops_total"] == both["flops_total_leaf"]).all()
    assert (both["correct"] == both["correct_leaf"]).all()
    item = (data.keys["benchmark"][0], data.keys["item_id"][0])
    assert np.allclose(data.h[0, data.h_index[3, 1]], features[item][1.0, 2]["h_last"].float().numpy())
    assert np.isnan(data.raw["kl_prev"][:, :, 0]).all()
    full = build_tree(grid, features, PARAMS, CTRL, k0s=[1.0])
    assert len(full) == len(features) * 5 and tree_data(full, features, 1.0, 1.0, "dev").correct.shape[1:] == (1, 5)
    print("✓ PASS tree: paths walk to the right leaves; no-look trees at k0 = 1")


def test_path_stats() -> None:
    assert path_actions("X@1") == (1, 0, 0, 1) and path_actions("X@3") == (1, 2, 0, 3)
    assert path_actions("L@2,X@3") == (1, 3, 1, 3)  # LOOP, LOOK | LOOP, LOOP, EXIT
    rows = pd.DataFrame({"path": ["X@1", "X@1", "X@1", "L@1,X@2"], "seed": [0, 0, 1, 1], "flops_total": 1, "cost_ref": 2})
    stats = path_stats(rows, 5, 0.95)
    assert (stats["fixed_seeds"], stats["n_seeds"], stats["top_share"], stats["look_items"]) == (1, 2, 75.0, 25.0)
    assert np.isclose(stats["EXIT"] + stats["LOOP"] + stats["LOOK"], 100) and stats["cost_frac"] == 0.5
    print("✓ PASS diagnostics: action counts from paths, a fixed seed is counted")


def test_carve_val() -> None:
    rows = pd.DataFrame([{"benchmark": b, "item_id": f"{split}{i}", "split": split, "R": R}
                         for b in ("gqa", "textvqa") for split in ("ctrl_train", "dev")
                         for i in range(20) for R in (1, 2)])
    images = {b: {f"{split}{i}": f"{split}-img{i // 2}" for split in ("ctrl_train", "dev") for i in range(20)}
              for b in ("gqa", "textvqa")}
    fit = carve_val(rows, images, "ctrl_train", "ctrl_val", 0.2, 0)
    assert fit.equals(carve_val(rows, images, "ctrl_train", "ctrl_val", 0.2, 0))
    assert (fit[rows["split"] == "dev"] == "dev").all()
    image = [images[b][i] for b, i in zip(rows["benchmark"], rows["item_id"])]
    table = rows.assign(fit=fit, image=image)[rows["split"] == "ctrl_train"]
    assert (table.groupby(["benchmark", "image"])["fit"].nunique() == 1).all()  # whole images
    held = table[table["fit"] == "ctrl_val"].groupby("benchmark")["image"].nunique()
    assert held.tolist() == [2, 2]  # 20% of 10 images per benchmark
    print("✓ PASS fit split: seeded, image-disjoint, 20% of the train images per benchmark")


def test_test_split_sealed() -> None:
    if (Path(__file__).resolve().parents[2] / "configs" / "prereg.yaml").is_file():
        return
    grid, features = synthetic_grid(n_items=1, splits=("test",))
    for call in (lambda: check_test_allowed(["test"]), lambda: build_tree(grid, features, PARAMS, CTRL)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("the test split must be refused without configs/prereg.yaml")
    print("✓ PASS tree: test split refused without prereg.yaml")


if __name__ == "__main__":
    test_leaves_match_grid()
    test_cost_arithmetic()
    test_walk_and_arrays()
    test_path_stats()
    test_carve_val()
    test_test_split_sealed()
