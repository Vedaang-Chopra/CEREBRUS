"""S11: tree outcomes and costs against the grid rows they come from, on a made-up grid. CPU only.

Run: `python tests/adaloop/test_tree.py` (or pytest) from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaloop.core.controller.inputs import tree_data
from adaloop.core.controller.rows import policy_rows
from adaloop.core.controller.tree import EXIT, LOOK, LOOP, action_mask, walk
from adaloop.core.cost.flops import (
    flops_features,
    flops_partial,
    flops_prefill,
    flops_readout,
    flops_total,
)
from adaloop.orchestration.tree import build_tree, check_test_allowed
from adaloop.schemas.cost import CostParams

PARAMS = CostParams(embed=0, prelude=600, core=1200, core_message=40, coda=600, lm_head=900, vit_patch=50,
                    vit_layers=300, vit_projector=80, core_start=2, core_end=4, attn_dims=(8,) * 6,
                    attn_windows=(None,) * 6, kv_dims=(4,) * 6, vit_n_layers=3, vit_hidden=16, vit_merge=2)
CTRL = {"k0": [0.25, 0.5], "look": {"mode": "restart", "k": 1.0}, "charge_features": True,
        "features": ["delta_norm", "kl_prev", "entropy", "top1_prob", "vis_attn_mass"]}
KS, RS, HIDDEN, HEADS = (0.25, 0.5, 1.0), (1, 2, 3, 4, 5), 24, 4


def synthetic_grid(n_items: int = 6, splits: tuple[str, ...] = ("ctrl_train", "dev"), d: int = HIDDEN,
                   seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """(grid rows, features) shaped like the S7 outputs, FLOPs from PARAMS. `n_items` per benchmark and split."""
    rng = np.random.default_rng(seed)
    rows, features = [], {}
    for split in splits:
        for benchmark in ("gqa", "textvqa"):
            for i in range(n_items):
                item_id = f"{split}{i}"
                n_vis_full, n_txt, hard = int(rng.integers(40, 120)), int(rng.integers(10, 30)), rng.random()
                feats = features[benchmark, item_id] = {}
                for k in KS:
                    n_vis = round(k * n_vis_full)
                    n = n_vis + n_txt
                    extra = flops_features(PARAMS, n, n_vis, HEADS, HIDDEN)
                    for R in RS:
                        skill = (k * 0.6 + R * 0.08) - hard * 0.7  # more tokens and loops help hard items more
                        correct = float(skill + rng.normal(0, 0.15) > 0)
                        flops = flops_total(PARAMS, n, 3, R, [(1, 8, 8)])
                        compression = 0 if k == 1.0 else 5000
                        rows.append({"item_id": item_id, "benchmark": benchmark, "split": split,
                                     "model_variant": "base", "rerun_id": 0, "k": k, "R": R, "correct": correct,
                                     "n_vis": n_vis, "n_vis_full": n_vis_full, "n_txt": n_txt,
                                     "flops_vit": flops["flops_vit"], "flops_prefill": flops["flops_prefill"],
                                     "flops_compression": compression,
                                     "flops_total": flops["flops_total"] + compression, "config_hash": "synthetic"})
                        h = torch.from_numpy(rng.normal(0, 1, d)).half()
                        h[:2] = torch.tensor([skill, hard]).half()  # something to learn
                        feats[k, R] = {"h_last": h, "delta_norm": float("nan") if R == 1 else rng.random(),
                                       "kl_prev": float("nan") if R == 1 else rng.random() / R,
                                       "entropy": float(max(0.0, 1 - skill)), "top1_prob": float(rng.random()),
                                       "vis_attn_mass": float(rng.random()), **extra}
    return pd.DataFrame(rows), features


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
    test_test_split_sealed()
