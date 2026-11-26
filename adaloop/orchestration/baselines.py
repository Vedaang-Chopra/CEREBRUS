"""S13: the baselines on the controller's items, written into the same policy_eval.parquet."""
from pathlib import Path

import numpy as np
import pandas as pd

from adaloop.core.controller.rows import POLICY_COLUMNS
from adaloop.orchestration.cell_policies import grid_cells
from adaloop.orchestration.choosers import chooser_rows
from adaloop.orchestration.controller import run_controller
from adaloop.orchestration.tree import build_tree
from adaloop.schemas.cost import CostParams

BASELINES = ("best_fixed", "best_fixed_cell", "early_exit_only", "tokens_only", "upfront_15way", "oracle", "random",
             "controller_no_h")
ITEM = ["benchmark", "item_id"]


def early_exit_rows(grid: pd.DataFrame, features: dict, params: CostParams, ctrl: dict,
                    weights_dir: str | Path | None = None) -> pd.DataFrame:
    """The k = 100% branch only, actions EXIT / LOOP: same MLP, features, objective and charges as the controller."""
    tree = build_tree(grid, features, params, ctrl, k0s=[ctrl["look"]["k"]])
    return run_controller(tree, features, ctrl, weights_dir=weights_dir, method="early_exit_only")


def no_h_rows(tree: pd.DataFrame, features: dict, ctrl: dict, weights_dir: str | Path | None = None) -> pd.DataFrame:
    """Ablation: the controller without h_last (scalar features only), same seeds, lambdas and charges."""
    return run_controller(tree, features, {**ctrl, "use_h_last": False}, weights_dir=weights_dir,
                          method="controller_no_h")


def tokens_only_rows(grid: pd.DataFrame, upfront: dict, ctrl: dict) -> pd.DataFrame:
    """Choose k per item at a fixed R from the upfront features."""
    settings = ctrl["baselines"]["tokens_only"]
    cells = [(float(k), settings["R"]) for k in settings["ks"]]
    return chooser_rows(grid, upfront, cells, ctrl, "tokens_only")


def upfront_15way_rows(grid: pd.DataFrame, upfront: dict, ctrl: dict) -> pd.DataFrame:
    """Choose one of all (k, R) cells per item from the upfront features."""
    return chooser_rows(grid, upfront, grid_cells(grid), ctrl, "upfront_15way")


def random_rows(tree: pd.DataFrame, controller: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The controller's own paths dealt to random items: per (k0, lambda, seed, split, benchmark) a seeded shuffle.

    So the mix of paths is the controller's, only who gets which is random. It reads no feature and pays for none
    (flops_cell + flops_discarded of the item's own leaf); at a matched mean cost it is read off its own curve.
    """
    leaves = tree.set_index(["k0", *ITEM, "path"])
    rng = np.random.default_rng(seed)
    out = []
    for _, part in controller.groupby(["k0", "lam", "seed", "split", "benchmark"], sort=True):
        part = part.sort_values("item_id")
        part = part.assign(path=part["path"].to_numpy()[rng.permutation(len(part))])
        leaf = leaves.loc[pd.MultiIndex.from_frame(part[["k0", *ITEM, "path"]])]
        out.append(part.assign(
            method="random", correct=leaf["correct"].to_numpy(), flops_cell=leaf["flops_cell"].to_numpy(),
            flops_discarded=leaf["flops_discarded"].to_numpy(), flops_readout=0, flops_state=0,
            flops_total=(leaf["flops_cell"] + leaf["flops_discarded"]).to_numpy()))
    return pd.concat(out, ignore_index=True)[POLICY_COLUMNS]


def save_baseline_rows(path: str | Path, parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Replace every baseline's rows in policy_eval.parquet with `parts` (controller and heuristic rows stay).

    A baseline that is not in `parts` (skipped) is removed, so no row of an older run is left behind.
    """
    old = pd.read_parquet(path)
    rows = pd.concat([old[~old["method"].isin(BASELINES)], *parts], ignore_index=True)
    rows.to_parquet(path, index=False)
    print(f"{len(rows)} policy rows -> {path} | methods {sorted(rows['method'].unique())}")
    return rows
