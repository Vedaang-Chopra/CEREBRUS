"""S11: load the grid and its pass features, build the trajectory trees, write tree.parquet. No GPU, no weights."""
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaloop.core.controller.fit_split import carve_val
from adaloop.core.controller.tree import charged, item_tree
from adaloop.core.cost.meta import cost_params_from_config
from adaloop.orchestration.features import load_features
from adaloop.orchestration.grid import grid_dir
from adaloop.orchestration.item_images import load_item_images
from adaloop.schemas.cost import CostParams
from adaloop.schemas.runs import TreeLeaf
from utils.paths import find_repo_root

ITEM = ["benchmark", "item_id"]


def check_test_allowed(splits: Sequence[str]) -> None:
    """The test split stays sealed until the preregistration file exists (DEC3 §1, §9)."""
    prereg = find_repo_root() / "configs" / "prereg.yaml"
    if "test" in set(splits) and not prereg.is_file():
        raise ValueError(f"the test split is sealed: write {prereg} before any stage touches test")


def controller_dir(cfg: dict, ctrl: dict, root: str | Path) -> Path:
    """<out_dir of mvp.yaml>/<out_dir of controller.yaml>."""
    return Path(root) / cfg["out_dir"] / ctrl["out_dir"]


def load_controller_inputs(cfg: dict, ctrl: dict, root: str | Path) -> tuple[pd.DataFrame, dict]:
    """(grid rows of the variant, main run; features[benchmark, item_id][k, p]) from the grid folder.

    With `splits.val` set, the rows get a `fit_split` column: `split` with the held-out train images renamed to val.
    `split` itself stays, so fixed-setting baselines still read the whole train split.
    """
    folder = grid_dir(cfg, root, ctrl["grid_name"])
    file = ctrl.get("grid_file", "grid.parquet")  # grid_partial.parquet: smoke test on a quick pass (nb11)
    grid = pd.read_parquet(folder / file)
    grid = grid[(grid["model_variant"] == ctrl["variant"]) & (grid["rerun_id"] == 0)]
    if grid.empty:
        raise ValueError(f"no main-run rows of variant {ctrl['variant']!r} in {folder / file}")
    check_test_allowed(grid["split"])
    splits, held = ctrl["splits"], ctrl.get("ctrl_val")
    if "val" in splits:
        grid = grid.assign(fit_split=carve_val(grid, load_item_images(cfg, ctrl, root), splits["train"],
                                               splits["val"], held["image_frac"], held["seed"]))
    features = {}
    first = grid.drop_duplicates(ITEM)
    for benchmark, item_id, digest in zip(first["benchmark"], first["item_id"], first["config_hash"]):
        path = folder / "features" / ctrl["variant"] / digest / f"{benchmark}_{item_id}.pt"
        features[benchmark, item_id] = load_features(path)
    print(f"{len(features)} items, {len(grid)} grid rows, features loaded from {folder}")
    return grid, features


def build_tree(grid: pd.DataFrame, features: dict, params: CostParams, ctrl: dict,
               k0s: Sequence[float] | None = None) -> pd.DataFrame:
    """tree.parquet rows: every leaf of every item at every k0 (default `ctrl["k0"]`; k0 = look.k: no look).

    `grid` holds one row per (item, k, R) of one model variant. flops_total charges the feature readouts / state sums
    that `ctrl["features"]` need (none if `charge_features` is false). A row's split is `fit_split` where the grid has it.
    """
    check_test_allowed(grid["split"])
    if ctrl["look"]["mode"] != "restart":
        raise ValueError(f"only look.mode 'restart' is built, got {ctrl['look']['mode']!r}")
    if grid["model_variant"].nunique() != 1 or grid.duplicated(ITEM + ["k", "R"]).any():
        raise ValueError("pass the main-run rows of one model variant")
    look_k = ctrl["look"]["k"]
    R_max = int(grid["R"].max())
    pays_readout, pays_state = charged(ctrl["features"])
    pays_readout, pays_state = pays_readout and ctrl["charge_features"], pays_state and ctrl["charge_features"]
    out = []
    for (benchmark, item_id), rows in grid.groupby(ITEM):
        cells = {(row["k"], row["R"]): row for row in rows.to_dict("records")}
        for k0 in k0s or ctrl["k0"]:
            lacking = [(k, R) for k in {k0, look_k} for R in range(1, R_max + 1) if (k, R) not in cells]
            if lacking:
                raise ValueError(f"{benchmark} {item_id}: incomplete grid, lacks cells {lacking}")
            first = cells[k0, 1]
            for leaf in item_tree(cells, features[benchmark, item_id], params, k0, look_k, R_max):
                total = (leaf["flops_cell"] + leaf["flops_discarded"] + pays_readout * leaf["flops_readout"]
                         + pays_state * leaf["flops_state"])
                out.append(asdict(TreeLeaf(item_id=item_id, benchmark=benchmark, split=first.get("fit_split", first["split"]),
                                           model_variant=first["model_variant"], k0=k0, flops_total=total, **leaf)))
    return pd.DataFrame(out)


def run_tree(cfg: dict, ctrl: dict, root: str | Path, params: CostParams | None = None) -> tuple[pd.DataFrame, dict]:
    """Load, build and write <controller_dir>/tree.parquet; returns (tree, features)."""
    grid, features = load_controller_inputs(cfg, ctrl, root)
    params = params or cost_params_from_config(cfg["model_id"])
    tree = build_tree(grid, features, params, ctrl)
    out = controller_dir(cfg, ctrl, root)
    out.mkdir(parents=True, exist_ok=True)
    tree.to_parquet(out / "tree.parquet", index=False)
    print(f"tree: {len(tree)} leaves -> {out / 'tree.parquet'}")
    return tree, features
