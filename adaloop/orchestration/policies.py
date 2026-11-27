"""S11 + S12 in one call: trees, the controller and the heuristic, policy_eval.parquet."""
from pathlib import Path

import pandas as pd

from adaloop.core.controller.rows import select_on_dev
from adaloop.orchestration.capacity import run_capacity
from adaloop.orchestration.controller import run_heuristic
from adaloop.orchestration.tree import check_test_allowed, controller_dir, run_tree
from adaloop.schemas.cost import CostParams


def run_policies(cfg: dict, ctrl: dict, root: str | Path, params: CostParams | None = None) -> pd.DataFrame:
    """tree.parquet, controller (architecture chosen on ctrl_val, `run_capacity`) and heuristic,
    policy_eval.parquet, and the dev choice of (k0, lambda) per cost cap."""
    if ctrl["splits"]["dev"] not in ctrl["splits"]["eval"]:
        raise ValueError("`splits.eval` must hold the dev split: (k0, lambda) per cost cap is chosen from its rows")
    check_test_allowed(ctrl["splits"]["eval"])
    tree, features = run_tree(cfg, ctrl, root, params)
    out = controller_dir(cfg, ctrl, root)
    learned, _, _ = run_capacity(tree, features, ctrl, out_dir=out)
    ruled, thresholds = run_heuristic(tree, features, ctrl)
    rows = pd.concat([learned, ruled], ignore_index=True)
    rows.to_parquet(out / "policy_eval.parquet", index=False)
    thresholds.to_csv(out / "heuristic_thresholds.csv", index=False)
    select_on_dev(rows, ctrl["cap_fracs"], ctrl["splits"]["dev"]).to_csv(out / "select_dev.csv", index=False)
    print(f"{len(rows)} policy rows -> {out / 'policy_eval.parquet'}")
    return rows
