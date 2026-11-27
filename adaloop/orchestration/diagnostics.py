"""Controller diagnostics: behaviour of the controller next to the hindsight-optimal policy of the same trees."""
import pandas as pd
import torch

from adaloop.core.controller.diagnostics import path_stats
from adaloop.core.controller.inputs import tree_data
from adaloop.core.controller.objectives import supervised_targets
from adaloop.core.controller.rows import policy_rows
from adaloop.core.controller.tree import action_mask, walk
from adaloop.orchestration.controller import charged_costs, tree_rewards


def collapse_table(rows: pd.DataFrame, tree: pd.DataFrame, features: dict, ctrl: dict,
                   method: str = "controller") -> pd.DataFrame:
    """Per (k0, lambda) on dev: one row for `method` and one for the oracle, columns as in `path_stats`.

    The oracle takes the best reachable action at every node for the same reward and charges (it reads the labels).
    `collapsed` (method rows): in some seed one fixed path takes more than `collapse_share` of the items while the
    oracle does not put that share on one path.
    """
    split, share = ctrl["splits"]["dev"], ctrl["collapse_share"]
    pays = charged_costs(ctrl)
    out = []
    for k0 in dict.fromkeys(tree["k0"]):
        data = tree_data(tree, features, k0, ctrl["look"]["k"], split)
        R_max = data.correct.shape[2]
        mask = torch.from_numpy(action_mask(*data.correct.shape[1:]))
        for lam in ctrl["lambdas"]:
            part = rows[(rows["method"] == method) & (rows["split"] == split) & (rows["k0"] == k0)
                        & (rows["lam"] == lam)]
            if part.empty:
                raise ValueError(f"no {split} rows of {method!r} at k0={k0}, lambda={lam}")
            best = walk(supervised_targets(tree_rewards(data, lam, pays), mask).numpy())
            oracle = policy_rows(data, best, "tree_oracle", lam, 0, *pays)
            learned, optimal = path_stats(part, R_max, share), path_stats(oracle, R_max, share)
            collapsed = learned["fixed_seeds"] > 0 and optimal["fixed_seeds"] == 0
            out.append({"k0": k0, "lam": lam, "who": method, "collapsed": collapsed, **learned})
            out.append({"k0": k0, "lam": lam, "who": "oracle", "collapsed": False, **optimal})
    return pd.DataFrame(out)
