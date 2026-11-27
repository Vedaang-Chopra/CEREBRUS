"""What a controller actually does, read from the paths of its rows: has it collapsed to one fixed path?"""
import re

import pandas as pd

from adaloop.core.controller.tree import ACTIONS

PATH = re.compile(r"(?:L@(\d+),)?X@(\d+)")


def path_actions(path: str) -> tuple[int, int, int, int]:
    """(EXIT, LOOP, LOOK counts over the visited nodes, exit pass) of "X@r" or "L@p,X@r"."""
    match = PATH.fullmatch(path)
    if match is None:
        raise ValueError(f"not a tree path: {path!r}")
    look, last = int(match[1] or 0), int(match[2])
    loops = (look - 1 if look else 0) + last - 1
    return 1, loops, int(look > 0), last


def path_stats(rows: pd.DataFrame, R_max: int, share: float) -> dict:
    """Behaviour of one method at one (k0, lambda): `rows` are its policy rows (every seed).

    EXIT / LOOP / LOOK: % of all visited nodes. exit1..exit{R_max}: % of items that exit at that pass of their last
    branch. look_items: % of items that ever look. cost_frac: mean FLOPs / mean cost_ref. top_path: the most common
    path; top_share: its % within a seed, averaged over seeds; fixed_seeds: seeds in which one path takes more than
    `share` of the items.
    """
    counts = pd.DataFrame([path_actions(p) for p in rows["path"]], columns=[*ACTIONS, "exit_pass"])
    visited = counts[list(ACTIONS)].sum()
    out = {name: 100.0 * visited[name] / visited.sum() for name in ACTIONS}
    out.update({f"exit{r}": 100.0 * (counts["exit_pass"] == r).mean() for r in range(1, R_max + 1)})
    tops = rows.groupby("seed")["path"].apply(lambda paths: paths.value_counts(normalize=True).iloc[0])
    out.update(look_items=100.0 * counts["LOOK"].mean(),
               cost_frac=rows["flops_total"].mean() / rows["cost_ref"].mean(),
               top_path=rows["path"].mode().iloc[0], top_share=100.0 * tops.mean(),
               fixed_seeds=int((tops > share).sum()), n_seeds=len(tops))
    return out
