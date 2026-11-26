"""From chosen tree leaves to policy_eval rows, and the choice of (k0, lambda) per cost cap on dev.

Shared by the controller, the heuristic and (S13) the baselines.
"""
import numpy as np
import pandas as pd

from adaloop.core.controller.tree import path_name
from adaloop.schemas.runs import PolicyRow, TreeData

POLICY_COLUMNS = list(PolicyRow.__dataclass_fields__)


def leaf_costs(data: TreeData, readout: bool, state: bool) -> np.ndarray:
    """(N, B, R) flops_total of every leaf for a method that pays for readouts and/or state sums."""
    return data.flops_cell + data.flops_discarded + readout * data.flops_readout + state * data.flops_state


def policy_rows(data: TreeData, exits: tuple[np.ndarray, np.ndarray], method: str, lam: float, seed: int,
                readout: bool, state: bool) -> pd.DataFrame:
    """One row per item for the exit nodes `exits` = (branch (N,), pass index (N,)) from `walk`."""
    b, q = exits
    item = np.arange(len(data.keys))

    def at(array: np.ndarray) -> np.ndarray:
        return array[item, b, q].round().astype(np.int64)

    table = data.keys.assign(
        method=method, lam=lam, seed=seed, k0=data.k0,
        path=[path_name(int(p), int(r) + 1) for p, r in zip(b, q)], correct=data.correct[item, b, q],
        flops_total=at(leaf_costs(data, readout, state)), flops_cell=at(data.flops_cell),
        flops_discarded=at(data.flops_discarded), flops_readout=at(readout * data.flops_readout),
        flops_state=at(state * data.flops_state), cost_ref=data.cost_ref.astype(np.int64))
    return table[POLICY_COLUMNS]


def select_on_dev(rows: pd.DataFrame, cap_fracs: list[float], split: str = "dev") -> pd.DataFrame:
    """Per method and cost cap: the (k0, lam) with the best dev accuracy whose mean FLOPs fit the cap.

    Accuracy, FLOPs and cost_ref are means per benchmark over items and seeds, then averaged over benchmarks
    (the prereg macro-average). cap = cap_frac x mean cost_ref. Ties: cheaper. NaN when nothing fits.
    """
    dev = rows[rows["split"] == split]
    if dev.empty:
        raise ValueError(f"no rows of split {split!r} to select on")
    columns = ["correct", "flops_total", "cost_ref"]
    per_bench = dev.groupby(["method", "k0", "lam", "benchmark"])[columns].mean()
    points = per_bench.groupby(["method", "k0", "lam"]).mean().reset_index()
    out = []
    for method, part in points.groupby("method"):
        for frac in cap_fracs:
            fits = part[part["flops_total"] <= frac * part["cost_ref"] * (1 + 1e-12)]
            row = {"method": method, "cap_frac": frac, "k0": np.nan, "lam": np.nan, "accuracy": np.nan,
                   "gflops": np.nan}
            if not fits.empty:
                best = fits.sort_values(["correct", "flops_total"], ascending=[False, True]).iloc[0]
                row.update(k0=best["k0"], lam=best["lam"], accuracy=100.0 * best["correct"],
                           gflops=best["flops_total"] / 1e9)
            out.append(row)
    return pd.DataFrame(out)
