"""S13 policies that pick fixed (k, R) cells: shared arrays and rows, best_fixed and the oracle per cost cap."""
import numpy as np
import pandas as pd

from adaloop.core.controller.baselines import best_fixed_cell
from adaloop.core.controller.rows import POLICY_COLUMNS
from adaloop.core.cost.flops import REF_R
from adaloop.core.eval.oracle import oracle_choices
from adaloop.orchestration.tree import check_test_allowed

ITEM = ["benchmark", "item_id"]


def grid_cells(grid: pd.DataFrame) -> list[tuple]:
    """Every (k, R) of the grid, sorted."""
    return sorted({(float(k), int(R)) for k, R in zip(grid["k"], grid["R"])})


def cell_arrays(grid: pd.DataFrame, split: str, cells: list[tuple]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray,
                                                                           np.ndarray]:
    """(keys, scores (N, C), FLOPs (N, C), cost_ref (N,)) of one split; items sorted by (benchmark, item_id).

    `grid`: main-run rows of one model variant. cost_ref: flops_total of the item's (k = 1, R = 3) cell.
    """
    check_test_allowed([split])
    rows = grid[grid["split"] == split]
    if rows.empty:
        raise ValueError(f"no grid rows of split {split!r}")
    wide = rows.pivot(index=ITEM, columns=["k", "R"], values=["correct", "flops_total"]).sort_index()
    columns = pd.MultiIndex.from_tuples(cells)
    values = wide["correct"].reindex(columns=columns).to_numpy(dtype=float)
    costs = wide["flops_total"].reindex(columns=columns).to_numpy(dtype=float)
    cost_ref = wide["flops_total"].reindex(columns=pd.MultiIndex.from_tuples([(1.0, REF_R)])).to_numpy(dtype=float)[:, 0]
    if np.isnan(values).any() or np.isnan(costs).any() or np.isnan(cost_ref).any():
        raise ValueError("incomplete grid: some item lacks a cell")
    first = rows.drop_duplicates(ITEM).set_index(ITEM).loc[wide.index, ["split", "model_variant"]]
    return first.reset_index(), values, costs, cost_ref


def cell_rows(keys: pd.DataFrame, cells: list[tuple], picks: np.ndarray, values: np.ndarray, costs: np.ndarray,
              cost_ref: np.ndarray, method: str, k0: float, lam: float = np.nan, cap_frac: float = np.nan,
              seed: int = 0, state: np.ndarray | None = None) -> pd.DataFrame:
    """policy_eval rows for item i running cell `cells[picks[i]]`; `state`: extra feature FLOPs per item."""
    item = np.arange(len(keys))
    cost = costs[item, picks].round().astype(np.int64)
    state = np.zeros(len(keys), dtype=np.int64) if state is None else np.asarray(state).astype(np.int64)
    table = keys.assign(
        method=method, lam=lam, cap_frac=cap_frac, seed=seed, k0=k0,
        path=[f"k{cells[j][0]:g},X@{cells[j][1]}" for j in picks], correct=values[item, picks],
        flops_total=cost + state, flops_cell=cost, flops_discarded=0, flops_readout=0, flops_state=state,
        cost_ref=cost_ref.astype(np.int64))
    return table[POLICY_COLUMNS]


def best_fixed_rows(grid: pd.DataFrame, ctrl: dict) -> pd.DataFrame:
    """Per cost cap one cell for every item: chosen on the train split among the cells that fit on the eval items.

    Cap = cap_frac x mean cost_ref of the evaluated split. A cap that no cell fits gives no rows (printed).
    """
    cells = grid_cells(grid)
    _, select_values, select_costs, _ = cell_arrays(grid, ctrl["splits"]["train"], cells)
    out = []
    for split in ctrl["splits"]["eval"]:
        keys, values, costs, cost_ref = cell_arrays(grid, split, cells)
        for frac in ctrl["cap_fracs"]:
            best = best_fixed_cell(select_values, select_costs, costs, frac * cost_ref.mean())
            if best is None:
                print(f"best_fixed {split}: no cell fits {frac} x cost_ref")
                continue
            picks = np.full(len(keys), best)
            out.append(cell_rows(keys, cells, picks, values, costs, cost_ref, "best_fixed", k0=cells[best][0],
                                 cap_frac=frac))
    return pd.concat(out, ignore_index=True)


def oracle_rows(grid: pd.DataFrame, ctrl: dict) -> pd.DataFrame:
    """Per cost cap the exact per-item oracle over the cells (mean FLOPs <= cap). It reads the labels: a bound only."""
    cells = grid_cells(grid)
    out = []
    for split in ctrl["splits"]["eval"]:
        keys, values, costs, cost_ref = cell_arrays(grid, split, cells)
        for frac in ctrl["cap_fracs"]:
            picks = oracle_choices(values, costs, frac * cost_ref.mean())
            if picks is None:
                print(f"oracle {split}: no choice fits {frac} x cost_ref")
                continue
            out.append(cell_rows(keys, cells, picks, values, costs, cost_ref, "oracle", k0=0.0, cap_frac=frac))
    return pd.concat(out, ignore_index=True)
