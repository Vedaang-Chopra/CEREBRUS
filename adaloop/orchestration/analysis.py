"""Tables and figure data from grid.parquet (DEC3 S14, grid part): cell means, re-run noise and headroom.

No model and no GPU. The test split is refused everywhere until stage S12.
"""
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from adaloop.core.controller.baselines import best_fixed_cell
from adaloop.core.cost.flops import REF_R
from adaloop.core.eval.bootstrap import diff_of_means_ci, paired_bootstrap_ci
from adaloop.core.eval.oracle import noise_oracle_gain, oracle_choices

ITEM = ["benchmark", "item_id"]
CELL = ["k", "R"]


def _refuse_test(splits: Sequence[str]) -> None:
    if "test" in set(splits):
        raise ValueError("the test split is sealed until stage S12")


def _variant_rows(grid: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Rows of one model variant of a grid table without test rows."""
    _refuse_test(grid["split"])
    rows = grid[grid["model_variant"] == variant]
    if rows.empty:
        raise ValueError(f"no rows of model_variant {variant!r} in the grid")
    if rows.duplicated(ITEM + CELL + ["rerun_id"]).any():
        raise ValueError("the grid repeats a cell of an item: merge it with api.merge_grid")
    return rows


def load_grid(path: str | Path) -> pd.DataFrame:
    """A saved grid.parquet; refuses a table that holds test rows."""
    grid = pd.read_parquet(path)
    _refuse_test(grid["split"])
    return grid


def cell_table(grid: pd.DataFrame, splits: Sequence[str], variant: str = "base") -> pd.DataFrame:
    """Per benchmark and (k, R), main run only: n, accuracy (%) and mean gflops_total over the items of `splits`."""
    _refuse_test(splits)
    rows = _variant_rows(grid, variant)
    rows = rows[(rows["rerun_id"] == 0) & rows["split"].isin(splits)]
    table = rows.groupby(["benchmark"] + CELL, as_index=False).agg(
        n=("correct", "size"), accuracy=("correct", "mean"), gflops_total=("flops_total", "mean"))
    table["accuracy"] *= 100.0
    table["gflops_total"] /= 1e9
    if table.groupby("benchmark")["n"].nunique().gt(1).any():
        raise ValueError(f"cells of a benchmark have different item counts (incomplete grid):\n{table}")
    return table


def noise_rate(grid: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    """Run 0 against run 1 on the same item and cell, per benchmark and for all: how often results move on their own.

    answer_flip_pct: parsed answers differ; score_change_pct: the score differs; accuracy_diff: run 1 minus run 0 (pts).
    """
    rows = _variant_rows(grid, variant)
    pair = rows[rows["rerun_id"] == 0].merge(rows[rows["rerun_id"] == 1], on=ITEM + CELL, suffixes=("", "_again"))
    if pair.empty:
        raise ValueError("no re-run rows (rerun_id = 1) that match a main row")
    pair = pair.assign(flip=pair["pred"].fillna("") != pair["pred_again"].fillna(""),
                       moved=(pair["correct"] - pair["correct_again"]).abs() > 1e-9)
    out = []
    for name, part in [*pair.groupby("benchmark"), ("all", pair)]:
        out.append({"benchmark": name, "n_items": len(part[ITEM].drop_duplicates()), "n_pairs": len(part),
                    "answer_flip_pct": 100.0 * part["flip"].mean(), "score_change_pct": 100.0 * part["moved"].mean(),
                    "accuracy": 100.0 * part["correct"].mean(), "accuracy_again": 100.0 * part["correct_again"].mean(),
                    "accuracy_diff": 100.0 * (part["correct_again"].mean() - part["correct"].mean())})
    return pd.DataFrame(out)


def _arrays(rows: pd.DataFrame, cells: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """(scores, FLOPs), each items x cells, items in a fixed order. Every item must have every cell."""
    wide = rows.pivot(index=ITEM, columns=CELL, values=["correct", "flops_total"]).sort_index()
    values = wide["correct"].reindex(columns=pd.MultiIndex.from_tuples(cells)).to_numpy(dtype=float)
    costs = wide["flops_total"].reindex(columns=pd.MultiIndex.from_tuples(cells)).to_numpy(dtype=float)
    if np.isnan(values).any() or np.isnan(costs).any():
        raise ValueError("incomplete grid: some item lacks a cell, a score or a FLOPs value")
    return values, costs


def _eligible_oracle(values: np.ndarray, costs: np.ndarray, cap: float) -> float:
    """Mean score of the per-item best over the cells whose OWN mean FLOPs fit the cap (nan if none fits)."""
    fits = costs.mean(axis=0) <= cap * (1 + 1e-12)
    return float(values[:, fits].max(axis=1).mean()) if fits.any() else float("nan")


def headroom_table(grid: pd.DataFrame, cfg: dict, variant: str = "base") -> pd.DataFrame:
    """T2: per benchmark and cost cap, best fixed, oracle, noise oracle and headroom (accuracy in %, gaps in points).

    Settings are `cfg["headroom"]`. The cap is cap_frac x cost_ref, cost_ref = mean flops_total of the (k=1, R=3)
    cell on the items in question.
    - best_fixed: one (k, R) chosen on `select_split`, scored on `report_split` (`best_fixed_cell`).
    - oracle: per report item its own cell, mean FLOPs of the picks <= cap, exact (`oracle_choices`).
    - oracle_eligible: the stricter reading, per-item best over the cells that fit the cap on their own (not used below).
    - noise_oracle: `noise_oracle_gain` on every re-run item of the benchmark, any grid split (the re-run sample holds
      only a few report items).
    - headroom = oracle - best_fixed - noise_oracle. CIs: bootstrap over items with the picks held fixed; the headroom
      CI resamples report items and re-run items separately.
    A cap that no cell fits gives nan.
    """
    settings = cfg["headroom"]
    select_split, report_split = settings["select_split"], settings["report_split"]
    _refuse_test([select_split, report_split])
    rows = _variant_rows(grid, variant)
    cells = sorted(set(zip(rows["k"], rows["R"])))
    if (1.0, REF_R) not in cells:
        raise ValueError(f"the grid lacks the reference cell k=1, R={REF_R}")
    ref = cells.index((1.0, REF_R))
    main, again = rows[rows["rerun_id"] == 0], rows[rows["rerun_id"] == 1]
    n_boot, seed = settings["n_boot"], cfg["splits"]["seed"]
    out = []
    for benchmark, bench in main.groupby("benchmark"):
        parts = {split: bench[bench["split"] == split] for split in (select_split, report_split)}
        empty = [split for split, part in parts.items() if part.empty]
        if empty:
            raise ValueError(f"{benchmark}: no grid rows of split {empty}")
        select_values, select_costs = _arrays(parts[select_split], cells)
        values, costs = _arrays(parts[report_split], cells)
        bench_again = again[again["benchmark"] == benchmark]
        if not bench_again.empty:
            rerun_main = bench.merge(bench_again[ITEM].drop_duplicates(), on=ITEM)
            noise_values, noise_costs = _arrays(rerun_main, cells)
            again_values, again_costs = _arrays(bench_again, cells)
            if len(noise_values) != len(again_values):
                raise ValueError(f"{benchmark}: some re-run items have no main run")
        for frac in settings["cap_fracs"]:
            cap = frac * costs[:, ref].mean()
            row = {"benchmark": benchmark, "cap_frac": frac, "cap_gflops": cap / 1e9, "n_items": len(values),
                   "n_select": len(select_values), "n_noise": 0 if bench_again.empty else len(noise_values)}
            best = best_fixed_cell(select_values, select_costs, costs, cap)
            picks = oracle_choices(values, costs, cap)
            row["oracle_eligible"] = 100.0 * _eligible_oracle(values, costs, cap)
            if picks is not None:
                item = np.arange(len(values))
                row.update(oracle=100.0 * values[item, picks].mean(), oracle_gflops=costs[item, picks].mean() / 1e9)
            gain = None
            if best is not None:  # then the oracle is feasible too: "this cell for every item" fits the cap
                gap = 100.0 * (values[np.arange(len(values)), picks] - values[:, best])
                row.update(best_k=cells[best][0], best_R=cells[best][1], best_fixed=100.0 * values[:, best].mean(),
                           best_fixed_gflops=costs[:, best].mean() / 1e9)
                row["oracle_gain"], row["oracle_gain_low"], row["oracle_gain_high"] = paired_bootstrap_ci(
                    100.0 * values[np.arange(len(values)), picks], 100.0 * values[:, best], n_boot, seed)
            if not bench_again.empty:
                gain = noise_oracle_gain(noise_values, noise_costs, again_values, again_costs,
                                         frac * noise_costs[:, ref].mean())
            if gain is not None:
                row["noise_oracle"], row["noise_oracle_low"], row["noise_oracle_high"] = paired_bootstrap_ci(
                    100.0 * gain, np.zeros(len(gain)), n_boot, seed)
            if best is not None and gain is not None:
                row["headroom"], row["headroom_low"], row["headroom_high"] = diff_of_means_ci(
                    gap, 100.0 * gain, n_boot, seed)
            out.append(row)
    return pd.DataFrame(out, columns=HEADROOM_COLUMNS)


HEADROOM_COLUMNS = [
    "benchmark", "cap_frac", "cap_gflops", "n_items", "n_select", "n_noise",
    "best_k", "best_R", "best_fixed", "best_fixed_gflops",
    "oracle", "oracle_gflops", "oracle_eligible",
    "oracle_gain", "oracle_gain_low", "oracle_gain_high",
    "noise_oracle", "noise_oracle_low", "noise_oracle_high",
    "headroom", "headroom_low", "headroom_high",
]
