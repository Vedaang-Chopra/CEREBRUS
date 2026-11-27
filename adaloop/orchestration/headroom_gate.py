"""S14 headroom gate: oracle - best fixed - noise oracle, with the noise taken as the larger of two measurements.

Best fixed is the hull mix of the headline (`hull_mix`); the single best cell (nb08's T2) is printed next to it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from adaloop.core.controller.baselines import hull_mix
from adaloop.core.cost.flops import REF_R
from adaloop.core.eval.bootstrap import diff_of_means_ci
from adaloop.core.eval.oracle import noise_oracle_gain, oracle_choices
from adaloop.orchestration.analysis import headroom_table
from adaloop.orchestration.cell_policies import cell_arrays, grid_cells
from adaloop.orchestration.grid import grid_dir


def load_batch_rows(cfg: dict, root: str | Path, variant: str, name: str = "grid") -> tuple | None:
    """(batch-1 rows, grid-batch rows) of nb08 Step-2b; None with a printed line if they are not there."""
    size = cfg["grid_batch"]["size"]
    folder = grid_dir(cfg, root, name) / f"b1_vs_b{size}"
    paths = [folder / f"b{n}" / "grid.parquet" for n in (1, size)]
    if not all(path.is_file() for path in paths):
        print(f"NOTE batch noise not available (nb08 Step-2b rows missing under {folder}): re-run noise only")
        return None
    return tuple(rows[(rows["model_variant"] == variant) & (rows["rerun_id"] == 0) & (rows["split"] == "dev")]
                 for rows in map(pd.read_parquet, paths))


def batch_noise(b1: pd.DataFrame, bn: pd.DataFrame, benchmark: str, frac: float) -> dict:
    """Noise oracle between batch 1 and the grid batch, on the items and cells Step-2b ran, per item in points.

    `at_cap`: under mean FLOPs <= frac x cost_ref (None if these few cells cannot fit it); `uncapped`: free choice.
    Limit: Step-2b has two cells (k = 25% and 100% at R = 3) on dev items, so the oracle has far fewer lucky picks
    than over the 15 cells of the grid; this number can understate the batch noise of the grid oracle.
    """
    on = ["benchmark", "item_id", "k", "R"]
    pair = b1[b1["benchmark"] == benchmark].merge(bn[on + ["correct"]], on=on, suffixes=("", "_bn"),
                                                  validate="one_to_one")
    if pair.empty or (1.0, REF_R) not in grid_cells(pair):
        raise ValueError(f"{benchmark}: Step-2b rows of the two batch sizes do not match, or lack k=1 R={REF_R}")
    cells = grid_cells(pair)
    _, values, costs, cost_ref = cell_arrays(pair, "dev", cells)
    _, again, _, _ = cell_arrays(pair.drop(columns="correct").rename(columns={"correct_bn": "correct"}), "dev", cells)
    out = {"n": len(values), "cells": len(cells)}
    for name, cap in (("at_cap", frac * cost_ref.mean()), ("uncapped", np.inf)):
        gain = noise_oracle_gain(values, costs, again, costs, cap)
        out[name] = None if gain is None else 100.0 * gain
    return out


def rerun_noise(rows: pd.DataFrame, variant: str, benchmark: str, frac: float) -> np.ndarray | None:
    """Per re-run item the noise-oracle gain in points, over all cells (as `headroom_table` computes it)."""
    part = rows[(rows["model_variant"] == variant) & (rows["benchmark"] == benchmark)]
    again = part[part["rerun_id"] == 1]
    if again.empty:
        return None
    first = part[part["rerun_id"] == 0].merge(again[["item_id"]].drop_duplicates(), on="item_id")
    cells = grid_cells(first)
    # re-run items come from several splits: they are pooled under one name for the arrays
    (_, values, costs, cost_ref), (_, values_again, costs_again, _) = (
        cell_arrays(run.assign(split="dev"), "dev", cells) for run in (first, again))
    gain = noise_oracle_gain(values, costs, values_again, costs_again, frac * cost_ref.mean())
    return None if gain is None else 100.0 * gain


def headroom_gate(cfg: dict, root: str | Path, variant: str = "base", name: str = "grid",
                  file: str = "grid.parquet") -> pd.DataFrame:
    """Per benchmark at `headroom.headline_frac`: headroom = oracle - best_fixed - max(re-run noise, batch noise).

    Dev numbers (`cfg["headroom"]`). `best_fixed` is the hull mix built per benchmark with the headline's data rules:
    accuracy from ctrl_train, costs of the report items, at mean FLOPs = the cap (above its most accurate cell it costs
    less). `headroom`, its CI and `met` use it. `best_fixed_cell`, `headroom_cell`, ... are the same with the single best
    cell <= cap, as in nb08 (T2). Both noise numbers are listed; the batch noise is the at-cap value, or the uncapped
    one when the two Step-2b cells cannot fit the cap (`noise_source` says which was used).
    """
    settings = cfg["headroom"]
    frac, split = settings["headline_frac"], settings["report_split"]
    grid = pd.read_parquet(grid_dir(cfg, root, name) / file)  # main and re-run rows
    rows = grid[grid["split"] != "test"]  # this gate is a dev measurement
    table = headroom_table(rows, cfg, variant)
    table = table[np.isclose(table["cap_frac"], frac)]
    batch = load_batch_rows(cfg, root, variant, name)
    main = rows[(rows["model_variant"] == variant) & (rows["rerun_id"] == 0)]
    out = []
    for row in table.itertuples():
        bench = main[main["benchmark"] == row.benchmark]
        cells = grid_cells(bench)
        _, select_values, _, _ = cell_arrays(bench, settings["select_split"], cells)
        _, values, costs, cost_ref = cell_arrays(bench, split, cells)
        cap = frac * cost_ref.mean()
        ### Step-1:- Noise: the larger of the two measurements
        gain = rerun_noise(rows, variant, row.benchmark, frac)
        record = {"benchmark": row.benchmark, "cap_frac": frac, "oracle": row.oracle, "best_fixed": np.nan,
                  "best_fixed_cell": row.best_fixed, "noise_rerun": row.noise_oracle, "noise_batch_at_cap": np.nan,
                  "noise_batch_uncapped": np.nan, "n_batch": 0, "noise_used": row.noise_oracle,
                  "noise_source": "none" if gain is None else "re-run"}
        if batch is not None:
            noise = batch_noise(*batch, row.benchmark, frac)
            record["n_batch"] = noise["n"]
            for key in ("at_cap", "uncapped"):
                if noise[key] is not None:
                    record[f"noise_batch_{key}"] = noise[key].mean()
            key = "at_cap" if noise["at_cap"] is not None else "uncapped"
            if gain is None or noise[key].mean() > gain.mean():
                gain = noise[key]
                record.update(noise_used=gain.mean(), noise_source=f"batch ({key})")
        ### Step-2:- Headroom over the hull mix and over the single cell
        picks = oracle_choices(values, costs, cap)
        weights = hull_mix(select_values, costs, cap)
        fixed = {"": None if weights is None else values @ weights,
                 "_cell": None if np.isnan(row.best_fixed) else values[:, cells.index((row.best_k, row.best_R))]}
        if weights is not None:
            record["best_fixed"] = 100.0 * fixed[""].mean()
        for suffix, scores in fixed.items():
            head = low = high = np.nan
            if scores is not None and picks is not None and gain is not None:
                gap = 100.0 * (values[np.arange(len(values)), picks] - scores)
                head, low, high = diff_of_means_ci(gap, gain, settings["n_boot"], cfg["splits"]["seed"])
            record.update({f"headroom{suffix}": head, f"headroom{suffix}_low": low, f"headroom{suffix}_high": high,
                           f"met{suffix}": bool(head >= settings["min_pts"])})
        record["split"] = f"{split} (optimistic: not the test split)"
        out.append(record)
    return pd.DataFrame(out)
