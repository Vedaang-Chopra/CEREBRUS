"""S14 headroom gate: oracle - best fixed - noise oracle, with the noise taken as the larger of two measurements."""
from pathlib import Path

import numpy as np
import pandas as pd

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


def headroom_gate(cfg: dict, root: str | Path, variant: str = "base", name: str = "grid") -> pd.DataFrame:
    """Per benchmark at `headroom.headline_frac`: headroom = oracle - best_fixed - max(re-run noise, batch noise).

    Dev numbers (`cfg["headroom"]`); best fixed is chosen per benchmark on ctrl_train, as in nb08 (T2), so it is not
    the pooled cell of the headline. Both noise numbers are listed. The batch noise is the at-cap value, or the
    uncapped one when the two Step-2b cells cannot fit the cap (`noise_source` says which was used).
    """
    settings = cfg["headroom"]
    frac, split = settings["headline_frac"], settings["report_split"]
    grid = pd.read_parquet(grid_dir(cfg, root, name) / "grid.parquet")  # main and re-run rows
    rows = grid[grid["split"] != "test"]  # this gate is a dev measurement
    base = headroom_table(rows, cfg, variant)
    base = base[np.isclose(base["cap_frac"], frac)]
    batch = load_batch_rows(cfg, root, variant, name)
    main = rows[(rows["model_variant"] == variant) & (rows["rerun_id"] == 0)]
    out = []
    for row in base.itertuples():
        record = {"benchmark": row.benchmark, "cap_frac": frac, "best_fixed": row.best_fixed, "oracle": row.oracle,
                  "noise_rerun": row.noise_oracle, "noise_batch_at_cap": np.nan, "noise_batch_uncapped": np.nan,
                  "n_batch": 0, "noise_used": row.noise_oracle, "noise_source": "re-run",
                  "headroom": row.headroom, "headroom_low": row.headroom_low, "headroom_high": row.headroom_high}
        if batch is not None:
            noise = batch_noise(*batch, row.benchmark, frac)
            record["n_batch"] = noise["n"]
            for key in ("at_cap", "uncapped"):
                if noise[key] is not None:
                    record[f"noise_batch_{key}"] = noise[key].mean()
            key = "at_cap" if noise["at_cap"] is not None else "uncapped"
            value = noise[key].mean()
            if not np.isnan(row.best_fixed) and not value <= row.noise_oracle:  # NaN re-run noise: batch is used
                cells = grid_cells(main)
                _, values, costs, cost_ref = cell_arrays(main[main["benchmark"] == row.benchmark], split, cells)
                picks = oracle_choices(values, costs, frac * cost_ref.mean())
                gap = 100.0 * (values[np.arange(len(values)), picks] - values[:, cells.index((row.best_k, row.best_R))])
                record.update(noise_used=value, noise_source=f"batch ({key})")
                record["headroom"], record["headroom_low"], record["headroom_high"] = diff_of_means_ci(
                    gap, noise[key], settings["n_boot"], cfg["splits"]["seed"])
        if np.isnan(record["noise_used"]):
            record["noise_source"] = "none"
        record["met"] = bool(record["headroom"] >= settings["min_pts"])
        record["split"] = f"{split} (optimistic: not the test split)"
        out.append(record)
    return pd.DataFrame(out)
