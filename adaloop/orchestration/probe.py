"""Headroom probe: a few grid cells on extra datasets, and what it says about where more passes or tokens pay off.

Settings: `cfg["extra_benchmarks"]` (configs/probe.yaml). The run itself is `run_grid` with `ks` / `Rs`.
"""
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from adaloop.data.extra import PROBE, load_extra
from adaloop.orchestration.grid import rerun_items
from adaloop.schemas.items import Item


def probe_items(cfg: dict, root: str | Path) -> list[Item]:
    """The probe questions of every extra dataset, in a fixed order."""
    return [item for name in cfg["extra_benchmarks"]["sets"] for item in load_extra(cfg, name, PROBE, root=root)]


def probe_rerun_items(items: Sequence[Item], cfg: dict) -> list[Item]:
    """`extra_benchmarks.n_rerun` items of each dataset, run a second time."""
    n, seed = cfg["extra_benchmarks"]["n_rerun"], cfg["splits"]["seed"]
    return [item for name in dict.fromkeys(item.benchmark for item in items)
            for item in rerun_items([i for i in items if i.benchmark == name], n, seed)]


def probe_report(grid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per dataset from a merged probe table; accuracies and shares in %, gaps in points.

    - acc_k<k>_R<R>: accuracy of each cell (main run).
    - depth_change: items whose correctness is not the same at every R, at k = `depth_k`.
      depth_noise: re-run items x those cells whose correctness differs between the two runs.
      depth_headroom = depth_change - depth_noise.
    - look_gain: items correct at k = `depth_k` and wrong at the smaller k, both at R = `look_R`.
      look_noise: the same re-run flip rate on those two cells. look_headroom = look_gain - look_noise.
    - oracle: each item takes its best of the cells. oracle_gain = oracle - the best single cell (not noise corrected).
    """
    extra = cfg["extra_benchmarks"]
    ks, Rs, full_k, look_R = extra["ks"], extra["Rs"], extra["depth_k"], extra["look_R"]
    small_k = min(ks)
    cells = [(k, R) for k in ks for R in Rs]
    out = []
    for name, rows in grid[grid["model_variant"] == "base"].groupby("benchmark"):
        rows = rows.assign(right=(rows["correct"] > 0.5).astype(float))
        main = rows[rows["rerun_id"] == 0].pivot(index="item_id", columns=["k", "R"], values="right")
        main = main.reindex(columns=pd.MultiIndex.from_tuples(cells))
        if main.isna().any().any():
            raise ValueError(f"{name}: some item lacks one of the cells {cells}; finish the probe run")
        again = rows[rows["rerun_id"] == 1].pivot(index="item_id", columns=["k", "R"], values="right")
        if again.empty:
            raise ValueError(f"{name}: no re-run rows (rerun_id = 1)")
        again = again.reindex(columns=main.columns)
        if again.isna().any().any():
            raise ValueError(f"{name}: some re-run item lacks one of the cells {cells}; finish the probe run")
        flips = again != main.loc[again.index]  # (re-run items, cells)

        def flip_pct(wanted: list[tuple]) -> float:
            return 100.0 * flips[wanted].to_numpy().mean()

        depth = main[[(full_k, R) for R in Rs]]
        full, small = main[full_k, look_R], main[small_k, look_R]
        row = {"benchmark": name, "n_items": len(main), "n_rerun": len(again)}
        row.update({f"acc_k{k}_R{R}": 100.0 * main[k, R].mean() for k, R in cells})
        row["depth_change"] = 100.0 * (depth.max(axis=1) != depth.min(axis=1)).mean()
        row["depth_noise"] = flip_pct([(full_k, R) for R in Rs])
        row["depth_headroom"] = row["depth_change"] - row["depth_noise"]
        row["look_gain"] = 100.0 * ((full == 1) & (small == 0)).mean()
        row["look_noise"] = flip_pct([(small_k, look_R), (full_k, look_R)])
        row["look_headroom"] = row["look_gain"] - row["look_noise"]
        row["best_cell"] = "k{}_R{}".format(*cells[int(main.mean().to_numpy().argmax())])
        row["best_cell_acc"] = 100.0 * main.mean().max()
        row["oracle"] = 100.0 * main.max(axis=1).mean()
        row["oracle_gain"] = row["oracle"] - row["best_cell_acc"]
        out.append(row)
    return pd.DataFrame(out)
