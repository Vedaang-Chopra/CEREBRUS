"""Pareto sweep (S5b, nb07 step 7): every method at every k, cls_iqr at every m; S5 rows of the same hash are reused."""
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from adaloop.orchestration.cell_runs import cell_lines, cell_status, run_batch1_cells
from adaloop.orchestration.compare import row_latency
from adaloop.orchestration.evaluate import read_results
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM


def _name(method: str, k: float) -> str:
    return f"{method}_k{round(100 * k)}"


def pareto_cells(cfg: dict) -> list[dict]:
    """The cells of `cfg["pareto"]`: name, method, m (cls_iqr only) and the config keys the cell sets.

    The full model (`none`, k = 1) once, every method at every k below 1 (at k = 1 each gives the full model's
    outputs), then cls_iqr at every m. cls_iqr's m sits under `compression.cls_iqr` of its cell config only, so it
    is part of that cell's hash and of no other.
    """
    pareto = cfg["pareto"]
    budget = [("none", 1.0)] + [(method, k) for method in pareto["methods"] for k in pareto["ks"] if k < 1]
    cells = [{"cell": _name(method, k), "method": method, "m": None, "keys": {"compressor": method, "k": k}}
             for method, k in budget]
    cells += [{"cell": f"cls_iqr_m{m}", "method": "cls_iqr", "m": m,
               "keys": {"compressor": "cls_iqr", "k": 1.0,
                        "compression": {**cfg["compression"], "cls_iqr": {"m": m}}}}
              for m in pareto["ms"]]
    return cells


def run_pareto(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cfg: dict, out_dir: str | Path,
               reuse_dir: str | Path | None = None, batch_size: int = 1) -> list[dict]:
    """Every cell on every benchmark's items (resumable; `batch_size` > 1 batches generation, accuracy only); return the rows with FLOPs and latency.

    `reuse_dir` is an S5 comparison folder: rows there with a cell's config hash are read, not run. Every answer found
    neither there nor in `out_dir` is run.
    """
    return run_batch1_cells(vlms, items, pareto_cells(cfg), cfg, cfg["pareto"]["R"], out_dir, reuse_dir, batch_size)


def pareto_status(items: Mapping[str, Sequence[Item]], cfg: dict, out_dir: str | Path,
                  reuse_dir: str | Path | None = None) -> list[dict]:
    """`cell_status` of the sweep's cells: what `run_pareto` with the same arguments would find and still has to run."""
    return cell_status(items, pareto_cells(cfg), cfg, cfg["pareto"]["R"], out_dir, reuse_dir)


def pareto_seconds_per_answer(cfg: dict, timing_dir: str | Path) -> tuple[float, str]:
    """(seconds per answer, where it comes from), for a time estimate.

    The median of the batch-1 rows at `res_px` in `timing_dir`; without any, `pareto.seconds_per_answer` of the config.
    """
    rows = [row for path in sorted(Path(timing_dir).glob("*.jsonl")) for row in read_results(path)]
    times = [row_latency(row) for row in rows if row["res_px"] == cfg["res_px"] and row["batch_size"] == 1]
    if times:
        return float(np.median(times)), f"median of {len(times)} batch-1 rows at {cfg['res_px']} px in {timing_dir}"
    return cfg["pareto"]["seconds_per_answer"], f"config: {cfg['pareto']['seconds_source']}"


def pareto_table(rows: Sequence[dict], latency_rows: Sequence[dict], cfg: dict) -> list[dict]:
    """One line per cell and benchmark (and "all"): cell, method, m, then the `comparison_table` columns.

    `k` is the mean kept / full, `n_vis` the mean kept count (the point of cls_iqr). `latency_from` is "reused" when the
    timing comes from the S5 files, "run" when from this sweep's latency pass ("reused+run": both), None without timing.
    """
    table = []
    for cell in pareto_cells(cfg):
        own = [row for row in rows if row["cell"] == cell["cell"]]
        if not own:
            continue
        timed = [row for row in latency_rows if row["cell"] == cell["cell"]]
        source = "+".join(sorted({row["source"] for row in timed}))
        for line in cell_lines(own, timed):
            table.append({"cell": cell["cell"], **line, "method": cell["method"], "m": cell["m"],
                          "latency_from": source if line["latency_n"] else None})
    return table
