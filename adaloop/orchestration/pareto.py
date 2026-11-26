"""Pareto sweep (S5b, nb07 step 7): every method at every k, cls_iqr at every m; the finished S5 cells are reused."""
from collections.abc import Mapping, Sequence
from pathlib import Path

from adaloop.orchestration.cell_runs import cell_lines, run_batch1_cells
from adaloop.orchestration.compare import comparison_cells
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


def pareto_reused_cells(cfg: dict) -> list[str]:
    """Names of the sweep's cells that the S5 comparison already ran (same method, k and R)."""
    if cfg["pareto"]["R"] != cfg["compare"]["R"]:
        return []
    names = {_name(method, k) for method, k in comparison_cells(cfg)}
    return [cell["cell"] for cell in pareto_cells(cfg) if cell["cell"] in names]


def run_pareto(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cfg: dict, out_dir: str | Path,
               reuse_dir: str | Path | None) -> list[dict]:
    """Every cell on every benchmark's items at batch 1 (resumable); return the rows with FLOPs and latency.

    `reuse_dir` is the finished S5 folder: its cells (`pareto_reused_cells`) are read from there and never run, and a
    missing S5 row is an error. Pass None only to run every cell from scratch.
    """
    must_reuse = pareto_reused_cells(cfg) if reuse_dir is not None else []
    return run_batch1_cells(vlms, items, pareto_cells(cfg), cfg, cfg["pareto"]["R"], out_dir, reuse_dir, must_reuse)


def pareto_table(rows: Sequence[dict], latency_rows: Sequence[dict], cfg: dict) -> list[dict]:
    """One line per cell and benchmark (and "all"): cell, method, m, then the `comparison_table` columns.

    `k` is the mean kept / full, `n_vis` the mean kept count (the point of cls_iqr). `latency_from` is "reused" when the
    timing comes from the S5 session, "run" when from this sweep's latency pass, None without timing.
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
