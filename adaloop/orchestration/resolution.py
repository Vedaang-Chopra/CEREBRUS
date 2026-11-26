"""Resolution check (nb04 step 5): a sharper image with DivPrune against 512 px with every token.

Batch 1 throughout, so the same function serves the accuracy run (several copies) and the latency pass (one copy).
One JSONL per (benchmark, cell); a cell the S5 comparison already ran (same config hash) is read from its files
(`cell_runs.run_batch1_cells`).
"""
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from adaloop.core.eval.bootstrap import paired_bootstrap_ci
from adaloop.orchestration.batching import input_key
from adaloop.orchestration.cell_runs import cell_lines, run_batch1_cells
from adaloop.orchestration.evaluate import evaluate_item
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.timing import peak_memory_gb


def resolution_cells(cfg: dict) -> list[dict]:
    """The cells of `cfg["resolution"]`: name, setting (full | matched | fixed_k) and the config keys it sets."""
    res = cfg["resolution"]
    cells = [{"cell": f"full_{px}", "setting": "full", "keys": {"compressor": "none", "res_px": px, "k": 1.0}}
             for px in res["res_px"]]
    cells += [{"cell": f"divprune_{px}_match{res['match_res_px']}", "setting": "matched",
               "keys": {"compressor": "divprune", "res_px": px, "match_res_px": res["match_res_px"]}}
              for px in res["matched_res_px"]]
    cells += [{"cell": f"divprune_{px}_k{round(100 * k)}", "setting": "fixed_k",
               "keys": {"compressor": "divprune", "res_px": px, "k": k}} for px, k in res["fixed_k"]]
    return cells


def resolution_token_counts(vlm: LoadedVLM, items: Sequence[Item], cfg: dict) -> list[dict]:
    """Full visual-token count of every item at every res_px (processor only, no model call)."""
    return [{"benchmark": item.benchmark, "item_id": item.item_id, "long_side": max(item.image.size),
             **{f"n_{px}": input_key(vlm, item, {**cfg, "res_px": px})[0] for px in cfg["resolution"]["res_px"]}}
            for item in items]


def resolution_memory(vlm: LoadedVLM, item: Item, cfg: dict) -> dict[int, float]:
    """Answer `item` with every token at the largest res_px; peak allocated GiB per GPU of this copy."""
    full = {**cfg, "compressor": "none", "k": 1.0, "res_px": max(cfg["resolution"]["res_px"])}
    evaluate_item(vlm, item, full, cfg["resolution"]["R"])  # generate resets the peak counters when it starts
    devices = range(torch.cuda.device_count()) if vlm.devices is None else vlm.devices
    return {i: peak_memory_gb([i]) for i in devices}


def run_resolution(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cfg: dict,
                   out_dir: str | Path, reuse_dir: str | Path | None = None) -> list[dict]:
    """Every cell on every benchmark's items at batch 1 (resumable); return the rows with FLOPs and latency.

    `reuse_dir` holds `run_comparison` files: rows of a cell with the same config hash are taken from there and never
    run again (`source` "reused"). Files are not named by the item count, so a sample run's rows count in the full run.
    """
    return run_batch1_cells(vlms, items, resolution_cells(cfg), cfg, cfg["resolution"]["R"], out_dir, reuse_dir)


def resolution_table(rows: Sequence[dict], latency_rows: Sequence[dict], cfg: dict) -> list[dict]:
    """`comparison_table` per cell (lines per benchmark and "all"), with cell, setting and res_px.

    `k` is the mean kept / full over the line's items; `fallback_rate` is the share (%) of matched items that had
    fewer tokens than asked for and kept them all.
    """
    table = []
    for cell in resolution_cells(cfg):
        name = cell["cell"]
        own = [row for row in rows if row["cell"] == name]
        if not own:
            continue
        for line in cell_lines(own, [row for row in latency_rows if row["cell"] == name]):
            table.append({"cell": name, "setting": cell["setting"], "res_px": cell["keys"]["res_px"], **line})
    return table


def resolution_gain(rows: Sequence[dict], cfg: dict, res_px: int) -> list[dict]:
    """Matched DivPrune at `res_px` minus full tokens at `match_res_px`, per benchmark and "all".

    Accuracy points with the paired 95% bootstrap interval over items; "beats" only when the interval is above 0.
    """
    match = cfg["resolution"]["match_res_px"]
    a_name, b_name = f"divprune_{res_px}_match{match}", f"full_{match}"
    scores = {name: {(r["benchmark"], r["item_id"]): r["correct"] for r in rows if r["cell"] == name}
              for name in (a_name, b_name)}
    a, b = scores[a_name], scores[b_name]
    if not a or a.keys() != b.keys():
        raise ValueError(f"{a_name} ({len(a)} items) and {b_name} ({len(b)}) must cover the same items")
    lines = []
    for benchmark in [*dict.fromkeys(bench for bench, _ in a), "all"]:
        ids = [key for key in a if benchmark in (key[0], "all")]
        diff, low, high = paired_bootstrap_ci([a[i] for i in ids], [b[i] for i in ids], cfg["compare"]["n_boot"],
                                              cfg["splits"]["seed"])
        verdict = "beats" if low > 0 else "worse" if high < 0 else "no clear difference"
        lines.append({"benchmark": benchmark, "n": len(ids), "pruned": a_name, "full": b_name,
                      "acc_pruned": 100.0 * float(np.mean([a[i] for i in ids])),
                      "acc_full": 100.0 * float(np.mean([b[i] for i in ids])),
                      "gain": 100.0 * diff, "ci_low": 100.0 * low, "ci_high": 100.0 * high, "verdict": verdict})
    return lines
