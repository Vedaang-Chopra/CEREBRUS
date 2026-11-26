"""Resolution check (nb04 step 5): a sharper image with DivPrune against 512 px with every token.

Batch 1 throughout, so the same function serves the accuracy run (several copies) and the latency pass (one copy).
One JSONL per (benchmark, cell); a cell the S5 comparison already ran (same config hash) is read from its files.
"""
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from adaloop.core.cost.flops import cost_params, row_flops
from adaloop.core.eval.bootstrap import paired_bootstrap_ci
from adaloop.orchestration.batching import input_key
from adaloop.orchestration.cells import generate_cell, map_copies
from adaloop.orchestration.compare import comparison_table, row_latency
from adaloop.orchestration.compress import compress
from adaloop.orchestration.evaluate import evaluate_item, read_results, result_row
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash
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


def _answer(vlm: LoadedVLM, item: Item, cell_cfg: dict, R: int) -> dict:
    """One row: compress (exact count for a matched cell), generate at batch 1, score."""
    n_keep = None
    if "match_res_px" in cell_cfg:  # the item's own full count at the smaller size
        n_keep = input_key(vlm, item, {**cell_cfg, "res_px": cell_cfg["match_res_px"]})[0]
    inputs, info = compress(vlm, item_inputs(vlm, item, cell_cfg), item, cell_cfg.get("k", 1.0), cell_cfg,
                            n_keep=n_keep)
    gen = generate_cell(vlm, [inputs], cell_cfg, R)[0]
    return result_row(item, cell_cfg, R, gen, batch_size=1, info=info)


def run_resolution(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cfg: dict,
                   out_dir: str | Path, reuse_dir: str | Path | None = None) -> list[dict]:
    """Every cell on every benchmark's items at batch 1 (resumable); return the rows with FLOPs and latency.

    `reuse_dir` holds `run_comparison` files: rows of a cell with the same config hash are taken from there and never
    run again (`source` "reused"). Files are not named by the item count, so a sample run's rows count in the full run.
    """
    R = cfg["resolution"]["R"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = cost_params(vlms[0].model)
    gpu_type, commit = torch.cuda.get_device_name(0), git_hash()
    rows = []
    for benchmark, bench_items in items.items():
        ### Step-1:- What is already there, per cell: reused rows first, then this folder's own
        cells = resolution_cells(cfg)
        for cell in cells:
            cell["cfg"] = {**cfg, **cell["keys"]}
            digest = config_hash(cell["cfg"])
            cell["path"] = out_dir / f"{benchmark}_{cell['cell']}_R{R}_{digest}.jsonl"
            old = []
            if reuse_dir is not None:  # the file name pattern of run_comparison
                method, k = cell["keys"]["compressor"], cell["keys"].get("k")
                if k is not None:
                    pattern = f"{benchmark}_{method}_k{round(100 * k)}_R{R}_n*_{digest}.jsonl"
                    old = [row for path in sorted(Path(reuse_dir).glob(pattern)) for row in read_results(path)]
            own = read_results(cell["path"]) if cell["path"].exists() else []
            if own:  # rewrite clean lines before appending
                cell["path"].write_text("".join(json.dumps(row) + "\n" for row in own))
            if {row["config_hash"] for row in old + own} - {digest}:
                raise ValueError(f"{cell['cell']} {benchmark}: rows from another config than {digest}")
            cell["done"] = {row["item_id"]: {**row, "source": "run"} for row in own}
            cell["done"].update({row["item_id"]: {**row, "source": "reused"} for row in old})

        ### Step-2:- Run what is missing, one item per copy at a time
        pending = [item for item in bench_items if any(item.item_id not in cell["done"] for cell in cells)]
        n_missing = sum(item.item_id not in cell["done"] for item in bench_items for cell in cells)
        print(f"[{benchmark}] {len(bench_items)} items x {len(cells)} cells | {n_missing} answers to run | "
              f"{len(vlms)} copies")

        def work(vlm: LoadedVLM, j: int, item: Item) -> list[tuple[dict, dict]]:
            out = []
            for cell in cells:
                if item.item_id not in cell["done"]:
                    meta = {"copy": j, "n_copies": len(vlms), "config_hash": config_hash(cell["cfg"]),
                            "git_hash": commit, "gpu_type": gpu_type, "seed": cfg["splits"]["seed"]}
                    out.append((cell, {**_answer(vlm, item, cell["cfg"], R), **meta}))
            return out

        for n, (_, results) in enumerate(map_copies(vlms, pending, work), 1):
            for cell, row in results:  # only this thread writes
                with cell["path"].open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                cell["done"][row["item_id"]] = {**row, "source": "run"}
            if n % 10 == 0 or n == len(pending):
                print(f"[{benchmark}] {n}/{len(pending)} items done")

        for cell in cells:
            for item in bench_items:
                row = cell["done"][item.item_id]
                rows.append({**row, "cell": cell["cell"], **row_flops(params, row), "latency_s": row_latency(row)})
    return rows


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
        timed = [{**row, "k": 0} for row in latency_rows if row["cell"] == name]
        # k = 0: one group per cell, although a matched cell's k differs from item to item
        for line in comparison_table([{**row, "k": 0} for row in own], timed):
            group = [row for row in own if line["benchmark"] in (row["benchmark"], "all")]
            table.append({"cell": name, "setting": cell["setting"], "res_px": cell["keys"]["res_px"], **line,
                          "k": float(np.mean([row["k"] for row in group]))})
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
