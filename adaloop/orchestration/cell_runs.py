"""Cells answered at batch 1, one JSONL per (benchmark, cell), resumable, finished S5 rows reused.

A cell is {"cell": name, "keys": the config keys it sets}. Used by the resolution check and the Pareto sweep; batch 1
throughout, so the same runner serves the accuracy run (several copies) and the latency pass (one copy).
"""
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from adaloop.core.cost.flops import cost_params, row_flops
from adaloop.orchestration.batching import input_key
from adaloop.orchestration.cells import generate_cell, map_copies
from adaloop.orchestration.compare import comparison_table, row_latency
from adaloop.orchestration.compress import compress
from adaloop.orchestration.evaluate import read_results, result_row
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash


def _answer(vlm: LoadedVLM, item: Item, cell_cfg: dict, R: int) -> dict:
    """One row: compress (exact count for a matched cell), generate at batch 1, score."""
    n_keep = None
    if "match_res_px" in cell_cfg:  # the item's own full count at the smaller size
        n_keep = input_key(vlm, item, {**cell_cfg, "res_px": cell_cfg["match_res_px"]})[0]
    inputs, info = compress(vlm, item_inputs(vlm, item, cell_cfg), item, cell_cfg.get("k", 1.0), cell_cfg,
                            n_keep=n_keep)
    gen = generate_cell(vlm, [inputs], cell_cfg, R)[0]
    return result_row(item, cell_cfg, R, gen, batch_size=1, info=info)


def run_batch1_cells(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cells: Sequence[dict],
                     cfg: dict, R: int, out_dir: str | Path, reuse_dir: str | Path | None = None,
                     must_reuse: Collection[str] = ()) -> list[dict]:
    """Every cell on every benchmark's items at batch 1 (resumable); return the rows with cell, FLOPs and latency.

    `reuse_dir` holds `run_comparison` files: rows of a cell with the same config hash are taken from there and never
    run again (`source` "reused"). Cells named in `must_reuse` are never run: a missing row is an error. Files are not
    named by the item count, so a sample run's rows count in the full run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = cost_params(vlms[0].model)
    gpu_type, commit = torch.cuda.get_device_name(0), git_hash()
    rows = []
    for benchmark, bench_items in items.items():
        ### Step-1:- What is already there, per cell: reused rows first, then this folder's own
        todo = [{**cell, "cfg": {**cfg, **cell["keys"]}} for cell in cells]
        for cell in todo:
            digest = config_hash(cell["cfg"])
            cell["path"] = out_dir / f"{benchmark}_{cell['cell']}_R{R}_{digest}.jsonl"
            old = []
            if reuse_dir is not None:  # the file name pattern of run_comparison
                method, k = cell["keys"]["compressor"], cell["keys"].get("k")
                if k is not None:
                    pattern = f"{benchmark}_{method}_k{round(100 * k)}_R{R}_n*_{digest}.jsonl"
                    old = [row for path in sorted(Path(reuse_dir).glob(pattern)) for row in read_results(path)]
            if cell["cell"] in must_reuse:
                missing = {item.item_id for item in bench_items} - {row["item_id"] for row in old}
                if missing:
                    raise ValueError(f"{cell['cell']} {benchmark}: {len(missing)} items have no finished row with hash "
                                     f"{digest} in {reuse_dir}; this cell is reused, never run again")
            own = read_results(cell["path"]) if cell["path"].exists() else []
            if own:  # rewrite clean lines before appending
                cell["path"].write_text("".join(json.dumps(row) + "\n" for row in own))
            if {row["config_hash"] for row in old + own} - {digest}:
                raise ValueError(f"{cell['cell']} {benchmark}: rows from another config than {digest}")
            cell["done"] = {row["item_id"]: {**row, "source": "run"} for row in own}
            cell["done"].update({row["item_id"]: {**row, "source": "reused"} for row in old})

        ### Step-2:- Run what is missing, one item per copy at a time
        pending = [item for item in bench_items if any(item.item_id not in cell["done"] for cell in todo)]
        n_missing = sum(item.item_id not in cell["done"] for item in bench_items for cell in todo)
        print(f"[{benchmark}] {len(bench_items)} items x {len(todo)} cells | {n_missing} answers to run | "
              f"{len(vlms)} copies")

        def work(vlm: LoadedVLM, j: int, item: Item) -> list[tuple[dict, dict]]:
            out = []
            for cell in todo:
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

        for cell in todo:
            for item in bench_items:
                row = cell["done"][item.item_id]
                rows.append({**row, "cell": cell["cell"], **row_flops(params, row), "latency_s": row_latency(row)})
    return rows


def cell_lines(rows: Sequence[dict], latency_rows: Sequence[dict]) -> list[dict]:
    """`comparison_table` lines (per benchmark and "all") of one cell's rows.

    `k` is the mean kept / full over the line's items: within a cell it may differ from item to item.
    """
    # k = 0: one group per cell, whatever each row's k is
    lines = comparison_table([{**row, "k": 0} for row in rows], [{**row, "k": 0} for row in latency_rows])
    return [{**line, "k": float(np.mean([r["k"] for r in rows if line["benchmark"] in (r["benchmark"], "all")]))}
            for line in lines]
