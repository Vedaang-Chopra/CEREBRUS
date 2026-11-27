"""Cells answered at batch 1 (or in generation batches, for accuracy only), one JSONL per (benchmark, cell), resumable, S5 rows of the same config hash reused.

A cell is {"cell": name, "keys": the config keys it sets}. Used by the resolution check and the Pareto sweep. The same
runner serves the accuracy run (several copies; `batch_size` > 1 batches generation) and the batch-1 latency pass.
"""
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from adaloop.core.cost.flops import cost_params, row_flops
from adaloop.orchestration.batching import input_key, item_chunks
from adaloop.orchestration.cells import generate_cell, map_copies
from adaloop.orchestration.compare import comparison_table, row_latency
from adaloop.orchestration.compress import compress
from adaloop.orchestration.evaluate import read_results, result_row
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash


def _answers(vlm: LoadedVLM, items: Sequence[Item], cell_cfg: dict, R: int) -> list[dict]:
    """Rows of one cell for one generation batch: compress each item (exact count for a matched cell), generate, score.

    One item takes the batch-1 path, which times it.
    """
    prepared = []
    for item in items:
        n_keep = None
        if "match_res_px" in cell_cfg:  # the item's own full count at the smaller size
            n_keep = input_key(vlm, item, {**cell_cfg, "res_px": cell_cfg["match_res_px"]})[0]
        prepared.append(compress(vlm, item_inputs(vlm, item, cell_cfg), item, cell_cfg.get("k", 1.0), cell_cfg,
                                 n_keep=n_keep))
    gens = generate_cell(vlm, [inputs for inputs, _ in prepared], cell_cfg, R)
    return [result_row(item, cell_cfg, R, gen, batch_size=len(items), info=info)
            for item, gen, (_, info) in zip(items, gens, prepared)]


def _cell_files(benchmark: str, cell: dict, cfg: dict, R: int, out_dir: Path, reuse_dir: str | Path | None) -> dict:
    """The cell with its config, own file, and the finished rows: `old` from `reuse_dir`, `own` from `out_dir`."""
    cell_cfg = {**cfg, **cell["keys"]}
    digest = config_hash(cell_cfg)
    path = out_dir / f"{benchmark}_{cell['cell']}_R{R}_{digest}.jsonl"
    old = []
    method, k = cell["keys"]["compressor"], cell["keys"].get("k")
    if reuse_dir is not None and k is not None:  # the file name pattern of run_comparison
        pattern = f"{benchmark}_{method}_k{round(100 * k)}_R{R}_n*_{digest}.jsonl"
        old = [row for found in sorted(Path(reuse_dir).glob(pattern)) for row in read_results(found)]
    own = read_results(path) if path.exists() else []
    if {row["config_hash"] for row in old + own} - {digest}:
        raise ValueError(f"{cell['cell']} {benchmark}: rows from another config than {digest}")
    return {**cell, "cfg": cell_cfg, "path": path, "old": old, "own": own}


def cell_status(items: Mapping[str, Sequence[Item]], cells: Sequence[dict], cfg: dict, R: int, out_dir: str | Path,
                reuse_dir: str | Path | None = None) -> list[dict]:
    """Per (benchmark, cell), counted from the files (nothing is written): of `n_items` answers, how many are
    `reused` (in `reuse_dir`), `own` (only in `out_dir`) and `missing`."""
    status = []
    for benchmark, bench_items in items.items():
        ids = {item.item_id for item in bench_items}
        for cell in cells:
            files = _cell_files(benchmark, cell, cfg, R, Path(out_dir), reuse_dir)
            reused = ids & {row["item_id"] for row in files["old"]}
            own = ids & {row["item_id"] for row in files["own"]} - reused
            status.append({"benchmark": benchmark, "cell": cell["cell"], "n_items": len(ids), "reused": len(reused),
                           "own": len(own), "missing": len(ids) - len(reused) - len(own)})
    return status


def run_batch1_cells(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cells: Sequence[dict],
                     cfg: dict, R: int, out_dir: str | Path, reuse_dir: str | Path | None = None,
                     batch_size: int = 1) -> list[dict]:
    """Every cell on every benchmark's items (resumable); return the rows with cell, FLOPs and latency.

    `reuse_dir` holds `run_comparison` files: rows of a cell with the same config hash are taken from there and never
    run again (`source` "reused"); whatever is missing there and in `out_dir` is run. Files are not named by the item
    count, so a sample run's rows count in the full run.

    `batch_size` > 1 batches generation (accuracy only): the batches are the `item_chunks` of all the benchmark's items,
    one cell per batch, and only the items still missing that cell are in it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = cost_params(vlms[0].model)
    gpu_type, commit = torch.cuda.get_device_name(0), git_hash()
    rows = []
    for benchmark, bench_items in items.items():
        ### Step-1:- What is already there, per cell: reused rows first, then this folder's own
        todo = [_cell_files(benchmark, cell, cfg, R, out_dir, reuse_dir) for cell in cells]
        for cell in todo:
            if cell["own"]:  # rewrite clean lines before appending
                cell["path"].write_text("".join(json.dumps(row) + "\n" for row in cell["own"]))
            cell["done"] = {row["item_id"]: {**row, "source": "run"} for row in cell["own"]}
            cell["done"].update({row["item_id"]: {**row, "source": "reused"} for row in cell["old"]})

        ### Step-2:- Run what is missing, one item per copy at a time
        pending = [item for item in bench_items if any(item.item_id not in cell["done"] for cell in todo)]
        n_missing = sum(item.item_id not in cell["done"] for item in bench_items for cell in todo)
        print(f"[{benchmark}] {len(bench_items)} items x {len(todo)} cells | {n_missing} answers to run | "
              f"{len(vlms)} copies")

        def work(vlm: LoadedVLM, j: int, chunk: list[Item]) -> list[tuple[dict, dict]]:
            out = []
            for cell in todo:
                lacking = [item for item in chunk if item.item_id not in cell["done"]]
                if lacking:
                    meta = {"copy": j, "n_copies": len(vlms), "config_hash": config_hash(cell["cfg"]),
                            "git_hash": commit, "gpu_type": gpu_type, "seed": cfg["splits"]["seed"]}
                    out += [(cell, {**row, **meta}) for row in _answers(vlm, lacking, cell["cfg"], R)]
            return out

        if batch_size == 1:
            jobs = [[item] for item in pending]
        else:  # cut from all items, so a sample and the full run share their batches
            ids = {item.item_id for item in pending}
            jobs = [chunk for chunk in item_chunks(vlms[0], bench_items, cfg, batch_size)
                    if any(item.item_id in ids for item in chunk)]
        for n, (_, results) in enumerate(map_copies(vlms, jobs, work), 1):
            for cell, row in results:  # only this thread writes
                with cell["path"].open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                cell["done"][row["item_id"]] = {**row, "source": "run"}
            if n % 10 == 0 or n == len(jobs):
                print(f"[{benchmark}] {n}/{len(jobs)} {'items' if batch_size == 1 else 'batches'} done")

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
