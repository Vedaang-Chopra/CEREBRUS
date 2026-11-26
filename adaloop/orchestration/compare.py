"""Compression comparison (S5): every method x k at one loop count; accuracy, FLOPs and latency per cell."""
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from adaloop.core.cost.flops import cost_params, row_flops
from adaloop.core.eval.bootstrap import paired_bootstrap_ci
from adaloop.orchestration.cells import ChunkInputs, generate_cell, map_copies
from adaloop.orchestration.evaluate import read_results, result_row
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash


def comparison_cells(cfg: dict) -> list[tuple[str, float]]:
    """(method, k): the full model once, then every method at every k below 1.

    At k = 1 every method gives the full model's outputs (nb07 step 3), so it is not run per method.
    """
    compare = cfg["compare"]
    return [("none", 1.0)] + [(method, k) for method in compare["methods"] for k in compare["ks"]]


def _cell_path(out_dir: Path, benchmark: str, method: str, k: float, R: int, n: int, cell_cfg: dict) -> Path:
    """One JSONL per (benchmark, method, k). Batch size and copy count are not part of the name or the hash."""
    return out_dir / f"{benchmark}_{method}_k{round(100 * k)}_R{R}_n{n}_{config_hash(cell_cfg)}.jsonl"


def _open_cells(items: Sequence[Item], cfg: dict, out_dir: Path, benchmark: str, R: int) -> tuple[dict, dict]:
    """({cell: path}, {cell: item ids already done}); unreadable lines are dropped, other configs are an error."""
    paths, done = {}, {}
    for method, k in comparison_cells(cfg):
        cell_cfg = {**cfg, "compressor": method, "k": k}
        path = _cell_path(out_dir, benchmark, method, k, R, len(items), cell_cfg)
        rows = read_results(path) if path.exists() else []
        stale = {row["config_hash"] for row in rows} - {config_hash(cell_cfg)}
        if stale:
            raise ValueError(f"{path} has rows from other configs {stale}")
        if rows:  # rewrite clean lines before appending
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[method, k], done[method, k] = path, {row["item_id"] for row in rows}
    return paths, done


def run_comparison(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cfg: dict,
                   out_dir: str | Path, batch_size: int | None = None) -> list[dict]:
    """Run every cell on every benchmark's items (resumable, one JSONL per cell); return all rows with FLOPs.

    Items are taken `batch_size` at a time (default `eval_batch_size`): the vision tower runs once per item and every
    cell reuses it (not `none` and `lowres`), each batch holds one cell. Rows already on disk are kept, whatever
    batch size or copy count wrote them. Batched rows have no time to first token and an averaged `generate_s`; with
    several copies the times include contention.
    """
    R = cfg["compare"]["R"]
    batch_size = batch_size or cfg["eval_batch_size"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = cost_params(vlms[0].model)
    gpu_type, commit = torch.cuda.get_device_name(0), git_hash()
    rows = []
    for benchmark, bench_items in items.items():
        paths, done = _open_cells(bench_items, cfg, out_dir, benchmark, R)
        pending = [item for item in bench_items if any(item.item_id not in done[cell] for cell in paths)]
        print(f"[{benchmark}] {len(bench_items)} items, {len(pending)} with cells to run | {len(paths)} cells | "
              f"batch {batch_size}, {len(vlms)} copies")
        chunks = [pending[first : first + batch_size] for first in range(0, len(pending), batch_size)]

        def work(vlm: LoadedVLM, j: int, chunk: list[Item]) -> list[tuple[tuple, dict]]:
            inputs = ChunkInputs(vlm, chunk, cfg)
            out = []
            for method, k in paths:
                todo = [i for i, item in enumerate(chunk) if item.item_id not in done[method, k]]
                if not todo:
                    continue
                cell_cfg = {**cfg, "compressor": method, "k": k}
                prepared = [inputs.compressed(i, method, k) for i in todo]
                gens = generate_cell(vlm, [packed for packed, _ in prepared], cell_cfg, R)
                meta = {"copy": j, "n_copies": len(vlms), "config_hash": config_hash(cell_cfg), "git_hash": commit,
                        "gpu_type": gpu_type, "seed": cfg["splits"]["seed"]}
                for i, (_, info), gen in zip(todo, prepared, gens):
                    row = result_row(chunk[i], cell_cfg, R, gen, batch_size=len(todo), info=info)
                    out.append(((method, k), {**row, **meta}))
            return out

        seen = 0
        for n, (_, chunk_rows) in enumerate(map_copies(vlms, chunks, work)):
            for cell, row in chunk_rows:  # only this thread writes
                with paths[cell].open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            seen += len(chunks[n])
            print(f"[{benchmark}] {seen}/{len(pending)} items done")

        wanted = {item.item_id for item in bench_items}
        for path in paths.values():
            for row in read_results(path):
                if row["item_id"] in wanted:
                    rows.append({**row, **row_flops(params, row), "latency_s": row_latency(row)})
    return rows


def row_latency(row: dict) -> float:
    """Seconds for one answer: vision tower (when run outside generate), compression, generate."""
    return (row.get("vit_s") or 0.0) + row["compression_s"] + row["generate_s"]


def comparison_table(rows: Sequence[dict]) -> list[dict]:
    """One line per (method, k, benchmark), and per (method, k) over both benchmarks (benchmark "all")."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        for benchmark in (row["benchmark"], "all"):
            groups.setdefault((row["selector"], row["k"], benchmark), []).append(row)

    def mean(group: list[dict], key: str) -> float:
        return float(np.mean([r[key] for r in group]))

    def latency(group: list[dict]) -> float:
        # a batched row's time is the batch's divided by its size, not a latency
        single = [r["latency_s"] for r in group if r["batch_size"] == 1]
        return float(np.mean(single)) if single else float("nan")

    return [
        {"method": method, "k": k, "benchmark": benchmark, "n": len(group),
         "accuracy": 100.0 * mean(group, "correct"), "n_vis": mean(group, "n_vis"),
         "n_vis_full": mean(group, "n_vis_full"), "gflops_total": mean(group, "flops_total") / 1e9,
         "gflops_vit": mean(group, "flops_vit") / 1e9, "gflops_compression": mean(group, "flops_compression") / 1e9,
         "latency_s": latency(group), "latency_n": sum(r["batch_size"] == 1 for r in group),
         "compression_s": mean(group, "compression_s"),
         "parse_rate": 100.0 * mean(group, "parse_ok"),
         "fallback_rate": 100.0 * float(np.mean([r.get("fallback") is not None for r in group]))}
        for (method, k, benchmark), group in groups.items()
    ]


def pick_winner(rows: Sequence[dict], cfg: dict) -> tuple[str, list[dict]]:
    """Suggested compressor and the evidence: one line per method, best first.

    Per item, the score is the mean over the k < 1 cells. A method is tied with the best when the 95% paired
    bootstrap interval of (best - method) contains 0; the winner is the tied method with the lowest mean FLOPs.
    Needs the same items in every cell.
    """
    compare = cfg["compare"]
    scores: dict[str, dict[str, list[float]]] = {}
    flops: dict[str, list[float]] = {}
    for row in rows:
        if row["k"] < 1:
            item = (row["benchmark"], row["item_id"])
            scores.setdefault(row["selector"], {}).setdefault(item, []).append(row["correct"])
            flops.setdefault(row["selector"], []).append(row["flops_total"])
    if not scores:
        raise ValueError("no rows with k < 1")
    ids = sorted(next(iter(scores.values())))
    per_item = {}
    for method, by_item in scores.items():
        if sorted(by_item) != ids or any(len(v) != len(compare["ks"]) for v in by_item.values()):
            raise ValueError(f"{method} does not have every item at every k; finish the run first")
        per_item[method] = [float(np.mean(by_item[i])) for i in ids]
    best = max(per_item, key=lambda m: np.mean(per_item[m]))
    lines = []
    for method in per_item:
        diff, low, high = paired_bootstrap_ci(per_item[best], per_item[method], compare["n_boot"],
                                              cfg["splits"]["seed"])
        lines.append({"method": method, "accuracy_low_k": 100.0 * float(np.mean(per_item[method])),
                      "behind_best": 100.0 * diff, "ci_low": 100.0 * low, "ci_high": 100.0 * high,
                      "tied": method == best or low <= 0 <= high,
                      "gflops_total": float(np.mean(flops[method])) / 1e9})
    lines.sort(key=lambda line: -line["accuracy_low_k"])
    winner = min((line for line in lines if line["tied"]), key=lambda line: line["gflops_total"])["method"]
    return winner, lines
