"""Compression comparison (S5): every method x k at one loop count; accuracy, FLOPs and latency per cell."""
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from adaloop.core.cost.flops import cost_params, row_flops
from adaloop.core.eval.bootstrap import paired_bootstrap_ci
from adaloop.orchestration.evaluate import read_results, run_evaluation
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash


def comparison_cells(cfg: dict) -> list[tuple[str, float]]:
    """(method, k): the full model once, then every method at every k below 1.

    At k = 1 every method gives the full model's outputs (nb07 step 3), so it is not run per method.
    """
    compare = cfg["compare"]
    return [("none", 1.0)] + [(method, k) for method in compare["methods"] for k in compare["ks"]]


def run_comparison(vlms: Sequence[LoadedVLM], items: Mapping[str, Sequence[Item]], cfg: dict,
                   out_dir: str | Path) -> list[dict]:
    """Run every cell on every benchmark's items (resumable, one JSONL per cell); return all rows with FLOPs.

    With several copies the times include contention between them; compare times only within one run.
    """
    R = cfg["compare"]["R"]
    params = cost_params(vlms[0].model)
    rows = []
    cells = comparison_cells(cfg)
    for n, (method, k) in enumerate(cells, 1):
        cell_cfg = {**cfg, "compressor": method, "k": k}
        for benchmark, bench_items in items.items():
            name = f"{benchmark}_{method}_k{round(100 * k)}_R{R}_n{len(bench_items)}_{config_hash(cell_cfg)}.jsonl"
            out = Path(out_dir) / name
            print(f"[{n}/{len(cells)}] {method} k={k} {benchmark} | {name}")
            run_evaluation(None, bench_items, cell_cfg, R, out, resume=True, vlms=vlms)
            wanted = {item.item_id for item in bench_items}
            for row in read_results(out):
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

    return [
        {"method": method, "k": k, "benchmark": benchmark, "n": len(group),
         "accuracy": 100.0 * mean(group, "correct"), "n_vis": mean(group, "n_vis"),
         "n_vis_full": mean(group, "n_vis_full"), "gflops_total": mean(group, "flops_total") / 1e9,
         "gflops_vit": mean(group, "flops_vit") / 1e9, "gflops_compression": mean(group, "flops_compression") / 1e9,
         "latency_s": mean(group, "latency_s"), "compression_s": mean(group, "compression_s"),
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
