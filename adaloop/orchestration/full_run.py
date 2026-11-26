"""Full baseline run: shard the items of each benchmark x split, one resumable JSONL per shard and R.

Shared by `scripts/mvp.py full` and `notebooks/06_full_baseline`, so file names match.
"""
import time
from collections.abc import Sequence
from pathlib import Path

from adaloop.core.eval.summary import summarize_rows
from adaloop.data.gqa import load_gqa
from adaloop.data.textvqa import load_textvqa
from adaloop.orchestration.evaluate import read_results, run_chunks, run_evaluation
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash

LOADERS = {"gqa": load_gqa, "textvqa": load_textvqa}


def shard_items(items: Sequence[Item], shard: int, n_shards: int) -> list[Item]:
    """Every `n_shards`-th item starting at `shard` (0-based)."""
    return list(items[shard::n_shards])


def full_run_config(cfg: dict, batch_size: int) -> dict:
    """The config a run is hashed and named by: batch size is part of it, placement is not."""
    return {**cfg, "batch_size": batch_size}


def result_path(cfg: dict, root: str | Path, benchmark: str, split: str, R: int,
                batch_size: int, shard: tuple[int, int] | None = None) -> Path:
    """JSONL path of one benchmark x split x R (x shard) run; the name ends with the run config hash."""
    shard_tag = "" if shard is None else f"_shard{shard[0]}of{shard[1]}"
    digest = config_hash(full_run_config(cfg, batch_size))
    name = f"{benchmark}_{split}_res{cfg['res_px']}_R{R}_b{batch_size}{shard_tag}_{digest}.jsonl"
    return Path(root) / cfg["out_dir"] / name


def _run_files(cfg: dict, root: str | Path, benchmark: str, split: str, R: int) -> list[Path]:
    """Every result file of one benchmark x split x R of this run config, whatever its shard count."""
    digest = config_hash(full_run_config(cfg, cfg["batch_size"]))
    pattern = f"{benchmark}_{split}_res{cfg['res_px']}_R{R}_*_{digest}.jsonl"
    return sorted((Path(root) / cfg["out_dir"]).glob(pattern))


def _done_elsewhere(cfg: dict, root: str | Path, benchmark: str, split: str, R: int, n_shards: int) -> set[str]:
    """item_ids finished in files of another shard count (files of this count hold other shards or are our own)."""
    return {row["item_id"] for path in _run_files(cfg, root, benchmark, split, R)
            if f"of{n_shards}_" not in path.name for row in read_results(path)}


def _check_scope(cfg: dict) -> None:
    """Refuse the frozen test split and pruning settings that are not wired in yet."""
    if "test" in cfg["run_splits"]:
        raise ValueError("the test split is frozen until stage S12")
    if list(cfg["k_grid"]) != [1.0]:
        raise ValueError(f"k_grid must be [1.0] until pruning is wired in, got {cfg['k_grid']}")


def run_full(vlm: LoadedVLM | None, cfg: dict, root: str | Path, shard: tuple[int, int] = (0, 1),
             vlms: Sequence[LoadedVLM] | None = None) -> None:
    """Run this shard's items for every benchmark x split x R in the config; resumable.

    Items finished in files of another shard count are skipped, so the shard count may change between runs
    (stop the old processes first).

    `vlms` uses several copies in this process (see `run_evaluation`); file names do not change.
    """
    _check_scope(cfg)
    shard_i, n_shards = shard
    batch_size = cfg["batch_size"]
    run_cfg = full_run_config(cfg, batch_size)
    for benchmark in cfg["run_benchmarks"]:
        for split in cfg["run_splits"]:
            items = shard_items(LOADERS[benchmark](cfg, split, root=root), shard_i, n_shards)
            for R in cfg["R_grid"]:
                out = result_path(cfg, root, benchmark, split, R, batch_size, shard)
                elsewhere = _done_elsewhere(cfg, root, benchmark, split, R, n_shards)
                todo = [item for item in items if item.item_id not in elsewhere]
                print(f"[{benchmark} {split}] R={R} | shard {shard_i}/{n_shards} | {len(todo)} items "
                      f"({len(items) - len(todo)} done under another shard count) | {out.name}")
                if todo:
                    run_evaluation(vlm, todo, run_cfg, R, out, resume=True, batch_size=batch_size, vlms=vlms)


def _remaining_items(cfg: dict, root: str | Path, shard: tuple[int, int]) -> dict[int, int]:
    """Per R, the items of this shard (all benchmarks x splits) that have no row yet."""
    batch_size = cfg["batch_size"]
    remaining = dict.fromkeys(cfg["R_grid"], 0)
    for benchmark in cfg["run_benchmarks"]:
        for split in cfg["run_splits"]:
            ids = {item.item_id for item in shard_items(LOADERS[benchmark](cfg, split, root=root), *shard)}
            for R in cfg["R_grid"]:
                out = result_path(cfg, root, benchmark, split, R, batch_size, shard)
                done = {row["item_id"] for row in read_results(out)} if out.exists() else set()
                remaining[R] += len(ids - done - _done_elsewhere(cfg, root, benchmark, split, R, shard[1]))
    return remaining


def estimate_full_run(vlm: LoadedVLM | None, cfg: dict, root: str | Path, shard: tuple[int, int] = (0, 1),
                      vlms: Sequence[LoadedVLM] | None = None, n_items: int = 16) -> dict:
    """Time `n_items` dev items at every R with the run's batch size and copies; estimate the time `run_full` needs.

    Writes no result files. Counts only items without a row yet; model and data loading are not included.
    Returns {"per_R": [one dict per R], "total_s", "total_h"}.
    """
    _check_scope(cfg)
    copies = list(vlms) if vlms else [vlm]
    batch_size = cfg["batch_size"]
    if n_items < len(copies) * batch_size:
        raise ValueError(f"n_items must be >= {len(copies) * batch_size} (one batch per copy), got {n_items}")
    run_cfg = full_run_config(cfg, batch_size)
    benchmarks = cfg["run_benchmarks"]
    # one block per benchmark, as run_full batches them
    items = [item for benchmark in benchmarks
             for item in LOADERS[benchmark](cfg, "dev", limit=n_items // len(benchmarks), root=root)]
    list(run_chunks(copies, items[: len(copies) * batch_size], run_cfg, cfg["R_grid"][0], batch_size))  # warmup

    remaining = _remaining_items(cfg, root, shard)
    per_R = []
    for R in cfg["R_grid"]:
        start = time.perf_counter()
        rows = [row for _, chunk in run_chunks(copies, items, run_cfg, R, batch_size) for row in chunk]
        s_per_item = (time.perf_counter() - start) / len(items)
        summary = summarize_rows(rows)
        per_R.append({"R": R, "s_per_item": s_per_item, "n_think_tokens_mean": summary["n_think_tokens_mean"],
                      "forced_rate": summary["forced_rate"], "remaining": remaining[R],
                      "est_s": s_per_item * remaining[R]})
        print(f"R={R} | {s_per_item:.2f} s/item | forced {summary['forced_rate']:.0f}% | {remaining[R]} items left")
    total_s = sum(row["est_s"] for row in per_R)
    print(f"estimated time left for shard {shard[0]}/{shard[1]}: {total_s / 3600:.1f} h")
    return {"per_R": per_R, "total_s": total_s, "total_h": total_s / 3600}


def summarize_full(cfg: dict, root: str | Path) -> list[dict]:
    """One row per benchmark x split x R over all shard files of this run config, de-duplicated."""
    _check_scope(cfg)
    table = []
    for benchmark in cfg["run_benchmarks"]:
        for split in cfg["run_splits"]:
            for R in cfg["R_grid"]:
                rows = {}
                for path in _run_files(cfg, root, benchmark, split, R):
                    for row in read_results(path):
                        rows[row["item_id"]] = row
                if not rows:
                    continue
                summary = summarize_rows(list(rows.values()))
                table.append({"benchmark": benchmark, "split": split, "R": R, "n": summary["n"],
                              "accuracy": summary["accuracy"], "accuracy_lmms": summary["accuracy_lmms"],
                              "parse_rate": summary["parse_rate"],
                              "n_think_tokens_mean": summary["n_think_tokens_mean"],
                              "forced_rate": summary["forced_rate"], "s_per_item": summary["generate_s_mean"]})
    return table
