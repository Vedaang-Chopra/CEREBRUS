"""Full baseline run: shard the items of each benchmark x split, one resumable JSONL per shard and R.

Shared by `scripts/mvp.py full` and `notebooks/06_full_baseline`, so file names match.
"""
from collections.abc import Sequence
from pathlib import Path

from adaloop.core.eval.summary import summarize_rows
from adaloop.data.gqa import load_gqa
from adaloop.data.textvqa import load_textvqa
from adaloop.orchestration.evaluate import read_results, run_evaluation
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


def _check_scope(cfg: dict) -> None:
    """Refuse the frozen test split and pruning settings that are not wired in yet."""
    if "test" in cfg["run_splits"]:
        raise ValueError("the test split is frozen until stage S12")
    if list(cfg["k_grid"]) != [1.0]:
        raise ValueError(f"k_grid must be [1.0] until pruning is wired in, got {cfg['k_grid']}")


def run_full(vlm: LoadedVLM | None, cfg: dict, root: str | Path, shard: tuple[int, int] = (0, 1),
             vlms: Sequence[LoadedVLM] | None = None) -> None:
    """Run this shard's items for every benchmark x split x R in the config; resumable.

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
                print(f"[{benchmark} {split}] R={R} | shard {shard_i}/{n_shards} | {len(items)} items | {out.name}")
                run_evaluation(vlm, items, run_cfg, R, out, resume=True, batch_size=batch_size, vlms=vlms)


def summarize_full(cfg: dict, root: str | Path) -> list[dict]:
    """One row per benchmark x split x R over all shard files of this run config, de-duplicated."""
    _check_scope(cfg)
    batch_size = cfg["batch_size"]
    digest = config_hash(full_run_config(cfg, batch_size))
    folder = Path(root) / cfg["out_dir"]
    table = []
    for benchmark in cfg["run_benchmarks"]:
        for split in cfg["run_splits"]:
            for R in cfg["R_grid"]:
                rows = {}
                for path in sorted(folder.glob(f"{benchmark}_{split}_res{cfg['res_px']}_R{R}_*_{digest}.jsonl")):
                    for row in read_results(path):
                        rows[row["item_id"]] = row
                if not rows:
                    continue
                summary = summarize_rows(list(rows.values()))
                table.append({"benchmark": benchmark, "split": split, "R": R, "n": summary["n"],
                              "accuracy": summary["accuracy"], "parse_rate": summary["parse_rate"],
                              "s_per_item": summary["generate_s_mean"]})
    return table
