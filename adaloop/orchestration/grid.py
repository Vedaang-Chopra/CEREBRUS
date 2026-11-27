"""The base grid (S7): every item at every k x R under one compressor, features on, resumable; then grid.parquet.

The batches are the chunks of `item_chunks` (at most `grid_batch.size`) over the whole item list, so they do not
depend on what is already done, on the shard or copy count, or on which items are written (`only`). A batch with a
missing row is computed again as a whole and only the missing rows are written; per item the features file is written
first, then its rows. Each worker (shard x model copy) appends to its own file. One folder holds one batch size (`grid_dir`).
"""
import json
import random
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import transformers

from adaloop.core.cost.flops import cost_params, kv_cache_bytes, row_flops
from adaloop.orchestration.batching import EQUAL_LENGTH, item_chunks
from adaloop.orchestration.evaluate import read_results
from adaloop.orchestration.features import run_items_cells, save_features
from adaloop.orchestration.full_run import LOADERS
from adaloop.schemas.cost import CostParams
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash
from utils.servers import server_name

COLUMNS = [
    "item_id", "benchmark", "split", "question_type", "model_variant", "selector", "k", "R",
    "pred", "correct", "correct_lmms", "parse_ok",
    "n_vis", "n_vis_full", "n_txt", "n_out_tokens",
    "flops_vit", "flops_prefill", "flops_decode", "flops_compression", "flops_total",
    "kv_cache_bytes",
    "t_vit_ms", "t_compression_ms", "t_prefill_ms", "t_decode_ms", "peak_mem_gb",
    "rerun_id", "n_copies", "batch_size", "gpu_type", "git_hash", "config_hash", "seed",
    "server", "torch_version", "transformers_version",  # empty in rows written before these columns
]


def grid_config(cfg: dict) -> dict:
    """The config of every grid run: `max_new_tokens` is `grid_batch.max_new_tokens`, so the grid's hash covers it."""
    return {**cfg, "max_new_tokens": cfg["grid_batch"]["max_new_tokens"]}


def check_grid_config(cfg: dict) -> None:
    """Refuse the sealed test split and a grid without a compressor."""
    if "test" in cfg["grid"]["splits"]:
        raise ValueError("the test split is sealed until stage S12")
    if cfg["compressor"] == "none" and list(cfg["k_grid"]) != [1.0]:
        raise ValueError("the grid prunes: freeze `compressor` (S5) in configs/mvp.yaml first")


def grid_items(cfg: dict, root: str | Path) -> list[Item]:
    """All items of the grid splits and benchmarks, in a fixed order."""
    check_grid_config(cfg)
    return [item for split in cfg["grid"]["splits"] for benchmark in cfg["grid"]["benchmarks"]
            for item in LOADERS[benchmark](cfg, split, root=root)]


def grid_dir(cfg: dict, root: str | Path, name: str = "grid") -> Path:
    """<out_dir>/<name>_b<grid batch size>, plus `_padded` when batches may mix prompt lengths."""
    size = cfg["grid_batch"]["size"]
    padded = size > 1 and cfg.get("batch_grouping") != EQUAL_LENGTH
    return Path(root) / cfg["out_dir"] / f"{name}_b{size}{'_padded' if padded else ''}"


def rerun_items(items: Sequence[Item], n: int, seed: int) -> list[Item]:
    """The `n` items that are run a second time (noise ceiling): a seeded sample, the same on every shard."""
    keys = sorted((item.benchmark, item.item_id) for item in items)
    picked = set(random.Random(seed).sample(keys, min(n, len(keys))))
    return [item for item in items if (item.benchmark, item.item_id) in picked]


def _grid_rows(rows: list[dict], cfg: dict, params, rerun_id: int, meta: dict) -> list[dict]:
    """Result rows of one item in the columns of grid.parquet."""
    def ms(seconds: float | None) -> float | None:
        return None if seconds is None else 1000.0 * seconds

    out = []
    for row in rows:
        row = {**row, **row_flops(params, row)}
        ttft, total = row["ttft_s"], row["generate_s"]  # batched rows: no ttft, generate_s = batch time / size
        row.update(
            n_txt=row["n_input"] - row["n_vis"],
            t_vit_ms=ms(row["vit_s"]),  # None for none/lowres: the vision tower is inside t_prefill_ms
            t_compression_ms=ms(row["compression_s"]),
            t_prefill_ms=ms(ttft),
            t_decode_ms=None if ttft is None else ms(total - ttft),
            rerun_id=rerun_id,
            **meta,
        )
        out.append({name: row[name] for name in COLUMNS})
    return out


def _done(out_dir: Path, digest: str) -> dict[tuple, set]:
    """(benchmark, item_id, rerun_id) -> the (k, R) cells already in any result file of this config."""
    done: dict[tuple, set] = {}
    for path in sorted(out_dir.glob(f"grid_*_{digest}.jsonl")):
        for row in read_results(path):
            done.setdefault((row["benchmark"], row["item_id"], row["rerun_id"]), set()).add((row["k"], row["R"]))
    return done


def _lacking(item: Item, wanted: set, cells: set, done: dict, rerun_id: int) -> set:
    """The cells of a wanted item that no result file holds yet."""
    key = (item.benchmark, item.item_id)
    return cells - done.get((*key, rerun_id), set()) if key in wanted else set()


def grid_remaining(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, out_dir: str | Path, rerun_id: int = 0,
                   only: Sequence[Item] | None = None) -> int:
    """Answers `run_grid` (one shard, default batch size) would still generate: every cell of every item of a batch
    that lacks a wanted row."""
    cells = {(k, R) for k in cfg["k_grid"] for R in cfg["R_grid"]}
    done = _done(Path(out_dir), config_hash(cfg))
    wanted = {(item.benchmark, item.item_id) for item in (items if only is None else only)}
    chunks = item_chunks(vlm, items, cfg, cfg["grid_batch"]["size"])
    return len(cells) * sum(len(chunk) for chunk in chunks
                            if any(_lacking(item, wanted, cells, done, rerun_id) for item in chunk))


def count_done(out_dir: str | Path, cfg: dict) -> int:
    """Rows (item x rerun x cell) of this config already in the result files of `out_dir`."""
    return sum(len(cells) for cells in _done(Path(out_dir), config_hash(cfg)).values())


def run_grid(vlms: Sequence[LoadedVLM], items: Sequence[Item], cfg: dict, out_dir: str | Path,
             shard: tuple[int, int] = (0, 1), rerun_id: int = 0, variant: str = "base",
             batch_size: int | None = None, only: Sequence[Item] | None = None) -> tuple[int, float]:
    """Run shard i of n of the batches of `items` over the model copies, one thread each; resumable.

    `items` defines the batches (`batch_size`, default `grid_batch.size`); `only` (default: all of `items`) are the
    items whose rows are written, so a re-run or a sample goes through the same batches as the full run.
    Writes `grid_<variant>_shard<i>of<n>_copy<j>_<hash>.jsonl` and, for `rerun_id` 0, one features file per item in
    `features/<variant>/<hash>/<benchmark>_<item_id>.pt`. With several copies the times include contention.
    Returns (answers generated by this call, including unwritten batch neighbours; seconds spent on the batches).
    """
    check_grid_config(cfg)
    batch_size = batch_size or cfg["grid_batch"]["size"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = config_hash(cfg)
    cells = {(k, R) for k in cfg["k_grid"] for R in cfg["R_grid"]}
    done = _done(out_dir, digest)
    wanted = {(item.benchmark, item.item_id) for item in (items if only is None else only)}
    unknown = wanted - {(item.benchmark, item.item_id) for item in items}
    if unknown:
        raise ValueError(f"`only` has items that are not in `items`: {sorted(unknown)[:5]}")

    def missing(item: Item) -> set:
        """The cells of this item still to be written."""
        return _lacking(item, wanted, cells, done, rerun_id)

    ### Step-1:- Batches from the whole list, then the shard, then the batches that still lack a row
    mine = item_chunks(vlms[0], items, cfg, batch_size)[shard[0] :: shard[1]]
    todo = [chunk for chunk in mine if any(missing(item) for item in chunk)]
    n_pending = sum(bool(missing(item)) for chunk in todo for item in chunk)
    n_wanted = sum((item.benchmark, item.item_id) in wanted for chunk in mine for item in chunk)
    print(f"grid rerun_id={rerun_id} shard {shard[0]}/{shard[1]}: {n_pending} of {n_wanted} items to write in "
          f"{len(todo)} batches ({sum(map(len, todo))} items computed), {len(cells)} cells each, batch {batch_size}, "
          f"{len(vlms)} copies")
    if not todo:
        return 0, 0.0
    meta = {"model_variant": variant, "n_copies": len(vlms), "gpu_type": torch.cuda.get_device_name(0),
            "git_hash": git_hash(), "config_hash": digest, "seed": cfg["splits"]["seed"], "server": server_name(),
            "torch_version": str(torch.__version__), "transformers_version": transformers.__version__}
    params = cost_params(vlms[0].model)
    stop = threading.Event()  # one failing worker stops the others after their current batch
    generated: list[int] = []  # answers per computed batch

    ### Step-2:- Each copy takes every n-th batch and writes the missing rows of its items
    def work(j: int) -> None:
        path = out_dir / f"grid_{variant}_shard{shard[0]}of{shard[1]}_copy{j}_{digest}.jsonl"
        kept = read_results(path) if path.exists() else []
        with path.open("w") as handle:  # rewrite the readable rows: a half-written last line is dropped
            handle.writelines(json.dumps(row) + "\n" for row in kept)
            chunks = todo[j :: len(vlms)]
            total, seen = sum(bool(missing(item)) for chunk in chunks for item in chunk), 0
            for chunk in chunks:
                if stop.is_set():
                    return
                try:
                    results = run_items_cells(vlms[j], chunk, cfg)
                    generated.append(len(chunk) * len(cells))
                    for item, (rows, features) in zip(chunk, results):
                        lacking = missing(item)
                        if not lacking:  # done before, or only here to keep the batch as it was
                            continue
                        if rerun_id == 0:
                            save_features(features, out_dir / "features" / variant / digest,
                                          f"{item.benchmark}_{item.item_id}")
                        new = [row for row in rows if (row["k"], row["R"]) in lacking]
                        for row in _grid_rows(new, cfg, params, rerun_id, meta):
                            handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        seen += 1
                except BaseException:
                    stop.set()
                    print(f"copy {j} failed on {[(i.benchmark, i.item_id) for i in chunk]}; stopping")
                    raise
                print(f"copy {j}: [{seen}/{total}] items")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(vlms)) as pool:
        try:
            list(pool.map(work, range(len(vlms))))
        except BaseException:
            stop.set()
            raise
    return sum(generated), time.perf_counter() - start


def merge_grid(out_dir: str | Path, cfg: dict, name: str = "grid.parquet",
               params: CostParams | None = None) -> pd.DataFrame:
    """All result files of this config as one table (a repeated cell keeps its last row), written to <out_dir>/name.

    With `params` (from `cost_params`), `kv_cache_bytes` is computed for every row, also rows written before the column
    existed; without it those rows have NaN there.
    """
    out_dir = Path(out_dir)
    rows = [row for path in sorted(out_dir.glob(f"grid_*_{config_hash(cfg)}.jsonl")) for row in read_results(path)]
    if not rows:
        raise ValueError(f"no result files of config {config_hash(cfg)} in {out_dir}")
    table = pd.DataFrame(rows, columns=COLUMNS)
    keys = ["benchmark", "item_id", "model_variant", "k", "R", "rerun_id"]
    table = table.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    if params is not None:  # analytic: the prompt after compression is text + kept visual tokens
        table["kv_cache_bytes"] = [kv_cache_bytes(params, int(t + v), int(R))
                                   for t, v, R in zip(table["n_txt"], table["n_vis"], table["R"])]
    table.to_parquet(out_dir / name, index=False)
    return table
