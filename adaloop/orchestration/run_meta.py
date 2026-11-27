"""What a grid run folder says about itself: meta.json (provenance, expected items, launches) and README.md.

meta.json is shared by every launch and shard of the folder: each launch adds one entry to `launches` (`end` stays null
if the process was killed). `complete` is set when every expected (benchmark, item, rerun, k, R) row is in the result
files; `scripts/check_run.py` is the full check.
"""
import fcntl
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from adaloop.data.splits import load_manifest
from adaloop.orchestration.evaluate import read_results
from adaloop.orchestration.grid import cell_subset
from adaloop.schemas.items import Item
from utils.hashing import config_hash, git_hash
from utils.servers import server_name

CARD = """# Grid run folder

One row per (benchmark, item_id, rerun_id, k, R) and one features file per (item, rerun). Provenance and the expected
items: meta.json. Check and checksums: `python scripts/check_run.py <this folder> [--partial]` (manifest.sha256).

## Files
- `grid_<variant>_shard<i>of<n>_copy<j>_<config_hash>.jsonl`: the rows, as written (source of truth).
- `grid.parquet` / `grid_partial.parquet`: the same rows merged; list columns are arrays when read back.
- `features/<variant>/<config_hash>/<benchmark>_<item_id>.pt`: features of the main run (rerun_id 0).
- `features_rerun1/...`: the same for the noise re-run (rerun_id 1).
- `meta.json`: config, config_hash, split manifest hashes, items, one entry per launch (commit, server, GPU, versions,
  start, end in UTC), `complete`.
- `upfront/<benchmark>_<item_id>.pt` (if `mvp.py upfront` ran): visual_mean, question_mean (fp32, d), n_vis_full,
  n_question_tokens, config_hash.
- `b1_vs_b<size>/b<n>/` and `b1_vs_b<size>.csv` (if `mvp.py grid-check` ran): batch-size check runs and their table.

## Columns
- Item: item_id, benchmark, split, question_type (empty for TextVQA), answers (gold strings).
- Cell: model_variant, selector (compressor), k (kept fraction), R (loop passes), res_px, rerun_id (0 main, 1 re-run).
- Answer: pred (parsed; null if not parsed), correct (primary score), correct_lmms, parse_ok, format_ok and
  answer_len_ok (null if the config sets no format / limit), raw_output (generated text, special tokens kept; the
  prefill is not in it), token_ids (every generated token), hit_limit, forced, fallback (null: none).
- Token counts: n_vis (visual tokens the decoder got), n_vis_full (before compression), n_txt, n_input (prompt),
  n_out_tokens, n_think_tokens, n_answer_tokens, n_reprefill_tokens.
- Cost: flops_vit, flops_prefill, flops_decode, flops_compression, flops_total, kv_cache_bytes (analytic). Inputs of the
  FLOPs: n_input, n_out_tokens, R, grid_hw (vision grid height, width), compression_flops, n_reprefill_tokens.
- Time and memory (include the feature hooks): t_vit_ms (null for none / lowres), t_compression_ms, t_prefill_ms,
  t_decode_ms and peak_mem_gb (null in batches); batched t_vit_ms and t_compression_ms are per item, generation is
  batch time / batch size.
- Provenance: n_copies, batch_size, gpu_type, git_hash, config_hash, seed, server, torch_version, transformers_version.

## Features file
`torch.load(path, weights_only=True)` gives {(k, R): dict}; the R run is the answer at pass R.
- h_last (fp16, d): last core layer at the last prompt position after pass R.
- top_logp (fp32, 32), top_ids (int32, 32): largest log-probs of the first answer token, largest first.
- tail_mass: probability outside those 32. entropy (nats), top1_prob: of the full distribution.
- kl_prev: KL(pass R || pass R-1) of the first answer token; NaN if R-1 was not run (R = 1).
- delta_norm: ||h_R - h_(R-1)|| / ||h_(R-1)||; NaN at R = 1.
- vis_attn_mass: attention of the last prompt position on the visual tokens, last core layer.
- flops_readout, flops_state: cost of computing the features. n_tokens (prompt), n_vis.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_meta(run_dir: str | Path) -> dict:
    """meta.json of a run folder."""
    path = Path(run_dir) / "meta.json"
    if not path.is_file():
        raise ValueError(f"no {path}: the folder was not written by `scripts/mvp.py grid`")
    return json.loads(path.read_text())


def expected_rows(meta: dict) -> set[tuple]:
    """(benchmark, item_id, rerun_id, k, R) of every row the folder should hold."""
    runs = ((0, meta["items"]), (1, meta["rerun_items"]))
    return {(item[0], item[1], rerun_id, k, R) for rerun_id, items in runs for item in items for k, R in meta["cells"]}


def _all_rows_written(run_dir: Path, meta: dict) -> bool:
    have = {(row["benchmark"], row["item_id"], row["rerun_id"], row["k"], row["R"])
            for path in run_dir.glob(f"grid_*_{meta['config_hash']}.jsonl") for row in read_results(path)}
    return expected_rows(meta) <= have


def _update(run_dir: Path, change: Callable[[dict | None], dict]) -> dict:
    """Read, change and write meta.json under a lock (shards share the folder); the write is atomic."""
    path = run_dir / "meta.json"
    with (run_dir / ".meta.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta = change(json.loads(path.read_text()) if path.is_file() else None)
        part = run_dir / "meta.json.part"
        part.write_text(json.dumps(meta, indent=1))
        part.replace(path)
    return meta


@contextmanager
def record_launch(run_dir: str | Path, cfg: dict, root: str | Path, items: Sequence[Item], again: Sequence[Item],
                  launch: dict, variant: str = "base", ks: Sequence[float] | None = None,
                  Rs: Sequence[int] | None = None) -> Iterator[None]:
    """Wrap one launch: write README.md and add a launch entry to meta.json; close it when the block ends.

    `items` and `again` are the main and re-run items the folder is meant to hold (added to what earlier launches
    named); `launch` holds the command's own settings (shard, limit, ...). `ks` / `Rs`: the cells the folder is meant to
    hold, when `run_grid` runs a subset of the grid.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    digest = config_hash(cfg)
    entry = {**launch, "commit": git_hash(), "server": server_name(), "gpu": torch.cuda.get_device_name(0),
             "torch": str(torch.__version__), "transformers": transformers.__version__, "start": _now(), "end": None,
             "ok": None}
    run_cfg = cell_subset(cfg, ks, Rs)
    benchmarks = {**cfg["benchmarks"], **cfg.get("extra_benchmarks", {}).get("sets", {})}
    manifests = {name: load_manifest(Path(root) / bench["manifest"])["hash"] for name, bench in benchmarks.items()}

    def begin(meta: dict | None) -> dict:
        if meta and meta["config_hash"] != digest:  # another config used this folder: keep its record aside
            (run_dir / f"meta_{meta['config_hash']}.json").write_text(json.dumps(meta, indent=1))
            meta = None
        meta = meta or {"config_hash": digest, "config": cfg, "variant": variant,
                        "cells": [[k, R] for k in run_cfg["k_grid"] for R in run_cfg["R_grid"]],
                        "items": [], "rerun_items": [], "launches": []}
        for key, new in (("items", items), ("rerun_items", again)):
            known = {(b, i): split for b, i, split in meta[key]}
            known.update({(item.benchmark, item.item_id): item.split for item in new})
            meta[key] = [[b, i, split] for (b, i), split in sorted(known.items())]
        meta.update(n_items=len(meta["items"]), n_rerun_items=len(meta["rerun_items"]), split_manifests=manifests,
                    complete=False)
        meta["launches"].append(entry)
        return meta

    index = len(_update(run_dir, begin)["launches"]) - 1
    (run_dir / "README.md").write_text(CARD)

    def finish(ok: bool) -> None:
        def end(meta: dict) -> dict:
            meta["launches"][index].update(end=_now(), ok=ok)
            meta["complete"] = _all_rows_written(run_dir, meta)
            return meta

        _update(run_dir, end)

    try:
        yield
    except BaseException:
        finish(False)
        raise
    finish(True)
