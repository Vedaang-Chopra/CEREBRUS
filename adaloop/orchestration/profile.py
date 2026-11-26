"""Where the time goes: per-item, per-stage times of `compress` and generation. Timing only, never accuracy.

Each (item, method, k) runs twice through the production functions: plain (the wall time) and with hooks (the stages).
Vision tower and decoder stages come from forward hooks (`record_stages`: GPU queue time between marks). Compression
selection and input building are synchronized wall time around the functions `compress` calls, CPU work included.
"""
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from adaloop.core.cost.flops import cost_params, kv_cache_bytes
from adaloop.core.model.loops import set_loops
from adaloop.core.model.stage_hooks import record_stages
from adaloop.orchestration import compress as compress_module
from adaloop.orchestration.cells import generate_cell
from adaloop.orchestration.evaluate import read_results
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash
from utils.timing import DeviceClock, cuda_sync, peak_memory_gb, reset_peak_memory

# functions of orchestration/compress.py timed in the hooked run, by column; the last two are parts of `select_ms`
TIMED_FUNCTIONS = {"select_ms": ("_select", "lowres_res_px"), "build_ms": ("prune_inputs", "item_inputs"),
                   "select_spacy_ms": ("question_nouns",), "select_attention_ms": ("unit_cls_attention",)}


def _wall(vlm: LoadedVLM) -> float:
    """Wall clock after the copy's GPUs are idle."""
    cuda_sync(vlm.devices)
    return time.perf_counter()


@contextmanager
def _time_compression(vlm: LoadedVLM) -> Iterator[dict]:
    """Yield column -> ms, filled by wrappers around the functions `compress` calls; the originals come back after."""
    times = {column: 0.0 for column in TIMED_FUNCTIONS}
    originals = {}

    def timed(column: str, function):
        def run(*args, **kwargs):
            start = _wall(vlm)
            result = function(*args, **kwargs)
            times[column] += 1000.0 * (_wall(vlm) - start)
            return result

        return run

    for column, names in TIMED_FUNCTIONS.items():
        for name in names:
            originals[name] = getattr(compress_module, name)
            setattr(compress_module, name, timed(column, originals[name]))
    try:
        yield times
    finally:
        for name, function in originals.items():
            setattr(compress_module, name, function)


def _run_item(vlm: LoadedVLM, item: Item, inputs: Mapping, cfg: dict, R: int) -> dict:
    """compress + generate as a batch-1 run does them: the generation, its prompt, wall ms and peak memory."""
    reset_peak_memory(vlm.devices)
    start = _wall(vlm)
    packed, info = compress_module.compress(vlm, inputs, item, cfg["k"], cfg)
    peak_compress = peak_memory_gb(vlm.devices)  # `generate` resets the counter
    gen = generate_cell(vlm, [packed], cfg, R)[0]
    wall_ms = 1000.0 * (_wall(vlm) - start)
    return {"gen": gen, "info": info, "n_input": packed["input_ids"].shape[1], "wall_ms": wall_ms,
            "peak_mem_gb": max(peak_compress, gen.peak_mem_gb)}


def profile_item(vlm: LoadedVLM, item: Item, cfg: dict, method: str, k: float, R: int,
                 clock: DeviceClock | None = None) -> dict:
    """One row of stage times (ms) for one item under one compressor and keep fraction.

    `wall_ms`, `ttft_ms` and memory are from the plain run, the stages from the hooked run (`wall_profiled_ms`).
    `stage_sum_ms` adds the stages; `overhead_ms` = wall_ms - stage_sum_ms: Python, launches, tokenizing, the work
    between hooked modules on another device (negative if the hooks slowed the stages by more than that).
    """
    cell_cfg = {**cfg, "compressor": method, "k": k}
    inputs = item_inputs(vlm, item, cell_cfg)  # the same for every method: not timed
    set_loops(vlm, R)
    plain = _run_item(vlm, item, inputs, cell_cfg, R)
    with _time_compression(vlm) as parts, record_stages(vlm, clock) as trace:
        hooked = _run_item(vlm, item, inputs, cell_cfg, R)
    totals = trace.totals()

    ### Step-1:- Stages: vision tower wherever it ran, decoder stages of the prefill, decode steps as one total
    phases = [totals["prefill"], totals["outside"]]
    stages = {f"{name}_ms": sum(phase.get(name, 0.0) for phase in phases)
              for name in ("vit", "projector", "prelude", *(f"core{p}" for p in range(1, R + 1)), "coda")}
    decode_ms = sum(totals["decode"].values())
    n_decode = totals["n_decode"]
    stage_sum = sum(stages.values()) + parts["select_ms"] + parts["build_ms"] + decode_ms

    gen, info = plain["gen"], plain["info"]
    ttft_ms = None if gen.ttft_s is None else 1000.0 * gen.ttft_s
    return {
        "item_id": item.item_id, "benchmark": item.benchmark, "method": method, "k": k, "R": R,
        "n_vis_full": info["n_vis_full"], "n_vis": info["n_vis"], "n_input": plain["n_input"],
        "n_new": len(gen.token_ids), "fallback": info["fallback"],
        "same_output": torch.equal(gen.token_ids, hooked["gen"].token_ids),
        "wall_ms": plain["wall_ms"], "wall_profiled_ms": hooked["wall_ms"], "ttft_ms": ttft_ms,
        **parts, **stages,
        "decode_ms": decode_ms, "n_decode_steps": n_decode,
        "decode_ms_per_token": decode_ms / n_decode if n_decode else float("nan"),
        "n_prefill": totals["n_prefill"], "handoff_ms": sum(totals["handoff_ms"].values()),
        "stage_sum_ms": stage_sum, "overhead_ms": plain["wall_ms"] - stage_sum,
        "peak_mem_gb": plain["peak_mem_gb"],
        "kv_cache_bytes": kv_cache_bytes(cost_params(vlm.model), plain["n_input"], R),
    }


def profile_cells(cfg: dict) -> list[tuple[str, float]]:
    """(method, k): the full model, then every method at every k of `cfg["profile"]` (k = 1: the packed path)."""
    profile = cfg["profile"]
    return [("none", 1.0)] + [(method, k) for method in profile["methods"] for k in profile["ks"]]


def profile_items(vlm: LoadedVLM, items: Mapping[str, Sequence[Item]], cfg: dict, out_dir: str | Path,
                  cells: Sequence[tuple[str, float]] | None = None) -> list[dict]:
    """`profile_item` for every item and cell (default `profile_cells`), one resumable JSONL; return the rows.

    Items are the outer loop, so a drift of the GPUs hits all cells alike. The first pending item runs every cell once
    untimed (warmup). Use one model copy and leave its GPUs otherwise idle.
    """
    R = cfg["profile"]["R"]
    cells = list(cells or profile_cells(cfg))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"profile_R{R}_{config_hash(cfg)}.jsonl"
    done = {(row["item_id"], row["method"], row["k"]): row for row in (read_results(path) if path.exists() else [])}
    if done:  # rewrite clean lines before appending
        path.write_text("".join(json.dumps(row) + "\n" for row in done.values()))
    every = [item for bench_items in items.values() for item in bench_items]
    pending = [item for item in every if any((item.item_id, *cell) not in done for cell in cells)]
    print(f"{len(every)} items x {len(cells)} cells at R={R} | {len(pending)} items with cells to run")
    gpu_type = torch.cuda.get_device_name(0)
    commit = git_hash()
    if pending:
        for method, k in cells:
            profile_item(vlm, pending[0], cfg, method, k, R)
    start = time.perf_counter()
    for n, item in enumerate(pending, 1):
        for method, k in cells:
            if (item.item_id, method, k) in done:
                continue
            row = {**profile_item(vlm, item, cfg, method, k, R),
                   "config_hash": config_hash({**cfg, "compressor": method, "k": k}), "git_hash": commit,
                   "gpu_type": gpu_type, "devices": vlm.devices}
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done[item.item_id, method, k] = row
        print(f"[{n}/{len(pending)}] {item.item_id} | {(time.perf_counter() - start) / n:.1f} s/item (all cells)")
    return [done[item.item_id, method, k] for item in every for method, k in cells]


def profile_table(rows: Sequence[dict]) -> list[dict]:
    """One line per (method, k): median ms per stage, its % of the median wall time, and the comparison with `none`.

    `compression_overhead_ms` = median of select + build (what compression adds); `saved_ms` = median over items of
    (wall of `none` - wall of the cell), so it is net of that overhead. Medians of parts need not add up to the
    median of the sum. `handoff_ms` and the two `select_*` parts are inside other columns, not extra.
    """
    base = {row["item_id"]: row for row in rows if row["method"] == "none"}
    if not base:
        raise ValueError("profile_table needs the rows of `none`")
    R = rows[0]["R"]
    columns = ["wall_ms", "wall_profiled_ms", "ttft_ms", "vit_ms", "projector_ms", "select_ms", "select_spacy_ms",
               "select_attention_ms", "build_ms", "prelude_ms", *(f"core{p}_ms" for p in range(1, R + 1)), "coda_ms",
               "decode_ms", "decode_ms_per_token", "handoff_ms", "stage_sum_ms", "overhead_ms"]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["k"]), []).append(row)

    def median(values: list) -> float:
        values = [v for v in values if v is not None and not np.isnan(v)]
        return float(np.median(values)) if values else float("nan")

    table = []
    for (method, k), group in groups.items():
        line = {"method": method, "k": k, "n": len(group), "n_vis": median([r["n_vis"] for r in group]),
                "n_new": median([r["n_new"] for r in group])}
        line.update({column: median([r[column] for r in group]) for column in columns})
        line.update({f"pct_{column.removesuffix('_ms')}": 100.0 * line[column] / line["wall_ms"]
                     for column in columns if column.endswith("_ms") and not column.startswith(("wall", "ttft"))})
        line["compression_overhead_ms"] = median([r["select_ms"] + r["build_ms"] for r in group])
        line["saved_ms"] = median([base[r["item_id"]]["wall_ms"] - r["wall_ms"] for r in group])
        line["saved_pct"] = 100.0 * line["saved_ms"] / median([base[r["item_id"]]["wall_ms"] for r in group])
        line["peak_mem_gb"] = median([r["peak_mem_gb"] for r in group])
        line["kv_cache_bytes"] = median([r["kv_cache_bytes"] for r in group])
        table.append(line)
    return table
