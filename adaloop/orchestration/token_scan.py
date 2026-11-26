"""Is the decoder faster with fewer visual tokens? Timing only: all tokens against `random` at exact kept counts."""
import json
import statistics
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from adaloop.core.model.loops import set_loops
from adaloop.core.model.stage_hooks import record_stages
from adaloop.core.tokens.prune import drop_visual_tokens
from adaloop.orchestration.cells import generate_cell
from adaloop.orchestration.compress import compress, encode_vision
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash
from utils.timing import DeviceClock


@contextmanager
def _exact_length(vlm: LoadedVLM, n_new: int) -> Iterator[None]:
    """Stop tokens are suppressed up to `n_new` tokens, so every answer decodes the same number of steps."""
    config = vlm.model.generation_config
    before = config.min_new_tokens
    config.min_new_tokens = n_new
    try:
        yield
    finally:
        config.min_new_tokens = before


def _variants(vlm: LoadedVLM, item: Item, cfg: dict, kept: Sequence[int]) -> dict[tuple, tuple[Mapping, int]]:
    """(method, kept) -> (decoder inputs, visual tokens in them); the vision tower runs here, once, untimed.

    `none` (kept None) is every token through the packed path. 0 kept is the prompt with the image placeholders
    removed and no vision features (`compress` always keeps at least one). An item with fewer tokens keeps them all.
    """
    full = item_inputs(vlm, item, cfg)
    vision = encode_vision(vlm, full)
    random_cfg = {**cfg, "compressor": "random"}
    packed, info = compress(vlm, full, item, 1.0, random_cfg, vision=vision)
    out = {("none", None): (packed, info["n_vis"])}
    for n in kept:
        if n == 0:
            out["random", 0] = (drop_visual_tokens(vlm, full), 0)
        else:
            packed, info = compress(vlm, full, item, 1.0, random_cfg, vision=vision, n_keep=n)
            out["random", n] = (packed, info["n_vis"])
    return out


def run_token_scan(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, out_dir: str | Path,
                   clock: DeviceClock | None = None) -> list[dict]:
    """Time prefill and decode over `cfg["profile"]["scan"]` (kept counts x R x batch size); return and save rows.

    One row per (method, kept, R, batch). Medians of `repeats` runs after one warmup: `prefill_ms` and
    `decode_ms_per_step` from the hooked runs (GPU stage times; one step = one token per row of the batch),
    `wall_ms` and `ttft_ms` (batch 1 only) from plain runs. Only the decoder runs: the vision tower is outside.
    A batch is the next `batch_size` items, left-padded to one length (`n_input`); a last short batch is dropped.
    """
    scan = cfg["profile"]["scan"]
    if len(items) < max(scan["batch_sizes"]):
        raise ValueError(f"the scan needs at least {max(scan['batch_sizes'])} items, got {len(items)}")
    n_new = scan["n_decode"]
    run_cfg = {**cfg, "max_new_tokens": n_new, "think_budget": None, "batch_grouping": None}
    print(f"preparing {len(items)} items x {1 + len(scan['kept'])} token settings")
    variants = [_variants(vlm, item, cfg, scan["kept"]) for item in items]
    commit = git_hash()
    rows = []
    with _exact_length(vlm, n_new):
        for setting in variants[0]:
            for R in scan["R_list"]:
                set_loops(vlm, R)
                for batch_size in scan["batch_sizes"]:
                    for first in range(0, len(items) - batch_size + 1, batch_size):
                        chunk = [v[setting] for v in variants[first:first + batch_size]]
                        inputs = [packed for packed, _ in chunk]
                        generate_cell(vlm, inputs, run_cfg, R)  # warmup
                        wall, ttft, prefill, decode = [], [], [], []
                        for _ in range(scan["repeats"]):
                            gens = generate_cell(vlm, inputs, run_cfg, R)
                            wall.append(1000.0 * gens[0].generate_s * len(gens))  # batched: batch time / size
                            ttft.append(gens[0].ttft_s)
                            with record_stages(vlm, clock) as trace:
                                generate_cell(vlm, inputs, run_cfg, R)
                            totals = trace.totals()
                            prefill.append(sum(totals["prefill"].values()))
                            decode.append(sum(totals["decode"].values()) / totals["n_decode"])
                        rows.append({
                            "method": setting[0], "kept": setting[1], "R": R, "batch_size": batch_size,
                            "first_item": items[first].item_id,
                            "n_vis": statistics.mean(n for _, n in chunk),
                            "n_input": max(packed["input_ids"].shape[1] for packed in inputs),
                            "n_new": min(len(gen.token_ids) for gen in gens), "n_decode_steps": totals["n_decode"],
                            "prefill_ms": statistics.median(prefill),
                            "decode_ms_per_step": statistics.median(decode),
                            "wall_ms": statistics.median(wall),
                            "ttft_ms": None if ttft[0] is None else 1000.0 * statistics.median(ttft),
                            "git_hash": commit,
                        })
                print(f"{setting[0]} kept {setting[1]} | R={R} | prefill {rows[-1]['prefill_ms']:.1f} ms, "
                      f"decode {rows[-1]['decode_ms_per_step']:.1f} ms/step (batch {rows[-1]['batch_size']})")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"token_scan_{config_hash(cfg)}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"saved {len(rows)} rows to {path}")
    return rows


def token_scan_table(rows: Sequence[dict]) -> list[dict]:
    """One line per (method, kept, R, batch size): medians over the batches (`n_vis`: mean)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["kept"], row["R"], row["batch_size"]), []).append(row)

    def median(group: list[dict], key: str) -> float:
        values = [r[key] for r in group if r[key] is not None]
        return statistics.median(values) if values else float("nan")

    return [{"method": method, "kept": kept, "R": R, "batch_size": batch_size, "batches": len(group),
             "n_vis": statistics.mean(r["n_vis"] for r in group), "n_input": median(group, "n_input"),
             "prefill_ms": median(group, "prefill_ms"), "decode_ms_per_step": median(group, "decode_ms_per_step"),
             "wall_ms": median(group, "wall_ms"), "ttft_ms": median(group, "ttft_ms")}
            for (method, kept, R, batch_size), group in groups.items()]
