"""Evaluate items at a fixed loop count, resumable, one JSONL row per item.

Adapted from Jiaqi-Gao/adaloop-vlm:adaloop/eval/runner.py @ e870bff.
Rows use the field names of `grid.parquet` (docs/DEC3_MVP.md §5) so later stages can reuse them.
"""
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from adaloop.core.eval.extract import extract_answer
from adaloop.core.eval.scorers import score
from adaloop.core.eval.summary import summarize_rows
from adaloop.core.model.inference import generate, generate_batch
from adaloop.core.model.loops import set_loops
from adaloop.core.model.prompting import prepare_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.hashing import config_hash, git_hash
from utils.images import resize_long_side


def _item_inputs(vlm: LoadedVLM, item: Item, cfg: dict) -> Mapping:
    """Model inputs for one item: resized image, question plus suffix, answer prefill."""
    prompt = item.question + cfg["prompt_suffix"]
    image = resize_long_side(item.image, cfg["res_px"])
    return prepare_inputs(vlm, prompt, image, prefill=cfg.get("answer_prefill"))


def _result_row(item: Item, cfg: dict, R: int, gen: GenerationResult, batch_size: int) -> dict:
    """Score one generation and lay it out as a result row."""
    pred = extract_answer(gen.raw_text, gen.hit_limit)
    return {
        "item_id": item.item_id,
        "benchmark": item.benchmark,
        "split": item.split,
        "question_type": item.question_type,
        "model_variant": "base",
        "k": 1.0,
        "R": R,
        "res_px": cfg["res_px"],
        "pred": pred,
        "answers": list(item.answers),
        "correct": score(item.benchmark, pred, item.answers),
        "parse_ok": pred is not None,
        "hit_limit": gen.hit_limit,
        "n_vis": gen.n_visual_tokens,
        "n_input": gen.n_input_tokens,
        "n_out_tokens": len(gen.token_ids),
        "ttft_s": gen.ttft_s,
        "generate_s": gen.generate_s,
        "peak_mem_gb": gen.peak_mem_gb,
        "batch_size": batch_size,  # > 1: generate_s is batch time / batch_size; ttft_s, peak_mem_gb are None
        "raw_output": gen.raw_text,
    }


def evaluate_item(vlm: LoadedVLM, item: Item, cfg: dict, R: int) -> dict:
    """Answer one item at loop count R with all visual tokens; return its result row."""
    set_loops(vlm, R)
    gen = generate(vlm, _item_inputs(vlm, item, cfg), cfg["max_new_tokens"])
    return _result_row(item, cfg, R, gen, batch_size=1)


def _evaluate_batch(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, R: int) -> list[dict]:
    """Like `evaluate_item` for several items generated as one batch."""
    set_loops(vlm, R)
    inputs_list = [_item_inputs(vlm, item, cfg) for item in items]
    gens = generate_batch(vlm, inputs_list, cfg["max_new_tokens"])
    return [_result_row(item, cfg, R, gen, batch_size=len(items)) for item, gen in zip(items, gens)]


def _read_rows(path: Path) -> list[dict]:
    """Rows already written to a JSONL file (a half-written last line is dropped)."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"dropping unreadable line in {path}: {line[:80]}")
    return rows


def run_evaluation(
    vlm: LoadedVLM,
    items: Sequence[Item],
    cfg: dict,
    R: int,
    out_path: str | Path,
    resume: bool = True,
    batch_size: int = 1,
) -> dict:
    """Evaluate `items` at loop count R, appending rows to `out_path`; return the summary.

    With `resume`, items whose `item_id` is already in the file are skipped; rows from a different
    config hash are an error. The summary is also written next to the JSONL as `.summary.json`.
    `batch_size` 1 is the reference path; larger values batch the generation of the pending items, for
    accuracy runs only (no per-item time to first token or memory).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # the batch size actually used is hashed (padding can shift numerics); plain batch-1 configs keep their hash
    hashed = {**cfg, "batch_size": batch_size} if batch_size > 1 or "batch_size" in cfg else cfg
    run_meta = {"config_hash": config_hash(hashed), "git_hash": git_hash(), "seed": cfg["splits"]["seed"],
                "gpu_type": torch.cuda.get_device_name(0)}

    rows = _read_rows(out_path) if out_path.exists() else []
    if rows and not resume:
        raise FileExistsError(f"{out_path} exists; pass resume=True to continue it")
    stale = {r["config_hash"] for r in rows} - {run_meta["config_hash"]}
    if stale:
        raise ValueError(f"{out_path} has rows from other configs {stale}; use a new output path")
    done = {r["item_id"] for r in rows}

    with out_path.open("w") as handle:  # rewrite clean rows, then append new ones
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        pending = [(n, item) for n, item in enumerate(items) if item.item_id not in done]
        for first in range(0, len(pending), batch_size):
            chunk = pending[first : first + batch_size]
            if batch_size == 1:
                results = [evaluate_item(vlm, chunk[0][1], cfg, R)]
            else:
                results = _evaluate_batch(vlm, [item for _, item in chunk], cfg, R)
            for (n, _), result in zip(chunk, results):
                row = {**result, **run_meta}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)
                if n == 0 or (n + 1) % 10 == 0:
                    acc = 100.0 * sum(r["correct"] for r in rows) / len(rows)
                    print(f"[{n + 1}/{len(items)}] R={R} | acc: {acc:.1f}")

    summary = {**summarize_rows(rows), "R": R, **run_meta}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def read_results(path: str | Path) -> list[dict]:
    """Load the rows of a results JSONL file."""
    return _read_rows(Path(path))
