"""Evaluate items at a fixed loop count, resumable, one JSONL row per item.

Adapted from Jiaqi-Gao/adaloop-vlm:adaloop/eval/runner.py @ e870bff.
Rows use the field names of `grid.parquet` (docs/DEC3_MVP.md §5) so later stages can reuse them.
"""
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from adaloop.core.eval.extract import extract_answer
from adaloop.core.eval.scorers import score
from adaloop.core.eval.summary import summarize_rows
from adaloop.core.model.inference import generate
from adaloop.core.model.loops import set_loops
from adaloop.core.model.prompting import prepare_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash
from utils.images import resize_long_side

logger = logging.getLogger(__name__)


def evaluate_item(vlm: LoadedVLM, item: Item, cfg: dict, R: int) -> dict:
    """Answer one item at loop count R with all visual tokens; return its result row."""
    set_loops(vlm, R)
    prompt = item.question + cfg["prompt_suffix"]
    image = resize_long_side(item.image, cfg["res_px"])
    inputs = prepare_inputs(vlm, prompt, image, prefill=cfg.get("answer_prefill"))
    gen = generate(vlm, inputs, cfg["max_new_tokens"])
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
        "raw_output": gen.raw_text,
    }


def _read_rows(path: Path) -> list[dict]:
    """Rows already written to a JSONL file (a half-written last line is dropped)."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("dropping unreadable line in %s: %.80s", path, line)
    return rows


def run_evaluation(
    vlm: LoadedVLM, items: Sequence[Item], cfg: dict, R: int, out_path: str | Path, resume: bool = True
) -> dict:
    """Evaluate `items` at loop count R, appending rows to `out_path`; return the summary.

    With `resume`, items whose `item_id` is already in the file are skipped; rows from a different
    config hash are an error. The summary is also written next to the JSONL as `.summary.json`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta = {"config_hash": config_hash(cfg), "git_hash": git_hash(), "seed": cfg["splits"]["seed"],
                "gpu_type": torch.cuda.get_device_name(0)}
    logger.info("evaluating %d items at R=%d -> %s (%s)", len(items), R, out_path, run_meta)

    rows = _read_rows(out_path) if out_path.exists() else []
    if rows and not resume:
        raise FileExistsError(f"{out_path} exists; pass resume=True to continue it")
    stale = {r["config_hash"] for r in rows} - {run_meta["config_hash"]}
    if stale:
        raise ValueError(f"{out_path} has rows from other configs {stale}; use a new output path")
    done = {r["item_id"] for r in rows}
    logger.info("resuming: %d of %d items already done", len(done), len(items))

    with out_path.open("w") as handle:  # rewrite clean rows, then append new ones
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        for n, item in enumerate(items):
            if item.item_id in done:
                continue
            row = {**evaluate_item(vlm, item, cfg, R), **run_meta}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            if n == 0 or (n + 1) % 10 == 0:
                logger.info("[%d/%d] R=%d accuracy so far %.1f", n + 1, len(items), R,
                            100.0 * sum(r["correct"] for r in rows) / len(rows))

    summary = {**summarize_rows(rows), "R": R, **run_meta}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def read_results(path: str | Path) -> list[dict]:
    """Load the rows of a results JSONL file."""
    return _read_rows(Path(path))
