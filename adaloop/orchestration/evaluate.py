"""Evaluate items at a fixed loop count, resumable, one JSONL row per item.

Adapted from Jiaqi-Gao/adaloop-vlm:adaloop/eval/runner.py @ e870bff.
Rows use the field names of `grid.parquet` (docs/DEC3_MVP.md §5) so later stages can reuse them.
"""
import json
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from adaloop.core.eval.extract import answer_is_open, extract_answer
from adaloop.core.eval.scorers import score
from adaloop.core.eval.summary import summarize_rows
from adaloop.core.model.inference import generate, generate_batch
from adaloop.core.model.loops import set_loops
from adaloop.core.model.prompting import prepare_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.hashing import config_hash, git_hash
from utils.images import resize_long_side


def item_prompt(item: Item, cfg: dict) -> str:
    """The question text sent to the model (capitalized for TextVQA) plus its benchmark's `prompt_suffix`."""
    # TextVQA: lmms-eval applies str.capitalize (first letter upper, the rest lower). Ported from
    # EvolvingLMMs-Lab/lmms-eval:lmms_eval/tasks/textvqa/utils.py::textvqa_doc_to_text
    # @ 9134731f4f617e62ee80bcae9d0d5d9695ea55ef.
    question = item.question.capitalize() if item.benchmark == "textvqa" else item.question
    return question + cfg["benchmarks"][item.benchmark]["prompt_suffix"]


def item_inputs(vlm: LoadedVLM, item: Item, cfg: dict) -> Mapping:
    """Model inputs for one item: resized image, prompt, answer prefill."""
    image = resize_long_side(item.image, cfg["res_px"])
    return prepare_inputs(vlm, item_prompt(item, cfg), image, prefill=cfg.get("answer_prefill"))


def _result_row(item: Item, cfg: dict, R: int, gen: GenerationResult, batch_size: int) -> dict:
    """Score one generation and lay it out as a result row."""
    pred = extract_answer(gen.raw_text, gen.hit_limit, prefilled=answer_is_open(cfg.get("answer_prefill")))
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
    gen = generate(vlm, item_inputs(vlm, item, cfg), cfg["max_new_tokens"])
    return _result_row(item, cfg, R, gen, batch_size=1)


def _evaluate_batch(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, R: int) -> list[dict]:
    """Like `evaluate_item` for several items generated as one batch."""
    set_loops(vlm, R)
    inputs_list = [item_inputs(vlm, item, cfg) for item in items]
    gens = generate_batch(vlm, inputs_list, cfg["max_new_tokens"])
    return [_result_row(item, cfg, R, gen, batch_size=len(items)) for item, gen in zip(items, gens)]


def _run_chunk(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, R: int, batch_size: int) -> list[dict]:
    """Rows of one chunk on one model copy: the batch-1 reference path, or one left-padded batch."""
    if batch_size == 1:
        return [evaluate_item(vlm, items[0], cfg, R)]
    return _evaluate_batch(vlm, items, cfg, R)


def check_distinct_copies(vlms: Sequence[LoadedVLM]) -> None:
    """Threads need their own model (loop count is model state) and processor (tokenizers are not shareable)."""
    for name in ("model", "processor"):
        if len({id(getattr(v, name)) for v in vlms}) != len(vlms):
            raise ValueError(f"every copy needs its own {name}; pass the list from load_models")


def run_chunks(vlms: Sequence[LoadedVLM], items: Sequence[Item], cfg: dict, R: int,
               batch_size: int = 1) -> Iterator[tuple[int, list[dict]]]:
    """Yield (copy index, rows) for consecutive chunks of `batch_size` items, in item order.

    One copy runs in the calling thread. Several copies take the chunks round-robin, one thread each, so
    the chunks (and so the batch padding) are the same as with one copy. An error in a chunk is raised when
    its turn comes: earlier chunks were already yielded, later ones are dropped.
    """
    chunks = [items[first : first + batch_size] for first in range(0, len(items), batch_size)]
    if len(vlms) == 1:
        for chunk in chunks:
            yield 0, _run_chunk(vlms[0], chunk, cfg, R, batch_size)
        return

    check_distinct_copies(vlms)
    pools = [ThreadPoolExecutor(max_workers=1) for _ in vlms]  # a copy is only ever used by its own thread
    try:
        waiting: deque = deque()  # (copy index, future), oldest chunk first
        for n, chunk in enumerate(chunks):
            j = n % len(vlms)
            waiting.append((j, pools[j].submit(_run_chunk, vlms[j], chunk, cfg, R, batch_size)))
            # at most 2 chunks per copy are in flight, which bounds the finished work a killed process loses
            if len(waiting) >= 2 * len(vlms):
                j, future = waiting.popleft()
                yield j, future.result()
        while waiting:
            j, future = waiting.popleft()
            yield j, future.result()
    finally:
        for pool in pools:  # queued chunks are cancelled; a running one is waited for
            pool.shutdown(wait=True, cancel_futures=True)


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
    vlm: LoadedVLM | None,
    items: Sequence[Item],
    cfg: dict,
    R: int,
    out_path: str | Path,
    resume: bool = True,
    batch_size: int = 1,
    vlms: Sequence[LoadedVLM] | None = None,
) -> dict:
    """Evaluate `items` at loop count R, appending rows to `out_path`; return the summary.

    With `resume`, items whose `item_id` is already in the file are skipped; rows from a different
    config hash are an error. The summary is also written next to the JSONL as `.summary.json`.
    `batch_size` 1 is the reference path; larger values batch the generation of the pending items, for
    accuracy runs only (no per-item time to first token or memory).

    `vlms` (from `load_models`) splits the pending chunks over several copies, one thread each; `vlm` must
    then be None or vlms[0]. Outputs, hash and resume are the same as with one copy; rows record `copy` and
    `n_copies`, and with n_copies > 1 their times include contention between the copies. This thread writes
    the rows, in item order. After an error, rows before the failed chunk are on disk; finished later chunks
    are dropped and rerun on resume (a killed process loses at most 2 chunks per copy).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if vlms and vlm is not None and vlm is not vlms[0]:
        raise ValueError("with vlms, pass vlm=None or vlms[0]")
    copies = list(vlms) if vlms else [vlm]
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
        positions = iter(n for n, _ in pending)  # place of each pending item in `items`, for the progress line
        for copy, results in run_chunks(copies, [item for _, item in pending], cfg, R, batch_size):
            for result in results:
                n = next(positions)
                row = {**result, "copy": copy, "n_copies": len(copies), **run_meta}
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
