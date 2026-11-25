"""Pass-state check (stage S3b): is pass p of a long run the same as the end of an R = p run?"""
import time
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from adaloop.core.eval.scorers import score
from adaloop.core.model.hooks import record_pass_states
from adaloop.core.model.budget import generate_answers
from adaloop.core.model.loops import MAX_LOOPS, set_loops
from adaloop.orchestration.evaluate import item_inputs, read_answer
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM


def _run_with_states(
    vlm: LoadedVLM, inputs, cfg: dict, R: int
) -> tuple[list[torch.Tensor], GenerationResult]:
    """Generate at loop count R; return (the R pass states of the prompt forward, the generation)."""
    set_loops(vlm, R)
    with record_pass_states(vlm) as states:
        gen = generate_answers(vlm, [inputs], cfg)[0]
    expected = 2 * R if gen.forced else R  # a forced answer runs the prompt again
    if len(states) != expected:
        raise RuntimeError(f"expected {expected} pass states, got {len(states)}")
    return states[:R], gen  # the first prompt forward


def _compare(
    item: Item,
    p: int,
    long_run: tuple[torch.Tensor, GenerationResult],
    short_run: tuple[torch.Tensor, GenerationResult],
    cfg: dict,
) -> dict:
    """Build the result row for pass p; each run is (state to compare, generation)."""
    long_h, long_gen = long_run  # (hidden_size,)
    short_h, short_gen = short_run
    pred_p = read_answer(short_gen, cfg)
    pred_full = read_answer(long_gen, cfg)
    return {
        "item_id": item.item_id,
        "p": p,
        "cosine": F.cosine_similarity(long_h, short_h, dim=0).item(),
        "rel_l2": ((long_h - short_h).norm() / short_h.norm()).item(),  # relative to the R = p state
        "h_equal": torch.equal(long_h, short_h),
        "pred_p": pred_p,
        "pred_full": pred_full,
        "parsed_both": pred_p is not None and pred_full is not None,
        "answer_agree": pred_p is not None and pred_p == pred_full,
        "tokens_equal": torch.equal(short_gen.token_ids, long_gen.token_ids),
        "correct_p": score(item.benchmark, pred_p, item.answers),
        "correct_full": score(item.benchmark, pred_full, item.answers),
    }


def check_pass_state(vlm: LoadedVLM, items: Sequence[Item], cfg: dict) -> list[dict]:
    """Compare one R = MAX_LOOPS run per item with a fresh run at every R = p; one row per (item, p)."""
    # The long run has no answer at pass p, so `pred_p` is the R = p answer and `pred_full` the long run's answer.
    # At p = MAX_LOOPS both runs use the same setting, so that row shows the rerun noise.
    rows = []
    for n, item in enumerate(items):
        start = time.perf_counter()
        ### Step-1:- One prompt per item, shared by every run
        inputs = item_inputs(vlm, item, cfg)
        ### Step-2:- The long run records the state of every pass
        long_states, long_gen = _run_with_states(vlm, inputs, cfg, MAX_LOOPS)
        ### Step-3:- A fresh run per p; compare its final state with pass p of the long run
        for p in range(1, MAX_LOOPS + 1):
            states, gen = _run_with_states(vlm, inputs, cfg, p)
            rows.append(_compare(item, p, (long_states[p - 1], long_gen), (states[-1], gen), cfg))
        worst = max(r["rel_l2"] for r in rows[-MAX_LOOPS:])
        print(f"[{n + 1}/{len(items)}] {item.item_id} | max rel L2: {worst:.2e} | {time.perf_counter() - start:.1f}s")
    return rows
