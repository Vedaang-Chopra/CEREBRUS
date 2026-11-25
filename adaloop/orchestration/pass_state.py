"""Pass-state check (stage S3b): is pass p of a long run the same as the end of an R = p run?"""
import logging
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from adaloop.core.eval.extract import extract_answer
from adaloop.core.eval.scorers import score
from adaloop.core.model.hooks import record_pass_states
from adaloop.core.model.inference import generate
from adaloop.core.model.loops import MAX_LOOPS, set_loops
from adaloop.core.model.prompting import prepare_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.images import resize_long_side

logger = logging.getLogger(__name__)


def _run_with_states(
    vlm: LoadedVLM, inputs, max_new_tokens: int, R: int
) -> tuple[list[torch.Tensor], GenerationResult]:
    """Generate at loop count R; return the R pass states of the prompt forward and the generation."""
    set_loops(vlm, R)
    with record_pass_states(vlm) as states:
        gen = generate(vlm, inputs, max_new_tokens)
    if len(states) != R:
        raise RuntimeError(f"expected {R} pass states, got {len(states)}")
    return states, gen


def check_pass_state(vlm: LoadedVLM, items: Sequence[Item], cfg: dict) -> list[dict]:
    """Compare one R = MAX_LOOPS run per item with a fresh run at every R = p; return one row per (item, p).

    `cosine` and `rel_l2` compare the pass-p state of the long run with the final state of the R = p run
    (`rel_l2` is relative to the R = p state). The long run has no answer at pass p, so `pred_p` is the
    answer of the R = p run and `pred_full` the answer of the long run. At p = MAX_LOOPS the two runs are
    identical, so that row shows the rerun noise.
    """
    rows = []
    for n, item in enumerate(items):
        image = resize_long_side(item.image, cfg["res_px"])
        prompt = item.question + cfg["prompt_suffix"]
        inputs = prepare_inputs(vlm, prompt, image, prefill=cfg.get("answer_prefill"))
        full_states, full = _run_with_states(vlm, inputs, cfg["max_new_tokens"], MAX_LOOPS)
        pred_full = extract_answer(full.raw_text, full.hit_limit)
        for p in range(1, MAX_LOOPS + 1):
            states, gen = _run_with_states(vlm, inputs, cfg["max_new_tokens"], p)
            long_h, short_h = full_states[p - 1], states[-1]
            pred = extract_answer(gen.raw_text, gen.hit_limit)
            rows.append({
                "item_id": item.item_id,
                "p": p,
                "cosine": F.cosine_similarity(long_h, short_h, dim=0).item(),
                "rel_l2": ((long_h - short_h).norm() / short_h.norm()).item(),
                "h_equal": torch.equal(long_h, short_h),
                "pred_p": pred,
                "pred_full": pred_full,
                "parsed_both": pred is not None and pred_full is not None,
                "answer_agree": pred is not None and pred == pred_full,
                "tokens_equal": torch.equal(gen.token_ids, full.token_ids),
                "correct_p": score(item.benchmark, pred, item.answers),
                "correct_full": score(item.benchmark, pred_full, item.answers),
            })
        logger.info("pass-state check %d/%d done", n + 1, len(items))
    return rows
