"""The generation path of every run: thinking budget with budget forcing (config `think_budget`).

Phase 1 generates with thinking on. A row whose thinking message is still open after `think_budget` new tokens stops
there; a row that closed it goes on to its own end token, up to `think_budget + answer_extra_tokens`.
Phase 2, for the stopped rows only: append `<|end_message|><|message_model|><|content_text|>` + `answer_format` and
generate the answer. The KV cache is not reused: phase 2 is a new `generate` over the whole sequence (vision tower and
prefill run again), counted in `n_reprefill_tokens`.
"""
from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

from adaloop.core.eval.extract import ANSWER, END, THINKING, count_message_tokens
from adaloop.core.model.inference import END_SAMPLING, generate, generate_batch, token_id
from adaloop.core.model.prompting import append_ids
from adaloop.core.model.stopping import thinking_open
from adaloop.schemas.vlm import GenerationResult, LoadedVLM

NEW_MESSAGE = "<|message_model|>"


def thinking_budget(cfg: Mapping) -> int | None:
    """The budget that applies to a config: None when it is off or a prefill is set."""
    return cfg.get("think_budget") if cfg.get("answer_prefill") is None else None


def _run(
    vlm: LoadedVLM, inputs_list: Sequence[Mapping], max_new_tokens: int, batched: bool, **stop
) -> list[GenerationResult]:
    """One generation phase: one left-padded batch, or the batch-1 reference path."""
    if batched:
        return generate_batch(vlm, inputs_list, max_new_tokens, **stop)
    if len(inputs_list) != 1 and inputs_list:
        raise ValueError(f"batched=False takes one item, got {len(inputs_list)}")
    return [generate(vlm, inputs, max_new_tokens, **stop) for inputs in inputs_list]


def _needs_forcing(vlm: LoadedVLM, gen: GenerationResult, budget: int) -> bool:
    """True when the row reached the budget inside its thinking message."""
    ids = gen.token_ids[:budget].tolist()
    return (len(ids) == budget and token_id(vlm, END_SAMPLING) not in ids
            and thinking_open(ids, token_id(vlm, THINKING), token_id(vlm, END)))


def _merge(
    vlm: LoadedVLM, first: GenerationResult, answer: GenerationResult, think_ids: torch.Tensor,
    suffix_ids: list[int], n_reprefill: int,
) -> GenerationResult:
    """One result for a forced row. `token_ids` holds generated tokens only; the texts include the appended ones."""
    shown = think_ids.tolist() + suffix_ids + answer.token_ids.tolist()
    stops = {token_id(vlm, END), token_id(vlm, END_SAMPLING)}
    n_think, _ = count_message_tokens(think_ids.tolist(), token_id(vlm, THINKING), token_id(vlm, ANSWER), stops)
    peaks = [p for p in (first.peak_mem_gb, answer.peak_mem_gb) if p is not None]
    return replace(
        first,
        token_ids=torch.cat([think_ids, answer.token_ids]),
        text=vlm.processor.decode(shown, skip_special_tokens=True),
        raw_text=vlm.processor.decode(shown, skip_special_tokens=False),
        hit_limit=answer.hit_limit,
        peak_mem_gb=max(peaks) if peaks else None,
        n_think_tokens=n_think,
        n_answer_tokens=sum(t not in stops for t in answer.token_ids.tolist()),
        forced=True,
        n_reprefill_tokens=n_reprefill,
    )


def generate_answers(
    vlm: LoadedVLM, inputs_list: Sequence[Mapping], cfg: Mapping, batched: bool = False
) -> list[GenerationResult]:
    """Generate one result per `prepare_inputs` output under the config's token limits and thinking budget.

    `batched` False takes one item (the reference path with per-item measurements). True runs phase 1 as one batch and
    the forced rows together as a second batch. `generate_s` covers both phases (batched: both batch times / batch
    size, the same for every row); `ttft_s` is phase 1's.
    """
    budget = thinking_budget(cfg)
    if budget is None:
        return _run(vlm, inputs_list, cfg["max_new_tokens"], batched)

    ### Step-1:- Think up to the budget; a row that closed its thinking may go on a little
    results = _run(vlm, inputs_list, budget + cfg["answer_extra_tokens"], batched, think_budget=budget)
    forced = [i for i, gen in enumerate(results) if _needs_forcing(vlm, gen, budget)]
    if not forced:
        return results

    ### Step-2:- Open the answer message for the rows still thinking, then generate their answers
    suffix = END + NEW_MESSAGE + ANSWER + cfg["answer_format"]
    suffix_ids = vlm.processor.tokenizer(suffix, add_special_tokens=False)["input_ids"]
    think_ids = [results[i].token_ids[:budget] for i in forced]  # batched rows carry padding after the budget
    longer = [append_ids(inputs_list[i], ids.tolist() + suffix_ids) for i, ids in zip(forced, think_ids)]
    stop_ids = [token_id(vlm, END), token_id(vlm, END_SAMPLING)]
    answers = _run(vlm, longer, cfg["forced_answer_tokens"], batched, stop_ids=stop_ids)

    ### Step-3:- Merge the two phases; the phase-2 time belongs to the whole batch
    for i, ids, inputs, answer in zip(forced, think_ids, longer, answers):
        results[i] = _merge(vlm, results[i], answer, ids, suffix_ids, inputs["input_ids"].shape[1])
    extra_s = sum(answer.generate_s for answer in answers) / len(results)
    return [replace(gen, generate_s=gen.generate_s + extra_s) for gen in results]
