"""Greedy generation and last-position logits.

Timing and memory measurement adapted from Jiaqi-Gao/adaloop-vlm:adaloop/models/inference.py @ e870bff.
"""
import time
from collections.abc import Mapping, Sequence

import torch
from transformers import StoppingCriteriaList

from adaloop.core.eval.extract import ANSWER, END, THINKING, count_message_tokens
from adaloop.core.model.stopping import StopOpenThinking
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.timing import FirstTokenTimer, cuda_sync, peak_memory_gb, reset_peak_memory

IMAGE_KEYS = ("pixel_values", "image_grid_thw")  # one row per image token / image, not per sequence
END_SAMPLING = "<|content_model_end_sampling|>"


def token_id(vlm: LoadedVLM, token: str) -> int:
    """Id of a special token, looked up by its string."""
    tokenizer = vlm.processor.tokenizer
    found = tokenizer.convert_tokens_to_ids(token)
    if found is None or found == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no token {token!r}")
    return found


def _answer_open(vlm: LoadedVLM, row: torch.Tensor) -> bool:
    """True when the last `<|content_text|>` of a prompt row (T,) has no `<|end_message|>` after it."""
    def last(marker: str) -> int:
        found = (row == token_id(vlm, marker)).nonzero()
        return int(found[-1]) if len(found) else -1

    return last(ANSWER) > last(END)


def _stop_ids(vlm: LoadedVLM, input_ids: torch.Tensor) -> tuple[list[int], dict]:
    """(stop token ids, extra `model.generate` kwargs) for prompts `input_ids` (B, T), padded on the left."""
    # Answer message already open: its <|end_message|> ends the answer, otherwise the model starts a new message.
    # Not open (thinking on): <|end_message|> also ends the thinking message, so the model's own stop is kept.
    is_open = torch.tensor([_answer_open(vlm, row) for row in input_ids])  # (B,)
    if is_open.all():
        ids = [token_id(vlm, END), token_id(vlm, END_SAMPLING)]
        return ids, {"eos_token_id": ids}
    if is_open.any():
        raise ValueError("a batch cannot mix prompts with and without an open answer message")
    eos = vlm.model.generation_config.eos_token_id
    return [] if eos is None else [eos] if isinstance(eos, int) else list(eos), {}


def _message_counts(vlm: LoadedVLM, token_ids: torch.Tensor, input_ids: torch.Tensor) -> tuple[int, int]:
    """(thinking tokens, answer tokens) of one generated row for its unpadded prompt `input_ids` (1, T)."""
    answer_id, thinking_id = token_id(vlm, ANSWER), token_id(vlm, THINKING)
    open_part = "answer" if _answer_open(vlm, input_ids[0]) else "think" if int(input_ids[0, -1]) == thinking_id else None
    return count_message_tokens(
        token_ids.tolist(),
        thinking_id=thinking_id,
        text_id=answer_id,
        end_ids={token_id(vlm, END), token_id(vlm, END_SAMPLING)},
        open_part=open_part,
    )


def _generate_kwargs(
    vlm: LoadedVLM, input_ids: torch.Tensor, think_budget: int | None, stop_ids: list[int] | None
) -> tuple[list[int], dict]:
    """(stop token ids, extra `model.generate` kwargs); `stop_ids` replaces the rule read from the prompt."""
    if stop_ids is None:
        stop_ids, kwargs = _stop_ids(vlm, input_ids)
    else:
        kwargs = {"eos_token_id": stop_ids}
    if think_budget is not None:
        criterion = StopOpenThinking(input_ids.shape[1], think_budget, token_id(vlm, THINKING), token_id(vlm, END))
        kwargs = {**kwargs, "stopping_criteria": StoppingCriteriaList([criterion])}
    return stop_ids, kwargs


@torch.inference_mode()
def generate(
    vlm: LoadedVLM,
    inputs: Mapping,
    max_new_tokens: int,
    think_budget: int | None = None,
    stop_ids: list[int] | None = None,
) -> GenerationResult:
    """Greedy-decode a single-item batch and measure it; works on full and pruned inputs.

    One phase only: `think_budget` ends the output when the thinking is still open there. Runs under a config go
    through `generate_answers` (core/model/budget.py), which adds the forced answer.
    """
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]
    n_visual = int((input_ids == vlm.model.config.image_token_id).sum())

    _, stop_kwargs = _generate_kwargs(vlm, input_ids, think_budget, stop_ids)
    devices = vlm.devices  # only this copy's GPUs: another copy may be running in another thread
    reset_peak_memory(devices)
    timer = FirstTokenTimer()
    cuda_sync(devices)
    start = time.perf_counter()
    timer.start()
    outputs = vlm.model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, streamer=timer, **stop_kwargs
    )
    cuda_sync(devices)
    generate_s = time.perf_counter() - start

    token_ids = outputs[0, prompt_len:].cpu()
    n_think, n_answer = _message_counts(vlm, token_ids, input_ids)
    result = GenerationResult(
        token_ids=token_ids,
        text=vlm.processor.decode(token_ids, skip_special_tokens=True),
        raw_text=vlm.processor.decode(token_ids, skip_special_tokens=False),
        n_input_tokens=prompt_len,
        n_visual_tokens=n_visual,
        hit_limit=len(token_ids) >= max_new_tokens,
        ttft_s=timer.ttft_s,
        generate_s=generate_s,
        peak_mem_gb=peak_memory_gb(devices),
        n_think_tokens=n_think,
        n_answer_tokens=n_answer,
    )
    return result


def _left_pad(inputs_list: Sequence[Mapping], pad_id: int) -> dict:
    """Merge single-item inputs into one left-padded batch; image tensors are concatenated in row order."""
    keys = set().union(*inputs_list)
    width = max(x["input_ids"].shape[1] for x in inputs_list)  # T
    batch = {}
    for key in keys:
        if key in IMAGE_KEYS:  # row order = placeholder order, which is how the model fills them
            batch[key] = torch.cat([x[key] for x in inputs_list if key in x])
            continue
        rows = []
        for inputs in inputs_list:
            value = inputs.get(key)
            if not torch.is_tensor(value) or value.shape != inputs["input_ids"].shape or value.shape[0] != 1:
                raise ValueError(f"generate_batch takes single-item `prepare_inputs` outputs; bad key {key!r}")
            fill = pad_id if key == "input_ids" else 0  # attention_mask and token types are 0 on padding
            pad = value.new_full((1, width - value.shape[1]), fill)
            rows.append(torch.cat([pad, value], dim=1))
        batch[key] = torch.cat(rows)  # (B, T)
    return batch


@torch.inference_mode()
def generate_batch(
    vlm: LoadedVLM,
    inputs_list: Sequence[Mapping],
    max_new_tokens: int,
    think_budget: int | None = None,
    stop_ids: list[int] | None = None,
) -> list[GenerationResult]:
    """Greedy-decode several single-item inputs as one left-padded batch; one result per item, in order.

    For accuracy runs. `generate` (batch 1) is the reference and the only path with per-item measurements:
    here `generate_s` is batch time / batch size, and `ttft_s` and `peak_mem_gb` are None.
    """
    batch = _left_pad(inputs_list, vlm.model.generation_config.pad_token_id)
    prompt_width = batch["input_ids"].shape[1]
    stop_ids, stop_kwargs = _generate_kwargs(vlm, batch["input_ids"], think_budget, stop_ids)
    eos_ids = torch.tensor(stop_ids, dtype=torch.long)

    cuda_sync(vlm.devices)
    start = time.perf_counter()
    outputs = vlm.model.generate(
        **batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, **stop_kwargs
    )
    cuda_sync(vlm.devices)
    batch_s = time.perf_counter() - start
    new_ids = outputs[:, prompt_width:].cpu()  # (B, new)

    results = []
    for inputs, token_ids in zip(inputs_list, new_ids):
        # a finished row is filled with pad up to the longest row: cut after its first stop token, as batch 1 does
        stops = torch.isin(token_ids, eos_ids).nonzero()
        if len(stops):
            token_ids = token_ids[: int(stops[0]) + 1]
        input_ids = inputs["input_ids"]
        n_think, n_answer = _message_counts(vlm, token_ids, input_ids)
        results.append(
            GenerationResult(
                token_ids=token_ids,
                text=vlm.processor.decode(token_ids, skip_special_tokens=True),
                raw_text=vlm.processor.decode(token_ids, skip_special_tokens=False),
                n_input_tokens=input_ids.shape[-1],  # unpadded
                n_visual_tokens=int((input_ids == vlm.model.config.image_token_id).sum()),
                hit_limit=len(token_ids) >= max_new_tokens,
                ttft_s=None,
                generate_s=batch_s / len(inputs_list),
                peak_mem_gb=None,
                n_think_tokens=n_think,
                n_answer_tokens=n_answer,
            )
        )
    return results


@torch.inference_mode()
def last_logits(vlm: LoadedVLM, inputs: Mapping) -> torch.Tensor:
    """Logits at the last input position, float32 on CPU."""
    out = vlm.model(**inputs, use_cache=False, logits_to_keep=1)
    logits = out.logits[0, -1].float().cpu()
    return logits
