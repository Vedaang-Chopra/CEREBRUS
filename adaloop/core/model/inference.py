"""Greedy generation and last-position logits.

Timing and memory measurement adapted from Jiaqi-Gao/adaloop-vlm:adaloop/models/inference.py @ e870bff.
"""
import time
from collections.abc import Mapping, Sequence

import torch

from adaloop.core.eval.extract import ANSWER, END
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.timing import FirstTokenTimer, cuda_sync, peak_memory_gb, reset_peak_memory

IMAGE_KEYS = ("pixel_values", "image_grid_thw")  # one row per image token / image, not per sequence
END_SAMPLING = "<|content_model_end_sampling|>"


def _token_id(vlm: LoadedVLM, token: str) -> int:
    """Id of a special token, looked up by its string."""
    tokenizer = vlm.processor.tokenizer
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no token {token!r}")
    return token_id


def _stop_ids(vlm: LoadedVLM, input_ids: torch.Tensor) -> tuple[list[int], dict]:
    """(stop token ids, extra `model.generate` kwargs) for prompts `input_ids` (B, T), padded on the left."""
    # Answer message already open: its <|end_message|> ends the answer, otherwise the model starts a new message.
    # Not open (thinking on): <|end_message|> also ends the thinking message, so the model's own stop is kept.
    is_open = input_ids[:, -1] == _token_id(vlm, ANSWER)  # (B,)
    if is_open.all():
        ids = [_token_id(vlm, END), _token_id(vlm, END_SAMPLING)]
        return ids, {"eos_token_id": ids}
    if is_open.any():
        raise ValueError("a batch cannot mix prompts with and without an open answer message")
    eos = vlm.model.generation_config.eos_token_id
    return [] if eos is None else [eos] if isinstance(eos, int) else list(eos), {}


@torch.inference_mode()
def generate(vlm: LoadedVLM, inputs: Mapping, max_new_tokens: int) -> GenerationResult:
    """Greedy-decode a single-item batch and measure it; works on full and pruned inputs."""
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]
    n_visual = int((input_ids == vlm.model.config.image_token_id).sum())

    _, stop_kwargs = _stop_ids(vlm, input_ids)
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
def generate_batch(vlm: LoadedVLM, inputs_list: Sequence[Mapping], max_new_tokens: int) -> list[GenerationResult]:
    """Greedy-decode several single-item inputs as one left-padded batch; one result per item, in order.

    For accuracy runs. `generate` (batch 1) is the reference and the only path with per-item measurements:
    here `generate_s` is batch time / batch size, and `ttft_s` and `peak_mem_gb` are None.
    """
    batch = _left_pad(inputs_list, vlm.model.generation_config.pad_token_id)
    prompt_width = batch["input_ids"].shape[1]
    stop_ids, stop_kwargs = _stop_ids(vlm, batch["input_ids"])
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
            )
        )
    return results


@torch.inference_mode()
def last_logits(vlm: LoadedVLM, inputs: Mapping) -> torch.Tensor:
    """Logits at the last input position, float32 on CPU."""
    out = vlm.model(**inputs, use_cache=False, logits_to_keep=1)
    logits = out.logits[0, -1].float().cpu()
    return logits
