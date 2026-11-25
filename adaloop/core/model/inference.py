"""Greedy generation and last-position logits.

Timing and memory measurement adapted from Jiaqi-Gao/adaloop-vlm:adaloop/models/inference.py @ e870bff.
"""
import time
from collections.abc import Mapping

import torch

from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.timing import FirstTokenTimer, cuda_sync, peak_memory_gb, reset_peak_memory


@torch.inference_mode()
def generate(vlm: LoadedVLM, inputs: Mapping, max_new_tokens: int) -> GenerationResult:
    """Greedy-decode a single-item batch and measure it; works on full and pruned inputs."""
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]
    n_visual = int((input_ids == vlm.model.config.image_token_id).sum())

    reset_peak_memory()
    timer = FirstTokenTimer()
    cuda_sync()
    start = time.perf_counter()
    timer.start()
    outputs = vlm.model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, streamer=timer
    )
    cuda_sync()
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
        peak_mem_gb=peak_memory_gb(),
    )
    return result


@torch.inference_mode()
def last_logits(vlm: LoadedVLM, inputs: Mapping) -> torch.Tensor:
    """Logits at the last input position, float32 on CPU."""
    out = vlm.model(**inputs, use_cache=False, logits_to_keep=1)
    logits = out.logits[0, -1].float().cpu()
    return logits
