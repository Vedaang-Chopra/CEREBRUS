"""Greedy generation and last-position logits."""
import logging
from collections.abc import Mapping

import torch

from adaloop.schemas.vlm import GenerationResult, LoadedVLM

logger = logging.getLogger(__name__)


@torch.inference_mode()
def generate(vlm: LoadedVLM, inputs: Mapping, max_new_tokens: int) -> GenerationResult:
    """Greedy-decode a single-item batch; works on full and pruned inputs."""
    prompt_len = inputs["input_ids"].shape[-1]
    logger.info("generating from %d tokens, max_new_tokens=%d", prompt_len, max_new_tokens)
    outputs = vlm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    token_ids = outputs[0, prompt_len:].cpu()
    logger.info("generated %d tokens", len(token_ids))
    return GenerationResult(token_ids, vlm.processor.decode(token_ids, skip_special_tokens=True))


@torch.inference_mode()
def last_logits(vlm: LoadedVLM, inputs: Mapping) -> torch.Tensor:
    """Logits at the last input position, float32 on CPU."""
    logger.info("forward on %d tokens", inputs["input_ids"].shape[-1])
    out = vlm.model(**inputs, use_cache=False, logits_to_keep=1)
    logits = out.logits[0, -1].float().cpu()
    logger.info("got logits over %d entries", logits.shape[-1])
    return logits
