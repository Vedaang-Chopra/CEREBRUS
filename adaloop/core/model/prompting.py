"""Prompt and input construction."""
import logging

import torch
from PIL import Image
from transformers import BatchFeature

from adaloop.schemas.vlm import LoadedVLM

logger = logging.getLogger(__name__)


def prepare_inputs(
    vlm: LoadedVLM, question: str, image: Image.Image | None = None, prefill: str | None = None
) -> BatchFeature:
    """Tokenize one user turn (image first, then question) onto the model device.

    `prefill` is the start of the model's reply (special-token markers included); its ids are appended
    after the user turn, so generation continues from it.
    """
    logger.info("preparing inputs, image=%s, prefill=%r", image is not None, prefill)
    content = [{"type": "text", "text": question}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    inputs = vlm.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        reasoning_effort="none",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlm.model.device)
    if prefill is not None:
        _append_prefill(vlm, inputs, prefill)
    logger.info("prepared %d input tokens", inputs["input_ids"].shape[-1])
    return inputs


def _append_prefill(vlm: LoadedVLM, inputs: BatchFeature, prefill: str) -> None:
    """Extend every per-token tensor of `inputs` in place with the prefill ids; image tensors are untouched."""
    shape = inputs["input_ids"].shape
    ids = vlm.processor.tokenizer(prefill, add_special_tokens=False)["input_ids"]
    for key, value in list(inputs.items()):
        if value.shape != shape:
            continue
        if key == "input_ids":
            tail = value.new_tensor([ids])
        else:  # attention_mask is 1; token-type tensors are 0 (text)
            tail = value.new_full((shape[0], len(ids)), int(key == "attention_mask"))
        inputs[key] = torch.cat([value, tail], dim=1)
    logger.info("appended %d prefill tokens", len(ids))
