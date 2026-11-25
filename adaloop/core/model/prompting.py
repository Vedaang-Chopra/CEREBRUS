"""Prompt and input construction."""
from collections.abc import Mapping

import torch
from PIL import Image
from transformers import BatchFeature

from adaloop.schemas.vlm import LoadedVLM


def prepare_inputs(
    vlm: LoadedVLM,
    question: str,
    image: Image.Image | None = None,
    prefill: str | None = None,
    system: str | None = None,
    reasoning_effort: float | str = "none",
) -> BatchFeature:
    """Tokenize one user turn (image first, then question) onto the model device.

    `system` becomes a system message before the template's own "Thinking effort level" system message.
    `reasoning_effort` only changes the number in that message (none 0 ... high 0.9, or a number in [0, 0.99]);
    "none" does not stop the model from thinking. The template adds no generation prompt.
    `prefill` is the start of the model's reply (special-token markers included); its ids are appended
    after the user turn, so generation continues from it.
    """
    content = [{"type": "text", "text": question}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    messages = [{"role": "user", "content": content}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
    inputs = vlm.processor.apply_chat_template(
        messages,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlm.model.device)
    if prefill is not None:
        ids = vlm.processor.tokenizer(prefill, add_special_tokens=False)["input_ids"]
        inputs.update(append_ids(inputs, ids))
    return inputs


def append_ids(inputs: Mapping, ids: list[int]) -> dict:
    """Copy of single-item `inputs` with `ids` added to every per-token tensor; image tensors are untouched."""
    shape = inputs["input_ids"].shape
    longer = dict(inputs)
    for key, value in inputs.items():
        if value.shape != shape:
            continue
        if key == "input_ids":
            tail = value.new_tensor([ids])
        else:  # attention_mask is 1; token-type tensors are 0 (text)
            tail = value.new_full((shape[0], len(ids)), int(key == "attention_mask"))
        longer[key] = torch.cat([value, tail], dim=1)
    return longer
