"""Prompt and input construction."""
import logging

from PIL import Image
from transformers import BatchFeature

from adaloop.schemas.vlm import LoadedVLM

logger = logging.getLogger(__name__)


def prepare_inputs(vlm: LoadedVLM, question: str, image: Image.Image | None = None) -> BatchFeature:
    """Tokenize one user turn (image first, then question) onto the model device."""
    logger.info("preparing inputs, image=%s", image is not None)
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
    logger.info("prepared %d input tokens", inputs["input_ids"].shape[-1])
    return inputs
