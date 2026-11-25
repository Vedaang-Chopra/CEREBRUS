"""Visual-token extraction and packed pruning.

Ported from notebooks/02_pruning.ipynb.
"""
import logging
from collections.abc import Mapping
from copy import copy

import torch
from transformers.utils import ModelOutput

from adaloop.schemas.vlm import LoadedVLM

logger = logging.getLogger(__name__)


@torch.inference_mode()
def extract_visual_tokens(vlm: LoadedVLM, inputs: Mapping) -> tuple[torch.Tensor, ModelOutput]:
    """Projected visual tokens [N, d] and the full vision output."""
    if "pixel_values" not in inputs:
        raise ValueError("inputs have no pixel_values; prepare_inputs needs an image")
    logger.info("extracting visual tokens, grid %s", inputs["image_grid_thw"].tolist())
    outputs = vlm.model.model.get_image_features(
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        return_dict=True,
    )
    visual_tokens = torch.cat(outputs.pooler_output, dim=0)
    logger.info("visual tokens %s", tuple(visual_tokens.shape))
    return visual_tokens, outputs


def prune_inputs(
    vlm: LoadedVLM, inputs: Mapping, vision_outputs: ModelOutput, indices: torch.Tensor
) -> dict:
    """Keep only the visual tokens at `indices` (sorted, unique). Kept positions are packed."""
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1 or len(vision_outputs.pooler_output) != 1:
        raise ValueError("prune_inputs supports one item with one image")
    if indices.numel() == 0 or not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("indices must be non-empty, sorted and unique")

    image_mask = input_ids[0] == vlm.model.config.image_token_id
    image_positions = image_mask.nonzero(as_tuple=True)[0]
    visual_tokens = vision_outputs.pooler_output[0]
    if len(image_positions) != len(visual_tokens):
        raise ValueError(f"{len(image_positions)} image placeholders vs {len(visual_tokens)} visual tokens")

    keep_mask = ~image_mask
    keep_mask[image_positions[indices.to(input_ids.device)]] = True
    pruned_vision = copy(vision_outputs)
    pruned_vision.pooler_output = (visual_tokens[indices.to(visual_tokens.device)],)
    logger.info("pruned visual tokens %d -> %d", len(visual_tokens), len(indices))
    return {
        "input_ids": input_ids[:, keep_mask],
        "attention_mask": inputs["attention_mask"][:, keep_mask],
        "mm_encoder_outputs": {"image": pruned_vision},
    }
