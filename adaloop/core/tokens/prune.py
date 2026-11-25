"""Visual-token extraction and packed pruning.

Ported from notebooks/02_pruning.ipynb.
"""
from collections.abc import Mapping
from copy import copy

import torch
from transformers.utils import ModelOutput

from adaloop.schemas.vlm import LoadedVLM


@torch.inference_mode()
def extract_visual_tokens(vlm: LoadedVLM, inputs: Mapping) -> tuple[torch.Tensor, ModelOutput]:
    """Return (projected visual tokens (N, d), the full vision output)."""
    if "pixel_values" not in inputs:
        raise ValueError("inputs have no pixel_values; prepare_inputs needs an image")
    outputs = vlm.model.model.get_image_features(
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        return_dict=True,
    )
    return torch.cat(outputs.pooler_output, dim=0), outputs


def prune_inputs(
    vlm: LoadedVLM, inputs: Mapping, vision_outputs: ModelOutput, indices: torch.Tensor
) -> dict:
    """Keep only the visual tokens at `indices` (sorted, unique). Kept positions are packed."""
    # Packed is the only option: the model builds token distances, masks and convolutions from the token index,
    # not from position_ids (docs/notes/modeling_code.md).
    input_ids = inputs["input_ids"]  # (1, T)
    if input_ids.shape[0] != 1 or len(vision_outputs.pooler_output) != 1:
        raise ValueError("prune_inputs supports one item with one image")
    if indices.numel() == 0 or not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("indices must be non-empty, sorted and unique")

    ### Step-1:- One placeholder in the prompt per visual token
    image_mask = input_ids[0] == vlm.model.config.image_token_id  # (T,)
    image_positions = image_mask.nonzero(as_tuple=True)[0]  # (N,)
    visual_tokens = vision_outputs.pooler_output[0]  # (N, d)
    if len(image_positions) != len(visual_tokens):
        raise ValueError(f"{len(image_positions)} image placeholders vs {len(visual_tokens)} visual tokens")

    ### Step-2:- Drop the same entries from the prompt and from the vision features
    keep_mask = ~image_mask
    keep_mask[image_positions[indices.to(input_ids.device)]] = True
    pruned_vision = copy(vision_outputs)  # the caller's vision output stays untouched
    pruned_vision.pooler_output = (visual_tokens[indices.to(visual_tokens.device)],)
    return {
        "input_ids": input_ids[:, keep_mask],
        "attention_mask": inputs["attention_mask"][:, keep_mask],
        "mm_encoder_outputs": {"image": pruned_vision},
    }
