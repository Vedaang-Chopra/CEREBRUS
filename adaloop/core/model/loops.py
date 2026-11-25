"""Loop-count control and verification.

Ported from notebooks/01_huggingfaceload.ipynb.
"""
import logging
from collections import Counter
from collections.abc import Mapping

import torch

from adaloop.schemas.vlm import LoadedVLM

logger = logging.getLogger(__name__)

MAX_LOOPS = 5  # upper limit of the varloop checkpoint, not a tunable


def set_loops(vlm: LoadedVLM, R: int) -> None:
    """Set the number of loop passes for all following forward calls."""
    if not 1 <= R <= MAX_LOOPS:
        raise ValueError(f"R must be in 1..{MAX_LOOPS}, got {R}")
    vlm.model.config.text_config.loop_num_passes = R
    vlm.model.model.language_model.config.loop_num_passes = R
    logger.info("loop passes set to %d", R)


def expected_layer_calls(vlm: LoadedVLM) -> list[int]:
    """Calls each decoder layer should receive in one forward at the current R."""
    language_model = vlm.model.model.language_model
    cfg = language_model.config
    return [
        cfg.loop_num_passes if cfg.loop_layer_index_start <= i < cfg.loop_layer_index_end else 1
        for i in range(len(language_model.layers))
    ]


@torch.inference_mode()
def count_layer_calls(vlm: LoadedVLM, inputs: Mapping) -> list[int]:
    """Run one forward and count how often each decoder layer executes."""
    layers = vlm.model.model.language_model.layers
    calls: Counter = Counter()

    def make_hook(index: int):
        def hook(module, args, output):
            calls[index] += 1

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        vlm.model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    result = [calls[i] for i in range(len(layers))]
    logger.info("total layer calls %d", sum(result))
    return result
