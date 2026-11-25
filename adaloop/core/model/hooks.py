"""Forward hooks that read the loop state.

Module path and loop structure: docs/notes/modeling_code.md.
"""
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from adaloop.schemas.vlm import LoadedVLM

logger = logging.getLogger(__name__)


@contextmanager
def record_pass_states(vlm: LoadedVLM) -> Iterator[list[torch.Tensor]]:
    """Record the last core layer's output at the last prompt position, once per core pass.

    Yields a list that fills with one float32 CPU vector [hidden_size] per pass (pass 1 first), so it has
    R entries after one forward or `generate` call. Only multi-token forwards are recorded: cached decode
    steps (length 1) are skipped. Use one context per call, one item per batch.
    """
    language_model = vlm.model.model.language_model
    index = language_model.config.loop_layer_index_end - 1
    states: list[torch.Tensor] = []

    def hook(module, args, output):
        if output.shape[0] != 1:
            raise ValueError(f"record_pass_states supports one item, got batch {output.shape[0]}")
        if output.shape[1] > 1:
            states.append(output[0, -1].detach().float().cpu())

    logger.info("recording pass states at layer %d", index)
    handle = language_model.layers[index].register_forward_hook(hook)
    try:
        yield states
    finally:
        handle.remove()
        logger.info("recorded %d pass states", len(states))
