"""Forward hooks that read the loop state. Module path and loop structure: docs/notes/modeling_code.md."""
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from adaloop.schemas.vlm import LoadedVLM


@contextmanager
def record_pass_states(vlm: LoadedVLM) -> Iterator[list[torch.Tensor]]:
    """Yield a list that gets the last core layer's output at the last prompt position, once per core pass."""
    # Each entry is a float32 CPU vector (hidden_size,), pass 1 first: R entries per forward or `generate`.
    # Use one context per call and one item per batch.
    language_model = vlm.model.model.language_model
    last_core_layer = language_model.layers[language_model.config.loop_layer_index_end - 1]
    states: list[torch.Tensor] = []

    def hook(module, args, output):
        B, T, _ = output.shape
        assert B == 1, f"record_pass_states supports one item, got batch {B}"
        if T > 1:  # the prompt forward; cached decode steps have T = 1 and are skipped
            states.append(output[0, -1].detach().float().cpu())

    handle = last_core_layer.register_forward_hook(hook)
    try:
        yield states
    finally:
        handle.remove()
