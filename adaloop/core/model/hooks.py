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


@contextmanager
def record_attention_mass(vlm: LoadedVLM, visual_mask: torch.Tensor) -> Iterator[list[float]]:
    """Yield a list that gets, per core pass, the attention of the last prompt position on the visual tokens."""
    # Layer: the last core layer (a full-attention one). Mean over its heads of the summed weights on `visual_mask`
    # (T,), the image positions of the prompt. The attention sink keeps some mass, so the value is below 1 - text.
    language_model = vlm.model.model.language_model
    attention = language_model.layers[language_model.config.loop_layer_index_end - 1].self_attn
    masses: list[float] = []

    def hook(module, args, output):
        weights = output[1]  # (B, heads, T_query, T_key)
        if weights is None:
            raise RuntimeError("the attention returned no weights; it needs the eager implementation")
        if weights.shape[2] == 1:  # cached decode step
            return
        if weights.shape[0] != 1 or weights.shape[3] != len(visual_mask):
            raise RuntimeError(f"attention weights {tuple(weights.shape)} do not match a prompt of {len(visual_mask)}")
        last = weights[0, :, -1].float()  # (heads, T)
        masses.append(last[:, visual_mask.to(last.device)].sum(dim=-1).mean().item())

    handle = attention.register_forward_hook(hook)
    try:
        yield masses
    finally:
        handle.remove()


@contextmanager
def record_first_logits(vlm: LoadedVLM) -> Iterator[list[torch.Tensor]]:
    """Yield a list that gets the logits of the first generated token (float32 on CPU), once."""
    logits: list[torch.Tensor] = []

    def hook(module, args, output):
        if not logits:
            if output.shape[0] != 1:
                raise RuntimeError(f"record_first_logits supports one item, got batch {output.shape[0]}")
            logits.append(output[0, -1].detach().float().cpu())

    handle = vlm.model.lm_head.register_forward_hook(hook)
    try:
        yield logits
    finally:
        handle.remove()
