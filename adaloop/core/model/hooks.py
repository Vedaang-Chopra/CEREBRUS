"""Forward hooks that read the loop state. Module path and loop structure: docs/notes/modeling_code.md."""
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from adaloop.schemas.vlm import LoadedVLM


@contextmanager
def record_pass_states(vlm: LoadedVLM, batch: bool = False) -> Iterator[list[torch.Tensor]]:
    """Yield a list that gets the last core layer's output at the last prompt position, once per core pass."""
    # Each entry is a float32 CPU vector (hidden_size,), pass 1 first: R entries per forward or `generate`.
    # With `batch` each entry is (B, hidden_size); prompts must be padded on the left. Use one context per call.
    language_model = vlm.model.model.language_model
    last_core_layer = language_model.layers[language_model.config.loop_layer_index_end - 1]
    states: list[torch.Tensor] = []

    def hook(module, args, output):
        B, T, _ = output.shape
        assert batch or B == 1, f"record_pass_states supports one item without `batch`, got batch {B}"
        if T > 1:  # the prompt forward; cached decode steps have T = 1 and are skipped
            last = output[:, -1] if batch else output[0, -1]
            states.append(last.detach().float().cpu())

    handle = last_core_layer.register_forward_hook(hook)
    try:
        yield states
    finally:
        handle.remove()


@contextmanager
def record_attention_mass(vlm: LoadedVLM, visual_mask: torch.Tensor) -> Iterator[list[float]]:
    """Yield a list that gets, per core pass, the attention of the last prompt position on the visual tokens.

    `visual_mask` (T,) gives one float per pass; (B, T), padded on the left, gives a tensor (B,) per pass.
    """
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
        rows = visual_mask if visual_mask.ndim == 2 else visual_mask[None]  # (B, T)
        if weights.shape[0] != rows.shape[0] or weights.shape[3] != rows.shape[1]:
            raise RuntimeError(f"attention weights {tuple(weights.shape)} do not match prompts {tuple(rows.shape)}")
        last = weights[:, :, -1].float()  # (B, heads, T)
        mass = (last * rows.to(last.device)[:, None]).sum(dim=-1).mean(dim=-1)  # (B,)
        masses.append(mass.cpu() if visual_mask.ndim == 2 else mass[0].item())

    handle = attention.register_forward_hook(hook)
    try:
        yield masses
    finally:
        handle.remove()


@contextmanager
def record_first_logits(vlm: LoadedVLM, batch: bool = False) -> Iterator[list[torch.Tensor]]:
    """Yield a list that gets the logits of the first generated token (float32 on CPU), once: (V,), or (B, V)."""
    logits: list[torch.Tensor] = []

    def hook(module, args, output):
        if not logits:
            if not batch and output.shape[0] != 1:
                raise RuntimeError(f"record_first_logits supports one item without `batch`, got {output.shape[0]}")
            last = output[:, -1] if batch else output[0, -1]
            logits.append(last.detach().float().cpu())

    handle = vlm.model.lm_head.register_forward_hook(hook)
    try:
        yield logits
    finally:
        handle.remove()
