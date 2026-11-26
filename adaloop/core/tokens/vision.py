"""Signals read from the vision tower. One 2x2 unit = 4 adjacent patches = one decoder token."""
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers.models.inkling2.modeling_inkling2 import apply_rotary_pos_emb_vision

from adaloop.schemas.vlm import LoadedVLM


@contextmanager
def capture_attention_inputs(vlm: LoadedVLM) -> Iterator[dict]:
    """Yield a dict that gets the inputs of the last ViT layer's attention (references only, nothing is computed)."""
    attention = vlm.model.model.vision_tower.layers[-1].self_attn
    captured: dict = {}

    def hook(module, args, kwargs):
        if len(kwargs["cu_seqlens"]) != 2:
            raise ValueError("capture_attention_inputs supports one image")
        captured["hidden"] = args[0] if args else kwargs["hidden_states"]  # (1 + P, D), CLS first
        captured["cos"], captured["sin"] = kwargs["position_embeddings"]

    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()


@torch.no_grad()
def unit_cls_attention(vlm: LoadedVLM, captured: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(CLS attention per unit (N,), keys per unit (1, N, head_dim)) of the last ViT layer from its captured inputs."""
    # Only the CLS row of the attention is computed (one query against all keys), from the same inputs the layer got,
    # so the model's attention implementation is not changed.
    tower = vlm.model.model.vision_tower
    module = tower.layers[-1].self_attn
    merge = tower.config.spatial_merge_size
    hidden = captured["hidden"]
    shape = (hidden.shape[0], module.num_heads, module.head_dim)
    q = module.q_proj(hidden).view(shape)
    k = module.k_proj(hidden).view(shape)
    q, k = apply_rotary_pos_emb_vision(q, k, captured["cos"], captured["sin"])
    q, k = q.float(), k.float()
    weights = torch.einsum("hd,phd->hp", q[0], k) * module.scaling  # (heads, 1 + P): the CLS query
    per_patch = weights.softmax(dim=-1)[:, 1:].sum(dim=0)  # (P,), summed over heads
    attention = per_patch.reshape(-1, merge**2).mean(dim=-1)  # (N,)
    keys = k[1:].reshape(-1, merge**2, module.num_heads, module.head_dim).mean(dim=1)  # (N, heads, head_dim)
    return attention, keys.mean(dim=1).unsqueeze(0)  # (1, N, head_dim)
