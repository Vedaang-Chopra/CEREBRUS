"""Signals read from the vision tower. One 2x2 unit = 4 adjacent patches = one decoder token."""
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers.models.inkling2.modeling_inkling2 import apply_rotary_pos_emb_vision

from adaloop.schemas.vlm import LoadedVLM


@contextmanager
def capture_last_attention(vlm: LoadedVLM) -> Iterator[dict]:
    """Yield a dict that gets `cls_attention` (N,) and `keys` (1, N, head_dim) per unit from the last ViT layer."""
    # Only the CLS row of the attention is computed (one query against all keys), from the same inputs the layer
    # gets, so the model's attention implementation is not changed. Single image only.
    tower = vlm.model.model.vision_tower
    attention = tower.layers[-1].self_attn
    merge = tower.config.spatial_merge_size
    captured: dict = {}

    def hook(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]  # (1 + P, D), CLS first
        cos, sin = kwargs["position_embeddings"]
        if len(kwargs["cu_seqlens"]) != 2:
            raise ValueError("capture_last_attention supports one image")
        shape = (hidden.shape[0], module.num_heads, module.head_dim)
        q = module.q_proj(hidden).view(shape)
        k = module.k_proj(hidden).view(shape)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k = q.float(), k.float()
        weights = torch.einsum("hd,phd->hp", q[0], k) * module.scaling  # (heads, 1 + P): the CLS query
        per_patch = weights.softmax(dim=-1)[:, 1:].sum(dim=0)  # (P,), summed over heads
        captured["cls_attention"] = per_patch.reshape(-1, merge**2).mean(dim=-1)  # (N,)
        keys = k[1:].reshape(-1, merge**2, module.num_heads, module.head_dim).mean(dim=1)  # (N, heads, head_dim)
        captured["keys"] = keys.mean(dim=1).unsqueeze(0)  # (1, N, head_dim)

    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()
