"""Per-token compute of the model, read from its parameter shapes (built by core/cost/flops.py)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CostParams:
    """Active parameters (multiply-adds per token) per part, plus what the attention terms need.

    MoE expert weights are scaled by experts_per_token / n_experts. `core` is one pass over the loop layers;
    `core_message` is the extra every pass after the first runs. `embed` is a lookup and adds no FLOPs.
    """

    embed: int
    prelude: int
    core: int
    core_message: int
    coda: int
    lm_head: int
    vit_patch: int  # patch embedding, per native patch
    vit_layers: int  # all vision layers, per native patch (and class token)
    vit_projector: int  # per language-model image token
    core_start: int
    core_end: int
    attn_dims: tuple[int, ...]  # per decoder layer: heads * head_dim
    attn_windows: tuple[int | None, ...]  # per decoder layer: sliding window, None = full attention
    kv_dims: tuple[int, ...]  # per decoder layer: key/value heads * head_dim (what the KV cache stores per token)
    vit_n_layers: int
    vit_hidden: int
    vit_merge: int
