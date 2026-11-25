"""Analytic FLOPs (2 per multiply-add) for the looped VLM. Cost model: docs/DEC3_MVP.md §6.

Counts the matrix products, short convolutions and attention products of the transformers `inkling2_loop` code.
Left out: norms, softmax, activations, router top-k, block and message attention residuals beyond their parameters.
Weights are counted as run: the NVFP4/MXFP8 experts of the checkpoint are dequantized to bf16 on load.
"""
import re
from collections.abc import Sequence

from transformers import PreTrainedModel

from adaloop.schemas.cost import CostParams

REF_R = 3  # loop count of the reference cost (DEC3 §6); a definition, not a tunable


def cost_params(model: PreTrainedModel) -> CostParams:
    """Count active parameters per part from parameter shapes; works for sharded and meta-device models."""
    text, vision = model.config.text_config, model.config.vision_config
    sliding = [kind == "hybrid_sliding" for kind in text.layer_types]
    heads = [text.swa_num_attention_heads if s else text.num_attention_heads for s in sliding]
    head_dims = [text.swa_head_dim if s else text.head_dim for s in sliding]
    expert_share = text.num_experts_per_tok / text.n_routed_experts

    parts = dict.fromkeys(
        ["embed", "prelude", "core", "core_message", "coda", "lm_head", "vit_patch", "vit_layers", "vit_projector"], 0
    )
    for name, param in model.named_parameters():
        n = param.numel()
        layer = re.search(r"language_model\.layers\.(\d+)\.", name)
        if "embed_tokens." in name:
            parts["embed"] += n if param.ndim > 1 else 0
            continue
        # 1D parameters are norms, scales and biases (elementwise)
        if param.ndim < 2 or "embeddings.position_embedding." in name:
            continue

        if "vision_tower.embeddings.patch_embedding." in name:
            parts["vit_patch"] += n
        elif "vision_tower.layers." in name:
            parts["vit_layers"] += n
        elif "vision_tower.merger." in name:
            parts["vit_projector"] += n
        elif "message_input." in name:
            parts["core_message"] += n
        elif "final_input." in name:
            parts["coda"] += n
        elif "lm_head." in name:
            parts["lm_head"] += n
        elif layer:
            i = int(layer.group(1))
            if "experts." in name:
                n = round(n * expert_share)
            elif name.endswith("relative_position_bias.proj"):
                n *= heads[i]  # (d_rel, rel_extent) bank, multiplied once per head
            if "message_gate." in name:
                parts["core_message"] += n
            elif i < text.loop_layer_index_start:
                parts["prelude"] += n
            elif i < text.loop_layer_index_end:
                parts["core"] += n
            else:
                parts["coda"] += n
        else:
            raise ValueError(f"cost_params does not know where parameter {name} {tuple(param.shape)} runs")

    empty = [part for part, n in parts.items() if n == 0]
    if empty:
        raise ValueError(f"no parameters found for {empty}; module names differ from the expected model")

    return CostParams(
        **parts,
        core_start=text.loop_layer_index_start,
        core_end=text.loop_layer_index_end,
        attn_dims=tuple(h * d for h, d in zip(heads, head_dims)),
        attn_windows=tuple(text.sliding_window_size if s else None for s in sliding),
        vit_n_layers=vision.num_hidden_layers,
        vit_hidden=vision.hidden_size,
        vit_merge=vision.spatial_merge_size,
    )


def _causal_pairs(n: int, window: int | None) -> int:
    """Query-key pairs of causal attention over n tokens."""
    if window is None or n <= window:
        return n * (n + 1) // 2
    return window * (window + 1) // 2 + (n - window) * window


def _attention(params: CostParams, layers: range, n_tokens: int) -> int:
    """Q·K and attention·V over one run of `layers`: 4 FLOPs per pair per (head, head_dim) entry."""
    return sum(4 * params.attn_dims[i] * _causal_pairs(n_tokens, params.attn_windows[i]) for i in layers)


def _through_pass(params: CostParams, n_tokens: int, p: int) -> int:
    """Prelude plus the first p core passes."""
    if p < 1:
        raise ValueError(f"loop passes must be >= 1, got {p}")
    dense = params.prelude + p * params.core + (p - 1) * params.core_message
    prelude = range(0, params.core_start)
    core = range(params.core_start, params.core_end)
    return 2 * n_tokens * dense + _attention(params, prelude, n_tokens) + p * _attention(params, core, n_tokens)


def _coda(params: CostParams, n_tokens: int) -> int:
    layers = range(params.core_end, len(params.attn_dims))
    return 2 * n_tokens * params.coda + _attention(params, layers, n_tokens)


def flops_prefill(params: CostParams, n_tokens: int, R: int) -> int:
    """Prompt forward at R passes, without lm_head (counted with the first output token in `flops_decode`)."""
    return _through_pass(params, n_tokens, R) + _coda(params, n_tokens)


def flops_partial(params: CostParams, n_tokens: int, p: int) -> int:
    """Prompt forward stopped after pass p (no coda)."""
    return _through_pass(params, n_tokens, p)


def flops_readout(params: CostParams, n_tokens: int) -> int:
    """Coda over the prompt plus lm_head at one position: the extra cost of reading logits at a pass."""
    return _coda(params, n_tokens) + 2 * params.lm_head


def flops_decode(params: CostParams, n_out: int, ctx: int, R: int) -> int:
    """n_out cached steps after a prompt of ctx tokens; every step runs all R passes and lm_head."""
    if R < 1:
        raise ValueError(f"loop passes must be >= 1, got {R}")
    dense = params.prelude + R * params.core + (R - 1) * params.core_message + params.coda + params.lm_head
    total = 2 * n_out * dense
    for i, (dim, window) in enumerate(zip(params.attn_dims, params.attn_windows)):
        passes = R if params.core_start <= i < params.core_end else 1
        keys = sum(ctx + step if window is None else min(ctx + step, window) for step in range(1, n_out + 1))
        total += 4 * dim * passes * keys
    return total


def flops_vit(params: CostParams, image_grid_thw: Sequence[Sequence[int]]) -> int:
    """Vision tower and projector on the full image(s); independent of how many tokens are kept later."""
    total = 0
    for row in image_grid_thw:  # (t, h, w) in language-model image tokens; a (n_images, 3) tensor works
        t, h, w = (int(x) for x in row)
        n_tokens = t * h * w
        n_patches = n_tokens * params.vit_merge**2
        seq = n_patches + 1  # one class token per image
        total += 2 * (n_patches * params.vit_patch + seq * params.vit_layers + n_tokens * params.vit_projector)
        total += 4 * params.vit_n_layers * params.vit_hidden * seq**2  # not causal: seq^2 pairs
    return total


def flops_total(
    params: CostParams, n_tokens: int, n_out: int, R: int, image_grid_thw: Sequence[Sequence[int]]
) -> dict[str, int]:
    """Cost of one answer. n_tokens is the prompt length after pruning (text + kept visual tokens)."""
    parts = {
        "flops_vit": flops_vit(params, image_grid_thw),
        "flops_prefill": flops_prefill(params, n_tokens, R),
        "flops_decode": flops_decode(params, n_out, n_tokens, R),
    }
    parts["flops_total"] = sum(parts.values())
    return parts


def cost_ref(params: CostParams, n_tokens_full: int, n_out: int, image_grid_thw: Sequence[Sequence[int]]) -> int:
    """Total FLOPs with all visual tokens (k = 1.0) at R = 3."""
    return flops_total(params, n_tokens_full, n_out, REF_R, image_grid_thw)["flops_total"]
