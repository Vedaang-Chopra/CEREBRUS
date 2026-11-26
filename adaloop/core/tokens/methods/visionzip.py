"""VisionZip: keep the dominant tokens by attention, merge each other token into its most similar contextual token."""
# Ported from dvlab-research/VisionZip:Qwen2_5_VL/qwen2_5vl_visionzip.py @ 8f86b55c6f00 (Apache-2.0), the block under
# `if select_pixel:`. The selection, key normalization, assignment and merge lines are unchanged. Changes:
#   - `dominant_num` and `contextual_num` are arguments (upstream: fixed fractions 0.65 / 0.05 of the token count)
#   - the lines are wrapped in a function on the visual tokens only: `contextual` replaces the slice of
#     `inputs_embeds` between the first and last image token; position_ids / input_ids editing is left to the caller
#   - `contextual_num` must be >= 1 and at most the number of non-dominant tokens
# Input scores (computed by `core/tokens/vision.py`): upstream sums eager attention over all queries of the last
# ViT block (Qwen2.5-VL has no CLS token). Our tower has one, so the score is the CLS attention, as in VisionZip's
# LLaVA path. Keys are the rotary-encoded keys of the last block, 2x2-unit mean, head mean, as upstream.
import torch


def visionzip_merge(
    attn_logits: torch.Tensor, attn_key: torch.Tensor, tokens: torch.Tensor, dominant_num: int, contextual_num: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (indices of the kept tokens, sorted; token features with the merged contextual tokens written in).

    attn_logits (N,) attention score per token, attn_key (1, N, D) mean key per token, tokens (N, d).
    """
    if not 1 <= contextual_num <= attn_logits.size(0) - dominant_num:
        raise ValueError(f"contextual_num {contextual_num} needs 1..{attn_logits.size(0) - dominant_num}")
    inputs_embeds = tokens.unsqueeze(0).clone()  # (1, N, d)
    topk_values, topk_indices = torch.topk(attn_logits, dominant_num)

    mask = torch.zeros_like(attn_logits, dtype=torch.bool)
    mask[topk_indices] = True
    contextual_mask = ~mask
    metric_filtered = attn_key[:,contextual_mask]
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)
    del attn_key, metric_filtered

    ## Contextual Visual Tokens
    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=attn_logits.dtype, device=metric_normalized.device)
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

    select_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
    select_mask[topk_indices] = True

    false_pos = (~select_mask).nonzero(as_tuple=True)[0]

    select_mask[false_pos[target_indices]] = True

    contexual_input_idx = false_pos[target_indices]
    hidden_states_filtered = inputs_embeds[:, contextual_mask.to(inputs_embeds.device)]
    assign_one_hot = assign_one_hot.to(hidden_states_filtered.device, hidden_states_filtered.dtype)
    counts = counts.to(hidden_states_filtered.device, hidden_states_filtered.dtype)
    target_indices = target_indices.to(hidden_states_filtered.device)
    hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
    target_hidden = hidden_states_filtered[:, target_indices, :]

    contextual_tokens = target_hidden + aggregated_hidden
    inputs_embeds[:, contexual_input_idx.to(inputs_embeds.device)] = contextual_tokens

    return select_mask.nonzero(as_tuple=True)[0], inputs_embeds[0]
