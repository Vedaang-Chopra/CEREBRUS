"""ReDiPrune: greedy relevance + diversity selection, before the projector."""
# Ported from UA-CVML/ReDiPrune:LLaVA-NeXT/llavavid/model/llava_arch.py @ ba04aa4427f7 (Apache-2.0)
# `_select_text_cond` is unchanged. `extract_question_embedding` changes:
#   - method -> function: `embed_tokens`, `image_token_id`, `hidden_size` replace `self` and IMAGE_TOKEN_INDEX
#   - token embeddings are cast to float32 (our embeddings are bfloat16)
#   - the attention mask must be boolean (as upstream passes it); the caller converts it
# Not ported: environment variables (alpha, tau come from the config; keep_ratio from k), the video/frame loop.
# Our use (docs/notes/compression_baselines.md): features are 2x2 units (mean of 4 patch states, 1024-d), and the
# 2304-d question embedding is brought to 1024-d by upstream's own dimension matching.
import torch
from torch.nn import functional as F

TEXT_QUERY_TAIL_MAX = 128
TEXT_QUERY_USE_COSINE = True


@torch.no_grad()
def _select_text_cond(
    visual_tokens: torch.Tensor,
    text_query: torch.Tensor,
    keep_ratio: float,
    alpha: float,
    tau: float
    ) -> torch.Tensor:
    """
    The method keeps a subset of visual tokens that are **diverse**
    (maximising pair-wise distance) and **relevant** to a text query.

    Scoring
    -------
    score = min_dist + alpha · relevance
        *min_dist*  – minimum distance to the current selection  
        *relevance* – cosine similarity with the query

    A relevance threshold *tau* optionally filters or top-k selects
    candidates before the greedy search.

    Parameters
    ----------
    visual_tokens : torch.Tensor
        Visual embeddings of shape ``(P, Cv)``.
    text_query : torch.Tensor
        Query embedding of shape ``(Dv,)`` or ``(1, Dv)``.
    keep_ratio : float
        Fraction of tokens to retain (0 < keep_ratio ≤ 1).
    alpha : float
        Weight of the relevance term in the scoring function.
    tau : float
        Relevance threshold. If ``tau > 0``, tokens with
        relevance < tau are discarded or subjected to a top-k filter.

    Returns
    -------
    torch.Tensor
        1-D tensor containing the indices of the retained tokens,
        sorted to preserve their original order.
    """
    P, Cv = visual_tokens.shape
    K = max(1, int(round(P * keep_ratio)))
    if K >= P:
        return torch.arange(P, device=visual_tokens.device)

    Vn = torch.nn.functional.normalize(visual_tokens, p=2, dim=1)

    # --- Match text-query dimensionality to Cv
    tq = text_query.unsqueeze(0) if text_query.dim() == 1 else text_query
    Dq = tq.shape[1]
    if Dq != Cv:
        if Dq > Cv:
            if Dq % Cv == 0:
                tq = tq.view(1, Cv, Dq // Cv).mean(dim=2)
            else:
                tq = torch.nn.functional.adaptive_avg_pool1d(tq.unsqueeze(1), Cv).squeeze(1)
        else:
            rep = (Cv + Dq - 1) // Dq
            tq = tq.repeat(1, rep)[:, :Cv]
    Tn = torch.nn.functional.normalize(tq, p=2, dim=1)

    # --- Relevance scores
    rel = (Vn @ Tn.t()).squeeze(1)

    # --- Candidate pre-selection
    if tau > 0:
        if K < P // 2:
            _, cand = torch.topk(rel, k=min(P, max(K * 3, K+1)))
        else:
            mask = (rel >= tau)
            if mask.sum() < K:
                _, cand = torch.topk(rel, K)
            else:
                cand = torch.where(mask)[0]
    else:
        cand = torch.arange(P, device=visual_tokens.device)

    # --- Greedy selection
    div_mat = 1.0 - (Vn[cand] @ Vn[cand].t())
    rel_c  = rel[cand]

    sel = torch.empty(K, dtype=torch.long, device=visual_tokens.device)
    sel[0] = torch.argmax(rel_c)

    Nc = cand.shape[0]
    for i in range(1, K):
        min_dist = div_mat[sel[:i]].min(dim=0).values
        score = min_dist + alpha * rel_c
        mask = torch.ones(Nc, dtype=torch.bool, device=visual_tokens.device)
        mask[sel[:i]] = False
        score[~mask] = -float('inf')
        sel[i] = torch.argmax(score)

    out_idx = cand[sel]
    out_idx, _ = torch.sort(out_idx)
    return out_idx


def extract_question_embedding(embed_tokens, image_token_id: int, hidden_size: int, input_ids, attention_mask):
    """Return one text-query embedding per sample.
    The embedding is computed from the *tail* of each prompt—that is,
    the tokens **after** the last image token.  If no image token is
    present, the entire sequence is used.
    Steps
    -----
    1.  Keep at most ``TEXT_QUERY_TAIL_MAX`` tail tokens.
    2.  Embed every token via the language model's token embedding.
    3.  If the tail has a single token, return its embedding.
    4.  Otherwise build a self-similarity matrix (cosine by default),
        derive an attention map, and compute a weighted sum
        (weights grow exponentially toward the end of the tail).
    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, D_text)`` where *B* is the batch size and
        *D_text* the model hidden size.
    """
    Lmax = TEXT_QUERY_TAIL_MAX
    use_cos = TEXT_QUERY_USE_COSINE

    H = hidden_size
    B = len(input_ids)
    out = []

    for i in range(B):
        ids = input_ids[i]
        mask = attention_mask[i] if attention_mask is not None else None
        if mask is not None:
            ids = ids[mask]

        # --- Tokens after the final image placeholder
        img_pos = torch.where(ids == image_token_id)[0]
        tail_ids = ids[img_pos[-1] + 1:] if len(img_pos)>0 else ids

        if tail_ids.numel() == 0:
            out.append(torch.zeros(H, device=ids.device, dtype=torch.float32))
            continue

        if Lmax > 0 and tail_ids.shape[0] > Lmax:
            tail_ids = tail_ids[-Lmax:]

        emb = embed_tokens(tail_ids).float()
        L, D = emb.shape

        if L == 1:
            out.append(emb.squeeze(0))
            continue

        # --- Self-similarity 
        if use_cos:
            emb_n = F.normalize(emb, dim=-1, eps=1e-12)
            scores = emb_n @ emb_n.T
        else:
            scores = (emb @ emb.T) / (D ** 0.5)

        scores = scores - scores.max(dim=-1, keepdim=True).values

        attn = torch.softmax(scores, dim=-1)
        context = attn @ emb

        # --- Exponentially increasing weights toward the tail end
        t = torch.linspace(0, 1, steps=L, device=emb.device, dtype=emb.dtype)
        w = torch.exp(3.0 * t)
        w = w / w.sum().clamp_min(1e-12)

        q_vec = (context * w.unsqueeze(1)).sum(dim=0)
        out.append(q_vec)

    return torch.stack(out, dim=0)
