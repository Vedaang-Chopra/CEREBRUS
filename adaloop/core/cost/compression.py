"""Analytic FLOPs of the compression step itself (2 per multiply-add). The vision tower and decoder are in flops.py.

N = visual tokens, K = kept, d = decoder width. Cheap bookkeeping (softmax, argmax, sorting)
and the spaCy tagger of crisp_lite are not counted.
"""


def _cosine(n: int, m: int, d: int) -> int:
    """Normalize n and m vectors of width d, then all n*m dot products."""
    return 4 * d * (n + m) + 2 * n * m * d


def compression_flops(method: str, N: int, K: int, d: int, n_words: int = 0, head_dim: int = 0,
                      n_heads: int = 0, n_patches_per_unit: int = 4, n_contextual: int = 0) -> int:
    """FLOPs of choosing K of N tokens with `method`; 0 when everything is kept (not cls_iqr: it always scores)."""
    # one query (CLS) against every patch key, all heads, plus the q/k that the layer already computes (not counted)
    cls_attention = 2 * (N * n_patches_per_unit + 1) * n_heads * head_dim
    if method == "cls_iqr":  # the count comes out of the scores; the quartiles are bookkeeping
        return cls_attention
    if K >= N or method in ("none", "random", "uniform", "lowres"):
        return 0
    greedy = N * K * K // 2  # min over the selected set, once per pick
    if method == "text_mean":
        return _cosine(N, 1, d)
    if method == "divprune":
        return _cosine(N, N, d) + greedy
    if method == "rediprune":  # n_words = question-tail tokens (query self-attention), then scores in d
        return 2 * _cosine(n_words, n_words, d) + _cosine(N, N, d) + _cosine(N, 1, d) + greedy
    if method == "crisp_lite":  # n_words = nouns; 0 means the CLS-attention fallback
        return _cosine(N, n_words, d) if n_words else cls_attention
    if method == "visionzip":
        merge = 2 * (N - K + n_contextual) * n_contextual * head_dim + (N - K + n_contextual) * d
        return cls_attention + merge
    raise ValueError(f"unknown compressor {method!r}")
