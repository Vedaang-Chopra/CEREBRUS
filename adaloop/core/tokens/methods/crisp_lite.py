"""crisp_lite: tokens most similar to the question's nouns, plus a spatially uniform fill."""
# Reimplemented from the CRISP paper (#052, arXiv 2607.16326); no official code exists. This is NOT full CRISP:
#   - kept from the paper: spaCy `en_core_web_sm` nouns; a noun's embedding is the mean of its subword decoder input
#     embeddings; cosine to the projected visual tokens; with no noun, rank by the CLS attention of the vision tower
#   - simplified: one top-k by the best noun score (no per-noun quota weighted by CLS attention, Eqs. 3-4)
#   - replaced: the second stage is a spatially uniform fill, not max-min feature diversity (Eq. 5)
#   - the share of the fill is `uniform_frac` (build brief: 25%), not 1 - alpha (paper: 70%)
from functools import lru_cache

import torch
from torch.nn import functional as F

NOUN_TAGS = ("NOUN", "PROPN")


@lru_cache(maxsize=1)
def _tagger(model_name: str):
    """The spaCy pipeline, loaded once."""
    import spacy  # imported here so that only this method needs spaCy

    return spacy.load(model_name)


def question_nouns(question: str, model_name: str) -> list[str]:
    """Nouns of the question, in order."""
    return [token.text for token in _tagger(model_name)(question) if token.pos_ in NOUN_TAGS]


@torch.inference_mode()
def word_embeddings(embed_tokens, tokenizer, words: list[str]) -> torch.Tensor:
    """One embedding per word: the mean of its subword input embeddings, (n_words, d)."""
    device = embed_tokens.weight.device
    rows = []
    for word in words:
        ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
        rows.append(embed_tokens(torch.tensor(ids, device=device)).float().mean(dim=0))
    return torch.stack(rows)


def noun_relevance(visual_tokens: torch.Tensor, nouns: torch.Tensor) -> torch.Tensor:
    """Per token, the largest cosine similarity to any noun, (N,)."""
    visual = F.normalize(visual_tokens.float(), dim=-1)
    words = F.normalize(nouns.to(visual.device), dim=-1)
    return (visual @ words.T).max(dim=1).values
