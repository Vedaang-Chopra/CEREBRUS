"""Visual-token selectors. Each returns sorted indices of the tokens to keep.

Ported from notebooks/02_pruning.ipynb.
"""
import logging

import torch
from torch.nn import functional as F

from adaloop.schemas.vlm import LoadedVLM

logger = logging.getLogger(__name__)


def _n_keep(n_tokens: int, k: float) -> int:
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}")
    return max(1, int(n_tokens * k))


def random_selector(visual_tokens: torch.Tensor, k: float, seed: int = 42) -> torch.Tensor:
    """Keep a seeded random fraction `k` of the tokens."""
    n_tokens = len(visual_tokens)
    n_keep = _n_keep(n_tokens, k)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_tokens, generator=generator)[:n_keep].sort().values
    logger.info("random selector kept %d/%d", n_keep, n_tokens)
    return indices.to(visual_tokens.device)


def uniform_selector(image_grid_thw: torch.Tensor, k: float) -> torch.Tensor:
    """Keep a spatially spread fraction `k` by farthest-point sampling on the grid."""
    t, h, w = image_grid_thw[0].tolist()
    if t != 1:
        raise ValueError(f"only single-frame images are supported, got t={t}")
    n_tokens = h * w
    n_keep = _n_keep(n_tokens, k)
    logger.info("uniform selector keeping %d/%d", n_keep, n_tokens)
    if n_keep == n_tokens:
        return torch.arange(n_tokens)

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([y, x], dim=-1).reshape(-1, 2).float()
    selected: list[int] = []
    distances = torch.full((n_tokens,), float("inf"))
    index = 0
    for _ in range(n_keep):
        selected.append(index)
        distances = torch.minimum(distances, ((coords - coords[index]) ** 2).sum(-1))
        distances[selected] = -1
        index = int(distances.argmax())
    return torch.tensor(sorted(selected))


@torch.inference_mode()
def question_embeddings(vlm: LoadedVLM, question: str) -> torch.Tensor:
    """Decoder input embeddings of the question tokens, [n_tokens, d]."""
    logger.info("embedding question of %d characters", len(question))
    layer = vlm.model.model.get_input_embeddings()
    ids = vlm.processor.tokenizer(question, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embeddings = layer(ids.to(layer.weight.device)).squeeze(0)
    logger.info("question embeddings %s", tuple(embeddings.shape))
    return embeddings


def text_mean_selector(
    visual_tokens: torch.Tensor, text_embeddings: torch.Tensor, k: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the tokens closest (cosine) to the mean question embedding; also return scores."""
    n_keep = _n_keep(len(visual_tokens), k)
    visual = F.normalize(visual_tokens.float(), dim=-1)
    query = text_embeddings.float().to(visual.device)
    query = F.normalize(query.mean(dim=0), dim=-1)
    scores = visual @ query
    logger.info("text-mean selector kept %d/%d", n_keep, len(visual_tokens))
    return scores.topk(n_keep).indices.sort().values, scores
