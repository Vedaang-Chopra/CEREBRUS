"""Teacher distributions (S8). `answer_log_probs` is the teacher-forced forward the student will use too."""
from collections.abc import Mapping

import torch

from adaloop.core.model.prompting import append_ids
from adaloop.schemas.vlm import LoadedVLM


def answer_log_probs(vlm: LoadedVLM, inputs: Mapping, answer_ids: torch.Tensor) -> torch.Tensor:
    """Teacher-forced log-probs (n, V) of the n answer positions; keeps gradients."""
    n = len(answer_ids)
    # The last answer token is only predicted, never fed in.
    tensors = {key: value for key, value in inputs.items() if torch.is_tensor(value)}
    extended = {**inputs, **append_ids(tensors, answer_ids[:-1].tolist())}  # (1, T + n - 1)
    out = vlm.model(**extended, use_cache=False, logits_to_keep=n)  # the last n positions predict the n tokens
    return out.logits[0].float().log_softmax(dim=-1)  # (n, V)


def split_top(logp: torch.Tensor, top_n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(top ids (n, top_n), their log-probs, log of the remaining probability mass (n,)) of `logp` (n, V)."""
    if logp.shape[-1] <= top_n:
        raise ValueError(f"top_n {top_n} must be smaller than the vocabulary {logp.shape[-1]}")
    top_logp, top_ids = logp.topk(top_n, dim=-1)
    tail_logp = logp.scatter(-1, top_ids, float("-inf")).logsumexp(dim=-1)  # exact, no 1 - sum rounding
    return top_ids, top_logp, tail_logp


def target_tensors(answer_ids: torch.Tensor, logits: torch.Tensor, top_n: int) -> dict[str, torch.Tensor]:
    """The stored tensors of one record, on CPU. `logits` (n, V) are those that produced the n answer tokens."""
    if len(logits) != len(answer_ids):
        raise ValueError(f"{len(logits)} logit rows for {len(answer_ids)} answer tokens")
    top_ids, top_logp, tail_logp = split_top(logits.float().log_softmax(dim=-1), top_n)
    # clone: a view would drag its whole storage into the file; inference tensors cannot enter a backward graph
    return {
        "answer_ids": answer_ids.to(torch.int32).cpu().clone(),
        "top_ids": top_ids.to(torch.int32).cpu().clone(),
        "top_logp": top_logp.cpu().clone(),
        "tail_logp": tail_logp.cpu().clone(),
    }
