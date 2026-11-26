"""Data contracts of LoRA distillation (stage S9)."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TeacherTarget:
    """One cached teacher record: the teacher's greedy answer on one item at loop count R.

    `answer_ids` (n,) are the generated ids, end token included. `top_ids` and `top_logp` (n, top_n) are the most
    likely tokens and their log-probs (full-vocabulary softmax); `tail_logp` (n,) is the log of the summed
    probability of every other token. All tensors are on CPU.
    """

    benchmark: str
    item_id: str
    R: int
    answer_ids: torch.Tensor
    top_ids: torch.Tensor
    top_logp: torch.Tensor
    tail_logp: torch.Tensor


@dataclass(frozen=True)
class TrainSample:
    """Training sample number `index`: item position, keep fraction, student loop count and teacher loop count."""

    index: int
    item_n: int
    k: float
    R: int
    R_teacher: int
