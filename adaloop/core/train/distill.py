"""Distillation loss of the student and the model state it is trained in."""
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch

from adaloop.core.model.loops import set_loops
from adaloop.core.train.teacher import answer_log_probs
from adaloop.schemas.train import TeacherTarget
from adaloop.schemas.vlm import LoadedVLM


def kl_topk(
    student_logp: torch.Tensor, top_ids: torch.Tensor, top_logp: torch.Tensor, tail_logp: torch.Tensor
) -> torch.Tensor:
    """KL(teacher || student) per answer position over top_n + 1 outcomes; returns (n,).

    `student_logp` (n, V); `top_ids`, `top_logp` (n, top_n); `tail_logp` (n,). The outcomes are the teacher's top
    tokens and one bucket "any other token". This is the exact KL of the coarsened distributions: >= 0, zero when
    student = teacher, and a lower bound of the full KL (how mass is spread inside the tail is not penalized).
    """
    ### Step-1:- Student log-probs of the teacher's top tokens, and of everything else together
    student_top = student_logp.gather(-1, top_ids)  # (n, top_n)
    student_tail = student_logp.scatter(-1, top_ids, float("-inf")).logsumexp(dim=-1, keepdim=True)  # (n, 1)
    s = torch.cat([student_top, student_tail], dim=-1)  # (n, top_n + 1)
    t = torch.cat([top_logp, tail_logp[:, None]], dim=-1)
    ### Step-2:- sum_i p_t(i) * (log p_t(i) - log p_s(i))
    return (t.exp() * (t - s)).sum(dim=-1)


def distill_loss(vlm: LoadedVLM, inputs: Mapping, target: TeacherTarget, R: int) -> torch.Tensor:
    """Mean KL over all answer positions (end token included); the student runs `inputs` with R loop passes.

    Teacher-forced on the teacher's answer. Gradients go through every pass: nothing is detached between passes.
    """
    set_loops(vlm, R)  # the student's R, which may differ from target.R
    logp = answer_log_probs(vlm, inputs, target.answer_ids.long())  # (n, V)
    device = logp.device  # the lm_head's GPU
    return kl_topk(logp, target.top_ids.long().to(device), target.top_logp.to(device),
                   target.tail_logp.to(device)).mean()


@contextmanager
def training_mode(vlm: LoadedVLM, activation_checkpointing: bool) -> Iterator[None]:
    """Train mode on the language model for the block; back to eval without checkpointing afterwards."""
    language_model = vlm.model.model.language_model
    # checkpointing only acts in train mode; with no dropout the outputs are those of eval mode
    if language_model.config.attention_dropout:
        raise ValueError(f"attention_dropout {language_model.config.attention_dropout} would add noise in train mode")
    language_model.train()
    try:
        if activation_checkpointing:
            language_model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
            # the enable call makes the embeddings require grad; non-reentrant checkpointing does not need that, and
            # it would send a backward pass through the frozen layers before the core
            language_model.disable_input_require_grads()
        yield
    finally:
        if activation_checkpointing:
            language_model.activation_checkpointing_disable()
        language_model.eval()
