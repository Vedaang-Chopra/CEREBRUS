"""Per-pass controller features. Definitions: docs/02_loop/adaloop_build_spec.md (feature table)."""
import torch
from torch.nn import functional as F


def pass_features(
    states: list[torch.Tensor], vis_attn_mass: float, logits: torch.Tensor, prev_logits: torch.Tensor | None
) -> dict:
    """Features of pass p = len(states) from the R = p run.

    states: last-token hidden state after each pass; logits: first answer-token logits of this run; prev_logits:
    the same from the R = p-1 run. `delta_norm` and `kl_prev` are NaN when there is no earlier pass.
    """
    h = states[-1]
    delta = float("nan")
    if len(states) > 1:
        delta = ((h - states[-2]).norm() / states[-2].norm()).item()  # ||h_p - h_{p-1}|| / ||h_{p-1}||
    log_p = F.log_softmax(logits.float(), dim=-1)  # (vocab,)
    p = log_p.exp()
    kl = float("nan")
    if prev_logits is not None:  # KL(p_p || p_{p-1}), as in the build spec's exit rule
        kl = (p * (log_p - F.log_softmax(prev_logits.float(), dim=-1))).sum().item()
    return {
        "h_last": h.half().clone(),
        "delta_norm": delta,
        "vis_attn_mass": vis_attn_mass,
        "entropy": -(p * log_p).sum().item(),  # nats
        "kl_prev": kl,
        "top1_prob": p.max().item(),
    }
