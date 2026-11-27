"""Per-pass controller features. Definitions: docs/02_loop/adaloop_build_spec.md (feature table)."""
import torch
from torch.nn import functional as F

TOP_N = 32  # first-answer-token log-probs kept per pass (a stored-data format, not a config value)


def pass_features(
    states: list[torch.Tensor], vis_attn_mass: float, logits: torch.Tensor, prev_logits: torch.Tensor | None
) -> dict:
    """Features of pass p = len(states) from the R = p run.

    states: last-token hidden state after each pass; logits: first answer-token logits of this run; prev_logits:
    the same from the R = p-1 run. `delta_norm` and `kl_prev` are NaN when there is no earlier pass.
    `top_logp` (fp32), `top_ids` (int32): the TOP_N largest log-probs of the first answer token, largest first;
    `tail_mass`: the probability outside them.
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
    top = log_p.topk(min(TOP_N, log_p.numel()))  # kept so other readout features never need a rerun
    return {
        "h_last": h.half().clone(),
        "top_logp": top.values.clone(),  # (TOP_N,)
        "top_ids": top.indices.to(torch.int32),  # (TOP_N,)
        "tail_mass": (1.0 - top.values.exp().sum()).clamp(min=0.0).item(),
        "delta_norm": delta,
        "vis_attn_mass": vis_attn_mass,
        "entropy": -(p * log_p).sum().item(),  # nats
        "kl_prev": kl,
        "top1_prob": p.max().item(),
    }
