"""Upfront choosers (S13: tokens_only, upfront_15way): one choice per item before the decoder runs."""
import torch
from torch import nn
from torch.nn import functional as F


class Chooser(nn.Module):
    """MLP on [projected mean visual token, projected mean question embedding] -> one logit per (k, R) choice.

    Same widths as the controller (`proj_dim`, `hidden`, `dropout`).
    """

    def __init__(self, d: int, n_choices: int, cfg: dict):
        super().__init__()
        self.visual = nn.Linear(d, cfg["proj_dim"])
        self.question = nn.Linear(d, cfg["proj_dim"])
        hidden, dropout = cfg["hidden"], cfg["dropout"]
        self.mlp = nn.Sequential(
            nn.Linear(2 * cfg["proj_dim"], hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_choices),
        )

    def forward(self, visual: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """(N, d), (N, d) -> (N, n_choices) logits."""
        return self.mlp(torch.cat([self.visual(visual), self.question(question)], dim=-1))


def chooser_loss(logits: torch.Tensor, rewards: torch.Tensor, entropy_bonus: float) -> torch.Tensor:
    """The controller's expected-reward objective for a one-step choice: -(sum_a pi(a) r(a) + alpha H), item mean."""
    log_probs = F.log_softmax(logits, dim=-1)  # (N, C)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return -((probs * rewards).sum(dim=-1) + entropy_bonus * entropy).mean()
