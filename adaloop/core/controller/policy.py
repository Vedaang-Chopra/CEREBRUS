"""The controller network: [projected h_last, node scalars] -> logits of EXIT, LOOP, LOOK (build spec §5.1)."""
import torch
from torch import nn


class Policy(nn.Module):
    """MLP [F] -> hidden -> hidden -> 3 with GELU and dropout; h_last is first projected to `proj_dim`."""

    def __init__(self, d_state: int, n_scalars: int, cfg: dict):
        """cfg: `use_h_last` (False: scalar features only), `proj_dim`, `hidden`, `dropout`."""
        super().__init__()
        self.project = nn.Linear(d_state, cfg["proj_dim"]) if cfg["use_h_last"] else None
        width = n_scalars + (cfg["proj_dim"] if cfg["use_h_last"] else 0)
        hidden, dropout = cfg["hidden"], cfg["dropout"]
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, h: torch.Tensor, scalars: torch.Tensor, h_index: torch.Tensor) -> torch.Tensor:
        """h (N, U, d), scalars (N, B, R, S), h_index (B, R) -> logits (N, B, R, 3), unmasked."""
        if self.project is not None:
            z = self.project(h)[:, h_index]  # (N, B, R, proj): each distinct state is projected once
            scalars = torch.cat([z, scalars], dim=-1)
        return self.mlp(scalars)
