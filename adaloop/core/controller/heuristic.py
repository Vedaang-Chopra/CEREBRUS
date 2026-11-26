"""Rule-based controller (build spec §5.1): thresholds on the readout features and the visual attention mass.

Pass 1:  EXIT if top1_prob >= tau1 (there is no earlier pass, so no KL exit).
Pass 2+: EXIT if kl_prev < eps.
Else:    LOOK if entropy > eta and vis_attn_mass > zeta and a look remains; else LOOP. Pass R_max: EXIT.
"""
from itertools import product

import numpy as np

from adaloop.core.controller.tree import EXIT, LOOK, LOOP, action_mask
from adaloop.schemas.runs import TreeData

HEURISTIC_FEATURES = ("kl_prev", "entropy", "top1_prob", "vis_attn_mass")  # what it reads, so what it pays for


def heuristic_actions(data: TreeData, eps: float, eta: float, zeta: float, tau1: float) -> np.ndarray:
    """(N, B, R) action at every node."""
    n, n_branches, R_max = data.correct.shape
    mask = action_mask(n_branches, R_max)
    first = np.arange(R_max) == 0
    with np.errstate(invalid="ignore"):  # kl_prev is NaN at pass 1
        stop = np.where(first, data.raw["top1_prob"] >= tau1, data.raw["kl_prev"] < eps)
        look = (data.raw["entropy"] > eta) & (data.raw["vis_attn_mass"] > zeta) & mask[:, :, LOOK]
    actions = np.where(look, LOOK, LOOP)
    return np.where(stop | ~mask[:, :, LOOP], EXIT, actions)


def threshold_grid(data: TreeData, grid_cfg: dict) -> list[dict]:
    """Every (eps, eta, zeta, tau1) to try. eps, eta, zeta are quantiles of the (dev) node values, tau1 is raw."""
    def quantiles(name: str, qs: list[float]) -> list[float]:
        values = data.raw[name][np.isfinite(data.raw[name])]
        return sorted(set(np.quantile(values, qs).tolist()))

    axes = [quantiles("kl_prev", grid_cfg["eps_quantiles"]), quantiles("entropy", grid_cfg["eta_quantiles"]),
            quantiles("vis_attn_mass", grid_cfg["zeta_quantiles"]), grid_cfg["tau1"]]
    return [dict(zip(("eps", "eta", "zeta", "tau1"), values)) for values in product(*axes)]
