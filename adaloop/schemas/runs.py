"""Records of the controller stages (S11, S12): tree leaves, policy results and the array form of the trees."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TreeLeaf:
    """One leaf of one item's tree (a row of tree.parquet).

    flops_total = flops_cell + flops_discarded + charged feature costs (flops_readout, flops_state).
    """

    item_id: str
    benchmark: str
    split: str
    model_variant: str
    k0: float
    path: str  # "X@r" or "L@p,X@r"
    look_pass: int  # 0: no look
    exit_pass: int
    correct: float
    flops_total: int
    flops_cell: int  # the grid cell that gives the answer
    flops_discarded: int  # low-token work thrown away by the look (compression + passes)
    flops_readout: int  # early readouts (coda + lm_head) before the exit pass
    flops_state: int  # delta_norm and attention mass at every pass with a decision
    cost_ref: int  # flops_total of the (k = 1, R = 3) cell of this item
    spent_frac: float  # controller input: work done when deciding here / (ViT + prefill at k = 1, R = 3)
    kept_frac: float  # controller input: kept / all visual tokens on this branch


@dataclass(frozen=True)
class PolicyRow:
    """One method's path on one item (a row of policy_eval.parquet). Costs as in `TreeLeaf`.

    Methods with a trade-off weight set `lam` (cap_frac NaN); best_fixed and oracle are defined per cost cap and set
    `cap_frac` (lam NaN; best_fixed, best_fixed_cell, oracle). `k0` is 0.0 when the method has no common first keep
    fraction (tokens_only, upfront_15way, best_fixed, oracle); best_fixed_cell holds its cell's k there. best_fixed is
    a random mix of two cells: its rows hold the expected `correct` and FLOPs of the item.
    """

    method: str
    lam: float
    cap_frac: float
    seed: int
    model_variant: str
    item_id: str
    benchmark: str
    split: str
    k0: float
    path: str
    correct: float
    flops_total: int
    flops_cell: int
    flops_discarded: int
    flops_readout: int  # 0 when the method is not charged for it
    flops_state: int
    cost_ref: int


@dataclass(frozen=True)
class TreeData:
    """The trees of N items at one k0 as arrays. Node (b, q): branch b, pass q+1.

    Branch 0 runs at k0; branch p >= 1 is the restart at full tokens after a look at pass p. B = R_max with looks,
    1 without. The exit leaf of node (b, q) holds correct[:, b, q] and the costs.
    """

    keys: pd.DataFrame  # benchmark, item_id, split, model_variant (N rows)
    k0: float
    h: np.ndarray  # (N, U, d) float32: last-token states, U = R_max (branch k0) or 2 * R_max (plus full branch)
    h_index: np.ndarray  # (B, R) int: which of the U states node (b, q) reads
    raw: dict[str, np.ndarray]  # feature name -> (N, B, R), NaN where undefined (delta_norm, kl_prev at pass 1)
    kept_frac: np.ndarray  # (N, B, R)
    spent_frac: np.ndarray  # (N, B, R)
    correct: np.ndarray  # (N, B, R)
    flops_cell: np.ndarray  # (N, B, R)
    flops_discarded: np.ndarray  # (N, B, R)
    flops_readout: np.ndarray  # (N, B, R)
    flops_state: np.ndarray  # (N, B, R)
    cost_ref: np.ndarray  # (N,)
