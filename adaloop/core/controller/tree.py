"""The trajectory tree of one item (build spec §5.2, DEC3 S11): actions, masks, paths and leaf costs.

Low branch: passes at k0. At pass p < R_max the controller may LOOK, which restarts the item with all tokens
(k = look_k) from pass 1; there it can only EXIT or LOOP. So every leaf is one cell of the fixed (k, R) grid.
"""
import numpy as np

from adaloop.core.cost.flops import REF_R, flops_partial, flops_prefill, flops_readout
from adaloop.schemas.cost import CostParams

EXIT, LOOP, LOOK = 0, 1, 2
ACTIONS = ("EXIT", "LOOP", "LOOK")
# need coda + lm_head at that pass (pred_changed also the previous pass's readout, which that pass's decision paid)
READOUT_FEATURES = ("entropy", "kl_prev", "top1_prob", "top2_margin", "pred_changed")
STATE_FEATURES = ("delta_norm", "vis_attn_mass")  # cheap sums over states / attention weights


def path_name(look_pass: int, exit_pass: int) -> str:
    """"X@r" without a look, "L@p,X@r" with a look after pass p."""
    return f"L@{look_pass},X@{exit_pass}" if look_pass else f"X@{exit_pass}"


def action_mask(n_branches: int, R_max: int) -> np.ndarray:
    """(B, R, 3) bool: no LOOP or LOOK at pass R_max; LOOK only on branch 0 and only if there are look branches."""
    mask = np.zeros((n_branches, R_max, 3), dtype=bool)
    mask[:, :, EXIT] = True
    mask[:, : R_max - 1, LOOP] = True
    if n_branches > 1:
        assert n_branches == R_max, f"one look branch per pass 1..R_max-1: B={n_branches}, R={R_max}"
        mask[0, : R_max - 1, LOOK] = True
    return mask


def walk(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Follow (N, B, R) chosen actions from node (0, 0); return the exit node (branch (N,), pass index (N,))."""
    n, n_branches, R_max = actions.shape
    item = np.arange(n)
    b, q, done = np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    mask = action_mask(n_branches, R_max)
    for _ in range(2 * R_max):
        a = np.where(done, EXIT, actions[item, b, q])
        assert mask[b, q, a].all(), "a policy chose a masked action"
        done |= a == EXIT
        look, loop = ~done & (a == LOOK), ~done & (a == LOOP)
        b = np.where(look, q + 1, b)  # a look after pass p = q + 1 goes to branch p, pass 1
        q = np.where(look, 0, q + loop)
    assert done.all(), "walk did not reach an exit"
    return b, q


def _checked_tokens(params: CostParams, cells: dict, feats: dict, k: float, R_max: int) -> int:
    """Prompt length of branch k; the analytic parameters must reproduce the saved grid and feature FLOPs."""
    n = int(cells[k, 1]["n_txt"] + cells[k, 1]["n_vis"])
    for R in range(1, R_max + 1):
        saved = (int(cells[k, R]["flops_prefill"]), int(feats[k, R]["flops_readout"]))
        again = (flops_prefill(params, n, R), flops_readout(params, n))
        if saved != again:
            raise ValueError(f"k={k} R={R}: saved (prefill, readout) FLOPs {saved} != {again} from the cost "
                             "parameters (wrong model config or a different cost model)")
    return n


def item_tree(cells: dict, feats: dict, params: CostParams, k0: float, look_k: float, R_max: int) -> list[dict]:
    """All leaves of one item. `cells[k, R]`: grid row, `feats[k, p]`: saved features. k0 == look_k: no look.

    Costs: flops_cell = the grid row that answers. flops_discarded (look only) = low-branch compression + prelude and
    p core passes (plus its vision run if it differs from the full one; otherwise the vision run is reused).
    flops_readout = every early readout used for a decision (the exit pass's readout is already in the cell).
    flops_state = delta_norm + attention mass at every pass with a decision (none at R_max, where EXIT is forced).
    """
    looks = k0 != look_k
    branches = [(0, k0)] + [(p, look_k) for p in range(1, R_max)] * looks
    tokens = {k: _checked_tokens(params, cells, feats, k, R_max) for k in {k for _, k in branches}}
    ref = cells[look_k, REF_R]
    spent_ref = ref["flops_vit"] + ref["flops_prefill"]
    leaves = []
    for p, k in branches:
        low = cells[k0, 1]  # compression and vision cost do not depend on R
        extra_vit = low["flops_vit"] if p and low["flops_vit"] != cells[k, 1]["flops_vit"] else 0
        discarded = flops_partial(params, tokens[k0], p) + low["flops_compression"] + extra_vit if p else 0
        before = {name: sum(feats[k0, q][name] for q in range(1, p + 1)) for name in ("flops_readout", "flops_state")}
        for r in range(1, R_max + 1):
            cell = cells[k, r]
            readout = before["flops_readout"] + sum(feats[k, q]["flops_readout"] for q in range(1, r))
            state = before["flops_state"] + sum(feats[k, q]["flops_state"] for q in range(1, min(r, R_max - 1) + 1))
            spent = discarded + cell["flops_vit"] + cell["flops_compression"] + flops_partial(params, tokens[k], r)
            leaves.append({
                "path": path_name(p, r), "look_pass": p, "exit_pass": r, "correct": float(cell["correct"]),
                "flops_cell": int(cell["flops_total"]), "flops_discarded": int(discarded),
                "flops_readout": int(readout), "flops_state": int(state), "cost_ref": int(ref["flops_total"]),
                "spent_frac": float(spent / spent_ref), "kept_frac": float(cell["n_vis"] / cell["n_vis_full"]),
            })
    return leaves


def charged(features: list[str] | tuple[str, ...]) -> tuple[bool, bool]:
    """(pays readouts, pays state sums) for a controller that reads `features`; h_last is a copy and free."""
    return bool(set(features) & set(READOUT_FEATURES)), bool(set(features) & set(STATE_FEATURES))
