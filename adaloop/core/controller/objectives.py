"""Training objectives of the controller on full trajectory trees (build spec §5.3, DEC3 S12). CORE: review this.

Every leaf of every tree is known, so nothing is sampled. Shapes: N items, B branches, R passes, A = 3 actions
(EXIT, LOOP, LOOK). Node (b, q) is pass q+1 of branch b. Its children:
    EXIT -> the leaf of (b, q)
    LOOP -> node (b, q+1)
    LOOK -> node (q+1, 0)   only from branch 0: branch p is the full-token restart after a look at pass p

Reward of a leaf (rule based, no learned reward model):
    r = correct - lam * cost(path) / cost_ref(item)
correct: the benchmark score in [0, 1]; cost(path): flops_total of the leaf (answering cell + discarded low-token
work + feature readouts used to get there); cost_ref(item): flops_total of the fixed (k = 100%, R = 3) run of the
same item. The safety and format penalties of build spec §4.2 are not used (mu = nu = 0).
"""
import torch
from torch.nn import functional as F

from adaloop.core.controller.tree import LOOK, LOOP

NODE_WEIGHTS = ("uniform", "reach")


def leaf_rewards(correct: torch.Tensor, cost: torch.Tensor, cost_ref: torch.Tensor, lam: float) -> torch.Tensor:
    """(N, B, R) rewards from (N, B, R) correct and cost and (N,) cost_ref."""
    return correct - lam * cost / cost_ref[:, None, None]


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log pi over the allowed actions; (..., 3). Forbidden actions get -inf (probability 0)."""
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def backward_values(r_exit: torch.Tensor, mask: torch.Tensor,
                    probs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact backward recursion over the tree: (Q (N, B, R, 3), V (N, B, R)).

    Q(node, EXIT) = r_exit(node);  Q(node, LOOP | LOOK) = V(child)
    V(node) = sum_a probs(a) Q(node, a)   with `probs` (N, B, R, 3): the value under that policy
    V(node) = max_a Q(node, a)            with probs None: the best reachable value (hindsight optimum)
    mask: (B, R, 3). Q of a forbidden action is -inf.
    """
    n_branches, R_max = r_exit.shape[1], r_exit.shape[2]
    q_nodes = [[None] * R_max for _ in range(n_branches)]
    v_nodes = [[None] * R_max for _ in range(n_branches)]
    ### Step-1:- Look branches first (branch 0 needs their root values); within a branch from the last pass back
    for b in [*range(1, n_branches), 0]:
        for q in reversed(range(R_max)):
            actions = [r_exit[:, b, q]]  # EXIT
            actions.append(v_nodes[b][q + 1] if mask[b, q, LOOP] else None)
            actions.append(v_nodes[q + 1][0] if mask[b, q, LOOK] else None)
            missing = torch.full_like(r_exit[:, b, q], float("-inf"))
            q_node = torch.stack([missing if a is None else a for a in actions], dim=-1)  # (N, 3)
            q_nodes[b][q] = q_node
            if probs is None:
                v_nodes[b][q] = q_node.max(dim=-1).values
            else:  # 0 * -inf is nan, so forbidden actions are zeroed before the product
                v_nodes[b][q] = (probs[:, b, q] * q_node.masked_fill(~mask[b, q], 0.0)).sum(dim=-1)
    q_all = torch.stack([torch.stack(row, dim=1) for row in q_nodes], dim=1)  # (N, B, R, 3)
    v_all = torch.stack([torch.stack(row, dim=1) for row in v_nodes], dim=1)  # (N, B, R)
    return q_all, v_all


def supervised_targets(r_exit: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(a) The argmax-Q action of every node under optimal play below it. Ties: EXIT, then LOOP, then LOOK."""
    q, _ = backward_values(r_exit, mask)
    return q.argmax(dim=-1)  # first maximum = lowest action index


def supervised_loss(logits: torch.Tensor, r_exit: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(a) Cross-entropy to the argmax-Q action, summed over nodes, averaged over items.

    Nodes with only EXIT allowed contribute 0.
    """
    targets = supervised_targets(r_exit, mask)  # (N, B, R)
    log_probs = masked_log_probs(logits, mask)
    return -log_probs.gather(-1, targets[..., None]).sum() / len(logits)


def reach_probabilities(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(N, B, R): probability of reaching each node from the root (branch 0, pass 1) under `probs` (N, B, R, 3)."""
    n_branches, R_max = probs.shape[1], probs.shape[2]
    nodes = [[None] * R_max for _ in range(n_branches)]
    nodes[0][0] = torch.ones_like(probs[:, 0, 0, 0])
    for b in range(n_branches):  # branch 0 first: it feeds the first node of every look branch
        if b:
            nodes[b][0] = nodes[0][b - 1] * probs[:, 0, b - 1, LOOK] * mask[0, b - 1, LOOK]
        for q in range(R_max - 1):
            nodes[b][q + 1] = nodes[b][q] * probs[:, b, q, LOOP] * mask[b, q, LOOP]
    return torch.stack([torch.stack(row, dim=1) for row in nodes], dim=1)


def expected_reward_loss(logits: torch.Tensor, r_exit: torch.Tensor, mask: torch.Tensor,
                         entropy_bonus: float, node_weight: str = "uniform") -> torch.Tensor:
    """(b) loss = -J - alpha * H, with J = mean_items sum_nodes sum_a pi(a | f_node) Q(node, a) (build spec §5.3).

    Q is computed bottom-up under the CURRENT policy and held fixed (detached): each node is pushed toward the
    actions that are better given how the policy behaves below it, and every node gets that push, also rarely
    visited ones. The gradient does not flow through the children's values into a parent's term.
    H: policy entropy summed the same way; alpha = `entropy_bonus`.

    `node_weight`, the weight of a node's term in both sums:
        "uniform" (default, build spec §5.3): 1. The sum runs over all tree nodes, never-reached ones included, and
            the full-token subtree counts once per look point.
        "reach" (not in the spec; an option to compare on ctrl_val): the probability of reaching the node under the
            current policy, held fixed. With alpha = 0 the gradient is then exactly that of -mean_items V(root).
    """
    log_probs = masked_log_probs(logits, mask)
    probs = log_probs.exp()
    q, _ = backward_values(r_exit, mask, probs.detach())
    q = q.masked_fill(~mask, 0.0)  # forbidden: probability 0, keep the product finite
    if node_weight not in NODE_WEIGHTS:
        raise ValueError(f"node_weight must be one of {NODE_WEIGHTS}, got {node_weight!r}")
    weight = reach_probabilities(probs.detach(), mask)[..., None] if node_weight == "reach" else 1.0
    gain = (weight * probs * q).sum()
    entropy = -(weight * probs * log_probs.masked_fill(~mask, 0.0)).sum()
    return -(gain + entropy_bonus * entropy) / len(logits)
