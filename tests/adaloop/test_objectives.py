"""S12 objectives against a tree worked out by hand. CPU only.

Run: `python tests/adaloop/test_objectives.py` (or pytest) from the repo root.

Tree with R_max = 2 (branches: 0 = low tokens, 1 = full-token restart after a look at pass 1), one item:

    node (0,0) pass 1 low   : EXIT r=0.2 | LOOP -> (0,1) | LOOK -> (1,0)
    node (0,1) pass 2 low   : EXIT r=0.4
    node (1,0) pass 1 full  : EXIT r=0.6 | LOOP -> (1,1)
    node (1,1) pass 2 full  : EXIT r=1.0

Optimal (max): V(0,1)=0.4, V(1,1)=1.0, V(1,0)=max(0.6, 1.0)=1.0, V(0,0)=max(0.2, 0.4, 1.0)=1.0
    targets: (0,0) LOOK, (0,1) EXIT, (1,0) LOOP, (1,1) EXIT
Uniform policy over allowed actions:
    V(1,0) = (0.6 + 1.0)/2 = 0.8,  V(0,0) = (0.2 + 0.4 + 0.8)/3 = 7/15
    J = sum over nodes of sum_a pi Q = V(0,0) + V(0,1) + V(1,0) + V(1,1) = 7/15 + 0.4 + 0.8 + 1.0
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaloop.core.controller.objectives import (
    backward_values,
    expected_reward_loss,
    leaf_rewards,
    masked_log_probs,
    reach_probabilities,
    supervised_loss,
    supervised_targets,
)
from adaloop.core.controller.tree import EXIT, LOOK, LOOP, action_mask

MASK = torch.from_numpy(action_mask(2, 2))
R_EXIT = torch.tensor([[[0.2, 0.4], [0.6, 1.0]]])  # (1, 2, 2)


def test_reward() -> None:
    r = leaf_rewards(torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[50.0, 200.0]]]), torch.tensor([100.0]), 0.2)
    assert torch.allclose(r, torch.tensor([[[0.9, -0.4]]]))
    print("✓ PASS objectives: reward = correct - lam * cost / cost_ref")


def test_optimal_values_and_targets() -> None:
    q, v = backward_values(R_EXIT, MASK)
    assert torch.allclose(v, torch.tensor([[[1.0, 0.4], [1.0, 1.0]]]))
    assert torch.allclose(q[0, 0, 0], torch.tensor([0.2, 0.4, 1.0]))
    assert q[0, 0, 1, LOOP] == float("-inf") and q[0, 1, 0, LOOK] == float("-inf")
    assert supervised_targets(R_EXIT, MASK).tolist() == [[[LOOK, EXIT], [LOOP, EXIT]]]
    assert supervised_targets(torch.ones(1, 2, 2), MASK).tolist() == [[[EXIT, EXIT], [EXIT, EXIT]]]  # ties: EXIT
    print("✓ PASS objectives: optimal values, argmax-Q targets, ties to EXIT")


def test_expected_values_and_losses() -> None:
    logits = torch.zeros(1, 2, 2, 3, requires_grad=True)  # uniform over the allowed actions
    probs = masked_log_probs(logits, MASK).exp()
    assert torch.allclose(probs[0, 0, 0], torch.full((3,), 1 / 3)) and probs[0, 0, 1].tolist() == [1.0, 0.0, 0.0]
    q, v = backward_values(R_EXIT, MASK, probs.detach())
    assert torch.allclose(v, torch.tensor([[[7 / 15, 0.4], [0.8, 1.0]]]))
    J = 7 / 15 + 0.4 + 0.8 + 1.0
    entropy = math.log(3) + math.log(2)  # only (0,0) and (1,0) have a choice
    assert abs(expected_reward_loss(logits, R_EXIT, MASK, 0.0).item() + J) < 1e-6
    assert abs(expected_reward_loss(logits, R_EXIT, MASK, 0.1).item() + J + 0.1 * entropy) < 1e-6
    ### Step-1:- Gradient of -J at (0,0): -pi(a) (Q(a) - V) with Q held fixed
    expected_reward_loss(logits, R_EXIT, MASK, 0.0).backward()
    want = -(1 / 3) * (torch.tensor([0.2, 0.4, 0.8]) - 7 / 15)
    assert torch.allclose(logits.grad[0, 0, 0], want, atol=1e-6)
    want_full = -0.5 * (torch.tensor([0.6, 1.0]) - 0.8)
    assert torch.allclose(logits.grad[0, 1, 0, :2], want_full, atol=1e-6) and logits.grad[0, 1, 0, LOOK] == 0
    assert torch.isfinite(logits.grad).all() and (logits.grad[0, 0, 1] == 0).all()
    assert abs(supervised_loss(logits.detach(), R_EXIT, MASK).item() - (math.log(3) + math.log(2))) < 1e-6
    print("✓ PASS objectives: policy values, expected-reward loss and gradient, supervised loss")


def test_reach_gradient_is_the_root_value_gradient() -> None:
    mask = torch.from_numpy(action_mask(3, 3))  # looks after pass 1 or 2
    generator = torch.Generator().manual_seed(0)
    r_exit = torch.rand(2, 3, 3, dtype=torch.float64, generator=generator)
    logits = torch.randn(2, 3, 3, 3, dtype=torch.float64, generator=generator).requires_grad_()

    def root_value(z: torch.Tensor) -> float:
        return backward_values(r_exit, mask, masked_log_probs(z, mask).exp())[1][:, 0, 0].mean().item()

    numeric = torch.zeros_like(logits)
    with torch.no_grad():
        for i in range(logits.numel()):
            step = torch.zeros(logits.numel(), dtype=torch.float64)
            step[i] = 1e-6
            step = step.view_as(logits)
            numeric.view(-1)[i] = (root_value(logits + step) - root_value(logits - step)) / 2e-6
    gradient = {}
    for weight in ("reach", "uniform"):
        gradient[weight], = torch.autograd.grad(expected_reward_loss(logits, r_exit, mask, 0.0, weight), logits)
    assert torch.allclose(gradient["reach"], -numeric, atol=1e-8)
    assert not torch.allclose(gradient["uniform"], -numeric, atol=1e-3)
    reach = reach_probabilities(masked_log_probs(logits, mask).exp(), mask)
    assert (reach[:, 0, 0] == 1).all() and (reach <= 1).all()
    print("✓ PASS objectives: reach weighting gives the exact gradient of the root value (finite differences)")


def test_gradient_descent_finds_the_best_path() -> None:
    for loss in (lambda z: expected_reward_loss(z, R_EXIT, MASK, 0.0), lambda z: supervised_loss(z, R_EXIT, MASK)):
        logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.2)
        for _ in range(100):
            optimizer.zero_grad()
            loss(logits).backward()
            optimizer.step()
        chosen = logits.detach().masked_fill(~MASK, float("-inf")).argmax(-1)
        assert chosen[0, 0, 0] == LOOK and chosen[0, 1, 0] == LOOP
    print("✓ PASS objectives: both objectives learn LOOK then LOOP on the tiny tree")


if __name__ == "__main__":
    test_reward()
    test_optimal_values_and_targets()
    test_expected_values_and_losses()
    test_reach_gradient_is_the_root_value_gradient()
    test_gradient_descent_finds_the_best_path()
