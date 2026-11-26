"""S12: train the controller and tune the heuristic per k0 (train on ctrl_train, select on dev), write policy_eval."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaloop.core.controller.heuristic import HEURISTIC_FEATURES, heuristic_actions, threshold_grid
from adaloop.core.controller.inputs import fit_normalizer, policy_inputs, tree_data
from adaloop.core.controller.objectives import expected_reward_loss, leaf_rewards, supervised_loss
from adaloop.core.controller.policy import Policy
from adaloop.core.controller.rows import leaf_costs, policy_rows, select_on_dev
from adaloop.core.controller.tree import action_mask, charged, walk
from adaloop.orchestration.tree import check_test_allowed, controller_dir, run_tree
from adaloop.schemas.cost import CostParams
from adaloop.schemas.runs import TreeData

OBJECTIVES = ("expected_reward", "supervised")


def _tensors(data: TreeData, norm: dict, ctrl: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h, node scalars, h_index) of the trees, standardized with the training statistics."""
    h, scalars = policy_inputs(data, ctrl["features"], norm)
    return torch.from_numpy(h), torch.from_numpy(scalars), torch.from_numpy(data.h_index)


def _pays(ctrl: dict) -> tuple[bool, bool]:
    readout, state = charged(ctrl["features"])
    return readout and ctrl["charge_features"], state and ctrl["charge_features"]


def _rewards(data: TreeData, lam: float, pays: tuple[bool, bool]) -> torch.Tensor:
    cost = torch.from_numpy(leaf_costs(data, *pays) / data.cost_ref[:, None, None]).float()
    correct = torch.tensor(data.correct, dtype=torch.float32)  # a copy: pandas arrays are read-only
    return leaf_rewards(correct, cost, torch.ones(len(data.cost_ref)), lam)  # cost already divided by cost_ref


def greedy_exits(policy: Policy, data: TreeData, norm: dict, ctrl: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exit nodes when the policy takes its most likely allowed action at every node."""
    policy.eval()
    mask = torch.from_numpy(action_mask(*data.correct.shape[1:]))
    with torch.no_grad():
        logits = policy(*_tensors(data, norm, ctrl))
    return walk(logits.masked_fill(~mask, float("-inf")).argmax(dim=-1).numpy())


def train_policy(train: TreeData, dev: TreeData, lam: float, seed: int, ctrl: dict) -> tuple[Policy, dict, dict]:
    """One controller for one (k0, lam, seed): (policy at its best dev epoch, normalizer, info).

    Mini-batches of whole trees; the epoch is chosen by the mean reward of the greedy policy on dev.
    """
    if ctrl["objective"] not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {ctrl['objective']!r}")
    torch.manual_seed(seed)  # weights and dropout
    order = torch.Generator().manual_seed(seed)  # batches
    norm = fit_normalizer(train, ctrl["features"])
    h, scalars, h_index = _tensors(train, norm, ctrl)
    mask = torch.from_numpy(action_mask(*train.correct.shape[1:]))
    rewards = _rewards(train, lam, _pays(ctrl))  # (N, B, R)
    dev_rewards = _rewards(dev, lam, _pays(ctrl)).numpy()
    policy = Policy(h.shape[-1], scalars.shape[-1], ctrl)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=ctrl["lr"], weight_decay=ctrl["weight_decay"])
    best = {"reward": -np.inf, "epoch": 0, "state": None}
    for epoch in range(1, ctrl["epochs"] + 1):
        policy.train()
        for batch in torch.randperm(len(h), generator=order).split(ctrl["batch_trees"]):
            logits = policy(h[batch], scalars[batch], h_index)
            if ctrl["objective"] == "supervised":
                loss = supervised_loss(logits, rewards[batch], mask)
            else:
                loss = expected_reward_loss(logits, rewards[batch], mask, ctrl["entropy_bonus"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        b, q = greedy_exits(policy, dev, norm, ctrl)
        reward = float(dev_rewards[np.arange(len(b)), b, q].mean())
        if reward > best["reward"] + 1e-9:
            best = {"reward": reward, "epoch": epoch,
                    "state": {name: value.clone() for name, value in policy.state_dict().items()}}
        elif epoch - best["epoch"] >= ctrl["patience"]:
            break
    policy.load_state_dict(best["state"])
    policy.eval()
    return policy, norm, {"best_epoch": best["epoch"], "dev_reward": best["reward"], "epochs_run": epoch}


def _splits(tree: pd.DataFrame, features: dict, ctrl: dict, k0: float) -> dict[str, TreeData]:
    """TreeData of the train, dev and evaluation splits at k0."""
    names = [ctrl["splits"]["train"], ctrl["splits"]["dev"], *ctrl["splits"]["eval"]]
    check_test_allowed(names)
    return {name: tree_data(tree, features, k0, ctrl["look"]["k"], name) for name in dict.fromkeys(names)}


def run_controller(tree: pd.DataFrame, features: dict, ctrl: dict, weights_dir: str | Path | None = None,
                   method: str = "controller") -> pd.DataFrame:
    """policy_eval rows of the learned controller: every k0 x lambda x seed on the `eval` splits."""
    # the gradient of h[:, h_index] accumulates in parallel on CPU unless deterministic algorithms are on
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    rows = []
    for k0 in dict.fromkeys(tree["k0"]):
        data = _splits(tree, features, ctrl, k0)
        train, dev = data[ctrl["splits"]["train"]], data[ctrl["splits"]["dev"]]
        for lam in ctrl["lambdas"]:
            for seed in ctrl["seeds"]:
                start = time.perf_counter()
                policy, norm, info = train_policy(train, dev, lam, seed, ctrl)
                print(f"{method} k0={k0} lam={lam} seed={seed}: epoch {info['best_epoch']}/{info['epochs_run']} | "
                      f"dev reward {info['dev_reward']:.4f} | {time.perf_counter() - start:.1f}s")
                if weights_dir is not None:
                    Path(weights_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({"state": policy.state_dict(), "norm": {k: torch.from_numpy(v) for k, v in norm.items()},
                                "ctrl": ctrl, **info}, Path(weights_dir) / f"{method}_k{k0}_lam{lam}_seed{seed}.pt")
                for split in ctrl["splits"]["eval"]:
                    exits = greedy_exits(policy, data[split], norm, ctrl)
                    rows.append(policy_rows(data[split], exits, method, lam, seed, *_pays(ctrl)))
    torch.use_deterministic_algorithms(previous)
    return pd.concat(rows, ignore_index=True)


def run_heuristic(tree: pd.DataFrame, features: dict, ctrl: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(policy_eval rows, chosen thresholds) of the heuristic: per k0 and lambda, the thresholds with the best
    mean reward on dev. It reads readout features and attention mass, so it pays for both."""
    readout, state = charged(HEURISTIC_FEATURES)
    pays = (readout and ctrl["charge_features"], state and ctrl["charge_features"])
    rows, chosen = [], []
    for k0 in dict.fromkeys(tree["k0"]):
        data = _splits(tree, features, ctrl, k0)
        dev = data[ctrl["splits"]["dev"]]
        grid = threshold_grid(dev, ctrl["heuristic"])
        item = np.arange(len(dev.keys))
        exits = [walk(heuristic_actions(dev, **thresholds)) for thresholds in grid]
        for lam in ctrl["lambdas"]:
            reward = _rewards(dev, lam, pays).numpy()
            means = [reward[item, b, q].mean() for b, q in exits]
            best = grid[int(np.argmax(means))]
            chosen.append({"k0": k0, "lam": lam, **best, "dev_reward": float(max(means))})
            for split in ctrl["splits"]["eval"]:
                picks = walk(heuristic_actions(data[split], **best))
                rows.append(policy_rows(data[split], picks, "heuristic", lam, 0, *pays))
        print(f"heuristic k0={k0}: {len(grid)} threshold sets tried on dev")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(chosen)


def run_policies(cfg: dict, ctrl: dict, root: str | Path, params: CostParams | None = None) -> pd.DataFrame:
    """S11 + S12: tree.parquet, controller and heuristic, policy_eval.parquet, dev selection per cost cap."""
    if ctrl["splits"]["dev"] not in ctrl["splits"]["eval"]:
        raise ValueError("`splits.eval` must hold the dev split: (k0, lambda) per cost cap is chosen from its rows")
    check_test_allowed(ctrl["splits"]["eval"])
    tree, features = run_tree(cfg, ctrl, root, params)
    out = controller_dir(cfg, ctrl, root)
    learned = run_controller(tree, features, ctrl, weights_dir=out / "weights")
    ruled, thresholds = run_heuristic(tree, features, ctrl)
    rows = pd.concat([learned, ruled], ignore_index=True)
    rows.to_parquet(out / "policy_eval.parquet", index=False)
    thresholds.to_csv(out / "heuristic_thresholds.csv", index=False)
    select_on_dev(rows, ctrl["cap_fracs"], ctrl["splits"]["dev"]).to_csv(out / "select_dev.csv", index=False)
    print(f"{len(rows)} policy rows -> {out / 'policy_eval.parquet'}")
    return rows
