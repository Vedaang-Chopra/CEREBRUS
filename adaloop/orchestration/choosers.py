"""S13 upfront baselines: train a chooser over fixed (k, R) cells on ctrl_train, pick its epoch on ctrl_val."""
import numpy as np
import pandas as pd
import torch

from adaloop.core.controller.chooser import Chooser, chooser_loss
from adaloop.orchestration.cell_policies import cell_arrays, cell_rows
from adaloop.orchestration.tree import check_test_allowed


def _upfront_arrays(keys: pd.DataFrame, upfront: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(visual (N, d), question (N, d), pooling FLOPs (N,)) in the order of `keys`.

    The FLOPs are the additions of the two means. The vision tower is in every cell's cost already, the embedding is a
    table lookup, and the chooser MLP is not counted (nor is the controller's).
    """
    items = [upfront[key] for key in zip(keys["benchmark"], keys["item_id"])]
    visual = np.stack([item["visual_mean"].float().numpy() for item in items])
    question = np.stack([item["question_mean"].float().numpy() for item in items])
    flops = np.array([(item["n_vis_full"] + item["n_question_tokens"]) * visual.shape[1] for item in items])
    return visual, question, flops


def train_chooser(train: tuple, val: tuple, seed: int, ctrl: dict) -> tuple[Chooser, np.ndarray]:
    """One chooser; `train` and `val` are (visual, question, rewards (N, C)) tensors.

    Returns (model at the epoch with the best mean greedy reward on `val`, that epoch).
    """
    torch.manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    visual, question, rewards = train
    model = Chooser(visual.shape[1], rewards.shape[1], ctrl)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ctrl["lr"], weight_decay=ctrl["weight_decay"])
    best = {"reward": -np.inf, "epoch": 0, "state": None}
    for epoch in range(1, ctrl["epochs"] + 1):
        model.train()
        for batch in torch.randperm(len(visual), generator=order).split(ctrl["batch_trees"]):
            loss = chooser_loss(model(visual[batch], question[batch]), rewards[batch], ctrl["entropy_bonus"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            picks = model(val[0], val[1]).argmax(dim=-1)
        reward = float(val[2][torch.arange(len(picks)), picks].mean())
        if reward > best["reward"] + 1e-9:
            best = {"reward": reward, "epoch": epoch,
                    "state": {name: value.clone() for name, value in model.state_dict().items()}}
        elif epoch - best["epoch"] >= ctrl["patience"]:
            break
    model.load_state_dict(best["state"])
    model.eval()
    return model, best["epoch"]


def chooser_rows(grid: pd.DataFrame, upfront: dict, cells: list[tuple], ctrl: dict, method: str) -> pd.DataFrame:
    """policy_eval rows of an upfront chooser over `cells` [(k, R)]: every lambda x seed on the `eval` splits.

    A `fit_split` column of the grid (see `load_controller_inputs`) replaces `split`, so train and val do not overlap.
    """
    splits = ctrl["splits"]
    if "fit_split" in grid:
        grid = grid.assign(split=grid["fit_split"])
    names = list(dict.fromkeys([splits["train"], splits["val"], *splits["eval"]]))
    check_test_allowed(names)
    data = {}
    for split in names:
        keys, values, costs, cost_ref = cell_arrays(grid, split, cells)
        visual, question, pooling = _upfront_arrays(keys, upfront)
        data[split] = (keys, values, costs, cost_ref, visual, question, pooling * ctrl["charge_features"])
    ### Step-1:- Standardize with the training items only
    stats = [(x.mean(axis=0), np.where(x.std(axis=0) > 1e-6, x.std(axis=0), 1.0)) for x in data[splits["train"]][4:6]]

    def inputs(split: str) -> list[torch.Tensor]:
        return [torch.tensor((x - mean) / std, dtype=torch.float32)
                for x, (mean, std) in zip(data[split][4:6], stats)]

    def rewards(split: str, lam: float) -> torch.Tensor:
        _, values, costs, cost_ref, _, _, pooling = data[split]
        return torch.tensor(values - lam * (costs + pooling[:, None]) / cost_ref[:, None], dtype=torch.float32)

    ### Step-2:- One chooser per lambda and seed
    rows = []
    for lam in ctrl["lambdas"]:
        for seed in ctrl["seeds"]:
            model, epoch = train_chooser((*inputs(splits["train"]), rewards(splits["train"], lam)),
                                         (*inputs(splits["val"]), rewards(splits["val"], lam)), seed, ctrl)
            print(f"{method} lam={lam} seed={seed}: best val epoch {epoch}")
            for split in splits["eval"]:
                keys, values, costs, cost_ref, _, _, pooling = data[split]
                with torch.no_grad():
                    picks = model(*inputs(split)).argmax(dim=-1).numpy()
                rows.append(cell_rows(keys, cells, picks, values, costs, cost_ref, method, k0=0.0, lam=lam, seed=seed,
                                      state=pooling))
    return pd.concat(rows, ignore_index=True)
