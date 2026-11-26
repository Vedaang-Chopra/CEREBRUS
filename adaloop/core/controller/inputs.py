"""Tree tables -> arrays (`TreeData`), and arrays -> the policy's node feature vectors."""
import numpy as np
import pandas as pd

from adaloop.core.controller.tree import path_name
from adaloop.schemas.runs import TreeData

FEATURES = ("delta_norm", "kl_prev", "entropy", "top1_prob", "vis_attn_mass")
NO_PREV = ("delta_norm", "kl_prev")  # NaN at pass 1: there is no earlier pass
KEYS = ["benchmark", "item_id"]
LEAF_ARRAYS = ("correct", "flops_cell", "flops_discarded", "flops_readout", "flops_state", "kept_frac", "spent_frac")


def tree_data(tree: pd.DataFrame, features: dict, k0: float, look_k: float, split: str) -> TreeData:
    """Arrays of the trees of `split` at `k0`. `features[benchmark, item_id][k, p]` are the saved pass features."""
    rows = tree[(tree["k0"] == k0) & (tree["split"] == split)]
    if rows.empty:
        raise ValueError(f"no tree rows for k0={k0}, split {split!r}")
    R_max = int(rows["exit_pass"].max())
    n_branches = int(rows["look_pass"].max()) + 1
    paths = [path_name(b, q + 1) for b in range(n_branches) for q in range(R_max)]
    wide = rows.set_index(KEYS + ["path"])[list(LEAF_ARRAYS) + ["cost_ref"]].unstack("path").sort_index()
    shape = (len(wide), n_branches, R_max)
    arrays = {name: wide[name][paths].to_numpy(dtype=float).reshape(shape) for name in LEAF_ARRAYS}
    if any(np.isnan(a).any() for a in arrays.values()):
        raise ValueError("some item lacks a tree leaf")
    keys = rows.drop_duplicates(KEYS).set_index(KEYS).loc[wide.index, ["split", "model_variant"]].reset_index()

    ks = [k0] if n_branches == 1 else [k0, look_k]  # the branches' keep fractions
    h_index = np.array([[(b > 0) * R_max + q for q in range(R_max)] for b in range(n_branches)])
    states = [(k, p) for k in ks for p in range(1, R_max + 1)]  # the U distinct pass states, in h_index order
    items = [features[key] for key in wide.index]
    h = np.stack([np.stack([item[s]["h_last"].float().numpy() for s in states]) for item in items])  # (N, U, d)
    raw = {}
    for name in FEATURES:
        table = np.array([[item[s][name] for s in states] for item in items], dtype=float)  # (N, U)
        raw[name] = table[:, h_index]  # (N, B, R)
    cost_ref = wide["cost_ref"][paths[0]].to_numpy(dtype=float)
    return TreeData(keys=keys, k0=k0, h=h, h_index=h_index, raw=raw, cost_ref=cost_ref, **arrays)


def node_scalars(data: TreeData, features: list[str]) -> np.ndarray:
    """(N, B, R, S): the chosen features (NaN kept), no_prev flag, kept_frac, spent_frac, pass one-hot."""
    unknown = set(features) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown controller features {sorted(unknown)}; choose from {FEATURES}")
    shape = data.correct.shape  # (N, B, R)
    columns = [data.raw[name] for name in features]
    first = np.zeros(shape)
    first[:, :, 0] = 1.0  # pass 1 has no previous pass
    if set(features) & set(NO_PREV):
        columns.append(first)
    columns += [data.kept_frac, data.spent_frac]
    onehot = np.broadcast_to(np.eye(shape[2]), shape + (shape[2],))
    return np.concatenate([np.stack(columns, axis=-1), onehot], axis=-1).astype(np.float32)


def fit_normalizer(data: TreeData, features: list[str]) -> dict[str, np.ndarray]:
    """Mean and std of the node scalars and of h_last, from the TRAINING trees only."""
    scalars = node_scalars(data, features).reshape(-1, node_scalars(data, features).shape[-1])
    states = data.h.reshape(-1, data.h.shape[-1])
    norm = {"mean": np.nanmean(scalars, axis=0), "std": np.nanstd(scalars, axis=0),
            "h_mean": states.mean(axis=0), "h_std": states.std(axis=0)}
    for name in ("std", "h_std"):
        norm[name] = np.where(norm[name] > 1e-6, norm[name], 1.0)  # constant columns stay as they are
    return {name: value.astype(np.float32) for name, value in norm.items()}


def policy_inputs(data: TreeData, features: list[str], norm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Standardized (h (N, U, d), scalars (N, B, R, S)); undefined values become 0 (= the training mean)."""
    scalars = (node_scalars(data, features) - norm["mean"]) / norm["std"]
    h = (data.h - norm["h_mean"]) / norm["h_std"]
    return h.astype(np.float32), np.nan_to_num(scalars, nan=0.0).astype(np.float32)
