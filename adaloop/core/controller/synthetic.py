"""A made-up grid shaped like the S7 outputs, for CPU tests and nb11's synthetic mode. Its numbers mean nothing."""
import numpy as np
import pandas as pd
import torch

from adaloop.core.cost.flops import flops_features, flops_total
from adaloop.schemas.cost import CostParams

SYNTHETIC_PARAMS = CostParams(
    embed=0, prelude=600, core=1200, core_message=40, coda=600, lm_head=900, vit_patch=50, vit_layers=300,
    vit_projector=80, core_start=2, core_end=4, attn_dims=(8,) * 6, attn_windows=(None,) * 6, kv_dims=(4,) * 6,
    vit_n_layers=3, vit_hidden=16, vit_merge=2)
KS, RS, HIDDEN, HEADS = (0.25, 0.5, 1.0), (1, 2, 3, 4, 5), 24, 4


def synthetic_grid(n_items: int = 6, splits: tuple[str, ...] = ("ctrl_train", "dev"), d: int = HIDDEN,
                   seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """(grid rows, features[benchmark, item_id][k, p]), FLOPs from SYNTHETIC_PARAMS. `n_items` per benchmark and split."""
    rng = np.random.default_rng(seed)
    params = SYNTHETIC_PARAMS
    rows, features = [], {}
    for split in splits:
        for benchmark in ("gqa", "textvqa"):
            for i in range(n_items):
                item_id = f"{split}{i}"
                n_vis_full, n_txt, hard = int(rng.integers(40, 120)), int(rng.integers(10, 30)), rng.random()
                feats = features[benchmark, item_id] = {}
                for k in KS:
                    n_vis = round(k * n_vis_full)
                    n = n_vis + n_txt
                    extra = flops_features(params, n, n_vis, HEADS, HIDDEN)
                    for R in RS:
                        skill = (k * 0.6 + R * 0.08) - hard * 0.7  # more tokens and loops help hard items more
                        correct = float(skill + rng.normal(0, 0.15) > 0)
                        flops = flops_total(params, n, 3, R, [(1, 2, 2)])  # tiny image: the decoder dominates, so caps bind
                        compression = 0 if k == 1.0 else 5000
                        rows.append({"item_id": item_id, "benchmark": benchmark, "split": split,
                                     "model_variant": "base", "rerun_id": 0, "k": k, "R": R, "correct": correct,
                                     "n_vis": n_vis, "n_vis_full": n_vis_full, "n_txt": n_txt,
                                     "flops_vit": flops["flops_vit"], "flops_prefill": flops["flops_prefill"],
                                     "flops_compression": compression,
                                     "flops_total": flops["flops_total"] + compression, "config_hash": "synthetic"})
                        h = torch.from_numpy(rng.normal(0, 1, d)).half()
                        h[:2] = torch.tensor([skill, hard]).half()  # something to learn
                        feats[k, R] = {"h_last": h, "delta_norm": float("nan") if R == 1 else rng.random(),
                                       "kl_prev": float("nan") if R == 1 else rng.random() / R,
                                       "entropy": float(max(0.0, 1 - skill)), "top1_prob": float(rng.random()),
                                       "vis_attn_mass": float(rng.random()), **extra}
    return pd.DataFrame(rows), features


def synthetic_upfront(features: dict, d: int = HIDDEN, seed: int = 1) -> dict:
    """Upfront features per item as `upfront_features` saves them; the item's difficulty is readable (noisily)."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, feats in features.items():
        visual, question = (torch.from_numpy(rng.normal(0, 1, d)).float() for _ in range(2))
        visual[0] = 3.0 * float(feats[1.0, 1]["h_last"][1]) + 0.3 * float(rng.normal())
        out[key] = {"visual_mean": visual, "question_mean": question, "n_vis_full": 80, "n_question_tokens": 8,
                    "config_hash": "synthetic"}
    return out
