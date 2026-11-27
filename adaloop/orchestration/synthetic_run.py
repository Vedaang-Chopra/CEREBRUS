"""Write a synthetic grid in the layout of the real one, so nb11 can run end to end before the grid exists."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaloop.core.controller.synthetic import synthetic_grid, synthetic_upfront
from adaloop.orchestration.features import save_features
from adaloop.orchestration.grid import grid_dir
from adaloop.orchestration.item_images import FILE as IMAGES_FILE


def write_synthetic_grid(cfg: dict, ctrl: dict, root: str | Path) -> dict:
    """grid.parquet (with re-run rows), pass features, upfront features and Step-2b rows under the synthetic out_dir.

    Returns `cfg` with `out_dir` set to `ctrl["synthetic"]["out_dir"]`: use it for every later step.
    """
    settings = ctrl["synthetic"]
    out = Path(settings["out_dir"])
    real = Path(cfg["out_dir"])
    if out == real or real in out.parents or out in real.parents:
        raise ValueError(f"synthetic out_dir {out} must be outside the real one {real}")
    cfg = {**cfg, "out_dir": str(out)}
    grid, features = synthetic_grid(n_items=settings["n_items"])
    folder = grid_dir(cfg, root, ctrl["grid_name"])
    folder.mkdir(parents=True, exist_ok=True)
    images = {}  # two items per image
    for benchmark, item_id in features:
        split = item_id.rstrip("0123456789")
        images.setdefault(benchmark, {})[item_id] = f"{split}_image{int(item_id[len(split):]) // 2}"
    (folder / IMAGES_FILE).write_text(json.dumps(images))
    rng = np.random.default_rng(2)
    again = grid.assign(rerun_id=1, correct=np.where(rng.random(len(grid)) < 0.03, 1.0 - grid["correct"],
                                                     grid["correct"]))
    full = pd.concat([grid, again[again["item_id"].str[-1].isin(["0", "1", "2"])]], ignore_index=True)
    full.to_parquet(folder / "grid.parquet", index=False)
    for (benchmark, item_id), feats in features.items():
        save_features(feats, folder / "features" / ctrl["variant"] / "synthetic", f"{benchmark}_{item_id}")
    for (benchmark, item_id), record in synthetic_upfront(features).items():
        save_features(record, folder / "upfront", f"{benchmark}_{item_id}")
    size = cfg["grid_batch"]["size"]
    check = grid[(grid["split"] == "dev") & (grid["R"] == 3) & grid["k"].isin([0.25, 1.0])]
    flipped = np.where(rng.random(len(check)) < 0.05, 1.0 - check["correct"], check["correct"])
    for n, rows in ((1, check), (size, check.assign(correct=flipped))):
        target = folder / f"b1_vs_b{size}" / f"b{n}"
        target.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(target / "grid.parquet", index=False)
    print(f"SYNTHETIC grid ({len(features)} items, numbers mean nothing) -> {folder}")
    return cfg
