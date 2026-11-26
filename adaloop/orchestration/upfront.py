"""Upfront features of the grid items (S13): vision tower + projector and question embeddings, no decoder pass."""
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from adaloop.core.tokens.prune import extract_visual_tokens
from adaloop.core.tokens.selectors import question_embeddings
from adaloop.orchestration.grid import grid_dir
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash


def upfront_dir(cfg: dict, root: str | Path, name: str = "grid") -> Path:
    """<grid folder>/upfront."""
    return grid_dir(cfg, root, name) / "upfront"


@torch.inference_mode()
def upfront_features(vlm: LoadedVLM, item: Item, cfg: dict) -> dict:
    """Mean projected visual token (all tokens, before compression) and mean question input embedding of one item.

    Inputs as in the grid (`item_inputs`: same image resize); the question text is `item.question`, as the
    text-guided selectors read it. The decoder does not run.
    """
    tokens, _ = extract_visual_tokens(vlm, item_inputs(vlm, item, cfg))  # (N, d)
    question = question_embeddings(vlm, item.question)  # (n_q, d)
    return {"visual_mean": tokens.float().mean(dim=0).cpu(), "question_mean": question.float().mean(dim=0).cpu(),
            "n_vis_full": len(tokens), "n_question_tokens": len(question), "config_hash": config_hash(cfg)}


def run_upfront(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, folder: str | Path, every: int = 50) -> int:
    """Write <folder>/<benchmark>_<item_id>.pt for every item; files of this config are kept (resumable).

    Returns the number of files written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    digest = config_hash(cfg)
    written = 0
    for i, item in enumerate(items, 1):
        path = folder / f"{item.benchmark}_{item.item_id}.pt"
        if not path.is_file() or torch.load(path, map_location="cpu", weights_only=True)["config_hash"] != digest:
            part = path.with_suffix(".tmp")
            torch.save(upfront_features(vlm, item, cfg), part)
            part.replace(path)  # an interrupted write leaves no half file
            written += 1
        if i % every == 0 or i == len(items):
            print(f"upfront [{i}/{len(items)}] items | {written} written")
    return written


def load_upfront(grid: pd.DataFrame, folder: str | Path) -> dict | None:
    """features[benchmark, item_id] for every item of `grid`; None (with a printed line) while files are missing."""
    folder = Path(folder)
    items = grid.drop_duplicates(["benchmark", "item_id"])
    paths = {(b, i): folder / f"{b}_{i}.pt" for b, i in zip(items["benchmark"], items["item_id"])}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        print(f"SKIPPED upfront features: {len(missing)} of {len(paths)} files missing in {folder} "
              "(run `python scripts/mvp.py upfront` on the GPU host)")
        return None
    out = {key: torch.load(path, map_location="cpu", weights_only=True) for key, path in paths.items()}
    for key, digest in zip(zip(items["benchmark"], items["item_id"]), items["config_hash"]):
        if out[key]["config_hash"] != digest:
            raise ValueError(f"{paths[key]} was made with config {out[key]['config_hash']}, the grid with {digest}: "
                             "run `python scripts/mvp.py upfront` again")
    return out
