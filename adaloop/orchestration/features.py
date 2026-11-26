"""One item over the (k, R) cells: result rows and the per-pass controller features."""
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn import functional as F

from adaloop.core.controller.features import pass_features
from adaloop.core.cost.flops import cost_params, flops_features
from adaloop.core.model.hooks import record_attention_mass, record_first_logits, record_pass_states
from adaloop.orchestration.batching import EQUAL_LENGTH, equal_key_chunks
from adaloop.orchestration.cells import ChunkInputs, generate_cell
from adaloop.orchestration.evaluate import result_row
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM


def run_items_cells(vlm: LoadedVLM, items: Sequence[Item], cfg: dict) -> list[tuple[list[dict], dict]]:
    """Every k in `cfg["k_grid"]` x R in `cfg["R_grid"]` for each item; per item (rows, {(k, p): features}).

    The items form one generation batch per cell; the vision tower runs once per item and one compression per k is
    shared by its R runs. Each R is generated on its own (the answer needs the coda after the last pass). `kl_prev`
    needs the R = p-1 run, so it is NaN when p-1 is not in `R_grid`. Needs `answer_prefill` (no thinking): the first
    generated token is then the first answer token. Features and `generate_s` carry the hooks' small overhead; batched
    values differ from batch 1 in the last bits (floating point), take latency from batch-1 runs without features.
    Under `batch_grouping: equal_length`, items whose compressed prompts differ in length run as separate batches.
    """
    if cfg.get("answer_prefill") is None:
        raise ValueError("features read the first answer token: set answer_prefill (no thinking)")
    params = cost_params(vlm.model)
    text = vlm.model.config.text_config
    image_id = vlm.model.config.image_token_id
    chunk = ChunkInputs(vlm, items, cfg)
    n_items = len(items)
    compressed = {k: [chunk.compressed(i, cfg["compressor"], k) for i in range(n_items)] for k in cfg["k_grid"]}
    if cfg.get("batch_grouping") == EQUAL_LENGTH:
        # the hooks need one generate call per cell, and `generate_answers` splits a batch of unequal prompt lengths
        lengths = [tuple(compressed[k][i][0]["input_ids"].shape[1] for k in cfg["k_grid"]) for i in range(n_items)]
        groups = equal_key_chunks(lengths, n_items)
        if len(groups) > 1:
            results = {i: result for group in groups
                       for i, result in zip(group, run_items_cells(vlm, [items[i] for i in group], cfg))}
            return [results[i] for i in range(n_items)]
    batched = n_items > 1
    out = [([], {}) for _ in items]
    for k in cfg["k_grid"]:
        prepared = compressed[k]
        masks = [inputs["input_ids"][0] == image_id for inputs, _ in prepared]  # (T_i,)
        width = max(len(mask) for mask in masks)
        padded = torch.stack([F.pad(mask, (width - len(mask), 0), value=False) for mask in masks])  # (B, T), left
        # the last core layer is a full-attention layer with num_attention_heads heads
        costs = [flops_features(params, len(mask), int(mask.sum()), text.num_attention_heads, text.hidden_size)
                 for mask in masks]
        previous: tuple[int, list] | None = None  # (R, first logits per item) of the last run
        for R in sorted(cfg["R_grid"]):
            with record_pass_states(vlm, batch=True) as states, record_attention_mass(vlm, padded) as masses, \
                    record_first_logits(vlm, batch=True) as logits:
                gens = generate_cell(vlm, [inputs for inputs, _ in prepared], cfg, R)
            if len(states) != R or len(masses) != R:
                raise RuntimeError(f"expected {R} pass states, got {len(states)} states and {len(masses)} masses")
            for i, (item, gen) in enumerate(zip(items, gens)):
                prev_logits = previous[1][i] if previous and previous[0] == R - 1 else None
                out[i][1][k, R] = {
                    **pass_features([state[i] for state in states], masses[-1][i].item(), logits[0][i], prev_logits),
                    **costs[i], "n_tokens": len(masks[i]), "n_vis": int(masks[i].sum())}
                row = result_row(item, cfg, R, gen, batch_size=n_items, info=prepared[i][1])
                out[i][0].append({**row, "features": True})
            previous = (R, list(logits[0]))
    return out


def run_item_cells(vlm: LoadedVLM, item: Item, cfg: dict) -> tuple[list[dict], dict]:
    """`run_items_cells` for one item (batch 1): (rows, {(k, p): features})."""
    return run_items_cells(vlm, [item], cfg)[0]


def save_features(features: dict, folder: str | Path, item_id: str) -> Path:
    """Write one item's {(k, p): features} to <folder>/<item_id>.pt."""
    path = Path(folder) / f"{item_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, path)
    return path


def load_features(path: str | Path) -> dict:
    """Read a file written by `save_features`."""
    return torch.load(path, map_location="cpu", weights_only=True)
