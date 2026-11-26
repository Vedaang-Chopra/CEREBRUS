"""One item over the (k, R) cells: result rows and the per-pass controller features."""
from pathlib import Path

import torch

from adaloop.core.controller.features import pass_features
from adaloop.core.cost.flops import cost_params, flops_features
from adaloop.core.model.budget import generate_answers
from adaloop.core.model.hooks import record_attention_mass, record_first_logits, record_pass_states
from adaloop.core.model.loops import set_loops
from adaloop.orchestration.compress import compress
from adaloop.orchestration.evaluate import result_row
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM


def run_item_cells(vlm: LoadedVLM, item: Item, cfg: dict) -> tuple[list[dict], dict]:
    """Run every k in `cfg["k_grid"]` x R in `cfg["R_grid"]`; return (rows, {(k, p): features of the R = p run}).

    Each R is generated on its own (the answer needs the coda after the last pass). One compression per k is shared
    by its R runs. `kl_prev` needs the R = p-1 run, so it is NaN when p-1 is not in `R_grid`. Needs
    `answer_prefill` (no thinking): the first generated token is then the first answer token.
    Rows of these runs carry the hooks' small overhead in `generate_s`; take latency from runs without features.
    """
    if cfg.get("answer_prefill") is None:
        raise ValueError("features read the first answer token: set answer_prefill (no thinking)")
    params = cost_params(vlm.model)
    text = vlm.model.config.text_config
    image_id = vlm.model.config.image_token_id
    full_inputs = item_inputs(vlm, item, cfg)
    rows, features = [], {}
    for k in cfg["k_grid"]:
        inputs, info = compress(vlm, full_inputs, item, k, cfg)
        visual_mask = inputs["input_ids"][0] == image_id  # (T,)
        n_tokens, n_vis = len(visual_mask), int(visual_mask.sum())
        cost = flops_features(params, n_tokens, n_vis, text.num_attention_heads, text.hidden_size)
        previous: tuple[int, torch.Tensor] | None = None  # (R, first logits) of the last run
        for R in sorted(cfg["R_grid"]):
            set_loops(vlm, R)
            with record_pass_states(vlm) as states, record_attention_mass(vlm, visual_mask) as masses, \
                    record_first_logits(vlm) as logits:
                gen = generate_answers(vlm, [inputs], cfg)[0]
            if len(states) != R or len(masses) != R:
                raise RuntimeError(f"expected {R} pass states, got {len(states)} states and {len(masses)} masses")
            prev_logits = previous[1] if previous and previous[0] == R - 1 else None
            features[k, R] = {**pass_features(states, masses[-1], logits[0], prev_logits), **cost,
                              "n_tokens": n_tokens, "n_vis": n_vis}
            previous = (R, logits[0])
            rows.append({**result_row(item, cfg, R, gen, batch_size=1, info=info), "features": True})
    return rows, features


def save_features(features: dict, folder: str | Path, item_id: str) -> Path:
    """Write one item's {(k, p): features} to <folder>/<item_id>.pt."""
    path = Path(folder) / f"{item_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, path)
    return path


def load_features(path: str | Path) -> dict:
    """Read a file written by `save_features`."""
    return torch.load(path, map_location="cpu", weights_only=True)
