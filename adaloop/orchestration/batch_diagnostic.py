"""Why batched output differs from batch 1: logit gaps per item, with and without padding."""
from collections.abc import Sequence

import torch

from adaloop.core.model.inference import generate_batch
from adaloop.core.model.loops import set_loops
from adaloop.orchestration.cells import ChunkInputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM


def _margin(logits: torch.Tensor) -> float:
    """Top-1 minus top-2 of one step's logits (V,)."""
    top = logits.float().topk(2).values
    return float(top[0] - top[1])


def _gap(alone: GenerationResult, logits_alone: torch.Tensor, other: GenerationResult,
         logits_other: torch.Tensor) -> dict:
    """Compare one row with its batch-1 run; logits are (n_new, V)."""
    def max_diff(step: int) -> float:
        return float((logits_alone[step].float() - logits_other[step].float().to(logits_alone.device)).abs().max())

    n = min(len(alone.token_ids), len(other.token_ids))
    differs = (alone.token_ids[:n] != other.token_ids[:n]).nonzero()
    # the prefix before the first differing token is the same, so logits at that step are comparable
    flip = int(differs[0]) if len(differs) else None
    return {"first_diff": max_diff(0), "flip_step": flip,
            "flip_margin": None if flip is None else _margin(logits_alone[flip]),
            "flip_diff": None if flip is None else max_diff(flip)}


def diagnose_batch(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, cells: Sequence[tuple[str, float]],
                   R: int, max_new_tokens: int = 32) -> list[dict]:
    """One row per (method, k, item): batch 1 against two batches of size len(items).

    `copies_*`: the item repeated len(items) times (batched, no padding).
    `mixed_*`: all items in one left-padded batch (batched, padding where lengths differ).
    `first_margin` is the batch-1 top-2 logit margin of the first answer token, `*_first_diff` the max |logit diff|
    there, and `*_flip_*` describe the first generated token that differs (None: same tokens).
    """
    set_loops(vlm, R)
    chunk = ChunkInputs(vlm, items, cfg)

    def run(batch: list) -> tuple[list[GenerationResult], list[torch.Tensor]]:
        return generate_batch(vlm, batch, max_new_tokens, return_logits=True)

    rows = []
    for method, k in cells:
        prepared = [chunk.compressed(i, method, k)[0] for i in range(len(items))]
        width = max(x["input_ids"].shape[1] for x in prepared)
        mixed, mixed_logits = run(prepared)
        for i, (item, inputs) in enumerate(zip(items, prepared)):
            alone, alone_logits = run([inputs])
            copies, copies_logits = run([inputs] * len(items))
            row = {"method": method, "k": k, "benchmark": item.benchmark, "item_id": item.item_id,
                   "n_visual": alone[0].n_visual_tokens, "n_pad": width - inputs["input_ids"].shape[1],
                   "first_margin": _margin(alone_logits[0][0])}
            others = {"copies": (copies[0], copies_logits[0]), "mixed": (mixed[i], mixed_logits[i])}
            for name, (gen, logits) in others.items():
                gap = _gap(alone[0], alone_logits[0], gen, logits)
                row.update({f"{name}_{key}": value for key, value in gap.items()})
            rows.append(row)
            print(f"[{len(rows)}/{len(cells) * len(items)}] {method} {item.item_id} | margin {row['first_margin']:.3f} "
                  f"| diff copies {row['copies_first_diff']:.3f} mixed {row['mixed_first_diff']:.3f} "
                  f"| flip copies {row['copies_flip_step']} mixed {row['mixed_flip_step']}")
    return rows
