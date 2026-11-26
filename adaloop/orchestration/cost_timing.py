"""Wall-clock measurements that go with the analytic cost model (stage S6)."""
import statistics
import time
from collections.abc import Mapping, Sequence

import torch

from adaloop.core.cost.flops import cost_params, row_flops
from adaloop.core.model.budget import generate_answers
from adaloop.core.model.inference import generate
from adaloop.core.model.loops import set_loops
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
from adaloop.orchestration.compress import compress
from adaloop.orchestration.features import run_items_cells
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM


def time_prefill(vlm: LoadedVLM, inputs: Mapping, R: int, repeats: int) -> float:
    """Median seconds of the prompt forward at R passes, without the vision tower; one warmup call first."""
    # all visual tokens kept: bit-exact with the full inputs, and the vision features are computed outside the timer
    visual, vision_outputs = extract_visual_tokens(vlm, inputs)
    full = prune_inputs(vlm, inputs, vision_outputs, torch.arange(len(visual)))
    set_loops(vlm, R)
    generate(vlm, full, max_new_tokens=1)  # warmup
    return statistics.median(generate(vlm, full, max_new_tokens=1).ttft_s for _ in range(repeats))


def time_throughput(vlm: LoadedVLM, inputs_list: Sequence[Mapping], R: int, batch_size: int, cfg: Mapping) -> dict:
    """Generate all inputs at `batch_size` as a run of `cfg` does and return wall-clock seconds per item."""
    set_loops(vlm, R)

    def run(chunk: Sequence[Mapping]) -> list:
        return generate_answers(vlm, chunk, cfg, batched=batch_size > 1)

    run(inputs_list[:batch_size])  # warmup
    start = time.perf_counter()
    results = []
    for first in range(0, len(inputs_list), batch_size):
        results += run(inputs_list[first : first + batch_size])
    total_s = time.perf_counter() - start  # both generate paths synchronize the GPUs before returning
    return {
        "R": R,
        "batch_size": batch_size,
        "n_items": len(inputs_list),
        "total_s": total_s,
        "s_per_item": total_s / len(inputs_list),
        "mean_out_tokens": statistics.mean(len(r.token_ids) for r in results),
    }


def cost_grid(vlm: LoadedVLM, item: Item, cfg: dict, n_out: int) -> list[dict]:
    """Analytic FLOPs of one item at every k x R of the config under `cfg["compressor"]`; `n_out` held fixed.

    The item is compressed for real at each k (the packed length and, for `lowres`, the image grid depend on it);
    nothing is generated.
    """
    params = cost_params(vlm.model)
    full_inputs = item_inputs(vlm, item, cfg)
    rows = []
    for k in cfg["k_grid"]:
        inputs, info = compress(vlm, full_inputs, item, k, cfg)
        for R in cfg["R_grid"]:
            row = {"k": k, "R": R, "n_vis": info["n_vis"], "n_input": inputs["input_ids"].shape[1],
                   "n_out_tokens": n_out, "grid_hw": info["grid_hw"], "compression_flops": info["compression_flops"]}
            rows.append({**row, **row_flops(params, row)})
    return rows


def time_grid(vlm: LoadedVLM, items: Sequence[Item], cfg: dict) -> tuple[list[dict], float]:
    """Run every cell on `items` as the grid does (features on, `eval_batch_size`); return (rows, seconds per item).

    One warmup batch is run first and not counted. The seconds cover all k x R cells of one item on one model copy.
    """
    size = cfg["eval_batch_size"]
    run_items_cells(vlm, items[:size], cfg)  # warmup
    rows = []
    start = time.perf_counter()
    for first in range(0, len(items), size):
        for item_rows, _ in run_items_cells(vlm, items[first : first + size], cfg):
            rows += item_rows
        done = min(first + size, len(items))
        print(f"[{done}/{len(items)}] {(time.perf_counter() - start) / done:.1f} s/item (all cells)")
    return rows, (time.perf_counter() - start) / len(items)


def estimate_grid_hours(s_per_item: float, n_items: int, n_rerun: int, n_copies: int) -> float:
    """Hours for the grid: `n_items` items plus `n_rerun` re-run items, spread over `n_copies` model copies.

    Assumes the copies do not slow each other down, so it is a lower bound.
    """
    return s_per_item * (n_items + n_rerun) / n_copies / 3600
