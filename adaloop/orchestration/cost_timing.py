"""Wall-clock measurements that go with the analytic cost model (stage S6)."""
import statistics
import time
from collections.abc import Mapping, Sequence

import torch

from adaloop.core.model.budget import generate_answers
from adaloop.core.model.inference import generate
from adaloop.core.model.loops import set_loops
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
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
