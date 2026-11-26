"""Shared pieces of the multi-cell runners (S5 comparison, S7 grid): compressed batches and copy scheduling."""
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

import torch

from adaloop.core.model.budget import generate_answers
from adaloop.core.model.loops import set_loops
from adaloop.orchestration.compress import compress, encode_vision
from adaloop.orchestration.evaluate import check_distinct_copies
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM


class ChunkInputs:
    """The items of one batch: their model inputs, built once, and one vision-tower run per item, made on demand."""

    def __init__(self, vlm: LoadedVLM, items: Sequence[Item], cfg: dict):
        self.vlm, self.items, self.cfg = vlm, list(items), cfg
        self.full = [item_inputs(vlm, item, cfg) for item in self.items]
        self.vision: dict[int, object] = {}

    def compressed(self, i: int, method: str, k: float) -> tuple[dict, dict]:
        """`compress` of item i; every method except none and lowres shares the item's vision run."""
        if method not in ("none", "lowres") and i not in self.vision:
            self.vision[i] = encode_vision(self.vlm, self.full[i])
        return compress(self.vlm, self.full[i], self.items[i], k, {**self.cfg, "compressor": method},
                        vision=self.vision.get(i))


def generate_cell(vlm: LoadedVLM, inputs: Sequence[dict], cfg: dict, R: int) -> list[GenerationResult]:
    """Answers of one (method, k, R) batch; a batch of one takes the batch-1 path, which times each item."""
    set_loops(vlm, R)
    return generate_answers(vlm, list(inputs), cfg, batched=len(inputs) > 1)


def map_copies(vlms: Sequence[LoadedVLM], jobs: Sequence, work: Callable) -> Iterator[tuple[int, object]]:
    """Yield (copy index, work(vlm, copy index, job)) for every job in order; copy j runs in its own thread.

    One copy runs in the calling thread. At most 2 jobs per copy are in flight.
    """
    if len(vlms) == 1:
        for job in jobs:
            yield 0, work(vlms[0], 0, job)
        return
    check_distinct_copies(vlms)
    pools = [ThreadPoolExecutor(max_workers=1) for _ in vlms]  # a copy is only ever used by its own thread
    try:
        waiting: deque = deque()
        for n, job in enumerate(jobs):
            j = n % len(vlms)
            waiting.append((j, pools[j].submit(work, vlms[j], j, job)))
            if len(waiting) >= 2 * len(vlms):
                j, future = waiting.popleft()
                yield j, future.result()
        while waiting:
            j, future = waiting.popleft()
            yield j, future.result()
    finally:
        for pool in pools:  # queued jobs are cancelled; a running one is waited for
            pool.shutdown(wait=True, cancel_futures=True)


def check_batched(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, cells: Sequence[tuple[str, float]],
                  R: int) -> list[dict]:
    """For every (method, k): all `items` as one batch against each item alone; one row per (method, k, item).

    The same compressed inputs are used for both, so only the batching differs.
    """
    chunk = ChunkInputs(vlm, items, cfg)
    rows = []
    for method, k in cells:
        prepared = [chunk.compressed(i, method, k)[0] for i in range(len(items))]
        batched = generate_cell(vlm, prepared, cfg, R)
        for item, inputs, together in zip(items, prepared, batched):
            alone = generate_cell(vlm, [inputs], cfg, R)[0]
            rows.append({"method": method, "k": k, "benchmark": item.benchmark, "item_id": item.item_id,
                         "raw_b1": alone.raw_text, "raw_batched": together.raw_text,
                         "ids_equal": torch.equal(alone.token_ids, together.token_ids)})
    return rows
