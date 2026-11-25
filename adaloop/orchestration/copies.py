"""Several model copies in one process: speed against one copy, and a check that outputs do not change."""
import statistics
import time
from collections.abc import Sequence

from adaloop.orchestration.evaluate import run_chunks
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM


def _rows(vlms: Sequence[LoadedVLM], items: Sequence[Item], cfg: dict, R: int, batch_size: int) -> list[dict]:
    """Result rows in item order, in memory, each tagged with the copy that produced it."""
    return [{**row, "copy": copy}
            for copy, rows in run_chunks(vlms, items, cfg, R, batch_size) for row in rows]


def _check_enough_items(n_items: int, n_copies: int, batch_size: int) -> None:
    """Every copy must get at least one full batch, or the comparison says nothing about it."""
    if n_items < n_copies * batch_size:
        raise ValueError(f"{n_copies} copies at batch {batch_size} need >= {n_copies * batch_size} items, got {n_items}")


def time_copies(vlms: Sequence[LoadedVLM], items: Sequence[Item], cfg: dict, R: int,
                batch_sizes: Sequence[int] = (1, 8)) -> list[dict]:
    """Wall-clock seconds per item with 1 copy and with all of `vlms`, at each batch size.

    Times what `run_evaluation` does per item (input preparation, generation, scoring; no file writing),
    over the same preloaded items, after one warmup batch per copy. `speedup` is relative to 1 copy.
    """
    table = []
    for batch_size in batch_sizes:
        _check_enough_items(len(items), len(vlms), batch_size)
        one_copy_s = None
        for n_copies in sorted({1, len(vlms)}):
            copies = vlms[:n_copies]
            _rows(copies, items[: n_copies * batch_size], cfg, R, batch_size)  # warmup
            start = time.perf_counter()
            rows = _rows(copies, items, cfg, R, batch_size)  # every generate call synchronizes its GPUs
            total_s = time.perf_counter() - start
            one_copy_s = one_copy_s or total_s
            table.append({
                "R": R,
                "batch_size": batch_size,
                "n_copies": n_copies,
                "n_items": len(items),
                "total_s": total_s,
                "s_per_item": total_s / len(items),
                "speedup": one_copy_s / total_s,
                "mean_out_tokens": statistics.mean(row["n_out_tokens"] for row in rows),
            })
            print(f"batch {batch_size} | n_copies={n_copies} | {total_s / len(items):.2f} s/item")
    return table


def compare_copies(vlms: Sequence[LoadedVLM], items: Sequence[Item], cfg: dict, R: int,
                   batch_size: int = 1) -> list[dict]:
    """One row per item: the answer from copy 0 alone against the answer with all copies in threads."""
    _check_enough_items(len(items), len(vlms), batch_size)
    single = _rows(vlms[:1], items, cfg, R, batch_size)
    multi = _rows(vlms, items, cfg, R, batch_size)
    return [
        {
            "item_id": a["item_id"],
            "copy": b["copy"],
            "pred_1": a["pred"],
            "pred_n": b["pred"],
            "same_pred": a["pred"] == b["pred"],
            "same_raw": a["raw_output"] == b["raw_output"],  # every generated token, special ones included
        }
        for a, b in zip(single, multi)
    ]
