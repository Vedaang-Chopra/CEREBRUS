"""Which items share a generation batch (config `batch_grouping`).

`equal_length`: only items with the same number of visual tokens and the same number of other tokens. Compression at
one k then keeps them equally long, so a batch holds no padding. Anything else: consecutive items, left-padded.
"""
from collections.abc import Hashable, Sequence

from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM

EQUAL_LENGTH = "equal_length"


def input_key(vlm: LoadedVLM, item: Item, cfg: dict) -> tuple[int, int]:
    """(visual tokens, other tokens) of the item's uncompressed prompt; processor only, no model call."""
    ids = item_inputs(vlm, item, cfg)["input_ids"]  # (1, T)
    n_visual = int((ids == vlm.model.config.image_token_id).sum())
    return n_visual, ids.shape[1] - n_visual


def equal_key_chunks(keys: Sequence[Hashable], batch_size: int) -> list[list[int]]:
    """Indices of `keys` in chunks of at most `batch_size` with one key each, ordered by their first index."""
    groups: dict = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    chunks = [group[first : first + batch_size] for group in groups.values()
              for first in range(0, len(group), batch_size)]
    return sorted(chunks, key=lambda chunk: chunk[0])


def item_chunks(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, batch_size: int) -> list[list[Item]]:
    """`items` cut into the generation batches of a run; every item appears once."""
    if batch_size == 1 or cfg.get("batch_grouping") != EQUAL_LENGTH:
        return [list(items[first : first + batch_size]) for first in range(0, len(items), batch_size)]
    chunks = equal_key_chunks([input_key(vlm, item, cfg) for item in items], batch_size)
    if chunks:
        print(f"equal-length batches: {len(items)} items in {len(chunks)} batches (mean {len(items) / len(chunks):.1f}, "
              f"{sum(len(c) == 1 for c in chunks)} alone), batch size {batch_size}")
    return [[items[i] for i in chunk] for chunk in chunks]


def batchable_items(vlm: LoadedVLM, pool: Sequence[Item], cfg: dict, n: int) -> list[Item]:
    """Up to `n` items of `pool` that have the most partners of equal length (for the batch checks)."""
    keys = [input_key(vlm, item, cfg) for item in pool]
    groups = sorted(equal_key_chunks(keys, len(pool)), key=len, reverse=True)
    return [pool[i] for group in groups for i in group][:n]
