"""ctrl_val: a part of the train split, held out by IMAGE, for early stopping and architecture choice."""
import random

import pandas as pd


def carve_val(rows: pd.DataFrame, images: dict[str, dict[str, str]], source: str, name: str, image_frac: float,
              seed: int) -> pd.Series:
    """The `split` of every row, with the rows of `image_frac` of the images of `source` renamed to `name`.

    rows: benchmark, item_id, split (any number of rows per item). images[benchmark][item_id] is the item's image.
    Per benchmark, a seeded sample of the images of the `source` items present, so no image is on both sides.
    """
    in_source = rows["split"] == source
    if not in_source.any():
        raise ValueError(f"no rows of split {source!r} to hold {name!r} out of")
    out = rows["split"].copy()
    for benchmark in sorted(rows.loc[in_source, "benchmark"].unique()):
        part = rows[in_source & (rows["benchmark"] == benchmark)]
        unknown = set(part["item_id"]) - set(images.get(benchmark, {}))
        if unknown:
            raise ValueError(f"no image id for {len(unknown)} {benchmark} items, e.g. {sorted(unknown)[:3]}: "
                             "rebuild the item -> image file (`build_item_images`)")
        image = part["item_id"].map(images[benchmark])
        pool = sorted(image.unique())
        if len(pool) < 2:
            raise ValueError(f"{benchmark}: {len(pool)} image in {source!r}, cannot hold images out")
        held = random.Random(seed).sample(pool, min(max(1, round(image_frac * len(pool))), len(pool) - 1))
        out[image.index[image.isin(held)]] = name
    return out
