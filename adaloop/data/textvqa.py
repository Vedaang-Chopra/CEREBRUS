"""TextVQA validation loader → list[Item].

Images, questions and the 10 reference answers are columns of the same rows.
"""
from functools import partial
from pathlib import Path

from datasets import load_dataset

from adaloop.data.rows import row_image
from adaloop.data.splits import build_manifest, check_sizes, grow_manifest, load_manifest
from adaloop.schemas.items import Item, LazyItem

BENCHMARK = "textvqa"


def build_textvqa_splits(cfg: dict, root: str | Path = ".", grow: bool = False) -> dict:
    """Create the image-disjoint TextVQA manifest if missing; return the manifest.

    An existing manifest must have the configured sizes; `grow` extends it to them, keeping its ids.
    """
    bcfg = cfg["benchmarks"][BENCHMARK]
    path = Path(root) / bcfg["manifest"]
    if path.exists() and not grow:
        manifest = load_manifest(path)
        check_sizes(manifest, cfg["splits"]["sizes"], path)
        return manifest
    if grow and not path.exists():
        raise ValueError(f"no split manifest {path} to grow")
    data = load_dataset(bcfg["source"], split=bcfg["hf_split"])
    # Read only the id columns, so no image is decoded.
    question_ids = [str(qid) for qid in data["question_id"]]
    if grow:
        return grow_manifest(path, question_ids, list(data["image_id"]), cfg["splits"])
    return build_manifest(path, question_ids, list(data["image_id"]), cfg["splits"], {"dataset": bcfg["source"]})


def load_textvqa(cfg: dict, split: str, limit: int | None = None, root: str | Path = ".",
                 shard: tuple[int, int] = (0, 1), lazy: bool = False) -> list[Item]:
    """Load the first `limit` questions of `split` (manifest order) as Items, images decoded as RGB.

    `shard` (i, n) keeps every n-th of them starting at i, so only those images are decoded.
    `lazy` decodes no image here: each read of `item.image` decodes it from the dataset (training).
    """
    bcfg = cfg["benchmarks"][BENCHMARK]
    manifest = build_textvqa_splits(cfg, root)
    ids = manifest["splits"][split][:limit][shard[0] :: shard[1]]

    data = load_dataset(bcfg["source"], split=bcfg["hf_split"])
    row_of_question = {str(qid): i for i, qid in enumerate(data["question_id"])}

    columns = data.remove_columns("image")  # reading a full row would decode its image
    items = []
    for qid in ids:
        row = columns[row_of_question[qid]]
        image = partial(row_image, data, row_of_question[qid])
        items.append((LazyItem if lazy else Item)(
            item_id=qid,
            benchmark=BENCHMARK,
            split=split,
            image_id=row["image_id"],
            image=image if lazy else image(),
            question=row["question"],
            answers=tuple(row["answers"]),
        ))
    return items
