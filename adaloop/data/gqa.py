"""GQA testdev-balanced loader → list[Item].

Adapted from Jiaqi-Gao/adaloop-vlm:adaloop/benchmarks/gqa.py @ e870bff.
"""
from pathlib import Path

from datasets import load_dataset

from adaloop.data.splits import build_manifest, load_manifest
from adaloop.schemas.items import Item

BENCHMARK = "gqa"


def build_gqa_splits(cfg: dict, root: str | Path = ".") -> dict:
    """Create the image-disjoint GQA manifest if missing; return the manifest."""
    bcfg = cfg["benchmarks"][BENCHMARK]
    path = Path(root) / bcfg["manifest"]
    if path.exists():
        return load_manifest(path)
    questions = load_dataset(bcfg["source"], bcfg["questions_config"], split=bcfg["hf_split"])
    meta = {"dataset": bcfg["source"], "config": bcfg["questions_config"]}
    return build_manifest(path, questions["id"], questions["imageId"], cfg["splits"], meta)


def load_gqa(cfg: dict, split: str, limit: int | None = None, root: str | Path = ".",
             shard: tuple[int, int] = (0, 1)) -> list[Item]:
    """Load the first `limit` questions of `split` (manifest order) as Items, images decoded as RGB.

    `shard` (i, n) keeps every n-th of them starting at i, so only those images are decoded.
    """
    bcfg = cfg["benchmarks"][BENCHMARK]
    manifest = build_gqa_splits(cfg, root)
    ids = manifest["splits"][split][:limit][shard[0] :: shard[1]]

    questions = load_dataset(bcfg["source"], bcfg["questions_config"], split=bcfg["hf_split"])
    images = load_dataset(bcfg["source"], bcfg["images_config"], split=bcfg["hf_split"])
    row_of_question = {qid: i for i, qid in enumerate(questions["id"])}
    row_of_image = {iid: i for i, iid in enumerate(images["id"])}

    items = []
    for qid in ids:
        row = questions[row_of_question[qid]]
        items.append(Item(
            item_id=qid,
            benchmark=BENCHMARK,
            split=split,
            image_id=row["imageId"],
            image=images[row_of_image[row["imageId"]]]["image"].convert("RGB"),
            question=row["question"],
            answers=(row["answer"],),
            question_type=row["types"]["structural"],
            meta={"semantic_type": row["types"]["semantic"], "detailed_type": row["types"]["detailed"]},
        ))
    return items
