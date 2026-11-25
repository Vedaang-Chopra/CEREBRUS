"""TextVQA validation loader → list[Item].

Images, questions and the 10 reference answers are columns of the same rows.
"""
import logging
from pathlib import Path

from datasets import load_dataset

from adaloop.data.splits import build_manifest, load_manifest
from adaloop.schemas.items import Item

logger = logging.getLogger(__name__)

BENCHMARK = "textvqa"


def build_textvqa_splits(cfg: dict, root: str | Path = ".") -> dict:
    """Create the image-disjoint TextVQA manifest if missing; return the manifest."""
    bcfg = cfg["benchmarks"][BENCHMARK]
    path = Path(root) / bcfg["manifest"]
    if path.exists():
        return load_manifest(path)
    logger.info("no manifest at %s, creating it", path)
    data = load_dataset(bcfg["source"], split=bcfg["hf_split"])
    # Read only the id columns, so no image is decoded.
    question_ids = [str(qid) for qid in data["question_id"]]
    return build_manifest(path, question_ids, list(data["image_id"]), cfg["splits"], {"dataset": bcfg["source"]})


def load_textvqa(cfg: dict, split: str, limit: int | None = None, root: str | Path = ".") -> list[Item]:
    """Load the first `limit` questions of `split` (manifest order) as Items, images decoded as RGB."""
    bcfg = cfg["benchmarks"][BENCHMARK]
    manifest = build_textvqa_splits(cfg, root)
    ids = manifest["splits"][split][:limit]
    logger.info("loading %d TextVQA %s items", len(ids), split)

    data = load_dataset(bcfg["source"], split=bcfg["hf_split"])
    row_of_question = {str(qid): i for i, qid in enumerate(data["question_id"])}

    items = []
    for qid in ids:
        row = data[row_of_question[qid]]
        items.append(Item(
            item_id=qid,
            benchmark=BENCHMARK,
            split=split,
            image_id=row["image_id"],
            image=row["image"].convert("RGB"),
            question=row["question"],
            answers=tuple(row["answers"]),
        ))
    logger.info("loaded %d TextVQA items", len(items))
    return items
