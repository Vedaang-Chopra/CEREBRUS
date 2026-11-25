"""GQA testdev-balanced loader → list[Item].

Adapted from Jiaqi-Gao/adaloop-vlm:adaloop/benchmarks/gqa.py @ e870bff.
"""
import logging
from pathlib import Path

from datasets import load_dataset

from adaloop.data.splits import load_manifest, make_splits, save_manifest
from adaloop.schemas.items import Item

logger = logging.getLogger(__name__)

BENCHMARK = "gqa"


def build_gqa_splits(cfg: dict, root: str | Path = ".") -> dict:
    """Create the image-disjoint GQA manifest if missing; return the manifest."""
    bcfg, scfg = cfg["benchmarks"][BENCHMARK], cfg["splits"]
    path = Path(root) / bcfg["manifest"]
    if path.exists():
        return load_manifest(path)
    logger.info("no manifest at %s, creating it", path)
    questions = load_dataset(bcfg["source"], bcfg["questions_config"], split=bcfg["hf_split"])
    splits = make_splits(questions["id"], questions["imageId"], scfg["sizes"], scfg["image_fracs"], scfg["seed"])
    image_of = dict(zip(questions["id"], questions["imageId"], strict=True))
    images = {name: sorted({image_of[q] for q in ids}) for name, ids in splits.items()}
    meta = {"dataset": bcfg["source"], "config": bcfg["questions_config"], "seed": scfg["seed"], "images": images}
    save_manifest(path, splits, meta)
    return load_manifest(path)


def load_gqa(cfg: dict, split: str, limit: int | None = None, root: str | Path = ".") -> list[Item]:
    """Load the first `limit` questions of `split` (manifest order) as Items, images decoded as RGB."""
    bcfg = cfg["benchmarks"][BENCHMARK]
    manifest = build_gqa_splits(cfg, root)
    ids = manifest["splits"][split][:limit]
    logger.info("loading %d GQA %s items", len(ids), split)

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
    logger.info("loaded %d GQA items", len(items))
    return items
