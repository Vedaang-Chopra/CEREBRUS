"""Image-disjoint splits, frozen to a JSON manifest.

Split method adapted from Jiaqi-Gao/adaloop-vlm:adaloop/benchmarks/gqa.py @ e870bff
(shuffle image ids with a fixed seed, assign whole images to splits, then sample questions).
"""
import hashlib
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def make_splits(
    question_ids: Sequence[str],
    image_ids: Sequence[str],
    sizes: dict[str, int | None],
    image_fracs: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Assign whole images to splits by fraction, then sample `sizes[split]` questions per split.

    `question_ids[i]` belongs to image `image_ids[i]`. A size of None keeps every question of that split.
    Splits share no image. Each id list is in seeded shuffled order, so a prefix (`ids[:limit]`) is a random
    subset; the same inputs and seed give the same lists.
    """
    if set(sizes) != set(image_fracs):
        raise ValueError(f"sizes and image_fracs need the same splits: {sorted(sizes)} vs {sorted(image_fracs)}")
    rng = random.Random(seed)
    images = sorted(set(image_ids))
    rng.shuffle(images)

    image_split: dict[str, str] = {}
    start = 0
    names = list(sizes)
    for n, name in enumerate(names):
        end = len(images) if n == len(names) - 1 else start + round(image_fracs[name] * len(images))
        for image in images[start:end]:
            image_split[image] = name
        start = end

    pools: dict[str, list[str]] = {name: [] for name in names}
    for qid, image in zip(question_ids, image_ids, strict=True):
        pools[image_split[image]].append(qid)

    splits = {}
    for name in names:
        pool = sorted(pools[name])
        size = sizes[name]
        if size is not None and size > len(pool):
            raise ValueError(f"split {name}: wants {size} questions, pool has {len(pool)}")
        splits[name] = rng.sample(pool, len(pool) if size is None else size)
        logger.info("split %s: %d questions from %d images", name, len(splits[name]),
                    sum(1 for s in image_split.values() if s == name))
    return splits


def manifest_hash(splits: dict[str, list[str]]) -> str:
    """Short hash of the split contents, stored in the manifest."""
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()[:12]


def save_manifest(path: str | Path, splits: dict[str, list[str]], meta: dict) -> None:
    """Write splits plus metadata (dataset, seed, hash) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "hash": manifest_hash(splits), "splits": splits}, indent=1))
    logger.info("saved split manifest %s (hash %s)", path, manifest_hash(splits))


def build_manifest(
    path: str | Path, question_ids: Sequence[str], image_ids: Sequence[str], split_cfg: dict, meta: dict
) -> dict:
    """Make the splits from `split_cfg`, save them with `meta` and each split's image ids, return the manifest."""
    splits = make_splits(question_ids, image_ids, split_cfg["sizes"], split_cfg["image_fracs"], split_cfg["seed"])
    image_of = dict(zip(question_ids, image_ids, strict=True))
    images = {name: sorted({image_of[q] for q in ids}) for name, ids in splits.items()}
    save_manifest(path, splits, {**meta, "seed": split_cfg["seed"], "images": images})
    return load_manifest(path)


def load_manifest(path: str | Path) -> dict:
    """Read a manifest and check its stored hash."""
    manifest = json.loads(Path(path).read_text())
    if manifest_hash(manifest["splits"]) != manifest["hash"]:
        raise ValueError(f"split manifest {path} was edited: hash mismatch")
    logger.info("loaded split manifest %s (hash %s)", path, manifest["hash"])
    return manifest
