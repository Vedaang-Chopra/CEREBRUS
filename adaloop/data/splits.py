"""Image-disjoint splits, frozen to a JSON manifest.

Split method adapted from Jiaqi-Gao/adaloop-vlm:adaloop/benchmarks/gqa.py @ e870bff
(shuffle image ids with a fixed seed, assign whole images to splits, then sample questions).
"""
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path


def question_pools(
    question_ids: Sequence[str],
    image_ids: Sequence[str],
    names: list[str],
    image_fracs: dict[str, float],
    rng: random.Random,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Assign whole images to splits by fraction; return each split's sorted question pool and image count."""
    images = sorted(set(image_ids))
    rng.shuffle(images)

    image_split: dict[str, str] = {}
    start = 0
    for n, name in enumerate(names):
        end = len(images) if n == len(names) - 1 else start + round(image_fracs[name] * len(images))
        for image in images[start:end]:
            image_split[image] = name
        start = end

    pools: dict[str, list[str]] = {name: [] for name in names}
    for qid, image in zip(question_ids, image_ids, strict=True):
        pools[image_split[image]].append(qid)
    n_images = {name: sum(1 for s in image_split.values() if s == name) for name in names}
    return {name: sorted(pool) for name, pool in pools.items()}, n_images


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
    names = list(sizes)
    pools, n_images = question_pools(question_ids, image_ids, names, image_fracs, rng)

    splits = {}
    for name in names:
        pool = pools[name]
        size = sizes[name]
        if size is not None and size > len(pool):
            raise ValueError(f"split {name}: wants {size} questions, pool has {len(pool)}")
        splits[name] = rng.sample(pool, len(pool) if size is None else size)
        print(f"split {name}: {len(splits[name])} questions from {n_images[name]} images")
    return splits


def manifest_hash(splits: dict[str, list[str]]) -> str:
    """Short hash of the split contents, stored in the manifest."""
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()[:12]


def save_manifest(path: str | Path, splits: dict[str, list[str]], meta: dict) -> None:
    """Write splits plus metadata (dataset, seed, hash) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "hash": manifest_hash(splits), "splits": splits}, indent=1))


def build_manifest(
    path: str | Path, question_ids: Sequence[str], image_ids: Sequence[str], split_cfg: dict, meta: dict
) -> dict:
    """Make the splits from `split_cfg`, save them with `meta` and each split's image ids, return the manifest."""
    splits = make_splits(question_ids, image_ids, split_cfg["sizes"], split_cfg["image_fracs"], split_cfg["seed"])
    image_of = dict(zip(question_ids, image_ids, strict=True))
    images = {name: sorted({image_of[q] for q in ids}) for name, ids in splits.items()}
    save_manifest(path, splits, {**meta, "seed": split_cfg["seed"], "images": images})
    return load_manifest(path)


def check_sizes(manifest: dict, sizes: dict[str, int | None], path: str | Path) -> None:
    """Refuse a manifest whose split sizes differ from the config (a size of None is not checked)."""
    wrong = {name: (len(manifest["splits"][name]), size) for name, size in sizes.items()
             if size is not None and len(manifest["splits"][name]) != size}
    if wrong:
        raise ValueError(f"split manifest {path} does not have the configured sizes (have, want): {wrong}; "
                         "grow it with `python scripts/mvp.py splits --grow`")


def grow_manifest(
    path: str | Path, question_ids: Sequence[str], image_ids: Sequence[str], split_cfg: dict
) -> dict:
    """Grow a saved manifest to `split_cfg["sizes"]`, keeping every old id; return the manifest.

    Old ids stay first and in their order, so `ids[:limit]` up to the old size picks the same items. New ids are a
    seeded sample of the same split's remaining pool, appended. Splits of size None keep their whole pool.
    """
    old = load_manifest(path)
    sizes, seed = split_cfg["sizes"], split_cfg["seed"]
    names = list(sizes)
    if set(names) != set(old["splits"]) or seed != old["seed"]:
        raise ValueError(f"{path}: split names or seed differ from the config, cannot grow")
    pools, n_images = question_pools(question_ids, image_ids, names, split_cfg["image_fracs"], random.Random(seed))

    ### Step-1:- Check every split before writing anything
    for name in names:
        have, pool, size = old["splits"][name], pools[name], sizes[name]
        print(f"{path} {name}: pool {len(pool)} questions from {n_images[name]} images")
        outside = set(have) - set(pool)
        if outside:
            raise ValueError(f"{path} {name}: {len(outside)} old ids are not in the pool (other data, seed or "
                             "image_fracs than the manifest was built with)")
        if size is None and len(have) != len(pool):
            raise ValueError(f"{path} {name}: keeps its whole pool ({len(pool)}) but has {len(have)} ids")
        if size is not None and size > len(pool):
            raise ValueError(f"{path} {name}: wants {size} questions, the maximum possible is {len(pool)}")
        if size is not None and size < len(have):
            raise ValueError(f"{path} {name}: has {len(have)} questions, growing cannot shrink it to {size}")

    ### Step-2:- Append new ids from each split's own pool
    splits = {}
    for name in names:
        have, size = list(old["splits"][name]), sizes[name]
        if size is not None and size > len(have):
            rest = sorted(set(pools[name]) - set(have))
            have += random.Random(f"{seed}:{name}").sample(rest, len(rest))[: size - len(old["splits"][name])]
        splits[name] = have

    ### Step-3:- Old ids kept, no duplicate ids, image-disjoint
    image_of = dict(zip(question_ids, image_ids, strict=True))
    images = {name: sorted({image_of[q] for q in ids}) for name, ids in splits.items()}
    for name in names:
        assert splits[name][: len(old["splits"][name])] == old["splits"][name], f"{name}: old ids moved"
        print(f"{path} {name}: {len(old['splits'][name])} -> {len(splits[name])} questions, "
              f"{len(old['images'][name])} -> {len(images[name])} images")
    all_ids = [q for ids in splits.values() for q in ids]
    assert len(all_ids) == len(set(all_ids)), "duplicate question ids"
    all_images = [i for ids in images.values() for i in ids]
    assert len(all_images) == len(set(all_images)), "an image is in two splits"
    print(f"✓ {path}: old ids kept, no duplicate ids, splits image-disjoint")

    if splits == old["splits"]:
        print(f"{path}: already at the configured sizes, not rewritten")
        return old
    meta = {key: value for key, value in old.items() if key not in ("hash", "splits")}
    save_manifest(path, splits, {**meta, "images": images, "grown_from": old["hash"]})
    return load_manifest(path)


def load_manifest(path: str | Path) -> dict:
    """Read a manifest and check its stored hash."""
    manifest = json.loads(Path(path).read_text())
    if manifest_hash(manifest["splits"]) != manifest["hash"]:
        raise ValueError(f"split manifest {path} was edited: hash mismatch")
    return manifest
