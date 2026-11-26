"""Hashes of every GQA / TextVQA eval image. Only image columns and image ids are read, never questions or answers."""
import json
from pathlib import Path

from datasets import load_dataset

from adaloop.data.post.images import dhash, phash
from adaloop.data.rows import row_image
from adaloop.data.splits import load_manifest

# benchmark -> (config key of its image table, id column)
_IMAGE_TABLES = {"gqa": ("images_config", "id"), "textvqa": (None, "image_id")}


def _eval_image_ids(manifest: dict, splits: list[str]) -> list[str]:
    return sorted({image_id for split in splits for image_id in manifest["images"][split]})


def _hash_benchmark(name: str, bench_cfg: dict, image_ids: list[str]) -> list[list[str]]:
    config_key, id_column = _IMAGE_TABLES[name]
    args = (bench_cfg["source"], bench_cfg[config_key]) if config_key else (bench_cfg["source"],)
    # select_columns: the questions and answers of the test split are never loaded into Python
    data = load_dataset(*args, split=bench_cfg["hf_split"]).select_columns([id_column, "image"])
    row_of = {}
    for index, image_id in enumerate(data[id_column]):
        row_of.setdefault(image_id, index)
    rows = []
    for image_id in image_ids:
        image = row_image(data, row_of[image_id])
        rows.append([f"{name}/{image_id}", f"{phash(image):016x}", f"{dhash(image):016x}"])
    print(f"eval hashes: {name} {len(rows)} images")
    return rows


def eval_hashes(bench_cfg: dict, splits: list[str], root: str | Path, cache: str | Path) -> list[tuple[int, int]]:
    """(pHash, dHash) of every image of `splits`, cached in `cache`; the cache is rebuilt if a manifest changed."""
    manifests = {name: load_manifest(Path(root) / bench_cfg["benchmarks"][name]["manifest"]) for name in _IMAGE_TABLES}
    key = {"splits": splits, **{name: manifest["hash"] for name, manifest in manifests.items()}}
    cache = Path(cache)
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    if saved.get("key") != key:
        rows = []
        for name, manifest in manifests.items():
            rows += _hash_benchmark(name, bench_cfg["benchmarks"][name], _eval_image_ids(manifest, splits))
        saved = {"key": key, "images": rows}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(saved))
    return [(int(p, 16), int(d, 16)) for _, p, d in saved["images"]]
