"""Loader of the headroom-probe datasets (`cfg["extra_benchmarks"]["sets"]`, configs/probe.yaml).

Only the listed parquet files are downloaded. Images that match a GQA or TextVQA eval image (pHash / dHash) are left
out of the manifest; a dataset's splits (`probe`, `pool`) share no image. Never the datasets' train splits.
"""
import hashlib
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download
from PIL import Image

from adaloop.data.extra_rows import IMAGE_COLUMN, IMAGE_NAME, ROWS
from adaloop.data.post.images import dhash, near_eval, phash
from adaloop.data.splits import build_manifest, check_sizes, load_manifest
from adaloop.schemas.items import Item

PROBE, POOL = "probe", "pool"


def _source(bcfg: dict) -> datasets.Dataset:
    """The listed files of `repo` (a Hub dataset id, or a local folder in tests) as one table."""
    repo = bcfg["repo"]
    paths = [str(Path(repo) / name) if Path(repo).is_dir() else hf_hub_download(repo, name, repo_type="dataset")
             for name in bcfg["files"]]
    return datasets.load_dataset("parquet", data_files=paths, split="train")


def _raw(data: datasets.Dataset, column: str) -> datasets.Dataset:
    """`data` with the image column left as stored ({bytes, path} or bytes), so reading a row decodes nothing."""
    if isinstance(data.features[column], datasets.Image):
        return data.cast_column(column, datasets.Image(decode=False))
    return data


def _image_bytes(value: dict | bytes) -> bytes:
    stored = value["bytes"] if isinstance(value, dict) else value
    if not stored:
        raise ValueError("a row holds no image bytes")
    return stored


def _decode(value: dict | bytes) -> Image.Image:
    """RGB image, decoded as `datasets` does it (EXIF orientation applied), like the GQA / TextVQA loaders."""
    return datasets.Image().decode_example({"bytes": _image_bytes(value), "path": None}).convert("RGB")


def _image_id(name: str, row: dict, value: dict | bytes) -> str:
    return row[IMAGE_NAME[name]] if name in IMAGE_NAME else hashlib.sha256(_image_bytes(value)).hexdigest()[:16]


def _set_split_cfg(cfg: dict, name: str) -> dict:
    """The `splits`-shaped settings of one dataset: `n_items` probe questions, everything else stays as the pool."""
    extra = cfg["extra_benchmarks"]
    frac = extra["sets"][name]["image_frac"]
    return {"seed": cfg["splits"]["seed"], "sizes": {PROBE: extra["n_items"], POOL: None},
            "image_fracs": {PROBE: frac, POOL: 1.0 - frac}}


def build_extra_splits(cfg: dict, name: str, eval_hashes: list[tuple[int, int]], dedupe: dict,
                       root: str | Path = ".") -> dict:
    """Create the manifest of one probe dataset if missing (else check its sizes); return it.

    `eval_hashes` are (pHash, dHash) of the eval images, `dedupe` the two distance limits.
    """
    bcfg = cfg["extra_benchmarks"]["sets"][name]
    path, split_cfg = Path(root) / bcfg["manifest"], _set_split_cfg(cfg, name)
    if path.exists():
        manifest = load_manifest(path)
        check_sizes(manifest, split_cfg["sizes"], path)
        return manifest
    column = IMAGE_COLUMN.get(name, "image")
    data = _raw(_source(bcfg), column)

    ### Step-1:- Question and image ids; each image is hashed once
    question_ids, image_ids, near = [], [], {}
    for index, row in enumerate(data):
        image_id = _image_id(name, row, row[column])
        if image_id not in near:
            image = _decode(row[column])
            near[image_id] = near_eval(phash(image), dhash(image), eval_hashes, dedupe)
        question_ids.append(ROWS[name](row, index)[0])
        image_ids.append(image_id)
    if len(set(question_ids)) != len(question_ids):
        raise ValueError(f"{name}: item ids are not unique")

    ### Step-2:- Leave out every question of an image that is close to an eval image
    dropped = sorted(image_id for image_id, close in near.items() if close)
    kept = [(q, i) for q, i in zip(question_ids, image_ids) if not near[i]]
    print(f"{name}: {len(question_ids)} questions, {len(near)} images | {len(dropped)} images "
          f"({len(question_ids) - len(kept)} questions) near a GQA / TextVQA image, left out")
    meta = {"dataset": bcfg["repo"], "files": bcfg["files"], "dedupe": dedupe, "n_eval_images": len(eval_hashes),
            "dropped_near_eval": dropped}
    return build_manifest(path, [q for q, _ in kept], [i for _, i in kept], split_cfg, meta)


def load_extra(cfg: dict, name: str, split: str = PROBE, limit: int | None = None,
               root: str | Path = ".") -> list[Item]:
    """The first `limit` questions of `split` of one probe dataset (manifest order) as Items."""
    bcfg = cfg["extra_benchmarks"]["sets"][name]
    path = Path(root) / bcfg["manifest"]
    if not path.exists():
        raise ValueError(f"no split manifest {path}: run `python scripts/mvp.py probe-splits`")
    manifest = load_manifest(path)
    check_sizes(manifest, _set_split_cfg(cfg, name)["sizes"], path)
    ids = manifest["splits"][split][:limit]
    column = IMAGE_COLUMN.get(name, "image")
    data = _raw(_source(bcfg), column)
    text = data.remove_columns(column)
    # `text` keeps BLINK's image_2, which its mapper checks
    rows = {record[0]: (index, record) for index, record in
            ((index, ROWS[name](row, index)) for index, row in enumerate(text))}

    items, images = [], set(manifest["images"][split])
    for item_id in ids:
        index, (_, question, answers, question_type) = rows[item_id]
        row = data[index]
        image_id = _image_id(name, row, row[column])
        if image_id not in images:
            raise ValueError(f"{name} {item_id}: its image is not the one in {path} (the source files changed)")
        items.append(Item(item_id=item_id, benchmark=name, split=split, image_id=image_id,
                          image=_decode(row[column]), question=question, answers=answers,
                          question_type=question_type))
    return items
