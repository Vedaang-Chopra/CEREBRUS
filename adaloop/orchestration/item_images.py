"""item_id -> image id of the grid items. The grid rows and the split manifests do not hold this pairing."""
import json
from collections.abc import Sequence
from pathlib import Path

from adaloop.data.splits import load_manifest
from adaloop.orchestration.grid import grid_dir

FILE = "item_images.json"  # a grid folder may carry its own (the synthetic one does)
ID_COLUMNS = {"gqa": ("id", "imageId"), "textvqa": ("question_id", "image_id")}


def _parquet_pattern(benchmark: str, bench: dict) -> str:
    if benchmark == "gqa":
        return f"datasets/{bench['source']}/{bench['questions_config']}/{bench['hf_split']}-*.parquet"
    return f"datasets/{bench['source']}/data/{bench['hf_split']}-*.parquet"


def build_item_images(cfg: dict, root: str | Path, path: str | Path,
                      splits: Sequence[str] = ("ctrl_train", "dev")) -> dict[str, dict[str, str]]:
    """Write {benchmark: {item_id: image_id}} of the manifest items of `splits` to `path` (needs network).

    Reads only the two id columns of the public Hub parquet files (a few MB, no images, nothing is cached). The images
    found for each split must be exactly the images the manifest lists for it.
    """
    import pyarrow.parquet as pq  # only this one-off builder needs the Hub and parquet
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=False)  # public data: no account needed
    out = {}
    for benchmark, bench in cfg["benchmarks"].items():
        manifest = load_manifest(Path(root) / bench["manifest"])
        question, image = ID_COLUMNS[benchmark]
        files = sorted(fs.glob(_parquet_pattern(benchmark, bench)))
        if not files:
            raise ValueError(f"{benchmark}: no parquet files at {_parquet_pattern(benchmark, bench)}")
        pairs = {}
        for file in files:
            table = pq.read_table(file, columns=[question, image], filesystem=fs)
            pairs.update(zip(map(str, table[question].to_pylist()), map(str, table[image].to_pylist())))
        out[benchmark] = {}
        for split in splits:
            found = {item_id: pairs[item_id] for item_id in manifest["splits"][split]}
            if set(found.values()) != set(manifest["images"][split]):
                raise ValueError(f"{benchmark} {split}: the images read do not match the manifest's")
            out[benchmark].update(found)
        print(f"{benchmark}: {len(out[benchmark])} items, {len(set(out[benchmark].values()))} images")
    Path(path).write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    return out


def load_item_images(cfg: dict, ctrl: dict, root: str | Path) -> dict[str, dict[str, str]]:
    """The pairing: <grid folder>/item_images.json if the folder has one, else `ctrl_val.item_images`."""
    local = grid_dir(cfg, root, ctrl["grid_name"]) / FILE
    path = local if local.is_file() else Path(root) / ctrl["ctrl_val"]["item_images"]
    if not path.is_file():
        raise ValueError(f"no item -> image file {path}: build it with `build_item_images`")
    return json.loads(path.read_text())
