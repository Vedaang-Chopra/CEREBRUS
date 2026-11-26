"""Remote parquet files read one row group and one column set at a time, and image URL downloads."""
import hashlib
import random
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfFileSystem

# eval image overlap (docs/notes/post_training_data.md); refused whatever the config says
EXCLUDED = ("textvqa", "textcaps", "textocr(gpt4v)", "localized_narratives", "tallyqa", "visual7w", "st_vqa")


def refuse_excluded(name: str) -> None:
    """Raise for a source that shares images with an eval benchmark."""
    if name.lower() in EXCLUDED:
        raise ValueError(f"source {name!r} is excluded: its images overlap GQA / TextVQA evaluation images")


def parquet_files(repo: str, folder: str, seed: str) -> tuple[fsspec.AbstractFileSystem, list[str]]:
    """The folder's train parquet files in seeded random order. `repo` is a Hub dataset id or a local folder."""
    if Path(repo).is_dir():
        fs, base = fsspec.filesystem("file"), str(Path(repo) / folder)
    else:
        fs, base = HfFileSystem(), f"datasets/{repo}/{folder}"
    files = sorted(path for path in fs.ls(base, detail=False)
                   if path.endswith(".parquet") and path.rsplit("/", 1)[-1].startswith("train"))
    if not files:
        raise ValueError(f"no train parquet files in {base}")
    random.Random(seed).shuffle(files)
    return fs, files


def retry(read: Callable, what: str, retries: int):
    """Call `read()` with retries; the last error is raised (the source then stays unfinished and resumes later)."""
    for attempt in range(1, retries + 1):
        try:
            return read()
        except Exception as exc:  # the Hub client's network error types differ between versions
            print(f"read failed ({attempt}/{retries}) {what}: {type(exc).__name__}: {exc}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def column_bytes(parquet: pq.ParquetFile, group: int, columns: list[str]) -> int:
    """Compressed size of those columns in one row group: what a projected read transfers."""
    meta = parquet.metadata.row_group(group)
    chunks = (meta.column(i) for i in range(meta.num_columns))
    return sum(c.total_compressed_size for c in chunks if c.path_in_schema.split(".")[0] in columns)


def row_groups(fs: fsspec.AbstractFileSystem, path: str, seed: str, retries: int) -> Iterator[tuple[pq.ParquetFile, int]]:
    """(open file, row group index) in seeded random order."""
    handle = retry(lambda: fs.open(path, "rb", block_size=2**16), path, retries)
    with handle:
        parquet = retry(lambda: pq.ParquetFile(handle), path, retries)
        order = list(range(parquet.metadata.num_row_groups))
        random.Random(f"{seed}:{path.rsplit('/', 1)[-1]}").shuffle(order)
        for group in order:
            yield parquet, group


def fetch_images(rows: list[dict], cfg: dict) -> list[bytes | None]:
    """Download each row's `image_url`; None on an error, a timeout or a sha256 mismatch."""
    def one(row: dict) -> bytes | None:
        try:
            reply = requests.get(row["image_url"], timeout=cfg["timeout_s"], headers={"User-Agent": cfg["user_agent"]})
            reply.raise_for_status()
        except (requests.RequestException, ValueError, UnicodeError):  # counted by the caller, which reports the rate
            return None
        return reply.content if hashlib.sha256(reply.content).hexdigest() == row["image_sha256"] else None

    with ThreadPoolExecutor(cfg["workers"]) as pool:
        return list(pool.map(one, rows))
