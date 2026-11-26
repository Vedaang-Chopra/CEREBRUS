"""The state of one source while it is read, and the image filters that turn a picked turn into a stored candidate."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from adaloop.data.post.images import dhash, jpeg_bytes, near_eval, phash, read_image, size_ok
from adaloop.data.post.store import write_shard
from adaloop.schemas.post_data import Candidate

# filters in order; each count is the rows still alive after that step
COUNTS = ["rows_read", "after_ratings", "after_answer", "after_download", "after_image", "after_eval", "after_dup",
          "unrated", "bytes_read"]


@dataclass
class SourceRun:
    """State of one source while it is read."""

    cfg: dict
    folder: Path
    bucket: str
    source: dict
    target: int
    blocked: list  # eval image hashes
    seen: set  # (phash, dhash) of every stored candidate
    counts: dict = field(default_factory=lambda: dict.fromkeys(COUNTS, 0))
    buffer: list = field(default_factory=list)
    shards: list = field(default_factory=list)
    files_used: int = 0
    skipped: str = ""
    files_total: int = 0

    @property
    def name(self) -> str:
        return self.source["config"]

    @property
    def full(self) -> bool:
        return self.counts["after_dup"] >= self.target

    @property
    def over_budget(self) -> bool:
        return self.counts["bytes_read"] >= self.source.get("max_gb", self.cfg["max_gb"]) * 1e9

    def flush(self) -> None:
        if self.buffer:
            self.shards.append(write_shard(self.folder, self.name, len(self.shards), self.buffer))
            self.buffer = []


def add_image(run: SourceRun, data: bytes, turn: dict, row_id: str) -> None:
    """Filters 3 and 4 and the duplicate check; keep the candidate if it survives."""
    image_cfg = run.cfg["image"]
    try:
        image, kept_format = read_image(data)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        print(f"{row_id}: image not readable: {type(exc).__name__}: {exc}")
        return
    if not size_ok(*image.size, run.name, image_cfg):
        return
    run.counts["after_image"] += 1
    p, d = phash(image), dhash(image)
    if near_eval(p, d, run.blocked, run.cfg["dedupe"]):
        return
    run.counts["after_eval"] += 1
    if (p, d) in run.seen:
        return
    run.counts["after_dup"] += 1
    run.seen.add((p, d))
    ratings = turn.pop("ratings")
    run.buffer.append(Candidate(
        id=row_id, bucket=run.bucket, source_config=run.name, ratings=json.dumps(ratings) if ratings else None,
        image=jpeg_bytes(image, data, kept_format, image_cfg["jpeg_quality"]), width=image.width,
        height=image.height, phash=f"{p:016x}", dhash=f"{d:016x}", **turn))
    if len(run.buffer) >= run.cfg["shard_rows"]:
        run.flush()
