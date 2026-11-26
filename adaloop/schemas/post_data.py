"""One post-training candidate: an image with one question-answer turn."""
from dataclasses import dataclass


@dataclass
class Candidate:
    """A stored row. `phash`/`dhash` are 64-bit hashes as 16 hex digits; `ratings` is a JSON string or None."""

    id: str
    bucket: str
    source_config: str
    question: str
    answer: str
    answer_full: str
    answer_type: str  # short | count | free
    ratings: str | None
    image: bytes  # JPEG
    width: int
    height: int
    phash: str
    dhash: str
    split: str = "train"
