"""Benchmark item: one image-question pair with its reference answers."""
from dataclasses import dataclass, field

from PIL import Image


@dataclass(frozen=True)
class Item:
    """One benchmark question.

    `answers` holds every reference answer (1 for GQA, 10 for TextVQA).
    `question_type` is the benchmark's own type label (GQA structural type), used in error analysis.
    """

    item_id: str
    benchmark: str
    split: str
    image_id: str
    image: Image.Image
    question: str
    answers: tuple[str, ...]
    question_type: str = ""
    meta: dict = field(default_factory=dict)
