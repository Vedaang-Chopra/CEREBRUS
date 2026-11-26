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


class LazyItem(Item):
    """An `Item` built with a zero-argument image loader: every read of `image` decodes anew, nothing is kept."""

    @property
    def image(self) -> Image.Image:
        source = self.__dict__["_image"]
        return source() if callable(source) else source  # dataclasses.replace passes a decoded image back in

    @image.setter  # reached only from the dataclass __init__; the instance stays frozen
    def image(self, source) -> None:
        self.__dict__["_image"] = source
