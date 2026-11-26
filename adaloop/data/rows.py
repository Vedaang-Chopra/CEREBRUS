"""Decode one image of a (memory-mapped) HF dataset."""
from datasets import Dataset
from PIL import Image


def row_image(data: Dataset, index: int) -> Image.Image:
    """The `image` of row `index` as RGB."""
    return data[index]["image"].convert("RGB")
