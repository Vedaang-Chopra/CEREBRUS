"""Image loading."""
import logging
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(url_or_path: str, max_side: int) -> Image.Image:
    """Load an image from a URL or path as RGB, with the long side at most `max_side`."""
    logger.info("loading image %s", url_or_path)
    if url_or_path.startswith(("http://", "https://")):
        response = requests.get(url_or_path, timeout=30)
        response.raise_for_status()
        source = BytesIO(response.content)
    else:
        source = url_or_path
    return resize_long_side(Image.open(source), max_side)


def resize_long_side(image: Image.Image, max_side: int) -> Image.Image:
    """RGB copy with the long side at most `max_side` (never upscales; keeps aspect ratio)."""
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    logger.info("image size %s", image.size)
    return image
