"""Image rules: size limits (filter 3), perceptual hashes (filter 4), JPEG bytes."""
import io

import numpy as np
from PIL import Image

_N = 32
# DCT-II basis, rows = frequencies
_DCT = np.cos(np.pi * (2 * np.arange(_N) + 1) * np.arange(_N)[:, None] / (2 * _N))


def _bits_to_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), "big")


def phash(image: Image.Image) -> int:
    """64-bit perceptual hash: 32x32 grey, 2D DCT-II, low 8x8 block against its median."""
    grey = np.asarray(image.convert("L").resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    low = (_DCT @ grey @ _DCT.T)[:8, :8]
    return _bits_to_int(low > np.median(low))


def dhash(image: Image.Image) -> int:
    """64-bit difference hash: 9x8 grey, is each pixel brighter than its left neighbour."""
    grey = np.asarray(image.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return _bits_to_int(grey[:, 1:] > grey[:, :-1])


def near_eval(p: int, d: int, eval_hashes: list[tuple[int, int]], cfg: dict) -> bool:
    """True if (pHash, dHash) is within either threshold of any eval image."""
    return any((p ^ ep).bit_count() <= cfg["phash_max"] or (d ^ ed).bit_count() <= cfg["dhash_max"]
               for ep, ed in eval_hashes)


def size_ok(width: int, height: int, source: str, cfg: dict) -> bool:
    """Short side, aspect, long side and the per-source height limit."""
    short, long = sorted((width, height))
    return (short >= cfg["min_short_side"] and long <= cfg["max_long_side"] and long / short <= cfg["max_aspect"]
            and height <= cfg.get("max_height", {}).get(source, height))


def read_image(data: bytes) -> tuple[Image.Image, str]:
    """Decode fully: (RGB image, original format, or "" if the bytes cannot be stored as they are)."""
    image = Image.open(io.BytesIO(data))
    kept_format = image.format if image.mode == "RGB" else ""
    if image.mode in ("RGBA", "LA", "P"):  # transparency on white, not black
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return image.convert("RGB"), kept_format


def jpeg_bytes(image: Image.Image, data: bytes, kept_format: str, quality: int) -> bytes:
    """The original bytes if they are an RGB JPEG, else a re-encoded JPEG."""
    if kept_format == "JPEG":
        return data
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()
