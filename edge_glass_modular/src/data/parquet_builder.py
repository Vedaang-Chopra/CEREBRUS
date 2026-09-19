"""Builders for local PixMo parquet datasets.

Provenance: salvaged from `older_code_base/v2_code_base/dataset_builders.py`
(`build_pixmocap_parquet`) and the v2 notebook
`00_build_alignment_datasets_offline.ipynb` (`build_pixmocap_offline_parquet`,
`_download_image_bytes`), plus the 70/15/15 train/val/test split logic from
`00_build_qa_decoder_alignment.ipynb`. Adapted to the current package style
(logging instead of prints, Path-based outputs); behavior preserved.

The current runtime datasets (`PixmoParquetImageTextDataset`,
`PixmoQADataset` in `dataset_builder.py`) consume parquet files with embedded
`image_bytes`. This module constructs those files; it does not load them.
"""

from __future__ import annotations

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

PIXMO_CAP_REPO = "allenai/pixmo-cap"


def _download_image_bytes(args: Tuple[int, str]) -> Tuple[int, Optional[bytes]]:
    """Worker: fetch image bytes from a URL. Returns (idx, None) on failure."""
    idx, url = args
    try:
        import requests

        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return idx, resp.content
    except Exception:
        return idx, None


def build_pixmocap_parquet(
    output_path: Path,
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle_seed: int = 42,
) -> None:
    """Save a subset of the PixMo-Cap HF split directly to Parquet.

    Keeps all original columns (e.g. `image_url`, `caption`); note that this
    does NOT embed image bytes. Use `build_pixmocap_offline_parquet` for a
    file with embedded `image_bytes` as consumed by
    `PixmoParquetImageTextDataset`.

    Args:
        output_path: Where to save the Parquet file.
        split: HF split to use (default: "train").
        max_samples: If provided, randomly select at most this many samples.
        shuffle_seed: Seed for shuffling before subsetting.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PixMo-Cap split='%s' from Hugging Face...", split)
    ds = load_dataset(PIXMO_CAP_REPO, split=split)
    logger.info("Total available samples: %s", f"{len(ds):,}")

    if max_samples is not None and max_samples < len(ds):
        logger.info("Shuffling and selecting first %s samples (seed=%s)...", f"{max_samples:,}", shuffle_seed)
        ds = ds.shuffle(seed=shuffle_seed).select(range(max_samples))
    else:
        logger.info("Using full split (no subsetting).")

    logger.info("Saving subset to Parquet: %s", output_path)
    ds.to_parquet(str(output_path))
    logger.info("Done: PixMo-Cap subset saved.")


def build_pixmocap_offline_parquet(
    output_path: Path,
    split: str = "train",
    max_successful_samples: int = 20_000,
    shuffle_seed: int = 42,
    chunk_size: int = 5_000,
    num_workers: int = 32,
) -> Path:
    """Build an offline PixMo-Cap subset with images embedded as raw bytes.

    Produces a Parquet file with columns:
      - `image_bytes`: raw bytes of the downloaded image
      - `caption`: text caption
      - `image_url`: original URL
      - `sample_id`: stable ID like "pixmo_0001234"

    This is the schema consumed by `PixmoParquetImageTextDataset`
    (`required_columns = ("image_bytes", "caption", "sample_id")`).

    Args:
        output_path: Where to save the Parquet file.
        split: HF split to use (default: "train").
        max_successful_samples: Target number of *successful* samples to keep.
        shuffle_seed: Seed for deterministic shuffling of the HF split.
        chunk_size: Number of HF rows processed per download chunk.
        num_workers: Parallel download workers (capped at 32).

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PixMo-Cap split='%s' from Hugging Face...", split)
    ds = load_dataset(PIXMO_CAP_REPO, split=split)
    total = len(ds)
    logger.info("Total available samples in HF: %s", f"{total:,}")

    ds = ds.shuffle(seed=shuffle_seed)

    target = max_successful_samples
    logger.info("Target successful samples: %s", f"{target:,}")

    kept: List[Dict[str, object]] = []
    start_idx = 0
    workers = min(num_workers, 32)

    while len(kept) < target and start_idx < total:
        end_idx = min(start_idx + chunk_size, total)
        logger.info("Processing chunk %s-%s ...", f"{start_idx:,}", f"{end_idx - 1:,}")

        batch = ds.select(range(start_idx, end_idx))

        download_tasks: List[Tuple[int, str]] = []
        captions: Dict[int, str] = {}
        urls: Dict[int, str] = {}

        for local_idx, row in enumerate(batch):
            global_idx = start_idx + local_idx
            image_url = row.get("image_url", None)
            caption = row.get("caption", "")

            if not image_url:
                logger.warning("Missing image_url at HF index %s, skipping.", global_idx)
                continue

            download_tasks.append((global_idx, image_url))
            captions[global_idx] = caption
            urls[global_idx] = image_url

        if not download_tasks:
            logger.warning("No valid URLs in this chunk, moving to next.")
            start_idx = end_idx
            continue

        logger.info("Starting image downloads with %s workers for %s candidates...", workers, f"{len(download_tasks):,}")

        with Pool(processes=workers) as pool:
            for idx, img_bytes in pool.imap_unordered(_download_image_bytes, download_tasks, chunksize=256):
                if img_bytes is None:
                    # download failed; skip (do not keep broken rows)
                    continue

                kept.append(
                    {
                        "image_bytes": img_bytes,
                        "caption": captions[idx],
                        "image_url": urls[idx],
                        "sample_id": f"pixmo_{idx:07d}",
                    }
                )

                if len(kept) >= target:
                    break

        logger.info("Successful so far: %s / %s", f"{len(kept):,}", f"{target:,}")
        start_idx = end_idx

    if len(kept) < target:
        logger.warning(
            "Only %s successful downloads out of requested %s. Proceeding with available samples.",
            f"{len(kept):,}",
            f"{target:,}",
        )

    logger.info("Saving %s samples to Parquet: %s", f"{len(kept):,}", output_path)
    df = pd.DataFrame(kept)
    df.to_parquet(output_path, index=False)
    logger.info("Done: PixMo-Cap offline Parquet saved.")
    return output_path


def split_parquet_train_val_test(
    parquet_path: Path,
    output_dir: Path,
    prefix: str = "pixmo",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    """Shuffle a parquet file and write 70/15/15 train/val/test splits.

    Split logic salvaged from v2 `00_build_qa_decoder_alignment.ipynb`
    (PixmoQAMixConfig flow); remainder row goes to test so all samples are used.

    Args:
        parquet_path: Source parquet file.
        output_dir: Directory for `<prefix>_train.parquet` / `_val` / `_test`.
        prefix: Output filename prefix.
        train_ratio: Train fraction (default 0.70).
        val_ratio: Val fraction (default 0.15); test gets the remainder.
        seed: Shuffle seed.

    Returns:
        (train_path, val_path, test_path)
    """
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    n_total = len(df)
    logger.info("Splitting %s rows from %s", f"{n_total:,}", parquet_path)

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_df = shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = shuffled.iloc[n_train + n_val :].reset_index(drop=True)

    train_path = output_dir / f"{prefix}_train.parquet"
    val_path = output_dir / f"{prefix}_val.parquet"
    test_path = output_dir / f"{prefix}_test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info("Splits written: train=%s val=%s test=%s", f"{len(train_df):,}", f"{len(val_df):,}", f"{len(test_df):,}")
    return train_path, val_path, test_path
