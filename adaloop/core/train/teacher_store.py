"""Files of the teacher cache: tensor parts per shard and R, and a JSONL index that says which records exist.

A part is written first (atomically), then its index lines. A record exists when the index names it and its part is
on disk, so a killed run loses only what was not yet indexed. Every writer (shard x copy) has its own index file.
"""
import json
import os
from pathlib import Path

import torch


def read_index(folder: str | Path) -> list[dict]:
    """Index lines of every record in the cache (unreadable lines and lines without their part are skipped)."""
    folder = Path(folder)
    rows = {}
    for path in sorted(folder.glob("index_*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping unreadable line in {path.name}: {line[:80]}")
                continue
            if (folder / row["part"]).is_file():
                rows[row["benchmark"], row["item_id"], row["R"]] = row
    return list(rows.values())


def repair_index(path: str | Path) -> None:
    """Drop a half-written last line before appending to an index file."""
    path = Path(path)
    if path.exists():
        good = []
        for line in path.read_text().splitlines():
            try:
                json.loads(line)
            except json.JSONDecodeError:
                print(f"dropping unreadable line in {path.name}: {line[:80]}")
                continue
            good.append(line + "\n")
        path.write_text("".join(good))


def write_part(folder: str | Path, writer: str, R: int, index_rows: list[dict], tensors: list[dict]) -> None:
    """Save one part of records at loop count R and index them in `index_<writer>.jsonl`."""
    folder = Path(folder)
    parts = folder / f"R{R}"
    parts.mkdir(parents=True, exist_ok=True)
    number = len(list(parts.glob(f"{writer}_part*.pt")))
    part = f"R{R}/{writer}_part{number:05d}.pt"
    tmp = folder / f"{part}.tmp{os.getpid()}"
    torch.save(tensors, tmp)
    tmp.replace(folder / part)
    with (folder / f"index_{writer}.jsonl").open("a") as handle:
        for row_number, row in enumerate(index_rows):
            handle.write(json.dumps({**row, "part": part, "row": row_number}) + "\n")


def load_records(folder: str | Path, rows: list[dict]) -> list[dict]:
    """The tensors of these index lines, in order; each part file is read once."""
    folder = Path(folder)
    parts: dict[str, list] = {}
    out = []
    for row in rows:
        if row["part"] not in parts:
            parts[row["part"]] = torch.load(folder / row["part"], map_location="cpu", weights_only=True)
        out.append(parts[row["part"]][row["row"]])
    return out
