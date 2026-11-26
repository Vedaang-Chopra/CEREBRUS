"""The candidate store: Parquet shards, one state file per finished source, the val split and the summary CSV."""
import csv
import hashlib
import json
import os
from dataclasses import asdict, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adaloop.schemas.post_data import Candidate

COLUMNS = [f.name for f in fields(Candidate)]
_SCHEMA = pa.schema([(name, pa.binary() if name == "image" else pa.int32() if name in ("width", "height")
                      else pa.string()) for name in COLUMNS])


def write_shard(folder: Path, source: str, index: int, candidates: list[Candidate]) -> str:
    """Write one shard (the only place rows are persisted); return its file name."""
    folder = Path(folder) / "shards"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{source}-{index:03d}.parquet"
    table = pa.Table.from_pylist([asdict(c) for c in candidates], schema=_SCHEMA)
    pq.write_table(table, folder / f"{name}.tmp", row_group_size=200)
    os.replace(folder / f"{name}.tmp", folder / name)
    return name


def clear_source(folder: Path, source: str) -> None:
    """Remove what an interrupted run of this source left behind, so `shards/` holds only live rows."""
    for path in Path(folder).glob(f"shards/{source}-[0-9][0-9][0-9].parquet*"):
        path.unlink()
    (Path(folder) / "state" / f"{source}.json").unlink(missing_ok=True)


def save_state(folder: Path, source: str, state: dict) -> None:
    """Mark a source finished."""
    path = Path(folder) / "state" / f"{source}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(json.dumps(state, indent=1))
    os.replace(path.with_suffix(".tmp"), path)


def load_states(folder: Path) -> dict[str, dict]:
    """Finished sources by name."""
    return {path.stem: json.loads(path.read_text()) for path in sorted(Path(folder).glob("state/*.json"))}


def shard_paths(folder: Path) -> list[Path]:
    """Shards of finished sources."""
    return [Path(folder) / "shards" / name for state in load_states(folder).values() for name in state["shards"]]


def load_post_data(folder: str | Path, columns: list[str] | None = None, ids: list[str] | None = None) -> pa.Table:
    """The store as a table. Default columns leave the images out; `ids` keeps only those rows."""
    columns = columns or [name for name in COLUMNS if name != "image"]
    filters = [("id", "in", ids)] if ids else None
    tables = [pq.read_table(path, columns=columns, filters=filters) for path in shard_paths(folder)]
    if not tables:
        raise ValueError(f"no finished shards in {folder}")
    return pa.concat_tables(tables)


def assign_splits(folder: Path, val_per_bucket: dict[str, int], seed: int) -> dict[str, int]:
    """Mark a seeded sample of each bucket as val (all others train). Images are unique in the store (dedupe), so
    the two splits share none. Shards are rewritten one at a time."""
    rows = load_post_data(folder, ["id", "bucket", "phash", "dhash"]).to_pylist()
    if len({(r["phash"], r["dhash"]) for r in rows}) != len(rows):
        raise ValueError("duplicate image hashes in the store: train and val could share an image")
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    val, counts = set(), dict.fromkeys(val_per_bucket, 0)
    for row in rows:
        if counts[row["bucket"]] < val_per_bucket[row["bucket"]]:
            counts[row["bucket"]] += 1
            val.add(row["id"])
    for path in shard_paths(folder):
        table = pq.read_table(path)
        split = pa.array(["val" if i in val else "train" for i in table["id"].to_pylist()], pa.string())
        pq.write_table(table.set_column(table.schema.get_field_index("split"), "split", split),
                       f"{path}.tmp", row_group_size=200)
        os.replace(f"{path}.tmp", path)
    return counts


def write_summary(folder: Path) -> Path:
    """One CSV line per finished source."""
    rows = [{k: " ".join(v) if k == "shards" else v for k, v in state.items()} for state in load_states(folder).values()]
    path = Path(folder) / "summary.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
