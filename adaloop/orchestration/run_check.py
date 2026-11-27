"""Check a grid run folder against its meta.json (rows, features files) and write its checksums. CPU only."""
import hashlib
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from adaloop.core.controller.features import TOP_N
from adaloop.orchestration.evaluate import read_results
from adaloop.orchestration.features import load_features
from adaloop.orchestration.grid import COLUMNS
from adaloop.orchestration.run_meta import expected_rows, read_meta

KEY = ("benchmark", "item_id", "rerun_id", "k", "R")
# columns that may be empty (None / NaN); every other column of COLUMNS must have a value
NULLABLE = {"pred", "fallback", "format_ok", "answer_len_ok"}
BATCHED_NULLABLE = {"t_prefill_ms", "t_decode_ms", "peak_mem_gb"}  # only when batch_size > 1
NO_VIT_SELECTORS = {"none", "lowres"}  # t_vit_ms is empty: the vision tower runs inside the prefill
SCALARS = ("delta_norm", "vis_attn_mass", "entropy", "kl_prev", "top1_prob", "tail_mass", "flops_readout",
           "flops_state", "n_tokens", "n_vis")
TENSORS = {"h_last": torch.float16, "top_logp": torch.float32, "top_ids": torch.int32}


def _empty(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _row_problems(row: dict) -> list[str]:
    """Columns of one row that are missing or empty although they must not be."""
    allowed = set(NULLABLE)
    if (row.get("batch_size") or 0) > 1:
        allowed |= BATCHED_NULLABLE
    if row.get("selector") in NO_VIT_SELECTORS:
        allowed.add("t_vit_ms")
    return [name for name in COLUMNS if name not in row or (_empty(row[name]) and name not in allowed)]


def _feature_problems(features: dict, cells: list[tuple], vocab_top: int) -> list[str]:
    """What is wrong in one item's {(k, R): features}."""
    out = [f"lacks cell {cell}" for cell in cells if cell not in features]
    ran = {R for _, R in cells}
    for k, R in [cell for cell in cells if cell in features]:
        cell = features[k, R]
        lacking = [name for name in (*TENSORS, *SCALARS) if name not in cell]
        if lacking:
            out.append(f"cell {(k, R)} lacks {lacking}")
            continue
        for name, dtype in TENSORS.items():
            value = cell[name]
            if value.dtype != dtype or value.ndim != 1 or (name != "h_last" and len(value) != vocab_top):
                out.append(f"cell {(k, R)} {name}: {value.dtype} {tuple(value.shape)}")
            elif not torch.isfinite(value).all():
                out.append(f"cell {(k, R)} {name}: not finite")
        may_be_nan = {"delta_norm"} if R == 1 else set()
        if R - 1 not in ran:
            may_be_nan.add("kl_prev")
        bad = [name for name in SCALARS if name not in may_be_nan and not math.isfinite(cell[name])]
        if bad:
            out.append(f"cell {(k, R)} not finite: {bad}")
    return out


def _write_manifest(run_dir: Path) -> int:
    """manifest.sha256: `<sha256>  <path>` for every file under the folder (check with `sha256sum -c`)."""
    skip = {"manifest.sha256", ".meta.lock"}
    lines = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name not in skip):
        with path.open("rb") as handle:
            lines.append(f"{hashlib.file_digest(handle, 'sha256').hexdigest()}  {path.relative_to(run_dir).as_posix()}\n")
    (run_dir / "manifest.sha256").write_text("".join(lines))
    return len(lines)


def check_run(run_dir: str | Path, partial: bool = False) -> bool:
    """Print what the folder holds and what is wrong; True if it passes (then manifest.sha256 is written).

    With `partial`, missing rows and the features of items without rows are reported but are not failures.
    """
    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    digest, variant = meta["config_hash"], meta["variant"]
    cells = [(k, R) for k, R in meta["cells"]]
    expected = expected_rows(meta)
    rows = [row for path in sorted(run_dir.glob(f"grid_*_{digest}.jsonl")) for row in read_results(path)]
    problems: list[str] = []

    ### Step-1:- Rows: no duplicate, nothing unexpected, nothing missing, no forbidden empty value
    counts = Counter(tuple(row.get(name) for name in KEY) for row in rows if row.get("model_variant") == variant)
    problems += [f"duplicate row {key} x{n}" for key, n in counts.items() if n > 1]
    problems += [f"row of another model_variant {row.get('model_variant')!r}" for row in rows
                 if row.get("model_variant") != variant]
    problems += [f"unexpected row {key}" for key in counts if key not in expected]
    for row in rows:
        bad = _row_problems(row)
        if bad:
            problems.append(f"row {tuple(row.get(name) for name in KEY)}: empty or absent {bad}")
    missing = expected - set(counts)
    for rerun_id, name in ((0, "main"), (1, "re-run")):
        n = sum(key[2] == rerun_id for key in expected)
        have = n - sum(key[2] == rerun_id for key in missing)
        print(f"{name}: {have} of {n} rows ({n // len(cells)} items x {len(cells)} cells)")
    if missing and not partial:
        problems.append(f"{len(missing)} rows missing, e.g. {sorted(missing)[:3]}")
    parquet = run_dir / "grid.parquet"
    if not partial and parquet.is_file() and len(pd.read_parquet(parquet, columns=["k"])) != len(expected):
        problems.append(f"grid.parquet does not hold {len(expected)} rows: merge again")

    ### Step-2:- One features file per item and run, with every cell and key
    with_rows = {key[:3] for key in counts}
    for rerun_id, items_key, folder in ((0, "items", "features"), (1, "rerun_items", "features_rerun1")):
        found = 0
        for benchmark, item_id, _ in meta[items_key]:
            path = run_dir / folder / variant / digest / f"{benchmark}_{item_id}.pt"
            if not path.is_file():
                if not partial or (benchmark, item_id, rerun_id) in with_rows:
                    problems.append(f"missing {path.relative_to(run_dir)}")
                continue
            try:
                bad = _feature_problems(load_features(path), cells, TOP_N)
            except Exception as exc:  # a half-written or damaged file: report it and go on
                bad = [f"unreadable ({type(exc).__name__}: {exc})"]
            found += 1
            problems += [f"{path.relative_to(run_dir)}: {text}" for text in bad]
        print(f"{folder}: {found} of {len(meta[items_key])} files")

    for text in problems[:30]:
        print("FAIL", text)
    if problems:
        print(f"FAILED: {len(problems)} problems in {run_dir}")
        return False
    n_files = _write_manifest(run_dir)
    state = "complete" if not missing else f"PARTIAL, {len(missing)} rows still missing"
    print(f"PASSED ({state}) | manifest.sha256: {n_files} files")
    return True
