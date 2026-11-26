"""Run provenance: config hash and git commit."""
import hashlib
import json
import subprocess
from pathlib import Path

# placement, sample size and how items are batched: they must not change a run's hash (or break resume)
# `resolution` only lists the cells of the resolution check; each cell's res_px / k / match_res_px is hashed
# `pareto` only lists the cells of the S5b sweep; each cell's compressor / k / compression.cls_iqr.m is hashed
# `profile` holds the settings of the timing-only profiling (nb06 step 7); its rows go to their own folder
# `headroom` holds settings of the analysis of a finished grid; it changes no model output
HASH_EXCLUDE = {"gpu_groups", "min_free_gib", "n_copies", "sample_items", "eval_batch_size", "latency_items",
                "batch_grouping", "resolution", "pareto", "headroom", "profile"}


def config_hash(cfg: dict) -> str:
    """Short stable hash of a config dict (key order does not matter; HASH_EXCLUDE keys are dropped)."""
    kept = {key: value for key, value in cfg.items() if key not in HASH_EXCLUDE}
    blob = json.dumps(kept, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_hash(repo: str | Path = ".") -> str:
    """Current commit of `repo`, with a `-dirty` suffix if tracked files changed."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"git hash unavailable for {repo}: {exc}")
        return "unknown"
    return f"{commit}-dirty" if dirty else commit
