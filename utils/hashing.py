"""Run provenance: config hash and git commit."""
import hashlib
import json
import subprocess
from pathlib import Path

# placement only: which GPUs a run uses must not change its hash (or break resume)
HASH_EXCLUDE = {"gpu_groups"}


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
