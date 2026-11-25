"""File input and output helpers."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict. A top-level `extends: <file>` loads the parent (relative to this file) and merges it under this file."""
    path = Path(path)
    with open(path) as handle:
        child = yaml.safe_load(handle)
    if not isinstance(child, dict) or "extends" not in child:
        return child
    parent = load_config(path.parent / child.pop("extends"))
    merged = {**parent, **child}
    for key in parent.keys() & child.keys():
        if isinstance(parent[key], dict) and isinstance(child[key], dict):
            merged[key] = {**parent[key], **child[key]}
    return merged
