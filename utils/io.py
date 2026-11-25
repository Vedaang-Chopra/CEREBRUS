"""File input and output helpers."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict.

    A top-level `extends: <file>` loads the parent and merges it under this file. A top-level `prompts: <file>` is
    replaced by that file's content, so the prompt text is part of the config hash. Both are relative to this file.
    """
    path = Path(path)
    with open(path) as handle:
        child = yaml.safe_load(handle)
    if not isinstance(child, dict):
        return child
    if isinstance(child.get("prompts"), str):
        child["prompts"] = load_config(path.parent / child["prompts"])
    if "extends" not in child:
        return child
    parent = load_config(path.parent / child.pop("extends"))
    merged = {**parent, **child}
    for key in parent.keys() & child.keys():
        if isinstance(parent[key], dict) and isinstance(child[key], dict):
            merged[key] = {**parent[key], **child[key]}
    return merged
