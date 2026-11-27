"""File input and output helpers."""
from pathlib import Path

import yaml

from utils.servers import apply_server_overlay


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict.

    A top-level `extends: <file>` loads the parent and merges it under this file. A top-level `prompts: <file>` is
    replaced by that file's content, so the prompt text is part of the config hash. Both are relative to this file.
    With ADALOOP_SERVER set, that server's placement keys are applied last (`apply_server_overlay`).
    """
    cfg = _read_config(Path(path))
    return apply_server_overlay(cfg) if isinstance(cfg, dict) else cfg


def _read_config(path: Path) -> dict:
    """One YAML file with its `prompts` and `extends` files merged in."""
    with open(path) as handle:
        child = yaml.safe_load(handle)
    if not isinstance(child, dict):
        return child
    if isinstance(child.get("prompts"), str):
        child["prompts"] = _read_config(path.parent / child["prompts"])
    if "extends" not in child:
        return child
    parent = _read_config(path.parent / child.pop("extends"))
    merged = {**parent, **child}
    for key in parent.keys() & child.keys():
        if isinstance(parent[key], dict) and isinstance(child[key], dict):
            merged[key] = {**parent[key], **child[key]}
    return merged
