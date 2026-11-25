"""File input and output helpers."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict."""
    with open(path) as handle:
        return yaml.safe_load(handle)
