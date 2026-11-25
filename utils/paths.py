"""Locate the repository root from this file, not from the working directory."""
from pathlib import Path


def find_repo_root() -> Path:
    """Return the nearest folder above this file that holds pyproject.toml."""
    here = Path(__file__).resolve()
    for folder in here.parents:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml above {here}")
