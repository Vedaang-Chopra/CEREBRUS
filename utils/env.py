"""Notebook environment setup."""
import os
import sys
import warnings
from pathlib import Path

import datasets
import transformers
from dotenv import load_dotenv

from utils.gpus import select_gpu_group
from utils.io import load_config


def _find_root() -> Path:
    """Return the first folder with pyproject.toml, searching from the cwd, then from this file."""
    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve()
    for folder in (cwd, *cwd.parents, *here.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml above {cwd} or {here}")


def setup_notebook(config: str = "configs/mvp.yaml", gpu_group: int | None = None) -> tuple[Path, dict]:
    """Prepare a notebook session and return (repo root, loaded config).

    Changes into the repo root, puts it on sys.path, loads env.env if present
    (variables already set are kept),
    and silences transformers/datasets logs, FutureWarning and UserWarning.
    A relative `config` path is resolved against the repo root.
    Restricts the process to one GPU group (`select_gpu_group`): `gpu_group` indexes cfg["gpu_groups"],
    None picks a free one. Call it before anything touches CUDA (even `torch.cuda.is_available`).
    """
    root = _find_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    load_dotenv(root / "env.env")
    # Keep notebook output readable: drop library warnings (tqdm progress bars stay).
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    for category in (FutureWarning, UserWarning):
        warnings.filterwarnings("ignore", category=category)
    print(f"repo root: {root}")
    cfg = load_config(root / config)
    select_gpu_group(cfg, gpu_group)
    return root, cfg
