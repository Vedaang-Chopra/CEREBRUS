"""Notebook environment setup."""
import os
import shutil
import sys
import warnings
from pathlib import Path

import datasets
import transformers
from dotenv import load_dotenv

from utils.gpus import all_gpus, matching_group
from utils.io import load_config


def _find_root() -> Path:
    """Return the first folder with pyproject.toml, searching from the cwd, then from this file."""
    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve()
    for folder in (cwd, *cwd.parents, *here.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml above {cwd} or {here}")


def _check_gpus(cfg: dict) -> None:
    """Report whether CUDA_VISIBLE_DEVICES is one of cfg["gpu_groups"] or all of them; changes nothing."""
    if shutil.which("nvidia-smi") is None:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    groups = cfg["gpu_groups"]
    if matching_group([*groups, all_gpus(groups)]) is None:
        print(f"WARNING: CUDA_VISIBLE_DEVICES={visible or 'unset'} is not one of gpu_groups {groups} or their union. "
              "Restart the kernel and call utils.gpus.select_gpu_group before importing adaloop.api.")
    else:
        print(f"CUDA_VISIBLE_DEVICES={visible}")


def setup_notebook(config: str = "configs/mvp.yaml", overrides: dict | None = None) -> tuple[Path, dict]:
    """Prepare a notebook session and return (repo root, loaded config).

    Changes into the repo root, puts it on sys.path, loads env.env if present
    (variables already set are kept),
    and silences transformers/datasets logs, FutureWarning and UserWarning.
    A relative `config` path is resolved against the repo root. `overrides` replaces top-level config keys.
    Does not select GPUs (importing this module is already too late): it only checks
    CUDA_VISIBLE_DEVICES, set earlier by `utils.gpus.select_gpu_group`.
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
    cfg = {**load_config(root / config), **(overrides or {})}
    _check_gpus(cfg)
    return root, cfg
