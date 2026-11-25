"""Notebook environment setup."""
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils.io import load_config

logger = logging.getLogger(__name__)


def _find_root() -> Path:
    """Return the first folder with pyproject.toml, searching from the cwd, then from this file."""
    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve()
    for folder in (cwd, *cwd.parents, *here.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml above {cwd} or {here}")


def setup_notebook(config: str = "configs/mvp.yaml") -> tuple[Path, dict]:
    """Prepare a notebook session and return (repo root, loaded config).

    Changes into the repo root, puts it on sys.path, loads env.env if present
    (variables already set are kept) and configures INFO logging.
    A relative `config` path is resolved against the repo root.
    """
    root = _find_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    env_loaded = load_dotenv(root / "env.env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    logger.info("setup_notebook root=%s env_file_loaded=%s", root, env_loaded)
    cfg = load_config(root / config)
    logger.info("setup_notebook done, config=%s", config)
    return root, cfg
