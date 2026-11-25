"""File input and output helpers."""
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict."""
    logger.info("loading config %s", path)
    with open(path) as handle:
        return yaml.safe_load(handle)
