"""Which server this is: the placement overlay named by ADALOOP_SERVER and the GPU host check. No torch import."""
import os
import socket

import yaml

from utils.hashing import HASH_EXCLUDE
from utils.paths import find_repo_root


def apply_server_overlay(cfg: dict) -> dict:
    """`cfg` with the placement keys of configs/servers/$ADALOOP_SERVER.yaml; unchanged if the variable is unset.

    Only run configs (those with `gpu_groups`) are touched. The overlay may only set unhashed keys, so a run's
    hash is the same on every server.
    """
    name = os.environ.get("ADALOOP_SERVER")
    if not name:
        return cfg
    folder = find_repo_root() / "configs" / "servers"
    path = folder / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"ADALOOP_SERVER={name!r}: no {path}; known: {sorted(p.stem for p in folder.glob('*.yaml'))}")
    overlay = yaml.safe_load(path.read_text()) or {}
    hashed = set(overlay) - HASH_EXCLUDE
    if hashed:
        raise ValueError(f"{path} sets {sorted(hashed)}: a server overlay may only set keys in HASH_EXCLUDE")
    return {**cfg, **overlay} if "gpu_groups" in cfg else cfg


def server_name() -> str:
    """ADALOOP_SERVER, else the hostname (provenance on result rows)."""
    return os.environ.get("ADALOOP_SERVER") or socket.gethostname()


def is_gpu_host() -> bool:
    """Whether this machine is Inara or a PACE node."""
    host = socket.gethostname()
    return "inara" in host or host.endswith(".pace.gatech.edu")
