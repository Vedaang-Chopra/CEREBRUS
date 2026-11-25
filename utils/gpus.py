"""Choose which physical GPUs this process may use. Call before anything initializes CUDA.

Notebooks import `select_gpu_group` from here, the one notebook import allowed outside `adaloop.api`:
importing `adaloop.api` already initializes CUDA, so the GPU choice must come first.
Nothing in this module's import chain may import torch or transformers.
"""
import os
import subprocess
import sys
from pathlib import Path

from utils.io import load_config
from utils.paths import find_repo_root

SMI_QUERY = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]


def free_gpu_mib() -> dict[int, int] | None:
    """Free memory (MiB) per physical GPU from nvidia-smi (no CUDA init); None if there is no nvidia-smi."""
    try:
        out = subprocess.run(SMI_QUERY, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        return None
    return {int(index): int(free) for index, free in (line.split(",") for line in out.strip().splitlines())}


def all_gpus(groups: list[list[int]]) -> list[int]:
    """Every GPU of `gpu_groups`, in config order."""
    return [gpu for ids in groups for gpu in ids]


def matching_group(groups: list[list[int]]) -> list[int] | None:
    """The group that CUDA_VISIBLE_DEVICES names (in any order), or None."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    for ids in groups:
        if sorted(visible.split(",")) == sorted(str(gpu) for gpu in ids):
            return list(ids)
    return None


def select_gpu_group(config: str | Path | dict, group: int | str | None = None) -> list[int]:
    """Restrict this process to one of the config's `gpu_groups`, or all of them; return the physical GPU ids.

    `config` is a config path (relative paths start at the repo root) or an already loaded dict.

    `group` is an index into `gpu_groups`; None takes the first group whose GPUs all have at least
    cfg["min_free_gib"] free; "all" takes every group (for one model copy per group, `load_models`) and
    needs that much free on every GPU. GPUs outside `gpu_groups` are never used. Once CUDA is initialized
    the devices are fixed: the visible set is returned if it is what was asked for (no memory test, this
    process may hold the model), else an error asks for a kernel restart. Without nvidia-smi nothing is
    set and [] is returned.
    """
    cfg = config if isinstance(config, dict) else load_config(find_repo_root() / config)
    groups = cfg["gpu_groups"]
    if group not in (None, "all") and group not in range(len(groups)):
        raise ValueError(f"gpu group {group!r} does not exist; use 0..{len(groups) - 1}, 'all' or None; "
                         f"gpu_groups = {groups}")
    free = free_gpu_mib()
    if free is None:
        print("no nvidia-smi: GPU selection skipped")
        return []
    allowed = sorted(set(all_gpus(groups)))
    missing = [gpu for gpu in allowed if gpu not in free]
    if missing:
        raise RuntimeError(f"gpu_groups names GPUs {missing} that nvidia-smi does not list")
    table = "\n".join(f"  GPU {gpu}: {free[gpu] / 1024:5.1f} GiB free" for gpu in allowed)

    if group is None:
        candidates = groups
    else:
        candidates = [all_gpus(groups)] if group == "all" else [groups[group]]

    # never import torch here: if it is not loaded yet, CUDA cannot have been initialized
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        chosen = _visible_group(candidates)
    else:
        chosen = _free_group(candidates, free, cfg["min_free_gib"], table)
        _set_visible(chosen)
    ids = ",".join(str(gpu) for gpu in chosen)
    print(f"using GPUs {ids} | free {', '.join(f'{free[gpu] / 1024:.1f}' for gpu in chosen)} GiB")
    print(table)
    return chosen


def select_for_notebook(config: str | Path) -> tuple[list[int], int]:
    """Select the GPUs for the config's `n_copies` (1: one free group, more: all groups); return (ids, n_copies)."""
    cfg = load_config(find_repo_root() / config)
    n_copies = cfg["n_copies"]
    return select_gpu_group(cfg, "all" if n_copies > 1 else None), n_copies


def _visible_group(wanted: list[list[int]]) -> list[int]:
    """The GPU set CUDA was already started on; it must be one of `wanted`."""
    chosen = matching_group(wanted)
    if chosen is not None:
        return chosen
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    raise RuntimeError(
        f"CUDA is already initialized with CUDA_VISIBLE_DEVICES={visible or 'unset'}, expected one of {wanted}. "
        "Devices cannot change now: restart the kernel and run select_gpu_group first."
    )


def _free_group(candidates: list[list[int]], free: dict[int, int],
                min_free_gib: float, table: str) -> list[int]:
    """The first candidate GPU set whose GPUs all have `min_free_gib` free."""
    for ids in candidates:
        if all(free[gpu] >= min_free_gib * 1024 for gpu in ids):
            return list(ids)
    raise RuntimeError(f"no GPU group in {candidates} has {min_free_gib} GiB free on every GPU:\n{table}")


def _set_visible(ids: list[int]) -> None:
    """Make only `ids` visible to CUDA, numbered in PCI bus order (the order nvidia-smi uses)."""
    value = ",".join(str(gpu) for gpu in ids)
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preset is not None and preset != value:
        print(f"overriding CUDA_VISIBLE_DEVICES={preset} with {value}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = value
