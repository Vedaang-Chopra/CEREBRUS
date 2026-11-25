"""Choose which physical GPUs this process may use. Call before anything initializes CUDA."""
import os
import subprocess

import torch

SMI_QUERY = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]


def free_gpu_mib() -> dict[int, int] | None:
    """Free memory (MiB) per physical GPU from nvidia-smi (no CUDA init); None if there is no nvidia-smi."""
    try:
        out = subprocess.run(SMI_QUERY, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        return None
    return {int(index): int(free) for index, free in (line.split(",") for line in out.strip().splitlines())}


def select_gpu_group(cfg: dict, group: int | None = None) -> list[int]:
    """Restrict this process to one of cfg["gpu_groups"] and return its physical GPU ids.

    `group` is an index into `gpu_groups`; None takes the first group whose GPUs all have at least
    cfg["min_free_gib"] free. GPUs outside `gpu_groups` are never used. Once CUDA is initialized the
    devices are fixed: the visible group is returned (no memory test, this process may hold the model)
    or an error asks for a kernel restart. Without nvidia-smi nothing is set and [] is returned.
    """
    groups = cfg["gpu_groups"]
    if group is not None and not 0 <= group < len(groups):
        raise ValueError(f"gpu group {group} does not exist; gpu_groups = {groups}")
    free = free_gpu_mib()
    if free is None:
        print("no nvidia-smi: GPU selection skipped")
        return []
    allowed = sorted({gpu for ids in groups for gpu in ids})
    missing = [gpu for gpu in allowed if gpu not in free]
    if missing:
        raise RuntimeError(f"gpu_groups names GPUs {missing} that nvidia-smi does not list")
    table = "\n".join(f"  GPU {gpu}: {free[gpu] / 1024:5.1f} GiB free" for gpu in allowed)

    if torch.cuda.is_initialized():
        chosen = _visible_group(groups, group)
    else:
        chosen = _free_group(groups, group, free, cfg["min_free_gib"], table)
        _set_visible(chosen)
    ids = ",".join(str(gpu) for gpu in chosen)
    print(f"using GPUs {ids} | free {', '.join(f'{free[gpu] / 1024:.1f}' for gpu in chosen)} GiB")
    print(table)
    return chosen


def _visible_group(groups: list[list[int]], group: int | None) -> list[int]:
    """The group CUDA was already started on; it must be a gpu_group (and the requested one)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    wanted = groups if group is None else [groups[group]]
    for ids in wanted:
        if visible is not None and sorted(visible.split(",")) == sorted(str(gpu) for gpu in ids):
            return list(ids)
    raise RuntimeError(
        f"CUDA is already initialized with CUDA_VISIBLE_DEVICES={visible or 'unset'}, expected one of {wanted}. "
        "Devices cannot change now: restart the kernel and run setup first."
    )


def _free_group(groups: list[list[int]], group: int | None, free: dict[int, int],
                min_free_gib: float, table: str) -> list[int]:
    """The requested group, or the first one, whose GPUs all have `min_free_gib` free."""
    candidates = groups if group is None else [groups[group]]
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
