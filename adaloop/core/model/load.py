"""Model loading: one copy, or one copy per `gpu_groups` entry in the same process.

Ported from notebooks/01_huggingfaceload.ipynb.
"""
import gc
import os

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from adaloop.core.model.cache import LOADED
from adaloop.schemas.vlm import LoadedVLM
from utils.gpus import all_gpus, matching_group


def load_model(cfg: dict, force_reload: bool = False) -> LoadedVLM:
    """Load the model frozen in eval mode, or reuse the one this process already holds.

    With several copies loaded (`load_models`) this returns copy 0 (the first gpu group) and keeps the
    others; `force_reload` drops all of them and loads copy 0 only.
    The loop count is not set here (a load-time override is ignored); use `set_loops`.
    A reused model keeps its last R.
    """
    return load_models(cfg, 1, force_reload)[0]


def load_models(cfg: dict, n_copies: int, force_reload: bool = False) -> list[LoadedVLM]:
    """`n_copies` independent models; copy j lives entirely on the j-th gpu group. Loaded copies are reused.

    More than one copy needs every GPU of `gpu_groups` visible: `select_gpu_group(config, "all")`.
    Each copy has its own processor, so threads never share a tokenizer. Cached copies beyond `n_copies`
    stay loaded; a different model, placement setting or visible GPU set drops everything first.
    """
    groups = cfg.get("gpu_groups", [])
    n_max = max(1, len(groups))
    if not 1 <= n_copies <= n_max:
        raise ValueError(f"n_copies must be in 1..{n_max} (one per gpu_groups entry), got {n_copies}")
    if n_copies > 1 and not _all_groups_visible(groups):
        raise RuntimeError(
            f"{n_copies} copies need every GPU of gpu_groups {groups} visible, CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}. Restart the kernel and run "
            "select_gpu_group(config, 'all') before importing adaloop.api."
        )
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    family = (cfg["model_id"], str(cfg["device_map"]), str(cfg["max_memory"]), str(groups), gpus)
    if force_reload or any(key[:-1] != family for key in LOADED):
        unload_model()
    copies = []
    for j in range(n_copies):
        key = (*family, j)
        if key in LOADED:
            print(f"reusing loaded model {cfg['model_id']} (copy {j}) on visible GPUs {gpus}")
        else:
            LOADED[key] = _load(cfg, _copy_max_memory(cfg, j))
        copies.append(LOADED[key])
    return copies


def unload_model() -> None:
    """Drop every cached copy and free GPU memory.

    Memory is only released once no variable (e.g. a notebook's `vlm` or `vlms`) still points to a model.
    """
    if not LOADED:
        return
    print(f"unloading {len(LOADED)} cached model(s) {next(iter(LOADED))[0]}")
    LOADED.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _all_groups_visible(groups: list[list[int]]) -> bool:
    """Whether CUDA_VISIBLE_DEVICES names exactly the union of `gpu_groups`."""
    return bool(groups) and matching_group([all_gpus(groups)]) is not None


def _copy_max_memory(cfg: dict, j: int) -> dict:
    """`max_memory` of copy j, keyed by visible-device index.

    cfg["max_memory"] is keyed by position inside a gpu group. With one group visible it is used as is.
    With all groups visible, copy j gets those budgets on the visible indices of group j.
    """
    groups = cfg.get("gpu_groups", [])
    budgets = cfg["max_memory"]
    if not _all_groups_visible(groups):
        return budgets
    visible = [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if len(budgets) != len(groups[j]):
        raise ValueError(f"max_memory {budgets} needs one entry per GPU of group {groups[j]}")
    # Other devices are left out, not set to 0: accelerate keeps headroom on the first listed device even
    # when its budget is 0, so zeros would split copy 1 differently from copy 0.
    return {visible.index(gpu): budgets[k] for gpu, k in zip(groups[j], sorted(budgets))}


def _load(cfg: dict, max_memory: dict) -> LoadedVLM:
    """Load processor and model from the Hub and check the model sits only on the `max_memory` devices."""
    model_id = cfg["model_id"]
    processor = AutoProcessor.from_pretrained(model_id)
    model, loading_info = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        device_map=cfg["device_map"],
        max_memory=max_memory,
        experts_implementation="grouped_mm",
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    model.eval()
    model.requires_grad_(False)
    # Check from Jiaqi-Gao/adaloop-vlm:adaloop/models/load.py @ e870bff: offloading would distort timings.
    offloaded = {name: dev for name, dev in model.hf_device_map.items() if str(dev) in ("cpu", "disk", "meta")}
    if offloaded:
        raise RuntimeError(f"model not fully on GPU; offloaded: {offloaded}")
    outside = {name: dev for name, dev in model.hf_device_map.items()
               if int(str(dev).removeprefix("cuda:")) not in max_memory}
    if outside:
        raise RuntimeError(f"model placed outside its GPUs {sorted(max_memory)}: {outside}")
    loaded = LoadedVLM(model=model, processor=processor, loading_info=loading_info)
    print(f"loaded {model_id} on visible GPUs {loaded.devices} | device map {model.hf_device_map} | "
          f"{model.get_memory_footprint() / 1024**3:.2f} GiB")
    return loaded
