"""Model loading.

Ported from notebooks/01_huggingfaceload.ipynb.
"""
import gc
import os

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from adaloop.core.model.cache import LOADED
from adaloop.schemas.vlm import LoadedVLM


def load_model(cfg: dict, force_reload: bool = False) -> LoadedVLM:
    """Load the model frozen in eval mode, or reuse the one this process already holds.

    The loop count is not set here (a load-time override is ignored); use `set_loops`.
    A reused model keeps its last R.
    """
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    key = (cfg["model_id"], str(cfg["device_map"]), str(cfg["max_memory"]), gpus)
    if key in LOADED and not force_reload:
        print(f"reusing loaded model {cfg['model_id']} on GPUs {gpus}")
        return LOADED[key]
    unload_model()  # two copies would not fit
    LOADED[key] = _load(cfg)
    return LOADED[key]


def unload_model() -> None:
    """Drop the cached model and free GPU memory.

    Memory is only released once no variable (e.g. a notebook's `vlm`) still points to the model.
    """
    if not LOADED:
        return
    print(f"unloading cached model {next(iter(LOADED))[0]}")
    LOADED.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _load(cfg: dict) -> LoadedVLM:
    """Load processor and model from the Hub and check nothing was offloaded."""
    model_id = cfg["model_id"]
    processor = AutoProcessor.from_pretrained(model_id)
    model, loading_info = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        device_map=cfg["device_map"],
        max_memory=cfg["max_memory"],
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
    print(f"loaded {model_id} | device map {model.hf_device_map} | {model.get_memory_footprint() / 1024**3:.2f} GiB")
    return LoadedVLM(model=model, processor=processor, loading_info=loading_info)
