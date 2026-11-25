"""Model loading.

Ported from notebooks/01_huggingfaceload.ipynb.
"""
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from adaloop.schemas.vlm import LoadedVLM


def load_model(cfg: dict) -> LoadedVLM:
    """Load the model frozen in eval mode.

    The loop count is not set here (a load-time override is ignored); use `set_loops`.
    """
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
