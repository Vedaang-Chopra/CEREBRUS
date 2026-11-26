"""Cost parameters without weights: the model is built from its config on the meta device."""
import torch
from transformers import AutoConfig, AutoModelForMultimodalLM

from adaloop.core.cost.flops import cost_params
from adaloop.schemas.cost import CostParams


def cost_params_from_config(model_id: str) -> CostParams:
    """`cost_params` of `model_id` from its config.json only (no weights, no GPU)."""
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForMultimodalLM.from_config(config)
    return cost_params(model)
