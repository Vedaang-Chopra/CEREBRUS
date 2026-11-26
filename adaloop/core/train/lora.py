"""The LoRA adapter on the core-layer attention projections (PEFT).

peft is imported inside the functions that need it, so `adaloop.api` imports without it.
The adapter is injected into `vlm.model` in place: keep using `vlm` for forward and generate, and the returned
PEFT object only for `disable_adapter()`, saving and `remove_lora`.
"""
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from adaloop.schemas.vlm import LoadedVLM

if TYPE_CHECKING:
    from peft import PeftModel

PROJECTIONS_PER_LAYER = 4  # q, k, v, o


def has_lora(vlm: LoadedVLM) -> bool:
    """True when an adapter has been injected into the model."""
    return any(hasattr(module, "lora_A") for module in vlm.model.modules())


def lora_target_names(vlm: LoadedVLM, cfg: dict) -> list[str]:
    """Names of the modules that cfg["lora"]["target_regex"] matches (full match, as peft does)."""
    regex = re.compile(cfg["lora"]["target_regex"])
    return [name for name, _ in vlm.model.named_modules() if regex.fullmatch(name)]


def expected_target_count(vlm: LoadedVLM) -> int:
    """q, k, v and o projection of every core (looped) layer."""
    text_cfg = vlm.model.model.language_model.config
    return PROJECTIONS_PER_LAYER * (text_cfg.loop_layer_index_end - text_cfg.loop_layer_index_start)


def target_layer(name: str) -> int:
    """Decoder layer index in a module or parameter name."""
    return int(re.search(r"\.layers\.(\d+)\.", name)[1])


def _check_targets(vlm: LoadedVLM, cfg: dict) -> None:
    """The regex must hit exactly the core-layer attention projections, all plain nn.Linear."""
    if has_lora(vlm):
        raise RuntimeError("this model already has an adapter; remove_lora() it first")
    text_cfg = vlm.model.model.language_model.config
    core = set(range(text_cfg.loop_layer_index_start, text_cfg.loop_layer_index_end))
    names = lora_target_names(vlm, cfg)
    if len(names) != expected_target_count(vlm) or {target_layer(name) for name in names} != core:
        raise ValueError(f"target_regex matches {len(names)} modules, expected {expected_target_count(vlm)} "
                         f"in layers {sorted(core)}: {names}")
    wrong = {name: type(vlm.model.get_submodule(name)).__name__ for name in names
             if type(vlm.model.get_submodule(name)) is not nn.Linear}
    if wrong:
        raise TypeError(f"LoRA targets must be plain nn.Linear, got {wrong}")


def attach_lora(vlm: LoadedVLM, cfg: dict) -> "PeftModel":
    """Inject a new, untrained adapter (output unchanged) and return its PEFT handle, in eval mode."""
    from peft import LoraConfig, get_peft_model

    _check_targets(vlm, cfg)
    lora = cfg["lora"]
    torch.manual_seed(lora["seed"])  # lora_A is random, lora_B starts at zero; the same init for every variant
    peft_config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
                             target_modules=lora["target_regex"])
    # peft freezes everything else, keeps adapter weights in float32 and returns the base layer's dtype (BF16)
    peft_model = get_peft_model(vlm.model, peft_config)
    peft_model.eval()
    return peft_model


def load_lora(vlm: LoadedVLM, adapter_dir: str | Path) -> "PeftModel":
    """Inject a saved adapter into the model, for inference."""
    from peft import PeftModel

    if has_lora(vlm):
        raise RuntimeError("this model already has an adapter; remove_lora() it first")
    peft_model = PeftModel.from_pretrained(vlm.model, str(adapter_dir))
    peft_model.eval()
    return peft_model


def save_lora(peft_model: "PeftModel", adapter_dir: str | Path) -> None:
    """Write adapter weights and adapter config (no base weights)."""
    peft_model.save_pretrained(str(adapter_dir), save_embedding_layers=False)


def remove_lora(peft_model: "PeftModel") -> None:
    """Take the adapter out again (nothing is merged), leaving the base model."""
    peft_model.unload()


def lora_parameters(vlm: LoadedVLM) -> dict[str, nn.Parameter]:
    """The trainable parameters; an error if there are none or any of them is not an adapter weight."""
    trainable = {name: p for name, p in vlm.model.named_parameters() if p.requires_grad}
    others = [name for name in trainable if "lora_" not in name]
    if not trainable or others:
        raise RuntimeError(f"only adapter weights may be trainable (call attach_lora); other trainable: {others[:5]}")
    return trainable


def lora_untrained(vlm: LoadedVLM) -> bool:
    """True when every lora_B is still zero, i.e. the adapter does not change any output."""
    return all(not p.any() for name, p in vlm.model.named_parameters() if "lora_B" in name)


@torch.no_grad()
def base_fingerprint(vlm: LoadedVLM) -> dict[str, tuple[float, float]]:
    """(sum, sum of squares) of every non-adapter parameter, to show that training leaves the base model alone."""
    out = {}
    for name, p in vlm.model.named_parameters():
        if "lora_" not in name:
            x = p.detach().float()
            # peft renames a wrapped layer's weight to <layer>.base_layer.weight
            out[name.replace(".base_layer.", ".")] = (x.sum().item(), x.square().sum().item())
    return out
