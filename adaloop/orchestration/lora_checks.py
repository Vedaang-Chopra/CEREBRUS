"""The S9 checks and the eval probe (notebook 10): small functions that return rows to assert on."""
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from adaloop.core.model.budget import generate_answers
from adaloop.core.model.loops import set_loops
from adaloop.core.train.distill import training_mode
from adaloop.core.train.lora import attach_lora, base_fingerprint, lora_parameters, remove_lora, target_layer
from adaloop.core.train.schedule import variant_name
from adaloop.orchestration.compress import compress
from adaloop.orchestration.evaluate import result_row
from adaloop.orchestration.inputs import item_inputs
from adaloop.orchestration.lora_data import Targets, item_loss
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM

if TYPE_CHECKING:
    from peft import PeftModel


@torch.no_grad()
def probe_loss(vlm: LoadedVLM, items: Sequence[Item], targets: Targets, cfg: dict,
               ks: Sequence[float] | None = None, Rs: Sequence[int] | None = None) -> list[dict]:
    """Loss of fixed samples (each item at every k and R), comparable before and after training; eval mode.

    Defaults: the training keep fractions and 1.0, at cfg["smoke"]["probe_R"].
    """
    ks = [*cfg["lora"]["k_train"], 1.0] if ks is None else ks
    Rs = cfg["smoke"]["probe_R"] if Rs is None else Rs
    return [{"benchmark": item.benchmark, "item_id": item.item_id, "k": k, "R": R,
             "loss": float(item_loss(vlm, item, targets, cfg, k, R))}
            for item in items for k in ks for R in Rs]


def lora_grad_norms(vlm: LoadedVLM, item: Item, targets: Targets, cfg: dict, R: int, k: float) -> list[dict]:
    """One backward pass at (k, R) as in training; the gradient norm of each LoRA module (A and B together).

    Weights do not change and gradients are cleared. Use k < 1: at k = 1 an untrained student equals its teacher
    and the gradient is zero. While lora_B is zero, only B gets a gradient.
    """
    params = lora_parameters(vlm)
    vlm.model.zero_grad(set_to_none=True)
    with training_mode(vlm, cfg["train"]["gradient_checkpointing"]):
        item_loss(vlm, item, targets, cfg, k, R).backward()
    squares: dict[str, float] = {}
    for name, p in params.items():
        module = name.split(".lora_")[0]
        squares[module] = squares.get(module, 0.0) + (0.0 if p.grad is None else p.grad.float().square().sum().item())
    vlm.model.zero_grad(set_to_none=True)
    return [{"R": R, "layer": target_layer(module + "."), "module": module, "grad_norm": square**0.5}
            for module, square in squares.items()]


def student_layer_calls(vlm: LoadedVLM, item: Item, targets: Targets, cfg: dict, R: int, k: float) -> list[int]:
    """Calls of each decoder layer during one student forward (training mode, no backward).

    Equals `expected_layer_calls(vlm)` afterwards: the loop count in use is the sample's R.
    """
    layers = vlm.model.model.language_model.layers
    calls = [0] * len(layers)

    def make_hook(index: int):
        def hook(module, args, output):
            calls[index] += 1

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        with training_mode(vlm, cfg["train"]["gradient_checkpointing"]):
            item_loss(vlm, item, targets, cfg, k, R)
    finally:
        for handle in handles:
            handle.remove()
    return calls


def one_step_check(vlm: LoadedVLM, items: Sequence[Item], targets: Targets, cfg: dict, R: int, k: float) -> dict:
    """One optimizer step on a throwaway adapter; what is trainable and what changed.

    The model must have no adapter; it has none again afterwards. `n_base_changed` must be 0.
    """
    peft_model = attach_lora(vlm, cfg)
    try:
        params = lora_parameters(vlm)
        trainable = [name for name, p in vlm.model.named_parameters() if p.requires_grad]
        base_before = base_fingerprint(vlm)
        lora_before = {name: p.detach().clone() for name, p in params.items()}
        optimizer = torch.optim.AdamW(params.values(), lr=cfg["train"]["lr"],
                                      weight_decay=cfg["train"]["weight_decay"])
        with training_mode(vlm, cfg["train"]["gradient_checkpointing"]):
            for item in items:
                (item_loss(vlm, item, targets, cfg, k, R) / len(items)).backward()
        optimizer.step()
        base_after = base_fingerprint(vlm)
        changed = {name.split(".lora_")[0] for name, p in params.items() if not torch.equal(p, lora_before[name])}
        return {
            "n_trainable_tensors": len(trainable),
            "n_trainable_not_lora": sum("lora_" not in name for name in trainable),
            "n_trainable_params": sum(p.numel() for p in params.values()),
            "n_base_params_checked": len(base_before),
            "n_base_changed": sum(base_before[name] != base_after[name] for name in base_before),
            "n_lora_modules_changed": len(changed),
        }
    finally:
        vlm.model.zero_grad(set_to_none=True)
        remove_lora(peft_model)


def lora_eval_probe(vlm: LoadedVLM, peft_model: "PeftModel", items: Sequence[Item], cfg: dict,
                    ks: Sequence[float] | None = None, Rs: Sequence[int] | None = None) -> list[dict]:
    """Greedy result rows of the same items with the adapter off ("base") and on (the variant's name).

    Defaults: cfg["smoke"]["eval_k"] and ["eval_R"]. Batch 1; never the test split.
    """
    if any(item.split == "test" for item in items):
        raise ValueError("the test split is frozen until stage S12")
    ks = cfg["smoke"]["eval_k"] if ks is None else ks
    Rs = cfg["smoke"]["eval_R"] if Rs is None else Rs
    rows = []
    for item in items:
        for k in ks:
            inputs, info = compress(vlm, item_inputs(vlm, item, cfg), item, k, cfg)
            for R in Rs:
                set_loops(vlm, R)
                for variant in ("base", variant_name(cfg)):
                    if variant == "base":
                        with peft_model.disable_adapter():
                            gen = generate_answers(vlm, [inputs], cfg)[0]
                    else:
                        gen = generate_answers(vlm, [inputs], cfg)[0]
                    row = result_row(item, cfg, R, gen, batch_size=1, info=info)
                    rows.append({**row, "model_variant": variant})
    print(f"eval probe: {len(items)} items x k {list(ks)} x R {list(Rs)} x base/{variant_name(cfg)}")
    return rows
