"""Train one budget-tolerance LoRA (S9) on lora_train items against the teacher cache; resumable.

One run is one variant (`variant_config`). Each sample is its own microbatch (one R, one item); `eff_batch`
samples are accumulated per optimizer step.
"""
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from adaloop.core.train.distill import training_mode
from adaloop.core.train.lora import lora_parameters, lora_untrained, save_lora
from adaloop.core.train.schedule import train_sample, variant_name
from adaloop.orchestration.lora_data import Targets, item_loss
from adaloop.orchestration.teacher_cache import TRAIN_SPLIT, teacher_hash
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash

if TYPE_CHECKING:
    from peft import PeftModel


def lora_hash(cfg: dict) -> str:
    """Hash of the config values that change the trained weights (the number of steps is not one: runs resume)."""
    method = cfg["compressor"]
    return config_hash({
        "teacher": teacher_hash(cfg), "lora": cfg["lora"],
        "train": {key: cfg["train"][key] for key in ("lr", "weight_decay", "max_grad_norm")},
        "compressor": method, "compression": cfg.get("compression", {}).get(method),
    })


def lora_run_dir(cfg: dict, root: str | Path, name: str) -> Path:
    """<root>/<train.out_dir>/<variant>/<name>_<hash>/; `name` says what the run is (smoke, full)."""
    return Path(root) / cfg["train"]["out_dir"] / variant_name(cfg) / f"{name}_{lora_hash(cfg)}"


def _train_step(vlm: LoadedVLM, items: Sequence[Item], targets: Targets, cfg: dict, step: int,
                eff_batch: int) -> dict:
    """Accumulate the gradient of `eff_batch` samples; return the step's mean loss (all, and k < 1 only)."""
    losses, pruned = [], []
    for index in range(step * eff_batch, (step + 1) * eff_batch):
        sample = train_sample(index, len(items), cfg)
        item = items[sample.item_n]
        loss = item_loss(vlm, item, targets, cfg, sample.k, sample.R)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is {loss.item()} at step {step}, item {item.item_id}, {sample}")
        (loss / eff_batch).backward()  # backward per sample: only one graph is alive at a time
        losses.append(loss.item())
        if sample.k < 1.0:
            pruned.append(losses[-1])
    return {"step": step, "loss": sum(losses) / len(losses),
            "loss_pruned": sum(pruned) / len(pruned) if pruned else None}


def _run_record(items: Sequence[Item], cfg: dict, eff_batch: int) -> dict:
    """What must not change when a run is resumed."""
    return {"variant": variant_name(cfg), "teacher_R": cfg["lora"]["teacher_R"], "config_hash": lora_hash(cfg),
            "seed": cfg["lora"]["seed"], "eff_batch": eff_batch, "n_items": len(items),
            "items_hash": config_hash({"items": [[item.benchmark, item.item_id] for item in items]})}


def _save_state(peft_model: "PeftModel", params: Mapping, optimizer: torch.optim.Optimizer, history: list[dict],
                out: Path) -> None:
    """Adapter folder for inference, plus one atomically written file with everything needed to resume."""
    save_lora(peft_model, out / "adapter")
    state = {"adapter": {name: p.detach().cpu() for name, p in params.items()},
             "optimizer": optimizer.state_dict(), "history": history}
    torch.save(state, out / "train_state.tmp")
    (out / "train_state.tmp").replace(out / "train_state.pt")
    (out / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))


def _load_state(vlm: LoadedVLM, params: Mapping, optimizer: torch.optim.Optimizer, record: dict,
                out: Path) -> list[dict]:
    """Restore the run found in `out` and return its history; a new run needs an untrained adapter."""
    if not (out / "train_state.pt").exists():
        if not lora_untrained(vlm):
            raise RuntimeError(f"new run in {out}, but the adapter has already been trained: remove_lora, attach_lora")
        return []
    saved = json.loads((out / "run.json").read_text())
    changed = {key: (saved.get(key), value) for key, value in record.items() if saved.get(key) != value}
    if changed:
        raise ValueError(f"{out} holds a different run (saved, now): {changed}")
    state = torch.load(out / "train_state.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(state["adapter"][name])
    optimizer.load_state_dict(state["optimizer"])
    print(f"resuming {out} at step {len(state['history'])}")
    return state["history"]


def train_lora(
    vlm: LoadedVLM,
    peft_model: "PeftModel",
    items: Sequence[Item],
    targets: Targets,
    cfg: dict,
    out_dir: str | Path,
    steps: int | None = None,
    eff_batch: int | None = None,
) -> list[dict]:
    """Train the adapter from `attach_lora` up to `steps` optimizer steps; return one history row per step.

    `cfg` is a `variant_config`. `out_dir` gets `adapter/`, `train_state.pt`, `history.jsonl` and `run.json`; a run
    found there is resumed. `steps` and `eff_batch` default to cfg["train"].
    """
    train = cfg["train"]
    steps = train["steps"] if steps is None else steps
    eff_batch = train["eff_batch"] if eff_batch is None else eff_batch
    if not items or any(item.split != TRAIN_SPLIT for item in items):
        raise ValueError(f"training takes {TRAIN_SPLIT} items only")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    params = lora_parameters(vlm)
    optimizer = torch.optim.AdamW(params.values(), lr=train["lr"], weight_decay=train["weight_decay"])
    record = _run_record(items, cfg, eff_batch)
    history = _load_state(vlm, params, optimizer, record, out)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    (out / "run.json").write_text(json.dumps({**record, "git_hash": git_hash(), "gpu_type": gpu, "steps": steps},
                                             indent=1))
    n_params = sum(p.numel() for p in params.values())
    print(f"{record['variant']}: {n_params / 1e6:.2f}M adapter params | {len(items)} items | step {len(history)} -> "
          f"{steps}, {eff_batch} samples each")

    with training_mode(vlm, train["gradient_checkpointing"]):
        for step in range(len(history), steps):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            row = _train_step(vlm, items, targets, cfg, step, eff_batch)
            row["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params.values(), train["max_grad_norm"]))
            optimizer.step()
            row["step_s"] = time.perf_counter() - start
            history.append(row)
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
                print(f"[{step + 1}/{steps}] loss {row['loss']:.4f} | grad {row['grad_norm']:.3f} | "
                      f"{row['step_s']:.1f} s")
            if (step + 1) % train["checkpoint_every"] == 0 or step + 1 == steps:
                _save_state(peft_model, params, optimizer, history, out)
    optimizer.zero_grad(set_to_none=True)
    return history


def lora_run_estimate(history: Sequence[dict], cfg: dict, eff_batch: int) -> dict:
    """Hours of one full variant run from the step times of a run with `eff_batch` samples per step."""
    times = [row["step_s"] for row in history[1:]] or [row["step_s"] for row in history]  # step 0 warms up
    s_per_sample = sum(times) / len(times) / eff_batch
    steps, full_batch = cfg["train"]["steps"], cfg["train"]["eff_batch"]
    return {"s_per_sample": s_per_sample, "full_steps": steps, "full_eff_batch": full_batch,
            "full_hours_per_variant": s_per_sample * full_batch * steps / 3600}
