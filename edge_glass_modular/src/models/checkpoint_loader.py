"""Shared VLM checkpoint loading helper.

Canonical checkpoint layouts produced by the retained training path
(`scripts/train/train_vlm_accelerate.py`, accelerate 1.10.1):

1. ``checkpoint_best/`` — weight-only save via
   ``unwrapped_model.save_pretrained(ckpt_dir, state_dict=accelerator.get_state_dict(model),
   safe_serialization=False)``. Because the model is PEFT-wrapped (LoRA on Qwen),
   PEFT filters the state dict to adapter keys: the directory contains
   ``adapter_model.bin`` + ``adapter_config.json`` (+ README.md). NOTE: plain
   (non-LoRA-target) modules such as ``vision_proj`` and TRM layers are NOT in
   this layout — known canonical artifact limitation, see AUDIT_LEDGER.md.

2. ``checkpoint-epoch-N/`` and ``checkpoint-final/`` and ``checkpoint-debug/`` —
   full-state saves via ``accelerator.save_state(ckpt_dir)``. Non-DeepSpeed
   layout (verified empirically on accelerate 1.10.1): ``model.safetensors``
   (full wrapped state dict, PEFT ``base_model.model.*`` keys included),
   ``optimizer.bin``, ``scheduler.bin``, ``random_states_0.pkl``.

3. Loose ``.pt`` files (alignment checkpoints like
   ``notebooks/checkpoints/pixmo_alignment/checkpoint_best.pt``): a dict with a
   ``model_state_dict`` key (possibly plus config/epoch metadata).

``load_vlm_checkpoint`` resolves a checkpoint path (file or directory) and
applies weights onto a ``QwenVLM`` (or any nn.Module) with ``strict=False``,
returning the load result for logging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
ADAPTER_FILES = ("adapter_model.bin", "adapter_model.safetensors")


def _resolve_pt_file(path: Path) -> torch.Tensor | Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_pt_state_dict(path: Path) -> Tuple[Dict[str, torch.Tensor], str]:
    """Load a .pt checkpoint file, unwrapping the 'model_state_dict' key if present."""
    payload = _resolve_pt_file(path)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], "pt(model_state_dict)"
    if isinstance(payload, dict):
        return payload, "pt(raw)"
    raise ValueError(f"Checkpoint file {path} does not contain a state dict")


def load_vlm_checkpoint(model, checkpoint_path: str, strict: bool = False) -> Tuple[Tuple, str]:
    """Load a VLM checkpoint produced by the canonical training path onto ``model``.

    Handles, in order of preference:
      1. Directory with ``pytorch_model.bin`` / ``model.safetensors``
         (full-state accelerate save, incl. LoRA-wrapped keys) — applied via
         accelerate's ``load_checkpoint_in_model`` (handles shards/offload),
         falling back to a plain ``load_state_dict`` if accelerate is unavailable.
      2. Directory with ``adapter_model.bin`` / ``adapter_model.safetensors``
         (PEFT weight-only save) — REJECTED when used as a full VLM checkpoint:
         PEFT ``save_pretrained`` stores only LoRA adapter keys, so non-PEFT
         QwenVLM components (``vision_proj``, TRM/refinement layers, ``z_init``)
         are absent and would be left randomly initialized. Loading raises a
         clear ``ValueError`` instead of silently proceeding.
      3. ``.pt`` file — unwrapped and applied with ``strict=False``.

    Args:
        model: nn.Module to load into (typically a ``QwenVLM``).
        checkpoint_path: path to a checkpoint file or directory.
        strict: passed through to ``load_state_dict`` (default False; matches
            all retained consumers).

    Returns:
        ((missing_keys, unexpected_keys), how) describing what was loaded.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.is_dir():
        # 1. Full-state layout (accelerate save_state / HF save)
        weight_file = next((f for f in WEIGHT_FILES if (path / f).is_file()), None)
        if weight_file is not None:
            try:
                from accelerate.utils import load_checkpoint_in_model

                load_checkpoint_in_model(model, str(path / weight_file), strict=strict)
                # load_checkpoint_in_model does not return missing/unexpected; report via state_dict diff
                return _diff_state_dict(model, str(path / weight_file)), f"dir:{weight_file}"
            except ImportError:
                if weight_file.endswith(".safetensors"):
                    state_dict = _load_shard(path / weight_file)
                else:
                    state_dict = _load_pt_state_dict(path / weight_file)[0]
                result = model.load_state_dict(state_dict, strict=strict)
                return result, f"dir:{weight_file}"

        # 2. PEFT adapter-only layout (historical checkpoint_best via save_pretrained)
        # INCOMPLETE for full VLM restoration: PEFT filters the state dict to
        # LoRA adapter keys, so vision_proj / TRM / z_init are missing and would
        # stay randomly initialized. Fail loudly instead of silently proceeding.
        adapter_file = next((f for f in ADAPTER_FILES if (path / f).is_file()), None)
        if adapter_file is not None:
            raise ValueError(
                f"Checkpoint {path} contains only PEFT adapter weights "
                f"({adapter_file}) and no full model state. It is INCOMPLETE for "
                "full VLM restoration: non-PEFT components (vision_proj, "
                "TRM/refinement layers, z_init) are absent and would be left "
                "randomly initialized. Use a full-state checkpoint directory "
                "instead (checkpoint-epoch-N/ or checkpoint-final/ containing "
                "model.safetensors, written by accelerator.save_state), or "
                "re-run training — the canonical training path now saves a "
                "full-state checkpoint_best/."
            )

        # 3. Sharded full-state layout (model.safetensors.index.json)
        shards = sorted(path.glob("*.safetensors"))
        index = next(iter(path.glob("*.index.json")), None)
        if shards and index is not None:
            state_dict: Dict[str, torch.Tensor] = {}
            for shard in shards:
                state_dict.update(_load_shard(shard))
            result = model.load_state_dict(state_dict, strict=strict)
            return result, "dir:sharded"

        raise FileNotFoundError(
            f"No recognized checkpoint payload in {path} "
            f"(looked for {WEIGHT_FILES + ADAPTER_FILES} or sharded *.safetensors)"
        )

    # 4. Plain file (alignment-style .pt or direct weight file)
    if path.suffix == ".pt" or path.suffix == ".bin":
        state_dict, how = _load_pt_state_dict(path)
        result = model.load_state_dict(state_dict, strict=strict)
        return result, f"file:{how}"
    if path.suffix == ".safetensors":
        state_dict = _load_shard(path)
        result = model.load_state_dict(state_dict, strict=strict)
        return result, "file:safetensors"

    raise ValueError(f"Unsupported checkpoint path: {path}")


def _load_shard(shard_path: Path) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file

        return load_file(str(shard_path))
    except ImportError:  # pragma: no cover
        raise RuntimeError("safetensors required to load sharded/safetensors checkpoints")


def _diff_state_dict(model, checkpoint_ref: str) -> Tuple:
    """Best-effort missing/unexpected report after load_checkpoint_in_model."""
    try:
        from safetensors.torch import load_file

        if os.path.isdir(checkpoint_ref):
            files = [f for f in os.listdir(checkpoint_ref) if f.endswith((".safetensors", ".bin"))]
        else:
            files = [checkpoint_ref]
        sd: Dict[str, torch.Tensor] = {}
        for f in files:
            if f.endswith(".safetensors"):
                sd.update(load_file(os.path.join(checkpoint_ref, f)))
            elif f.endswith(".bin"):
                payload = _resolve_pt_state_dict_path(os.path.join(checkpoint_ref, f))
                sd.update(payload)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(sd.keys())
        return (sorted(model_keys - ckpt_keys), sorted(ckpt_keys - model_keys))
    except Exception:
        return (["<unavailable>"], ["<unavailable>"])


def _resolve_pt_state_dict_path(p: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload
