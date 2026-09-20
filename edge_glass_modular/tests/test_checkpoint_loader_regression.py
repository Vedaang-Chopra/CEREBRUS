"""Regression test: checkpoint loading in inference_vlm.py must handle the canonical
training-produced checkpoint layouts.

Regression for the checkpoint-format mismatch recorded in AUDIT_LEDGER.md:
training (train_vlm_accelerate.py) saves DIRECTORY checkpoints
(``checkpoint_best/`` via PEFT save_pretrained -> adapter_model.bin;
``checkpoint-epoch-N/``, ``checkpoint-final/`` via accelerate.save_state ->
model.safetensors), while the old inference loader called ``torch.load(path)``
on the path directly — raising IsADirectoryError for every checkpoint the
canonical training path produces.

This test builds SYNTHETIC checkpoint directories with the same file
names/layouts as the canonical artifacts (verified empirically against the
installed accelerate 1.10.1 + peft), and asserts the loader restores weights.
No Qwen weights are downloaded.
"""

import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in {str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT.parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.checkpoint_loader import load_vlm_checkpoint  # noqa: E402


class _StubQwenVLM(torch.nn.Module):
    """Mimics QwenVLM key layout: vision_proj + a PEFT-style qwen submodule prefix."""

    def __init__(self):
        super().__init__()
        self.vision_proj = torch.nn.Linear(4, 4)
        self.qwen_model = torch.nn.Linear(4, 4)

    # emulate the wrapped PEFT prefix naming used by the canonical checkpoints
    def wrapped_state_dict(self):
        sd = self.state_dict()
        return {f"qwen.model.base_model.model.{k}": v for k, v in sd.items() if k.startswith("qwen_model.")}


def test_loads_accelerate_full_state_dir():
    """checkpoint-epoch-N layout: dir with model.safetensors (full wrapped state)."""
    torch.manual_seed(0)
    model = _StubQwenVLM()
    ref = _StubQwenVLM()
    with torch.no_grad():
        ref.vision_proj.weight.fill_(3.0)

    # build the synthetic accelerate-style checkpoint: keys as accelerate saves them
    full_sd = {**ref.state_dict(), **{f"qwen.model.{k}": v.clone() for k, v in ref.state_dict().items() if k.startswith("qwen_model.")}}
    d = tempfile.mkdtemp(prefix="ckpt-epoch-")
    from safetensors.torch import save_file
    save_file(full_sd, str(Path(d) / "model.safetensors"))
    (Path(d) / "optimizer.bin").touch()      # completeness markers, same as real save_state
    (Path(d) / "scheduler.bin").touch()

    before = model.vision_proj.weight.mean().item()
    (missing, unexpected), how = load_vlm_checkpoint(model, d, strict=False)
    after = model.vision_proj.weight.mean().item()

    assert how == "dir:model.safetensors", how
    assert abs(after - 3.0) < 1e-6, f"weights not restored: {before} -> {after}"
    # keys present in checkpoint but not in model are 'unexpected' — loader must not crash
    assert any("qwen.model." in k for k in unexpected) or unexpected == []


def test_loads_peft_adapter_dir():
    """checkpoint_best layout: dir with adapter_model.bin (LoRA keys only)."""
    torch.manual_seed(0)
    model = _StubQwenVLM()
    ref = _StubQwenVLM()

    adapter_sd = {
        "qwen.model.base_model.model.qwen_model.lora_A.weight": torch.ones(2, 4),
        "qwen.model.base_model.model.qwen_model.lora_B.weight": torch.full((4, 2), 2.0),
    }
    d = tempfile.mkdtemp(prefix="ckpt-best-")
    torch.save(adapter_sd, Path(d) / "adapter_model.bin")
    (Path(d) / "adapter_config.json").touch()
    (Path(d) / "README.md").touch()

    (missing, unexpected), how = load_vlm_checkpoint(model, d, strict=False)
    assert how == "dir:adapter_model.bin", how
    # strict=False: adapter keys not in stub model land in 'unexpected' without raising
    assert len(unexpected) == 2, unexpected


def test_loads_pt_file_with_model_state_dict():
    """Alignment-style .pt: dict with model_state_dict key."""
    torch.manual_seed(0)
    model = _StubQwenVLM()
    d = tempfile.mkdtemp()
    ref_sd = model.state_dict()
    with torch.no_grad():
        ref_sd["vision_proj.weight"] = torch.full((4, 4), 5.0)
    p = Path(d) / "checkpoint_best.pt"
    torch.save({"model_state_dict": ref_sd, "epoch": 1}, p)

    (missing, unexpected), how = load_vlm_checkpoint(model, str(p), strict=False)
    assert how == "file:pt(model_state_dict)", how
    assert torch.allclose(model.vision_proj.weight, torch.full((4, 4), 5.0))


def test_directory_checkpoint_rejected_by_old_loader():
    """Proves the OLD behavior (torch.load on a directory) fails — the regression."""
    d = tempfile.mkdtemp(prefix="ckpt-epoch-")
    (Path(d) / "model.safetensors").touch()
    try:
        torch.load(d, map_location="cpu")
        raise AssertionError("old loader unexpectedly succeeded on a directory")
    except (IsADirectoryError, OSError, RuntimeError, ValueError):
        pass  # expected under the old loader


def test_missing_checkpoint_raises():
    try:
        load_vlm_checkpoint(_StubQwenVLM(), "/nonexistent/ckpt", strict=False)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    test_loads_accelerate_full_state_dir()
    print("PASS loads_accelerate_full_state_dir")
    test_loads_peft_adapter_dir()
    print("PASS loads_peft_adapter_dir")
    test_loads_pt_file_with_model_state_dict()
    print("PASS loads_pt_file_with_model_state_dict")
    test_directory_checkpoint_rejected_by_old_loader()
    print("PASS directory_checkpoint_rejected_by_old_loader")
    test_missing_checkpoint_raises()
    print("PASS missing_checkpoint_raises")
    print("ALL CHECKPOINT-LOADER REGRESSION TESTS PASSED")
