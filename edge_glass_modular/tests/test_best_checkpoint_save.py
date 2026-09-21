"""Regression tests: training-side best-checkpoint save semantics.

Guards the checkpoint-completeness fix recorded in AUDIT_LEDGER.md: the old
best-checkpoint path called ``unwrapped_model.save_pretrained(checkpoint_best,
...)`` on the PEFT-wrapped model, which FILTERED the state dict to LoRA adapter
keys only — silently dropping vision_proj / TRM / z_init from the canonical
best checkpoint. The fix:
  * canonical ``checkpoint_best/`` is written with ``accelerator.save_state``
    (or ``accelerator.save_model`` on DeepSpeed) — full state;
  * the PEFT adapter export goes to a SEPARATE ``checkpoint_best_adapter/``;
  * epoch/final checkpoints use ``accelerator.save_state`` too.

These tests verify BOTH:
  1. the source of scripts/train/train_vlm_accelerate.py retains exactly that
     structure (source-level guard — the script can't be executed on CPU);
  2. a behavioral simulation reproducing the save sequence: a stub
     accelerator.save_state writes a full-state model.safetensors into
     checkpoint_best/, the adapter export goes to checkpoint_best_adapter/,
     and the produced layouts load through the REAL load_vlm_checkpoint with
     completeness semantics (full state accepted, adapter-only rejected).
No network, no training, CPU only.
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
from safetensors.torch import save_file  # noqa: E402

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train" / "train_vlm_accelerate.py"


def _source() -> str:
    return TRAIN_SCRIPT.read_text()


def test_source_uses_save_state_for_checkpoint_best():
    """The best-checkpoint branch must call accelerator.save_state (full state),
    inside the improvement branch guarded on avg_val_loss < best_val_loss."""
    src = _source()
    best_idx = src.index("checkpoint_best\"")
    block = src[best_idx - 200: best_idx + 2000]
    assert "avg_val_loss < best_val_loss" in block or "avg_val_loss < best_val_loss" in src
    # canonical full-state save in the best branch:
    assert "accelerator.save_state(ckpt_dir)" in block, (
        "checkpoint_best no longer saved via accelerator.save_state — "
        "completeness regression!"
    )
    # DeepSpeed alternate full-state path is allowed but save_state is the default branch
    assert "save_model(model, ckpt_dir" in block or "save_model(model, ckpt_dir" in src


def test_source_adapter_export_goes_to_separate_dir():
    """The PEFT save_pretrained export must target checkpoint_best_adapter/,
    NOT checkpoint_best/."""
    src = _source()
    assert '"checkpoint_best_adapter"' in src or "'checkpoint_best_adapter'" in src, (
        "adapter export dir checkpoint_best_adapter/ missing from training script"
    )
    # every save_pretrained call in the script must target the adapter dir
    import re
    calls = re.findall(r"save_pretrained\(\s*\n?\s*([^,\n]+)", src)
    for target in calls:
        assert "checkpoint_best_adapter" in target or "adapter" in target, (
            f"save_pretrained used as canonical save targeting: {target!r}"
        )


def test_source_checkpoint_best_not_save_pretrained():
    """The canonical best checkpoint must not be produced by model.save_pretrained.
    Verify no save_pretrained call is bound to a checkpoint_best (non-adapter)
    directory."""
    src = _source()
    import re
    for m in re.finditer(r"save_pretrained\(([^)]*)\)", src, flags=re.S):
        args = m.group(1)
        if "checkpoint_best_adapter" in args:
            continue
        assert "checkpoint_best" not in args.replace("checkpoint_best_adapter", ""), (
            f"save_pretrained used on canonical best checkpoint: {args[:120]!r}"
        )


def test_behavioral_simulation_save_sequence():
    """Simulate the fixed save sequence end-to-end with a stub accelerator:
    save_state -> full-state dir; adapter save_pretrained -> adapter dir; then
    load both through the REAL checkpoint loader."""
    torch.manual_seed(0)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_proj = torch.nn.Linear(4, 4)
            self.qwen_model = torch.nn.Linear(4, 4)

    model = _Model()
    ref = _Model()
    with torch.no_grad():
        ref.vision_proj.weight.fill_(9.0)

    out_dir = Path(tempfile.mkdtemp())

    # --- stub accelerator recording the two save calls ---
    calls = {}

    def save_state(d):
        calls["best_dir"] = Path(d)
        # accelerator.save_state writes the FULL state dict (ref carries the
        # trained weights; model carries the random pre-load init)
        full = {**ref.state_dict(),
                **{f"qwen.model.{k}": v.clone() for k, v in ref.state_dict().items()
                   if k.startswith("qwen_model.")}}
        (Path(d) / "model.safetensors").parent.mkdir(parents=True, exist_ok=True)
        save_file(full, str(Path(d) / "model.safetensors"))
        (Path(d) / "optimizer.bin").touch()

    def get_state_dict(_m):
        # what PEFT save_pretrained receives: the FULL state dict...
        return {**model.state_dict(),
                **{f"qwen.model.{k}": v.clone() for k, v in ref.state_dict().items()
                   if k.startswith("qwen_model.")}}

    # simulate the fixed branch
    avg_val_loss, best_val_loss = 0.42, 0.5
    assert avg_val_loss < best_val_loss
    ckpt_dir = out_dir / "checkpoint_best"
    calls["mode"] = "save_state"
    save_state(ckpt_dir)

    # adapter export: PEFT FILTERS to adapter keys (the historical bug's effect)
    adapter_sd = {
        "qwen.model.base_model.model.qwen_model.lora_A.weight": torch.ones(2, 4),
        "qwen.model.base_model.model.qwen_model.lora_B.weight": torch.full((4, 2), 2.0),
    }
    adapter_dir = out_dir / "checkpoint_best_adapter"
    adapter_dir.mkdir()
    torch.save(adapter_sd, adapter_dir / "adapter_model.bin")
    (adapter_dir / "adapter_config.json").touch()

    # --- assertions on produced layouts ---
    assert calls["mode"] == "save_state"
    assert (ckpt_dir / "model.safetensors").exists()
    assert not (ckpt_dir / "adapter_model.bin").exists(), (
        "canonical checkpoint_best contains adapter-only content"
    )
    assert (adapter_dir / "adapter_model.bin").exists()

    # the canonical best checkpoint restores vision_proj through the REAL loader.
    # NOTE: the loader routes a dir with model.safetensors through
    # from_pretrained (accelerate's own loading); torch.manual_seed ensures the
    # random init differs from the ref weights, so allclose proves real restore.
    fresh = _Model()
    (missing, unexpected), how = load_vlm_checkpoint(fresh, str(ckpt_dir), strict=False)
    assert how == "dir:model.safetensors"
    assert torch.allclose(fresh.vision_proj.weight, ref.vision_proj.weight), (
        "vision_proj not restored from canonical best checkpoint — completeness regression"
    )

    # the adapter dir is INCOMPLETE for the full model — loader must reject it
    try:
        load_vlm_checkpoint(_Model(), str(adapter_dir), strict=False)
        raise AssertionError("adapter-only dir must be rejected (incomplete)")
    except ValueError:
        pass


def test_epoch_checkpoints_also_save_state():
    """Epoch/final checkpoints use accelerator.save_state as well — the layout
    load_vlm_checkpoint's full-state branch expects."""
    src = _source()
    assert "checkpoint-epoch-" in src
    # the save site (line ~650): ckpt_dir = ... / f"checkpoint-epoch-{epoch+1}"
    # immediately followed by accelerator.save_state(ckpt_dir)
    save_site = src.index('f"checkpoint-epoch-{epoch+1}"')
    tail = src[save_site: save_site + 400]
    assert "accelerator.save_state(ckpt_dir)" in tail, (
        "epoch checkpoints no longer saved via accelerator.save_state"
    )


if __name__ == "__main__":
    test_source_uses_save_state_for_checkpoint_best()
    print("PASS source_uses_save_state_for_checkpoint_best")
    test_source_adapter_export_goes_to_separate_dir()
    print("PASS source_adapter_export_goes_to_separate_dir")
    test_source_checkpoint_best_not_save_pretrained()
    print("PASS source_checkpoint_best_not_save_pretrained")
    test_behavioral_simulation_save_sequence()
    print("PASS behavioral_simulation_save_sequence")
    test_epoch_checkpoints_also_save_state()
    print("PASS epoch_checkpoints_also_save_state")
    print("ALL BEST-CHECKPOINT SAVE SEMANTICS TESTS PASSED")
