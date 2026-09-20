"""Regression tests: device-placement policy in the VLM inference path.

Root cause (see AUDIT_LEDGER.md, device-placement audit): inference called
``model.to(model_device)`` on the FULL QwenVLM — which contains the Qwen
backbone possibly quantized (bitsandbytes), device_map-managed ('auto') and
PEFT-wrapped. HF/bitsandbytes forbid nn.Module.to() on a loaded quantized
model, and moving a device_map-sharded model breaks its dispatch map.
Additionally, QwenVLM.forward()/generate() mutated modules at runtime
(re-assigning self.vision_proj / self.down_proj / self.up_proj, calling
self.trm_layers.to(...), and rewriting self.z_init.data on every call).

The fix: setup-time alignment of ONLY the custom non-HF modules
(`align_custom_modules()`), no backbone movement, and zero module mutation
inside forward()/generate() — tensors only.

These tests use a mock decoder whose .to() RAISES, proving the old path would
crash on a quantized/device-map-managed backbone and the new path succeeds.
No Qwen weights are downloaded.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# trm_qwen_vlm imports `from src.decoders...` -> parent of src/ needed on path
for p in {str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT.parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.trm_qwen_vlm import QwenVLM  # noqa: E402


class _ToForbiddenError(RuntimeError):
    """Raised when the (quantized/device-map-managed) backbone is .to()-moved."""


class _ForbiddenToLinear(torch.nn.Linear):
    """A Linear that rejects device migration — mimics a bnb-quantized /
    device_map-managed module, where HuggingFace/bitsandbytes forbid post-load
    device changes. Overrides _apply (the single funnel for Module.to/_apply)
    so even Module.to's internal dispatch is caught."""

    def to(self, *args, **kwargs):
        raise _ToForbiddenError("module.to() forbidden on quantized/device-mapped model")

    def _apply(self, fn, recurse=True):
        raise _ToForbiddenError("device migration forbidden on quantized/device-mapped model")


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class _QuantizedLikeDecoder(torch.nn.Module):
    """Mock QwenDecoder: backbone with a .to() that raises, plus a frozen dtype.
    The backbone's forward returns a minimal CausalLM-like namespace so the real
    QwenVLM.forward runs end-to-end (baseline mode; no TRM refinement asserted
    on real hidden states beyond shape flow)."""

    def __init__(self):
        super().__init__()
        self.model = _ForbiddenToLinear(8, 8)  # the "Qwen" backbone
        self.model.dtype = torch.bfloat16  # attribute, not a .to() call
        self.model.lm_head = _ForbiddenToLinear(8, 8)  # head, same migration guard

    tokenizer = _StubTokenizer()  # pad/eos ids used by generate's mask logic

    @property
    def hidden_dim(self):
        return 8

    @property
    def vocab_size(self):
        return 8

    def forward(self, input_ids=None, attention_mask=None, prefix_embeds=None,
                labels=None, output_hidden_states=False, return_dict=True, **kwargs):
        from types import SimpleNamespace
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, 8, dtype=self.model.dtype)
        logits = torch.zeros(batch, seq, self.vocab_size, dtype=self.model.dtype)
        hs = (hidden,) if output_hidden_states else None
        return SimpleNamespace(loss=None, logits=logits,
                               hidden_states=hs if output_hidden_states else None)


def _build_vlm(use_trm: bool) -> QwenVLM:
    """Real QwenVLM around the forbidden-to backbone — full __init__, no weights."""
    from decoders.trm import TRMConfig

    trm_cfg = TRMConfig(hidden_dim=8, num_layers=2, num_heads=2, max_seq_len=64)
    return QwenVLM(
        qwen_decoder=_QuantizedLikeDecoder(),
        vision_token_dim=8,
        use_trm_recursion=use_trm,
        trm_config=trm_cfg,
        num_recursion_steps=1,
    )


def test_align_custom_modules_never_touches_backbone():
    """align_custom_modules moves ONLY custom modules; the backbone's .to()
    (which raises for quantized/device-map models) is never called."""
    vlm = _build_vlm(use_trm=True)
    # All custom modules start on CPU; backbone is 'quantized-like' on CPU too.
    vlm.align_custom_modules()  # must not raise
    assert vlm.vision_proj.weight.dtype == torch.bfloat16
    # down_proj/up_proj are Identity when dims match (no parameters to check)
    assert isinstance(vlm.down_proj, torch.nn.Identity)
    assert isinstance(vlm.up_proj, torch.nn.Identity)
    assert vlm.z_init.dtype == torch.bfloat16
    for layer in vlm.trm_layers:
        for p in layer.parameters():
            assert p.dtype == torch.bfloat16


def test_old_full_model_to_would_crash_on_quantized():
    """Proves the OLD inference behavior (model.to(device)) crashes against a
    quantized/device-map-managed backbone — this is why it was removed."""
    vlm = _build_vlm(use_trm=False)
    try:
        vlm.to("cpu")  # old inference_vlm.py line 115
        raise AssertionError("expected _ToForbiddenError from full-model .to()")
    except _ToForbiddenError:
        pass
    except Exception as e:
        raise AssertionError(f"unexpected error type from full-model .to(): {e!r}")


def test_forward_and_generate_mutate_no_modules():
    """forward() and generate() must not call .to() on ANY module (old code
    re-assigned self.vision_proj and called self.trm_layers.to() mid-forward).
    Instrument every submodule's .to to raise; run both paths; expect success."""
    vlm = _build_vlm(use_trm=True)
    vlm.align_custom_modules()
    vlm.eval()

    calls = {"n": 0}
    original_to = torch.nn.Module.to

    def counting_to(self, *args, **kwargs):
        calls["n"] += 1
        return original_to(self, *args, **kwargs)

    import models.trm_qwen_vlm as tqv

    # patch Module.to globally during the call and assert ZERO module .to()s
    # from forward/generate (tensor .to() is Tensor.to, untouched)
    torch.nn.Module.to = counting_to
    try:
        question_ids = torch.tensor([[2, 3, 4]])
        vision_tokens = torch.randn(1, 1, 8)
        with torch.no_grad():
            # full forward path with TRM recursion (exercises vision_proj + TRM
            # module-touching sections of the old code)
            vlm(vision_tokens=vision_tokens, question_ids=question_ids)
            # generation path (baseline branch also touched vision_proj before)
            vlm.generate(vision_tokens, question_ids, max_new_tokens=1)
    finally:
        torch.nn.Module.to = original_to
    assert calls["n"] == 0, (
        f"forward/generate mutated modules: {calls['n']} Module.to() calls"
    )


def test_custom_module_tensor_placement_consistent():
    """vision_proj / TRM tensors and forward intermediates must all live on the
    backbone device after setup-time alignment; forward output must flow."""
    vlm = _build_vlm(use_trm=True)
    vlm.align_custom_modules()
    backbone_dev = next(vlm.qwen.parameters()).device
    assert vlm.vision_proj.weight.device == backbone_dev
    assert vlm.z_init.device == backbone_dev

    question_ids = torch.tensor([[2, 3, 4]])
    vision_tokens = torch.randn(1, 1, 8)
    with torch.no_grad():
        out = vlm(vision_tokens=vision_tokens, question_ids=question_ids)
    assert out.logits is not None  # forward completes end-to-end


def test_generate_baseline_path_tensor_only():
    """Baseline generate: mocked decoder generate sees prefix embeds already in
    the backbone dtype; no module .to() attempted on the backbone."""
    vlm = _build_vlm(use_trm=False)
    vlm.align_custom_modules()

    captured = {}

    def fake_generate(input_ids=None, prefix_embeds=None, **kwargs):
        captured["dtype"] = prefix_embeds.dtype
        return torch.tensor([[5, 6, 7]])

    vlm.qwen.generate = fake_generate
    question_ids = torch.tensor([[2, 3, 4]])
    vision_tokens = torch.randn(1, 1, 8)
    out = vlm.generate(vision_tokens, question_ids, max_new_tokens=3)
    assert out.tolist() == [[5, 6, 7]]
    assert captured["dtype"] == torch.bfloat16, captured["dtype"]


if __name__ == "__main__":
    test_align_custom_modules_never_touches_backbone()
    print("PASS align_custom_modules_never_touches_backbone")
    test_old_full_model_to_would_crash_on_quantized()
    print("PASS old_full_model_to_would_crash_on_quantized")
    test_forward_and_generate_mutate_no_modules()
    print("PASS forward_and_generate_mutate_no_modules")
    test_custom_module_tensor_placement_consistent()
    print("PASS custom_module_tensor_placement_consistent")
    test_generate_baseline_path_tensor_only()
    print("PASS generate_baseline_path_tensor_only")
    print("ALL DEVICE-PLACEMENT REGRESSION TESTS PASSED")
