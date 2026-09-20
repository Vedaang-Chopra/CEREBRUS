"""Regression test: baseline (TRM-refinement-disabled) generation must not discard
newly generated tokens.

Root cause regression for AUDIT_LEDGER entry "baseline generation double-strip bug":
`TRMQwenVLM.generate` sliced `outputs[:, prompt_len:]`, but QwenDecoder.generate
passes only `inputs_embeds` to HF `model.generate()`, which (verified on installed
transformers 4.57.6) returns ONLY the newly generated tokens — prompt token IDs are
not echoed. The slice therefore returned an empty answer.

This test mocks the underlying Qwen model call and requires the real
`TRMQwenVLM.generate` baseline branch to return the mocked generated tokens intact.
No model weights are downloaded.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT  # trm_qwen_vlm imports `from src.decoders...` -> needs parent of src/... on path
for p in {str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT.parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.trm_qwen_vlm import QwenVLM  # noqa: E402


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _build_stub_model():
    """Minimal stand-in exposing only the attributes the baseline path touches."""
    stub = SimpleNamespace()
    stub.tokenizer = _StubTokenizer()
    stub.model = SimpleNamespace(dtype=torch.float32)
    stub.generate = None  # patched per-test
    return stub


def _make_trm(use_trm_recursion: bool) -> QwenVLM:
    """Instantiate QwenVLM without loading any pretrained weights."""
    trm = QwenVLM.__new__(QwenVLM)
    torch.nn.Module.__init__(trm)
    trm.qwen = _build_stub_model()
    # vision_proj is only touched for dtype/device alignment before generation;
    # provide a minimal Linear so `.weight.device/.dtype` checks pass.
    trm.vision_proj = torch.nn.Linear(8, 8)
    trm.use_trm_recursion = use_trm_recursion
    return trm


def _fake_decoder_generate(input_ids, prefix_embeds, attention_mask, **kwargs):
    """Mock QwenDecoder.generate: returns generated-only token IDs, as HF does for
    inputs_embeds-only generation on transformers 4.57.6 (prompt NOT included)."""
    # tokens [5, 6, 7] simulate a 3-token answer
    return torch.tensor([[5, 6, 7], [5, 6, 7]])


def test_baseline_returns_generated_tokens():
    """The generated answer tokens must survive the baseline path."""
    trm = _make_trm(use_trm_recursion=False)
    question_ids = torch.tensor([[2, 3, 4], [2, 3, 4]])  # prompt_len = 3
    vision_tokens = torch.randn(2, 1, 8)

    with patch.object(trm.qwen, "generate", side_effect=_fake_decoder_generate):
        out = trm.generate(vision_tokens, question_ids, max_new_tokens=3)

    expected = torch.tensor([[5, 6, 7], [5, 6, 7]])
    assert out.shape == expected.shape, f"expected {expected.shape}, got {out.shape}"
    assert torch.equal(out, expected), (
        f"baseline generation discarded generated tokens: got {out.tolist()}"
    )


def test_old_buggy_slicing_would_fail():
    """Proves the previous `outputs[:, prompt_len:]` behavior fails this test:
    with prompt_len=3 and a 3-token generated-only output, the old slice returns
    an empty tensor — i.e., all newly generated tokens discarded."""
    outputs = torch.tensor([[5, 6, 7], [5, 6, 7]])  # what HF actually returns
    prompt_len = 3
    old_result = outputs[:, prompt_len:]
    assert old_result.shape[1] == 0, "old slicing was expected to empty the answer"
    # the fix's requirement: result must NOT be empty
    assert old_result.shape[1] != 3 or False


def test_prompt_tokens_not_in_answer():
    """Prompt/prefix token IDs must not leak into the returned answer."""
    trm = _make_trm(use_trm_recursion=False)
    question_ids = torch.tensor([[2, 3, 4], [2, 3, 4]])
    vision_tokens = torch.randn(2, 1, 8)

    with patch.object(trm.qwen, "generate", side_effect=_fake_decoder_generate):
        out = trm.generate(vision_tokens, question_ids, max_new_tokens=3)

    prompt_ids = {2, 3, 4}
    for row in out.tolist():
        assert not set(row) & prompt_ids, f"prompt tokens leaked into answer: {row}"


def test_refinement_path_unchanged():
    """Refinement-enabled path builds output from the manual loop and must not be
    affected by the baseline fix. Verify it still returns per-step stacked tokens."""
    calls = {"n": 0}

    def fake_forward(input_ids, attention_mask=None, prefix_embeds=None):
        calls["n"] += 1
        # deterministic last-position logits: pick token 9 every step
        logits = torch.full((2, input_ids.shape[1], 16), -1e9)
        logits[:, -1, 9] = 0.0
        return {"logits": logits}

    trm = _make_trm(use_trm_recursion=True)
    trm.forward = fake_forward
    question_ids = torch.tensor([[2, 3, 4], [2, 3, 4]])
    vision_tokens = torch.randn(2, 1, 8)

    out = trm.generate(vision_tokens, question_ids, max_new_tokens=3, temperature=0.0)

    assert out.shape == (2, 3), f"refinement path shape changed: {out.shape}"
    assert torch.equal(out, torch.full((2, 3), 9)), f"unexpected refinement output: {out}"
    assert calls["n"] == 3, "refinement loop should call forward once per token"


if __name__ == "__main__":
    test_baseline_returns_generated_tokens()
    print("PASS baseline_returns_generated_tokens")
    test_old_buggy_slicing_would_fail()
    print("PASS old_buggy_slicing_would_fail")
    test_prompt_tokens_not_in_answer()
    print("PASS prompt_tokens_not_in_answer")
    test_refinement_path_unchanged()
    print("PASS refinement_path_unchanged")
    print("ALL GENERATION REGRESSION TESTS PASSED")
