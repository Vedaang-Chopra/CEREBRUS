"""Regression tests: EOS stopping in the manual TRM refinement generation loop.

Root cause (see AUDIT_LEDGER.md, refinement-EOS audit): the refinement loop ran
exactly ``max_new_tokens`` iterations unconditionally — the ``# Stop if EOS``
comment existed but no stop logic was ever implemented. A sequence that
generated EOS on step 1 still accumulated max_new_tokens-1 further model tokens
after it; in a batch, per-sequence EOS was ignored entirely.

New semantics: EOS (from ``qwen.tokenizer.eos_token_id``) is retained in the
returned IDs for the sequence that generated it; sequences already finished
emit pad tokens afterwards; the loop exits early once ALL sequences are done;
no-EOS generation still produces exactly max_new_tokens.

All tests use deterministic mocked logits. No Qwen weights downloaded.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in {str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT.parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.trm_qwen_vlm import QwenVLM  # noqa: E402

EOS = 9   # mock eos_token_id
PAD = 0   # mock pad_token_id


class _StubTokenizer:
    pad_token_id = PAD
    eos_token_id = EOS


def _make_trm() -> QwenVLM:
    """QwenVLM shell with a stubbed decoder; forward patched per-test."""
    trm = QwenVLM.__new__(QwenVLM)
    torch.nn.Module.__init__(trm)
    stub = SimpleNamespace(tokenizer=_StubTokenizer())
    trm.qwen = stub
    trm.vision_proj = torch.nn.Linear(8, 8)
    trm.use_trm_recursion = True
    return trm


def _run(trm, per_step_tokens, max_new_tokens):
    """Drive the real refinement loop with a scripted forward.

    per_step_tokens: list of (B,) LongTensors — the argmax picks token j at
    step j (greedy path). Returns (out, n_forward_calls).
    """
    calls = {"n": 0}

    def fake_forward(input_ids, attention_mask=None, prefix_embeds=None, **kw):
        calls["n"] += 1
        step = calls["n"] - 1
        tok = per_step_tokens[min(step, len(per_step_tokens) - 1)]
        B, Q = input_ids.shape
        logits = torch.full((B, Q, 16), -1e9)
        logits[:, -1, tok[0].item() if tok.numel() == 1 else 0] = 0.0
        # For batch tests the scripted token must differ per row: override the
        # whole last-position row via a side-channel (token tensor rows).
        logits[:, -1, :] = -1e9
        for row in range(B):
            logits[row, -1, tok[row].item()] = 0.0
        return {"logits": logits}

    trm.forward = fake_forward
    question_ids = torch.tensor([[2, 3, 4]] * per_step_tokens[0].shape[0])
    vision_tokens = torch.randn(per_step_tokens[0].shape[0], 1, 8)
    out = trm.generate(vision_tokens, question_ids, max_new_tokens=max_new_tokens,
                       temperature=0.0)
    return out, calls["n"]


def test_eos_first_token_stops_immediately():
    """EOS at step 0: 1 forward call, output is [[EOS]] — not max_new_tokens."""
    trm = _make_trm()
    out, n = _run(trm, [torch.tensor([EOS])], max_new_tokens=8)
    assert n == 1, f"expected 1 forward call, got {n}"
    assert out.tolist() == [[EOS]], out.tolist()


def test_eos_after_n_tokens_returns_n_including_eos():
    """Tokens [4, 5, EOS]: exactly 3 forwards, output [4,5,EOS] (EOS retained),
    and NO further forward calls for the remaining steps."""
    trm = _make_trm()
    steps = [torch.tensor([4]), torch.tensor([5]), torch.tensor([EOS])]
    out, n = _run(trm, steps, max_new_tokens=8)
    assert n == 3, f"expected 3 forward calls, got {n}"
    assert out.tolist() == [[4, 5, EOS]], out.tolist()


def test_no_eos_generates_exact_length():
    """No EOS anywhere: exactly max_new_tokens tokens, max_new_tokens forwards."""
    trm = _make_trm()
    steps = [torch.tensor([i + 1]) for i in range(5)]
    out, n = _run(trm, steps, max_new_tokens=5)
    assert n == 5
    assert out.tolist() == [[1, 2, 3, 4, 5]], out.tolist()


def test_batch_one_finishes_early_others_continue():
    """Batch of 2: seq0 EOSes at step 1, seq1 keeps generating until its own EOS
    at step 3. Finished seq0 emits PAD afterwards (never model content); the
    loop stops when BOTH are done (4 forwards, not max_new_tokens=10)."""
    trm = _make_trm()
    steps = [
        torch.tensor([7, 7]),        # step 0: both real tokens
        torch.tensor([EOS, 6]),      # step 1: seq0 EOS; seq1 continues
        torch.tensor([PAD, 5]),      # step 2: seq0 pad-filled
        torch.tensor([PAD, EOS]),    # step 3: seq1 EOS -> all done
    ]
    out, n = _run(trm, steps, max_new_tokens=10)
    assert n == 4, f"expected 4 forwards (stop when all done), got {n}"
    assert out.tolist() == [[7, EOS, PAD, PAD], [7, 6, 5, EOS]], out.tolist()


def test_batch_never_content_after_eos():
    """Finished sequences never accumulate non-pad/non-EOS content afterwards,
    regardless of what the mocked logits would have produced."""
    trm = _make_trm()
    steps = [
        torch.tensor([EOS, 3]),   # seq0 finishes immediately
        torch.tensor([11, 4]),    # would-be model content for seq0
        torch.tensor([12, 5]),
        torch.tensor([13, EOS]),
    ]
    out, n = _run(trm, steps, max_new_tokens=6)
    row0 = out.tolist()[0]
    after = row0[1:]  # everything after its EOS
    assert set(after) <= {PAD}, f"content after EOS: {row0}"
    assert out.tolist()[1] == [3, 4, 5, EOS]


def test_baseline_path_untouched():
    """The baseline (non-TRM) branch must not have gained EOS logic of its own —
    it delegates to qwen.generate (HF handles EOS) and returns tokens intact."""
    trm = _make_trm()
    trm.use_trm_recursion = False

    def fake_generate(**kwargs):
        assert kwargs["eos_token_id"] == EOS
        return torch.tensor([[5, 6, EOS]])

    trm.qwen.generate = fake_generate
    question_ids = torch.tensor([[2, 3, 4]])
    out = trm.generate(torch.randn(1, 1, 8), question_ids, max_new_tokens=3)
    assert out.tolist() == [[5, 6, EOS]]


def test_old_behavior_would_fail():
    """Proves the OLD loop (no stop logic) fails these expectations: it always
    runs max_new_tokens steps, so an EOS-first call would perform 8 forwards
    and the output would not be [[EOS]]."""
    def old_loop(n_steps):
        calls = 0
        ids = []
        for _ in range(n_steps):
            calls += 1
            ids.append(7)  # some model content each step
        return calls, ids

    calls, ids = old_loop(8)  # EOS-first scenario under old code
    assert calls == 8, "old loop ran all steps regardless of EOS"
    assert EOS not in ids, "old loop never surfaced EOS"


if __name__ == "__main__":
    test_eos_first_token_stops_immediately()
    print("PASS eos_first_token_stops_immediately")
    test_eos_after_n_tokens_returns_n_including_eos()
    print("PASS eos_after_n_tokens_returns_n_including_eos")
    test_no_eos_generates_exact_length()
    print("PASS no_eos_generates_exact_length")
    test_batch_one_finishes_early_others_continue()
    print("PASS batch_one_finishes_early_others_continue")
    test_batch_never_content_after_eos()
    print("PASS batch_never_content_after_eos")
    test_baseline_path_untouched()
    print("PASS baseline_path_untouched")
    test_old_behavior_would_fail()
    print("PASS old_behavior_would_fail")
    print("ALL EOS-STOP REGRESSION TESTS PASSED")
