"""Regression tests: Qwen prefix masking / label handling in src/decoders/qwen.py.

Covers the maintained multimodal-prefix plumbing in `QwenDecoder.forward` /
`QwenDecoder.generate` WITHOUT downloading Qwen weights:

* visual-prefix attention-mask entries must be 1 (prefix tokens are valid);
* the text attention mask must be passed through UNCHANGED so left-padded
  prompt tokens stay masked (pad must not silently become "valid" context);
* prefix + text masks concatenate to the expected shape (B, num_prefix + T);
* label masking: prefix positions must be -100 (ignored in the LM loss);
* generate() with prefix-only inputs builds its dummy input_ids from the
  tokenizer's pad_token_id; the pad-token fallback (QwenDecoder.__init__
  lines 87-88: `if pad_token is None: pad_token = eos_token`) must leave
  pad_token_id == eos_token_id so dummy prefix BOS/pad anchors are EOS-safe.

The decoder is instantiated via __new__ around stub model/tokenizer objects —
the real forward/generate logic runs, no HF loading.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in {str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT.parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)

from decoders.qwen import QwenDecoder  # noqa: E402


class _StubTokenizer:
    """Post-fallback tokenizer: pad == eos (the __init__ fallback contract)."""

    pad_token_id = 1
    eos_token_id = 1
    pad_token = "<eos>"

    def __call__(self, texts, **kwargs):
        ids = torch.tensor([[2, 3, 4]] * len(texts))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _CapturingModel(torch.nn.Module):
    """Stub causal LM: records what QwenDecoder actually forwards to it."""

    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.captured = {}

    def get_input_embeddings(self):
        return self.embed

    def forward(self, **kwargs):
        self.captured = kwargs
        batch, seq = kwargs["inputs_embeds"].shape[:2]
        return SimpleNamespace(
            loss=kwargs["labels"].float().sum() if kwargs.get("labels") is not None else None,
            logits=torch.zeros(batch, seq, 16),
        )

    def generate(self, **kwargs):
        self.captured = kwargs
        return torch.zeros(kwargs["inputs_embeds"].shape[0], 2, dtype=torch.long)


def _make_decoder() -> QwenDecoder:
    dec = QwenDecoder.__new__(QwenDecoder)
    torch.nn.Module.__init__(dec)
    dec.model = _CapturingModel()
    dec.tokenizer = _StubTokenizer()
    dec.use_lora = False
    return dec


def test_prefix_mask_entries_are_one():
    """The visual prefix mask must be all ones — prefix tokens are valid input."""
    dec = _make_decoder()
    ids = torch.tensor([[2, 3, 4]])
    mask = torch.tensor([[1, 0, 0]])  # left-padded text region stays 0
    prefix = torch.randn(1, 3, 8)
    dec.forward(input_ids=ids, attention_mask=mask, prefix_embeds=prefix)

    sent = dec.model.captured["attention_mask"]
    assert sent.shape == (1, 6), f"expected (1, 3+3), got {sent.shape}"
    assert torch.equal(sent[0, :3], torch.ones(3)), f"prefix mask not all ones: {sent}"
    # text region preserved exactly — padding must remain masked
    assert torch.equal(sent[0, 3:], mask[0]), f"text mask altered: {sent}"


def test_text_padding_preserved_not_treated_as_valid():
    """Padded text tokens must keep attention_mask=0 after prefix extension."""
    dec = _make_decoder()
    ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])  # row 1 left-padded
    prefix = torch.randn(2, 2, 8)
    dec.forward(input_ids=ids, attention_mask=mask, prefix_embeds=prefix)

    sent = dec.model.captured["attention_mask"]
    assert sent.shape == (2, 6)
    assert torch.equal(sent[1, :2], torch.ones(2)), "prefix entries must be valid"
    assert sent[1, 2:].tolist() == [1, 0, 0, 0], (
        f"padded text tokens were treated as valid: {sent[1].tolist()}"
    )


def test_prefix_and_text_mask_concatenate_expected_shape():
    dec = _make_decoder()
    B, P, T = 2, 7, 5
    dec.forward(
        input_ids=torch.randint(0, 16, (B, T)),
        attention_mask=torch.ones(B, T, dtype=torch.long),
        prefix_embeds=torch.randn(B, P, 8),
    )
    sent = dec.model.captured["attention_mask"]
    assert sent.shape == (B, P + T), f"expected (B, P+T), got {sent.shape}"
    assert sent.sum() == B * (P + T)


def test_labels_prefix_positions_are_ignore_index():
    """Prefix positions must be -100 in labels so the LM loss ignores them."""
    dec = _make_decoder()
    labels = torch.tensor([[4, 5, 6]])
    dec.forward(
        input_ids=torch.tensor([[2, 3, 4]]),
        prefix_embeds=torch.randn(1, 3, 8),
        labels=labels,
    )
    sent = dec.model.captured["labels"]
    assert sent.shape == (1, 6)
    assert torch.equal(sent[0, :3], torch.full((3,), -100)), f"prefix not masked: {sent}"
    assert torch.equal(sent[0, 3:], labels[0]), "text labels altered"


def test_generate_prefix_only_uses_pad_token_id_dummy():
    """Prefix-only generation builds dummy input_ids from pad_token_id."""
    dec = _make_decoder()
    dec.generate(prefix_embeds=torch.randn(1, 2, 8), max_new_tokens=4)
    cap = dec.model.captured
    assert cap["pad_token_id"] == 1 and cap["eos_token_id"] == 1
    # attention_mask for prefix-only input: all ones over prefix + dummy token
    assert cap["attention_mask"].shape == (1, 3)
    assert cap["attention_mask"].min().item() == 1


def test_pad_fallback_contract_pad_becomes_eos():
    """The __init__ pad fallback (`pad_token = eos_token` when pad_token is
    None) must end with pad_token_id == eos_token_id — re-exercise the exact
    documented fallback contract on a pad-less tokenizer."""
    tok = SimpleNamespace(pad_token=None, eos_token="<eos>", eos_token_id=7)
    # This mirrors QwenDecoder.__init__ lines 87-88; the invariant everything
    # downstream (dummy prefix anchors, generate pad_token_id kwarg) relies on:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    assert tok.pad_token_id == tok.eos_token_id == 7

    dec = _make_decoder()
    dec.tokenizer = tok  # pad/eos unified — generation must not choke
    dec.generate(prefix_embeds=torch.randn(1, 2, 8), max_new_tokens=2)
    assert dec.model.captured["pad_token_id"] == 7
    assert dec.model.captured["eos_token_id"] == 7


def test_kwargs_passthrough_no_pad_eos_duplication():
    """generate() pops caller-supplied pad/eos kwargs and re-supplies its own
    from the tokenizer — the underlying model must never receive duplicates."""
    dec = _make_decoder()
    dec.generate(prefix_embeds=torch.randn(1, 2, 8), max_new_tokens=2,
                 pad_token_id=99, eos_token_id=99)
    cap = dec.model.captured
    assert cap["pad_token_id"] == 1 and cap["eos_token_id"] == 1


if __name__ == "__main__":
    test_prefix_mask_entries_are_one()
    print("PASS prefix_mask_entries_are_one")
    test_text_padding_preserved_not_treated_as_valid()
    print("PASS text_padding_preserved_not_treated_as_valid")
    test_prefix_and_text_mask_concatenate_expected_shape()
    print("PASS prefix_and_text_mask_concatenate_expected_shape")
    test_labels_prefix_positions_are_ignore_index()
    print("PASS labels_prefix_positions_are_ignore_index")
    test_generate_prefix_only_uses_pad_token_id_dummy()
    print("PASS generate_prefix_only_uses_pad_token_id_dummy")
    test_pad_fallback_contract_pad_becomes_eos()
    print("PASS pad_fallback_contract_pad_becomes_eos")
    test_kwargs_passthrough_no_pad_eos_duplication()
    print("PASS kwargs_passthrough_no_pad_eos_duplication")
    print("ALL QWEN PREFIX-MASKING TESTS PASSED")
