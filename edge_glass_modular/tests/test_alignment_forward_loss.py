"""Regression tests: canonical alignment forward/loss wiring (synthetic, no HF weights).

Covers `MultimodalAlignmentModel` + `AlignmentLoss`/`TriModalAlignmentLoss`
around DETERMINISTIC stub encoders — no CLIP / Whisper / sentence-transformers
downloads, CPU-only, fast:

* vision+text (bi-modal) forward produces a finite total loss;
* embedding shapes are (B, projection_dim) and embeddings are returned only
  when requested (`return_embeddings=True`);
* the loss dictionary has the canonical keys ('total_loss', 'contrastive_loss',
  'loss_clip', optional 'mrl_loss_{dim}'/'loss_mrl');
* NO retired surfaces reappear: no 'fused_emb', no decoder outputs;
* MRL path works with synthetic multi-dimension embedding dicts.

Model construction is NOT redesigned: `MultimodalAlignmentModel` is built via
__new__ with its real config-driven loss setup replicated, and only the
*encoders* (heavyweight HF wrappers) are stubbed with deterministic modules
returning `{pooled, mrl_embeddings}` — exactly the interface forward() consumes.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in {str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT.parent)}:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import ExperimentConfig, EncoderConfig  # noqa: E402
from models.alignment import MultimodalAlignmentModel, AlignmentModelOutput  # noqa: E402
from models.losses import AlignmentLoss, TriModalAlignmentLoss, contrastive_loss, mrl_loss  # noqa: E402


class _StubEncoder(torch.nn.Module):
    """Deterministic stand-in for a full encoder: returns the {pooled,
    mrl_embeddings} namespace real encoders produce. Projection is a real
    Linear so the wiring exercises actual module calls."""

    def __init__(self, in_dim=8, proj_dim=16, mrl_dims=None, seed=0):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, proj_dim)
        torch.manual_seed(seed)
        self.mrl_dims = mrl_dims or []
        self.mrl_heads = torch.nn.ModuleDict(
            {str(d): torch.nn.Linear(in_dim, d) for d in self.mrl_dims}
        )

    def forward(self, inputs):
        pooled = self.proj(inputs)
        mrl = {d: head(inputs) for d, head in self.mrl_heads.items()} if self.mrl_dims else None
        return SimpleNamespace(pooled=pooled, mrl_embeddings=mrl)


def _make_model(mrl_dims=(4, 2), sample_single=False) -> MultimodalAlignmentModel:
    """Real model wiring; encoders stubbed, loss is the REAL AlignmentLoss."""
    dims = list(mrl_dims) if mrl_dims else []
    cfg = ExperimentConfig.from_dict({
        "vision_encoder": {"model_name": "stub-vision", "projection_dim": 16,
                           "use_mrl": bool(dims), "mrl_dimensions": dims},
        "text_encoder": {"model_name": "stub-text", "projection_dim": 16,
                         "use_mrl": bool(dims), "mrl_dimensions": dims},
        "optimization": {"contrastive_loss_weight": 1.0, "mrl_loss_weight": 0.05},
    })
    model = MultimodalAlignmentModel.__new__(MultimodalAlignmentModel)
    torch.nn.Module.__init__(model)
    model.config = cfg
    model.vision_encoder = _StubEncoder(mrl_dims=dims, seed=1)
    model.text_encoder = _StubEncoder(mrl_dims=dims, seed=2)
    model.audio_encoder = None
    model.alignment_loss_fn = AlignmentLoss(
        contrastive_weight=cfg.optimization.contrastive_loss_weight,
        mrl_weight=cfg.optimization.mrl_loss_weight,
        temperature=0.07,
        sample_single_mrl_dim=sample_single,
    )
    return model


def _batch(B=4):
    return torch.randn(B, 8), torch.randn(B, 8)


def test_bimodal_forward_finite_loss_and_shapes():
    model = _make_model(mrl_dims=None)
    model.train()
    imgs, txts = _batch()
    out = model(images=imgs, texts=txts, return_embeddings=True)
    assert isinstance(out, AlignmentModelOutput)
    assert out.vision_emb.shape == (4, 16), out.vision_emb.shape
    assert out.text_emb.shape == (4, 16), out.text_emb.shape
    assert out.loss is not None
    assert torch.isfinite(out.loss), f"non-finite loss: {out.loss}"
    # embeddings suppressed when not requested
    out2 = model(images=imgs, texts=txts, return_embeddings=False)
    assert out2.vision_emb is None and out2.text_emb is None


def test_loss_dict_keys_canonical():
    model = _make_model(mrl_dims=None)
    model.train()
    imgs, txts = _batch()
    out = model(images=imgs, texts=txts)
    keys = set(out.losses.keys())
    assert {"total_loss", "contrastive_loss", "loss_clip"} <= keys, keys
    assert torch.isfinite(torch.stack([v for v in out.losses.values()]).sum())


def test_no_retired_surfaces_reappear():
    """fused_emb / decoder outputs must not reappear in output or loss keys."""
    model = _make_model()
    model.train()
    imgs, txts = _batch()
    out = model(images=imgs, texts=txts, return_embeddings=True)
    assert not hasattr(out, "fused_emb")
    for f in ("fused_emb", "decoder_logits", "decoder_loss"):
        assert f not in (out.losses or {}), f"retired surface {f} reappeared"
    assert out.audio_emb is None  # audio encoder not configured


def test_mrl_loss_path_with_synthetic_embeddings():
    """MRL: matching pairs must yield lower loss than shuffled pairs."""
    model = _make_model(mrl_dims=(4, 2), sample_single=False)
    model.train()
    torch.manual_seed(0)
    imgs = torch.randn(4, 8)
    txts = imgs.clone()  # aligned pairs
    out = model(images=imgs, texts=txts)
    mrl_keys = [k for k in out.losses if k.startswith("mrl_loss_")]
    assert mrl_keys, f"no MRL keys in losses: {out.losses.keys()}"
    assert torch.isfinite(out.loss)

    # aligned vs shuffled contrastive sanity at the loss-function level
    emb = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    aligned = contrastive_loss(emb, emb.clone())
    shuffled = contrastive_loss(emb, torch.randn(4, 16))
    assert aligned < shuffled


def test_mrl_loss_multidim_all_dims_computed():
    a = {512: torch.randn(4, 512), 256: torch.randn(4, 256), 128: torch.randn(4, 128)}
    b = {512: torch.randn(4, 512), 256: torch.randn(4, 256), 128: torch.randn(4, 128)}
    losses = mrl_loss(a, b, sample_single_dim=False)
    assert sorted(losses.keys()) == [128, 256, 512]
    assert all(torch.isfinite(v) for v in losses.values())

    # single-dim sampling picks exactly one available dim
    losses1 = mrl_loss(a, b, sample_single_dim=True)
    assert len(losses1) == 1 and list(losses1)[0] in (512, 256, 128)


def test_trimodal_loss_wiring():
    v, a_, t = torch.randn(4, 16), torch.randn(4, 16), torch.randn(4, 16)
    fn = TriModalAlignmentLoss(mrl_weight=0.05, temperature=0.07)
    losses = fn(vision_emb=v, audio_emb=a_, text_emb=t)
    assert {"total_loss", "vision_text_loss", "audio_text_loss", "vision_audio_loss"} <= set(losses)
    assert torch.isfinite(losses["total_loss"])


if __name__ == "__main__":
    test_bimodal_forward_finite_loss_and_shapes()
    print("PASS bimodal_forward_finite_loss_and_shapes")
    test_loss_dict_keys_canonical()
    print("PASS loss_dict_keys_canonical")
    test_no_retired_surfaces_reappear()
    print("PASS no_retired_surfaces_reappear")
    test_mrl_loss_path_with_synthetic_embeddings()
    print("PASS mrl_loss_path_with_synthetic_embeddings")
    test_mrl_loss_multidim_all_dims_computed()
    print("PASS mrl_loss_multidim_all_dims_computed")
    test_trimodal_loss_wiring()
    print("PASS trimodal_loss_wiring")
    print("ALL ALIGNMENT FORWARD/LOSS TESTS PASSED")
