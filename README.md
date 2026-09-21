# Project CERBERUS: Edge Glass Assistant

A modular multimodal / VLM **research** repository (alignment of frozen encoders, multi-scale representations, and a Qwen VLM path with TRM-inspired iterative refinement). This is a research prototype, not a production or proven tri-modal system.

The maintained code is [`edge_glass_modular/`](edge_glass_modular/README.md) — see that README for the current architecture, install/train/eval/test instructions, and result provenance. Provenance decisions for every retained result live in [`AUDIT_LEDGER.md`](AUDIT_LEDGER.md).

---

## Key resources

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Maintained code** | The active modular codebase (canonical alignment path, Perceiver, Qwen VLM with TRM-inspired iterative refinement, retained experimental audio). | [`edge_glass_modular/`](edge_glass_modular/README.md) |
| **Provenance ledger** | Authoritative record of provenance decisions for all retained results. | [`AUDIT_LEDGER.md`](AUDIT_LEDGER.md) |
| **Resume context** | Externally safe claims, verified metrics, and claims to avoid. | [`PROJECT_RESUME_CONTEXT.md`](PROJECT_RESUME_CONTEXT.md) |
| **Research paper** | Historical project paper (claims in it are not authoritative over the audit). | [`CEREBRUS.pdf`](CEREBRUS.pdf) |
| **Retained results** | Curated alignment / training-history / EDA artifacts. | [`edge_glass_modular/results/`](edge_glass_modular/results/) |

---

## Repository layout

- `edge_glass_modular/` — the maintained package (see its README).
- `older_code_base/` (untracked) and the archive branch `archive/historical-code-pre-retirement` — historical material, not the current implementation. Retired subsystems (standalone TRM decoder, fusion, Path C / `MultimodalProjector`) live only there.
- `requirements.txt` / `requirements.lock.txt` — exact-reproduction environment lock.

---

## Quick start

```bash
cd edge_glass_modular
pip install -e .          # validated compatibility ranges (see pyproject.toml)
pytest tests/             # 48-test regression suite
# For exact reproduction: pip install -r ../requirements.txt
```

See **[edge_glass_modular/README.md](edge_glass_modular/README.md)** for training, evaluation, inference, and checkpoint semantics.

---

## Project highlights

- **Canonical alignment path**: frozen CLIP ViT-L/14-336 + MPNet → Linear → SimpleAttentionPooling → L2-normalized 4096-d embeddings with MRL prefix representations (2048→128).
- **Retained paired-retrieval results**: 77.7% I2T / 77.5% T2I Recall@5 at 4096-d on a persisted PixMo validation evaluation, with a 4096→128 multiscale ablation (see `results/alignment/alignment_eval_final/`; provenance in `AUDIT_LEDGER.md`).
- **Perceiver (experimental)**: packaged implementation canonicalized to the executed V3 training architecture; the historical Perceiver metric is provenance-compromised pending a controlled rerun.
- **Qwen VLM path**: aligned visual tokens → vision_proj → Qwen2.5-3B (LoRA) → TRM-inspired iterative refinement (fixed 4 iterations, shared TRMLayers) → LM head.
- **48-test regression suite** covering generation, refinement EOS, checkpoint loading, device placement, prefix masking, alignment losses, config parsing, and checkpoint semantics.
