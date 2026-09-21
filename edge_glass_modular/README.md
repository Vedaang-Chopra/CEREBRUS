# Edge Glass Modular (CERBERUS)

A modular multimodal / VLM **research** codebase focused on:

- **frozen-encoder alignment** — bridging frozen CLIP vision and frozen sentence-transformer text backbones into one shared, L2-normalized embedding space;
- **compact / multi-scale representations** — Matryoshka-style nested prefix embeddings (4096 → 128 dims) supervised and evaluated at multiple scales;
- **VLM integration** — feeding aligned visual tokens to a LoRA-adapted Qwen2.5 model for image-question answering;
- **TRM-inspired iterative refinement experiments** — an iterative y/z refinement loop over Qwen hidden states, inspired by (but *not* a faithful implementation of) TinyRecursiveModels.

This is a research prototype, not a production or proven tri-modal system. Provenance decisions for every retained result live in the top-level `AUDIT_LEDGER.md`.

---

## Current maintained architecture

### Canonical alignment path (authoritative)

This is the path that produced the authoritative retained alignment results:

```
image → CLIP ViT-L/14-336 (frozen)
      → Linear projection (1024 → 4096)
      → SimpleAttentionPooling (learnable, over patch tokens)
      → L2-normalized aligned embedding
      → MRL prefix representations (2048 / 1024 / 512 / 256 / 128)
```

Text side: `sentence-transformers/all-mpnet-base-v2` (frozen) → Linear projection → L2 norm → the same MRL prefixes. Trained with a symmetric InfoNCE/CLIP loss plus MRL contrastive loss on sampled prefix dimensions (`src/models/alignment.py`, `src/models/losses.py`, `src/encoders/{vision,text,mrl,pooling}.py`; driven by `configs/pixmo_alignment.yaml`).

**The retained paired-retrieval results under `results/alignment/alignment_eval_final/` come from this path and are the authoritative alignment numbers** (see `AUDIT_LEDGER.md` for the provenance chain and qualifications).

### Perceiver (canonical implementation, provenance-compromised historical result)

`src/encoders/perceiver.py` now implements the architecture that the **executed V3 training run** actually used (`scripts/experiments/train_perceiver_v3.py`): per-layer cross-attention with separate query/KV pre-normalization, latent **self-attention**, FFN branch with its own pre-norm, residual connections, final latent LayerNorm, and KV-mask support. This was verified to be a 0.0-diff extract of the training computation under strict weight copy.

Important provenance caveat — **do not quote the historical Perceiver number as an architecture verdict**:

- The historical Perceiver evaluation architecture **differed** from the training architecture.
- The historical checkpoint was loaded with `strict=False`.
- The historical retained Perceiver metric (near-random paired R@5 in `results/alignment/alignment_eval_final/`) is therefore **provenance-compromised**: it is retained as a recorded historical outcome, not as a valid comparison of architectures.
- A **controlled rerun** (matching train/eval architecture, strict loading, fixed text backbone) is required before drawing any Perceiver-vs-attention-pooling conclusion.

The historical number itself is preserved unchanged in the retained CSVs.

### VLM path (Qwen2.5-3B + TRM-inspired iterative refinement)

The executed VLM graph:

```
image
  → aligned visual tokens (from the canonical alignment encoder)
  → vision_proj
  → Qwen2.5-3B (LoRA)
  → final hidden states
  → TRM-inspired y/z iterative refinement (shared TRMLayers, fixed 4 iterations)
  → residual add
  → LM head
```

Precisely stated:

- Refinement reuses a **shared stack of `TRMLayers`** (`src/models/trm_qwen_vlm.py`); the loop alternates y-updates and z-updates over the final hidden states for a **fixed 4 iterations**.
- This is **TRM-inspired iterative refinement**, *not* a faithful implementation of [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels): there is no adaptive halting, no standalone `TRMDecoder`, and the recursion depth is fixed rather than learned.
- **No standalone `TRMDecoder` exists anymore** — the standalone decoder path and its config surface were retired. `src/decoders/trm.py` retains only the `TRMLayer` building block and a retirement note.
- No quantitative VLM improvement is claimed: no retained, provenance-clean evaluation artifact supports one.

### Audio (retained experimental support)

Audio was an **early / experimental scope**:

- `AudioEncoder` (frozen Whisper encoder) and the tri-modal loss scaffolding are retained in the tree.
- **No retained experiment demonstrates successful tri-modal alignment or tri-modal VLM training.** Do not infer audio results from the vision-text results.
- The fusion subsystem has been **retired** and is not part of the maintained architecture.

---

## Repository structure

```
edge_glass_modular/
├── src/
│   ├── config.py              # dataclass-based YAML config loading
│   ├── data/                  # parquet datasets, datamodule, transforms, downloader, parquet_builder
│   ├── encoders/              # vision, text, audio, pooling, perceiver, mrl
│   ├── decoders/              # qwen (LoRA/quantized Qwen), trm (TRMLayer building block only)
│   ├── models/                # alignment.py, losses.py, trm_qwen_vlm.py, checkpoint_loader.py
│   ├── training/              # improved_trainer.py (canonical alignment trainer)
│   ├── evaluation/            # retrieval metrics, QA metrics, benchmark/explainability/zero-shot (experimental)
│   └── utils/                 # logging, visualization, registry, distributed
├── scripts/
│   ├── train/                 # train_vlm_accelerate.py, run_vlm_experiments.sh
│   ├── eval/                  # evaluate_vlm_accelerate.py, inference_vlm.py, evaluate_freeze_align.py
│   ├── data/                  # download_datasets.py, download_pixmo_audio.py
│   └── experiments/           # train_pixmo_alignment.py, train_perceiver.py, train_perceiver_v3.py
├── configs/                   # 9 experiment YAMLs
├── tests/                     # 48-test permanent regression suite
├── results/                   # curated retained artifacts (alignment / train_history / eda)
├── notebooks/                 # experiments/, analysis/, archive/, wandb/, system_architecture.md
├── reports/                   # historical report snapshots (see headers inside each file)
└── pyproject.toml             # dependency contract
```

Historical code (v0–v2 codebases, pre-retirement sources, retired subsystems) is preserved on the archive branch
`archive/historical-code-pre-retirement` and under the untracked `older_code_base/` at the repo root. It is **not** part of the maintained code.

---

## Installation / reproducibility

```bash
cd edge_glass_modular
pip install -e .
```

- `pip install -e .` installs dependencies **within the validated compatibility ranges** declared in `pyproject.toml`
  (transformers `>=4.57,<4.58`, peft `>=0.17,<0.19`, accelerate `>=1.10,<1.12`, bitsandbytes `>=0.48,<0.49`,
  torch `>=2.8,<2.10`, sentence-transformers `>=5.1,<5.2`, safetensors, huggingface_hub, etc.).
- For **exact reproduction**, use the repository lock files at the repo root:
  `pip install -r ../requirements.txt` (equivalently `../requirements.lock.txt`).
- Validated core environment: **transformers 4.57.x, PEFT 0.17–0.18, Accelerate 1.10–1.11** (torch 2.9.x).
  Do not perform unconstrained dependency upgrades — an unconstrained `sentence-transformers` install has
  previously forced transformers to 5.x and broken the Qwen generation path.

---

## Testing

```bash
pytest tests/
```

48 tests, CPU-only and network-free. Key regression coverage:

- **generation** (baseline Qwen generation, no prompt-echo / double-strip);
- **refinement EOS** (per-sequence EOS stopping, pad-filling, early exit in the refinement loop);
- **checkpoint loading** (full-state directories, adapter-only rejection, `.pt` unwrapping);
- **device placement** (quantized/device_map/PEFT backbone never moved; custom modules aligned at setup);
- **Qwen prefix masking** (prefix mask/labels semantics in `QwenDecoder.forward/generate`);
- **alignment forward/loss** (bi-modal + MRL + tri-modal wiring, canonical loss keys);
- **config parsing** (all 9 YAMLs load; retired surfaces verified absent);
- **best-checkpoint semantics** (full-state `checkpoint_best/` save contract, adapter-only rejection end to end).

---

## Training

### Alignment

The historically successful canonical alignment training ran through the **`ImprovedMultimodalTrainer` /
notebook workflow** (`src/training/improved_trainer.py`, driven from `notebooks/experiments/01_pixmo_vision_text_alignment.ipynb`).
There is currently **no polished canonical alignment CLI**: the old trainer CLI family was retired, and
`scripts/experiments/train_pixmo_alignment.py` exists as an experiment script, not a maintained entry point.
Reproducing the alignment run today means driving `ImprovedMultimodalTrainer` from a notebook or script of your own.

### VLM (Qwen2.5-3B QA)

The maintained training entry point is the Accelerate script:

```bash
accelerate launch scripts/train/train_vlm_accelerate.py \
    --config configs/trm_vlm_qa_qwen2.5-3b_regularized.yaml \
    --alignment_config configs/pixmo_alignment.yaml \
    --checkpoint_path <path-to-alignment-checkpoint.pt> \
    --output_dir <output-dir> \
    --use_wandb
```

Notes:

- Alignment-checkpoint lookup falls back to candidates relative to the script, plus a hardcoded HPC-cluster path (last-resort candidate, retained intentionally).
- Refinement is enabled by default; `--disable_trm` disables it.
- `scripts/train/run_vlm_experiments.sh` holds the historically used launch invocations (demo / verification / full runs).

### Evaluation / inference

```bash
# QA evaluation
accelerate launch scripts/eval/evaluate_vlm_accelerate.py \
    --config configs/trm_vlm_qa_qwen2.5-3b_regularized.yaml \
    --checkpoint_path <path-to-full-state-checkpoint-dir> \
    --output_file eval_results.json

# Interactive inference
python scripts/eval/inference_vlm.py \
    --image_path <image> --question "<question>" \
    --checkpoint_path best   # or 'latest', or an explicit path
```

Checkpoint paths may be a full-state checkpoint directory, a sharded safetensors directory, or a loose `.pt` file.

### Checkpoint semantics

- `checkpoint_best/` (new format) is a **full Accelerate full-state directory** (`model.safetensors` containing
  LoRA + `vision_proj` + refinement layers + `z_init`) — a complete, restorable VLM.
- `checkpoint_best_adapter/` is a **LoRA-only** export and **cannot** restore the complete VLM by itself.
- Historical adapter-only `checkpoint_best/` artifacts (LoRA only; `vision_proj`/refinement/`z_init` missing) are
  **incomplete for full-VLM inference and intentionally rejected** by `src/models/checkpoint_loader.load_vlm_checkpoint`
  with an explicit error naming the missing components. Use a full-state `checkpoint-epoch-N/` / `checkpoint-final/`
  directory, or re-run the fixed training path.
- Device placement is handled at load time: the HF backbone (possibly 4/8-bit quantized, device_map-sharded, PEFT-wrapped)
  is never moved after load; custom modules are aligned once to the backbone's device/dtype.

---

## Results

Curated retained artifacts:

- `results/alignment/alignment_eval_final/` — final paired PixMo retrieval tables (attention vs. historical
  Perceiver arm; multi-scale 4096→128 prefix ablation; plots). The **attention-pooling (canonical alignment path)
  rows are the authoritative retained alignment results** per `AUDIT_LEDGER.md`. The Perceiver rows are a
  **provenance-compromised historical outcome** (see the Perceiver section above) — retain as recorded, do not
  interpret as an architecture verdict.
- `results/train_history/` — training curves / loss components / LR schedules for the `pixmo_alignment` and
  `perceiver_mrl_alignment` runs.
- `results/eda/` — PixMo dataset split-distribution plots.

No provenance-free alignment result is included; a conflicting, unprovenanced metrics CSV was previously removed
(see `AUDIT_LEDGER.md`).

---

## Status & known debt

Current at commit `ca641a8` (sanity sweep); see `AUDIT_LEDGER.md` for the full audit trail. Known intentional debt:

1. `decoder.num_layers_baseline` / `decoder.use_learned_z_init` — dead config fields retained for compatibility.
2. `src/utils/registry.py` — public extension surface with zero current consumers.
3. Hardcoded HPC fallback path in `train_vlm_accelerate.py` checkpoint lookup.
4. `ImprovedMultimodalTrainer` has no CLI (canonical alignment CLI is planned, not built).
5. Legacy alias config fields (`lr`, `gradient_clip`, `grad_accum_steps`, `trainer.num_epochs`) kept because retained configs use them.
6. Git history carries large historical blobs (repo-history slimming deliberately not performed).

Audio / tri-modal remains retained-but-experimental; fusion and Path C (`MultimodalProjector`) are retired.
