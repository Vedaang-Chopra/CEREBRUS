# AUDIT_LEDGER.md — working scratch file (not polished docs)

## 1. Repository / version map

| Area | Path | Status | Notes |
|---|---|---|---|
| Repo root | `/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Edge Assistant` | git `main`, 1 commit ahead of `origin/main` | Large unstaged deletion set: old `code_base/v0–v2_code_base` files deleted in working tree (apparently relocated to `older_code_base/`, which is untracked). Also untracked: `architectures/`, `PROJECT_RESUME_CONTEXT.md`, `GRA - Research Questions.pdf` |
| **Current canonical code** | `edge_glass_modular/` | tracked/untracked mix (`.pi-subagents/` untracked) | Modular package: `src/{data, encoders, models, decoders, training, evaluation, utils}`, `configs/*.yaml`, `scripts/`, `notebooks/` (incl. `archive/`), `outputs/`, `reports/` |
| v0 legacy | `older_code_base/v0_code_base/` | untracked (git-deleted at old `code_base/` path) | First modularization: `architecture/` (audio/image/text encoders + text decoder + base), `data/`, `utils/` (config, io, similarity), `tests/`, `scripts/`, notebooks |
| v1 legacy | `older_code_base/v1_code_base/` | untracked | Alignment-training era: `imports/` (config, dataset, encoders, model, perceiver, `align_training/{heads,losses,steps,training,eval,text_encoder}`), `modular implementation/`, `reference_poc/` (4 PoC dirs), notebooks, `edge_glass/features/` |
| v2 legacy | `older_code_base/v2_code_base/` | untracked | Dataset-builder + multimodal alignment era: `imports/` (core, data, `in_memory_datasets`, `llm_integration`, `multimodal_alignment_perceiver`, train scripts, `trm_vlm_helper`), `dataset_builders.py`, many numbered notebooks, `notebooks/` subdir, `data/` subsets |
| Other | `older_code_base/{dataset,experiments,additional_code}` | untracked | Non-code extras (experiments artifacts, external `TinyRecursiveModels` reference repo) |

## 2. Subsystem map (code only)

| Subsystem | edge_glass_modular | v0 | v1 | v2 |
|---|---|---|---|---|
| Configuration | `src/config.py`, `configs/*.yaml` | `utils/config.py` | `imports/configs/`, `imports/config.yaml` | (inline in notebooks/scripts) |
| Data | `src/data/{datamodule,dataset_builder,downloader,transforms}.py` | `data/` (+ olmo shards) | `data/`, dataset code in `imports/dataset.py` | `imports/{data,in_memory_datasets}.py`, `dataset_builders.py`, dataset-build notebooks |
| Encoders / representation | `src/encoders/{audio,vision,text,mrl,perceiver,pooling}.py` | `architecture/{image,audio,text}_encoder.py` | `imports/encoders.py`, `imports/perceiver.py`, `imports/align_training/text_encoder.py` | inside `imports/multimodal_alignment_perceiver.py`, `core.py` |
| Models / architecture | `src/models/` | `architecture/base.py` | `imports/model.py` | `imports/core.py` (partial) |
| Decoders | `src/decoders/{trm,qwen}.py` | `architecture/text_decoder.py` | `reference_poc/04_decoder_poc/` (notebooks) | `imports/llm_integration.py`, `trm_vlm_helper.py` |
| Training | `src/training/` | (in notebooks) | `align_training/{training,steps,losses}.py`, `align_training.py`, `modular implementation/train.py` | `imports/train*.py` |
| Evaluation / inference | `src/evaluation/` (7 modules) | (in notebooks) | `03_eval_embeddings.ipynb`, `inference_eval.ipynb`, `freeze_align_eval.ipynb` | `03a_eval_*` notebooks, `05/06 inference` notebooks, `imports/align_training/eval.py` (v1-style) |
| Utilities | `src/utils/` | `utils/{config,io,similarity}.py` | `extract_code.py`, `run_notebook.py` misc scripts | `imports/core.py` helpers |
| Tests | (none found at light scan) | `tests/` + `pytest.ini` | `test_device_fix.py` | `imports/test_in_memory_datasets.py` |
| Notebooks / experiments | `notebooks/` + `scripts/` helpers + `outputs/`, `reports/` | root notebooks + `notebooks/` | many root + PoC notebooks | many numbered notebooks + `notebooks/` |

## 3. Directories to IGNORE during code audit

- `**/wandb/`, `**/__pycache__/`, `**/*.egg-info/`
- `older_code_base/v1_code_base/venv/`
- Data/checkpoint/cache dirs: `older_code_base/v2_code_base/cache/`, `v2_code_base/data/`, `v0_code_base/pixmo/`, `v0_code_base/data/olmo_data/`, `features_vision/`, `features_audio_librispeech/`, `runs_perceiver_mrl_qwen/` (all copies), root `wandb/`
- Notebooks *outputs* (e.g. `edge_glass_modular/notebooks/outputs/`, `eda_outputs/`, `outputs/`) — audit outputs dir names only, not contents
- Images/media: `architectures/`, `images/`, `arch_plots/`, `Papers/`, `*.pdf`, `*.wav`, root-level binary artifacts
- `older_code_base/additional_code/other_repo/` (external vendored repo)
- `edge_glass_modular/.pi-subagents/` (agent artifacts)

## 4. Obvious cross-version areas requiring comparison

1. **Encoders/perceiver**: v0 `architecture/*_encoder.py` vs v1 `imports/encoders.py` + `perceiver.py` vs v2 perceiver-alignment vs current `src/encoders/` (incl. MRL/pooling).
2. **Configuration**: v0 `utils/config.py` vs v1 `imports/configs/config.py` + yaml vs current `src/config.py` + `configs/*.yaml`.
3. **Training/losses/steps**: v1 `align_training/*` (incl. `_bkp` duplicates) vs v2 `imports/train*.py` vs current `src/training/`.
4. **Data pipeline**: v0 `data/` vs v1 `imports/dataset.py` vs v2 `in_memory_datasets`/`dataset_builders` vs current `src/data/`.
5. **Decoders / LLM integration**: v0 `text_decoder.py` → v1 decoder PoC notebooks → v2 `llm_integration.py`/`trm_vlm_helper.py` → current `src/decoders/{trm,qwen}.py`.
6. **Evaluation/retrieval metrics**: v1 eval notebooks + `align_training/eval.py` vs v2 `03a_eval_*` notebooks vs current `src/evaluation/`.
7. **Notebook duplication within versions**: v1 has `_bkp`/`-1`/`-2` notebook duplicates; v2 has near-duplicate alignment/training notebook variants (MLP vs perceiver, online vs offline).
8. **Legacy relocation consistency**: git shows deletions under `code_base/v{0,1,2}_code_base/` while untracked copies exist under `older_code_base/` — verify contents actually match (otherwise git history is the only backup of some files). **RESOLVED 2026-09-19 (baseline commit `467cbba`):** content-hash comparison of HEAD blobs vs. working files confirmed 190/190 non-wandb files identical (`__init__.py` empty-blob collisions verified individually); the only deletions not matching under `older_code_base/` were `wandb/` run logs (intentionally dropped). Staged as 191 renames + 228 wandb deletions and committed as the relocation baseline. HEAD `467cbba` = `origin/main` (`693e43b`) + relocation commit + prior audit commit. Still untracked (never tracked): `older_code_base/additional_code/` (incl. vendored `TinyRecursiveModels` w/ its own `.git`), `older_code_base/v2_code_base/data/`, `architectures/`, `PROJECT_RESUME_CONTEXT.md`, `AUDIT_LEDGER.md`.

---

## Data subsystem (audited 2026-09-19)

Scope: data source code only (`edge_glass_modular/src/data/` vs. data code in v0/v1/v2). No dataset contents, W&B, caches, or generated features inspected.

### Canonical candidates (current implementation)

| File | Contents | Verdict |
|---|---|---|
| `edge_glass_modular/src/data/dataset_builder.py` (435 L) | `ImageTextDataset` (JSON metadata + image path), `AudioTextDataset` (waveform pad/truncate to 10 s @16 kHz), `InstructionDataset` (passthrough), `TriModalDataset` (zips vision+audio meta by index), `PixmoParquetImageTextDataset` (parquet with embedded `image_bytes`, verifies/drops invalid images, optional text dropout), `PixmoQADataset` (parquet `image_bytes`+question+answer, tokenized Q/A), builders `build_datasets` (metadata.json-driven) and `build_image_datasets_from_parquet` (train/val/test) | **KEEP** — this is the live surface. Actively consumed by `scripts/train_alignment.py`, `scripts/evaluate_freeze_align.py`, `scripts/train_vlm_accelerate.py`, `scripts/evaluate_vlm_accelerate.py`, `scripts/generate_eval_notebooks.py` and ~10 notebooks |
| `edge_glass_modular/src/data/datamodule.py` (44 L) | `MultimodalDataModule` — wraps `build_datasets`, single collate (stack image/audio, list text/sample_id), `train_dataloader()` prefers tri_modal else vision_text | **KEEP**; note: only exposes a train loader; no val/test loader path |
| `edge_glass_modular/src/data/transforms.py` (74 L) | CLIP-normalize vision pipeline (`build_transforms`) + train/eval variants with aug (`get_image_transforms`); torchaudio resample→16 kHz loader + 128-mel spec transform | **KEEP** |
| `edge_glass_modular/src/data/downloader.py` (197 L) | `download_pixmo_subset` (HF pixmo-cap → URL fetch via mp Pool → metadata.json), `download_common_voice_subset` (multi-version fallback loader), `build_instruction_corpus` (AudioCaps + OpenOrca), `download_instruction_dataset` (OpenOrca streaming) | **KEEP / LATER REFACTOR** — no val split support; no parquet output (current datasets moved to parquet); dead-ish relative to parquet flow |

### Older alternatives examined

| Version | File | What it does | Equivalent in current? |
|---|---|---|---|
| v0 | `data/image_data.py` (440 L) | `ImageCaptionDataModule` + config dataclass: generic HF image-caption loading, row filtering, train/val subsetting, **feature caching to .pt with index JSON** (`cache_image_features`), `ImageCaptionFeaturesDataset` reader | No — current has no HF-URL→cache-feature pipeline at all |
| v0 | `data/audio_data.py` (377 L) | `AudioDataModule` + `AudioFeaturesDataset`: mirror of image module for audio (filter, subset, cache Whisper-style features to .pt) | No |
| v1 | `imports/dataset.py` (418 L) | `PixmoFeatureDataset` / `LibriSpeechFeatureDataset` (read cached .pt blobs, elaborate multi-strategy path fixing), `collate_alignment` (padding + mask + modality ids), `PixmoVisionDataset` / `LibriSpeechAudioDataset` (on-the-fly encode inside `__getitem__` w/ retry-skip logic) | Partially — no .pt feature datasets or `collate_alignment` in current |
| v2 | `imports/data.py` (368 L) | `FeatureDataset` (generic .pt, modality param), `ImageTextDataset` (HF on-the-fly, URL or PIL, retry), `collate_features`/`collate_images`, `create_dataloader` factory, `SimpleImageTextDataset` | Partially — generic .pt feature dataset absent from current |
| v2 | `imports/in_memory_datasets.py` (435 L) | `InMemoryImageTextDataset` / `InMemoryAudioTextDataset`: mp-pool preload of all images/audio into RAM (resize LANCZOS, resample via librosa, blank-image fallback, failed-index tracking) | **No** — nothing equivalent in current |
| v2 | `dataset_builders.py` (88 L) | `build_pixmocap_parquet` / `build_musiccaps_parquet`: HF → shuffled subset → **local parquet** | Partially — current consumes parquet but has no builder for it (except ad-hoc notebooks) |

### Unique salvageable behavior (not in `edge_glass_modular/src/data`)

1. **Feature caching pipeline** (v0 `image_data.py`/`audio_data.py`, v1 feature datasets): encode once → .pt blobs + index JSON → reload for training. Current code always recomputes encoder passes; this is the biggest salvage candidate if pre-extracted-feature training is ever resumed.
2. **In-memory preloading** (v2 `in_memory_datasets.py`): parallel preload with graceful per-image failure fallback (blank gray image + `failed_indices`); relevant for repeated-epoch Mac training.
3. **Retry-skip-on-failure `__getitem__`** (v1 `PixmoVisionDataset`, v2 `ImageTextDataset`): current `PixmoParquetImageTextDataset` *raises* on unreadable image at getitem time despite `_drop_invalid_images` pre-filter — v1/v2 skip-and-advance behavior is more robust.
4. **HF→parquet subset builders** (v2 `dataset_builders.py`): current flow consumes parquet (pixmo_alignment configs) but the builder lives only in v2 + notebooks.
5. **`collate_alignment`** (v1): variable-length feature padding + mask + modality ids — needed by any perceiver alignment run over features.
6. **Multi-version Common Voice fallback loader** (current `downloader.py`) — already salvaged into current; nothing more to do.

### Duplicated code

- `_drop_invalid_images` duplicated verbatim inside `PixmoParquetImageTextDataset` and `PixmoQADataset` (same file).
- Path-fixing logic repeated 3× across v0/v1/v2 feature datasets (`data/pixmo → data/data/pixmo` etc.).
- `ImageTextDataset` name collision: current `dataset_builder.py` vs. v2 `imports/data.py` — different semantics, same name.
- URL-fetch + PIL decode block appears in v0, v1 (×2), v2 (×3).

### Suspicious / broken code

- `scripts/create_eval_notebook.py` imports `from src.data.pixmo_dataset import ...` — **module does not exist** (stale generator script).
- v1 `imports/dataset.py:125` uses `dict[str, Any]` but **never imports `Any`** → NameError at runtime.
- `TriModalDataset` pairs vision/audio metadata **by index** with no ID matching — likely incorrect pairing unless metadata was pre-zipped.
- Current datasets truncate text by **characters** (`text[:max_text_length]`) while `PixmoQADataset` truncates by tokens — inconsistent semantics.
- `PixmoParquetImageTextDataset.__getitem__` raises `UnidentifiedImageError` for rows that survived `_drop_invalid_images` (which uses `verify()` only — `verify()` does not catch all corrupt images).
- Downloader seeds differ silently (pixmo seed 13, Common Voice seed 17) with no config passthrough.
- `downloader.build_instruction_corpus` writes `instruction.json` consumed by `build_datasets` via `dataset_cfg.text_root`, but record schema (`instruction/input/output/modality`) differs from `download_instruction_dataset` (`instruction/input/output/sample_id`) — InstructionDataset is schema-agnostic passthrough, so downstream must know which producer was used.

### Recommended future file organization (proposal only — NOT executed)

```
edge_glass_modular/src/data/
├── __init__.py            # explicit public exports only
├── datasets/
│   ├── image_text.py      # ImageTextDataset, PixmoParquetImageTextDataset, TriModalDataset
│   ├── qa.py              # PixmoQADataset (+ shared parquet base class w/ single _drop_invalid_images)
│   ├── audio_text.py      # AudioTextDataset
│   ├── features.py        # (SALVAGE) generic .pt FeatureDataset + index-based readers from v0/v1/v2
│   └── instruction.py     # InstructionDataset
├── collate.py             # MultimodalDataModule.collate + (SALVAGE) collate_alignment
├── transforms.py          # unchanged
├── datamodule.py          # MultimodalDataModule (+ val/test loader methods)
└── downloaders/
    ├── pixmo.py           # download_pixmo_subset + (SALVAGE) build_pixmocap_parquet
    ├── audio.py           # common voice, audiocaps, musiccaps parquet builder
    └── instructions.py    # instruction corpus builders (unify record schema)
```

Rule: `transforms.py`/`datasets/` stay import-cycle-free (transforms imported by datasets, never reverse).

### Unresolved questions

1. Is pre-extracted-feature training (.pt blob flow) still an active direction, or fully superseded by parquet+on-the-fly? Determines whether `features.py` salvage is worth doing now or should be HISTORICAL.
2. `TriModalDataset` index-zip pairing — was upstream metadata guaranteed aligned, or is this an acknowledged bug? (affects KEEP vs. fix-later)
3. Should `downloader.py` gain val-split + parquet output now (to retire v2 `dataset_builders.py` + ad-hoc notebook builders), or stay metadata-only?

### Data-only coherence pass (2026-09-19, post-baseline `467cbba`)

**Changes actually made (no behavior change intended):**
1. `scripts/create_eval_notebook.py`: fixed stale import `src.data.pixmo_dataset` → `src.data.dataset_builder` (PixmoQADataset) + `src.data.transforms` (get_image_transforms). Unambiguous replacement; generator script now emits a valid import.
2. `dataset_builder.py`: consolidated the verbatim-duplicated `_drop_invalid_images` + column validation + bytes→tensor decode (previously copy-pasted in `PixmoParquetImageTextDataset` and `PixmoQADataset`) into one shared base class `_ParquetImageDataset(Dataset)` with `required_columns` class attr and `_load_image()` helper. Both datasets now subclass it; `__getitem__` bodies, sample dicts, truncation, dropout, and warning text semantics unchanged. Hoisted `import torchvision.transforms as T` to module level (was previously re-imported inside `__getitem__`).
3. Public API preserved: `from src.data import *` and `from src.data.dataset_builder import ...` names unchanged; verified importable in a torch-capable venv.

**Parquet-builder determination:** YES — required. Live configs (`pixmo_alignment.yaml`, `pixmo_alignment_1.yaml`, `trm_vlm_qa*.yaml`) and notebooks consume `final_dataset/pixmo/pixmo_{train,val,test}.parquet` (embedded `image_bytes`). Current repo has no HF→parquet builder. Smallest salvage target: `older_code_base/v2_code_base/dataset_builders.py` (`build_pixmocap_parquet`, 35 L) and the QA split logic in `00_build_qa_decoder_alignment.ipynb` (PixmoQAMixConfig → train/val/test parquet). Defer actual salvage; mark HISTORICAL only the `.pt+index` flow.

**Functionality deliberately left historical (no active current consumer found):**
- `.pt + index JSON` pre-extracted-feature pipeline (v0 `*_data.py` cache modules, v1/v2 `FeatureDataset`/`PixmoFeatureDataset`, `collate_alignment`) — no current config/script/notebook reads `.pt` feature indices.
- v2 `in_memory_datasets.py` preloading and v1/v2 skip-on-failure `__getitem__` — current parquet datasets already pre-drop corrupt rows; skip-on-failure is a behavior change, not adopted.
- `ImageCaptionDataModule`/`AudioDataModule` (v0) generic HF datamodules — superseded by metadata/parquet flow.

**Remaining data issues (untouched, need decisions or later pass):**
- `TriModalDataset` pairs vision/audio metadata by raw index (no ID matching) — suspected behavior bug; NOT changed (see Questions).
- Text truncation inconsistency: image-text datasets truncate by **characters**, `PixmoQADataset` by **tokens**.
- `PixmoQADataset` docstring mentions `source` column but it is not in `required_columns` (benign).
- `downloader.py` has no val-split/parquet output and hard-coded differing seeds (13/17).
- `MultimodalDataModule` exposes only `train_dataloader()`; no val/test loader.
- `datamodule.collate` cannot batch tokenized QA batches (only raw text) — QA training path constructs its own loaders in scripts, bypassing the DataModule.

**Behavior changes requiring a later decision (NOT implemented):**
1. Skip-on-failure `__getitem__` in parquet datasets (v1/v2 style) vs. current raise-on-corrupt.
2. `TriModalDataset` ID-based pairing.
3. Unifying char- vs token-based text truncation.

Phase A — pure extraction, zero behavior change:
1. Split `dataset_builder.py` into `datasets/{image_text,qa,audio_text,instruction}.py`; hoist the duplicated `_drop_invalid_images` + parquet-loading into one shared base class. Update `__init__.py` re-exports (all existing import paths keep working).
2. Add `collate.py`; move `MultimodalDataModule.collate` logic there (module keeps thin wrapper).

Phase B — small salvages:
3. Port v1/v2 skip-on-failure `__getitem__` behavior into parquet datasets (opt-in flag).
4. Port v2 `build_pixmocap_parquet` / `build_musiccaps_parquet` into `downloaders/`; make seeds/splits configurable.

Phase C — deferred:
5. Decide on `features.py` salvage (pending Q1).
6. Add val/test loader methods to `MultimodalDataModule`.

Phase A is mechanical and reversible (git-diffable per file); B/C each touch behavior and should be separate commits.

## Parquet builder salvage (2026-09-19)

**Done.** Created `edge_glass_modular/src/data/parquet_builder.py`; completed the data subsystem item marked "YES — required" above.

### Provenance (code salvaged from v2)

| Salvaged function | Source | Notes |
|---|---|---|
| `build_pixmocap_parquet` | `older_code_base/v2_code_base/dataset_builders.py:15-50` | Near-verbatim (HF split → optional shuffle+subset → `to_parquet`); prints → logging. Keeps original columns; does NOT embed image bytes. |
| `build_pixmocap_offline_parquet` + `_download_image_bytes` | `00_build_alignment_datasets_offline.ipynb` cells 2/4 | Chunked mp-Pool URL download → rows with `image_bytes`/`caption`/`image_url`/`sample_id` (`pixmo_%07d`). This is the schema `PixmoParquetImageTextDataset` consumes. Adapted: logging, `num_workers` param, returns path. |
| `split_parquet_train_val_test` | `00_build_qa_decoder_alignment.ipynb` cell 12 (PixmoQAMixConfig flow, 70/15/15) | Generalized to any parquet + prefix; deterministic via seed; remainder row → test. |

Not copied: `build_musiccaps_parquet` (no MusicCaps consumer in current configs), `build_musiccaps_waveform_parquet` (yt-dlp audio pipeline, no active current consumer), QA-source streaming loaders (pixmo-cap-qa/ask-anything/point-explanations/count — QA build remains ad-hoc/notebook-only).

### Files changed
- `edge_glass_modular/src/data/parquet_builder.py` (NEW)
- `edge_glass_modular/src/data/__init__.py` (+3 exports: `build_pixmocap_parquet`, `build_pixmocap_offline_parquet`, `split_parquet_train_val_test`)

### Validation performed
- `python -m compileall -q src/data` — OK.
- `import src.data` + direct `parquet_builder` imports in a torch/datasets-capable venv — OK; all `__all__` exports resolve.
- Smoke test (no network, synthetic 20-row image-bytes parquet): `split_parquet_train_val_test` → (14,3,3) as expected, deterministic across two calls with same seed; split output loads successfully through the real `PixmoParquetImageTextDataset` (required columns + decode path exercised); `_download_image_bytes` failure path returns `(idx, None)` without raising. PASSED.

### TriModalDataset usage determination
**INACTIVE** in practice: the only construction site is `build_datasets` (fires only when `dataset_cfg.image_root` AND `audio_root` are both set), and **no current config, script, or notebook sets `image_root`/`audio_root`** — every live config path consumes parquet directly via `build_image_datasets_from_parquet` / `PixmoQADataset`, bypassing `MultimodalDataModule` entirely. `MultimodalDataModule.train_dataloader()` would prefer `tri_modal` if present, but nothing instantiates the DataModule except the known-broken `scripts/train_alignment.py`. Implementation untouched.

### Remaining data issues (unchanged)
- `TriModalDataset` raw-index pairing (suspected bug) — untouched, INACTIVE.
- Char-vs-token truncation inconsistency — untouched by design.
- Corrupt-parquet rows still raise at `__getitem__` post-`verify()` prefilter — skip-on-failure deliberately NOT adopted.
- `downloader.py` still metadata-only (no val split, hard-coded seeds 13/17); parquet construction now lives in `parquet_builder.py` but is not wired into `downloader` flows.

### Intentionally left historical
- v2 `build_musiccaps_parquet` / `build_musiccaps_waveform_parquet` (no current consumer).
- v2 QA streaming loaders + `PixmoQAMixConfig` (QA parquet build stays notebook-only; only the generic split logic was salvaged).
- `.pt + index JSON` feature pipeline (per prior decision).

## Encoder / representation subsystem audit (2026-09-19)

Scope: `edge_glass_modular/src/encoders/{vision,text,audio,pooling,perceiver,mrl,__init__}.py` + driving config fields. Audit only — no code changed.

### Current implementation inventory

| File | Classes | Backbone | Input → Output | Token/pooled | Internal projection | Trainable | Consumers |
|---|---|---|---|---|---|---|---|
| `vision.py` | `VisionEncoder`, `MLPAdapter`, `VisionEncoderOutput` | `CLIPVisionModel` (frozen; final config `clip-vit-large-patch14-336`, hidden 1024) | `(B,3,336,336)` float tensor → pooled `(B, projection_dim)` normalized | token-level inside, pooled out | perceiver path: `MLPAdapter`(1024→512) → Perceiver → mean-pool → `LayerNorm+Linear`(512→4096/512); direct path: `Linear`(1024→4096) | projector/pooling/perceiver trainable; CLIP frozen (`requires_grad=False`) but **no `no_grad`/`eval()` in forward** — autograd still traverses the frozen backbone | `models/alignment.py:54` |
| `text.py` | `TextEncoder` | dual backend: `SentenceTransformer` (MPNet in final config) or `CLIPTextModel` if "clip" in name | list[str] or tokenized dict → pooled `(B, projection_dim)` normalized | pooled only (ST) | `Linear(hidden→projection)` | projector trainable; backbone frozen; ST path guarded by `no_grad` (inference mode) + `.clone()` | `models/alignment.py` |
| `audio.py` | `AudioEncoder` | `WhisperModel.encoder` (frozen) | mel features `(B, n_mels, T)` → pooled `(B, projection_dim)` normalized | token-level inside, pooled out | `Linear(d_model→projection)` or perceiver path | projector/perceiver trainable; backbone frozen | `models/alignment.py:68` (no active config) |
| `pooling.py` | `AttentionPooling` (multihead), `SimpleAttentionPooling` | n/a | `(B,L,D)[+mask]` → `(B,D)` | pooled | none | fully trainable | `vision.py` (`Simple` used by final config; `AttentionPooling` **never referenced by any config — dead option**) |
| `perceiver.py` | `PerceiverResampler`, `PerceiverResamplerLayer`, `PerceiverAttention`, `FeedForward` | n/a | `(B,T,dim)` → `(B,num_latents,dim)` | token-level | none (dim-preserving) | fully trainable | `vision.py`, `audio.py` |
| `mrl.py` | `MatryoshkaProjection`, `matryoshka_loss` | n/a | `(B,input_dim)` → `{dim: (B,dim) normalized}` | pooled | **none — pure prefix slicing + re-L2-normalize; zero parameters** | n/a (no params) | `vision.py`, `text.py`, `audio.py`; losses in `models/losses.py` |

Config dependencies: `configs/pixmo_alignment.yaml` = attention path (proj 4096, `SimpleAttentionPooling`, MRL [2048…128]) — **artifact-backed final path**. `configs/perceiver_mrl_alignment.yaml` = perceiver path (proj 512, latent_dim 512, 64 latents, 4 layers, 8 heads, MRL [256,128,64], attention pooling off).

### Component-by-component historical comparison

**Vision.** V0 `architecture/image_encoder.py`: generic `AutoModel` wrapper, `feature_strategy` (auto/image_embeds/layers_concat) + pool (mean/cls/none), `@torch.no_grad()`, fp16, device self-managed. V1 `imports/encoders.py`: DINOv2-style **two-layer concat** (`hidden_states[1]` + `[-2]`, CLS removed) → `(B,T,2D)` patch features + mask; careful MPS/fp16 fallback + device-sync logic. V2: notebook-era CLIP patch tokens → MLPAdapter → Perceiver → `AlignmentProjector` (mean pool → LN+Linear). Modular `vision.py` = V2 lineage folded into one class, with attention pooling added. **Lost in migration:** V1's multi-layer feature concat + mask, V0's feature-strategy flexibility, and the disciplined device/dtype/no_grad handling.

**Text.** V0: AutoModel mean/CLS pool, batched, frozen. V1 `align_training/text_encoder.py`: HFTextEncoder, CLS pool, fp32, frozen. Modular: adds MPNet `SentenceTransformer` backend (final-config backbone) + projection + MRL + L2 normalize. Modular is the only version with a projection head; closest to final experiments. `SequenceTransformer` sequence output unsupported (documented).

**Audio.** V0/V1: Whisper wrappers with strategy/pooling options and device handling. Modular: mean-pool (mask-aware) or Perceiver; MRL option. **Bug:** `audio.py:161-167` computes `mrl_embeddings` from the **un-normalized** pooled vector, then L2-normalizes `pooled` afterwards — order inverted vs. `vision.py`/`text.py` (which normalize before MRL). Not exercised by any current config.

**Pooling.** No historical standalone equivalent (V2 `AlignmentProjector` had inline mean/first/linear-attention options). Modular `SimpleAttentionPooling` (learnable scalar-attention query, LN, weighted sum) is the **artifact-backed** pooling of the final 77.7%/77.5% R@5 result. `AttentionPooling` (multihead, `nn.MultiheadAttention`) is unused by any config.

**Perceiver — three distinct designs exist:**
1. **v1 `imports/perceiver.py` (`PerceiverLatentEncoder`)**: latents 64×512; per block: cross-attn (with `enc_proj` input projection + KV mask) → **self-attn** → FFN(mlp_ratio 4), all pre-LN + residual; d_input explicit; dropout param; explicit input-dim assert.
2. **v2 `imports/multimodal_alignment_perceiver.py` (`PerceiverResampler`)**: 64×512, 4 layers, pre-LN **inside** attention (ln_q/ln_kv), **self-attn present**, KV mask support, latent init `randn*dim**-0.5`, `ln_out` final norm; MLPAdapter upstream; `AlignmentProjector` (mean pool + LN+Linear) downstream. This is the design the packaged lineage imitates.
3. **Modular `perceiver.py`**: cross-attn (single shared `norm1` module applied to both x and latents) → FFN. **No self-attention block, no mask support, dropout defaults 0.0** (configs pass nothing for dropout → 0), final `norm` present.
4. **Notebook-final `PerceiverV3Model`** (`07_alignment_final_report.ipynb` cell 4): self-contained — perceiver **with self-attention** and pre-LN in attention, LN+Linear adapters, `d_vision=1024, d_text=384, d_model=512, d_out=512`, mean pool over latents, then LN+Linear projection. **This is the implementation that produced the failed Perceiver arm (0.20%/0.17% R@5) — not the packaged module.** Note its `d_text=384` implies MiniLM text features, not the MPNet backbone in `perceiver_mrl_alignment.yaml`.

**MRL.** Modular `MatryoshkaProjection` = **prefix slicing + re-L2-normalization only** — no learned projection, no separate nested heads, no internal loss; supervision lives in `models/losses.py:mrl_loss` (reuses the same `contrastive_loss` at sampled dims, `sample_single_dim=True` per final config). This matches the final multiscale **evaluation** semantics (`evaluation/metrics.py` re-slices and re-normalizes the same learned 4096-d vector) and the resume-context claim ("normalized prefix-slicing, not a learned cascade"). v1 `matryoshka_contrastive_loss` differs: averages symmetric loss over **all** radii and validates finiteness; modular samples **one** dim per batch. v2 config `mrl_dims=(64,128,256,512)` over a 512-d align space (fractional prefixes), vs modular's [2048…128] over 4096-d. `mrl.py:matryoshka_loss` is a duplicate of `models/losses.mrl_loss` with no current consumer.

### Architecture-boundary findings (flagged, not fixed)

1. **`vision.py` orchestrates everything**: encoder + projection + perceiver + pooling + normalization + MRL in one forward. Projection arguably belongs in a head module (v2 kept them separate). The two paths even differ in *where* projection happens: direct path projects before pooling; perceiver path projects after mean-pooling.
2. **Perceiver path bypasses the attention-pooling module entirely** (`use_attention_pooling and not use_perceiver`) — mean-pooling hard-coded for latents; config `pooling_type` is silently ignored in perceiver mode.
3. **`audio.py` normalize/MRL ordering bug** (above).
4. **`mrl.py` contains a loss function** (`matryoshka_loss`) — loss logic belongs with `models/losses.py`; it's dead code duplicating `mrl_loss`.
5. **No `no_grad`/`eval()` on frozen backbones in `vision.py`/`audio.py`** — frozen CLIP/Whisper forward runs inside the autograd graph every step (only weights are `requires_grad=False`); wasteful and inconsistent with `text.py`.
6. **`VisionEncoder` keeps `output_hidden_states=True`** on every forward but never uses `hidden_states` — wasted compute.
7. **`AttentionPooling` (multihead) is dead** — no config or code path selects it.
8. **Perceiver layer reuses one `norm1` module** for both input and latents (works, but weights are shared where v1/v2 use separate norms; also pre-norms the *input* on every layer repeatedly).

### Code-quality classification

| Component | Classification | Rationale |
|---|---|---|
| `pooling.py` (Simple) | **KEEP** | Artifact-backed; clean; single responsibility. |
| `text.py` | **KEEP** | Canonical MPNet path; sound; only minor cleanup (ST inference-mode/clone workaround). |
| `mrl.py` (`MatryoshkaProjection`) | **KEEP** | Matches research experiment exactly; parameter-free. |
| `mrl.py` (`matryoshka_loss`) | **HISTORICAL** (dead duplicate) | Superseded by `models/losses.mrl_loss`. |
| `vision.py` | **CLEAN LATER** | Structurally sound and artifact-backed on the direct path, but boundary violations + no_grad/hidden_states waste. |
| `audio.py` | **REPAIR LATER** | Normalize/MRL ordering bug; no active consumer to validate against. |
| `pooling.py` (`AttentionPooling`) | **EXPERIMENTAL** | Unexercised multihead variant; keep or delete in a later pass. |
| `perceiver.py` | **MERGE WITH OLDER IMPLEMENTATION** | Missing self-attention + mask support that v1/v2 AND the notebook-final V3 all have; if perceiver work resumes, base it on v2 `multimodal_alignment_perceiver.py` (the design family the final experiment actually used), not the current reduced module. Do **not** simply import the notebook V3 (unproven, failed, but architecturally closest). |
| V0 wrappers (`image/text/audio_encoder.py`) | **HISTORICAL** | Feature-strategy flexibility + device/dtype discipline worth borrowing, not restoring wholesale. |
| V1 `encoders.py` | **HISTORICAL** (salvage: multi-layer concat + MPS/device handling) | Superseded by parquet/CLIP flow. |

### Worth salvaging if encoders are touched later
- V1's device/dtype discipline (MPS fp16 fallback, model-device-vs-input sync) — current modules have none.
- v2 `PerceiverLayer` self-attention + mask support + per-attention pre-LN — required to make the packaged Perceiver comparable to the notebook V3 that was actually evaluated.
- V0 `feature_strategy`/`layers_concat` flexibility if multi-layer features are ever revisited.

### Unresolved questions
1. Which Perceiver is "the" canonical target: packaged v2-style (with self-attn) vs notebook V3 (which produced the recorded negative result)? Any future Perceiver rerun must pick one and reconcile the dimension labeling (512-d actual vs 4096-d table labels).
2. Notebook V3 used MiniLM (384-d) text features; `perceiver_mrl_alignment.yaml` specifies MPNet. The final Perceiver-vs-Attention comparison is therefore also a text-backbone confound — cannot be resolved from repo evidence.
3. Was `AttentionPooling` (multihead) ever trained anywhere? No checkpoint artifact or config reference was found.

### Decision
Modular encoders remain canonical (`vision.py` direct path + `pooling.py` + `mrl.py` = artifact-backed). No restructure performed in this pass.

## Core model + loss subsystem audit (2026-09-19)

Scope: `edge_glass_modular/src/models/{alignment,projector,fusion,losses,trm_qwen_vlm,__init__}.py` + decoder/config interfaces. Audit only — no code changed.

### 1. Reconstructed current forward graph

**A. Artifact-backed alignment path** (`configs/pixmo_alignment.yaml` — the 77.7/77.5% R@5 result):

```
images (B,3,336,336)
  → VisionEncoder: frozen CLIP ViT-L/14-336 → patch tokens (B,577,1024)
  → Linear proj 1024→4096  (projection happens BEFORE pooling)
  → SimpleAttentionPooling (B,4096)
  → L2 normalize (vision.py:224)
  → MatryoshkaProjection prefix-slice+re-normalize → {2048: (B,2048) … 128: (B,128)}
texts (list[str])
  → TextEncoder: frozen MPNet → (B,384) → Linear 384→4096 → L2 normalize → MRL slice
→ AlignmentLoss: 1.0×contrastive (InfoNCE, T=0.07, re-normalizes inside) + 0.5×mrl_loss(sampled single dim)
```
No fusion (every config sets `fusion: null`), no decoder in alignment configs. `MultimodalAlignmentModel.forward` wires this exactly; consumers: `improved_trainer.py:315` (`self.model(images=…, texts=…)`), `scripts/evaluate_freeze_align.py`, notebook 07, `generate_eval_notebooks.py`. Gradients flow through both modalities (symmetric CE).

**B. VLM/QA path actually executed** (`scripts/train_vlm_accelerate.py` + notebooks 03/05):

```
frozen aligned vision encoder → return_sequence=True → sequence (B,577,4096)
  → QwenVLM.vision_proj Linear 4096→hidden (per-token projection, NOT MultimodalProjector)
  → prefix_embeds prepended inside QwenDecoder.forward (inputs_embeds concat, mask extended, labels -100-padded)
  → Qwen backbone (Qwen2.5, LoRA r=64 on q/k/v/o_proj, 4-bit nf4) → last_hidden (B, L+prefix, H)
  → [if TRM] down_proj H→trm_dim → y/z dual-latent recursion (num_recursion_steps=4, cat along seq dim)
    → up_proj → residual add onto qwen_hidden → lm_head → manual shifted CE (prefix logits sliced off)
```
Labels: question tokens −100, answer tokens supervised. Trainable: LoRA + vision_proj + TRM layers/norms/z_init; vision encoder frozen. Note: **QA configs configure `use_perceiver: true` (latent_dim 1024)** for the vision encoder while the loaded checkpoint (`pixmo_alignment/checkpoint_best.pt`, strict=False) is the attention-pooling model — perceiver weights don't exist in the checkpoint and attention-pool weights are silently dropped (strict=False), i.e. the "aligned" vision features in the QA path are likely not aligned at all. See §5.

**C. Never-executed hybrid path** (`MultimodalAlignmentModel` + decoder, alignment.py:120–155): `llm_projector = MultimodalProjector(proj_dim→hidden×8, num_tokens=8)` producing 8 soft prompt tokens from a *pooled* embedding. No script or config instantiates this combination (QA path uses QwenVLM instead; the known-broken `scripts/train_alignment.py` uses no decoder). Text is never offered as a decoder prefix — `prefix_emb` selection is fused→vision→audio only (alignment.py:267–274).

### 2. Model-component findings

1. **`fusion.py` is dead in practice**: all 9 configs set `fusion: null`; it only instantiates when >1 encoder AND fusion set; no artifact, no notebook, no experiment ever exercised it. `CrossModalAttentionLayer` is actually *self*-attention over modality tokens (mislabeled). Not necessary for the final VLM path.
2. **`projector.py` duplication**: `MultimodalProjector` (path C) and `VisionToLLMProjector` (v2 `llm_integration.py:159` equivalent, no consumer in current code) vs the QA path's bare `nn.Linear` in `trm_qwen_vlm.py:41`. Three projection-to-LLM designs; only the bare Linear runs.
3. **`alignment.py` decoder-branch fragility**: `hasattr(self.decoder, "model")` to distinguish Qwen vs TRM; TRM branch reads `decoder_output.get("loss"/"logits")` — depends on TRMDecoder return type (verify in decoder task). Dead commented-out loss block (lines 304–315) duplicated by the live one (316–325). Custom `.to()` override for quantized decoders (drops decoder during move) is sound.
4. **Perceiver plumbing**: `alignment.py` passes perceiver params straight from config to `VisionEncoder`/`AudioEncoder`; the expected Perceiver interface is the packaged `perceiver.py` `PerceiverResampler: (B,T,dim) → (B,num_latents,dim)` — alignment.py itself is agnostic; the contract lives inside the encoders.
5. **Silently ignored config/args**: `confidence_threshold` (QwenVLM ctor arg, default 0.75) is **stored and never read** — no adaptive halting exists despite TRM framing; `num_trm_layers=4`/`num_recursion_steps=4` hard-coded in `train_vlm_accelerate.py:280` ("Configurable?"), not read from config; `pooling_type` ignored in perceiver mode (encoder-level, recorded earlier).
6. **`trm_qwen_vlm.py` runtime anti-patterns**: modules re-assigned in place during forward (`self.vision_proj = self.vision_proj.to(...)`, lines 104–113 and 177–183) — mutates the module tree per step, incompatible with DDP static graphs/Accelerate prepare assumptions. **`generate()` TRM branch does `out['logits']` (line 363) but `forward` returns a `CausalLMOutputWithPast` object in both branches → TypeError on the first generated token** (TRM generate cannot execute). TRM generate also applies fixed 4-step recursion of the full prefix+context sequence per token (no halting, O(T²)-ish cost) and ignores `confidence_threshold`.
7. **TRM's actual role**: recursive refinement of Qwen's final hidden states (dual y/z latents, sequence-dim concat, residual add) — NOT a standalone decoder, NOT a per-layer adapter, NOT the paper TRM (no deep supervision, no halt head). `z_init` is a single learnable row expanded across all positions.
8. **Loss-path divergence in QwenVLM**: non-TRM branch delegates loss to Qwen internals (prefix labels padded to −100 inside QwenDecoder); TRM branch recomputes shifted CE manually after slicing prefix logits. Likely equivalent but untested and structurally divergent.
9. **Two VLM compositions coexist** (B and C) with different token-injection designs (8 pooled soft tokens vs per-patch 577-token prefix). Only B has execution evidence (notebook traces; no verified result).
10. **alignment.py:318**: `alignment_losses.get("total_loss", 0)` silently yields int 0 when the tri-modal branch conditions miss (e.g. TriModal loss attached but only 2 modalities passed) — crashes later at `loss.item()` with a misleading error.

### 3. Loss audit (`models/losses.py`)

| Loss | Purpose | Shapes | Normalization | Temp | Pos/neg | Both sides? | Called by |
|---|---|---|---|---|---|---|---|
| `contrastive_loss` | symmetric InfoNCE/CLIP | (B,d)×(B,d)→scalar | F.normalize inside (idempotent; inputs already normalized upstream) | 0.07 | diagonal=positive, in-batch negatives | yes (a→b, b→a averaged) | `AlignmentLoss`, `TriModalAlignmentLoss` |
| `mrl_loss` | nested-prefix supervision | Dict[dim→(B,dim)] | inputs already prefix-normalized by `MatryoshkaProjection` | 0.07 | same, per sampled dim | yes | `AlignmentLoss.forward` — **canonical** |
| `AlignmentLoss` | weighted combo | — | — | 0.07 | — | — | `MultimodalAlignmentModel` (bimodal); trainer logs `loss_clip`/`loss_mrl` |
| `TriModalAlignmentLoss` | pairwise tri-modal | — | — | 0.07 | — | yes | 3-encoder case only; **no artifact-backed run** |
| `encoders/mrl.py:matryoshka_loss` | duplicate MRL | — | re-normalizes inside | 0.07 | — | yes | **NO CONSUMER** — dead duplicate |

**Duplication resolution:** `models/losses.py:mrl_loss` is canonical (consumed via `AlignmentLoss`; matches final config `sample_single_mrl_dim: true` + weights). `encoders/mrl.py:matryoshka_loss` differs materially (computes ALL dims every call, no sampling, no weights) and has no caller in src/scripts/notebooks → HISTORICAL. Confirms the encoder-audit classification.

Additional loss findings:
1. **NameError bug, losses.py:171**: `if mrl_losses:` executes unconditionally at the end of `AlignmentLoss.forward`, but `mrl_losses` is only bound inside `if embeddings_a_mrl is not None and …` — with MRL dicts omitted (`use_mrl: false`), forward raises NameError instead of returning contrastive-only losses. Latent today (both live alignment configs set `use_mrl: true`).
2. **Temperature hard-coded 0.07** at `alignment.py:165,172`, not plumbed from config (no temperature field exists in `optimization`). `AlignmentLoss.__init__` default `contrastive_weight=0.25` with stale "# Updated: clip_weight = 0.25" comment contradicts both live configs (pixmo 1.0, perceiver 0.25); config wins, defaults are misleading.
3. **Historical v1 `matryoshka_contrastive_loss`** (`v1 align_training/losses.py:94`): averages symmetric loss over ALL radii + finite validation — the only materially different MRL-loss implementation anywhere; unused. v1 `clip_contrastive_loss` is the same math family as current `contrastive_loss` (explicit cosine matrix / T).
4. No other loss implementations exist in v0/v2/training/notebooks; notebook 07 re-implements only evaluation-side prefix re-normalization, not a loss.
5. Gradient flow verified correct on the canonical path: encoders trainable (projector/pooling), losses symmetric, MRL prefixes carry gradient into the same 4096-d projection.

### 4. Alignment model vs experiments

- Current `MultimodalAlignmentModel` (attention path) **matches** the successful direct-alignment experiment (notebook 07 instantiates it and loads `checkpoint_best.pt`; config values match). ✔
- The Perceiver branch of `VisionEncoder` (MLPAdapter→packaged PerceiverResampler→mean-pool→LN+Linear) **matches nothing ever run**: the failed experiment used notebook-local `PerceiverV3Model` (self-contained, self-attention, d_text=384 MiniLM, d_out=512), while `perceiver_mrl_alignment.yaml` targets the packaged module with different dims (proj 512, latent 512). No artifact exists for a packaged-perceiver run.
- Text/vision symmetry: both end Linear→L2-norm; vision uniquely has attention pooling, and in perceiver mode projection placement flips (after mean-pool vs before pooling) — asymmetric, but that asymmetry is covered by the artifacts.
- Trainable set: projector/pooling/perceiver/MRL(no-op) trainable; frozen backbones still traverse autograd (no `no_grad`, recorded earlier).

### 5. Perceiver evidence summary (unresolved choice)

- `alignment.py` expects the packaged `PerceiverResampler` interface, only indirectly through `VisionEncoder`/`AudioEncoder` ctor args. Replacing the packaged module with v1 `PerceiverLatentEncoder` or v2 `PerceiverResampler` is **interface-compatible** (same (B,T,dim)→(B,latents,dim) contract); only constructor wiring in the encoders changes — no model-interface change required.
- Experiments: successful runs never instantiated any Perceiver; the failed run instantiated notebook V3. QA configs (`trm_vlm_qa*`) nominally select the packaged perceiver with **`perceiver_latent_dim: 1024`** — matching no other config (perceiver_mrl uses 512) and incompatible with the attention-path checkpoint they load with `strict=False`. Evidence the QA-path configs were never validated as written.
- No repository evidence makes the canonical choice unequivocal; v2-style (with self-attention) remains the design family closest to what was actually evaluated. No change made.

### 6. Classification

| Component | Classification | Rationale |
|---|---|---|
| `losses.py` — `contrastive_loss`, `mrl_loss`, `AlignmentLoss` | **KEEP** | Artifact-backed; canonical. Fix NameError + config-plumb temperature later. |
| `losses.py` — `TriModalAlignmentLoss` | **EXPERIMENTAL** | Wired for 3-encoder case; no artifact-backed run; hard-coded weights. |
| `encoders/mrl.py:matryoshka_loss` | **HISTORICAL** (duplicate) | Dead; superseded by `models/losses.mrl_loss`. |
| `alignment.py` — model class + canonical graph | **CLEAN LATER** | Correct on path A; dead commented block, hard-coded temperature, fragile decoder dispatch, silent `0` fallback. |
| `alignment.py` — decoder/`llm_projector` branch (path C) | **EXPERIMENTAL** | Never executed by any script/config; superseded in practice by QwenVLM. |
| `fusion.py` | **EXPERIMENTAL** (near-dead) | Zero consumers, zero artifacts; not necessary for the final VLM path. |
| `projector.py` — `MultimodalProjector` | **CLEAN LATER** | Used only by dead path C; keep/retire with path C decision. |
| `projector.py` — `VisionToLLMProjector` | **HISTORICAL** | Mirror of v2 `llm_integration.py:159`; no current consumer. |
| `trm_qwen_vlm.py` — QwenVLM | **REFACTOR LATER** | The actually-executed VLM composition, but TRM-generate cannot run, in-place module mutation, unused `confidence_threshold`, hard-coded knobs, divergent loss paths. TRM role = hidden-state refiner. |
| `decoders/qwen.py` / `trm.py` | Deferred (interfaces only) | QwenDecoder prefix-injection correct; TRM internals deferred to decoder task. |
| Path A end-to-end | **KEEP** | Matches the successful direct-alignment experiment. |

### 7. Unresolved questions (repository evidence cannot resolve)

1. Was the VLM/QA path ever trained with a checkpoint-matching vision encoder (attention config), or always with the mismatched perceiver config + `strict=False`? Determines whether the QA direction is salvageable as-is or the configs must be rewritten before any rerun.
2. Is path C (pooled 8-token soft-prefix via `MultimodalProjector`) an intentional future design or abandoned after the QwenVLM per-patch-prefix path won? Decides KEEP vs MERGE for `MultimodalProjector` + the alignment.py decoder branch.
3. Should adaptive halting (`confidence_threshold`) be implemented for TRM (the name implies intent), or is fixed 4-step recursion the actual design? No artifact answers this.

### Recommended next step
Fix the mechanical model/loss defects before any training repair: (a) `losses.py:171` `mrl_losses` NameError, (b) QwenVLM TRM-generate object-vs-dict indexing, (c) QA configs' vision-encoder/checkpoint mismatch. Behavior-preserving except where current code cannot execute at all.

## Decoder subsystem audit (2026-09-19)

Scope: `edge_glass_modular/src/decoders/{qwen,trm,__init__}.py` + `src/models/trm_qwen_vlm.py` as consumer/interface. Audit only — no code changed. Corrects one finding from the 2026-09-19 core-model pass (see §4.1).

### 1. Qwen audit (`decoders/qwen.py`)

1. **Wrapped class**: `AutoModelForCausalLM.from_pretrained` (Qwen2.5 family per configs: 1.5B/1.8B/3B/7B-Instruct; final QA config `Qwen/Qwen2.5-3B-Instruct`), `dtype=bfloat16`, optional BitsAndBytes 8-bit/4-bit-nf4, optional PEFT LoRA (r=64, α=128, dropout 0.2 on q/k/v/o_proj in the regularized config). Tokenizer loaded with `padding_side="left"`, pad=eos fallback.
2. **Dimensions**: `hidden_dim` = HF `config.hidden_size` (Qwen2.5-3B: 2048); `vocab_size` from config. Visual input expected as `prefix_embeds (B, num_prefix, hidden_dim)` — any upstream projection must produce hidden_dim. `QwenVLM.vision_proj` (4096→2048 per patch token) provides this.
3. **Insertion mechanism**: **projected patch embeddings as a soft prefix prepended to `inputs_embeds`** — NOT token replacement, NOT 8 pooled soft tokens (that is the unused path-C design). Concat order: `[prefix_embeds, text_embeds]`.
4. **Construction**: `inputs_embeds = get_input_embeddings()(input_ids)` then prefix cat; attention mask extended with a ones-block of `num_prefix` on the left; labels extended with −100 over the prefix positions so the HF-internal shifted CE ignores prefix logits. All three extensions are correct and mutually consistent (prefix count matches in mask, labels, and embeds).
5. **Frozen/trainable**: backbone frozen except LoRA adapters when `use_lora`; embedding table and lm_head untouched (frozen unless LoRA targets them — it does not). No explicit `requires_grad` audit of quantized weights; BitsAndBytes freezes quantized weights by default.
6. **Weight tying**: never touched by the wrapper — HF's own tie (Qwen2.5 ties `lm_head` to `embed_tokens`) is preserved. Nothing in QwenDecoder or QwenVLM breaks it. (Contrast: `TRMDecoder` ties manually, `trm.py:248`.)
7. **Generation vs training**: training forward uses `inputs_embeds` path; `generate()` also builds `inputs_embeds` (with a dummy pad token if no input_ids) and calls HF `.generate(inputs_embeds=…)`. **Consequence: HF returns only the NEWLY generated ids when given `inputs_embeds` (no prompt-echo)**. `QwenVLM.generate` baseline branch then does `outputs[:, prompt_len:]` (trm_qwen_vlm.py:314) — this slices `prompt_len` tokens off a tensor that contains only generated tokens, silently truncating/wrecking the answer. Confirmed code-level bug in the non-TRM baseline branch; unexercised there (notebook 05 ran with `use_trm=True`).
8. **Interface compatibility with `trm_qwen_vlm.py`**: compatible — QwenVLM calls `self.qwen(input_ids, attention_mask, prefix_embeds, labels, output_hidden_states=True, return_dict=True)` and reads `.hidden_states[-1]`, `.loss`, `.logits`, `model.lm_head`; `hidden_dim`/`vocab_size`/`tokenizer`/`encode_text` all exposed. QwenDecoder passes `**kwargs` straight to the HF model, so `output_hidden_states` flows through. Note `labels` passed twice effectively (inside kwargs and explicit) — fine, explicit wins.
9. **dtype/device/mutation**: QwenDecoder itself is clean (casts prefix embeds to embed dtype/device before cat). Debug `print` calls with forced config overrides (`num_key_value_heads` etc., qwen.py:64-71) are leftover debugging — harmless but ugly. The mutation problems live in the consumer (see §4.2).
10. **Runtime evidence**: notebooks 03_FIXED (18 executed cells, 2 errors — later load failures), 04 (missing checkpoint error), 05 (10 cells, 0 errors, real generation ran). QwenDecoder is exercised; its forward/generate are the proven surface.

### 2. TRM audit (`decoders/trm.py` — `TRMDecoder`)

**What it actually is: a plain small causal transformer decoder — there is NO recursion in it at all.** Components: token embedding → N× pre-LN RMSNorm transformer blocks (self-attn with optional RoPE, GELU MLP) → final RMSNorm → lm_head **tied** to the embedding. No y/z states, no iterative update, no deep supervision, no halt head, no carry. The "TRM" recursion exists only in the consumer `trm_qwen_vlm.py`.

1. **Provenance verdict**: `TRMDecoder` is **materially different from and loosely inspired by** TRM — it is a standard LLM with an TRM-branded name. The vendored original (`older_code_base/additional_code/other_repo/TinyRecursiveModels/models/recursive_reasoning/trm.py`) has: ACT halt head (`q_head`), puzzle embeddings, inner/outer carry with **detached carries between steps**, deep supervision, single-network dual-latent (z_H/z_L) recursion. The current module preserves **none** of that machinery.
2. **What v2 preserved/removed**: v2 `imports/trm_vlm_helper.py:TRMDecoderLLM` is a genuine TRM adaptation: x-embedding input, y init = x clone, z from learnable `z_init`, per-cycle n_recursions with **input injection (x+y added to z)**, y soft-updated via `probs @ token_embedding.weight`, deep-supervision CE averaged over `t_cycles` logits, **grad only on the last cycle** (`set_grad_enabled(t == last)` + detach between cycles). The current `decoders/trm.py` keeps only the layer primitives (RMSNorm/RoPE/attention) and drops input injection, soft y-update, deep supervision, and last-cycle-only grad. `QwenVLM`'s recursion re-invents a *different* variant (see §3).
3. **`confidence_threshold`**: stale configuration. Defined in QwenVLM ctor and in notebooks (`CONFIDENCE_THRESHOLD = 0.75`) but never read by any code path — no adaptive halting was ever implemented anywhere in the repo. The name implies intended ACT-style halting that was never built.
4. **Fixed 4-step recursion**: hard-coded at the call sites (`NUM_RECURSION_STEPS=4` in notebooks; `num_recursion_steps=4` in `train_vlm_accelerate.py:281`); QwenVLM accepts it as a ctor arg but no YAML config carries it. Config-controlled only via Python defaults.
5. **Gradients**: QwenVLM's recursion propagates through **all** 4 steps (no detach, no last-cycle-only) — gradient flow is correct but memory-heavy; this differs from both v2 (last-cycle-only) and the vendored original (detached carries). Training==generation computation in QwenVLM's TRM branch: generation re-runs the identical full forward per token (no KV cache), so yes, same computation — O(T²) cost.
6. **`TRMDecoder`-specific bugs**: (a) `TRMAttention` else-branch does `attn_weights + attention_mask` (trm.py:153) assuming an **additive** mask, but every caller pattern (alignment.py path C passes a 0/1 mask) supplies a multiplicative 0/1 mask → adding 1 everywhere is wrong (no masking + logits offset). Latent: path C never runs. (b) `generate()` drops `prefix_embeds` after the first iteration (trm.py:405) while recomputing the whole sequence each step without cache — prefix conditioning silently vanishes after token 1. (c) RoPE applied to query/key of dim `head_dim` with cached tables sized to `max_seq_len=2048`; `RotaryEmbedding.forward` silently truncates beyond that (no error).

### 3. The actual recursion in `trm_qwen_vlm.py` (QwenVLM TRM branch)

Input = Qwen's **final-layer hidden states** `qwen_hidden (B, L+P, H)` (language embeddings over image-prefix+text, NOT visual latents, NOT logits). Algorithm: `x = down_proj(qwen_hidden)`; learnable `z_init (1,1,trm_dim)` expanded to (B,L,trm_dim); `y = x`; loop 4×: `combined = cat([y, z], dim=1)` (sequence concat, doubling length) → N TRMLayer blocks → chunk back into y,z. After loop: `y_up = up_proj(y)`; **`final_hidden = qwen_hidden + y_up`** (residual against the original Qwen states); `logits = lm_head(final_hidden)`; manual shifted CE on answer tokens with prefix logits sliced off. Trainable: down/up_proj, TRM layers, trm_norm (present but **never applied** — `self.trm_norm` created and moved around but no call site uses it), z_init, vision_proj; Qwen otherwise LoRA/frozen. This is a *different* algorithm from v2's helper (no input injection, no soft y-update, no deep supervision, no last-cycle grad) — a third, previously undocumented variant.

### 4. Resolution of prior findings

1. **`out['logits']` on `CausalLMOutputWithPast` — NOT a bug (prior finding corrected).** HF `ModelOutput.__getitem__` supports string keys, so `out['logits']` is valid. Runtime-confirmed: notebook 05 executed the TRM generate path end-to-end without error. Downgrade to harmless style inconsistency (dict access on an object).
2. **In-place module `.to()` during forward (trm_qwen_vlm.py:104-113, 177-183)** — confirmed anti-pattern. Works in single-device/notebook contexts (evidence: notebook 05), but re-assigning `self.vision_proj`/`self.down_proj` per forward mutates the module tree, breaks DDP/Accelerate parameter registration, and forces host-side device/dtype churn every step. Requires training-context evidence for whether it ever corrupted a DDP run; structurally it must be removed before any distributed rerun.
3. **Two VLM projection/injection strategies** — confirmed and now fully characterized: (A) per-patch 577-token prefix via `QwenVLM.vision_proj` (the executed one) vs (B) pooled 8-soft-token prefix via `alignment.py` path C / `MultimodalProjector` (zero execution evidence). v2's `VisionToLLMProjector` (num_tokens pooled prefix) is the historical ancestor of B; the executed design A descends from notebook-era per-patch injection. Not a bug — an unresolved design fork.
4. **`confidence_threshold` unused** — confirmed stale configuration (see §2.3); no halting implementation exists anywhere in repo history except the vendored original's ACT head.
5. **New confirmed bug (from this audit)**: QwenVLM baseline-branch `outputs[:, prompt_len:]` double-strips against HF's inputs_embeds-only-generated-ids behavior (§1.7). And the previously recorded QA-config perceiver/checkpoint mismatch stands (use_perceiver:true, latent 1024 vs attention checkpoint, strict=False).

### 5. Historical comparison summary

| Implementation | Recursion | Halt/ACT | Deep supervision | Injection | Grad control | Status |
|---|---|---|---|---|---|---|
| Vendored TinyRecursiveModels `trm.py` | z_H/z_L dual, T/N loops | q_head ACT halt | yes | input concat | detached carries | reference only, never imported |
| v2 `trm_vlm_helper.TRMDecoderLLM` | y/z, n_recursions + t_cycles | no | yes (avg over cycles) | x+y added to z | last-cycle only | standalone LM; superseded |
| v1 `04_decoder_poc/` notebooks | — | — | — | prefix finetuning PoC | — | exploratory |
| v0 `text_decoder.DecoderLLM` | — | — | — | task-prompt wrapper | — | prompt-engineering era |
| **current `decoders/trm.py`** | **none** | no | no | prefix concat | n/a (plain LM) | misnamed plain decoder |
| **current QwenVLM TRM branch** | y/z seq-concat ×4 | no | no | residual add to Qwen hidden | all steps | executed (degenerate output) |

Lost functionality worth noting: v2's deep supervision + last-cycle-only grad (memory/perf discipline), the vendored ACT halt (the only real "recursive reasoning with halting" in the repo), and v2's input-injection design. Runtime evidence for QwenVLM+TRM generation exists but the outputs were degenerate ("It is a picture of a picture." loops) — consistent with the unaligned vision features concern from the core-model pass.

### 6. Classification

| Component | Classification | Rationale |
|---|---|---|
| `qwen.py` — QwenDecoder | **KEEP** | Correct prefix injection, mask/label handling; proven surface. Clean debug prints + config-override hack later. |
| `qwen.py` — `generate()` inputs_embeds interaction | **REPAIR LATER** (in consumer) | The double-strip bug is in QwenVLM, not here; document the generated-ids-only contract. |
| `trm.py` — `TRMDecoder` | **REFACTOR LATER / rename** | Not recursive despite the name; additive-mask bug latent in path C; generate drops prefix after step 1. Salvageable as a plain small LM. |
| `trm.py` — RMSNorm/RoPE/attention primitives | **KEEP** | Sound primitives shared with any future real TRM port. |
| `trm_qwen_vlm.py` — QwenVLM TRM branch | **REPAIR LATER** | Executed but a third undocumented variant; missing v2's deep supervision/grad discipline; `trm_norm` dead; `.to()` mutation; memory-heavy. |
| `trm_qwen_vlm.py` — baseline generate `outputs[:, prompt_len:]` | **REPAIR LATER** | Confirmed double-strip bug vs HF inputs_embeds semantics. |
| `confidence_threshold` plumbing | **HISTORICAL** (stale) | Never implemented anywhere. |
| Vendored TinyRecursiveModels | **HISTORICAL** (reference) | The only true TRM; import-reference for any future real port. |
| v2 `trm_vlm_helper.TRMDecoderLLM` | **HISTORICAL** (salvage patterns) | Deep supervision + last-cycle grad + input injection worth porting if TRM work resumes. |
| `__init__.py` exports | **KEEP** | Minimal, correct. |

### 7. Unresolved questions

1. Did any actual distributed/accelerate training run execute with the `.to()`-mutation code (i.e. did it ever corrupt optimizer state), or were all runs single-process notebooks/scripts? Determines urgency of the refactor.
2. Was the baseline (non-TRM) generate path ever used for evaluation? The double-strip bug would have silently truncated all baseline answers — if `04_vlm_qa_evaluation` ever produced numbers through that branch they are suspect (it errored on missing checkpoint, so likely no).
3. Should a real TRM (vendored ACT machinery or v2-style deep supervision) ever be ported, or is the current "hidden-state refiner" framing the intended final design? No artifact answers intent.

## Training / checkpoint / experiment-wiring audit (2026-09-19)

Scope: `src/training/{trainer,improved_trainer,callbacks}.py`, `src/utils/checkpoint.py`, `scripts/{train_alignment,train_vlm_accelerate,evaluate_vlm_accelerate,run_vlm_experiments.sh}`, driving configs + execution evidence in notebooks 01/04/05/07. Audit only — no code changed. **Resolves the suspected QA perceiver/checkpoint mismatch from the core-model pass — the mismatch does not occur in the script path (see §B.1).**

### A. Alignment training — actual path

```
notebooks/01_pixmo_vision_text_alignment.ipynb (24 executed cells)
→ load_config(pixmo_alignment[_1].yaml) → build_image_datasets_from_parquet (Pixmo Parquet loaders)
→ MultimodalAlignmentModel(cfg)  [vision CLIP frozen / Linear 1024→4096 / SimpleAttentionPooling trainable;
   text MPNet frozen / Linear 384→4096 trainable; MRL parameter-free; no fusion; no decoder]
→ ImprovedMultimodalTrainer(cfg, model, train_loader, val_loader, use_wandb=True)
→ AdamW(model.parameters(), lr, wd, betas=(0.9,0.95))   ← NOTE: NOT filtered by requires_grad
→ warmup+cosine LambdaLR; bf16 autocast; GradScaler(enabled=fp16=False) → inert
→ grad-accum 2 (config), clip 1.0; symmetric InfoNCE 1.0 + MRL 0.5 (sampled single dim)
→ validate(): val loss + i2t R@1/5/10 (single-direction; only I2T, not T2I)
→ save: checkpoint_latest.pt / checkpoint_best.pt / checkpoint_epoch_N.pt
   payload = {epoch, global_step, best_val_loss, model_state_dict (unwrapped), history, config dataclass,
              optimizer/scheduler/scaler state unless save_optimizer_state=False}
```

Findings:
1. **Optimizer includes frozen parameters.** `AdamW(self.model.parameters(), …)` (improved_trainer.py:96-102) is not filtered by `requires_grad`. Frozen CLIP/MPNet weights enter the optimizer; they receive no grads so they are never updated (AdamW skips `p.grad is None`), but optimizer state memory is inflated and this diverges from the QA script which does filter. Harmless-but-wasteful.
2. **Resume path is `strict=False` with keys logged** (improved_trainer.py:267-273): missing keys warned, unexpected keys counted at INFO. Optimizer-state load failure caught and logged. Sound.
3. `scripts/train_alignment.py` (the only CLI entry) remains **BROKEN EXPERIMENT WIRING**: it uses `MultimodalTrainer`, which calls `self.model(batch)` and indexes `outputs["loss"]` on a dataclass (trainer.py:55-56) — raises immediately. Also its `MultimodalDataModule` path cannot consume Parquet configs. All real alignment training happened in notebook 01.
4. `callbacks.py`/`utils/checkpoint.py` (`save_checkpoint` tmp+atomic-rename, `step_*.pt`) are consumed only by the broken `MultimodalTrainer` — effectively dead for the canonical path.

### B. VLM/QA training — actual path

```
run_vlm_experiments.sh production run: accelerate launch --num_processes 4
  train_vlm_accelerate.py --config trm_vlm_qa_qwen2.5-3b.yaml --batch_size 4 --gradient_accumulation_steps 4 --use_wandb
→ Accelerator(mixed_precision=bf16, DeepSpeedPlugin(zero_stage=2, accum=4, clip=1.0))
→ alignment_config = --alignment_config (DEFAULT: pixmo_alignment.yaml) with decoder=None, text_encoder=None
   → MultimodalAlignmentModel: vision = CLIP + Linear 1024→4096 + SimpleAttentionPooling, use_perceiver=FALSE
→ torch.load(checkpoint_best.pt) → load_state_dict(strict=False)  [silent — no missing/unexpected logging]
→ aligned_model.eval(); all params requires_grad=False; vision_encoder moved to accelerator device
→ encode_images(): vision_encoder(images, return_sequence=True) → sequence (B,577,4096) under no_grad
→ QwenDecoder(Qwen2.5-3B-Instruct, 4-bit nf4, LoRA r=64 q/k/v/o) → QwenVLM(vision_proj 4096→2048, use_trm_recursion=TRUE default)
→ optimizer = AdamW(filter(requires_grad, model.parameters()), lr=1e-4)  ← correctly filtered
→ warmup+cosine (floor 0.1); accelerator.prepare(model, optimizer, loaders, scheduler)
→ loss = QwenVLM.forward(...).loss (HF CE over answer tokens; question −100)
→ save: save_pretrained/checkpoint_best (unwrapped state_dict) + accelerator.save_state per epoch (DeepSpeed format,
   rotation keeps last) + checkpoint-final
```

Findings:
1. **QA-config `vision_encoder.use_perceiver: true` is dead config in the script path.** The script constructs the vision encoder exclusively from `--alignment_config` (default `pixmo_alignment.yaml`, `use_perceiver: false`); the QA YAML's vision block is only read for `projection_dim` (train_vlm_accelerate.py:272) and similar scalar fields. **The suspected perceiver/random-weights scenario therefore did not occur in script-driven runs** — the vision encoder was the attention-path architecture matching `checkpoint_best.pt`. (Notebook variants that build the model differently were not evidence-checked cell-by-cell; notebook 04 errored on a missing checkpoint and never trained.)
2. **`answer_mask` is accepted by `QwenVLM.forward` and never used** (trm_qwen_vlm.py:83 — parameter bound, no downstream reference). The training loop passes it every step. Consequence: `labels = cat(−100, answer_ids)` supervises **pad_token_id positions as real targets** in the CE loss, and no `attention_mask` is passed to Qwen for the text block. Training/val loss therefore includes padding supervision; plausibly contributes to the degenerate repetitive generation observed in notebook 05. CONFIRMED training-semantics bug.
3. Trainable-parameter map (QA path): Qwen base weights FROZEN+4-bit (not in optimizer ✓); LoRA adapters TRAINABLE+optimizer ✓; `vision_proj` TRAINABLE+optimizer ✓; TRM down/up_proj+layers+z_init TRAINABLE+optimizer ✓ (when use_trm; **`trm_norm` created trainable but never called → dead trainable param, still in optimizer with zero grads**); `vision_proj` gradient flows (vision encoder output under no_grad; projection trainable) ✓; frozen vision encoder excluded from optimizer ✓; alignment text encoder not instantiated ✓. Alignment path trainable map: vision projector+SimpleAttentionPooling and text projector trainable, both in optimizer (along with everything frozen — A.1); MRL parameter-free.
4. Distributed: 4-process launch + DeepSpeed ZeRO-2 plugin. The in-forward `.to()` mutations in QwenVLM are **no-ops on a homogeneous single-node 4×GPU run** (device/dtype already match after the first step), which is why no corruption was observed — but they remain DDP-unsafe: re-assigning `self.vision_proj` after `accelerator.prepare` desynchronizes module references across ranks and would break under any heterogeneous placement. Severity for actual runs: LOW (no-op in the executed configuration); structurally still must go.
5. Checkpoint resume: auto-detect `checkpoint-epoch-*` by `scheduler.bin` presence (completeness proxy) → `accelerator.load_state(strict=False / load_module_strict=False)`. Epoch parsed from directory name. Reasonable; strict=False again silent.

### C. Checkpoint audit

| Load site | Producer | Consumer arch | strict | Handling | Risk |
|---|---|---|---|---|---|
| notebook 07 ← attention `checkpoint_best.pt` | ImprovedMultimodalTrainer (notebook 01) | MultimodalAlignmentModel + `pixmo_alignment_1.yaml` | False, silent | none | LOW: `pixmo_alignment_1.yaml` is byte-identical to `pixmo_alignment.yaml` (diff verified empty) → architecture matches producer. |
| notebook 07 ← `perceiver_v2/checkpoint-best/pytorch_model/mp_rank_00_model_states.pt` | **DeepSpeed ZeRO run of the notebook-local PerceiverV3Model** | PerceiverV3Model (notebook-local) | False + key remap | silent | The Perceiver arm of the final table came from a DeepSpeed run of notebook V3 — still NOT the packaged `perceiver.py`. Modular Perceiver remains run-orphaned. |
| train_vlm_accelerate ← `pixmo_alignment/checkpoint_best.pt` | ImprovedMultimodalTrainer | attention-path vision encoder (script-enforced) | False, silent | not logged | Text-encoder/decoder keys unexpected (dropped); vision keys match; shape mismatches would still raise. LOW. |
| improved_trainer resume | itself | itself | False | missing warned, unexpected counted | Sound. |
| evaluate_vlm_accelerate ← VLM ckpt | train_vlm_accelerate | same construction path | False | silent | Consistent. |

1. `checkpoint_best.pt` schema: dict with `model_state_dict`, `config` (ExperimentConfig dataclass), epoch/step/best_val_loss — matches notebook 07's `ckpt.get('model_state_dict', ckpt)` read. ✔
2. `strict=False` at every load site without producer/consumer fingerprinting (no arch hash, no config echo) — provenance rests entirely on path naming.
3. No checkpoint artifacts exist in-repo (paths are cluster-absolute `/home/hice1/...`); correspondence claims rest on notebook code, not retrievable files.

### D. Experiment evidence

1. **Alignment**: notebook 01 (24 executed cells, ImprovedMultimodalTrainer, pixmo parquet) produced `checkpoint_best.pt`; notebook 07 evaluated it under the identical config. The 77.7/77.5% artifact chain is **config-consistent end-to-end** (provenance still limited by absent files/paths).
2. **Perceiver arm**: trained under DeepSpeed (mp_rank_00 state file), evaluated via notebook-local V3 — never the packaged module. Confirms and strengthens the prior Perceiver finding.
3. **VLM/QA production run** (`run_vlm_experiments.sh` #3): `trm_vlm_qa_qwen2.5-3b.yaml`, no TRM flags → **use_trm_recursion=True by default** — the production run was TRM-enabled, 4 processes, ZeRO-2. No result artifacts persist.
4. **Baseline (use_trm=false) generation**: the no_TRM notebook has **0 executed cells**; `run_vlm_experiments.sh` has no baseline eval entry; `04_vlm_qa_evaluation` errored on a missing checkpoint. **No evidence the baseline branch (with its double-strip bug) ever produced evaluation numbers.** The double-strip bug is confirmed-but-never-exercised.
5. Distributed execution: confirmed used (accelerate launch --num_processes 4). `.to()` mutation severity downgraded to LOW as above.

### E. Trustworthiness of reported experiment paths

| Path | Trustworthy? |
|---|---|
| Attention alignment training + final retrieval CSV | YES — config-consistent chain; normal loss semantics. |
| Perceiver negative table | Outcome trustworthy as a V3-notebook result; architecture attribution to the packaged module remains INVALID. |
| VLM/QA training loss/perplexity | QUALIFIED — losses include pad-token supervision (answer_mask bug), TRM-variant only; no artifact persists anyway. |
| VLM/QA generated-text metrics (val BLEU/ROUGE) | WEAK — only first 5 batches, exceptions swallowed (`except: pass`), generation via `inner_model.qwen.generate` bypassing QwenVLM.generate. |
| Anything citing paper Table VII BLEU/ROUGE | Still UNVERIFIED (unchanged). |

### F. Classification

| Component | Classification |
|---|---|
| `improved_trainer.py` | **KEEP** (REFACTOR LATER: unfiltered optimizer params; DDP guard OK) |
| `trainer.py` + `train_alignment.py` + `callbacks.py` + `utils/checkpoint.py` | **BROKEN EXPERIMENT WIRING** (CLI path; repair or retire in favor of the notebook-01 flow) |
| `train_vlm_accelerate.py` | **REFACTOR LATER** (works; silent strict=False; swallowed eval exceptions; dead config fields) |
| `answer_mask` non-consumption in QwenVLM | **REPAIR LATER** (confirmed training-semantics bug) |
| `trm_norm` dead trainable module | **REPAIR LATER** (apply or remove) |
| Alignment checkpoint chain (01→07) | **KEEP** (provenance-qualified) |
| Perceiver v2 DeepSpeed checkpoint + key-remap eval | **HISTORICAL** |
| `run_vlm_experiments.sh` | **CLEAN LATER** (comment-documented script; hardcoded run names) |

### G. Unresolved questions

1. Which exact `checkpoint_best.pt` (which epoch/run of notebook 01) fed the final CSV and the QA runs — single checkpoint or overwritten across sessions? Path evidence only; no file hash exists.
2. Did any QA run complete multiple epochs on the cluster (checkpoint rotation deleted evidence)? `training_progress.log`/W&B summaries are not in-repo.
3. Was DeepSpeed ZeRO-2 actually active at runtime (plugin passed, but requires the deepspeed env) or did it fall back to plain DDP? Matters only for rerun fidelity.

## Training-path low-risk fixes (2026-09-19)

Scope: masking semantics in `src/models/trm_qwen_vlm.py`, optimizer filter in `src/training/improved_trainer.py`, checkpoint diagnostics in `scripts/{train_vlm_accelerate,evaluate_vlm_accelerate}.py`. No other behavior touched.

### Masking semantics discovered (pre-fix trace)

`answer_mask` is created in `scripts/train_vlm_accelerate.py:collate_fn` (lines 109-115): `1 = real answer token, 0 = answer padding` — a **valid-token/padding mask**, NOT an answer-only loss mask. `PixmoQADataset` per-item `answer_mask` (1s only) was overwritten by this collate mask. Pre-fix `QwenVLM.forward` accepted `answer_mask` and **never read it**: labels were `cat(−100·question, answer_ids)` (padding supervised), and no attention mask was passed to Qwen for the text block. The collate answer padding and the question tokens both entered CE. Separation now applied: `answer_mask` drives BOTH (a) the text attention mask (padding semantics) and (b) label −100 filling (loss semantics); the question block remains −100 (no loss) as before — no intended-loss-token change.

### Fixes made

1. `trm_qwen_vlm.py` `forward` (question_ids+answer_ids branch): builds `attention_mask = cat(ones(question), answer_keep)`; labels become `cat(−100·question, masked_fill(answer_keep==0 → −100))`. Missing `answer_mask` falls back to all-valid (unchanged behavior for such callers).
2. `trm_qwen_vlm.py` manual TRM-branch loss: replaced the `logits.shape[1] > labels.shape[1]` heuristic (which mis-detected batch-length padding) with explicit prefix-aware label padding — labels are padded with −100 over the prefix region, so shapes always align and the prefix contributes nothing to CE.
3. `improved_trainer.py`: `AdamW((p for p in model_params if p.requires_grad), …)` — frozen CLIP/MPNet backbones no longer enter the optimizer.
4. Checkpoint diagnostics at script `strict=False` load sites (train + evaluate): missing/unexpected keys captured and logged at WARNING; all-match case logged at INFO; strictness unchanged.

### Validation performed (synthetic, no downloads; venv torch 2.9.1)

`/tmp/validate_training_fixes.py` — ALL PASSED:
1. Padded answer labels become −100 (per-batch variable answer lengths).
2. Attention-mask length = question+answer; padding entries 0, valid entries 1; with the prefix block the full mask and labels align (prefix −100 / ones).
3. Non-padding intended targets unchanged; questions remain −100; `CrossEntropyLoss` default `ignore_index=-100` verified equal to manual mean over valid positions.
4. Missing-`answer_mask` fallback = all-valid.
5. Optimizer filter: 2/4 tensors in optimizer; gradients flow only to the trainable module.
6. Checkpoint diagnostics: matching load reports no keys; producer-only keys (simulated decoder) captured as unexpected while load still succeeds; `strict=False` preserved.
7. `py_compile` + AST parse of all four modified files; edited regions verified present.

Full-module import in the v1 venv is blocked by missing `sentence_transformers`/`transformers` deps there (pre-existing environment gap, not a code issue); compile/AST checks + the synthetic logic checks cover the changed code paths.

### Intentionally deferred (unchanged per task constraints)

- `trm_norm` dead module; `confidence_threshold` stale arg; TRM architecture; decoder naming.
- Baseline generate double-strip bug (`outputs[:, prompt_len:]`).
- In-forward `.to()` mutations in QwenVLM.
- Broken legacy CLI (`trainer.py`/`train_alignment.py`/`callbacks.py`).
- Evaluation behavior (incl. the 5-batch BLEU/ROUGE subset and swallowed exceptions).
- Notebook load sites (07) — not scripts; untouched.

## Evaluation / inference subsystem audit (2026-09-19)

Scope: `src/evaluation/{metrics,benchmark,qa_metrics,simple_metrics,zero_shot,templates,explainability,__init__}.py`, `scripts/{evaluate_vlm_accelerate,evaluate_freeze_align,inference_vlm}.py`, notebook 07 (final results) + 05 (executed inference). Audit only — no code changed.

### 1. Alignment evaluation

Three separate retrieval-metric implementations exist:

| Implementation | Where | Math | Used for |
|---|---|---|---|
| Notebook 07 `compute_metrics` (cell 6, self-contained) | final CSV producer | per-dim truncate → F.normalize → sim → argsort → ranks (1-indexed) → R@1/5/10, MRR, MedR; i2t via `sim`, t2i via `sim.T`; diagonal targets | **THE final artifact** (`alignment_metrics_all_models.csv` + derived tables) |
| `src/evaluation/metrics.py:compute_retrieval_metrics` | module | normalize inside → sort → `ranks = (indices == targets).nonzero()` → R@K×100, mean/median rank, mAP@K, NDCG@K | `AlignmentBenchmark` / `evaluate_freeze_align.py` / `__init__` exports |
| `improved_trainer.validate` inline | training loop | matmul → argsort → R@1/5/10 (I2T only) | training-time val logging |

All three share the same semantics: cosine on L2-normalized embeddings, **diagonal-only ground truth** (paired protocol), i2t and t2i by transposing the similarity matrix. Correct but triplicated. `metrics.py` extras (mAP/NDCG) are computed but no artifact uses them.

Chain of trust (attention arm): notebook 01 checkpoint → notebook 07 attention eval under `pixmo_alignment_1.yaml` (byte-identical to `pixmo_alignment.yaml`, verified in training audit) → final CSV. **Trustworthy as paired-retrieval numbers**, same qualification as before (cluster-absolute paths, val split, N≈2984 inferred, no persisted N/provenance). Note notebook 07 embeddings ARE re-saved (`embeddings_val_attention.pt` etc. — cell 8), but those artifacts are also not in-repo.

`scripts/evaluate_freeze_align.py`: loads `checkpoint_best.pt` with **strict=True** (line 53 — the only strict load in the repo; would raise on any key mismatch), then `AlignmentBenchmark.run_full_evaluation` (retrieval + MRL dims — but `AlignmentBenchmark(model, device=device)` is constructed **without `mrl_dims`**, so the MRL branch silently never runs) + zero-shot CIFAR10/100 with try/except skip. Whole PixMo section wrapped in try/except that prints and continues. Config hard-coded to `pixmo_alignment.yaml`; checkpoint fallback `notebooks/checkpoints/...` → `outputs/pixmo_alignment/...`. Usable but never produced a retained artifact.

`zero_shot.py`/`templates.py`/`explainability.py`: no persisted result artifact found for any zero-shot or explainability run in-repo; CIFAR evaluation existed as notebook cells (07 cell 16) but outputs only transient. EXPERIMENTAL.

### 2. VLM / QA evaluation

`scripts/evaluate_vlm_accelerate.py` (the only batch QA evaluator):
1. **Calls `QwenVLM.generate`** with `--use_trm` **defaulting to False** → runs the **baseline branch** → **directly hits the confirmed double-strip bug** (`outputs[:, prompt_len:]` vs HF inputs_embeds-only-generated-ids). Any prediction would lose its first `prompt_len` (~577) tokens — i.e. essentially every answer truncated to garbage. The evaluator as written cannot produce meaningful EM/F1 with default flags. REPAIR required before any result from it can be trusted.
2. TRM path (`--use_trm`) would use the working TRM branch (returns generated-only ids), but with `confidence_threshold` ignored and O(T²) per-token recursion.
3. **Full val set** by default (`--max_test_samples` optional Subset of the first N — the *first* N, not random); greedy T=0.0, max_new_tokens=128; per-process predictions then `gather_object` (guarded by hasattr) → metrics on main process only. Saved to `--output_file` (default `eval_results.json`). **No such JSON exists in-repo** → no persisted VLM/QA eval result was ever produced/retained through this script.
4. No swallowed exceptions in the eval loop itself (unlike training-time val BLEU/ROUGE which keeps `except: pass` and only 5 batches — that remains a WEAK in-training signal, recorded previously).
5. Prompt handling: raw question text re-tokenized with the Qwen tokenizer (left-padded) — no chat template is applied anywhere (train or eval): questions are fed as raw token sequences. Consistent train/eval, but means the model was never trained with any instruct/chat formatting.
6. Metrics: `qa_metrics.py` EM/F1 correct (normalized, SQuAD-style); BLEU-4 crude-but-sane (no smoothing → 0 when any n-gram precision is 0); ROUGE-L F1 correct LCS. `simple_metrics.py` is a near-duplicate with **precision/recall orientation swapped** (ref/cand reversed vs qa_metrics; F1 value unaffected, p/r reported values would differ).

### 3. Inference (`scripts/inference_vlm.py`)

1. Model construction path is coherent with training (alignment config default → vision-only load, QwenDecoder + QwenVLM, same `get_image_transforms(is_training=False)` preprocessing — train/eval preprocessing consistent; tokenizer identical).
2. **Checkpoint format mismatch breaks it**: training saves `checkpoint_best` / `checkpoint-epoch-N` as **directories** (`save_pretrained` / `accelerator.save_state`), but inference does `torch.load(checkpoint_path)` (line 149) — `torch.load` on a directory raises IsADirectoryError. Auto-resolved 'latest'/'best' therefore always fail; only an ad-hoc `.pt` file would work, and training never produces one. Inference cannot load anything the training pipeline saved. BROKEN.
3. `model.to(model_device)` (line 115) on QwenVLM containing a 4-bit/`device_map="auto"` decoder — QwenVLM has **no** quantization-safe `.to()` override (only `MultimodalAlignmentModel` does) → would attempt to move quantized/meta params. Risk of crash or silent corruption with the default configs (`load_in_4bit: true`).
4. Minor: duplicate `transform = get_image_transforms(...)` lines (158-159); default config `trm_vlm_qa_qwen1.5.yaml` while production used 2.5-3b; `num_key_value_heads`/`intermediate_size` kwargs forced into QwenDecoder's config-override hack; generation temperature 0.2 sampling (vs eval greedy).
5. Notebook 05 remains the only executed inference evidence (TRM branch, degenerate outputs — previously recorded).

### 4. Classification

| Component | Classification |
|---|---|
| `metrics.py` (`compute_retrieval_metrics` + MRL perf + sim stats) | **KEEP** (canonical module; retire duplicates into it later) |
| Notebook 07 self-contained metrics | **HISTORICAL** (artifact producer; do not reuse as code) |
| `improved_trainer.validate` inline retrieval | **MERGE LATER** (third duplicate; keep only logging behavior) |
| `benchmark.py` (`AlignmentBenchmark`) | **CLEAN LATER** (works; default `mrl_dims=[]` silently disables MRL; embedding collection solid) |
| `evaluate_freeze_align.py` | **CLEAN LATER** (strict=True load inconsistency; broad try/except; no artifact) |
| `qa_metrics.py` | **KEEP** (correct EM/F1/BLEU/ROUGE; BLEU lacks smoothing — acceptable, document) |
| `simple_metrics.py` | **REMOVE LATER** (duplicate; swapped p/r orientation) |
| `zero_shot.py` / `templates.py` / `explainability.py` | **EXPERIMENTAL** (no retained result artifacts) |
| `evaluate_vlm_accelerate.py` | **REPAIR LATER** (baseline-generate double-strip on default path; otherwise sound loop) |
| `inference_vlm.py` | **REPAIR LATER** (checkpoint-format mismatch; unguarded `.to()` on quantized decoder) |
| Training-time val BLEU/ROUGE (5 batches, except-pass) | **REPAIR LATER** (already recorded; confirm here) |

### 5. Unresolved questions

1. Did any `evaluate_vlm_accelerate.py` run ever complete off-cluster and write an `eval_results.json` not present in the repo? If so, its default-flags baseline predictions would be double-strip-corrupted — such numbers should be discarded if they ever surface.
2. Is the intended QA eval protocol with-chat-template (Qwen2.5 instruct) or raw-question? Current code trains and evaluates raw — but instruct models typically need templates; intent cannot be recovered from the repo.

### Recommended next step
Repair the two broken QA-facing entry points as one commit: fix the double-strip in `QwenVLM.generate` baseline branch (making `evaluate_vlm_accelerate.py` trustworthy), and give `inference_vlm.py` a directory-aware checkpoint loader plus a quantization-safe `.to()` path (or reuse pattern from `MultimodalAlignmentModel.to`).

## Notebooks / scripts classification (2026-09-19)

Scope: `notebooks/*.ipynb`, `notebooks/*.py` helpers, `notebooks/ddsp_scripts/`, `notebooks/archive/`, `scripts/`. Classification by filename, execution counts (cell-level exec/error scan), imports, and overlap with audited `src/` modules. No code changed. Historical v0/v1/v2 notebooks NOT re-inventoried (covered by the data audit's salvage list).

### 1. Notebook classification (all execution counts verified)

| Notebook | Exec cells / errors | Task | Classification | Evidence retained? | Unique code? | Keep? |
|---|---|---|---|---|---|---|
| `01_pixmo_vision_text_alignment.ipynb` | 24 / 0 | Alignment training (ImprovedMultimodalTrainer, pixmo parquet) → produced `checkpoint_best.pt` | **CANONICAL EXPERIMENT** | YES — checkpoint fed the final CSV chain (no artifact in-repo) | No — imports `models`, `data.dataset_builder` from src; notebook-only: training-loop wiring + `retrieve_top_k` (superseded by improved_trainer + metrics.py) | **YES** — provenance of the 77.7/77.5% result |
| `02_perceiver_mrl_vision_text_alignment.ipynb` | 2 / 1 | Attempted packaged-Perceiver MRL training; aborted after 2 cells | **DEBUGGING / HISTORICAL** | Training-curves PNGs in `notebooks/outputs/perceiver_mrl_alignment/` (provenance unclear — likely from ddsp_scripts runs, not this notebook) | No | **ARCHIVE** |
| `03_trm_vlm_qa_training.ipynb` | 3 / 1 | Early TRM-VLM QA training; **notebook-local `QwenVLM`, `TRMVLMWithConfidence`, `TinyVLMDecoder` classes** | **HISTORICAL / SUPERSEDED** | No | YES — earliest QwenVLM variant, but superseded by `src/models/trm_qwen_vlm.py` (04-era). No retained results | **ARCHIVE** (but see salvage note below) |
| `03_trm_vlm_qa_training_FIXED.ipynb` | 18 / 1 | TRM-VLM QA training (imports `models.trm_qwen_vlm` from src) | **ABLATION** (TRM-enabled production run lineage) | Weak — load/epoch traces only; no committed results | Notebook-only: `collate_fn` with answer_mask construction (already mirrored in train_vlm_accelerate.py) | **YES** — executed evidence for E-experiment |
| `03_trm_vlm_qa_training_MODULAR.ipynb` | 16 / 0 | QA captioning variant (evaluate_caption; no TRM import) | **ABLATION** | No results retained | No unique logic | **ARCHIVE** |
| `03_trm_vlm_qa_training_no_TRM.ipynb` | **0 / 0** | Baseline (non-TRM) QA training | **HISTORICAL / GENERATED** | None — never executed (confirms training-audit finding that the baseline arm never ran) | No | **ARCHIVE** |
| `04_pixmo_dataset_eda.ipynb` | 13 / 0 | PixMo + QA EDA (splits, n-grams, aspect ratios); wrote `notebooks/eda_outputs/*.png` | **DATA PREPARATION / ANALYSIS** | YES — eda_outputs PNGs | No unique modeling code | **YES** (analysis value) |
| `04_vlm_qa_evaluation.ipynb` | 2 / 1 | VLM QA evaluation — **errored on missing VLM checkpoint** | **DEBUGGING** | No (the FileNotFoundError IS the evidence) | No | **ARCHIVE** |
| `05_alignment_evaluation.ipynb` | 4 / 1 | Alignment retrieval eval attempt (src.evaluation.benchmark; cluster paths) | **HISTORICAL / SUPERSEDED** by 07 | Partial | No | **ARCHIVE** |
| `05_vlm_qualitative_inference.ipynb` | 10 / 0 | Qualitative VLM inference (TRM branch; degenerate outputs) | **ANALYSIS / EVALUATION** | YES — the ONLY executed inference evidence | No — imports from src; loops are notebook-only glue | **YES** |
| `07_alignment_final_report.ipynb` | 8 / 0 | Final attention-vs-Perceiver report + multiscale tables → `outputs/alignment_eval_final/` CSVs | **CANONICAL EXPERIMENT** | YES — the final CSV artifact chain | **YES — notebook-local `PerceiverV3Model` + full Perceiver stack (FeedForward/PerceiverAttention/PerceiverLayer/PerceiverResampler) + `EvalConfig` + `compute_metrics`; NONE of it exists in `src/`** | **YES** — but flag for salvage (see §5) |
| `archive/04_alignment_evaluation.ipynb` | — | Superseded eval notebook | **HISTORICAL** | No | No | Already archived; leave |

### 2. Notebook-support scripts inside `notebooks/` (top level)

`merge_notebooks.py`, `modify_notebook{,_v2,_v3,_v4,_v5}.py`, `patch_eval_notebook.py` — all are **GENERATED / TEMPORARY** one-shot notebook JSON editors (file mtime cluster Dec 10 2025). No unique implementation logic. → archive/remove later.

### 3. `notebooks/ddsp_scripts/` — DeepSpeed/Accelerate launcher-era scripts (IMPORTANT, previously unclassified)

| Script | What it is | Classification |
|---|---|---|
| `train_perceiver.py` (261 L) | Accelerate trainer importing packaged `models.alignment.MultimodalAlignmentModel` | **ABLATION** (packaged-perceiver lineage; notebook 02 abandoned path) |
| `train_perceiver_v2.py` (0 lines) | Empty file | **GENERATED / TEMPORARY** |
| `train_perceiver_v3.py` (736 L) | Self-contained Accelerate+DeepSpeed trainer with **its own inline `MultimodalAlignmentConfig` + `MultimodalAlignmentModel` + Perceiver stack** (d_text=384 MiniLM) — this is the DeepSpeed ZeRO run family that produced the `perceiver_v2/checkpoint-best` mp_rank state consumed by notebook 07 | **ABLATION — CONTAINS THE ONLY EXECUTED PERCEIVER TRAINING CODE** (kernel of notebook 07's PerceiverV3Model, pre-eval) |
| `train_pixmo_alignment.py` (716 L) | Standalone W&B trainer importing packaged `MultimodalAlignmentModel` | **ABLATION** (pre-notebook-01 CLI-era training attempt) |
| `debug_gradients.py` (102 L) | Gradient-flow inspection | **DEBUGGING** |

These are the missing provenance link for the Perceiver arm: DeepSpeed-run training code lives HERE and in notebook 07's eval-time `PerceiverV3Model`, never in `src/`. → **retain** (move under a scripts/experiments destination), do not delete.

### 4. `scripts/` classification (grouped)

**Real training entry points**
- `train_alignment.py` — BROKEN experiment wiring (audited: trainer.py interface mismatch). Retain as entry point **only after repair or retirement** in favor of notebook-01/improved_trainer flow.
- `train_vlm_accelerate.py` — KEEP (canonical QA training; REFACTOR LATER findings already recorded).
- `train_instruction.py` — imports broken `MultimodalTrainer` (same wiring defect family as train_alignment.py); never produced artifacts → **EXPERIMENTAL/likely obsolete**.
- `run_vlm_experiments.sh` — KEEP (documented production run history; CLEAN LATER).

**Real evaluation/inference entry points**
- `evaluate_vlm_accelerate.py` — KEEP (REPAIR LATER: double-strip default path).
- `evaluate_freeze_align.py` — KEEP (CLEAN LATER).
- `inference_vlm.py` — KEEP (REPAIR LATER: checkpoint-format break).

**Dataset/download utilities**
- `download_datasets.py`, `download_pixmo_audio.py` — KEEP (wrap src/data/downloader).

**Notebook-generation utilities** (all GENERATED/TEMPORARY dev artifacts; safely removable once notebooks are curated)
- `generate_eval_notebooks.py`, `create_eval_notebook.py` (stale-import history, fixed in data pass), `create_alignment_eval_notebook.py` (hard-coded cluster paths), `add_checkpointing_to_notebook.py`, `add_test_split_to_notebook.py`, `insert_notebook_cell.py`, `update_notebook_structure.py`, `finalize_notebook_updates.py`, `fix_eval_notebook.py`, `fix_notebook_transforms.py`, `fix_val_transforms.py`, `fix_collate_fn.py`, `fix_training_loop.py`, `force_replace_qwenvlm.py`, `optimize_gpu_usage.py`, `map_notebook.py` (has the known compileall f-string SyntaxError).

**Debug/inspection**
- `debug_ckpt_loading.py`, `inspect_checkpoint_simple.py`, `inspect_checkpoint_v2.py`, `debug_perceiver.py` — one-time diagnostics; archive/remove later.

**No dataset-repair, reproducibility, or patch scripts found beyond the above** — the `fix_*` family here is uniformly notebook-editing, not code repair.

### 5. Notebook-only salvageable code (§4 of task)

| Notebook/script path | Component | Why it matters | Extract to src/? |
|---|---|---|---|
| `notebooks/07_alignment_final_report.ipynb` (cell 4) + kernel in `ddsp_scripts/train_perceiver_v3.py` | `PerceiverV3Model` + Perceiver stack (`PerceiverLayer` w/ self-attention, per-attn pre-LN) + `EvalConfig` | THE implementation that produced the recorded Perceiver negative result and the DeepSpeed checkpoint consumed by notebook 07. Never in `src/`; packaged `perceiver.py` lacks self-attention/mask (encoder-audit finding) | YES — if Perceiver work resumes: merge V3 architecture into packaged `perceiver.py` (design-family source), reconciling 512-d labeling + MiniLM text confound |
| `notebooks/07_alignment_final_report.ipynb` (cell 6) | Self-contained `compute_metrics` (final-CSV producer) | Duplicate of `src/evaluation/metrics.py` semantics — do NOT extract; recorded as HISTORICAL artifact producer only | NO |
| `notebooks/01_pixmo_vision_text_alignment.ipynb` | Training-loop wiring for ImprovedMultimodalTrainer (data→trainer glue) | The only executed alignment-training path; `scripts/train_alignment.py` is broken. Useful as the template for repairing the CLI | YES-LATER — the repair spec for train_alignment.py should be derived from this notebook's wiring |
| `notebooks/03_trm_vlm_qa_training.ipynb` | Earliest notebook-local `QwenVLM`/`TinyVLMDecoder` variants | Superseded by `src/models/trm_qwen_vlm.py`; only provenance value (evolution of the prefix-injection design). No extraction | NO (archive with history) |

### 6. Likely final category structure (proposal only — NOT executed)

```
edge_glass_modular/
├── notebooks/
│   ├── experiments/    # 01 alignment training, 03_FIXED, 07 final report
│   ├── analysis/       # 04 EDA, 05_vlm_qualitative_inference
│   └── archive/        # 02, 03 (original), 03_MODULAR, 03_no_TRM, 04_vlm_qa_evaluation, 05_alignment_evaluation + modify_*/patch_* helpers
├── scripts/
│   ├── train/          # train_vlm_accelerate.py (+ repaired train_alignment.py), run_vlm_experiments.sh
│   ├── eval/           # evaluate_vlm_accelerate.py, evaluate_freeze_align.py, inference_vlm.py
│   ├── data/           # download_datasets.py, download_pixmo_audio.py
│   └── experiments/    # notebooks/ddsp_scripts/* (perceiver/pixmo trainers) — or src-adjacent if salvaged
└── (removed later)     # notebook-generation + debug/inspect one-shots
```

### 7. Ambiguities requiring user input

1. `ddsp_scripts/train_perceiver_v3.py` vs notebook 07 `PerceiverV3Model`: keep BOTH as experiment provenance, or extract the V3 architecture into `src/encoders/perceiver.py` now (it is the executed design) and archive both? Ledger default: keep both, extract later.
2. `train_instruction.py` + `train_alignment.py`: repair the broken CLI family, or retire the CLI path and declare notebook-01/improved_trainer the canonical entry (ledger default: retire-then-rebuild)?
3. Notebook-generation scripts (`create_*notebook`, `fix_*`, `modify_*` family): delete outright at restructure, or move to `archive/`? They have zero unique logic.
4. `notebooks/wandb/` dir (46 run dirs) inside notebooks/ — previously out of scope; confirm it can be excluded/ignored at restructure rather than relocated.
