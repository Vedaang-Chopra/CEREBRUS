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
