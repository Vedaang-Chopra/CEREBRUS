> **Historical audit (August 2026) — superseded in part.** This is the repository audit
> produced 2026-08-17 (`4113a26`), before the release-verification and regression-test
> work. Several release blockers and cleanup items documented here were subsequently
> resolved (see `AUDIT_LEDGER.md`). `AUDIT_LEDGER.md`, the current source, `tests/`, and
> the current README are authoritative for the present state. This report is retained for
> its unique architecture/metrics provenance evidence, not as a current verdict.

# CERBERUS code, claims, and release audit

**Audit date:** 2026-08-17  
**Scope:** `CEREBRUS.pdf`, top-level project documentation, `architectures/`, the
`edge_glass_modular/` package, notebooks, configurations, committed outputs, and
local W&B/run metadata. This is an evidence audit, not a retraining attempt.

## Executive assessment

CERBERUS has a real and reasonably modular vision-text alignment implementation:
frozen CLIP and sentence-transformer backbones feed trainable projections,
attention pooling or a Perceiver branch, nested prefix embeddings, and a
symmetric contrastive/MRL objective. The checked-in final retrieval table supports
the paper's attention-model and 128-dimension retrieval values, and it also
supports a near-random Perceiver result. However, the paper's strongest
experimental evidence is not reproducible from this repository as shipped. The
final evaluator depends on absolute paths and unavailable checkpoints, silently
permits partial checkpoint loads, and defines a separate Perceiver model inside a
notebook. The reported TRM ROUGE-L comparison has no surviving committed result
artifact. The project is therefore **not ready for arXiv or an open-source release
without an evidence/reproducibility cleanup**.

## Phase 1 hypothesis tested

The paper presents CERBERUS as a lightweight, frozen-encoder alignment front-end:
CLIP (vision), SBERT (text), and optional Whisper (audio) produce features that
are compressed by attention pooling by default or a Perceiver Resampler ablation,
then aligned with contrastive loss and multi-scale Matryoshka prefixes from 4096
to 128 dimensions. These latents are intended to condition Qwen and a tiny
recursive decoder (TRM). Its testable claims are that PixMo attention pooling
reaches about 78% R@5 at 4096-d, that the 128-d prefix remains useful at about
64% R@5, that the Perceiver branch is a negative result, and that the visual TRM
improves ROUGE-L over a text-only baseline. The architecture diagrams agree with
this broad two-stage story, but some depict a Qwen-1.5B/4096-to-1536 Stage-2
bridge and others depict different Perceiver designs, so they are not a single
versioned experimental specification.

## Architecture match/mismatch table

| Paper / documentation claim | Code and artifact evidence | Verdict |
|---|---|---|
| Frozen CLIP, SBERT, and optional Whisper encoders | `src/encoders/vision.py:102-104`, `text.py:67-69`, and `audio.py:74-76` set the backbone parameters `requires_grad=False`; all three modules exist. Main final retrieval uses vision/text only. | **Matches architecture; audio result claims remain unverified.** |
| Attention pooling is the default main vision-text compression backend | `vision.py:143-159` implements attention pooling for the non-Perceiver path. `configs/pixmo_alignment.yaml:27-32` selects it. The final CSV labels its successful rows `attention`. | **Matches.** |
| Perceiver compression is an interchangeable packaged backend | `vision.py:112-140` packages an MLP adapter, `PerceiverResampler`, mean pooling, and final projection. But `07_alignment_final_report.ipynb:166-188` defines a different local `PerceiverV3Model`; it is the model evaluated in the final table. | **Mismatch for the reported ablation.** The code package and final experiment are not the same implementation. |
| Packaged Perceiver has the documented cross-attention + self-attention stack | `src/encoders/perceiver.py:96-111` has cross-attention plus feed-forward only. The notebook-local V3 model has cross- and self-attention. | **Mismatch.** Documentation/diagrams describe more than the shipped reusable Perceiver module implements. |
| MRL learns usable nested 4096-to-128 embeddings | `src/encoders/mrl.py:48-80` returns L2-normalized prefixes of the 4096-d projected vector and `models/losses.py:154-167` supervises a sampled shared dimension. | **Substantively matches, with terminology caveat.** `MatryoshkaProjection` contains no learned projection; it slices the encoder's already learned projector output. |
| The same latent conditions Qwen and TRM through prefix tokens | `models/alignment.py:120-167, 268-298`, `decoders/qwen.py:133-161`, and `decoders/trm.py:285-323` implement a learned prefix-token bridge. `models/trm_qwen_vlm.py:8-74` also implements a distinct Qwen-plus-optional-TRM-recursion design. | **Partially matches.** Two incompatible decoder paths exist; the paper must identify which one produced each reported result. |
| Lightweight TRM is the reported decoder | A standalone randomly initialized `TRMDecoder` exists in `decoders/trm.py`. The active modular QA notebook instead reports a Qwen-2.5-3B model plus optional recursive layers (`03_trm_vlm_qa_training_MODULAR.ipynb` execution output). | **Mismatch / ambiguous provenance.** The paper's "TRM" needs an exact implementation and checkpoint identifier. |
| Offline-first PixMo-Cap/PixMo-QA pipeline | Parquet dataset classes exist in `data/dataset_builder.py:102-194,268-379`; configs hardcode the author's cluster data paths. No data manifest, downloader-to-Parquet recipe, or released data/checkpoint is present. | **Partial implementation, not reproducible release infrastructure.** |
| Production-ready DDP training, checkpointing, and recovery | `training/improved_trainer.py` implements these features and its notebook was executed. The advertised CLI calls `training/trainer.py`, whose loop calls `self.model(batch)` and indexes a dataclass as `outputs["loss"]` (`trainer.py:52-56`), neither matching `MultimodalAlignmentModel.forward`. | **Mismatch; release-blocking entrypoint defect.** |
| Retrieval evaluation is symmetric Recall@K over paired PixMo rows | `evaluation/metrics.py:51-102` normalizes embeddings, uses diagonal pairs as positives, and reports both directions. The final notebook independently implements equivalent ranking at `07_alignment_final_report.ipynb:555`. | **Matches the stated protocol, conditional on one positive caption/image and retained row order.** |

### Module-by-module audit

| Area | What is implemented | Match to paper/docs |
|---|---|---|
| `src/encoders/vision.py`, `text.py` | Frozen HF backbones, 4096-d trainable projections, L2 normalization, and MRL prefixes; vision supports attention pooling and Perceiver. | **Main vision-text design matches.** Actual main config uses `all-mpnet-base-v2`, while the paper says MiniLM; the final Perceiver evaluator uses MiniLM. State the backbone per experiment. |
| `src/encoders/audio.py` | Frozen Whisper, projection, optional Perceiver and MRL. | **Partial.** There is no audited audio result artifact. It applies MRL before normalizing the pooled vector (`audio.py:161-167`), unlike vision/text and the claimed common normalization pipeline. |
| `src/encoders/perceiver.py`, `pooling.py`, `mrl.py` | Functional tensor modules with expected shapes and nested-prefix MRL. | **Mixed.** Packaged Perceiver is not the V3 implementation used in final evaluation; MRL is prefix slicing, not an extra learned head. |
| `src/models/alignment.py`, `projector.py`, `fusion.py`, `losses.py` | Bimodal/trimodal loss orchestration, fusion options, LLM prefix projector, InfoNCE/MRL losses. | **Mostly matches.** It instantiates a decoder whenever config provides one, including pure alignment config, creating unnecessary Qwen dependency/memory for the claimed retrieval workflow. |
| `src/models/trm_qwen_vlm.py`, `src/decoders/trm.py`, `qwen.py` | Qwen LoRA/quantization bridge, custom TRM transformer, and a separate Qwen-plus-recursion wrapper. | **Architecture exists, experiment identity unclear.** Debug `print`s remain in `qwen.py:64-71`. |
| `src/training/trainer.py`, `improved_trainer.py`, `callbacks.py` | A more complete notebook-oriented trainer plus a broken minimal CLI trainer. | **Does not meet "production-ready" claim.** Use one tested trainer behind the canonical command. |
| `src/evaluation/metrics.py`, `benchmark.py`, `qa_metrics.py`, `zero_shot.py`, `simple_metrics.py` | Retrieval/MRL metrics, benchmark wrapper, handwritten QA scores, and zero-shot utilities. | **Retrieval matches.** `qa_metrics.py:126-137` returns ROUGE-L multiplied by 100, while Table VII reports decimal values, so metric scale/protocol must be reconciled. |
| `src/data/dataset_builder.py`, `datamodule.py`, `transforms.py`, `downloader.py` | Parquet PixMo datasets, transforms, and web/HF download helpers. | **Partial.** Required runtime packages `requests` and `soundfile` are not declared; no validation/test dataloaders are exposed by `datamodule.py`. |
| Notebooks `01`-`07` | Executed training/evaluation history exists, but there are duplicate/fixed/modular/no-TRM variants and stale references. | **Not a clean pipeline.** README names a nonexistent `02_pixmo_vision_text_alignment.ipynb`; actual notebooks use a different numbering scheme. |

## Metrics verification

"Verified" below means a committed result artifact numerically supports the
claim. It does **not** mean the artifact can be independently rerun from the
repository: the source data and checkpoints used by the final notebook are not
committed.

| Claimed number | Evidence path | Verified? | Audit finding |
|---|---|---:|---|
| PixMo 4096-d R@5 about 78% | `edge_glass_modular/outputs/alignment_eval_final/alignment_metrics_all_models.csv`, attention/4096 rows | **Yes, artifact-backed** | I2T R@5 = 77.681%; T2I R@5 = 77.547%. `07_alignment_final_report.ipynb` saved the table after evaluating `split='val'`. |
| 128-d R@5 about 64% after 32x compression | Same CSV, attention/128 rows; `table_multiscale_combined.csv` | **Yes, artifact-backed** | I2T = 62.936%; T2I = 64.042%. This is a 4096-to-128 (32x) prefix truncation. |
| TRM ROUGE-L 0.24 vs text-only 0.08 | Paper Table VII only; no committed CSV/JSON/notebook output/usable W&B summary containing 0.2377 or 0.0819 | **No** | `qa_metrics.py` provides scoring code, but no auditable predictions, targets, checkpoint, exact split, or result artifact survives. Remove the numbers or regenerate and commit a provenance bundle. |
| Perceiver Resampler degraded retrieval | Same final CSV, perceiver/4096 rows; `table_4096_detailed.csv`; executed final notebook | **Yes for observed negative outcome; no for packaged-architecture attribution** | I2T R@5 = 0.201%; T2I = 0.168%, consistent with the paper. The final notebook's local Perceiver is only 512-d (`PerceiverV3Model`), so calling this a 4096-d packaged-Perceiver result is inaccurate. |

Additional provenance issues:

- `07_alignment_final_report.ipynb` points to uncommitted absolute PixMo and
  checkpoint paths (`:82-88`) and loads both models with `strict=False`
  (`:196,206`). The CSV has no checkpoint hash, commit SHA, dataset fingerprint,
  row count, seed, or environment lockfile.
- The notebook execution output says 16 invalid validation images were removed.
  The final CSV does not persist the final sample count, so the evaluation
  population cannot be identified exactly from the artifact.
- `outputs/alignment_evaluation_results.csv` reports conflicting R@5 values
  (85.69/86.80). It should be archived, labelled as a different run, or removed
  from release assets to avoid accidental citation.

## Open-source readiness checklist

### Release blockers

- [ ] **Replace all personal/cluster paths.** `configs/pixmo_alignment.yaml`,
  `configs/perceiver_mrl_alignment.yaml`, and all QA configs embed
  `/home/hice1/...` paths. `07_alignment_final_report.ipynb:82-88` embeds the
  same paths and paths to absent checkpoints. Use relative configuration, CLI
  arguments, environment-variable defaults, and a documented data directory.
- [ ] **Make one canonical command work.** `README.md:83` uses unsupported
  `--dataset`/`--num_samples` flags; `scripts/download_pixmo_audio.py:22-26`
  requires `--pixmo-out` and `--audio-out`. README also references missing
  `trimodal_qwen.yaml`, `trimodal_trm.yaml`, and
  `02_pixmo_vision_text_alignment.ipynb` (`README.md:40,53,97,103,109`).
- [ ] **Repair or retire the default trainer.** The public
  `scripts/train_alignment.py` uses the incompatible `MultimodalTrainer`.
  Add a smoke test that runs one CPU batch through the released entrypoint.
- [ ] **Release result provenance.** Publish dataset build manifest/split hashes,
  config copies, seed, package and CUDA lockfile, exact checkpoint hashes (or a
  download manifest), evaluator command, final sample count, and raw metric
  output. Fail on missing or unexpected checkpoint keys instead of `strict=False`.
- [ ] **Resolve Perceiver identity.** Move the V3 architecture into the package
  or re-evaluate the packaged resampler. Label the output dimension honestly;
  the current final Perceiver is 512-d while the table is titled 4096-d.
- [ ] **Reproduce the decoder experiment.** Commit a non-sensitive metric JSON/
  CSV and prediction sample, identify the decoder path and baseline, specify
  ROUGE implementation/scale, and link both results to immutable checkpoints.
- [ ] **Add tests and CI.** No project tests, pytest configuration, or CI workflow
  were found under `edge_glass_modular/`. At minimum test config loading,
  metrics, MRL nesting, checkpoint compatibility, dataset collation, and one
  offline forward pass.

### Required cleanup before public release

- [ ] Add missing runtime dependencies to `pyproject.toml`: `requests`,
  `soundfile`, `matplotlib`, `seaborn`, and `scikit-learn`; add `nbformat` if
  notebook-generation scripts are retained. Pin tested version ranges and
  provide a reproducible lock/conda environment. The existing metadata only
  declares the first-party ML stack (`pyproject.toml:11-28`).
- [ ] Delete, archive, or move notebook-editing/debug scripts out of the public
  runnable surface: e.g. `scripts/finalize_notebook_updates.py`,
  `update_notebook_structure.py`, `fix_*`, `debug_*`, and checkpoint inspectors.
  Several embed absolute paths and mutate notebooks.
- [ ] Remove debug prints from production modules (`decoders/qwen.py:64-71`),
  and replace library-level prints in benchmark/explainability/data code with
  structured logging or optional verbosity.
- [ ] Reconcile stale documentation. `PROJECT_SUMMARY.md` and
  `IMPLEMENTATION_GUIDE.md` still describe template/TODO components while
  `FINAL_STATUS.md` and README call the package complete/production-ready.
- [ ] Consolidate notebooks to a numbered, documented sequence, remove duplicate
  `03_*` variants, and clear outputs from release notebooks unless the output is
  a deliberately versioned result artifact.
- [ ] Add a security policy and `.gitignore`; document that W&B/HF credentials
  are supplied through environment/login mechanisms. The static scan found no
  hard-coded API key/token pattern in text source, but current W&B logs and
  user-specific run metadata should not be released without review.

## Recommended paper updates

1. **Narrow the central claim.** Present CERBERUS as a *vision-text alignment
   prototype with an audited attention-pooling result*, not a production-ready
   multimodal/edge VLM. State that audio and tri-modal alignment are incomplete.
2. **State exact experimental identities.** Name the attention checkpoint/config,
   text backbone, PixMo split size after invalid-row removal, seed, and evaluation
   command. Do not mix MPNet, MiniLM, and CLIP-text configurations under one
   undifferentiated SBERT claim.
3. **Correct the Perceiver table and narrative.** Report it as a 512-d
   notebook-local V3 Perceiver ablation unless it is re-run with the packaged
   4096-d implementation. The negative outcome is valuable, but it needs a
   faithful architecture and reproducible provenance.
4. **Remove or qualify Table VII until rerun.** The ROUGE-L values lack evidence
   and the decoder naming is ambiguous. Replace the table with an explicitly
   preliminary, non-comparative statement or regenerate the experiment with
   released artifacts and a pre-registered metric protocol.
5. **Avoid unsupported performance language.** Remove "production-ready,"
   "full crash recovery," and edge-efficiency conclusions until the public CLI,
   environment, data pipeline, checkpoints, and hardware measurements work from
   a clean clone. Avoid comparisons to prior work that use different datasets or
   protocols except as clearly labelled qualitative context.
6. **Make the MRL description precise.** Say that the model trains a 4096-d
   projected embedding with losses on normalized prefix slices; do not imply a
   separate learned Matryoshka projection cascade unless one is implemented.

## Validation performed

- Read the requested documentation in order, extracted and visually reviewed the
  14-page paper, and reviewed all `architectures/` image assets (architecture,
  MRL, Perceiver-vs-MLP, decoder, training, and loss figures).
- Reviewed all requested source modules, configs, notebooks `01` through `07`,
  output tables, and available run/W&B metadata.
- Ran `python3 -m compileall -q edge_glass_modular/src edge_glass_modular/scripts`
  successfully.
- Loaded every shipped YAML configuration through `load_config` successfully.
- Performed text-only hard-coded-path, TODO/debug-print, dependency, and
  credential-pattern scans. No API key/token literal was found by the scan.

No retraining, model downloads, checkpoint reconstruction, or source-code changes
were performed. Existing uncommitted/deleted legacy-worktree changes were left
untouched.
