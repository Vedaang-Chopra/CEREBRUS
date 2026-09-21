# PROJECT_RESUME_CONTEXT — CERBERUS / Edge Glass

**Last audited: 2026-09-21** (documentation reconciliation pass on `main`, following the architecture-cleanup series that ended at commit `ca641a8`)  
**Evidence status: repository-audited; resume claims must derive from this file or newer verified evidence.**

> **Audit rule:** Persisted experiment artifacts and current code were treated as stronger evidence than project prose, paper claims, or old resume-like text. `VERIFIED_IMPLEMENTED` means source exists and is integrated enough to inspect/compile or load configuration; it does **not** by itself prove a successful production run. `VERIFIED_EXPERIMENT` means an executed notebook/output artifact persists. Repository artifacts do not include the data, model checkpoints, source commit SHA, seed, or environment needed to independently reproduce the main results.

---

## 1. Project Identity

| Field | Audited value | Evidence / confidence |
|---|---|---|
| Project name | **CERBERUS: Edge Glass Assistant** | Top-level `README.md`; paper title. High. |
| Alternate names | CERBERUS; Edge Glass; Edge Glass Modular; `edge-glass-modular` package | `README.md`, `edge_glass_modular/pyproject.toml`. High. |
| Repository | `Vedaang-Chopra/Edge-Glass` (paper URL) / local repository root | `CEREBRUS.pdf`; Git metadata. High for repository ownership; do not claim public release readiness. |
| Institution | Georgia Institute of Technology | `CEREBRUS.pdf:5-9`. High. |
| Primary contributor / role | **Vedaang Chopra — primary repository author and research/engineering implementer.** Git history contains many commits authored by Vedaang across data, alignment, decoder, reports, results, and audit work; one visible commit is authored by Shourya Teja Adluri. | `git shortlog -sne --all`; recent history. High for primary contribution; do not imply sole authorship. |
| Dates evidenced | Main implementation/experimentation: Nov–Dec 2025; documentation/paper sync: Jan 2026; repository audit: Aug 2026; this resume-context audit: Sep 2026. | Git log and commit timestamps. High. |
| Project type | Academic/research engineering prototype; personal/research repository. | Paper, Git ownership, artifact state. High. |
| Current status | **PARTIAL / research prototype.** The audited, artifact-backed component is vision–text retrieval. Modular code also contains retained experimental audio, Qwen, TRM-inspired-refinement, and distributed-training paths, but these are not all validated end to end. | `AUDIT_REPORT.md`, current code, outputs. High. |
| Current one-sentence description | A modular PyTorch research prototype that freezes CLIP and sentence-transformer backbones, trains pooling/projection heads with symmetric contrastive and nested-prefix objectives, and evaluates multi-scale PixMo image–text retrieval. | Current config, encoders, losses, final CSV. High. |

**Working-tree caveat:** At audit time, many legacy `code_base/v0_code_base`, `v1_code_base`, and `v2_code_base` files were deleted in the working tree and `older_code_base/` was untracked. They are not the current canonical implementation and should not be used to inflate current capabilities. The current implementation surface is `edge_glass_modular/`.

---

## 2. Problem and Research/Engineering Objective

CERBERUS investigates whether heterogeneous, frozen multimodal encoders can be bridged by comparatively lightweight learned components instead of retraining a monolithic VLM end to end. The concrete verified focus is **image–text retrieval**: map CLIP image patch features and sentence-transformer caption features into one normalized embedding space where paired images and captions rank near one another.

The problem is difficult because the source representations differ in modality, sequence length, dimensionality, and pooling semantics, while downstream retrieval and generation interfaces need fixed-size vectors. The project further asks whether one learned high-dimensional representation can remain useful when truncated to smaller nested prefixes, allowing a cost/quality trade-off without retraining.

**Core hypothesis, conservatively stated:** frozen vision and text backbones plus trainable projection/pooling heads and a symmetric contrastive objective can produce usable 4096-dimensional PixMo retrieval embeddings; supervising normalized prefixes can retain substantial retrieval quality at smaller dimensions.

**Differentiation actually supported by the repository:**

- Encoder-frozen **CLIP ViT-L/14-336** vision and **`sentence-transformers/all-mpnet-base-v2`** text backbones are projected to a common 4096-dimensional space in the primary configuration. This is the **canonical alignment path** and produced the authoritative retained alignment results.
- Vision pooling is learned attention pooling in the successful artifact-backed path. The historical Perceiver-style alternative arm produced a near-random result whose train/eval architecture chain is provenance-compromised (see below); it is retained as a recorded historical outcome, **not** as a valid architecture verdict. A controlled rerun is required before drawing any Perceiver conclusion.
- The "MRL" implementation is specifically a normalized **prefix-slicing** scheme over the learned projected vector, supervised with contrastive loss at sampled dimensions. It is not a separately learned cascade of Matryoshka heads.
- The decoder-facing extension is a Qwen2.5-3B VLM path with **TRM-inspired iterative refinement**: a fixed 4-iteration y/z loop over shared `TRMLayer`s applied to Qwen final hidden states. It is *not* a faithful implementation of TinyRecursiveModels and there is no standalone `TRMDecoder` anymore. Repository evidence does not establish a reproducible, quantitative end-to-end generation result.

---

## 3. Current System Architecture

### Canonical, artifact-backed path — `VERIFIED_IMPLEMENTED` + `VERIFIED_EXPERIMENT`

`PixMo-Cap Parquet image bytes + caption → CLIP ViT-L/14-336 (frozen) + SentenceTransformer MPNet (frozen) → trainable projections → learned attention pooling over vision tokens → L2-normalized 4096-d vectors → nested normalized prefixes (2048/1024/512/256/128) → symmetric InfoNCE retrieval/MRL losses → cosine-similarity ranking → I2T and T2I Recall@K/MRR/median rank CSVs`

| Subsystem | Current behavior | Status | Evidence |
|---|---|---|---|
| Inputs/data | `PixmoParquetImageTextDataset` loads Parquet rows containing embedded image bytes, captions, and sample IDs; invalid images are removed at load time. | VERIFIED_IMPLEMENTED; exact artifact data files are absent. | `src/data/dataset_builder.py:102-195`; `configs/pixmo_alignment.yaml:4-17`. |
| Vision encoder | Frozen `openai/clip-vit-large-patch14-336`; direct linear patch projection to 4096 dimensions; simple learned attention pooling selected by config. | VERIFIED_IMPLEMENTED; used by final retrieval configuration. | `configs/pixmo_alignment.yaml:19-28`; `src/encoders/vision.py:98-104,136-167,183-233`. |
| Text encoder | Frozen `sentence-transformers/all-mpnet-base-v2`, linear projection to 4096 dimensions. | VERIFIED_IMPLEMENTED; configured for main path. | `configs/pixmo_alignment.yaml:30-35`; `src/encoders/text.py:43-94,130-169`. |
| Representation | `MatryoshkaProjection` slices and re-normalizes the first N coordinates of a projected vector; no learned parameters inside the MRL module. | VERIFIED_IMPLEMENTED; retrieval evaluation persists. | `src/encoders/mrl.py:35-80`; final CSV. |
| Objective | L2-normalized symmetric CLIP/InfoNCE loss; MRL loss invokes the same contrastive loss for sampled common prefix dimension(s). | VERIFIED_IMPLEMENTED. | `src/models/losses.py:9-40,43-98,101-175`. |
| Evaluation | Pairwise cosine-similarity matrix, diagonal pairing assumption, I2T/T2I Recall@K, MRR, and median rank. | VERIFIED_IMPLEMENTED + VERIFIED_EXPERIMENT for final CSVs. | `src/evaluation/metrics.py:51-114`; `results/alignment/alignment_eval_final/alignment_metrics_all_models.csv`. |
| Outputs | Versioned final CSV tables and plots for attention vs. Perceiver and prefix dimensions. | VERIFIED_EXPERIMENT. | `results/alignment/alignment_eval_final/` (curated location; was `results/alignment/alignment_eval_final/` before the artifact reorganization). |

### Alternative and extension subsystems

| Subsystem | Actual implementation | Status and qualification |
|---|---|---|
| Perceiver compression | `src/encoders/perceiver.py` now implements the **executed V3-training architecture** (latent self-attention, in-attention pre-norms, mask support), canonicalized from `scripts/experiments/train_perceiver_v3.py`. | PARTIAL / **provenance-compromised historical result**. The historical Perceiver evaluation architecture differed from training and the checkpoint was loaded with `strict=False`; the near-random retained metric must not be interpreted as a valid architecture verdict. A controlled rerun is required. The historical number itself is preserved unchanged in the final CSVs. `AUDIT_LEDGER.md` (Perceiver canonicalization + evaluation-chain entries). |
| Audio | Frozen Whisper encoder, optional Perceiver and prefix slicing; tri-modal loss scaffolding retained. | **Retained experimental support.** No retained artifact establishes audio retrieval, ASR preservation, or tri-modal results; audio results must not be inferred from vision–text results. `src/encoders/audio.py`; `src/models/losses.py` (tri-modal loss). |
| Fusion | **Retired.** `MultimodalFusion` / cross-attention fusion were removed from the active tree. | Historical code only (archive branch). No retained result was ever attributable to fusion. |
| Decoder Path C (`MultimodalProjector`) | **Retired.** The pooled 8-token prefix fork was never executed and was removed. | Historical code only (archive branch). No retained result came from Path C's forward. |
| Qwen integration | Qwen2.5-3B is the configured VLM backbone (older Qwen 1.5B/1.8B/7B IDs persist in legacy configs). Qwen decoder supports 4/8-bit loading, LoRA, and prepended learned prefix embeddings. | VERIFIED_IMPLEMENTED as code/config; **UNVERIFIED** as an artifact-backed end-to-end evaluated result. `src/decoders/qwen.py`; configs. |
| TRM / iterative refinement | **TRM-inspired iterative refinement** in `src/models/trm_qwen_vlm.py`: shared `TRMLayer` stack, fixed 4 y/z iterations over Qwen final hidden states, residual add, LM head. No standalone `TRMDecoder` exists (retired); `src/decoders/trm.py` retains only the `TRMLayer` building block. | PARTIAL. This is not a faithful TinyRecursiveModels implementation (no adaptive halting, fixed depth). It is not evidence of a reproduced "Tiny Recursive Model" result or deployed edge decoder. `src/models/trm_qwen_vlm.py`. |
| Alignment training | `ImprovedMultimodalTrainer` contains warmup/cosine scheduling, gradient accumulation/clipping, optional DDP, checkpoint save/load, validation retrieval, W&B calls. | PARTIAL. Notebook-oriented trainer has evidence of past use and is the canonical alignment trainer; it has **no CLI**. The old trainer CLI family was retired. `src/training/improved_trainer.py`. |
| Public alignment CLI | The old `scripts/train_alignment.py` / `trainer.py` CLI was retired (it was broken: `MultimodalTrainer` indexed the model's returned dataclass as a dict). | **No canonical alignment CLI currently exists.** Reproducing alignment training means driving `ImprovedMultimodalTrainer` from a notebook/script. `src/training/improved_trainer.py`; `AUDIT_LEDGER.md` (CLI retirement entry). |
| VLM training/eval scripts | Accelerate + optional DeepSpeed scripts load an alignment checkpoint, Qwen, a QA Parquet dataset, and optional refinement. Checkpoint loading is now directory-aware (`src/models/checkpoint_loader.py`); adapter-only `checkpoint_best/` artifacts are explicitly rejected as incomplete. | PARTIAL. Scripts depend on unavailable checkpoint/data paths, and no provenance-clean persisted evaluation JSON/predictions is retained. `scripts/train/train_vlm_accelerate.py`; `scripts/eval/evaluate_vlm_accelerate.py`; `scripts/eval/inference_vlm.py`. |
| Serving/deployment | No API/service, container, health check, deployment manifest, or retained device benchmark was found. | UNVERIFIED / not implemented as a product capability. |
| Tests/CI | `edge_glass_modular/tests/` now holds a 48-test permanent regression suite (generation, refinement EOS, checkpoint loading, device placement, Qwen prefix masking, alignment forward/loss, config parsing, best-checkpoint semantics). No CI workflow is configured. | VERIFIED_IMPLEMENTED (tests exist and pass); CI itself remains absent. |

### Deprecated / non-canonical architecture

- `older_code_base/`, legacy code-base directories, and the archive branch `archive/historical-code-pre-retirement` are historical material, not the current package. Retired from the active tree: the standalone TRM decoder, the fusion subsystem, decoder Path C / `MultimodalProjector`, and the broken trainer CLI family.
- The package README has been rewritten as current documentation (no longer "production-ready" marketing); `reports/` files are historical snapshots carrying explicit historical headers. Neither overstates current capabilities.
- The historical Perceiver result is a **provenance-compromised** historical outcome (evaluation architecture differed from training; `strict=False` checkpoint loading). Treat it as a recorded historical ablation, not a validated architecture comparison; a controlled rerun is required. The packaged Perceiver now matches the executed V3 training architecture.

---

## 4. My Verified Contributions

### Modeling / Training

| Contribution | Why it mattered | Technology/method | Evidence | Status |
|---|---|---|---|---|
| Authored the primary repository implementation across frozen-encoder alignment, configs, outputs, paper/report synchronization, and later release audit. | Establishes primary ownership of the research-engineering prototype while preserving one collaborator-authored commit in Git history. | Git, PyTorch research code, experiment configs. | `git shortlog -sne --all`; commits from Nov 2025–Aug 2026. | VERIFIED_IMPLEMENTED / high-confidence attribution. |
| Built/configured frozen CLIP vision and SentenceTransformer text encoders with trainable common-space projections and vision attention pooling. | Makes cross-modal retrieval trainable without encoder fine-tuning. | PyTorch, Hugging Face Transformers, SentenceTransformers, attention pooling. | `configs/pixmo_alignment.yaml`; `src/encoders/vision.py`; `src/encoders/text.py`; authored commits. | VERIFIED_IMPLEMENTED and final result uses this path. |
| Implemented symmetric contrastive loss plus sampled nested-prefix supervision. | Supports the project’s multi-scale retrieval study. | InfoNCE/CLIP-style loss; normalized prefix embedding evaluation. | `src/models/losses.py`; `src/encoders/mrl.py`; final multi-scale CSV. | VERIFIED_IMPLEMENTED + VERIFIED_EXPERIMENT. |
| Implemented an improved training loop with optimizer/scheduler, mixed precision, optional distributed wrapping, gradient accumulation/clipping, checkpoint serialization, and W&B hooks. | Supplies research training infrastructure beyond notebook-only forward passes. | PyTorch, DDP, BF16/FP16, AdamW, cosine warmup, W&B. | `src/training/improved_trainer.py`; `git log` includes `ba01b82 Bug fix improved trainer`. | PARTIAL: feature code exists, but full public CLI/reproducibility is unverified. |

### Evaluation / Experimentation

| Contribution | Why it mattered | Technology/method | Evidence | Status |
|---|---|---|---|---|
| Produced and committed the final paired PixMo retrieval tables for attention pooling and Perceiver variants at six embedding dimensions. | Provides the strongest surviving quantitative evidence. | Cosine retrieval; bidirectional Recall@K, MRR, median rank; MRL truncation ablation. | Commit `05479e3`; `results/alignment/alignment_eval_final/`; `notebooks/experiments/07_alignment_final_report.ipynb`. | VERIFIED_EXPERIMENT. |
| Conducted a negative Perceiver ablation and documented likely failure modes. | Preserves a useful falsified direction rather than reporting only positive results. | Perceiver-style latent compression, loss/gradient diagnostics, hyperparameter analysis. | Final CSV; `notebooks/experiments/PERCEIVER_TRAINING_ISSUE_ANALYSIS.md`. | VERIFIED_EXPERIMENT for failure outcome; root-cause attribution is PARTIAL/hypothesis; the train/eval provenance chain is compromised (see §3). |
| Created retrieval evaluation utilities and visualization/report artifacts. | Enables analysis beyond one scalar: bidirectional recall, rank statistics, similarity plots, and t-SNE assets. | PyTorch similarity ranking, NumPy, Matplotlib/Seaborn/sklearn imports/plots. | `src/evaluation/metrics.py`; `results/alignment/`; `results/alignment/pixmo_alignment_eval/visualizations/`. | VERIFIED_IMPLEMENTED; only final retrieval tables are resume-safe metrics. |

### Data / Datasets

| Contribution | Why it mattered | Technology/method | Evidence | Status |
|---|---|---|---|---|
| Implemented local Parquet loaders for embedded image bytes/captions and PixMo QA image-question-answer rows, including invalid-image filtering and CLIP-style transforms. | Supports offline-oriented local training inputs and avoids runtime image URL dependence once a Parquet file has been built. | Pandas/Parquet, PIL, torchvision, PyTorch Dataset/DataLoader. | `src/data/dataset_builder.py:102-195,268-380`; `reports/dataset_construction_report.md`. | VERIFIED_IMPLEMENTED; retained Parquet datasets are absent. |
| Added data-download/build scaffolding for PixMo and audio-oriented data sources. | Enables future dataset acquisition and multimodal extensions. | Hugging Face datasets, `requests`, multiprocessing, yt-dlp/ffmpeg references. | `src/data/downloader.py`, scripts, dataset report. | PARTIAL; do not claim a validated released dataset pipeline or completed audio corpus. |

### Systems / Infrastructure / Decoder Interfaces

| Contribution | Why it mattered | Technology/method | Evidence | Status |
|---|---|---|---|---|
| Implemented Qwen LoRA/quantization prefix-conditioning interfaces and the **TRM-inspired iterative refinement** path (shared `TRMLayer`s, fixed 4-iteration y/z loop over Qwen hidden states). | Explores iterative refinement as a decoder-conditioning alternative; an inspired design, not a faithful TinyRecursiveModels implementation. | Hugging Face Transformers, PEFT/LoRA, BitsAndBytes, prefix embeddings, residual refinement loop. | `src/decoders/qwen.py`; `src/models/trm_qwen_vlm.py`; recent regression tests (`test_eos_stop.py`, `test_device_placement.py`, `test_checkpoint_loader_regression.py`). | VERIFIED_IMPLEMENTED (code + regression tests); end-to-end generation quality UNVERIFIED. |

---

## 5. Verified Results and Metrics

**Protocol qualification for every retrieval number:** The evaluator normalizes embeddings, builds an $N\times N$ similarity matrix, and treats only the same row index as relevant (`src/evaluation/metrics.py:66-88`). Thus these are **paired retrieval metrics**, not end-to-end VLM generation accuracy, QA accuracy, real-world search relevance, latency, or production quality.

| Metric | Result | Experiment / dataset | Evidence | Confidence | Resume-safe wording |
|---|---:|---|---|---|---|
| I2T Recall@5 at 4096-d | **77.681%** | Attention-pooling vision–text model; PixMo validation split | `results/alignment/alignment_eval_final/alignment_metrics_all_models.csv:2` | High artifact confidence; moderate reproducibility confidence | “Achieved 77.7% image-to-text Recall@5 on a persisted PixMo validation retrieval evaluation.” |
| T2I Recall@5 at 4096-d | **77.547%** | Same attention model / split | Same CSV:3 | High artifact confidence; moderate reproducibility confidence | “Achieved 77.5% text-to-image Recall@5 on the paired PixMo validation evaluation.” |
| I2T/T2I Recall@1 at 4096-d | **50.871% / 50.871%** | Same | Same CSV:2–3 | High artifact confidence | Use only if room permits; clarify paired retrieval. |
| I2T/T2I Recall@10 at 4096-d | **85.958% / 85.891%** | Same | Same CSV:2–3 | High artifact confidence | Use only alongside the protocol qualifier. |
| I2T/T2I R@5 at 512-d | **73.727% / 74.933%** | Attention model prefix truncation, 8× smaller than 4096-d | Same CSV:8–9; `table_multiscale_combined.csv:5` | High artifact confidence | “Retained ~74% paired Recall@5 after truncating 4096-d embeddings to 512 dimensions.” |
| I2T/T2I R@5 at 128-d | **62.936% / 64.042%** | Attention model prefix truncation, 32× smaller than 4096-d | Same CSV:12–13; `table_multiscale_combined.csv:7` | High artifact confidence | “Observed 62.9%/64.0% I2T/T2I Recall@5 at a 128-d prefix in the persisted multiscale retrieval evaluation.” |
| Attention → 128-d R@5 absolute change | **−14.13 points I2T; −13.51 points T2I** vs. 4096-d | Same multiscale evaluation | Derived directly from persisted final CSV; summary CSV records I2T average-style drop. | High arithmetic/artifact confidence | “Quantified the retrieval–dimension trade-off across 4096→128-d nested prefixes.” |
| Perceiver I2T/T2I R@5 | **0.201% / 0.168%** | Final notebook-local Perceiver V3 negative ablation — **provenance-compromised** (eval architecture ≠ training architecture; `strict=False` loading; retained as recorded history, not an architecture verdict) | Final CSV:14–15; `table_4096_detailed.csv:4–5` | High outcome confidence; low architecture-provenance confidence | “Documented a historical failed Perceiver compression ablation (near-random paired R@5) with a provenance-compromised train/eval chain; a controlled rerun is required before reading it as an architecture comparison.” |

### Sample-size and baseline interpretation

- Documentation says PixMo-Cap was split 14,000 / 3,000 / 3,000 train/validation/test (`CEREBRUS.pdf`, `reports/results_report.md`). The final CSV itself has **no explicit N**. The final notebook/audit reports 16 invalid validation images removed; the Perceiver R@1 of 0.033512% equals one hit out of **2,984**, consistent with 3,000 − 16. Treat **N≈2,984** as an inferred effective evaluation size, not a fully provenance-backed immutable count.
- Baseline in the persisted comparison is attention pooling versus a failed Perceiver variant—not a published external benchmark or general CLIP baseline. The 4096-d full vector is the internal dimension baseline for prefix truncation.
- the previously deleted provenance-free `alignment_evaluation_results.csv` (removed; git-history recoverable) contains higher values (I2T/T2I R@5 85.69/86.80) plus CIFAR and MRL lines but lacks experiment provenance and conflicts with the final committed table. It is **not resume-safe**.
- The historical `notebooks/outputs/pixmo_alignment/metrics.csv` (superseded by `results/train_history/pixmo_alignment/` after artifact curation) reports a single training-summary row including `best_r5=0.8190`, but no dataset/checkpoint/run identity makes it less authoritative than the final result tables. It is **not resume-safe**.
- The paper’s TRM/Baseline BLEU and ROUGE-L values lack committed predictions, targets, evaluation JSON, checkpoint, and reconciled metric scale. They are **not resume-safe**.

---

## 6. Experiments Actually Run

### A. Final PixMo attention-pooling retrieval evaluation — `VERIFIED_EXPERIMENT`

- **Question:** Can frozen CLIP vision and MPNet text encoders plus learned attention pooling and contrastive/MRL supervision yield useful shared image–text retrieval embeddings?
- **Setup:** 4096-d projected vision/text embeddings; frozen CLIP ViT-L/14-336 and `all-mpnet-base-v2`; learned attention pooling on vision; normalized cosine ranking.
- **Dataset/sample size:** PixMo validation. Documentation says 3,000 validation examples. Final effective N appears to be 2,984 after invalid-image filtering, but the result artifact does not persist N, data fingerprint, or split hash.
- **Baseline/arms:** Attention path; evaluated I2T and T2I.
- **Metrics:** Recall@1/5/10, MRR, median rank.
- **Result:** At 4096-d, I2T R@1/R@5/R@10 = 50.871/77.681/85.958%; T2I = 50.871/77.547/85.891%; MRR = 0.629/0.631; median rank = 1/1.
- **Interpretation:** Strong paired retrieval within this specific retained evaluation artifact; it establishes an alignment result, **not** grounded QA/generation capability or deployment readiness.
- **Limitations:** Checkpoints and datasets are absent; notebook has absolute paths and loads with `strict=False`; exact reproducibility is not supported by the repository.
- **Artifacts:** `edge_glass_modular/results/alignment/alignment_eval_final/alignment_metrics_all_models.csv`; `table_4096_detailed.csv`; `notebooks/experiments/07_alignment_final_report.ipynb`.

### B. Matryoshka prefix-truncation retrieval study — `VERIFIED_EXPERIMENT`

- **Question:** Does the same attention model preserve paired retrieval quality as its 4096-d vector is truncated?
- **Setup:** Re-rank normalized first N dimensions for N = 4096, 2048, 1024, 512, 256, 128.
- **Dataset/sample size:** Same final PixMo validation evaluation; effective N not persisted.
- **Baseline/arms:** 4096-d full embedding versus nested prefix sizes.
- **Metrics:** I2T/T2I Recall@5, plus R@1/R@10/MRR/median rank in the complete CSV.
- **Result:** R@5 falls gradually from 77.681/77.547% (4096) to 73.727/74.933% (512), 70.409/70.643% (256), and 62.936/64.042% (128).
- **Interpretation:** This is direct evidence of a dimension/retrieval trade-off in one aligned model. It supports saying prefixes remained useful on this evaluation; it does **not** establish measured storage cost, inference latency, edge-device throughput, or decoder quality.
- **Limitations:** Prefix evaluation re-normalizes slices; there is no released checkpoint/data provenance or multi-seed confidence interval.
- **Artifacts:** `results/alignment/alignment_eval_final/alignment_metrics_all_models.csv`; `table_multiscale_combined.csv`; `plot_r5_dim_comparison_all.png`.

### C. Perceiver compression ablation — `VERIFIED_EXPERIMENT` (negative result)

- **Question:** Can a Perceiver-style learned latent compressor replace the attention-pooling projection path?
- **Setup:** A final notebook-local Perceiver V3 model was evaluated against attention outputs. Packaged configuration uses 64 latents and 512-d latent/projection dimensions; final table labels are misleadingly 4096-d.
- **Dataset/sample size:** Same paired PixMo validation evaluation; inferred N≈2,984.
- **Baseline/arms:** Attention pooling vs. Perceiver.
- **Metrics:** R@1/5/10, MRR, median rank.
- **Result:** Perceiver I2T/T2I R@5 = 0.201/0.168%; I2T/T2I median rank = 1517/1555. This is near random and far below attention pooling.
- **Interpretation:** Negative result: the evaluated Perceiver path did not learn usable retrieval alignment under this run/configuration. It justifies retaining attention pooling as the artifact-backed primary path.
- **Limitations:** The final notebook model differs from reusable code; the suggested initialization/learning-rate/mean-pooling explanations are diagnostic hypotheses, not controlled causal proof; no rerun after proposed changes persists.
- **Artifacts:** `results/alignment/alignment_eval_final/alignment_metrics_all_models.csv:14-25`; `notebooks/experiments/PERCEIVER_TRAINING_ISSUE_ANALYSIS.md`; `notebooks/experiments/07_alignment_final_report.ipynb`.

### D. Alignment training summary — `PARTIAL`

- **Question:** Did a PixMo alignment training run produce logged loss/retrieval history?
- **Setup/evidence:** A one-row persisted `metrics.csv` lists a 4096-d run, MRL dimensions, final train loss, final validation loss, and retrieval values.
- **Result:** `final_train_loss=1.11396`, `final_val_loss=0.76460`, `best_r1=0.56870`, `best_r5=0.81903`, `best_r10=0.89176`.
- **Why not resume-safe:** The artifact does not record split, N, checkpoint, commit, seed, or the directionality/protocol behind `best_r*`; it conflicts with the final result table. Do not cite its numbers.
- **Artifact:** the historical `notebooks/outputs/pixmo_alignment/metrics.csv` (superseded by `results/train_history/pixmo_alignment/`).

### E. Qwen/TRM/PixMo-QA work — `PARTIAL / UNVERIFIED`

- **Evidence:** Multiple QA configs, training/evaluation scripts, notebook outputs, and a Qwen+optional-recursion code path exist. `03_trm_vlm_qa_training_FIXED.ipynb` shows execution traces of checkpoint/model loading and a configured 10-epoch run; `04_vlm_qa_evaluation.ipynb` also retains a `FileNotFoundError` for a missing VLM checkpoint.
- **No verified claim:** No committed prediction set, raw targets, result JSON/CSV, usable checkpoint, defined test population, or reproducible decoder comparison was found. Do not cite paper Table VII BLEU/ROUGE-L, “grounded generation,” Qwen improvement, TRM improvement, decoder readiness, or QA accuracy.
- **Artifacts/code:** `notebooks/experiments/03_trm_vlm_qa_training_FIXED.ipynb`; `scripts/train/train_vlm_accelerate.py`; `scripts/eval/evaluate_vlm_accelerate.py`; `configs/trm_vlm_qa_*.yaml`.

### F. Audio / tri-modal experiments — `UNVERIFIED`

- **Evidence:** Audio encoder, tri-modal loss, dataset notes, and paper claims exist.
- **No verified claim:** No retained result artifact supports WER 3.18%, ESC-50 3% accuracy, audio–text retrieval, audio–vision retrieval, or a trained tri-modal system. Treat all as unverified until an immutable result bundle is retained.

---

## 7. Current Research / Engineering Bottlenecks

1. **No polished canonical alignment CLI.** The broken `scripts/train_alignment.py` / `MultimodalTrainer` CLI family was retired; the canonical alignment trainer (`ImprovedMultimodalTrainer`) is notebook/script-driven only. A maintained alignment CLI remains unbuilt.
2. **Reproducibility gap.** Final results lack source dataset/split manifest, sample count, seed, environment lock, commit SHA, checkpoint hash, runnable relative paths, and raw evaluation outputs. Final notebook relies on unavailable absolute paths and permissive `strict=False` loading.
3. **Perceiver provenance gap.** The historical Perceiver evaluation architecture differed from training and the checkpoint was loaded with `strict=False`; the near-random retained metric is provenance-compromised. The packaged Perceiver now matches the executed V3 training architecture, but no controlled rerun has validated it.
4. **Decoder results unsubstantiated.** Qwen / TRM-inspired-refinement code and fixes (EOS stopping, device placement, checkpoint integrity) do not substitute for validated generative metrics; no provenance-clean QA evaluation result is retained.
5. **Audio/tri-modal gap.** Retained experimental audio support exists in code but has no retained current result artifact; do not infer audio results from vision–text results.
6. **Data pipeline paths.** Primary configs point to user-specific absolute Parquet paths (HPC cluster) that do not exist in the repository; a full local rerun requires rebuilding the parquet datasets via `src/data/parquet_builder.py` / `scripts/data/`.
7. **Testing/quality status.** A 48-test permanent regression suite now exists and passes (`pytest tests/`), but there is no CI workflow wired up.
8. **Dependency status.** `pyproject.toml` now declares validated compatibility ranges (transformers 4.57.x, PEFT 0.17–0.18, Accelerate 1.10–1.11, bnb 0.48.x, sentence-transformers 5.1.x); the repo-root lock files remain the exact-reproduction path. Some runtime/tooling dependencies (e.g. `requests`, `soundfile`) are still not declared.
9. **No measured edge or production performance.** No benchmark artifact reports GPU count/type used, throughput, latency, memory footprint, quantized model size, device deployment, uptime, or serving reliability.

---

## 8. Active Work vs Future Roadmap

### Currently Developing / with implementation evidence

| Item | Evidence | Status |
|---|---|---|
| Modularize frozen-encoder alignment, retrieval evaluation, and offline Parquet loaders | `edge_glass_modular/src/`, configs, final outputs | PARTIAL: code and some result artifacts exist; canonical runnable workflow is unresolved. |
| Diagnose Perceiver failure | Negative result artifact and issue-analysis document | PARTIAL: diagnosis exists; no verified repair/retraining. |
| Qwen QA training/inference interface with optional recursion | QA configs/scripts/notebooks | PARTIAL: implementation and notebook activity exist; no verified evaluation. |
| Checkpointing/Accelerate/DeepSpeed/DDP-aware infrastructure | improved trainer and VLM scripts | PARTIAL: feature code exists; no verified clean multi-GPU end-to-end run. |

### Planned / proposed

Every item below is **DO NOT CLAIM AS COMPLETED ON RESUME**.

- **Fix and rerun Perceiver experiments** with an unambiguous packaged architecture, controlled initialization/pooling/hyperparameters, and result provenance.
- **Release a reproducible experiment bundle**: relative config paths, dataset/split manifests, environment lock, immutable checkpoints, evaluator command, raw predictions/ranks, seed, commit SHA.
- **Complete and evaluate PixMo-QA decoder experiments** with archived predictions, exact EM/F1/BLEU/ROUGE protocol, test split, baseline, and checkpoint identifiers.
- **Train/evaluate tri-modal alignment** and any audio retrieval/classification with retained data and results.
- **Measure edge deployment properties** such as model footprint, prefix-storage benefit, latency, throughput, energy, and actual device support.
- **Harden the package** by repairing the public CLI, adding tests/CI, reconciling dependencies, and removing absolute paths/debug utilities.
- **Production deployment/serving.** No current evidence supports a deployed service.

---

## 9. Technology and Skills Evidence

### Languages

- **Python — STRONG:** primary implementation, scripts, notebooks, data/evaluation/training modules.
- **YAML — USED:** nine experiment configuration files load successfully.
- **Jupyter notebooks — USED:** execution outputs persist, although notebooks are not a clean reproducible pipeline.

### ML / Deep Learning

- **PyTorch — STRONG:** modules, loss functions, training loops, tensor evaluation.
- **Torchvision — USED:** image loading/transforms.
- **Hugging Face Transformers — STRONG:** CLIP, Whisper, Qwen wrappers/model loading.
- **SentenceTransformers — USED:** MPNet/SBERT text encoding.
- **PEFT / LoRA — USED in implementation/config:** Qwen code applies LoRA; no completed fine-tuning result claim.
- **BitsAndBytes quantization — USED in implementation/config:** 4/8-bit loading code/config exists; no inference-memory or performance result.
- **Matryoshka/prefix embeddings — STRONG:** implemented and artifact-backed for retrieval truncation.
- **Perceiver-style cross-attention — LIMITED:** implemented and evaluated as a failed/ambiguous ablation.

### Agentic AI

- **None resume-safe from this project.** This is not evidence of multi-agent orchestration, tool-use agents, memory, RAG, planning, self-correction, or agent evaluation.

### Training / Inference / Serving

- **AdamW, warmup + cosine schedule, gradient accumulation/clipping — USED in code/config.**
- **BF16/FP16 mixed-precision support — USED in code/config; no resume-safe hardware-efficiency claim.**
- **PyTorch DDP — LIMITED/PARTIAL:** wrappers/utilities exist; no retained successful distributed run evidence.
- **Accelerate / optional DeepSpeed ZeRO — LIMITED/PARTIAL:** VLM scripts construct these paths; do not claim a deployed or validated distributed-training system.
- **Checkpointing and W&B hooks — PARTIAL:** implemented, but checkpoint provenance and usable W&B run summaries are not retained.
- **Serving/deployment — OMIT:** no evidence.

### Evaluation / Data / Systems

- **Bidirectional image–text retrieval evaluation — STRONG:** cosine similarity, Recall@K, MRR, median rank; final CSVs persist.
- **Controlled prefix-dimension ablation — STRONG:** six dimensions on the same artifact-backed evaluation.
- **Failure analysis — STRONG:** Perceiver negative outcome and documented diagnostic hypotheses.
- **Pandas / Apache Parquet / PIL — USED:** local embedded-media dataset loaders.
- **NumPy / Matplotlib / Seaborn / scikit-learn — USED/LIMITED:** analyses/visualizations and present environment; package metadata is incomplete.
- **Hugging Face datasets / requests / multiprocessing — LIMITED:** download/build code and documentation exist; not a released/reproducible pipeline.

### Research methods

- **Frozen-backbone adaptation — STRONG.**
- **Symmetric contrastive retrieval learning — STRONG.**
- **Embedding normalization and cosine ranking — STRONG.**
- **Ablation / failure analysis — STRONG, with provenance limitations.**
- **Reproducibility engineering — LIMITED:** configuration-first intent is real, but current artifact provenance is insufficient.

---

## 10. Research Concepts Demonstrated

These are technically defensible if described with their limits:

1. **Cross-modal representation alignment:** learned projections map frozen image and text encoder outputs into a shared retrieval space.
2. **Contrastive representation learning:** symmetric InfoNCE/CLIP-style objective over paired batch examples.
3. **Attention-based token pooling:** learned query-like attention pools CLIP patch-token sequences in the successful vision path.
4. **Nested / Matryoshka-style embeddings:** re-normalized coordinate prefixes are supervised/evaluated at multiple dimensions.
5. **Inference-time representation trade-offs:** the retrieval study compares the same learned vector at 4096→128 dimensions; no latency claim.
6. **Parameter-efficient adaptation design:** frozen backbones and trainable heads; LoRA/quantization are decoder-interface options, not verified outcome claims.
7. **Multimodal prefix conditioning:** aligned embeddings are projected into decoder-token space in code.
8. **Ablation design and negative-result reporting:** attention versus Perceiver outcome, paired directionality, and dimension study.
9. **Offline embedded-media dataset design:** Parquet rows carry image bytes with data-quality filtering.
10. **Training-systems primitives:** optimizer scheduling, gradient accumulation/clipping, checkpoints, optional DDP/Accelerate/DeepSpeed, with implementation—not successful-scale-run—scope.

---

## 11. Resume-Safe Claims

| Claim | Evidence | Confidence | Best applicable resume type |
|---|---|---|---|
| Built a frozen-backbone vision–text alignment prototype using CLIP ViT-L/14-336, MPNet sentence embeddings, learned attention pooling, and symmetric contrastive training. | Main config; encoder/loss code; final attention retrieval artifacts. | High | Extended ATS; Research Engineer; Applied Scientist; MLE/AI Engineer. |
| Evaluated bidirectional PixMo image–text retrieval and persisted 77.7% I2T R@5 / 77.5% T2I R@5 at 4096 dimensions. | Final CSV. | High artifact confidence; qualify protocol. | Research Engineer; Applied Scientist; Extended ATS. |
| Quantified a multiscale retrieval trade-off: 512-d prefixes retained ~74% R@5 and 128-d prefixes retained ~63–64% R@5 in the same paired validation evaluation. | Final CSV/table. | High artifact confidence. | Research Engineer; Applied Scientist; MLE/AI Engineer. |
| Ran and documented a negative Perceiver compression ablation with near-random R@5, retaining attention pooling as the artifact-backed primary path. | Final CSV and issue report. | High for outcome; moderate for diagnosis. | Research Engineer; Applied Scientist. |
| Implemented modular PyTorch components for image/text/audio encoders, multiscale retrieval evaluation, Parquet datasets, checkpointing, and optional Qwen prefix conditioning. | Current source. | High for code existence; phrase as “implemented,” not “deployed” or “trained successfully.” | Extended ATS; Research Engineer; MLE/AI Engineer. |
| Implemented offline-oriented Parquet image-text and image-QA dataset loaders with embedded media decoding and invalid-image filtering. | Dataset builder. | High | MLE/AI Engineer; Research Engineer. |
| Implemented optional LoRA/quantization and Accelerate/DeepSpeed/DDP code paths for decoder experiments. | Qwen and VLM scripts. | Moderate; implementation only. | Extended ATS / MLE only, with “implemented support” language. |

---

## 12. Claims to Avoid / Correct

| Old / risky claim | Verdict | Why | Correct replacement |
|---|---|---|---|
| “Production-ready” / “production-grade” VLM framework | CONTRADICTED | Broken advertised alignment CLI, absolute paths, missing data/checkpoints, no tests/CI, compile failure in a script, no service/deployment evidence. | “Modular multimodal alignment research prototype.” |
| “Deployed edge-compatible assistant/system” | UNVERIFIED | No deployment, API, device benchmark, latency, model-size, or reliability artifact. | “Designed to study compact, multi-scale representations for future resource-constrained use.” |
| “Built an agentic / multimodal agent framework” | MISLEADING | No agent loop, tool use, memory, orchestration, planning, or agent evaluation. | “Built a multimodal alignment and decoder-conditioning prototype.” |
| “Trained a full tri-modal vision-audio-text model” | UNVERIFIED | Audio/tri-modal modules exist, but no artifact-backed current result. | “Implemented optional Whisper/audio and tri-modal-loss components; core verified experiments are vision–text.” |
| “Perceiver Resampler achieved/was tested at 4096-d” | MISLEADING | Final table labels 4096, but final notebook-local Perceiver is 512-d and not the packaged module. | “A Perceiver-style compression ablation failed near random retrieval; architecture provenance/dimension labeling needs reconciliation.” |
| “MRL learned a separate projection cascade/head” | MISLEADING | `MatryoshkaProjection` only slices and L2-normalizes an already learned projected vector. | “Supervised normalized nested prefixes of a learned 4096-d projection.” |
| “MRL yielded 87.5% storage reduction / faster inference” | PARTIAL / unsupported outcome claim | 4096→512 is mathematically 87.5% fewer vector coordinates, but no storage-system, latency, cost, or throughput measurement persists. | “Evaluated 8× shorter embedding prefixes with ~74% paired R@5; no latency or storage-system benchmark was run.” |
| “Qwen/TRM improved BLEU/ROUGE-L / converted an LLM into a VLM” | UNVERIFIED | Paper Table VII has no retained predictions, targets, checkpoint, result file, split protocol, or reconciled metric scale. | “Implemented Qwen/TRM prefix-conditioning paths; generation metrics require rerun and provenance.” |
| “TRM is a validated recursive reasoning model” | UNVERIFIED / misleading | There is no standalone TRM decoder anymore; the VLM path is **TRM-inspired iterative refinement** (shared `TRMLayer`s, fixed 4 iterations) and is explicitly not a faithful TinyRecursiveModels implementation. No validated result remains. | “Implemented TRM-inspired iterative refinement over Qwen hidden states (fixed 4-iteration y/z loop); it is not a TinyRecursiveModels implementation and its quality impact is unverified.” |
| “Used 2×H200 / H100/A100 clusters in completed training” | UNVERIFIED | Docs/configs contain hardware recommendations/claims, but no immutable run metadata or throughput evidence verifies hardware use. | “Implemented optional DDP/Accelerate/DeepSpeed-aware paths; do not cite hardware counts.” |
| “Distributed training works” | PARTIAL | DDP code exists; `ImprovedMultimodalTrainer` conditionally wraps DDP, but no retained successful distributed run or test exists. | “Implemented optional DDP-aware training infrastructure.” |
| “W&B-tracked experiments” | PARTIAL | W&B integration is coded; useful corresponding run summary/provenance is not retained in current package. | “Integrated optional W&B logging hooks.” |
| “R@5 = 85.69/86.80” | UNVERIFIED / conflicting | the previously deleted provenance-free `alignment_evaluation_results.csv` (removed; git-history recoverable) lacks provenance and conflicts with final committed table. | Use only final table’s 77.681/77.547% values with paired-retrieval qualifier. |
| “PixMo dataset was 20k exactly for final evaluation” | PARTIAL | Docs specify 14k/3k/3k, but final result artifact omits actual post-filter count/split fingerprint. | “Documentation describes a 14k/3k/3k PixMo subset; final effective validation N is inferred at ~2,984 after filtering.” |
| “Published/submitted paper” | UNVERIFIED | A local paper PDF exists; no venue submission, DOI, arXiv ID, acceptance, or publication evidence found. | “Authored a project paper/draft” only if applicable and accurate outside this repository. |
| “No encoder fine-tuning” | VERIFIED for primary config, not global codebase | Main config freezes vision/text encoders; do not generalize to every future config/path. | “Primary PixMo alignment config freezes CLIP and MPNet backbones.” |

---

## 13. Resume Positioning by Role

### A. Extended ATS Resume

Preserve: frozen encoder adaptation; CLIP; SentenceTransformers; PyTorch; contrastive learning; multimodal retrieval; Matryoshka/nested embeddings; LoRA/quantization/Accelerate as implementation keywords; Parquet data pipeline; evaluation metrics. Keep the one artifact-backed R@5 pair and dimension study. Omit production, deployment, agent/RAG/RL claims and unverified Qwen/TRM performance.

### B. Research Engineer Resume

Emphasize the architectural separation of frozen backbones and trainable alignment heads, attention pooling, symmetric InfoNCE, nested-prefix supervision, and the controlled attention-vs-Perceiver / dimensional ablations. Lead with measurement discipline: paired retrieval protocol, directions, artifacts, and negative result. De-emphasize generic “VLM framework” language and unvalidated deployment.

### C. Agentic AI / Agentic Research Resume

This project is a weak fit for agentic positioning. The defensible overlap is modular model routing/interface design—one latent interface can feed candidate decoder architectures—but **there is no verified agent behavior**. Use this project as a supporting multimodal-representation project, not evidence of autonomous agents, tool use, memory, orchestration, planning, self-correction, or agent evaluation. Prefer another project for the core agentic bullet.

### D. Applied Scientist Resume

Emphasize hypothesis-driven experiments: whether attention pooling aligns frozen encoders; whether prefixes preserve retrieval utility; whether Perceiver compression works. Include quantitative results with the paired-validation qualifier, negative Perceiver outcome, and explicit limitations/provenance. Omit infrastructure adjectives that obscure experimental scope.

### E. Machine Learning Engineer / AI Engineer Resume

Emphasize implemented systems: PyTorch modules, YAML configs, Parquet/PIL data loading, invalid-image handling, metrics, checkpoints, optional DDP/Accelerate/DeepSpeed, LoRA/quantization hooks. Mention the public CLI/reproducibility limitations only in interview—not the resume. Do not position it as production deployment or full end-to-end VLM serving.

---

## 14. Candidate Resume Bullets

### Extended ATS

- Built a PyTorch vision–text alignment prototype with frozen CLIP ViT-L/14-336 and MPNet backbones, learned attention pooling, symmetric contrastive loss, and nested 4096→128-dimensional embeddings.
- Evaluated paired PixMo retrieval in both directions, reaching 77.7% image-to-text and 77.5% text-to-image Recall@5 at 4096 dimensions.
- Ran a multiscale embedding ablation: 512-dimensional prefixes retained ~74% Recall@5 and 128-dimensional prefixes retained 62.9–64.0% Recall@5 on the persisted validation evaluation.
- Implemented Parquet-based embedded-image and PixMo-QA loaders, PyTorch evaluation utilities, YAML experiment configs, checkpointing, and optional Qwen LoRA/quantization integration.
- Diagnosed and documented a failed Perceiver compression ablation with near-random retrieval, retaining learned attention pooling as the validated path.

### Research Engineer

- Designed a frozen-backbone cross-modal alignment pipeline that projects CLIP image patches and MPNet captions into a normalized 4096-d retrieval space using learned attention pooling and symmetric InfoNCE.
- Produced artifact-backed PixMo retrieval results of 77.7% I2T and 77.5% T2I Recall@5, including rank statistics and cosine-similarity analysis.
- Evaluated normalized nested-prefix embeddings from 4096 to 128 dimensions, characterizing retrieval degradation under 8× and 32× representation truncation.
- Preserved a negative Perceiver-style compression result (0.20%/0.17% I2T/T2I R@5) and documented the architecture/provenance issues requiring a controlled rerun.

### Agentic Research

- Implemented a modular multimodal latent interface that maps frozen CLIP/text representations into decoder-ready prefix embeddings for Qwen and a custom lightweight decoder.
- Built and evaluated a shared multi-resolution embedding representation for selecting different latent sizes under varying compute budgets; validated this only for retrieval, not agent behavior.
- Implemented optional Qwen LoRA/quantization and recursive-decoder code paths for multimodal QA experimentation; end-to-end generation metrics remain unverified.

### Applied Scientist

- Tested whether frozen multimodal backbones plus learned alignment heads can support paired image–text retrieval, obtaining 77.7%/77.5% I2T/T2I Recall@5 in a retained PixMo validation artifact.
- Conducted a six-arm embedding-dimension ablation (4096→128) and measured a gradual R@5 decline rather than assuming compression preserved quality.
- Conducted a negative attention-pooling versus Perceiver compression comparison; the Perceiver arm remained near random, informing the retained architecture.
- Built bidirectional cosine-ranking evaluation with Recall@K, MRR, and median rank rather than treating representation similarity as an unmeasured proxy.

### MLE / AI Engineer

- Implemented modular PyTorch data, encoder, loss, training, and evaluation components for an offline-oriented Parquet image–text retrieval pipeline.
- Added configuration-driven experiments, embedded-media validation, checkpointing, gradient accumulation/clipping, and optional DDP/Accelerate/DeepSpeed and Qwen LoRA/quantization support.
- Built retrieval evaluation and result serialization for normalized image↔text ranking across multiple embedding dimensions.

---

## 15. Skills Placement

### Extended ATS

- **Put in bullets:** CLIP, SentenceTransformers/MPNet, PyTorch, contrastive learning/InfoNCE, multimodal retrieval, Matryoshka/nested embeddings, Recall@K, Parquet.
- **Put in Skills:** Transformers, Hugging Face, PEFT/LoRA, BitsAndBytes, Accelerate, DeepSpeed, DDP, W&B, Pandas, PIL, YAML.
- **Omit:** deployment/serving, agent orchestration, RAG, RLHF/DPO/PPO/GRPO, production reliability, hardware claims.

### Research Engineer

- **Put in bullets:** frozen-backbone alignment, attention pooling, symmetric contrastive objective, prefix truncation ablation, I2T/T2I retrieval, Perceiver negative result.
- **Put in Skills:** PyTorch, Transformers, SentenceTransformers, NumPy, Pandas, Matplotlib, Jupyter, Parquet.
- **Omit:** generic “AI agent,” production/edge deployment, unverified decoder evaluation.

### Agentic Research

- **Put in bullets:** decoder-ready prefix embeddings, modular decoder interface, Qwen integration code, multiscale latent selection.
- **Put in Skills:** PyTorch, Transformers, PEFT/LoRA, BitsAndBytes.
- **Omit:** agent frameworks, tool use, planners, memory systems, RAG, autonomous multi-agent systems, self-correction/evaluation claims.

### Applied Scientist

- **Put in bullets:** hypothesis/ablation, attention vs. Perceiver, Recall@K/MRR, prefix-dimension study, negative result.
- **Put in Skills:** PyTorch, experiment design, contrastive learning, representation learning, retrieval evaluation, statistical analysis/visualization (only if separately defensible).
- **Omit:** production software claims, Qwen/TRM performance, hardware/scale claims.

### MLE / AI Engineer

- **Put in bullets:** PyTorch pipeline, Parquet embedded-media data loader, config-driven experiments, checkpointing, evaluation, DDP/Accelerate implementation.
- **Put in Skills:** Python, PyTorch, Transformers, Pandas, Parquet, PIL, YAML, PEFT/LoRA, BitsAndBytes, Accelerate, DeepSpeed, W&B.
- **Omit:** Kubernetes, Docker/containerization, serving APIs, MLOps production operations, distributed-scale completion, edge deployment unless newer evidence exists.

---

## 16. Interview Narrative

I built CERBERUS/Edge Glass to test an alignment-first alternative to retraining a monolithic vision-language model. The core problem was turning frozen CLIP image-patch features and sentence-transformer caption embeddings into one fixed-size representation that supports paired retrieval and can later feed a VLM. I implemented the PyTorch pipeline: frozen encoders, trainable projections, learned attention pooling (the canonical alignment path), normalized nested prefixes from 4096 to 128 dimensions, and a symmetric contrastive objective. I also built Parquet-based image/QA loaders, retrieval evaluation, configs, and a Qwen2.5 VLM path with TRM-inspired iterative refinement (a fixed 4-iteration y/z refinement loop, not a faithful TinyRecursiveModels implementation).

The strongest result is the persisted PixMo paired retrieval evaluation: 77.7% image-to-text and 77.5% text-to-image Recall@5 at 4096 dimensions. The most informative experiment was the prefix ablation: 512-dimensional prefixes still reached roughly 74% Recall@5, while 128 dimensions fell to about 63–64%. I also kept a negative result: the historical Perceiver-style compression arm was near random — with the important caveat that its evaluation architecture differed from training and the checkpoint was loaded permissively, so the number is provenance-compromised and needs a controlled rerun before it can be read as an architecture verdict; attention pooling remains the artifact-backed primary path. The main limitation is reproducibility and scope: data/checkpoints and decoder metrics are not retained cleanly, there is no polished alignment CLI, and audio/tri-modal and end-to-end generation remain experimental rather than demonstrated.

---

## 17. Evidence Index

| Claim / Area | Evidence path(s) |
|---|---|
| Project title, author, institution, research framing | `CEREBRUS.pdf`; `README.md` |
| Primary repository authorship and dates | Git log; `git shortlog -sne --all`; commits `05479e3`, `693e43b`, `4113a26` |
| Prior critical repository audit | `AUDIT_REPORT.md` |
| Primary current package metadata/dependencies | `edge_glass_modular/pyproject.toml`; `requirements.txt` |
| Primary PixMo configuration / actual primary model IDs | `edge_glass_modular/configs/pixmo_alignment.yaml` |
| Perceiver configuration / its 512-d dimensions | `edge_glass_modular/configs/perceiver_mrl_alignment.yaml` |
| Qwen QA configuration/model ID and planned hardware-style settings | `edge_glass_modular/configs/trm_vlm_qa_qwen2.5-3b_regularized.yaml` and sibling QA configs |
| Configuration loading behavior | `edge_glass_modular/src/config.py`; audit-time successful load of all nine YAML configs |
| Vision architecture and frozen CLIP/attention/Perceiver paths | `edge_glass_modular/src/encoders/vision.py` |
| Text architecture and frozen SentenceTransformer/CLIP-text option | `edge_glass_modular/src/encoders/text.py` |
| Audio architecture | `edge_glass_modular/src/encoders/audio.py` |
| Nested-prefix/MRL implementation | `edge_glass_modular/src/encoders/mrl.py` |
| Contrastive and tri-modal losses | `edge_glass_modular/src/models/losses.py` |
| Main model and decoder prefix bridge | `edge_glass_modular/src/models/alignment.py`; `src/models/trm_qwen_vlm.py` |
| Qwen LoRA/quantization/prefix code | `edge_glass_modular/src/decoders/qwen.py` |
| TRMLayer building block (refinement) | `edge_glass_modular/src/decoders/trm.py` (standalone TRMDecoder retired; building block only) |
| Parquet/PixMo/PixMo-QA data loader behavior | `edge_glass_modular/src/data/dataset_builder.py` |
| Retired trainer CLI (historical reference) | archive branch `archive/historical-code-pre-retirement` |
| Improved trainer/checkpoint/DDP/W&B implementation | `edge_glass_modular/src/training/improved_trainer.py` |
| Accelerate/DeepSpeed QA scripts | `edge_glass_modular/scripts/train/train_vlm_accelerate.py`; `scripts/eval/evaluate_vlm_accelerate.py`; `scripts/eval/inference_vlm.py` |
| Checkpoint loading / device placement / EOS semantics | `edge_glass_modular/src/models/checkpoint_loader.py`; regression tests in `edge_glass_modular/tests/` |
| Retrieval evaluation semantics | `edge_glass_modular/src/evaluation/metrics.py` |
| QA metric implementation and scale caveat | `edge_glass_modular/src/evaluation/qa_metrics.py` |
| Final artifact-backed attention/Perceiver results | `edge_glass_modular/results/alignment/alignment_eval_final/alignment_metrics_all_models.csv` |
| Final attention vs. Perceiver table | `edge_glass_modular/results/alignment/alignment_eval_final/table_4096_detailed.csv` |
| Final multi-scale R@5 table | `edge_glass_modular/results/alignment/alignment_eval_final/table_multiscale_combined.csv` |
| Final visualization assets | `edge_glass_modular/results/alignment/alignment_eval_final/plot_tsne_comparison.png`; `plot_r5_dim_comparison_all.png`; `plot_cosine_hist_comparison.png` |
| Noncanonical training-history metrics to avoid citing | `edge_glass_modular/results/train_history/pixmo_alignment/metrics.csv` (no run identity; conflicts with final table) |
| Final evaluation notebook/provenance caveats | `edge_glass_modular/notebooks/experiments/07_alignment_final_report.ipynb`; `AUDIT_LEDGER.md` (evaluation-chain entries) |
| Perceiver negative-result diagnosis (historical) | `edge_glass_modular/notebooks/experiments/PERCEIVER_TRAINING_ISSUE_ANALYSIS.md` |
| Executed Perceiver V3 training code (provenance) | `edge_glass_modular/scripts/experiments/train_perceiver_v3.py` |
| QA training/evaluation evidence and missing-checkpoint failure | `edge_glass_modular/notebooks/experiments/03_trm_vlm_qa_training_FIXED.ipynb`; `notebooks/analysis/05_vlm_qualitative_inference.ipynb` |
| Dataset construction intent / paths / limitations | `edge_glass_modular/reports/dataset_construction_report.md` (historical snapshot); `reports/results_report.md` (historical snapshot) |
| Dependency contract / reproducibility | `edge_glass_modular/pyproject.toml`; repo-root `requirements.txt` / `requirements.lock.txt` |
| Provenance ledger (authoritative) | `AUDIT_LEDGER.md` |
| Audit-time code/config checks | `pytest edge_glass_modular/tests/` → 48/48; `compileall src scripts tests` OK; all 9 YAML configs load through `load_config` |

---

## Canonical-use rule

Future resume text should cite only claims marked **resume-safe** here, unless a newer commit adds an immutable result bundle with exact data split, checkpoint/model identity, environment, evaluator command, raw outputs, and verified read-back. If newer evidence conflicts with this document, record the conflict and update this file before changing resume claims.
