# PROJECT_RESUME_CONTEXT — CERBERUS / Edge Glass

**Last audited: 2026-09-13**  
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
| Current status | **PARTIAL / research prototype.** The audited, artifact-backed component is vision–text retrieval. Modular code also contains audio, Qwen, TRM, and distributed-training paths, but these are not all validated end to end. | `AUDIT_REPORT.md`, current code, outputs. High. |
| Current one-sentence description | A modular PyTorch research prototype that freezes CLIP and sentence-transformer backbones, trains pooling/projection heads with symmetric contrastive and nested-prefix objectives, and evaluates multi-scale PixMo image–text retrieval. | Current config, encoders, losses, final CSV. High. |

**Working-tree caveat:** At audit time, many legacy `code_base/v0_code_base`, `v1_code_base`, and `v2_code_base` files were deleted in the working tree and `older_code_base/` was untracked. They are not the current canonical implementation and should not be used to inflate current capabilities. The current implementation surface is `edge_glass_modular/`.

---

## 2. Problem and Research/Engineering Objective

CERBERUS investigates whether heterogeneous, frozen multimodal encoders can be bridged by comparatively lightweight learned components instead of retraining a monolithic VLM end to end. The concrete verified focus is **image–text retrieval**: map CLIP image patch features and sentence-transformer caption features into one normalized embedding space where paired images and captions rank near one another.

The problem is difficult because the source representations differ in modality, sequence length, dimensionality, and pooling semantics, while downstream retrieval and generation interfaces need fixed-size vectors. The project further asks whether one learned high-dimensional representation can remain useful when truncated to smaller nested prefixes, allowing a cost/quality trade-off without retraining.

**Core hypothesis, conservatively stated:** frozen vision and text backbones plus trainable projection/pooling heads and a symmetric contrastive objective can produce usable 4096-dimensional PixMo retrieval embeddings; supervising normalized prefixes can retain substantial retrieval quality at smaller dimensions.

**Differentiation actually supported by the repository:**

- Encoder-frozen **CLIP ViT-L/14-336** vision and **`sentence-transformers/all-mpnet-base-v2`** text backbones are projected to a common 4096-dimensional space in the primary configuration.
- Vision pooling is learned attention pooling in the successful artifact-backed path; a Perceiver-style alternative was evaluated and failed.
- The “MRL” implementation is specifically a normalized **prefix-slicing** scheme over the learned projected vector, supervised with contrastive loss at sampled dimensions. It is not a separately learned cascade of Matryoshka heads.
- A prefix-token bridge for Qwen and a custom small decoder exist as architectural extensions, but repository evidence does not establish a reproducible, quantitative end-to-end generation result.

Do **not** describe this as a completed edge deployment, a production VLM, an agent system, an omni-model, or a proven multimodal generation system.

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
| Evaluation | Pairwise cosine-similarity matrix, diagonal pairing assumption, I2T/T2I Recall@K, MRR, and median rank. | VERIFIED_IMPLEMENTED + VERIFIED_EXPERIMENT for final CSVs. | `src/evaluation/metrics.py:51-114`; `outputs/alignment_eval_final/alignment_metrics_all_models.csv`. |
| Outputs | Versioned final CSV tables and plots for attention vs. Perceiver and prefix dimensions. | VERIFIED_EXPERIMENT. | `outputs/alignment_eval_final/`. |

### Alternative and extension subsystems

| Subsystem | Actual implementation | Status and qualification |
|---|---|---|
| Perceiver compression | `PerceiverResampler` is packaged; vision path uses MLP adapter → Perceiver → mean pooling → final projection. | PARTIAL. A **notebook-local** `PerceiverV3Model` rather than the packaged module generated the final negative table; output is 512-d in its configuration despite table labels showing 4096. Do not attribute the final negative result cleanly to the reusable packaged Perceiver. `AUDIT_REPORT.md:44-45,79,115-117`. |
| Audio | Frozen Whisper encoder, optional Perceiver and prefix slicing; tri-modal loss can sum vision–text, audio–text, and vision–audio losses. | PARTIAL / UNVERIFIED_EXPERIMENT for current repo. No retained artifact establishes audio retrieval, ASR preservation, or tri-modal results. `src/encoders/audio.py`; `src/models/losses.py:178-274`. |
| Fusion | Concatenation, cross-attention, and gated fusion module definitions exist. | PARTIAL. No current configuration/artifact demonstrates a completed fusion experiment. `src/models/fusion.py`; `src/models/alignment.py:90-114`. |
| Qwen integration | Qwen 1.5B, 1.8B, 3B, and 7B model IDs appear across configs. Qwen decoder supports 4/8-bit loading, LoRA, and prepended learned prefix embeddings. | VERIFIED_IMPLEMENTED as code/config; **UNVERIFIED** as an artifact-backed end-to-end evaluated result. `src/decoders/qwen.py`; configs. |
| TRM integration | A custom small causal transformer-like decoder with RMSNorm, RoPE, self-attention, and prefix embedding support; separate Qwen+optional-recursion wrapper also exists. | PARTIAL. It is not evidence of a reproduced “Tiny Recursive Model” result or deployed edge decoder. `src/decoders/trm.py`; `src/models/trm_qwen_vlm.py`. |
| Alignment training | `ImprovedMultimodalTrainer` contains warmup/cosine scheduling, gradient accumulation/clipping, optional DDP, checkpoint save/load, validation retrieval, W&B calls. | PARTIAL. Notebook-oriented trainer has evidence of past use; no clean command/data/checkpoint bundle is retained. `src/training/improved_trainer.py`. |
| Public alignment CLI | `scripts/train_alignment.py` wires the data module to `MultimodalTrainer`. | CONTRADICTED as runnable canonical training path: the trainer calls `self.model(batch)` and indexes the returned dataclass as `outputs["loss"]`, incompatible with `MultimodalAlignmentModel.forward`. `src/training/trainer.py:52-56`; `src/models/alignment.py:175-184`; `AUDIT_REPORT.md:50,108-110`. |
| VLM training/eval scripts | Accelerate + optional DeepSpeed scripts load an alignment checkpoint, Qwen, a QA Parquet dataset, and optional recursion. | PARTIAL. Scripts depend on unavailable checkpoint/data paths, allow `strict=False`, and lack persisted evaluation JSON/predictions. `scripts/train_vlm_accelerate.py`; `scripts/evaluate_vlm_accelerate.py`. |
| Serving/deployment | No API/service, container, health check, deployment manifest, or retained device benchmark was found. | UNVERIFIED / not implemented as a product capability. |
| Tests/CI | No project test suite or CI workflow was found under `edge_glass_modular/`. One file named `add_test_split_to_notebook.py` is a notebook-editing script, not a test. | UNVERIFIED. |

### Deprecated / non-canonical architecture

- `older_code_base/` and legacy code-base directories are historical material, not the current package.
- Top-level and package READMEs still call the system “production-ready,” reference missing configs/notebooks, and overstate infrastructure readiness. They are **not authoritative** over the 2026 audit, current source, or persisted outputs.
- The final Perceiver evaluation implementation differs from the packaged Perceiver module; treat it as a historical negative ablation, not a validated architecture comparison.

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
| Produced and committed the final paired PixMo retrieval tables for attention pooling and Perceiver variants at six embedding dimensions. | Provides the strongest surviving quantitative evidence. | Cosine retrieval; bidirectional Recall@K, MRR, median rank; MRL truncation ablation. | Commit `05479e3`; `outputs/alignment_eval_final/`; `notebooks/07_alignment_final_report.ipynb`. | VERIFIED_EXPERIMENT. |
| Conducted a negative Perceiver ablation and documented likely failure modes. | Preserves a useful falsified direction rather than reporting only positive results. | Perceiver-style latent compression, loss/gradient diagnostics, hyperparameter analysis. | Final CSV; `notebooks/PERCEIVER_TRAINING_ISSUE_ANALYSIS.md`. | VERIFIED_EXPERIMENT for failure outcome; root-cause attribution is PARTIAL/hypothesis. |
| Created retrieval evaluation utilities and visualization/report artifacts. | Enables analysis beyond one scalar: bidirectional recall, rank statistics, similarity plots, and t-SNE assets. | PyTorch similarity ranking, NumPy, Matplotlib/Seaborn/sklearn imports/plots. | `src/evaluation/metrics.py`; `outputs/alignment_eval_final/`; `outputs/pixmo_alignment_eval/visualizations/`. | VERIFIED_IMPLEMENTED; only final retrieval tables are resume-safe metrics. |

### Data / Datasets

| Contribution | Why it mattered | Technology/method | Evidence | Status |
|---|---|---|---|---|
| Implemented local Parquet loaders for embedded image bytes/captions and PixMo QA image-question-answer rows, including invalid-image filtering and CLIP-style transforms. | Supports offline-oriented local training inputs and avoids runtime image URL dependence once a Parquet file has been built. | Pandas/Parquet, PIL, torchvision, PyTorch Dataset/DataLoader. | `src/data/dataset_builder.py:102-195,268-380`; `reports/dataset_construction_report.md`. | VERIFIED_IMPLEMENTED; retained Parquet datasets are absent. |
| Added data-download/build scaffolding for PixMo and audio-oriented data sources. | Enables future dataset acquisition and multimodal extensions. | Hugging Face datasets, `requests`, multiprocessing, yt-dlp/ffmpeg references. | `src/data/downloader.py`, scripts, dataset report. | PARTIAL; do not claim a validated released dataset pipeline or completed audio corpus. |

### Systems / Infrastructure / Decoder Interfaces

| Contribution | Why it mattered | Technology/method | Evidence | Status |
|---|---|---|---|---|
| Implemented a learned multimodal-projector/prefix-token interface for decoder conditioning, including Qwen LoRA/quantization options. | Defines an architectural bridge from retrieval latents to language-model embeddings. | Hugging Face Transformers, PEFT/LoRA, BitsAndBytes, prefix embeddings. | `src/models/projector.py`; `src/decoders/qwen.py`; `src/models/alignment.py:116-155,262-325`. | VERIFIED_IMPLEMENTED; end-to-end generation quality UNVERIFIED. |
| Implemented custom TRM-style decoder code and an optional recursive Qwen-VLM wrapper. | Explores a lower-capacity decoder interface in the architecture. | PyTorch, RMSNorm, RoPE, causal attention, autoregressive generation. | `src/decoders/trm.py`; `src/models/trm_qwen_vlm.py`. | PARTIAL; no artifact-backed resume claim of successful TRM improvement. |

---

## 5. Verified Results and Metrics

**Protocol qualification for every retrieval number:** The evaluator normalizes embeddings, builds an $N\times N$ similarity matrix, and treats only the same row index as relevant (`src/evaluation/metrics.py:66-88`). Thus these are **paired retrieval metrics**, not end-to-end VLM generation accuracy, QA accuracy, real-world search relevance, latency, or production quality.

| Metric | Result | Experiment / dataset | Evidence | Confidence | Resume-safe wording |
|---|---:|---|---|---|---|
| I2T Recall@5 at 4096-d | **77.681%** | Attention-pooling vision–text model; PixMo validation split | `outputs/alignment_eval_final/alignment_metrics_all_models.csv:2` | High artifact confidence; moderate reproducibility confidence | “Achieved 77.7% image-to-text Recall@5 on a persisted PixMo validation retrieval evaluation.” |
| T2I Recall@5 at 4096-d | **77.547%** | Same attention model / split | Same CSV:3 | High artifact confidence; moderate reproducibility confidence | “Achieved 77.5% text-to-image Recall@5 on the paired PixMo validation evaluation.” |
| I2T/T2I Recall@1 at 4096-d | **50.871% / 50.871%** | Same | Same CSV:2–3 | High artifact confidence | Use only if room permits; clarify paired retrieval. |
| I2T/T2I Recall@10 at 4096-d | **85.958% / 85.891%** | Same | Same CSV:2–3 | High artifact confidence | Use only alongside the protocol qualifier. |
| I2T/T2I R@5 at 512-d | **73.727% / 74.933%** | Attention model prefix truncation, 8× smaller than 4096-d | Same CSV:8–9; `table_multiscale_combined.csv:5` | High artifact confidence | “Retained ~74% paired Recall@5 after truncating 4096-d embeddings to 512 dimensions.” |
| I2T/T2I R@5 at 128-d | **62.936% / 64.042%** | Attention model prefix truncation, 32× smaller than 4096-d | Same CSV:12–13; `table_multiscale_combined.csv:7` | High artifact confidence | “Observed 62.9%/64.0% I2T/T2I Recall@5 at a 128-d prefix in the persisted multiscale retrieval evaluation.” |
| Attention → 128-d R@5 absolute change | **−14.13 points I2T; −13.51 points T2I** vs. 4096-d | Same multiscale evaluation | Derived directly from persisted final CSV; summary CSV records I2T average-style drop. | High arithmetic/artifact confidence | “Quantified the retrieval–dimension trade-off across 4096→128-d nested prefixes.” |
| Perceiver I2T/T2I R@5 | **0.201% / 0.168%** | Final notebook-local Perceiver V3 negative ablation | Final CSV:14–15; `table_4096_detailed.csv:4–5` | High outcome confidence; low architecture-provenance confidence | “Documented a failed Perceiver compression ablation (near-random paired R@5), motivating attention pooling as the retained path.” |

### Sample-size and baseline interpretation

- Documentation says PixMo-Cap was split 14,000 / 3,000 / 3,000 train/validation/test (`CEREBRUS.pdf`, `reports/results_report.md`). The final CSV itself has **no explicit N**. The final notebook/audit reports 16 invalid validation images removed; the Perceiver R@1 of 0.033512% equals one hit out of **2,984**, consistent with 3,000 − 16. Treat **N≈2,984** as an inferred effective evaluation size, not a fully provenance-backed immutable count.
- Baseline in the persisted comparison is attention pooling versus a failed Perceiver variant—not a published external benchmark or general CLIP baseline. The 4096-d full vector is the internal dimension baseline for prefix truncation.
- `outputs/alignment_evaluation_results.csv` contains higher values (I2T/T2I R@5 85.69/86.80) plus CIFAR and MRL lines but lacks experiment provenance and conflicts with the final committed table. It is **not resume-safe**.
- `notebooks/outputs/pixmo_alignment/metrics.csv` reports a single training-summary row including `best_r5=0.8190`, but no dataset/checkpoint/run identity makes it less authoritative than the final result tables. It is **not resume-safe**.
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
- **Artifacts:** `edge_glass_modular/outputs/alignment_eval_final/alignment_metrics_all_models.csv`; `table_4096_detailed.csv`; `notebooks/07_alignment_final_report.ipynb`.

### B. Matryoshka prefix-truncation retrieval study — `VERIFIED_EXPERIMENT`

- **Question:** Does the same attention model preserve paired retrieval quality as its 4096-d vector is truncated?
- **Setup:** Re-rank normalized first N dimensions for N = 4096, 2048, 1024, 512, 256, 128.
- **Dataset/sample size:** Same final PixMo validation evaluation; effective N not persisted.
- **Baseline/arms:** 4096-d full embedding versus nested prefix sizes.
- **Metrics:** I2T/T2I Recall@5, plus R@1/R@10/MRR/median rank in the complete CSV.
- **Result:** R@5 falls gradually from 77.681/77.547% (4096) to 73.727/74.933% (512), 70.409/70.643% (256), and 62.936/64.042% (128).
- **Interpretation:** This is direct evidence of a dimension/retrieval trade-off in one aligned model. It supports saying prefixes remained useful on this evaluation; it does **not** establish measured storage cost, inference latency, edge-device throughput, or decoder quality.
- **Limitations:** Prefix evaluation re-normalizes slices; there is no released checkpoint/data provenance or multi-seed confidence interval.
- **Artifacts:** `outputs/alignment_eval_final/alignment_metrics_all_models.csv`; `table_multiscale_combined.csv`; `plot_r5_dim_comparison_all.png`.

### C. Perceiver compression ablation — `VERIFIED_EXPERIMENT` (negative result)

- **Question:** Can a Perceiver-style learned latent compressor replace the attention-pooling projection path?
- **Setup:** A final notebook-local Perceiver V3 model was evaluated against attention outputs. Packaged configuration uses 64 latents and 512-d latent/projection dimensions; final table labels are misleadingly 4096-d.
- **Dataset/sample size:** Same paired PixMo validation evaluation; inferred N≈2,984.
- **Baseline/arms:** Attention pooling vs. Perceiver.
- **Metrics:** R@1/5/10, MRR, median rank.
- **Result:** Perceiver I2T/T2I R@5 = 0.201/0.168%; I2T/T2I median rank = 1517/1555. This is near random and far below attention pooling.
- **Interpretation:** Negative result: the evaluated Perceiver path did not learn usable retrieval alignment under this run/configuration. It justifies retaining attention pooling as the artifact-backed primary path.
- **Limitations:** The final notebook model differs from reusable code; the suggested initialization/learning-rate/mean-pooling explanations are diagnostic hypotheses, not controlled causal proof; no rerun after proposed changes persists.
- **Artifacts:** `outputs/alignment_eval_final/alignment_metrics_all_models.csv:14-25`; `notebooks/PERCEIVER_TRAINING_ISSUE_ANALYSIS.md`; `notebooks/07_alignment_final_report.ipynb`.

### D. Alignment training summary — `PARTIAL`

- **Question:** Did a PixMo alignment training run produce logged loss/retrieval history?
- **Setup/evidence:** A one-row persisted `metrics.csv` lists a 4096-d run, MRL dimensions, final train loss, final validation loss, and retrieval values.
- **Result:** `final_train_loss=1.11396`, `final_val_loss=0.76460`, `best_r1=0.56870`, `best_r5=0.81903`, `best_r10=0.89176`.
- **Why not resume-safe:** The artifact does not record split, N, checkpoint, commit, seed, or the directionality/protocol behind `best_r*`; it conflicts with the final result table. Do not cite its numbers.
- **Artifact:** `edge_glass_modular/notebooks/outputs/pixmo_alignment/metrics.csv`.

### E. Qwen/TRM/PixMo-QA work — `PARTIAL / UNVERIFIED`

- **Evidence:** Multiple QA configs, training/evaluation scripts, notebook outputs, and a Qwen+optional-recursion code path exist. `03_trm_vlm_qa_training_FIXED.ipynb` shows execution traces of checkpoint/model loading and a configured 10-epoch run; `04_vlm_qa_evaluation.ipynb` also retains a `FileNotFoundError` for a missing VLM checkpoint.
- **No verified claim:** No committed prediction set, raw targets, result JSON/CSV, usable checkpoint, defined test population, or reproducible decoder comparison was found. Do not cite paper Table VII BLEU/ROUGE-L, “grounded generation,” Qwen improvement, TRM improvement, decoder readiness, or QA accuracy.
- **Artifacts/code:** `notebooks/03_trm_vlm_qa_training*.ipynb`; `scripts/train_vlm_accelerate.py`; `scripts/evaluate_vlm_accelerate.py`; `configs/trm_vlm_qa_*.yaml`.

### F. Audio / tri-modal experiments — `UNVERIFIED`

- **Evidence:** Audio encoder, tri-modal loss, dataset notes, and paper claims exist.
- **No verified claim:** No retained result artifact supports WER 3.18%, ESC-50 3% accuracy, audio–text retrieval, audio–vision retrieval, or a trained tri-modal system. Treat all as unverified until an immutable result bundle is retained.

---

## 7. Current Research / Engineering Bottlenecks

1. **Canonical training entrypoint is broken.** `scripts/train_alignment.py` calls `MultimodalTrainer`, which passes a batch dict as one positional model argument and indexes a dataclass like a dictionary. This blocks a clean published/run-anywhere alignment command.
2. **Reproducibility gap.** Final results lack source dataset/split manifest, sample count, seed, environment lock, commit SHA, checkpoint hash, runnable relative paths, and raw evaluation outputs. Final notebook relies on unavailable absolute paths and permissive `strict=False` loading.
3. **Perceiver ambiguity and failure.** The package Perceiver differs from the notebook-local V3 evaluated in the negative table; reported table labeling obscures its 512-d implementation. No validated fix/rerun exists.
4. **Decoder results unsubstantiated.** Qwen/TRM code/configurations do not substitute for validated generative metrics. The remaining evaluation notebook shows a missing checkpoint failure.
5. **Audio/tri-modal gap.** Audio support exists in code but has no retained current result artifact. Audio representation normalization also differs from vision/text ordering (`audio.py:161-167`).
6. **Data pipeline mismatch.** Primary config points to unavailable user-specific absolute Parquet paths. `MultimodalDataModule` calls legacy metadata-oriented `build_datasets`, while primary PixMo Parquet builders are separate; this reinforces that the advertised CLI path is not canonical.
7. **Testing/quality gap.** No unit/integration test suite or CI was found. Audit-time `compileall` fails on `scripts/map_notebook.py` with an f-string backslash `SyntaxError`; all YAML configs did load successfully.
8. **Dependency/packaging gaps.** `pyproject.toml` omits imports used by runtime/tooling paths (for example `requests`, `soundfile`, plotting/scikit-learn dependencies); README references missing configs/notebooks and unsupported downloader flags.
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
| “TRM is a validated recursive reasoning model” | UNVERIFIED / misleading | The custom decoder is causal transformer-like; recursion claims are ambiguous across separate code paths and no validated result remains. | “Implemented a custom lightweight decoder and a separate optional recursive Qwen-VLM wrapper.” |
| “Used 2×H200 / H100/A100 clusters in completed training” | UNVERIFIED | Docs/configs contain hardware recommendations/claims, but no immutable run metadata or throughput evidence verifies hardware use. | “Implemented optional DDP/Accelerate/DeepSpeed-aware paths; do not cite hardware counts.” |
| “Distributed training works” | PARTIAL | DDP code exists; `ImprovedMultimodalTrainer` conditionally wraps DDP, but no retained successful distributed run or test exists. | “Implemented optional DDP-aware training infrastructure.” |
| “W&B-tracked experiments” | PARTIAL | W&B integration is coded; useful corresponding run summary/provenance is not retained in current package. | “Integrated optional W&B logging hooks.” |
| “R@5 = 85.69/86.80” | UNVERIFIED / conflicting | `outputs/alignment_evaluation_results.csv` lacks provenance and conflicts with final committed table. | Use only final table’s 77.681/77.547% values with paired-retrieval qualifier. |
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

I built CERBERUS/Edge Glass to test an alignment-first alternative to retraining a monolithic vision-language model. The core problem was turning frozen CLIP image-patch features and sentence-transformer caption embeddings into one fixed-size representation that supports paired retrieval and can later be passed to different decoders. I implemented the PyTorch pipeline: frozen encoders, trainable projections, learned attention pooling, normalized nested prefixes from 4096 to 128 dimensions, and a symmetric contrastive objective. I also built Parquet-based image/QA loaders, retrieval evaluation, configs, and optional Qwen-prefix and lightweight-decoder interfaces.

The strongest result is the persisted PixMo paired retrieval evaluation: 77.7% image-to-text and 77.5% text-to-image Recall@5 at 4096 dimensions. The most informative experiment was the prefix ablation: 512-dimensional prefixes still reached roughly 74% Recall@5, while 128 dimensions fell to about 63–64%. I also kept a negative result: the Perceiver-style compression arm was near random, so attention pooling is the validated path. The main limitation is reproducibility and scope: data/checkpoints and decoder metrics are not retained cleanly, the public alignment CLI needs repair, and audio/tri-modal and end-to-end generation remain ongoing rather than completed.

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
| Main model, fusion, and decoder prefix bridge | `edge_glass_modular/src/models/alignment.py`; `src/models/projector.py`; `src/models/fusion.py` |
| Qwen LoRA/quantization/prefix code | `edge_glass_modular/src/decoders/qwen.py` |
| Custom lightweight decoder code | `edge_glass_modular/src/decoders/trm.py`; `src/models/trm_qwen_vlm.py` |
| Parquet/PixMo/PixMo-QA data loader behavior | `edge_glass_modular/src/data/dataset_builder.py` |
| Data-module/primary CLI mismatch | `edge_glass_modular/src/data/datamodule.py`; `src/data/dataset_builder.py`; `scripts/train_alignment.py` |
| Broken public trainer interface | `edge_glass_modular/src/training/trainer.py`; `src/models/alignment.py`; `scripts/train_alignment.py` |
| Improved trainer/checkpoint/DDP/W&B implementation | `edge_glass_modular/src/training/improved_trainer.py` |
| Accelerate/DeepSpeed QA scripts | `edge_glass_modular/scripts/train_vlm_accelerate.py`; `scripts/evaluate_vlm_accelerate.py` |
| Retrieval evaluation semantics | `edge_glass_modular/src/evaluation/metrics.py` |
| QA metric implementation and scale caveat | `edge_glass_modular/src/evaluation/qa_metrics.py` |
| Final artifact-backed attention/Perceiver results | `edge_glass_modular/outputs/alignment_eval_final/alignment_metrics_all_models.csv` |
| Final attention vs. Perceiver table | `edge_glass_modular/outputs/alignment_eval_final/table_4096_detailed.csv` |
| Final multi-scale R@5 table | `edge_glass_modular/outputs/alignment_eval_final/table_multiscale_combined.csv` |
| Final visualization assets | `edge_glass_modular/outputs/alignment_eval_final/plot_tsne_comparison.png`; `plot_r5_dim_comparison_all.png`; `plot_cosine_hist_comparison.png` |
| Conflicting/noncanonical results to avoid | `edge_glass_modular/outputs/alignment_evaluation_results.csv`; `edge_glass_modular/notebooks/outputs/pixmo_alignment/metrics.csv` |
| Final evaluation notebook/provenance caveats | `edge_glass_modular/notebooks/07_alignment_final_report.ipynb`; `AUDIT_REPORT.md:81-92` |
| Perceiver negative-result diagnosis | `edge_glass_modular/notebooks/PERCEIVER_TRAINING_ISSUE_ANALYSIS.md` |
| QA training/evaluation evidence and missing-checkpoint failure | `edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb`; `03_trm_vlm_qa_training_MODULAR.ipynb`; `04_vlm_qa_evaluation.ipynb`; `05_vlm_qualitative_inference.ipynb` |
| Dataset construction intent / paths / limitations | `edge_glass_modular/reports/dataset_construction_report.md`; `reports/results_report.md` |
| Stale production/readiness claims | `README.md`; `edge_glass_modular/README.md`; `PROJECT_SUMMARY.md`; `edge_glass_modular/COMPLETE_CODEBASE.md`; `QUICK_REFERENCE.md` |
| Audit-time code/config checks | `python -m compileall -q edge_glass_modular/src edge_glass_modular/scripts` (failed only on `scripts/map_notebook.py` f-string syntax); all YAML configs loaded successfully through `load_config` |

---

## Canonical-use rule

Future resume text should cite only claims marked **resume-safe** here, unless a newer commit adds an immutable result bundle with exact data split, checkpoint/model identity, environment, evaluator command, raw outputs, and verified read-back. If newer evidence conflicts with this document, record the conflict and update this file before changing resume claims.
