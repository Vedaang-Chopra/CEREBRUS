# RESUME_PROJECT_BRIEF — CERBERUS / Edge Glass Assistant

**Purpose:** Evidence-backed handoff for a model-training-focused Research Engineer resume.  
**Prepared:** 2026-10-20. **Research result snapshot:** 2025-12-11 (commit `05479e3`). **Latest separately audited local-code snapshot:** 2026-09-21 (through `ca641a8`, with root-housekeeping state reported at `ab3036e`).  
**Scope/verification limit:** The accessible GitHub repository `Vedaang-Chopra/CEREBRUS` has HEAD `693e43b` (2026-01-06). A later, detailed *local repository audit* (`PROJECT_RESUME_CONTEXT.md`, dated 2026-09-21), local `README.md`, and housekeeping audit describe newer changes not pushed to that remote. This document reconciles those records with GitHub source, commits, and saved CSVs; the latest local checkout, checkpoint weights, and raw data are **not directly accessible for fresh execution**. Treat later local state as **previously audited**, not independently re-tested today. Current source and `AUDIT_LEDGER.md` take priority if they differ.

**Status vocabulary:** **Implemented** = inspected source, or implementation established by the September local audit; **Experimentally validated** = surviving output artifact / previously executed test; **In progress** = partial code or incomplete proof; **Planned** = proposal without validated implementation. Code existence alone does not prove successful training.

## A. Project Overview

**Problem.** Can frozen vision/text encoders be aligned using comparatively small trainable modules, yielding useful image–caption retrieval embeddings and nested representations for potential resource-constrained use without end-to-end backbone retraining?

**Artifact-backed approach (Stage 1).** PixMo-Cap image/caption pairs from local Parquet → frozen **CLIP ViT-L/14-336** patch features (1024-dimensional) and frozen **SentenceTransformer `all-mpnet-base-v2`** caption features → trainable projections into a shared **4096-dimensional** space → **SimpleAttentionPooling** over image tokens → L2 normalization → normalized prefixes at 2048/1024/512/256/128 dimensions. Optimize symmetric CLIP-style **InfoNCE** over paired examples plus contrastive supervision at sampled prefix dimensions (Matryoshka-style representation learning). Evaluate cosine-similarity retrieval in both image→text (I2T) and text→image (T2I) directions.

**Experimental extensions (Stage 2).** A Qwen2.5-3B-Instruct VLM wrapper maps aligned visual features to decoder prefix embeddings via `vision_proj`, with optional 4/8-bit loading and PEFT/LoRA. An optional **TRM-inspired** refinement branch applies a shared `TRMLayer` stack to **Qwen's final hidden states** using a fixed **four-iteration y/z loop**, residual addition, and the LM head. This is an external refinement module, **not** a modification to Qwen attention layers, not a standalone TinyRecursiveModel, and not demonstrated as improved reasoning or QA.

**Scope:** A multimodal **research prototype**, with credible vision–text retrieval evidence; audio/Whisper and tri-modal losses remain experimental support. No demonstrated edge deployment, tool-using agent, ML-security evaluation, or RL-based post-training.

**Stack:** Python, PyTorch, Hugging Face Transformers, SentenceTransformers, torchvision, Pandas/Parquet, PIL, NumPy, YAML/Jupyter, PEFT/LoRA, bitsandbytes; optional PyTorch DDP, Accelerate, DeepSpeed, BF16/FP16, W&B. H200/A100 references are setup/configuration context, **not verified benchmark hardware**.

## B. Technical Contributions

| Contribution | Technical mechanism / significance | Evidence level |
|---|---|---|
| Frozen-backbone alignment heads | Project CLIP patch features and MPNet captions into a normalized 4096-d retrieval space; learn image attention pooling rather than relying solely on CLS/mean pooling. Adapts representations without updating the primary encoder weights. | **Implemented + experimentally validated** for paired retrieval. |
| Contrastive and nested-prefix training | Symmetric in-batch cross entropy on image–text similarity and contrastive supervision of L2-normalized coordinate prefixes; sampled MRL dimension reduces per-step auxiliary-loss computation. The MRL module **slices** an existing learned embedding; it is **not** a trained cascade of separate projection heads. | **Implemented + experimentally validated** via dimension ablation. |
| Perceiver alternative | Learned latent-query compression, cross-attention, latent self-attention/FFN in later canonicalized Perceiver code. Historical evaluation and training implementations were inconsistent, with permissive checkpoint loading. | **Implemented; historical experiment recorded but comparison provenance compromised**. |
| Qwen visual-prefix bridge | Project aligned vision-token features into Qwen2.5-3B hidden size, prepend prefix embeddings and mask non-answer positions; optional LoRA/quantization; TRM-inspired hidden-state refinement in fixed four steps. | **Implemented; generation quality unverified**. |
| Training/evaluation/data tooling | Parquet-embedded image/caption and QA datasets, image validation/transforms, AdamW/warmup-cosine training, accumulation/clipping, checkpoint support, bidirectional Recall@K/MRR/median-rank evaluation; optional distributed scripts. | **Implemented**; the September audit reports **48 passing CPU-only regression tests**, but does **not** verify a clean distributed end-to-end training run. |

**Ownership note:** Git history/audit identify Vedaang as the primary implementation contributor, but at least one collaborator-authored commit exists. Do not imply exclusive authorship.

## C. Latest Developments

**Chronology and authority.**

- **2025-12-09 to 2025-12-11 (GitHub commits):** training/trainer bug fixes (`ba01b82`), TRM fixes (`a96d13b`), reports/configs (`5229a91`, `b6e3113`), and final alignment tables/notebook (`05479e3`). **2026-01-06:** `693e43b` synchronizes README/paper/images; this is the most recent accessible GitHub HEAD, **not** the latest local code state.
- **2026-08-17 onward (later local audits):** `AUDIT_REPORT.md` (`4113a26`) found source/evaluation provenance gaps; the newer `AUDIT_LEDGER.md`, `PROJECT_RESUME_CONTEXT.md`, and rewritten `README.md` supersede some historical blockers.
- **Through 2026-09-21 (`ca641a8`; separately reported root state `ab3036e`):** canonical main path retained; old broken alignment CLI/trainer family, unused fusion and Path-C projector, and standalone `TRMDecoder` retired from the **audited local active tree**. Packaged Perceiver implementation aligned to the executed V3 training computation; regression coverage expanded to **48 tests**, covering generation/EOS, checkpoint integrity, device placement, prefix masking, alignment losses, and config parsing. Canonical `ImprovedMultimodalTrainer` remains **notebook/script-driven, without a maintained CLI**. Existing result CSVs were preserved rather than silently recalculated.

**Implementation states:**

- **Experimentally validated:** Attention-pooling PixMo paired retrieval, six prefix-size retrieval measurements, persisted plots/CSV; previously executed 48-test regression suite (software correctness, not research efficacy).
- **Implemented, not outcome-validated:** Qwen prefix embedding/LoRA/quantization interfaces; four-step recursive hidden-state refinement; Whisper encoder, tri-modal loss scaffolding; Accelerate/DeepSpeed/DDP-aware code paths.
- **In progress / evidence gaps:** Strict, matched Perceiver rerun; provenance-complete alignment reproducibility; checkpoint/data reconstruction; end-to-end QA baseline/refinement evaluation.
- **Planned or unsupported:** New clean alignment CLI, proven tri-modal training, inference/edge benchmarks, public serving, agents/tool use, security/robustness studies, RL/DPO/GRPO post-training.

**Outdated statements:** The accessible GitHub README and historical `FINAL_STATUS.md` claim “production-ready,” a standalone custom TRM decoder, old fusion/config profiles, and prospective efficiency numbers. These are **not** current resume claims. The later local README/audit explicitly correct them.

## D. Experimental Results

**Evaluation protocol:** Paired PixMo validation image/caption retrieval with normalized embeddings and an `N × N` cosine similarity matrix; each row's same-index example is the assumed positive. Recall@K percentages are **retrieval**, not generation accuracy. Documentation describes ~3,000 validation examples; ~2,984 effective examples is **inferred**, not recorded by the final result file. No checkpoint checksum, exact split manifest, evaluator environment, seed provenance, confidence interval, or raw ranked-prediction bundle survives.

| Model / representation | I2T R@1 | I2T R@5 | T2I R@5 | Status |
|---|---:|---:|---:|---|
| **Attention pooling, 4096-d** | **50.87%** | **77.68%** | **77.55%** | **Verified saved result**; I2T/T2I R@10 = 85.96%/85.89%; MRR = .629/.631. |
| Attention pooling, 2048-d prefix | 50.37% | 76.64% | 77.75% | Verified same-model prefix ablation. |
| Attention pooling, 1024-d prefix | 48.12% | 75.67% | 76.81% | Verified same-model prefix ablation. |
| **Attention pooling, 512-d prefix** | 46.45% | **73.73%** | **74.93%** | Verified: 8× fewer embedding coordinates than 4096-d; **not** measured 8× latency/cost savings. |
| Attention pooling, 256-d prefix | 42.19% | 70.41% | 70.64% | Verified same-model prefix ablation. |
| **Attention pooling, 128-d prefix** | 33.45% | **62.94%** | **64.04%** | Verified: 32× fewer coordinates, with meaningful recall loss. |
| Historical Perceiver-style arm | 0.034% at reported 4096 label | 0.201% | 0.168% | **Recorded negative outcome only.** Actual arm is 512-d; 4096-d label is misleading; evaluation architecture/checkpoint loading mismatched training. **Do not use as a causal architecture comparison.** |

**Experimental interpretation:** A trained, non-Perceiver attention/projection path produces useful paired retrieval; prefix truncation retains substantial utility through 512 dimensions but degrades further at 128. The historical Perceiver result is near random yet **provenance-compromised**, not proof that Perceiver compression is intrinsically worse.

**Other data to exclude:** `alignment_evaluation_results.csv` (61.03% I2T R@1, 85.69% R@5) conflicts with the final authoritative table and lacks sufficient run identity; a one-row historical training `metrics.csv` similarly lacks a reproducible split/metric protocol. Small-sample audio retrieval, WER, TRM exact match/F1, and paper BLEU/ROUGE improvements lack a provenance-clean current result bundle. Do not mix them into the reported final experiment.

**Reproducibility limitation:** Final evaluation notebook relied on absolute paths and historically used `strict=False` checkpoint loading; required data and complete checkpoints are not in the accessible repository. Accordingly call these **saved, artifact-backed results**, not freshly reproduced scores.

## E. Research Engineer Relevance

1. **High — Model architectures / representation learning:** learned attention pooling; frozen-backbone adapters; shared normalized cross-modal space; nested coordinate-prefix supervision.
2. **High — Model training:** implemented symmetric InfoNCE and sampled-prefix loss, optimization/training loop, checkpoint/config wiring. This is **alignment-head training**, not CLIP/MPNet full fine-tuning or foundation-model pretraining.
3. **High — Evaluation / research methodology:** bidirectional retrieval, six-scale ablation, reported negative experiment plus failure/provenance analysis.
4. **Medium — Decoder architecture / parameter-efficient adaptation:** Qwen prefix conditioning, LoRA/quantization support, TRM-inspired recursive hidden-state refinement. **No resume-safe improved-QA metric**.
5. **Supporting — ML systems:** loaders, mixed precision, optional distributed orchestration, 48-test regression suite; no demonstrated throughput, speedup, deployment, or distributed-scale result.
6. **Not supported here — ML security / agents / RL:** no attacks, defenses, verifiers, tool-use loops, RAG, agent orchestration, or RL post-training evidenced. Use other projects for these specialties.

## F. Resume-Ready Contributions

Choose **3–4** of these for a research-training CV; the fifth is optional.

- Designed and trained a **frozen-backbone vision–text alignment pipeline** with CLIP ViT-L/14-336, MPNet, learned attention pooling, and symmetric InfoNCE to learn normalized 4096-d retrieval embeddings.
- Evaluated **bidirectional PixMo image–caption retrieval**, obtaining **77.7% image→text and 77.5% text→image Recall@5** at 4096 dimensions in persisted validation results.
- Implemented **Matryoshka-style contrastive supervision** on nested 4096→128-dimensional prefixes; measured **73.7%/74.9% I2T/T2I Recall@5** using 512-d representations versus 77.7%/77.5% at full dimension.
- Implemented a **Qwen2.5-3B multimodal prefix-conditioning path** with optional LoRA/quantized loading and **four-step TRM-inspired hidden-state refinement**; added regression coverage for generation, EOS handling, and checkpoint restoration. *(Implementation result, not a claim of QA improvement.)*
- Built the **PyTorch training/evaluation pipeline** for Parquet-embedded PixMo data, configurable contrastive losses, checkpointing, and bidirectional retrieval metrics; retained auditable CSVs and analyzed a **provenance-limited Perceiver ablation**.

**Suggested short set:** bullets 1–3 plus bullet 4 only when the role emphasizes VLM architectures/post-training implementation. The implementation of a refinement loop is not evidence of RL training.

## G. Evidence and Claim Audit

Paths below are relative to the local project root unless marked **GitHub historical**. The later local audit moved output artifacts from the historical GitHub `outputs/` folder into curated `results/`.

| Claim | Primary evidence | Verdict |
|---|---|---|
| CLIP/MPNet frozen, 4096-d projections, attention pooling | `edge_glass_modular/configs/pixmo_alignment.yaml`; `src/encoders/{vision,text,pooling}.py`; `src/models/alignment.py`; GitHub historical config `configs/pixmo_alignment_1.yaml` | **Verified implemented** |
| Symmetric contrastive and MRL-prefix objective | `edge_glass_modular/src/models/losses.py`; `src/encoders/mrl.py` | **Verified implemented** |
| Final I2T/T2I 4096-d R@5 | `edge_glass_modular/results/alignment/alignment_eval_final/alignment_metrics_all_models.csv`; `table_4096_detailed.csv`; commit `05479e3` | **Verified persisted experiment; not independently reproduced** |
| Six-dimensional prefix ablation | Same final CSV; `table_multiscale_combined.csv`; `notebooks/experiments/07_alignment_final_report.ipynb` | **Verified persisted experiment** |
| Historical Perceiver near-random | Same CSV; `notebooks/experiments/PERCEIVER_TRAINING_ISSUE_ANALYSIS.md`; `scripts/experiments/train_perceiver_v3.py`; `AUDIT_LEDGER.md` | **Verified recorded outcome; architecture conclusion invalid without rerun** |
| Qwen LoRA/4-bit/prefix conditioning and iterative refinement | `edge_glass_modular/src/decoders/{qwen,trm}.py`; `src/models/trm_qwen_vlm.py`; `configs/trm_vlm_qa_qwen2.5-3b_regularized.yaml`; regression tests | **Verified implementation; no validated quality lift** |
| Optimizer, checkpoints, optional DDP/Accelerate | `src/training/improved_trainer.py`; `scripts/train/train_vlm_accelerate.py`; `src/models/checkpoint_loader.py` | **Implemented support; successful full distributed run unverified** |
| Regression-test cleanup/current architecture | `edge_glass_modular/tests/`; local `README.md`, `AUDIT_LEDGER.md`, `PROJECT_RESUME_CONTEXT.md` (2026-09-21 audit) | **Previously audited 48 passing tests; not rerun here** |
| Exact model/data/seed/checkpoint/environment provenance | Missing immutable run bundle, Parquet manifests, complete checkpoint identity | **Partially supported / unresolved** |
| Qwen fine-tuning success, TRM/Perceiver generation lift, full tri-modal training | Training plans and notebook traces but no provenance-clean committed end-to-end outputs | **Unverified — do not claim** |
| Edge deployment, inference savings, H200 scaling, RL post-training, autonomous agents, ML security, published paper | No supporting execution/deployment/benchmark/publication evidence | **Do not claim** |

**Conflict/retirement audit:** Historical `README.md`, `FINAL_STATUS.md`, `PROJECT_SUMMARY.md`, and older GitHub code advertise more than current evidence supports. The September 2026 local audit supersedes them: fusion, Path C, standalone TRM decoder, and old CLI are **retired locally**; the public GitHub snapshot still contains older versions of some modules. Do not infer a currently maintained feature merely because it remains in that older remote.

**Evidence sources / commit anchors:**
- Audited local: `README.md`; `PROJECT_RESUME_CONTEXT.md` (2026-09-21); `AUDIT_LEDGER.md`; `AUDIT_REPORT.md` (historical 2026-08-17); `edge_glass_modular/tests/`.
- Persisted local canonical artifacts: `edge_glass_modular/results/alignment/alignment_eval_final/{alignment_metrics_all_models.csv,table_multiscale_combined.csv,table_4096_detailed.csv}`.
- Historical, directly inspected GitHub artifacts: `edge_glass_modular/outputs/alignment_eval_final/` (same final CSV values); `src/encoders/{vision,text,mrl,perceiver}.py`, `src/models/{alignment,losses,trm_qwen_vlm}.py`, `src/decoders/{qwen,trm}.py`, `src/training/improved_trainer.py`, `src/evaluation/metrics.py`.
- GitHub: `05479e3` (2025-12-11 final results), `693e43b` (2026-01-06 remote HEAD). Later local audit: `4113a26` (Aug audit), `ca641a8` (Sep cleanup), `ab3036e` (Sep root-housekeeping snapshot).

## H. Recommended Resume Emphasis

**Ranked top three:** **(1)** Frozen CLIP/MPNet + learned attention-pooling alignment architecture with symmetric contrastive training; **(2)** final measured I2T/T2I PixMo paired retrieval; **(3)** multi-scale MRL prefix training and six-dimension retrieval ablation.

**Emphasize:** architectural ownership of trainable alignment modules, loss design, artifact-backed retrieval results, and controlled representation-size trade-offs. **Compress:** broad modularity, tool/library lists, trainer infrastructure; mention Qwen/refinement only in a secondary architecture bullet. **Omit:** audio/tri-modal success, Perceiver superiority claims, decoder quality gains, RL/post-training-as-RL, agents/verifiers/security, hardware speedups, production/edge deployment, and publication claims.

**Next evidence needed to upgrade claims:** exact dataset split hash/effective N, model/checkpoint identity and complete weights, seeded evaluator command/environment, per-query rank/prediction archive, strict architecture-matched Perceiver rerun, and paired Qwen-with/without-refinement QA predictions with baseline/metric definitions. Until these exist, retain the conservative wording above.
