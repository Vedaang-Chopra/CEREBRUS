# Handoff — main-discussion (supervisor) session, Nov 26, 2026 (~04:00 EST)

Paste the block below into a new Claude Code session in this repo. Name the session `main-discussion`.

---

You are the **supervisor / planning session** for AdaLoop-VLM (repo `cerebrus`, branch `main`). You plan with
Vedaang, brief the implementation agent, review its commits, sync code to Inara and read run results. **You don't
write project code yourself**; small docs, configs, notebook flag edits and tooling are fine.

**Talk to Vedaang** in very simple language, bullet points, short tables, short precise answers. Challenge his
proposals when evidence says otherwise; don't just agree.

## Read first, in order
`AGENTS.md` (rules: notebooks are the tests, his notebook/code style, prints not logging, GPU rule, sync) →
`docs/STATUS.md` (run list "Today", stage table, later to-do) → `docs/decisions.md` (every decision with numbers; read
fully) → `docs/DEC3_MVP.md` (Dec 3 spec; S8/S9 rows are now under redesign, see below) → `docs/notes/s8_design.md`
(post-training design review, superseded in parts by §5 below) → `docs/notes/compression_baselines.md` →
`docs/notes/evaluation.md` (eval protocol, batch nondeterminism) → `docs/notes/modeling_code.md` → `docs/FIGURES.md`
→ `notebooks/README.md` → `docs/ai_context/CODEBASE_MAP.md`. Your auto-memory has Inara details and working rules.

## 1. How we work
- Vedaang decides. One step at a time; he runs notebooks himself on Inara; you read results over ssh.
- **Implementation agent** = Claude session `build-minimal-version-project` (ListAgents → SendMessage). Its context was
  cleared several times: each brief must be self-contained (point to `docs/04_prompts/*.md`). Rule (Vedaang): it
  **supervises only**; code is written by its subagents (Sonnet for model-critical code, Haiku for mechanical edits,
  Hermes delegate skill for independent pieces in parallel); it reviews diffs, runs CPU checks, commits `agent:` to
  main, reports hashes. Never pushes, never runs on Inara, asks before `rm -rf`.
- **Review → sync:** `./scripts/sync_inara.sh` (pushes committed main to `/coc/scratch/vchopra/cerberus_v2`, backs up
  executed notebooks to `execution_results/notebook_runs/`). It aborts if a notebook on Inara changed in the last
  20 min; then back up by hand (`cp` to notebook_runs + `git checkout --`) **only when no run is active** (check
  `nvidia-smi`). Never sync while a notebook runs (autoreload would change code under it).
- **Before saying a notebook is ready:** set its flags in the file for the agreed run, sync, then grep the flags on
  Inara and list them for Vedaang. He should only press Run All.
- **Read results:** `ssh inara 'cd /coc/scratch/vchopra/cerberus_v2 && python3 scripts/nb_report.py notebooks/<nb>.ipynb --all'`
  and files under `execution_results/`.
- **Terminal runs on Inara:** never `source Deep-Learning/with-hf-cache.sh` without arguments (it execs `jupyter
  lab`). Use `/coc/scratch/vchopra/learning_env/Deep-Learning/with-hf-cache.sh /coc/scratch/vchopra/learning_env/learning_env/bin/python scripts/mvp.py ...`.
- **GPUs:** 8× A40 (46 GB). One BF16 model copy = 2 GPUs (~28 GB each). Notebook checks/small runs: GPUs 0–3
  (`gpu_groups`). Big runs (Pareto sweep, grid, full baseline, S8, S9 training, S10, final tests): all 0–7
  (`full_gpu_groups`, 4 copies). Placement keys are unhashed.
- Budget: subagents only for real research/audits (max ~3 at once); ask before installs, pushes, secrets.

## 2. Fixed decisions (details and numbers in decisions.md)
| Topic | Decision |
|---|---|
| Answer path (S2) | No thinking for the POC: prefill `<|message_model|><|content_text|>ANSWER:`, strict `ANSWER:` extraction. Thinking (larger budget) only for final results |
| Resolution | **res_px 768** (TextVQA 55.8 → 68.8 at full tokens vs 512) |
| Pruning | packed (original positions unsupported); k=100% bit-exact; pass-p state of R=5 = R=p run (bit-identical) |
| Compressor (S5 winner) | **divprune** (re-check at 768 pending in nb07) |
| Decoding | greedy, batch 1 everywhere (batched outputs differ: 13/128 even with equal-length batches; BF16 kernel effect, Thinking Machines "Defeating Nondeterminism") |
| Test split | sealed until S12; grids on ctrl_train + dev |
| Eval data role | **GQA + TextVQA are for evaluation / hypothesis testing only** (Vedaang, Nov 26), not for training |
| Methods | controls random, uniform, text_mean, lowres; crisp_lite (reimplemented), rediprune (post-projector), divprune, visionzip (official code); dynamic `cls_iqr` (TRIM IQR rule on CLS attention). All budget-based except cls_iqr |

## 3. Results so far (dev, R=3 unless noted)
- S2: P (no thinking) GQA 60 / TextVQA 59, ~2 s/item; T (thinking 256) 60 / 63, ~90 s/item, 30–35% forced.
- nb05: GQA accuracy R=1..5 = 50/60/60/65/65 (n=20). nb06: noisy at n=20; FLOPs model within ±25% of measured.
- S5 at 512 px (n=100/benchmark): DivPrune best (25%: 47.7 all; 50%: 52.2), full 57.4; text_mean below random;
  TextVQA loses most. Batch-1 latency barely follows FLOPs (overhead-bound).
- Resolution: 768 full 62.9 all (TextVQA 68.8); DivPrune 768 → 512-px count 59.5 (TextVQA 63.1 vs 55.8 at 512 full).

## 4. Where runs stand (at handoff)
| Notebook | State |
|---|---|
| nb07 (Pareto at 768: 8 methods × k 10/25/50/75/100% + cls_iqr m 0–3, 200 dev items, 4 copies, batch 1; winner re-check; F06) | **running** on Inara (started ~03:25; ~1,060 rows written at 04:00). When done: read the winner check at 768, the Pareto table, the batch diagnostic (`nb07/batch_diag.csv`); log in decisions.md |
| nb08 (S7 grid: 1 item → 20 → full ctrl_train+dev 15 cells + 100-item re-run; F05 from k=100% cells, F07, F08, noise rate, headroom T2/F09) | **prepared** in commit `7046a70` (ALL_GPUS True, RUN_FULL True, batch check off), **not synced** yet (nb07 running). Sync after nb07, then grep flags |
| nb06 (profiling: per-stage times, token-count scan, torch trace; RUN_PROFILE True, RUN_ESTIMATE False) | prepared in `7046a70`; run **alone** after nb08 (timing needs idle GPUs) |
| nb04 resolution | done |
| nb09 / nb10 (S8 teacher cache, S9 LoRA) | **paused**: post-training redesign (§5) |

Open housekeeping: a stray `jupyter-lab` (PID 3822961) started by a research agent may still run on Inara; ask
Vedaang to `kill 3822961` if it does (Vedaang's own server is PID 2367500). Uncommitted local nb04 key-order diff and
`docs/notes/inference_server_setup_generic.md` are not ours: leave them. `configs/prereg.yaml` does not exist (needs
Vedaang's "what changed since Oct 1" sentence before S12).

## 5. Post-training: the open design (the main topic of the new session)
Goal: a LoRA on the shared core layers (6–17 attention) so the looped VLM tolerates 25/50% visual tokens at every
depth R=1..5, keeping full-token accuracy; experiments A (depth-matched teacher) vs B (fixed R=3 teacher, per
Vedaang's spec; S9 code default is `fixed:5` and must change to `fixed:3`).

**Vedaang's direction (Nov 26):**
- Training data must be a **varied mix** (documents, charts, counting, OCR, general VQA, free-form; e.g. PixMo
  family), few samples per source, varying question types; GQA/TextVQA stay evaluation-only.
- He wants to vary loops across the data (e.g. some sources at R=3, others at R=5).
- He leans to **on-policy** distillation (student answers, frozen full-token teacher scores it): then the work is the
  dataset strategy + loss, and training is "inference + scoring". Offline's advantage: 4 model copies generate all
  teacher data up front, training just reads it. Decide from hardware and cost.
- Question raised: mix SFT (CE on ground truth) with KL?

**Supervisor positions so far (to discuss, not decided):**
1. Data: verify candidates before choosing (licenses, image availability, sizes, answer lengths). PixMo ships image
   URLs (link rot risk); The Cauldron (HuggingFaceM4) bundles ~50 datasets with embedded images (VQAv2, DocVQA,
   ChartQA, TextVQA, OCR-VQA, AI2D, …); counting: PixMo-Count / TallyQA. Official GQA train_balanced (943k q, 72k
   images, ~10 GB) and TextVQA train (34.6k q, cached on Inara) are also available. Audit of the old eval-carved
   `lora_train`: GQA had only 198 images for 6,056 questions → not usable for training.
2. Loops: for a clean A vs B, R must be sampled per sample identically in both runs and independent of the data
   source; per-dataset depth schedules = a later curriculum experiment.
3. Loss: KL(teacher ‖ student) + λ·CE(ground truth), same λ in A and B, ablation λ=0 (precedents: TBD CE+KL, LFM2
   KL+CE, ViCO KL only). Forward KL for short teacher-forced answers; reverse KL / JSD for on-policy and long answers
   (MiniLLM, GKD, Glance-ETD uses JSD α=0.5 on top-100 + tail).
4. No verifier is needed for distillation (offline or on-policy): the teacher gives per-token log-probs. Verifiers
   only for reporting accuracy, RL rewards (GRPO) or filtering.
5. Teacher implementation: one model per process, teacher = LoRA disabled (`with peft_model.disable_adapter(),
   torch.no_grad()`), 100% tokens at the teacher's R; student = LoRA on, pruned (Token-Budget Distillation 2608.28138
   §3.1 does this). Full-vocabulary KL, so top-K/K choice disappears (top-K + residual = fallback if too slow).
   Teacher-forced scoring must use the same forward path as the student (current S8 code uses generation logits —
   fix). Batch 1 (no padding).
6. On-policy evidence: Thinking Machines OPD blog (per-token reverse KL as reward, 9–30× cheaper than RL/SFT);
   LT-OPD 2609.32353 (offline fixed-prefix 77.6% vs on-policy 81.5–82.3% retention at 5% tokens, long answers, 14k
   prompts, 175 updates, 15 h on 8 A100); Glance-ETD 2611.20006 (on-policy self-distillation, 5% tokens, 14k prompts,
   4.7 h on 8 A800). For 1–5-token answers exposure bias is small (inference), but a varied mix with free-form
   answers makes on-policy more relevant.
7. Hardware sketch: per training run one copy (2 GPUs) holds student+teacher via the adapter toggle → A and B can train
   in parallel on 4 GPUs, 4 GPUs left for eval; on-policy adds ≤ a few decode steps per sample (short answers) or
   more for free-form data.

Research already done (summaries in s8_design.md): KL truncation (LFM2 2511.23404 §3.3 decoupled top-K K=32; Sparse
Logit Sampling 2503.16870 bias of top-K, importance sampling; Gemma 3 2503.19786 256 sampled logits; Efficient KD
2608.03796 offline top-K ≈ online; MiniLLM; GKD 2306.13649), loop distillation (LoopOPD 2610.10623, LoopFormer
2602.11451, ELT 2604.09168, MELT 2605.07721 — all deep-teacher, none with token budgets), compression post-training
(ViCO / InternVL3.5 2508.18265, TBD 2608.28138, GRIP-VLM 2605.13375, StepPrune 2609.13804 — selector only).

**Code that exists (main, unused, paused):** S8 teacher cache (`core/train/teacher.py`, `teacher_store.py`,
`orchestration/teacher_cache.py`, nb09) and S9 LoRA A/B (`core/train/lora_budget.py`, `orchestration/train_lora.py`,
`configs/train_lora.yaml` with `lora.teacher_R` = same | fixed:N | plus:N, nb10, `scripts/mvp.py train --variant`).
peft 0.22.0 installed on Inara. It will need rework for the chosen design (data mix, online teacher, CE term,
on-policy option).

## 6. First steps in the new session
1. Tell Vedaang in 5 lines where things stand; check nb07 (read results if done; then sync nb08 and give flags).
2. Start the post-training design discussion from §5: data strategy first (one research agent to verify the
   training-mix datasets), then offline vs on-policy with a cost estimate on our hardware, then loss.
3. Write the agreed design into `docs/DEC3_MVP.md` S8/S9/S10 + `decisions.md` before any code brief.
