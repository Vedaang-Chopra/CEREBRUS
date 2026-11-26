# Status — where we are and what is next

Updated by main-discussion after every stage. Last update: **Wed Nov 25, 2026**.
**How to build the project:** `docs/DEC3_MVP.md` is the plan until Dec 3 (stages S0–S17, each with a Done-when). After Dec 3: `docs/IMPLEMENTATION.md` (full paper). This file only tracks progress.

## Your loop for every stage
1. An agent writes the code + its notebook; main-discussion reviews and syncs it to Inara.
2. **You run the notebook** on Inara (VS Code, `learning_env` kernel, GPUs 2–3), save it, and say "NN done".
3. main-discussion reads the saved outputs, checks the Done-when, and asks you for any decision. Decisions go into `docs/decisions.md`.

## The plan now (Vedaang, Nov 25 evening): small steps, 20 samples, then post-training

One step at a time. Each step = one notebook run on ~20 items per benchmark, read by main-discussion, one decision.
No code is written for a step before the previous step's result is in. Infrastructure for full-scale runs goes to teammates.

| # | Step | Notebook | Question it answers | State |
|---|---|---|---|---|
| 1 | S2 answer format | 04 | P (no thinking) or T (thinking, 256 budget)? Parse ≥95%? | **done: P for the POC**, thinking for final results |
| 2 | S3/S3b checks | 05 | k=100% pruning exact? pass p of R=5 = fresh R=p? | **done: both yes (bit-exact)** |
| 3 | Loops at full tokens | 06 (Step-1..3 on 20 items) | does accuracy change with R = 1..5? batched = batch 1? | **done**: batched = batch 1; R trend within noise at n=20; FLOPs model within ±25% |
| 4 | Tokens × loops, small | small k × R table on 20 items (k ∈ 25/50/100%, R ∈ 1/3/5) | how much does accuracy drop at low k? (this is what the LoRA must fix) | not built: one small addition after step 3 |
| 5 | Post-training (S8 teacher cache, S9 budget-tolerance LoRA) | 10 | does LoRA recover accuracy at low k without hurting k=100%? | after step 4; draft code parked on branch `wip/post-training` |

**For teammates (later, full scale):** S7 grid runner at scale, S4/S5 features + selectors, S6 cost ledger details, S11–S13 controller + baselines (draft on branch `wip/controller-teammates`), S14–S15 analysis. Roles in `docs/TEAM_PLAN.md`.

## Stages (DEC3_MVP §7)

| Stage | What | Notebook | State | Next action |
|---|---|---|---|---|
| S0 | Env, package, model load, loop control, pruning port | 01, 02 | code done; env pinned to `learning_env` | archive 01/02 once you commit your edits; `scripts/mvp.py smoke` not run yet |
| S1 | GQA + TextVQA loaders, splits, scorers, eval runner | 03 | **data checks PASS** (splits frozen, `configs/splits/`) | you: finish the evaluation section |
| S2 | Thinking off, `res_px` | 04 | **done**: no thinking for the POC (P: GQA 60, TextVQA 59 on 20 items) | res_px 768/1024 not compared → **resolution check today** |
| S3 | Original positions after pruning | 05 | **done**: not supported → pruning stays packed; k=100% bit-exact | — |
| S3b | Layer-call count, pass-p state vs fresh R=p run | 05 | **done**: bit-identical for p=1..5 (n=20) | — |
| S4 | Per-pass features + compression methods (crisp_lite, ReDiPrune, DivPrune, VisionZip + controls) | 07 | **done** (1-item checks on real weights, features saved) | — |
| S5 | Compression comparison (R=3, k 25/50/100%, accuracy + FLOPs + latency) | 07 | **done: divprune frozen** in `configs/mvp.yaml` (see docs/decisions.md) | |
| S6 | FLOPs model + throughput → item counts | 06 | FLOPs ratio check PASS, throughput measured | grid timing (`RUN_GRID_TIME`) → budget line |
| S7 | Base grid 15 cells (divprune) + noise re-run | 08 | code ready | 1 item → 20 → full (ctrl_train + dev) |
| S8 | Multi-depth teacher cache (100% tokens, R=1..5, all lora_train) | 09 | code ready, not run on GPU | you: run 09, paste Step-2; then the full run (GPUs 0–7 allowed for it) |
| S9 | Two LoRAs: A depth-matched (teacher R = student R), B `fixed:5` teacher for compressed samples (default, to be confirmed; `fixed:3` still works) | 10 | code written, CPU-checked on a tiny model, not run on GPU; training waits for S8 validation + your approval | you: approve the B depth; run 10 on Inara and paste the outputs; peft==0.22.0 pinned in `environment.yml` / `pyproject.toml`; training (`scripts/mvp.py train --variant A\|B`) and nb10 refuse `compressor: none` (frozen to divprune, S5) |
| S10 | Grids for base, A, B (15 cells each) | — | not started | |
| S11 | Action tree from grid | — | not started | |
| S12 | Controller (heuristic + learned) | — | not started | |
| S13 | Baselines | — | not started | |
| S14 | Analysis, headline, prereg verdict | — | not started | `configs/prereg.yaml` before S12 |
| S15 | Ablations + error analysis | — | not started | |
| S16 | `mvp_results.ipynb` | — | not started | |
| S17 | README_MVP, release, repro | — | not started | |

**Day plan (DEC3_MVP §7):** Wed 25: S0–S3b · Thu 26: S4, S5, S6, S8 · Fri 27: S7 launch, S9 · Sat 28: S9, S10, S11 · Sun 29: S12–S14 · Mon 30: S15, S16, results freeze · Tue Dec 1: S17 + report · Wed Dec 2: video · Thu Dec 3: submit.

**S5b prep (code):** `kv_cache_bytes` (analytic KV cache size after prefill) is on evaluation, S5 and grid rows; checked on the tiny model only. The S5b Pareto sweep and `cls_iqr` are built (nb07 step 7, `RUN_PARETO`), tiny-model check only.

## Open decisions (yours)
- S2: `answer_prefill`, `res_px`, `max_new_tokens` (after notebook 04).
- S3: log "pruning packed; original positions unsupported" + the S3b table in `docs/decisions.md` (after notebook 05).
- Sync rule: let implementation sessions run `scripts/sync_inara.sh` themselves?
- Hermes: approve the install in the `basic-code-changes` session; commit `.claude/skills/` or keep local?
- `configs/prereg.yaml`: the "what changed since Oct 1" sentence (needed before S12).

## Later to-do (run separately, after the main path)
- **S5b Pareto sweep (built, nb07 step 7):** every method at k ∈ {10, 25, 50, 75, 100}% plus `cls_iqr` (TRIM IQR rule on CLS attention; kept count per image) at m ∈ {0, 0.5, 1, 1.5, 3}, R = 3, dev, 200 items → accuracy vs GFLOPs, kept tokens and KV cache bytes with the Pareto front (F06). S5 25 / 50 / 100% rows are reused; 21 new cells = 4,200 answers + 420 for latency, about 45 min at batch 1 on 2 copies. Run after the S8 cache.

## Today (Nov 25–26): inference and baselines through S8 — run list
Vedaang decision: no post-training today; finish every inference baseline and the S8 dataset; S9 starts tomorrow.

| # | Run | Where | Gives |
|---|---|---|---|
| 1 | ~~nb09~~ **paused**: S8 dataset design (top-32 distillation vs SFT / reward data) under review by Vedaang | — | — |
| 2 | nb07 `RUN_BATCH_CHECK` + `RUN_BATCH_DIAG` | nb07 | may the grid batch? |
| 3 | `scripts/mvp.py smoke` | terminal | S0 Done-when |
| 4 | nb06 `RUN_GRID_TIME` | nb06 | S6 budget line, grid item counts |
| 5 | nb06 `RUN_FULL` (k=100%, R=1..5, dev + ctrl_train) | nb06 / CLI | F05 accuracy vs R |
| 6 | Resolution check (512 / 768 / 1024 px, full and DivPrune at matched tokens) | nb04 step 5, `RUN_RESOLUTION` (built, tiny-model check only; ready to run) | does a sharper image + pruning beat 512 px? |
| 7 | S5b Pareto sweep + KV bytes; profiling | nb07 step 7, `RUN_PARETO` (built, ready to run) / nb06 profiling (not built) | F06 curves, where time goes |
| 8 | nb08 1 item → 20 → full grid + re-run | nb08 / CLI, GPUs 0–7 | S7 grid.parquet, noise rate, F07, F08 |
| 9 | ~~Full S8 cache~~ **paused** (see #1) | — | — |
| 10 | Analysis from the grid: best_fixed, oracle, noise oracle, headroom (T2, F09) | code (to build), seconds | headroom claim |
Not today (planned later): S5 extra 10/75% curve is inside #7; POPE/hallucination test, evidence retention, thinking study (post-MVP / final runs).
