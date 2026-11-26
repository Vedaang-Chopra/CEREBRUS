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
| S2 | Thinking off, `res_px` | 04 | **done**: no thinking for the POC (P: GQA 60, TextVQA 59 on 20 items, ~2 s/item) | thinking with a larger budget only for final results |
| S3 | Original positions after pruning | 05 | code done: **original positions not supported by the model** (distances, masks, convs use token index; see `docs/notes/modeling_code.md`) → pruning stays packed (DEC3 fallback) | you: run 05, confirm k=100% bit-exact |
| S3b | Layer-call count, pass-p state vs fresh R=p run | 05 | code done (theory: pass p of R=5 equals R=p) | you: run 05 Step-8 table on real weights |
| S4 | Per-pass features + compression baselines from official code (crisp_lite, ReDiPrune, DivPrune, VisionZip + controls incl. lowres) | 07 | code done (f5e18fb, bca6a36), not run on real weights | you: run nb07, paste per-method output |
| S5 | Compression comparison (R=3, k 25/50/100%, accuracy + FLOPs + latency) | 07 | agent building | |
| S6 | FLOPs model + throughput → item counts | — | not started | can start now (no dependency on S2) |
| S7 | Base grid 15 cells + noise re-run | — | not started | needs S2–S6 |
| S8 | Teacher cache | — | not started | needs S2, S5 |
| S9 | Budget-tolerance LoRA | — | not started | needs S8; peft not yet installed |
| S10 | LoRA grid | — | not started | |
| S11 | Action tree from grid | — | not started | |
| S12 | Controller (heuristic + learned) | — | not started | |
| S13 | Baselines | — | not started | |
| S14 | Analysis, headline, prereg verdict | — | not started | `configs/prereg.yaml` before S12 |
| S15 | Ablations + error analysis | — | not started | |
| S16 | `mvp_results.ipynb` | — | not started | |
| S17 | README_MVP, release, repro | — | not started | |

**Day plan (DEC3_MVP §7):** Wed 25: S0–S3b · Thu 26: S4, S5, S6, S8 · Fri 27: S7 launch, S9 · Sat 28: S9, S10, S11 · Sun 29: S12–S14 · Mon 30: S15, S16, results freeze · Tue Dec 1: S17 + report · Wed Dec 2: video · Thu Dec 3: submit.

## Open decisions (yours)
- S2: `answer_prefill`, `res_px`, `max_new_tokens` (after notebook 04).
- S3: log "pruning packed; original positions unsupported" + the S3b table in `docs/decisions.md` (after notebook 05).
- Sync rule: let implementation sessions run `scripts/sync_inara.sh` themselves?
- Hermes: approve the install in the `basic-code-changes` session; commit `.claude/skills/` or keep local?
- `configs/prereg.yaml`: the "what changed since Oct 1" sentence (needed before S12).
