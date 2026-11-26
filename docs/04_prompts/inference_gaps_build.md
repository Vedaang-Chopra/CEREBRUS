# Build brief: inference gaps before S9 (resolution check, grid figures, headroom analysis)

From main-discussion, Nov 25–26, 2026. Approved by Vedaang: finish all inference baselines through S8 today; S9 tomorrow.
Queue after `s5b_profile_build.md` parts 2–4. Same working style: subagents write code (Sonnet for model/eval code,
Haiku for mechanical edits, Hermes for independent pieces); you review, run CPU checks, commit per part, report hashes.
Never run on Inara, never push, never touch test, ask before `rm -rf`. Run list: `docs/STATUS.md` "Today".

## 1. Resolution check (nb04, replace the skipped Step-5)
Cells, both benchmarks, dev 100 per benchmark, R = 3, no thinking, batch 1, 2 copies, own folder, resumable:
- `none` at res_px ∈ {512, 768, 1024} (all tokens);
- `divprune` at 768 and 1024 with k set per item so the kept count equals that item's 512 px full count (matched tokens);
- `divprune` at 1024 with k = 25% (≈ 512 px full count, as a fixed-k sanity row).
Columns as S5 (accuracy, GFLOPs incl. vision tower at that resolution, latency from a 10-item batch-1 pass,
kept tokens, kv_cache_bytes). Check memory at 1024 px on 2 A40s first (1 item, print peak GB). Figure F09b
(accuracy vs visual tokens per resolution, full vs DivPrune) via a plot function; decisions lines printed.
Answer to report: does 1024 px + DivPrune at 512-px token count beat 512 px full, per benchmark, with paired bootstrap CI?

## 2. Grid figures and noise rate (nb08)
Plot functions + nb08 figure steps: F07 (k × R accuracy heatmap per benchmark), F08 (15 cells accuracy vs GFLOPs with
the Pareto front). Print the nondeterminism rate from the `rerun_id` 0 vs 1 rows (answer flip %, accuracy diff).

## 3. Headroom analysis from the grid (inference-only, S14 inputs)
`adaloop/core/analysis/` (or the planned module path in DEC3 §3): best_fixed per cost cap (cell chosen on ctrl_train,
reported on dev), oracle per cap (per-item best over cells with mean FLOPs ≤ cap), noise oracle (same on `rerun_id=1`),
headroom = oracle − best_fixed − noise_oracle, with paired bootstrap CIs. Table T2 + figure F09. nb08 final section
(runs on the saved grid in seconds). Test stays sealed: refuse split "test".

## 4. Small leftovers
- nb03: confirm the TextVQA 20-item scoring section runs (flag), so S1's Done-when has TextVQA evidence.
- nb06 `RUN_FULL` must use the CLI path with shards over gpu_groups 0–7 (big run) and resume.
