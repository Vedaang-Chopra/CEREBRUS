# Decisions log

One entry per gate or design decision. Newest at the bottom. Format:

```
## <date> · <step / gate ID> · <short title>
- Result: <numbers, with file path to the run>
- Decision: <what we do now>
- Changes: <which later steps change>
```

---

## 2026-11-24 · Planning · Docs reorganized; IMPLEMENTATION.md is the master reference
- Decision: one master doc (`docs/IMPLEMENTATION.md`); older plans kept as background in `02_loop/` and `03_encoder/`.
- Decision: post-training (PT-1 LoRA-insert, PT-2 QAFE, PT-3 joint RL) is part of the paper's method; the frozen-model controller is the first system and a baseline.
- Decision: all mid-loop operations are implemented as schedules via mask emulation on HF modules (no reimplementation of Inkling2Loop); measured latency comes from compacted runs.
- Decision: TRL is not used for the per-pass controller RL (actions are not tokens); custom GRPO loop. TRL only for the optional answer-level GRPO.
- Decision: QAFE is built after Gate G1 and Q1; its loop coupling is the encoder's main contribution.

## Gates (fill in)
- G0 backbone: _pending_
- G1b early readout: _pending_
- G1 look-again: _pending_
- G2 headroom: _pending_
- Q1 encoder viability: _pending_
- P1 LoRA-insert: _pending_
- Q2 encoder training signal: _pending_
- Q3 encoder vs baselines: _pending_
- P3 joint RL: _pending_
- G3 final: _pending_

## Nov 25, 2026 — test split sealed
- Decision: the test split stays sealed until the LoRA, selector, controller, λ and thresholds are frozen. S7 and S10 grids run on ctrl_train + dev only; test rows are added once, after that freeze, before S12–S14 report on test. Reason: DEC3 S7/S10 ran on test, which conflicts with §9 rule 2 and would let test results steer choices. `configs/full_baseline.yaml` already uses `run_splits: [dev, ctrl_train]` and `full_run` refuses test.

## Nov 25, 2026 — S4/S5 revised: compression baselines from official code
- Decision: S4 integrates official code of CDPruner, ReDiPrune, TRIM, ToMe (+ DivPrune) instead of crisp_lite; S5 compares them at R = 3, k ∈ {25, 50, 100}% on dev with accuracy, FLOPs, latency, compression time and token count. Details, commits, licenses, glue and blockers: `docs/notes/compression_baselines.md`.
- Blockers: faithful TRIM and CDPruner relevance need a CLIP text tower our model lacks → `TRIM-style`, `CDPruner-style`. ToMe merges 2×2 units (the merger needs 2×2 groups). CRISP and HQAP have no official code → excluded.

## Nov 25, 2026 · S2 · answer path: no thinking for the POC (git 1e456ae, nb04)
- Dev n=20 per benchmark, R=3, res_px 512, batch 1. P (no thinking, prefill "ANSWER:"): GQA 60.0, TextVQA 59.0, parse 100%, 1.5-2.6 s/item. T (thinking, effort 0.9, budget 256): GQA 60.0, TextVQA 63.0, parse 100%, 30-35% forced, 89-98 s/item.
- Decision (Vedaang): build and test the whole pipeline (S4 onward, incl. LoRA) with **no thinking** (`answer_prefill` = `no_think_prefill`). Final results add a **thinking** evaluation with a larger budget (forcing cut 30-35% of answers at 256) and the strict parser.
- Open for the final runs: think budget size; LoRA is trained without thinking, so base vs LoRA under thinking must be compared in the same format; thinking costs ~50x time per item. res_px stays 512 (768/1024 not compared).

## Nov 25, 2026 · S3 · token positions after pruning (git 3600fa8, nb05)
- Decision: pruning stays packed. Inkling2Loop builds token distances, masks and convolutions from the token index, so original positions are not supported (`docs/notes/modeling_code.md`). On real weights: k=100% packed pruning is bit-exact; original position_ids change nothing at k=50%.

## Nov 25, 2026 · S3b · loop control and pass state (git 3600fa8, nb05)
- Layer calls = 6 + 12R + 6 for R=1..5; logits differ between R=1 and R=3.
- Pass state, GQA dev n=20, no thinking: the last-core-layer state at pass p of an R=5 run is **bit-identical** to the final pass of an R=p run for every p=1..5 (cosine 1.0, rel L2 0, h_equal 20/20).
- Decision: the tree assumption holds (stopping at pass p costs exactly the R=p run). Answers still need the coda after the last pass, so each R is generated on its own; per-pass states/features can come from one R=5 run.
- First look at loops (GQA dev n=20, k=100%): accuracy R=1 50, R=2 60, R=3 60, R=4 65, R=5 65; answer agreement with R=5: 75 / 85 / 85 / 95 / 100%.

## Nov 25, 2026 — S8/S9 distillation design: under discussion (not yet in DEC3)
- Proposal (Vedaang): depth-matched distillation (teacher 100% tokens at the student's R) instead of a fixed R=3 teacher. Literature check (LoopOPD 2610.10623, LoopFormer 2602.11451, ELT 2604.09168, MELT 2605.07721, InternVL3.5 ViCO 2508.18265): looped papers use the deepest loop as teacher to make short loops match long ones; ViCO (forward KL, full-token teacher, same model) is the direct precedent for depth-matched budget tolerance; none compares same-depth vs deep teacher or combines token budget with loop depth (within these 5 papers).
- main-discussion recommendation, pending Vedaang's OK: depth-matched main, cross-depth with R=5 teacher as the only ablation (drop fixed R=3); S8 caches only the planned (item, R) training samples; S10 grid for base + both LoRAs; S15 adds A7 (distillation target).

## Nov 25, 2026 · nb06 · loops at full tokens, batch check, cost model (git bca6a36, nb06)
- Dev n=20 per benchmark, k=100%, no thinking, batch 8. GQA accuracy R=1..5: 55 / 60 / 55 / 65 / 60. TextVQA: 53 / 63 / 54.5 / 54 / 59. Parse 95-100%. At n=20 the R trend is within noise (1 item = 5 pts); the full dev / grid decides.
- Batch 8 = batch 1 predictions on 8 items: batched runs are allowed.
- Cost: measured R5/R1 prefill time ratio 3.41 vs analytic 3.05 (within ±25%): the FLOPs model holds. Throughput at R=3: batch 1 1.08 s/item, batch 8 0.37 s/item per copy (2x A40).

## Nov 25, 2026 · S8 · multi-depth teacher cache approved
- Decision (Vedaang): S8 caches the frozen full-token teacher at R = 1..5 for all 8,549 lora_train items (42,745 records), schema and checks in DEC3 S8 and `docs/04_prompts/s8_build.md`. GPUs 0–7 may be used for the S8 cache run (exception to the 0–3 rule, S8 only). Estimate from nb06 throughput (0.37 s/item at R=3, batch 8): ~2 h on 2 copies, ~1 h on 4.
- S9 (not started until S8 is validated and Vedaang approves): A = depth-matched (teacher R = student R), B = fixed R=3 teacher; 10% full-token samples depth-matched in both; forward KL top-32 + tail bucket; identical training settings. S10 compares base, A, B on the 15-cell grid.

## Nov 25, 2026 · S5 · batching off for compressed runs
- nb07 batch check (8 items × rediprune, visionzip at k=25%, R=3): batched generation changed 2 of 16 outputs vs batch 1 (gqa 20442346 rediprune yes→no; textvqa 35744 visionzip). At k=100% (nb06) batch 8 = batch 1 on 8/8.
- Decision: `eval_batch_size: 1` for S5 (the rows already done were batch 1, so all methods are compared under the same regime). Speed comes from 2 copies. Root cause (BF16 numerics with padding vs a bug) is checked before the S7 grid uses batching.

## Nov 25, 2026 · GPUs for big runs
- Decision (Vedaang): big runs (S5b Pareto sweep in nb07, S7 grid, S8 cache, S9 full training runs A and B, S10 grids, final test runs) may use all GPUs 0–7 (4 model copies, `gpu_groups: [[0,1],[2,3],[4,5],[6,7]]` in the run's config). Notebook checks and small runs stay on 0–3.

## Nov 25, 2026 · S9 · teacher depth is a config rule
- Teacher depth of a compressed student comes from `lora.teacher_R` in `configs/train_lora.yaml`: `same` (R_teacher = R_student), `fixed:N` (N loops) or `plus:N` (min(R + N, 5)). Full-token samples (k = 1) always use `same`.
- Variants: A = `same`; B = `fixed:5` (default, to be confirmed by Vedaang; `fixed:3` still works). A later extension `plus:2` is possible. Changing B means editing `lora_variants` only.
- Code: `adaloop/core/train/schedule.py`. Written and CPU-checked on a tiny model; not run on GPU. Training waits for S8 validation and Vedaang's approval.
- peft pinned at `==0.22.0` (`environment.yml`, `pyproject.toml`).

## Nov 25, 2026 · S5 · compression comparison (git 69c92a9, nb07, dev n = 100 per benchmark, R = 3, batch 1, no thinking)
- Decision (Vedaang): compressor = **divprune**. Best accuracy at k = 25% and 50%. The automatic rule suggested visionzip (tied within CI, 1% fewer FLOPs); overruled because DivPrune is +2.4 pts on the mean of both k and the cost gap is ~1%.
- Latency: median of 10 items per benchmark, batch 1, one copy. Table also in `execution_results/tables/S5_compression.csv` on Inara.
- `configs/full_baseline.yaml` pins `compressor: none`, so its config hash is unchanged. The S5 cell hashes and the S8 teacher hash do not depend on this key.

| method | k | all | GQA | TextVQA | GFLOPs | latency s | kept tokens |
|---|---|---|---|---|---|---|---|
| divprune | 0.25 | 47.7 | 58 | 37.4 | 2112.7 | 0.78 | 31 |
| visionzip | 0.25 | 43.0 | 54 | 32.0 | 2118.9 | 0.77 | 31 |
| rediprune | 0.25 | 41.2 | 46 | 36.4 | 2105.1 | 0.78 | 31 |
| crisp_lite | 0.25 | 40.9 | 51 | 30.8 | 2093.4 | 0.78 | 31 |
| uniform | 0.25 | 40.4 | 57 | 23.7 | 2100.6 | 0.78 | 31 |
| random | 0.25 | 36.4 | 49 | 23.7 | 2099.2 | 0.78 | 31 |
| lowres | 0.25 | 35.8 | 49 | 22.5 | 1819.9 | 0.79 | 29 |
| text_mean | 0.25 | 31.4 | 46 | 16.7 | 2075.8 | 0.77 | 31 |
| divprune | 0.5 | 52.2 | 57 | 47.3 | 2393.6 | 0.91 | 63 |
| visionzip | 0.5 | 52.1 | 54 | 50.2 | 2343.0 | 0.79 | 63 |
| rediprune | 0.5 | 50.8 | 56 | 45.6 | 2390.4 | 0.92 | 63 |
| random | 0.5 | 49.2 | 53 | 45.4 | 2373.4 | 0.92 | 63 |
| uniform | 0.5 | 48.8 | 56 | 41.6 | 2338.6 | 0.91 | 63 |
| lowres | 0.5 | 48.5 | 54 | 42.9 | 2143.1 | 0.79 | 60 |
| crisp_lite | 0.5 | 47.5 | 53 | 41.9 | 2365.0 | 0.80 | 63 |
| text_mean | 0.5 | 43.5 | 53 | 34.0 | 2362.6 | 0.78 | 63 |
| none | 1.0 | 57.4 | 59 | 55.8 | 2925.6 | 1.02 | 126 |

- Findings: question-conditioned scoring with decoder input embeddings hurts (text_mean below random); TextVQA loses far more than GQA at 25%; batch-1 latency barely follows FLOPs (overhead-bound; profiled in S5b part 4).

## Nov 25, 2026 · S8/S9 paused
- Vedaang paused S8 (teacher cache) and S9: the dataset design is under review (no top-32 log-prob distillation as planned; SFT, distillation data and reward modeling need more focus). Vedaang will share a report. S8/S9 code stays on main unused; nothing is run.

## Nov 25, 2026 · S14 inputs · headroom definitions (main-discussion; for Vedaang's review before any number is quoted)
- Oracle per cost cap: per-item cell choice with **mean** FLOPs ≤ cap (exact DP). The stricter per-item reading is reported as `oracle_eligible`, not used.
- Noise oracle: the part of the oracle score that does not survive a re-run (choose cells on one run, score on the other, both directions averaged), on all re-run items (ctrl_train + dev). Chosen over the build-spec §2.4 simulated flip-rate version because it uses measured data only. If greedy runs are deterministic it is ~0; item-sampling noise is covered by paired bootstrap CIs.
- Caps {0.25 … 1.5} × cost_ref (k=100%, R=3), headline 0.5. FLOPs include the vision tower; by the nb06 cost table, k=25–50% at R=1–2 should fall under 0.5 (to confirm on the grid).

## Nov 26, 2026 · S2 · resolution check (git f613eb4, nb04, dev n=100 per benchmark, R=3, batch 1, no thinking)
| cell | GQA | TextVQA | all | visual tokens (mean) | GFLOPs |
|---|---|---|---|---|---|
| full 512 | 59.0 | 55.8 | 57.4 | 126 | 2926 |
| full 768 | 57.0 | **68.8** | **62.9** | 233 | 4243 |
| full 1024 | 57.0 | 68.3 | 62.6 | 339 | 5593 |
| DivPrune 768 → 512-px count | 56.0 | 63.1 | 59.5 | 126 | 3253 |
| DivPrune 1024 → 512-px count | 56.0 | 49.7 | 52.8 | 126 | 3633 |
| DivPrune 1024, k=25% | 52.0 | 50.7 | 51.4 | 85 | 3258 |
- GQA images are small (long side ~616 px): 768 and 1024 give the same input (178 tokens); 15/200 items do not grow at all. The GQA differences are noise.
- TextVQA gains +13 pts from 512 → 768 full (126 → 288 tokens). DivPrune at the 512-px token count from a 768 image: +7.3 vs 512 full [CI −2.2, +16.9]; from 1024 (keeping 25%): −6.1 [−17.4, +5.0].
- Latency of the reused 512 rows comes from another session (1.02 s vs 0.80 s here): not comparable across sessions.
- Decision on res_px: pending Vedaang (see main-discussion recommendation).
- **Decision (Vedaang, Nov 26): res_px = 768** for every run from now on (S5b Pareto, S6 timing, S7 grid, S8, S9, S10). 512-px results stay as the S5 record and an appendix comparison. The S5 winner is re-checked at 768 in the Pareto sweep (k = 25/50%).

## Nov 26, 2026 · batching is off for every run
- nb07 equal-length batch check (768 px, batch up to 8, no padding inside a batch, 16 dev items per benchmark × rediprune/visionzip/text_mean at k=25% + none at k=100%): **13 of 128 batched outputs differ from batch 1**, also at k=100% with no pruning (e.g. "sitting on a bench" vs "sitting on bench", "6:00" vs "6:30", "Firefox" vs "firefox"). Padding is not the cause; batch shape changes the BF16 kernels and flips near-ties.
- Decision: batch 1 everywhere (`eval_batch_size: 1`, full_baseline `batch_size: 1`, teacher `batch_size: 1`). Speed comes from model copies (4 on GPUs 0–7 for big runs). Equal-length grouping code stays, unused. The earlier nb06 "batch 8 = batch 1 on 8/8" was a small sample.
