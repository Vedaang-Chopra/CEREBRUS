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
