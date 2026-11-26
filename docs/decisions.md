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
