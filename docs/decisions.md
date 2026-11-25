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
