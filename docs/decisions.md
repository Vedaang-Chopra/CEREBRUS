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
