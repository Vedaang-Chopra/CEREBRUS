# Build brief: S9 two budget-tolerance LoRAs (code only, no training run)

From main-discussion, Nov 25, 2026. Vedaang approved writing the code. **Training starts only after S8 is validated and
Vedaang approves.** Spec: `docs/DEC3_MVP.md` S9 row and `docs/decisions.md` (S8/S9 entries). Read `AGENTS.md`,
`docs/04_prompts/s8_build.md`, `adaloop/orchestration/teacher_cache.py` (the cache this reads).

## Working style (Vedaang's rule)
The implementation agent **supervises**: every code change is written by a subagent (Sonnet for model-critical code:
loss, recurrence, gradients, PEFT; Haiku for mechanical edits: exports, docs, notebook cells; Hermes allowed). The main
agent keeps its context free, reviews each subagent's diff, runs the CPU checks itself, then reports to main-discussion.
Independent pieces may run in parallel through the Hermes delegate skill (`~/agent-governance/skills/optional/hermes-delegate/SKILL.md`,
profile `cerberus`); the main agent still reviews every diff before committing.

## Start from the parked draft
`wip/post-training` @ `6e2ddf8`: `core/train/lora_budget.py` (PEFT attach, `kl_topk`, `compute_loss`, sample schedule),
`orchestration/train_lora.py`, `schemas/train.py`, `configs/train_lora.yaml`, `notebooks/10_post_training.ipynb`.
Bring over what S9 needs onto main and adapt:

1. **Two variants** `lora_depth_matched` (A) and `lora_cross_depth` (B), from config `lora_variants` (A: `same`, B: `fixed:5`); the teacher rule is `lora.teacher_R` (`same` | `fixed:N` | `plus:N`).
   Each starts from the base model; separate adapter dirs and run names.
2. **Teacher lookup from the S8 cache** by (benchmark, item_id, R_teacher): A uses R_teacher = R_student;
   B uses the teacher rule of `lora_variants.B` (default `fixed:5`) for compressed samples. Full-token samples (10%, k = 1.0) use R_teacher = R_student in both.
3. **Sample schedule** depends only on seed and index: R_student ~ U{1..5}, k ~ {0.25, 0.5} (90%) or 1.0 (10%).
   Identical in A and B (same seed), so the only difference is the teacher depth.
4. **Student inputs** use `compress(vlm, inputs, item, k, cfg)` with `cfg["compressor"]` (the S5 winner; text_mean until
   frozen). lowres must work too (it changes pixel inputs, not token selection).
5. **Loss**: forward KL(teacher top-32 + tail bucket ‖ student) at every answer position incl. the end token,
   teacher-forced on the cached teacher answer ids of that R_teacher; prompt = S2 setup (`no_think_prefill`).
   No gold CE, GRPO or cost term.
6. **Recurrence**: loop count set per microbatch (one R per microbatch); gradients through all executed passes into the
   shared LoRA (no detach); activation checkpointing; BF16; 2-GPU model parallel.
7. **Config** `configs/train_lora.yaml`: LoRA r 16, α 32, dropout 0, target q/k/v/o `.linear` of layers 6–17
   (48 modules, checked), lr 1e-4, eff. batch 16, steps, seeds, `lora_train` split; smoke settings
   (50 items, 100 steps per variant).

## Checks (notebook `10_post_training.ipynb`, Vedaang runs it; CPU tiny model first)
- PEFT targets = 48 modules; only LoRA params require grad; base params unchanged after a step.
- Gradient reaches every LoRA module at R = 1, 3, 5 (non-zero grad norm per core layer).
- Loop count during the student forward = sampled R (layer-call count 6 + 12R + 6).
- Teacher lookup: A pairs R_t = R_s; B pairs R_t = 5 (`fixed:5`) for k < 1 and R_t = R_s for k = 1.
- Loss is 0 when the student = the teacher (k = 1, LoRA zero-initialized, same R) up to BF16 noise.
- Smoke (behind a flag): 100 steps per variant, loss falls; measured step time → full-run hours estimate.
- Eval probe (behind a flag): dev 20 items at k ∈ {0.25, 1.0}, R ∈ {1, 3, 5}, base vs adapter.

## Rules
Commit per logical piece, report hashes. Never run on Inara, never push, never touch the test split, ask before any
`rm -rf`. `peft` goes into `environment.yml` / `pyproject.toml` (install on Inara is main-discussion's job, after
Vedaang's OK). Update `CODEBASE_MAP.md`, `adaloop/README.md`, `notebooks/README.md`, `STATUS.md` S9 row.
