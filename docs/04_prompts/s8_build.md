# Build brief: S8 multi-depth teacher cache

From main-discussion, Nov 25, 2026. Approved by Vedaang. Only S8: no S9 training code changes beyond what S8 needs.
Read first: `AGENTS.md`, `docs/DEC3_MVP.md` (S8 row), `docs/decisions.md` (last entries), `docs/notes/modeling_code.md`.

## Start from the parked draft
Branch `wip/post-training` @ `6e2ddf8` has `core/train/teacher.py`, `orchestration/teacher_cache.py`, `schemas/train.py`,
`configs/train_lora.yaml`, `notebooks/10_post_training.ipynb` (unreviewed). Bring over only what S8 needs, onto main, and adapt:
- Teacher at **R = 1..5** (`teacher.R_list: [1, 2, 3, 4, 5]`), not R = 3 only.
- Prefill = `no_think_prefill` from `mvp.yaml` (the draft's prefill lacks "ANSWER:"; extraction is strict).
- Leave LoRA / training code on the branch.

## Record: one per (benchmark, item_id, R)
`benchmark, item_id, image_id, question, answers, R, pred, answer_ids (greedy answer tokens incl. the end token),
n_answer_tokens, parse_ok, hit_limit, correct (official scorer), top_ids (n, 32) int32, top_logp (n, 32),
tail_logp (n,) = log of the probability mass outside the top 32, config_hash, git_hash, model revision, prompt hash, code_version`.
Log-probs: log-softmax over the full vocabulary, then take the top 32. Storage: tensors in `.pt` per shard and R plus a
JSONL index, under `execution_results/teacher/<config_hash>/`.

## Run
- Frozen model, eval mode, `torch.inference_mode`. Greedy. S2 setup (no thinking, `ANSWER:` prefill, res_px 512).
- Vision tower once per item; reuse its output for R = 1..5. Each R is generated on its own (no readout from inside a
  longer run).
- Batches contain one R only (loop count is model-wide). Batched generation is allowed (nb06: batch 8 = batch 1).
- Shards over all `gpu_groups` given in the S8 config. Vedaang allowed GPUs 0–7 for S8 only: put
  `gpu_groups: [[0,1],[2,3],[4,5],[6,7]]` in `configs/train_lora.yaml` (not in mvp.yaml).
- Resumable: existing (benchmark, item_id, R) records are skipped; a cut file is recovered.
- Logged: teacher accuracy per R, cross-R answer agreement, items/s, bytes per record.

## Notebook `09_teacher_cache.ipynb` (Vedaang runs it)
1. One item: teacher answer at R = 1..5, top-5 tokens with probabilities, tail mass (pasted before anything else).
2. Cache = live: for 20 items, a fresh teacher-forced forward on prompt + cached answer ids reproduces `top_logp`
   within BF16 tolerance (state the tolerance), at every R.
3. `exp(top_logp).sum + exp(tail_logp) ≈ 1`.
4. Resume: re-running a finished shard adds no records.
5. 20 items per benchmark summary: accuracy per R, agreement between R pairs, parse rate, throughput, storage
   estimate for the full run.
6. Full run behind `RUN_FULL = False`, plus a terminal command (`scripts/mvp.py teacher --shard i/n`) for long runs.

## Rules
One commit (or two: code, then notebook), report hashes to main-discussion. Sonnet subagents for the teacher
generation / top-k code, Haiku for mechanical edits; Hermes allowed. CPU check on the tiny random model. Never run on
Inara, never push, never touch the test split, ask before any `rm -rf`. Update `CODEBASE_MAP.md`,
`adaloop/README.md`, `notebooks/README.md`, `STATUS.md` S8 row.
