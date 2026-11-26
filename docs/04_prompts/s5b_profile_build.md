# Build brief: S5 winner freeze, S5b Pareto sweep, KV cache bytes, inference profiling

From main-discussion, Nov 25, 2026. Approved by Vedaang. Working style: you supervise; subagents write the code
(Sonnet for timing/profiling/model code, Haiku for mechanical edits; Hermes delegate for independent pieces); you
review diffs, run CPU checks, commit, report hashes. Never run on Inara, never push, ask before `rm -rf`.

## 1. Freeze the S5 winner
`compressor: divprune` in `configs/mvp.yaml` (Vedaang's decision; nb07 full dev: DivPrune best at k = 25% and 50%).
Before committing, verify and report:
- the finished S5 cell files (`execution_results/mvp/nb07/compare_dev`) keep their hashes, because each cell sets its own method;
- the S8 teacher cache hash does not depend on `compressor` (the teacher never compresses); exclude it if it does;
- S7 grid and S9 now read divprune.
Log the decision in `docs/decisions.md` with the S5 table (git 69c92a9, dev n = 100 per benchmark).

## 2. KV cache bytes (analytic, every result row)
New column `kv_cache_bytes` = bytes of the decoder KV cache at the end of prefill for that row: token count after
compression × (prelude + coda layers + core layers × R cache copies, as the cache is actually allocated; read it from
the cache class, `docs/notes/modeling_code.md`) × 2 (K, V) × kv heads × head dim × 2 bytes (BF16), with sliding
layers capped at their window. Verify once against the real cache object on the tiny model. Not hashed into configs.

## 3. S5b Pareto sweep (nb07 new section, own config key `pareto.ks`, own folder)
All 8 methods + `none` at k ∈ {0.10, 0.25, 0.50, 0.75, 1.0}, R = 3, dev 100 per benchmark, batch 1, 2 copies.
Reuse the finished 25 / 50 / 100% cells (same hashes). Plot accuracy vs GFLOPs, vs kept tokens, vs kv_cache_bytes,
one line per method, Pareto front marked (F06 becomes these panels).

## 4. Profiling: where the time goes (nb06 new step, 20 items per benchmark, batch 1, 1 copy, flag `RUN_PROFILE`)
For `none` and each method at k ∈ {0.10, 0.25, 0.50, 1.0}, R = 3, per item, CUDA-synchronized timers (CUDA events):
- vision tower; merger/projector; compression selection (incl. CPU parts: spaCy for crisp_lite, attention capture
  for visionzip); building the pruned inputs;
- decoder prefill split into prelude, each core pass, coda + lm_head (forward hooks with CUDA events);
- decode: time per generated token and token count; GPU-to-GPU hand-off of the 2-GPU split if measurable;
- wall time minus summed GPU time = Python/launch overhead;
- peak GPU memory (measured) and kv_cache_bytes.
Table per (method, k): median ms per stage, % of total, overhead added by compression, time saved vs `none`.

Plus a token-count scan answering "if we cut N tokens, is the decoder faster?": `none` vs `random` at kept visual
tokens {0, 16, 32, 64, 126}, R ∈ {1, 3, 5}, batch 1 and batch 8 (timing only): prefill ms and decode ms per token
vs token count. One `torch.profiler` trace of one item (k = 1 vs k = 0.25, R = 3) saved to execution_results, with a
printed top-15 kernel table (GPU busy vs idle). Plot: prefill / decode time vs token count per R and batch size.

## Rules
Timing never changes accuracy rows (separate folders, unhashed keys). One commit per part (1, 2, 3, 4).
Update CODEBASE_MAP, READMEs, notebooks/README, STATUS.

## 3b. One dynamic method: `cls_iqr` (TRIM's IQR rule on CLS attention)
The 8 methods are budget-based. Add one whose kept count is decided per image, as a step toward QAFE (the encoder
decides how much to keep). Score = the CLS attention we already compute for VisionZip (unit level, last ViT layer).
Keep the tokens whose score is an outlier by TRIM's rule: `score > Q3 + m·IQR`, ported from
FreedomIntelligence/TRIM:llava/model/multimodal_encoder/clip_encoder.py @ f2032236583b (IQR block, l.93-107, Apache-2.0),
unchanged except the score source; at least 1 token. No mean token of the rest (our decoder never saw one; note it).
`m` is the knob instead of k: sweep m ∈ {0.0, 0.5, 1.0, 1.5 (TRIM default), 3.0} → one Pareto point per m, with the
actual mean kept count per benchmark. Name in tables: `cls_iqr` ("TRIM IQR rule on CLS attention", not TRIM).
Rows log `n_vis` per item (varies), so FLOPs / KV / accuracy are per item as usual.

## Reuse rule (all parts)
Never rerun a finished row to fill a new column. `kv_cache_bytes` and any other analytic column are computed from the
saved row fields (token counts, R, config) when tables are built; resume treats rows without them as complete.
Only genuinely new cells (k = 10%, 75%, `cls_iqr` per m) run.
