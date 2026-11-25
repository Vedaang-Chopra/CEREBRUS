# Build brief: S4–S7 (compression baselines, comparison, cost, grid)

From main-discussion, Nov 25, 2026. Approved by Vedaang. Read first: `AGENTS.md`, `docs/STATUS.md`, `docs/DEC3_MVP.md`
(S4–S7 rows, §9 rules), `docs/notes/compression_baselines.md` (vision tower facts, upstream commits, glue notes),
`docs/notes/modeling_code.md`, `docs/ai_context/CODEBASE_MAP.md`.

## Fixed decisions
- **No thinking** for every POC run: `answer_prefill` = `no_think_prefill` (already in `configs/mvp.yaml`).
- Test split sealed. Runs use dev (S5) and ctrl_train + dev (S7). GPUs 0–3 only, 2 copies via `n_copies`.
- spaCy 3.8.18 + `en_core_web_sm` are installed in `learning_env` on Inara. Add `spacy` to `environment.yml` / `pyproject.toml`.

## Methods (final set)
| Name (`compressor`) | Kind | Source |
|---|---|---|
| `none` | bypass, k = 100% | — |
| `random`, `uniform`, `text_mean` | controls | existing `selectors.py` |
| `lowres` | control: smaller `res_px` so the token count ≈ k × full count (report the actual count) | config only |
| `crisp_lite` | question nouns (spaCy) → decoder input embeddings, top-k by cosine to projected tokens + 25% spatially uniform fill | reimplemented from the CRISP paper (#052, `Papers/.../Shortlisted_Literature`), no official code; say so in the provenance header |
| `rediprune` | relevance + diversity, greedy | official: UA-CVML/ReDiPrune @ `ba04aa4427f7`, `_select_text_cond` + `extract_question_embedding` |
| `divprune` | diversity only (max-min) | official: vbdi/divprune @ `799e2d950aa0` |
| `visionzip` | dominant tokens by CLS attention in the vision tower + contextual merge | official: dvlab-research/VisionZip @ `8f86b55c6f00` (has a Qwen2.5-VL path). Needs attention weights from the last ViT layer(s): eager attention or a q/k hook on those layers only |
Optional, only if it takes < 1 h: `cdpruner_style` (official DPP core @ `9541616c40fc`, relevance = cosine to mean question embedding). Not now: ToMe, TRIM.

Integration rules: clone each upstream repo read-only to `old_code/third_party/<name>/` at the commit above. Copy the
core algorithm into `adaloop/core/tokens/methods/<name>.py` unchanged under a header
`# Ported from <repo>:<path> @ <commit> (<license>)`; glue below it. Cleaning up upstream code is fine (drop env-var
plumbing, unused branches) as long as the algorithm is the same; list every change in the header. Any deviation from the
paper's method (e.g. a different query signal, scoring on 2×2 units) is written in `docs/notes/compression_baselines.md`.
Pre-merger scoring works on 2×2 units (4 consecutive patch states), because the merger takes 2×2 groups.

One interface for all: `compress(vlm, inputs, item, k, cfg) -> (model inputs for generation, info)`; `info` has kept
visual-token count, compression seconds, compression FLOPs (analytic where possible). `cfg["compressor"]` picks the
method. The per-pass loop-state extraction (`record_pass_states`, features) runs after compression, unchanged.

## Stages: one commit each, report each hash to main-discussion before moving on
1. **S4a methods.** The methods above + `compress` + api exports. Notebook `07_compression.ipynb`: per method, a
   1-item-per-benchmark check that prints kept count, raw_output, pred, gold (Vedaang pastes it before any batch);
   asserts: exact kept count at k = 25/50% (lowres: within one 2×2 row), `none` and every method at k = 100% give
   the same generated ids as no compression.
2. **S4b features.** `hooks.py` + `features.py` per DEC3 S4 (per-pass entropy, kl_prev, top1_prob, h_last, vis_attn_mass),
   counted in FLOPs. Section in nb07: features for 5 items × 15 cells saved.
3. **S5 comparison.** Runner over method × k ∈ {25, 50, 100}% at R = 3 on dev: accuracy, FLOPs (ViT + decoder +
   compression), latency, compression time, actual token count. nb07: `sample_items` first (flag), then full dev (flag).
   F06 plot (accuracy vs FLOPs per method) via `adaloop/core/plots/`. Writes the table for `decisions.md`.
4. **S6 cost.** Extend `flops.py` / `cost_timing.py` and nb06 Step 4: ViT FLOPs (full for all post-ViT methods; lowres
   smaller), compression overhead, analytic vs measured ratio check (±25%), throughput → item counts for S7.
5. **S7 grid.** Extend `full_run.py` (today it refuses `k_grid != [1.0]`): cells = `compressor` (S5 winner, a config
   value) × k ∈ {25, 50, 100}% × R ∈ 1..5, features on, shards over copies, ctrl_train + dev, noise re-run
   (`rerun_id=1`). Use one R = 5 run for all R only if nb05 confirms the pass-state check; keep a config switch.
   Notebook `08_grid.ipynb`: 1-item check, 20-item sample, full run behind `RUN_FULL = False`.

Each stage updates `notebooks/README.md`, `docs/ai_context/CODEBASE_MAP.md`, `adaloop/README.md`, and
`docs/notes/compression_baselines.md` where it applies. `code_version` bumps when outputs change.

## Working rules
- Sonnet subagents for model-critical code (VisionZip attention, ReDiPrune/DivPrune ports, grid runner, FLOPs);
  Haiku for mechanical edits (exports, docs, notebook cells). Hermes may be used (Vedaang approved).
- Never run on Inara, never push. CPU checks with a tiny random model where possible.
- Notebook style per AGENTS.md; config values in `configs/`, not in cells.
