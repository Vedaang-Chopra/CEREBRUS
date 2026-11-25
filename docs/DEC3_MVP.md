# AdaLoop-VLM — Dec 3 Minimal Version (agent build spec)

v1 · Nov 25, 2026 · owner: Vedaang · deadline: **Thu Dec 3, 11:59 PM ET** (CSE 8803 final: report + video + rerunnable code)
This file is the build spec for the course submission. It is a **subset** of `docs/IMPLEMENTATION.md`: same package layout, fewer steps. Where they conflict before Dec 3, **this file wins**.

> **Amendment (Nov 25):** the agent implements **everything**, CORE items included (Vedaang reviews instead of writing). **Tests = notebook execution:** `notebooks/mvp.ipynb` runs every stage in `smoke` mode with `assert` checks and loads full-run artifacts in `full` mode (see `docs/04_prompts/agent_build_mvp.md`). Where this conflicts with §3, §8 or §9, the amendment wins.

---

## 0. What we are building

**Research question.** On one weight-shared looped VLM (`thinkingmachines/TML-d24-varloop-research`), does a **per-pass controller that reads the loop state** and chooses **exit / loop again / look again** beat the best **fixed (visual-token budget k, loop count R)** setting at matched FLOPs? And does a **budget-tolerance LoRA** (post-training for reduced visual tokens) move that trade-off?

**Deliverables**
1. Python package `adaloop/` (subset of the full layout), with one CLI: `python scripts/mvp.py <stage> --config configs/mvp.yaml`.
2. **One results notebook** `notebooks/mvp_results.ipynb`. It only reads saved artifacts and shows every table and figure for the report.
3. A top-level `README_MVP.md` with setup + the exact commands that reproduce the headline number.

**Key simplification (why this fits in 8 days).** With greedy decoding, every controller path is assembled from **fixed runs**:
- **Exit after pass r** = the fixed run at (k0, R = r).
- **Look again at pass p** = restart with all tokens. The outcome is the fixed run at (k = 100%, R = r′); the cost is the partial low-token run through pass p + the full run.
- So the **whole action tree comes from the 15-cell grid**. No mid-loop insertion, no masking engine, no custom forward pass.
- **Assumption to verify (Stage S3b):** the pass-p state inside a longer run equals the pass-p state of an R = p run. If it doesn't hold, we say so and charge each R as its own run.

**In scope:**
- GQA + TextVQA
- 3 × 5 grid
- oracle + noise ceiling
- heuristic + learned controller
- 4 baselines
- budget-tolerance LoRA
- ablations, seeds, bootstrap CIs

**Out of scope (report as future work):**
- mid-loop insertion LoRA
- question-aware encoder (QAFE)
- joint GRPO
- LoopZip
- safety
- MMMU and ChartQA (ChartQA only if Stage S15 has slack)

---

## 1. Preregistration (write to `configs/prereg.yaml` before Stage S10 touches test)

```yaml
metric: accuracy on test items (GQA testdev-balanced + TextVQA val, macro-average of the two) at mean FLOPs = 0.5 × cost(k=100%, R=3)
baseline: best fixed (k, R) cell with mean FLOPs <= that cost, from our own grid, same items, same code
decision_rule: supported if (controller - best_fixed) >= 1.0 pt and the 95% paired-bootstrap CI (10k resamples over items) excludes 0
secondary: same comparison on the LoRA model (model_variant=lora)
headroom_check: oracle - best_fixed - noise_oracle >= 3 pts on >= 1 benchmark (else report the controller as exploratory)
changed_from_oct1_proposal: "<VEDAANG: one sentence on what changed and why>"
```

---

## 2. Facts the agent must respect (from notebooks 01/02)

| Fact | Action |
|---|---|
| Load with `AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=bf16, device_map=..., experts_implementation="grouped_mm")`. Load-time loop override is ignored. | Set R at runtime: `model.config.text_config.loop_num_passes = R` **and** `model.model.language_model.config.loop_num_passes = R`. |
| Visual tokens: `model.model.get_image_features(pixel_values, image_grid_thw).pooler_output`; forward accepts `mm_encoder_outputs={"image": ...}`. | Pruning = drop image placeholders + matching rows (notebook `prune_inputs`). |
| `prune_inputs` **packs** kept tokens (positions shift). | Stage S3 tests original positions; use them if the model accepts `position_ids`. |
| Thinking is on by default (`<|message_model|><|content_thinking|>…`). | Stage S2 turns it off; every run uses the same prompt. |
| k = 100% via `prune_inputs` is bit-exact. | Keep as a test. |
| Python 3.13 + transformers 5.x on the GPU host; `pyproject.toml` pins are stale. | Fix pins first (S0). |
| Code doesn't run on the laptop. | Every stage prints the exact GPU-host command; results go to `execution_results/mvp/`. |

---

## 3. Package layout (build only these files)

Follows `design-layered-module`: notebook → `api.py` → `orchestration/` → `core/` → `schemas/` + `utils/`. The notebook imports **only** `adaloop.api`.

```
utils/                      io.py (parquet read/write + schema check)  seeds.py  hashing.py (config_hash, git_hash)  timing.py (CUDA events)  stats.py (paired bootstrap, McNemar)
adaloop/
  api.py                    # public: load_model, load_items, run_cell, run_grid, build_tree, train_lora, train_controller, evaluate, load_results
  schemas/
    items.py                # Item, EvidenceBox
    runs.py                 # GridRow, FeatureRecord, TreeLeaf, PolicyRow + pyarrow schemas
  data/
    gqa.py  textvqa.py      # → list[Item]
    splits.py               # image-disjoint splits, frozen to configs/splits/mvp_*.json
  core/
    model/   load.py  loops.py (set_loops, call counter)  prompting.py (thinking off)  hooks.py (per-pass last-token state, visual attention mass)
    tokens/  selectors.py (random, text_mean, crisp_lite)  prune.py (prune_inputs with original positions)
    cost/    flops.py       # analytic FLOPs per (n_tokens, R, partial_pass) from the model config
    eval/    extract.py  scorers.py (GQA exact match, TextVQA VQA-acc)  oracle.py
    controller/
      features.py           # per-pass feature vectors
      tree.py               # assemble leaves from the grid
      objectives.py         # supervised + full-information expected reward   ← CORE (Vedaang)
      policy.py             # MLP + masks
      heuristic.py
      baselines.py          # best_fixed, early_exit_only, tokens_only, upfront_15way
    train/
      teacher.py            # cache full-token answers + top-k log-probs
      lora_budget.py        # PEFT config + compute_loss                       ← CORE (Vedaang)
  orchestration/
    grid.py  tree.py  train_lora.py  controller.py  analysis.py (tables + figures data)
configs/  mvp.yaml  prereg.yaml  prompts.yaml  splits/
scripts/  mvp.py            # thin CLI: stage name → adaloop.api call
tests/adaloop/  test_identity.py  test_loop_control.py  test_prune_positions.py  test_flops.py  test_tree.py  test_objectives.py  test_schemas.py
notebooks/  mvp_results.ipynb   (old notebooks → notebooks/archive/, unchanged)
README_MVP.md
```

**CORE (Vedaang)** items: the agent writes the typed signature, docstring, a test that defines the behaviour, and `raise NotImplementedError("TODO(vedaang)")`. Implement them only if Vedaang says "implement CORE <name>".

---

## 4. Config (`configs/mvp.yaml`)

```yaml
model_id: thinkingmachines/TML-d24-varloop-research
device_map: <balanced | custom>          # decide in S0, one policy
res_px: <RES_PX>                        # long side; TextVQA needs readable text — test 768 vs 1024 on dev in S2
prompt: configs/prompts.yaml#short_answer
max_new_tokens: 16
k_grid: [0.25, 0.5, 1.0]
R_grid: [1, 2, 3, 4, 5]
selector: <text_mean | crisp_lite>       # frozen after S5
benchmarks:
  gqa:     {source: <HF_ID verified in S1>, split: testdev_balanced}
  textvqa: {source: <HF_ID verified in S1>, split: validation}
splits: {test: 500, ctrl_train: 300, dev: 100, lora_train: rest}   # per benchmark, image-disjoint
lora: {r: 16, alpha: 32, targets: <regex from state dict>, layers: "6-17", lr: 1e-4, eff_batch: 16, steps: <LORA_STEPS>, k_train: [0.25, 0.5], identity_frac: 0.1, seed: 0}
controller: {hidden: 512, epochs: 50, lr: 1e-3, seeds: [0, 1, 2], lambdas: [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]}
n_rerun: 100                              # items re-run per cell for the noise ceiling
out_dir: execution_results/mvp
```

---

## 5. Data contracts (`adaloop/schemas/runs.py`)

**`grid.parquet`** — one row per (item, k, R, model_variant)
```
item_id, benchmark, split, question_type, model_variant (base|lora), selector, k, R,
pred, correct (float; VQA-acc for TextVQA), parse_ok,
n_vis, n_txt, n_out_tokens,
flops_vit, flops_prefill, flops_decode, flops_total,
t_vit_ms, t_prefill_ms, t_decode_ms, peak_mem_gb,
rerun_id (0 = main, 1 = noise re-run), gpu_type, git_hash, config_hash, seed
```
**`features/<variant>/<item_id>.pt`** — per (k, pass p) from the R=p fixed run
```
{ (k, p): {"h_last": [d] (fp16), "delta_norm": float, "vis_attn_mass": float,
           "entropy": float, "kl_prev": float, "top1_prob": float} }
```
`entropy`, `kl_prev` and `top1_prob` come from the first answer-token logits of the R = p run (= an early readout; its coda cost is charged in the tree when used).

**`tree.parquet`** — one row per (item, leaf, variant)
```
item_id, benchmark, split, model_variant, k0, path ("X@r" | "L@p,X@r'"), look_pass, exit_pass, correct, flops_total
```
**`policy_eval.parquet`** — one row per (method, λ, seed, item)
```
method, lam, seed, model_variant, item_id, benchmark, path, correct, flops_total
```

---

## 6. Cost model (`core/cost/flops.py`)

- Active params per module from the **config + state dict**. Dense params count fully; MoE expert params count × (experts_per_token / n_experts). Split into: embed/vision projector, prelude (layers 0–5), core (6–17, per pass), coda (18–23), lm_head.
- `flops_prefill(n_tokens, R) = 2 · n_tokens · (P_prelude + R · P_core + P_coda) + attention term (2 · n_layers_called · n_tokens² · d_attn)`.
- `flops_partial(n_tokens, p) = 2 · n_tokens · (P_prelude + p · P_core)` (a pass-p state with no coda).
- `flops_readout(n_tokens) = 2 · n_tokens · P_coda + 2 · P_lm_head`.
- `flops_decode(n_out, ctx, R) = n_out · 2 · (P_prelude + R · P_core + P_coda + P_lm_head)`.
- `flops_vit(image)`: independent of k (pruning happens after the projector). Report it separately, and say so in the report.
- `cost_ref(item) = flops_total(k=1.0, R=3)`.
- **Test:** the ratio R = 5 / R = 1 of measured A40 prefill time is within ±25% of the analytic ratio, on 20 items.

**Path costs in the tree**
- `X@r` with k0: `flops_total(k0, r)` (+ the readout cost at each earlier pass whose features used logits).
- `L@p, X@r'`: `flops_partial(k0 tokens, p)` + readouts through p + `flops_total(1.0, r')`.

---

## 7. Stages (do them in order; each has a Done-when)

Owner columns match `docs/TEAM_PLAN.md` roles: **V** = Vedaang, **A** = data/eval teammate, **B** = experiments teammate.

| Stage | Owner | What | Done when |
|---|---|---|---|
| **S0** | A | Env pins from the GPU host's `pip freeze`; package skeleton; move old notebooks to `notebooks/archive/`; port `load_model`, `set_loops`, `prepare_inputs`, `extract_visual_tokens`, `prune_inputs`, selectors into `core/` | `python scripts/mvp.py smoke` loads the model and answers 1 item |
| **S1** | A | Verify HF dataset IDs and splits; loaders → `Item`; image-disjoint splits frozen; scorers (GQA normalized exact match; TextVQA VQA-accuracy with lmms-eval normalization, provenance comment) | `tests/adaloop/test_schemas.py` passes; 20 items/benchmark score with plausible numbers |
| **S2** | A | **Thinking off.** Inspect chat template + special tokens; prefill an empty thinking block + answer start. Pick `res_px` (768 vs 1024 on 50 TextVQA dev items) | ≥ 95% parseable short answers on 20 GQA + 20 TextVQA. Decision in `docs/decisions.md`. **Deadline: Wed noon.** |
| **S3** | B | Original positions: read the modeling code for `position_ids` / `relative_distance`; `prune.py` keeps original positions if supported. Compare packed vs positioned at k = 50% on 50 dev items | Decision logged; `test_prune_positions.py` + `test_identity.py` (k = 100% bit-exact) pass |
| **S3b** | B | Loop control test (calls = 6 + 12R + 6). **Pass-state check:** h_last at pass p from hooks in an R = 5 run vs an R = p run, cosine + answer agreement over p, 30 items | Numbers in `decisions.md`. Drives the tree assumption. |
| **S4** | B | `hooks.py` + `features.py`; selectors `text_mean` (notebook v0) and `crisp_lite` (spaCy nouns → decoder input embeddings, top-k by cosine + 25% spatially uniform fill) | Features saved for 5 items × 15 cells |
| **S5** | B | Selector choice on dev at R = 3, k ∈ {25, 50}%: random vs text_mean vs crisp_lite. Freeze the winner in `mvp.yaml` | Small table in `decisions.md` |
| **S6** | A | `flops.py` + timing; **throughput:** 50 dev items × 15 cells → s/item → final item counts | `test_flops.py` passes; budget line in `decisions.md` |
| **S7** | A | **Base grid** on test + ctrl_train + dev, all 15 cells, features on. Shard by item over replicas; merge shards. Plus noise re-run: `n_rerun` items × 15 cells with `rerun_id=1` | `grid.parquet` complete; row counts checked |
| **S8** | V | Teacher cache on `lora_train`: full tokens, R = 3, greedy answer ids + top-32 log-probs per answer token | `teacher/*.pt` for all lora_train items; cached ≈ live on 20 items |
| **S9** | V | **Budget-tolerance LoRA** (`train/lora_budget.py`, CORE): PEFT LoRA on core-layer attention projections (layers 6–17; confirm module names from the state dict). Per sample: k ~ `k_train` (10% k = 1.0 identity guard), R ~ U{1..5}, frozen selector; loss = KL(teacher top-32 ‖ student) on teacher answer ids, teacher-forced. HF `Trainer` with custom `compute_loss` or a short loop; model-parallel over 2 GPUs; log to wandb | Smoke: 50 items × 100 steps, loss falls. Full run: dev accuracy at k = 25% improves vs base; k = 100% at R = 3 drops ≤ 0.5 pts |
| **S10** | A | **LoRA grid** (`model_variant=lora`): same items and cells as S7 (test + ctrl_train + dev) | `grid.parquet` has lora rows |
| **S11** | B | `tree.py`: leaves from the grid, per variant. k0 ∈ {25, 50}%, ≤ 1 look, look target = k = 100%. 5 exit leaves + 4 × 5 look leaves per k0. `test_tree.py`: costs and outcomes match the grid rows they come from | `tree.parquet` for base + lora |
| **S12** | V + B | **Controller.** `objectives.py` (CORE, V): (a) supervised to the argmax-Q action; (b) full-information expected reward by exact backward recursion (`docs/02_loop/adaloop_build_spec.md` §5.3). `policy.py` (B): MLP `[F] → 512 → 512 → 3` with masks (LOOP forbidden at r = 5; LOOK only on the low-k branch and only once). Train on ctrl_train, select λ/epochs on dev, 3 seeds. `heuristic.py` (B): thresholds on `top1_prob`, `kl_prev`, `vis_attn_mass`, tuned on dev | `policy_eval.parquet` on test for every λ × seed |
| **S13** | B | **Baselines** on the same items: best_fixed (per cost cap); early_exit_only (k = 100% branch, actions {EXIT, LOOP}, same MLP + features); tokens_only (3-way k at R = 3, features = mean visual token + mean question embedding, no decoder pass); upfront_15way (15-way chooser, same upfront features, expected-reward objective); oracle | All in `policy_eval.parquet` |
| **S14** | A | **Analysis** (`orchestration/analysis.py`): (1) iso-cost curves per method (accuracy vs mean FLOPs over λ); (2) headline = accuracy at 0.5 × cost_ref by linear interpolation along each curve; (3) paired bootstrap vs best_fixed and vs every baseline, Holm over the headline claims; (4) oracle per cost cap; noise oracle from re-runs; headroom; (5) seed mean ± std | `tables/*.csv` + `figures_data/*.parquet` |
| **S15** | B | **Ablations** (all re-scored from existing data unless noted): A1 controller features: all / − h_last / − readout features (entropy, kl_prev, top1_prob; also drops their cost) / − vis_attn_mass. A2 look on vs off. A3 selector random vs chosen (R = 3 only; extra cells k ∈ {25, 50}% × random). A4 LoRA off vs on (base vs lora trees). A5 k0 = 25% vs 50%. A6 tree assumption: if S3b failed, cost each R as its own run and recompute. **Error analysis:** per GQA `question_type` breakdown; 6 qualitative examples (exit-early correct, look rescued, look wasted, loop helped) | `tables/ablations.csv` |
| **S16** | A | `notebooks/mvp_results.ipynb` (§8) built from artifacts only | Runs top to bottom on a CPU-only machine |
| **S17** | A | Repro: `README_MVP.md` (setup, data download, one command per table, headline command), LoRA + controller weights on HF Hub (< 5 GB), versions + GPU hours + seeds in `execution_results/mvp/repro.json`; test the zip in a clean env | Headline reproduced from the zip within bootstrap noise |

**Dependency order:** S0 → S1, S2 (parallel) → S3, S3b, S4 → S5 → S6 → S7 → S11 → S12, S13 → S14 → S15 → S16 → S17.
LoRA track: S2 + S5 → S8 → S9 → S10 → (S11 lora) → S12/S13/S14 lora rows.

**Day plan:**

| Day | Stages |
|---|---|
| Wed 25 | S0, S1, S2, S3, S3b |
| Thu 26 | S4, S5, S6, S8 |
| Fri 27 | S7 launch, S9 smoke → full |
| Sat 28 | S9 finish, S10, S11 |
| Sun 29 | S12, S13, S14 |
| Mon 30 | S15, S16, **results freeze** |
| Tue Dec 1 | S17 + report |
| Wed Dec 2 | video |
| Thu Dec 3 | submit |

---

## 8. The results notebook (`notebooks/mvp_results.ipynb`)

Inspection only: imports `adaloop.api`, loads `execution_results/mvp/*`, no GPU. Sections mirror the report:

| § | Title | Shows |
|---|---|---|
| 0 | Setup | config, `config_hash`, `git_hash`, versions, item counts per split |
| 1 | Sanity | identity test result, loop-call counts, thinking-off parse rate, packed vs positioned, S3b pass-state check, nondeterminism rate |
| 2 | Cost model | FLOPs per (k, R) table; analytic vs measured time ratio; ViT share of cost |
| 3 | The k × R surface | accuracy heatmaps per benchmark (base and lora); accuracy vs FLOPs scatter of the 15 cells |
| 4 | Headroom | oracle per cost cap vs best fixed vs noise oracle; headroom table (prereg check) |
| 5 | **Headline** | iso-cost curves (controller, heuristic, 4 baselines, oracle); headline table at 0.5 × cost_ref, mean ± std over seeds, CI, p-value; **prereg verdict line** |
| 6 | Post-training | base vs LoRA surface; controller on base vs LoRA; does the controller gain survive the LoRA? |
| 7 | Policy behaviour | action histograms (exit pass, look rate) per benchmark and question type |
| 8 | Ablations | A1–A6 table |
| 9 | Error analysis | per-question-type table; 6 qualitative examples with image, question, path, answers |
| 10 | Reproducibility | GPU type and count, total GPU hours per stage, seeds, hyperparameters (copied into the report) |

**Figures** saved to `paper/figures/` (PDF + PNG). Tables to `paper/tables/` as CSV + LaTeX.

---

## 9. Rules for the agent

1. One stage at a time. Before coding a stage: a layer plan (files, functions, signatures), then wait for approval.
2. Never touch the **test** split before S12. `configs/prereg.yaml` is filled before S12 runs on test.
3. Every run row carries `git_hash`, `config_hash`, `seed`, `gpu_type`. All config values live in `configs/mvp.yaml`.
4. Code doesn't run locally: give the exact GPU-host command and the expected output; don't report a pass without pasted output.
5. Long jobs are sharded by item (`--shard i/n`), resumable (skip existing rows), and checkpointed.
6. CORE (Vedaang) items: signature + docstring + defining test + `TODO(vedaang)`.
7. After each stage: tick it in this file, add a line to `docs/decisions.md` if it decided something, update `docs/ai_context/CODEBASE_MAP.md`, commit with an `agent:` prefix, ask before pushing.
8. If a stage fails its Done-when, stop and report. Don't work around it silently. Fallbacks:
   - thinking can't be turned off → cap at 64 tokens and log `n_think` as cost;
   - positions unsupported → packed, and say so;
   - S3b fails → A6 costing;
   - LoRA hurts k = 100% → raise `identity_frac`, lower lr.

---

## 10. Agent prompt (paste this)

> You are building the Dec 3 minimal version of AdaLoop-VLM in this repo. Read `docs/DEC3_MVP.md` fully (it wins over other docs until Dec 3), then `AGENTS.md` and `docs/ai_context/CODEBASE_MAP.md`. Build the package in §3, the CLI `scripts/mvp.py`, and the single results notebook in §8. Work one stage at a time from §7, in order. For each stage, first give me a layer plan and wait for my OK. Leave CORE (Vedaang) items as signatures + docstrings + defining tests + `TODO(vedaang)`. Code runs only on the GPU host: give me exact commands and tell me what output to paste back. Start with S0.
