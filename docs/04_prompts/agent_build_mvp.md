# Agent prompt — build the Dec 3 AdaLoop-VLM system + test notebook

You are the implementation agent for AdaLoop-VLM in this repo. Your job: **write all the code for the Dec 3 minimal version and one notebook that runs and checks every feature.** I run the notebook on the GPU host; you cannot run code here.

## Read first (in this order)
1. `docs/DEC3_MVP.md`: the build spec (stages S0–S17, layout §3, config §4, data contracts §5, cost model §6, analysis §7 S14). Read its **Amendment** at the top.
2. `AGENTS.md`: repo rules (branches, commits, provenance comments).
3. `notebooks/01_huggingfaceload.ipynb` and `notebooks/02_pruning.ipynb`: working code for loading, loop control, visual-token extraction and pruning. Reuse it; do not reinvent it.
4. `docs/02_loop/adaloop_build_spec.md` §5.3: the full-information controller objective.

Nothing in `docs/IMPLEMENTATION.md` has been implemented yet. Start from an empty package.

## What to build
- The package exactly as in `DEC3_MVP.md` §3 (`utils/`, `adaloop/` with `api.py`, `schemas/`, `data/`, `core/…`, `orchestration/`), plus `configs/mvp.yaml`, `configs/prompts.yaml`, `configs/prereg.yaml`, `scripts/mvp.py` and `README_MVP.md`.
- **Implement everything**, including the items marked CORE (`lora_budget.compute_loss`, `controller/objectives.py`). Write them clearly, with comments explaining each step of the math; I will review them.
- Use the libraries already chosen: transformers + PyTorch hooks (never reimplement the Inkling model), datasets, peft, HF `Trainer` or a short training loop, pandas/pyarrow, numpy, spaCy, matplotlib, wandb.
- Follow the `design-layered-module` skill: notebook → `adaloop.api` → `orchestration/` → `core/` → `schemas/`, `utils/`. No logic in the notebook. Configs hold every constant.

## Testing = running the notebook
There is no separate pytest suite. **`notebooks/mvp.ipynb` is the test.** Follow the `write-notebook` skill.

- **Parameters cell** (papermill-tagged): `MODE = "smoke"` or `"full"`, `CONFIG = "configs/mvp.yaml"`, `STAGES = [...]` (which sections to run), `N_SMOKE = 8`.
- **One section per stage S0–S15**, in order. Each section:
  1. a markdown cell: what the stage does and what "pass" means;
  2. a call to one `adaloop.api` function (with `MODE` and `N_SMOKE`);
  3. cells that **display intermediate outputs** (shapes, token counts, sample predictions, a heatmap, a table head);
  4. **`assert` checks** that encode the stage's Done-when from `DEC3_MVP.md` §7. Each check prints `PASS <stage>: <what>` or fails with a clear message.
- **`smoke` mode:** every stage really runs end to end on `N_SMOKE` items per benchmark, with tiny settings (LoRA 20 steps, controller 5 epochs, 1 seed, R and k grids as configured). Outputs go to `execution_results/mvp/smoke/`. Smoke mode must finish in **< 60 min on 2 A40s**.
- **`full` mode:** heavy stages (S7 grid, S9 LoRA, S10 LoRA grid) are launched from `scripts/mvp.py` (sharded, resumable). In this mode the notebook **loads** their artifacts from `execution_results/mvp/full/` and runs the same asserts on them, plus every analysis, table and figure from `DEC3_MVP.md` §8.
- **Final section "Results"**: headline table, iso-cost figure, preregistration verdict, ablations, error analysis, reproducibility info. Figures saved to `paper/figures/`, tables to `paper/tables/`.
- **Last cell:** a summary table of every check (stage, check, PASS/FAIL).
- Provide `scripts/run_notebook.sh <smoke|full>`, which runs papermill and writes the executed copy to `execution_results/mvp/<mode>/mvp_executed.ipynb`.

Checks that must exist (minimum):
- S0: model loads; R set at runtime; one answer generated.
- S1: items load; splits are image-disjoint; scorers give the expected score on hand-written examples.
- S2: ≥ 95% parseable short answers; no thinking tokens in output.
- S3: k = 100% is bit-exact vs the unmodified model; packed vs positioned results shown.
- S3b: per-layer call counts = 6 + 12R + 6 for R = 1..5; pass-state agreement table shown.
- S4–S5: features have the §5 shapes; selector table shown.
- S6: FLOPs increase with k and R; analytic vs measured R5/R1 ratio within ±25%.
- S7/S10: grid rows = items × 15 cells; schema valid.
- S8: cached teacher log-probs ≈ live (atol 1e-3).
- S9: LoRA loss decreases; LoRA off ⇒ base outputs unchanged.
- S11: every tree leaf's correctness and cost match the grid rows it was built from.
- S12–S13: λ = 0 ⇒ the controller reaches the best-accuracy leaf; large λ ⇒ it exits at pass 1; every baseline produces a row per test item.
- S14–S15: tables and figures exist; bootstrap CIs computed; prereg verdict printed.

## How to work
1. First reply: a **build plan**: file tree, public `api.py` functions with signatures, the notebook outline (sections + checks), and the order you will write things. Wait for my OK once.
2. Then build **everything** in one pass, stage by stage, without stopping for approval between stages. Commit after each stage with an `agent:` prefix on branch `v/mvp-build`. Don't push.
3. Get the details right:
   - Verify HF dataset IDs and module names (LoRA targets, special tokens) from the actual model and processor in code, and print them in the notebook rather than hard-coding guesses.
   - Every output row carries `git_hash`, `config_hash`, `seed`, `gpu_type`.
   - Long jobs are sharded (`--shard i/n`), resumable and checkpointed.
   - Never touch the test split in smoke mode.
4. When done: tell me the exact commands to (a) create the env, (b) run `scripts/run_notebook.sh smoke`, (c) launch the full-mode jobs, (d) run `scripts/run_notebook.sh full`. List any decision you had to make (thinking-off token, positions, `res_px`) and where it's configured.
5. When I paste a failing notebook output, fix the cause (not the assert) and tell me what changed.
