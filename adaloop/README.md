# adaloop

Public entry points: `adaloop/api.py`. Notebooks and `scripts/mvp.py` import only that.

| Step | Functions | File |
|---|---|---|
| GPUs | `select_gpu_group` (one of `gpu_groups`, or `"all"` of them, with `min_free_gib` free per GPU; run it before `adaloop.api` is imported, from `utils.gpus` (the one notebook import outside `adaloop.api`: the GPU choice must precede CUDA init); `setup_notebook(config)` only checks the result) | `utils/gpus.py` |
| Load | `setup_notebook`, `load_config`, `load_model` (reuses the loaded model in this process; copy 0 if several are loaded), `load_models(cfg, n_copies)` (copy j on the j-th gpu group; needs `"all"`), `unload_model` (frees all) | `utils/env.py`, `utils/io.py`, `core/model/load.py`, `core/model/cache.py` |
| Inputs | `load_image`, `resize_long_side`, `prepare_inputs` (`system`, `reasoning_effort`, `prefill`); benchmark items go through `item_inputs` | `utils/images.py`, `core/model/prompting.py` |
| Loops | `set_loops`, `count_layer_calls`, `expected_layer_calls` | `core/model/loops.py` |
| Pass states | `record_pass_states` (last core layer, last prompt position, per pass), `check_pass_state` | `core/model/hooks.py`, `orchestration/pass_state.py` |
| Comparison (S5) | `comparison_cells`, `run_comparison` (method x k from `cfg["compare"]`, resumable), `comparison_table`, `pick_winner` (paired bootstrap, `paired_bootstrap_ci`), `row_flops`, `plot_compression` (F06) | `orchestration/compare.py`, `core/eval/bootstrap.py`, `core/plots/compression.py` |
| Teacher cache (S8) | `teacher_items` (lora_train, per shard), `build_teacher_cache` (R in `teacher.R_list`, vision once per item, log-probs from the generation's logits; resumable), `teacher_records`, `read_index`, `load_records`, `check_teacher_live`, `teacher_summary`, `teacher_dir`, `teacher_total_records`, `answer_log_probs`, `split_top`; config `configs/train_lora.yaml` (GPUs 0-7, S8 only); CLI `scripts/mvp.py teacher --shard i/n` | `orchestration/teacher_cache.py`, `core/train/` |
| Grid (S7) | `grid_items`, `rerun_items`, `run_grid` (resumable per item; one JSONL per shard x copy; features under `features/<variant>/<hash>/<benchmark>_<item_id>.pt`), `merge_grid` → `grid.parquet`; CLI `scripts/mvp.py grid --shard i/n` | `orchestration/grid.py` |
| Features (S4b) | `run_item_cells(vlm, item, cfg)` → (rows, {(k, p): features}) over `k_grid` x `R_grid`; `pass_features`, `record_attention_mass`, `record_first_logits`, `flops_features`, `save_features`, `load_features`. Needs `answer_prefill` | `orchestration/features.py`, `core/controller/features.py`, `core/model/hooks.py` |
| Inference | `generate` (with TTFT, time, peak memory), `generate_batch` (left-padded, accuracy runs only), `last_logits`; `generate_answers` (a run under a config: thinking budget with a forced answer, no cache reuse) | `core/model/inference.py`, `core/model/budget.py`, `core/model/stopping.py` |
| Pruning | `extract_visual_tokens`, selectors, `n_keep_for`, `prune_inputs` | `core/tokens/` |
| Compression (S4) | `compress(vlm, inputs, item, k, cfg)` → (model inputs, info), method = `cfg["compressor"]` (`METHODS`); `evaluate_item` calls it with `cfg["k"]` (default 1.0). Latency = `vit_s` + `compression_s` + `generate_s` (`vit_s` is None for `none`/`lowres`, where the vision tower runs inside `generate`). Batches only at k = 1 | `orchestration/compress.py`, `core/tokens/methods/`, `core/tokens/vision.py`, `core/cost/compression.py` |
| Data | `build_gqa_splits`, `load_gqa`, `build_textvqa_splits`, `load_textvqa` → `Item`; `make_splits`, `load_manifest` | `data/gqa.py`, `data/textvqa.py`, `data/splits.py` |
| Scoring | `extract_answer` (`answer_format`), `read_answer` (a generation under a config), `score`, `score_gqa` (official, primary), `score_textvqa`, `score_lmms`, `score_gqa_lmms`, `summarize_rows` (`docs/notes/evaluation.md`) | `core/eval/` |
| Evaluation | `item_prompt` (question + `prompts.question_suffix`), `item_inputs` (system prompt, effort, prefill), `evaluate_item`, `run_evaluation` (resumable JSONL; `batch_size` > 1 batches generation; `vlms` runs one thread per copy), `read_results` | `orchestration/inputs.py`, `orchestration/evaluate.py` |
| Model copies | `time_copies` (s/item, 1 copy vs all), `compare_copies` (same outputs per item) | `orchestration/copies.py` |
| Audit | `wrong_predictions` (saved rows re-scored with the current scorers) | `orchestration/audit.py` |
| Cost model (S6) | `cost_params` → `CostParams`, `flops_prefill`, `flops_partial`, `flops_readout`, `flops_decode`, `flops_vit`, `flops_total`, `cost_ref` (analytic); `time_prefill`, `time_throughput` (measured); `row_flops` (a result row incl. compression), `cost_grid` (FLOPs of every k x R cell), `time_grid` (s/item over all cells, as the grid runs), `estimate_grid_hours` | `core/cost/flops.py`, `orchestration/cost_timing.py` |
| Figures and tables | `set_paper_style`, `METHOD_COLORS` (fixed Okabe-Ito colour per method), `VARIANT_LINESTYLES` (base solid, lora dashed), `save_figure` (PDF + PNG), `save_table` (CSV + LaTeX); drafts to `execution_results/`, finals (`final=True`) to `paper/` | `core/plots/style.py`, `core/plots/colors.py`, `core/plots/save.py`, `utils/paths.py` |
| Paper figures | `plot_thinking_off` (F02), `plot_pass_state` (F03), `plot_cost_model` (F04), `plot_accuracy_vs_r` (F05): DataFrame in, figure out | `core/plots/` |
| Full baseline | `shard_items`, `full_run_config`, `result_path`, `run_full` (one shard, all benchmark x split x R; `vlms` for copies in one process), `estimate_full_run` (16-item timing pass, time left), `summarize_full` (all shards, de-duplicated) | `orchestration/full_run.py` |

Notebooks: `01_huggingfaceload` (load, loops), `02_pruning` (tokens), `03_data_and_scoring` (S1 + runner), `04_thinking_off` (S2), `05_positions_and_loops` (S3, S3b), `06_full_baseline` (batch check, full run shards, summary, S6 cost model).
CLI: `python scripts/mvp.py {smoke,splits,eval,full,grid,teacher}`. `full --shard i/n` runs shard i on `gpu_groups[i]`; `smoke` and `eval` pick a free group (physical GPUs 0-3 only on Inara; `gpu_groups` and `min_free_gib` are not hashed).

Two copies in one process (4 GPUs): `select_gpu_group(CONFIG, "all")` before importing `adaloop.api`, `vlms = api.load_models(cfg, 2)`, then `run_evaluation(None, ..., vlms=vlms)` or `run_full(None, cfg, ROOT, vlms=vlms)`; single-item steps use `vlms[0]`. Chunks go round-robin to the copies, so chunk contents, config hash and resume match the one-copy run; rows carry `copy` and `n_copies` (not hashed). Per-item times and TTFT of `n_copies` > 1 rows include contention between the threads: take latency numbers from one-copy runs. After an interrupt, wait for the cell to stop before rerunning (a worker finishes its current chunk).

Limits: one image per item; `batch_size` 1 is the reference (batched rows have no TTFT or peak memory, and `generate_s` is batch time / size; batched logits are not bit-exact with batch 1); pruning packs positions (the model does not support original ones, `docs/notes/modeling_code.md`); the model thinks by default (`answer_prefill: null`).
Prompt text is `configs/prompts.yaml`; only the last `ANSWER:` line outside thinking is graded. `think_budget` caps thinking; `forced` rows had the answer opened for them. Rows log `format_ok`, `answer_len_ok`, `n_think_tokens`, `n_answer_tokens` (message contents, markers excluded); `n_out_tokens` counts every generated token.
`prepare_inputs(prefill=...)` appends the start of the model reply (the chat template adds none).
`set_loops` changes model state until the next call.
GQA loading, splits, extraction, scoring, timing and the runner are merged from Jiaqi-Gao/adaloop-vlm @ e870bff
(provenance comment in each file). TextVQA accuracy is ported from lmms-eval @ 9134731. Its hard-coded FLOPs estimate was not merged; `core/cost/flops.py` replaces it.
FLOPs are analytic (2 per multiply-add) for dense bf16 weights, count causal attention pairs (eager attention executes more),
and include the message gates of passes after the first. `flops_total` includes the vision tower.
