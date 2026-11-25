# adaloop

Public entry points: `adaloop/api.py`. Notebooks and `scripts/mvp.py` import only that.

| Step | Functions | File |
|---|---|---|
| GPUs | `select_gpu_group` (one of `gpu_groups`, or `"all"` of them, with `min_free_gib` free per GPU; run it before `adaloop.api` is imported, from `utils.gpus` (the one notebook import outside `adaloop.api`: the GPU choice must precede CUDA init); `setup_notebook(config)` only checks the result) | `utils/gpus.py` |
| Load | `setup_notebook`, `load_config`, `load_model` (reuses the loaded model in this process; copy 0 if several are loaded), `load_models(cfg, n_copies)` (copy j on the j-th gpu group; needs `"all"`), `unload_model` (frees all) | `utils/env.py`, `utils/io.py`, `core/model/load.py`, `core/model/cache.py` |
| Inputs | `load_image`, `resize_long_side`, `prepare_inputs` | `utils/images.py`, `core/model/prompting.py` |
| Loops | `set_loops`, `count_layer_calls`, `expected_layer_calls` | `core/model/loops.py` |
| Pass states | `record_pass_states` (last core layer, last prompt position, per pass), `check_pass_state` | `core/model/hooks.py`, `orchestration/pass_state.py` |
| Inference | `generate` (with TTFT, time, peak memory), `generate_batch` (left-padded, accuracy runs only), `last_logits` | `core/model/inference.py` |
| Pruning | `extract_visual_tokens`, selectors, `prune_inputs` | `core/tokens/` |
| Data | `build_gqa_splits`, `load_gqa`, `build_textvqa_splits`, `load_textvqa` → `Item`; `make_splits`, `load_manifest` | `data/gqa.py`, `data/textvqa.py`, `data/splits.py` |
| Scoring | `extract_answer`, `score`, `score_gqa`, `score_textvqa`, `summarize_rows` | `core/eval/` |
| Evaluation | `item_prompt` (question + the benchmark's `prompt_suffix`), `item_inputs`, `evaluate_item`, `run_evaluation` (resumable JSONL; `batch_size` > 1 batches generation; `vlms` runs one thread per copy), `read_results` | `orchestration/evaluate.py` |
| Model copies | `time_copies` (s/item, 1 copy vs all), `compare_copies` (same outputs per item) | `orchestration/copies.py` |
| Audit | `wrong_predictions` (saved rows re-scored with the current scorers) | `orchestration/audit.py` |
| Cost model (S6) | `cost_params` → `CostParams`, `flops_prefill`, `flops_partial`, `flops_readout`, `flops_decode`, `flops_vit`, `flops_total`, `cost_ref` (analytic); `time_prefill`, `time_throughput` (measured) | `core/cost/flops.py`, `orchestration/cost_timing.py` |
| Figures and tables | `set_paper_style`, `METHOD_COLORS` (fixed Okabe-Ito colour per method), `VARIANT_LINESTYLES` (base solid, lora dashed), `save_figure` (PDF + PNG), `save_table` (CSV + LaTeX); drafts to `execution_results/`, finals (`final=True`) to `paper/` | `core/plots/style.py`, `core/plots/colors.py`, `core/plots/save.py`, `utils/paths.py` |
| Paper figures | `plot_thinking_off` (F02), `plot_pass_state` (F03), `plot_cost_model` (F04), `plot_accuracy_vs_r` (F05): DataFrame in, figure out | `core/plots/` |
| Full baseline | `shard_items`, `full_run_config`, `result_path`, `run_full` (one shard, all benchmark x split x R; `vlms` for copies in one process), `summarize_full` (all shards, de-duplicated) | `orchestration/full_run.py` |

Notebooks: `01_huggingfaceload` (load, loops), `02_pruning` (tokens), `03_data_and_scoring` (S1 + runner), `04_thinking_off` (S2), `05_positions_and_loops` (S3, S3b), `06_full_baseline` (batch check, full run shards, summary, S6 cost model).
CLI: `python scripts/mvp.py {smoke,splits,eval,full}`. `full --shard i/n` runs shard i on `gpu_groups[i]`; `smoke` and `eval` pick a free group (physical GPUs 0-3 only on Inara; `gpu_groups` and `min_free_gib` are not hashed).

Two copies in one process (4 GPUs): `select_gpu_group(CONFIG, "all")` before importing `adaloop.api`, `vlms = api.load_models(cfg, 2)`, then `run_evaluation(None, ..., vlms=vlms)` or `run_full(None, cfg, ROOT, vlms=vlms)`; single-item steps use `vlms[0]`. Chunks go round-robin to the copies, so chunk contents, config hash and resume match the one-copy run; rows carry `copy` and `n_copies` (not hashed). Per-item times and TTFT of `n_copies` > 1 rows include contention between the threads: take latency numbers from one-copy runs. After an interrupt, wait for the cell to stop before rerunning (a worker finishes its current chunk).

Limits: one image per item; `batch_size` 1 is the reference (batched rows have no TTFT or peak memory, and `generate_s` is batch time / size; batched logits are not bit-exact with batch 1); pruning packs positions (the model does not support original ones, `docs/notes/modeling_code.md`); thinking stays on until `answer_prefill` is set in the config (S2).
`prepare_inputs(prefill=...)` appends the start of the model reply (the chat template adds none).
`set_loops` changes model state until the next call.
GQA loading, splits, extraction, scoring, timing and the runner are merged from Jiaqi-Gao/adaloop-vlm @ e870bff
(provenance comment in each file). TextVQA accuracy is ported from lmms-eval @ 9134731. Its hard-coded FLOPs estimate was not merged; `core/cost/flops.py` replaces it.
FLOPs are analytic (2 per multiply-add) for dense bf16 weights, count causal attention pairs (eager attention executes more),
and include the message gates of passes after the first. `flops_total` includes the vision tower.
