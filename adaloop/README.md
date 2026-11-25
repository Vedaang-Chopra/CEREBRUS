# adaloop

Public entry points: `adaloop/api.py`. Notebooks and `scripts/mvp.py` import only that.

| Step | Functions | File |
|---|---|---|
| Load | `setup_notebook`, `load_config`, `load_model` (reuses the loaded model in this process), `unload_model` | `utils/env.py`, `utils/io.py`, `core/model/load.py`, `core/model/cache.py` |
| Inputs | `load_image`, `resize_long_side`, `prepare_inputs` | `utils/images.py`, `core/model/prompting.py` |
| Loops | `set_loops`, `count_layer_calls`, `expected_layer_calls` | `core/model/loops.py` |
| Pass states | `record_pass_states` (last core layer, last prompt position, per pass), `check_pass_state` | `core/model/hooks.py`, `orchestration/pass_state.py` |
| Inference | `generate` (with TTFT, time, peak memory), `generate_batch` (left-padded, accuracy runs only), `last_logits` | `core/model/inference.py` |
| Pruning | `extract_visual_tokens`, selectors, `prune_inputs` | `core/tokens/` |
| Data | `build_gqa_splits`, `load_gqa`, `build_textvqa_splits`, `load_textvqa` → `Item`; `make_splits`, `load_manifest` | `data/gqa.py`, `data/textvqa.py`, `data/splits.py` |
| Scoring | `extract_answer`, `normalize_answer`, `score`, `score_gqa`, `score_textvqa`, `summarize_rows` | `core/eval/` |
| Evaluation | `evaluate_item`, `run_evaluation` (resumable JSONL; `batch_size` > 1 batches generation), `read_results` | `orchestration/evaluate.py` |

Notebooks: `01_huggingfaceload` (load, loops), `02_pruning` (tokens), `03_data_and_scoring` (S1 + runner), `04_thinking_off` (S2), `05_positions_and_loops` (S3, S3b).
CLI: `python scripts/mvp.py {smoke,splits,eval}`.

Limits: one image per item; `batch_size` 1 is the reference (batched rows have no TTFT or peak memory, and `generate_s` is batch time / size; batched logits are not bit-exact with batch 1); pruning packs positions (the model does not support original ones, `docs/notes/modeling_code.md`); thinking stays on until `answer_prefill` is set in the config (S2).
`prepare_inputs(prefill=...)` appends the start of the model reply (the chat template adds none).
`set_loops` changes model state until the next call.
GQA loading, splits, extraction, scoring, timing and the runner are merged from Jiaqi-Gao/adaloop-vlm @ e870bff
(provenance comment in each file). TextVQA accuracy is ported from lmms-eval @ 9134731. Its hard-coded FLOPs estimate was not merged; S6 replaces it.
