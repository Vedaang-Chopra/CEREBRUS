# adaloop

Public entry points: `adaloop/api.py`. Notebooks and `scripts/mvp.py` import only that.

| Step | Functions | File |
|---|---|---|
| Load | `load_config`, `load_model` | `utils/io.py`, `core/model/load.py` |
| Inputs | `load_image`, `resize_long_side`, `prepare_inputs` | `utils/images.py`, `core/model/prompting.py` |
| Loops | `set_loops`, `count_layer_calls`, `expected_layer_calls` | `core/model/loops.py` |
| Inference | `generate` (with TTFT, time, peak memory), `last_logits` | `core/model/inference.py` |
| Pruning | `extract_visual_tokens`, selectors, `prune_inputs` | `core/tokens/` |
| Data | `build_gqa_splits`, `load_gqa` → `Item`; `make_splits`, `load_manifest` | `data/gqa.py`, `data/splits.py` |
| Scoring | `extract_answer`, `normalize_answer`, `score`, `score_gqa`, `summarize_rows` | `core/eval/` |
| Evaluation | `evaluate_item`, `run_evaluation` (resumable JSONL), `read_results` | `orchestration/evaluate.py` |

Notebooks: `01_huggingfaceload` (load, loops), `02_pruning` (tokens), `03_data_and_scoring` (S1 + runner).
CLI: `python scripts/mvp.py {smoke,splits,eval}`.

Limits: one item, one image per batch; pruning packs positions; thinking is still on (S2).
`set_loops` changes model state until the next call.
GQA loading, splits, extraction, scoring, timing and the runner are merged from Jiaqi-Gao/adaloop-vlm @ e870bff
(provenance comment in each file). Its hard-coded FLOPs estimate was not merged; S6 replaces it.
