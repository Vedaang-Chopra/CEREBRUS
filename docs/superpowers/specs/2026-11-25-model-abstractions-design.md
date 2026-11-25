# Model abstractions: design (short)

Approved by Vedaang, 2026-11-25. Part of stage S0 in `docs/DEC3_MVP.md`.

## Intent
Move loading, inference, loop control and pruning out of `notebooks/01_huggingfaceload.ipynb`
and `02_pruning.ipynb` into importable code, so notebooks and grid runs share one path.

## Goals / non-goals
- Goals: reusable typed functions; no hidden globals; same behaviour as the notebooks.
- Non-goals: thinking-off prompting (S2), original positions (S3), attention selector,
  heatmap plots, scorers, datasets, FLOPs.
- Anti-goal: a class that collects every model operation.

## Constraints
- Layout from `docs/DEC3_MVP.md` §3; notebooks import only `adaloop.api`.
- Config values in `configs/mvp.yaml`. Code runs only on the GPU host.

## Approaches
- **A (picked): small dataclass + plain functions.** One handle, no globals, each step runs alone.
- B: class with methods. Neat calls, grows into a god class with hooks, pruning, LoRA.
- C: bare functions taking model and processor. Simplest, noisy signatures.

## Design
```
adaloop/api.py             re-exports only
adaloop/core/model/
  load.py        load_model(cfg) -> LoadedVLM(model, processor)
  loops.py       set_loops(vlm, R), count_layer_calls(vlm, inputs)
  prompting.py   prepare_inputs(vlm, question, image=None)
  inference.py   generate(vlm, inputs, max_new_tokens), last_logits(vlm, inputs)
adaloop/core/tokens/
  prune.py       extract_visual_tokens(vlm, inputs), prune_inputs(...)
  selectors.py   random, uniform, text_mean
utils/images.py            load_image(url_or_path, max_side)
configs/mvp.yaml           model_id, memory limits, res_px, max_new_tokens
```
`inference.py` is an addition to the spec's file list.

## Fixes made while porting
- Loader takes no `R` (load-time override is ignored); only `set_loops` sets it.
- Drop the unused `DEVICE_MAP`; placement is `device_map` + memory limits from config.
- `verify_loops` and `test_loop_execution` become `count_layer_calls`.
- Pruning keeps the current packed positions.

## Verification (GPU host)
Layer calls match R; k=100% pruning gives identical logits; logits differ between R=1 and R=3.

## Left to the builder
Exact types, config loading, logging detail.
