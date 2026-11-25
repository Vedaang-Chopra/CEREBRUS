# adaloop

Public entry points: `adaloop/api.py`. Notebooks import only that. Test notebook: `notebooks/mvp.ipynb`.

| Step | Functions | File |
|---|---|---|
| Load | `load_config`, `load_model` | `utils/io.py`, `core/model/load.py` |
| Inputs | `load_image`, `prepare_inputs` | `utils/images.py`, `core/model/prompting.py` |
| Loops | `set_loops`, `count_layer_calls`, `expected_layer_calls` | `core/model/loops.py` |
| Inference | `generate`, `last_logits` | `core/model/inference.py` |
| Pruning | `extract_visual_tokens`, selectors, `prune_inputs` | `core/tokens/` |

Limits: one item, one image per batch; pruning packs positions; thinking is still on.
`set_loops` changes model state until the next call.
