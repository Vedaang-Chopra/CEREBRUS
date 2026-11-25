# notebooks

Stages are from `docs/DEC3_MVP.md`. Notebooks import only `adaloop.api`.

| Notebook | DEC3 stage | `api` functions used |
|---|---|---|
| `01_huggingfaceload` | S0 (load, loops) | `load_config`, `load_model`, `load_image`, `prepare_inputs`, `set_loops`, `count_layer_calls`, `expected_layer_calls`, `generate` |
| `02_pruning` | S0 (visual tokens, selectors, pruning) | `load_config`, `load_model`, `load_image`, `prepare_inputs`, `set_loops`, `generate`, `last_logits`, `extract_visual_tokens`, `prune_inputs`, `random_selector`, `uniform_selector`, `question_embeddings`, `text_mean_selector` |
| `03_data_and_scoring` | S1 (GQA part) | `setup_notebook`, `config_hash`, `git_hash`, `build_gqa_splits`, `load_manifest`, `load_gqa`, `extract_answer`, `score_gqa`, `build_textvqa_splits`, `load_textvqa`, `score_textvqa`, `load_model`, `run_evaluation`, `read_results` |
