# notebooks

Stages are from `docs/DEC3_MVP.md`. Notebooks import only `adaloop.api`.

| Notebook | DEC3 stage | `api` functions used |
|---|---|---|
| `01_huggingfaceload` | S0 (load, loops) | `load_config`, `load_model`, `load_image`, `prepare_inputs`, `set_loops`, `count_layer_calls`, `expected_layer_calls`, `generate` |
| `02_pruning` | S0 (visual tokens, selectors, pruning) | `load_config`, `load_model`, `load_image`, `prepare_inputs`, `set_loops`, `generate`, `last_logits`, `extract_visual_tokens`, `prune_inputs`, `random_selector`, `uniform_selector`, `question_embeddings`, `text_mean_selector` |
| `03_data_and_scoring` | S1 | `setup_notebook`, `config_hash`, `git_hash`, `build_gqa_splits`, `load_manifest`, `load_gqa`, `extract_answer`, `score_gqa`, `build_textvqa_splits`, `load_textvqa`, `score_textvqa`, `load_model`, `run_evaluation`, `read_results` |
| `04_thinking_off` | S2 (thinking off, `res_px`) | `setup_notebook`, `config_hash`, `git_hash`, `load_model`, `load_gqa`, `load_textvqa`, `prepare_inputs`, `set_loops`, `generate`, `extract_answer`, `resize_long_side`, `run_evaluation`, `read_results` |
| `05_positions_and_loops` | S3, S3b (positions after pruning, loop control, pass state) | `setup_notebook`, `config_hash`, `git_hash`, `load_model`, `load_gqa`, `resize_long_side`, `prepare_inputs`, `extract_visual_tokens`, `set_loops`, `last_logits`, `prune_inputs`, `question_embeddings`, `text_mean_selector`, `count_layer_calls`, `expected_layer_calls`, `record_pass_states`, `generate`, `check_pass_state` |
