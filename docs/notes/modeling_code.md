# Inkling2Loop modeling code: positions and loop passes (stages S3, S3b)

Source read: `huggingface/transformers` @ `2f1bc5e321b9a736200a590d6e7743ccdeb60772` (the `pyproject.toml` pin),
`src/transformers/models/inkling2_loop/modeling_inkling2_loop.py` (the checkpoint's `model_type` is `inkling2_loop`,
not `inkling2`). `main` @ `b62646bbda836d467060b7e3d64028202079b77d` differs only by refactoring in these parts.
Config: `thinkingmachines/TML-d24-varloop-research` @ `91a57fe2b779bbd12c0f74d1f518cbd5b84e8d0d`.

## Modules and layout
- `Inkling2LoopForConditionalGeneration.model` (`Inkling2LoopModel`) `.language_model` (`Inkling2LoopTextModel`) `.layers`,
  so the path is `vlm.model.model.language_model.layers[i]`.
- 24 layers: prelude 0-5, core 6-17 (`loop_layer_index_start` / `_end`), coda 18-23. Layers 5, 11, 17, 23 use full
  attention, the rest sliding attention (window 1024). A decoder layer returns one tensor.

## Positions
| Structure | Built from | Where |
|---|---|---|
| Relative position bias (all layers) | token **index**: cache length and offset | `get_relative_distance`, `Inkling2LoopRelativePositionBias` |
| Causal and sliding masks | index; `position_ids` only to detect packed sequences when there is **no** attention mask and **no** cache | `create_causal_mask`, `masking_utils._preprocess_mask_arguments` |
| Short convolutions on keys, values, MLP/attention outputs (kernel 4) | neighbouring **index** | `Inkling2LoopShortConvolution`, `Inkling2LoopDynamicShortConvolution` |
| Log-scaling of queries and relative states (full layers only) | `position_ids`; factor `1 + 0.1 * log(max((pos + 1) / 8192, 1))` | `Inkling2LoopLogScale` |

- `forward` (text model, multimodal model, conditional generation) accepts `position_ids`; if None, the text model
  counts them from the cache length. There is no relative-distance argument. The masks, including the conv padding
  mask, can be replaced by passing `attention_mask` as a dict keyed by mask type.
- The multimodal wrapper computes no positions (no 2D or multimodal position scheme); it passes `position_ids` through.
  `forward` also accepts `mm_encoder_outputs`, which pruning uses.
- `generate` creates `position_ids` from the attention mask if none are given, and extends given ones by +1 from the last.

**Conclusion (S3): original positions are not supported; pruning stays packed.** Passing the original `position_ids`
for kept tokens changes nothing below position 8192, because token distances, masks and convolutions never read them.
So `keep_positions` was not added: it would label a packed run as positioned. Supporting it would mean patching
`get_relative_distance` and building custom sliding masks, and convolutions would still treat kept tokens as neighbours unless the
dropped positions stayed in the sequence. That is a change to the model, not built.

Consequence: packing shortens the distance between kept visual tokens, and from the question back to them.

## Prompt and thinking (chat template of the checkpoint)
- Order: the system prompt, then the template's own system message `Thinking effort level: <x>`, then the user image
  and question messages. No generation prompt is added; `add_generation_prompt` is not read.
- `reasoning_effort`: none/minimal/low/medium/high/xhigh/max or a number in [0, 0.99]; it only changes `<x>`.
- A thinking reply is a `<|content_thinking|>` message ended by `<|end_message|>`, then a `<|content_text|>` message,
  then `<|content_model_end_sampling|>` (the `eos_token_id` of `generation_config.json`).
- Thinking budget (`core/model/budget.py`): phase 2 does not reuse the KV cache; it runs the vision tower and the
  prompt forward again over prompt + thinking + appended ids (`n_reprefill_tokens`).
- Thinking tokens are decode cost: FLOPs take all generated tokens. Thinking length may change with R, so rows log it.

## Loop passes (S3b)
- `Inkling2LoopTextModel.forward`: prelude once; first core pass without messages; each later pass restarts from the
  prelude output and gets messages from the previous pass (`message_input`, queries indexed by **layer**, not pass).
  Coda and `final_input` once, after the last pass.
- Nothing inside a core pass reads the total R or uses the pass index in a computation. The pass index only selects
  cache slots (`cache_offset`; R - 1 copies of the core cache layers). The only difference between passes is
  "first pass (no messages)" vs later passes. So pass p of an R = 5 run should equal the end of an R = p run.
- The coda needs the block states, cache, masks and distances of the pass, so there is no simple correct readout at
  pass p of a longer run. Notebook 05 compares the R = p answer with the R = 5 answer instead.
- The checkpoint config has `loop_config.is_varloop: true`; the transformers code never reads it. Not checked against
  a reference implementation.

## Checked on a tiny random model (CPU, same commit, torch 2.15)
Not the real weights. Layer calls = 6 + 12R + 6 for R = 1..5; k = 100% pruning bit-exact; original `position_ids`
at k = 50% give identical logits and identical generated ids; shifting them by 100000 changes logits; without an
attention mask and cache, gapped `position_ids` change logits (packed-sequence masking); pass-p states of an R = 5 run
equal (`torch.equal`) the final states of R = p runs.

## Teacher cache (S8)
`orchestration/teacher_cache.py`. Teacher inputs: `item_inputs` with `answer_prefill` = `no_think_prefill`, all other
settings from `mvp.yaml` (including `reasoning_effort`, so the system text matches evaluation), `max_new_tokens` 32. The
vision tower runs once per item (`mm_encoder_outputs`, packed with every token); batches carry one R. Log-probs are the
log-softmax of the generation's own logits (batched, KV cache); nb09 compares them with a teacher-forced full forward.
Layout: `execution_results/teacher/<hash>/R<r>/shard<i>of<n>_copy<j>_part<k>.pt` + `index_shard<i>of<n>_copy<j>.jsonl`.
`n_answer_tokens` = number of answer ids, end token included. The hash covers model, res_px, prompts, answer format,
prefill, reasoning effort, `code_version`, `top_n` and `max_new_tokens`: bumping `code_version` starts a new cache.
