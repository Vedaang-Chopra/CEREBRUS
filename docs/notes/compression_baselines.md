# Visual-token compression baselines (S4 / S5)

**Final set for the Dec 3 POC (Vedaang, Nov 25 evening):** controls `random`, `uniform`, `text_mean`, `lowres`; methods
`crisp_lite` (reimplemented, no official code), `rediprune`, `divprune`, `visionzip` (official code). Optional
`cdpruner_style`. ToMe and TRIM later. They form a 2×2: question relevance only (crisp_lite), relevance + diversity
(rediprune), diversity only (divprune), visual saliency without the question (visionzip). Build brief:
`docs/04_prompts/s4_s7_build.md`. The table below is the full research record.

Decided Nov 25, 2026 (Vedaang). Goal: strong training-free baselines for QAFE, built from the **official code** of each
method. No method is replaced by a simplified version without saying so: a method whose scorer had to change is named
`<method>-style` in code, tables and the paper. Methods without official code stay in the literature review only.

## Our vision tower (what the methods plug into)
`Inkling2VisionModel` (transformers `models/inkling2/modeling_inkling2.py` @ `2f1bc5e321b9`): 23 pre-norm ViT layers,
hidden 1024, 16 heads, patch 20, one CLS per image, learned 32×32 absolute positions + axial 2D RoPE, eager/sdpa/flash.
No final norm. Then `Inkling2VisionMerger`: groups of **4 adjacent patches (2×2)** → Linear 4096→4096, GELU, → 2304 =
one decoder token. **No aligned text encoder** (no CLIP text tower, no `visual_projection`).

Consequences:
- Selecting whole 2×2 units before the merger = selecting projected tokens after it. Pre- vs post-projector changes
  only the features a method **scores** on, not what the decoder sees.
- Only in-ViT methods (ToMe) save vision-tower FLOPs. All others save decoder FLOPs only.
- Methods that need a CLIP text tower cannot run as published: their relevance term is replaced (→ `-style`).

## Methods

| Method | Official code @ commit | License | Reused verbatim | Glue we write | Runs as |
|---|---|---|---|---|---|
| random, uniform, text_mean | ours | — | — | — | controls |
| **CDPruner** | Theia-4869/CDPruner @ `9541616c40fc` | Apache-2.0 | DPP kernel `L = diag(r) S diag(r)` + greedy MAP (`llava/model/llava_arch.py` `encode_images` l.141-187) | relevance `r` = cosine to mean question input embedding, min-max normalized (upstream uses the CLIP text tower); also `r = 1` (DPP diversity only) | **CDPruner-style**, post-projector |
| **ReDiPrune** | UA-CVML/ReDiPrune @ `ba04aa4427f7` | Apache-2.0 | `_select_text_cond` (l.38-127) and `extract_question_embedding` (~l.293) of `LLaVA-NeXT/llavavid/model/llava_arch.py`, incl. its query-dimension matching | k instead of env vars; features = pre-merger 2×2 units (mean of the 4 patch states, 1024-d), as upstream scores before the projector | **ReDiPrune** (pre-merger units); note: upstream is LLaVA-NeXT-Video only |
| **TRIM** | FreedomIntelligence/TRIM @ `f2032236583b` | Apache-2.0 | selection + one mean token of the unselected (`clip_encoder.py` `token_reduction` l.67-134) | scorer = question input embeddings vs projected tokens (not CLIP-aligned); k fixed (no IQR) | **TRIM-style**; faithful TRIM **blocked** (needs CLIP text tower). Outputs keep + 1 tokens |
| **ToMe** | facebookresearch/ToMe @ `af95e4b1befa` | CC BY-NC 4.0 | `tome/merge.py` (`bipartite_soft_matching`, `merge_wavg`, `merge_source`), `parse_r` | our ViT layer/attention forward with size-weighted proportional attention, per-image loop, merged positions; **merges 2×2 units, not patches**, so the merger gets the input it was trained on and the final count is exact | **ToMe (unit-level)**; per-patch ToMe at exact 25/50% is not possible with a 2×2 merger |
| DivPrune | vbdi/divprune @ `799e2d950aa0` | CC BY-NC 4.0 | max-min diversity selection | none (post-projector) | optional, cheap; ReDiPrune builds on it |
| CRISP / crisp_lite | none found | — | — | — | **excluded**, literature only |
| HQAP | none found | — | — | — | **excluded**, literature only |

Not added for Dec 3 (official code, but they prune inside the decoder or need ViT attention): FastV, SparseVLM, DART,
VisionZip, PruMerge. Candidates for the full paper.
CC BY-NC code (ToMe, DivPrune): fine for the course paper; check before any public release.

## Code layout
- Upstream repos cloned read-only to `old_code/third_party/<name>/` (git-ignored), at the commits above.
- Reused code copied to `adaloop/core/tokens/methods/<name>.py` with a provenance header
  (`# Ported from <repo>:<path> @ <commit>, <license>`), the copied part unchanged, glue below it.
- One interface for every method: `compress(vlm, inputs, question, k, cfg) -> (mm_encoder_outputs, info)`;
  `info` = kept token count, compression time, extra FLOPs. Selected by `cfg["compressor"]`; `none` = bypass (k = 100%).
- Loop-state extraction (`record_pass_states`, per-pass features) is unchanged: it reads decoder states after
  compression.

## S5 protocol
Frozen model, dev split, GQA + TextVQA, R = 3, answer format from S2, k ∈ {25, 50, 100}%.
Per method × k: accuracy, FLOPs (ViT + decoder + compression overhead), latency, compression time, actual visual
token count. 20 items per benchmark first (check), then the full dev split (100 per benchmark).
Winner = best accuracy at k = 25% and 50% within noise, ties broken by cost. Frozen in `configs/mvp.yaml` → used by S7–S10.

## As built (S4a)
Interface: `compress(vlm, inputs, item, k, cfg) -> (model inputs for generation, info)` in
`adaloop/orchestration/compress.py`. Kept count `K = max(1, int(N * k))` for every method except `lowres`.
At `K = N` nothing is scored: the same packed path runs with every token. Parameters: `compression:` in `configs/mvp.yaml`.
Upstream clones: `old_code/third_party/{rediprune,divprune,visionzip}`; each ported file's header lists the changes.

| Method | What runs here (deviations from the paper) |
|---|---|
| random, uniform, text_mean | random: seed = split seed + crc32(item id). text_mean: mean embedding of the question without the instruction suffix |
| lowres | long side (step 4 px, at most `res_px`) whose `ceil(w/40) * ceil(h/40)` is closest to k × full count; the count is not exact |
| crisp_lite | **not full CRISP.** spaCy nouns (NOUN, PROPN) → mean subword input embeddings → best cosine per projected token; top 75% of K, then 25% farthest-point fill. No per-noun quota (Eqs. 3-4), no feature-diversity stage (Eq. 5). No noun: rank by CLS attention (as the paper) |
| rediprune | upstream `_select_text_cond` byte-identical, α = 0.5, τ = 0. Features = 2×2 units of the last ViT states (1024-d). The 2304-d question-tail embedding is cut to 1024-d by upstream's `adaptive_avg_pool1d` branch: the two spaces are not aligned, so relevance carries little meaning on this model |
| divprune | upstream max-min on the projected tokens (cosine distance). An index picked twice (only possible with identical tokens) raises |
| visionzip | upstream Qwen2.5-VL merge on the projected tokens. Score = CLS attention of the last ViT layer (our tower has a CLS token; upstream's Qwen path has none and sums over all queries), 2×2 mean; keys = rotary keys, 2×2 and head mean. Contextual = `int(0.05 * N)` tokens, dominant = the rest of K (upstream at 50%: 45% + 5%) |

CDPruner-style (optional) is not built.

FLOPs of the step (`core/cost/compression.py`): cosine scores, greedy loops, the question self-attention (rediprune), the
CLS attention row and merge (visionzip). Not counted: softmax/argmax/sort, the spaCy tagger. Wall time `compression_s`
includes all of them.

Checked on a tiny random model (CPU): k = 100% gives identical last logits and ids for every method (R = 1, 3); exact
kept counts at 25/50%; the CLS attention equals the layer's eager attention weights; VisionZip changes only the
contextual target tokens. Not yet run on the real weights (nb07).

## Features (S4b)
`orchestration/features.py::run_item_cells`; one record per (k, p), from the R = p run, in `features/<variant>/<config_hash>/<item_id>.pt`:
`h_last` (fp16, last core layer, last prompt position), `delta_norm` = ‖h_p − h_{p−1}‖ / ‖h_{p−1}‖, `vis_attn_mass`
(last core layer 17, heads mean, attention of the last prompt position on the kept visual tokens), `entropy`,
`top1_prob`, `kl_prev` = KL(p_p ‖ p_{p−1}) from first answer-token logits (R = p vs R = p − 1 runs), `flops_readout`,
`flops_state`, `n_tokens`, `n_vis`. NaN where there is no earlier pass.
The grid generates every R on its own, so all features of pass p come from that R = p run (nb05: its states equal pass p
of an R = 5 run bit for bit).
