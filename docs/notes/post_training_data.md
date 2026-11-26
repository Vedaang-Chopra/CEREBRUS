# Post-training data: research and proposal (Nov 26, 2026; DRAFT, not approved)

Purpose: the training set for the budget-tolerance LoRA (S8/S9 redesign). GQA and TextVQA stay evaluation only.
Sources: two read-only research passes by main-discussion subagents (Nov 26). "Verified" = checked in the paper or the
dataset card; "inferred" = from memory or reasoning.

## What the closest papers trained on
| Paper | Data | Size | Selection |
|---|---|---|---|
| ViCO / InternVL3.5 (2508.18265 §2.3, verified) | same as SFT (~56M samples) | ~30B tokens; router 270K OCR/VQA | none beyond SFT mix |
| LT-OPD (2609.32353 §3.4, App. A Tab. 8, verified) | OneThinker (MC, math, numeric, OCR, regression), PixMo-AMA 3k, LLaVA-1.5 mix 2.8k, Vision-OPD MC 1k | 14k | balanced quotas, leakage check, no quality filter |
| Glance-ETD (2611.20006 §4, App. B.5/D.3, verified) | LT-OPD-14K + 1.5k spatial MC | 14k | **700–7K coverage subsets retain 90.0–90.7% vs 90.4% for 14K**; dedupe dHash ≤6, pHash ≤10 |
| Token-Budget Distillation (2608.28138, verified) | LLaVA-Video-178K | n/a | KL only where the teacher is correct and confident |
| GRIP-VLM (2605.13375, verified) | LLaVA-665K | 10 epochs | selector only |

Findings: volume is not the limit (Glance); pruning hurts reading/document questions most and training helps them
most (Glance §4: DivPrune TextVQA 76.8 → 70.8); no paper ablates the mix composition.

## Source collection
**HuggingFaceM4/FineVision** (arXiv 2510.17269, verified): embedded images, `source` field, per-turn 1–5 ratings
(relevance, visual_dependency, image_correspondence, formatting; judges Qwen3-32B / Qwen2.5-VL-32B), 185 subsets
incl. CoSyn-400K. Rating filters did not beat unfiltered data (App. A.4): use them only to drop the worst items.
Licences per source (card states none). PixMo-Count / PixMo-AMA are not in FineVision (URLs + sha256; ~85–97% live).

**Excluded (eval image overlap):** `textvqa`, `textcaps`, `textocr(gpt4v)`, `localized_narratives` (OpenImages);
`tallyqa`, `visual7w`, `st_vqa` (Visual Genome). COCO subsets allowed only after a pHash check against GQA eval images.
**Excluded (resolution):** DocVQA / InfographicVQA images taller than ~1,536 px (unreadable at 768 px).

## Proposed mix (~12k items, one turn per image, 1 epoch)
| Bucket | Source (FineVision unless noted) | n |
|---|---|---|
| General VQA | `vqav2`, `aokvqa`, `vsr` (pHash vs eval) | 2,500 |
| Charts | `chartqa`, `dvqa`, `plotqa`, `CoSyn_400k_chart` (≤500 each dvqa/plotqa) | 2,500 |
| Docs / tables | `CoSyn_400k_document`, `_table`, `_nutrition`; `docvqa` only if height ≤1,536 px | 2,000 |
| OCR / scene text | `ocrvqa`, `est_vqa`, `cocotext` | 2,000 |
| Diagrams / science | `ai2d_merged`, `scienceqa`, `iconqa` | 1,000 |
| Counting | `clevr` + PixMo-Count | 1,000 |
| Free-form | PixMo-AMA or `LLaVA_Instruct_150K`, answer cap ~64 tokens | 1,000 |

## Filters, in order
1. Ratings: `relevance_min`, `image_correspondence_min`, `formatting_min` ≥ 4.
2. Short-answer turns ≤ 8 tokens (except free-form); strip style suffixes, normalize to our `ANSWER:` format.
3. Images: short side ≥ 224, aspect ≤ 2.5, long side ≤ 1,536 px.
4. pHash/dHash dedupe vs all GQA and TextVQA eval images (Glance thresholds).
5. Teacher pass (full tokens, R = 3): keep items the teacher answers correctly. Optional: prefer items whose answer
   changes at 25% tokens (the items that carry the signal; GRIP LV2/LV3 idea, inferred).
6. Drop multi-reference answers without a clear majority.

Open (Vedaang): approve the mix; R sampling (uniform vs 1:2:6:2:1); data-only control run.

## Deviations (candidate store build)
- Ratings filter applies to the chosen turn (`<key>_ratings`), not the row's `<key>_min` columns: the `_min` columns
  passed almost nothing in several configs.
- Counting falls back to `clevr`: PixMo-Count downloads fail (HTTP 429, sha256 mismatches).
- `est_vqa` (about 80% Chinese) has an English filter: a turn is kept only if the question and the answer are mostly
  ASCII letters (digits and punctuation ignored; a number answer is fine). If fewer than 30% of the first 500 turns
  pass, the whole source is dropped and `ocrvqa` takes its share. Thresholds: `configs/post_training_data.yaml`.
