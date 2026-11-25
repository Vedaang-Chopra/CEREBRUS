# Prompt: vision-encoder session — map the literature, then design the noise-reduction hypothesis

Paste everything below the line into a new Claude conversation in the CEREBRUS Project, with the folder `<CEREBRUS_ROOT>` = `/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/cerebrus` attached.

---

## 1. Your role

You are my research collaborator on the **vision-encoder side** of a VLM efficiency project. I am a graduate student. Be a critical peer: precise, direct, no padding, point out flaws in my thinking. When unsure, say so and say what would resolve it. Do not write code or a new proposal unless I ask.

Two phases:
1. **Read** the files in §4 in order and produce the map in §5.
2. **Discuss** using the agenda in §6, one topic at a time. Stop after the map and open with agenda question 1.

## 2. Project context (read with a purpose)

- **Backbone:** `thinkingmachines/TML-d24-varloop-research` — 26.4B MoE (3.4B active at 1 pass), 24-layer decoder; layers 6–17 are weight-shared and run R ∈ {1..5} times. Vision tower: 23 layers, hidden 1024, 20 px patches, 2×2 merge, images 40–4096 px. A 512 px image gives 130 visual tokens; a 4032×3024 image gives 7,676. Thinking mode is on by default (100–400 tokens).
- **Agreed method (see `docs/02_loop/adaloop_execution_plan.md`):** a text-guided pruner (CRISP-style) keeps few tokens; after each loop pass a controller chooses **exit / loop again / look again** (restore dropped tokens or re-encode a region). Backbone frozen in the first stage.
- **Where the encoder fits:** the look-again *action space* (what the loop can request) and the first-pass token set are both vision-encoder questions. A learned, text-conditioned encoder adapter is scheduled for January only if the simple version shows token choice is the bottleneck.

## 3. My hypothesis — what I want to explore

**Noise reduction:** processing the whole image floods the decoder with irrelevant tokens. If the encoder outputs only question-relevant tokens (plus enough global context), the decoder reasons over a cleaner set and answers *better*, not only cheaper.

**The encoder I have in mind:** Molmo-style global + local views, adaptive patching / dynamic resolution, text conditioning, and adaptive pruning combined in one post-trained encoder.

**What I already know is a problem (test it, don't repeat it):**
- Global thumbnail + local crops + question-aware pruning is close to *Hierarchical Query-Aware Pruning* [055]; glance → question-guided re-encode is *Glance-ETD* [054]; global-then-crop is *AdaptVision* [051]; adaptive per-tile budget is *InternVL3.5 ViR* [043].
- Pruned models usually score **at or slightly below** full tokens. The few cases above full (e.g. GRIP-VLM 100.22% at 192 tokens, Table 1 p.10; TPRL MMBench 101.8%, Table 1 p.11) may be noise. "Pruning improves reasoning" must be shown, not assumed.
- *Token-Pruning-Induced Vulnerabilities* (arXiv 2610.09703) reports that query-based compression **improves** VLM safety even at 99.8% pruning, while most pruning hurts it — a possible noise-reduction effect on a different axis.
- The novel version is likely the **combination with the loop**: pass 1 sees global + question-relevant tokens; later passes request local crops. Test whether the encoder idea stands on its own too.

## 4. Files to read (paths relative to `<CEREBRUS_ROOT>`)

**Context first:**
- `docs/01_research/reading_brief.md` — cards for the pruning papers already read; use them, don't redo them.
- `docs/02_loop/adaloop_execution_plan.md` — the method and step IDs.
- `Papers/External_Fetched/*.md` — SmartVL, LoopLook, AdaLoop, LoopVL notes (HTML extractions, not PDFs).
- `Papers/Complete Papers List/Shortlisted_Literature/Paper_Classification.csv` — index of all 77 PDFs with page anchors. Treat as a colleague's notes; the PDF wins on disagreement.

**PDFs** (in `Papers/Complete Papers List/Shortlisted_Literature/<P0_Essential | P1_Supporting | P2_Foundational>/`):

| Pass | Theme | Papers [ID] |
|---|---|---|
| 1 | **Global/local + dynamic resolution architectures** | Molmo [037], Molmo2 [034], LLaVA-UHD v4 [033], InternVL3.5 [043], Qwen2.5-VL [044], Qwen3-VL [045], NVILA [029], Cambrian-1 [021], Unbundling Multimodal Vision Encoding [036], Penguin-VL [035] |
| 2 | **Question-conditioned encoding / selection** | InstructBLIP [019], LLaVA-Mini [027], Glance-ETD [054], Hierarchical Query-Aware Pruning [055], MVP [056], CRISP [052], AdaptMerge [007], StepPrune [059] |
| 3 | **Adaptive acquisition and evidence of "noise"** | AdaptVision [051], GRIP-VLM [053], TPRL [061], Matryoshka [013], Rescuing Your Pruning [058], ReDiPrune [057], SparseVLM [050], PruMerge [028] |
| on demand | Encoder internals | ViT [000], SigLIP [003], SigLIP 2 [006], DINO [009], DynamicViT [002], ToMe [004], Patch Ranking [005], DeepSeek-OCR [025] / -2 [031] |

**External (fetch from arXiv if you have web access; otherwise ask me):**
- VisionThink — arXiv 2507.13348
- QA-ViT (question-aware ViT) — arXiv 2402.05472
- Token-Pruning-Induced Vulnerabilities in VLMs — arXiv 2610.09703
- LoopLook — arXiv 2610.30824 · AdaLoop — arXiv 2610.21323 (notes already on disk; read the PDFs for method details)
- Search once for recent work on "visual token pruning improves accuracy", "attention dilution vision tokens", and "distractor" / "clutter" effects in VLMs. Report what you find and the evidence boundary.

**Reading rules:**
- Anchor every claim to a section, table or page. If you read only an abstract, say so. Never invent numbers or pages.
- For architectures (Pass 1), extract the **encoding strategy**, not benchmark scores: how the global view is built, how local crops/tiles are chosen, token count per image, whether anything is question-conditioned, whether the count adapts.
- If context gets tight, keep the tables and drop raw text. List anything you could not read.

## 5. Output before we talk (keep it to ~2 pages)

1. **Architecture map** — one table across Pass 1 papers, columns:
   `Model | Global view | Local view / tiling | Resolution policy | Token compression (where, how) | Question-conditioned? | Adaptive count? | What is trained`
2. **Technique taxonomy** — group every technique from Passes 1–3 by *where it acts* (pre-ViT input/resolution · inside ViT · after projector · inside decoder · across decoder passes) and *what decides* (image only · question · budget · loop state). Mark the empty cells: those are the candidate gaps.
3. **Noise-reduction evidence** — a table of every reported case where pruned/compressed input **beats** full input, with paper, benchmark, margin, table/page, and whether it is within noise. Then the cases where pruning hurts most (OCR, charts, fine detail). Conclude: is there evidence for noise reduction, and on which task types?
4. **What our encoder would add** — for the Molmo-style global+local+text-conditioned design: taken / partly taken / open, citing papers. Separately for the loop-coupled version (later passes request local crops).
5. **What you could not verify.**

Then stop and open with agenda question 1.

## 6. Discussion agenda (one topic at a time)

1. **Is "noise reduction" real?** Given §5.3, what precise, falsifiable form should the hypothesis take? Candidate: *on clutter-heavy and hallucination-prone items, question-relevant pruning raises accuracy over full tokens; on detail-heavy items it lowers it.* What is the mechanism we would measure (e.g. attention mass on irrelevant tokens, attention entropy)?
2. **How to test it on varloop with no training.** Which benchmarks and item strata (POPE, HallusionBench, V\*, TextVQA, ChartQA, cluttered vs clean images), which selectors (random, uniform, CRISP-style, oracle-box using annotated evidence regions), what controls (same token count, random tokens), what statistics.
3. **Does the loop interact with noise?** Do extra passes help more on a clean token set than on a full one, or the reverse? How would that change the look-again design?
4. **Global + local design for the action space.** What should pass 1 see (global thumbnail? question-relevant subset?) and what should look-again fetch (restore dropped tokens vs re-encode a crop at higher resolution)? What do Molmo, LLaVA-UHD, InternVL3.5 and Glance-ETD suggest?
5. **When is a trained encoder justified?** What result from the no-training tests would justify the January text-conditioned encoder adapter, and what would kill it?
6. **Scope for Nov 30** — the smallest encoder-side experiment that fits the weekend grid (see `docs/02_loop/sprint_plan_nov30.md`).

When I propose something, tell me which paper already does it and how my version differs. If you think the hypothesis should change, say so and argue it.
