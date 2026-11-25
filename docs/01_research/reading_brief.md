# Reading brief — Adaptive Looped VLM (Proposal_v2)

2026-11-24. Evidence boundary: the proposal, the varloop model card, the earlier review notes, 15 P0 + 2 P1 corpus PDFs (page-numbered text; page = 1-based PDF page), 4 external papers read as arXiv HTML extractions (arXiv blocked for download; notes in `Papers/External_Fetched/`), and 10 cited papers checked at abstract level. Nothing here is a field-wide novelty claim.

---

## 1. Cards

### Pass 1 + SmartVL (full)

**StepPrune [059]** (arXiv 2609.13804, Sep 2026)
- Method: cross-modal pointer decoder selects visual tokens one at a time, conditioned on question and prior picks; learned STOP sets K (§3.1–3.2, pp.3–4).
- Acts: pre-LLM, after projector. Question-conditioned: yes (text jointly encoded, Eq.3, p.4).
- Budget: varies per input, **learned** (STOP); λ=0.03 length penalty sets the mean (Eq.10, p.5; §4.6, p.11).
- Trained: encoder, pointer decoder, denoiser, START/STOP. Frozen: ViT, projector, LLM (p.6). Objective: answer CE + expected length, noise-gated surrogate; **not RL**.
- Headline: LLaVA-1.5-7B, 88.9% pruned → 94.6% retention (Table 1, p.8); prefill 59.95→40.05 ms (§4.4, p.9).
- **Matched-mean adaptive vs fixed K: −0.57 to +0.90 pp** across 6 benchmarks (Fig.3c, p.10).
- Takes: "learned per-input visual budget with frozen decoder." Leaves: depth axis; recurrence. Its Fig.3c also calibrates expectations for H1/H3: realized per-input allocation gains are ≲1 pp.

**LoopZip [038]** (arXiv 2611.22728, 17 Nov 2026)
- Method: training-free; VisionZip-style aggregation N→K0 before backbone, then token selection from mean instruction attention over the **first two calls of a shared layer**, value correction fitted to both calls' outputs, compaction to K1 for later calls (§3.2–4.5, pp.2–5; Fig.3 p.4).
- Acts: pre-LLM + **inside the loop, across passes**. Question-conditioned: yes (instruction attention).
- Budget: externally supplied average pruning ratio; **fixed loop schedule**, no learned K or R (§3.1, p.2).
- Trained: nothing. Backbones: LoopVL-1B and **SenseNova-Loop-9B-Preview** (middle 24 layers ×2, fixed; §5.1, p.5).
- Headline: LoopVL-1B 50% pruning 60.58→59.02 avg, 97.16% rel. (Table 1, p.6); 2.01–2.47× end-to-end, −58% peak memory at 8192 tokens (§5.3, p.7). Token importance shifts across calls (Fig.2a, p.3).
- Side result that matters: **SparseVLM is among the weakest selectors in looped VLMs** (87.5% / 90.9% rel. at 50%, Table 1 p.6).
- Takes: "visual compression in a looped VLM." Leaves: variable R; per-item allocation; k×R interaction.

**TPRL [061]** (arXiv 2603.13394, Mar 2026)
- Method: autoencoder-compressed state + query embedding → policy outputs per-token keep probs; multi-step at training, **one-shot threshold τ at inference** (§3, Eq.16, p.7; p.8).
- Acts: pre-LLM. Question-conditioned: yes (q in state, p.5). Budget: per-example via threshold; τ tuned to hit an average.
- Trained: policy (BC init → PPO), **and the LLM in the headline**; frozen-LLM ablation separate (Table 5, p.12). Reward: score improvement + compression (Eq.4–6, p.5).
- Headline: LLaVA-1.5-7B, 192* tokens → 99.3%, −54.2% FLOPs (Table 4, p.13; Table 1, p.11). Frozen vs fine-tuned: 98.8 vs 99.6% VQAv2 (Table 5, p.12). PPO ≈ GRPO ≈ MAPO (Fig.3, p.13–14).
- Takes: "RL, question-conditioned token budget." Leaves: depth axis.

**AdaptMerge [007]**
- Method: training-free merging in ViT path (diversity, ρ=0.05) and after projector by text relevance (τ=0.15) using the jointly-trained SigLIP text encoder (§3, pp.3–5).
- Acts: encoder + connector. Question-conditioned: yes. Budget: varies per input, set by thresholds (not learned).
- Trained: nothing. Headline: Gemma-3-4B AI2D 68.10→65.12 at 7.91→5.12 TFLOPs (Table 1, p.6).
- **Cost lesson:** vision encoder 0.532 s vs LLM 0.172 s per sample on Gemma-3-4B (Table 2, p.7) — post-projector pruning cannot touch the dominant term.
- Takes: frozen, query-aware variable-count reduction. Leaves: depth axis.

**GRIP-VLM [053]** (arXiv 2605.13375, May 2026)
- Method: FiLM budget-aware scorer (inputs: token reps, current and target ratio), SFT distillation warm-up then GRPO, G=16 (§4, Eq.6–7, pp.8–9).
- Acts: encoder-side pre-LLM or decoder-side. Question-conditioned: only implicitly (decoder variant reads LLM hidden states); explicit question input not documented.
- Budget: **supplied**. Trained: scorer only; LLM frozen (p.10). Objective: SFT then GRPO with task + over-budget reward.
- Headline: 64 tokens encoder+RL 94.14% vs decoder+RL 90.71% (Table 1, p.10). **RL over its own SFT: +1.0–1.5 pp (decoder), +0.1–0.4 pp (encoder)** (Table 1, p.10).
- Takes: "GRPO scorer with frozen backbone." Leaves: budget choice; depth.

**SmartVL** (external, arXiv 2607.20357, Jul 2026; HTML read, no tables in HTML)
- Method: budget token in ViT input → Gumbel-sigmoid token mask; budget token in prompt → per-layer (L) or per-head-group (LH) gates after P full prefix layers; differentiable FLOPs estimator (§3.1–3.3, Eq.5–10).
- Acts: inside ViT (mask) + inside decoder (layer/head skip). **Question conditioning: token mask NO (image + budget only, §3.1); compute gates indirectly via prefix attention (§3.2).**
- Budget: **supplied** target b, sampled at training; allocation per input. Trained: LLM, projector, controllers. Frozen: ViT. Objective: supervised + FLOPs penalty; not RL.
- Headline: LLaVA-1.5-7B VQAv2 @50% FLOPs 74.4 vs AdaLLaVA 67.9, vs AdaLLaVA-PruMerge 74.5 (§4.2); TextVQA @50% 54.4 vs 29.8.
- **§4.4 / Fig.6: manual token-ratio × compute-ratio grid; POPE best at T=0.8,C=0.5; VQAv2/TextVQA at T=0.9,C=0.8 (7 TFLOPs); "no single configuration … optimal" across tasks.** Prefill-only; decode named as open; no wall-clock.
- Takes: joint token + compute allocation; dataset-level token×compute grid. Leaves: weight-shared recurrence; item-level causal analysis; frozen decoder.

### One line each

| Paper | Line (anchor) |
|---|---|
| InstructBLIP [019] | Instruction-aware Q-Former, ViT+LLM frozen, fixed 32 queries; instruction-awareness GQA 48.4 vs 45.9 (Table 2, p.7). Takes question-conditioned frozen-decoder connector. |
| Matryoshka M3 [013] | Nested pooled scales, LLM fine-tuned. **Per-item oracle over 5 scales: MMMU 50.44 vs ~34–36 single-scale; POPE 94.29 vs ~87** (Table 2, p.6) — oracle inflation precedent. |
| LLaVA-Mini [027] | Text pre-fusion + 1 visual token; VQAv2 77.6 vs LLaVA-1.5 78.5 (Table 1, p.7); decoder trained after alignment. |
| InternVL3.5 [043] | ViR router 256/64 tokens per tile, router-only stage with frozen backbone; 80.2→79.8 avg (Table 17, p.24); 2.71→10.97 throughput is DvD+ViR combined (Table 18, p.24). |
| RecursiveVLM [039] | Whole decoder re-run with feedback connector, per-pass CE, backbone trained; pass-2 62.60 vs 58.60 (Table 1, p.6); more depth can hurt (Table 6, p.8). |
| AdaptVision [051] | Low-res + RL crop tool, policy fine-tuned; 33% tokens at 97.9% (Table 1, p.6). |
| Glance-ETD [054] | Question evidence scorer at 5% budget; frozen scorer 82.0 vs self-distilled 91.6 retention; OCRBench 590 vs 848 dense (Table 1, p.6). |
| HQAP [055] | Training-free; image complexity sets K, question sets allocation; adaptive vs fixed **97.8→98.0** (Table I, p.5). |
| MVP [056] | Frozen-VLM question pruner; fixed-64 96.38% vs threshold-variable mean 63.7 → 94.42%; TextVQA 56.3→43.6 (App. Table 6, p.14). Variable K can lose to fixed K. |
| TOP-RL [060] | PPO pruning inside decoder layers; 192/576 → 97.6% (Table 1, p.6); weak cross-task transfer (Table 4, p.7). |
| SparseVLM [050] | Training-free, text-raters' attention inside LLM + recycling. Weak in looped VLMs (LoopZip Table 1, p.6). |
| CRISP [052] | Pre-LLM: cosine of projected visual tokens vs LLM input embeddings of extracted nouns, then diversity fill; CLS-attention fallback (§III-B/C, p.3); up to 99.5% retention (p.1). ≈ the proposal's "text-sim" selector. |
| LoopLook (ext.) | Loops ViT layers with question FiLM + learned exit, LLM frozen; **fine-grained tasks get more encoder loops** (V* 2.62 vs POPE 1.28, Fig.5); Qwen2.5-VL-3B avg +4.6 (Table 1). |
| AdaLoop (ext.) | Loops frozen ViT span with question AdaLN, deep supervision, answer-stability halting; LM LoRA-trained; Qwen ID 60.0→64.6 over single-pass (Table 1). |
| LoopVL (ext.) | 1B, two 16-layer shared stacks, H2L3 schedule; **each schedule trained separately; off-schedule inference collapses** (MMStar H2L3 63.47 → H4L3 24.00; Table 4, §5.3). |

---

## 2. Novelty verdict (§9.1 contributions)

| Contribution | Verdict | Evidence |
|---|---|---|
| (1) k×R grid on one set of looped weights | **Partly taken** | SmartVL Fig.6 does a token×(non-recurrent) compute grid at dataset level and reports task-dependent optima. LoopZip compresses tokens in looped VLMs at a fixed schedule. Varloop card / LoopVL Table 4 sweep R at full tokens. No paper found crossing variable *weight-shared* R with k. Open only in that narrow form, and time-sensitive (model released Nov 20). |
| (2) H2 substitution (reasoning vs perception) | **Partly taken (dataset level) / open (item level)** | SmartVL Fig.6, Matryoshka Table 2 (dataset-level token bias), LoopLook Fig.5 and AdaLoop (perception-heavy tasks want more *encoder* loops). Open: item-level, non-circular labels, decoder recurrence specifically. |
| (3) Controller (H3) | **Taken** | Learned per-input k with frozen decoder (StepPrune); joint token+compute (SmartVL); RL budget/pruning (TPRL, GRIP-VLM, AdaptVision, VisionThink, AdaVSkip). Full-table training is weighted classification. |
| (4) Safety on both axes (H4) | **Partly taken** | Loop axis, text: Running Fewer Loops, SafeBridge. Token axis, VLM: Token-Pruning-Induced Vulnerabilities. Open: joint, multimodal, capability-conditioned, GUI injection. |

---

## 3. Three strongest reviewer objections

1. **"This is one checkpoint, and it cannot generalise."** Varloop was post-trained at R=3, and the card warns of unpredictable behaviour at other R. No second depth-elastic VLM exists in what I read: LoopVL collapses off-schedule (Table 4), and SenseNova-Loop runs a fixed 2× schedule (LoopZip §5.1, p.5). Any k×R surface confounds "looping" with this checkpoint's 1:2:6:2:1 training mix, and §9.5's own remedy (a second backbone) has no viable candidate.
2. **"The headroom is noise, and the controller gain will be within error bars."** A per-item oracle over 15 binary cells is an OR over noisy outcomes. Matryoshka's 5-scale oracle already adds about 16 points on MMMU (Table 2, p.6). Realised adaptive gains in the closest work are ≤0.9 pp (StepPrune Fig.3c), 0.2 pp (HQAP Table I) or negative (MVP App. Table 6). RL over supervised is ≤1.5 pp (GRIP Table 1). Against ±3-point CIs at n=1000, H1 can pass spuriously and H3's gap-closed ratio has an ill-conditioned denominator.
3. **"The cost model is wrong for this model."** Thinking adds 100–400 decode tokens and changes with k and R, so prefill FLOPs (§1.1) are not the cost. SmartVL itself names decode as unsolved. Post-projector pruning saves no ViT cost, and the ViT can dominate (AdaptMerge Table 2, p.7) at 7,676 tokens per image. With prelude+coda ≈2.4B against the core's 1.0B per pass, the 3×5 grid has few iso-cost pairs, so H2's iso-FLOP comparison has little to compare.

---

## 4. Corrections

**Proposal_v2**
- §2.2 novelty matrix: SmartVL "text-conditioned: Yes" is wrong for the token mask, which uses the image and budget only (§3.1). Only the compute gates see the question, indirectly.
- §2.3: "SmartVL: POPE prefers tokens, TextVQA prefers compute." Fig.6 shows TextVQA/VQAv2 move to *both* high (T=0.9, C=0.8). It comes from a manual grid ablation, not the learned controller.
- §2 omits StepPrune, LoopZip, TPRL, GRIP-VLM, AdaptMerge, LoopLook, AdaLoop and LoopVL. All of them bear directly on the claims.
- §1.3 frames the safety priors as opposite *across* axes. On the loop axis alone they conflict: SafeBridge reports attack success *rising* at deeper loops. Varloop's own card shows StrongREJECT at 98.8 at 1 pass against 98.2 at 3 (range 98.2–99.4 across R) and FORTRESS-adversarial flat at 89.0–90.4, so "fewer loops removes refusal" may not transfer to this model.
- Running Fewer Loops reports that per-request depth selection does not recover refusal. A (k,R) controller is a per-request depth selector, so this bears directly on H4.
- AdaVSkip "text-conditioned … GRPO": neither is in the abstract. Unverified.
- TL;DR contribution (3) "reward-trained controller" contradicts §9.1 ("weak novelty") and §13.2 (Dec method is not RL).
- §1.4 "loop count set via text_config — Verified": your Nov 24 notebook shows it was ignored.
- Venues (ICML/ICLR/NeurIPS) for Token-Pruning Vulnerabilities, VPI-Bench, LoopFormer, MoR and VisionThink are not stated on arXiv. Unverified.

**Earlier review (`docs/literature/*`)**
- It treats LoopVL as a candidate backbone "after validation". The paper already shows it is not depth-elastic (Table 4), so it is not usable for R sweeps.
- It never mentions SenseNova-Loop-9B (in LoopZip), the only other open looped VLM in the corpus.
- Page anchors are off by 1–2 pages in places: TPRL Table 1 is p.11 (review: p.9) and Table 5 is p.12 (review: p.11); StepPrune Table 1 is p.8 (review: p.7); TOP-RL Table 1 is p.6 (review: p.5).
- It misses SmartVL Fig.6, the closest existing result to the k×R grid.

---

## 5. Not verified

- External papers: read as HTML extractions, not PDFs. SmartVL has no table numbers in HTML. LoopVL §6–8 and appendices are unread. LoopLook and AdaLoop were read up to ~100k characters.
- The 10 cited papers: abstracts only. RR-VLM details (layers 4–9, ×6, 60B tokens, KL-exit, a sampled-depth variant) are unverified, which matters because §9.5 names RR-VLM as a second backbone.
- Whether SenseNova-Loop-9B supports variable loop counts: only LoopZip's description was read.
- Pass 2/3 papers: headline numbers confirmed on the cited pages, not full reads.
- Not read: Literature_Review.md (beyond overlap), the Inkling-2-Mini article, the MiniCPM-V card, and the HF blog behind the text_config claim.
