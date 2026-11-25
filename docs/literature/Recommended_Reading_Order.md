# Recommended reading order

**Hand-off rule after receiving the proposal:** give Claude `Project_Proposal_Reference.txt`, `Research_Strategy.md`, and `Research_Gap_Analysis.md` **first**. For the December 3 Version B work, attach the five Pass-1 P0 papers relevant to token×loop novelty **plus SparseVLM [050] and CRISP [052] from P1 when selecting the frozen text-guided pruner**; add the other P0 papers only when discussing their methods. The 15-paper P0 set covers the *full research arc*, including January Version C, and is **not** a request to preload 15 long PDFs into one context. The 12 P1 PDFs are optional implementation/baseline references; the CSV indexes all 77 files. Each P0 appears once below.

## Pass 1 — try to disprove the idea first

1. **StepPrune** `[059]` — learned question-conditioned STOP, variable K, frozen vision/projector/decoder. This destroys the broad “frozen-decoder question-aware adaptive count” claim. Read §§3–4 and matched-mean Fig.3 (PDF pp.3–10).
2. **LoopZip** `[038]` — already compresses actual looped-VLM visual states using instruction attention across shared-layer calls. Read Algorithm 1/Eq.4 and Tables 1–3 (PDF pp.2–8).
3. **TPRL** `[061]` — query-guided PPO pruning and score/compression reward. Read §§3.1–3.5, Table 5 frozen-LLM ablation (PDF pp.4–13).
4. **AdaptMerge** `[007]` — frozen, training-free language-guided variable-count merging. Read §§3–4 and AI2D accuracy/latency (PDF pp.3–8).
5. **GRIP-VLM** `[053]` — frozen-backbone GRPO scorer and budget curriculum, but externally supplied K. Read §4 Eqs.4–9 and Tables 1–3 (PDF pp.8–12).

**Gate:** If the planned pitch is merely “query-aware token count plus GRPO with a frozen model,” stop here and reframe it. The external *SmartVL* paper adds a strong joint visual-token/decoder-compute collision; see the primary-source citations in `Research_Gap_Analysis.md`.

## Pass 2 — architecture and frozen-backbone baselines

6. **InstructBLIP** `[019]` — instruction-aware connector with frozen vision and language. Read §2.2–2.6 and Table 2 (PDF pp.4–7).
7. **Matryoshka Multimodal Models** `[013]` — one multi-resolution model, but no deployed automatic budget selector and full LLM fine-tuning. Read §3, Tables 1–2, Table 7 (PDF pp.4–9).
8. **LLaVA-Mini** `[027]` — fixed one-token pre-fusion alternative; read §4, Tables 1–3, Appendix B training stages (PDF pp.5–9 and appendix).
9. **InternVL3.5** `[043]` — consistency training followed by a separate frozen-backbone visual router; read §§2.1–2.3 and Tables 15–18 (PDF pp.3–7,22–24).
10. **RecursiveVLM** `[039]` — distinguish actual repeated decoder passes, feedback state, stepwise losses, and no learned halt. Read §§3–4 and Tables 1,5,6 (PDF pp.3–8).

**Gate:** verify that `thinkingmachines/TML-d24-varloop-research` actually loads and varies R on the available A40s, then build the 3×5 grid and full-information controller **without** backpropagation through the backbone. The supplied TML variable-loop model card and the November 22 proposal are context, not PDF papers. Frozen-decoder gradients matter only for later Version C; decoder LoRA in January is a distinct non-frozen condition.

## Pass 3 — benchmark design and edge cases

11. **AdaptVision** `[051]` — question-aware RL visual acquisition and real crop/tool overhead. Read §§4–5 and Table 1/Fig.4 (PDF pp.3–7).
12. **Glance-ETD** `[054]` — supervised question evidence, extreme-compression OCR caveats, frozen-scorer versus decoder-trained headline; read §3, Table 1 and supplement S11/S15 (PDF pp.3–7,23–26).
13. **Hierarchical Query-Aware Pruning** `[055]` — separate image-dependent total K from question-dependent *allocation*; read Eqs.1–8 and Table I (PDF pp.2–5).
14. **MVP** `[056]` — question-conditioned frozen-VLM pruner; fixed-K versus threshold-variable-K failure illustrates calibration need. Read §4, Tables 1–2 and Appendix Table 6 (PDF pp.4–8,13–14).
15. **TOP-RL** `[060]` — PPO in decoder layers, task-transfer weakness, and incomplete supplied appendix. Read method and Tables 1–4 (PDF pp.3–7); do not infer missing training settings.

## Pull from P1 only as needed (12 total)

- **Selector mechanics:** DynamicViT `[002]`, Patch Ranking `[005]`, PruMerge `[028]`, ReDiPrune `[057]`, CRISP `[052]`.
- **Token retention vs merging and baseline strength:** ToMe `[004]`, SparseVLM `[050]`, *Rescuing Your Pruning* `[058]`.
- **Frozen connector and placement/encoder compute:** BLIP-2 `[018]`, Cambrian-1 `[021]`, LLaVA-UHD v4 `[033]`, *Unbundling Multimodal Vision Encoding* `[036]`.

## Optional topical appendices, not part of the main reading packet

- **Robustness/safety:** *MM-SafetyBench* `[063]`, visual adversarial examples `[062]`, *FigStep* `[064]` are P2 for a later safety study, not primary token-compression methods (source PDF pp.8–15,4–9,3–12 respectively).
- **Foundational:** ViT `[000]`, CLIP `[001]`, SigLIP `[003]` remain P2; read only to resolve implementation details, not to establish adaptive-token novelty.
- **External papers absent from the supplied PDFs:** *SmartVL* (joint token and model compute), *LoopLook* (frozen-decoder recurrent vision with learned halt), *AdaLoop* (question-conditioned recurrent vision), and *LoopVL* (candidate actual looped backbone). Bibliographic primary links and the precise boundary are in `Research_Gap_Analysis.md`; obtain their PDFs before any claim about the broader field.
