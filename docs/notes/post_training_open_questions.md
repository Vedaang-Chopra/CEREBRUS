# Post-training design: open questions (for external review, Nov 26, 2026)

Self-contained brief. Please answer each question in §4 with a recommendation and a short reason, and flag anything in §1–3
that looks wrong.

## 1. Project in one paragraph
AdaLoop-VLM studies a **looped vision-language model** (TML-d24-varloop-research, Thinking Machines: 24-layer decoder,
sparse MoE (128 experts, 8 active), 26.4B total / 3.4B active parameters; the middle 12 layers (6–17) are shared and can
run R = 1..5 times at inference; trained at R = 1..5 in a 1:2:6:2:1 ratio during midtraining and post-trained at R = 3).
We cut visual tokens after the projector (DivPrune, keeping k = 25% or 50% of image tokens) and vary R. A controller later
decides per question: exit, loop again, or look again with more tokens. Goal of post-training: make the model **tolerate
25–50% visual tokens at every depth R = 1..5 while keeping full-token accuracy**. Deadline: Dec 3 (proof of concept + paper);
today is Nov 26.

## 2. Fixed facts and constraints
- Inference setup: no thinking, short answers (prefill `ANSWER:`), greedy, image res 768 px; eval benchmarks GQA and
  TextVQA (evaluation only, never trained on).
- Measured (dev, R = 3, 768 px, n = 100 per benchmark): full tokens 62.9 avg (GQA 57, TextVQA 68.8). At 512 px DivPrune
  k = 25% = 47.7 vs 57.4 full; TextVQA loses most. Full Pareto at 768 px is running now.
- Greedy answers change with batch shape (13/128 answers differ batch 8 vs batch 1, BF16 kernels): reported numbers all
  use one regime per comparison; latency at batch 1.
- Hardware: Inara 8× A40 46 GB (reliable; training + all reported eval). One BF16 model copy = 2 A40s (~28 GB each).
  PACE: 2× A100 40 GB + 1× A100 80 GB, 8 h job limit, not always available. Other servers usable for inference.
- Whole-model fine-tuning does not fit: 26.4B params → ~53 GB weights + ~53 GB grads + ~320 GB Adam/master + teacher
  ≈ 480 GB > 368 GB on Inara, and it would block all GPUs.

## 3. Current proposal (main-discussion, not yet approved)
- **Data:** ~12k items from HuggingFaceM4/FineVision (embedded images, source tags, per-turn quality ratings), 7 buckets:
  general VQA 2.5k, charts 2.5k, docs/tables 2k, OCR/scene text 2k, diagrams/science 1k, counting 1k, free-form 1k.
  Excluded: every subset whose images overlap GQA (Visual Genome) or TextVQA (OpenImages); dense docs too tall for 768 px.
  Filters: ratings ≥ 4, short answers ≤ 8 tokens (except free-form ≤ 64), image size limits, perceptual-hash dedupe vs eval
  images, keep items the full-token teacher answers correctly. Precedent: Glance-ETD (2611.20006) retains the same with
  700–7K samples as with 14K; LT-OPD (2609.32353) used 14K mixed prompts.
- **Teacher:** same frozen model, 100% tokens. Off-policy (train on the teacher's greedy answer, teacher-forced). Teacher
  distributions computed **offline** in one inference pass and stored (full vocabulary is affordable: ~40k answer
  positions). The same pass gives the correctness filter.
- **What is trained:** full-rank update of the attention projections (q/k/v/o) of the 12 shared loop layers; experts,
  routers, vision tower, projector frozen. Fallback: LoRA r = 16 on the same modules.
- **Loss:** forward KL(teacher ‖ student) on every answer position incl. end token + λ·CE(ground truth), λ = 0.5.
- **Per sample:** R and k fixed in advance (seeded, identical across runs); k ∈ {25%, 50%} for 90%, k = 100% for 10%.
- **Runs:** A depth-matched (teacher R = student R), B fixed teacher R = 3, C data-only control (same data, k = 100% only,
  to separate "learned the data/format" from "learned to tolerate pruning"). 1 epoch, lr 1e-4 (LoRA) / lower for
  full-rank, effective batch 16, BF16, gradient checkpointing.
- **Checks:** smoke run (loss falls, only trained params change, peak memory at R = 5); success = dev accuracy at k = 25%
  above base and above C; k = 100% drop ≤ 0.5 pts at every R; thinking-mode check on 20 items.

## 4. Questions
1. **Off-policy vs on-policy.** Answers are 1–3 tokens except ~8% free-form. Is off-policy teacher-forced KL enough, or
   is on-policy (student samples, teacher scores, reverse KL / JSD as in GKD, MiniLLM, Thinking Machines OPD) worth the
   cost here? Mixed (on-policy only for free-form)?
2. **What to train.** Full-rank attention of the loop layers vs LoRA vs also experts/routers. Risk of forgetting
   (thinking mode, long answers) and router drift with ~12k samples? Learning rate for full-rank attention?
3. **Loss.** Forward KL vs reverse KL vs JSD for 1–3-token answers; is λ·CE on ground truth helpful or harmful when the
   teacher is filtered to be correct (CE can shift answer style: "Net." vs "net")? λ value? KL only where the teacher is
   confident (Token-Budget Distillation 2608.28138)?
4. **Depth sampling.** Student R uniform over 1..5 vs the model's own 1:2:6:2:1 prior. The model card warns that
   R ≠ 3 may behave unpredictably; training at R = 1 and 5 may also move R = 3.
5. **Teacher depth.** A (teacher at the student's R) vs B (teacher always R = 3) vs deepest (R = 5, as in looped-model
   distillation papers LoopOPD, LoopFormer, ELT, MELT). Which is the most informative comparison for a 3-run budget?
6. **Data.** Is ~12k enough / too much? Should we oversample items whose answer flips at 25% tokens (the items that carry
   the signal) rather than a balanced mix? Keep or drop items the teacher gets wrong?
7. **Gradients through loops.** Backprop through all R passes of the shared layers (memory, instability at R = 5) vs
   truncated backprop through the last pass(es)?
8. **Evaluation power.** With ~100 dev items per benchmark the CI is about ±7 pts. Minimum eval size to separate A, B, C
   with one seed each? Second seed or more eval items first?
9. **Anything we are missing** for a 5-day timeline (tuning shortcuts, known failure modes of distillation under token
   pruning, MoE-specific issues).
