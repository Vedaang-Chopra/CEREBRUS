# S8 design review — teacher data for budget-tolerance distillation

Status: **proposal for Vedaang's approval** (Nov 26, 2026). Nothing is generated or trained until approved.
Sources: three literature/data reviews (Nov 26), the S8 code on main (`adaloop/core/train/teacher.py`,
`orchestration/teacher_cache.py`), the nb05/nb07 results. Claims from papers are cited by section; items marked
*(inference)* are our reasoning, not a published result.

## 0. Summary of what changes

| Topic | Current S8 plan | Proposal | Why |
|---|---|---|---|
| Training data | `lora_train` carved from GQA testdev + TextVQA val (8,549 q) | **Official train splits**: GQA train_balanced + TextVQA train, ~5k q each, stratified, 1–2 q per image | GQA `lora_train` has **198 images** for 6,056 questions; training on eval splits is non-standard |
| Teacher distribution | Cached offline, top-32 + residual, taken from generation logits | **Online teacher** (same model, LoRA disabled, full vocabulary, exact KL) for training; S8 caches answer trajectories + diagnostics only | Answers are 2–6 tokens: one extra forward per step removes the K question, supports any teacher R and on-policy later (TBD §3.1 does exactly this) |
| Fallback if online is too slow | — | Offline top-K + residual, K chosen by the pilot | Pilot measures both |
| Supervised sequence | Teacher-specific greedy answer per R | **Main: ground-truth answer trajectory (same for A and B)**; teacher-specific trajectory as ablation | Removes the sequence-selection confound between A and B |
| Teacher scoring | Logits of the KV-cached generation | **Teacher-forced forward**, the same path as the student | Cache = live by construction |
| Wrong teacher answers | Kept | Kept for the KL (no depth-dependent filtering); imitation and accuracy reported separately; correctness mask = optional ablation | Your rule; TBD shows filtering adds ~0.1 pt |

## 1. Dataset quality and composition (audit, Nov 26)

Current `lora_train` (eval-carved, image-disjoint from test / ctrl_train / dev — verified 0 shared images):

| | GQA | TextVQA |
|---|---|---|
| questions / images | 6,056 / **198** | 2,493 / 1,582 |
| answer length | 1.06 words, 4.7 chars | 1.56 words, 8.5 chars |
| skew | yes/no 35%, top answer 18%, yes+no+left+right ≈ 41% | flat prior; 4% placeholder answers ("unanswerable", "answering does not require reading text") |
| skills | relations 43%, attributes 41%, object/category 15%, logical+compare 18%; **no counting** | OCR (answer in OCR tokens 47%); "how many" 1.9% |
| issues | 4 duplicate (image, question) pairs, 1 with conflicting answers | 9% of items have ≤ 3/10 annotator agreement |

Problems: 198 GQA images is too little visual diversity for a vision-compression LoRA (it can memorize images);
training on the benchmark's evaluation split is not the standard protocol even when image-disjoint.

Official train splits (verified loadable): GQA `train_balanced` 943k q / 72k images (not cached, ~10 GB images; a
subset can stream parquet shards); TextVQA `train` 34.6k q / 22k images (**cached on Inara**, 0 overlap with val images).

**Proposal**: `lora_train` = GQA train_balanced ~5,000 q on ~3,000–5,000 images + TextVQA train ~5,000 q, stratified:
GQA by structural × semantic type (floors for logical/compare; cap yes/no ≤ 35%, any single answer ≤ 5%); TextVQA by
answer type (number / brand-name / other) and answer-in-OCR, agreement ≥ 5/10, placeholders dropped; ≤ 2 questions per
image. ~10k questions is in the range of prior compression post-training (LT-OPD / Glance-ETD: 14k prompts).
ctrl_train / dev / test stay as they are (eval-carved, frozen). Counting is not covered: we state it as a limitation.
Cost: a new split builder for train sources (loaders already take `hf_split` / config names); GQA image subset download.

## 2. Teacher generation (multi-depth)

- Five execution configurations of one frozen model: k = 100%, R ∈ {1..5}, greedy, batch 1, 768 px, no thinking.
- Vision tower once per item, reused across R (exact: the vision tower does not depend on R). Intermediate loop
  states are **not** reused for answers: nb05 shows pass-p states equal the R=p run, but the coda and decoding need
  their own run (and our batch tests show the generation path itself is sensitive).
- Per (item, R): greedy answer ids (incl. end token), parse status, correctness, then a **teacher-forced** forward on
  (a) its own answer and (b) the ground-truth answer, giving per-position entropy, top-1 probability, margin,
  log-prob of the target token. With the online design, full distributions are not stored beyond the pilot.

## 3. Teacher reliability (measured in the pilot, reported per R and benchmark)

Accuracy per R; cross-R answer agreement; entropy / top-1 / margin; calibration (ECE of top-1 prob vs correct);
items where more loops help or hurt; items wrong at every R; answer length and parse rate. Note: the grid's k = 100%
cells (nb08) give teacher accuracy per R on dev/ctrl_train already; the pilot adds it on the training distribution.

## 4–5. KL objective and vocabulary truncation

Forward KL (teacher ‖ student) at temperature 1, fp32 log-softmax over the full vocabulary, gathered at the
teacher's support. Reasons: answers are 1–5 tokens and teacher-forced (no rollout, no exposure bias — *inference*);
reverse KL / on-policy matters for long generations (MiniLLM; GKD §4.1: divergence matters little under greedy).

| Variant | Exact? | Note |
|---|---|---|
| Full-vocab KL (online teacher) | yes | proposal for training |
| Top-K renormalized | biased (Sparse Logit Sampling §2.2: grad = (Σ_K t)·p − t, overconfident head) | not used |
| Top-K + residual bucket | proper divergence on the (K+1)-partition, lower bound of full KL; = LFM2 §3.3 decoupled loss at τ=1 *(our derivation)*; misses how the student spreads tail mass | offline fallback |
| Sampled logits (Sparse Logit Sampling §4, Gemma 3 §2.2) | unbiased (SLS) / biased (Gemma: no IS correction) | high variance on peaked short answers *(inference)* |

Storage (≈ 240k answer positions at 10k items × 5 R): top-K K=32 ≈ 46 MB, K=256 ≈ 0.4 GB; full vocab (~200k, bf16)
≈ 96 GB. The pilot keeps full vocab (≈ 3 GB) to measure every K.

Numerics: residual as `logsumexp` of the complement (already in `split_top`, avoids 1 − Σ rounding); clamp q_other
≥ 1e-12 in fp32; student probabilities gathered at the **teacher's** top-K ids.

## 6. Teacher–student alignment (correctness requirements)

- Same prompt (system + image + question + `ANSWER:` prefill) for teacher and student; only the visual tokens differ.
- Teacher-forced: feed target ids[:-1] after the prompt; the last n logit rows predict the n target ids (incl. end
  token) — as `answer_log_probs` already does; unit test with a shifted-by-one check.
- Mask: answer positions only; prompt and image positions never in the loss.
- Batch 1 throughout (batching changes outputs, decisions Nov 26), so no padding in training or teacher passes.
- Pruning changes only the image span; answer positions are counted from the end, so alignment is unaffected.
- **Trajectory choice**: main = ground-truth trajectory for every sample (identical for A and B, so the only
  difference is the teacher's depth); ablation = teacher-specific greedy trajectory (A: answer at R_s; B: answer at
  R=3). Without this control, A vs B mixes teacher depth with which answer string is supervised.
- Checks: cached = live (pilot), teacher with LoRA disabled = base model (bit-exact), loss = 0 at k=1 with a zero LoRA.

## 7. Offline vs on-policy

| | Offline (teacher-forced) | On-policy (student generates, teacher scores) |
|---|---|---|
| Evidence | ViCO (InternVL3.5 §2.3), TBD | Thinking Machines OPD blog; LT-OPD (+4 pts retention over offline at 5% tokens, long answers); Glance-ETD (JSD on top-100 + tail) |
| Exposure bias | none at train; small for 1–5-token answers *(inference)* | removed |
| Cost / step | student fwd+bwd + teacher fwd | + student decode (≤ 5 steps) |
| Storage | trajectories only (online teacher) | none |

One physical model serves both roles: `with peft_model.disable_adapter(), torch.no_grad():` teacher pass, then the
student pass with the adapter (TBD §3.1). Pitfalls: no checkpointing around the teacher pass; reset loop state / R
per call; teacher R set independently of student R. Plan: offline (teacher-forced, online teacher) is the S9
baseline; the code exposes `teacher_logits(model, inputs, target_ids, R, adapter=False)` and `student_generate(...)`
so on-policy is a later switch, not a rewrite.

## 8. Closest prior work (verified IDs)

ViCO (InternVL3.5, 2508.18265): forward KL full-token → compressed, teacher-forced on SFT data. TBD (2608.28138):
same model with LoRA off as teacher, CE + KL on correct-and-confident positions, gains over LoRA+CE small (58.3 →
58.9). LT-OPD (2609.32353) and Glance-ETD (2611.20006): on-policy self-distillation from a full-token copy, 14k
prompts. GRIP-VLM (2605.13375) and StepPrune (2609.13804) train only a selector (not comparable). Looped papers
(LoopOPD, LoopFormer, ELT, MELT) distill across depth, never with token budgets. Our contribution: the interaction of
distillation, visual compression and weight-shared depth (A vs B), not KL itself.

## 9. Schema

`items.parquet` (once per item): benchmark, item_id, image_id, image sha1, source split, question, question type
(GQA structural/semantic; TextVQA answer type, in-OCR), ground-truth answers, gt target ids.
`teacher.parquet` (per item × R): R, k=1.0, n_visual, pred, answer ids, parse_ok, correct, per-position
entropy / top1 / margin / logp(target) for own and gt trajectory, n_new, latency.
`pilot_logits/` (pilot only): full-vocab fp16 log-probs for own and gt trajectory.
Run metadata (once per run): model revision, tokenizer/processor versions, res_px, prompt hash, precision, generation
settings, config + git hash, code_version. Sharded by item, resumable, verified by a checksum per part.

## 10. Pilot (nb09 rewritten)

500 items (250 GQA + 250 TextVQA, stratified as in §1) × R 1..5 = 2,500 teacher records, 4 copies, batch 1.
Outputs: §3 reliability table; K study for K ∈ {16, 32, 64, 128, 256, full} — retained mass (mean, 5th pct, worst),
KL error vs full, logit-gradient cosine and norm ratio vs full, broken down by R, benchmark, correctness, position;
cached = live check; online-teacher step time vs offline; a 20-step backward pass for A and B (loss falls, grads reach
LoRA at R=1,3,5). Estimated time: generation + scoring ≈ 20–30 min on 4 copies; K study on CPU minutes; ~3 GB storage.

## 11. Acceptance criteria for scaling (and for S9)

1. Pilot cached = live: max |Δ log p| ≤ 1e-2 on target tokens; argmax equal.
2. Trajectory alignment tests pass; LoRA-off = base, bit-exact.
3. Teacher parse rate ≥ 98% at every R.
4. Online-teacher step ≤ 1.6× the student-only step, or else offline top-K with K = smallest K whose retained mass ≥
   0.99 at the 5th percentile, KL error < 1% and gradient cosine ≥ 0.99.
5. Backward smoke for A and B: loss falls over 20 steps, k=1 loss ≈ 0 at init.

## 12. Decisions for Vedaang

1. **Training data: official train splits** (recommended) or keep the eval-carved `lora_train` (state it in the paper)?
2. **Online teacher** (full vocab, no cache) as the main design, offline top-K as fallback?
3. **Supervised trajectory: ground truth** for both A and B (main) + teacher-specific as ablation?
4. B's teacher depth: R=3 (your spec) — confirmed?
5. Pilot of 500 items as specified, then the S9 smoke?

Mandatory for Dec 3: data switch (or documented keep), pilot §10, A and B with one trajectory choice. Optional: the
trajectory ablation, correctness-mask ablation, K study beyond the pilot. Future: on-policy, reverse KL / JSD,
counting data, thinking-mode teacher.
