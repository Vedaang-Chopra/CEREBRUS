# Study brief: post-training (S8–S10) and the controller (S11–S14) of AdaLoop-VLM

Paste this into a chat assistant. I want help **learning the background** to drive these stages: what to read, the
core ideas, and small practice experiments on a ~1B model. Please teach step by step, correct anything below that
looks wrong, and connect every concept back to this project.

## 1. The project in one paragraph
AdaLoop-VLM studies a **looped vision-language model**: TML-d24-varloop-research (Thinking Machines). 24-layer
decoder, sparse MoE (128 experts, 8 active per token), 26.4B total / 3.4B active parameters. The **middle 12 layers
(6–17) are shared weights that can run R = 1..5 times** (R = number of passes, "depth"). Trained at R = 1..5 in a 1:2:6:2:1
ratio during midtraining, post-trained at R = 3. We also **cut visual tokens** after the vision projector (DivPrune,
keeping k = 25%, 50% or 100% of image tokens). So every question can be answered at 15 settings (3 k × 5 R), each with
its own cost (FLOPs) and accuracy. Goal: answer each question **as cheaply as possible without losing accuracy**.
Eval: GQA and TextVQA (short answers, greedy, no thinking, image 768 px). Deadline Dec 3 (proof of concept + course
paper).

## 2. Stages done (S0–S7)
- Model loading, loop control (R changes the logits), pruning methods compared (DivPrune won), resolution chosen.
- **S7 grid:** every dev and ctrl_train item run at all 15 (k, R) cells; for each item and cell we store correct/wrong,
  FLOPs, latency, and per-pass internal signals ("features", §4.2). This grid is the dataset for everything below.
- Example numbers (dev, R = 3, 768 px): full tokens 62.9% (GQA 57, TextVQA 68.8). At 25% tokens TextVQA drops most.

## 3. Post-training (S8–S10), after Dec 3
**Aim:** make the model **tolerate 25–50% visual tokens at every depth R** while keeping full-token accuracy.

- **S8, teacher:** the same frozen model with 100% tokens produces answers and token distributions (logits) on ~12k
  training items (from HuggingFaceM4/FineVision; never GQA/TextVQA). Stored offline.
- **S9, student training:** the same model, but seeing pruned tokens (k = 25/50%) at a sampled R, learns to match the
  teacher. Loss: forward KL(teacher ‖ student) on the answer tokens (+ maybe a small cross-entropy on the gold answer).
  This is **knowledge distillation / self-distillation**, not RL. Variants: A teacher at the same R as the student;
  B teacher always at R = 3; C data-only control (train at k = 100%, to separate "learned the format" from "learned to
  handle pruning").
- **What gets trained:** the whole model does not fit in memory (~480 GB needed vs 368 GB available). Options: a
  full-rank update of the attention projections (q/k/v/o) in the 12 shared layers, or LoRA on the same modules. Experts,
  routers, vision tower frozen.
- **S10:** rerun the 15-cell grid with each trained model; compare to the base.

**My open questions** (want to understand these deeply):
1. Off-policy (teacher-forced on the teacher's answer) vs on-policy distillation (student samples, teacher scores;
   GKD, MiniLLM, on-policy distillation). Does it matter for 1–3-token answers?
2. Forward KL vs reverse KL vs JSD; when to add cross-entropy on gold answers.
3. Training a weight-shared looped block: backprop through all R passes vs truncated; stability at R = 5.
4. Full-rank attention vs LoRA vs training experts/routers; forgetting and router drift in MoE fine-tuning.
5. Could RL (GRPO, DPO, PPO-style) help here, e.g. reward = correct at low k? When is RL better than distillation for
   short-answer VQA?

## 4. Controller (S11–S14), the Dec 3 target (on the frozen model)
**Aim:** a small network that decides **per question, between loop passes**, whether to stop or spend more compute.

### 4.1 How one question is answered
1. Start cheap: k0 = 25% (or 50%) of image tokens.
2. After each pass p of the shared block, the controller looks at features and picks one action:
   - **EXIT:** answer now (read out at depth p).
   - **LOOP:** run one more pass (not allowed at p = 5).
   - **LOOK:** restart with 100% of image tokens (at most once; then only EXIT/LOOP).
3. Cost = FLOPs of everything executed, including the discarded cheap run if it looked again.

### 4.2 Controller inputs (per pass)
- `h_last`: hidden state of the last token after this pass.
- `delta_norm`: ||h_p − h_{p−1}|| / ||h_{p−1}|| (is the state still changing?).
- `kl_prev`: KL between the answer distribution at pass p and pass p−1 (did the answer change?).
- `entropy`, `top1_prob`: confidence of the current answer.
- `vis_attn_mass`: how much attention goes to image tokens.
- Plus: kept-token fraction, compute spent so far, pass number. Reading features also costs FLOPs, and we charge it.

### 4.3 Model and training
- Policy: MLP [features] → 512 → 512 → 3 actions, with masks for forbidden actions.
- **Training is offline and full-information:** the S7 grid tells us the outcome (correct?, cost) of *every* path for
  every training item. So we build a **trajectory tree** per item (S11: 25 nodes), and compute exact values by
  backward recursion: Q(node, action), V(node) = max over allowed actions.
- Reward of a finished path: `correct − λ · cost / cost_ref` (cost_ref = cost at k = 100%, R = 3). Sweeping λ traces an
  accuracy-vs-cost curve.
- Two objectives: (a) supervised: cross-entropy to the best action (argmax Q); (b) expected reward: maximize
  Σ_nodes Σ_a π(a | features) · Q(node, a) plus a small entropy bonus. No sampling is needed, because Q is known exactly.
- Also a hand-tuned **heuristic**: thresholds on top1_prob, kl_prev, entropy, vis_attn_mass.

### 4.4 Baselines and the claim (S13–S14)
- **best_fixed:** the best mix of fixed (k, R) cells at the same average cost (the fair baseline). Also the single best cell.
- early_exit_only (only EXIT/LOOP at 100% tokens), tokens_only (choose k up front), upfront_15way (choose 1 of 15 cells
  before running anything), oracle (perfect per-item choice), random.
- **Headline:** accuracy at half the cost of (k = 100%, R = 3). Claim if controller − best_fixed ≥ 1 pt, with a 95%
  paired-bootstrap CI excluding 0 (Holm correction across baselines).
- **Headroom gate:** oracle − best_fixed − noise ≥ 3 pts on at least one benchmark, else adaptivity cannot help.

### 4.5 My intuition to check
"The controller looks like a mixture-of-experts gate." Please compare it carefully with:
- **MoE routers** (per token, inside a layer, choosing experts, trained jointly with the model, load-balancing loss);
- **adaptive computation / halting**: ACT (Graves 2016), PonderNet, Universal Transformers, early-exit networks
  (BranchyNet, CALM, DeeBERT), Mixture-of-Depths;
- **model cascades / routing between models** (FrugalGPT, RouteLLM) and **learning to defer**;
- **contextual bandits / offline RL** (we know every outcome, so is this plain cost-sensitive classification?);
- **adaptive visual token budgets** in VLMs (asking for more image detail when unsure).
Where is it similar, where is it different, and what does each field teach for designing our controller?

**Controller questions:**
1. Is the expected-reward objective on full trees the right choice vs supervised argmax, or vs a policy-gradient method?
2. With only ~600 training items (300 per benchmark), how do we avoid overfitting a 512-wide MLP on a 6k-dim hidden state?
3. Which features are known to predict "the answer will not change with more compute" (calibration, confidence)?
4. How do papers report adaptive-compute results fairly (iso-cost curves, convex hull of fixed baselines)?

## 5. What I want from you
1. **Reading list** for each block (post-training, controller), 5–8 items each, most important first, with one line on
   why each matters to this project. Mark papers vs blog posts vs code.
2. **Concept lessons** in order, short and with small math: distillation losses (KL directions, temperature,
   top-k teacher), LoRA vs full fine-tuning, on-policy distillation, GRPO/DPO/PPO basics, adaptive computation and
   halting, offline policy learning from full-information trees, iso-cost evaluation.
3. **Hands-on practice** I can do in a few days on one GPU with a ~0.5–1.5B model (for example Qwen2.5-0.5B/1.5B-Instruct
   or a small VLM such as SmolVLM): 
   - distill a bigger teacher into a small student (TRL `GKDTrainer` or a hand-written KL loop);
   - LoRA vs full fine-tune on a small dataset; watch forgetting;
   - a small GRPO run (TRL `GRPOTrainer`) on a verifiable task (e.g. GSM8K-style math or short-answer VQA);
   - an early-exit toy: run a small model with exits at several layers, record per-layer confidence/answers, then
     train a tiny exit controller with the same tree + backward-recursion objective as §4.3, and compare to
     fixed-depth baselines at equal cost.
   For each: dataset, steps, expected runtime, what to plot, and what result would teach me something.
4. Tell me what I am missing or misunderstanding.
