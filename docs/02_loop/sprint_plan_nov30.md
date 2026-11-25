# AdaLoop-VLM — Sprint Plan to Nov 30

Tue Nov 24 → Mon Nov 30, 2026 · two parallel tracks · companion to `docs/02_loop/adaloop_execution_plan.md` (step IDs A1–C4 refer to it)

> **Calendar check:** Nov 26 is Thanksgiving. Nov 30 is a **Monday**, used for analysis and write-up. Code freeze is **Fri Nov 27, 12:00**. Weekend jobs launch **Fri evening**.

---

## 0. Scope for this sprint

| In | Cut or reduced |
|---|---|
| Phase A complete on 3 benchmarks (GQA, TextVQA, ChartQA), 500 test items each | MMMU only if throughput allows |
| A3 feasibility result (does look-again work?) | — |
| Per-pass trajectories + learned controller (B4–B5) vs baselines (B6) | Full LoopZip → **LoopZip-lite** (cross-pass selection, no value correction), labelled as such |
| **C1** LoRA so the model accepts tokens added mid-loop (distillation, no generation) | **C3 joint GRPO → pilot only** (generation is too slow for a weekend) |
| **C2** text-conditioned encoder adapter — only if code passes smoke test by Fri 12:00 | Safety grid → 100-item subset, only if replicas are idle |

**Accepted risk:** training starts before Gate G2 (headroom) is known. If G2 fails on Monday, the controller results become a negative result, not a loss of work.

---

## 1. Who owns what

```mermaid
flowchart LR
  subgraph INF["Inference team"]
    A1[A1 load + call counts] --> A2[A2 thinking rule] --> A3[A3 feasibility] --> A4[A4 eval pipeline] --> A5[A5 loop sweep] --> A6[A6 selector] --> GRID[Weekend: grid + trajectories] --> A7[A7 analysis]
  end
  subgraph TRN["Training (Vedaang)"]
    T1[T1 controller on mock data] --> T2[T2 LoRA insertion distill] --> T3[T3 encoder adapter] --> W[Weekend: train controller + LoRA]
  end
  A3 -. G1 result .-> T2
  A4 -. schemas + hooks .-> T1
  GRID -. trajectories.parquet .-> W
```

| Track | Owner | Steps | Output contract |
|---|---|---|---|
| Inference | Team | A1–A7, B1 (features), B2 (actions), B3 (heuristic), B6 baselines | `runs.parquet`, `trajectories.parquet`, `features/*.pt` |
| Training | Vedaang | B4 reader, B5 controller, C1 LoRA, C2 adapter (optional), C3 pilot | `ckpt/*`, `controller_results.parquet` |
| Shared | Both | `adaloop/hooks.py`, `adaloop/actions.py`, `configs/` | Owned by inference; training imports only, does not edit |

---

## 2. Interfaces — agree on these **tonight** (Tue)

Both tracks build against these schemas. Changing them after Wed noon needs both sides to agree.

**`runs.parquet`** — one row per (item, k, R, policy)
```
item_id, benchmark, split, k, R, policy, selector, pred, correct,
n_vis_tokens, n_txt_tokens, n_think_tokens,
t_vit_ms, t_prefill_ms, t_decode_ms, peak_mem_gb, flops_vit, flops_prefill, flops_decode,
gpu_type, git_hash, config_hash
```

**`trajectories.parquet`** — one row per (item, action path)
```
item_id, benchmark, split, path            # e.g. "L,L,X" or "L,A,L,X"  (L=loop, A=look again, X=exit)
pass_exit, n_look, correct, cost_flops, cost_ms
```

**`features/<item_id>.pt`** — per-pass controller features (B1)
```
{ "pass": [1..R_max], "h_last": [R_max, d], "delta_norm": [R_max], "kl_prev": [R_max],
  "entropy": [R_max], "vis_attn_mass": [R_max] }
```

**Splits:** `dev` (200/benchmark: selector + thresholds), `train` (controller + LoRA), `test` (500/benchmark: never touched until Monday).

---

## 3. Hardware allocation

All GPUs run **BF16** (NVFP4/MXFP8 kernels are Blackwell-only; A40, A100 and H200 do not have them). The BF16 model is ~53 GB.

| GPU | Count | Fits model on | Role |
|---|---|---|---|
| H200 (141 GB) | `<N_H200>` | 1 GPU | **Training:** C1 LoRA (fits model + LoRA + activations on one GPU) |
| A100 80 GB | `<N_A100>` | 1 GPU (27 GB left for KV/activations) | Inference replicas: grid + trajectories |
| A100 40 GB | — | 2 GPUs | Inference replicas |
| A40 (48 GB) | 4 | 2 GPUs → 2 replicas | Inference replicas **+ all latency measurements** |

- **Latency is measured on A40 only**, so the cost numbers are comparable. Correctness can come from any GPU, but log `gpu_type` and spot-check 50 items across types for agreement.
- If no H200: C1 runs on 2× A100 80 GB with FSDP; inference loses one replica.

---

## 4. Day by day

### Tue Nov 24 (tonight)
| Inference team | Vedaang |
|---|---|
| Agree schemas (§2). Set up env + model weights on every node (A40, A100, H200). | Agree schemas. Write `configs/` structure and `adaloop/controller/` skeleton. |

### Wed Nov 25 — feasibility day
| Inference team | Vedaang (T1) |
|---|---|
| **A1:** per-layer call counts (`6 + 12R + 6`), runtime R control, hooks for per-pass hidden states. | Controller code on a **synthetic** trajectories table: dataset, 2-layer MLP, three objectives (supervised cheapest-correct, full-information expected reward, λ sweep), eval at matched cost. Unit tests. |
| **A2:** can thinking be skipped via the chat template? Decide by noon. | LoRA scaffold for C1: student = mid-loop insertion, teacher = full-token run, loss = KL on answer logits. |
| **A3:** early-readout equivalence, drop-after-pass-2, add-after-pass-2 (insert vs restart), 50 items each. | — |
| **Gate G1 by Wed night:** insert / restart / drop-only. | Receives G1 → decides whether C1 is needed. |

### Thu Nov 26 — pipeline day (Thanksgiving: lighter)
| Inference team | Vedaang (T2) |
|---|---|
| **A4:** loaders (GQA, TextVQA, ChartQA), scorers, cost logging, `runs.parquet` writer. Nondeterminism re-run on 50 items. | Plug controller into a **real 50-item trajectory pilot** (from inference team). |
| **A5 milestone:** 200 GQA × R = 1..5 at k = 100%. | **C1 smoke test** on H200: 50 items, 100 steps, loss goes down, insert-mode accuracy moves toward restart-mode. |
| **A6:** selectors on dev (random, uniform, CRISP-style, resolution); freeze one. **Keep original positions.** | Start **C2** adapter code (FiLM at ViT layer ℓ + token scorer + distillation). Drop it if not working by Fri 12:00. |
| **Measure throughput** (s/item per GPU type) → fills §5 budget. | — |

### Fri Nov 27 — freeze and launch
| Time | Inference team | Vedaang |
|---|---|---|
| By 12:00 | **Code freeze.** B1 features, B2 actions, B3 heuristic, LoopZip-lite. | Code freeze. C1 config final. C2 go/no-go. |
| 12:00–17:00 | **Dry run:** every weekend job on 20 items end-to-end, outputs validated against §2 schemas. | Dry run: controller trains on the dry-run trajectories; C1 runs 50 steps. |
| Evening | **Launch** grid + trajectories on all inference replicas. | **Launch** C1 on H200 (and C2 if go). |

### Sat Nov 28 — run and monitor
| Inference team | Vedaang |
|---|---|
| Watch jobs every few hours; restart failures; merge parquet shards. | Train controller on whatever trajectories have landed (cheap: minutes on 1 GPU). Watch C1. |
| Start **A7** on finished benchmarks. | C3 GRPO **pilot** only if a GPU is idle (≤ 50 prompts, to estimate cost for January). |

### Sun Nov 29 — results
| Inference team | Vedaang |
|---|---|
| Grid complete. **A7:** oracle per cost cap, noise ceiling, evidence-retention analysis. **Gate G2.** | Final controller training on full `train` split; λ sweep. |
| Run B6 baselines + B3 heuristic on `test`. | Evaluate C1 (insert-mode with LoRA vs restart vs no-LoRA). Hand controller to inference team for `test` runs. |

### Mon Nov 30 — analysis and write-up
- Headline table: controller vs every B6 baseline at matched measured cost, 3 benchmarks, bootstrap CIs.
- Figures: iso-cost plot, accuracy surface, action histogram, evidence-retention result.
- One-page decision: what goes to January (C2, C3, safety, second benchmark set).

---

## 5. Weekend compute budget (fill in Thursday)

```
runs_grid   = items × benchmarks × 15 cells          (500 × 3 × 15 = 22,500)
runs_traj   = train_items × benchmarks × leaves/item (e.g. 300 × 3 × ~10)
hours       = (runs_grid + runs_traj) × <s_per_item> / 3600 / <N_REPLICAS>
available   ≈ 60 h (Fri 20:00 → Mon 08:00)
```

With thinking on and HF eager (~0.13 s/token, 100–400 tokens), one item ≈ 13–52 s. **22,500 runs ≈ 80–325 replica-hours** — it does not fit unless something gives. Levers, in order:

1. **Skip thinking (A2).** Biggest lever: cuts decode to a few tokens.
2. **Early-readout trick (A3 test 1).** If it passes, one R = 5 run gives all 5 R values → grid shrinks 5×.
3. **Serve with SGLang / vLLM** if they accept per-request loop count.
4. **Cut**, in this order: MMMU → items to 300 → drop k = 50%.

Decide the cut on **Thursday night** from measured s/item, not on Sunday.

---

## 6. Risks and fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| Look-again fails (A3) | Wed night | Scope = exit + drop-only; C1 becomes the only route to look-again (January) |
| Grid does not fit the weekend | Thu throughput | Apply §5 cuts Thursday night |
| Thinking cannot be disabled | Wed noon | Cap at `<T_MAX>`, log thinking tokens as a cost axis; grid uses the early-readout trick where possible |
| Mixed GPU types change outputs | Thu 50-item spot check | Pin each benchmark to one GPU type |
| Schema drift between tracks | Fri dry run | Schema in §2 is the contract; dry run validates it |
| Holiday availability | Thu | Thursday tasks are front-loaded on Wed; Fri morning is buffer |
| No H200 | Tue | C1 on 2× A100 80 GB with FSDP; one fewer inference replica |

---

## 7. Definition of done (Mon Nov 30)

- [ ] G1 result written up (insert / restart / drop-only).
- [ ] 15-cell grid on 3 benchmarks × 500 test items, with cost logs.
- [ ] Oracle + noise ceiling + evidence-retention analysis (G2).
- [ ] Per-pass controller trained and evaluated against fixed, upfront 15-way, tokens-only, early-exit-only, LoopZip-lite, heuristic.
- [ ] C1 LoRA result (insertion with vs without LoRA).
- [ ] C2 and C3: status note (done / pilot / moved to January).
