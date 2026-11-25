# QAFE — Low-Level Build Spec (agent-ready)

v1 · Nov 24, 2026 · companion to `docs/03_encoder/qafe_spec.md` (the why and the phases)
Follow the steps **in order**. Each step lists: **Goal · Depends on · Do · Interface · Tests · Output · Feeds decision**. Do not start a step until its dependencies' tests pass.

Placeholders: `<REPO>`, `<MODEL_DIR>`, `<DATA_DIR>`, `<RESULTS_DIR>`, `<CKPT_DIR>`, `<HOST>` ∈ {`varloop`, `qwen3vl`}. Thresholds in `<ANGLE_BRACKETS>` go in `configs/prereg.yaml` **before** any test-split run.

---

## 0. Conventions for the agent

1. **Package:** `<REPO>/qafe/`, a sibling of `<REPO>/adaloop/`. QAFE never edits `adaloop/`; AdaLoop imports QAFE through `qafe/api.py` only.
2. **Layers** (strict, top to bottom):
   - `scripts/` and `notebooks/` (entry points, inspection only)
   - `qafe/api.py` (public entry points)
   - `qafe/orchestration/` (experiments, trainers)
   - `qafe/core/` (modules)
   - `qafe/schemas/` + `qafe/utils/`
   
   Notebooks call only `api.py`. No logic in notebooks.
3. **Identity rule:** every QAFE module must be bypassable. With QAFE disabled, or with all regions forced fine and gates at zero, outputs must be **token-exact** equal to the unmodified host on 20 items.
4. **Determinism:** greedy decoding, fixed seeds, short-answer mode (thinking skipped or capped per AdaLoop decision A2).
5. **Logging:** every evaluation row is written to `qafe_runs.parquet` (§2.3). Every decision goes in `<RESULTS_DIR>/qafe/decisions.md`.
6. **Never touch the test split** until Gate Q3.

---

## 1. Repository layout

```
<REPO>/
  qafe/
    api.py                      # build_encoder(), encode(), upgrade(), zoom(), run_experiment()
    schemas/
      items.py                  # Item, EvidenceBox
      tokens.py                 # VisualTokenSet, RouteMap, StemCache
      runs.py                   # parquet schemas + validators
    hosts/
      base.py                   # HostAdapter protocol
      varloop.py                # VarloopHost
      qwen3vl.py                # Qwen3VLHost
    core/
      grid.py                   # patch/unit/block index math, positions
      question.py               # QuestionPooler, NullQuestion
      condition.py              # AdaLNModulator (zero-init), QAViTConcat (ablation)
      router.py                 # Router, budget FiLM, gumbel/hard decisions, budget enforcement
      merge.py                  # QuadtreeMerger: build coarse virtual patch groups
      global_tokens.py          # GlobalPool, ThumbnailGlobal (ablation)
      scale_embed.py            # ScaleEmbedding (zero-init)
      encoder.py                # QAFEEncoder: assembles everything
      selectors.py              # emulation selectors for D1–D3: uniform, crisp, random, oracle_box
      emulate.py                # post-projector merge emulation (D2)
    orchestration/
      exp_d1_foveation.py
      exp_d2_tolerance.py
      exp_d3_noise.py
      exp_d4_probe.py
      teacher_cache.py          # cache full-token teacher logits
      labels_lossratio.py       # ViR-style router labels
      train_distill.py
      train_sft.py
      train_router_sup.py
      train_grpo.py
      bakeoff.py                # D6 driver: same setup, all objectives
      ablations.py
      eval_qafe.py              # matched-budget evaluation + baselines
    data/
      loaders.py                # benchmark + training loaders → Item
      evidence.py               # box extraction per dataset, evidence-area ratio
      strata.py                 # stratum assignment
    utils/
      timing.py  stats.py  io.py  seeds.py
  configs/qafe/
    base.yaml  prereg.yaml  hosts/{varloop,qwen3vl}.yaml  exp/*.yaml  train/*.yaml
  tests/qafe/
    test_grid.py test_identity.py test_router_budget.py test_merge_shapes.py
    test_positions.py test_schemas.py test_teacher_cache.py test_upgrade.py
  notebooks/qafe/
    01_host_inspect.ipynb  02_d1_d3_results.ipynb  03_probe.ipynb  04_bakeoff.ipynb
```

---

## 2. Data contracts

### 2.1 Items (`qafe/schemas/items.py`)
```python
@dataclass
class EvidenceBox:
    x0: float; y0: float; x1: float; y1: float      # pixels, original image
    source: str                                     # "textvqa_ocr" | "vstar" | "gqa_sg" | "viscot" | "refcoco"

@dataclass
class Item:
    item_id: str; benchmark: str; split: str        # dev | train | test
    image_path: str; image_hw: tuple[int, int]
    question: str; answers: list[str]
    evidence: list[EvidenceBox] | None
    evidence_area_ratio: float | None               # union box area / image area
    stratum: str                                    # small_target | general | halluc | reasoning | grounding
```
Item manifests are stored as `<DATA_DIR>/qafe/manifests/<benchmark>_<split>.jsonl`, one `Item` per line. Split IDs are frozen and hashed into `configs/qafe/splits.lock`.

### 2.2 Tokens (`qafe/schemas/tokens.py`)
```python
@dataclass
class RouteMap:                    # per image
    unit_hw: tuple[int, int]       # merge-unit grid (Hu, Wu)
    level: Tensor                  # [Hu, Wu] int8 in {0: fine, 1: 2x2-merged, 2: 4x4-merged}
    block_id: Tensor               # [Hu, Wu] int32, id of the output token covering each unit
    scores: Tensor | None          # [Hu, Wu] router relevance (for logging / upgrade)

@dataclass
class StemCache:                   # saved for upgrade() in the loop
    layer: int                     # ℓ
    feats: Tensor                  # [Hp*Wp, d_vit] patch features at layer ℓ (all patches)
    pos: Tensor                    # [Hp*Wp, 2]
    q: Tensor | None               # pooled question vector used

@dataclass
class VisualTokenSet:              # what the decoder receives (superset of adaloop VisualTokens)
    feats: Tensor                  # [N_out, d_dec] after host merge + projector
    pos: Tensor                    # [N_out, 2] float (row, col) in unit coordinates (block centre)
    level: Tensor                  # [N_out] int8 (-1 = global token)
    is_global: Tensor              # [N_out] bool
    route: RouteMap
    stem: StemCache | None
    n_vit_tokens: int              # tokens processed by ViT layers ≥ ℓ (cost)
```

### 2.3 Run rows (`qafe_runs.parquet`)
```
item_id, benchmark, split, stratum, evidence_area_ratio,
host, method, budget_b, n_out_tokens, n_global, frac_fine,
evidence_recall,               # fraction of evidence units at level 0 (NaN if no boxes)
pred, correct, yes_ratio_flag, # POPE: 1 if pred == "yes"
t_vit_ms, t_prefill_ms, t_decode_ms, ttft_ms, peak_mem_gb,
flops_vit, flops_prefill,
exp_id, config_hash, git_hash, gpu_type, seed
```
`tests/qafe/test_schemas.py` validates every writer.

### 2.4 Other artefacts
| File | Content |
|---|---|
| `<RESULTS_DIR>/qafe/teacher/<host>/<item_id>.pt` | `{"answer_ids": [T], "topk_ids": [T,32], "topk_logp": [T,32]}` from the full-token teacher |
| `<RESULTS_DIR>/qafe/labels/lossratio_<host>.parquet` | `item_id, block_r, block_c, level_tested, kl_increase` |
| `<RESULTS_DIR>/qafe/probe/<host>_layer<ℓ>.parquet` | `item_id, unit_r, unit_c, score, is_evidence` |
| `<RESULTS_DIR>/qafe/decisions.md` | one entry per gate / decision |

---

## 3. Host adapter (the only host-specific code)

```python
# qafe/hosts/base.py
class HostAdapter(Protocol):
    patch_px: int                  # varloop 20; qwen3vl <CHECK>
    merge: int                     # 2 (2x2 merge before projector)
    n_vit_layers: int              # varloop 23
    d_vit: int; d_dec: int

    def preprocess(self, image, max_px: int) -> tuple[Tensor, tuple[int,int]]: ...   # pixels, (Hp, Wp)
    def patch_embed(self, pixels) -> tuple[Tensor, Tensor]: ...                      # feats [Hp*Wp, d_vit], pos [Hp*Wp, 2]
    def vit_layers(self, x, pos, start: int, end: int, layer_hook=None) -> Tensor: ...  # variable-length OK
    def merge_project(self, x_groups: Tensor) -> Tensor: ...                         # [N, 4, d_vit] -> [N, d_dec]
    def build_inputs(self, vts: "VisualTokenSet", prompt: str) -> dict: ...          # splice tokens at their positions
    def generate(self, inputs: dict, max_new_tokens: int) -> str: ...
    def answer_logprobs(self, inputs: dict, answer_ids: Tensor) -> Tensor: ...       # teacher-forced, [T, V] or top-k
    def embed_text(self, ids: Tensor) -> Tensor: ...                                 # decoder input embeddings (frozen)
```

**Host notes the agent must resolve in step S0.1 and record in `docs/notes/qafe_hosts.md`:**

| Item | varloop | qwen3vl |
|---|---|---|
| Patch size, merge | 20 px, 2×2 | `<CHECK>`, 2×2 |
| ViT attention | full, per image, 2D rotary (NeMo docs) | `<CHECK>` window attention? |
| Accepts float / arbitrary 2D positions in ViT | `<CHECK>` | `<CHECK>` |
| How visual positions reach the decoder | `<CHECK>` (relative bias + short convs; keep original positions) | M-RoPE (t, h, w) `<CHECK>` |
| Extra streams to merge consistently | — | **DeepStack** features from intermediate ViT layers → LLM layers 1–3 `<CHECK layer ids>` |
| Separate image segment possible (for thumbnail global) | `<CHECK multi-image>` | yes (multi-image) |

---

## 4. Index math (`qafe/core/grid.py`)

- Patch grid (Hp, Wp). **Unit** = merge × merge patches (2×2). Unit grid Hu = Hp/2, Wu = Wp/2. Pad to a multiple of 4 units with a pad mask.
- **Levels:** L0 = 1 unit → 1 output token; L1 = 2×2 units → 1 token; L2 = 4×4 units → 1 token.
- **Inside the ViT, every output token is represented by 4 "virtual patches"** (a 2×2 group), so the host projector is unchanged. For a block at level L, virtual patch (i, j) = mean of the patches in quadrant (i, j) of the block.
  - ViT tokens after ℓ = 4 × N_out (excluding global).
  - Virtual patch position = quadrant centre (float).
- **Output token position** = block centre in unit coordinates.
- Required functions:
```python
def unit_grid(Hp, Wp, merge=2) -> tuple[int,int]: ...
def blocks_from_levels(level: Tensor) -> tuple[Tensor, Tensor]: ...  # block_id map, block list [(r0,c0,size)]
def virtual_patches(feats, pos, blocks, Hp, Wp) -> tuple[Tensor, Tensor]: ...   # [4*N, d], [4*N, 2]
def evidence_units(boxes, image_hw, Hu, Wu, patch_px, merge) -> Tensor: ...     # [Hu, Wu] bool
```
- **Tests (`test_grid.py`):** all-L0 levels reproduce the original patch order and positions; any level map tiles the image exactly once; token count formula is correct; evidence-unit mapping round-trips on synthetic boxes.

---

## 5. Build steps

### Phase P0 — Infrastructure

**S0.1 Host adapters**
- **Goal:** both hosts runnable through `HostAdapter`.
- **Do:** implement `VarloopHost` (reuse `adaloop/load.py` handles; R fixed at 3) and `Qwen3VLHost`. Fill the §3 table.
- **Tests:** `test_identity.py::host_passthrough` — `build_inputs` + `generate` with the unmodified token set equals the host's own `generate` (token-exact, 20 items per host).
- **Output:** `qafe/hosts/*.py`, `docs/notes/qafe_hosts.md`.

**S0.2 Data + evidence + strata**
- **Do:** loaders for V*, HR-Bench, TextVQA (OCR boxes), DocVQA, ChartQA, OCRBench, GQA (scene-graph boxes for objects named in the question), MMBench, MMStar, POPE, HallusionBench, MMMU, RefCOCO; training sources (Visual-CoT boxes, LLaVA-665K subset, TextVQA/DocVQA/ChartQA train, GQA train, RefCOCO train). Compute `evidence_area_ratio` and `stratum`. Freeze splits: dev 200, test 500 per benchmark.
- **Tests:** boxes inside image bounds; ratio ∈ (0, 1]; no item ID shared between train and dev/test (also across sources: image-hash check).
- **Output:** manifests, `splits.lock`.

**S0.3 Evaluation harness**
- **Do:** `orchestration/eval_qafe.py`. Takes `(host, method, budget, items)` → rows (§2.3). Official scorers (reuse `adaloop/eval/scorers.py`). Timing on A40 at batch 1 via `utils/timing.py` (CUDA events, warm-up 5).
- **Tests:** full-token run on 50 dev items reproduces public numbers within plausible range; nondeterminism re-run on 50 items logged.

### Phase P1 — Decision experiments

**S1.1 Selectors and emulation (`core/selectors.py`, `core/emulate.py`)**
- **Selectors** (all return a per-unit relevance map):
  - `uniform`: constant
  - `random`: seeded
  - `crisp`: cosine of projected visual tokens vs decoder input embeddings of question nouns (spaCy), + diversity fill — same definition as AdaLoop A6
  - `oracle_box`: `evidence_units`
- **Emulation** (`emulate.py`, post-projector, no training): given a level map, pool decoder tokens of merged blocks (mean over the units' projector outputs), position = block centre. Optional global tokens: pool to G×G.
- **Tests:** at budget 1.0 every selector reproduces the full run.

**S1.2 D1 — Foveation oracle** (`exp_d1_foveation.py`)
- **Conditions at matched token count N_b** (b ∈ {0.15, 0.25}):
  - (a) uniform downscale to N_b
  - (b) thumbnail + crop around the oracle box
  - (c) thumbnail + crop around the `crisp` peak
  - (d) thumbnail + random crop
  - (e) full tokens (reference)
- **Thumbnail:** downscale to `<THUMB_TOKENS>` (e.g. 64). Crop fills the rest at native resolution. Crop box = evidence box expanded to the budget area, aspect clamped, inside the image.
- **Thumbnail + crop format:**
  - Qwen3-VL: two images.
  - varloop: two images if supported (S0.1); otherwise a side-by-side composite (logged as such).
- **Items:** dev splits of V*, TextVQA, GQA, ChartQA (300 each if dev is too small, drawn from the train manifest, marked `split=probe`).
- **Analysis:** accuracy by condition × stratum × evidence-area-ratio bins; paired bootstrap of (b − a), (c − a), (c − d).
- **Decision (Gate Q1 input):** H-F holds if (b − a) ≥ `<Y_F>` on small-target, CI excluding 0.

**S1.3 D2 — Tolerance to coarse tokens** (`exp_d2_tolerance.py`)
- At b ∈ {0.25, 0.5}, with the `crisp` and `oracle_box` level maps, compare:
  - (i) **drop** non-selected
  - (ii) **merge** non-selected into L1/L2 (emulation)
  - (iii) merge + 16 global tokens (pooled in-grid)
  - (iv) merge + thumbnail segment (if supported)
  - (v) full
- **Decision:** merge vs drop (H-T); global-token placement.

**S1.4 D3 — Noise with controls** (`exp_d3_noise.py`)
- **Items:** POPE (all three splits, dev subset), HallusionBench, and a clutter stratum (GQA items with ≥ `<N_OBJ>` scene-graph objects and evidence ratio < 0.05).
- **Conditions at k ∈ {0.25, 0.5}:** full · `crisp` · `oracle_box` · `random` · **mean-token replacement** (non-selected tokens replaced by the image-mean projector token; count unchanged).
- **Log:** accuracy, F1, yes-ratio.
- **Decision:** H-N holds only if selected > full **and** > random **and** > mean-replacement, with yes-ratio difference < `<YES_DELTA>`. Feeds the D6 decision rule (headroom above the teacher).

**S1.5 D4 — Relevance probe** (`exp_d4_probe.py`)
- Frozen host ViT. For layers ℓ ∈ {4, 8, 12, 16, 20}, extract unit features (mean of the 4 patches).
- Probe = `MLP([unit_feat ‖ q_pool ‖ unit_feat ⊙ W q_pool]) → logit`, where q_pool = mean of decoder input embeddings of the question.
- Train on 5K train items with evidence boxes (Visual-CoT, TextVQA, GQA); evaluate on 1K held-out.
- **Metrics:** AUROC; **evidence recall@b** = fraction of evidence units in the top-b units, b ∈ {0.1, 0.25}.
- **Decision:** choose ℓ = the earliest layer with recall@0.25 ≥ `<RECALL_MIN>`. If none ≤ ⅔ depth → **two-stage design** (record it in decisions.md; P2 then builds a glance → re-encode router).

**S1.6 Gate Q1** — write `decisions.md`: full QAFE / efficiency-only / stop; merge vs drop; global placement; router layer ℓ (or two-stage); H-N result.

### Phase P2 — Build QAFE modules

**S2.1 `QuestionPooler` (`core/question.py`)**
```python
class QuestionPooler(nn.Module):
    # in: decoder input embeddings of question ids [Tq, d_dec]; out: q [d_q]
    def __init__(self, d_dec: int, d_q: int = 512, n_heads: int = 8): ...
    def forward(self, q_emb: Tensor, q_mask: Tensor) -> Tensor: ...  # 1 learned query, attention pool, LayerNorm, Linear
class NullQuestion(nn.Module): ...  # learned vector used when no question
```

**S2.2 `AdaLNModulator` (`core/condition.py`)**
```python
class AdaLNModulator(nn.Module):
    # per ViT layer in [ℓ, L): produces (shift, scale, gate) for attn and mlp branches from q (and optional budget b)
    def __init__(self, d_q: int, d_vit: int, layers: list[int]): ...   # final Linear zero-init ⇒ identity at init
    def forward(self, q: Tensor, layer: int) -> dict[str, Tensor]: ...
```
- Applied through a `layer_hook` in `HostAdapter.vit_layers`: `x_attn_in = LN(x)·(1+scale)+shift`; residual branch × (1 + gate).
- Ablation: `QAViTConcat` (append projected question tokens to attention keys/values in the top layers; outputs for question positions discarded).
- **Test:** zero-init ⇒ output identical to host (atol 1e-5).

**S2.3 `Router` (`core/router.py`)**
```python
class Router(nn.Module):
    # scores 4x4-unit blocks (L2 split?) and 2x2-unit blocks (L1 split?)
    def __init__(self, d_vit: int, d_q: int, hidden: int = 256): ...
    def forward(self, unit_feats: Tensor, q: Tensor, b: float) -> dict:  # unit_feats [Hu, Wu, d_vit]
        # returns logits_split_L2 [Hu/4, Wu/4], logits_split_L1 [Hu/2, Wu/2], unit_score [Hu, Wu]
    def decide(self, out: dict, b: float, mode: Literal["gumbel", "hard"], tau: float) -> RouteMap: ...
```
- Input features: unit features at layer ℓ (mean of 4 patches) → `[feat ‖ FiLM_b(feat) ⊙ proj(q)]` → MLP. Block logits = pooled unit logits + a learned bias per level.
- **Training (`gumbel`):** Bernoulli split decisions with Gumbel-sigmoid + straight-through; expected token count `E[n]` is differentiable.
- **Inference (`hard`):** split greedily by score until n ≤ b·N_full (tie-break by score); deterministic.
- **Tests (`test_router_budget.py`):** hard mode never exceeds the budget; b = 1 ⇒ all L0; b → min ⇒ all L2 + global; E[n] matches the Monte-Carlo mean within 2%.

**S2.4 `QuadtreeMerger` (`core/merge.py`)**
```python
class QuadtreeMerger(nn.Module):
    def forward(self, patch_feats, patch_pos, route: RouteMap, Hp, Wp) -> tuple[Tensor, Tensor, Tensor]:
        # returns vit_tokens [4*N, d_vit], vit_pos [4*N, 2], token_level [N]
```
- Default operator: mean pooling per quadrant. Ablation: attention pooling with a mean query (Molmo style), initialised to mean.
- `ScaleEmbedding` (`core/scale_embed.py`): learned vector per level, zero-init, added to the virtual patches.
- **Test (`test_merge_shapes.py`):** all-L0 ⇒ identical tensors; shapes; position centres.

**S2.5 `GlobalPool` (`core/global_tokens.py`)**
- Default: adaptive average pool of layer-ℓ patch features to (2G × 2G) → G×G tokens of 4 virtual patches each, flagged `is_global`.
- **Position:** per the D2 decision (separate segment, or in-grid with a global flag embedding).
- Ablation: `ThumbnailGlobal` runs a separate low-res ViT pass.

**S2.6 `QAFEEncoder` (`core/encoder.py`) and `api.py`**
```python
class QAFEEncoder(nn.Module):
    def __init__(self, host: HostAdapter, cfg): ...   # owns QuestionPooler, AdaLN, Router, Merger, ScaleEmbed, GlobalPool
    def forward(self, image, question: str | None, b: float, mode="hard") -> VisualTokenSet:
        # 1 patch_embed → 2 vit_layers[0, ℓ) on all patches (cache StemCache)
        # 3 q = pooler(embed_text(question)) or null
        # 4 route = router.decide(router(unit_feats, q, b), b, mode)
        # 5 tokens = merger(stem feats, route) + global tokens
        # 6 vit_layers[ℓ, L) with AdaLN(q) hooks → 7 merge_project → VisualTokenSet

# qafe/api.py
def build_encoder(host_name: str, cfg_path: str, ckpt: str | None) -> QAFEEncoder: ...
def encode(enc, image, question, b) -> VisualTokenSet: ...
def upgrade(enc, vts: VisualTokenSet, units: Tensor, q_override: Tensor | None) -> VisualTokenSet: ...  # §7
def zoom(enc, image, box, scale: float, question) -> VisualTokenSet: ...                                 # §7
```
- **Tests (`test_identity.py`):** `force_all_fine=True` with gates at zero ⇒ token-exact host outputs (20 items); `n_vit_tokens` equals the formula; timing shows ViT time falls with b.

### Phase P3a — Pilots that choose conditioning and the training signal

**Common pilot setup** (fixed for every arm):
- Development host `<DEV_HOST>`; 20K training samples (stratified mix, §6); 1 epoch; same trainable modules; same LR schedule; same seed set {0, 1}.
- Dev evaluation at b ∈ {0.2, 0.35, 0.5} on TextVQA, V*, GQA, ChartQA, POPE dev.

**S3.1 Teacher cache** (`teacher_cache.py`)
- Full-token, full-resolution host; greedy answer; store top-32 log-probs per answer token (§2.4). Also store the teacher's correctness.
- **Test (`test_teacher_cache.py`):** cached log-probs match live within 1e-3 on 100 items.

**S3.2 Loss-ratio router labels** (`labels_lossratio.py`, 5K-item subset)
- For each L2 block: KL increase on the answer tokens when that block alone is merged to L2 (everything else fine).
- Batch block variants per image. Normalise per image to percentiles.

**S3.3 Trainers** (one file each, sharing `orchestration/_train_common.py`)

| Arm | Loss | Notes |
|---|---|---|
| **DIST** | `KL(teacher_topk ‖ student) + β·max(0, E[n]/N − b)²` | Teacher-forced on teacher answer ids |
| **SFT** | `CE(gt_answer) + β·budget` | Teacher-forced on ground truth |
| **DIST+SFT** | `KL + α·CE + β·budget` | α ∈ {0.1, 0.5} |
| **RSUP** | DIST + `γ·BCE(unit_score, label)` | Labels: oracle boxes where available, else loss-ratio |
| **GRPO** | Policy = router (Bernoulli splits); G = `<G>` (8) samples per item; reward `1[correct] − λ·n/N`; group-normalised advantage; KL to the init router β_KL | Decoder frozen, greedy short answers; correctness by the benchmark scorer; **no learned reward model** |
| **NOCOND** | Best of the above with AdaLN disabled | Answers D5 / H-C |

- b sampled per batch from {0.1, 0.2, 0.35, 0.5}. Gumbel τ annealed 1.0 → 0.3.
- **Diagnostics logged every `<EVAL_EVERY>` steps** (to `<RESULTS_DIR>/qafe/bakeoff/<arm>.parquet`):
  - dev accuracy per b
  - budget calibration |n/N − b|
  - evidence recall
  - teacher agreement
  - accuracy on teacher-wrong items
  - GRPO only: zero-advantage group fraction, reward mean / variance, mean n/N (collapse check)

**S3.4 Bake-off driver + decision** (`bakeoff.py`)
- Runs all arms × 2 seeds. Produces one table: accuracy at matched b with bootstrap CIs, and the sample-efficiency curve.
- **Apply the decision rule from the high-level spec §5.1** and write it to `decisions.md`:
  1. No headroom in D3 → best dev objective; ties → the cheapest.
  2. Headroom → include an outcome term; RL only if ≥ `<RL_MARGIN>` better and zero-advantage < `<ZA_MAX>`.
  3. Low evidence recall → include RSUP.
- **D5 decision:** keep conditioning if best-arm vs NOCOND ≥ `<COND_MIN>` with CI excluding 0; otherwise drop AdaLN (router still uses q).

### Phase P3b — Full training and ablations (development host)

**S3.5 Full training**
- Chosen objective; full mix ~250K samples (§6); `<EPOCHS>` = 1; checkpoints every `<CKPT_EVERY>`.
- Trainable: pooler, AdaLN, router, scale embed, global pool (+ LoRA r = 16 on ViT layers ≥ ℓ if ablation A8 says so).

**S3.6 Evaluation vs baselines** (`eval_qafe.py`, test split only after Gate Q2)
- **Baselines** (all at matched N_out):

| Baseline | Implementation |
|---|---|
| full | host, unmodified |
| uniform | host downscaled to N_out |
| pool | fixed 2× / 4× pooling of all units |
| ToMe | in-ViT bipartite merging to N_out |
| uhd_lite | uniform in-ViT merge at layer ℓ (QAFE with a constant router) |
| crisp | post-projector selection (AdaLoop A6) |
| hqap_lite | training-free hierarchy: complexity-scaled budget + question allocation |
| divprune | diversity selection |
| glance_etd | only if code is released |
| oracle | `oracle_box` router |

- **Statistics:** paired bootstrap (10K); Holm across the headline claims; everything else labelled exploratory.

**S3.7 Ablations** (`ablations.py`; dev, then test for the final table)

| ID | Variant |
|---|---|
| A1 | Conditioning only, all units fine (no merge) |
| A2 | Merge with image-only router (q = null) |
| A3 | Full QAFE |
| A4 | No global tokens |
| A5 | ℓ ∈ {4, 8, 12, 16} |
| A6 | Pooled global vs thumbnail global |
| A7 | Router supervision none / oracle / loss-ratio |
| A8 | Frozen upper ViT vs LoRA |
| A9 | Best non-RL vs GRPO |
| A10 | Fixed b vs router-chosen n (add a STOP-style budget head) |
| A11 | Mean vs attention pooling in the merger |

**S3.8 Gate Q3** — criteria in the high-level spec §6. Write to `decisions.md`.

### Phase P4 — Port to varloop and couple with the loop

**S4.1 Train QAFE on varloop** (R = 3 fixed) with the chosen recipe. Repeat S3.6 on varloop at R = 3.

**S4.2 Loop actions** (`qafe/api.py` → used by `adaloop/actions.py`)
```python
def upgrade(enc, vts, units, q_override=None) -> VisualTokenSet:
    # 1 pick coarse blocks containing `units` → set to L0 in RouteMap
    # 2 build their fine virtual patches from vts.stem (no stem recompute)
    # 3 run vit_layers[ℓ, L) only for the new tokens (with AdaLN(q_override or q))
    # 4 merge_project → return ONLY the new tokens (+ ids of the coarse tokens they replace)
def zoom(enc, image, box, scale, question) -> VisualTokenSet:
    # crop box, resize ×scale, full QAFE encode at budget b_zoom, positions mapped to original image coords
```
- **AdaLoop side** (owned by the inference team; QAFE provides the functions): `actions.restore(...)` gains `mode ∈ {insert, restart}` × `source ∈ {pool, upgrade, zoom}`.
- **Which units:** (a) top router scores among coarse blocks recomputed with `q_override = W·pool(h_pass_r)` (a small linear map, trained in P4 by distillation on items where the full-token teacher is right and pass-1 QAFE is wrong); (b) attention mass from the last token to coarse tokens (no training).
- **Tests (`test_upgrade.py`):**
  - Upgrading every unit then restarting equals a full-fine encode (token-exact on 10 items).
  - Ledger counts upgrade ViT cost.

**S4.3 Controller integration**
- AdaLoop B1 feature `vis_attn_mass_coarse` and B2 LOOK variants.
- Trajectory tree (`adaloop/controller/tree.py`) adds LOOK = upgrade / zoom branches.
- Evaluation per AdaLoop B6/B7 at matched measured cost.

**S4.4 Gate Q4** — loop-coupled QAFE vs encoder-only QAFE vs AdaLoop baselines.

### Phase P5 — Optional
- Joint GRPO over controller + router (AdaLoop C3), only if Q4 passes and S3.4 showed RL is not starved.

---

## 6. Training data mix (P3a uses 20K stratified, P3b ~250K)

| Source | Share | Boxes | Role |
|---|---|---|---|
| LLaVA-665K subset | 40% | — | General |
| TextVQA / DocVQA / ChartQA / OCR-VQA train | 24% | TextVQA OCR | Detail pressure |
| Visual-CoT | 20% | yes | Evidence supervision |
| GQA train | 8% | scene graph | Compositional |
| RefCOCO/+/g | 4% | yes | Grounding guardrail |
| Caption / no-question samples | 4% | — | Null-question path |

Every sample is converted to **short-answer form**. Drop items whose image hash appears in any dev/test split.

---

## 7. Hyperparameters (starting points)

| Item | Value |
|---|---|
| d_q | 512 |
| Router hidden | 256 |
| G (global grid) | 4 → 16 tokens |
| LR (adapters) | 1e-4 AdamW, cosine, warm-up 3% |
| LR (LoRA) | 5e-5 |
| Batch | 16 (grad-accum to 64) |
| Budget penalty β | 1.0 (sweep 0.3–3) |
| Gumbel τ | 1.0 → 0.3 linear |
| DIST temperature | 1.0 |
| GRPO | G = 8, λ = 0.2, β_KL = 0.02, LR 1e-5, ≤ `<GRPO_STEPS>` |
| Probe | 2-layer MLP, 256 hidden, 5 epochs |

---

## 8. Experiment registry

| Exp | Script | Config | Output | Decision |
|---|---|---|---|---|
| D1 | `exp_d1_foveation.py` | `exp/d1.yaml` | `qafe_runs.parquet` (exp_id = d1) | Q1: viability |
| D2 | `exp_d2_tolerance.py` | `exp/d2.yaml` | same | merge vs drop; global placement |
| D3 | `exp_d3_noise.py` | `exp/d3.yaml` | same | headroom above teacher; noise claim |
| D4 | `exp_d4_probe.py` | `exp/d4.yaml` | `probe/*.parquet` | router layer ℓ / two-stage |
| D5 + D6 | `bakeoff.py` | `train/bakeoff.yaml` | `bakeoff/*.parquet` | conditioning; training signal |
| Main | `eval_qafe.py` | `exp/main_<host>.yaml` | `qafe_runs.parquet` | Q3 |
| Ablations | `ablations.py` | `exp/abl_*.yaml` | same | paper tables |
| Loop | AdaLoop `eval_policy.py` | AdaLoop configs | AdaLoop `runs.parquet` | Q4 |

---

## 9. Implementation hazards

| Hazard | Why | Check |
|---|---|---|
| Positions of merged tokens | Hosts may assume an integer grid; float block centres may break RoPE or relative bias | S0.1 table; `test_positions.py` compares integer-rounded vs float on dev |
| Qwen3-VL DeepStack | Intermediate-layer features are injected separately and must use the same RouteMap | Apply `QuadtreeMerger` to every DeepStack stream; identity test |
| Window attention (if present) | Variable token sets break window partitioning | Use only full-attention hosts, or move routing after the last window layer |
| Pad units | Padded units must never be L0 or counted | Pad mask in router + merger |
| Teacher / student tokenisation drift | KL needs identical answer ids | Teacher-force the student on the teacher's ids |
| Budget collapse in GRPO | Reward hacking toward minimum tokens | Monitor n/N; floor b_min |
| varloop thumbnail segment | Multi-image support unknown | Fall back to in-grid global tokens |
| Cost accounting | Upgrade and zoom must be charged | Every ViT call appends to the AdaLoop `CostLedger` |

---

## 10. Definition of done per phase

- **P0:** S0.1–S0.3 tests pass; host notes filled.
- **P1:** D1–D4 tables in `notebooks/qafe/02_*` and `03_*`; Gate Q1 entry written.
- **P2:** all `tests/qafe/*` pass; ViT time falls monotonically as b falls.
- **P3a:** bake-off table; decision entries for conditioning and the training signal.
- **P3b:** main table vs all baselines; ablation tables; Gate Q3 entry.
- **P4:** upgrade / zoom tests pass; loop-coupled results; Gate Q4 entry.
