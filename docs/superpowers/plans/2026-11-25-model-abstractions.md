# Model Abstractions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move model loading, inference, loop control and pruning from the two notebooks into `adaloop/`, behind `adaloop.api`.

**Architecture:** One frozen dataclass `LoadedVLM` passed to plain functions. `api.py` only re-exports. `notebooks/mvp.ipynb` is the test.

**Tech Stack:** Python 3.13, PyTorch, transformers 5.x, PIL, PyYAML.

**Spec:** `docs/superpowers/specs/2026-11-25-model-abstractions-design.md` (parent: `docs/DEC3_MVP.md`)

## Global Constraints
- Source behaviour is `notebooks/01_huggingfaceload.ipynb` and `02_pruning.ipynb`. Do not edit or move them.
- Notebooks import only `adaloop.api`. Constants live in `configs/mvp.yaml`.
- No pytest suite (DEC3_MVP amendment): assert cells in `notebooks/mvp.ipynb` are the tests, run by Vedaang on the GPU host.
- Code does not run locally. Local check is `python3 -m py_compile` only; never claim GPU behaviour as verified.
- Public functions: type hints, docstring, `logging` (no `print` in modules).
- Branch `v/mvp-build`; commits prefixed `agent:`; no push.
- Each passing check prints `PASS S0: <what>`.

## Review Focus
- `k <= 0` or `k > 1` in any selector → `ValueError`.
- `indices` unsorted or duplicated in `prune_inputs` → `ValueError` (silently misaligns tokens and embeddings).
- Text-only inputs to `extract_visual_tokens` → `ValueError` naming the missing `pixel_values`, not a `KeyError`.
- Forward fails inside `count_layer_calls` → hooks still removed.
- Bad image URL or missing path in `load_image` → error raised, never a blank image.

## Step Map
| Step | Function | File | Input → output | Notebook section |
|---|---|---|---|---|
| 1 | `load_config`, `load_model` | `utils/io.py`, `core/model/load.py` | path → dict → `LoadedVLM` | Load |
| 2 | `load_image`, `prepare_inputs` | `utils/images.py`, `core/model/prompting.py` | question, image → inputs | Inputs |
| 3 | `set_loops`, `count_layer_calls` | `core/model/loops.py` | R, inputs → calls per layer | Loops |
| 4 | `generate`, `last_logits` | `core/model/inference.py` | inputs → result / logits | Inference |
| 5 | `extract_visual_tokens`, selectors, `prune_inputs` | `core/tokens/` | inputs, k → pruned inputs | Pruning |

No `orchestration/` yet: nothing composes steps until the grid stage.

---

### Task 1: Package setup, config and loader

**Files:**
- Create: `configs/mvp.yaml`, `utils/__init__.py`, `utils/io.py`, `adaloop/schemas/__init__.py`, `adaloop/schemas/vlm.py`, `adaloop/core/__init__.py`, `adaloop/core/model/__init__.py`, `adaloop/core/model/load.py`, `adaloop/api.py`, `notebooks/mvp.ipynb`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict`; `LoadedVLM(model, processor, loading_info: dict)` frozen dataclass; `load_model(cfg: dict) -> LoadedVLM`.

- [ ] **Step 1:** `git checkout -b v/mvp-build`
- [ ] **Step 2:** `pyproject.toml`: `requires-python = ">=3.10,<3.14"`, `"transformers>=5"`, add `"requests"`, package include `["adaloop*", "utils*"]`. Leave other pins; real pins need the host's `pip freeze`.
- [ ] **Step 3:** `configs/mvp.yaml`:
```yaml
model_id: thinkingmachines/TML-d24-varloop-research
device_map: balanced
max_memory: {0: 35GiB, 1: 35GiB}
res_px: 512
max_new_tokens: 128   # thinking is still on; S2 lowers this to 16
```
- [ ] **Step 4:** Notebook: papermill-tagged parameters cell (`CONFIG = "configs/mvp.yaml"`, resolved from the repo root) and section Load:
```python
vlm = load_model(load_config(CONFIG))
assert not vlm.loading_info["missing_keys"] and not vlm.loading_info["unexpected_keys"]
assert all(str(d) not in ("cpu", "disk", "meta") for d in vlm.model.hf_device_map.values())
print("PASS S0: model loads fully on GPU")
```
- [ ] **Step 5:** Implement the three interfaces and re-export from `api.py`. `load_model` uses `AutoModelForMultimodalLM.from_pretrained(..., dtype=torch.bfloat16, experts_implementation="grouped_mm", output_loading_info=True)`, then `eval()` and `requires_grad_(False)`. No `R` argument.
- [ ] **Step 6:** `python3 -m py_compile utils/*.py adaloop/api.py adaloop/schemas/*.py adaloop/core/model/*.py` → no output.
- [ ] **Step 7:** Commit `agent: add config and model loader`.

### Task 2: Inputs, loop control, inference

**Files:**
- Create: `utils/images.py`, `adaloop/core/model/prompting.py`, `loops.py`, `inference.py`; `adaloop/schemas/vlm.py` gains `GenerationResult`
- Modify: `adaloop/api.py`, `notebooks/mvp.ipynb`

**Interfaces:**
- Consumes: `LoadedVLM`.
- Produces:
  - `load_image(url_or_path: str, max_side: int) -> PIL.Image.Image` (RGB, thumbnail)
  - `prepare_inputs(vlm, question: str, image: Image | None = None) -> BatchFeature` (one user turn, image before text, `reasoning_effort="none"`, on `vlm.model.device`)
  - `set_loops(vlm, R: int) -> None` (1..5 else `ValueError`; sets both `model.config.text_config` and `model.model.language_model.config`)
  - `count_layer_calls(vlm, inputs) -> list[int]`; `expected_layer_calls(vlm) -> list[int]` (R inside `loop_layer_index_start..end`, else 1)
  - `GenerationResult(token_ids: Tensor, text: str)` (new tokens only, CPU; special tokens skipped in `text`)
  - `generate(vlm, inputs, max_new_tokens: int) -> GenerationResult` (greedy, cache on)
  - `last_logits(vlm, inputs) -> Tensor` (float32, CPU, `use_cache=False`, `logits_to_keep=1`)

- [ ] **Step 1:** Notebook sections Inputs, Loops, Inference, using the candy image URL and question "What animal is on the candy?" from notebook 02:
```python
for R in range(1, 6):
    set_loops(vlm, R)
    calls = count_layer_calls(vlm, inputs)
    assert calls == expected_layer_calls(vlm) and sum(calls) == 6 + 12 * R + 6
set_loops(vlm, 1); a = last_logits(vlm, inputs)
set_loops(vlm, 3); b = last_logits(vlm, inputs)
assert not torch.equal(a, b)
assert generate(vlm, inputs, cfg["max_new_tokens"]).text.strip()
for bad in (0, 6):
    try: set_loops(vlm, bad); raise AssertionError("accepted bad R")
    except ValueError: pass
```
Plus a cell asserting `load_image("missing.png", 512)` raises.
- [ ] **Step 2:** Implement the interfaces; `count_layer_calls` removes hooks in `finally`. Re-export from `api.py`.
- [ ] **Step 3:** `py_compile` the new files → no output. Commit `agent: add inputs, loop control and inference`.

### Task 3: Visual tokens, selectors, pruning

**Files:**
- Create: `adaloop/core/tokens/__init__.py`, `prune.py`, `selectors.py`
- Modify: `adaloop/api.py`, `notebooks/mvp.ipynb`

**Interfaces:**
- Consumes: `LoadedVLM`, `prepare_inputs`, `last_logits`, `generate`.
- Produces:
  - `extract_visual_tokens(vlm, inputs) -> tuple[Tensor, ModelOutput]` ([N, 2304], full vision output)
  - `random_selector(visual_tokens, k: float, seed: int = 42) -> Tensor`
  - `uniform_selector(image_grid_thw, k: float) -> Tensor` (farthest-point, as notebook 02)
  - `question_embeddings(vlm, question: str) -> Tensor`
  - `text_mean_selector(visual_tokens, text_embeddings, k: float) -> tuple[Tensor, Tensor]` (indices, scores)
  - `prune_inputs(vlm, inputs, vision_outputs, indices) -> dict` (`input_ids`, `attention_mask`, `mm_encoder_outputs={"image": ...}`; packed; batch of 1)
  - All selectors return sorted indices, `n_keep = max(1, int(N * k))`.

- [ ] **Step 1:** Notebook section Pruning:
```python
set_loops(vlm, 3)
vis, out = extract_visual_tokens(vlm, inputs)
n = vis.shape[0]
full = prune_inputs(vlm, inputs, out, random_selector(vis, 1.0))
assert torch.equal(last_logits(vlm, inputs), last_logits(vlm, full))
for k in (0.5, 0.25):
    for idx in (random_selector(vis, k), uniform_selector(inputs["image_grid_thw"], k),
                text_mean_selector(vis, question_embeddings(vlm, question), k)[0]):
        assert len(idx) == max(1, int(n * k)) and torch.equal(idx, idx.unique())
        pruned = prune_inputs(vlm, inputs, out, idx)
        assert pruned["input_ids"].shape[-1] == inputs["input_ids"].shape[-1] - n + len(idx)
```
Plus cells expecting `ValueError` for `k=0`, `k=1.5`, `indices=torch.tensor([1, 0])`, `torch.tensor([0, 0])`, and text-only inputs to `extract_visual_tokens`. Notebook 02 reported k=100% as bit-exact; if `torch.equal` fails on host, report the max difference, do not loosen silently.
- [ ] **Step 2:** Implement; re-export from `api.py`.
- [ ] **Step 3:** `py_compile` → no output. Commit `agent: add visual-token selectors and pruning`.

### Task 4: Docs and host handoff

**Files:**
- Create: `adaloop/README.md`, `utils/README.md`
- Modify: `docs/ai_context/CODEBASE_MAP.md`

- [ ] **Step 1:** READMEs: public entry points, step map, restrictions. CODEBASE_MAP: set `adaloop/api.py` and `utils/` rows to created, add `notebooks/mvp.ipynb`, add a small Mermaid diagram of api → core → schemas/utils.
- [ ] **Step 2:** Commit `agent: document model modules`.
- [ ] **Step 3:** Give Vedaang the host commands and state that GPU checks are unrun:
```bash
pip install -e . --no-deps
jupyter nbconvert --to notebook --execute notebooks/mvp.ipynb --output-dir execution_results/mvp
```
