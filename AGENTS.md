# AGENTS.md — AdaLoop-VLM

Project overrides only. Universal rules: `~/agent-governance/AGENTS.md` (read it first; do not duplicate it here).

## Project
AdaLoop-VLM: a looped VLM (varloop backbone) whose loop state decides after each pass to exit, loop again, or look again;
then post-trained (PT-1 LoRA-insert, PT-2 question-aware encoder QAFE, PT-3 joint RL). Reported as accuracy vs FLOPs and
measured latency, plus safety. **Master reference: `docs/IMPLEMENTATION.md`** (wins on conflicts; index in `docs/README.md`).
Controller-feature cost is counted in FLOPs from day one.

## Goal and priority
1. **Dec 3 first:** build `docs/DEC3_MVP.md` (S0–S17) to a working POC plus the course paper. It wins over other docs until Dec 3.
2. **Then the full project:** if time remains, continue with `docs/IMPLEMENTATION.md` steps not covered by the MVP, to add results to the paper.
The agent builds everything, CORE items included; Vedaang reviews.

## Notebooks are the tests
- Python packages (`adaloop/`, `qafe/`, `utils/`) hold all logic. Notebooks import them through `api.py` and use them to run the features and the research.
- **Every feature has a notebook** that runs it and checks it works: intermediate outputs shown, `assert` checks that print `PASS <feature>: <what>`.
- **Keep the set small and fixed.** One notebook per build stage group (list in `docs/DEC3_MVP.md` / the plan), not one per change. A new feature goes into the matching notebook as a section; a changed feature updates that section in the same commit. Replace outdated cells; do not stack new checks on old ones.
- Checks cover only the stage's Done-when, a few asserts each.
- A feature is done only when its notebook ran on the GPU host and the output was pasted back.
- **Vedaang runs the notebooks himself**, step by step, on Inara. Agents write the code, say which notebook is ready, and wait for the output. Agents do not execute notebooks unless asked.
- **Keep it simple.** Notebooks are for understanding the project: short cells, one call per step, plain prints. Code reused in two places moves to a small function in the package and is exposed through `api.py`. No clever abstractions.
- Sometimes Vedaang builds a notebook himself: then guide (which `api` function, what to print, what to assert) instead of writing it.

### Notebook style (Vedaang's pattern; reference: `model_training/post_training/03-sft/01_a_instruct_tuning.ipynb`, `pre_training/llm/02-multi-architecture/02_qwen3.ipynb`)
- **Header cells, in order:** `%load_ext autoreload` + `%autoreload 2` → stdlib imports → `ROOT, cfg = api.setup_notebook(...)` printing the root → host check (`"inara" in socket.gethostname()`) + GPU list (name, GiB) → third-party imports (torch, pandas, matplotlib).
- **Key settings visible near the top** as UPPER_CASE names read from `cfg` (e.g. `MODEL_ID = cfg["model_id"]`, `R = 3`, `N_ITEMS = 20`). Values still live in `configs/`.
- **Sections:** markdown `### Step-N:- <Title>` with 1–3 sentences on *what* and *why* (the notebook teaches the project). Sub-steps `#### <Title>`.
- **Many small cells:** one step per cell; show results by leaving the object as the last line (`items[0]`, `df.head()`, `summary`) rather than wrapping in prints. Import the `api` names used by a section at the start of that section if it helps reading.
- **The notebook composes the workflow visibly:** load → prepare → run one sample → inspect → then the batch. The package provides the building blocks; don't hide a whole stage behind one call when its steps are worth seeing.
- **Sample run before the full run:** a tiny run (1 item / 1 batch) with checks, then the real run.
- **Checks:** at the end of a step, plain `assert` lines then `print(f"✓ <what> ...")`. A `[STAGE] <name>` print starts a step that runs model work. No check-registry helpers or summary tables.
- **Outputs** go to `ROOT / "execution_results" / ...`; never next to the notebook.
- **Don't copy:** deep imports into package internals (use `adaloop.api`), hard-coded hyperparameters in cells, commented-out code.
- Notebook names: `notebooks/NN_<feature>.ipynb`, in build order. `notebooks/mvp_results.ipynb` only reads saved artifacts and shows the paper tables and figures.

## Branches
| Branch / ref | Purpose | Rule |
|---|---|---|
| `main` | Active AdaLoop-VLM work. Clean scaffold on top of the old history. | Commit small steps, `agent:` prefix. Ask before every push. |
| `cerebrus-v0` | All old CEREBRUS code and history (frozen). | Read-only reference. Do not commit. |
| `archive/historical-code-pre-retirement` | Old V0/V1/V2 source. | Read-only. |
| tag `cerebrus-v1-final` | Last CEREBRUS commit before the reset. | Never move. |
| `master` | Legacy remote branch. | Leave untouched. |

Never force-push or rewrite history without explicit approval. Do not merge `cerebrus-v0` into `main`.
To see old code: `git show cerebrus-v0:<path>`; do not restore old trees into `main`.

## Layout (tracked)
```
adaloop/ qafe/ utils/  packages, layered per docs/IMPLEMENTATION.md §4.2 (api → orchestration → core → schemas/utils)
tests/              optional pytest for pure CPU logic; notebooks are the main tests (see below)
configs/            YAML configs
notebooks/          one notebook per feature; they are the tests; no production logic
scripts/            thin CLI entry points
docs/               IMPLEMENTATION.md (master), README.md (index), decisions.md,
                    01_research/ 02_loop/ 03_encoder/ 04_prompts/ notes/ ai_context/
docs/proposal/      Proposal_v2.docx (superseded), GRA research questions
docs/cerebrus/      AUDIT_LEDGER.md, PROJECT_RESUME_CONTEXT.md, RESUME_PROJECT_BRIEF.md
docs/literature/    paper classification CSVs, reading notes, model cards (varloop card here)
pyproject.toml, environment.yml
```

## Local-only (git-ignored, never commit)
- `old_code/` — all old code in one place, read-only reference for porting:
  - `old_code/cerebrus-v0/` — git worktree of branch `cerebrus-v0` (CEREBRUS source, `edge_glass_modular/` etc.).
  - `old_code/archive-v0-v2/` — git worktree of `archive/historical-code-pre-retirement` (V0/V1/V2 source).
  - `old_code/third_party/TinyRecursiveModels/` — vendored upstream TRM repo (not ours, not in git).
  - Do not commit from these worktrees. Remove with `git worktree remove old_code/<name>`.
- `old_data/` — non-code leftovers (~25 GB): `legacy_data/` (v1/v2 datasets, checkpoints), `wandb/` (all wandb runs, by source), `logs/`, `plots/`, `v2_notebooks/`, `misc/`.
- `Papers/` — PDFs (~500 MB). Canonical set: `Papers/Complete Papers List/Shortlisted_Literature/` (P0–P3 tiers; `Paper_Classification.csv` maps each ID to its file).
- New wandb dirs, logs and run outputs go to `execution_results/` or `old_data/`, never the repo root.

## Code rules
- Port from old code only these: `qa_metrics`, bootstrap CI/stats utilities, trimmed config loading, LoRA/4-bit loading helpers. Do not port encoders, MRL, Perceiver, alignment, TRM, trainers.
- Every ported file keeps a provenance comment (`# Ported from cerebrus-v0:<path> @ <commit>`).
- `qa_metrics`: ROUGE-L is multiplied by 100.
- Gate 0 order: loop control test (logits R=1 vs R=3 differ) → readout-equivalence check (pass r == fresh run at R=r) → only then any grid run.
- Code does not run locally. All runs happen on the Inara GPU host (`conda env create -f environment.yml`). Run outputs go to `execution_results/` (ignored).
- Public functions: type hints and docstring. Config values live in `configs/`, not in code.
