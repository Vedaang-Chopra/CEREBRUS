# AGENTS.md — AdaLoop-VLM

Project overrides only. Universal rules: `~/agent-governance/AGENTS.md` (read it first; do not duplicate it here).

## Project
AdaLoop-VLM: a looped VLM (varloop backbone) whose loop state decides after each pass to exit, loop again, or look again;
then post-trained (PT-1 LoRA-insert, PT-2 question-aware encoder QAFE, PT-3 joint RL). Reported as accuracy vs FLOPs and
measured latency, plus safety. **Master reference: `docs/IMPLEMENTATION.md`** (wins on conflicts; index in `docs/README.md`).
Controller-feature cost is counted in FLOPs from day one.

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
tests/              pytest; one test file per ported or new piece
configs/            YAML configs
notebooks/          inspection only, no production logic
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
