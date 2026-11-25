# AGENTS.md — AdaLoop-VLM

Project overrides only. Universal rules: `~/agent-governance/AGENTS.md` (read it first; do not duplicate it here).

## Project
Adaptive Looped VLM: a controller picks visual-token budget `k` and loop count `R` on the
varloop backbone. Reported as accuracy vs FLOPs, plus safety. Spec: `docs/proposal/Proposal_v2.docx`.
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
adaloop/            package: load.py, hooks.py, controller/, data/, eval/
tests/              pytest; one test file per ported or new piece
configs/            YAML configs
notebooks/          inspection only, no production logic
scripts/            thin CLI entry points
docs/proposal/      Proposal_v2.docx, GRA research questions
docs/cerebrus/      AUDIT_LEDGER.md, PROJECT_RESUME_CONTEXT.md, RESUME_PROJECT_BRIEF.md
docs/literature/    paper classification CSVs, reading notes, model cards (varloop card here)
pyproject.toml, environment.yml
```

## Local-only (git-ignored, never commit)
- `old_data/` — old checkpoints, wandb, plots, logs, old notebooks (26 GB).
- `edge_glass_modular/` — local copy of old modular code; reference for porting only.
- `Papers/` — PDFs (~500 MB). Canonical set: `Papers/Complete Papers List/Shortlisted_Literature/` (P0–P3 tiers; `Paper_Classification.csv` maps each ID to its file).

## Code rules
- Port from old code only these: `qa_metrics`, bootstrap CI/stats utilities, trimmed config loading, LoRA/4-bit loading helpers. Do not port encoders, MRL, Perceiver, alignment, TRM, trainers.
- Every ported file keeps a provenance comment (`# Ported from cerebrus-v0:<path> @ <commit>`).
- `qa_metrics`: ROUGE-L is multiplied by 100.
- Gate 0 order: loop control test (logits R=1 vs R=3 differ) → readout-equivalence check (pass r == fresh run at R=r) → only then any grid run.
- Code does not run locally. All runs happen on the Inara GPU host (`conda env create -f environment.yml`). Run outputs go to `execution_results/` (ignored).
- Public functions: type hints and docstring. Config values live in `configs/`, not in code.
