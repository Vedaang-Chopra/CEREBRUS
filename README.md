# AdaLoop-VLM

Adaptive Looped VLM: a controller chooses visual-token budget `k` and loop count `R` on the
varloop backbone. Reported as accuracy vs FLOPs, plus safety. Spec: `docs/proposal/Proposal_v2.docx`.

- Previous project (CEREBRUS): branch `cerebrus-v0`, tag `cerebrus-v1-final`.
- CEREBRUS audit notes: `docs/cerebrus/`.
- Runs happen on the Inara GPU host, not locally. Setup: `conda env create -f environment.yml`.

Layout: `adaloop/{load,hooks}.py`, `adaloop/{controller,data,eval}/`, `tests/`, `configs/`, `notebooks/`, `scripts/`.
