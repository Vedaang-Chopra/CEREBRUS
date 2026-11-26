# Figures and tables plan (Dec 3 paper)

Every paper figure is made by a notebook, from saved results, through one plotting function in the package.
Notebooks show the figure inline and save it. Small-sample runs save drafts; full runs save the paper version.

**How a figure is made**
- Plot code lives in `adaloop/core/plots/` (one small function per figure: takes a DataFrame, returns a matplotlib `fig`), exposed through `adaloop.api`.
- `api.save_figure(fig, "F05_kr_surface", final=False)` writes PDF + PNG: drafts to `execution_results/figures/`, finals (`final=True`, only from full runs) to `paper/figures/`.
- One shared style (`api.set_paper_style()`): one font, consistent colours per method across all figures (controller, heuristic, each baseline, oracle always the same colour), colour-blind safe, readable at one-column width.
- Tables: `api.save_table(df, "T1_headline")` → CSV + LaTeX in `paper/tables/` (finals) or `execution_results/tables/` (drafts).

## Figures

| ID | Figure | Paper section | Made in | Data | Main / appendix |
|---|---|---|---|---|---|
| F01 | Method diagram: prelude → core pass → controller {exit, loop, look} → coda | Method | hand-made (not a notebook) | — | main |
| F02 | Thinking-off: parse rate + accuracy per prefill variant; 768 vs 1024 px on TextVQA | Setup | `04_thinking_off` | nb04 results | appendix |
| F03 | Pass-state check: cosine and answer agreement of pass p (in R=5) vs fresh R=p, per p | Setup (tree assumption) | `05_positions_and_loops` | nb05 table | appendix |
| F04 | Cost model: FLOPs per (k, R) heatmap; analytic vs measured time ratio | Cost | `06_full_baseline` (S6) | flops + timing | appendix |
| F05 | Accuracy vs R at full tokens, per benchmark (the loop-depth sweep) | Measurement | `06_full_baseline` | full baseline | main |
| F06 | Compression comparison at R=3, k ∈ {25, 50, 100}%: accuracy vs FLOPs per method (random, uniform, text_mean, lowres, crisp_lite, ReDiPrune, DivPrune, VisionZip) | Measurement | S5 notebook | selector runs | appendix |
| F07 | k × R accuracy surface (heatmap) per benchmark | Measurement | S7 grid notebook | `grid.parquet` | main |
| F08 | Accuracy vs FLOPs for the 15 cells, with the Pareto front | Measurement | S7 grid notebook | `grid.parquet` | main |
| F09 | Headroom: oracle per cost cap vs best fixed vs noise oracle | Measurement | S14 analysis notebook | grid + reruns | main |
| F10 | **Iso-cost curves**: controller, heuristic, 4 baselines, oracle (accuracy vs mean FLOPs), headline cost marked | Results (headline) | S14 analysis notebook | `policy_eval.parquet` | main |
| F11 | Base vs LoRA: surface difference (LoRA − base) and controller curve on each | Post-training | S10/S14 notebooks | base + lora grids | main |
| F12 | Policy behaviour: exit-pass and look-rate histograms per benchmark and GQA question type | Analysis | S12/S14 notebook | `policy_eval.parquet` | main |
| F13 | Ablations A1–A6 (bars with CIs) | Ablations | S15 notebook | ablations | main |
| F14 | Qualitative examples: image, question, path taken, answers (exit-early correct, look rescued, look wasted, loop helped) | Analysis | S15 notebook | tree + items | appendix |

## Tables

| ID | Table | Made in |
|---|---|---|
| T1 | Headline at 0.5 × cost_ref: accuracy (mean ± std over seeds), Δ vs best fixed, 95% CI, p-value, prereg verdict | S14 |
| T2 | Headroom per benchmark (oracle, best fixed, noise oracle, headroom) | S14 |
| T3 | Baselines at matched cost | S14 |
| T4 | Ablations A1–A6 | S15 |
| T5 | Per-question-type accuracy (GQA) | S15 |
| T6 | Setup and reproducibility (model, precision BF16, GPUs, hours, seeds, hyperparameters) | S16 |

## Rules
- A notebook that produces a figure has a `### Step-N:- Figure Fxx` section near the end: build the DataFrame, call the plot function, show it, save it.
- Small-sample notebooks (01–05 series) save drafts only. Paper finals come from full runs.
- Changing a figure means changing its plot function; notebooks stay thin.
