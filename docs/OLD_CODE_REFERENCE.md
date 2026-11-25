# Old Code Reference (CEREBRUS)

Index of the old code. Nothing here is ported. Port decisions, interfaces and structure come from the owner, case by case.

## Where it lives

| Ref | What | Local path (git-ignored) |
|---|---|---|
| branch `cerebrus-v0` (`b13eff8`) | Full CEREBRUS project at its final state | `old_code/cerebrus-v0/` |
| tag `cerebrus-v1-final` (`cc0e589`) | Last CEREBRUS commit before the reset (same tree, without the merged resume brief) | |
| branch `archive/historical-code-pre-retirement` (`134539e`) | Old V0/V1/V2 source (`older_code_base/`) | `old_code/archive-v0-v2/` |
| upstream TRM repo (not ours, not in git) | Vendored TinyRecursiveModels | `old_code/third_party/TinyRecursiveModels/` |
| non-code data, checkpoints, wandb, plots | Local only, ~25 GB | `old_data/` |

Read without a worktree: `git show cerebrus-v0:edge_glass_modular/src/<file>`.

## Docs about the old code (on `main`)
- `docs/cerebrus/AUDIT_LEDGER.md` — every cleanup/fix commit with root cause and validation.
- `docs/cerebrus/PROJECT_RESUME_CONTEXT.md`, `RESUME_PROJECT_BRIEF.md` — project summary and claims.
- Also on `cerebrus-v0` only: `AUDIT_REPORT.md` (claims vs code audit, release blockers), `edge_glass_modular/COMPLETE_CODEBASE.md`, `edge_glass_modular/README.md`, `notebooks/system_architecture.md`.

## Source map: `edge_glass_modular/src/` (paths relative to that folder)

| Path | Lines | What it is |
|---|---|---|
| `config.py` | 441 | Dataclass configs (`EncoderConfig`, `DecoderConfig`, `DatasetConfig`, `OptimizationConfig`, `TrainingConfig`, `ExperimentConfig`, …) and `load_config(path)` for YAML |
| `evaluation/qa_metrics.py` | 139 | `normalize_answer`, `compute_exact_match`, `compute_f1`, `compute_bleu`, `compute_rouge_l`, `evaluate_qa_metrics` |
| `evaluation/metrics.py`, `benchmark.py`, `zero_shot.py`, `explainability.py`, `templates.py` | 86–303 | Retrieval/alignment metrics, benchmark suite, zero-shot, explainability |
| `decoders/qwen.py` | 273 | `QwenDecoder`: Qwen load with 4-bit (`BitsAndBytesConfig`) and LoRA (`LoraConfig`, `get_peft_model`), `forward`, `generate`, `encode_text` |
| `decoders/trm.py` | 221 | TRM-inspired primitive layers |
| `models/trm_qwen_vlm.py` | 487 | `QwenVLM`: vision prefix + Qwen, y/z iterative refinement loop, `forward`, `generate`, `align_custom_modules` |
| `models/checkpoint_loader.py` | 184 | `load_vlm_checkpoint` (directory full-state, PEFT adapter dir, sharded safetensors, loose `.pt`) |
| `models/alignment.py`, `losses.py` | 215, 275 | Multimodal alignment model and losses (MRL, contrastive) |
| `encoders/` | 96–254 | `vision.py` (CLIP/ViT), `text.py`, `audio.py` (Whisper), `perceiver.py`, `mrl.py`, `pooling.py` |
| `data/` | 44–400 | `datamodule.py`, `dataset_builder.py`, `downloader.py`, `parquet_builder.py` (PixMo), `transforms.py` |
| `training/improved_trainer.py` | 591 | DDP-ready trainer (`ImprovedMultimodalTrainer`) |
| `utils/` | 26–488 | `distributed.py`, `logging.py`, `registry.py`, `visualization.py` |

Other folders under `edge_glass_modular/`:
- `configs/` — 9 YAMLs (TRM/no-TRM VLM QA on Qwen 1.5/1.8/2.5-3b, Perceiver+MRL alignment, PixMo alignment).
- `scripts/` — `train/train_vlm_accelerate.py`, `train/run_vlm_experiments.sh`, `eval/evaluate_vlm_accelerate.py`, `eval/inference_vlm.py`, `eval/evaluate_freeze_align.py`, `experiments/train_*.py`, `data/download_*.py`.
- `tests/` — 8 files (alignment forward/loss, best-checkpoint save, checkpoint loader, config loading, device placement, EOS stop, generation regression, Qwen prefix masking).
- `notebooks/`, `results/` (alignment, eda, train_history).

## Facts worth knowing before porting
- `qa_metrics.evaluate_qa_metrics` is defined twice in the same file (line 37 and line 126). The second one wins. It returns EM, F1, BLEU and ROUGE-L, each multiplied by 100.
- The old code has no bootstrap CI or stats utilities. A search of the `cerebrus-v0` code for "bootstrap" and confidence-interval functions found nothing, so anything of that kind is new work.
- `config.py` is mostly encoder/MRL/TRM/alignment/trainer dataclasses. Only a small part of it is generic YAML loading.
- 4-bit + LoRA loading is inside `QwenDecoder.__init__`, not a standalone helper.
- Known fixed bugs and their root causes are in `AUDIT_LEDGER.md` (checkpoint loading, generation double-strip, EOS stop, device placement).
- ROUGE-L's ×100 scale is easy to get wrong: any new metric code should keep it explicit.
