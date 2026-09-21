> **Historical snapshot — not the current canonical architecture.** Archived with the pre-retirement
> notebooks; superseded by current source, `tests/`, and `AUDIT_LEDGER.md`.

# Quick Start Guide - Modular VLM Training

## TL;DR

```bash
# 1. Update config with your dataset paths
vim edge_glass_modular/configs/trm_vlm_qa.yaml

# 2. Open the modular notebook
jupyter notebook edge_glass_modular/notebooks/03_trm_vlm_qa_training_MODULAR.ipynb

# 3. Run all cells
# Done!
```

## What You Need to Update

### REQUIRED: Validation and Test Dataset Paths

Edit [edge_glass_modular/configs/trm_vlm_qa.yaml](../configs/trm_vlm_qa.yaml):

```yaml
dataset:
  train_parquet: /home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pimo-alignment/pixmo_qa_mixed_with_bytes.parquet
  val_parquet: /path/to/your/validation.parquet  # ← UPDATE THIS
  test_parquet: /path/to/your/test.parquet  # ← UPDATE THIS
```

## Files Overview

| File | Purpose |
|------|---------|
| [`03_trm_vlm_qa_training_MODULAR.ipynb`](./03_trm_vlm_qa_training_MODULAR.ipynb) | **Main notebook** - Run this! |
| [`TRM_VLM_QA_MODULAR_README.md`](./TRM_VLM_QA_MODULAR_README.md) | Detailed documentation |
| [`REFACTORING_SUMMARY.md`](./REFACTORING_SUMMARY.md) | What changed and why |
| [`QUICK_START.md`](./QUICK_START.md) | This file |
| [`../configs/trm_vlm_qa.yaml`](../configs/trm_vlm_qa.yaml) | Configuration file |

## Architecture at a Glance

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Vision Encoder  │ ← From src/encoders/vision.py (Frozen)
│ (CLIP+Perceiver)│
└──────┬──────────┘
       │ (B, 64, 4096)
       ▼
┌─────────────────┐
│Vision Projection│ ← Trainable linear layer
└──────┬──────────┘
       │ (B, 64, qwen_dim)
       │
       ├─────┐ Question tokens (B, L_q)
       │     │
       ▼     ▼
┌────────────────────┐
│   Qwen Decoder     │ ← From src/decoders/qwen.py (LoRA trainable)
│   with LoRA        │
└─────────┬──────────┘
          │
          ▼
    Answer tokens
```

## Dataset Structure

**Training Dataset**: `pixmo_qa_mixed_with_bytes.parquet` (12,000 samples)

| Column | Type | Example |
|--------|------|---------|
| `image_bytes` | bytes | Raw PNG/JPEG bytes |
| `question` | str | "[USER]Why is John Travolta wearing..." |
| `answer` | str | "This image is likely from a movie..." |
| `source` | str | "pixmo-cap-qa" |

## Modular Components Used

| Component | File | Purpose |
|-----------|------|---------|
| `PixmoQADataset` | `src/data/dataset_builder.py` | Load QA pairs from parquet |
| `VisionEncoder` | `src/encoders/vision.py` | Encode images (CLIP+Perceiver+MRL) |
| `QwenDecoder` | `src/decoders/qwen.py` | Qwen2.5 with LoRA |
| `get_image_transforms` | `src/data/transforms.py` | Image preprocessing |
| `load_config` | `src/config.py` | Load YAML config |

## Training Parameters (Default)

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-4 |
| LoRA rank | 32 |
| Max question tokens | 128 |
| Max answer tokens | 256 |
| Vision encoder | Frozen |
| Decoder quantization | 8-bit |

## Expected Output

```
Loaded config: trm_vlm_qa

Vision Encoder:
  Model: openai/clip-vit-large-patch14
  Projection dim: 4096
  Use Perceiver: True
  Trainable params: 0

Qwen Decoder:
  Model: Qwen/Qwen2.5-7B-Instruct
  Use LoRA: True
  Trainable params: 42,893,312

Dataset:
  Train samples: 12,000

Training Setup:
  Num epochs: 3
  Learning rate: 2e-4
  Total steps: 2,250

Epoch 1/3: 100%|████| 750/750 [12:34<00:00, loss=2.12, lr=1.5e-4]
✓ Saved best checkpoint (loss: 2.1234)
```

## Common Issues

### OOM Error
**Solution**: Reduce batch size in config:
```yaml
dataset:
  batch_size: 8  # Reduce from 16
```

### Dataset Not Found
**Solution**: Check the path in config matches your dataset location:
```bash
ls -la /home/hice1/vchopra37/scratch/projects/edge_glass/dataset/final_dataset/pimo-alignment/pixmo_qa_mixed_with_bytes.parquet
```

### Import Errors
**Solution**: Make sure you're in the correct directory and added `src/` to path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
```

## Next Steps After Training

1. **Load checkpoint**:
   ```python
   checkpoint = torch.load("checkpoints/trm_vlm_qa/checkpoint_best.pt")
   ```

2. **Generate answers**:
   ```python
   generated_ids = decoder.generate(
       input_ids=question_ids,
       prefix_embeds=vision_prefix,
       max_new_tokens=256,
   )
   ```

3. **Evaluate** (once you have val/test sets):
   - Exact Match (EM)
   - F1 Score
   - BLEU, ROUGE, METEOR

## Experiment Ideas

1. **Try different decoders**:
   ```yaml
   decoder:
     model_name: "meta-llama/Llama-2-7b-chat-hf"
   ```

2. **Unfreeze vision encoder**:
   ```yaml
   vision_encoder:
     freeze: false
   ```

3. **Adjust LoRA rank**:
   ```yaml
   decoder:
     lora_r: 64  # Higher = more trainable params
   ```

4. **Use 4-bit quantization** (less memory):
   ```yaml
   decoder:
     load_in_8bit: false
     load_in_4bit: true
   ```

## Questions?

1. **Read**: [TRM_VLM_QA_MODULAR_README.md](./TRM_VLM_QA_MODULAR_README.md) for detailed docs
2. **Check**: [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md) for what changed
3. **Review**: Config at `configs/trm_vlm_qa.yaml`
4. **Inspect**: Source code in `src/encoders/`, `src/decoders/`, `src/data/`

## Summary

- ✅ Use `03_trm_vlm_qa_training_MODULAR.ipynb`
- ✅ Update `configs/trm_vlm_qa.yaml` with your dataset paths
- ✅ Run all cells
- ✅ Monitor training with W&B or console logs
- ✅ Checkpoints saved to `checkpoints/trm_vlm_qa/`
- ✅ All components imported from `src/` (modular!)

Happy training! 🚀
