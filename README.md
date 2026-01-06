# Project CEREBRUS: Edge Glass Assistant

**Production-ready, highly modular multimodal alignment framework (Vision-Lanuage Models)**

This repository contains the complete research and implementation for **Project Cerebrus (Edge Glass)**, an advanced multimodal agent framework designed for efficient training and inference on edge-compatible architectures.

---

## 📚 Key Resources

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Research Paper** | The full scientific paper detailing the methodology and results. | [📄 **CEREBRUS.pdf**](CEREBRUS.pdf) |
| **Production Code** | The final, modular, production-ready codebase. | [💻 **edge_glass_modular/**](edge_glass_modular/) |
| **Final Status** | Detailed status report confirming completion of all objectives. | [✅ **FINAL_STATUS.md**](FINAL_STATUS.md) |
| **Project Summary** | Comprehensive 18-page overview of the entire project. | [📖 **PROJECT_SUMMARY.md**](PROJECT_SUMMARY.md) |
| **Implementation** | Technical guide for the modular architecture. | [🛠️ **IMPLEMENTATION_GUIDE.md**](IMPLEMENTATION_GUIDE.md) |

---

## 🏗️ Codebase Structure

The project has been consolidated into a single, modular package.

### ✅ `edge_glass_modular/` (Primary)
**This is the only code you need.** It contains:
- **Source Code**: Fully modularized encoders (Vision, Audio, Text), decoders (Qwen, TRM), and fusion modules.
- **Training**: Distributed Data Parallel (DDP) trainer, checkpointing, and mixed precision support.
- **Configs**: YAML-based configuration for all 10 experiments.
- **Documentation**: Its own standalone `README.md` and setup guides.

### 🗃️ Archived Code
The following directories contain legacy code and are **no longer needed**:
- `v2_code_base/`: Early research prototypes.
- `v3_code_base/`: Interim development version.

---

## 🚀 Quick Start

To start working with the project, navigate to the modular codebase and install it:

```bash
# 1. Enter the production codebase
cd edge_glass_modular

# 2. Install dependencies
pip install -e .

# 3. Follow the specific guide
cat README.md
```

For detailed instructions on running experiments (e.g., Vision-Text Alignment, Tri-Modal Training), refer to **[edge_glass_modular/README.md](edge_glass_modular/README.md)**.

---

## 🏆 Project Highlights

- **Matryoshka Representation Learning (MRL)**: Multi-scale embeddings for flexible retrieval.
- **Perceiver Resampler**: Efficient compression of high-dimensional inputs (Audio/Vision).
- **Modular Decoders**: Support for state-of-the-art **Qwen-14B** and custom lightweight **TRM** models.
- **Production Infrastructure**: Built for stability on H100/A100 clusters with full crash recovery.

---

*Generated for Project Cerebrus.*