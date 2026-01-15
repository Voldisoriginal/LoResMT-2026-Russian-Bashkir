# DevLake: Pre-training and Model Scale in Low-Resource Translation (LoResMT 2026)

[![ODS.ai Competition](https://img.shields.io/badge/Competition-ODS.ai-0057ff?style=flat&logo=kaggle)](https://ods.ai/competitions/ru-bashkir-lores-mt)
[![Task](https://img.shields.io/badge/Task-Russian%20to%20Bashkir-orange)](https://ods.ai/competitions/ru-bashkir-lores-mt)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the implementation of the **DevLake** system for the **[Russian-Bashkir Machine Translation (LoResMT 2026)](https://ods.ai/competitions/ru-bashkir-lores-mt)** shared task hosted on ODS.ai. Our submission explores the trade-offs between fine-tuning massive multilingual models and training compact architectures from scratch.

Our best system achieved a **CHRF++ score of 52.67**, securing a top position on the leaderboard.

## 📄 Abstract

Bashkir is a low-resource Turkic language with rich morphology. We conducted a comparative study of three architectures:
1.  **NLLB-200 (1.3B):** Fine-tuned via QLoRA.
2.  **M2M-100 (418M):** Fine-tuned via LoRA.
3.  **MarianMT (77M):** Full fine-tuning with vocabulary expansion.

We demonstrate that combining semantic data filtering (using BERT) with parameter-efficient fine-tuning of large models significantly outperforms traditional compact baselines.

## 🏆 Results

| System | Architecture | Params | Method | CHRF++ | Hugging Face Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DevLake-Large** | **NLLB-200** | **1.3B** | **QLoRA** | **52.67** | [Link](https://huggingface.co/Voldis/nllb-1.3b-rus-bak) |
| DevLake-Medium | M2M-100 | 418M | LoRA | 48.80 | [Link](https://huggingface.co/Voldis/m2m100-rus-bak) |
| DevLake-Small | MarianMT | 77M | Full FT | 43.15 | [Link](https://huggingface.co/Voldis/marian-rus-bak) |

## 🛠️ Installation

1. Install PyTorch with CUDA support (follow instructions at [pytorch.org](https://pytorch.org/get-started/locally/)).
2. Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Reproduction

### 1. Data Filtering
We filter the dataset using a domain-specific BERT model to remove noise and misalignment.
```bash
python scripts/data_filtering.py
```

### 2. Training
Run the training script for the desired architecture:

**System 1: NLLB-1.3B (Best Performance)**
```bash
python scripts/train_nllb_1.3b.py
```

**System 2: M2M-100**
```bash
python scripts/train_m2m_418m.py
```

**System 3: MarianMT**
```bash
python scripts/train_marian_77m.py
```

## 📚 Citation

If you use this code or models, please cite our paper:

```bibtex
@article{devlake2026loresmt,
  title={DevLake at LoResMT 2026: The Impact of Pre-training and Model Scale on Russian-Bashkir Low-Resource Translation},
  author={DevLake Team},
  journal={Proceedings of LoResMT 2026},
  year={2026}
}