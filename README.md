# fashion-forge

AI-powered fashion design studio.  
Describe a garment in plain text — the model retrieves visually similar references and generates 3 novel design proposals using RAG + Conditional StyleGAN2-lite.

---

## Quick start

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Dataset

**Option A — Fashion Product Images via HuggingFace**

```bash
python download_dataset.py
```

**Option B — Fashion-MNIST**

```bash
# No setup needed — the dataset is downloaded automatically:
python train_local.py --dataset fmnist
```

Fashion-MNIST is greyscale 28x28. Good for quickly verifying the pipeline,
but output images will be less detailed than the colour dataset.

---

## Training

```bash
# Real colour dataset, GPU
python train_local.py

# Quick CPU smoke-test
python train_local.py --dataset fmnist --epochs 5 --size 32 --batch 8

# Show all options
python train_local.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 15 | Number of training epochs |
| `--size` | 64 | Output image resolution in pixels (32 / 64 / 128) |
| `--batch` | 16 | Batch size — reduce to 8 if you run out of memory on CPU |
| `--dataset` | fashion | `fashion` (HuggingFace) or `fmnist` (auto-download) |
| `--no-eval` | off | Skip post-training evaluation |
| `--interactive` | off | Launch the interactive terminal loop after training |

The trained checkpoint is saved to `output/fastfashion_checkpoint.pt`.

---

## Interactive terminal loop

```bash
python train_local.py --interactive

# Or, if you already have a checkpoint:
python train_local.py --epochs 0 --interactive
```

---

## Web app

```bash
# Checkpoint at output/fastfashion_checkpoint.pt
mkdir -p static && cp index.html static/
python app.py
```

### How it works

1. Type a garment description in the text field
2. Press **Generate** (or Enter)
3. Receive 3 AI-generated design proposals
4. Click your favourite, rate each with stars, and submit
5. All feedback is saved to `output/human_feedback.json`

### UI preview

```
┌──────────────────────────────────────────────────────────┐
│ fashion-forge           RAG + Conditional StyleGAN2-lite │
├──────────────────────────────────────────────────────────┤
│  Describe your garment:                                  │
│  [elegant red evening dress with floral embroidery ...]  │
│                                              [Generate]  │
├──────────────────────────────────────────────────────────┤
│  3 Proposals — "elegant red evening dress..."            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │          │  │ SELECTED │  │          │                │
│  │ Proposal │  │ Proposal │  │ Proposal │                │
│  │    1     │  │    2  ✓  │  │    3     │                │
│  └──────────┘  └──────────┘  └──────────┘                │ 
│  [Submit feedback]  [Skip]                               │
└──────────────────────────────────────────────────────────┘
```

---

## Evaluation metrics

Computed automatically after training and saved to `output/eval/eval_report.json`.

| Metric             | Measures                                      |
|--------------------|-----------------------------------------------|
| **FID**            | Generated vs real distribution quality        |
| **IS**             | Sharpness + class diversity (Inception Score) |
| **CLIP Score**     | Semantic alignment: query ↔ generated image   |
| **Diversity**      | Pairwise CLIP distance (anti mode-collapse)   |
| **RAG Precision**  | Relevance of retrieved reference garmentsv    |

---

## Architecture

```
Text query
    │
    ▼
CLIP ViT-B/32 (text encoder)
    │  512-dim embedding
    ▼
FAISS IndexFlatIP ──── Top-K reference garments
    │                        │
    │              mean CLIP embeddings
    │                        │
    └────────────────────────┘
             style_cond [1, 512]
                    │
                    ▼
         MappingNetwork (LayerNorm + 4× Linear)
                    │  w [B, 256]
                    ▼
         Const 4×4 → GenBlock (AdaIN) ×4 → 64×64
                    │
                    ▼
              RGB image [-1, 1]
```

**Discriminator**: 4× DiscBlock (stride-2) + MinibatchStd + R1 regularisation  
**Training loss**: Non-saturating GAN loss + R1 gradient penalty (γ=10)  
**Conditioning**: CLIP embedding injected at every generator layer via AdaIN

---

## License

© 2026 Michael Corelli. All rights reserved.

This repository is publicly visible.
No licence is granted to use, copy, modify, or distribute this code.

---

## Author

Michael Corelli