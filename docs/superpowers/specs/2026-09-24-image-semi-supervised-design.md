# Image Semi-Supervised Pipeline Design

**Date:** 2026-09-24  
**Author:** dg-ng  
**Status:** Approved — ready for implementation planning

---

## Overview

A semi-supervised image classification pipeline for 10 animal classes (5,500 images, 550/class), structured as a direct parallel to the existing text semi-supervised track (nb04, nb05, nb05b, nb08). Ground-truth labels are derived from folder names. The semi-supervised setting is simulated by using only 5% of images (~275 total, ~27–28/class) as the labeled seed.

**Dataset:** `data/image-data/<class>/` — 10 classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel.

**Backbone model:** `openai/clip-vit-base-patch32` (HuggingFace) — used for all embedding extraction and text-based labeling functions. CPU-compatible, 512-dim output.

---

## Notebook Structure

| Notebook | Purpose | Text parallel |
|---|---|---|
| `nb_img_00` | Data loading, manifest CSV, stratified 5% seed split | nb00 |
| `nb_img_01` | CLIP embedding extraction → `image_embeddings.npy` | nb01 |
| `nb_img_04` | Weak supervision (CLIP zero-shot LFs → Snorkel) | nb04 |
| `nb_img_05` | Pseudo-labeling — CLIP + MLP config A | nb05 |
| `nb_img_05b` | Pseudo-labeling — CLIP + MLP config B | nb05b |
| `nb_img_08` | Label propagation on CLIP embeddings | nb08 |
| `nb_img_07` | Cross-method comparison (ACC, F1, NMI) | nb07 |

The unsupervised track (clustering + description generation) is out of scope for this spec and will be covered separately.

---

## Data Flow

```
data/image-data/<class>/        (raw images, read-only)
        ↓ nb_img_00
data/image_manifest.csv         (filepath, true_label, label_idx, is_seed)
        ↓ nb_img_01
data/image_embeddings.npy       (5500 × 512, float32, row order matches manifest)
        ↓
┌──────────────────────────────────────────┐
│  nb_img_04    nb_img_05/05b    nb_img_08  │
│  weak sup     pseudo-label    label prop  │
└──────────────────────────────────────────┘
        ↓
results/image_<method>_predictions.csv     (filepath, true_label, predicted_label)
        ↓ nb_img_07
results/image_comparison.csv              (method, ACC, F1-macro, NMI)
```

**Seed split:** stratified, fixed `random_state=42`, exactly 5% per class (~27–28 images). `is_seed` column in manifest flags labeled rows.

---

## Method Details

### nb_img_04 — Weak Supervision

- For each of the 10 classes, define one CLIP-based labeling function (LF):
  - Compute cosine similarity between image embedding and text embedding of `"a photo of a {class}"`
  - If similarity ≥ threshold → assign label; else abstain (`-1`)
  - Threshold tuned per class using seed images
- Feed all 10 LF output matrices into Snorkel `LabelModel`
- `LabelModel` produces probabilistic labels for unlabeled set
- Train final MLP on probabilistic labels; evaluate on full dataset
- **Dependency:** `snorkel` (already in project), `transformers`

### nb_img_08 — Label Propagation

- Load `image_embeddings.npy`; extract seed rows as labeled nodes
- Build k-NN graph (k=15, cosine metric) via `sklearn`
- Run `sklearn.semi_supervised.LabelSpreading`
- Output: predicted label per image
- No new dependencies — identical pipeline to nb08, different embeddings

### nb_img_05 — Pseudo-Labeling (Config A)

- Train MLP (2 hidden layers, 256→128 units, ReLU) on 275 seed embeddings
- Predict on unlabeled set; retain predictions with confidence ≥ 0.9 as pseudo-labels
- Retrain MLP on seed + pseudo-labeled set
- Evaluate on full dataset (using true labels from manifest)
- Config A: confidence threshold = 0.9, MLP depth = 2 layers

### nb_img_05b — Pseudo-Labeling (Config B)

- Same pipeline as nb_img_05
- Config B: confidence threshold = 0.85, MLP depth = 3 layers (256→128→64)
- Mirrors the DistilBERT/ELECTRA split in the text track — two configs, same method

**Note:** Both pseudo-labeling notebooks use frozen CLIP embeddings as input — no full ViT fine-tuning. MLP training on ~275–2000 samples is seconds on CPU.

---

## Outputs

Each method produces:
- `results/image_<method>_predictions.csv` — `filepath, true_label, predicted_label`
- Printed metrics: ACC, F1-macro, NMI

nb_img_07 aggregates all into `results/image_comparison.csv` with a summary table matching nb07's format.

---

## Dependencies

| Package | Already in project? | Used by |
|---|---|---|
| `transformers` | Yes | CLIP model loading |
| `torch` (cpu) | Yes | CLIP forward pass, MLP training |
| `snorkel` | Yes | nb_img_04 LabelModel |
| `scikit-learn` | Yes | label propagation, k-NN graph |
| `Pillow` | Likely yes | image loading |
| `numpy` | Yes | embedding storage |

No new dependencies required.

---

## Out of Scope

- Unsupervised image track (KMeans/Agglom/DEC on CLIP embeddings) — separate spec
- Image description generation / BLIP captioning — separate spec
- Full ViT fine-tuning — excluded due to CPU-only constraint
- Data augmentation — not needed for embedding-based methods
