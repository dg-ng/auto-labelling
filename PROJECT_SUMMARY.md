# Auto-Labeling for Text Classification — AG News

**Project summary for presentation**
Date: 2026-08-16

---

## 1. Goal

Explore how far you can get at labeling news articles into 4 categories
(**World, Sports, Business, Sci/Tech**) **without full human annotation** —
comparing fully unsupervised methods, semi-supervised methods that use only a
small labeled seed set, and a fully supervised upper-bound baseline. Ground
truth labels are only ever used for *evaluation*, never given to the
unsupervised methods during training.

## 2. Dataset

**AG News** — a standard news classification benchmark.

| Property | Value |
|---|---|
| Classes | 4 (World, Sports, Business, Sci/Tech), balanced |
| Train rows | 120,000 (30,000/class) |
| Test rows | 7,600 (1,900/class) |
| Text used | `title + description` |

For semi-supervised methods, 5% of the train labels are kept (`LABEL_FRACTION
= 0.05`) and the rest are hidden (`label = -1`) to simulate a realistic
"small labeled seed set" scenario.

> **Scale note:** results below are from a **development-scale run** —
> embeddings/clustering used a stratified sample of 8,000 train rows (full
> 7,600-row test set always used in full); frozen RoBERTa embedding used 500
> rows; DistilBERT fine-tuning used 150 rows — chosen to keep CPU-only
> iteration fast. A larger-scale run is a documented follow-up (Section 6).

## 3. Pipeline / Tooling

- Environment: `uv` (`pyproject.toml` + `uv.lock`), pure CPU (no GPU).
- Shared `utils/` package (`config`, `data`, `embeddings`, `metrics`,
  `modeling`) imported by 8 sequential, independently re-runnable Jupyter
  notebooks (`00`–`07`), each caching its expensive outputs to disk.
- `SEED = 42` everywhere for reproducibility.

## 4. Approaches Implemented

### A. Unsupervised auto-labeling (no labels used during training)

Ground-truth labels are only used afterward, to score how well the discovered
clusters line up with the real categories (via Hungarian-matching for
accuracy).

| Method | Embedding | Clustering |
|---|---|---|
| TF-IDF + KMeans / HDBSCAN | Bag-of-words TF-IDF | KMeans (k=4) & HDBSCAN |
| MiniLM + KMeans / HDBSCAN | Sentence-transformer (`all-MiniLM-L6-v2`) | KMeans (k=4) & HDBSCAN |
| RoBERTa + KMeans / HDBSCAN | Frozen `roberta-base` (mean-pooled) | KMeans (k=4) & HDBSCAN |
| OpenAI + KMeans / HDBSCAN | `text-embedding-3-small` (optional, needs API key) | KMeans (k=4) & HDBSCAN |
| BERTopic | MiniLM (internal) | UMAP + HDBSCAN (internal), forced to 4 topics |

All embeddings are UMAP-reduced to 50 dims before clustering. Evaluated with
Hungarian-matched accuracy, NMI, ARI, FMI, homogeneity/completeness/V-measure,
silhouette, Davies-Bouldin, and coverage (fraction of points HDBSCAN didn't
call noise).

### B. Semi-supervised auto-labeling (5% labels guide the rest)

| Method | Idea |
|---|---|
| **Weak supervision** (Snorkel) | 6 hand-written labeling functions (keyword lists per class + regex heuristics for "has a %" → Business, "has a score like 3-2" → Sports) vote per example; Snorkel's generative `LabelModel` combines the noisy votes into probabilistic labels — no labeled data required to *fit* the label model, but the 5% seed is used to sanity-check LF empirical accuracy. |
| **Pseudo-labeling** (self-training) | Fine-tune DistilBERT on the small labeled seed → predict on the unlabeled pool → keep only high-confidence (≥0.90) predictions → add them to the labeled pool → repeat for 3 iterations → final fine-tune on the grown pool. |

### C. Baseline

| Method | Idea |
|---|---|
| **Full supervised** | Same DistilBERT architecture, fine-tuned on 100% of the (sampled) labeled train data — the upper-bound reference point everything else is measured against. |

## 5. Results (dev-scale run)

**Unsupervised — clustering quality vs. hidden ground truth**

| Method | ACC (Hungarian) | NMI | ARI | Coverage |
|---|---|---|---|---|
| **MiniLM + KMeans** | **0.830** | 0.620 | 0.629 | 1.00 |
| TF-IDF + KMeans | 0.808 | 0.530 | 0.567 | 1.00 |
| BERTopic | 0.717 | 0.661 | 0.613 | 0.64 |
| RoBERTa (frozen) + KMeans | 0.518 | 0.218 | 0.214 | 1.00 |
| MiniLM + HDBSCAN | 0.493 | 0.497 | 0.308 | 1.00 |
| TF-IDF + HDBSCAN | 0.262 | 0.004 | 0.00 | 0.96 |
| RoBERTa + HDBSCAN | 0.00 | 0.00 | 0.00 | 0.00 |
| OpenAI + KMeans / HDBSCAN | — | — | — | *pending — no API key configured for this run* |

**Semi-supervised & supervised — test-set performance**

| Method | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **Full supervised (100% labels)** | **0.862** | 0.862 | Upper bound |
| Pseudo-labeling (5% seed, self-trained) | 0.828 | 0.825 | Reaches ~96% of the full-supervised Macro F1 using only 5% of labels |
| Weak supervision (Snorkel, label quality only*) | — | — | Label Accuracy 0.464, Label Macro F1 0.444, Coverage 0.788 |

\*Weak supervision produces *pseudo-labels*, not a trained classifier, so it's
scored on label quality against hidden ground truth rather than test-set
accuracy.

### Key takeaways

1. **Sentence embeddings (MiniLM) + simple KMeans is a surprisingly strong,
   fully unsupervised baseline** — 83% clustering accuracy with zero labels,
   beating both TF-IDF and frozen RoBERTa embeddings.
2. **Frozen RoBERTa embeddings underperformed** MiniLM/TF-IDF here — likely
   because raw mean-pooled RoBERTa isn't optimized for semantic similarity
   the way a sentence-transformer is, combined with a much smaller sample
   size (500 vs 8,000 rows) for this method specifically.
3. **HDBSCAN was consistently worse than KMeans** on this data — it doesn't
   know there are exactly 4 classes and tends to either over-fragment or
   (in RoBERTa's case) call everything noise. KMeans' assumption of `k=4`
   fits AG News's known, balanced class count well.
4. **Pseudo-labeling with only 5% labeled data reached ~96% of the fully
   supervised Macro F1** (0.825 vs 0.862) — the semi-supervised self-training
   loop captured most of the value of full labeling at a fraction of the
   annotation cost.
5. **Weak supervision (keyword heuristics) had good coverage (79%) but low
   label accuracy (46%)** — hand-written keyword rules are noisy and not by
   themselves a substitute for a trained classifier, though they're a
   reasonable zero-labeled-data starting point.

## 6. Limitations / Next Steps

- **Dev-scale sampling**: current numbers use 8,000/500/150-row samples (not
  the full 120,000/7,600), chosen to keep CPU-only iteration fast. A
  larger-scale or full-data run is scoped and costed out already (see
  `docs/superpowers/plans/2026-08-15-autolabel-notebooks.md`, "Final Run"
  section) — estimated **~5–10+ days unattended** for a literal full-data run
  on this CPU-only machine, vs. a few hours for a "scaled-up bounded" run
  (e.g. 20–30k rows) that would be far more representative than the current
  numbers while staying practical.
- **OpenAI embeddings** were not run in this dataset snapshot (no API key
  configured) — rows show as "pending."
- A known fix is needed before scaling up: `silhouette_score`/
  `davies_bouldin_score` are O(n²) and must be sub-sampled before running at
  much larger scale.

## 7. Repo Map

```
utils/            shared code: config, data loading/splitting, embeddings
                   (TF-IDF/MiniLM/RoBERTa/OpenAI, disk-cached), metrics,
                   modeling (fine-tuning + pseudo-label self-training)
notebooks/
  00_data_transform.ipynb          raw CSV → cleaned/split parquet
  01_embeddings.ipynb              generate & cache all embedding methods
  02_unsupervised_clustering.ipynb KMeans/HDBSCAN over each embedding method
  03_bertopic.ipynb                BERTopic topic discovery
  04_weak_supervision.ipynb        Snorkel labeling functions + LabelModel
  05_pseudo_labeling.ipynb         DistilBERT self-training loop
  06_full_supervised_baseline.ipynb fully supervised upper bound
  07_comparison.ipynb              assembles results/comparison_table.csv
results/comparison_table.csv       final results table (source for Section 5)
autolabel_project_spec.md          original project spec (approach/metric defs)
docs/superpowers/                  implementation plan + design doc
```
