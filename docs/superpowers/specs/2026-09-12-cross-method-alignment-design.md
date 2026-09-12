# Cross-Method Cluster Alignment — Design Spec

**Date:** 2026-09-12
**Notebook:** `notebooks/10_cross_method_alignment.ipynb`

---

## Goal

Measure how well clusters produced by different embedding models and
clustering algorithms agree on the same groups of news articles, using
centroid similarity in a shared UMAP embedding space.

## Motivation

The six unsupervised methods (KMeans/Agglomerative/DEC × MiniLM/RoBERTa)
each produce 16 clusters with arbitrary IDs. MiniLM centroids are 384-dim
and RoBERTa centroids are 768-dim — incommensurable spaces. To ask "does
MiniLM KMeans cluster 3 and RoBERTa DEC cluster 7 cover the same topic?"
we need a single shared embedding space where all centroids can be compared.

---

## Architecture

### Shared Embedding Space

For each of the 3,824 test articles, concatenate its L2-normalised MiniLM
embedding (384-dim) and its L2-normalised RoBERTa embedding (768-dim) into
a single 1,152-dim vector. Fit one UMAP on those 3,824 × 1,152 vectors to
produce a 50-dim shared space.

```
article i → [L2(minilm_i) | L2(roberta_i)]  (1152-dim)
                    ↓ UMAP(n_components=50, metric='cosine', random_state=42)
article i → shared_i  (50-dim)
```

Every cluster centroid — regardless of which of the six methods produced it
— is the mean of its member articles' shared-space vectors. Cosine
similarity between any two centroids is then a meaningful cross-family
metric.

### Centroid Matrix

96 centroids total (6 methods × 16 clusters). Stack into a (96 × 50)
matrix, L2-normalise each row, then compute the full (96 × 96) cosine
similarity matrix via matrix multiply.

### Alignment Per Method Pair

For each of the 15 method pairs (6 choose 2), extract the (16 × 16)
sub-matrix. This is the pairwise alignment heatmap. Apply Hungarian
matching (scipy.optimize.linear_sum_assignment on the negated sub-matrix)
to find the globally optimal 1-to-1 cluster pairing and compute:

- **Mean best-match similarity**: average cosine similarity of the 16
  matched pairs.
- **Strong alignments**: number of matched pairs with similarity > 0.5.

---

## Data Inputs

| File | Shape | Notes |
|---|---|---|
| `embeddings_cache/minilm_test_full_summary.npy` | (3824, 384) | MiniLM embeddings, test split |
| `embeddings_cache/roberta_test_full_summary.npy` | (3824, 768) | RoBERTa embeddings, test split |
| `results/cluster_labels_{name}.npy` | (3824,) | One per method, values 0–15 |

Six label files: `minilm_kmeans`, `roberta_kmeans`, `minilm_agglomerative`,
`roberta_agglomerative`, `minilm_dec`, `roberta_dec`.

---

## Outputs

| File | Description |
|---|---|
| `results/centroid_alignment_summary.csv` | 15-row table: method_a, method_b, mean_best_match, strong_alignments |
| `results/alignment_heatmap_all.png` | 5×3 grid of 15 heatmaps (one per method pair), 16×16 each, annotated with matched diagonal from Hungarian |
| `results/centroid_scatter_2d.png` | 2D UMAP of all 96 centroids; coloured by method, labelled by cluster ID |

---

## Notebook Structure

### Cell 1 — Imports & config
```python
METHODS = [
    "minilm_kmeans", "roberta_kmeans",
    "minilm_agglomerative", "roberta_agglomerative",
    "minilm_dec", "roberta_dec",
]
N_CLUSTERS = 16
UMAP_SHARED_DIM = 50
UMAP_VIZ_DIM = 2
STRONG_ALIGN_THRESHOLD = 0.5
```

### Cell 2 — Load embeddings & cluster labels
Load both `.npy` embedding files. L2-normalise each. Concatenate to
(3824, 1152). Load all six label arrays.

### Cell 3 — Fit shared UMAP
```python
reducer = umap.UMAP(n_components=UMAP_SHARED_DIM, metric='cosine',
                    random_state=42)
shared = reducer.fit_transform(combined)   # (3824, 50)
```

### Cell 4 — Compute centroids
For each method and each cluster ID 0–15, compute the mean of member rows
in `shared`. Result: dict mapping method name → (16, 50) array.

### Cell 5 — Build 96×96 cosine similarity matrix
Stack all centroid arrays → (96, 50). L2-normalise. Dot product →
(96, 96) similarity matrix.

### Cell 6 — Per-pair alignment heatmaps (5×3 grid)
Iterate over 15 method pairs. Extract (16, 16) sub-matrix. Run Hungarian
matching. Plot heatmap with seaborn; annotate the matched diagonal with
white squares. Save `results/alignment_heatmap_all.png`.

### Cell 7 — Summary table
Build 15-row DataFrame with columns: method_a, method_b, mean_best_match,
strong_alignments. Print and save `results/centroid_alignment_summary.csv`.

### Cell 8 — 2D centroid scatter
Fit a second UMAP(n_components=2) on the 96 × 50 centroid matrix. Scatter
plot; colour by method (6 colours); label each point with its cluster ID.
Save `results/centroid_scatter_2d.png`.

---

## Global Constraints

- Python ≥ 3.10; all dependencies already in `pyproject.toml`
  (`umap-learn`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`, `scipy`)
- `random_state=42` on all UMAP calls
- `N_CLUSTERS = 16` — never hardcode 16 inline; use the constant
- Notebook must be runnable via `uv run jupyter nbconvert --to notebook
  --execute` without error
- No model re-runs — all inputs are pre-computed `.npy` files
- Figures saved to `results/`; `results/` is gitignored, so the notebook
  itself is the artefact tracked in git
- No new utility modules required; all logic lives in the notebook
- `scipy.optimize.linear_sum_assignment` for Hungarian matching (already
  available via scipy, a transitive dependency)

---

## Success Criteria

1. Notebook executes end-to-end without error.
2. `results/centroid_alignment_summary.csv` has exactly 15 rows and 4
   columns (`method_a`, `method_b`, `mean_best_match`, `strong_alignments`).
3. `results/alignment_heatmap_all.png` exists and shows a 5×3 grid.
4. `results/centroid_scatter_2d.png` exists.
5. Same-family pairs (e.g. minilm_kmeans vs minilm_agglomerative) show
   higher mean best-match than cross-family pairs — a sanity check that
   the shared space is meaningful.
