# Cross-Method Cluster Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/10_cross_method_alignment.ipynb` that projects all six clustering methods' centroids into a shared UMAP space and measures pairwise alignment via cosine similarity and Hungarian matching.

**Architecture:** Concatenate L2-normalised MiniLM (384-dim) and RoBERTa (768-dim) train embeddings per article to form a 1,152-dim combined vector; fit one UMAP to obtain a 50-dim shared space; compute all 96 cluster centroids (6 methods × 16 clusters) in that space; build a 96×96 cosine similarity matrix; extract 15 pairwise 16×16 sub-matrices for heatmaps and Hungarian matching.

**Tech Stack:** Python 3.10+, numpy, umap-learn, scikit-learn, scipy, pandas, matplotlib, seaborn

**Spec:** `docs/superpowers/specs/2026-09-12-cross-method-alignment-design.md`

## Global Constraints

- Python ≥ 3.10; all dependencies already in `pyproject.toml` (`umap-learn`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`, `scipy`)
- `random_state=42` on every UMAP call — no exceptions
- `N_CLUSTERS = 16` — use the constant; never hardcode 16 inline
- Notebook must run to completion via `uv run jupyter nbconvert --to notebook --execute notebooks/10_cross_method_alignment.ipynb --output-dir notebooks` without error
- No model re-runs — all inputs are pre-computed `.npy` files in `embeddings_cache/` and `results/`
- Figures saved to `results/`; `results/` is gitignored, so the notebook is the artefact tracked in git
- No new utility modules — all logic lives inside the notebook cells
- `scipy.optimize.linear_sum_assignment` for Hungarian matching (scipy is a transitive dependency)
- **Spec correction:** use `minilm_train_full_summary.npy` and `roberta_train_full_summary.npy` (both 3,824 rows), not the `test` variants (957 rows). Cluster labels are (3824,).
- Notebook follows existing repo style: first cell = markdown description, second cell = imports + REPO_ROOT path setup + data loading, helper functions in their own cells, config constants at the top of cell 2

---

### Task 1: Scaffold, data loading, and shared UMAP (cells 1–3)

**Files:**
- Create: `notebooks/10_cross_method_alignment.ipynb`

**Interfaces:**
- Produces: `shared` — numpy array shape (3824, 50), the shared UMAP space used by all later tasks

- [ ] **Step 1: Create the notebook file with cell 1 (markdown description)**

Write `notebooks/10_cross_method_alignment.ipynb` as a valid Jupyter notebook
JSON. Cell 1 is a markdown cell:

```markdown
# 10 — Cross-Method Cluster Alignment

Projects all six unsupervised clustering methods into a **shared 50-dim UMAP space**
by concatenating L2-normalised MiniLM (384-dim) and RoBERTa (768-dim) train embeddings
per article (→ 1,152-dim), then fitting one UMAP on the combined matrix.

In this shared space, every cluster centroid — regardless of which method produced it —
is comparable via cosine similarity. The result is a 96×96 similarity matrix
(6 methods × 16 clusters), from which we extract 15 pairwise 16×16 alignment heatmaps
and apply Hungarian matching to find the best 1-to-1 cluster pairing across methods.

Inputs (pre-computed, no re-running):
- `embeddings_cache/minilm_train_full_summary.npy`  — (3824, 384)
- `embeddings_cache/roberta_train_full_summary.npy` — (3824, 768)
- `results/cluster_labels_{name}.npy`               — (3824,) for each of 6 methods

Outputs:
- `results/centroid_alignment_summary.csv`  — 15-row summary table
- `results/alignment_heatmap_all.png`       — 5×3 grid of 16×16 heatmaps
- `results/centroid_scatter_2d.png`         — 2D scatter of all 96 centroids
```

- [ ] **Step 2: Add cell 2 — imports, config, data loading**

```python
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import config

# ── Config ───────────────────────────────────────────────────────────────────
METHODS = [
    "minilm_kmeans", "roberta_kmeans",
    "minilm_agglomerative", "roberta_agglomerative",
    "minilm_dec", "roberta_dec",
]
N_CLUSTERS = 16
UMAP_SHARED_DIM = 50
UMAP_VIZ_DIM = 2
STRONG_ALIGN_THRESHOLD = 0.5

# ── Load embeddings ───────────────────────────────────────────────────────────
minilm_emb = np.load(REPO_ROOT / "embeddings_cache" / "minilm_train_full_summary.npy")
roberta_emb = np.load(REPO_ROOT / "embeddings_cache" / "roberta_train_full_summary.npy")

assert minilm_emb.shape == (3824, 384), f"Unexpected MiniLM shape: {minilm_emb.shape}"
assert roberta_emb.shape == (3824, 768), f"Unexpected RoBERTa shape: {roberta_emb.shape}"

# L2-normalise each before concatenating so neither family dominates by scale
minilm_norm = normalize(minilm_emb.astype(np.float32))   # (3824, 384)
roberta_norm = normalize(roberta_emb.astype(np.float32)) # (3824, 768)
combined = np.hstack([minilm_norm, roberta_norm])         # (3824, 1152)

# ── Load cluster labels ───────────────────────────────────────────────────────
labels = {}
for name in METHODS:
    path = REPO_ROOT / "results" / f"cluster_labels_{name}.npy"
    assert path.exists(), f"Missing: {path} — run notebook 02 first"
    arr = np.load(path).astype(int)
    assert arr.shape == (3824,), f"{name}: unexpected label shape {arr.shape}"
    assert arr.min() >= 0 and arr.max() < N_CLUSTERS, \
        f"{name}: label values out of range [0, {N_CLUSTERS})"
    labels[name] = arr

print(f"Combined shape: {combined.shape}")
print(f"Loaded labels for: {list(labels.keys())}")
```

- [ ] **Step 3: Add cell 3 — fit shared UMAP (this takes ~2 min, print progress)**

```python
print("Fitting shared UMAP on 3824 × 1152 matrix …")
reducer = umap.UMAP(
    n_components=UMAP_SHARED_DIM,
    metric="cosine",
    random_state=42,
    verbose=True,
)
shared = reducer.fit_transform(combined)  # (3824, 50)
assert shared.shape == (3824, UMAP_SHARED_DIM), f"Unexpected shared shape: {shared.shape}"
print(f"Shared UMAP complete: {shared.shape}")
```

- [ ] **Step 4: Smoke-test the notebook runs to cell 3 without error**

Run only the first 3 cells:
```
uv run jupyter nbconvert --to notebook --execute notebooks/10_cross_method_alignment.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600
```
Expected: exits with code 0, notebook written to `notebooks/10_cross_method_alignment.ipynb`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/10_cross_method_alignment.ipynb
git commit -m "feat(nb10): scaffold + data loading + shared UMAP (cells 1-3)"
```

---

### Task 2: Centroid computation, 96×96 similarity matrix, and heatmaps (cells 4–6)

**Files:**
- Modify: `notebooks/10_cross_method_alignment.ipynb` (append cells 4–6)

**Interfaces:**
- Consumes: `shared` (3824, 50), `labels` dict, `METHODS`, `N_CLUSTERS` — from Task 1
- Produces:
  - `centroids` — dict: method name → np.ndarray (16, 50)
  - `all_centroids` — np.ndarray (96, 50), rows ordered as METHODS × cluster 0–15
  - `method_index` — dict: method name → (start_row, end_row) slice into `all_centroids`
  - `sim_matrix` — np.ndarray (96, 96) cosine similarity, symmetric, diagonal ≈ 1.0
  - `results/alignment_heatmap_all.png` saved to disk

- [ ] **Step 1: Add cell 4 — compute centroids**

```python
def compute_centroids(shared_emb: np.ndarray, cluster_labels: np.ndarray,
                      n_clusters: int) -> np.ndarray:
    """
    Compute mean centroid per cluster in the shared embedding space.

    Args:
        shared_emb: (N, D) float32 — shared UMAP space
        cluster_labels: (N,) int — cluster assignments, values in [0, n_clusters)
        n_clusters: int — number of clusters (16)

    Returns:
        centroids: (n_clusters, D) float32 — one centroid per cluster
    """
    D = shared_emb.shape[1]
    centroids = np.zeros((n_clusters, D), dtype=np.float32)
    for k in range(n_clusters):
        mask = cluster_labels == k
        if mask.sum() == 0:
            # empty cluster: centroid stays zero (will have zero cosine sim with all)
            continue
        centroids[k] = shared_emb[mask].mean(axis=0)
    return centroids

centroids = {}
for name in METHODS:
    centroids[name] = compute_centroids(shared, labels[name], N_CLUSTERS)
    assert centroids[name].shape == (N_CLUSTERS, UMAP_SHARED_DIM), \
        f"{name}: centroid shape mismatch {centroids[name].shape}"

print("Centroid shapes OK.")
for name, c in centroids.items():
    print(f"  {name}: {c.shape}")
```

- [ ] **Step 2: Add cell 5 — build 96×96 cosine similarity matrix**

```python
# Stack all centroids in METHODS order: 6 methods × 16 clusters = 96 rows
all_centroids = np.vstack([centroids[name] for name in METHODS])  # (96, 50)
assert all_centroids.shape == (len(METHODS) * N_CLUSTERS, UMAP_SHARED_DIM)

# Build index: method name → (start, end) row slice
method_index = {}
for i, name in enumerate(METHODS):
    method_index[name] = (i * N_CLUSTERS, (i + 1) * N_CLUSTERS)

# L2-normalise each centroid row, then dot-product → cosine similarity
all_centroids_norm = normalize(all_centroids)   # (96, 50)
sim_matrix = all_centroids_norm @ all_centroids_norm.T  # (96, 96)

# Sanity: diagonal should be ≈ 1.0, matrix should be symmetric
assert np.allclose(np.diag(sim_matrix), 1.0, atol=1e-5), "Diagonal not ≈ 1.0"
assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-6), "Matrix not symmetric"
print(f"sim_matrix shape: {sim_matrix.shape}, range: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")
```

- [ ] **Step 3: Add cell 6 — 15 pairwise heatmaps in a 5×3 grid**

Note: `combinations` is already imported in cell 2. `pairs` is defined here and reused in cell 7 (Task 3) — cells share kernel state.

```python
# pairs defined here; cell 7 reuses it without recomputing
pairs = list(combinations(METHODS, 2))   # 15 pairs
assert len(pairs) == 15

fig, axes = plt.subplots(5, 3, figsize=(18, 28))
axes_flat = axes.flatten()

for ax, (method_a, method_b) in zip(axes_flat, pairs):
    sa, ea = method_index[method_a]
    sb, eb = method_index[method_b]
    sub = sim_matrix[sa:ea, sb:eb]   # (16, 16)

    # Hungarian matching: maximise similarity → minimise negative similarity
    row_ind, col_ind = linear_sum_assignment(-sub)
    mean_best = float(sub[row_ind, col_ind].mean())
    strong = int((sub[row_ind, col_ind] > STRONG_ALIGN_THRESHOLD).sum())

    sns.heatmap(
        sub,
        ax=ax,
        vmin=0, vmax=1,
        cmap="YlOrRd",
        cbar=True,
        xticklabels=range(N_CLUSTERS),
        yticklabels=range(N_CLUSTERS),
        linewidths=0,
    )

    # Annotate the Hungarian-matched pairs with white squares
    for r, c in zip(row_ind, col_ind):
        ax.add_patch(plt.Rectangle(
            (c, r), 1, 1,
            fill=False, edgecolor="white", lw=2,
        ))

    ax.set_title(
        f"{method_a}\nvs {method_b}\n"
        f"mean best-match={mean_best:.3f}  strong={strong}/{N_CLUSTERS}",
        fontsize=9,
    )
    ax.set_xlabel(method_b, fontsize=8)
    ax.set_ylabel(method_a, fontsize=8)
    ax.tick_params(labelsize=6)

fig.suptitle(
    "Cross-Method Cluster Alignment\n(cosine similarity of centroids in shared UMAP space)",
    fontsize=14, y=1.01,
)
plt.tight_layout()

out_path = REPO_ROOT / "results" / "alignment_heatmap_all.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
```

- [ ] **Step 4: Verify heatmap file is created**

After running the notebook to cell 6:
```python
import os
assert os.path.exists("results/alignment_heatmap_all.png"), "Heatmap not saved"
```
(This check lives in the next cell; confirm the file appears in `results/`.)

- [ ] **Step 5: Commit**

```bash
git add notebooks/10_cross_method_alignment.ipynb
git commit -m "feat(nb10): centroid computation + 96×96 similarity + heatmaps (cells 4-6)"
```

---

### Task 3: Summary table, 2D scatter, final verification, and execution (cells 7–9)

**Files:**
- Modify: `notebooks/10_cross_method_alignment.ipynb` (append cells 7–9)

**Interfaces:**
- Consumes: `sim_matrix`, `method_index`, `METHODS`, `N_CLUSTERS`, `STRONG_ALIGN_THRESHOLD`, `all_centroids` — from Tasks 1–2
- Produces:
  - `results/centroid_alignment_summary.csv` — 15 rows, 4 columns
  - `results/centroid_scatter_2d.png`

- [ ] **Step 1: Add cell 7 — summary table**

```python
rows = []
for method_a, method_b in pairs:
    sa, ea = method_index[method_a]
    sb, eb = method_index[method_b]
    sub = sim_matrix[sa:ea, sb:eb]   # (16, 16)

    row_ind, col_ind = linear_sum_assignment(-sub)
    matched_sims = sub[row_ind, col_ind]
    rows.append({
        "method_a": method_a,
        "method_b": method_b,
        "mean_best_match": float(matched_sims.mean()),
        "strong_alignments": int((matched_sims > STRONG_ALIGN_THRESHOLD).sum()),
    })

summary_df = pd.DataFrame(rows)
assert summary_df.shape == (15, 4), f"Expected (15,4), got {summary_df.shape}"
assert list(summary_df.columns) == ["method_a", "method_b", "mean_best_match", "strong_alignments"]

out_csv = REPO_ROOT / "results" / "centroid_alignment_summary.csv"
summary_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
print(summary_df.sort_values("mean_best_match", ascending=False).to_string(index=False))
```

- [ ] **Step 2: Add cell 8 — 2D centroid scatter**

```python
# Fit a second UMAP on the 96 centroid vectors (50-dim → 2-dim) for visualisation
viz_reducer = umap.UMAP(n_components=UMAP_VIZ_DIM, metric="cosine", random_state=42)
centroids_2d = viz_reducer.fit_transform(all_centroids_norm)  # (96, 2)

# Colour by method (6 distinct colours)
METHOD_COLORS = {
    "minilm_kmeans":        "#1f77b4",
    "roberta_kmeans":       "#ff7f0e",
    "minilm_agglomerative": "#2ca02c",
    "roberta_agglomerative":"#d62728",
    "minilm_dec":           "#9467bd",
    "roberta_dec":          "#8c564b",
}

fig, ax = plt.subplots(figsize=(12, 9))
for i, name in enumerate(METHODS):
    start = i * N_CLUSTERS
    end = (i + 1) * N_CLUSTERS
    xs = centroids_2d[start:end, 0]
    ys = centroids_2d[start:end, 1]
    color = METHOD_COLORS[name]
    ax.scatter(xs, ys, c=color, s=80, label=name, alpha=0.85, zorder=3)
    for k, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x + 0.03, y + 0.03, str(k), fontsize=7, color=color, alpha=0.9)

ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.set_title(
    "2D UMAP of All 96 Cluster Centroids\n(coloured by method, labelled by cluster ID)",
    fontsize=12,
)
ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")
plt.tight_layout()

out_scatter = REPO_ROOT / "results" / "centroid_scatter_2d.png"
fig.savefig(out_scatter, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {out_scatter}")
```

- [ ] **Step 3: Add cell 9 — final verification cell**

```python
import os

checks = [
    ("results/centroid_alignment_summary.csv", True),
    ("results/alignment_heatmap_all.png",      True),
    ("results/centroid_scatter_2d.png",         True),
]
for path, expected in checks:
    full = REPO_ROOT / path
    exists = full.exists()
    status = "✓" if exists == expected else "✗"
    print(f"{status} {path}")

# Sanity: same-family pairs should have higher mean_best_match than cross-family
same_family = summary_df[
    summary_df["method_a"].str.split("_").str[0] ==
    summary_df["method_b"].str.split("_").str[0]
]["mean_best_match"].mean()
cross_family = summary_df[
    summary_df["method_a"].str.split("_").str[0] !=
    summary_df["method_b"].str.split("_").str[0]
]["mean_best_match"].mean()
print(f"\nSame-family mean best-match:  {same_family:.3f}")
print(f"Cross-family mean best-match: {cross_family:.3f}")
assert same_family > cross_family, (
    f"Sanity check FAILED: same-family ({same_family:.3f}) ≤ cross-family ({cross_family:.3f}). "
    "The shared UMAP space may not be meaningful."
)
print("Sanity check passed ✓")
```

- [ ] **Step 4: Execute the complete notebook end-to-end**

From the repo root:
```bash
uv run jupyter nbconvert --to notebook --execute \
    notebooks/10_cross_method_alignment.ipynb \
    --output-dir notebooks \
    --ExecutePreprocessor.timeout=600
```

Expected: exits code 0. All three output files appear in `results/`. Sanity check prints "Sanity check passed ✓".

If UMAP takes >10 min total, that is unexpected — check that `umap-learn` is installed in the venv (`uv run python -c "import umap; print(umap.__version__)"`).

- [ ] **Step 5: Commit final notebook**

```bash
git add notebooks/10_cross_method_alignment.ipynb
git commit -m "feat(nb10): summary table + 2D scatter + final verification (cells 7-9)"
```
