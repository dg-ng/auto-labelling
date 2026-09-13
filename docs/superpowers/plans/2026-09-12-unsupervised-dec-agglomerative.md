# Unsupervised Track — DEC + Agglomerative Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HDBSCAN/BERTopic/TF-IDF in the unsupervised clustering track with Agglomerative Clustering and Deep Embedded Clustering (DEC), producing 6 clean outcomes: KMeans × {MiniLM, RoBERTa}, Agglomerative × {MiniLM, RoBERTa}, DEC × {MiniLM, RoBERTa} — all on generated summaries, all with explicit k=16.

**Architecture:** Pre-trained dense embeddings (MiniLM 384-dim, RoBERTa 768-dim) serve as input to all three clustering algorithms. KMeans and Agglomerative both use UMAP(n_components=50) reduction first (same as the existing notebook 02 pipeline). DEC learns its own latent space (encoder MLP → 64-dim bottleneck) and refines cluster centers jointly via KL divergence — no UMAP needed. All 6 outcomes save metric JSON files + full-row CSVs + cluster label arrays for notebook 09.

**Tech Stack:** Python 3, PyTorch (CPU), scikit-learn (KMeans, AgglomerativeClustering), UMAP (umap-learn), sentence-transformers (MiniLM cache), transformers (RoBERTa cache), Jupyter notebooks, existing `utils/` helpers.

**Spec:** `docs/superpowers/specs/2026-09-05-master-data-pipeline-design.md` (see also: chat discussion 2026-09-12 finalizing the 6-outcome design and removing auto-k methods).

## Global Constraints

- CPU-only machine (Intel UHD 620, no CUDA) — all PyTorch ops run on CPU; `torch.no_grad()` everywhere inference happens.
- Python env: `.venv/` at repo root; activate before running notebooks or scripts.
- Embeddings are already cached in `embeddings_cache/` as `{method}_train_full_summary.npy` (MiniLM = 384-dim, RoBERTa = 768-dim) — never regenerate them.
- `config.SEED = 42` — pass to every stochastic call: KMeans, UMAP, AgglomerativeClustering (no randomness, but document it), DEC torch/numpy seeds.
- `config.NUM_CLASSES = 16` — this is k for every method; never hardcode 16, always use `config.NUM_CLASSES`.
- Every clustering method saves: `results/metrics_{name}.json`, `results/full_labels_{name}.csv` (text, summary, predicted_label, true_label), `results/cluster_labels_{name}.npy` (raw integer array, length = n_train rows), `results/clusters_{name}.csv` (KeyBERT-based interpretability table).
- Method names (exact, case-sensitive): `minilm_kmeans`, `roberta_kmeans`, `minilm_agglomerative`, `roberta_agglomerative`, `minilm_dec`, `roberta_dec`.
- Drop from active methods: `tfidf_kmeans`, `tfidf_hdbscan`, `minilm_hdbscan`, `roberta_hdbscan`, `openai_kmeans`, `openai_hdbscan`. The BERTopic notebook (03) is kept as-is but removed from the comparison.
- `utils/dec.py` is a new file — do not modify any existing `utils/` files in Tasks 1–3.
- Notebook cells must be runnable top-to-bottom with no hidden state — each cell block re-imports what it uses.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `utils/dec.py` | **Create** | DEC encoder, autoencoder pretraining, DEC refinement loop, `train_dec()` entry point |
| `notebooks/02_unsupervised_clustering.ipynb` | **Rebuild** | KMeans + Agglomerative + DEC × MiniLM + RoBERTa; saves all 6 outcome artifacts |
| `notebooks/09_cluster_similarity.ipynb` | **Rebuild** | Load cluster labels from `results/cluster_labels_*.npy`; compute cosine similarity stats; heatmaps + bar chart |
| `notebooks/07_comparison.ipynb` | **Modify** | Remove hardcoded OPENAI_METHODS pending rows; auto-picks up new methods from `results/metrics_*.json` |
| `PROJECT_SUMMARY.md` | **Modify** | Update Section 3.2 flow diagram, Section 4.1 approach text, Section 5.1 results table, Section 9 limitations, Section 10 repo map |
| `results/metrics_tfidf_*.json` etc. | **Delete** | Remove stale artifacts from dropped methods so notebook 07 stays clean |

---

## Task 1: Add `utils/dec.py` — Deep Embedded Clustering

**Files:**
- Create: `utils/dec.py`

**Interfaces:**
- Produces:
  - `train_dec(embeddings: np.ndarray, n_clusters: int, latent_dim: int = 64, pretrain_epochs: int = 50, dec_epochs: int = 150, batch_size: int = 256, lr_pretrain: float = 1e-3, lr_dec: float = 1e-4, update_interval: int = 5, tol: float = 1e-3, seed: int = 42) -> np.ndarray` — returns integer cluster label array of shape `(n_samples,)`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_dec.py`:
```python
import numpy as np
import pytest
from utils.dec import train_dec


def test_train_dec_output_shape_and_range():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((120, 32)).astype(np.float32)  # 120 rows, 32-dim
    labels = train_dec(
        emb,
        n_clusters=4,
        latent_dim=16,
        pretrain_epochs=3,
        dec_epochs=10,
        batch_size=32,
        seed=0,
    )
    assert labels.shape == (120,), f"expected (120,), got {labels.shape}"
    assert labels.dtype in (np.int32, np.int64), f"expected int dtype, got {labels.dtype}"
    assert set(labels).issubset(set(range(4))), f"labels out of range: {set(labels)}"


def test_train_dec_reproducible():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((80, 16)).astype(np.float32)
    a = train_dec(emb, n_clusters=3, latent_dim=8, pretrain_epochs=2, dec_epochs=5, seed=7)
    b = train_dec(emb, n_clusters=3, latent_dim=8, pretrain_epochs=2, dec_epochs=5, seed=7)
    np.testing.assert_array_equal(a, b, err_msg="DEC not reproducible with same seed")
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd C:\Users\ACER\OneDrive\Documents\final-project
.venv\Scripts\activate
pytest tests/test_dec.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils.dec'`

- [ ] **Step 3: Write `utils/dec.py`**

```python
"""
Deep Embedded Clustering (DEC) — Xie et al., 2016.

Adapted for pre-trained dense embeddings (MiniLM / RoBERTa) as input:
skips the raw-input-to-feature-extractor step and directly learns a
compressed latent space (MLP encoder, 2-layer) plus cluster centers.

Entry point: train_dec(embeddings, n_clusters, ...) -> np.ndarray of int labels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def _build_encoder(input_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),
    )


def _build_decoder(input_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, input_dim),
    )


# ---------------------------------------------------------------------------
# Soft assignment and target distribution
# ---------------------------------------------------------------------------

def _soft_assignment(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Student's t-distribution soft assignment q_ij.

    Args:
        z: (N, latent_dim) latent representations
        centers: (K, latent_dim) cluster center parameters
        alpha: degrees of freedom (1 by default as in Xie et al.)

    Returns:
        q: (N, K) soft assignments, each row sums to 1
    """
    diff = z.unsqueeze(1) - centers.unsqueeze(0)          # (N, K, D)
    dist_sq = (diff ** 2).sum(dim=2)                        # (N, K)
    q = (1.0 + dist_sq / alpha) ** (-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True)


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Sharpened target distribution p_ij.

    p_ij = (q_ij^2 / sum_i q_ij) / (sum_j q_ij^2 / sum_i q_ij)
    """
    numerator = q ** 2 / q.sum(dim=0, keepdim=True)        # (N, K)
    return numerator / numerator.sum(dim=1, keepdim=True)   # (N, K)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _pretrain_autoencoder(
    X: torch.Tensor,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> nn.Sequential:
    """Train an MLP autoencoder on X; return the encoder half."""
    torch.manual_seed(seed)
    input_dim = X.shape[1]
    encoder = _build_encoder(input_dim, latent_dim)
    decoder = _build_decoder(input_dim, latent_dim)
    optim = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_seen = 0
        for (xb,) in loader:
            z = encoder(xb)
            recon = decoder(z)
            loss = F.mse_loss(recon, xb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)
            n_seen += len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  [DEC pretrain] epoch {epoch + 1}/{epochs}  "
                  f"recon_loss={epoch_loss / n_seen:.5f}")
    return encoder


def train_dec(
    embeddings: np.ndarray,
    n_clusters: int,
    latent_dim: int = 64,
    pretrain_epochs: int = 50,
    dec_epochs: int = 150,
    batch_size: int = 256,
    lr_pretrain: float = 1e-3,
    lr_dec: float = 1e-4,
    update_interval: int = 5,
    tol: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """
    Train Deep Embedded Clustering on pre-computed dense embeddings.

    Pipeline:
      1. Pretrain MLP autoencoder on `embeddings` with MSE reconstruction loss.
      2. Run KMeans on the encoder's latent space to initialise K cluster centers.
      3. Jointly refine encoder weights and cluster centers by minimising
         KL(P || Q) where Q is the soft assignment and P is the sharpened
         target distribution.
      4. Return hard cluster labels (argmax of final Q) as an int array.

    Args:
        embeddings: (N, D) float32 array of pre-trained embedding vectors.
        n_clusters: number of clusters k (must set explicitly; DEC does not
                    discover k automatically).
        latent_dim: bottleneck size for the encoder MLP.
        pretrain_epochs: autoencoder pretraining epochs.
        dec_epochs: maximum DEC refinement epochs.
        batch_size: mini-batch size for both pretraining and refinement.
        lr_pretrain: Adam learning rate for autoencoder pretraining.
        lr_dec: Adam learning rate for DEC refinement.
        update_interval: recompute P (and check convergence) every N epochs.
        tol: stop early if fraction of label changes < tol.
        seed: random seed for torch, numpy, and KMeans.

    Returns:
        labels: (N,) int32 array of cluster assignments in [0, n_clusters).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(embeddings, dtype=torch.float32)

    # 1. Pretrain encoder via autoencoder
    print(f"[DEC] Pretraining autoencoder ({pretrain_epochs} epochs)...")
    encoder = _pretrain_autoencoder(
        X, latent_dim=latent_dim, epochs=pretrain_epochs,
        batch_size=batch_size, lr=lr_pretrain, seed=seed,
    )

    # 2. KMeans initialisation in latent space
    print("[DEC] Initialising cluster centers with KMeans...")
    encoder.eval()
    with torch.no_grad():
        Z_init = encoder(X).numpy()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    km.fit(Z_init)
    centers = nn.Parameter(
        torch.tensor(km.cluster_centers_, dtype=torch.float32)
    )

    # 3. DEC refinement: KL(P || Q)
    print(f"[DEC] Refinement (up to {dec_epochs} epochs, tol={tol})...")
    optim_dec = Adam(list(encoder.parameters()) + [centers], lr=lr_dec)
    loader_ordered = DataLoader(
        TensorDataset(X), batch_size=batch_size, shuffle=False
    )

    prev_labels: np.ndarray | None = None
    final_labels: np.ndarray = km.labels_.astype(np.int32)

    for epoch in range(dec_epochs):
        # Recompute P from full Q every update_interval epochs
        if epoch % update_interval == 0:
            encoder.eval()
            with torch.no_grad():
                q_all = _soft_assignment(encoder(X), centers)
                p_all = _target_distribution(q_all).detach()
            curr_labels = q_all.argmax(dim=1).numpy().astype(np.int32)
            final_labels = curr_labels

            if prev_labels is not None:
                delta = float((curr_labels != prev_labels).mean())
                print(f"  [DEC] epoch {epoch:4d}  label-change delta={delta:.4f}")
                if delta < tol:
                    print(f"  [DEC] Converged (delta < {tol}) at epoch {epoch}.")
                    break
            prev_labels = curr_labels.copy()

        # Mini-batch KL divergence update
        encoder.train()
        for i, (xb,) in enumerate(loader_ordered):
            start = i * batch_size
            end = start + len(xb)
            p_b = p_all[start:end]
            q_b = _soft_assignment(encoder(xb), centers)
            # KL(P||Q): sum_j p_j * log(p_j / q_j)
            loss = F.kl_div(q_b.log(), p_b, reduction="batchmean")
            optim_dec.zero_grad()
            loss.backward()
            optim_dec.step()

    print(f"[DEC] Done. {len(np.unique(final_labels))} non-empty clusters.")
    return final_labels
```

- [ ] **Step 4: Run the tests**

```
pytest tests/test_dec.py -v
```
Expected output:
```
tests/test_dec.py::test_train_dec_output_shape_and_range PASSED
tests/test_dec.py::test_train_dec_reproducible PASSED
```

If `test_train_dec_reproducible` fails: the `DataLoader(shuffle=True)` in pretraining uses a seeded Generator but the encoder state still varies slightly by batch order across runs — fix by also seeding `torch.manual_seed(seed)` at the very top of `train_dec()` before creating the DataLoader (already in the code above; verify it's there).

- [ ] **Step 5: Commit**

```
git add utils/dec.py tests/test_dec.py
git commit -m "feat: add utils/dec.py — Deep Embedded Clustering for unsupervised track

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01StACqfxGJJkBzxocBh6gxU"
```

---

## Task 2: Rebuild `notebooks/02_unsupervised_clustering.ipynb`

**Files:**
- Modify: `notebooks/02_unsupervised_clustering.ipynb`

**Interfaces:**
- Consumes:
  - `embeddings_cache/minilm_train_full_summary.npy` — shape (3824, 384)
  - `embeddings_cache/roberta_train_full_summary.npy` — shape (3824, 768)
  - `data/processed/train_clean.parquet` — columns: text, summary, label
  - `utils/dec.train_dec(embeddings, n_clusters, ...) -> np.ndarray`
  - `utils/metrics.evaluate_unsupervised(true_labels, cluster_labels, embeddings, ...) -> dict`
  - `utils/metrics.hungarian_match_predictions(true_labels, cluster_labels) -> np.ndarray`
  - `utils/interpretability.summarize_clusters(summaries, cluster_labels, true_labels, emb_reduced, class_names) -> pd.DataFrame`
  - `utils/samples.save_full_output(texts, predicted, true_labels, class_names, extra_columns, path)`
- Produces (for each of the 6 method names):
  - `results/metrics_{name}.json`
  - `results/full_labels_{name}.csv`
  - `results/clusters_{name}.csv`
  - `results/cluster_labels_{name}.npy` — raw integer cluster labels (NEW; used by notebook 09)

**Active methods (exact names used as dict keys and file name suffixes):**
```python
METHODS = ["minilm", "roberta"]
CLUSTERERS = ["kmeans", "agglomerative", "dec"]
# → 6 combinations: minilm_kmeans, minilm_agglomerative, minilm_dec,
#                   roberta_kmeans, roberta_agglomerative, roberta_dec
```

- [ ] **Step 1: Delete the old notebook content**

Open `notebooks/02_unsupervised_clustering.ipynb` and clear all cells (or delete and recreate). The new notebook has 7 cells total.

- [ ] **Step 2: Write Cell 0 — markdown header**

```markdown
# 02 — Unsupervised Clustering (v2)

KMeans, Agglomerative Clustering, and Deep Embedded Clustering (DEC) over
MiniLM and RoBERTa **summary** embeddings — 6 outcomes total
(3 clusterers × 2 embedding methods), all with explicit k=16.

KMeans and Agglomerative use UMAP(n_components=50, metric="cosine") reduction
first. DEC learns its own 64-dim latent space (MLP encoder) and does not need
UMAP.

Every method saves:
- `results/metrics_{name}.json` — clustering quality metrics
- `results/cluster_labels_{name}.npy` — raw integer cluster assignments
- `results/clusters_{name}.csv` — per-cluster KeyBERT interpretation table
- `results/full_labels_{name}.csv` — every row with predicted/true label
```

- [ ] **Step 3: Write Cell 1 — imports and data loading**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering

from utils import config
from utils.dec import train_dec
from utils.embeddings import load_cached
from utils.interpretability import summarize_clusters
from utils.metrics import evaluate_unsupervised, hungarian_match_predictions
from utils.samples import save_full_output

train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
true_labels = train_clean["label"].to_numpy()
texts = train_clean["text"].tolist()
summaries = train_clean["summary"].tolist()

suffix = "full"  # embeddings cached with this suffix (no sample cap)

METHODS = ["minilm", "roberta"]
embeddings_by_method = {}
for method in METHODS:
    arr = load_cached(f"{method}_train_{suffix}_summary")
    assert arr is not None, (
        f"Missing cached embeddings for '{method}' — run 01_embeddings.ipynb first"
    )
    embeddings_by_method[method] = arr
    print(f"Loaded {method}: shape {arr.shape}")

print(f"\nRows: {len(train_clean)} | k={config.NUM_CLASSES}")
```

- [ ] **Step 4: Write Cell 2 — UMAP reduction helper**

```python
def umap_reduce(emb: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    """Reduce embeddings to 50-dim cosine-space via UMAP (same params as prior notebook)."""
    reducer = umap.UMAP(n_components=50, metric="cosine", random_state=seed)
    return reducer.fit_transform(emb)

# Pre-reduce for methods that need it (KMeans, Agglomerative).
# DEC skips UMAP — it learns its own latent representation.
print("Running UMAP reduction for KMeans / Agglomerative...")
reduced_by_method = {}
for method, emb in embeddings_by_method.items():
    print(f"  UMAP {method}...")
    reduced_by_method[method] = umap_reduce(emb)
    print(f"  {method} reduced: {reduced_by_method[method].shape}")
print("UMAP done.")
```

- [ ] **Step 5: Write Cell 3 — KMeans and Agglomerative clustering**

```python
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
all_results = {}  # method_name -> metrics dict (for display)

def run_and_save(name: str, cluster_labels: np.ndarray, emb_original: np.ndarray) -> None:
    """Evaluate, interpret, and save all artifacts for one clustering result."""
    metrics = evaluate_unsupervised(
        true_labels, cluster_labels, emb_original,
        metric_sample_size=config.SILHOUETTE_SAMPLE_SIZE, seed=config.SEED,
    )
    all_results[name] = metrics

    # Save metrics JSON
    with open(config.RESULTS_DIR / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save raw cluster label array (used by notebook 09)
    np.save(config.RESULTS_DIR / f"cluster_labels_{name}.npy", cluster_labels)

    # Cluster interpretation table (KeyBERT)
    interp = summarize_clusters(
        summaries, cluster_labels, true_labels,
        emb_original,  # pass original embedding for centroid proximity, not UMAP-reduced
        config.CLASS_NAMES,
    )
    if not interp.empty:
        interp.to_csv(config.RESULTS_DIR / f"clusters_{name}.csv", index=False)

    # Full-row output CSV
    predicted = hungarian_match_predictions(true_labels, cluster_labels)
    save_full_output(
        texts, predicted, true_labels, config.CLASS_NAMES,
        extra_columns={"summary": summaries},
        path=config.RESULTS_DIR / f"full_labels_{name}.csv",
    )

    print(f"  {name}: ACC={metrics['ACC (Hungarian)']:.4f}  "
          f"F1={metrics['Macro F1']:.4f}  "
          f"NMI={metrics['NMI']:.4f}  "
          f"coverage={metrics['Coverage']:.2f}")


# ── KMeans ──────────────────────────────────────────────────────────────────
print("=== KMeans (k={}) ===".format(config.NUM_CLASSES))
for method in METHODS:
    emb_red = reduced_by_method[method]
    emb_orig = embeddings_by_method[method]
    km = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
    labels = km.fit_predict(emb_red)
    run_and_save(f"{method}_kmeans", labels, emb_orig)

# ── Agglomerative Clustering ─────────────────────────────────────────────────
print("\n=== Agglomerative Clustering (k={}, ward linkage) ===".format(config.NUM_CLASSES))
for method in METHODS:
    emb_red = reduced_by_method[method]
    emb_orig = embeddings_by_method[method]
    agg = AgglomerativeClustering(
        n_clusters=config.NUM_CLASSES,
        linkage="ward",         # minimises variance within merges; good for dense embeddings
        metric="euclidean",     # ward requires euclidean
    )
    labels = agg.fit_predict(emb_red)
    run_and_save(f"{method}_agglomerative", labels, emb_orig)
```

- [ ] **Step 6: Write Cell 4 — DEC clustering**

```python
# ── Deep Embedded Clustering (DEC) ──────────────────────────────────────────
# DEC learns its own latent space — no UMAP reduction needed.
# Uses raw (pre-UMAP) MiniLM/RoBERTa embeddings directly.
print("=== Deep Embedded Clustering (DEC, k={}) ===".format(config.NUM_CLASSES))
for method in METHODS:
    emb_orig = embeddings_by_method[method]
    print(f"\n[DEC] Training on {method} embeddings (shape {emb_orig.shape})...")
    labels = train_dec(
        emb_orig,
        n_clusters=config.NUM_CLASSES,
        latent_dim=64,
        pretrain_epochs=50,
        dec_epochs=150,
        batch_size=256,
        lr_pretrain=1e-3,
        lr_dec=1e-4,
        update_interval=5,
        tol=1e-3,
        seed=config.SEED,
    )
    run_and_save(f"{method}_dec", labels, emb_orig)
```

- [ ] **Step 7: Write Cell 5 — summary table**

```python
print("\n=== All results ===")
summary_rows = []
for name, metrics in all_results.items():
    summary_rows.append({
        "method": name,
        "ACC (Hungarian)": round(metrics["ACC (Hungarian)"], 4),
        "Macro F1": round(metrics["Macro F1"], 4),
        "NMI": round(metrics["NMI"], 4),
        "ARI": round(metrics["ARI"], 4),
        "Silhouette": round(metrics["Silhouette Score"], 4),
        "Coverage": round(metrics["Coverage"], 4),
    })
summary_df = pd.DataFrame(summary_rows).sort_values("ACC (Hungarian)", ascending=False)
print(summary_df.to_string(index=False))
print(f"\nAll artifacts saved to {config.RESULTS_DIR}")
```

- [ ] **Step 8: Run the notebook end-to-end**

Open the notebook in Jupyter and run all cells, or:
```
cd C:\Users\ACER\OneDrive\Documents\final-project
.venv\Scripts\activate
jupyter nbconvert --to notebook --execute notebooks/02_unsupervised_clustering.ipynb --output notebooks/02_unsupervised_clustering.ipynb --ExecutePreprocessor.timeout=1800
```
DEC on CPU for 3,824 rows × 384-dim (MiniLM) should take ~5–15 min; RoBERTa (768-dim) ~10–25 min. Check the output summary table. All 6 results should have Coverage=1.0 (KMeans and Agglomerative always cover all rows; DEC does too since we take argmax).

- [ ] **Step 9: Verify all 6 artifact sets exist**

```python
import os
from pathlib import Path
results = Path("results")
methods = ["minilm_kmeans", "roberta_kmeans",
           "minilm_agglomerative", "roberta_agglomerative",
           "minilm_dec", "roberta_dec"]
for m in methods:
    for suffix in [f"metrics_{m}.json", f"cluster_labels_{m}.npy",
                   f"full_labels_{m}.csv", f"clusters_{m}.csv"]:
        assert (results / suffix).exists(), f"Missing: {suffix}"
print("All 24 artifacts present ✓")
```

- [ ] **Step 10: Commit**

```
git add notebooks/02_unsupervised_clustering.ipynb results/
git commit -m "feat: rebuild notebook 02 — KMeans + Agglomerative + DEC x MiniLM/RoBERTa (6 outcomes)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01StACqfxGJJkBzxocBh6gxU"
```

---

## Task 3: Update `notebooks/09_cluster_similarity.ipynb`

Replace the current approach (re-running KMeans/HDBSCAN/BERTopic from scratch) with loading the saved `cluster_labels_{name}.npy` files — faster, consistent with notebook 02's actual results, and no heavy model re-runs.

**Files:**
- Modify: `notebooks/09_cluster_similarity.ipynb`

**Interfaces:**
- Consumes:
  - `results/cluster_labels_{name}.npy` for each of the 6 methods (produced by Task 2)
  - `embeddings_cache/{method}_train_full_summary.npy` for cosine similarity computation in original space
  - `results/clusters_{name}.csv` for majority-label lookup per cluster
  - `data/processed/train_clean.parquet` for summaries
- Produces:
  - `results/cluster_similarity_summary.csv`
  - `results/cluster_sim_heatmap_{name}.png` × 6
  - `results/cluster_similarity_comparison.png`

- [ ] **Step 1: Clear all cells in the notebook**

Delete all existing cells. New notebook has 7 cells.

- [ ] **Step 2: Write Cell 0 — markdown header**

```markdown
# 09 — Intra/Inter-Cluster Similarity (v2)

Measures how **cohesive** each cluster is and how **separated** clusters are
from each other, for all 6 unsupervised methods:
minilm_kmeans, roberta_kmeans, minilm_agglomerative, roberta_agglomerative,
minilm_dec, roberta_dec.

Cluster assignments are loaded from `results/cluster_labels_{name}.npy`
(saved by notebook 02) — no re-running of clustering algorithms here.
Cosine similarity is always computed in the **original embedding space**
(pre-UMAP, pre-DEC encoder) so that distances are comparable across methods.
```

- [ ] **Step 3: Write Cell 1 — imports and data loading**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from utils import config
from utils.embeddings import load_cached

train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
true_labels = train_clean["label"].to_numpy()

METHODS = ["minilm", "roberta"]
CLUSTERERS = ["kmeans", "agglomerative", "dec"]
METHOD_NAMES = [f"{m}_{c}" for m in METHODS for c in CLUSTERERS]

suffix = "full"
embeddings_by_method = {}
for method in METHODS:
    arr = load_cached(f"{method}_train_{suffix}_summary")
    assert arr is not None, f"Missing cached embeddings for '{method}'"
    embeddings_by_method[method] = arr

print(f"Methods: {METHOD_NAMES}")
print(f"Rows: {len(train_clean)}")
for m, e in embeddings_by_method.items():
    print(f"  {m}: shape {e.shape}")
```

- [ ] **Step 4: Write Cell 2 — cluster_similarity_stats helper**

```python
def cluster_similarity_stats(embeddings: np.ndarray, cluster_labels: np.ndarray):
    """
    Compute intra-cluster cohesion and inter-cluster centroid similarity.

    Args:
        embeddings: (N, D) float32 — original embedding space (pre-UMAP)
        cluster_labels: (N,) int — cluster assignments; -1 = noise (excluded)

    Returns:
        per_cluster_df: DataFrame with columns [cluster, size, mean_intra_sim,
                        min_intra_sim, max_intra_sim]
        inter_mat: (K, K) float32 centroid-to-centroid cosine similarity matrix
        cluster_ids: sorted list of non-noise cluster ids
        mean_inter_sim: mean off-diagonal inter-cluster similarity (nan if K < 2)
    """
    unique_clusters = sorted(c for c in np.unique(cluster_labels) if c != -1)
    if not unique_clusters:
        return pd.DataFrame(), np.array([[]]), [], float("nan")

    rows = []
    centroids = []
    for cid in unique_clusters:
        mask = cluster_labels == cid
        members = embeddings[mask].astype(np.float32)
        centroid = members.mean(axis=0, keepdims=True)
        centroids.append(centroid[0])
        sims = cosine_similarity(members, centroid).ravel()
        rows.append({
            "cluster": cid,
            "size": int(mask.sum()),
            "mean_intra_sim": float(sims.mean()),
            "min_intra_sim": float(sims.min()),
            "max_intra_sim": float(sims.max()),
        })

    per_cluster_df = pd.DataFrame(rows)
    cent_matrix = np.stack(centroids).astype(np.float32)
    inter_mat = cosine_similarity(cent_matrix)

    n = len(unique_clusters)
    mean_inter = float(inter_mat[~np.eye(n, dtype=bool)].mean()) if n > 1 else float("nan")

    return per_cluster_df, inter_mat, unique_clusters, mean_inter

print("cluster_similarity_stats() defined.")
```

- [ ] **Step 5: Write Cell 3 — compute similarity for all 6 methods**

```python
all_stats = {}       # method_name -> (per_cluster_df, inter_mat, cluster_ids)
method_summary = []  # one row per method for the cross-method table

# Load majority-label lookup from clusters_{name}.csv (saved by notebook 02)
cluster_name_map = {}
for name in METHOD_NAMES:
    path = config.RESULTS_DIR / f"clusters_{name}.csv"
    if path.exists():
        cdf = pd.read_csv(path)
        cluster_name_map[name] = dict(zip(cdf["cluster"], cdf["majority_true_label"]))

for name in METHOD_NAMES:
    embedding_key = name.split("_")[0]  # "minilm" or "roberta"
    emb = embeddings_by_method[embedding_key]

    label_path = config.RESULTS_DIR / f"cluster_labels_{name}.npy"
    assert label_path.exists(), (
        f"Missing {label_path.name} — run notebook 02 first to generate cluster labels"
    )
    cluster_labels = np.load(label_path).astype(int)

    per_df, inter_mat, cids, mean_inter = cluster_similarity_stats(emb, cluster_labels)

    if per_df.empty:
        print(f"{name}: no non-noise clusters — skipping")
        continue

    mean_intra = float(per_df["mean_intra_sim"].mean())
    n_noise = int((cluster_labels == -1).sum())
    coverage = 1.0 - n_noise / len(cluster_labels)
    ratio = mean_intra / mean_inter if mean_inter and not np.isnan(mean_inter) else float("nan")

    all_stats[name] = (per_df, inter_mat, cids)
    method_summary.append({
        "method": name,
        "n_clusters": len(cids),
        "coverage": coverage,
        "mean_intra_sim": mean_intra,
        "mean_inter_sim": mean_inter,
        "separation_ratio": ratio,
    })
    print(f"{name}: intra={mean_intra:.3f}  inter={mean_inter:.3f}  "
          f"ratio={ratio:.3f}  coverage={coverage:.2f}  ({len(cids)} clusters)")
```

- [ ] **Step 6: Write Cell 4 — per-method cluster detail printout**

```python
for method_name, (cdf, _, _) in all_stats.items():
    cname_map = cluster_name_map.get(method_name, {})
    display = cdf.copy()
    display.insert(1, "majority_label",
                   display["cluster"].map(cname_map).fillna("?"))
    print(f"\n=== {method_name} ===")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
```

- [ ] **Step 7: Write Cell 5 — save summary CSV + heatmaps**

```python
summary_df = pd.DataFrame(method_summary).sort_values("separation_ratio", ascending=False)
print("=== Cross-method similarity comparison ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(config.RESULTS_DIR / "cluster_similarity_summary.csv", index=False)
print(f"\nSaved: results/cluster_similarity_summary.csv")

# Per-method centroid heatmaps
for method_name, (cdf, inter_mat, cids) in all_stats.items():
    if len(cids) < 2:
        print(f"{method_name}: too few clusters for heatmap — skipping")
        continue

    cname_map = cluster_name_map.get(method_name, {})
    tick_labels = [cname_map.get(c, str(c)) for c in cids]
    n = len(cids)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.55)))
    im = ax.imshow(inter_mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{inter_mat[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=6 if n > 10 else 7,
                    color="white" if inter_mat[i, j] > 0.6 else "black")
    ax.set_title(f"{method_name} — cluster centroid cosine similarity", pad=10)
    plt.tight_layout()
    out = config.RESULTS_DIR / f"cluster_sim_heatmap_{method_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")
```

- [ ] **Step 8: Write Cell 6 — cross-method bar chart**

```python
plot_df = summary_df[summary_df["mean_intra_sim"].notna()].set_index("method")
x = np.arange(len(plot_df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - width / 2, plot_df["mean_intra_sim"], width,
       label="Intra-cluster (↑ tight)", color="steelblue")
ax.bar(x + width / 2, plot_df["mean_inter_sim"], width,
       label="Inter-cluster centroid (↓ separated)", color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Mean cosine similarity")
ax.set_title("Intra- vs. inter-cluster similarity by method")
ax.set_ylim(0, 1.05)
ax.legend()

for i, (_, row) in enumerate(plot_df.iterrows()):
    if not np.isnan(row["separation_ratio"]):
        ax.text(i, max(row["mean_intra_sim"], row["mean_inter_sim"]) + 0.02,
                f"×{row['separation_ratio']:.2f}",
                ha="center", fontsize=7, color="#333")

plt.tight_layout()
out = config.RESULTS_DIR / "cluster_similarity_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out.name}")
```

- [ ] **Step 9: Run notebook 09 end-to-end**

```
jupyter nbconvert --to notebook --execute notebooks/09_cluster_similarity.ipynb --output notebooks/09_cluster_similarity.ipynb --ExecutePreprocessor.timeout=600
```
Verify: all 6 heatmaps exist in `results/`, `cluster_similarity_summary.csv` has 6 rows.

- [ ] **Step 10: Commit**

```
git add notebooks/09_cluster_similarity.ipynb results/
git commit -m "feat: rebuild notebook 09 — load cluster labels from npy, 6 new methods

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01StACqfxGJJkBzxocBh6gxU"
```

---

## Task 4: Clean up stale results + update `notebooks/07_comparison.ipynb`

Remove old metric/label files for dropped methods so notebook 07's auto-scan doesn't pick them up. Update notebook 07 to remove the hardcoded OPENAI pending rows (those methods are dropped entirely).

**Files:**
- Modify: `notebooks/07_comparison.ipynb`
- Delete: `results/metrics_tfidf_kmeans.json`, `results/metrics_tfidf_hdbscan.json`, `results/metrics_minilm_hdbscan.json`, `results/metrics_roberta_hdbscan.json`, `results/metrics_bertopic.json` (if exists separately), `results/full_labels_tfidf_*.csv`, `results/full_labels_*hdbscan*.csv`, `results/cluster_labels_*hdbscan*.npy`, `results/cluster_labels_tfidf*.npy`, `results/clusters_*hdbscan*.csv`, `results/clusters_tfidf*.csv`, `results/cluster_sim_heatmap_*hdbscan*.png`, `results/cluster_sim_heatmap_tfidf*.png`

- [ ] **Step 1: Delete stale result files**

```python
# Run this snippet in a Python shell or notebook cell to remove dropped-method artifacts
from pathlib import Path
results = Path("C:/Users/ACER/OneDrive/Documents/final-project/results")

DROPPED_PREFIXES = [
    "tfidf_kmeans", "tfidf_hdbscan",
    "minilm_hdbscan", "roberta_hdbscan",
]
DROPPED_ARTIFACTS = [
    "bertopic",   # BERTopic is no longer in the active comparison
]

all_dropped = DROPPED_PREFIXES + DROPPED_ARTIFACTS

deleted = []
for pattern in all_dropped:
    for f in results.glob(f"*{pattern}*"):
        print(f"Deleting: {f.name}")
        f.unlink()
        deleted.append(f.name)

print(f"\nDeleted {len(deleted)} files.")
```

Verify after deletion: `results/metrics_*.json` should have exactly these files:
`metrics_minilm_kmeans.json`, `metrics_roberta_kmeans.json`,
`metrics_minilm_agglomerative.json`, `metrics_roberta_agglomerative.json`,
`metrics_minilm_dec.json`, `metrics_roberta_dec.json`,
plus the semi-supervised ones (`metrics_pseudo_labeling.json`, `metrics_weak_supervision.json`, etc.).

- [ ] **Step 2: Update notebook 07 — remove OPENAI_METHODS pending rows**

Find the cell in `07_comparison.ipynb` that contains:
```python
OPENAI_METHODS = ["openai_kmeans", "openai_hdbscan"]
for method in OPENAI_METHODS:
    if method not in rows:
        rows[method] = {"Status": "pending"}
```
Delete those 3 lines entirely. The notebook auto-scans `results/metrics_*.json` — with those files gone and the pending injection removed, the comparison table will be clean.

- [ ] **Step 3: Run notebook 07**

```
jupyter nbconvert --to notebook --execute notebooks/07_comparison.ipynb --output notebooks/07_comparison.ipynb --ExecutePreprocessor.timeout=120
```
Verify: the output comparison table shows the 6 new unsupervised methods alongside the existing semi-supervised methods, with no `pending` rows and no NaN-only rows from dropped methods.

- [ ] **Step 4: Commit**

```
git add notebooks/07_comparison.ipynb results/
git commit -m "chore: remove stale HDBSCAN/TF-IDF/BERTopic artifacts, clean notebook 07 pending rows

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01StACqfxGJJkBzxocBh6gxU"
```

---

## Task 5: Update `PROJECT_SUMMARY.md`

**Files:**
- Modify: `PROJECT_SUMMARY.md`

Do this **after** Tasks 2–4 are complete and you have actual metric numbers from the notebook runs.

- [ ] **Step 1: Update Section 3.2 — flow diagram**

Replace the existing unsupervised flow diagram. Find:
```
    S --> V1[TF-IDF vector]
    S --> V2[MiniLM embedding]
    S --> V3[RoBERTa embedding]
    S --> V4[OpenAI embedding - pending, no API key]
    V1 --> CL[KMeans k=16 / HDBSCAN / BERTopic]
    V2 --> CL
    V3 --> CL
    V4 --> CL
    CL --> M[Hungarian-match cluster id -> class name]
    M --> R[results/full_labels_*.csv]
```
Replace with:
```
    S --> V2[MiniLM embedding all-MiniLM-L6-v2]
    S --> V3[RoBERTa embedding roberta-base]
    V2 --> CL[KMeans k=16 / Agglomerative k=16 / DEC k=16]
    V3 --> CL
    CL --> M[Hungarian-match cluster id -> class name]
    M --> R[results/full_labels_*.csv\n6 outcomes: 3 clusterers x 2 embeddings]
```

- [ ] **Step 2: Update Section 4.1 — approach text**

Replace the paragraph starting "Every row's `text` is summarized..." with:

```
Every row's `text` is summarized into a real sentence with
`facebook/bart-large-cnn`, then that **summary** (not the raw text) is
embedded two ways — MiniLM (`all-MiniLM-L6-v2`, 384-dim) and RoBERTa
(`roberta-base`, 768-dim) — and clustered with three algorithms, all
requiring explicit k=16:

- **KMeans** (`sklearn.cluster.KMeans`, n_init=10): runs on
  UMAP(n_components=50)-reduced embeddings. Standard baseline.
- **Agglomerative Clustering** (`sklearn.cluster.AgglomerativeClustering`,
  ward linkage): hierarchical bottom-up clustering on UMAP-reduced
  embeddings. Ward linkage minimises within-cluster variance at each merge.
- **DEC** (Deep Embedded Clustering, Xie et al. 2016, `utils/dec.py`):
  trains a 2-layer MLP encoder (input_dim → 256 → 64) directly on the
  pre-trained embeddings, initialises K=16 cluster centers from KMeans on
  the latent space, then refines both encoder and centers jointly by
  minimising KL(P‖Q) where Q is the student-t soft assignment and P is
  the sharpened target distribution. No UMAP reduction needed — DEC learns
  its own compressed representation.

This gives **6 outcomes** (3 clusterers × 2 embeddings). Cluster ids are
Hungarian-matched to the 16 class names for accuracy scoring.
```

- [ ] **Step 3: Update Section 5.1 results table**

Replace the old table (which had HDBSCAN/BERTopic rows) with the actual numbers from your notebook 02 run output. The table format to use:

```markdown
| Method | ACC (Hungarian) | Macro F1 | NMI | ARI | Silhouette | Coverage |
|---|---|---|---|---|---|---|
| **minilm_dec** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | 1.00 |
| **minilm_kmeans** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | 1.00 |
| **minilm_agglomerative** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | 1.00 |
| **roberta_dec** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | 1.00 |
| **roberta_kmeans** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | 1.00 |
| **roberta_agglomerative** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | 1.00 |
```
(Sort by ACC descending. Fill in actual numbers — do not leave X.XXXX.)

- [ ] **Step 4: Update Section 9 limitations**

Remove the paragraph about HDBSCAN's `min_cluster_size`. Remove the paragraph about BERTopic's noise-topic coverage. Remove the paragraph about OpenAI embeddings pending. Keep and update the k=16 assumption note:

```markdown
- **k=16 assumed, not swept.** Every clustering run above fixes k=16 (the
  known class count). All three algorithms (KMeans, Agglomerative, DEC)
  require explicit k. A real deployment without ground truth would need to
  sweep k and choose it by an unsupervised metric (e.g. silhouette or
  elbow). Deferred to future work.
```

- [ ] **Step 5: Update Section 10 repo map — notebooks list**

Remove `03_bertopic.ipynb` from the active pipeline description. Add:

```
  utils/
    dec.py              # DEC model, autoencoder pretraining, KL divergence refinement
```

Update the notebook 02 description line:
```
  notebooks/
    02_unsupervised_clustering.ipynb  # KMeans/Agglomerative/DEC × MiniLM/RoBERTa (6 outcomes)
```

- [ ] **Step 6: Update Section 3.1 overall pipeline diagram**

Find the line:
```
    D --> E[embed summaries TF-IDF/MiniLM/RoBERTa\n-> cluster KMeans k=16 / HDBSCAN / BERTopic]
```
Replace with:
```
    D --> E[embed summaries MiniLM/RoBERTa\n-> cluster KMeans/Agglomerative/DEC k=16\n6 outcomes]
```

- [ ] **Step 7: Commit**

```
git add PROJECT_SUMMARY.md
git commit -m "docs: update PROJECT_SUMMARY for 6-outcome unsupervised track (DEC + Agglomerative)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01StACqfxGJJkBzxocBh6gxU"
```

---

## Self-Review

### Spec coverage check

| Requirement (from chat + memory) | Covered by task |
|---|---|
| Summaries as unsupervised input | ✅ Task 2 Cell 1 loads `summary` column |
| MiniLM + RoBERTa embeddings only | ✅ Task 2 METHODS = ["minilm", "roberta"] |
| KMeans with explicit k | ✅ Task 2 Cell 3 |
| Agglomerative with explicit k | ✅ Task 2 Cell 3 |
| DEC with explicit k | ✅ Task 2 Cell 4 + Task 1 |
| 6 outcomes total | ✅ 3 clusterers × 2 embeddings |
| No HDBSCAN, no BERTopic, no TF-IDF clustering, no OpenAI | ✅ Tasks 2 + 4 |
| Full-row CSVs per method | ✅ `run_and_save()` calls `save_full_output()` |
| Cluster labels saved as npy for notebook 09 | ✅ `run_and_save()` calls `np.save()` |
| Notebook 09 loads from npy (no model re-run) | ✅ Task 3 |
| Notebook 07 clean (no pending rows) | ✅ Task 4 |
| PROJECT_SUMMARY updated | ✅ Task 5 |
| All metrics via existing `evaluate_unsupervised()` | ✅ All clusterers use same function |

### Placeholder scan
None found — all code blocks are complete and runnable.

### Type consistency
- `train_dec()` → `np.ndarray` of int32 ✅ matches `cluster_labels` type expected by `evaluate_unsupervised()` and `hungarian_match_predictions()`
- `run_and_save(name, cluster_labels, emb_original)` — called the same way for all 6 methods ✅
- `cluster_similarity_stats(embeddings, cluster_labels)` — same signature in Task 3 as expected ✅
- `embedding_key = name.split("_")[0]` in Task 3 — works for all 6 names: `"minilm_kmeans".split("_")[0]` → `"minilm"` ✅ (Note: `"roberta_agglomerative".split("_")[0]` → `"roberta"` ✅, `"minilm_dec".split("_")[0]` → `"minilm"` ✅)
