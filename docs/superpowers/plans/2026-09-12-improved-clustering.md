# Improved Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/11_improved_clustering.ipynb` with 3 improvement experiments (MPNet embeddings, UMAP tuning, title embeddings) that add 8 new methods to the comparison table without touching any existing notebook or result.

**Architecture:** One notebook, three clearly-labelled experiment sections; shared helper functions (`umap_reduce`, `run_and_save`) replicated from notebook 02; each method produces the same 4-artifact schema (metrics JSON, cluster labels npy, clusters CSV, full-labels CSV) so notebook 07 picks them up automatically.

**Tech Stack:** Python 3.10+, sentence-transformers, umap-learn, scikit-learn, numpy, pandas, existing `utils/` modules

**Spec:** `docs/superpowers/specs/2026-09-12-improved-clustering-design.md`

## Global Constraints

- Python ≥ 3.10; all dependencies already in `pyproject.toml`
- `random_state=42` / `SEED = config.SEED` (= 42) on every stochastic call — never hardcode 42 inline; use `config.SEED`
- `N_CLUSTERS = config.NUM_CLASSES` (= 16) — never hardcode 16 inline
- Notebook runnable via `uv run jupyter nbconvert --to notebook --execute notebooks/11_improved_clustering.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=1800`
- No modification to existing notebooks, result files, or utility modules
- Embedding cache files: skip encoding if file already exists (`Path(cache_path).exists()`)
- `run_and_save(name, cluster_labels, emb_original)` — exact same signature and artifact schema as notebook 02
- UMAP default params for experiments A and C: `n_neighbors=15, min_dist=0.1, n_components=50, metric="cosine", random_state=config.SEED`
- Method names must exactly match: `mpnet_kmeans`, `mpnet_agglomerative`, `mpnet_dec`, `minilm_kmeans_umap`, `minilm_agglomerative_umap`, `title_minilm_kmeans`, `title_minilm_agglomerative`, `title_minilm_dec`
- DEC params (same as notebook 02): `latent_dim=64, pretrain_epochs=50, dec_epochs=150, batch_size=256, lr_pretrain=1e-3, lr_dec=1e-4, update_interval=5, tol=1e-3`
- Silhouette subsampling: `config.SILHOUETTE_SAMPLE_SIZE` (= 2000)

---

### Task 1: Scaffold — markdown description, imports, config, and helper functions (cells 1–3)

**Files:**
- Create: `notebooks/11_improved_clustering.ipynb`

**Interfaces:**
- Produces (kernel state for all later tasks):
  - `true_labels`: np.ndarray (3824,) int
  - `texts`: list[str] len 3824
  - `summaries`: list[str] len 3824
  - `titles`: list[str] len 3824
  - `all_results`: dict (accumulates metrics across all tasks)
  - `umap_reduce(emb, n_neighbors, min_dist, n_components, seed)` → np.ndarray
  - `run_and_save(name, cluster_labels, emb_original)` → None

- [ ] **Step 1: Create notebook file with cell 0 (markdown description)**

Write `notebooks/11_improved_clustering.ipynb` as valid Jupyter notebook JSON with `nbformat=4`. Cell 0 is a markdown cell:

```markdown
# 11 — Improved Unsupervised Clustering

Three targeted experiments on top of the notebook 02 baseline (MiniLM KMeans/Agglomerative ~45% ACC):

**Experiment A — MPNet embeddings:** replace `all-MiniLM-L6-v2` (384-dim) with
`all-mpnet-base-v2` (768-dim) on the same BART summaries. Run KMeans, Agglomerative, DEC.

**Experiment B — UMAP hyperparameter tuning:** keep MiniLM embeddings; sweep
n_neighbors=[15,30,50] × min_dist=[0.0,0.05,0.1] (9 combos), pick best by silhouette.
Run KMeans and Agglomerative with best params. (DEC has its own dim-reduction; excluded.)

**Experiment C — Title embeddings:** embed raw article titles with MiniLM instead of BART
summaries. Run KMeans, Agglomerative, DEC.

New methods (8 total): mpnet_kmeans, mpnet_agglomerative, mpnet_dec,
minilm_kmeans_umap, minilm_agglomerative_umap,
title_minilm_kmeans, title_minilm_agglomerative, title_minilm_dec

All methods produce the same artifact schema as notebook 02 and are automatically
included in the comparison table when notebook 07 is re-run.
```

- [ ] **Step 2: Add cell 1 — imports, config, data loading**

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
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from utils import config
from utils.dec import train_dec
from utils.embeddings import load_cached
from utils.interpretability import summarize_clusters
from utils.metrics import evaluate_unsupervised, hungarian_match_predictions
from utils.samples import save_full_output

# ── Data ─────────────────────────────────────────────────────────────────────
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
true_labels  = train_clean["label"].to_numpy()
texts        = train_clean["text"].tolist()
summaries    = train_clean["summary"].tolist()
titles       = train_clean["title"].tolist()

assert len(true_labels) == 3824, f"Expected 3824 rows, got {len(true_labels)}"
print(f"Rows: {len(train_clean)} | k={config.NUM_CLASSES}")

# ── Accumulator ───────────────────────────────────────────────────────────────
all_results = {}   # method_name -> metrics dict
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Add cell 2 — helper functions**

```python
def umap_reduce(
    emb: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 50,
    metric: str = "cosine",
    seed: int = config.SEED,
) -> np.ndarray:
    """Reduce embeddings via UMAP. Parameterised to support grid search."""
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(emb)


def run_and_save(name: str, cluster_labels: np.ndarray, emb_original: np.ndarray) -> None:
    """Evaluate and save all artifacts for one clustering result (same schema as notebook 02)."""
    metrics = evaluate_unsupervised(
        true_labels, cluster_labels, emb_original,
        metric_sample_size=config.SILHOUETTE_SAMPLE_SIZE, seed=config.SEED,
    )
    all_results[name] = metrics

    with open(config.RESULTS_DIR / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(config.RESULTS_DIR / f"cluster_labels_{name}.npy", cluster_labels)

    interp = summarize_clusters(
        summaries, cluster_labels, true_labels,
        emb_original,
        config.CLASS_NAMES,
    )
    interp.to_csv(config.RESULTS_DIR / f"clusters_{name}.csv", index=False)

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


print("Helpers defined: umap_reduce(), run_and_save()")
```

- [ ] **Step 4: Verify notebook structure is valid JSON**

```bash
uv run python -c "
import json
nb = json.load(open('notebooks/11_improved_clustering.ipynb'))
print('cells:', len(nb['cells']), '| nbformat:', nb['nbformat'])
assert nb['nbformat'] == 4
assert nb['cells'][0]['cell_type'] == 'markdown'
assert nb['cells'][1]['cell_type'] == 'code'
assert nb['cells'][2]['cell_type'] == 'code'
print('Structure OK')
"
```
Expected: `cells: 3 | nbformat: 4` then `Structure OK`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/11_improved_clustering.ipynb
git commit -m "feat(nb11): scaffold — markdown + imports + helper functions (cells 0-2)"
```

---

### Task 2: Experiment A — MPNet embeddings (KMeans, Agglomerative, DEC) (cells 3–5)

**Files:**
- Modify: `notebooks/11_improved_clustering.ipynb` (append cells 3–5)

**Interfaces:**
- Consumes: `true_labels`, `summaries`, `texts`, `run_and_save`, `umap_reduce`, `config`, `train_dec`, `KMeans`, `AgglomerativeClustering` — from Task 1 kernel state
- Produces: 3 method artifacts on disk: `mpnet_kmeans`, `mpnet_agglomerative`, `mpnet_dec`
- Produces kernel var: `mpnet_emb` (3824, 768) — raw MPNet embeddings

- [ ] **Step 1: Add cell 3 — encode summaries with MPNet, cache result**

```python
# ════════════════════════════════════════════════════════════════════
# EXPERIMENT A — MPNet embeddings (all-mpnet-base-v2, 768-dim)
# ════════════════════════════════════════════════════════════════════
MPNET_CACHE = REPO_ROOT / "embeddings_cache" / "mpnet_train_full_summary.npy"

if MPNET_CACHE.exists():
    print(f"Loading cached MPNet embeddings from {MPNET_CACHE.name}...")
    mpnet_emb = np.load(MPNET_CACHE).astype(np.float32)
else:
    print("Encoding summaries with all-mpnet-base-v2 (first run — takes ~5 min)...")
    model_mpnet = SentenceTransformer("all-mpnet-base-v2")
    mpnet_emb = model_mpnet.encode(
        summaries,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(MPNET_CACHE, mpnet_emb)
    print(f"Saved to {MPNET_CACHE.name}")

assert mpnet_emb.shape == (3824, 768), f"Unexpected shape: {mpnet_emb.shape}"
print(f"MPNet embeddings ready: {mpnet_emb.shape}")
```

- [ ] **Step 2: Add cell 4 — MPNet KMeans and Agglomerative**

```python
print("=== [A] MPNet KMeans ===")
mpnet_reduced = umap_reduce(mpnet_emb)   # default params: n_neighbors=15, min_dist=0.1
print(f"  UMAP reduced: {mpnet_reduced.shape}")

km = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
run_and_save("mpnet_kmeans", km.fit_predict(mpnet_reduced), mpnet_emb)

print("\n=== [A] MPNet Agglomerative ===")
agg = AgglomerativeClustering(
    n_clusters=config.NUM_CLASSES,
    linkage="ward",
    metric="euclidean",
)
run_and_save("mpnet_agglomerative", agg.fit_predict(mpnet_reduced), mpnet_emb)
```

- [ ] **Step 3: Add cell 5 — MPNet DEC**

```python
print("\n=== [A] MPNet DEC ===")
print(f"Training DEC on mpnet embeddings {mpnet_emb.shape} — expect ~10-15 min on CPU...")
labels_dec = train_dec(
    mpnet_emb,
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
run_and_save("mpnet_dec", labels_dec, mpnet_emb)
```

- [ ] **Step 4: Verify Experiment A artifacts exist**

```bash
uv run python -c "
import numpy as np, json
from pathlib import Path
for name in ['mpnet_kmeans', 'mpnet_agglomerative', 'mpnet_dec']:
    m = json.load(open(f'results/metrics_{name}.json'))
    lbl = np.load(f'results/cluster_labels_{name}.npy')
    assert lbl.shape == (3824,), f'{name}: bad label shape {lbl.shape}'
    print(f'{name}: ACC={m[\"ACC (Hungarian)\"]:.4f}  shape={lbl.shape}')
"
```
Expected: 3 lines, all ACC values printed, no assertion errors.

- [ ] **Step 5: Commit**

```bash
git add notebooks/11_improved_clustering.ipynb
git commit -m "feat(nb11): Experiment A — MPNet KMeans/Agglomerative/DEC (cells 3-5)"
```

---

### Task 3: Experiment B — UMAP hyperparameter grid search (cells 6–7)

**Files:**
- Modify: `notebooks/11_improved_clustering.ipynb` (append cells 6–7)

**Interfaces:**
- Consumes: `umap_reduce`, `run_and_save`, `config`, `KMeans`, `AgglomerativeClustering`, `silhouette_score` — from Task 1 kernel state
- Consumes: MiniLM embeddings loaded fresh in this task (not from Task 2)
- Produces: 2 method artifacts on disk: `minilm_kmeans_umap`, `minilm_agglomerative_umap`
- Produces kernel vars: `best_n_neighbors` (int), `best_min_dist` (float)

- [ ] **Step 1: Add cell 6 — load MiniLM embeddings and run 3×3 grid**

```python
# ════════════════════════════════════════════════════════════════════
# EXPERIMENT B — UMAP hyperparameter tuning on MiniLM embeddings
# ════════════════════════════════════════════════════════════════════
minilm_emb = np.load(
    REPO_ROOT / "embeddings_cache" / "minilm_train_full_summary.npy"
).astype(np.float32)
assert minilm_emb.shape == (3824, 384), f"Unexpected: {minilm_emb.shape}"
print(f"MiniLM embeddings loaded: {minilm_emb.shape}")

N_NEIGHBORS_GRID = [15, 30, 50]
MIN_DIST_GRID    = [0.0, 0.05, 0.1]

print("\nUMAP grid search (3×3 = 9 combos, scored by KMeans silhouette):")
print(f"{'n_neighbors':>12} {'min_dist':>9} {'silhouette':>12}")
print("-" * 36)

best_score, best_n_neighbors, best_min_dist = -1.0, 15, 0.1

for nn in N_NEIGHBORS_GRID:
    for md in MIN_DIST_GRID:
        reduced = umap_reduce(minilm_emb, n_neighbors=nn, min_dist=md)
        km_tmp = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
        labels_tmp = km_tmp.fit_predict(reduced)
        score = silhouette_score(
            reduced, labels_tmp,
            sample_size=config.SILHOUETTE_SAMPLE_SIZE,
            random_state=config.SEED,
        )
        marker = " ◀ best" if score > best_score else ""
        print(f"{nn:>12} {md:>9.2f} {score:>12.4f}{marker}")
        if score > best_score:
            best_score, best_n_neighbors, best_min_dist = score, nn, md

print(f"\nBest params: n_neighbors={best_n_neighbors}, min_dist={best_min_dist} "
      f"(silhouette={best_score:.4f})")
```

- [ ] **Step 2: Add cell 7 — run KMeans and Agglomerative with best params**

```python
print("\n=== [B] MiniLM KMeans (tuned UMAP) ===")
minilm_reduced_tuned = umap_reduce(
    minilm_emb, n_neighbors=best_n_neighbors, min_dist=best_min_dist
)
print(f"  UMAP reduced: {minilm_reduced_tuned.shape}")

km = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
run_and_save("minilm_kmeans_umap", km.fit_predict(minilm_reduced_tuned), minilm_emb)

print("\n=== [B] MiniLM Agglomerative (tuned UMAP) ===")
agg = AgglomerativeClustering(
    n_clusters=config.NUM_CLASSES,
    linkage="ward",
    metric="euclidean",
)
run_and_save("minilm_agglomerative_umap", agg.fit_predict(minilm_reduced_tuned), minilm_emb)
```

- [ ] **Step 3: Verify Experiment B artifacts**

```bash
uv run python -c "
import numpy as np, json
for name in ['minilm_kmeans_umap', 'minilm_agglomerative_umap']:
    m = json.load(open(f'results/metrics_{name}.json'))
    lbl = np.load(f'results/cluster_labels_{name}.npy')
    assert lbl.shape == (3824,)
    print(f'{name}: ACC={m[\"ACC (Hungarian)\"]:.4f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/11_improved_clustering.ipynb
git commit -m "feat(nb11): Experiment B — UMAP grid search + tuned KMeans/Agglomerative (cells 6-7)"
```

---

### Task 4: Experiment C — Title embeddings (KMeans, Agglomerative, DEC) (cells 8–10)

**Files:**
- Modify: `notebooks/11_improved_clustering.ipynb` (append cells 8–10)

**Interfaces:**
- Consumes: `titles`, `true_labels`, `summaries`, `texts`, `run_and_save`, `umap_reduce`, `config`, `train_dec`, `KMeans`, `AgglomerativeClustering` — from Task 1 kernel state
- Produces: 3 method artifacts on disk: `title_minilm_kmeans`, `title_minilm_agglomerative`, `title_minilm_dec`
- Produces kernel var: `title_emb` (3824, 384)

- [ ] **Step 1: Add cell 8 — encode titles with MiniLM, cache result**

```python
# ════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Title embeddings (MiniLM on raw article titles)
# ════════════════════════════════════════════════════════════════════
TITLE_CACHE = REPO_ROOT / "embeddings_cache" / "minilm_train_title.npy"

if TITLE_CACHE.exists():
    print(f"Loading cached title embeddings from {TITLE_CACHE.name}...")
    title_emb = np.load(TITLE_CACHE).astype(np.float32)
else:
    print("Encoding titles with all-MiniLM-L6-v2 (first run — takes ~1 min)...")
    model_minilm = SentenceTransformer("all-MiniLM-L6-v2")
    title_emb = model_minilm.encode(
        titles,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(TITLE_CACHE, title_emb)
    print(f"Saved to {TITLE_CACHE.name}")

assert title_emb.shape == (3824, 384), f"Unexpected shape: {title_emb.shape}"
print(f"Title embeddings ready: {title_emb.shape}")
```

- [ ] **Step 2: Add cell 9 — Title KMeans and Agglomerative**

```python
print("=== [C] Title MiniLM KMeans ===")
title_reduced = umap_reduce(title_emb)   # default params
print(f"  UMAP reduced: {title_reduced.shape}")

km = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
run_and_save("title_minilm_kmeans", km.fit_predict(title_reduced), title_emb)

print("\n=== [C] Title MiniLM Agglomerative ===")
agg = AgglomerativeClustering(
    n_clusters=config.NUM_CLASSES,
    linkage="ward",
    metric="euclidean",
)
run_and_save("title_minilm_agglomerative", agg.fit_predict(title_reduced), title_emb)
```

- [ ] **Step 3: Add cell 10 — Title DEC**

```python
print("\n=== [C] Title MiniLM DEC ===")
print(f"Training DEC on title embeddings {title_emb.shape} — expect ~10-15 min on CPU...")
labels_title_dec = train_dec(
    title_emb,
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
run_and_save("title_minilm_dec", labels_title_dec, title_emb)
```

- [ ] **Step 4: Verify Experiment C artifacts**

```bash
uv run python -c "
import numpy as np, json
for name in ['title_minilm_kmeans', 'title_minilm_agglomerative', 'title_minilm_dec']:
    m = json.load(open(f'results/metrics_{name}.json'))
    lbl = np.load(f'results/cluster_labels_{name}.npy')
    assert lbl.shape == (3824,)
    print(f'{name}: ACC={m[\"ACC (Hungarian)\"]:.4f}')
"
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/11_improved_clustering.ipynb
git commit -m "feat(nb11): Experiment C — title embeddings KMeans/Agglomerative/DEC (cells 8-10)"
```

---

### Task 5: Results summary, full execution, and verification (cell 11)

**Files:**
- Modify: `notebooks/11_improved_clustering.ipynb` (append cell 11)

**Interfaces:**
- Consumes: `all_results` dict (accumulated by `run_and_save` across Tasks 2–4)
- Produces: printed results table; confirms all 8 methods in `all_results`

- [ ] **Step 1: Add cell 11 — results summary table**

```python
# ════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY — all 8 new methods vs. baseline
# ════════════════════════════════════════════════════════════════════
BASELINE = {"name": "minilm_agglomerative (baseline)", "ACC (Hungarian)": 0.4506,
            "Macro F1": None, "NMI": None, "Silhouette": None}

ALL_NEW_METHODS = [
    "mpnet_kmeans", "mpnet_agglomerative", "mpnet_dec",
    "minilm_kmeans_umap", "minilm_agglomerative_umap",
    "title_minilm_kmeans", "title_minilm_agglomerative", "title_minilm_dec",
]

assert set(ALL_NEW_METHODS) == set(all_results.keys()), (
    f"Missing methods: {set(ALL_NEW_METHODS) - set(all_results.keys())}"
)

print(f"{'Method':<35} {'ACC':>7} {'F1':>7} {'NMI':>7} {'Silhouette':>11}")
print("-" * 70)
print(f"{'minilm_agglomerative (baseline)':<35} {'0.4506':>7}   —       —        —")
for name in ALL_NEW_METHODS:
    m = all_results[name]
    beat = " ✓" if m["ACC (Hungarian)"] > 0.4506 else ""
    print(
        f"{name:<35} {m['ACC (Hungarian)']:>7.4f} "
        f"{m['Macro F1']:>7.4f} {m['NMI']:>7.4f} "
        f"{m['Silhouette']:>11.4f}{beat}"
    )
print("\n✓ = beats baseline (minilm_agglomerative ACC=0.4506)")
print(f"\nAll {len(ALL_NEW_METHODS)} new methods complete.")
```

- [ ] **Step 2: Execute the complete notebook end-to-end**

This will take up to 30 minutes (two DEC runs). Run from repo root:

```bash
uv run jupyter nbconvert --to notebook --execute \
    notebooks/11_improved_clustering.ipynb \
    --output-dir notebooks \
    --ExecutePreprocessor.timeout=1800
```

Expected: exits code 0. All 8 methods printed in the summary table.

If it times out (unlikely under 30 min), increase `--ExecutePreprocessor.timeout=3600`.

- [ ] **Step 3: Verify all 8 artifacts exist and shapes are correct**

```bash
uv run python -c "
import numpy as np, json
from pathlib import Path

methods = [
    'mpnet_kmeans', 'mpnet_agglomerative', 'mpnet_dec',
    'minilm_kmeans_umap', 'minilm_agglomerative_umap',
    'title_minilm_kmeans', 'title_minilm_agglomerative', 'title_minilm_dec',
]
for name in methods:
    m = json.load(open(f'results/metrics_{name}.json'))
    lbl = np.load(f'results/cluster_labels_{name}.npy')
    assert lbl.shape == (3824,), f'{name}: bad shape {lbl.shape}'
    assert Path(f'results/clusters_{name}.csv').exists(), f'{name}: clusters CSV missing'
    assert Path(f'results/full_labels_{name}.csv').exists(), f'{name}: full CSV missing'
    print(f'OK {name}: ACC={m[\"ACC (Hungarian)\"]:.4f}')

print(f'\nAll {len(methods)} methods verified.')
"
```

Expected: 8 `OK` lines, no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/11_improved_clustering.ipynb
git commit -m "feat(nb11): summary cell + full execution verified (cell 11)"
```
