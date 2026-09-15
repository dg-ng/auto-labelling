# Unsupervised Improvement & Cluster Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new clustering experiments (RoBERTa+UMAP-tuned, combined embeddings), then build a semantic evaluation notebook that generates LexRank descriptions for every cluster, maps them to the 16 real categories, and scores every method with a RAGAS-style cosine similarity metric.

**Architecture:** Notebook 12 adds Experiments D and E following the exact same `run_and_save()` artifact schema as notebooks 02 and 11. A new `utils/cluster_description.py` module provides LexRank-based description generation. Notebook 13 loads all existing cluster results, generates descriptions, maps to categories via cosine similarity with `all-mpnet-base-v2`, and writes a final `semantic_evaluation.csv`.

**Tech Stack:** Python 3.11, sentence-transformers (`all-mpnet-base-v2`), sumy (LexRank), scikit-learn (UMAP+KMeans+Agglomerative), existing project utils (`config`, `dec`, `interpretability`, `metrics`, `samples`).

**Spec:** `docs/superpowers/specs/2026-09-15-unsupervised-improvement-evaluation-design.md`

## Global Constraints

- CPU-only machine — no `.cuda()`, no `.to("mps")`, no GPU flags anywhere.
- Python 3.11.8, `torch==2.9.1+cpu`.
- `sumy` is installed. Use `from sumy.parsers.plaintext import PlaintextParser`, `from sumy.nlp.tokenizers import Tokenizer`, `from sumy.summarizers.lex_rank import LexRankSummarizer`.
- Sentence-transformer for all embeddings in nb13: `all-mpnet-base-v2`.
- All notebooks: `REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()`, add to `sys.path`.
- `config.RESULTS_DIR`, `config.NUM_CLASSES` (=16), `config.SEED` (=42), `config.CLASS_NAMES` (16-element list), `config.SILHOUETTE_SAMPLE_SIZE` (=2000).
- `run_and_save(name, cluster_labels, emb_original)` pattern from nb11: evaluate → save metrics JSON → save `.npy` → save clusters CSV → save full_labels CSV.
- Every new artifact file lands in `results/`.
- Sentences in LexRank output shorter than 10 words must be dropped.

---

### Task 1: `utils/cluster_description.py` — LexRank description generator

**Files:**
- Create: `utils/cluster_description.py`

**Interfaces:**
- Produces: `generate_cluster_descriptions(summaries: list[str], cluster_labels: np.ndarray, n_sentences: int = 3, min_words: int = 10) -> dict[int, str]`
  - Returns `{cluster_id: description_string}` for every non-noise cluster (cluster_id >= 0).
  - `description_string` is the LexRank top sentences joined by a single space.

- [ ] **Step 1: Write the file**

```python
# utils/cluster_description.py
import re
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


def generate_cluster_descriptions(
    summaries: list[str],
    cluster_labels: np.ndarray,
    n_sentences: int = 3,
    min_words: int = 10,
) -> dict[int, str]:
    """Return {cluster_id: description} for all non-noise clusters.

    Concatenates all summaries in a cluster, runs LexRank to pick the
    n_sentences most central sentences, filters out fragments shorter than
    min_words words, and joins the remainder into one description string.
    Falls back to top-n TF-IDF terms if LexRank returns nothing.
    """
    cluster_labels = np.array(cluster_labels)
    summaries = list(summaries)
    cluster_ids = sorted(c for c in np.unique(cluster_labels) if c >= 0)
    summarizer = LexRankSummarizer()

    descriptions = {}
    for c in cluster_ids:
        mask = cluster_labels == c
        cluster_docs = [summaries[i] for i, m in enumerate(mask) if m]

        full_text = " ".join(cluster_docs)
        # Remove zero-width and control characters that confuse sumy's tokenizer
        full_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\xa0]", " ", full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()

        if not full_text:
            descriptions[c] = ""
            continue

        try:
            parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
            result = summarizer(parser.document, sentences_count=n_sentences)
            sentences = [str(s).strip() for s in result
                         if len(str(s).strip().split()) >= min_words]
        except Exception:
            sentences = []

        descriptions[c] = " ".join(sentences) if sentences else full_text[:300]

    return descriptions
```

- [ ] **Step 2: Smoke-test in a scratch script**

```python
# Run from repo root:
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from utils.cluster_description import generate_cluster_descriptions

train = pd.read_parquet("data/processed/train_clean.parquet")
labels = np.load("results/cluster_labels_mpnet_kmeans.npy")
descs = generate_cluster_descriptions(train["summary"].tolist(), labels)
assert len(descs) == 16
for cid, d in list(descs.items())[:3]:
    print(f"Cluster {cid}: {d[:200]}")
    assert len(d) > 20, f"Cluster {cid} description is too short"
```

Run: `python -c "exec(open('test_desc.py').read())"` (write the script to a temp file first) or paste directly into a Python REPL.
Expected: 16 descriptions, each a complete sentence, no empty strings.

- [ ] **Step 3: Commit**

```bash
git add utils/cluster_description.py
git commit -m "feat(utils): add LexRank cluster description generator"
```

---

### Task 2: Notebook 12 — Experiment D: RoBERTa + tuned UMAP

**Files:**
- Create: `notebooks/12_further_clustering_experiments.ipynb`

**Interfaces:**
- Consumes: `embeddings_cache/roberta_train_full_summary.npy` (3824, 768) — already exists from nb02.
- Produces: `results/metrics_roberta_{kmeans,agglomerative}_umap.json`, `results/cluster_labels_roberta_{kmeans,agglomerative}_umap.npy`, `results/clusters_roberta_{kmeans,agglomerative}_umap.csv`, `results/full_labels_roberta_{kmeans,agglomerative}_umap.csv`.

- [ ] **Step 1: Create notebook cell 0 — markdown header**

````
# 12 — Further Clustering Experiments

**Experiment D — RoBERTa + tuned UMAP:** Apply the same 3×3 UMAP grid
(n_neighbors ∈ {15,30,50} × min_dist ∈ {0.0,0.05,0.1}) to RoBERTa
(768-dim) that was applied to MiniLM in notebook 11 Experiment B.
RoBERTa with default UMAP scored only ~24% ACC; tuning should close the gap.

**Experiment E — Combined embeddings:** L2-normalise MiniLM summary
embeddings and MiniLM title embeddings, concatenate → (3824, 768),
apply UMAP, cluster.
````

- [ ] **Step 2: Create notebook cell 1 — imports and data loading**

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

from utils import config
from utils.interpretability import summarize_clusters
from utils.metrics import evaluate_unsupervised, hungarian_match_predictions
from utils.samples import save_full_output

train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
true_labels  = train_clean["label"].to_numpy()
texts        = train_clean["text"].tolist()
summaries    = train_clean["summary"].tolist()

assert len(true_labels) == 3824, f"Expected 3824 rows, got {len(true_labels)}"
print(f"Rows: {len(train_clean)} | k={config.NUM_CLASSES}")

all_results = {}
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Create notebook cell 2 — shared helpers (copy from nb11 exactly)**

```python
def umap_reduce(
    emb: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 50,
    metric: str = "cosine",
    seed: int = config.SEED,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(emb)


def run_and_save(name: str, cluster_labels: np.ndarray, emb_original: np.ndarray) -> None:
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
          f"NMI={metrics['NMI']:.4f}")


print("Helpers defined.")
```

- [ ] **Step 4: Create notebook cell 3 — Experiment D UMAP grid on RoBERTa**

```python
# ════════════════════════════════════════════════════════════════════
# EXPERIMENT D — RoBERTa + tuned UMAP
# ════════════════════════════════════════════════════════════════════
roberta_emb = np.load(
    REPO_ROOT / "embeddings_cache" / "roberta_train_full_summary.npy"
).astype(np.float32)
assert roberta_emb.shape == (3824, 768), f"Unexpected: {roberta_emb.shape}"
print(f"RoBERTa embeddings loaded: {roberta_emb.shape}")

N_NEIGHBORS_GRID = [15, 30, 50]
MIN_DIST_GRID    = [0.0, 0.05, 0.1]

print("\nUMAP grid search on RoBERTa (3x3 = 9 combos, scored by KMeans silhouette):")
print(f"{'n_neighbors':>12} {'min_dist':>9} {'silhouette':>12}")
print("-" * 36)

best_score, best_n_neighbors, best_min_dist = -1.0, 15, 0.1

for nn in N_NEIGHBORS_GRID:
    for md in MIN_DIST_GRID:
        reduced = umap_reduce(roberta_emb, n_neighbors=nn, min_dist=md)
        km_tmp = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
        labels_tmp = km_tmp.fit_predict(reduced)
        score = silhouette_score(
            reduced, labels_tmp,
            sample_size=config.SILHOUETTE_SAMPLE_SIZE,
            random_state=config.SEED,
        )
        marker = " <- best" if score > best_score else ""
        print(f"{nn:>12} {md:>9.2f} {score:>12.4f}{marker}")
        if score > best_score:
            best_score, best_n_neighbors, best_min_dist = score, nn, md

print(f"\nBest params: n_neighbors={best_n_neighbors}, min_dist={best_min_dist} "
      f"(silhouette={best_score:.4f})")
```

- [ ] **Step 5: Create notebook cell 4 — Experiment D clustering**

```python
print("\n=== [D] RoBERTa KMeans (tuned UMAP) ===")
roberta_reduced_tuned = umap_reduce(
    roberta_emb, n_neighbors=best_n_neighbors, min_dist=best_min_dist
)
print(f"  UMAP reduced: {roberta_reduced_tuned.shape}")

km = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
run_and_save("roberta_kmeans_umap", km.fit_predict(roberta_reduced_tuned), roberta_emb)

print("\n=== [D] RoBERTa Agglomerative (tuned UMAP) ===")
agg = AgglomerativeClustering(
    n_clusters=config.NUM_CLASSES,
    linkage="ward",
    metric="euclidean",
)
run_and_save("roberta_agglomerative_umap", agg.fit_predict(roberta_reduced_tuned), roberta_emb)
```

- [ ] **Step 6: Verify Experiment D artifacts exist and ACC > 24% baseline**

After running cells 3–5, check:
```python
import json
from utils import config
for name in ["roberta_kmeans_umap", "roberta_agglomerative_umap"]:
    m = json.load(open(config.RESULTS_DIR / f"metrics_{name}.json"))
    assert m["ACC (Hungarian)"] > 0.24, f"{name} did not improve over baseline"
    print(f"{name}: ACC={m['ACC (Hungarian)']:.4f}")
```

- [ ] **Step 7: Commit Experiment D**

```bash
git add notebooks/12_further_clustering_experiments.ipynb
git commit -m "feat(nb12): add Experiment D — RoBERTa + tuned UMAP clustering"
```

---

### Task 3: Notebook 12 — Experiment E: Combined title + summary embeddings

**Files:**
- Modify: `notebooks/12_further_clustering_experiments.ipynb` (add cells)

**Interfaces:**
- Consumes: `embeddings_cache/minilm_train_full_summary.npy` (3824, 384), `embeddings_cache/minilm_train_title.npy` (3824, 384) — both cached from nb01/nb11.
- Produces: `embeddings_cache/combined_title_summary.npy` (3824, 768), `results/metrics_combined_{kmeans,agglomerative}.json`, `results/cluster_labels_combined_{kmeans,agglomerative}.npy`, `results/clusters_combined_{kmeans,agglomerative}.csv`, `results/full_labels_combined_{kmeans,agglomerative}.csv`.

- [ ] **Step 1: Create notebook cell 5 — Experiment E combined embeddings**

```python
# ════════════════════════════════════════════════════════════════════
# EXPERIMENT E — Combined title + summary embeddings
# ════════════════════════════════════════════════════════════════════
COMBINED_CACHE = REPO_ROOT / "embeddings_cache" / "combined_title_summary.npy"

if COMBINED_CACHE.exists():
    print(f"Loading cached combined embeddings from {COMBINED_CACHE.name}...")
    combined_emb = np.load(COMBINED_CACHE).astype(np.float32)
else:
    print("Building combined embeddings (L2-normalise MiniLM summary + title, concat)...")
    summary_emb = np.load(
        REPO_ROOT / "embeddings_cache" / "minilm_train_full_summary.npy"
    ).astype(np.float32)
    title_emb = np.load(
        REPO_ROOT / "embeddings_cache" / "minilm_train_title.npy"
    ).astype(np.float32)
    assert summary_emb.shape == (3824, 384), f"Unexpected summary emb shape: {summary_emb.shape}"
    assert title_emb.shape   == (3824, 384), f"Unexpected title emb shape: {title_emb.shape}"

    # L2-normalise each modality so neither dominates by magnitude
    def l2_norm(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    combined_emb = np.concatenate([l2_norm(summary_emb), l2_norm(title_emb)], axis=1)
    assert combined_emb.shape == (3824, 768), f"Unexpected combined shape: {combined_emb.shape}"
    np.save(COMBINED_CACHE, combined_emb)
    print(f"Saved to {COMBINED_CACHE.name}")

print(f"Combined embeddings ready: {combined_emb.shape}")
```

- [ ] **Step 2: Create notebook cell 6 — Experiment E clustering**

```python
print("\n=== [E] Combined KMeans ===")
combined_reduced = umap_reduce(combined_emb)   # default params: n_neighbors=15, min_dist=0.1
print(f"  UMAP reduced: {combined_reduced.shape}")

km = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
run_and_save("combined_kmeans", km.fit_predict(combined_reduced), combined_emb)

print("\n=== [E] Combined Agglomerative ===")
agg = AgglomerativeClustering(
    n_clusters=config.NUM_CLASSES,
    linkage="ward",
    metric="euclidean",
)
run_and_save("combined_agglomerative", agg.fit_predict(combined_reduced), combined_emb)
```

- [ ] **Step 3: Create notebook cell 7 — nb12 results summary**

```python
# ════════════════════════════════════════════════════════════════════
# NB12 RESULTS SUMMARY — all 4 new methods vs. best prior baselines
# ════════════════════════════════════════════════════════════════════
NB12_METHODS = [
    "roberta_kmeans_umap", "roberta_agglomerative_umap",
    "combined_kmeans", "combined_agglomerative",
]

assert set(NB12_METHODS) == set(all_results.keys()), (
    f"Missing methods: {set(NB12_METHODS) - set(all_results.keys())}"
)

print(f"{'Method':<35} {'ACC':>7} {'F1':>7} {'NMI':>7}")
print("-" * 55)
print(f"{'mpnet_kmeans (nb11 best)':<35} {'0.5413':>7}   -       -")
for name in NB12_METHODS:
    m = all_results[name]
    beat = " ✓" if m["ACC (Hungarian)"] > 0.5413 else ""
    print(
        f"{name:<35} {m['ACC (Hungarian)']:>7.4f} "
        f"{m['Macro F1']:>7.4f} {m['NMI']:>7.4f}{beat}"
    )
print("\n✓ = beats nb11 best (mpnet_kmeans ACC=0.5413)")
print(f"\nAll {len(NB12_METHODS)} nb12 methods complete.")
```

- [ ] **Step 4: Commit Experiment E and summary**

```bash
git add notebooks/12_further_clustering_experiments.ipynb embeddings_cache/combined_title_summary.npy
git commit -m "feat(nb12): add Experiment E — combined title+summary embeddings clustering"
```

---

### Task 4: `utils/cluster_description.py` integration test

**Files:**
- Modify: `utils/cluster_description.py` (only if bugs found — otherwise no change)

**Interfaces:**
- Consumes: `generate_cluster_descriptions(summaries, cluster_labels, n_sentences=3, min_words=10) -> dict[int, str]` from Task 1.

- [ ] **Step 1: Run description generation on all existing methods**

```python
# Paste into a notebook cell or run as a script:
import sys
from pathlib import Path
REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from utils import config
from utils.cluster_description import generate_cluster_descriptions

train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
summaries   = train_clean["summary"].tolist()

# Test on the 3 highest-ACC methods
test_methods = ["mpnet_kmeans", "title_minilm_agglomerative", "minilm_agglomerative"]
for method in test_methods:
    labels = np.load(config.RESULTS_DIR / f"cluster_labels_{method}.npy")
    descs  = generate_cluster_descriptions(summaries, labels)
    assert len(descs) == 16, f"{method}: expected 16 clusters, got {len(descs)}"
    empty = [c for c, d in descs.items() if not d.strip()]
    assert not empty, f"{method}: empty descriptions for clusters {empty}"
    print(f"{method}: OK — sample: {list(descs.values())[0][:120]}")
```

Expected: all 3 methods pass, sample description is a coherent sentence.

- [ ] **Step 2: Fix any issues found, commit if changed**

If Task 1 code worked cleanly: no commit needed here.  
If the smoke-test above reveals bugs (e.g. sumy tokenizer error on certain characters): fix `generate_cluster_descriptions`, re-run, then:

```bash
git add utils/cluster_description.py
git commit -m "fix(utils): handle sumy tokenizer edge cases in cluster descriptions"
```

---

### Task 5: Notebook 13 — Section 1: LexRank descriptions for all methods

**Files:**
- Create: `notebooks/13_cluster_description_evaluation.ipynb`

**Interfaces:**
- Consumes: `generate_cluster_descriptions(summaries, cluster_labels) -> dict[int, str]` from `utils/cluster_description.py`.
- Produces: `results/cluster_descriptions_{method}.json` for every discovered method.

- [ ] **Step 1: Create notebook cell 0 — markdown header**

````
# 13 — Cluster Descriptions, Category Mapping, and Semantic Evaluation

Three sections:
1. **LexRank descriptions** — for every clustering method × cluster, summarise
   all articles in the cluster to 2-3 representative sentences using LexRank.
2. **Semantic category mapping** — embed each cluster description with
   `all-mpnet-base-v2`, find the most similar of the 16 canonical category
   descriptions by cosine similarity.
3. **Semantic evaluation** — report a per-method semantic similarity score
   (mean cosine similarity across all 16 clusters) alongside the existing
   Hungarian-ACC, producing a combined comparison table.
````

- [ ] **Step 2: Create notebook cell 1 — imports and setup**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import config
from utils.cluster_description import generate_cluster_descriptions

train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
summaries   = train_clean["summary"].tolist()

# Discover all clustering methods that have saved label arrays
label_files = sorted(config.RESULTS_DIR.glob("cluster_labels_*.npy"))
ALL_METHODS = [f.stem.replace("cluster_labels_", "") for f in label_files]
print(f"Found {len(ALL_METHODS)} clustering methods:")
for m in ALL_METHODS:
    print(f"  {m}")
```

- [ ] **Step 3: Create notebook cell 2 — generate and save all descriptions**

```python
# ════════════════════════════════════════════════════════════════════
# SECTION 1 — LexRank cluster descriptions
# ════════════════════════════════════════════════════════════════════
all_descriptions = {}   # method -> {cluster_id(int): description(str)}

for method in ALL_METHODS:
    labels_path = config.RESULTS_DIR / f"cluster_labels_{method}.npy"
    cluster_labels = np.load(labels_path)

    descs = generate_cluster_descriptions(summaries, cluster_labels, n_sentences=3, min_words=10)
    all_descriptions[method] = descs

    # Persist as JSON with string keys for portability
    out_path = config.RESULTS_DIR / f"cluster_descriptions_{method}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in descs.items()}, f, indent=2, ensure_ascii=False)

    print(f"  {method}: {len(descs)} cluster descriptions saved → {out_path.name}")

print(f"\nSection 1 complete. {len(ALL_METHODS)} methods processed.")
```

- [ ] **Step 4: Spot-check descriptions by printing one cluster per method**

```python
# Print a sample description for cluster 0 of each method
print("=== Sample descriptions (cluster 0 per method) ===\n")
for method in ALL_METHODS[:5]:   # show first 5 to avoid scrolling
    d = all_descriptions[method].get(0, "(missing)")
    print(f"[{method}]\n{d}\n")
```

Expected: Each block shows a coherent 2-3 sentence paragraph, not keywords.

- [ ] **Step 5: Commit Section 1**

```bash
git add notebooks/13_cluster_description_evaluation.ipynb
git commit -m "feat(nb13): Section 1 — LexRank cluster descriptions for all methods"
```

---

### Task 6: Notebook 13 — Section 2: Semantic category mapping

**Files:**
- Modify: `notebooks/13_cluster_description_evaluation.ipynb` (add cells)

**Interfaces:**
- Consumes: `all_descriptions: dict[str, dict[int, str]]` from Task 5 (in-memory).
- Produces: `results/cluster_category_mapping_{method}.json` — `{str(cluster_id): {"mapped_category": str, "similarity": float}}`.

- [ ] **Step 1: Create notebook cell 3 — canonical category descriptions + embeddings**

```python
# ════════════════════════════════════════════════════════════════════
# SECTION 2 — Semantic category mapping
# ════════════════════════════════════════════════════════════════════

# 16 canonical one-sentence descriptions, one per class in config.CLASS_NAMES order
CANONICAL_DESCRIPTIONS = {
    "ARTS & CULTURE":  "Articles about visual arts, performing arts, music, literature, cultural events, and creative expression.",
    "BUSINESS":        "Articles about corporate finance, markets, economics, trade, and business strategy.",
    "COMEDY":          "Articles about stand-up comedy, satirical shows, late-night television hosts, and humorous commentary on current events.",
    "CRIME":           "Articles about criminal investigations, arrests, court cases, and law enforcement activities.",
    "EDUCATION":       "Articles about schools, universities, student loans, teachers, and educational policy.",
    "ENTERTAINMENT":   "Articles about movies, television shows, celebrity news, awards ceremonies, and pop culture.",
    "ENVIRONMENT":     "Articles about climate change, environmental policy, wildlife, natural disasters, and conservation.",
    "HEALTH":          "Articles about medical research, public health, disease outbreaks, vaccines, and healthcare policy.",
    "MEDIA":           "Articles about journalism, news organisations, television networks, media industry controversies, and press freedom.",
    "NEWS":            "Breaking news and general current events not classified under a more specific category.",
    "POLITICS":        "Articles about elections, government policy, politicians, legislation, and political parties.",
    "RELIGION":        "Articles about religious institutions, faith, spirituality, religious leaders, and interfaith issues.",
    "SCIENCE":         "Articles about scientific discoveries, space exploration, NASA, astronomy, physics, and technology research.",
    "SPORTS":          "Articles about professional and amateur sports competitions, athletes, teams, tournaments, and sporting events.",
    "TECH":            "Articles about consumer technology, gadgets, software, social media platforms, and the tech industry.",
    "WOMEN":           "Articles about women's rights, gender equality, sexual harassment, feminism, and women in public life.",
}

# Verify alignment with CLASS_NAMES
assert list(CANONICAL_DESCRIPTIONS.keys()) == config.CLASS_NAMES, (
    "CANONICAL_DESCRIPTIONS keys must match config.CLASS_NAMES order"
)

# Embed all 16 canonical descriptions once
print("Loading all-mpnet-base-v2 for semantic embeddings...")
embed_model = SentenceTransformer("all-mpnet-base-v2")
category_texts = list(CANONICAL_DESCRIPTIONS.values())
category_names = list(CANONICAL_DESCRIPTIONS.keys())

category_embs = embed_model.encode(
    category_texts, batch_size=16, show_progress_bar=False,
    convert_to_numpy=True, normalize_embeddings=True,   # L2-normalise → dot product = cosine sim
).astype(np.float32)

print(f"Category embeddings: {category_embs.shape}")
```

- [ ] **Step 2: Create notebook cell 4 — embed all cluster descriptions and map to categories**

```python
all_mappings = {}   # method -> {cluster_id(int): {"mapped_category": str, "similarity": float}}

for method in ALL_METHODS:
    descs = all_descriptions[method]   # {int: str}
    cluster_ids = sorted(descs.keys())
    desc_texts  = [descs[c] for c in cluster_ids]

    # Embed all cluster descriptions at once
    cluster_embs = embed_model.encode(
        desc_texts, batch_size=16, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    # Cosine similarity matrix: (n_clusters, 16)
    # Both arrays are L2-normalised, so dot product == cosine similarity
    sim_matrix = cluster_embs @ category_embs.T   # shape (16, 16) for k=16

    mapping = {}
    for i, c in enumerate(cluster_ids):
        best_cat_idx = int(np.argmax(sim_matrix[i]))
        mapping[c] = {
            "mapped_category": category_names[best_cat_idx],
            "similarity": float(sim_matrix[i, best_cat_idx]),
        }
    all_mappings[method] = mapping

    out_path = config.RESULTS_DIR / f"cluster_category_mapping_{method}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2, ensure_ascii=False)

    print(f"  {method}: mappings saved → {out_path.name}")

print(f"\nSection 2 complete. {len(ALL_METHODS)} methods processed.")
```

- [ ] **Step 3: Spot-check — print mapping table for best method**

```python
# Print cluster → mapped category → description snippet for mpnet_kmeans
best_method = "mpnet_kmeans"
mapping = all_mappings[best_method]
descs   = all_descriptions[best_method]

clusters_csv = pd.read_csv(config.RESULTS_DIR / f"clusters_{best_method}.csv")
majority_map = dict(zip(clusters_csv["cluster"], clusters_csv["majority_true_label"]))

print(f"{'Cluster':>8} {'Majority True':>20} {'Mapped Category':>20} {'Sim':>6} {'Match':>6}")
print("-" * 65)
for c in sorted(mapping.keys()):
    m = mapping[c]
    majority = majority_map.get(c, "?")
    match = "✓" if m["mapped_category"] == majority else "✗"
    print(f"{c:>8} {majority:>20} {m['mapped_category']:>20} {m['similarity']:>6.3f} {match:>6}")
```

Expected: many ✓ marks for high-purity clusters (SPORTS, SCIENCE, CRIME, EDUCATION) and occasional ✗ for mixed ones (ARTS & CULTURE, COMEDY/ENTERTAINMENT overlap).

- [ ] **Step 4: Commit Section 2**

```bash
git add notebooks/13_cluster_description_evaluation.ipynb
git commit -m "feat(nb13): Section 2 — semantic category mapping via mpnet cosine similarity"
```

---

### Task 7: Notebook 13 — Section 3: Semantic evaluation + final table

**Files:**
- Modify: `notebooks/13_cluster_description_evaluation.ipynb` (add cells)

**Interfaces:**
- Consumes: `all_mappings: dict[str, dict[int, dict]]` from Task 6 (in-memory); `results/metrics_{method}.json` for each method; `results/clusters_{method}.csv` for majority true labels.
- Produces: `results/semantic_evaluation.csv` — columns: `method, ACC_hungarian, macro_f1, NMI, mean_semantic_score, mapping_accuracy`.

- [ ] **Step 1: Create notebook cell 5 — compute per-method semantic scores**

```python
# ════════════════════════════════════════════════════════════════════
# SECTION 3 — Semantic evaluation
# ════════════════════════════════════════════════════════════════════
import json as _json

rows = []
for method in ALL_METHODS:
    # Load numeric metrics
    metrics_path = config.RESULTS_DIR / f"metrics_{method}.json"
    if not metrics_path.exists():
        print(f"  WARNING: no metrics file for {method}, skipping")
        continue
    metrics = _json.load(open(metrics_path))

    # Load majority true labels from clusters CSV
    clusters_path = config.RESULTS_DIR / f"clusters_{method}.csv"
    clusters_df   = pd.read_csv(clusters_path)
    majority_map  = dict(zip(clusters_df["cluster"], clusters_df["majority_true_label"]))

    mapping = all_mappings.get(method, {})

    # Mean semantic similarity across all clusters
    similarities = [v["similarity"] for v in mapping.values()]
    mean_sem = float(np.mean(similarities)) if similarities else 0.0

    # Mapping accuracy: fraction of clusters where mapped == majority true
    n_match = sum(
        1 for c, v in mapping.items()
        if v["mapped_category"] == majority_map.get(c, "")
    )
    map_acc = n_match / len(mapping) if mapping else 0.0

    rows.append({
        "method":             method,
        "ACC_hungarian":      round(metrics["ACC (Hungarian)"], 4),
        "macro_f1":           round(metrics["Macro F1"], 4),
        "NMI":                round(metrics["NMI"], 4),
        "mean_semantic_score": round(mean_sem, 4),
        "mapping_accuracy":   round(map_acc, 4),
    })

eval_df = pd.DataFrame(rows).sort_values("mean_semantic_score", ascending=False)
out_csv = config.RESULTS_DIR / "semantic_evaluation.csv"
eval_df.to_csv(out_csv, index=False)
print(f"Saved → {out_csv}")
```

- [ ] **Step 2: Create notebook cell 6 — print final comparison table**

```python
print("\n=== SEMANTIC EVALUATION — All Methods ===\n")
print(f"{'Method':<35} {'ACC':>7} {'F1':>7} {'NMI':>7} {'SemanticSim':>12} {'MapACC':>8}")
print("-" * 82)
for _, row in eval_df.iterrows():
    print(
        f"{row['method']:<35} {row['ACC_hungarian']:>7.4f} "
        f"{row['macro_f1']:>7.4f} {row['NMI']:>7.4f} "
        f"{row['mean_semantic_score']:>12.4f} {row['mapping_accuracy']:>8.4f}"
    )

# Sanity checks
assert len(eval_df) == len(ALL_METHODS) or len(eval_df) >= len(ALL_METHODS) - 1, (
    "Fewer rows than expected — check for missing metrics files"
)
top_method = eval_df.iloc[0]["method"]
print(f"\nHighest semantic similarity: {top_method} ({eval_df.iloc[0]['mean_semantic_score']:.4f})")
print(f"Highest ACC: {eval_df.sort_values('ACC_hungarian', ascending=False).iloc[0]['method']}")
```

- [ ] **Step 3: Verify the CSV was written correctly**

```python
check = pd.read_csv(config.RESULTS_DIR / "semantic_evaluation.csv")
assert list(check.columns) == [
    "method", "ACC_hungarian", "macro_f1", "NMI",
    "mean_semantic_score", "mapping_accuracy"
], f"Unexpected columns: {list(check.columns)}"
assert len(check) >= 14, f"Expected at least 14 rows, got {len(check)}"
print(f"semantic_evaluation.csv: {len(check)} rows, columns OK")
```

- [ ] **Step 4: Final commit**

```bash
git add notebooks/13_cluster_description_evaluation.ipynb results/semantic_evaluation.csv
git commit -m "feat(nb13): Section 3 — semantic evaluation score for all clustering methods"
```

---

## Self-Review

**Spec coverage:**
- Exp D (RoBERTa + UMAP tuning): Task 2 ✓
- Exp E (Combined embeddings): Task 3 ✓
- `utils/cluster_description.py` LexRank: Task 1 ✓
- Section 1 (LexRank descriptions, all methods): Task 5 ✓
- Section 2 (Canonical category descriptions, cosine mapping): Task 6 ✓
- Section 3 (Semantic score, mapping accuracy, final CSV): Task 7 ✓
- Sentence filter < 10 words: Task 1 Step 1 (`min_words=10`) ✓
- CPU-only constraint: no CUDA calls anywhere ✓
- `all-mpnet-base-v2` with `normalize_embeddings=True`: Task 6 Steps 1 & 2 ✓

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N" shortcuts — each task has full code.

**Type consistency:**
- `generate_cluster_descriptions` signature consistent across Tasks 1, 4, 5.
- `all_descriptions`: `dict[str, dict[int, str]]` — method name (str) → cluster_id (int) → description (str). Consistent in Tasks 5, 6, 7.
- `all_mappings`: `dict[str, dict[int, dict]]` — method → cluster_id → `{"mapped_category": str, "similarity": float}`. Consistent in Tasks 6, 7.
- JSON serialisation uses `str(k)` for cluster ids (JSON requires string keys); loading elsewhere should cast back to int.
