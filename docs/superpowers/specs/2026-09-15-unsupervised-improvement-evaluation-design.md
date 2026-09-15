# Unsupervised Improvement & Cluster Evaluation Design

**Date:** 2026-09-15  
**Status:** Approved

## Goal

Improve the best unsupervised clustering accuracy (currently 54.13% mpnet_kmeans), then build a semantic evaluation framework that generates a natural-language description for every cluster, maps it to one of the 16 real categories using cosine similarity, and scores each method with a RAGAS-style semantic similarity metric.

## Background

Notebooks 02 and 11 produced 14 clustering methods (MiniLM/RoBERTa/MPNet/Title × KMeans/Agglomerative/DEC). Two clear gaps remain:
- **RoBERTa** uses default UMAP and scores only ~24% ACC. Notebook 11 Experiment B showed UMAP tuning lifted MiniLM from 45% → 47%; the same treatment on RoBERTa is likely to produce a large gain.
- **No joint embedding** has been tried. Title embeddings (51–52%) and summary embeddings (54%) capture complementary signals; concatenating them may outperform either alone.

Separately, all existing evaluation is numeric (ACC, NMI, ARI). There is no way to see *what the model thinks a cluster is about* or how well that understanding aligns with the true categories.

## Scope

### Notebook 12 — Further Clustering Experiments

**Experiment D — RoBERTa + tuned UMAP**  
Cached: `embeddings_cache/roberta_train_full_summary.npy` (3824, 768).  
Run the same 3×3 UMAP grid as nb11 Experiment B (n_neighbors ∈ {15,30,50} × min_dist ∈ {0.0, 0.05, 0.1}).  
Score each combination by KMeans silhouette. Run KMeans (k=16) + Agglomerative (k=16) with best params. Skip DEC (does not use UMAP).  
New method names: `roberta_kmeans_umap`, `roberta_agglomerative_umap`.

**Experiment E — Combined title + summary embeddings**  
Load `embeddings_cache/minilm_train_full_summary.npy` (3824, 384) and `embeddings_cache/minilm_train_title.npy` (3824, 384).  
L2-normalise each array independently, then concatenate along axis=1 → (3824, 768).  
Save the combined array to `embeddings_cache/combined_title_summary.npy`.  
Apply default UMAP (n_components=50, metric="cosine") → run KMeans (k=16) + Agglomerative (k=16).  
New method names: `combined_kmeans`, `combined_agglomerative`.

Both experiments follow the exact same `run_and_save()` artifact schema as notebooks 02 and 11 (metrics JSON, cluster labels `.npy`, clusters CSV, full_labels CSV).

### Notebook 13 — Cluster Descriptions, Category Mapping, and Semantic Evaluation

Operates on **all** clustering methods: the 6 from nb02, 8 from nb11, and up to 4 from nb12 — whichever `cluster_labels_*.npy` files exist in `results/`.

**Section 1 — LexRank Cluster Descriptions**  
For each method × cluster (up to 18 methods × 16 clusters):
- Collect all BART summaries in the cluster.
- Concatenate into one document, split into sentences, filter out sentences shorter than 10 words (removes truncated BART artifacts like "The former won.").
- Run `sumy` LexRankSummarizer with `sentences_count=3`.
- Save `results/cluster_descriptions_{method}.json` — dict mapping `str(cluster_id)` → description string.

**Section 2 — Semantic Category Mapping**  
16 canonical category descriptions (one sentence each, defined inline in the notebook):

| Category | Canonical Description |
|---|---|
| ARTS & CULTURE | Articles about visual arts, performing arts, music, literature, cultural events, and creative expression. |
| BUSINESS | Articles about corporate finance, markets, economics, trade, and business strategy. |
| COMEDY | Articles about stand-up comedy, satirical shows, late-night television hosts, and humorous commentary on current events. |
| CRIME | Articles about criminal investigations, arrests, court cases, and law enforcement activities. |
| EDUCATION | Articles about schools, universities, student loans, teachers, and educational policy. |
| ENTERTAINMENT | Articles about movies, television shows, celebrity news, awards ceremonies, and pop culture. |
| ENVIRONMENT | Articles about climate change, environmental policy, wildlife, natural disasters, and conservation. |
| HEALTH | Articles about medical research, public health, disease outbreaks, vaccines, and healthcare policy. |
| MEDIA | Articles about journalism, news organisations, television networks, media industry controversies, and press freedom. |
| NEWS | Breaking news and general current events not classified under a more specific category. |
| POLITICS | Articles about elections, government policy, politicians, legislation, and political parties. |
| RELIGION | Articles about religious institutions, faith, spirituality, religious leaders, and interfaith issues. |
| SCIENCE | Articles about scientific discoveries, space exploration, NASA, astronomy, physics, and technology research. |
| SPORTS | Articles about professional and amateur sports competitions, athletes, teams, tournaments, and sporting events. |
| TECH | Articles about consumer technology, gadgets, software, social media platforms, and the tech industry. |
| WOMEN | Articles about women's rights, gender equality, sexual harassment, feminism, and women in public life. |

Embed all cluster descriptions **and** all 16 canonical descriptions with `all-mpnet-base-v2` (already installed).  
Cosine similarity → argmax → mapped category name per cluster.  
Save `results/cluster_category_mapping_{method}.json` — dict mapping `str(cluster_id)` → `{"mapped_category": str, "similarity": float}`.

**Section 3 — Semantic Evaluation**  
For each cluster: semantic score = cosine similarity between cluster description embedding and its mapped category embedding (already computed in Section 2).  
Per-method aggregate: mean semantic score across all 16 clusters.  
Also compute: mapping accuracy = fraction of clusters where `mapped_category == majority_true_label` (from `clusters_{method}.csv`).  
Final output: `results/semantic_evaluation.csv` — columns: `method, ACC_hungarian, macro_f1, mean_semantic_score, mapping_accuracy`, sorted by `mean_semantic_score` descending.  
Print a combined table in the notebook.

## New Utility Module

`utils/cluster_description.py` — one function:

```python
def generate_cluster_descriptions(
    summaries: list[str],
    cluster_labels: np.ndarray,
    n_sentences: int = 3,
    min_words: int = 10,
) -> dict[int, str]:
    """Return {cluster_id: description} for all non-noise clusters using LexRank."""
```

## Artifacts Produced

| File | Producer |
|---|---|
| `results/metrics_roberta_{kmeans,agglomerative}_umap.json` | nb12 Exp D |
| `results/cluster_labels_roberta_{kmeans,agglomerative}_umap.npy` | nb12 Exp D |
| `results/clusters_roberta_{kmeans,agglomerative}_umap.csv` | nb12 Exp D |
| `results/full_labels_roberta_{kmeans,agglomerative}_umap.csv` | nb12 Exp D |
| `embeddings_cache/combined_title_summary.npy` | nb12 Exp E |
| `results/metrics_combined_{kmeans,agglomerative}.json` | nb12 Exp E |
| `results/cluster_descriptions_{method}.json` | nb13 §1 |
| `results/cluster_category_mapping_{method}.json` | nb13 §2 |
| `results/semantic_evaluation.csv` | nb13 §3 |

## Constraints

- CPU-only machine (no CUDA/MPS). All inference must work on CPU.
- Python 3.11.8, PyTorch 2.9.1+cpu.
- `sumy` must be installed (`pip install sumy`). Already installed in this session.
- `gensim` 4.4.0 is installed but not used (summarization removed in v4).
- All notebooks follow the existing pattern: load from `train_clean.parquet`, use `config.RESULTS_DIR`, call `run_and_save()`.
- Sentence-transformer model for embeddings: `all-mpnet-base-v2` (already downloaded).
- All methods from nb02/nb11 are available via their `cluster_labels_*.npy` files.
