"""Qualitative inspection of unsupervised clustering results.

The metrics in `utils/metrics.py` (ACC, NMI, ARI, ...) are permutation-invariant
number-matching between cluster IDs and true label IDs — they never surface
*what a cluster is actually about*. This module produces the human-readable
side: representative terms per cluster, example documents, and a crosstab
against the true labels, so a cluster like "soccer, goal, league, match" can
be recognized as a good "Sports" cluster even where the numeric metrics alone
wouldn't make that legible.

Eval-only — nothing here is used as a training/loop-control signal.
"""

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.feature_extraction.text import TfidfVectorizer


def top_terms_per_cluster(texts, cluster_labels, n_terms=10, exclude_noise=True) -> dict:
    """Representative TF-IDF terms per cluster.

    Fits one TF-IDF vectorizer over the whole corpus (so term weights are
    comparable across clusters), then ranks terms within each cluster by
    their mean TF-IDF weight. Works regardless of what the clustering itself
    ran on (TF-IDF, MiniLM, RoBERTa, BERTopic) since this is post-hoc.

    Returns {cluster_id: [term, ...]}, noise cluster (-1) excluded by default.
    """
    texts = list(texts)
    cluster_labels = np.array(cluster_labels)

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2)
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        # min_df=2 pruned the vocab to nothing (tiny/sparse corpus) — fall
        # back to no pruning rather than crash the notebook.
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=1)
        tfidf = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    cluster_ids = sorted(np.unique(cluster_labels))
    if exclude_noise:
        cluster_ids = [c for c in cluster_ids if c >= 0]

    terms_by_cluster = {}
    for c in cluster_ids:
        mask = cluster_labels == c
        if mask.sum() == 0:
            terms_by_cluster[c] = []
            continue
        mean_weights = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = mean_weights.argsort()[::-1][:n_terms]
        terms_by_cluster[c] = vocab[top_idx].tolist()
    return terms_by_cluster


def example_docs_per_cluster(texts, cluster_labels, embeddings, n_examples=3, exclude_noise=True) -> dict:
    """The `n_examples` documents nearest each cluster's centroid (cosine).

    Faster for a human to eyeball than a term list alone. Returns
    {cluster_id: [text, ...]}.
    """
    texts = np.array(texts, dtype=object)
    cluster_labels = np.array(cluster_labels)
    embeddings = np.asarray(embeddings)

    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit_embeddings = embeddings / norm

    cluster_ids = sorted(np.unique(cluster_labels))
    if exclude_noise:
        cluster_ids = [c for c in cluster_ids if c >= 0]

    examples_by_cluster = {}
    for c in cluster_ids:
        mask = cluster_labels == c
        if mask.sum() == 0:
            examples_by_cluster[c] = []
            continue
        centroid = unit_embeddings[mask].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        sims = unit_embeddings[mask] @ centroid
        top_local_idx = sims.argsort()[::-1][:n_examples]
        examples_by_cluster[c] = texts[mask][top_local_idx].tolist()
    return examples_by_cluster


def cluster_label_crosstab(cluster_labels, true_labels, class_names) -> pd.DataFrame:
    """Cluster x true-class counts, plus majority class and purity per cluster.

    Purity = (# docs in cluster matching its own majority class) / (cluster size).
    Noise cluster (-1) is included as its own row so its size/spread is visible.
    """
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)

    rows = []
    for c in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == c
        size = int(mask.sum())
        counts = np.bincount(true_labels[mask], minlength=len(class_names))
        majority_idx = int(counts.argmax())
        rows.append({
            "cluster": c,
            "size": size,
            "majority_true_label": "noise" if c < 0 else class_names[majority_idx],
            "purity": counts[majority_idx] / size if size else 0.0,
            **{f"n_{name}": int(n) for name, n in zip(class_names, counts)},
        })
    return pd.DataFrame(rows)


def summarize_clusters(texts, cluster_labels, true_labels, embeddings, class_names,
                        n_terms=10, n_examples=3) -> pd.DataFrame:
    """One row per non-noise cluster: size, majority true label, purity,
    top TF-IDF terms, and example documents nearest the centroid.

    This is the table to print/eyeball — it's the qualitative counterpart to
    `utils.metrics.evaluate_unsupervised`'s aggregate scores.
    """
    terms = top_terms_per_cluster(texts, cluster_labels, n_terms=n_terms)
    examples = example_docs_per_cluster(texts, cluster_labels, embeddings, n_examples=n_examples)
    crosstab = cluster_label_crosstab(cluster_labels, true_labels, class_names)
    crosstab = crosstab[crosstab["cluster"] >= 0].copy()

    crosstab["top_terms"] = crosstab["cluster"].map(lambda c: ", ".join(terms.get(c, [])))
    crosstab["example_docs"] = crosstab["cluster"].map(
        lambda c: [t[:120] for t in examples.get(c, [])])

    cols = ["cluster", "size", "majority_true_label", "purity", "top_terms", "example_docs"]
    return crosstab[cols].sort_values("cluster").reset_index(drop=True)
