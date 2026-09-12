from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


def clustering_accuracy(true_labels, cluster_labels) -> float:
    """Hungarian-matched best cluster-to-class accuracy. Excludes noise (-1)."""
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    class_ids = np.unique(true_labels)
    cluster_ids = np.unique(cluster_labels[cluster_labels >= 0])
    n_classes = len(class_ids)
    n_clusters = len(cluster_ids)
    size = max(n_classes, n_clusters)

    cost_matrix = np.zeros((size, size))
    for ci, c in enumerate(cluster_ids):
        for ki, k in enumerate(class_ids):
            cost_matrix[ci, ki] = np.sum(
                (cluster_labels == c) & (true_labels == k))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    correct = cost_matrix[row_ind, col_ind].sum()
    total = (cluster_labels >= 0).sum()
    if total == 0:
        return 0.0
    return correct / total


def evaluate_unsupervised(
    true_labels,
    cluster_labels,
    embeddings,
    metric_sample_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Evaluate unsupervised clustering quality.

    Args:
        true_labels: Ground-truth integer class labels.
        cluster_labels: Predicted cluster labels (noise = -1).
        embeddings: Original embedding matrix (rows match true_labels).
        metric_sample_size: If set, subsample this many rows for Silhouette /
            Davies-Bouldin computation (expensive on large datasets).
        seed: Random seed used when subsampling.
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    mask = cluster_labels >= 0

    # Handle edge case: if insufficient non-noise clusters for silhouette/davies-bouldin
    # (both require at least 2 distinct cluster labels), return default values
    if mask.sum() == 0 or len(np.unique(cluster_labels[mask])) < 2:
        return {
            "ACC (Hungarian)": 0.0,
            "Macro F1": 0.0,
            "NMI": 0.0,
            "ARI": 0.0,
            "FMI": 0.0,
            "Homogeneity": 0.0,
            "Completeness": 0.0,
            "V-Measure": 0.0,
            "Silhouette Score": 0.0,
            "Davies-Bouldin": 0.0,
            "Coverage": mask.sum() / len(cluster_labels),
        }

    # Hungarian-matched accuracy and F1
    acc = clustering_accuracy(true_labels[mask], cluster_labels[mask])
    matched_pred = _hungarian_remap(true_labels[mask], cluster_labels[mask])
    macro_f1 = f1_score(true_labels[mask], matched_pred, average="macro", zero_division=0)

    # Subsample for expensive distance-based metrics
    emb_masked = np.array(embeddings)[mask]
    cl_masked = cluster_labels[mask]
    if metric_sample_size is not None and len(cl_masked) > metric_sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(cl_masked), size=metric_sample_size, replace=False)
        emb_s, cl_s = emb_masked[idx], cl_masked[idx]
    else:
        emb_s, cl_s = emb_masked, cl_masked

    return {
        "ACC (Hungarian)": acc,
        "Macro F1": macro_f1,
        "NMI": normalized_mutual_info_score(true_labels[mask], cluster_labels[mask]),
        "ARI": adjusted_rand_score(true_labels[mask], cluster_labels[mask]),
        "FMI": fowlkes_mallows_score(true_labels[mask], cluster_labels[mask]),
        "Homogeneity": homogeneity_score(true_labels[mask], cluster_labels[mask]),
        "Completeness": completeness_score(true_labels[mask], cluster_labels[mask]),
        "V-Measure": v_measure_score(true_labels[mask], cluster_labels[mask]),
        "Silhouette Score": silhouette_score(emb_s, cl_s, metric="cosine"),
        "Davies-Bouldin": davies_bouldin_score(emb_s, cl_s),
        "Coverage": mask.sum() / len(cluster_labels),
    }


def _hungarian_remap(true_labels: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """Return cluster_labels remapped to class indices via Hungarian matching."""
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    class_ids = np.unique(true_labels)
    cluster_ids = np.unique(cluster_labels)
    size = max(len(class_ids), len(cluster_ids))

    cost_matrix = np.zeros((size, size))
    for ci, c in enumerate(cluster_ids):
        for ki, k in enumerate(class_ids):
            cost_matrix[ci, ki] = np.sum((cluster_labels == c) & (true_labels == k))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class = {int(cluster_ids[r]): int(class_ids[c])
                        for r, c in zip(row_ind, col_ind)
                        if r < len(cluster_ids) and c < len(class_ids)}

    remapped = np.array([cluster_to_class.get(int(cl), -1) for cl in cluster_labels])
    return remapped


def hungarian_match_predictions(true_labels, cluster_labels) -> np.ndarray:
    """Remap cluster_labels to class indices via Hungarian assignment.

    Handles *all* rows (including noise / unassigned rows with label -1).
    Noise rows are left as -1 in the output.

    Args:
        true_labels: Ground-truth integer class labels (all rows).
        cluster_labels: Raw cluster label array (all rows; -1 = noise).

    Returns:
        Integer array of the same length with cluster IDs replaced by the
        best-matching class IDs per the Hungarian algorithm.
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    mask = cluster_labels >= 0

    result = np.full_like(cluster_labels, fill_value=-1)
    if mask.sum() > 0:
        result[mask] = _hungarian_remap(true_labels[mask], cluster_labels[mask])
    return result


def majority_vote_mapping(cluster_labels, true_labels, n_clusters) -> dict:
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)
    mapping = {}
    for c in range(n_clusters):
        cluster_mask = cluster_labels == c
        if cluster_mask.sum() == 0:
            continue
        mapping[c] = int(mode(true_labels[cluster_mask], keepdims=True).mode[0])
    return mapping


def evaluate_semisupervised(true_labels, predicted_labels, class_names, save_path=None):
    results = {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "Macro F1": f1_score(true_labels, predicted_labels, average="macro"),
        "Weighted F1": f1_score(true_labels, predicted_labels, average="weighted"),
        "Macro Precision": precision_score(true_labels, predicted_labels, average="macro"),
        "Macro Recall": recall_score(true_labels, predicted_labels, average="macro"),
        "Cohen's Kappa": cohen_kappa_score(true_labels, predicted_labels),
    }

    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    cm = confusion_matrix(true_labels, predicted_labels)

    if save_path is not None:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    return results, report, cm


def evaluate_label_quality(true_labels, pseudo_labels, confidence_scores=None) -> dict:
    true_labels = np.array(true_labels)
    pseudo_labels = np.array(pseudo_labels)
    mask = pseudo_labels >= 0

    # Handle edge case: no pseudo-labels were absorbed (e.g. confidence
    # threshold never met, or pseudo_labels is empty) — accuracy_score/f1_score
    # raise on empty input, so short-circuit with explicit zeros instead.
    if mask.sum() == 0:
        results = {"Label Accuracy": 0.0, "Label Macro F1": 0.0, "Coverage": 0.0}
        if confidence_scores is not None:
            results["Mean Confidence"] = 0.0
            results["Median Confidence"] = 0.0
        return results

    results = {
        "Label Accuracy": accuracy_score(true_labels[mask], pseudo_labels[mask]),
        "Label Macro F1": f1_score(true_labels[mask], pseudo_labels[mask], average="macro"),
        "Coverage": mask.sum() / len(pseudo_labels),
    }

    if confidence_scores is not None:
        confidence_scores = np.array(confidence_scores)
        results["Mean Confidence"] = float(confidence_scores[mask].mean())
        results["Median Confidence"] = float(np.median(confidence_scores[mask]))

    return results
