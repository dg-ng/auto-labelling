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


def evaluate_unsupervised(true_labels, cluster_labels, embeddings) -> dict:
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    mask = cluster_labels >= 0

    return {
        "ACC (Hungarian)": clustering_accuracy(true_labels[mask], cluster_labels[mask]),
        "NMI": normalized_mutual_info_score(true_labels[mask], cluster_labels[mask]),
        "ARI": adjusted_rand_score(true_labels[mask], cluster_labels[mask]),
        "FMI": fowlkes_mallows_score(true_labels[mask], cluster_labels[mask]),
        "Homogeneity": homogeneity_score(true_labels[mask], cluster_labels[mask]),
        "Completeness": completeness_score(true_labels[mask], cluster_labels[mask]),
        "V-Measure": v_measure_score(true_labels[mask], cluster_labels[mask]),
        "Silhouette Score": silhouette_score(embeddings[mask], cluster_labels[mask], metric="cosine"),
        "Davies-Bouldin": davies_bouldin_score(embeddings[mask], cluster_labels[mask]),
        "Coverage": mask.sum() / len(cluster_labels),
    }


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
