# utils/image_pseudo_labeling.py
"""MLP-based pseudo-labeling (self-training) on frozen CLIP embeddings.

No full ViT fine-tuning — just a small MLP head. CPU-feasible even at
5,500 images: training on ~275–2,000 samples takes seconds.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              hidden_sizes: tuple = (256, 128), seed: int = 42) -> Pipeline:
    """Fit MLP on (X_train, y_train). Returns sklearn Pipeline (scaler + MLP)."""
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation="relu",
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])
    clf.fit(X_train, y_train)
    return clf


def pseudo_label_loop(
    embeddings: np.ndarray,
    seed_mask: np.ndarray,
    seed_labels: np.ndarray,
    hidden_sizes: tuple = (256, 128),
    threshold: float = 0.9,
    max_iters: int = 10,
    seed: int = 42,
    n_classes: int = 10,
) -> tuple:
    """Self-training loop: train MLP on seed, absorb high-confidence predictions,
    retrain, repeat until no new pseudo-labels or max_iters.

    Returns:
        pred_labels: (N,) int array — final predicted label for every row
        history: list of {iteration, new_pseudo_labels, cumulative_coverage}
    """
    present = set(np.unique(seed_labels))
    expected = set(range(n_classes))
    assert present == expected, f"Seed missing classes: {expected - present}"

    labeled_mask = seed_mask.copy()
    labeled_labels = np.full(len(embeddings), -1, dtype=int)
    labeled_labels[seed_mask] = seed_labels

    history = []

    for it in range(max_iters):
        X_train = embeddings[labeled_mask]
        y_train = labeled_labels[labeled_mask]

        clf = train_mlp(X_train, y_train, hidden_sizes=hidden_sizes, seed=seed + it)

        unlabeled_mask = ~labeled_mask
        if unlabeled_mask.sum() == 0:
            break

        proba = clf.predict_proba(embeddings[unlabeled_mask])
        confidence = proba.max(axis=1)
        predictions = proba.argmax(axis=1)

        high_conf = confidence >= threshold
        unlabeled_indices = np.where(unlabeled_mask)[0]
        new_indices = unlabeled_indices[high_conf]

        labeled_labels[new_indices] = predictions[high_conf]
        labeled_mask[new_indices] = True

        coverage = labeled_mask.mean()
        history.append({
            "iteration": it,
            "new_pseudo_labels": int(high_conf.sum()),
            "cumulative_coverage": float(coverage),
        })
        print(f"  Iter {it}: +{high_conf.sum()} pseudo-labels  coverage={coverage:.1%}")

        if high_conf.sum() == 0:
            break

    # Final retrain on all labeled data
    clf_final = train_mlp(
        embeddings[labeled_mask], labeled_labels[labeled_mask],
        hidden_sizes=hidden_sizes, seed=seed + max_iters,
    )
    # Predict for any remaining unlabeled rows
    still_unlabeled = ~labeled_mask
    if still_unlabeled.sum() > 0:
        labeled_labels[still_unlabeled] = clf_final.predict(embeddings[still_unlabeled])

    return labeled_labels, history
