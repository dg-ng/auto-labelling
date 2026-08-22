"""Label propagation / label spreading — a semi-supervised method with no
fine-tuning loop: build a k-NN graph over embeddings and propagate the small
labeled seed's labels through the graph algebraically.

See `docs/semi_supervised_methods.md` for why this method was chosen.
"""

import numpy as np


def run_label_propagation(labeled_embeddings, labeled_labels, unlabeled_embeddings,
                           kernel="knn", n_neighbors=7, gamma=20, max_iter=1000):
    """Fit `sklearn.semi_supervised.LabelSpreading` over the combined
    labeled+unlabeled feature matrix and return predictions for the
    unlabeled rows only.

    LabelSpreading (rather than LabelPropagation) is used because it's
    regularized against noisy edges in the k-NN graph, at the cost of not
    perfectly preserving the original seed labels — an acceptable trade-off
    here since the seed labels themselves are just 5% samples, not curated.

    Returns (predicted_labels, confidence) — confidence is each unlabeled
    row's max label-distribution probability.
    """
    from sklearn.semi_supervised import LabelSpreading

    labeled_embeddings = np.asarray(labeled_embeddings)
    unlabeled_embeddings = np.asarray(unlabeled_embeddings)
    labeled_labels = np.asarray(labeled_labels)

    X = np.vstack([labeled_embeddings, unlabeled_embeddings])
    y = np.concatenate([labeled_labels, np.full(len(unlabeled_embeddings), -1)])

    model = LabelSpreading(kernel=kernel, n_neighbors=n_neighbors, gamma=gamma, max_iter=max_iter)
    model.fit(X, y)

    n_labeled = len(labeled_embeddings)
    predicted_labels = model.transduction_[n_labeled:]
    confidence = model.label_distributions_[n_labeled:].max(axis=1)
    return predicted_labels, confidence
