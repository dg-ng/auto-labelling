# utils/image_weak_supervision.py
"""CLIP-based zero-shot labeling functions for Snorkel weak supervision.

Each LF computes cosine similarity between an image's CLIP embedding and
a text embedding of "a photo of a {class}". If similarity >= threshold,
it votes for that class; else abstains (-1).
"""
import numpy as np


def tune_thresholds(
    seed_embeddings: np.ndarray,
    seed_labels: np.ndarray,
    n_classes: int,
    text_embeddings: np.ndarray,
    percentile: float = 30.0,
) -> list[float]:
    """For each class, set threshold = percentile of cosine sims for seed rows of that class.

    A lower percentile means the LF fires more often (higher recall, lower precision).
    30th percentile keeps enough coverage while avoiding noise.
    """
    # L2-normalise image embeddings for cosine sim via dot product
    norms = np.linalg.norm(seed_embeddings, axis=1, keepdims=True)
    emb_norm = seed_embeddings / np.maximum(norms, 1e-8)

    thresholds = []
    for c in range(n_classes):
        mask = seed_labels == c
        if mask.sum() == 0:
            thresholds.append(0.0)
            continue
        sims = emb_norm[mask] @ text_embeddings[c]   # (n_seed_c,)
        thresholds.append(float(np.percentile(sims, percentile)))
    return thresholds


def build_clip_lfs(
    image_embeddings: np.ndarray,
    class_names: list[str],
    text_embeddings: np.ndarray,
    thresholds: list[float],
):
    """Build one callable per class. Each callable takes a row index (int)
    and returns the class label int or -1 (abstain).

    Returns list of callables (not Snorkel LabelingFunction wrappers —
    those are built in the notebook where Snorkel is imported).
    """
    norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    emb_norm = (image_embeddings / np.maximum(norms, 1e-8)).astype(np.float32)

    lfs = []
    for c, (cls_name, threshold) in enumerate(zip(class_names, thresholds)):
        text_vec = text_embeddings[c].astype(np.float32)

        def _make_lf(label, vec, thresh):
            def lf(row_idx: int) -> int:
                sim = float(emb_norm[row_idx] @ vec)
                return label if sim >= thresh else -1
            return lf

        lfs.append(_make_lf(c, text_vec, threshold))
    return lfs
