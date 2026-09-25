# tests/test_image_weak_supervision.py
import numpy as np
import pytest


def _dummy_embeddings(n=20, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def test_build_clip_lfs_returns_one_lf_per_class():
    from utils.image_weak_supervision import build_clip_lfs

    image_embs = _dummy_embeddings(10)
    class_names = ["cat", "dog"]
    text_embs = _dummy_embeddings(2)
    thresholds = [0.5, 0.5]

    lfs = build_clip_lfs(image_embs, class_names, text_embs, thresholds)
    assert len(lfs) == 2


def test_clip_lf_returns_label_or_abstain():
    from utils.image_weak_supervision import build_clip_lfs

    # Make text_embs identical to image_embs[0] to guarantee a high sim for class 0
    image_embs = _dummy_embeddings(3)
    text_embs = np.vstack([image_embs[0:1], _dummy_embeddings(1, seed=99)])
    thresholds = [0.5, 0.5]

    lfs = build_clip_lfs(image_embs, ["cat", "dog"], text_embs, thresholds)
    # lf(row_idx) returns int label or -1
    for i in range(3):
        for lf in lfs:
            result = lf(i)
            assert result in [-1, 0, 1], f"Unexpected LF output: {result}"


def test_tune_thresholds_returns_per_class_float():
    from utils.image_weak_supervision import tune_thresholds

    seed_embs = _dummy_embeddings(10)
    seed_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    text_embs = _dummy_embeddings(2)

    thresholds = tune_thresholds(seed_embs, seed_labels, n_classes=2, text_embeddings=text_embs)
    assert len(thresholds) == 2
    assert all(0.0 <= t <= 1.0 for t in thresholds)
