import numpy as np
import pytest


def _make_embeddings(n=100, dim=512, n_classes=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = (np.arange(n) % n_classes).astype(int)
    return X, y


def test_train_mlp_predicts_correct_shape():
    from utils.image_pseudo_labeling import train_mlp

    X, y = _make_embeddings(60, dim=64, n_classes=3)
    clf = train_mlp(X, y, hidden_sizes=(32, 16), seed=42)
    preds = clf.predict(X)
    assert preds.shape == (60,)


def test_train_mlp_predicts_proba_correct_shape():
    from utils.image_pseudo_labeling import train_mlp

    X, y = _make_embeddings(60, dim=64, n_classes=3)
    clf = train_mlp(X, y, hidden_sizes=(32, 16), seed=42)
    proba = clf.predict_proba(X)
    assert proba.shape == (60, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_pseudo_label_loop_returns_history():
    from utils.image_pseudo_labeling import pseudo_label_loop

    X, y = _make_embeddings(100, dim=32, n_classes=3, seed=7)
    seed_mask = np.zeros(100, dtype=bool)
    seed_mask[:15] = True   # 5 per class
    seed_labels = y[seed_mask]

    pred_labels, history = pseudo_label_loop(
        embeddings=X,
        seed_mask=seed_mask,
        seed_labels=seed_labels,
        hidden_sizes=(16, 8),
        threshold=0.5,
        max_iters=3,
        seed=42,
        n_classes=3,
    )
    assert pred_labels.shape == (100,)
    assert isinstance(history, list)
    assert len(history) >= 1
    assert "iteration" in history[0]
    assert "new_pseudo_labels" in history[0]
    assert "cumulative_coverage" in history[0]
