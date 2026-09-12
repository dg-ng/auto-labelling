import numpy as np
import pytest
from utils.dec import train_dec


def test_train_dec_output_shape_and_range():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((120, 32)).astype(np.float32)  # 120 rows, 32-dim
    labels = train_dec(
        emb,
        n_clusters=4,
        latent_dim=16,
        pretrain_epochs=3,
        dec_epochs=10,
        batch_size=32,
        seed=0,
    )
    assert labels.shape == (120,), f"expected (120,), got {labels.shape}"
    assert labels.dtype in (np.int32, np.int64), f"expected int dtype, got {labels.dtype}"
    assert set(labels).issubset(set(range(4))), f"labels out of range: {set(labels)}"


def test_train_dec_reproducible():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((80, 16)).astype(np.float32)
    a = train_dec(emb, n_clusters=3, latent_dim=8, pretrain_epochs=2, dec_epochs=5, seed=7)
    b = train_dec(emb, n_clusters=3, latent_dim=8, pretrain_epochs=2, dec_epochs=5, seed=7)
    np.testing.assert_array_equal(a, b, err_msg="DEC not reproducible with same seed")
