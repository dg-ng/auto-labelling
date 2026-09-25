# tests/test_image_embeddings.py
import numpy as np
import pytest
from pathlib import Path


def test_get_clip_image_embeddings_shape(tmp_path, monkeypatch):
    """CLIP image embeddings return (N, 512) float32."""
    import pandas as pd
    from PIL import Image

    # Create 3 tiny dummy images across 2 classes
    for cls in ["cat", "dog"]:
        (tmp_path / cls).mkdir()
    for i, cls in enumerate(["cat", "cat", "dog"]):
        img = Image.new("RGB", (64, 64), color=(i * 80, 0, 0))
        img.save(tmp_path / cls / f"{i}.jpeg")

    manifest = pd.DataFrame([
        {"filepath": str(tmp_path / "cat" / "0.jpeg"), "true_label": "cat", "label_idx": 0, "is_seed": True},
        {"filepath": str(tmp_path / "cat" / "1.jpeg"), "true_label": "cat", "label_idx": 0, "is_seed": False},
        {"filepath": str(tmp_path / "dog" / "2.jpeg"), "true_label": "dog", "label_idx": 1, "is_seed": True},
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    embeddings_path = tmp_path / "embeddings.npy"

    from utils.image_embeddings import get_clip_image_embeddings
    emb = get_clip_image_embeddings(manifest_path, embeddings_path, batch_size=2)

    assert emb.shape == (3, 512)
    assert emb.dtype == np.float32
    assert embeddings_path.exists()


def test_get_clip_text_embeddings_normalised():
    """CLIP text embeddings are L2-normalised (unit vectors)."""
    from utils.image_embeddings import get_clip_text_embeddings

    texts = ["a photo of a cat", "a photo of a dog"]
    emb = get_clip_text_embeddings(texts)

    assert emb.shape == (2, 512)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_get_clip_image_embeddings_cache(tmp_path):
    """Second call loads from cache without re-running CLIP."""
    import pandas as pd
    from PIL import Image

    (tmp_path / "cat").mkdir()
    img = Image.new("RGB", (64, 64))
    img.save(tmp_path / "cat" / "0.jpeg")
    manifest = pd.DataFrame([
        {"filepath": str(tmp_path / "cat" / "0.jpeg"), "true_label": "cat", "label_idx": 0, "is_seed": True}
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    embeddings_path = tmp_path / "embeddings.npy"

    from utils.image_embeddings import get_clip_image_embeddings
    emb1 = get_clip_image_embeddings(manifest_path, embeddings_path)
    # overwrite embeddings with sentinel to confirm cache path is used
    np.save(embeddings_path, np.zeros((1, 512), dtype=np.float32))
    emb2 = get_clip_image_embeddings(manifest_path, embeddings_path)
    np.testing.assert_array_equal(emb2, np.zeros((1, 512), dtype=np.float32))
