"""Text inference — lazy-loaded models, called from app.py."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.config import CLASS_NAMES

MODELS_DIR = REPO_ROOT / "models"

_text_unsup_cache: Optional[dict] = None
_text_semisup_cache: Optional[dict] = None


def _load_text_unsup() -> dict:
    global _text_unsup_cache
    if _text_unsup_cache is not None:
        return _text_unsup_cache
    missing = [p for p in [
        MODELS_DIR / "text_umap.pkl",
        MODELS_DIR / "text_kmeans.pkl",
        MODELS_DIR / "text_cluster_names.json",
    ] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {[str(p) for p in missing]}. "
            "Run: uv run python scripts/export_models.py"
        )
    from sentence_transformers import SentenceTransformer
    with open(MODELS_DIR / "text_umap.pkl", "rb") as f:
        umap_reducer = pickle.load(f)
    with open(MODELS_DIR / "text_kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(MODELS_DIR / "text_cluster_names.json") as f:
        cluster_names = json.load(f)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    _text_unsup_cache = {
        "encoder": encoder,
        "umap": umap_reducer,
        "kmeans": kmeans,
        "cluster_names": cluster_names,
    }
    return _text_unsup_cache


def _load_text_semisup() -> dict:
    global _text_semisup_cache
    if _text_semisup_cache is not None:
        return _text_semisup_cache
    model_dir = MODELS_DIR / "text_distilbert"
    required = [model_dir / "config.json", model_dir / "tokenizer_config.json"]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete text_distilbert artifact (missing: {[p.name for p in missing]}). "
            "Run: uv run python scripts/export_models.py"
        )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    _text_semisup_cache = {"model": model, "tokenizer": tokenizer}
    return _text_semisup_cache


def predict_text_unsupervised(text: str) -> str:
    """Embed text with MiniLM → UMAP → KMeans → return cluster label name."""
    m = _load_text_unsup()
    emb = m["encoder"].encode([text], show_progress_bar=False)  # (1, 384)
    reduced = m["umap"].transform(emb)                          # (1, 50)
    cluster_id = int(m["kmeans"].predict(reduced)[0])
    return m["cluster_names"].get(str(cluster_id), CLASS_NAMES[cluster_id % len(CLASS_NAMES)])


def predict_text_semisupervised(text: str) -> str:
    """Tokenize text → DistilBERT → argmax → return class name."""
    import torch
    m = _load_text_semisup()
    inputs = m["tokenizer"](
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        logits = m["model"](**inputs).logits
    class_id = int(logits.argmax(dim=-1).item())
    return CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cluster-{class_id}"
