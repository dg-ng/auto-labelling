"""Image inference — lazy-loaded CLIP + DEC and CLIP + MLP, called from app.py."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR = REPO_ROOT / "models"

_clip_model_cache: Optional[dict] = None
_image_unsup_cache: Optional[dict] = None
_image_semisup_cache: Optional[dict] = None


def _clip_embed(image) -> np.ndarray:
    """Return (1, 512) CLIP embedding for a PIL Image."""
    global _clip_model_cache
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from utils.image_config import CLIP_MODEL_NAME

    if _clip_model_cache is None:
        _clip_model_cache = {
            "model": CLIPModel.from_pretrained(CLIP_MODEL_NAME),
            "processor": CLIPProcessor.from_pretrained(CLIP_MODEL_NAME),
        }
        _clip_model_cache["model"].eval()

    model = _clip_model_cache["model"]
    processor = _clip_model_cache["processor"]
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output if hasattr(features, "pooler_output") else features[0]
    return features.cpu().numpy().astype(np.float32)  # (1, 512)


def _load_image_unsup() -> dict:
    global _image_unsup_cache
    if _image_unsup_cache is not None:
        return _image_unsup_cache
    missing = [p for p in [
        MODELS_DIR / "image_dec_encoder.pt",
        MODELS_DIR / "image_dec_arch.json",
        MODELS_DIR / "image_dec_centers.npy",
        MODELS_DIR / "image_dec_cluster_names.json",
    ] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing image unsup artifacts: {[str(p) for p in missing]}. "
            "Run: uv run python scripts/export_models.py"
        )
    import torch
    from utils.dec import _build_encoder  # intentional: must match dec.py's architecture
    with open(MODELS_DIR / "image_dec_arch.json") as f:
        arch = json.load(f)
    encoder = _build_encoder(arch["input_dim"], arch["latent_dim"])
    encoder.load_state_dict(torch.load(
        MODELS_DIR / "image_dec_encoder.pt", map_location="cpu", weights_only=True
    ))
    encoder.eval()
    centers = np.load(MODELS_DIR / "image_dec_centers.npy")  # (10, latent_dim)
    with open(MODELS_DIR / "image_dec_cluster_names.json") as f:
        cluster_names = json.load(f)
    _image_unsup_cache = {"encoder": encoder, "centers": centers, "cluster_names": cluster_names}
    return _image_unsup_cache


def _load_image_semisup() -> dict:
    global _image_semisup_cache
    if _image_semisup_cache is not None:
        return _image_semisup_cache
    missing = [p for p in [
        MODELS_DIR / "image_mlp.pkl",
        MODELS_DIR / "image_mlp_label_names.json",
    ] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing image semisup artifacts: {[str(p) for p in missing]}. "
            "Run: uv run python scripts/export_models.py"
        )
    with open(MODELS_DIR / "image_mlp.pkl", "rb") as f:
        mlp = pickle.load(f)
    with open(MODELS_DIR / "image_mlp_label_names.json") as f:
        label_names = json.load(f)
    _image_semisup_cache = {"mlp": mlp, "label_names": label_names}
    return _image_semisup_cache


def predict_image_unsupervised(image) -> str:
    """CLIP embed → DEC encoder → nearest center → return cluster name."""
    import torch
    m = _load_image_unsup()
    clip_emb = _clip_embed(image)                          # (1, 512)
    z = m["encoder"](torch.tensor(clip_emb))               # (1, latent_dim)
    z_np = z.detach().cpu().numpy()                        # (1, latent_dim)
    dists = np.linalg.norm(z_np - m["centers"], axis=1)   # (10,)
    cluster_id = int(np.argmin(dists))
    return m["cluster_names"].get(str(cluster_id), str(cluster_id))


def predict_image_semisupervised(image) -> str:
    """CLIP embed → MLP pipeline → argmax → return label name."""
    m = _load_image_semisup()
    clip_emb = _clip_embed(image)      # (1, 512)
    pred = int(m["mlp"].predict(clip_emb)[0])
    return m["label_names"][pred] if pred < len(m["label_names"]) else f"cluster-{pred}"
