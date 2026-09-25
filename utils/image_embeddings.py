# utils/image_embeddings.py
from pathlib import Path
import numpy as np
import pandas as pd
from utils.image_config import CLIP_MODEL_NAME, IMAGE_EMBEDDINGS_PATH, IMAGE_MANIFEST_PATH


def get_clip_image_embeddings(
    manifest_path: Path = IMAGE_MANIFEST_PATH,
    embeddings_path: Path = IMAGE_EMBEDDINGS_PATH,
    model_name: str = CLIP_MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract CLIP image embeddings for all rows in manifest. Cache to disk.

    Returns (N, 512) float32 array in manifest row order.
    """
    embeddings_path = Path(embeddings_path)
    if embeddings_path.exists():
        return np.load(embeddings_path)

    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    manifest = pd.read_csv(manifest_path)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    filepaths = manifest["filepath"].tolist()

    for i in range(0, len(filepaths), batch_size):
        batch_paths = filepaths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # transformers >=5.x may return an output object rather than a raw tensor
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output if hasattr(features, "pooler_output") else features[0]
        all_embeddings.append(features.cpu().numpy())
        if i % (batch_size * 10) == 0:
            print(f"  {i}/{len(filepaths)} images embedded")

    arr = np.vstack(all_embeddings).astype(np.float32)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, arr)
    print(f"Saved embeddings to {embeddings_path} — shape {arr.shape}")
    return arr


def get_clip_text_embeddings(texts: list[str], model_name: str = CLIP_MODEL_NAME) -> np.ndarray:
    """Encode texts with CLIP text encoder. Returns L2-normalised (N, 512) float32."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    # transformers >=5.x may return an output object rather than a raw tensor
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output if hasattr(features, "pooler_output") else features[0]
    features = features.cpu().numpy().astype(np.float32)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)
