# Image Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full image auto-labeling pipeline (unsupervised + semi-supervised) for 10 animal classes that mirrors the text pipeline's structure and produces comparable results.

**Architecture:** CLIP ViT-B/32 extracts 512-dim embeddings for all 5,500 images once (nb_img_01) and saves them to disk; all downstream notebooks load from that cache. Unsupervised clustering uses KMeans/Agglomerative/DEC (nb_img_02) followed by BLIP caption-based cluster description and CLIP zero-shot label matching (nb_img_03b). Semi-supervised methods (nb_img_04, nb_img_05, nb_img_05b, nb_img_08) all consume the same frozen embeddings via a 5% labeled seed split built in nb_img_00.

**Tech Stack:** `transformers` (CLIP, BLIP), `torch` (cpu), `Pillow`, `scikit-learn`, `snorkel`, `sumy` (LexRank), `numpy`, `pandas`

**Spec:** `docs/superpowers/specs/2026-09-24-image-semi-supervised-design.md`

## Global Constraints

- CPU-only machine (no CUDA/MPS); no full ViT fine-tuning
- CLIP model: `openai/clip-vit-base-patch32` (512-dim output)
- BLIP model: `Salesforce/blip-image-captioning-base`
- Seed: `random_state=42`, stratified 5% split (~27–28 images/class)
- k=10 for all clustering (known class count)
- All results saved to `results/` with prefix `image_`
- Embeddings cached at `data/image_embeddings.npy`
- Manifest at `data/image_manifest.csv`
- Notebooks numbered: `nb_img_00` through `nb_img_08`
- Follow existing `utils/` patterns (cache-on-first-run, float32 numpy arrays)

---

## File Map

**New utils (create):**
- `utils/image_config.py` — image paths, class names, CLIP model name, seed constants
- `utils/image_embeddings.py` — CLIP image + text embedding helpers (cache-aware)
- `utils/image_weak_supervision.py` — CLIP zero-shot labeling functions for Snorkel
- `utils/image_pseudo_labeling.py` — MLP classifier + pseudo-label loop on CLIP embeddings

**New tests (create):**
- `tests/test_image_embeddings.py` — shape/dtype checks for CLIP embeddings
- `tests/test_image_weak_supervision.py` — LF output values, Snorkel integration
- `tests/test_image_pseudo_labeling.py` — MLP training, pseudo-label loop coverage

**New notebooks (create):**
- `notebooks/nb_img_00.ipynb` — manifest + seed split
- `notebooks/nb_img_01.ipynb` — CLIP embedding extraction
- `notebooks/nb_img_02.ipynb` — KMeans / Agglomerative / DEC clustering
- `notebooks/nb_img_03b.ipynb` — BLIP captions + LexRank descriptions + label matching
- `notebooks/nb_img_04.ipynb` — weak supervision
- `notebooks/nb_img_05.ipynb` — pseudo-labeling config A
- `notebooks/nb_img_05b.ipynb` — pseudo-labeling config B
- `notebooks/nb_img_08.ipynb` — label propagation
- `notebooks/nb_img_07.ipynb` — cross-method comparison

**Existing utils reused (no changes):**
- `utils/dec.py` — `train_dec()` works on any float32 numpy embeddings
- `utils/label_propagation.py` — `run_label_propagation()` works on any embeddings
- `utils/cluster_description.py` — `generate_cluster_descriptions()` works on any text
- `utils/metrics.py` — shared metric computation

---

## Task 1: Image config + manifest + seed split (nb_img_00)

**Files:**
- Create: `utils/image_config.py`
- Create: `notebooks/nb_img_00.ipynb`

**Interfaces:**
- Produces:
  - `IMAGE_DATA_DIR`, `IMAGE_CLASS_NAMES`, `IMAGE_MANIFEST_PATH`, `IMAGE_EMBEDDINGS_PATH`, `CLIP_MODEL_NAME`, `BLIP_MODEL_NAME`, `IMAGE_SEED`, `IMAGE_LABEL_FRACTION`, `IMAGE_NUM_CLASSES` from `utils/image_config.py`
  - `data/image_manifest.csv` with columns: `filepath` (str, absolute path), `true_label` (str), `label_idx` (int 0–9), `is_seed` (bool)

- [ ] **Step 1: Create `utils/image_config.py`**

```python
# utils/image_config.py
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

IMAGE_DATA_DIR = DATA_DIR / "image-data"
IMAGE_MANIFEST_PATH = DATA_DIR / "image_manifest.csv"
IMAGE_EMBEDDINGS_PATH = DATA_DIR / "image_embeddings.npy"

# Alphabetical so label_idx is deterministic and independent of filesystem order
IMAGE_CLASS_NAMES = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel",
]
IMAGE_NUM_CLASSES = len(IMAGE_CLASS_NAMES)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

IMAGE_SEED = 42
IMAGE_LABEL_FRACTION = 0.05   # ~27–28 images/class as labeled seed
```

- [ ] **Step 2: Create `notebooks/nb_img_00.ipynb` — build manifest and seed split**

The notebook has three sections. Write each section as its own cell block.

**Section 1 — Build manifest CSV:**
```python
import pandas as pd
from pathlib import Path
from utils.image_config import IMAGE_DATA_DIR, IMAGE_CLASS_NAMES, IMAGE_MANIFEST_PATH

rows = []
for class_name in IMAGE_CLASS_NAMES:
    label_idx = IMAGE_CLASS_NAMES.index(class_name)
    class_dir = IMAGE_DATA_DIR / class_name
    for img_path in sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
        rows.append({
            "filepath": str(img_path),
            "true_label": class_name,
            "label_idx": label_idx,
            "is_seed": False,
        })

manifest = pd.DataFrame(rows)
print(f"Total images: {len(manifest)}")
print(manifest["true_label"].value_counts())
```

**Section 2 — Stratified 5% seed split:**
```python
from sklearn.model_selection import train_test_split
from utils.image_config import IMAGE_LABEL_FRACTION, IMAGE_SEED

seed_idx, _ = train_test_split(
    manifest.index,
    test_size=1 - IMAGE_LABEL_FRACTION,
    stratify=manifest["label_idx"],
    random_state=IMAGE_SEED,
)
manifest.loc[seed_idx, "is_seed"] = True
manifest.to_csv(IMAGE_MANIFEST_PATH, index=False)
print(f"Seed images: {manifest['is_seed'].sum()} ({manifest['is_seed'].mean():.1%})")
print(manifest.groupby("true_label")["is_seed"].sum())
```

**Section 3 — Verify:**
```python
loaded = pd.read_csv(IMAGE_MANIFEST_PATH)
assert len(loaded) == 5500, f"Expected 5500 rows, got {len(loaded)}"
assert set(loaded["true_label"].unique()) == set(IMAGE_CLASS_NAMES)
assert loaded["is_seed"].sum() >= 270  # ~5% of 5500
assert loaded.groupby("true_label")["is_seed"].sum().min() >= 25
print("Manifest OK")
```

- [ ] **Step 3: Run nb_img_00 top-to-bottom, verify `data/image_manifest.csv` created with ~5,500 rows and ~275 seed images.**

- [ ] **Step 4: Commit**

```bash
git add utils/image_config.py notebooks/nb_img_00.ipynb data/image_manifest.csv
git commit -m "feat(img): nb_img_00 — image manifest and stratified 5% seed split"
```

---

## Task 2: CLIP embedding extraction (nb_img_01)

**Files:**
- Create: `utils/image_embeddings.py`
- Create: `tests/test_image_embeddings.py`
- Create: `notebooks/nb_img_01.ipynb`

**Interfaces:**
- Consumes: `IMAGE_MANIFEST_PATH`, `IMAGE_EMBEDDINGS_PATH`, `CLIP_MODEL_NAME` from `utils/image_config.py`
- Produces:
  - `get_clip_image_embeddings(manifest_path, embeddings_path, model_name, batch_size) -> np.ndarray` shape `(N, 512)` float32
  - `get_clip_text_embeddings(texts, model_name) -> np.ndarray` shape `(len(texts), 512)` float32, L2-normalised
  - `data/image_embeddings.npy` — (5500, 512) float32, row order matches manifest CSV

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_image_embeddings.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` — `utils/image_embeddings.py` doesn't exist yet.

- [ ] **Step 3: Create `utils/image_embeddings.py`**

```python
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
        features = model.get_text_features(**inputs).cpu().numpy().astype(np.float32)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_image_embeddings.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Create `notebooks/nb_img_01.ipynb`**

The notebook has two sections.

**Section 1 — Extract and cache embeddings:**
```python
from utils.image_embeddings import get_clip_image_embeddings
from utils.image_config import IMAGE_MANIFEST_PATH, IMAGE_EMBEDDINGS_PATH

embeddings = get_clip_image_embeddings(
    manifest_path=IMAGE_MANIFEST_PATH,
    embeddings_path=IMAGE_EMBEDDINGS_PATH,
    batch_size=32,
)
print(f"Embeddings shape: {embeddings.shape}")   # (5500, 512)
print(f"dtype: {embeddings.dtype}")              # float32
```

**Section 2 — Sanity checks:**
```python
import numpy as np
import pandas as pd

manifest = pd.read_csv(IMAGE_MANIFEST_PATH)
assert embeddings.shape == (len(manifest), 512), "Row count mismatch"
assert embeddings.dtype == np.float32
norms = np.linalg.norm(embeddings, axis=1)
print(f"Embedding norm range: {norms.min():.2f} – {norms.max():.2f}")
print("nb_img_01 OK")
```

- [ ] **Step 6: Run nb_img_01 top-to-bottom. Confirm `data/image_embeddings.npy` saved, shape (5500, 512). Expected runtime: ~10–15 min on CPU.**

- [ ] **Step 7: Commit**

```bash
git add utils/image_embeddings.py tests/test_image_embeddings.py notebooks/nb_img_01.ipynb data/image_embeddings.npy
git commit -m "feat(img): nb_img_01 + utils — CLIP image embedding extraction and cache"
```

---

## Task 3: Unsupervised clustering (nb_img_02)

**Files:**
- Create: `notebooks/nb_img_02.ipynb`

**Interfaces:**
- Consumes: `data/image_embeddings.npy` (5500, 512), `data/image_manifest.csv`
- Consumes: `utils/dec.py` — `train_dec(embeddings, n_clusters, ...) -> np.ndarray`
- Produces:
  - `results/image_full_labels_unsup_{method}.csv` for method in `{kmeans, agglomerative, dec}` — columns: `filepath, true_label, predicted_label`
  - `results/image_metrics_unsup_{method}.json` — `{acc, f1, nmi, ari, silhouette}`
  - `results/image_cluster_assignments_{method}.npy` — integer array (5500,) for use by nb_img_03b

- [ ] **Step 1: Create `notebooks/nb_img_02.ipynb`**

**Section 1 — Load embeddings and manifest:**
```python
import numpy as np
import pandas as pd
from utils.image_config import IMAGE_MANIFEST_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_CLASS_NAMES, IMAGE_NUM_CLASSES, IMAGE_SEED, RESULTS_DIR

manifest = pd.read_csv(IMAGE_MANIFEST_PATH)
embeddings = np.load(IMAGE_EMBEDDINGS_PATH)
print(f"Loaded {len(manifest)} rows, embeddings {embeddings.shape}")
```

**Section 2 — UMAP reduction (shared for KMeans and Agglomerative):**
```python
import umap

reducer = umap.UMAP(n_components=30, metric="cosine", random_state=IMAGE_SEED)
embeddings_reduced = reducer.fit_transform(embeddings).astype(np.float32)
print(f"UMAP reduced: {embeddings_reduced.shape}")
```

**Section 3 — KMeans clustering:**
```python
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from sklearn.metrics import silhouette_score
import json

def hungarian_match(pred_labels, true_labels, n_classes):
    """Map cluster ids to class ids via Hungarian algorithm. Returns remapped pred_labels."""
    cost = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost[i, j] = -np.sum((pred_labels == i) & (true_labels == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {row_ind[k]: col_ind[k] for k in range(len(row_ind))}
    return np.array([mapping.get(p, 0) for p in pred_labels])

true_labels = manifest["label_idx"].to_numpy()

km = KMeans(n_clusters=IMAGE_NUM_CLASSES, n_init=10, random_state=IMAGE_SEED)
km_raw = km.fit_predict(embeddings_reduced)
km_labels = hungarian_match(km_raw, true_labels, IMAGE_NUM_CLASSES)

acc_km = (km_labels == true_labels).mean()
print(f"KMeans ACC: {acc_km:.4f}")
```

**Section 4 — Agglomerative clustering:**
```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=IMAGE_NUM_CLASSES, linkage="ward")
agg_raw = agg.fit_predict(embeddings_reduced)
agg_labels = hungarian_match(agg_raw, true_labels, IMAGE_NUM_CLASSES)

acc_agg = (agg_labels == true_labels).mean()
print(f"Agglomerative ACC: {acc_agg:.4f}")
```

**Section 5 — DEC clustering:**
```python
from utils.dec import train_dec

dec_raw = train_dec(embeddings, n_clusters=IMAGE_NUM_CLASSES, seed=IMAGE_SEED)
dec_labels = hungarian_match(dec_raw, true_labels, IMAGE_NUM_CLASSES)

acc_dec = (dec_labels == true_labels).mean()
print(f"DEC ACC: {acc_dec:.4f}")
```

**Section 6 — Save all results:**
```python
RESULTS_DIR.mkdir(exist_ok=True)
sil_sample = min(2000, len(embeddings_reduced))
sil_idx = np.random.default_rng(IMAGE_SEED).choice(len(embeddings_reduced), sil_sample, replace=False)

for method_name, pred_labels, emb_for_sil in [
    ("kmeans", km_labels, embeddings_reduced),
    ("agglomerative", agg_labels, embeddings_reduced),
    ("dec", dec_labels, embeddings),
]:
    # predictions CSV
    out_df = manifest[["filepath", "true_label"]].copy()
    out_df["predicted_label"] = [IMAGE_CLASS_NAMES[i] for i in pred_labels]
    out_df.to_csv(RESULTS_DIR / f"image_full_labels_unsup_{method_name}.csv", index=False)

    # cluster assignments .npy (raw cluster ids before Hungarian, needed by nb_img_03b)
    if method_name == "kmeans":
        np.save(RESULTS_DIR / f"image_cluster_assignments_kmeans.npy", km_raw)
    elif method_name == "agglomerative":
        np.save(RESULTS_DIR / f"image_cluster_assignments_agglomerative.npy", agg_raw)
    else:
        np.save(RESULTS_DIR / f"image_cluster_assignments_dec.npy", dec_raw)

    # metrics JSON
    sil = silhouette_score(emb_for_sil[sil_idx], pred_labels[sil_idx], metric="cosine")
    metrics = {
        "acc": float((pred_labels == true_labels).mean()),
        "f1": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "silhouette": float(sil),
    }
    with open(RESULTS_DIR / f"image_metrics_unsup_{method_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"{method_name}: ACC={metrics['acc']:.4f}  F1={metrics['f1']:.4f}  NMI={metrics['nmi']:.4f}")
```

- [ ] **Step 2: Run nb_img_02 top-to-bottom. Confirm 3 CSVs and 3 JSON files created in `results/`. KMeans and Agglomerative should finish in <2 min; DEC may take ~10–15 min on CPU.**

- [ ] **Step 3: Commit**

```bash
git add notebooks/nb_img_02.ipynb results/image_full_labels_unsup_*.csv results/image_metrics_unsup_*.json results/image_cluster_assignments_*.npy
git commit -m "feat(img): nb_img_02 — unsupervised clustering KMeans/Agglom/DEC on CLIP embeddings"
```

---

## Task 4: BLIP captions + cluster descriptions + label matching (nb_img_03b)

**Files:**
- Create: `notebooks/nb_img_03b.ipynb`

**Interfaces:**
- Consumes: `results/image_cluster_assignments_{method}.npy`, `data/image_manifest.csv`, `data/image_embeddings.npy`
- Consumes: `utils/cluster_description.py` — `generate_cluster_descriptions(texts_by_cluster) -> dict`
- Consumes: `utils/image_embeddings.py` — `get_clip_text_embeddings(texts) -> np.ndarray`
- Produces:
  - `results/image_cluster_descriptions_{method}.json` — `{cluster_id: description_text}`
  - `results/image_cluster_mapping_{method}.json` — `{cluster_id: predicted_class_name}`

- [ ] **Step 1: Create `notebooks/nb_img_03b.ipynb`**

**Section 1 — Caption central images with BLIP:**
```python
import numpy as np
import pandas as pd
import json
import torch
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.image_config import (
    IMAGE_MANIFEST_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_CLASS_NAMES,
    IMAGE_NUM_CLASSES, BLIP_MODEL_NAME, RESULTS_DIR
)

manifest = pd.read_csv(IMAGE_MANIFEST_PATH)
embeddings = np.load(IMAGE_EMBEDDINGS_PATH)

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
blip_model.eval()

def caption_images(filepaths: list[str], max_new_tokens: int = 30) -> list[str]:
    """Generate one caption per image path using BLIP."""
    captions = []
    for fp in filepaths:
        img = Image.open(fp).convert("RGB")
        inputs = blip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
        captions.append(blip_processor.decode(out[0], skip_special_tokens=True))
    return captions
```

**Section 2 — For each method: find central images, caption them, build descriptions:**
```python
from utils.cluster_description import generate_cluster_descriptions
from utils.image_embeddings import get_clip_text_embeddings

CENTRAL_IMAGES_PER_CLUSTER = 8   # caption only 8 × 10 = 80 images total

def get_central_image_indices(embeddings, cluster_assignments, n_clusters, n_central=8):
    """Return dict {cluster_id: [row_indices of n_central images closest to centroid]}."""
    central = {}
    for c in range(n_clusters):
        mask = cluster_assignments == c
        if mask.sum() == 0:
            central[c] = []
            continue
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        top_local = np.argsort(dists)[:n_central]
        global_indices = np.where(mask)[0][top_local]
        central[c] = global_indices.tolist()
    return central

for method in ["kmeans", "agglomerative", "dec"]:
    assignments = np.load(RESULTS_DIR / f"image_cluster_assignments_{method}.npy")

    # Caption central images per cluster
    central_idx = get_central_image_indices(embeddings, assignments, IMAGE_NUM_CLASSES, CENTRAL_IMAGES_PER_CLUSTER)
    captions_by_cluster = {}
    for c, indices in central_idx.items():
        fps = manifest.iloc[indices]["filepath"].tolist()
        captions_by_cluster[c] = caption_images(fps)
        print(f"  {method} cluster {c}: {captions_by_cluster[c][:2]}")

    # LexRank descriptions on captions
    descriptions = generate_cluster_descriptions(
        {str(c): caps for c, caps in captions_by_cluster.items()},
        n_sentences=2,
    )
    with open(RESULTS_DIR / f"image_cluster_descriptions_{method}.json", "w") as f:
        json.dump(descriptions, f, indent=2)

    # CLIP zero-shot label matching
    class_prompts = [f"a photo of a {cls}" for cls in IMAGE_CLASS_NAMES]
    text_embs = get_clip_text_embeddings(class_prompts)  # (10, 512) L2-normalised

    mapping = {}
    for c, desc in descriptions.items():
        desc_emb = get_clip_text_embeddings([desc])           # (1, 512)
        sims = (desc_emb @ text_embs.T).ravel()               # (10,)
        best_class = IMAGE_CLASS_NAMES[int(sims.argmax())]
        mapping[c] = best_class

    with open(RESULTS_DIR / f"image_cluster_mapping_{method}.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"{method} mapping: {mapping}")
```

- [ ] **Step 2: Run nb_img_03b top-to-bottom. Confirm 6 JSON files created (3 descriptions + 3 mappings). BLIP captioning ~80 images on CPU takes ~5–10 min.**

- [ ] **Step 3: Commit**

```bash
git add notebooks/nb_img_03b.ipynb results/image_cluster_descriptions_*.json results/image_cluster_mapping_*.json
git commit -m "feat(img): nb_img_03b — BLIP captions + LexRank descriptions + CLIP zero-shot label matching"
```

---

## Task 5: CLIP zero-shot weak supervision (nb_img_04)

**Files:**
- Create: `utils/image_weak_supervision.py`
- Create: `tests/test_image_weak_supervision.py`
- Create: `notebooks/nb_img_04.ipynb`

**Interfaces:**
- Consumes: `data/image_embeddings.npy`, `data/image_manifest.csv`
- Consumes: `utils/image_embeddings.py` — `get_clip_text_embeddings(texts) -> np.ndarray` (L2-normalised, (N, 512) float32)
- Produces:
  - `build_clip_lfs(image_embeddings, class_names, thresholds) -> list[LabelingFunction]` — one Snorkel LF per class; LF returns class index or -1 (abstain)
  - `tune_thresholds(seed_embeddings, seed_labels, class_names, text_embeddings) -> list[float]` — per-class cosine similarity threshold tuned on seed
  - `results/image_full_labels_weak_supervision.csv` — `filepath, true_label, predicted_label`
  - `results/image_metrics_weak_supervision.json` — `{label_acc, label_f1, coverage}`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_image_weak_supervision.py -v
```
Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Create `utils/image_weak_supervision.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_image_weak_supervision.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Create `notebooks/nb_img_04.ipynb`**

**Section 1 — Load data:**
```python
import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score
from utils.image_config import (
    IMAGE_MANIFEST_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_CLASS_NAMES,
    IMAGE_NUM_CLASSES, IMAGE_SEED, RESULTS_DIR
)
from utils.image_embeddings import get_clip_text_embeddings
from utils.image_weak_supervision import tune_thresholds, build_clip_lfs

manifest = pd.read_csv(IMAGE_MANIFEST_PATH)
embeddings = np.load(IMAGE_EMBEDDINGS_PATH)

seed_mask = manifest["is_seed"].to_numpy().astype(bool)
seed_embs = embeddings[seed_mask]
seed_labels = manifest.loc[seed_mask, "label_idx"].to_numpy()
unlabeled_embs = embeddings[~seed_mask]
```

**Section 2 — Tune thresholds on seed, build LFs:**
```python
class_prompts = [f"a photo of a {cls}" for cls in IMAGE_CLASS_NAMES]
text_embs = get_clip_text_embeddings(class_prompts)   # (10, 512) L2-normalised

thresholds = tune_thresholds(seed_embs, seed_labels, IMAGE_NUM_CLASSES, text_embs)
print("Per-class thresholds:", {IMAGE_CLASS_NAMES[i]: f"{t:.3f}" for i, t in enumerate(thresholds)})

lfs = build_clip_lfs(embeddings, IMAGE_CLASS_NAMES, text_embs, thresholds)
```

**Section 3 — Apply LFs to full dataset, run Snorkel LabelModel:**
```python
from snorkel.labeling import LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel

# Wrap callables as Snorkel LabelingFunctions
def _make_snorkel_lf(name, fn):
    return LabelingFunction(name=name, f=lambda x, _fn=fn: _fn(x.row_idx))

snorkel_lfs = [_make_snorkel_lf(f"lf_{IMAGE_CLASS_NAMES[i]}", lfs[i]) for i in range(IMAGE_NUM_CLASSES)]

# Apply to full dataset (add row_idx column for LF access)
df_for_lf = manifest.reset_index().rename(columns={"index": "row_idx"})
applier = PandasLFApplier(lfs=snorkel_lfs)
L = applier.apply(df_for_lf)
print(f"LF matrix shape: {L.shape}")  # (5500, 10)
coverage = (L != -1).any(axis=1).mean()
print(f"Coverage (at least one LF fires): {coverage:.1%}")
```

**Section 4 — Fit LabelModel, predict, save results:**
```python
label_model = LabelModel(cardinality=IMAGE_NUM_CLASSES, verbose=True)
label_model.fit(L_train=L, n_epochs=500, seed=IMAGE_SEED)

pred_labels = label_model.predict(L=L, tie_break_policy="abstain")
# -1 means all LFs abstained — treat as no prediction
covered_mask = pred_labels != -1
print(f"Final coverage: {covered_mask.mean():.1%}")

true_labels = manifest["label_idx"].to_numpy()
label_acc = (pred_labels[covered_mask] == true_labels[covered_mask]).mean()
label_f1 = f1_score(true_labels[covered_mask], pred_labels[covered_mask], average="macro", zero_division=0)
print(f"Label ACC (covered): {label_acc:.4f}  F1: {label_f1:.4f}")

out_df = manifest[["filepath", "true_label"]].copy()
out_df["predicted_label"] = [IMAGE_CLASS_NAMES[p] if p != -1 else "ABSTAIN" for p in pred_labels]
RESULTS_DIR.mkdir(exist_ok=True)
out_df.to_csv(RESULTS_DIR / "image_full_labels_weak_supervision.csv", index=False)

metrics = {"label_acc": float(label_acc), "label_f1": float(label_f1), "coverage": float(covered_mask.mean())}
with open(RESULTS_DIR / "image_metrics_weak_supervision.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(metrics)
```

- [ ] **Step 6: Run nb_img_04 top-to-bottom. Confirm `results/image_full_labels_weak_supervision.csv` and `results/image_metrics_weak_supervision.json` created.**

- [ ] **Step 7: Commit**

```bash
git add utils/image_weak_supervision.py tests/test_image_weak_supervision.py notebooks/nb_img_04.ipynb results/image_full_labels_weak_supervision.csv results/image_metrics_weak_supervision.json
git commit -m "feat(img): nb_img_04 + utils — CLIP zero-shot weak supervision via Snorkel"
```

---

## Task 6: Label propagation on CLIP embeddings (nb_img_08)

**Files:**
- Create: `notebooks/nb_img_08.ipynb`

**Interfaces:**
- Consumes: `data/image_embeddings.npy`, `data/image_manifest.csv`
- Consumes: `utils/label_propagation.py` — `run_label_propagation(labeled_embeddings, labeled_labels, unlabeled_embeddings, kernel, n_neighbors, ...) -> (predicted_labels, confidence)`
- Produces:
  - `results/image_full_labels_label_propagation.csv` — `filepath, true_label, predicted_label, confidence`
  - `results/image_metrics_label_propagation.json` — `{label_acc, label_f1, coverage}`

- [ ] **Step 1: Create `notebooks/nb_img_08.ipynb`**

**Section 1 — Load data:**
```python
import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score
from utils.image_config import IMAGE_MANIFEST_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_CLASS_NAMES, RESULTS_DIR
from utils.label_propagation import run_label_propagation

manifest = pd.read_csv(IMAGE_MANIFEST_PATH)
embeddings = np.load(IMAGE_EMBEDDINGS_PATH)

seed_mask = manifest["is_seed"].to_numpy().astype(bool)
seed_embs = embeddings[seed_mask]
seed_labels = manifest.loc[seed_mask, "label_idx"].to_numpy()
unlabeled_embs = embeddings[~seed_mask]

print(f"Seed: {seed_mask.sum()}  Unlabeled: {(~seed_mask).sum()}")
```

**Section 2 — Run label propagation:**
```python
pred_unlabeled, confidence_unlabeled = run_label_propagation(
    labeled_embeddings=seed_embs,
    labeled_labels=seed_labels,
    unlabeled_embeddings=unlabeled_embs,
    kernel="knn",
    n_neighbors=15,
)

# Reconstruct full-dataset predictions (seed rows get their true label)
all_pred = np.empty(len(manifest), dtype=int)
all_pred[seed_mask] = seed_labels
all_pred[~seed_mask] = pred_unlabeled

all_conf = np.empty(len(manifest), dtype=float)
all_conf[seed_mask] = 1.0
all_conf[~seed_mask] = confidence_unlabeled

true_labels = manifest["label_idx"].to_numpy()
label_acc = (all_pred == true_labels).mean()
label_f1 = f1_score(true_labels, all_pred, average="macro", zero_division=0)
print(f"Label ACC: {label_acc:.4f}  F1: {label_f1:.4f}  Coverage: 100%")
```

**Section 3 — Save results:**
```python
out_df = manifest[["filepath", "true_label"]].copy()
out_df["predicted_label"] = [IMAGE_CLASS_NAMES[p] for p in all_pred]
out_df["confidence"] = all_conf
RESULTS_DIR.mkdir(exist_ok=True)
out_df.to_csv(RESULTS_DIR / "image_full_labels_label_propagation.csv", index=False)

metrics = {"label_acc": float(label_acc), "label_f1": float(label_f1), "coverage": 1.0}
with open(RESULTS_DIR / "image_metrics_label_propagation.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(metrics)
```

- [ ] **Step 2: Run nb_img_08 top-to-bottom. Confirm output files created. LabelSpreading on 5,500 rows with k=15 may take ~2–5 min on CPU.**

- [ ] **Step 3: Commit**

```bash
git add notebooks/nb_img_08.ipynb results/image_full_labels_label_propagation.csv results/image_metrics_label_propagation.json
git commit -m "feat(img): nb_img_08 — label propagation on CLIP embeddings"
```

---

## Task 7: Pseudo-labeling MLP (nb_img_05 + nb_img_05b)

**Files:**
- Create: `utils/image_pseudo_labeling.py`
- Create: `tests/test_image_pseudo_labeling.py`
- Create: `notebooks/nb_img_05.ipynb`
- Create: `notebooks/nb_img_05b.ipynb`

**Interfaces:**
- Consumes: `data/image_embeddings.npy`, `data/image_manifest.csv`
- Produces:
  - `train_mlp(X_train, y_train, hidden_sizes, seed) -> trained sklearn Pipeline`
  - `pseudo_label_loop(embeddings, seed_mask, seed_labels, hidden_sizes, threshold, max_iters, seed) -> (all_pred_labels, history)`
    - `history`: list of dicts `{iteration, new_pseudo_labels, cumulative_coverage}`
  - `results/image_full_labels_pseudo_labeling_A.csv` — `filepath, true_label, predicted_label`
  - `results/image_metrics_pseudo_labeling_A.json` — `{label_acc, label_f1, coverage, history}`
  - `results/image_full_labels_pseudo_labeling_B.csv` — same schema
  - `results/image_metrics_pseudo_labeling_B.json` — same schema

- [ ] **Step 1: Write failing tests**

```python
# tests/test_image_pseudo_labeling.py
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_image_pseudo_labeling.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Create `utils/image_pseudo_labeling.py`**

```python
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
) -> tuple[np.ndarray, list[dict]]:
    """Self-training loop: train MLP on seed, absorb high-confidence predictions,
    retrain, repeat until no new pseudo-labels or max_iters.

    Returns:
        pred_labels: (N,) int array — final predicted label for every row
        history: list of {iteration, new_pseudo_labels, cumulative_coverage}
    """
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_image_pseudo_labeling.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Create `notebooks/nb_img_05.ipynb` (Config A: threshold=0.9, hidden=(256, 128))**

**Section 1 — Load data:**
```python
import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score
from utils.image_config import IMAGE_MANIFEST_PATH, IMAGE_EMBEDDINGS_PATH, IMAGE_CLASS_NAMES, IMAGE_SEED, RESULTS_DIR
from utils.image_pseudo_labeling import pseudo_label_loop

manifest = pd.read_csv(IMAGE_MANIFEST_PATH)
embeddings = np.load(IMAGE_EMBEDDINGS_PATH)

seed_mask = manifest["is_seed"].to_numpy().astype(bool)
seed_labels = manifest.loc[seed_mask, "label_idx"].to_numpy()
true_labels = manifest["label_idx"].to_numpy()

print(f"Seed: {seed_mask.sum()}  Unlabeled: {(~seed_mask).sum()}")
```

**Section 2 — Run pseudo-label loop (Config A):**
```python
pred_labels, history = pseudo_label_loop(
    embeddings=embeddings,
    seed_mask=seed_mask,
    seed_labels=seed_labels,
    hidden_sizes=(256, 128),
    threshold=0.9,
    max_iters=10,
    seed=IMAGE_SEED,
    n_classes=len(IMAGE_CLASS_NAMES),
)
print("History:", history)
final_coverage = (pred_labels != -1).mean()
label_acc = (pred_labels == true_labels).mean()
label_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
print(f"Config A — ACC: {label_acc:.4f}  F1: {label_f1:.4f}  Coverage: {final_coverage:.1%}")
```

**Section 3 — Save results:**
```python
out_df = manifest[["filepath", "true_label"]].copy()
out_df["predicted_label"] = [IMAGE_CLASS_NAMES[p] for p in pred_labels]
RESULTS_DIR.mkdir(exist_ok=True)
out_df.to_csv(RESULTS_DIR / "image_full_labels_pseudo_labeling_A.csv", index=False)

metrics = {
    "config": "A",
    "hidden_sizes": [256, 128],
    "threshold": 0.9,
    "label_acc": float(label_acc),
    "label_f1": float(label_f1),
    "coverage": float(final_coverage),
    "history": history,
}
with open(RESULTS_DIR / "image_metrics_pseudo_labeling_A.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(metrics)
```

- [ ] **Step 6: Create `notebooks/nb_img_05b.ipynb` (Config B: threshold=0.85, hidden=(256, 128, 64))**

This notebook is identical to nb_img_05 except for two parameters. Replace Section 2 with:

```python
pred_labels, history = pseudo_label_loop(
    embeddings=embeddings,
    seed_mask=seed_mask,
    seed_labels=seed_labels,
    hidden_sizes=(256, 128, 64),
    threshold=0.85,
    max_iters=10,
    seed=IMAGE_SEED,
    n_classes=len(IMAGE_CLASS_NAMES),
)
print("History:", history)
final_coverage = (pred_labels != -1).mean()
label_acc = (pred_labels == true_labels).mean()
label_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
print(f"Config B — ACC: {label_acc:.4f}  F1: {label_f1:.4f}  Coverage: {final_coverage:.1%}")
```

Section 3 saves to `image_full_labels_pseudo_labeling_B.csv` and `image_metrics_pseudo_labeling_B.json` with `"config": "B"`, `"hidden_sizes": [256, 128, 64]`, `"threshold": 0.85`.

- [ ] **Step 7: Run nb_img_05 and nb_img_05b top-to-bottom. Confirm 4 output files created in `results/`.**

- [ ] **Step 8: Commit**

```bash
git add utils/image_pseudo_labeling.py tests/test_image_pseudo_labeling.py notebooks/nb_img_05.ipynb notebooks/nb_img_05b.ipynb results/image_full_labels_pseudo_labeling_*.csv results/image_metrics_pseudo_labeling_*.json
git commit -m "feat(img): nb_img_05/05b + utils — MLP pseudo-labeling on CLIP embeddings (configs A and B)"
```

---

## Task 8: Cross-method comparison (nb_img_07)

**Files:**
- Create: `notebooks/nb_img_07.ipynb`

**Interfaces:**
- Consumes: all `results/image_metrics_*.json` files
- Produces: `results/image_comparison.csv` — `method, acc, f1_macro, nmi, coverage`

- [ ] **Step 1: Create `notebooks/nb_img_07.ipynb`**

**Section 1 — Load all metrics:**
```python
import json
import pandas as pd
from pathlib import Path
from utils.image_config import RESULTS_DIR

METHOD_FILES = {
    "unsup_kmeans":          "image_metrics_unsup_kmeans.json",
    "unsup_agglomerative":   "image_metrics_unsup_agglomerative.json",
    "unsup_dec":             "image_metrics_unsup_dec.json",
    "weak_supervision":      "image_metrics_weak_supervision.json",
    "label_propagation":     "image_metrics_label_propagation.json",
    "pseudo_labeling_A":     "image_metrics_pseudo_labeling_A.json",
    "pseudo_labeling_B":     "image_metrics_pseudo_labeling_B.json",
}

rows = []
for method, fname in METHOD_FILES.items():
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        print(f"MISSING: {fname} — skipping")
        continue
    with open(fpath) as f:
        m = json.load(f)
    rows.append({
        "method": method,
        "acc":      m.get("acc",        m.get("label_acc", float("nan"))),
        "f1_macro": m.get("f1",         m.get("label_f1",  float("nan"))),
        "nmi":      m.get("nmi",        float("nan")),
        "coverage": m.get("coverage",   1.0),
    })

comparison = pd.DataFrame(rows).sort_values("acc", ascending=False).reset_index(drop=True)
print(comparison.to_string(index=False))
```

**Section 2 — Save and display:**
```python
comparison.to_csv(RESULTS_DIR / "image_comparison.csv", index=False)
print(f"\nSaved to results/image_comparison.csv")
print(f"\n10-class chance baseline: 10.0%")
```

- [ ] **Step 2: Run nb_img_07 top-to-bottom. Confirm `results/image_comparison.csv` created with one row per method.**

- [ ] **Step 3: Update `PROJECT_SUMMARY_IMAGE.md` Section 5 with the actual results table from `image_comparison.csv`.**

- [ ] **Step 4: Commit**

```bash
git add notebooks/nb_img_07.ipynb results/image_comparison.csv PROJECT_SUMMARY_IMAGE.md
git commit -m "feat(img): nb_img_07 — cross-method comparison table + fill results in PROJECT_SUMMARY_IMAGE"
```

---

## Self-Review

**Spec coverage check:**
- ✅ nb_img_00 — manifest + stratified 5% seed split
- ✅ nb_img_01 — CLIP embedding extraction
- ✅ nb_img_02 — KMeans / Agglomerative / DEC (k=10)
- ✅ nb_img_03b — BLIP captions + LexRank descriptions + CLIP zero-shot label matching
- ✅ nb_img_04 — weak supervision (CLIP zero-shot LFs → Snorkel)
- ✅ nb_img_05 — pseudo-labeling config A (threshold=0.9, hidden=(256,128))
- ✅ nb_img_05b — pseudo-labeling config B (threshold=0.85, hidden=(256,128,64))
- ✅ nb_img_08 — label propagation (k=15, cosine)
- ✅ nb_img_07 — comparison table

**Type/name consistency:**
- `get_clip_image_embeddings` / `get_clip_text_embeddings` used consistently across Tasks 2, 4, 5
- `IMAGE_CLASS_NAMES`, `IMAGE_NUM_CLASSES`, `IMAGE_SEED`, `RESULTS_DIR` from `utils/image_config.py` used consistently
- `run_label_propagation` from existing `utils/label_propagation.py` — signature unchanged
- `pseudo_label_loop` returns `(np.ndarray, list[dict])` — used consistently in Tasks 7–8

**Placeholder scan:** No TBDs, no "implement later", all code blocks are complete.

**CPU constraint:** All notebooks note expected runtimes. No full ViT fine-tuning anywhere.
