# Auto-Labeling Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared `utils/` Python package plus eight sequential Jupyter notebooks that transform AG News, generate embeddings, run unsupervised clustering + BERTopic, run weak supervision + pseudo-labeling, train a full-supervised baseline, and produce a final comparison table — all CPU-friendly via a sample-size toggle.

**Architecture:** Pure-Python logic (data loading, metrics, embedding caching, model training/inference) lives in a `utils/` package. Notebooks are thin orchestration layers that call `utils/` functions, cache expensive artifacts to disk, and assert sanity conditions inline. Every notebook is independently re-runnable because it checks caches before recomputing.

**Tech Stack:** pandas, numpy, scikit-learn, PyTorch (CPU wheel), Hugging Face `transformers`, `sentence-transformers`, `umap-learn`, `hdbscan`, `bertopic`, `snorkel`, `openai`, `python-dotenv`, Jupyter. Environment and dependency management via **uv** (`pyproject.toml` + `uv.lock`, all commands run through `uv run ...`).

## Global Constraints

- `SEED = 42` used everywhere randomness occurs (spec: "Use `random_state=42` everywhere for reproducibility").
- `SAMPLE_SIZE = 8000` is the dev-time cap on **train** rows only; set to `None` for the full 120,000-row final run. Test set (7,600 rows) is always used in full — it's small enough to embed/evaluate on CPU without sampling.
- `CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"` replaces `roberta-base` for fine-tuning tasks (05, 06) per the approved design — CPU-runtime decision. Frozen RoBERTa is still used as an embedding method in notebook 01.
- CSV loading uses `header=0` (the data files have a header row: `Class Index,Title,Description`) — **not** `header=None` as in the original `autolabel_project_spec.md`.
- `CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]`, `NUM_CLASSES = 4`, 0-indexed labels (`label = class_index - 1`).
- `LABEL_FRACTION = 0.05` for the semi-supervised labeled/unlabeled split.
- HDBSCAN/BERTopic outlier points use label `-1` and must be excluded from metric denominators but their exclusion rate ("coverage") must be reported.
- Every notebook must be re-runnable top-to-bottom without manual cleanup (idempotent via cache-file checks).
- `.env` (gitignored) holds `OPENAI_API_KEY`; OpenAI steps must degrade gracefully (skip with a printed message) when the key is absent.
- **No pytest / no `tests/` directory.** `utils/` functions are verified with one-off `uv run python` smoke checks (heredoc scripts run once, not kept as a suite) when first written; the notebooks' own inline assertions are the ongoing regression check.
- **`uv` is the only environment/dependency tool** — no `pip install`, no manually-activated venv. All Python/Jupyter commands are prefixed `uv run`.
- Source reference: `autolabel_project_spec.md` (approach/metric definitions) and `docs/superpowers/specs/2026-08-15-autolabel-notebooks-design.md` (this project's approved design).

---

## Task 1: Project Scaffolding + `utils/config.py`

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `utils/__init__.py`
- Create: `utils/config.py`
- Modify: `.gitignore` (append `.venv/`, `_tmp_trainer/`)

**Interfaces:**
- Produces: `utils.config.{CLASS_NAMES, NUM_CLASSES, SEED, LABEL_FRACTION, SAMPLE_SIZE, CLASSIFIER_MODEL_NAME, REPO_ROOT, DATA_DIR, PROCESSED_DIR, CACHE_DIR, RESULTS_DIR, OPENAI_API_KEY}` — every later task imports from here.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "autolabel-project"
version = "0.1.0"
description = "Auto-labeling for text: unsupervised and semi-supervised AG News classification"
requires-python = ">=3.10"
dependencies = [
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "sentence-transformers",
    "umap-learn",
    "hdbscan",
    "bertopic",
    "snorkel",
    "openai",
    "python-dotenv",
    "matplotlib",
    "seaborn",
    "tabulate",
    "pyarrow",
    "jupyter",
    "nbconvert",
]

[tool.uv]
package = false

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

- [ ] **Step 2: Create `.env.example`**

```
OPENAI_API_KEY=sk-your-key-here
```

- [ ] **Step 3: Create `utils/__init__.py`** (empty file)

- [ ] **Step 4: Append to `.gitignore`**

```
.venv/
_tmp_trainer/
```

- [ ] **Step 5: Sync the environment**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`, resolves and installs all dependencies (large download — torch, transformers, bertopic; several minutes).

- [ ] **Step 6: Write `utils/config.py`**

```python
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = REPO_ROOT / "embeddings_cache"
RESULTS_DIR = REPO_ROOT / "results"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = 4
SEED = 42
LABEL_FRACTION = 0.05
SAMPLE_SIZE = 8000  # train-set dev cap; set to None for the full-data final run
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

- [ ] **Step 7: Verify with a smoke check**

Run:
```bash
uv run python - <<'EOF'
from utils import config

assert config.CLASS_NAMES == ["World", "Sports", "Business", "Sci/Tech"]
assert config.NUM_CLASSES == 4
assert config.SEED == 42
assert config.LABEL_FRACTION == 0.05
assert config.DATA_DIR == config.REPO_ROOT / "data"
assert config.PROCESSED_DIR == config.DATA_DIR / "processed"
assert config.CACHE_DIR == config.REPO_ROOT / "embeddings_cache"
assert config.RESULTS_DIR == config.REPO_ROOT / "results"
print("OK: utils.config")
EOF
```
Expected: prints `OK: utils.config`, no traceback.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .env.example .gitignore utils/__init__.py utils/config.py
git commit -m "Add uv-managed project scaffolding and utils.config"
```

---

## Task 2: `utils/data.py`

**Files:**
- Create: `utils/data.py`

**Interfaces:**
- Produces: `load_raw(path) -> pd.DataFrame`, `build_text_column(df) -> pd.DataFrame`, `make_splits(train_df, label_fraction, seed) -> (labeled_df, unlabeled_df)`, `stratified_sample(df, sample_size, seed, label_col="label") -> pd.DataFrame` — used by every notebook task (6–13).

- [ ] **Step 1: Write `utils/data.py`**

```python
import pandas as pd


def load_raw(path) -> pd.DataFrame:
    """Load an AG News CSV. Files have a header row: Class Index,Title,Description."""
    return pd.read_csv(path, header=0, names=["label", "title", "description"])


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title+description into `text` and 0-index the label column."""
    df = df.copy()
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["label"] = df["label"] - 1
    return df


def make_splits(train_df: pd.DataFrame, label_fraction: float, seed: int):
    """Split into a small labeled pool and a large unlabeled pool.

    The unlabeled pool's real label is kept as `true_label` for evaluation
    only — `label` is set to -1 to simulate it being unavailable to any
    training algorithm.
    """
    labeled_df = train_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=label_fraction, random_state=seed))
    unlabeled_df = train_df.drop(labeled_df.index).copy()
    unlabeled_df["true_label"] = unlabeled_df["label"]
    unlabeled_df["label"] = -1
    return labeled_df.reset_index(drop=True), unlabeled_df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, sample_size, seed: int, label_col: str = "label") -> pd.DataFrame:
    """Return a stratified sample of `sample_size` rows, or the full df if
    sample_size is None or >= len(df)."""
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if label_col not in df.columns:
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    frac = sample_size / len(df)
    sampled = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=seed))
    return sampled.reset_index(drop=True)
```

- [ ] **Step 2: Verify with a smoke check**

Run:
```bash
uv run python - <<'EOF'
import pandas as pd
from utils.data import build_text_column, make_splits, stratified_sample

raw = pd.DataFrame({
    "label": [1, 1, 2, 2, 3, 3, 4, 4],
    "title": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
    "description": ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"],
})

df = build_text_column(raw)
assert list(df["text"]) == ["t1 d1", "t2 d2", "t3 d3", "t4 d4",
                             "t5 d5", "t6 d6", "t7 d7", "t8 d8"]
assert set(df["label"]) == {0, 1, 2, 3}

raw2 = raw.copy()
raw2.loc[0, "description"] = None
df2 = build_text_column(raw2)
assert df2.loc[0, "text"] == "t1 "

labeled, unlabeled = make_splits(df, label_fraction=0.5, seed=42)
assert len(labeled) + len(unlabeled) == len(df)
assert (unlabeled["label"] == -1).all()
assert "true_label" in unlabeled.columns
assert labeled["label"].value_counts().to_dict() == {0: 1, 1: 1, 2: 1, 3: 1}

full = stratified_sample(df, sample_size=None, seed=42)
assert len(full) == len(df)

sample_a = stratified_sample(df, sample_size=4, seed=42)
sample_b = stratified_sample(df, sample_size=4, seed=42)
assert len(sample_a) == 4
assert sorted(sample_a["text"]) == sorted(sample_b["text"])

oversized = stratified_sample(df, sample_size=1000, seed=42)
assert len(oversized) == len(df)

print("OK: utils.data")
EOF
```
Expected: prints `OK: utils.data`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add utils/data.py
git commit -m "Add utils.data: loading, text column, splits, sampling"
```

---

## Task 3: `utils/metrics.py`

**Files:**
- Create: `utils/metrics.py`

**Interfaces:**
- Produces: `clustering_accuracy(true_labels, cluster_labels) -> float`, `evaluate_unsupervised(true_labels, cluster_labels, embeddings) -> dict`, `majority_vote_mapping(cluster_labels, true_labels, n_clusters) -> dict[int, int]`, `evaluate_semisupervised(true_labels, predicted_labels, class_names, save_path=None) -> (dict, str, np.ndarray)`, `evaluate_label_quality(true_labels, pseudo_labels, confidence_scores=None) -> dict` — used by notebook tasks 8, 9, 10, 11, 12, 13.

- [ ] **Step 1: Write `utils/metrics.py`**

```python
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)


def clustering_accuracy(true_labels, cluster_labels) -> float:
    """Hungarian-matched best cluster-to-class accuracy. Excludes noise (-1)."""
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    n_classes = len(np.unique(true_labels))
    n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
    size = max(n_classes, n_clusters)

    cost_matrix = np.zeros((size, size))
    for c in range(n_clusters):
        for k in range(n_classes):
            cost_matrix[c, k] = np.sum((cluster_labels == c) & (true_labels == k))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    correct = cost_matrix[row_ind, col_ind].sum()
    total = (cluster_labels >= 0).sum()
    return correct / total


def evaluate_unsupervised(true_labels, cluster_labels, embeddings) -> dict:
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    mask = cluster_labels >= 0

    return {
        "ACC (Hungarian)": clustering_accuracy(true_labels[mask], cluster_labels[mask]),
        "NMI": normalized_mutual_info_score(true_labels[mask], cluster_labels[mask]),
        "ARI": adjusted_rand_score(true_labels[mask], cluster_labels[mask]),
        "FMI": fowlkes_mallows_score(true_labels[mask], cluster_labels[mask]),
        "Homogeneity": homogeneity_score(true_labels[mask], cluster_labels[mask]),
        "Completeness": completeness_score(true_labels[mask], cluster_labels[mask]),
        "V-Measure": v_measure_score(true_labels[mask], cluster_labels[mask]),
        "Silhouette Score": silhouette_score(embeddings[mask], cluster_labels[mask], metric="cosine"),
        "Davies-Bouldin": davies_bouldin_score(embeddings[mask], cluster_labels[mask]),
        "Coverage": mask.sum() / len(cluster_labels),
    }


def majority_vote_mapping(cluster_labels, true_labels, n_clusters) -> dict:
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)
    mapping = {}
    for c in range(n_clusters):
        cluster_mask = cluster_labels == c
        if cluster_mask.sum() == 0:
            continue
        mapping[c] = int(mode(true_labels[cluster_mask], keepdims=True).mode[0])
    return mapping


def evaluate_semisupervised(true_labels, predicted_labels, class_names, save_path=None):
    results = {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "Macro F1": f1_score(true_labels, predicted_labels, average="macro"),
        "Weighted F1": f1_score(true_labels, predicted_labels, average="weighted"),
        "Macro Precision": precision_score(true_labels, predicted_labels, average="macro"),
        "Macro Recall": recall_score(true_labels, predicted_labels, average="macro"),
        "Cohen's Kappa": cohen_kappa_score(true_labels, predicted_labels),
    }

    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    cm = confusion_matrix(true_labels, predicted_labels)

    if save_path is not None:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    return results, report, cm


def evaluate_label_quality(true_labels, pseudo_labels, confidence_scores=None) -> dict:
    true_labels = np.array(true_labels)
    pseudo_labels = np.array(pseudo_labels)
    mask = pseudo_labels >= 0

    results = {
        "Label Accuracy": accuracy_score(true_labels[mask], pseudo_labels[mask]),
        "Label Macro F1": f1_score(true_labels[mask], pseudo_labels[mask], average="macro"),
        "Coverage": mask.sum() / len(pseudo_labels),
    }

    if confidence_scores is not None:
        confidence_scores = np.array(confidence_scores)
        results["Mean Confidence"] = float(confidence_scores[mask].mean())
        results["Median Confidence"] = float(np.median(confidence_scores[mask]))

    return results
```

- [ ] **Step 2: Verify with a smoke check**

Run:
```bash
uv run python - <<'EOF'
import numpy as np
from utils.metrics import (
    clustering_accuracy, evaluate_unsupervised, majority_vote_mapping,
    evaluate_semisupervised, evaluate_label_quality,
)

true_labels = np.array([0, 0, 1, 1])
cluster_labels = np.array([1, 1, 0, 0])
assert clustering_accuracy(true_labels, cluster_labels) == 1.0

true2 = np.array([0, 0, 1, 1, 0])
cluster2 = np.array([0, 0, 1, 1, -1])
assert clustering_accuracy(true2, cluster2) == 1.0

true3 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
cluster3 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
emb3 = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1],
                  [5, 5], [5.1, 5], [5, 5.1], [5.1, 5.1]])
results = evaluate_unsupervised(true3, cluster3, emb3)
for key in ["ACC (Hungarian)", "NMI", "ARI", "FMI", "Homogeneity",
            "Completeness", "V-Measure", "Silhouette Score",
            "Davies-Bouldin", "Coverage"]:
    assert key in results
assert results["Coverage"] == 1.0
assert results["ACC (Hungarian)"] == 1.0

mapping = majority_vote_mapping(
    cluster_labels=np.array([0, 0, 0, 1, 1]),
    true_labels=np.array([2, 2, 3, 1, 1]),
    n_clusters=2)
assert mapping[0] == 2 and mapping[1] == 1

sr, report, cm = evaluate_semisupervised(
    [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 2],
    ["World", "Sports", "Business", "Sci/Tech"])
assert sr["Accuracy"] == 7 / 8
assert cm.shape == (4, 4)

lq = evaluate_label_quality(
    true_labels=np.array([0, 1, 2, 3, 0]),
    pseudo_labels=np.array([0, 1, -1, 3, -1]),
    confidence_scores=np.array([0.9, 0.8, 0.0, 0.95, 0.0]))
assert lq["Coverage"] == 3 / 5
assert lq["Label Accuracy"] == 1.0
assert abs(lq["Mean Confidence"] - (0.9 + 0.8 + 0.95) / 3) < 1e-9

print("OK: utils.metrics")
EOF
```
Expected: prints `OK: utils.metrics`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add utils/metrics.py
git commit -m "Add utils.metrics: unsupervised, semi-supervised, label-quality evaluation"
```

---

## Task 4: `utils/embeddings.py`

**Files:**
- Create: `utils/embeddings.py`

**Interfaces:**
- Consumes: `utils.config.CACHE_DIR`
- Produces: `load_cached(name) -> np.ndarray | None`, `save_cache(name, arr) -> None`, `get_tfidf_embeddings(texts, cache_name, max_features=5000) -> np.ndarray`, `get_sentence_embeddings(texts, cache_name, model_name="all-MiniLM-L6-v2", batch_size=64) -> np.ndarray`, `get_bert_embeddings(texts, cache_name, model_name="roberta-base", batch_size=64, max_length=128) -> np.ndarray`, `get_openai_embeddings(texts, cache_name, model="text-embedding-3-small", batch_size=100) -> np.ndarray` — used by notebook tasks 7, 8.

- [ ] **Step 1: Write `utils/embeddings.py`**

```python
from pathlib import Path

import numpy as np

from utils.config import CACHE_DIR


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npy"


def load_cached(name: str):
    path = _cache_path(name)
    if path.exists():
        return np.load(path)
    return None


def save_cache(name: str, arr: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_cache_path(name), arr)


def get_tfidf_embeddings(texts, cache_name: str, max_features: int = 5000) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features)
    arr = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    save_cache(cache_name, arr)
    return arr


def get_sentence_embeddings(texts, cache_name: str, model_name: str = "all-MiniLM-L6-v2",
                             batch_size: int = 64) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    arr = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)
    save_cache(cache_name, arr)
    return arr


def get_bert_embeddings(texts, cache_name: str, model_name: str = "roberta-base",
                         batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    arr = np.vstack(all_embeddings).astype(np.float32)
    save_cache(cache_name, arr)
    return arr


def get_openai_embeddings(texts, cache_name: str, model: str = "text-embedding-3-small",
                           batch_size: int = 100) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    from openai import OpenAI

    client = OpenAI()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([r.embedding for r in response.data])

    arr = np.array(all_embeddings, dtype=np.float32)
    save_cache(cache_name, arr)
    return arr
```

- [ ] **Step 2: Verify with a smoke check**

This checks the cache round-trip and TF-IDF (fast, no network). MiniLM/RoBERTa/OpenAI embedding functions are exercised for real in notebook 01 (Task 7) — downloading models just to smoke-test the wrapper isn't worth the time here.

Run:
```bash
uv run python - <<'EOF'
import numpy as np
from utils import embeddings

arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
assert embeddings.load_cached("__smoke_test_missing__") is None
embeddings.save_cache("__smoke_test_arr__", arr)
loaded = embeddings.load_cached("__smoke_test_arr__")
assert np.array_equal(loaded, arr)
embeddings._cache_path("__smoke_test_arr__").unlink()

texts = ["the stock market rose today", "the team won the game",
         "scientists launched a new satellite", "the election results are in"]
tfidf_arr = embeddings.get_tfidf_embeddings(texts, cache_name="__smoke_test_tfidf__")
assert tfidf_arr.shape[0] == len(texts)
assert not np.isnan(tfidf_arr).any()
embeddings._cache_path("__smoke_test_tfidf__").unlink()

print("OK: utils.embeddings (cache + tfidf; minilm/bert/openai exercised in notebook 01)")
EOF
```
Expected: prints `OK: utils.embeddings ...`, no traceback, no leftover `__smoke_test_*.npy` files in `embeddings_cache/`.

- [ ] **Step 3: Commit**

```bash
git add utils/embeddings.py
git commit -m "Add utils.embeddings: TF-IDF/MiniLM/BERT/OpenAI with disk caching"
```

---

## Task 5: `utils/modeling.py`

**Files:**
- Create: `utils/modeling.py`

**Interfaces:**
- Consumes: `utils.config.{CLASSIFIER_MODEL_NAME, NUM_CLASSES, SEED}`
- Produces: `TextClassificationDataset`, `train_model(labeled_df, model_name=CLASSIFIER_MODEL_NAME, epochs=3, output_dir="./_tmp_trainer", max_length=128, batch_size=16) -> (model, tokenizer)`, `get_predictions(model, tokenizer, texts, batch_size=64, max_length=128) -> np.ndarray`, `pseudo_label_loop(labeled_df, unlabeled_df, model_name=CLASSIFIER_MODEL_NAME, n_iterations=3, confidence_threshold=0.90, epochs=3) -> (model, tokenizer, current_labeled_df, history: list[dict])` — used by notebook tasks 11, 12.

The smoke check below downloads a tiny (<1MB) test-only model from the Hugging Face Hub on first run (network required) — this is far cheaper than testing against the real `distilbert-base-uncased`, while still exercising the actual `Trainer`/tokenizer wiring.

- [ ] **Step 1: Write `utils/modeling.py`**

```python
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.config import CLASSIFIER_MODEL_NAME, NUM_CLASSES, SEED


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def train_model(labeled_df, model_name=CLASSIFIER_MODEL_NAME, epochs=3,
                 output_dir="./_tmp_trainer", max_length=128, batch_size=16):
    """Fine-tune a sequence classifier on a labeled DataFrame with text/label columns."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_CLASSES)

    encodings = tokenizer(
        labeled_df["text"].tolist(), padding=True, truncation=True,
        max_length=max_length, return_tensors="pt")
    dataset = TextClassificationDataset(encodings, labeled_df["label"].to_numpy())

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        seed=SEED,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    return model, tokenizer


def get_predictions(model, tokenizer, texts, batch_size=64, max_length=128) -> np.ndarray:
    """Return softmax class probabilities, shape (len(texts), NUM_CLASSES)."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        all_probs.append(softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


def pseudo_label_loop(labeled_df, unlabeled_df, model_name=CLASSIFIER_MODEL_NAME,
                       n_iterations=3, confidence_threshold=0.90, epochs=3):
    """Iteratively: train on labeled set, predict on remaining unlabeled set,
    absorb high-confidence predictions, repeat. Returns the final model,
    tokenizer, the grown labeled DataFrame, and a per-iteration history log."""
    current_labeled = labeled_df.copy()
    remaining_unlabeled = unlabeled_df.copy()
    history = []

    for iteration in range(n_iterations):
        if len(remaining_unlabeled) == 0:
            break

        model, tokenizer = train_model(current_labeled, model_name=model_name, epochs=epochs)

        probs = get_predictions(model, tokenizer, remaining_unlabeled["text"].tolist())
        confidence = probs.max(axis=1)
        predicted_labels = probs.argmax(axis=1)

        high_conf_mask = confidence >= confidence_threshold
        newly_labeled = remaining_unlabeled[high_conf_mask].copy()
        newly_labeled["label"] = predicted_labels[high_conf_mask]

        history.append({
            "iteration": iteration,
            "new_labels": int(high_conf_mask.sum()),
            "labeled_size": len(current_labeled),
            "unlabeled_size": len(remaining_unlabeled),
        })

        drop_cols = [c for c in ["true_label"] if c in newly_labeled.columns]
        current_labeled = pd.concat(
            [current_labeled, newly_labeled.drop(columns=drop_cols)],
            ignore_index=True)
        remaining_unlabeled = remaining_unlabeled[~high_conf_mask]

    final_model, final_tokenizer = train_model(current_labeled, model_name=model_name, epochs=epochs)
    return final_model, final_tokenizer, current_labeled, history
```

- [ ] **Step 2: Verify with a smoke check**

Run:
```bash
uv run python - <<'EOF'
import pandas as pd
from utils.modeling import train_model, get_predictions, pseudo_label_loop

TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"

def toy_df(n_per_class):
    texts_per_class = {
        0: ["world news happened today", "government election update"],
        1: ["team won the game", "player scored a goal"],
        2: ["stock market rises", "company reports profit"],
        3: ["new software released", "scientists discover space"],
    }
    rows = [{"text": t, "label": label}
            for label, texts in texts_per_class.items()
            for t in texts[:n_per_class]]
    return pd.DataFrame(rows)

df = toy_df(2)
model, tokenizer = train_model(df, model_name=TINY_MODEL, epochs=1, batch_size=4)
probs = get_predictions(model, tokenizer, df["text"].tolist(), batch_size=4)
assert probs.shape == (len(df), 4)
for row_sum in probs.sum(axis=1):
    assert abs(row_sum - 1.0) < 1e-3

labeled = toy_df(1)
unlabeled = toy_df(2)
unlabeled["label"] = -1
_, _, final_labeled, history = pseudo_label_loop(
    labeled, unlabeled, model_name=TINY_MODEL, n_iterations=2,
    confidence_threshold=0.0, epochs=1)
assert len(history) >= 1
assert len(final_labeled) == len(labeled) + len(unlabeled)

print("OK: utils.modeling")
EOF
```
Expected: prints `OK: utils.modeling`, no traceback.

- [ ] **Step 3: Clean up trainer scratch directory and commit**

Run: `rm -rf _tmp_trainer`
```bash
git add utils/modeling.py
git commit -m "Add utils.modeling: fine-tuning, inference, pseudo-label self-training loop"
```

---

## Task 6: Notebook `00_data_transform.ipynb`

**Files:**
- Create: `notebooks/00_data_transform.ipynb`

**Interfaces:**
- Consumes: `utils.data.{load_raw, build_text_column, make_splits}`, `utils.config`
- Produces: `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet`, `data/processed/labeled.parquet`, `data/processed/unlabeled.parquet` — consumed by notebook tasks 7–13.

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 00 — Data Transform

Loads raw AG News CSVs, builds the `text` column, 0-indexes labels, and
creates the semi-supervised labeled/unlabeled splits. Run this first —
every other notebook reads from `data/processed/`.
```

Code cell (path setup):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import config
from utils.data import load_raw, build_text_column, make_splits
```

Code cell (load + transform):
```python
train_raw = load_raw(config.DATA_DIR / "train.csv")
test_raw = load_raw(config.DATA_DIR / "test.csv")

train_clean = build_text_column(train_raw)
test_clean = build_text_column(test_raw)

train_clean.head()
```

Code cell (sanity assertions):
```python
assert len(train_clean) == 120_000, f"expected 120000 train rows, got {len(train_clean)}"
assert len(test_clean) == 7_600, f"expected 7600 test rows, got {len(test_clean)}"
assert train_clean["text"].isna().sum() == 0
assert test_clean["text"].isna().sum() == 0

train_counts = train_clean["label"].value_counts().sort_index()
test_counts = test_clean["label"].value_counts().sort_index()
assert (train_counts == 30_000).all(), train_counts
assert (test_counts == 1_900).all(), test_counts

print("Train class distribution:\n", train_counts)
print("\nTest class distribution:\n", test_counts)
print("\nSample rows:\n", train_clean.sample(3, random_state=config.SEED))
```

Code cell (semi-supervised splits):
```python
labeled_df, unlabeled_df = make_splits(
    train_clean, label_fraction=config.LABEL_FRACTION, seed=config.SEED)

assert len(labeled_df) + len(unlabeled_df) == len(train_clean)
assert (unlabeled_df["label"] == -1).all()
print(f"Labeled pool: {len(labeled_df)} rows ({config.LABEL_FRACTION:.0%})")
print(f"Unlabeled pool: {len(unlabeled_df)} rows (true_label hidden for eval only)")
```

Code cell (save):
```python
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_clean.to_parquet(config.PROCESSED_DIR / "train_clean.parquet", index=False)
test_clean.to_parquet(config.PROCESSED_DIR / "test_clean.parquet", index=False)
labeled_df.to_parquet(config.PROCESSED_DIR / "labeled.parquet", index=False)
unlabeled_df.to_parquet(config.PROCESSED_DIR / "unlabeled.parquet", index=False)
print("Saved processed splits to", config.PROCESSED_DIR)
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_transform.ipynb`
Expected: exits 0, no cell raises an exception (assertion cells would raise `AssertionError` and fail the command).

- [ ] **Step 3: Verify outputs exist**

Run: `ls data/processed/`
Expected: `train_clean.parquet`, `test_clean.parquet`, `labeled.parquet`, `unlabeled.parquet` all present.

- [ ] **Step 4: Commit**

```bash
git add notebooks/00_data_transform.ipynb
git commit -m "Add 00_data_transform notebook"
```

---

## Task 7: Notebook `01_embeddings.ipynb`

**Files:**
- Create: `notebooks/01_embeddings.ipynb`

**Interfaces:**
- Consumes: `data/processed/{train_clean,test_clean}.parquet` (Task 6), `utils.data.stratified_sample`, `utils.embeddings.{get_tfidf_embeddings, get_sentence_embeddings, get_bert_embeddings, get_openai_embeddings}`, `utils.config.ROBERTA_SAMPLE_SIZE` (new constant, added by this task)
- Produces: cached `.npy` files in `embeddings_cache/` named `{method}_{split}_{suffix}.npy` where `method ∈ {tfidf, minilm, roberta, openai}`, `split ∈ {train, test}`. For `tfidf`/`minilm`/`openai`: `suffix` is `n{SAMPLE_SIZE}` for train or `full` for test. For `roberta`: `suffix` is `n{ROBERTA_SAMPLE_SIZE}` for **both** train and test (see note below) — consumed by notebook tasks 8, 9 (task 9/BERTopic doesn't use cached RoBERTa embeddings, only task 8 does).

**Runtime note — RoBERTa is embedded on a separate, much smaller sample than the other methods.** Measured on this machine, frozen RoBERTa CPU inference runs at roughly 0.5-1s/text once warmed up (not the ~99s/20-texts a cold first batch suggested) — but embedding the full `SAMPLE_SIZE=8000` train rows + 7,600 test rows would still take on the order of 2+ hours, far past what's practical for an interactive dev pass. TF-IDF, MiniLM, and OpenAI are all fast (TF-IDF/MiniLM are local and cheap; OpenAI is a fast hosted API call) and stay at the full `config.SAMPLE_SIZE`/full test set. RoBERTa alone uses a new `config.ROBERTA_SAMPLE_SIZE = 500` cap on **both** its train and test slices, so this notebook's RoBERTa cell completes in single-digit minutes. `ROBERTA_SAMPLE_SIZE` is added to `utils/config.py` as a new constant in this task's Step 1 (Task 1's config.py already exists and is being extended, not replaced — add the one line, don't touch anything else in that file). For the eventual full-scale final run (see "Final Run" section at the end of this plan), raise `ROBERTA_SAMPLE_SIZE` to `None` (full data) same as `SAMPLE_SIZE`, and budget hours, not minutes, run unattended.

- [ ] **Step 1: Add `ROBERTA_SAMPLE_SIZE` to `utils/config.py`**

Add this line to `utils/config.py` (from Task 1), directly after the existing `SAMPLE_SIZE = 8000` line — don't change anything else in the file:
```python
ROBERTA_SAMPLE_SIZE = 500  # separate, smaller cap for CPU-bound frozen RoBERTa embedding; None = full data
```

- [ ] **Step 2: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 01 — Embeddings

Generates and caches TF-IDF, MiniLM, RoBERTa (frozen), and OpenAI embeddings.
TF-IDF/MiniLM/OpenAI run on the (sampled) train set and full test set.
RoBERTa runs on its own smaller `ROBERTA_SAMPLE_SIZE` cap (both train and
test) since CPU inference for a transformer is far slower than the other
methods. Downstream notebooks load from `embeddings_cache/` rather than
recomputing.
```

Code cell (path setup + imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

from utils import config
from utils.data import stratified_sample
from utils.embeddings import (
    get_tfidf_embeddings,
    get_sentence_embeddings,
    get_bert_embeddings,
    get_openai_embeddings,
)
```

Code cell (load + sample):
```python
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")

train_sample = stratified_sample(train_clean, config.SAMPLE_SIZE, seed=config.SEED)
suffix = f"n{config.SAMPLE_SIZE}" if config.SAMPLE_SIZE else "full"

roberta_train_sample = stratified_sample(train_clean, config.ROBERTA_SAMPLE_SIZE, seed=config.SEED)
roberta_test_sample = stratified_sample(test_clean, config.ROBERTA_SAMPLE_SIZE, seed=config.SEED)
roberta_suffix = f"n{config.ROBERTA_SAMPLE_SIZE}" if config.ROBERTA_SAMPLE_SIZE else "full"

print(f"Embedding {len(train_sample)} train rows ({suffix}) and {len(test_clean)} test rows "
      f"for tfidf/minilm/openai")
print(f"Embedding {len(roberta_train_sample)} train rows and {len(roberta_test_sample)} test rows "
      f"({roberta_suffix}) for roberta")
```

Code cell (TF-IDF):
```python
train_tfidf = get_tfidf_embeddings(train_sample["text"].tolist(), cache_name=f"tfidf_train_{suffix}")
test_tfidf = get_tfidf_embeddings(test_clean["text"].tolist(), cache_name="tfidf_test_full")
assert train_tfidf.shape[0] == len(train_sample)
assert test_tfidf.shape[0] == len(test_clean)
print("TF-IDF dims:", train_tfidf.shape[1])
```

Code cell (MiniLM):
```python
train_minilm = get_sentence_embeddings(train_sample["text"].tolist(), cache_name=f"minilm_train_{suffix}")
test_minilm = get_sentence_embeddings(test_clean["text"].tolist(), cache_name="minilm_test_full")
assert train_minilm.shape[0] == len(train_sample)
print("MiniLM dims:", train_minilm.shape[1])
```

Code cell (RoBERTa, frozen — separate smaller sample, see runtime note above):
```python
train_roberta = get_bert_embeddings(roberta_train_sample["text"].tolist(), cache_name=f"roberta_train_{roberta_suffix}")
test_roberta = get_bert_embeddings(roberta_test_sample["text"].tolist(), cache_name=f"roberta_test_{roberta_suffix}")
assert train_roberta.shape[0] == len(roberta_train_sample)
assert test_roberta.shape[0] == len(roberta_test_sample)
print("RoBERTa dims:", train_roberta.shape[1])
```

Code cell (OpenAI, guarded):
```python
if config.OPENAI_API_KEY:
    train_openai = get_openai_embeddings(train_sample["text"].tolist(), cache_name=f"openai_train_{suffix}")
    test_openai = get_openai_embeddings(test_clean["text"].tolist(), cache_name="openai_test_full")
    assert train_openai.shape[0] == len(train_sample)
    print("OpenAI dims:", train_openai.shape[1])
else:
    print("OPENAI_API_KEY not set in .env — skipping OpenAI embeddings.")
```

Code cell (NaN sanity check):
```python
for name, arr in [("tfidf", train_tfidf), ("minilm", train_minilm), ("roberta", train_roberta)]:
    assert not np.isnan(arr).any(), f"{name} embeddings contain NaNs"
print("No NaNs in any computed embedding matrix.")
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 3: Execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_embeddings.ipynb --ExecutePreprocessor.timeout=1800`
Expected: exits 0, completes in well under 30 minutes now that RoBERTa uses the small `ROBERTA_SAMPLE_SIZE` cap (~500 rows each side, single-digit minutes at measured throughput) instead of the full 8000+7600.

- [ ] **Step 4: Verify cache files exist**

Run: `ls embeddings_cache/`
Expected: at least `tfidf_train_n8000.npy`, `tfidf_test_full.npy`, `minilm_train_n8000.npy`, `minilm_test_full.npy`, `roberta_train_n500.npy`, `roberta_test_n500.npy` (plus `openai_*` if an API key was configured). The pre-existing `tfidf_*`/`minilm_*` cache files from an earlier run are still valid and will be reused, not recomputed — that's expected.

- [ ] **Step 5: Commit**

```bash
git add utils/config.py notebooks/01_embeddings.ipynb
git commit -m "Add 01_embeddings notebook; add ROBERTA_SAMPLE_SIZE config"
```

---

## Task 8: Notebook `02_unsupervised_clustering.ipynb`

**Files:**
- Create: `notebooks/02_unsupervised_clustering.ipynb`

**Interfaces:**
- Consumes: cached embeddings from Task 7 via `utils.embeddings.load_cached`, `utils.metrics.evaluate_unsupervised`
- Produces: `results/metrics_{method}_kmeans.json`, `results/metrics_{method}_hdbscan.json` for each embedding method — consumed by notebook task 13.

**Important — RoBERTa uses a different, smaller sample than the other methods (see Task 7's runtime note).** Its cached embeddings only cover `config.ROBERTA_SAMPLE_SIZE` rows, not `config.SAMPLE_SIZE`. The `true_labels` array used to score each method's clustering must be sampled the same way and the same size as that method's embeddings, or `evaluate_unsupervised` will silently misalign labels against embeddings (or crash on a shape mismatch). Load `roberta`'s true labels from a separately-sampled `roberta_train_sample`, not from the general `train_sample`.

**Also important — gate OpenAI inclusion on cache existence, not just key presence.** Task 7's OpenAI cell degrades gracefully on API errors (e.g. quota exhaustion) — a configured `OPENAI_API_KEY` does not guarantee `openai_train_{suffix}.npy` actually exists. Check `load_cached(...) is not None` before adding `"openai"` to `METHODS`, not just `if config.OPENAI_API_KEY`.

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 02 — Unsupervised Clustering

KMeans and HDBSCAN over each cached embedding method (UMAP-reduced first),
scored against the hidden ground-truth labels via the Hungarian-matched
accuracy and standard clustering metrics.
```

Code cell (path setup + imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json

import hdbscan
import pandas as pd
import umap
from sklearn.cluster import KMeans

from utils import config
from utils.data import stratified_sample
from utils.embeddings import load_cached
from utils.metrics import evaluate_unsupervised
```

Code cell (load embeddings + true labels):
```python
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")

train_sample = stratified_sample(train_clean, config.SAMPLE_SIZE, seed=config.SEED)
suffix = f"n{config.SAMPLE_SIZE}" if config.SAMPLE_SIZE else "full"
true_labels = train_sample["label"].to_numpy()

# RoBERTa was embedded on a separate, smaller sample (see Task 7) — its
# true_labels must come from that same sample, not train_sample above.
roberta_train_sample = stratified_sample(train_clean, config.ROBERTA_SAMPLE_SIZE, seed=config.SEED)
roberta_suffix = f"n{config.ROBERTA_SAMPLE_SIZE}" if config.ROBERTA_SAMPLE_SIZE else "full"
roberta_true_labels = roberta_train_sample["label"].to_numpy()

METHODS = ["tfidf", "minilm", "roberta"]
if config.OPENAI_API_KEY and load_cached(f"openai_train_{suffix}") is not None:
    # Gate on cache existence, not just key presence — the OpenAI embedding
    # cell in 01_embeddings degrades gracefully on API errors (e.g. quota
    # exhaustion), so a configured key doesn't guarantee the cache exists.
    METHODS.append("openai")

embeddings_by_method = {}
labels_by_method = {}
for method in METHODS:
    method_suffix = roberta_suffix if method == "roberta" else suffix
    arr = load_cached(f"{method}_train_{method_suffix}")
    assert arr is not None, f"Missing cached embeddings for '{method}' — run 01_embeddings.ipynb first"
    embeddings_by_method[method] = arr
    labels_by_method[method] = roberta_true_labels if method == "roberta" else true_labels
    assert arr.shape[0] == len(labels_by_method[method]), \
        f"{method}: embeddings ({arr.shape[0]} rows) and labels ({len(labels_by_method[method])} rows) size mismatch"
```

Code cell (cluster + evaluate + save):
```python
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
all_results = {}

for method, emb in embeddings_by_method.items():
    method_labels = labels_by_method[method]

    reducer = umap.UMAP(n_components=50, metric="cosine", random_state=config.SEED)
    emb_reduced = reducer.fit_transform(emb)

    kmeans = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
    km_labels = kmeans.fit_predict(emb_reduced)
    km_metrics = evaluate_unsupervised(method_labels, km_labels, emb_reduced)
    all_results[f"{method}_kmeans"] = km_metrics

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric="euclidean",
                                 cluster_selection_method="eom")
    hdb_labels = clusterer.fit_predict(emb_reduced)
    hdb_metrics = evaluate_unsupervised(method_labels, hdb_labels, emb_reduced)
    all_results[f"{method}_hdbscan"] = hdb_metrics

    print(f"{method}: KMeans ACC={km_metrics['ACC (Hungarian)']:.3f} | "
          f"HDBSCAN coverage={hdb_metrics['Coverage']:.2f} ACC={hdb_metrics['ACC (Hungarian)']:.3f}")

for name, metrics in all_results.items():
    with open(config.RESULTS_DIR / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

print(f"Saved {len(all_results)} result files to {config.RESULTS_DIR}")
```

Markdown cell (stretch goal, no code required for the deliverable):
```markdown
### Optional stretch: LLM cluster naming

Instead of majority-vote mapping (which uses hidden ground truth and is
eval-only), an LLM could name each cluster from its top-N nearest documents
via `openai_client.chat.completions.create(...)`, following the
`llm_name_cluster` pattern in `autolabel_project_spec.md`. Not required for
the core comparison table — skipped here.
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_unsupervised_clustering.ipynb --ExecutePreprocessor.timeout=1200`
Expected: exits 0.

- [ ] **Step 3: Verify outputs**

Run: `ls results/metrics_*kmeans.json results/metrics_*hdbscan.json`
Expected: one KMeans + one HDBSCAN result file per method in `METHODS`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_unsupervised_clustering.ipynb
git commit -m "Add 02_unsupervised_clustering notebook"
```

---

## Task 9: Notebook `03_bertopic.ipynb`

**Files:**
- Create: `notebooks/03_bertopic.ipynb`

**Interfaces:**
- Consumes: `data/processed/train_clean.parquet` (Task 6), `utils.data.stratified_sample`, `utils.metrics.evaluate_unsupervised`
- Produces: `results/metrics_bertopic.json` — consumed by notebook task 13.

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 03 — BERTopic

BERTopic owns its own embedding (MiniLM) and clustering (UMAP + HDBSCAN
internally), so it's kept separate from notebook 02 rather than reusing
its cached embeddings.
```

Code cell (path setup + imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP

from utils import config
from utils.data import stratified_sample
from utils.metrics import evaluate_unsupervised
```

Code cell (load + sample):
```python
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
train_sample = stratified_sample(train_clean, config.SAMPLE_SIZE, seed=config.SEED)
texts = train_sample["text"].tolist()
true_labels = train_sample["label"].to_numpy()
```

Code cell (fit BERTopic):
```python
umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=config.SEED)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model,
                        nr_topics=config.NUM_CLASSES, calculate_probabilities=False)

topics, _ = topic_model.fit_transform(texts)
print(topic_model.get_topic_info())
```

Code cell (evaluate + save):
```python
topics_arr = np.array(topics)
topic_embeddings = sentence_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
bertopic_metrics = evaluate_unsupervised(true_labels, topics_arr, topic_embeddings)
print(bertopic_metrics)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_bertopic.json", "w") as f:
    json.dump(bertopic_metrics, f, indent=2)
print("Saved BERTopic metrics.")
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_bertopic.ipynb --ExecutePreprocessor.timeout=1200`
Expected: exits 0.

- [ ] **Step 3: Verify output**

Run: `cat results/metrics_bertopic.json`
Expected: valid JSON with the same metric keys as `evaluate_unsupervised` (ACC, NMI, ARI, etc.).

- [ ] **Step 4: Commit**

```bash
git add notebooks/03_bertopic.ipynb
git commit -m "Add 03_bertopic notebook"
```

---

## Task 10: Notebook `04_weak_supervision.ipynb`

**Files:**
- Create: `notebooks/04_weak_supervision.ipynb`

**Interfaces:**
- Consumes: `data/processed/{unlabeled,labeled}.parquet` (Task 6), `utils.data.stratified_sample`, `utils.metrics.evaluate_label_quality`
- Produces: `results/metrics_weak_supervision.json` — consumed by notebook task 13.

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 04 — Weak Supervision (Snorkel)

Six labeling functions (keyword + regex heuristics) vote on a label;
Snorkel's `LabelModel` combines them probabilistically. If Snorkel fails
to install (numpy/pandas version pinning issues are common on Windows),
see the commented Plan B cell at the bottom for a plain-pandas fallback.
```

Code cell (path setup + imports + label constants):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json
import re

import pandas as pd
from snorkel.labeling import LFAnalysis, PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

from utils import config
from utils.data import stratified_sample
from utils.metrics import evaluate_label_quality

ABSTAIN = -1
WORLD, SPORTS, BUSINESS, SCITECH = 0, 1, 2, 3  # must match utils.config.CLASS_NAMES order
```

Code cell (labeling functions):
```python
WORLD_KEYWORDS = ["war", "government", "election", "military", "president",
                   "troops", "peace", "nato", "un ", "united nations"]
SPORTS_KEYWORDS = ["game", "player", "coach", "season", "championship",
                    "league", "tournament", "score", "athlete", "nfl", "nba"]
BUSINESS_KEYWORDS = ["stock", "market", "shares", "revenue", "profit",
                      "ceo", "earnings", "investor", "company", "merger"]
SCITECH_KEYWORDS = ["software", "technology", "research", "scientist",
                     "space", "nasa", "quantum", "ai", "computer", "drug"]


def keyword_lf(text, keywords, label):
    text_lower = text.lower()
    return label if any(kw in text_lower for kw in keywords) else ABSTAIN


@labeling_function()
def lf_world(x): return keyword_lf(x.text, WORLD_KEYWORDS, WORLD)


@labeling_function()
def lf_sports(x): return keyword_lf(x.text, SPORTS_KEYWORDS, SPORTS)


@labeling_function()
def lf_business(x): return keyword_lf(x.text, BUSINESS_KEYWORDS, BUSINESS)


@labeling_function()
def lf_scitech(x): return keyword_lf(x.text, SCITECH_KEYWORDS, SCITECH)


@labeling_function()
def lf_has_percentage(x):
    return BUSINESS if re.search(r'\d+\.?\d*\%', x.text) else ABSTAIN


@labeling_function()
def lf_has_score(x):
    return SPORTS if re.search(r'\b\d{1,2}-\d{1,2}\b', x.text) else ABSTAIN


lfs = [lf_world, lf_sports, lf_business, lf_scitech, lf_has_percentage, lf_has_score]
```

Code cell (load + sample + apply LFs):
```python
unlabeled_df = pd.read_parquet(config.PROCESSED_DIR / "unlabeled.parquet")
labeled_df = pd.read_parquet(config.PROCESSED_DIR / "labeled.parquet")

unlabeled_sample = stratified_sample(unlabeled_df, config.SAMPLE_SIZE, seed=config.SEED)

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=unlabeled_sample)
L_dev = applier.apply(df=labeled_df)

print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
print(LFAnalysis(L=L_dev, lfs=lfs).lf_summary(Y=labeled_df["label"].to_numpy()))
```

Code cell (coverage sanity check):
```python
overall_coverage = (L_train != ABSTAIN).any(axis=1).mean()
assert overall_coverage > 0.05, f"LF coverage suspiciously low: {overall_coverage:.2%}"
print(f"Overall LF coverage on unlabeled sample: {overall_coverage:.2%}")
```

Code cell (fit LabelModel + predict):
```python
label_model = LabelModel(cardinality=config.NUM_CLASSES, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr=0.001, seed=config.SEED)

proba_labels = label_model.predict_proba(L=L_train)
hard_labels = label_model.predict(L=L_train)
confidence = proba_labels.max(axis=1)
```

Code cell (evaluate against hidden ground truth + save):
```python
label_quality = evaluate_label_quality(
    true_labels=unlabeled_sample["true_label"].to_numpy(),
    pseudo_labels=hard_labels,
    confidence_scores=confidence)
print("Weak supervision label quality:", label_quality)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_weak_supervision.json", "w") as f:
    json.dump(label_quality, f, indent=2)
```

Markdown + code cell (Plan B fallback, commented — only run manually if Snorkel install fails):
```markdown
### Plan B: if Snorkel won't install

Replace the `LabelModel` fit/predict cell above with this plain-pandas
weighted-vote combiner (less rigorous — no learned per-LF weights — but
dependency-free):
```
```python
# import numpy as np
#
# def weighted_vote_labels(L):
#     votes = np.zeros((L.shape[0], config.NUM_CLASSES))
#     for col in range(L.shape[1]):
#         valid = L[:, col] != ABSTAIN
#         votes[valid, L[valid, col]] += 1
#     hard = votes.argmax(axis=1)
#     hard[votes.sum(axis=1) == 0] = ABSTAIN
#     return hard
#
# hard_labels = weighted_vote_labels(L_train)
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_weak_supervision.ipynb --ExecutePreprocessor.timeout=900`
Expected: exits 0. If Snorkel fails to install in Task 1 Step 5, apply the Plan B cell manually and re-run.

- [ ] **Step 3: Verify output**

Run: `cat results/metrics_weak_supervision.json`
Expected: valid JSON with `Label Accuracy`, `Label Macro F1`, `Coverage`, `Mean Confidence`, `Median Confidence`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_weak_supervision.ipynb
git commit -m "Add 04_weak_supervision notebook"
```

---

## Task 11: Notebook `05_pseudo_labeling.ipynb`

**Files:**
- Create: `notebooks/05_pseudo_labeling.ipynb`

**Interfaces:**
- Consumes: `data/processed/{labeled,unlabeled,test_clean}.parquet` (Task 6), `utils.modeling.{pseudo_label_loop, get_predictions}` (Task 5), `utils.metrics.{evaluate_label_quality, evaluate_semisupervised}` (Task 3), `utils.config.CLASSIFIER_SAMPLE_SIZE` (new constant, added by this task)
- Produces: `results/metrics_pseudo_labeling.json`, `results/confusion_matrix_pseudo_labeling.png` — consumed by notebook task 13.

**Runtime note — fine-tuning uses its own, much smaller sample than the embedding/clustering tasks.** Measured on this machine: `pseudo_label_loop`'s 4 DistilBERT fine-tuning calls (3 self-training iterations + 1 final retrain) plus 3 prediction passes did not finish within a 3600-second (60 minute) cell timeout at the original scale (~6000 labeled rows, up to `SAMPLE_SIZE=8000` unlabeled rows) — actual fine-tuning throughput is far more expensive than frozen embedding, and per-call model/tokenizer construction overhead (~85-90s) compounds across 4 calls. `utils/config.py` gets a new constant `CLASSIFIER_SAMPLE_SIZE = 500`, added in this task's Step 1, capping **both** `labeled_df` and `unlabeled_sample` for this notebook (and reused by Task 12). This is independent of `SAMPLE_SIZE` (used by TF-IDF/MiniLM/OpenAI) and `ROBERTA_SAMPLE_SIZE` (used by frozen RoBERTa embedding) — three separate dials for three different cost profiles. For the eventual full-scale final run, raise `CLASSIFIER_SAMPLE_SIZE` to `None` alongside the others, and budget several hours, unattended.

- [ ] **Step 1: Add `CLASSIFIER_SAMPLE_SIZE` to `utils/config.py`**

Add this line to `utils/config.py`, directly after the existing `ROBERTA_SAMPLE_SIZE = 500` line (from Task 7) — don't change anything else in the file:
```python
CLASSIFIER_SAMPLE_SIZE = 500  # separate, smaller cap for DistilBERT fine-tuning (tasks 11, 12); None = full data
```

- [ ] **Step 2: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 05 — Pseudo-Labeling (Self-Training)

Iteratively fine-tunes `utils.config.CLASSIFIER_MODEL_NAME` on the labeled
pool, predicts on the unlabeled pool, and absorbs high-confidence
predictions each round. Uses `CLASSIFIER_SAMPLE_SIZE` (much smaller than
the embedding tasks' `SAMPLE_SIZE`) since fine-tuning is far more
CPU-expensive than frozen embedding — this is still the most expensive
notebook in the plan. The full-data final run should be kicked off
unattended and budgeted in hours, not minutes.
```

Code cell (path setup + imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json

import pandas as pd

from utils import config
from utils.data import stratified_sample
from utils.metrics import evaluate_label_quality, evaluate_semisupervised
from utils.modeling import get_predictions, pseudo_label_loop
```

Code cell (load + sample + leakage check):
```python
labeled_df = pd.read_parquet(config.PROCESSED_DIR / "labeled.parquet")
unlabeled_df = pd.read_parquet(config.PROCESSED_DIR / "unlabeled.parquet")
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")

labeled_sample = stratified_sample(labeled_df, config.CLASSIFIER_SAMPLE_SIZE, seed=config.SEED)
unlabeled_sample = stratified_sample(unlabeled_df, config.CLASSIFIER_SAMPLE_SIZE, seed=config.SEED)

overlap = set(unlabeled_sample["text"]) & set(test_clean["text"])
assert len(overlap) == 0, f"{len(overlap)} rows leaked between train pool and test set"
print(f"Labeled sample: {len(labeled_sample)} | Unlabeled sample: {len(unlabeled_sample)} | Test: {len(test_clean)}")
print("No train/test text overlap confirmed.")
```

Code cell (run self-training loop):
```python
final_model, final_tokenizer, current_labeled, history = pseudo_label_loop(
    labeled_sample, unlabeled_sample,
    model_name=config.CLASSIFIER_MODEL_NAME, n_iterations=3,
    confidence_threshold=0.90, epochs=3)

for h in history:
    print(h)
```

Code cell (pseudo-label quality against hidden ground truth):
```python
pseudo_only = current_labeled.iloc[len(labeled_sample):]
merged = pseudo_only.merge(unlabeled_sample[["text", "true_label"]], on="text", how="left")

label_quality = evaluate_label_quality(
    true_labels=merged["true_label"].to_numpy(),
    pseudo_labels=merged["label"].to_numpy())
print("Pseudo-label quality:", label_quality)
```

Code cell (final test evaluation + save):
```python
test_probs = get_predictions(final_model, final_tokenizer, test_clean["text"].tolist())
test_preds = test_probs.argmax(axis=1)

semisup_results, report, cm = evaluate_semisupervised(
    test_clean["label"].to_numpy(), test_preds, config.CLASS_NAMES,
    save_path=config.RESULTS_DIR / "confusion_matrix_pseudo_labeling.png")
print(report)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_pseudo_labeling.json", "w") as f:
    json.dump({"test_metrics": semisup_results, "label_quality": label_quality, "history": history},
               f, indent=2)
print("Saved pseudo-labeling results.")
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 3: Execute the notebook end-to-end**

Run this as a background command (NOT a single foreground command with a fixed wait — even at `CLASSIFIER_SAMPLE_SIZE=500` this involves 4 real fine-tuning runs and may take 15-30+ minutes; do not assume it will finish within any tool's default foreground timeout):
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_pseudo_labeling.ipynb --ExecutePreprocessor.timeout=3600`
Launch it as a single background process (no double-backgrounding — don't combine shell-level `&`/`nohup` with a tool-level background flag, pick exactly one backgrounding mechanism) and poll for `results/metrics_pseudo_labeling.json` to appear, or watch for the process to exit. Expected: exits 0, well under the 3600s cell timeout at this sample size.

- [ ] **Step 4: Verify output**

Run: `cat results/metrics_pseudo_labeling.json`
Expected: valid JSON with `test_metrics`, `label_quality`, `history` keys; `results/confusion_matrix_pseudo_labeling.png` exists.

- [ ] **Step 5: Commit**

```bash
git add utils/config.py notebooks/05_pseudo_labeling.ipynb
git commit -m "Add 05_pseudo_labeling notebook; add CLASSIFIER_SAMPLE_SIZE config"
```

---

## Task 12: Notebook `06_full_supervised_baseline.ipynb`

**Files:**
- Create: `notebooks/06_full_supervised_baseline.ipynb`

**Interfaces:**
- Consumes: `data/processed/{train_clean,test_clean}.parquet` (Task 6), `utils.modeling.{train_model, get_predictions}` (Task 5), `utils.metrics.evaluate_semisupervised` (Task 3), `utils.config.CLASSIFIER_SAMPLE_SIZE` (added by Task 11)
- Produces: `results/metrics_full_supervised.json`, `results/confusion_matrix_full_supervised.png` — consumed by notebook task 13.

**Uses `CLASSIFIER_SAMPLE_SIZE`, not `SAMPLE_SIZE`** — same reasoning as Task 11: DistilBERT fine-tuning is far more expensive than frozen embedding, so this task trains on the smaller `CLASSIFIER_SAMPLE_SIZE`-capped sample rather than the full `SAMPLE_SIZE=8000`.

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 06 — Full Supervised Baseline

Trains `utils.config.CLASSIFIER_MODEL_NAME` on 100% of available labels —
the upper-bound target every other method is compared against. Uses
`CLASSIFIER_SAMPLE_SIZE` (see Task 11's runtime note) since fine-tuning is
CPU-expensive.
```

Code cell (path setup + imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json

import pandas as pd

from utils import config
from utils.data import stratified_sample
from utils.metrics import evaluate_semisupervised
from utils.modeling import get_predictions, train_model
```

Code cell (load + sample):
```python
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")

train_sample = stratified_sample(train_clean, config.CLASSIFIER_SAMPLE_SIZE, seed=config.SEED)
print(f"Training on {len(train_sample)} fully-labeled rows (upper bound baseline)")
```

Code cell (train + evaluate + save):
```python
model, tokenizer = train_model(train_sample, model_name=config.CLASSIFIER_MODEL_NAME, epochs=3)

test_probs = get_predictions(model, tokenizer, test_clean["text"].tolist())
test_preds = test_probs.argmax(axis=1)

results, report, cm = evaluate_semisupervised(
    test_clean["label"].to_numpy(), test_preds, config.CLASS_NAMES,
    save_path=config.RESULTS_DIR / "confusion_matrix_full_supervised.png")
print(report)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_full_supervised.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved full-supervised baseline results.")
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_full_supervised_baseline.ipynb --ExecutePreprocessor.timeout=1800`
Expected: exits 0. Only one fine-tuning call this time (vs. Task 11's four), so this should complete in single-digit minutes at `CLASSIFIER_SAMPLE_SIZE=500` — but if running it as a single foreground command, use a generous timeout (at least 600000ms) and prefer a background launch with a single backgrounding mechanism if there's any doubt, per Task 11's lesson.

- [ ] **Step 3: Verify output**

Run: `cat results/metrics_full_supervised.json`
Expected: valid JSON with `Accuracy`, `Macro F1`, etc.; `results/confusion_matrix_full_supervised.png` exists.

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_full_supervised_baseline.ipynb
git commit -m "Add 06_full_supervised_baseline notebook"
```

---

## Task 13: Notebook `07_comparison.ipynb`

**Files:**
- Create: `notebooks/07_comparison.ipynb`

**Interfaces:**
- Consumes: every `results/metrics_*.json` file produced by Tasks 8–12
- Produces: `results/comparison_table.csv`, `results/comparison_bar_chart.png`

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 07 — Final Comparison

Aggregates every method's saved metrics into one table and bar chart. Run
this last, after notebooks 02–06.
```

Code cell (path setup + imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json

import matplotlib.pyplot as plt
import pandas as pd

from utils import config
```

Code cell (load all results):
```python
UNSUPERVISED_KEYS = ["ACC (Hungarian)", "NMI", "ARI", "Coverage"]
SEMISUPERVISED_KEYS = ["Accuracy", "Macro F1", "Coverage"]

result_files = sorted(config.RESULTS_DIR.glob("metrics_*.json"))
assert len(result_files) > 0, "No results found — run notebooks 02-06 first"

rows = {}
for path in result_files:
    name = path.stem.replace("metrics_", "")
    with open(path) as f:
        data = json.load(f)

    if "test_metrics" in data:  # pseudo-labeling nests its results
        flat = {**data["test_metrics"], "Coverage": data["label_quality"]["Coverage"]}
    else:
        flat = data
    rows[name] = flat
```

Code cell (sanity assertion + build table):
```python
for name, metrics in rows.items():
    missing_unsup = [k for k in UNSUPERVISED_KEYS if k not in metrics]
    missing_semisup = [k for k in SEMISUPERVISED_KEYS if k not in metrics]
    assert not (missing_unsup and missing_semisup), \
        f"'{name}' is missing all expected metric keys — got {list(metrics.keys())}"

df_results = pd.DataFrame(rows).T
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df_results.to_csv(config.RESULTS_DIR / "comparison_table.csv")
print(df_results.to_markdown())
```

Code cell (bar chart):
```python
metric_to_plot = "Macro F1" if "Macro F1" in df_results.columns else "ACC (Hungarian)"
plot_df = df_results[metric_to_plot].dropna().sort_values()

fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(plot_df))))
plot_df.plot(kind="barh", ax=ax)
ax.set_xlabel(metric_to_plot)
ax.set_title(f"Method comparison — {metric_to_plot}")
plt.tight_layout()
plt.savefig(config.RESULTS_DIR / "comparison_bar_chart.png", dpi=150)
plt.show()
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_comparison.ipynb`
Expected: exits 0.

- [ ] **Step 3: Verify outputs**

Run: `cat results/comparison_table.csv`
Expected: one row per method (tfidf_kmeans, tfidf_hdbscan, minilm_kmeans, ..., bertopic, weak_supervision, pseudo_labeling, full_supervised) with metric columns; `results/comparison_bar_chart.png` exists.

- [ ] **Step 4: Commit**

```bash
git add notebooks/07_comparison.ipynb
git commit -m "Add 07_comparison notebook"
```

---

## Final Run (not a task — a follow-up procedure)

Once all 13 tasks pass with `SAMPLE_SIZE = 8000`, set `SAMPLE_SIZE = None` in
`utils/config.py`, delete `embeddings_cache/*n8000*.npy` (so full-data
embeddings get computed fresh under new cache names), and re-run notebooks
01, 02, 03, 04, 05, 06, 07 in order — ideally unattended (05 and 06 will take
the longest). The resulting `results/comparison_table.csv` is the final
deliverable.
