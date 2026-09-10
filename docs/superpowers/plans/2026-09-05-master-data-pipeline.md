# Master-Data Pipeline Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the AG News pipeline with `data/master_data.csv` (16 categories, ~4,750 rows after cleaning) — a summary-first unsupervised clustering track and a raw-text semi-supervised track (weak supervision, label propagation, pseudo-labeling with DistilBERT + ELECTRA-small), all running on the full dataset with no dev-scale sampling caps.

**Architecture:** Rebuild the existing `utils/` package + 12-notebook pipeline (`00`–`09`, `05b`/`06b`) in place. `00_data_transform` becomes the single place that cleans `master_data.csv`, splits it, and generates a summary sentence per row (via `facebook/bart-large-cnn`) that every downstream notebook can read off `train_clean`/`test_clean` parquet without recomputing. The unsupervised track (`01`–`03`) embeds and clusters those summaries; the semi-supervised track (`04`, `05`/`05b`, `08`) and baseline (`06`/`06b`) train on raw text, joining the summary column in only for their new full-output CSVs. `09_summarization_labeling.ipynb` is retired (its role is subsumed by the new summary-clustering flow).

**Tech Stack:** `uv` (`pyproject.toml`/`uv.lock`), pure CPU (no GPU — confirmed no CUDA/MPS on this machine), Hugging Face `transformers` (`facebook/bart-large-cnn` summarizer, DistilBERT/ELECTRA-small classifiers), `sentence-transformers`, `snorkel`, `scikit-learn`, `umap-learn`, `hdbscan`, `bertopic`, `keybert`.

**Spec:** `docs/superpowers/specs/2026-09-05-master-data-pipeline-design.md`

## Global Constraints

- `SEED = 42` everywhere randomness occurs (existing project convention).
- No pytest / no `tests/` directory — matches this repo's existing convention. `utils/` functions are verified with one-off `uv run python` smoke checks when first written; notebooks' own inline `assert`s are the ongoing regression check, executed end-to-end via `jupyter nbconvert`.
- **`uv` is the only environment/dependency tool** — no `pip install`. All Python/Jupyter commands prefixed `uv run`.
- Every notebook execution (`nbconvert --execute`) that could run more than a couple minutes MUST be launched as a background process, not a blocking foreground call. Poll for the expected output file or process exit; do not combine shell-level backgrounding with a tool-level background flag — pick exactly one.
- `CLASS_NAMES` = the 16 `data/master_data.csv` categories, alphabetical (`utils/config.py`); `NUM_CLASSES = 16`. Labels are 0-indexed positions into this list.
- **Full data everywhere** — no sampling caps. `config.SAMPLE_SIZE`, `ROBERTA_SAMPLE_SIZE`, `CLASSIFIER_SAMPLE_SIZE` all default to `None` (kept as tunable knobs, not deleted, in case fast dev-scale iteration is needed again later).
- Unsupervised track (notebooks `01`, `02`, `03`) trains/clusters on the **generated `summary` column**. Semi-supervised track and baseline (`04`, `05`, `05b`, `06`, `06b`, `08`) train on the **raw `text` column**; `summary` is joined into their output only for the new full-output CSVs, never used as training input there.
- Every method notebook saves two output CSVs: the existing small qualitative sample (`utils.samples.save_label_samples`, 2 examples/class) **and** a new full-row output (`utils.samples.save_full_output`, Task 3) covering every row it produced a prediction for, columns `text, summary, predicted_label, true_label[, confidence]`.

---

## Task 0: Clear AG News leftovers

**Files:**
- Delete (finalize already-removed-on-disk deletion): `data/test.csv`, `data/train.csv`

**Interfaces:** none — pure repo hygiene so the AG News CSVs (superseded by `data/master_data.csv`) aren't left as an untracked-deletion in `git status` for the rest of this plan.

- [ ] **Step 1: Stage the deletion**

```bash
git rm data/test.csv data/train.csv
```
(These already show as deleted-on-disk in `git status`; this just stages that deletion so it's part of history rather than a dangling working-tree change.)

- [ ] **Step 2: Commit**

```bash
git commit -m "Remove AG News train/test.csv, superseded by data/master_data.csv"
```

---

## Task 1: Config, data utilities, and summarization model swap

**Files:**
- Modify: `utils/config.py`
- Modify: `utils/data.py`
- Modify: `utils/summarization.py`

**Interfaces:**
- Produces: `utils.config.{CLASS_NAMES (16-item), NUM_CLASSES=16, MASTER_DATA_PATH, TEST_FRACTION, MIN_WORDS, SILHOUETTE_SAMPLE_SIZE, SAMPLE_SIZE=None, ROBERTA_SAMPLE_SIZE=None, CLASSIFIER_SAMPLE_SIZE=None, SUMMARIZATION_MODEL_NAME="facebook/bart-large-cnn"}` — consumed by every later task.
- Produces: `utils.data.load_master_data(path, class_names, min_words=5) -> pd.DataFrame`, `utils.data.stratified_train_test_split(df, test_fraction, seed, label_col="label") -> (train_df, test_df)` — consumed by Task 5 (notebook `00`).
- Removes: `utils.data.{load_raw, build_text_column}` (AG-News-specific, unused after this task), `utils.config.{ZERO_SHOT_MODEL_NAME, SUMMARIZATION_SAMPLE_SIZE}`, `utils.summarization.{zero_shot_label, true_class_scores}` (only used by the retired notebook `09`; removed now so nothing downstream imports a name that no longer exists).
- Keeps unchanged: `utils.data.{make_splits, stratified_sample}` (already generic), `utils.summarization.{summarize_texts, generate_titles}` (pick up the new model via their `SUMMARIZATION_MODEL_NAME` default — no code change needed there).

- [ ] **Step 1: Rewrite `utils/config.py`**

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

MASTER_DATA_PATH = DATA_DIR / "master_data.csv"

# 16 balanced categories in data/master_data.csv, alphabetical so the
# label<->name mapping is deterministic and independent of CSV row order.
CLASS_NAMES = [
    "ARTS & CULTURE", "BUSINESS", "COMEDY", "CRIME", "EDUCATION",
    "ENTERTAINMENT", "ENVIRONMENT", "HEALTH", "MEDIA", "NEWS",
    "POLITICS", "RELIGION", "SCIENCE", "SPORTS", "TECH", "WOMEN",
]
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42
LABEL_FRACTION = 0.05
TEST_FRACTION = 0.20  # stratified train/test split of master_data.csv
MIN_WORDS = 5  # drop degenerate rows (e.g. a body of just "(CNN)") below this word count

# master_data.csv is only ~4,750 rows after cleaning (vs. AG News's
# 120,000) — small enough that every method runs on the FULL dataset.
# None = no cap. Kept as tunable knobs (not deleted) in case a future
# dev-scale iteration needs fast sampling again.
SAMPLE_SIZE = None
ROBERTA_SAMPLE_SIZE = None
CLASSIFIER_SAMPLE_SIZE = None

# silhouette_score/davies_bouldin_score are O(n^2) in the number of points
# — sub-sample before computing them now that clustering runs full-data.
SILHOUETTE_SAMPLE_SIZE = 2000

CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"

# Additional semi-supervised model (mentor feedback item 1) — compared
# *alongside* CLASSIFIER_MODEL_NAME, not replacing it.
CLASSIFIER_MODEL_NAME_ALT = "google/electra-small-discriminator"

# Summarization model feeding the unsupervised track (raw text -> summary
# sentence -> embed -> cluster). Swapped from the distilled
# sshleifer/distilbart-cnn-6-6 to the full BART-large model it was
# distilled from — more fluent, coherent multi-sentence summaries, at the
# cost of slower per-row CPU inference (a one-time cost, cached to disk
# via train_clean/test_clean's saved `summary` column).
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"
```

- [ ] **Step 2: Rewrite `utils/data.py`**

```python
import pandas as pd


def load_master_data(path, class_names, min_words: int = 5) -> pd.DataFrame:
    """Load and clean `data/master_data.csv` (columns: category, title,
    summary, text). Returns a DataFrame with `text` (title + body,
    deduplicated), `category` (cleaned string), and `label` (0-indexed
    position into `class_names`).

    Cleaning: drops the source `summary` column (populated for only a
    minority of rows/categories — every row gets a freshly generated
    summary later, in 00_data_transform, instead of reusing this patchy
    one), combines title+text into one `text` field, drops duplicate
    `text` rows (keep first), and drops degenerate rows below `min_words`
    words (e.g. a body of just "(CNN)").
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["summary"], errors="ignore")
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df.drop_duplicates(subset="text", keep="first")

    word_count = df["text"].str.split().str.len()
    df = df[word_count >= min_words].copy()

    df["category"] = df["category"].str.strip()
    unknown = set(df["category"].unique()) - set(class_names)
    assert not unknown, f"Unexpected categories not in class_names: {unknown}"

    label_by_name = {name: i for i, name in enumerate(class_names)}
    df["label"] = df["category"].map(label_by_name)
    return df.reset_index(drop=True)


def stratified_train_test_split(df: pd.DataFrame, test_fraction: float, seed: int,
                                 label_col: str = "label"):
    """Stratified train/test split by `label_col` (every class split at the
    same fraction). Returns (train_df, test_df), both index-reset."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_fraction, stratify=df[label_col], random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_splits(train_df: pd.DataFrame, label_fraction: float, seed: int):
    """Split into a small labeled pool and a large unlabeled pool.

    The unlabeled pool's real label is kept as `true_label` for evaluation
    only — `label` is set to -1 to simulate it being unavailable to any
    training algorithm.
    """
    indexed_df = train_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=label_fraction, random_state=seed))
    sampled_index = indexed_df.index
    labeled_df = train_df.loc[sampled_index]
    unlabeled_df = train_df.drop(sampled_index).copy()
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
    indexed = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=seed))
    return df.loc[indexed.index].reset_index(drop=True)
```

- [ ] **Step 3: Edit `utils/summarization.py`** — remove the now-unused zero-shot pieces

Change the top import line from:
```python
from utils.config import SUMMARIZATION_MODEL_NAME, ZERO_SHOT_MODEL_NAME
```
to:
```python
from utils.config import SUMMARIZATION_MODEL_NAME
```

Delete the `zero_shot_label` and `true_class_scores` functions entirely (everything from `def zero_shot_label(...)` to the end of the file). Keep `summarize_texts` and `generate_titles` unchanged — they already read `SUMMARIZATION_MODEL_NAME` from config, so they pick up `facebook/bart-large-cnn` automatically.

- [ ] **Step 4: Verify**

```bash
uv run python -c "
from utils import config
print(config.CLASS_NAMES)
assert config.NUM_CLASSES == 16
assert config.SUMMARIZATION_MODEL_NAME == 'facebook/bart-large-cnn'
assert config.SAMPLE_SIZE is None and config.CLASSIFIER_SAMPLE_SIZE is None
print('config OK')
"
uv run python -c "
import pandas as pd
from utils.data import load_master_data, stratified_train_test_split
from utils import config

df = load_master_data(config.MASTER_DATA_PATH, config.CLASS_NAMES, min_words=config.MIN_WORDS)
print(len(df), 'rows after cleaning')
assert df['text'].duplicated().sum() == 0
assert df['label'].isna().sum() == 0
train_df, test_df = stratified_train_test_split(df, test_fraction=config.TEST_FRACTION, seed=config.SEED)
print('train', len(train_df), 'test', len(test_df))
assert len(train_df) + len(test_df) == len(df)
print('OK')
"
uv run python -c "from utils.summarization import summarize_texts, generate_titles; print('summarization imports OK')"
```

Expected: all three print `OK`/expected values with no errors; row count after cleaning should be a bit below 4,800 (duplicates + degenerate rows dropped).

- [ ] **Step 5: Commit**

```bash
git add utils/config.py utils/data.py utils/summarization.py
git commit -m "Rebuild config/data utils for master_data.csv (16-class, full-data), swap summarizer to bart-large-cnn"
```

---

## Task 2: `utils/metrics.py` — subsample silhouette/Davies-Bouldin at full-data scale

**Files:**
- Modify: `utils/metrics.py`

**Interfaces:**
- Modifies: `evaluate_unsupervised(true_labels, cluster_labels, embeddings, metric_sample_size=None, seed=None) -> dict` — two new optional keyword params, default `None` preserves old behavior (compute on the full non-noise set). Consumed by Task 7 (notebook `02`) and Task 8 (notebook `03`) with `metric_sample_size=config.SILHOUETTE_SAMPLE_SIZE`.

- [ ] **Step 1: Edit `evaluate_unsupervised`**

Replace the function body (keep the early-return edge case unchanged) — change the signature and the final `return` block:

```python
def evaluate_unsupervised(true_labels, cluster_labels, embeddings, metric_sample_size=None, seed=None) -> dict:
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)
    mask = cluster_labels >= 0

    # Handle edge case: if insufficient non-noise clusters for silhouette/davies-bouldin
    # (both require at least 2 distinct cluster labels), return default values
    if mask.sum() == 0 or len(np.unique(cluster_labels[mask])) < 2:
        return {
            "ACC (Hungarian)": 0.0,
            "Macro F1": 0.0,
            "NMI": 0.0,
            "ARI": 0.0,
            "FMI": 0.0,
            "Homogeneity": 0.0,
            "Completeness": 0.0,
            "V-Measure": 0.0,
            "Silhouette Score": 0.0,
            "Davies-Bouldin": 0.0,
            "Coverage": mask.sum() / len(cluster_labels),
        }

    matched_predictions = hungarian_match_predictions(true_labels[mask], cluster_labels[mask])

    # silhouette_score/davies_bouldin_score are O(n^2) in the number of
    # points — sub-sample the non-noise rows before computing just these
    # two descriptive-geometry metrics (not part of the Hungarian-matched
    # accuracy pipeline), so full-data runs stay tractable.
    embeddings_masked = np.asarray(embeddings)[mask]
    cluster_labels_masked = cluster_labels[mask]
    n_masked = mask.sum()
    if metric_sample_size is not None and metric_sample_size < n_masked:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(n_masked, size=metric_sample_size, replace=False)
        # Guard against an unlucky sub-sample collapsing to <2 clusters,
        # which would make silhouette_score raise.
        if len(np.unique(cluster_labels_masked[sub_idx])) < 2:
            sub_idx = np.arange(n_masked)
    else:
        sub_idx = np.arange(n_masked)

    return {
        "ACC (Hungarian)": clustering_accuracy(true_labels[mask], cluster_labels[mask]),
        "Macro F1": f1_score(true_labels[mask], matched_predictions, average="macro"),
        "NMI": normalized_mutual_info_score(true_labels[mask], cluster_labels[mask]),
        "ARI": adjusted_rand_score(true_labels[mask], cluster_labels[mask]),
        "FMI": fowlkes_mallows_score(true_labels[mask], cluster_labels[mask]),
        "Homogeneity": homogeneity_score(true_labels[mask], cluster_labels[mask]),
        "Completeness": completeness_score(true_labels[mask], cluster_labels[mask]),
        "V-Measure": v_measure_score(true_labels[mask], cluster_labels[mask]),
        "Silhouette Score": silhouette_score(
            embeddings_masked[sub_idx], cluster_labels_masked[sub_idx], metric="cosine"),
        "Davies-Bouldin": davies_bouldin_score(
            embeddings_masked[sub_idx], cluster_labels_masked[sub_idx]),
        "Coverage": mask.sum() / len(cluster_labels),
    }
```

- [ ] **Step 2: Smoke-test**

```bash
uv run python -c "
import numpy as np
from utils.metrics import evaluate_unsupervised

rng = np.random.default_rng(42)
true_labels = rng.integers(0, 4, size=5000)
cluster_labels = true_labels.copy()  # perfect clustering
embeddings = rng.normal(size=(5000, 10))

full = evaluate_unsupervised(true_labels, cluster_labels, embeddings)
sub = evaluate_unsupervised(true_labels, cluster_labels, embeddings, metric_sample_size=500, seed=42)
print('full ACC', full['ACC (Hungarian)'], 'sub ACC', sub['ACC (Hungarian)'])
assert full['ACC (Hungarian)'] == 1.0 == sub['ACC (Hungarian)']
assert full['Silhouette Score'] != 0.0 and sub['Silhouette Score'] != 0.0
print('OK')
"
```

Expected: prints `OK`, both ACC values are `1.0` (perfect clustering), silhouette computed in both cases without error.

- [ ] **Step 3: Commit**

```bash
git add utils/metrics.py
git commit -m "Add optional metric_sample_size to evaluate_unsupervised for full-data silhouette/Davies-Bouldin"
```

---

## Task 3: `utils/samples.py` — full-row output CSV helper

**Files:**
- Modify: `utils/samples.py`

**Interfaces:**
- Produces: `save_full_output(texts, predicted_labels, true_labels, class_names, confidence=None, extra_columns=None, path=None) -> pd.DataFrame` — the full-row (no sampling) counterpart to `save_label_samples`. Consumed by Tasks 7-12 (every method notebook).

- [ ] **Step 1: Add `save_full_output` to `utils/samples.py`**

Add the import and new function (keep `save_label_samples` unchanged):

```python
from pathlib import Path

import pandas as pd
```

```python
def save_full_output(texts, predicted_labels, true_labels, class_names,
                      confidence=None, extra_columns=None, path=None) -> pd.DataFrame:
    """Save one row per input (no per-class sampling) — the full-row
    counterpart to `save_label_samples`'s small qualitative spot-check.
    Same column contract: predicted/true label names, a `correct` flag,
    optional `confidence`, optional `extra_columns` (e.g. {"summary": [...]})
    merged in row-aligned. Unlike `save_label_samples`, `text` is NOT
    truncated here — this file is meant for full inspection, not a
    print-friendly table.
    """
    def name_or_abstain(label):
        return class_names[label] if label >= 0 else "ABSTAIN"

    df = pd.DataFrame({
        "text": list(texts),
        "predicted_label": [name_or_abstain(p) for p in predicted_labels],
        "true_label": [name_or_abstain(t) for t in true_labels],
    })
    df["correct"] = df["predicted_label"] == df["true_label"]
    if confidence is not None:
        df["confidence"] = confidence
    if extra_columns:
        for col_name, values in extra_columns.items():
            df[col_name] = list(values)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df
```

- [ ] **Step 2: Smoke-test**

```bash
uv run python -c "
from utils.samples import save_full_output

texts = ['a', 'b', 'c']
preds = [0, 1, -1]
trues = [0, 1, 1]
extra = {'summary': ['sum a', 'sum b', 'sum c']}
df = save_full_output(texts, preds, trues, ['X', 'Y'], extra_columns=extra)
print(df)
assert len(df) == 3
assert df['predicted_label'].tolist() == ['X', 'Y', 'ABSTAIN']
assert 'summary' in df.columns
print('OK')
"
```

Expected: prints the 3-row DataFrame (no sampling — all rows present, including the abstained one), then `OK`.

- [ ] **Step 3: Commit**

```bash
git add utils/samples.py
git commit -m "Add save_full_output: full-row (unsampled) label output CSV for every method"
```

---

## Task 4: `utils/weak_supervision.py` — auto-derived per-class labeling functions

**Files:**
- Create: `utils/weak_supervision.py`

**Interfaces:**
- Produces: `derive_class_keywords(labeled_df, class_names, text_col="text", label_col="label", top_n=15, max_features=20000) -> dict[int, list[str]]`, `build_keyword_lfs(keywords_by_class) -> list[snorkel.labeling.LabelingFunction]` — consumed by Task 9 (notebook `04`).

- [ ] **Step 1: Write `utils/weak_supervision.py`**

```python
"""Auto-derived weak-supervision labeling functions.

The AG News version of this project hand-wrote one keyword list per class
(4 classes). At 16 classes — several overlapping in subject matter
(POLITICS vs. NEWS vs. WOMEN vs. CRIME vs. MEDIA could all describe the
same story) — hand-authoring doesn't scale. Instead, each class's most
TF-IDF-distinctive terms (relative to the rest of the 5% labeled seed) are
turned into a keyword-vote labeling function automatically.
"""

import numpy as np


def derive_class_keywords(labeled_df, class_names, text_col="text", label_col="label",
                           top_n=15, max_features=20000) -> dict:
    """For each class, return its `top_n` most distinctive keywords: the
    terms with the largest (mean TF-IDF weight within the class) minus
    (mean TF-IDF weight in the rest of the seed).

    Returns {label_int: [term, ...]}.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=1)
    tfidf = vectorizer.fit_transform(labeled_df[text_col])
    vocab = np.array(vectorizer.get_feature_names_out())
    labels = labeled_df[label_col].to_numpy()

    keywords_by_class = {}
    for label in range(len(class_names)):
        mask = labels == label
        if mask.sum() == 0:
            keywords_by_class[label] = []
            continue
        class_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        rest_mask = ~mask
        rest_mean = (np.asarray(tfidf[rest_mask].mean(axis=0)).ravel()
                     if rest_mask.sum() else np.zeros_like(class_mean))
        distinctiveness = class_mean - rest_mean
        top_idx = distinctiveness.argsort()[::-1][:top_n]
        keywords_by_class[label] = vocab[top_idx].tolist()
    return keywords_by_class


def build_keyword_lfs(keywords_by_class):
    """Build one Snorkel `LabelingFunction` per class from its keyword
    list. Each LF votes for its class if any of its keywords appears in
    `x.text` (case-insensitive substring match), else abstains (-1).

    Returns a list of `snorkel.labeling.LabelingFunction`, skipping any
    class with an empty keyword list (e.g. zero seed rows for that class).
    """
    from snorkel.labeling import LabelingFunction

    ABSTAIN = -1

    def _make_fn(label, keywords):
        def _fn(x):
            text_lower = x.text.lower()
            return label if any(kw in text_lower for kw in keywords) else ABSTAIN
        return _fn

    lfs = []
    for label, keywords in keywords_by_class.items():
        if not keywords:
            continue
        lfs.append(LabelingFunction(name=f"lf_class_{label}", f=_make_fn(label, keywords)))
    return lfs
```

- [ ] **Step 2: Smoke-test**

```bash
uv run python -c "
import pandas as pd
from utils.weak_supervision import derive_class_keywords, build_keyword_lfs

labeled_df = pd.DataFrame({
    'text': [
        'the team won the championship game with a great goal',
        'the striker scored twice in the soccer match',
        'the stock market rallied as shares of the company rose',
        'quarterly earnings and revenue beat analyst expectations',
    ],
    'label': [0, 0, 1, 1],
})
keywords = derive_class_keywords(labeled_df, ['Sports', 'Business'], top_n=5)
print(keywords)
assert set(keywords.keys()) == {0, 1}
lfs = build_keyword_lfs(keywords)
assert len(lfs) == 2

class Row:
    text = 'the goal in the championship match was incredible'
assert lfs[0].f(Row()) == 0
print('OK')
"
```

Expected: prints the derived keyword dict (sports-flavored terms for class 0, business-flavored for class 1), then `OK`.

- [ ] **Step 3: Commit**

```bash
git add utils/weak_supervision.py
git commit -m "Add utils/weak_supervision.py: auto-derive per-class keyword labeling functions"
```

---

## Task 5: Notebook `00_data_transform.ipynb` — master_data.csv, split, generated summaries

**Files:**
- Modify: `notebooks/00_data_transform.ipynb`

**Interfaces:**
- Consumes: `utils.data.{load_master_data, stratified_train_test_split, make_splits}`, `utils.summarization.summarize_texts`, `utils.config.{MASTER_DATA_PATH, CLASS_NAMES, MIN_WORDS, TEST_FRACTION, LABEL_FRACTION, SEED, PROCESSED_DIR}` (Task 1)
- Produces: `data/processed/{train_clean,test_clean,labeled,unlabeled}.parquet`, each now carrying a `summary` column — consumed by every later task.

- [ ] **Step 1: Replace the notebook's cells, in order**

Markdown cell:
```markdown
# 00 — Data Transform

Loads `data/master_data.csv` (16 categories, 4,800 raw rows), cleans it
(dedup, drop degenerate rows), splits into a stratified train/test set,
creates the 5%-per-class semi-supervised labeled/unlabeled split, and
generates a real summary sentence for every row (`facebook/bart-large-cnn`
— feeds the unsupervised track's summary-embedding clustering in
`01`-`03`; carried along on every row for every other method's full-output
CSV too, though they train on raw `text`, not `summary`). Run this first —
every other notebook reads from `data/processed/`.
```

Code cell (imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import config
from utils.data import load_master_data, stratified_train_test_split, make_splits
from utils.summarization import summarize_texts
```

Code cell (load + clean):
```python
master_df = load_master_data(config.MASTER_DATA_PATH, config.CLASS_NAMES, min_words=config.MIN_WORDS)

assert master_df["text"].isna().sum() == 0
assert master_df["text"].duplicated().sum() == 0
assert set(master_df["category"].unique()) == set(config.CLASS_NAMES)

class_counts = master_df["label"].value_counts().sort_index()
print(f"Loaded {len(master_df)} cleaned rows (from 4,800 raw) across {config.NUM_CLASSES} classes")
print(class_counts)
assert (class_counts >= 250).all(), f"unexpectedly large class-size drop after cleaning:\n{class_counts}"
```

Code cell (split):
```python
train_clean, test_clean = stratified_train_test_split(
    master_df, test_fraction=config.TEST_FRACTION, seed=config.SEED)

assert len(train_clean) + len(test_clean) == len(master_df)
print(f"Train: {len(train_clean)} | Test: {len(test_clean)}")
print("Train class distribution:\n", train_clean["label"].value_counts().sort_index())
print("\nTest class distribution:\n", test_clean["label"].value_counts().sort_index())
```

Code cell (generate summaries — the long-running step):
```python
# Generate a real summary sentence per row for the WHOLE dataset. One-time
# cost on this CPU-only machine — budget hours, not minutes (see Step 2).
# batch_size=4 (not the default 8) to bound peak memory with bart-large-cnn.
train_clean = train_clean.copy()
test_clean = test_clean.copy()
train_clean["summary"] = summarize_texts(train_clean["text"].tolist(), batch_size=4)
test_clean["summary"] = summarize_texts(test_clean["text"].tolist(), batch_size=4)

assert train_clean["summary"].isna().sum() == 0
assert test_clean["summary"].isna().sum() == 0
assert (train_clean["summary"].str.len() > 0).all()
print("Example summary:\n ", train_clean["summary"].iloc[0])
```

Code cell (labeled/unlabeled split):
```python
labeled_df, unlabeled_df = make_splits(
    train_clean, label_fraction=config.LABEL_FRACTION, seed=config.SEED)

assert len(labeled_df) + len(unlabeled_df) == len(train_clean)
assert (unlabeled_df["label"] == -1).all()
print(f"Labeled pool: {len(labeled_df)} rows ({config.LABEL_FRACTION:.0%})")
print(f"Unlabeled pool: {len(unlabeled_df)} rows (true_label hidden for eval only)")
print("Labeled seed per-class counts:\n", labeled_df["label"].value_counts().sort_index())
```

Code cell (save):
```python
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_clean.to_parquet(config.PROCESSED_DIR / "train_clean.parquet", index=False)
test_clean.to_parquet(config.PROCESSED_DIR / "test_clean.parquet", index=False)
labeled_df.to_parquet(config.PROCESSED_DIR / "labeled.parquet", index=False)
unlabeled_df.to_parquet(config.PROCESSED_DIR / "unlabeled.parquet", index=False)
print("Saved processed splits (with generated summaries) to", config.PROCESSED_DIR)
```

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

First clear any stale AG-News-era embedding cache so nothing downstream can accidentally load mismatched vectors:
```bash
rm -rf embeddings_cache/*
```

Run as a single background process (bart-large-cnn on ~4,750 rows is the slowest cell here — budget generously):
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_transform.ipynb --ExecutePreprocessor.timeout=36000
```
Poll for `data/processed/train_clean.parquet` to appear or the process to exit. Expected: exits 0. First run also downloads `facebook/bart-large-cnn` (~1.6GB) — a one-time multi-minute cost before summarization itself starts.

- [ ] **Step 3: Verify output**

```bash
uv run python -c "
import pandas as pd
train = pd.read_parquet('data/processed/train_clean.parquet')
test = pd.read_parquet('data/processed/test_clean.parquet')
labeled = pd.read_parquet('data/processed/labeled.parquet')
unlabeled = pd.read_parquet('data/processed/unlabeled.parquet')
for name, df in [('train', train), ('test', test), ('labeled', labeled), ('unlabeled', unlabeled)]:
    assert 'summary' in df.columns, f'{name} missing summary column'
    assert df['summary'].isna().sum() == 0, f'{name} has null summaries'
print('train', len(train), 'test', len(test), 'labeled', len(labeled), 'unlabeled', len(unlabeled))
print('OK')
"
```

Expected: prints row counts and `OK`, no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/00_data_transform.ipynb data/processed/*.parquet
git commit -m "Rebuild 00_data_transform for master_data.csv: clean, split, generate summaries"
```

---

## Task 6: Notebook `01_embeddings.ipynb` — embed summaries, full data

**Files:**
- Modify: `notebooks/01_embeddings.ipynb`

**Interfaces:**
- Consumes: `data/processed/{train_clean,test_clean}.parquet` (Task 5), `utils.embeddings.*` (unchanged), `utils.config.{SAMPLE_SIZE, OPENAI_API_KEY}` (Task 1)
- Produces: cached embeddings under `embeddings_cache/{tfidf,minilm,roberta,openai}_{train,test}_full_summary.npy` — consumed by Task 7 (notebook `02`).

- [ ] **Step 1: Replace the notebook's cells, in order**

Markdown cell:
```markdown
# 01 — Embeddings

Generates and caches TF-IDF, MiniLM, RoBERTa (frozen), and OpenAI
embeddings of the **generated summary sentences** (not raw text) —
`00_data_transform.ipynb`'s `summary` column — for the unsupervised
clustering track. Full dataset, no sampling cap (`config.SAMPLE_SIZE`).
Downstream notebooks load from `embeddings_cache/` rather than
recomputing.
```

Code cell (imports):
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

Code cell (load):
```python
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")

# config.SAMPLE_SIZE is None -> full dataset for every method (master_data
# is only ~4,750 rows total, so there's no separate smaller cap for the
# CPU-heavier RoBERTa embedding like AG News needed).
train_sample = stratified_sample(train_clean, config.SAMPLE_SIZE, seed=config.SEED)
suffix = f"n{config.SAMPLE_SIZE}" if config.SAMPLE_SIZE else "full"

print(f"Embedding {len(train_sample)} train summaries ({suffix}) and {len(test_clean)} test summaries")
```

Code cell (TF-IDF):
```python
train_tfidf = get_tfidf_embeddings(train_sample["summary"].tolist(), cache_name=f"tfidf_train_{suffix}_summary")
test_tfidf = get_tfidf_embeddings(test_clean["summary"].tolist(), cache_name="tfidf_test_full_summary")
assert train_tfidf.shape[0] == len(train_sample)
assert test_tfidf.shape[0] == len(test_clean)
print("TF-IDF dims:", train_tfidf.shape[1])
```

Code cell (MiniLM):
```python
train_minilm = get_sentence_embeddings(train_sample["summary"].tolist(), cache_name=f"minilm_train_{suffix}_summary")
test_minilm = get_sentence_embeddings(test_clean["summary"].tolist(), cache_name="minilm_test_full_summary")
assert train_minilm.shape[0] == len(train_sample)
print("MiniLM dims:", train_minilm.shape[1])
```

Code cell (RoBERTa):
```python
train_roberta = get_bert_embeddings(train_sample["summary"].tolist(), cache_name=f"roberta_train_{suffix}_summary")
test_roberta = get_bert_embeddings(test_clean["summary"].tolist(), cache_name="roberta_test_full_summary")
assert train_roberta.shape[0] == len(train_sample)
print("RoBERTa dims:", train_roberta.shape[1])
```

Code cell (OpenAI, optional):
```python
if config.OPENAI_API_KEY:
    try:
        train_openai = get_openai_embeddings(train_sample["summary"].tolist(), cache_name=f"openai_train_{suffix}_summary")
        test_openai = get_openai_embeddings(test_clean["summary"].tolist(), cache_name="openai_test_full_summary")
    except Exception as e:
        print(f"OpenAI embedding failed ({type(e).__name__}): {e}")
        print("Skipping OpenAI embeddings.")
        train_openai = None
    else:
        assert train_openai.shape[0] == len(train_sample)
        print("OpenAI dims:", train_openai.shape[1])
else:
    print("OPENAI_API_KEY not set in .env — skipping OpenAI embeddings.")
```

Code cell (sanity check):
```python
for name, arr in [("tfidf", train_tfidf), ("minilm", train_minilm), ("roberta", train_roberta)]:
    assert not np.isnan(arr).any(), f"{name} embeddings contain NaNs"
print("No NaNs in any computed embedding matrix.")
```

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_embeddings.ipynb --ExecutePreprocessor.timeout=3600
```
Poll for the process to exit. Expected: exits 0 — full-data at ~4,750 rows is fast for TF-IDF/MiniLM, RoBERTa (frozen, no fine-tuning) is the slower cell but still well under an hour at this scale.

- [ ] **Step 3: Verify output**

```bash
uv run python -c "
from utils.embeddings import load_cached
for name in ['tfidf_train_full_summary', 'minilm_train_full_summary', 'roberta_train_full_summary',
             'tfidf_test_full_summary', 'minilm_test_full_summary', 'roberta_test_full_summary']:
    arr = load_cached(name)
    assert arr is not None, f'{name} missing'
    print(name, arr.shape)
print('OK')
"
```

Expected: prints all 6 array shapes and `OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_embeddings.ipynb embeddings_cache/
git commit -m "Rebuild 01_embeddings: embed generated summaries, full data, no sampling caps"
```

---

## Task 7: Notebook `02_unsupervised_clustering.ipynb` — 16-way clustering, full-output CSVs

**Files:**
- Modify: `notebooks/02_unsupervised_clustering.ipynb`

**Interfaces:**
- Consumes: `embeddings_cache/*_full_summary.npy` (Task 6), `utils.metrics.evaluate_unsupervised(..., metric_sample_size=..., seed=...)` (Task 2), `utils.samples.save_full_output` (Task 3), `utils.metrics.hungarian_match_predictions` (existing), `utils.interpretability.summarize_clusters` (existing, unchanged)
- Produces: `results/metrics_{method}_{kmeans,hdbscan}.json`, `results/clusters_{method}_{kmeans,hdbscan}.csv`, `results/full_labels_{method}_{kmeans,hdbscan}.csv` — consumed by Task 14 (notebook `07`).

- [ ] **Step 1: Replace the notebook's cells, in order**

Markdown cell:
```markdown
# 02 — Unsupervised Clustering

KMeans (k=16, matching the known class count for now — a future pass
should sweep k and pick the value maximizing clustering quality rather
than assuming 16 is optimal) and HDBSCAN over each cached **summary**
embedding (UMAP-reduced first), scored against the hidden 16-way
ground-truth labels via Hungarian-matched accuracy and standard
clustering metrics. Silhouette/Davies-Bouldin are sub-sampled
(`config.SILHOUETTE_SAMPLE_SIZE`) since they're O(n^2). Each method also
saves a full-row output CSV (`text, summary, predicted_label, true_label`
for every row), not just the qualitative cluster-inspection sample.
```

Code cell (imports):
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
from utils.interpretability import summarize_clusters
from utils.metrics import evaluate_unsupervised, hungarian_match_predictions
from utils.samples import save_full_output
```

Code cell (load):
```python
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")

train_sample = stratified_sample(train_clean, config.SAMPLE_SIZE, seed=config.SEED)
suffix = f"n{config.SAMPLE_SIZE}" if config.SAMPLE_SIZE else "full"
true_labels = train_sample["label"].to_numpy()
texts = train_sample["text"].tolist()
summaries = train_sample["summary"].tolist()

METHODS = ["tfidf", "minilm", "roberta"]
if config.OPENAI_API_KEY and load_cached(f"openai_train_{suffix}_summary") is not None:
    METHODS.append("openai")

embeddings_by_method = {}
for method in METHODS:
    arr = load_cached(f"{method}_train_{suffix}_summary")
    assert arr is not None, f"Missing cached embeddings for '{method}' — run 01_embeddings.ipynb first"
    embeddings_by_method[method] = arr

print(f"Clustering {len(train_sample)} rows across {len(embeddings_by_method)} embedding methods, k={config.NUM_CLASSES}")
```

Code cell (cluster + score):
```python
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
all_results = {}
runs = {}  # name -> (cluster_labels, emb_reduced) for the interpretability step below

for method, emb in embeddings_by_method.items():
    reducer = umap.UMAP(n_components=50, metric="cosine", random_state=config.SEED)
    emb_reduced = reducer.fit_transform(emb)

    kmeans = KMeans(n_clusters=config.NUM_CLASSES, random_state=config.SEED, n_init=10)
    km_labels = kmeans.fit_predict(emb_reduced)
    km_metrics = evaluate_unsupervised(true_labels, km_labels, emb_reduced,
                                        metric_sample_size=config.SILHOUETTE_SAMPLE_SIZE, seed=config.SEED)
    all_results[f"{method}_kmeans"] = km_metrics
    runs[f"{method}_kmeans"] = (km_labels, emb_reduced)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom")
    hdb_labels = clusterer.fit_predict(emb_reduced)
    hdb_metrics = evaluate_unsupervised(true_labels, hdb_labels, emb_reduced,
                                         metric_sample_size=config.SILHOUETTE_SAMPLE_SIZE, seed=config.SEED)
    all_results[f"{method}_hdbscan"] = hdb_metrics
    runs[f"{method}_hdbscan"] = (hdb_labels, emb_reduced)

    print(f"{method}: KMeans ACC={km_metrics['ACC (Hungarian)']:.3f} | "
          f"HDBSCAN coverage={hdb_metrics['Coverage']:.2f} ACC={hdb_metrics['ACC (Hungarian)']:.3f}")

for name, metrics in all_results.items():
    with open(config.RESULTS_DIR / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

print(f"Saved {len(all_results)} result files to {config.RESULTS_DIR}")
```

Markdown cell:
```markdown
### Cluster inspection + full-row output

The metrics above are permutation-invariant number-matching between
cluster IDs and true label IDs — they don't show *what a cluster is
about*. This prints, per method+algorithm, each cluster's size, majority
true label, purity, top KeyBERT key-phrases (from the summaries), and
example summaries nearest its centroid, then saves both the qualitative
sample and a full-row `text, summary, predicted_label, true_label` CSV
for every document (predicted label = the cluster's Hungarian-matched
class name).
```

Code cell (interpretability + full output):
```python
for name, (cluster_labels, emb_reduced) in runs.items():
    summary = summarize_clusters(summaries, cluster_labels, true_labels, emb_reduced, config.CLASS_NAMES)

    print(f"\n=== {name} ===")
    if summary.empty:
        print("(no non-noise clusters — everything was noise)")
    else:
        with pd.option_context("display.max_colwidth", 60):
            print(summary.drop(columns="example_docs").to_string(index=False))
        for _, row in summary.iterrows():
            print(f"  cluster {row['cluster']} examples:")
            for doc in row["example_docs"]:
                print(f"    - {doc}")
        summary.to_csv(config.RESULTS_DIR / f"clusters_{name}.csv", index=False)

    predicted = hungarian_match_predictions(true_labels, cluster_labels)
    save_full_output(
        texts, predicted, true_labels, config.CLASS_NAMES,
        extra_columns={"summary": summaries},
        path=config.RESULTS_DIR / f"full_labels_{name}.csv")

print(f"\nSaved per-cluster qualitative summaries and full-row outputs to {config.RESULTS_DIR}")
```

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_unsupervised_clustering.ipynb --ExecutePreprocessor.timeout=3600
```
Poll for the process to exit. Expected: exits 0.

- [ ] **Step 3: Verify output**

```bash
uv run python -c "
import json, pandas as pd
m = json.load(open('results/metrics_minilm_kmeans.json'))
print(m)
assert 'ACC (Hungarian)' in m
full = pd.read_csv('results/full_labels_minilm_kmeans.csv')
assert set(['text','summary','predicted_label','true_label','correct']) <= set(full.columns)
print(len(full), 'full-output rows')
print('OK')
"
```

Expected: prints the MiniLM+KMeans metrics dict, full-output row count, and `OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_unsupervised_clustering.ipynb results/metrics_*.json results/clusters_*.csv results/full_labels_*.csv
git commit -m "Rebuild 02_unsupervised_clustering for 16-way summary-embedding clustering, full-output CSVs"
```

---

## Task 8: Notebook `03_bertopic.ipynb` — 16-way BERTopic on summaries

**Files:**
- Modify: `notebooks/03_bertopic.ipynb`

**Interfaces:**
- Consumes: `data/processed/train_clean.parquet` (Task 5), `utils.metrics.{evaluate_unsupervised, hungarian_match_predictions}` (Task 2), `utils.samples.save_full_output` (Task 3)
- Produces: `results/metrics_bertopic.json`, `results/clusters_bertopic.csv`, `results/full_labels_bertopic.csv` — consumed by Task 14.

- [ ] **Step 1: Replace the notebook's cells, in order**

Markdown cell:
```markdown
# 03 — BERTopic

BERTopic owns its own embedding (MiniLM) and clustering (UMAP + HDBSCAN
internally), so it's kept separate from notebook 02. Runs on the
**generated summary sentences**, forced to 16 topics
(`config.NUM_CLASSES`), with `KeyBERTInspired` topic representations for
readable labels.
```

Code cell (imports — unchanged from before):
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
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from utils import config
from utils.data import stratified_sample
from utils.interpretability import summarize_clusters
from utils.metrics import evaluate_unsupervised, hungarian_match_predictions
from utils.samples import save_full_output
```

Code cell (load — summaries, not text):
```python
train_clean = pd.read_parquet(config.PROCESSED_DIR / "train_clean.parquet")
train_sample = stratified_sample(train_clean, config.SAMPLE_SIZE, seed=config.SEED)
texts = train_sample["text"].tolist()
summaries = train_sample["summary"].tolist()
true_labels = train_sample["label"].to_numpy()
```

Code cell (fit BERTopic on summaries):
```python
umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=config.SEED)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
representation_model = KeyBERTInspired()
topic_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model,
                        vectorizer_model=vectorizer_model,
                        representation_model=representation_model,
                        nr_topics=config.NUM_CLASSES, calculate_probabilities=False)

topics, _ = topic_model.fit_transform(summaries)
print(topic_model.get_topic_info())
```

Code cell (score):
```python
topics_arr = np.array(topics)
topic_embeddings = sentence_model.encode(summaries, show_progress_bar=True, convert_to_numpy=True)
bertopic_metrics = evaluate_unsupervised(true_labels, topics_arr, topic_embeddings,
                                          metric_sample_size=config.SILHOUETTE_SAMPLE_SIZE, seed=config.SEED)
print(bertopic_metrics)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_bertopic.json", "w") as f:
    json.dump(bertopic_metrics, f, indent=2)
print("Saved BERTopic metrics.")
```

Markdown cell (unchanged in spirit):
```markdown
### Topic inspection + full-row output

Same purity/majority-label crosstab and example-summary view used in
`02_unsupervised_clustering.ipynb`, plus a full-row
`text, summary, predicted_label, true_label` CSV over every document.
```

Code cell (interpretability + full output):
```python
summary = summarize_clusters(summaries, topics_arr, true_labels, topic_embeddings, config.CLASS_NAMES)

if summary.empty:
    print("(no non-noise topics — everything was noise)")
else:
    with pd.option_context("display.max_colwidth", 60):
        print(summary.drop(columns="example_docs").to_string(index=False))
    for _, row in summary.iterrows():
        print(f"  topic {row['cluster']} examples:")
        for doc in row["example_docs"]:
            print(f"    - {doc}")
    summary.to_csv(config.RESULTS_DIR / "clusters_bertopic.csv", index=False)
    print(f"\nSaved per-topic qualitative summary to {config.RESULTS_DIR / 'clusters_bertopic.csv'}")

predicted = hungarian_match_predictions(true_labels, topics_arr)
save_full_output(
    texts, predicted, true_labels, config.CLASS_NAMES,
    extra_columns={"summary": summaries},
    path=config.RESULTS_DIR / "full_labels_bertopic.csv")
print("Saved full-row output for bertopic.")
```

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_bertopic.ipynb --ExecutePreprocessor.timeout=1800
```
Poll for the process to exit. Expected: exits 0.

- [ ] **Step 3: Verify output**

```bash
uv run python -c "
import json, pandas as pd
m = json.load(open('results/metrics_bertopic.json'))
assert 'ACC (Hungarian)' in m
full = pd.read_csv('results/full_labels_bertopic.csv')
assert set(['text','summary','predicted_label','true_label']) <= set(full.columns)
print(m)
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/03_bertopic.ipynb results/metrics_bertopic.json results/clusters_bertopic.csv results/full_labels_bertopic.csv
git commit -m "Rebuild 03_bertopic for 16-way topics on generated summaries, full-output CSV"
```

---

## Task 9: Notebook `04_weak_supervision.ipynb` — auto-derived labeling functions

**Files:**
- Modify: `notebooks/04_weak_supervision.ipynb`

**Interfaces:**
- Consumes: `data/processed/{labeled,unlabeled}.parquet` (Task 5), `utils.weak_supervision.{derive_class_keywords, build_keyword_lfs}` (Task 4), `utils.samples.save_full_output` (Task 3)
- Produces: `results/metrics_weak_supervision.json`, `results/sample_labels_weak_supervision.csv`, `results/full_labels_weak_supervision.csv` — consumed by Task 14.

- [ ] **Step 1: Replace the notebook's cells, in order**

Markdown cell:
```markdown
# 04 — Weak Supervision (Snorkel)

Per class, the most TF-IDF-distinctive keywords are auto-derived from that
class's 5% labeled seed (`utils.weak_supervision.derive_class_keywords`) —
a programmatic replacement for hand-written rules, needed because
hand-authoring 16 classes' worth of keyword lists (several overlapping in
subject matter) doesn't scale the way it did for AG News's 4 classes.
Each class's keyword list becomes one labeling function
(`build_keyword_lfs`); Snorkel's `LabelModel` combines all 16 LFs' votes
into probabilistic labels, trained on raw text, no fine-tuning.
```

Code cell (imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json

import pandas as pd
from snorkel.labeling import LFAnalysis, PandasLFApplier
from snorkel.labeling.model import LabelModel

from utils import config
from utils.data import stratified_sample
from utils.metrics import evaluate_label_quality
from utils.samples import save_full_output
from utils.weak_supervision import derive_class_keywords, build_keyword_lfs

ABSTAIN = -1
```

Code cell (derive LFs from the seed):
```python
labeled_df = pd.read_parquet(config.PROCESSED_DIR / "labeled.parquet")
unlabeled_df = pd.read_parquet(config.PROCESSED_DIR / "unlabeled.parquet")
unlabeled_sample = stratified_sample(unlabeled_df, config.SAMPLE_SIZE, seed=config.SEED, label_col="true_label")

keywords_by_class = derive_class_keywords(labeled_df, config.CLASS_NAMES, top_n=15)
lfs = build_keyword_lfs(keywords_by_class)
print(f"Built {len(lfs)} auto-derived labeling functions from the {len(labeled_df)}-row seed")
for label, keywords in keywords_by_class.items():
    print(f"  {config.CLASS_NAMES[label]}: {keywords[:8]}")
```

Code cell (apply + coverage check):
```python
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=unlabeled_sample)
L_dev = applier.apply(df=labeled_df)

print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
print(LFAnalysis(L=L_dev, lfs=lfs).lf_summary(Y=labeled_df["label"].to_numpy()))

overall_coverage = (L_train != ABSTAIN).any(axis=1).mean()
# Lower sanity bound than AG News's 4-class version (0.05) — 16
# auto-derived, narrower per-class keyword lists over overlapping
# subject-matter classes are expected to abstain more often.
assert overall_coverage > 0.02, f"LF coverage suspiciously low: {overall_coverage:.2%}"
print(f"Overall LF coverage on unlabeled sample: {overall_coverage:.2%}")
```

Code cell (fit LabelModel):
```python
label_model = LabelModel(cardinality=config.NUM_CLASSES, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr=0.001, seed=config.SEED)

proba_labels = label_model.predict_proba(L=L_train)
hard_labels = label_model.predict(L=L_train)
confidence = proba_labels.max(axis=1)
```

Code cell (evaluate + save metrics):
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

Code cell (sample + full output):
```python
from utils.samples import save_label_samples

save_label_samples(
    unlabeled_sample["text"], hard_labels, unlabeled_sample["true_label"].to_numpy(),
    config.CLASS_NAMES, confidence=confidence, n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_weak_supervision.csv")
print("Saved sample generated labels for weak_supervision.")

save_full_output(
    unlabeled_sample["text"], hard_labels, unlabeled_sample["true_label"].to_numpy(),
    config.CLASS_NAMES, confidence=confidence,
    extra_columns={"summary": unlabeled_sample["summary"].tolist()},
    path=config.RESULTS_DIR / "full_labels_weak_supervision.csv")
print("Saved full-row output for weak_supervision.")
```

Delete the old "Plan B" markdown + commented-out fallback cell at the end (it referenced the old 4-class `ABSTAIN`/hand-written-LF setup; the auto-derived LFs above are the only path now — if Snorkel fails to install, re-derive a plain-pandas weighted-vote fallback from `keywords_by_class`/`L_train` at that point rather than keeping stale commented code).

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_weak_supervision.ipynb --ExecutePreprocessor.timeout=1800
```
Poll for the process to exit. Expected: exits 0. If the `overall_coverage > 0.02` assertion fails, raise `top_n` in `derive_class_keywords` (e.g. to 25) and re-run before proceeding — don't lower the assertion further without understanding why coverage is that low.

- [ ] **Step 3: Verify output**

```bash
cat results/metrics_weak_supervision.json
head -3 results/full_labels_weak_supervision.csv
```

Expected: valid JSON with `Label Accuracy`, `Label Macro F1`, `Coverage`; full-output CSV has a header including `text,predicted_label,true_label,correct,confidence,summary`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_weak_supervision.ipynb results/metrics_weak_supervision.json results/sample_labels_weak_supervision.csv results/full_labels_weak_supervision.csv
git commit -m "Rebuild 04_weak_supervision with auto-derived per-class LFs for 16 classes"
```

---

## Task 10: Notebooks `05_pseudo_labeling.ipynb` & `05b_pseudo_labeling_electra.ipynb` — full data, 16-way

**Files:**
- Modify: `notebooks/05_pseudo_labeling.ipynb`
- Modify: `notebooks/05b_pseudo_labeling_electra.ipynb`

**Interfaces:**
- Consumes: `data/processed/{labeled,unlabeled,test_clean}.parquet` (Task 5), `utils.modeling.{pseudo_label_loop, get_predictions}` (unchanged), `utils.samples.save_full_output` (Task 3)
- Produces: `results/metrics_pseudo_labeling{,_electra}.json`, `results/confusion_matrix_pseudo_labeling{,_electra}.png`, `results/sample_labels_pseudo_labeling{,_electra}_{train_pool,test}.csv`, `results/full_labels_pseudo_labeling{,_electra}_{train_pool,test}.csv` — consumed by Task 14.

- [ ] **Step 1: Replace `05_pseudo_labeling.ipynb`'s cells, in order**

Markdown cell:
```markdown
# 05 — Pseudo-Labeling (Self-Training) — DistilBERT

Iteratively fine-tunes `utils.config.CLASSIFIER_MODEL_NAME` (DistilBERT) on
the 5%-per-class labeled seed (~15/class on master_data.csv, much thinner
than AG News's ~1,500/class), predicts on the full unlabeled pool, and
absorbs high-confidence predictions each round, targeting 98% coverage.
Starts from the AG-News-tuned confidence threshold (0.80) — at this much
thinner 16-way seed the round-0 behavior may differ; if round 0 stalls (0
labels absorbed), Step 2 below re-runs at a lower threshold, following the
same tuning methodology documented previously.
```

Code cell (imports):
```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import json

import pandas as pd

from utils import config
from utils.metrics import evaluate_label_quality, evaluate_semisupervised
from utils.modeling import get_predictions, pseudo_label_loop
from utils.samples import save_full_output, save_label_samples
```

Code cell (load — full pool, no sample cap):
```python
labeled_df = pd.read_parquet(config.PROCESSED_DIR / "labeled.parquet")
unlabeled_df = pd.read_parquet(config.PROCESSED_DIR / "unlabeled.parquet")
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")

overlap = set(unlabeled_df["text"]) & set(test_clean["text"])
assert len(overlap) == 0, f"{len(overlap)} rows leaked between train pool and test set"
print(f"Labeled seed: {len(labeled_df)} | Unlabeled pool: {len(unlabeled_df)} | Test: {len(test_clean)}")
```

Code cell (run self-training loop — CONFIDENCE_THRESHOLD as a named constant so Step 2 can retune it):
```python
CONFIDENCE_THRESHOLD = 0.80  # starting point from the AG News tuning story; retuned in Step 2 if round 0 stalls

final_model, final_tokenizer, current_labeled, history = pseudo_label_loop(
    labeled_df, unlabeled_df,
    model_name=config.CLASSIFIER_MODEL_NAME,
    confidence_threshold=CONFIDENCE_THRESHOLD, epochs=3,
    target_coverage=0.98, max_iterations=10)

for h in history:
    print(h)
```

Code cell (pseudo-label quality + full pool output, including unresolved rows):
```python
pseudo_only = current_labeled.iloc[len(labeled_df):]
merged = pseudo_only.merge(unlabeled_df[["text", "true_label", "summary"]], on="text", how="left")
unresolved = unlabeled_df[~unlabeled_df["text"].isin(pseudo_only["text"])]

label_quality = evaluate_label_quality(
    true_labels=merged["true_label"].to_numpy(),
    pseudo_labels=merged["label"].to_numpy())
print("Pseudo-label quality:", label_quality)

save_label_samples(
    merged["text"], merged["label"].to_numpy(), merged["true_label"].to_numpy(),
    config.CLASS_NAMES, n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_pseudo_labeling_train_pool.csv")

full_pool_texts = pd.concat([merged["text"], unresolved["text"]], ignore_index=True)
full_pool_predicted = pd.concat(
    [merged["label"], pd.Series(-1, index=unresolved.index)], ignore_index=True)
full_pool_true = pd.concat([merged["true_label"], unresolved["true_label"]], ignore_index=True)
full_pool_summary = pd.concat([merged["summary"], unresolved["summary"]], ignore_index=True)

save_full_output(
    full_pool_texts, full_pool_predicted.to_numpy(), full_pool_true.to_numpy(), config.CLASS_NAMES,
    extra_columns={"summary": full_pool_summary.tolist()},
    path=config.RESULTS_DIR / "full_labels_pseudo_labeling_train_pool.csv")
print("Saved sample + full-row train-pool outputs for pseudo_labeling.")
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
    json.dump({"test_metrics": semisup_results, "label_quality": label_quality,
               "history": history, "confidence_threshold": CONFIDENCE_THRESHOLD}, f, indent=2)
print("Saved pseudo-labeling results.")

save_label_samples(
    test_clean["text"], test_preds, test_clean["label"].to_numpy(),
    config.CLASS_NAMES, confidence=test_probs.max(axis=1), n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_pseudo_labeling_test.csv")
save_full_output(
    test_clean["text"], test_preds, test_clean["label"].to_numpy(), config.CLASS_NAMES,
    confidence=test_probs.max(axis=1), extra_columns={"summary": test_clean["summary"].tolist()},
    path=config.RESULTS_DIR / "full_labels_pseudo_labeling_test.csv")
print("Saved sample + full-row test outputs for pseudo_labeling.")
```

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute `05_pseudo_labeling.ipynb`, retuning if round 0 stalls**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_pseudo_labeling.ipynb --ExecutePreprocessor.timeout=7200
```
Poll for `results/metrics_pseudo_labeling.json` or process exit. Read the printed `history` list (from the executed notebook's outputs, e.g. via `jupyter nbconvert --to script --stdout` or opening the saved `.ipynb`'s cell outputs): if iteration 0 shows `new_labels: 0` (the loop stalled immediately, mirroring the AG News "Run 1" story), edit the `CONFIDENCE_THRESHOLD` cell to `0.50`, re-run the same command, and repeat this check. Once a run absorbs a nonzero number of pseudo-labels in at least one round, that run's result stands — note the threshold actually used.

- [ ] **Step 3: Verify output**

```bash
cat results/metrics_pseudo_labeling.json
head -3 results/full_labels_pseudo_labeling_test.csv
```

Expected: valid JSON with `test_metrics`, `label_quality`, `history`, `confidence_threshold`; full-output CSV has `text,predicted_label,true_label,correct,confidence,summary`.

- [ ] **Step 4: Commit `05_pseudo_labeling.ipynb`**

```bash
git add notebooks/05_pseudo_labeling.ipynb results/metrics_pseudo_labeling.json results/confusion_matrix_pseudo_labeling.png results/sample_labels_pseudo_labeling_*.csv results/full_labels_pseudo_labeling_*.csv
git commit -m "Rebuild 05_pseudo_labeling (DistilBERT) for full-data 16-way seed, full-output CSVs"
```

- [ ] **Step 5: Apply the identical rewrite to `05b_pseudo_labeling_electra.ipynb`**

Same cells as Steps 1-3 above, with these differences:
- Markdown cell's title/model name says "ELECTRA-small" (`config.CLASSIFIER_MODEL_NAME_ALT`) instead of DistilBERT.
- The self-training-loop cell passes `model_name=config.CLASSIFIER_MODEL_NAME_ALT` instead of `config.CLASSIFIER_MODEL_NAME`.
- **Use `05_pseudo_labeling.ipynb`'s already-tuned `CONFIDENCE_THRESHOLD` value from Step 2 as ELECTRA-small's starting point too** (not necessarily 0.80 — whatever DistilBERT's run actually converged on), since both notebooks should stay comparable at the same threshold where possible; still retune per Step 2's stall-check if ELECTRA-small's own round 0 stalls at that value.
- All file paths gain an `_electra` suffix: `metrics_pseudo_labeling_electra.json`, `confusion_matrix_pseudo_labeling_electra.png`, `sample_labels_pseudo_labeling_electra_{train_pool,test}.csv`, `full_labels_pseudo_labeling_electra_{train_pool,test}.csv`.

Execute:
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05b_pseudo_labeling_electra.ipynb --ExecutePreprocessor.timeout=7200
```
Apply the same stall-check/retune loop as Step 2 if needed.

- [ ] **Step 6: Verify output**

```bash
cat results/metrics_pseudo_labeling_electra.json
head -3 results/full_labels_pseudo_labeling_electra_test.csv
```

- [ ] **Step 7: Commit `05b_pseudo_labeling_electra.ipynb`**

```bash
git add notebooks/05b_pseudo_labeling_electra.ipynb results/metrics_pseudo_labeling_electra.json results/confusion_matrix_pseudo_labeling_electra.png results/sample_labels_pseudo_labeling_electra_*.csv results/full_labels_pseudo_labeling_electra_*.csv
git commit -m "Rebuild 05b_pseudo_labeling_electra (ELECTRA-small) for full-data 16-way seed, full-output CSVs"
```

---

## Task 11: Notebooks `06_full_supervised_baseline.ipynb` & `06b_full_supervised_baseline_electra.ipynb` — full data, 16-way

**Files:**
- Modify: `notebooks/06_full_supervised_baseline.ipynb`
- Modify: `notebooks/06b_full_supervised_baseline_electra.ipynb`

**Interfaces:**
- Consumes: `data/processed/{train_clean,test_clean}.parquet` (Task 5), `utils.modeling.{train_model, get_predictions}` (unchanged), `utils.samples.save_full_output` (Task 3)
- Produces: `results/metrics_full_supervised{,_electra}.json`, `results/confusion_matrix_full_supervised{,_electra}.png`, `results/sample_labels_full_supervised{,_electra}.csv`, `results/full_labels_full_supervised{,_electra}.csv` — consumed by Task 14.

- [ ] **Step 1: Replace `06_full_supervised_baseline.ipynb`'s cells, in order**

Markdown cell:
```markdown
# 06 — Full Supervised Baseline — DistilBERT

Trains `utils.config.CLASSIFIER_MODEL_NAME` on 100% of the train split
(full data, `config.CLASSIFIER_SAMPLE_SIZE=None`) — the upper-bound
reference every other method is compared against, 16-way.
```

Code cell (imports):
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
from utils.samples import save_full_output, save_label_samples
```

Code cell (load — full data):
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

Code cell (sample + full output):
```python
save_label_samples(
    test_clean["text"], test_preds, test_clean["label"].to_numpy(),
    config.CLASS_NAMES, confidence=test_probs.max(axis=1), n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_full_supervised.csv")
save_full_output(
    test_clean["text"], test_preds, test_clean["label"].to_numpy(), config.CLASS_NAMES,
    confidence=test_probs.max(axis=1), extra_columns={"summary": test_clean["summary"].tolist()},
    path=config.RESULTS_DIR / "full_labels_full_supervised.csv")
print("Saved sample + full-row outputs for full_supervised.")
```

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_full_supervised_baseline.ipynb --ExecutePreprocessor.timeout=3600
```
Poll for the process to exit. Expected: exits 0 — dominated by the full ~950-row test-set inference pass plus fine-tuning on ~3,800 train rows, a few-minutes-to-tens-of-minutes job at this scale.

- [ ] **Step 3: Verify output**

```bash
cat results/metrics_full_supervised.json
head -3 results/full_labels_full_supervised.csv
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_full_supervised_baseline.ipynb results/metrics_full_supervised.json results/confusion_matrix_full_supervised.png results/sample_labels_full_supervised.csv results/full_labels_full_supervised.csv
git commit -m "Rebuild 06_full_supervised_baseline (DistilBERT) for full-data 16-way"
```

- [ ] **Step 5: Apply the identical rewrite to `06b_full_supervised_baseline_electra.ipynb`**

Same cells as Step 1, with: markdown says "ELECTRA-small"; `train_model(train_sample, model_name=config.CLASSIFIER_MODEL_NAME_ALT, epochs=3)`; all output paths gain `_electra`: `metrics_full_supervised_electra.json`, `confusion_matrix_full_supervised_electra.png`, `sample_labels_full_supervised_electra.csv`, `full_labels_full_supervised_electra.csv`.

Execute:
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06b_full_supervised_baseline_electra.ipynb --ExecutePreprocessor.timeout=3600
```

- [ ] **Step 6: Verify output**

```bash
cat results/metrics_full_supervised_electra.json
head -3 results/full_labels_full_supervised_electra.csv
```

- [ ] **Step 7: Commit**

```bash
git add notebooks/06b_full_supervised_baseline_electra.ipynb results/metrics_full_supervised_electra.json results/confusion_matrix_full_supervised_electra.png results/sample_labels_full_supervised_electra.csv results/full_labels_full_supervised_electra.csv
git commit -m "Rebuild 06b_full_supervised_baseline_electra (ELECTRA-small) for full-data 16-way"
```

---

## Task 12: Notebook `08_label_propagation.ipynb` — full pool, 16-way

**Files:**
- Modify: `notebooks/08_label_propagation.ipynb`

**Interfaces:**
- Consumes: `data/processed/{labeled,unlabeled,test_clean}.parquet` (Task 5), `utils.embeddings.get_sentence_embeddings`, `utils.label_propagation.run_label_propagation` (both unchanged), `utils.samples.save_full_output` (Task 3)
- Produces: `results/metrics_label_propagation.json`, `results/sample_labels_label_propagation.csv`, `results/full_labels_label_propagation.csv` — consumed by Task 14.

- [ ] **Step 1: Replace the notebook's cells, in order**

Markdown cell (unchanged in spirit, updated for full pool):
```markdown
# 08 — Label Propagation

No fine-tuning loop: embed the 5%-per-class labeled seed and the full
unlabeled pool with MiniLM (on raw text), build a k-NN graph over all of
them, and propagate the seed labels through the graph via
`sklearn.semi_supervised.LabelSpreading`. Full unlabeled pool, no sampling
cap (`config.SAMPLE_SIZE`).
```

Code cell (imports — unchanged):
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
from utils.embeddings import get_sentence_embeddings
from utils.label_propagation import run_label_propagation
from utils.metrics import evaluate_label_quality
from utils.samples import save_full_output, save_label_samples
```

Code cell (load — full pool):
```python
labeled_df = pd.read_parquet(config.PROCESSED_DIR / "labeled.parquet")
unlabeled_df = pd.read_parquet(config.PROCESSED_DIR / "unlabeled.parquet")
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")

unlabeled_sample = stratified_sample(unlabeled_df, config.SAMPLE_SIZE, seed=config.SEED, label_col="true_label")

overlap = set(unlabeled_sample["text"]) & set(test_clean["text"])
assert len(overlap) == 0, f"{len(overlap)} rows leaked between train pool and test set"
print(f"Labeled seed: {len(labeled_df)} | Unlabeled pool: {len(unlabeled_sample)} | Test: {len(test_clean)}")
```

Code cell (embed):
```python
cache_name = f"minilm_labelprop_L{len(labeled_df)}_U{len(unlabeled_sample)}"
combined_texts = labeled_df["text"].tolist() + unlabeled_sample["text"].tolist()
combined_embeddings = get_sentence_embeddings(combined_texts, cache_name)

labeled_embeddings = combined_embeddings[:len(labeled_df)]
unlabeled_embeddings = combined_embeddings[len(labeled_df):]
assert unlabeled_embeddings.shape[0] == len(unlabeled_sample)
print(f"Embedded {len(combined_texts)} texts (MiniLM, {combined_embeddings.shape[1]}-d)")
```

Code cell (propagate + evaluate + save):
```python
predicted_labels, confidence = run_label_propagation(
    labeled_embeddings, labeled_df["label"].to_numpy(), unlabeled_embeddings,
    kernel="knn", n_neighbors=7)

label_quality = evaluate_label_quality(
    true_labels=unlabeled_sample["true_label"].to_numpy(),
    pseudo_labels=predicted_labels,
    confidence_scores=confidence)
print("Label propagation quality:", label_quality)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_label_propagation.json", "w") as f:
    json.dump(label_quality, f, indent=2)
print("Saved label propagation results.")
```

Code cell (sample + full output):
```python
save_label_samples(
    unlabeled_sample["text"], predicted_labels, unlabeled_sample["true_label"].to_numpy(),
    config.CLASS_NAMES, confidence=confidence, n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_label_propagation.csv")
save_full_output(
    unlabeled_sample["text"], predicted_labels, unlabeled_sample["true_label"].to_numpy(),
    config.CLASS_NAMES, confidence=confidence,
    extra_columns={"summary": unlabeled_sample["summary"].tolist()},
    path=config.RESULTS_DIR / "full_labels_label_propagation.csv")
print("Saved sample + full-row outputs for label_propagation.")
```

Use the `NotebookEdit` tool to replace the existing cells with these, in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/08_label_propagation.ipynb --ExecutePreprocessor.timeout=1800
```
Poll for the process to exit. Expected: exits 0.

- [ ] **Step 3: Verify output**

```bash
cat results/metrics_label_propagation.json
head -3 results/full_labels_label_propagation.csv
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/08_label_propagation.ipynb results/metrics_label_propagation.json results/sample_labels_label_propagation.csv results/full_labels_label_propagation.csv
git commit -m "Rebuild 08_label_propagation for full unlabeled pool, 16-way, full-output CSV"
```

---

## Task 13: Retire `09_summarization_labeling.ipynb`

**Files:**
- Delete: `notebooks/09_summarization_labeling.ipynb`
- Delete: `results/metrics_summarization.json`, `results/confusion_matrix_summarization.png`, `results/sample_labels_summarization.csv` (if present)

**Interfaces:** none — this notebook's role (summary-based auto-labeling) is subsumed by the new summary-embedding-clustering flow (Tasks 6-8). Its stale AG-News-era result files would otherwise still be picked up by `07_comparison.ipynb`'s glob in Task 14.

- [ ] **Step 1: Remove the notebook and its stale results**

```bash
git rm notebooks/09_summarization_labeling.ipynb
git rm -f results/metrics_summarization.json results/confusion_matrix_summarization.png results/sample_labels_summarization.csv
```

- [ ] **Step 2: Verify**

```bash
uv run python -c "
from pathlib import Path
assert not Path('notebooks/09_summarization_labeling.ipynb').exists()
assert not Path('results/metrics_summarization.json').exists()
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git commit -m "Retire 09_summarization_labeling: superseded by summary-embedding clustering (02/03)"
```

---

## Task 14: Re-run `07_comparison.ipynb`

**Files:**
- None modified — `07_comparison.ipynb` already globs `results/metrics_*.json` and needs no code changes for the new class count/method set. This task is execution-only, run last (after Tasks 5-13 have all produced their `metrics_*.json` files).

**Interfaces:**
- Consumes: every `results/metrics_*.json` file left after Task 13's deletions
- Produces: refreshed `results/comparison_table.csv`, `results/comparison_bar_chart.png` — consumed by Task 15.

- [ ] **Step 1: Execute**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_comparison.ipynb --ExecutePreprocessor.timeout=300
```
Expected: exits 0 quickly (pure aggregation, no heavy compute).

- [ ] **Step 2: Verify**

```bash
cat results/comparison_table.csv
```

Expected: one row per method — `tfidf_kmeans, tfidf_hdbscan, minilm_kmeans, minilm_hdbscan, roberta_kmeans, roberta_hdbscan, bertopic, weak_supervision, label_propagation, pseudo_labeling, pseudo_labeling_electra, full_supervised, full_supervised_electra` (plus `openai_kmeans`/`openai_hdbscan` as `pending` if no API key was configured) — no `summarization` row (retired in Task 13), no AG-News-era rows.

- [ ] **Step 3: Commit**

```bash
git add results/comparison_table.csv results/comparison_bar_chart.png
git commit -m "Re-run 07_comparison for the master_data.csv 16-way pipeline"
```

---

## Task 15: Archive the old summary, write a brand-new `PROJECT_SUMMARY.md` with flow diagrams and per-loop results

**Files:**
- Create: `docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md` (the old AG News summary, moved not deleted)
- Create (fresh, not a patch): `PROJECT_SUMMARY.md`
- Modify: `docs/semi_supervised_methods.md`

**Interfaces:**
- Consumes: `results/comparison_table.csv` and every `results/metrics_*.json`/`results/full_labels_*.csv`/`results/sample_labels_*.csv` produced by Tasks 5-14, plus each pseudo-labeling notebook's saved `history` list (Task 10) for the per-round tables.
- Produces: no new interfaces — the documentation-facing task every prior implementation effort in this repo's history ends with. Old `PROJECT_SUMMARY.md` content is preserved at its new path, not lost.

This task's exact numbers depend on Tasks 5-14's actual results, not known until they've run — read the generated files first, then transcribe real values. Do not fabricate numbers. Run this task **only after Task 14 has completed** (every notebook re-run, `07_comparison.ipynb` refreshed).

- [ ] **Step 1: Archive the current `PROJECT_SUMMARY.md` (AG News) — don't delete it**

```bash
git mv PROJECT_SUMMARY.md docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md
```

Add one line at the very top of the moved file (above its `# Auto-Labeling...` heading):
```markdown
> **Archived 2026-09-05.** This is the AG News (4-class) version of this
> project, kept for historical reference. The current project uses
> `data/master_data.csv` (16 classes) — see `../PROJECT_SUMMARY.md`.
```

- [ ] **Step 2: Read the generated results**

```bash
cat results/comparison_table.csv
for f in results/metrics_*.json; do echo "== $f =="; cat "$f"; done
head -10 results/full_labels_minilm_kmeans.csv
```

Also open each pseudo-labeling metrics file's `history` list (`results/metrics_pseudo_labeling.json` and `results/metrics_pseudo_labeling_electra.json`) — this is the per-round data for Step 4's tables below.

- [ ] **Step 3: Write the new `PROJECT_SUMMARY.md` from scratch**

A fresh document (not an edited copy of the archived one), covering:

- **Goal**: 16-category `master_data.csv` auto-labeling, same "no full human annotation" framing as before.
- **Dataset**: `data/master_data.csv` provenance (`data/raw/` sources), 16 classes, cleaning steps (dedup, degenerate-row drop, dropped patchy source `summary` column), stratified 80/20 split, full-data-everywhere note (every method runs on the whole dataset, no sampling caps — replacing the old AG News sampling-cap table entirely).
- **Pipeline / data flow diagrams** (Mermaid — GitHub/most Markdown viewers render ` ```mermaid ` fences natively):
  - One overall flowchart showing the whole pipeline branching into three tracks from the same cleaned+split data, e.g.:
    ```mermaid
    flowchart TD
        A[master_data.csv, 16 classes] --> B[clean + dedup + stratified 80/20 split]
        B --> C[train_clean / test_clean]
        C --> D[generate summary sentence per row\nfacebook/bart-large-cnn]
        D --> E[Unsupervised track:\nembed summaries -> cluster k=16]
        C --> F[5% labeled seed + 95% unlabeled pool]
        F --> G[Semi-supervised track:\nweak supervision / label propagation / pseudo-labeling\ntrained on raw text]
        C --> H[Full-supervised baseline:\n100% labels, raw text]
        E --> I[results/comparison_table.csv]
        G --> I
        H --> I
    ```
  - One flowchart per specific approach that has internal structure worth drawing — at minimum the **pseudo-labeling self-training loop** (both DistilBERT and ELECTRA-small follow this same loop shape):
    ```mermaid
    flowchart TD
        S[5% labeled seed] --> T[fine-tune classifier]
        T --> P[predict on remaining unlabeled pool]
        P --> C{confidence >= threshold?}
        C -->|yes| ABS[absorb as pseudo-label]
        C -->|no| LEFT[leave unlabeled, try next round]
        ABS --> COV{coverage >= 98% OR\nno unlabeled left OR\n0 new this round OR\nmax_iterations?}
        LEFT --> COV
        COV -->|no| T
        COV -->|yes| FINAL[final fine-tune on grown pool]
        FINAL --> EVAL[evaluate on held-out test set]
    ```
  - Optionally, a small diagram for the unsupervised summary→cluster→name flow (raw text -> summary sentence -> embedding -> KMeans/HDBSCAN/BERTopic -> Hungarian-match to class name), if it adds clarity beyond the overall flowchart.
- **Approaches**: unsupervised (TF-IDF/MiniLM/RoBERTa/OpenAI × KMeans(k=16)/HDBSCAN + BERTopic, all on summaries) and semi-supervised (auto-derived weak supervision, label propagation, pseudo-labeling × {DistilBERT, ELECTRA-small}) sections, written fresh for the new methods/scale — don't just copy the old AG News wording.
- **Results**: full new tables with real 16-way numbers from Step 2.
- **Per-loop results for iterative approaches**: for pseudo-labeling, a round-by-round table for **each** model (DistilBERT and ELECTRA-small), in the style of the old tuning story, built from that run's actual `history` list — columns: round, new pseudo-labels absorbed, cumulative coverage. State whatever `CONFIDENCE_THRESHOLD` value each model's notebook actually converged on (Task 10), and if retuning was needed (round-0 stall), narrate that the same way the old confidence-threshold-tuning story did.
- **Sample generated labels**: refresh the per-method example tables using real rows pulled from `results/sample_labels_*.csv` / `results/full_labels_*.csv`.
- **Full-output CSV convention**: document that every method saves both a small qualitative sample and a full-row `results/full_labels_<method>.csv` (`text, summary, predicted_label, true_label[, confidence]`), and where to find them.
- **Limitations / Next steps**: k=16-now/k-sweep-later (cluster count should be swept and chosen by best metric, not assumed from the known class count); note the deferred semantic-closeness scoring item only if still relevant to this pipeline.
- **Repo map**: updated notebook list (09 removed), `utils/` line (add `weak_supervision`, note `summarization` trimmed to summarize/title generation only, no zero-shot pieces).
- A pointer at the top or bottom to `docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md` for the retired AG News version.

- [ ] **Step 4: Update `docs/semi_supervised_methods.md`**

Add a short new section documenting: (a) the dataset swap to master_data.csv (16 classes); (b) weak supervision's labeling functions are now auto-derived from the seed rather than hand-written, with the reasoning (16 overlapping classes don't scale to hand-authoring); (c) pointer to the new `PROJECT_SUMMARY.md`'s results section for the real numbers.

- [ ] **Step 5: Commit**

```bash
git add PROJECT_SUMMARY.md docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md docs/semi_supervised_methods.md
git commit -m "Archive AG News summary; write new PROJECT_SUMMARY.md with flow diagrams and per-loop pseudo-labeling results"
```
