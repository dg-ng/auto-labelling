# Auto-Labeling Project: Unsupervised & Semi-Supervised NLP
## AG News Classification Dataset

---

## 📌 Project Overview

This project explores **automatic label generation** for text data without (or with minimal) human annotation, using the AG News dataset as a benchmark. We implement and compare:

- **Unsupervised auto-labeling** — no labels used during training
- **Semi-supervised auto-labeling** — a small fraction of labels used during training

The AG News dataset has 4 ground-truth classes (World, Sports, Business, Sci/Tech), which we hide during training and use only for evaluation.

---

## 📂 Dataset: AG News Classification

| Property | Value |
|---|---|
| Classes | 4 (World, Sports, Business, Sci/Tech) |
| Train samples | 120,000 (30,000 per class) |
| Test samples | 7,600 (1,900 per class) |
| Input columns | `class_index` (1–4), `title`, `description` |
| Label file | `classes.txt` |

### Class Mapping
```
1 → World
2 → Sports
3 → Business
4 → Sci/Tech
```

### Data Loading
```python
import pandas as pd

train_df = pd.read_csv("train.csv", header=None,
                        names=["label", "title", "description"])
test_df  = pd.read_csv("test.csv",  header=None,
                        names=["label", "title", "description"])

# Combine title + description into one text field
train_df["text"] = train_df["title"] + " " + train_df["description"]
test_df["text"]  = test_df["title"]  + test_df["description"]

# Shift labels to 0-indexed
train_df["label"] = train_df["label"] - 1
test_df["label"]  = test_df["label"]  - 1

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = 4
```

### Semi-Supervised Split (mask labels)
```python
import numpy as np

LABEL_FRACTION = 0.05  # use 5% of training labels

labeled_df   = train_df.groupby("label", group_keys=False)\
                        .apply(lambda x: x.sample(frac=LABEL_FRACTION, random_state=42))
unlabeled_df = train_df.drop(labeled_df.index).copy()
unlabeled_df["label"] = -1  # -1 means unlabeled

# For unsupervised: drop labels entirely
unsup_df = train_df.drop(columns=["label"])
```

---

## 🗂️ Project Structure

```
autolabel_project/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── classes.txt
│
├── embeddings/
│   ├── embed_bert.py          # BERT/RoBERTa embeddings
│   ├── embed_openai.py        # OpenAI text-embedding
│   └── embed_tfidf.py         # TF-IDF baseline
│
├── unsupervised/
│   ├── clustering.py          # KMeans, HDBSCAN
│   ├── bertopic_model.py      # BERTopic topic discovery
│   └── label_assignment.py    # Cluster → label name via LLM or majority vote
│
├── semi_supervised/
│   ├── weak_supervision.py    # Snorkel / labeling functions
│   └── pseudo_labeling.py     # Self-training loop
│
├── evaluation/
│   └── metrics.py             # All evaluation metrics
│
├── results/
│   └── comparison_table.csv   # Final results across all methods
│
└── README.md                  # This file
```

---

## ⚙️ Approach 1: Unsupervised Auto-Labeling

### No labels used during training. Ground truth used only at evaluation.

---

### 1A. Embedding → Clustering Pipeline

**Step 1: Generate Embeddings**

```python
# --- BERT / RoBERTa ---
from transformers import AutoTokenizer, AutoModel
import torch

def get_bert_embeddings(texts, model_name="roberta-base", batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling over token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)
```

```python
# --- OpenAI Embeddings ---
from openai import OpenAI

client = OpenAI()

def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=model)
        embeddings = [r.embedding for r in response.data]
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)
```

**Step 2: Dimensionality Reduction (optional but recommended)**

```python
import umap

reducer = umap.UMAP(n_components=50, metric="cosine", random_state=42)
embeddings_reduced = reducer.fit_transform(embeddings)
```

**Step 3: Clustering**

```python
from sklearn.cluster import KMeans
import hdbscan

# KMeans (requires knowing K)
kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=10)
cluster_labels_km = kmeans.fit_predict(embeddings_reduced)

# HDBSCAN (discovers K automatically)
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric="euclidean",
                              cluster_selection_method="eom")
cluster_labels_hdb = clusterer.fit_predict(embeddings_reduced)
# Note: HDBSCAN assigns -1 to noise points
```

**Step 4: Auto-assign label names to clusters**

```python
# Option A: Majority vote from ground truth (only for evaluation)
from scipy.stats import mode

def majority_vote_mapping(cluster_labels, true_labels, n_clusters):
    mapping = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        majority = mode(true_labels[mask], keepdims=True).mode[0]
        mapping[c] = majority
    return mapping

# Option B: LLM names the cluster using sample documents
def llm_name_cluster(sample_texts, candidate_labels, llm_client):
    prompt = f"""
    Given these text samples from one cluster:
    {chr(10).join(f'- {t}' for t in sample_texts[:5])}

    Choose the best label from: {candidate_labels}
    Respond with only the label name.
    """
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

---

### 1B. BERTopic

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model,
                        nr_topics=NUM_CLASSES,
                        calculate_probabilities=True)

topics, probs = topic_model.fit_transform(texts)
topic_info = topic_model.get_topic_info()
print(topic_info)
```

---

## ⚙️ Approach 2: Semi-Supervised Auto-Labeling

### A small labeled set (5–10%) guides labeling of the unlabeled pool.

---

### 2A. Weak Supervision (Snorkel / Labeling Functions)

Define heuristic labeling functions (LFs) that each vote on a label. Snorkel's label model combines them probabilistically.

```python
import re
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

ABSTAIN = -1
WORLD, SPORTS, BUSINESS, SCITECH = 0, 1, 2, 3

# --- Define Labeling Functions ---

WORLD_KEYWORDS    = ["war", "government", "election", "military", "president",
                      "troops", "peace", "nato", "un ", "united nations"]
SPORTS_KEYWORDS   = ["game", "player", "coach", "season", "championship",
                      "league", "tournament", "score", "athlete", "nfl", "nba"]
BUSINESS_KEYWORDS = ["stock", "market", "shares", "revenue", "profit",
                      "ceo", "earnings", "investor", "company", "merger"]
SCITECH_KEYWORDS  = ["software", "technology", "research", "scientist",
                      "space", "nasa", "quantum", "ai", "computer", "drug"]

def keyword_lf(text, keywords, label):
    text_lower = text.lower()
    return label if any(kw in text_lower for kw in keywords) else ABSTAIN

@labeling_function()
def lf_world(x):    return keyword_lf(x.text, WORLD_KEYWORDS,    WORLD)

@labeling_function()
def lf_sports(x):   return keyword_lf(x.text, SPORTS_KEYWORDS,   SPORTS)

@labeling_function()
def lf_business(x): return keyword_lf(x.text, BUSINESS_KEYWORDS, BUSINESS)

@labeling_function()
def lf_scitech(x):  return keyword_lf(x.text, SCITECH_KEYWORDS,  SCITECH)

@labeling_function()
def lf_has_percentage(x):
    # Financial texts often contain percentages
    return BUSINESS if re.search(r'\d+\.?\d*\%', x.text) else ABSTAIN

@labeling_function()
def lf_has_score(x):
    # Sports scores: "3-2", "21-14"
    return SPORTS if re.search(r'\b\d{1,2}-\d{1,2}\b', x.text) else ABSTAIN

# --- Apply LFs and Train Label Model ---
lfs = [lf_world, lf_sports, lf_business, lf_scitech,
        lf_has_percentage, lf_has_score]

applier  = PandasLFApplier(lfs=lfs)
L_train  = applier.apply(df=unlabeled_df)
L_dev    = applier.apply(df=labeled_df)

# Analyze LF quality
print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

# Train generative label model
label_model = LabelModel(cardinality=NUM_CLASSES, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr=0.001, seed=42)

# Generate probabilistic labels for unlabeled data
proba_labels = label_model.predict_proba(L=L_train)  # shape: (N, 4)
hard_labels  = label_model.predict(L=L_train)        # shape: (N,)
```

---

### 2B. Pseudo-Labeling (Self-Training Loop)

Uses a model trained on labeled data to iteratively label unlabeled examples with high confidence.

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer)
from torch.nn.functional import softmax
import torch

MODEL_NAME = "roberta-base"

def train_model(labeled_df, model_name=MODEL_NAME, epochs=3):
    """Fine-tune RoBERTa on labeled data."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_CLASSES)

    # ... (tokenize, create Dataset, set up Trainer)
    # Returns trained model and tokenizer
    return model, tokenizer

def get_predictions(model, tokenizer, texts, batch_size=64):
    """Return softmax probabilities for each text."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

def pseudo_label_loop(labeled_df, unlabeled_df, n_iterations=3,
                       confidence_threshold=0.90):
    """
    Iterative self-training:
    1. Train on labeled set
    2. Predict on unlabeled set
    3. Add high-confidence predictions to labeled set
    4. Repeat
    """
    current_labeled = labeled_df.copy()
    remaining_unlabeled = unlabeled_df.copy()

    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration+1} ---")
        print(f"Labeled size:   {len(current_labeled)}")
        print(f"Unlabeled size: {len(remaining_unlabeled)}")

        # Step 1: Train
        model, tokenizer = train_model(current_labeled)

        # Step 2: Predict on unlabeled
        probs = get_predictions(model, tokenizer,
                                 remaining_unlabeled["text"].tolist())
        confidence = probs.max(axis=1)
        predicted_labels = probs.argmax(axis=1)

        # Step 3: Select high-confidence samples
        high_conf_mask = confidence >= confidence_threshold
        newly_labeled = remaining_unlabeled[high_conf_mask].copy()
        newly_labeled["label"] = predicted_labels[high_conf_mask]
        newly_labeled["confidence"] = confidence[high_conf_mask]

        print(f"New pseudo-labels: {high_conf_mask.sum()} "
              f"(threshold={confidence_threshold})")

        # Step 4: Update sets
        current_labeled    = pd.concat([current_labeled, newly_labeled],
                                        ignore_index=True)
        remaining_unlabeled = remaining_unlabeled[~high_conf_mask]

        if len(remaining_unlabeled) == 0:
            print("All samples labeled. Stopping early.")
            break

    # Final model trained on full pseudo-labeled set
    final_model, final_tokenizer = train_model(current_labeled)
    return final_model, final_tokenizer, current_labeled
```

---

## 📏 Evaluation Metrics

### For Unsupervised Methods

These metrics compare discovered clusters against hidden ground truth.

```python
from sklearn.metrics import (
    normalized_mutual_info_score,   # NMI
    adjusted_rand_score,            # ARI
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    fowlkes_mallows_score           # FMI
)
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(true_labels, cluster_labels):
    """
    Uses the Hungarian algorithm to find the best
    cluster-to-class mapping, then computes accuracy.
    """
    true_labels    = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    n_classes  = len(np.unique(true_labels))
    n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
    size = max(n_classes, n_clusters)

    cost_matrix = np.zeros((size, size))
    for c in range(n_clusters):
        for k in range(n_classes):
            cost_matrix[c, k] = np.sum(
                (cluster_labels == c) & (true_labels == k))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    correct = cost_matrix[row_ind, col_ind].sum()
    total   = (cluster_labels >= 0).sum()  # exclude noise (-1) from HDBSCAN
    return correct / total

def evaluate_unsupervised(true_labels, cluster_labels, embeddings):
    mask = cluster_labels >= 0  # exclude HDBSCAN noise points

    results = {
        # External metrics (require ground truth)
        "ACC (Hungarian)":   clustering_accuracy(
                                true_labels[mask], cluster_labels[mask]),
        "NMI":               normalized_mutual_info_score(
                                true_labels[mask], cluster_labels[mask]),
        "ARI":               adjusted_rand_score(
                                true_labels[mask], cluster_labels[mask]),
        "FMI":               fowlkes_mallows_score(
                                true_labels[mask], cluster_labels[mask]),
        "Homogeneity":       homogeneity_score(
                                true_labels[mask], cluster_labels[mask]),
        "Completeness":      completeness_score(
                                true_labels[mask], cluster_labels[mask]),
        "V-Measure":         v_measure_score(
                                true_labels[mask], cluster_labels[mask]),

        # Internal metrics (no ground truth needed — real-world proxy)
        "Silhouette Score":  silhouette_score(
                                embeddings[mask], cluster_labels[mask],
                                metric="cosine"),
        "Davies-Bouldin":    davies_bouldin_score(
                                embeddings[mask], cluster_labels[mask]),

        # Coverage (important for HDBSCAN)
        "Coverage":          mask.sum() / len(cluster_labels),
    }
    return results
```

#### Metric Guide (Unsupervised)

| Metric | Range | Better | What it measures |
|--------|-------|--------|-----------------|
| ACC | 0–1 | Higher | Best-matched cluster accuracy |
| NMI | 0–1 | Higher | Mutual info between clusters and true labels |
| ARI | -1–1 | Higher | Agreement beyond chance |
| FMI | 0–1 | Higher | Geometric mean of precision & recall of pairs |
| Homogeneity | 0–1 | Higher | Each cluster contains only one class |
| Completeness | 0–1 | Higher | All members of a class are in one cluster |
| V-Measure | 0–1 | Higher | Harmonic mean of homogeneity & completeness |
| Silhouette | -1–1 | Higher | Intra-cluster cohesion vs inter-cluster separation |
| Davies-Bouldin | 0–∞ | Lower | Average cluster similarity ratio |
| Coverage | 0–1 | Higher | Fraction of points assigned (HDBSCAN) |

---

### For Semi-Supervised Methods

After pseudo-labels or weak supervision labels are generated, train a final classifier and evaluate on the test set.

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate_semisupervised(true_labels, predicted_labels,
                             class_names=CLASS_NAMES):
    results = {
        "Accuracy":          accuracy_score(true_labels, predicted_labels),
        "Macro F1":          f1_score(true_labels, predicted_labels,
                                       average="macro"),
        "Weighted F1":       f1_score(true_labels, predicted_labels,
                                       average="weighted"),
        "Macro Precision":   precision_score(true_labels, predicted_labels,
                                              average="macro"),
        "Macro Recall":      recall_score(true_labels, predicted_labels,
                                           average="macro"),
        "Cohen's Kappa":     cohen_kappa_score(true_labels, predicted_labels),
    }

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels,
                                 target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()

    return results
```

#### Label Quality Metrics (for pseudo-labels themselves)

```python
def evaluate_label_quality(true_labels, pseudo_labels,
                            confidence_scores=None):
    """
    Evaluate how good the auto-generated labels are
    compared to ground truth — before training a classifier.
    """
    mask = pseudo_labels >= 0  # only labeled samples

    results = {
        "Label Accuracy":    accuracy_score(
                                true_labels[mask], pseudo_labels[mask]),
        "Label Macro F1":    f1_score(
                                true_labels[mask], pseudo_labels[mask],
                                average="macro"),
        "Coverage":          mask.sum() / len(pseudo_labels),
    }

    if confidence_scores is not None:
        results["Mean Confidence"] = confidence_scores[mask].mean()
        results["Median Confidence"] = np.median(confidence_scores[mask])

    return results
```

#### Metric Guide (Semi-Supervised)

| Metric | Range | Better | What it measures |
|--------|-------|--------|-----------------|
| Accuracy | 0–1 | Higher | Overall correct predictions |
| Macro F1 | 0–1 | Higher | F1 averaged equally across classes |
| Weighted F1 | 0–1 | Higher | F1 weighted by class frequency |
| Macro Precision | 0–1 | Higher | Avg precision across classes |
| Macro Recall | 0–1 | Higher | Avg recall across classes |
| Cohen's Kappa | -1–1 | Higher | Agreement beyond chance |
| Label Accuracy | 0–1 | Higher | Quality of pseudo-labels vs ground truth |
| Coverage | 0–1 | Higher | Fraction of unlabeled data that got a label |

> **Note:** Use **Macro F1** as the primary metric since AG News is balanced. Use **Weighted F1** only if class imbalance becomes an issue.

---

## 📊 Final Comparison Table

Collect all results here for easy comparison.

```python
import pandas as pd

results_table = {
    # Unsupervised
    "BERT + KMeans":           {...},
    "RoBERTa + KMeans":        {...},
    "OpenAI + KMeans":         {...},
    "RoBERTa + HDBSCAN":       {...},
    "OpenAI + HDBSCAN":        {...},
    "BERTopic":                {...},

    # Semi-Supervised
    "Weak Supervision (5%)":   {...},
    "Pseudo-Labeling (5%)":    {...},
    "Pseudo-Labeling (10%)":   {...},

    # Baselines
    "TF-IDF + KMeans":         {...},   # unsupervised baseline
    "Full Supervised (100%)":  {...},   # upper bound
}

df_results = pd.DataFrame(results_table).T
df_results.to_csv("results/comparison_table.csv")
print(df_results.to_markdown())
```

Expected output format:

| Method | ACC | NMI | ARI | Macro F1 | Coverage |
|--------|-----|-----|-----|----------|----------|
| TF-IDF + KMeans | — | — | — | — | — |
| BERT + KMeans | — | — | — | — | — |
| RoBERTa + KMeans | — | — | — | — | — |
| OpenAI + KMeans | — | — | — | — | — |
| BERTopic | — | — | — | — | — |
| Weak Supervision (5%) | — | — | — | — | — |
| Pseudo-Labeling (5%) | — | — | — | — | — |
| Pseudo-Labeling (10%) | — | — | — | — | — |
| **Full Supervised (100%)** | — | — | — | — | — |

---

## 🔧 Environment Setup

```bash
pip install torch transformers sentence-transformers
pip install scikit-learn umap-learn hdbscan bertopic
pip install snorkel openai pandas numpy matplotlib seaborn
pip install datasets accelerate tabulate
```

---

## 🚀 Run Order

```
1. data_loader.py          → load and preprocess AG News
2. embeddings/embed_*.py   → generate all embeddings and cache to disk
3. unsupervised/           → run clustering experiments
4. semi_supervised/        → run weak supervision and pseudo-labeling
5. evaluation/metrics.py   → compute and print all metrics
6. results/                → compare all methods in one table
```

---

## 📝 Notes for Claude Code

- Always cache embeddings to disk (numpy `.npy` files) after generation — embedding is the slowest step.
- Use `random_state=42` everywhere for reproducibility.
- For HDBSCAN, noise points (label = -1) must be excluded from metric calculations.
- For pseudo-labeling, track both **label accuracy** and **coverage** across iterations — there is a trade-off between the two as you lower the confidence threshold.
- The full supervised baseline (100% labels, fine-tuned RoBERTa) is the upper-bound target. Report how close each method gets.
- For weak supervision, run `LFAnalysis` and report LF coverage, conflict rate, and empirical accuracy before training the label model.
- Save all plots (confusion matrix, cluster UMAP visualization, metric comparison bar chart) to the `results/` folder.
