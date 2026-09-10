# Mentor Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ELECTRA-small comparison model (alongside DistilBERT), a summarize-then-zero-shot-classify labeling method, and KeyBERT-based meaningful cluster labels — addressing the 4 points of mentor feedback without disturbing any existing DistilBERT results.

**Architecture:** Purely additive changes layered on the existing `utils/` package + 9-notebook pipeline. New logic lives in two new `utils/` modules (`summarization.py`, plus an addition to `interpretability.py`); new notebooks (`05b`, `06b`, `09`) follow the exact cell structure of their existing siblings; existing notebooks (`02`, `03`, `07`, `08`) are re-run, not restructured, to pick up new labels/rows.

**Tech Stack:** Hugging Face `transformers` (`pipeline("summarization")`, `pipeline("zero-shot-classification")`, `AutoModelForSequenceClassification` for ELECTRA), `keybert` (new dependency), `sentence-transformers` (already present, reused by KeyBERT), `uv` for environment/dependency management (all commands via `uv run ...` / `uv add ...`).

**Spec:** `docs/superpowers/specs/2026-08-26-mentor-feedback-design.md`

## Global Constraints

- `SEED = 42` everywhere randomness occurs (existing project convention, `utils/config.py`).
- **Additive only for item 1:** `utils.config.CLASSIFIER_MODEL_NAME` (`distilbert-base-uncased`) and notebooks `05`/`06` are never modified — ELECTRA-small is a new, separate `CLASSIFIER_MODEL_NAME_ALT` constant and new `05b`/`06b` notebooks, so DistilBERT results stay intact for comparison.
- No pytest / no `tests/` directory — matches this repo's existing convention (`docs/superpowers/plans/2026-08-15-autolabel-notebooks.md`'s Global Constraints: "`utils/` functions are verified with one-off `uv run python` smoke checks ... when first written; the notebooks' own inline assertions are the ongoing regression check"). Explicitly confirmed for this plan too.
- **`uv` is the only environment/dependency tool** — no `pip install`. All Python/Jupyter commands prefixed `uv run`; new dependency added via `uv add`.
- Every long-running notebook execution (`nbconvert --execute`) MUST be launched as a background process, not a blocking foreground call — some of these run 20-60+ minutes on this CPU-only machine. Poll for the expected output file or process exit; do not combine shell-level backgrounding (`&`/`nohup`) with a tool-level background flag — pick exactly one mechanism.
- `CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]`, `NUM_CLASSES = 4`, labels 0-indexed — unchanged, reused throughout.
- Item 3 (label-closeness comparison) is explicitly out of scope for this plan — deferred to a documentation note only (Task 11).

---

## Task 1: Config constants + `keybert` dependency

**Files:**
- Modify: `utils/config.py`
- Modify: `pyproject.toml` (via `uv add`, not manual edit)

**Interfaces:**
- Produces: `utils.config.{CLASSIFIER_MODEL_NAME_ALT, SUMMARIZATION_MODEL_NAME, ZERO_SHOT_MODEL_NAME, SUMMARIZATION_SAMPLE_SIZE}` — consumed by Tasks 2, 4, 5, 6, 7.

- [ ] **Step 1: Add new constants to `utils/config.py`**

Append after the existing `CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"` line (keep that line unchanged):

```python
# Additional semi-supervised model (mentor feedback item 1) — compared
# *alongside* CLASSIFIER_MODEL_NAME in new 05b/06b notebooks, not replacing
# it, so existing DistilBERT results stay intact. ELECTRA's replaced-token-
# detection pretraining is a distinct family from BERT/DistilBERT/RoBERTa's
# masked-LM objective. ~14M params — expected comparable-or-faster CPU
# fine-tuning than DistilBERT-base's 66M.
CLASSIFIER_MODEL_NAME_ALT = "google/electra-small-discriminator"

# Summarize-then-zero-shot-classify labeling method (item 2, notebook 09).
SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
ZERO_SHOT_MODEL_NAME = "valhalla/distilbart-mnli-12-3"
# Separate cap for notebook 09 — two chained CPU generation/inference
# passes per row (summarization + zero-shot), no fine-tuning. Starting
# value; Task 2's smoke-test measures actual per-row throughput on this
# machine and this value should be adjusted before Task 4's full run if
# the measured throughput makes 200 rows clearly too slow/fast for a
# practical unattended budget (same measure-then-set pattern already used
# for ROBERTA_SAMPLE_SIZE/CLASSIFIER_SAMPLE_SIZE above).
SUMMARIZATION_SAMPLE_SIZE = 200
```

- [ ] **Step 2: Add the `keybert` dependency**

Run: `uv add keybert`
Expected: `pyproject.toml`'s `dependencies` gains `"keybert"`, `uv.lock` updates, installs into `.venv/` (pulls in a few small transitive deps — no torch/transformers re-download since already present).

- [ ] **Step 3: Verify**

Run: `uv run python -c "from utils import config; print(config.CLASSIFIER_MODEL_NAME_ALT, config.SUMMARIZATION_MODEL_NAME, config.ZERO_SHOT_MODEL_NAME, config.SUMMARIZATION_SAMPLE_SIZE)"`
Expected: prints `google/electra-small-discriminator sshleifer/distilbart-cnn-6-6 valhalla/distilbart-mnli-12-3 200` with no errors.

Run: `uv run python -c "import keybert; print(keybert.__version__)"`
Expected: prints a version string, no import error.

- [ ] **Step 4: Commit**

```bash
git add utils/config.py pyproject.toml uv.lock
git commit -m "Add ELECTRA-small, summarization/zero-shot model config, keybert dependency"
```

---

## Task 2: `utils/summarization.py`

**Files:**
- Create: `utils/summarization.py`

**Interfaces:**
- Consumes: `utils.config.{SUMMARIZATION_MODEL_NAME, ZERO_SHOT_MODEL_NAME}` (Task 1)
- Produces: `summarize_texts(texts, model_name=..., max_length=60, min_length=8, batch_size=8) -> list[str]`, `generate_titles(texts, model_name=..., max_length=12, min_length=3, batch_size=8) -> list[str]`, `zero_shot_label(texts, class_names, model_name=..., batch_size=8) -> (np.ndarray[int], np.ndarray[float])` — consumed by Task 4 (notebook 09).

- [ ] **Step 1: Write `utils/summarization.py`**

```python
import numpy as np

from utils.config import SUMMARIZATION_MODEL_NAME, ZERO_SHOT_MODEL_NAME


def summarize_texts(texts, model_name=SUMMARIZATION_MODEL_NAME, max_length=60,
                     min_length=8, batch_size=8) -> list:
    """Summarize each text with a CPU-friendly distilled summarization model.

    Returns one summary string per input text, same order.
    """
    from transformers import pipeline

    summarizer = pipeline("summarization", model=model_name)
    outputs = summarizer(list(texts), max_length=max_length, min_length=min_length,
                          truncation=True, batch_size=batch_size)
    return [o["summary_text"].strip() for o in outputs]


def generate_titles(texts, model_name=SUMMARIZATION_MODEL_NAME, max_length=12,
                     min_length=3, batch_size=8) -> list:
    """Generate a short headline-length title per text.

    Same summarization model as `summarize_texts`, just a much shorter
    output length — reused rather than a separate pipeline.
    """
    return summarize_texts(texts, model_name=model_name, max_length=max_length,
                            min_length=min_length, batch_size=batch_size)


def zero_shot_label(texts, class_names, model_name=ZERO_SHOT_MODEL_NAME,
                     batch_size=8):
    """Zero-shot-classify each text into one of `class_names` via an NLI model.

    Returns (predicted_labels, confidence) as parallel numpy arrays —
    predicted_labels are indices into class_names, confidence is the top
    label's entailment score. Same shape/contract as utils.modeling's
    prediction outputs so callers can reuse utils.metrics unchanged.
    """
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model=model_name)
    results = classifier(list(texts), candidate_labels=list(class_names),
                          batch_size=batch_size)
    if isinstance(results, dict):
        results = [results]

    predicted_labels = np.array([class_names.index(r["labels"][0]) for r in results])
    confidence = np.array([r["scores"][0] for r in results])
    return predicted_labels, confidence
```

- [ ] **Step 2: Smoke-test and measure throughput**

Run (this downloads `sshleifer/distilbart-cnn-6-6` and `valhalla/distilbart-mnli-12-3` on first use — expect a one-time multi-minute download, then fast on repeat runs since HF caches models under `~/.cache/huggingface`):

```bash
uv run python -c "
import time
from utils.summarization import summarize_texts, generate_titles, zero_shot_label
from utils.config import CLASS_NAMES

texts = [
    'Apple unveiled its newest iPhone today, featuring a faster chip and improved camera system for photography enthusiasts.',
    'The home team won the championship game in overtime last night after a dramatic final-minute comeback.',
    'Stock markets rallied this week as investors reacted positively to the central bank interest rate decision.',
]
start = time.time()
summaries = summarize_texts(texts)
titles = generate_titles(texts)
labels, conf = zero_shot_label(summaries, CLASS_NAMES)
elapsed = time.time() - start
for t, s, ti, l, c in zip(texts, summaries, titles, labels, conf):
    print('TEXT:', t[:60])
    print('  SUMMARY:', s)
    print('  TITLE:', ti)
    print('  LABEL:', CLASS_NAMES[l], f'({c:.2f})')
print(f'Elapsed: {elapsed:.1f}s for {len(texts)} rows -> {elapsed/len(texts):.2f}s/row')
"
```

Expected: no errors; each row prints a non-empty summary, a short title, and one of `World/Sports/Business/Sci/Tech` as the label (the sports/business/tech examples above should plausibly land on their matching class, though this is a smoke test, not a scored eval). Note the printed `s/row` figure — if it projects `SUMMARIZATION_SAMPLE_SIZE=200` rows (2 pipelines per row) to well over ~45 minutes, lower `SUMMARIZATION_SAMPLE_SIZE` in `utils/config.py` accordingly before Task 4; if it's comfortably faster, leave it as-is (no need to raise it — 200 rows already gives a reasonable eval sample).

- [ ] **Step 3: Commit**

```bash
git add utils/summarization.py
git commit -m "Add utils/summarization.py: summarize, generate titles, zero-shot label"
```

---

## Task 3: Extend `utils/samples.py` for extra columns

**Files:**
- Modify: `utils/samples.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `save_label_samples(..., extra_columns: dict[str, list] = None, ...)` — new optional keyword-only-in-practice parameter (not enforced keyword-only, but callers use it as such); existing callers (05, 06, 08 notebooks) are unaffected since it defaults to `None`. Consumed by Task 4 (notebook 09) to attach `summary`/`generated_title` columns.

- [ ] **Step 1: Edit `utils/samples.py`**

Change the signature and insert the new columns before the per-class sampling loop, so they stay row-aligned:

```python
def save_label_samples(texts, predicted_labels, true_labels, class_names,
                        confidence=None, extra_columns=None, n_per_class=2,
                        seed=None, path=None):
    """Save a small, stratified qualitative sample of (text, predicted label,
    true label, correct[, confidence][, extra columns]) to CSV.

    Meant for presentation: a handful of actual generated labels per
    predicted class, alongside the aggregate metrics already reported
    elsewhere. Not a substitute for the metrics — just a spot-check a human
    reader can eyeball.

    `extra_columns`, if given, is a dict of {column_name: values} (same
    length/order as `texts`) merged in before sampling so it stays
    row-aligned — e.g. {"summary": [...], "generated_title": [...]} for
    the summarization-labeling method. Unlike `text`, extra columns are
    not truncated.
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

    samples = []
    for cls in class_names:
        subset = df[df["predicted_label"] == cls]
        if len(subset) == 0:
            continue
        take = min(n_per_class, len(subset))
        samples.append(subset.sample(n=take, random_state=seed))
    sample_df = pd.concat(samples, ignore_index=True) if samples else df.head(0)
    sample_df["text"] = sample_df["text"].str.slice(0, 140)

    if path is not None:
        sample_df.to_csv(path, index=False)
    return sample_df
```

- [ ] **Step 2: Smoke-test**

```bash
uv run python -c "
from utils.samples import save_label_samples

texts = ['a', 'b', 'c', 'd']
preds = [0, 1, 0, 1]
trues = [0, 1, 1, 1]
extra = {'summary': ['sum a', 'sum b', 'sum c', 'sum d']}
df = save_label_samples(texts, preds, trues, ['X', 'Y'], extra_columns=extra, n_per_class=2, seed=42)
print(df)
assert 'summary' in df.columns
assert len(df) == 4
print('OK')
"
```

Expected: prints a 4-row DataFrame with a `summary` column correctly aligned to each row's original text, then `OK`.

- [ ] **Step 3: Commit**

```bash
git add utils/samples.py
git commit -m "Add extra_columns support to save_label_samples for summary/title output"
```

---

## Task 4: Notebook `09_summarization_labeling.ipynb`

**Files:**
- Create: `notebooks/09_summarization_labeling.ipynb`

**Interfaces:**
- Consumes: `data/processed/test_clean.parquet`, `utils.data.stratified_sample`, `utils.summarization.{summarize_texts, generate_titles, zero_shot_label}` (Task 2), `utils.metrics.evaluate_semisupervised`, `utils.samples.save_label_samples` (Task 3), `utils.config.SUMMARIZATION_SAMPLE_SIZE` (Task 1)
- Produces: `results/metrics_summarization.json`, `results/confusion_matrix_summarization.png`, `results/sample_labels_summarization.csv` — consumed by Task 10 (`07_comparison.ipynb`).

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 09 — Summarization → Zero-Shot Labeling

A different approach to auto-labeling: instead of clustering embeddings or
fine-tuning a classifier, summarize each article with a text-summarization
model, then zero-shot-classify the summary into one of the 4 AG News
classes. Uses `utils.config.SUMMARIZATION_MODEL_NAME`
(`sshleifer/distilbart-cnn-6-6`) and `utils.config.ZERO_SHOT_MODEL_NAME`
(`valhalla/distilbart-mnli-12-3`) — both open-source, CPU-runnable, no
fine-tuning. Also generates a short headline per article (same summarizer,
shorter max_length) alongside its summary, saved together in the
sample-labels CSV. Scored against the hidden true label like every other
method in `results/comparison_table.csv`.
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
from utils.samples import save_label_samples
from utils.summarization import generate_titles, summarize_texts, zero_shot_label
```

Code cell (load + sample):
```python
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")
test_sample = stratified_sample(test_clean, config.SUMMARIZATION_SAMPLE_SIZE, seed=config.SEED)
print(f"Summarizing + zero-shot labeling {len(test_sample)} test rows")
```

Code cell (summarize + titles + zero-shot classify):
```python
summaries = summarize_texts(test_sample["text"].tolist())
generated_titles = generate_titles(test_sample["text"].tolist())
predicted_labels, confidence = zero_shot_label(summaries, config.CLASS_NAMES)

print("Example:")
print(" text:", test_sample["text"].iloc[0][:100])
print(" summary:", summaries[0])
print(" generated title:", generated_titles[0])
print(" predicted:", config.CLASS_NAMES[predicted_labels[0]], f"(confidence {confidence[0]:.2f})")
```

Code cell (evaluate + save metrics):
```python
results, report, cm = evaluate_semisupervised(
    test_sample["label"].to_numpy(), predicted_labels, config.CLASS_NAMES,
    save_path=config.RESULTS_DIR / "confusion_matrix_summarization.png")
print(report)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_summarization.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved summarization + zero-shot labeling results.")
```

Code cell (save sample labels with summary + generated title):
```python
save_label_samples(
    test_sample["text"], predicted_labels, test_sample["label"].to_numpy(),
    config.CLASS_NAMES, confidence=confidence, n_per_class=2, seed=config.SEED,
    extra_columns={"summary": summaries, "generated_title": generated_titles},
    path=config.RESULTS_DIR / "sample_labels_summarization.csv")
print("Saved sample generated labels (with summaries + generated titles) for summarization_zero_shot.")
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run as a single background process (per Global Constraints — do not block foreground, do not double-background):
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/09_summarization_labeling.ipynb --ExecutePreprocessor.timeout=5400`
Poll for `results/metrics_summarization.json` to appear or the process to exit. Expected: exits 0. Budget generously (5400s) since Task 2's throughput measurement is only a 3-row smoke test, not a guarantee at 200 rows.

- [ ] **Step 3: Verify output**

Run: `cat results/metrics_summarization.json`
Expected: valid JSON with `Accuracy`, `Macro F1`, etc.

Run: `head -3 results/sample_labels_summarization.csv`
Expected: header row includes `text,predicted_label,true_label,correct,confidence,summary,generated_title`, plus 2 data rows with non-empty `summary`/`generated_title` values.

Confirm `results/confusion_matrix_summarization.png` exists.

- [ ] **Step 4: Commit**

```bash
git add notebooks/09_summarization_labeling.ipynb results/metrics_summarization.json results/confusion_matrix_summarization.png results/sample_labels_summarization.csv
git commit -m "Add 09_summarization_labeling notebook: summarize + zero-shot classify"
```

---

## Task 5: Notebook `05b_pseudo_labeling_electra.ipynb`

**Files:**
- Create: `notebooks/05b_pseudo_labeling_electra.ipynb`

**Interfaces:**
- Consumes: `data/processed/{labeled,unlabeled,test_clean}.parquet`, `utils.modeling.{pseudo_label_loop, get_predictions}`, `utils.metrics.{evaluate_label_quality, evaluate_semisupervised}`, `utils.samples.save_label_samples`, `utils.config.CLASSIFIER_MODEL_NAME_ALT` (Task 1)
- Produces: `results/metrics_pseudo_labeling_electra.json`, `results/confusion_matrix_pseudo_labeling_electra.png`, `results/sample_labels_pseudo_labeling_electra_train_pool.csv`, `results/sample_labels_pseudo_labeling_electra_test.csv` — consumed by Task 10.

This mirrors `notebooks/05_pseudo_labeling.ipynb` exactly, at the same
`PSEUDO_LABEL_SAMPLE_SIZE=400` and hyperparameters (0.80 confidence
threshold, 0.98 target coverage — the tuned values from
`PROJECT_SUMMARY.md` §5's tuning story), swapping only the model name and
output filenames, so the two models are directly comparable.

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 05b — Pseudo-Labeling (Self-Training) — ELECTRA-small

Same self-training loop as `05_pseudo_labeling.ipynb`, run with
`utils.config.CLASSIFIER_MODEL_NAME_ALT` (`google/electra-small-discriminator`)
instead of DistilBERT, at the same sample size and tuned hyperparameters
(0.80 confidence threshold, 0.98 target coverage), so the two models are
directly comparable side by side in `07_comparison.ipynb`. Mentor
feedback item 1 — an addition, not a replacement; `05_pseudo_labeling.ipynb`
and its DistilBERT results are untouched.
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

Code cell (load + sample):
```python
labeled_df = pd.read_parquet(config.PROCESSED_DIR / "labeled.parquet")
unlabeled_df = pd.read_parquet(config.PROCESSED_DIR / "unlabeled.parquet")
test_clean = pd.read_parquet(config.PROCESSED_DIR / "test_clean.parquet")

PSEUDO_LABEL_SAMPLE_SIZE = 400  # matches 05_pseudo_labeling.ipynb for a fair comparison
labeled_sample = stratified_sample(labeled_df, PSEUDO_LABEL_SAMPLE_SIZE, seed=config.SEED)
unlabeled_sample = stratified_sample(unlabeled_df, PSEUDO_LABEL_SAMPLE_SIZE, seed=config.SEED)

overlap = set(unlabeled_sample["text"]) & set(test_clean["text"])
assert len(overlap) == 0, f"{len(overlap)} rows leaked between train pool and test set"
print(f"Labeled sample: {len(labeled_sample)} | Unlabeled sample: {len(unlabeled_sample)} | Test: {len(test_clean)}")
```

Code cell (run self-training loop, ELECTRA-small):
```python
final_model, final_tokenizer, current_labeled, history = pseudo_label_loop(
    labeled_sample, unlabeled_sample,
    model_name=config.CLASSIFIER_MODEL_NAME_ALT,
    confidence_threshold=0.80, epochs=3,
    target_coverage=0.98, max_iterations=10)

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
print("Pseudo-label quality (ELECTRA-small):", label_quality)
```

Code cell (save train-pool sample labels):
```python
from utils.samples import save_label_samples

save_label_samples(
    merged["text"], merged["label"].to_numpy(), merged["true_label"].to_numpy(),
    config.CLASS_NAMES, n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_pseudo_labeling_electra_train_pool.csv")
print("Saved sample generated pseudo-labels (train pool) for pseudo_labeling_electra.")
```

Code cell (final test evaluation + save):
```python
test_probs = get_predictions(final_model, final_tokenizer, test_clean["text"].tolist())
test_preds = test_probs.argmax(axis=1)

semisup_results, report, cm = evaluate_semisupervised(
    test_clean["label"].to_numpy(), test_preds, config.CLASS_NAMES,
    save_path=config.RESULTS_DIR / "confusion_matrix_pseudo_labeling_electra.png")
print(report)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_pseudo_labeling_electra.json", "w") as f:
    json.dump({"test_metrics": semisup_results, "label_quality": label_quality, "history": history},
               f, indent=2)
print("Saved pseudo-labeling (ELECTRA-small) results.")

save_label_samples(
    test_clean["text"], test_preds, test_clean["label"].to_numpy(),
    config.CLASS_NAMES, confidence=test_probs.max(axis=1), n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_pseudo_labeling_electra_test.csv")
print("Saved sample generated labels (test set) for pseudo_labeling_electra.")
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run as a single background process:
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05b_pseudo_labeling_electra.ipynb --ExecutePreprocessor.timeout=5400`
Poll for `results/metrics_pseudo_labeling_electra.json` to appear or the process to exit. Expected: exits 0, likely comparable to or faster than `05_pseudo_labeling.ipynb`'s original ~1hr run at the same sample size (ELECTRA-small has ~5x fewer parameters than DistilBERT-base), but budget the same 5400s timeout for safety margin.

- [ ] **Step 3: Verify output**

Run: `cat results/metrics_pseudo_labeling_electra.json`
Expected: valid JSON with `test_metrics`, `label_quality`, `history` keys; `results/confusion_matrix_pseudo_labeling_electra.png` exists.

- [ ] **Step 4: Commit**

```bash
git add notebooks/05b_pseudo_labeling_electra.ipynb results/metrics_pseudo_labeling_electra.json results/confusion_matrix_pseudo_labeling_electra.png results/sample_labels_pseudo_labeling_electra_train_pool.csv results/sample_labels_pseudo_labeling_electra_test.csv
git commit -m "Add 05b_pseudo_labeling_electra notebook (ELECTRA-small, additive comparison)"
```

---

## Task 6: Notebook `06b_full_supervised_baseline_electra.ipynb`

**Files:**
- Create: `notebooks/06b_full_supervised_baseline_electra.ipynb`

**Interfaces:**
- Consumes: `data/processed/{train_clean,test_clean}.parquet`, `utils.modeling.{train_model, get_predictions}`, `utils.metrics.evaluate_semisupervised`, `utils.samples.save_label_samples`, `utils.config.CLASSIFIER_MODEL_NAME_ALT` (Task 1)
- Produces: `results/metrics_full_supervised_electra.json`, `results/confusion_matrix_full_supervised_electra.png`, `results/sample_labels_full_supervised_electra.csv` — consumed by Task 10.

Mirrors `notebooks/06_full_supervised_baseline.ipynb` at the same
`CLASSIFIER_SAMPLE_SIZE`, swapping only the model name and output
filenames.

- [ ] **Step 1: Create the notebook with these cells, in order**

Markdown cell:
```markdown
# 06b — Full Supervised Baseline — ELECTRA-small

Same as `06_full_supervised_baseline.ipynb`, run with
`utils.config.CLASSIFIER_MODEL_NAME_ALT` (`google/electra-small-discriminator`)
instead of DistilBERT, at the same `CLASSIFIER_SAMPLE_SIZE`, for a direct
side-by-side comparison in `07_comparison.ipynb`.
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
print(f"Training on {len(train_sample)} fully-labeled rows (ELECTRA-small upper bound baseline)")
```

Code cell (train + evaluate + save):
```python
model, tokenizer = train_model(train_sample, model_name=config.CLASSIFIER_MODEL_NAME_ALT, epochs=3)

test_probs = get_predictions(model, tokenizer, test_clean["text"].tolist())
test_preds = test_probs.argmax(axis=1)

results, report, cm = evaluate_semisupervised(
    test_clean["label"].to_numpy(), test_preds, config.CLASS_NAMES,
    save_path=config.RESULTS_DIR / "confusion_matrix_full_supervised_electra.png")
print(report)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.RESULTS_DIR / "metrics_full_supervised_electra.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved full-supervised baseline (ELECTRA-small) results.")
```

Code cell (save sample labels):
```python
from utils.samples import save_label_samples

save_label_samples(
    test_clean["text"], test_preds, test_clean["label"].to_numpy(),
    config.CLASS_NAMES, confidence=test_probs.max(axis=1), n_per_class=2, seed=config.SEED,
    path=config.RESULTS_DIR / "sample_labels_full_supervised_electra.csv")
print("Saved sample generated labels for full_supervised_electra.")
```

Use the `NotebookEdit` tool to create the file and add each cell in this order.

- [ ] **Step 2: Execute the notebook end-to-end**

Run as a single background process:
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06b_full_supervised_baseline_electra.ipynb --ExecutePreprocessor.timeout=1800`
Poll for `results/metrics_full_supervised_electra.json` to appear or the process to exit. Expected: exits 0, budget ~20-30 min (dominated by the fixed full 7,600-row test-set inference pass, same as `06_full_supervised_baseline.ipynb`'s original run).

- [ ] **Step 3: Verify output**

Run: `cat results/metrics_full_supervised_electra.json`
Expected: valid JSON with `Accuracy`, `Macro F1`, etc.; `results/confusion_matrix_full_supervised_electra.png` exists.

- [ ] **Step 4: Commit**

```bash
git add notebooks/06b_full_supervised_baseline_electra.ipynb results/metrics_full_supervised_electra.json results/confusion_matrix_full_supervised_electra.png results/sample_labels_full_supervised_electra.csv
git commit -m "Add 06b_full_supervised_baseline_electra notebook (ELECTRA-small, additive comparison)"
```

---

## Task 7: `utils/interpretability.py` — KeyBERT-based cluster labels

**Files:**
- Modify: `utils/interpretability.py`

**Interfaces:**
- Consumes: `keybert.KeyBERT`, `sentence_transformers.SentenceTransformer` (both already dependencies as of Task 1)
- Produces: `meaningful_terms_per_cluster(texts, cluster_labels, n_terms=5, exclude_noise=True, embedding_model_name="all-MiniLM-L6-v2") -> dict[int, list[str]]` — new function. `summarize_clusters()` is updated to call it instead of `top_terms_per_cluster()` for its `top_terms` column (existing `top_terms_per_cluster()` is kept, unchanged, still importable). Consumed by Tasks 8 and 9 (notebooks 02, 03, 08, all of which already call `summarize_clusters()` — no notebook code changes needed beyond Task 8's BERTopic-specific addition).

- [ ] **Step 1: Add `meaningful_terms_per_cluster` to `utils/interpretability.py`**

Insert after `top_terms_per_cluster` (keep that function as-is):

```python
def meaningful_terms_per_cluster(texts, cluster_labels, n_terms=5, exclude_noise=True,
                                  embedding_model_name="all-MiniLM-L6-v2") -> dict:
    """Meaningful multi-word key-phrases per cluster via KeyBERT, instead of
    single TF-IDF top words (`top_terms_per_cluster`). Reuses the same
    MiniLM sentence-transformer already used elsewhere in the repo
    (`utils.embeddings.get_sentence_embeddings`) so phrase-embedding
    similarity is consistent with the rest of the pipeline.

    Returns {cluster_id: [phrase, ...]}, noise cluster (-1) excluded by
    default, same contract as `top_terms_per_cluster`.
    """
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    texts = np.array(texts, dtype=object)
    cluster_labels = np.array(cluster_labels)
    kw_model = KeyBERT(model=SentenceTransformer(embedding_model_name))

    cluster_ids = sorted(np.unique(cluster_labels))
    if exclude_noise:
        cluster_ids = [c for c in cluster_ids if c >= 0]

    terms_by_cluster = {}
    for c in cluster_ids:
        mask = cluster_labels == c
        if mask.sum() == 0:
            terms_by_cluster[c] = []
            continue
        # Cap concatenated docs per cluster so KeyBERT stays fast even on
        # large clusters — a representative sample, not the whole cluster.
        cluster_text = " ".join(texts[mask][:200])
        keywords = kw_model.extract_keywords(
            cluster_text, keyphrase_ngram_range=(1, 3), stop_words="english",
            top_n=n_terms, use_mmr=True, diversity=0.5)
        terms_by_cluster[c] = [phrase for phrase, score in keywords]
    return terms_by_cluster
```

- [ ] **Step 2: Update `summarize_clusters` to use it**

Change:
```python
    terms = top_terms_per_cluster(texts, cluster_labels, n_terms=n_terms)
```
to:
```python
    terms = meaningful_terms_per_cluster(texts, cluster_labels, n_terms=n_terms)
```
(one line, inside `summarize_clusters`, no other changes to that function).

- [ ] **Step 3: Smoke-test**

```bash
uv run python -c "
from utils.interpretability import meaningful_terms_per_cluster

texts = [
    'Soccer team wins championship match in dramatic overtime finish',
    'Basketball player scores winning goal in playoff game',
    'Stock market rallies as investors react to interest rate decision',
    'Company reports strong quarterly earnings and revenue growth',
]
labels = [0, 0, 1, 1]
result = meaningful_terms_per_cluster(texts, labels, n_terms=3)
print(result)
assert set(result.keys()) == {0, 1}
assert all(len(v) > 0 for v in result.values())
print('OK')
"
```

Expected: prints a dict like `{0: ['soccer team wins', ...], 1: ['stock market rallies', ...]}` (exact phrases vary), then `OK`. This downloads `all-MiniLM-L6-v2` if not already cached (already used elsewhere in the repo, so likely a cache hit).

- [ ] **Step 4: Commit**

```bash
git add utils/interpretability.py
git commit -m "Add KeyBERT-based meaningful_terms_per_cluster; use it in summarize_clusters"
```

---

## Task 8: Notebook `03_bertopic.ipynb` — meaningful topic representations

**Files:**
- Modify: `notebooks/03_bertopic.ipynb`

**Interfaces:**
- Consumes: `bertopic.representation.KeyBERTInspired` (from the existing `bertopic` dependency — no new install)
- Produces: updated `results/metrics_bertopic.json`, `results/clusters_bertopic.csv` (KeyBERT-inspired topic representations instead of raw c-TF-IDF words) — consumed by Task 10.

- [ ] **Step 1: Edit the BERTopic setup cell**

In the cell that currently reads:
```python
umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=config.SEED)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# stop_words="english" so BERTopic's own c-TF-IDF representation surfaces
# meaningful topic words ("soccer", "goal", ...) instead of "the, to, of, in"
vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
topic_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model,
                        vectorizer_model=vectorizer_model,
                        nr_topics=config.NUM_CLASSES, calculate_probabilities=False)

topics, _ = topic_model.fit_transform(texts)
print(topic_model.get_topic_info())
```

Replace with:
```python
umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=config.SEED)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# stop_words="english" so BERTopic's own c-TF-IDF representation surfaces
# meaningful topic words ("soccer", "goal", ...) instead of "the, to, of, in"
vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
# KeyBERTInspired refines the raw c-TF-IDF words into more meaningful,
# semantically-coherent topic representations (mentor feedback item 4).
representation_model = KeyBERTInspired()
topic_model = BERTopic(embedding_model=sentence_model, umap_model=umap_model,
                        vectorizer_model=vectorizer_model,
                        representation_model=representation_model,
                        nr_topics=config.NUM_CLASSES, calculate_probabilities=False)

topics, _ = topic_model.fit_transform(texts)
print(topic_model.get_topic_info())
```

Also add the import in the imports cell — change:
```python
from bertopic import BERTopic
```
to:
```python
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
```

Use the `NotebookEdit` tool to apply both edits.

- [ ] **Step 2: Execute the notebook end-to-end**

Run as a single background process:
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_bertopic.ipynb --ExecutePreprocessor.timeout=1200`
Poll for the process to exit. Expected: exits 0, similar runtime to before (representation model refinement is cheap relative to UMAP+HDBSCAN).

- [ ] **Step 3: Verify output**

Run: `head -3 results/clusters_bertopic.csv`
Expected: `top_terms` column now shows multi-word/refined phrases (from `meaningful_terms_per_cluster`, Task 7 — reused here too, since this cell calls `summarize_clusters`), not raw single c-TF-IDF words.

Run: `cat results/metrics_bertopic.json`
Expected: still valid JSON with the same keys as before (clustering quality metrics are on the same topic assignments — only representation changed, not topics themselves — so numbers should be identical or very close to the prior run).

- [ ] **Step 4: Commit**

```bash
git add notebooks/03_bertopic.ipynb results/metrics_bertopic.json results/clusters_bertopic.csv
git commit -m "Use KeyBERTInspired representation model in 03_bertopic for meaningful topic labels"
```

---

## Task 9: Re-run `02_unsupervised_clustering.ipynb` and `08_label_propagation.ipynb`

**Files:**
- None modified — both notebooks already call `utils.interpretability.summarize_clusters()`, which Task 7 updated internally. This task is execution-only.

**Interfaces:**
- Consumes: `utils.interpretability.summarize_clusters` (Task 7, updated)
- Produces: refreshed `results/clusters_*.csv` (one per clustering method in 02, plus label-propagation's if it writes one) with KeyBERT-based `top_terms` — consumed by Task 10 (no metrics changed, only the qualitative `top_terms` column, so `07_comparison.ipynb`'s numeric table is unaffected by this task).

- [ ] **Step 1: Execute `02_unsupervised_clustering.ipynb`**

Run as a single background process:
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_unsupervised_clustering.ipynb --ExecutePreprocessor.timeout=1200`
Poll for the process to exit. Expected: exits 0.

- [ ] **Step 2: Verify**

Run: `head -3 results/clusters_minilm_kmeans.csv`
Expected: `top_terms` column shows multi-word key-phrases, not single TF-IDF words.

- [ ] **Step 3: Execute `08_label_propagation.ipynb`**

Run as a single background process:
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/08_label_propagation.ipynb --ExecutePreprocessor.timeout=1200`
Poll for the process to exit. Expected: exits 0.

- [ ] **Step 4: Verify**

Run: `cat results/metrics_label_propagation.json`
Expected: unchanged numeric values from before this plan's changes (this notebook doesn't call `summarize_clusters`, so this is a regression check — confirm the re-run reproduces the same 87.2% label accuracy from `PROJECT_SUMMARY.md` §5, within floating-point noise).

- [ ] **Step 5: Commit**

```bash
git add results/clusters_*.csv results/metrics_label_propagation.json
git commit -m "Re-run 02/08 to pick up KeyBERT-based cluster labels"
```

---

## Task 10: Re-run `07_comparison.ipynb`

**Files:**
- None modified — `07_comparison.ipynb` already globs `results/metrics_*.json`, so it picks up the new files from Tasks 4, 5, 6 automatically. This task is execution-only.

**Interfaces:**
- Consumes: every `results/metrics_*.json` file (now including `metrics_summarization.json`, `metrics_pseudo_labeling_electra.json`, `metrics_full_supervised_electra.json`)
- Produces: refreshed `results/comparison_table.csv`, `results/comparison_bar_chart.png` — consumed by Task 11.

- [ ] **Step 1: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_comparison.ipynb --ExecutePreprocessor.timeout=300`
Expected: exits 0 quickly (this notebook just aggregates already-computed JSON files, no heavy compute).

- [ ] **Step 2: Verify**

Run: `cat results/comparison_table.csv`
Expected: 3 new rows present — `summarization` (or however `metrics_summarization.json`'s stem resolves — check the actual row label matches `summarization`), `pseudo_labeling_electra`, `full_supervised_electra` — alongside all pre-existing rows (nothing dropped).

- [ ] **Step 3: Commit**

```bash
git add results/comparison_table.csv results/comparison_bar_chart.png
git commit -m "Re-run 07_comparison: add ELECTRA-small and summarization rows"
```

---

## Task 11: Update documentation with real results

**Files:**
- Modify: `PROJECT_SUMMARY.md`
- Modify: `docs/semi_supervised_methods.md`

**Interfaces:**
- Consumes: `results/comparison_table.csv`, `results/metrics_summarization.json`, `results/metrics_pseudo_labeling_electra.json`, `results/metrics_full_supervised_electra.json`, `results/sample_labels_summarization.csv`, `results/clusters_*.csv` (Tasks 4-10's outputs)
- Produces: no new interfaces — this is the documentation-facing task each prior implementation task in this repo's history ends with (see git log: "Rework pseudo-labeling stop criteria; refresh full comparison table", "Add label propagation as a new semi-supervised method").

This task's exact numbers depend on Tasks 4-10's actual results, not known
until they've run — read the generated files first, then transcribe real
values into the docs below. Do not fabricate numbers.

- [ ] **Step 1: Read the generated results**

```bash
cat results/comparison_table.csv
cat results/metrics_summarization.json
cat results/metrics_pseudo_labeling_electra.json
cat results/metrics_full_supervised_electra.json
head -10 results/sample_labels_summarization.csv
```

- [ ] **Step 2: Update `PROJECT_SUMMARY.md`**

- §4 "Approaches Implemented": add a row to the semi-supervised table (§B) for "Summarize + zero-shot" describing the approach (mirroring the existing row style for weak supervision/label propagation/pseudo-labeling), and a note that ELECTRA-small pseudo-labeling/full-supervised are now run alongside DistilBERT (§C or a new subsection) as an additive comparison, referencing `05b`/`06b`.
- §5 "Results": add the 3 new rows (summarization+zero-shot, pseudo-labeling ELECTRA-small, full-supervised ELECTRA-small) to the semi-supervised/supervised results table, with the actual `Test Accuracy`/`Macro F1`/`Label Accuracy` values read in Step 1. Add a short prose paragraph comparing DistilBERT vs. ELECTRA-small at the same sample size (which model won, by how much) — this directly answers the original mentor ask ("choose another model ... we can still compare it with distilbert later").
- §6 "Sample Generated Labels": add a short subsection for the summarization method, in the same style as the existing per-method example tables (text / summary / generated title / predicted / true / correct), pulling 2-4 real rows from `results/sample_labels_summarization.csv`.
- §7 "Limitations / Next Steps": add the deferred item-3 future-work note — semantic-closeness/embedding-similarity scoring between a generated free-text label and the true class name, as a complement to exact-match accuracy (not built this pass, per explicit scope decision during design).
- §8 "Repo Map": add `notebooks/05b_pseudo_labeling_electra.ipynb`, `06b_full_supervised_baseline_electra.ipynb`, `09_summarization_labeling.ipynb` to the notebook list; add `summarization` to the `utils/` line description.

- [ ] **Step 3: Update `docs/semi_supervised_methods.md`**

Add a short new section (after the existing "Decision" section) documenting: (a) ELECTRA-small was added as an additive comparison model per mentor feedback, with a pointer to `PROJECT_SUMMARY.md` §5 for numbers; (b) the summarize→zero-shot-classify method was added as a new labeling approach (notebook 09); (c) the deferred semantic-closeness future-work note (same content as `PROJECT_SUMMARY.md` §7's addition, so both docs agree).

- [ ] **Step 4: Commit**

```bash
git add PROJECT_SUMMARY.md docs/semi_supervised_methods.md
git commit -m "Update docs with ELECTRA-small comparison and summarization-labeling results"
```
