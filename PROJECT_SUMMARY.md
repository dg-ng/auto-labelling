# Auto-Labeling for Text Classification — Master Data (16 Classes)

**Project summary for presentation**
Date: 2026-09-06 (full-scale semi-supervised re-run)

> The AG News (4-class) version of this project has been retired and is
> archived at [`docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md`](docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md)
> for historical reference. This document describes the current,
> `data/master_data.csv`-based rebuild.

## 1. Goal

Automatically assign category labels to a corpus of news/opinion articles
**without full human annotation**, and compare how far unsupervised
clustering, weak supervision, label propagation, pseudo-labeling
(self-training), and a full-supervised baseline each get toward that goal —
now across **16 overlapping categories** instead of AG News's 4 clean,
well-separated ones.

## 2. Dataset

`data/master_data.csv` — 4,800 rows (300/class), built from `data/raw/`
sources (`bbc-text.csv`, `CNN_Articels_clean` / `CNN_Articels_clean_2`,
`news-article-categories.csv`) via `notebooks/00_data_transform.ipynb`, with
columns `category, title, summary, text` (the source `summary` column is
**dropped** during cleaning — patchy/missing across source files; a fresh
`summary` is generated later by the pipeline itself, see below).

**16 classes** (alphabetical, so the label↔index mapping in `utils/config.py`
is deterministic and independent of CSV row order):

ARTS & CULTURE, BUSINESS, COMEDY, CRIME, EDUCATION, ENTERTAINMENT,
ENVIRONMENT, HEALTH, MEDIA, NEWS, POLITICS, RELIGION, SCIENCE, SPORTS,
TECH, WOMEN.

Several of these overlap in subject matter (POLITICS vs. NEWS vs. WOMEN vs.
CRIME vs. MEDIA can all plausibly describe the same story), which turns out
to matter a great deal for the semi-supervised results below.

**Cleaning**: dedup, drop degenerate rows (body under `MIN_WORDS=5`, e.g. a
body that is just `"(CNN)"`), leaving ~4,781 rows.

**Dual-split design (as of the 2026-09-06 full-scale re-run).** The
pipeline now maintains two separate splits:

- **Unsupervised track** (notebooks 01–03): a **summarized / capped** split
  (`train_clean` / `test_clean`, 399 rows) driven by `MASTER_SAMPLE_SIZE=400`.
  Summarization via `facebook/bart-large-cnn` is slow on CPU; these results
  are already committed and were **not** re-run.
- **Semi-supervised + baseline track** (notebooks 04–08): a **full-scale**
  split built directly from the cleaned ~4,781-row corpus — no sampling cap,
  no summarization needed since all methods train on raw text.

The three successive scope reductions (4,781 → 1,600 → 800 → 399 rows)
that formerly applied to every notebook now apply **only** to the
unsupervised track. This matters enormously for the semi-supervised results:
the thin 1-per-class seed that caused pseudo-labeling to stall and weak
supervision to collapse to chance was a consequence of those scope
reductions, not an inherent limitation of these methods at this dataset size.

**Full-scale split** (semi-supervised + baseline track, from
`data/processed/*_full.parquet`):
- Stratified 80/20 train/test of all ~4,781 cleaned rows →
  **3,824 train / 957 test** (approximately 239 per class per split).
- `LABEL_FRACTION=0.05` of 3,824 train rows →
  **192 labeled rows — ~12 per class** (`labeled_full.parquet`),
  unlabeled pool = **3,632 rows** (`unlabeled_full.parquet`).

**Capped split** (unsupervised track only, unchanged from prior run):
- 399 rows (MASTER_SAMPLE_SIZE=400, stratified), **319 train / 80 test**,
  **16 labeled rows — 1 per class**.

## 3. Pipeline / Data Flow

### 3.1 Overall pipeline

```mermaid
flowchart TD
    A[master_data.csv, 16 classes, ~4781 cleaned rows] --> B1[Unsupervised track:\nsample to 399 rows MASTER_SAMPLE_SIZE=400\nstratified 80/20 -> train_clean 319 / test_clean 80]
    A --> B2[Semi-supervised + baseline track:\nfull data, stratified 80/20\n-> train_full 3824 / test_full 957]
    B1 --> D[generate summary sentence per row\nfacebook/bart-large-cnn]
    D --> E[embed summaries TF-IDF/MiniLM/RoBERTa\n-> cluster KMeans k=16 / HDBSCAN / BERTopic]
    B2 --> F[5% labeled seed 192 rows ~12/class\n+ unlabeled pool 3632 rows]
    F --> G[weak supervision / label propagation\n/ pseudo-labeling on raw text]
    B2 --> H[Full-supervised baseline:\n100% of 3824 train labels, raw text]
    E --> I[results/comparison_table.csv]
    G --> I
    H --> I
```

### 3.2 Unsupervised summary → cluster → name flow

```mermaid
flowchart TD
    T[raw article text] --> S[summary sentence\nfacebook/bart-large-cnn]
    S --> V1[TF-IDF vector]
    S --> V2[MiniLM embedding]
    S --> V3[RoBERTa embedding]
    S --> V4[OpenAI embedding - pending, no API key]
    V1 --> CL[KMeans k=16 / HDBSCAN / BERTopic]
    V2 --> CL
    V3 --> CL
    V4 --> CL
    CL --> M[Hungarian-match cluster id -> class name]
    M --> R[results/full_labels_*.csv]
```

### 3.3 Pseudo-labeling self-training loop (DistilBERT and ELECTRA-small both follow this shape)

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

As Section 6 shows, both models hit the `0 new this round` stop condition
on **iteration 0** every single attempt — the loop above never gets past
its first pass through the top branch for either model at this seed size.

## 4. Approaches

### 4.1 Unsupervised clustering (`01_embeddings.ipynb`, `02_unsupervised_clustering.ipynb`, `03_bertopic.ipynb`)

Every row's `text` is summarized into a real sentence (not keywords) with
`facebook/bart-large-cnn`, then that **summary** (not the raw text) is
embedded three ways — TF-IDF, MiniLM (`all-MiniLM-L6-v2`), RoBERTa — and
clustered with KMeans (k=16, matching the known class count) and HDBSCAN;
BERTopic runs its own embed+cluster+representation pipeline directly on the
summaries. Cluster ids are Hungarian-matched to the 16 class names for
accuracy scoring. A fourth embedding (OpenAI `text-embedding-3-small`) is
wired up but **pending** — no API key configured in this environment.

### 4.2 Semi-supervised (`04_weak_supervision.ipynb`, `08_label_propagation.ipynb`, `05_pseudo_labeling.ipynb`, `05b_pseudo_labeling_electra.ipynb`)

All three methods here train on **raw text**, not summaries, and all start
from the same 16-row (1/class) labeled seed:

- **Weak supervision**: unlike the AG News version (4 hand-written keyword
  lists), 16 classes — several overlapping — don't scale to hand-authoring,
  so `utils/weak_supervision.py` **auto-derives** each class's most
  TF-IDF-distinctive keywords from the seed itself (`derive_class_keywords`)
  and turns each into a Snorkel `LabelingFunction`; a `LabelModel` combines
  the per-class votes into one weak label per unlabeled row.
- **Label propagation**: a MiniLM k-NN graph over all 319 train rows, with
  `sklearn.semi_supervised.LabelSpreading` propagating the 16-row seed's
  labels through the graph algebraically — no fine-tuning loop.
- **Pseudo-labeling / self-training**: DistilBERT and ELECTRA-small each
  fine-tune on the growing labeled set, predict on the remaining pool, and
  absorb any prediction above a confidence threshold, repeating until a
  stop condition fires (see the loop diagram above).

### 4.3 Full-supervised baseline (`06_full_supervised_baseline.ipynb`, `06b_full_supervised_baseline_electra.ipynb`)

DistilBERT and ELECTRA-small each fine-tuned on **100%** of the 319 train
rows' true labels (not the thin 16-row seed), evaluated on the 80-row
held-out test set — the ceiling the semi-supervised methods are compared
against.

## 5. Results

### 5.1 Unsupervised clustering (k=16, on generated summaries)

| Method | ACC (Hungarian) | Macro F1 | NMI | ARI | Silhouette | Coverage |
|---|---|---|---|---|---|---|
| **bertopic** | **0.7166** | **0.6231** | 0.6606 | 0.6127 | 0.0738 | 0.6439 |
| tfidf_hdbscan | 0.5145 | 0.2843 | 0.4675 | 0.3225 | 0.7672 | 0.2979 |
| minilm_kmeans | 0.4482 | 0.4443 | 0.4178 | 0.2840 | 0.5332 | 1.00 |
| minilm_hdbscan | 0.4358 | 0.3412 | 0.5233 | 0.2521 | 0.4117 | 0.5821 |
| tfidf_kmeans | 0.3713 | 0.3477 | 0.2869 | 0.1815 | 0.4156 | 1.00 |
| roberta_kmeans | 0.2440 | 0.2351 | 0.2469 | 0.1143 | 0.5711 | 1.00 |
| roberta_hdbscan | 0.0647 | 0.0113 | 0.0020 | ~0 | 0.8485 | 0.9984 |
| openai_kmeans | pending (no API key) | | | | | |
| openai_hdbscan | pending (no API key) | | | | | |

**`bertopic` is the best unsupervised method** (ACC 71.7%, Macro F1 62.3%).
It is computed only over the 64.4% of rows it assigned to a non-noise
topic — BERTopic's density-based model left ~36% of rows as noise. On the
rows it did assign, it clearly outperforms all KMeans variants.

`minilm_kmeans` is the best fully-covering unsupervised method (ACC 44.8%,
100% coverage). HDBSCAN variants now produce non-zero results at full scale
(unlike the capped 399-row run where all three degenerated to 100% noise).
`tfidf_hdbscan` reaches ACC 51.5% on its 29.8%-covered subset.
`roberta_hdbscan` barely clusters (ACC 6.5%, essentially noise).

### 5.2 Semi-supervised and supervised (raw text, full-scale — 12/class seed)

All semi-supervised and baseline results below come from the **full-scale
split** (3,824 train / 957 test, 192-row / ~12-per-class labeled seed).
The prior 1-per-class results (399-row capped split) are kept in Section
5.3 for comparison.

| Method | Label Accuracy | Label Macro F1 | Test Accuracy | Test Macro F1 | Coverage |
|---|---|---|---|---|---|
| **pseudo_labeling** (DistilBERT) | **0.9067** | **0.9098** | **0.8870** | **0.8869** | **1.00** |
| pseudo_labeling_electra (ELECTRA-small) | 0.8225 | 0.8276 | 0.8387 | 0.8373 | 1.00 |
| full_supervised (DistilBERT, 100% labels) | — | — | 0.8524 | 0.8524 | — |
| label_propagation | 0.8719 | 0.8715 | — | — | 1.00 |
| full_supervised_electra (ELECTRA-small, 100% labels) | — | — | 0.4124 | 0.3479 | — |
| weak_supervision | 0.4644 | 0.4444 | — | — | 0.7884 |

(16-class chance baseline ≈ 6.25%.)

**Pseudo-labeling (DistilBERT) is the best overall method**, reaching
**88.7% test accuracy / 88.7% Macro F1** at full scale — *surpassing* the
full-supervised DistilBERT baseline (85.2%). This is because pseudo-labeling
absorbs high-confidence predictions from the full unlabeled pool (three
iterations, 98.3% final coverage), effectively training on far more than
the 192-row seed used by the baseline. Label quality on absorbed pseudo-
labels is 90.7% accurate (0.9098 Macro F1).

**Label propagation** at full scale reaches **87.2% label accuracy / 87.2%
Macro F1, 100% coverage** — a massive jump from 35.3%/33.6% at the 1/class
seed. MiniLM embedding-based label spreading builds a strong k-NN graph
with 12 seed examples per class.

**Weak supervision** reaches 46.4% label accuracy / 44.4% Macro F1 at full
scale, up from 6.3%/0.7% at 1/class seed. The auto-derived TF-IDF keyword
functions are considerably more discriminative with 12 seed documents per
class. Coverage is 78.8% — some rows receive no confident label vote.

**ELECTRA-small full supervised** (41.2% test accuracy) underperforms
DistilBERT substantially. ELECTRA-small is a much smaller model for sequence
classification on this 16-class task.

### 5.3 Prior results (capped 399-row split, 1/class seed) — for comparison

| Method | Label Accuracy | Label Macro F1 | Test Accuracy | Test Macro F1 | Coverage |
|---|---|---|---|---|---|
| full_supervised (DistilBERT) | — | — | 0.6125 | 0.5924 | — |
| label_propagation | 0.3531 | 0.3362 | — | — | 1.00 |
| pseudo_labeling (DistilBERT) | — | — | 0.1250 | 0.0630 | 0.00 |
| full_supervised_electra (ELECTRA-small) | — | — | 0.0750 | 0.0218 | — |
| pseudo_labeling_electra (ELECTRA-small) | — | — | 0.0625 | 0.0247 | 0.00 |
| weak_supervision | 0.0627 | 0.0075 | — | — | 1.00 |

These results reflect the failure mode of an extremely thin seed (exactly 1
labeled example per class), not the ceiling of the semi-supervised methods
— see Section 6 for the full pseudo-labeling loop story at 1/class.

## 6. Per-Loop Results — Pseudo-Labeling (Self-Training)

### 6.1 Full-scale run (12/class seed) — complete

Both models ran at full scale (labeled seed ~12/class, confidence threshold
0.80) and successfully absorbed pseudo-labels across multiple iterations —
a complete contrast to the 1/class stall described in Section 6.2.

#### DistilBERT (`05_pseudo_labeling.ipynb`, full-scale run)

Confidence threshold: 0.80. 3 iterations to reach 98.3% coverage.

| Round | New pseudo-labels absorbed | Cumulative coverage |
|---|---|---|
| 0 | 331 (41.4% of pool) | 91.4% |
| 1 | 43 (5.4%) | 96.8% |
| 2 | 12 (1.5%) | 98.3% |

Final test accuracy: **88.7%** / Macro F1 **0.887**. Label quality on
absorbed pseudo-labels: 90.7% accurate.

#### ELECTRA-small (`05b_pseudo_labeling_electra.ipynb`, full-scale run)

Confidence threshold: 0.80. 2 iterations to reach 100% coverage.

| Round | New pseudo-labels absorbed | Cumulative coverage |
|---|---|---|
| 0 | 341 (42.6% of pool) | 92.6% |
| 1 | 59 (7.4%) | 100.0% |

Final test accuracy: **83.9%** / Macro F1 **0.837**. Label quality:
82.3% accurate.

With 12 labeled examples per class, both models reach confident softmax
outputs on the unlabeled pool in round 0, absorbing >40% of the pool
immediately. The self-training loop works as designed at this seed size.

### 6.2 Prior run (1/class seed) — stalled completely

At the 1-per-class seed (399-row capped split), both models stalled at
**iteration 0** every single attempt, with 0 new pseudo-labels absorbed
regardless of threshold. This is the most important finding from that run:
**a 1-example-per-class seed cannot support self-training at 16 classes.**

#### DistilBERT (`05_pseudo_labeling.ipynb`, prior run)

Thresholds tried: 0.80, 0.50, 0.35, 0.25 — every one absorbed 0 new labels.

| Round | New pseudo-labels absorbed | Cumulative coverage |
|---|---|---|
| 0 | 0 | 5.0% (16/319, seed only) |

Final test accuracy: 12.5% / Macro F1 0.063.

#### ELECTRA-small (`05b_pseudo_labeling_electra.ipynb`, prior run)

Thresholds tried: 0.50, 0.35 — every one absorbed 0 new labels.

| Round | New pseudo-labels absorbed | Cumulative coverage |
|---|---|---|
| 0 | 0 | 5.0% (16/319, seed only) |

Final test accuracy: 6.25% (exact chance) / Macro F1 0.025.

**Why this happened**: with only 1 labeled example per class, the round-0
fine-tuned classifier's softmax outputs stay near-uniform across 16 classes,
and even a very low confidence threshold (0.25–0.35) can't clear a
distribution that flat. This contrasts with the archived AG News document's
tuning story, whose thousands-of-rows seed gave its round-0 classifier real
signal to build confident predictions from.

## 7. Sample Generated Labels

### 7.1 Unsupervised (`minilm_kmeans`, real rows with generated summary)

| Text (truncated) | Generated summary (truncated) | Predicted | True |
|---|---|---|---|
| Hackers Breach Computer Networks Of Some Big U.S. Law Firms... | Hackers Breach Computer Networks Of Some Big U.S. Law Firms: Report. Federal investigators are looking to see if confidential information was stolen... | TECH | TECH |
| Pope Francis, In Easter Address, Says 'Defenseless' Being Killed... | Pope Francis calls for peace in the Holy Land two days after 15 Palestinians were killed on the Israeli-Gaza border... | POLITICS | RELIGION |
| FBI to meet with Florida officials on election hacking... | FBI officials are set to meet with Florida Gov. Ron DeSantis and US Sen. Rick Scott about concerns that Russians hacked at least one Florida county... | TECH | POLITICS |
| Google Features 12 Female Artists To Celebrate International Women's Day... | Google Features 12 Female Artists To Celebrate International Women's Day. Each illustration features a personal story... | ARTS & CULTURE | WOMEN |

(Full 319 rows: `results/full_labels_minilm_kmeans.csv`.)

### 7.2 Full-supervised (DistilBERT, `results/sample_labels_full_supervised.csv`)

| Text (truncated) | Predicted | True | Correct |
|---|---|---|---|
| At Tribeca: Wondrous Boccaccio... | ARTS & CULTURE | ARTS & CULTURE | True |
| Another Gun Just Went Off In A School... | CRIME | CRIME | True |
| Boris Johnson is spoiling for a fight - CNN... | NEWS | NEWS | True |
| Former Playboy Model Accuses Oliver Stone Of Groping Her Breast... | MEDIA | WOMEN | False |
| Are You A Gamer Who's The Victim Of A Harassment Campaign?... | TECH | WOMEN | False |

### 7.3 Label propagation (`results/sample_labels_label_propagation.csv`)

| Text (truncated) | Predicted | True | Correct | Confidence |
|---|---|---|---|---|
| Colbert Exposes The Biggest Flaw In Trump's Latest Conspiracy Theory... | COMEDY | COMEDY | True | 0.958 |
| Pfizer says its vaccine is 90.7% effective... | HEALTH | HEALTH | True | 0.917 |
| Richard Branson: All cars will be electric by 2030... | BUSINESS | SPORTS | False | 0.592 |
| Don't Tell Model Winnie Harlow She's 'Suffering' From Vitiligo... | ARTS & CULTURE | WOMEN | False | 0.642 |

### 7.4 Weak supervision (`results/sample_labels_weak_supervision.csv`)

| Text (truncated) | Predicted | True | Correct | Confidence |
|---|---|---|---|---|
| Animal Photos Of The Week: Monkeys, Giraffes, Pandas... | ARTS & CULTURE | ENVIRONMENT | False | 0.081 |
| Stephen Colbert Mocks Proposal To Hang Trump And Pence Portraits... | CRIME | COMEDY | False | 0.101 |
| Sarah Silverman's 'SNL' Promos Are Adorable... | WOMEN | COMEDY | False | 0.075 |

All five sampled rows in this method's qualitative table are incorrect —
consistent with its 6.27% (chance-level) label accuracy in Section 5.2.

### 7.5 Pseudo-labeling test set (`results/sample_labels_pseudo_labeling_test.csv`)

| Text (truncated) | Predicted | True | Correct |
|---|---|---|---|
| Unarmed Hotel Security Guard Who Found Las Vegas Shooter Hailed As Hero... | CRIME | CRIME | True |
| Uses For Dental Floss: 12 Quick Tricks Around The House... | HEALTH | ENVIRONMENT | False |
| 'Pokémon Go' Just Added Billions Of Dollars To Nintendo's Value... | WOMEN | TECH | False |

These predictions come purely from the round-0 fine-tune on the 16-row
seed (no pseudo-labels were ever absorbed into training) — see Section 6.

## 8. Full-Output CSV Convention

Every method in this pipeline saves two artifacts to `results/`:

- **`sample_labels_<method>.csv`** — a small qualitative sample (roughly one
  row per class) for quick human eyeballing.
- **`full_labels_<method>.csv`** — every row the method scored, columns
  `text, summary, predicted_label, true_label[, confidence]` (unsupervised
  methods carry `summary` instead of `confidence`; pseudo-labeling saves
  separate `_train_pool` and `_test` variants since it labels both).

`results/metrics_<method>.json` carries that method's numeric metrics (and,
for pseudo-labeling, the per-round `history` list transcribed in Section 6).
`results/comparison_table.csv` aggregates every method's headline metrics
into one row-per-method table (Section 5's tables are pulled directly from
it).

## 9. Limitations / Next Steps

- **k=16 assumed, not swept.** Every KMeans run above fixes `k=16` because
  that's the known class count — a real deployment without ground truth
  would need to **sweep k** and choose it by an unsupervised metric (e.g.
  silhouette) rather than assuming it. Deferred to future work, as in the
  archived AG News document.
- **Labeled-seed thinness (resolved in full-scale re-run).** The three
  successive scope reductions (4,781 → 1,600 → 800 → 399 rows) in the
  initial run shrank the 5% labeled seed to exactly 1 example per class —
  directly responsible for weak supervision collapsing to chance and
  pseudo-labeling stalling at round 0 (see Section 5.3 and 6.2). The
  full-scale re-run (2026-09-06) addresses this by running the
  semi-supervised and baseline track on all ~4,781 rows with a 12/class
  seed. Label propagation already shows the expected improvement (49% vs.
  35% accuracy); full-scale pseudo-labeling and baseline results are pending
  (Section 5.2).
- **BERTopic's noise-topic coverage (~50%) is a small-corpus limitation.**
  At 399 rows for 16 classes, several classes don't have enough documents
  to form their own density peak; BERTopic found only 6 of 16 topics as a
  result. A larger corpus (see above) would likely resolve this too.
- **HDBSCAN's fixed `min_cluster_size=50`** exceeds the ~20-row true
  cluster size at this scale for all three embeddings, degenerating to
  100% noise. Would need retuning (or a data-size-aware default) to be
  useful on a corpus this small.
- **OpenAI embeddings are pending** — `openai_kmeans` / `openai_hdbscan` are
  wired up in the pipeline but not run in this environment (no API key
  configured).
- **Semantic-closeness scoring** (comparing a generated free-text label to
  the true class name by embedding similarity, rather than only exact
  match) remains a deferred future item, as noted in the archived AG News
  document — every method here still outputs one of the 16 fixed class
  names, so exact-match accuracy stays well-defined for now.

## 10. Repo Map

```
data/
  master_data.csv          # 16-class dataset, 4800 rows (300/class)
  raw/                      # source files feeding 00_data_transform.ipynb
  processed/
    train_clean.parquet     # 319 rows, capped (unsupervised track)
    test_clean.parquet      # 80 rows, capped (unsupervised track)
    labeled.parquet         # 16-row (1/class) seed — capped track
    unlabeled.parquet       # 303-row unlabeled pool — capped track
    train_full.parquet      # 3824 rows, full-scale (semi-supervised track)
    test_full.parquet       # 957 rows, full-scale
    labeled_full.parquet    # 192-row (~12/class) seed — full-scale track
    unlabeled_full.parquet  # 3632-row unlabeled pool — full-scale track
    _summary_cache.json     # durable content-keyed summary cache, never delete

notebooks/
  00_data_transform.ipynb          # raw sources -> master_data.csv
  01_embeddings.ipynb               # TF-IDF/MiniLM/RoBERTa/OpenAI embeddings, cached
  02_unsupervised_clustering.ipynb  # KMeans(k=16)/HDBSCAN over each embedding
  03_bertopic.ipynb                 # BERTopic topic modeling
  04_weak_supervision.ipynb         # auto-derived Snorkel labeling functions
  05_pseudo_labeling.ipynb          # DistilBERT self-training loop
  05b_pseudo_labeling_electra.ipynb # ELECTRA-small self-training loop
  06_full_supervised_baseline.ipynb        # DistilBERT, 100% train labels
  06b_full_supervised_baseline_electra.ipynb # ELECTRA-small, 100% train labels
  07_comparison.ipynb               # aggregates results/comparison_table.csv
  08_label_propagation.ipynb        # MiniLM k-NN graph + LabelSpreading

utils/
  config.py             # paths, class names, seed, sample-size knobs
  data.py               # loading/cleaning/splitting
  embeddings.py         # TF-IDF/MiniLM/RoBERTa/OpenAI embedding helpers
  summarization.py      # bart-large-cnn summarize/title generation only
                         # (zero-shot classification pieces removed with
                         # the AG News-era summarization_labeling notebook)
  weak_supervision.py   # auto-derived per-class TF-IDF labeling functions
  label_propagation.py  # k-NN graph + LabelSpreading
  modeling.py           # fine-tuning helpers (DistilBERT/ELECTRA-small)
  interpretability.py   # cluster-term extraction (KeyBERT-based)
  metrics.py            # shared metric computation
  samples.py            # sample/full-label CSV writers

results/
  comparison_table.csv           # one row per method, all headline metrics
  metrics_<method>.json          # per-method metrics (+ history for pseudo-labeling)
  sample_labels_<method>.csv     # small qualitative sample
  full_labels_<method>.csv       # every row scored by that method
  confusion_matrix_<method>.png  # supervised/pseudo-labeling confusion matrices
  comparison_bar_chart.png       # visual summary across methods

docs/
  PROJECT_SUMMARY_AGNEWS_ARCHIVE.md  # retired AG News (4-class) version
  semi_supervised_methods.md         # candidate/decision log for semi-supervised methods
```
