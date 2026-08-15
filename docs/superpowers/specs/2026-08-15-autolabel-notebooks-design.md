# Auto-Labeling Notebooks: Design

**Date:** 2026-08-15
**Status:** Approved
**Source spec:** `autolabel_project_spec.md` (project-level approach, code snippets, metrics)

## Purpose

Turn `autolabel_project_spec.md` into a runnable notebook-based project comparing
unsupervised and semi-supervised auto-labeling methods on AG News, under real
constraints: local CPU-only execution, and code shared across notebooks instead
of duplicated.

## Deviations from `autolabel_project_spec.md`

1. **Data loading fix.** The data files (`data/train.csv`, `data/test.csv`) have
   a header row (`Class Index,Title,Description`). The spec's loader uses
   `header=None`, which would treat that header as a data row. Notebooks use
   `header=0` and rename columns.
2. **No `classes.txt`.** The spec's file tree references it; it doesn't exist in
   `data/`. Not needed — `CLASS_NAMES` is hardcoded in `utils/config.py`.
3. **DistilBERT instead of RoBERTa-base** for the pseudo-labeling self-training
   loop (`05`) and the supervised baseline (`06`). Fine-tuning RoBERTa-base for
   3 epochs × 3 self-training iterations on CPU is too slow for iterative
   development. `utils/config.py` exposes `CLASSIFIER_MODEL_NAME` as a single
   constant — swapping back to `roberta-base` on a GPU machine is a one-line
   change. BERT/RoBERTa is still used (frozen, no fine-tuning) as one of the
   embedding methods in `01_embeddings`, where it's much cheaper.
4. **Sample-size toggle instead of fixed subsampling.** `utils/config.py` sets
   `SAMPLE_SIZE = 8000` for development. Every notebook samples via
   `utils.data.stratified_sample(df, SAMPLE_SIZE, seed=SEED)`, which returns
   the full `df` unchanged when `SAMPLE_SIZE is None`. The final comparison run
   sets `SAMPLE_SIZE = None` and reruns the heavy notebooks (`01`, `05`, `06`)
   unattended.

## Non-goals

- No web app, API, or serving layer — this is an evaluation/comparison project.
- No CI/automated test suite — correctness is enforced by in-notebook sanity
  assertions (see below), consistent with a data-science notebook workflow.
- LLM cluster-naming (`llm_name_cluster` in the source spec) is a documented
  stretch goal in `02_unsupervised_clustering`, not a required deliverable.

## Project Structure

```
final-project/
├── data/
│   ├── train.csv                  # existing, raw
│   ├── test.csv                   # existing, raw
│   └── processed/                 # written by 00_data_transform
│       ├── train_clean.parquet
│       ├── test_clean.parquet
│       ├── labeled.parquet        # semi-supervised labeled pool
│       └── unlabeled.parquet      # semi-supervised unlabeled pool (label = -1)
├── utils/
│   ├── __init__.py
│   ├── config.py                  # CLASS_NAMES, NUM_CLASSES, SEED, SAMPLE_SIZE,
│   │                               # CLASSIFIER_MODEL_NAME, paths, .env loading
│   ├── data.py                    # load_raw(), build_text_column(),
│   │                               # make_splits(), stratified_sample()
│   ├── embeddings.py               # get_tfidf_embeddings(), get_sentence_embeddings(),
│   │                               # get_bert_embeddings(), get_openai_embeddings(),
│   │                               # cache_embeddings(), load_cached_embeddings()
│   ├── metrics.py                  # evaluate_unsupervised(), evaluate_semisupervised(),
│   │                                # evaluate_label_quality(), clustering_accuracy()
│   └── modeling.py                 # train_model(), get_predictions(), pseudo_label_loop() —
│                                    # shared by notebooks 05 and 06 so the fine-tuning
│                                    # loop isn't duplicated between them (planning-stage
│                                    # addition, DRY)
├── embeddings_cache/               # *.npy, gitignored
├── results/                        # metrics json/csv + plots, gitignored except comparison_table.csv
├── notebooks/
│   ├── 00_data_transform.ipynb
│   ├── 01_embeddings.ipynb
│   ├── 02_unsupervised_clustering.ipynb
│   ├── 03_bertopic.ipynb
│   ├── 04_weak_supervision.ipynb
│   ├── 05_pseudo_labeling.ipynb
│   ├── 06_full_supervised_baseline.ipynb
│   └── 07_comparison.ipynb
├── requirements.txt
├── .env                            # OPENAI_API_KEY (gitignored)
└── .gitignore
```

## Notebooks

| Notebook | Input | Output | Notes |
|---|---|---|---|
| `00_data_transform` | `data/train.csv`, `data/test.csv` | `data/processed/*.parquet` | Fix header, build `text` column, 0-index labels, create labeled/unlabeled/unsupervised splits (5% label fraction, stratified, seed 42). Sanity assertions on row counts, nulls, class balance. |
| `01_embeddings` | processed parquet | `embeddings_cache/*.npy` | TF-IDF, MiniLM (sentence-transformers, frozen), RoBERTa/BERT (frozen), OpenAI `text-embedding-3-small`. Cache per method/split/sample-size in filename. |
| `02_unsupervised_clustering` | cached embeddings | metrics json, UMAP plots | KMeans + HDBSCAN per embedding method; Hungarian-matched accuracy, NMI, ARI, FMI, homogeneity, completeness, V-measure, silhouette, Davies-Bouldin, HDBSCAN coverage. LLM cluster-naming as optional stretch cell. |
| `03_bertopic` | processed text | metrics json | BERTopic with MiniLM embedding model, `nr_topics=NUM_CLASSES`. Separate from `02` since it owns its own embedding+clustering. |
| `04_weak_supervision` | processed text | probabilistic + hard labels, LF report | Snorkel labeling functions (keyword + regex heuristics from source spec) → `LabelModel`. Print `LFAnalysis` coverage/conflict/empirical-accuracy before trusting output. Plan B if Snorkel fails to install: weighted-vote combiner in plain pandas, documented inline. |
| `05_pseudo_labeling` | `labeled.parquet`, `unlabeled.parquet` | pseudo-labeled pool, per-iteration metrics | Self-training loop fine-tuning `CLASSIFIER_MODEL_NAME` (DistilBERT). Logs label accuracy + coverage per iteration; asserts no test-set leakage. |
| `06_full_supervised_baseline` | full labeled train | trained model, test metrics | Same architecture as `05`'s classifier, 100% labels — upper bound. |
| `07_comparison` | all `results/*.json` | `results/comparison_table.csv`, bar chart | Asserts every method has the expected metric keys before building the table. |

## Shared `utils/` Package

- **`config.py`** — single source of truth for `CLASS_NAMES`, `NUM_CLASSES`,
  `SEED = 42`, `SAMPLE_SIZE = 8000` (→ `None` for final run),
  `CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"`, path constants
  (`DATA_DIR`, `PROCESSED_DIR`, `CACHE_DIR`, `RESULTS_DIR`), and `.env` loading
  via `python-dotenv` for `OPENAI_API_KEY`.
- **`data.py`** — raw loading with correct header handling, text-column
  construction, stratified label-masking split, and the reusable
  `stratified_sample()` used by every notebook's sampling toggle.
- **`embeddings.py`** — one function per embedding method, each accepting
  texts + a cache key and returning `(embeddings, from_cache: bool)`; callers
  don't need to know the cache path convention.
- **`metrics.py`** — verbatim the metric functions from the source spec
  (`clustering_accuracy` via Hungarian algorithm, `evaluate_unsupervised`,
  `evaluate_semisupervised`, `evaluate_label_quality`), imported by every
  notebook that reports results, so metric definitions can't drift between
  notebooks.

## Caching & Idempotency

Every expensive artifact (embeddings, trained models, pseudo-labels, metric
dicts) is written to disk under a name encoding its config, e.g.
`embeddings_cache/minilm_train_n8000.npy`,
`results/metrics_kmeans_roberta_n8000.json`. Each producing cell checks for
the cache file first and skips recomputation if present — notebooks are safe
to re-run top-to-bottom. `07_comparison` reads only from `results/`.

## Sanity Checks (per notebook)

- `00`: row counts match spec (120,000 train / 7,600 test), no nulls in
  `text`, balanced class distribution, printed sample rows.
- `01`: embedding row count matches input row count, no NaNs, dim printed.
- `02`/`03`: cluster count sanity check; HDBSCAN coverage (fraction not -1)
  reported explicitly, not silently dropped from denominators.
- `04`: `LFAnalysis` coverage/conflict table printed before trusting the
  `LabelModel` output.
- `05`/`06`: per-iteration label accuracy + coverage logged; assertion that
  the pseudo-labeled pool never includes test-set rows.
- `07`: assertion that every method's result dict has all expected metric
  keys before writing the final table.

## Environment

Dependency and environment management uses **uv**, not pip/requirements.txt.
`pyproject.toml` declares dependencies (Snorkel constrains numpy/pandas
compatibility — versions get resolved and pinned in `uv.lock`, which is
committed for reproducibility). Torch is pulled from the PyTorch CPU wheel
index (`tool.uv.sources` / `tool.uv.index`) to avoid downloading the much
larger CUDA build on this CPU-only machine. All commands run via `uv run
...` (e.g. `uv run jupyter nbconvert ...`) rather than activating a venv
manually. `.env` holds `OPENAI_API_KEY`, loaded via `python-dotenv` in
`utils/config.py`; `.gitignore` excludes `.env`, `.venv/`,
`embeddings_cache/`, `data/processed/`, and `results/*` except
`comparison_table.csv`.

No pytest/automated test suite — `utils/` functions are verified with
one-off `uv run python` smoke checks when first written, and are exercised
for real by the notebooks' own inline sanity assertions thereafter.

## Known Risks

1. Snorkel install on Windows can fail due to numpy/pandas pinning conflicts.
   Plan B (weighted-vote combiner) is documented inline in `04` if this
   happens.
2. OpenAI embedding calls cost money and require network access — `01`
   isolates them in their own cache-checked cell so a missing/invalid API key
   doesn't block the rest of the notebook.
3. CPU-only fine-tuning (`05`, `06`) is still slow even with DistilBERT on the
   full dataset — plan to run these unattended for the final (non-sampled)
   run.
