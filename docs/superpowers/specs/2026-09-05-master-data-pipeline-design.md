# Master-Data Pipeline Redesign — Design Doc

**Date:** 2026-09-05
**Status:** Approved by user in chat; awaiting spec review before implementation planning.

## 1. Goal

Replace the AG News-based pipeline with a new dataset, `data/master_data.csv`
(16 categories, 4,800 rows, curated from `data/raw/` sources), and restructure
the flow so that:

- **Unsupervised track**: raw text → generate a real, meaningful summary
  sentence per article (not keyword lists) → embed the *summaries* → cluster
  → a human inspects each cluster's example summaries/key-phrases and assigns
  it a real category name, scored against the hidden true label.
- **Semi-supervised track**: raw text + a 5%-per-class labeled seed → learn
  to label the rest, targeting ≥98% coverage of the whole pool — but trained
  directly on **raw text**, not summaries (summaries are generated once for
  the whole dataset and joined into semi-supervised methods' output files for
  inspection only, not used as training input).
- **Full supervised baseline**: unchanged in spirit — 100% of labels, upper
  bound reference — now on raw text, 16-way, full data (no sampling cap).

Ground-truth labels are still only used for scoring, never as unsupervised
training input, matching the project's existing evaluation philosophy.

This is a **full replacement** of the AG News pipeline (confirmed with user)
— AG News notebooks/results are superseded, not kept side by side. Existing
notebooks (`00`–`09`, `05b`/`06b`) are rebuilt in place for the new
dataset/flow rather than duplicated.

## 2. Dataset

**`data/master_data.csv`** — 4,800 rows, columns `category, title, summary,
text`.

| Property | Value |
|---|---|
| Classes | 16 (BUSINESS, RELIGION, POLITICS, NEWS, ARTS & CULTURE, COMEDY, ENTERTAINMENT, MEDIA, HEALTH, TECH, CRIME, SCIENCE, EDUCATION, ENVIRONMENT, WOMEN, SPORTS), 300 rows each, balanced |
| Source | Assembled from `data/raw/` (bbc-text.csv, two CNN article dumps, news-article-categories.csv) — provenance/assembly method not re-derived here (already done); this design starts from the finished `master_data.csv` |
| Text used | `title + text` (mirrors old `title + description` convention) |

### Data-quality findings (from inspection) and cleaning decisions

- 40 duplicate `text` values → dropped (keep first occurrence).
- At least one degenerate row (`text` = `"(CNN)"`, effectively empty) → drop
  rows below a minimum word-count threshold (e.g. < 5 words after cleaning).
- The existing `summary` column is populated for only 1,322/4,800 rows,
  concentrated in 6 of 16 categories (artifact of source assembly, e.g. CNN
  rows carrying a pre-existing lede/summary field that other sources lack) →
  **dropped entirely**; every row gets a freshly generated summary via the
  model instead, for consistency across all 16 classes.
- Stratified 80/20 train/test split by `category` → ~3,840 train / ~960 test,
  all 16 classes balanced in both splits (input is exactly 300/class).
- `LABEL_FRACTION = 0.05` of train per class → ~15/class labeled seed
  (~192 total) + ~3,648-row unlabeled pool, feeding the semi-supervised
  methods (weak supervision, label propagation, pseudo-labeling).
- `CLASS_NAMES` = the 16 category strings (as they appear in the CSV,
  uppercase); `NUM_CLASSES = 16`.

### Full-data scope (no more dev-scale sampling)

AG News's sampling caps (`SAMPLE_SIZE=8000`, `CLASSIFIER_SAMPLE_SIZE=150`,
etc.) existed purely to keep CPU-only compute tractable against a
120,000-row pool. At 4,800 rows total, **every method runs on the full
dataset** — no sampling caps, no dev-scale caveats in the results. Machine
is confirmed CPU-only (Intel UHD Graphics 620 integrated GPU, no CUDA/MPS;
`torch==2.9.1+cpu`) but full-data runs at this scale (fine-tuning on ~3,840
rows, or ~192-row seeds for semi-supervised) are a few-minutes-to-tens-of-
minutes job, not the multi-day AG-News-at-full-scale estimate.

## 3. Summarization (shared, unsupervised-track input)

- **Model swap**: `SUMMARIZATION_MODEL_NAME` changes from
  `sshleifer/distilbart-cnn-6-6` to **`facebook/bart-large-cnn`** — the full
  BART-large model the distilled version was distilled from, producing
  fluent, coherent multi-sentence summaries rather than choppier distilled
  output. Slower per row on CPU, but a one-time cost over 4,800 rows, cached
  to disk afterward.
- Every row's `text` is summarized once (full dataset, not just a sample) and
  cached to `data/processed/summaries.parquet` (or a `summary` column joined
  onto the cleaned parquet), keyed by row id, reused by every notebook that
  wants it (unsupervised clustering as training input; semi-supervised
  methods' full-output files for inspection only).
- `generate_titles` (short headline) — kept as-is, same model, for the
  qualitative sample tables.
- `ZERO_SHOT_MODEL_NAME` (`valhalla/distilbart-mnli-12-3`) — retained;
  the summarize+zero-shot method (old notebook `09`) is superseded by the
  new summary-embedding-clustering flow (Section 4) as the unsupervised
  track's summarization-based method; whether `09`'s standalone approach is
  kept as an additional comparison point or retired is an open item for the
  implementation plan to resolve pragmatically (default: retire it, since
  its role is now subsumed by the summary-clustering flow — flag this for
  confirmation during planning if it seems worth keeping as one more data
  point).

## 4. Unsupervised track (`01_embeddings`, `02_unsupervised_clustering`, `03_bertopic`)

- Embed the **generated summaries** (not raw text) with each method: TF-IDF,
  MiniLM (`all-MiniLM-L6-v2`), RoBERTa (frozen, mean-pooled), OpenAI
  (`text-embedding-3-small`, if `OPENAI_API_KEY` present) — full dataset, no
  sampling cap (previous `ROBERTA_SAMPLE_SIZE=500` cap dropped along with the
  others per Section 2).
- Cluster each embedding with **KMeans (k=16)** and **HDBSCAN**. Same metric
  suite as before: Hungarian-matched ACC, Macro F1 (from the same matching),
  NMI, ARI, FMI, homogeneity/completeness/V-measure, silhouette,
  Davies-Bouldin, coverage (fraction HDBSCAN didn't call noise).
  - `silhouette_score`/`davies_bouldin_score` are O(n²) — sub-sample before
    computing them now that full data is actually being used (previously a
    documented-but-unrealized fix; now required).
- **BERTopic** on summaries, forced to 16 topics, `KeyBERTInspired`
  representation model for readable topic labels (kept from the mentor-
  feedback work).
- **k=16 now, sweep later**: k is a parameter, not a hardcoded permanent
  choice. This pass uses k=16 (matching the known class count). A follow-up
  should sweep k and pick the value maximizing clustering-quality metrics
  rather than assuming the true class count is the best cluster count — not
  built this pass, documented as explicit future work (Section 7 of
  `PROJECT_SUMMARY.md`, mirroring the existing "known fix needed" pattern).
- Cluster-naming: per cluster, show example summaries + KeyBERT key-phrases
  (`meaningful_terms_per_cluster`, already exists) so a human can assign a
  real category name; Hungarian-matched majority label used for the
  aggregate metrics, exactly as before.
- **Full-output CSV** per embedding/cluster method (new, in addition to the
  existing small qualitative `results/clusters_*.csv`): one row per input
  document — `text, summary, predicted_label, true_label` — predicted_label
  is the cluster's Hungarian-matched class name. Saved to e.g.
  `results/full_labels_<method>.csv`.

## 5. Semi-supervised track (`04_weak_supervision`, `05`/`05b_pseudo_labeling`, `08_label_propagation`)

All three trained/evaluated on **raw text**, full data, 16-way. All three
kept (not consolidated to pseudo-labeling only), per user's explicit
decision — this widens the comparison but each method needs re-validation
at 16 classes / thin seed (~15/class), which is expected to be harder than
AG News's 4-class / 1,500-per-class seed.

- **Weak supervision**: per class, auto-derive a distinctive keyword list
  from that class's 5% labeled-seed docs via TF-IDF (top-N terms distinctive
  to that class vs. the rest of the seed) — a programmatic replacement for
  hand-written rules, scaling to 16 classes without manual authoring. Each
  class's keyword list becomes one labeling function (text contains any term
  → vote for that class); Snorkel's `LabelModel` combines the 16 LFs' votes
  into probabilistic labels, same mechanism as before. Scored against true
  labels (label accuracy, Macro F1, coverage) — expect this to be a genuinely
  harder regime than AG News (16 overlapping classes vs. 4 well-separated
  ones); report honestly, don't force a good number.
- **Label propagation**: MiniLM-embed the 5% seed + unlabeled pool (raw
  text), k-NN graph over the embeddings, `sklearn.semi_supervised.LabelSpreading`
  — same mechanism, 16-way, full unlabeled pool (no `SAMPLE_SIZE` cap).
- **Pseudo-labeling** (self-training): fine-tune on the ~192-row labeled
  seed → predict on the ~3,648-row unlabeled pool → keep high-confidence
  predictions → add to labeled pool → repeat until `target_coverage=0.98`,
  pool exhaustion, a stalled round, or a max-iteration safety cap — run for
  **both** DistilBERT (`05_pseudo_labeling.ipynb`) and ELECTRA-small
  (`05b_pseudo_labeling_electra.ipynb`), full unlabeled pool (no
  `PSEUDO_LABEL_SAMPLE_SIZE` cap). Confidence threshold/target coverage may
  need retuning at this much smaller, 16-way seed — reuse the same tuning
  discipline documented in `PROJECT_SUMMARY.md`'s existing tuning story
  (start from the previously-tuned 0.80/0.98 values, adjust based on what
  actually happens in round 0, document the tuning story again if it
  changes materially).
- Each method saves both the existing small qualitative sample CSV
  (`save_label_samples`, 2 examples/class) and a new **full-output CSV**:
  `text, summary, predicted_label, true_label` — `summary` is joined in from
  Section 3's generated summaries even though these methods don't train on
  it, purely so every method's full-output file has a consistent schema for
  side-by-side inspection.

## 6. Full supervised baseline (`06`/`06b_full_supervised_baseline`)

- DistilBERT (`06`) and ELECTRA-small (`06b`), both trained on 100% of the
  ~3,840-row train set (no `CLASSIFIER_SAMPLE_SIZE` cap), evaluated on the
  full ~960-row test set, 16-way. Same full-output CSV convention as
  Section 5.

## 7. Comparison & documentation (`07_comparison`, `PROJECT_SUMMARY.md`)

- `07_comparison.ipynb` needs no structural changes — it already globs
  `results/metrics_*.json`; it will pick up the new 16-way numbers
  automatically. Re-run to regenerate `results/comparison_table.csv` /
  `results/comparison_bar_chart.png`.
- `PROJECT_SUMMARY.md` gets a substantial rewrite (not incremental patch):
  - Dataset section: `master_data.csv` provenance, 16 classes, cleaning
    steps (Section 2 above).
  - Pipeline section: summarize-first flow for the unsupervised track,
    raw-text flow for semi-supervised/baseline, updated repo map (model
    swap to `facebook/bart-large-cnn`, any notebook retirement decisions
    from Section 3).
  - Full-data note replacing the old sampling-cap caveats/table.
  - New results tables for all methods at 16-way, full-data.
  - k=16-now/k-sweep-later noted as future work (Section 7 of the doc).
  - Weak supervision's auto-derived-LF approach documented in place of the
    old hand-written-rules description.
  - Full-output CSV convention documented (schema, one file per method,
    where they live).

## 8. Open items for the implementation plan to resolve

- Whether old notebook `09` (summarize+zero-shot) is retired or kept as an
  additional unsupervised comparison point alongside the new summary-
  clustering flow (default: retire — see Section 3).
- Exact minimum word-count threshold for dropping degenerate rows (Section
  2) — pick a reasonable value (e.g. 5 words) and document it, not a
  user-facing decision.
- Confidence-threshold/target-coverage values for pseudo-labeling at the new
  16-way/thin-seed regime (Section 5) — start from prior tuned values,
  adjust based on observed round-0 behavior, following the existing
  tuning-story pattern.
