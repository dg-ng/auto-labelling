# Mentor Feedback Implementation: Design

**Date:** 2026-08-26
**Status:** Approved
**Source:** Mentor feedback (4 points), relayed 2026-08-26

## Purpose

Address 4 pieces of mentor feedback on the AG News auto-labeling project:

1. Add a semi-supervised classifier model outside the BERT family (not
   DistilBERT), as a comparison point alongside the existing DistilBERT runs.
2. Add a new labeling approach: summarize raw text, then generate a label
   from the summary.
3. Compare generated labels against true labels using a library/tool.
4. Generate meaningful cluster labels/titles (not just top words) — for
   whole clusters, and per-document titles.

All 4 are addressed as a coordinated set of additions layered onto the
existing 9-notebook pipeline (see `PROJECT_SUMMARY.md` §3, §8), without
disturbing the existing DistilBERT results — everything is additive so old
and new numbers sit side by side for comparison.

## Constraints carried over from the base project

- CPU-only, no GPU (`PROJECT_SUMMARY.md` §3). Every new model choice below
  is picked for CPU feasibility, not best possible quality.
- `SEED = 42` everywhere for reproducibility (`utils/config.py`).
- Existing pattern: shared `utils/` package + one notebook per
  method/pipeline stage, each caching expensive outputs to disk
  (`embeddings_cache/`, `results/`).
- `PROJECT_SUMMARY.md` and `docs/semi_supervised_methods.md` are living
  documents updated alongside each change (see git history) — this work
  follows that convention.

## Item 1: ELECTRA-small as an additional semi-supervised model

**Not a replacement.** DistilBERT stays as-is (`05_pseudo_labeling.ipynb`,
`06_full_supervised_baseline.ipynb`, `CLASSIFIER_MODEL_NAME` unchanged) so
existing results remain comparable, per explicit instruction: "I think
google electra small discrimination is an addition we can still compare it
with distilbert later."

- `utils/config.py`: add `CLASSIFIER_MODEL_NAME_ALT =
  "google/electra-small-discriminator"` (~14M params). ELECTRA's
  replaced-token-detection pretraining is architecturally distinct from
  BERT/DistilBERT/RoBERTa's masked-LM objective — satisfies "not BERT
  family." Small size keeps CPU fine-tuning time comparable to or faster
  than DistilBERT-base (66M params).
- New notebooks, parallel to the existing ones, calling the same
  `utils/modeling.py` functions (`train_model`, `pseudo_label_loop`) with
  `model_name=CLASSIFIER_MODEL_NAME_ALT`:
  - `notebooks/05b_pseudo_labeling_electra.ipynb`
  - `notebooks/06b_full_supervised_baseline_electra.ipynb`
  - Same sample-size caps as their DistilBERT counterparts
    (`CLASSIFIER_SAMPLE_SIZE`, `PSEUDO_LABEL_SAMPLE_SIZE`) so the two model
    rows are apples-to-apples.
- `utils/modeling.py` needs no code changes — `model_name` is already a
  parameter on every function.
- `07_comparison.ipynb`: add "Pseudo-labeling (ELECTRA-small)" and "Full
  supervised (ELECTRA-small)" rows alongside the existing DistilBERT rows.

## Item 2: Summarize → zero-shot label

A new labeling method: summarize each article, then zero-shot-classify the
summary into one of the 4 AG News classes. Produces a real class
prediction, directly scorable against ground truth like every other method
in the comparison table — chosen over free-text keyword extraction because
it fits the existing scoring pipeline (`utils/metrics.py`) without new
machinery.

- New `utils/summarization.py`:
  - `summarize_texts(texts, model_name=SUMMARIZATION_MODEL_NAME,
    max_length=...)` — wraps `transformers.pipeline("summarization")`
    with a CPU-friendly distilled model, `sshleifer/distilbart-cnn-6-6`.
  - `zero_shot_label(texts, class_names, model_name=ZERO_SHOT_MODEL_NAME)`
    — wraps `transformers.pipeline("zero-shot-classification")` with a
    distilled NLI model, `valhalla/distilbart-mnli-12-3`, run against
    `CLASS_NAMES`. Returns predicted label + score per row.
  - `generate_titles(texts, model_name=SUMMARIZATION_MODEL_NAME,
    max_length=<short>)` — same summarizer, a much shorter `max_length`,
    to produce a generated headline per article (reused by item 4's
    per-document title output — see below).
- `utils/config.py`: add `SUMMARIZATION_MODEL_NAME`, `ZERO_SHOT_MODEL_NAME`,
  and a new sample-size cap `SUMMARIZATION_SAMPLE_SIZE` (both pipelines are
  CPU-heavy chained transformer calls; expect this cap to be small relative
  to `SAMPLE_SIZE` — actual value determined empirically during
  implementation, following the project's existing pattern of measuring
  throughput on this machine before fixing a cap, per
  `utils/config.py`'s existing comments on `ROBERTA_SAMPLE_SIZE` /
  `CLASSIFIER_SAMPLE_SIZE`).
- New `notebooks/09_summarization_labeling.ipynb`:
  1. Sample `SUMMARIZATION_SAMPLE_SIZE` rows from the test set (has ground
     truth, matches the pattern of other scored methods).
  2. `summarize_texts()` → `zero_shot_label()` → `generate_titles()`.
  3. Score predicted vs. true label with existing
     `utils/metrics.py` machinery.
  4. Save `results/sample_labels_summarization.csv` (text, summary,
     generated_title, predicted_label, true_label, correct, confidence) —
     same shape as existing `results/sample_labels_*.csv` files, via
     `utils/samples.py`'s `save_label_samples` (extended if needed to carry
     the extra `summary`/`generated_title` columns).
- `07_comparison.ipynb`: add a "Summarization + Zero-shot" row.

## Item 3: Label-closeness comparison — deferred

Clarified in discussion: this isn't exact-match scoring (already covered
by existing metrics) but *semantic* closeness — does a generated free-text
output mean the same thing as the true category, even if not a literal
match. Explicitly deferred: "we can temporarily skip this I can examine
the closeness by looking at it."

No code this pass. Documented as future work in both
`docs/semi_supervised_methods.md` and `PROJECT_SUMMARY.md`'s Limitations
section: a candidate approach would be embedding-similarity scoring
(cosine similarity between a generated label's embedding — reusing the
project's existing MiniLM infra — and the true class name's embedding),
as a complement to exact-match accuracy.

## Item 4: Meaningful cluster labels + per-document titles

**Cluster-level labels.** Two separate mechanisms, because BERTopic's
representation-model machinery only plugs into BERTopic's own pipeline and
can't be dropped onto arbitrary KMeans/HDBSCAN cluster assignments without
significant internal-API shimming:

- `notebooks/03_bertopic.ipynb`: add `representation_model=
  KeyBERTInspired()` to the `BERTopic()` constructor. Native support,
  replaces raw c-TF-IDF top-word topic labels with meaningful key-phrases.
- `utils/interpretability.py`: new `meaningful_terms_per_cluster(texts,
  cluster_labels, embedding_model, n_terms=10, exclude_noise=True)` using
  the `keybert` library (`KeyBERT(model=<existing cached MiniLM
  SentenceTransformer>)`), run against each cluster's concatenated/sampled
  text. Works for any clustering method (02, 08), not just BERTopic — a
  drop-in richer alternative to the existing `top_terms_per_cluster()`
  (kept, not removed, so old TF-IDF-word behavior remains available/
  comparable). `summarize_clusters()` updated to call
  `meaningful_terms_per_cluster()` instead of `top_terms_per_cluster()` for
  its `top_terms` column.
- `notebooks/02_unsupervised_clustering.ipynb` and
  `notebooks/08_label_propagation.ipynb`: re-run the cluster-summary cells
  that call `summarize_clusters()` to pick up the new KeyBERT-based labels
  in `results/clusters_*.csv`.

**Per-document titles.** Not a separate pipeline — per explicit
instruction ("You can put the generated title with the summarization
text"), the generated title lives alongside its summary in
`09_summarization_labeling.ipynb`'s output (`generated_title` column next
to `summary`, described under item 2 above).

## New dependency

`keybert` added to `pyproject.toml`. Summarization and zero-shot
classification use `transformers.pipeline(...)`, no new dependency beyond
the already-present `transformers`.

## Non-goals

- No change to the existing DistilBERT-based results in
  `05_pseudo_labeling.ipynb` / `06_full_supervised_baseline.ipynb` — item 1
  is additive only.
- No semantic-closeness/embedding-similarity scoring implementation this
  pass (item 3, deferred — see above).
- No full-data final run scope change — this work stays at whatever
  dev-scale sampling convention the base project uses; the existing
  full-data final-run question (`docs/superpowers/plans/
  2026-08-15-autolabel-notebooks.md`, "Final Run" section) is untouched
  and out of scope here.

## Verification

- Re-run `05b`, `06b` unattended and confirm they complete within a
  reasonable budget (expect comparable to or faster than DistilBERT's
  existing runtime, given ELECTRA-small's smaller size) — record actual
  wall-clock time.
- Run `09_summarization_labeling.ipynb` end-to-end on its sample cap;
  confirm `results/sample_labels_summarization.csv` is produced with all
  expected columns and non-trivial (not-all-identical) predictions.
- Re-run `03_bertopic.ipynb` and confirm topic labels are readable
  key-phrases, not raw single TF-IDF words.
- Re-run the cluster-summary cells in `02_unsupervised_clustering.ipynb`
  and `08_label_propagation.ipynb`; confirm `results/clusters_*.csv` show
  KeyBERT-based `top_terms`.
- Re-run `07_comparison.ipynb`; confirm `results/comparison_table.csv`
  gains the 3 new rows (ELECTRA pseudo-labeling, ELECTRA full-supervised,
  summarization+zero-shot) without losing any existing rows.
- Update `PROJECT_SUMMARY.md` (§4 Approaches, §5 Results, §6 Sample
  Generated Labels, §8 Repo Map) and `docs/semi_supervised_methods.md`
  (item-3 future-work note) to reflect all of the above.
