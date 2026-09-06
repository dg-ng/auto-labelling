# Additional Semi-Supervised Methods — Candidates

Options to extend `04_weak_supervision.ipynb` / `05_pseudo_labeling.ipynb`
beyond the current weak-supervision (Snorkel) and pseudo-labeling
(DistilBERT self-training) approaches. Only **label propagation** has been
built out so far (`utils/label_propagation.py`, `notebooks/08_label_propagation.ipynb`
— 87.2% label accuracy, 100% coverage, no fine-tuning loop). The rest are
still candidates; pick one (or a few) and it can be built out as a new
`notebooks/0X_*.ipynb` + `utils/` additions, following the same pattern.

Original weak spot this list responded to: a **fixed** confidence threshold
(0.90) stalled completely at small sample sizes (`CLASSIFIER_SAMPLE_SIZE=150`
absorbed 0 pseudo-labels across 3 rounds). **Resolved directly in
`05_pseudo_labeling.ipynb` (2026-08-22/23)** without needing a new method:
lowering the threshold to 0.60 confirmed it was the bottleneck, and
subsequent tuning (0.80 threshold + 0.98 target coverage at a 400-row
sample) produced a clean 3-round self-training curve (331 → 43 → 12
pseudo-labels absorbed) at 88.7% test accuracy — see `PROJECT_SUMMARY.md`
Section 5 for the full tuning story. FlexMatch/SetFit below remain
un-built alternatives if a smarter per-class threshold is wanted later.

## Options

| Method | Idea | Why it'd fit this project | Rough cost |
|---|---|---|---|
| **SetFit** | Contrastive-fine-tune a sentence-transformer on pairs drawn from the tiny labeled seed, then a lightweight classifier head on top — no large-batch fine-tuning loop | Purpose-built for "few labels, small model"; likely beats DistilBERT self-training at 5% labels, and doesn't depend on a brittle confidence threshold | Low — single short training pass, CPU-friendly |
| **FlexMatch / curriculum pseudo-labeling** | Per-class adaptive confidence threshold instead of one fixed 0.90 (classes the model is currently worse at get a lower bar) | Directly fixes the "threshold too strict → nothing absorbed" stall seen in the dev-scale run | Low — same loop, smarter threshold logic |
| **Noisy Student** | Same self-training loop, but inject data augmentation / dropout noise into the student each round, optionally growing model capacity | More robust to pseudo-label noise than plain self-training | Medium — needs an augmentation step |
| **Label propagation / label spreading** (`sklearn.semi_supervised`) — **✅ built, `08_label_propagation.ipynb`** | Build a k-NN graph over embeddings (already computed in `01_embeddings.ipynb`) and propagate the 5% seed labels through the graph algebraically — no fine-tuning loop at all | Cheap, fast, reuses existing embedding cache; good complementary baseline against pseudo-labeling/weak supervision | Low — no model training, just graph algebra |
| **Co-training / tri-training** | Train 2–3 classifiers on different "views" (e.g. TF-IDF vs. MiniLM embeddings) and let them pseudo-label for each other, keeping only points where they agree | Uses embedding variety already in the repo; agreement-based acceptance is a natural confidence proxy that needs no ground truth | Medium — multiple models trained per round |
| **Consistency regularization (UDA-style)** | Penalize the model for disagreeing between an example and an augmented/back-translated version of it | Sidesteps the confidence-threshold problem entirely — no accept/reject step | Medium-high — needs a text augmentation pipeline |
| **Ensemble weak supervision + pseudo-labeling** | Only accept a self-training pseudo-label when it also agrees with Snorkel's `LabelModel` probabilistic output | Combines the two semi-supervised methods already in the repo instead of treating them as independent comparison-table rows | Low — reuses both existing pipelines |

## Stopping-criterion notes (applies to any method chosen)

- `true_label` is only available here because AG News ships full ground
  truth that's artificially hidden for the experiment — a real deployment
  can't check "% correctly labeled" live. Legitimate no-ground-truth stop
  signals: **convergence** (new pseudo-labels absorbed this round ≈ 0),
  **held-out validation** (small slice carved from the original seed,
  never trained on, checked each round), or **target coverage** (current
  approach in `05_pseudo_labeling.ipynb` as of 2026-08-23: stop at 98% of
  the pool labeled, tuned up from an original 95% specifically so an early
  single-round coverage hit at a loose threshold doesn't mask genuine
  multi-round self-training — see the tuning story above).
- Label-quality-vs-ground-truth (`evaluate_label_quality`) should stay an
  *evaluation-only* metric computed after the loop, not a loop-control
  signal, for any of the options above too.

## Decision

**Label propagation was chosen and built** (`utils/label_propagation.py`,
`notebooks/08_label_propagation.ipynb`) — cheap, reuses existing embedding
infra, and gives a strong complementary baseline (87.2% label accuracy,
100% coverage) against pseudo-labeling. The other options remain un-built
candidates for future work; none are blocked on anything.

## Mentor feedback follow-ups (2026-08-28)

Two additions in response to mentor review, both additive — nothing above
was replaced or re-run.

**(a) ELECTRA-small as an additive comparison model.** Mentor feedback item
1 ("choose another model, we can still compare it with DistilBERT later")
was addressed by running `google/electra-small-discriminator` through the
same pseudo-labeling and full-supervised pipelines as DistilBERT
(`notebooks/05b_pseudo_labeling_electra.ipynb`,
`notebooks/06b_full_supervised_baseline_electra.ipynb`), at matching sample
sizes for each method. See `PROJECT_SUMMARY.md` Section 5 for the full
head-to-head numbers and discussion — short version: DistilBERT wins at
both sample sizes tested, but the gap narrows sharply (44 points → 5
points) as sample size grows, from the 150-row full-supervised regime to
the 800-row pseudo-labeling pool. One notebook-specific wrinkle worth
flagging here since it's a stopping-criterion detail: `05b`'s confidence
threshold had to be retuned down to 0.30 (from `05`'s tuned 0.80) because
ELECTRA-small fine-tuned on only the 400-row labeled seed is far less
confident than DistilBERT at this sample size — its round-0 softmax
confidence on the unlabeled pool tops out around 0.33 empirically, so
DistilBERT's threshold stalls the loop at zero absorbed labels, the same
"model too underconfident to progress" failure mode documented above for
DistilBERT's original 0.90 threshold. This means the two notebooks are
*not* threshold-matched — read ELECTRA-small's self-training numbers as
"calibrated to what this smaller, less-confident model can actually do at
this sample size," not as an apples-to-apples confidence-threshold
comparison with DistilBERT.

**(b) Summarize → zero-shot-classify as a new labeling approach.** Mentor
feedback item 2 was addressed by adding a genuinely different labeling
method rather than another clustering or fine-tuning variant:
`notebooks/09_summarization_labeling.ipynb` summarizes each article with an
open-source summarizer (`sshleifer/distilbart-cnn-6-6`), then zero-shot
classifies the summary into one of the 4 AG News classes with an
NLI-based zero-shot model (`valhalla/distilbart-mnli-12-3`) — no
fine-tuning at all, two off-the-shelf model passes per row. It also
generates a short headline per article for qualitative inspection. Result:
67.5% test accuracy / 64.1% Macro F1 on a 200-row test sample — well below
the fine-tuned methods, which is expected since this approach never sees a
single labeled AG News example. See `PROJECT_SUMMARY.md` Sections 5–6 for
numbers and example rows.

**Deferred future work (explicit scope decision, not built this pass):**
semantic-closeness / embedding-similarity scoring between a generated
free-text label (e.g. the per-article generated headline from notebook 09)
and the true class name, as a complement to exact-match accuracy. The
summarize + zero-shot method's classifier always outputs one of the 4 fixed
class names, so exact-match accuracy is well-defined as-is today — but a
similarity-based score would generalize to any future method that produces
more free-form generated labels. Same note appears in `PROJECT_SUMMARY.md`
Section 7 so both docs agree.

## Dataset pivot to master_data.csv (2026-09-05)

Everything above describes the retired AG News (4-class) version of this
project — see `docs/PROJECT_SUMMARY_AGNEWS_ARCHIVE.md`. The project has
since pivoted to `data/master_data.csv`, a **16-class** dataset (ARTS &
CULTURE, BUSINESS, COMEDY, CRIME, EDUCATION, ENTERTAINMENT, ENVIRONMENT,
HEALTH, MEDIA, NEWS, POLITICS, RELIGION, SCIENCE, SPORTS, TECH, WOMEN),
several of which overlap in subject matter. Three successive user-directed
scope reductions during this rebuild shrank the working dataset from the
full ~4,781 cleaned rows to a final 399 (319 train / 80 test), with a 5%
labeled seed of just 16 rows — 1 example per class.

Two methods changed as a direct result of the class-count and scale change:

- **Weak supervision's labeling functions are now auto-derived, not
  hand-written.** The AG News version hand-wrote one keyword list per class
  (4 classes, easy to author and validate by eye). At 16 overlapping
  classes, hand-authoring a distinct keyword list per class doesn't scale —
  a human author can't reliably pick keywords that separate POLITICS from
  NEWS from MEDIA from WOMEN from CRIME by inspection. `utils/weak_supervision.py`
  instead derives each class's most TF-IDF-distinctive keywords directly
  from the labeled seed (`derive_class_keywords`) and builds one Snorkel
  `LabelingFunction` per class automatically. At the 1-example-per-class
  seed size this rebuild ended up at, the auto-derived keywords pick up
  that single document's idiosyncrasies rather than real class signal —
  weak supervision lands at exactly chance accuracy (6.27% vs. a 6.25%
  16-class baseline). See `PROJECT_SUMMARY.md` Section 5 for the numbers.
- **Pseudo-labeling / self-training stalls completely at this seed size**,
  for both DistilBERT and ELECTRA-small — every confidence threshold tried
  absorbs exactly 0 new pseudo-labels at round 0. This is a much starker
  failure than anything seen in the AG News version (whose thousands-of-rows
  seed produced a real multi-round absorption curve). See
  `PROJECT_SUMMARY.md` Section 6 for the per-round tables (a single stalled
  row per model) and Section 9 for why a less aggressively capped seed size
  would be needed to actually observe these dynamics.

For every real number from this dataset pivot — clustering, weak
supervision, label propagation, pseudo-labeling, and the full-supervised
baseline — see `PROJECT_SUMMARY.md`'s Results section (Section 5) and its
per-loop pseudo-labeling tables (Section 6). This file's role from here on
is unchanged: a candidate/decision log for semi-supervised methods, not a
results doc.
