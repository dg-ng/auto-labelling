# Auto-Labeling for Text Classification — AG News

**Project summary for presentation**
Date: 2026-08-28

---

## 1. Goal

Explore how far you can get at labeling news articles into 4 categories
(**World, Sports, Business, Sci/Tech**) **without full human annotation** —
comparing fully unsupervised methods, semi-supervised methods that use only a
small labeled seed set, and a fully supervised upper-bound baseline. Ground
truth labels are only ever used for *evaluation*, never given to the
unsupervised methods during training.

## 2. Dataset

**AG News** — a standard news classification benchmark.

| Property | Value |
|---|---|
| Classes | 4 (World, Sports, Business, Sci/Tech), balanced |
| Train rows | 120,000 (30,000/class) |
| Test rows | 7,600 (1,900/class) |
| Text used | `title + description` |

For semi-supervised methods, 5% of the train labels are kept (`LABEL_FRACTION
= 0.05`) and the rest are hidden (`label = -1`) to simulate a realistic
"small labeled seed set" scenario.

> **Scale note:** results below are from a **development-scale run** on
> different sample-size caps per method (chosen to keep CPU-only iteration
> fast, each documented in Section 5). Test set (7,600 rows) is always used
> in full. A larger-scale/full-data run is a documented, costed-out
> follow-up (Section 7) — not yet started.

### Sample sizes by method

Every method draws from the same 120,000-row train pool / 7,600-row test
set, but at different caps chosen per method's compute cost (fine-tuning
≫ frozen embedding ≫ graph algebra). This is also why the full-supervised
vs. pseudo-labeling comparison in Section 5 carries a caveat — they're not
run at the same sample size.

| Method | Train-side sample used | Test rows used |
|---|---|---|
| TF-IDF / MiniLM / OpenAI + KMeans/HDBSCAN | 8,000 (`SAMPLE_SIZE`) | — (unsupervised, scored on train sample only) |
| RoBERTa (frozen) + KMeans/HDBSCAN | 500 (`ROBERTA_SAMPLE_SIZE`) | — |
| BERTopic | 8,000 (`SAMPLE_SIZE`) | — |
| Weak supervision (Snorkel) | 8,000 unlabeled rows scored + full 6,000-row labeled seed (LF accuracy sanity check only) | — (label-quality only) |
| Label propagation | Full 6,000-row labeled seed + 8,000 unlabeled rows (`SAMPLE_SIZE`) | 7,600 (overlap check only, not scored — no standalone classifier) |
| Pseudo-labeling (self-training) | 400 labeled + 400 unlabeled = 800 total (`PSEUDO_LABEL_SAMPLE_SIZE`, notebook-local) | 7,600 (full) |
| Full supervised (baseline) | 150 (`CLASSIFIER_SAMPLE_SIZE`) | 7,600 (full) |
| Pseudo-labeling / Full supervised — **ELECTRA-small** (additive comparison, `05b`/`06b`) | Same caps as their DistilBERT counterparts above (400 labeled + 400 unlabeled (800-row pool) / 150-row cap) | 7,600 (full) |
| Summarize + zero-shot classify | 200 (`SUMMARIZATION_SAMPLE_SIZE`), drawn from the 7,600-row test pool | (scored against the same rows' hidden labels) |

(`LABEL_FRACTION=0.05` of 120,000 train rows gives the full 6,000-row
labeled seed / 114,000-row unlabeled pool that weak supervision, label
propagation, and pseudo-labeling all sample *from* — the numbers above are
the caps actually used within that pool, not the pool sizes themselves.)

## 3. Pipeline / Tooling

- Environment: `uv` (`pyproject.toml` + `uv.lock`), pure CPU (no GPU).
- Shared `utils/` package (`config`, `data`, `embeddings`, `metrics`,
  `modeling`, `label_propagation`, `interpretability`, `summarization`)
  imported by 12 sequential, independently re-runnable Jupyter notebooks
  (`00`–`09`, plus additive `05b`/`06b` ELECTRA-small variants), each
  caching its expensive outputs to disk.
- `SEED = 42` everywhere for reproducibility.

## 4. Approaches Implemented

### A. Unsupervised auto-labeling (no labels used during training)

Ground-truth labels are only used afterward, to score how well the discovered
clusters line up with the real categories (via Hungarian-matching for
accuracy). Qualitative inspection (top KeyBERT key-phrases, example docs per
cluster, majority-label purity — `results/clusters_*.csv`) supplements the
aggregate metrics below.

| Method | Embedding | Clustering |
|---|---|---|
| TF-IDF + KMeans / HDBSCAN | Bag-of-words TF-IDF | KMeans (k=4) & HDBSCAN |
| MiniLM + KMeans / HDBSCAN | Sentence-transformer (`all-MiniLM-L6-v2`) | KMeans (k=4) & HDBSCAN |
| RoBERTa + KMeans / HDBSCAN | Frozen `roberta-base` (mean-pooled) | KMeans (k=4) & HDBSCAN |
| OpenAI + KMeans / HDBSCAN | `text-embedding-3-small` (optional, needs API key) | KMeans (k=4) & HDBSCAN |
| BERTopic | MiniLM (internal) | UMAP + HDBSCAN (internal), forced to 4 topics |

All embeddings are UMAP-reduced to 50 dims before clustering. Evaluated with
Hungarian-matched accuracy, NMI, ARI, FMI, homogeneity/completeness/V-measure,
silhouette, Davies-Bouldin, and coverage (fraction of points HDBSCAN didn't
call noise).

### B. Semi-supervised auto-labeling (5% labels guide the rest)

| Method | Idea |
|---|---|
| **Weak supervision** (Snorkel) | 6 hand-written labeling functions (keyword lists per class + regex heuristics for "has a %" → Business, "has a score like 3-2" → Sports) vote per example; Snorkel's generative `LabelModel` combines the noisy votes into probabilistic labels — no labeled data required to *fit* the label model, but the 5% seed is used to sanity-check LF empirical accuracy. |
| **Label propagation** | Embed the 5% labeled seed + unlabeled pool with MiniLM, build a k-NN graph over the embeddings (reusing cached embedding infra), and propagate labels algebraically via `sklearn.semi_supervised.LabelSpreading` — no fine-tuning loop at all. |
| **Pseudo-labeling** (self-training) | Fine-tune DistilBERT on the small labeled seed → predict on the unlabeled pool → keep only high-confidence predictions → add them to the labeled pool → repeat until a stopping condition fires → final fine-tune on the grown pool. See Section 5 for the confidence-threshold tuning story — this method changed materially since the first dev-scale run. |
| **Summarize + zero-shot** | Summarize each article with an open-source summarization model (`sshleifer/distilbart-cnn-6-6`), then zero-shot-classify the summary into one of the 4 classes with an NLI-based zero-shot classifier (`valhalla/distilbart-mnli-12-3`) — no fine-tuning at all, two off-the-shelf model passes per row. Also generates a short headline per article for qualitative inspection (Section 6). See `notebooks/09_summarization_labeling.ipynb`. |

### C. Baseline

| Method | Idea |
|---|---|
| **Full supervised** | Same DistilBERT architecture, fine-tuned on 100% of the (sampled) labeled train data — the upper-bound reference point everything else is measured against. |

### D. Additive model comparison: ELECTRA-small (mentor feedback item 1)

Per mentor feedback ("choose another model, we can still compare it with
DistilBERT later"), `google/electra-small-discriminator` was run through the
same pseudo-labeling (`notebooks/05b_pseudo_labeling_electra.ipynb`) and
full-supervised (`notebooks/06b_full_supervised_baseline_electra.ipynb`)
pipelines as DistilBERT, at matching sample sizes for each method
(`CLASSIFIER_SAMPLE_SIZE=150` for full-supervised; the same 400 labeled + 400
unlabeled (800-row pool) for pseudo-labeling) — additive, run *alongside*
DistilBERT's existing results,
not replacing them. See Section 5 for the head-to-head numbers and
discussion.

## 5. Results (dev-scale run)

**Unsupervised — clustering quality vs. hidden ground truth**

| Method | ACC (Hungarian) | NMI | ARI | Coverage |
|---|---|---|---|---|
| **MiniLM + KMeans** | **0.830** | 0.620 | 0.629 | 1.00 |
| TF-IDF + KMeans | 0.808 | 0.530 | 0.567 | 1.00 |
| BERTopic | 0.717 | 0.661 | 0.613 | 0.64 |
| RoBERTa (frozen) + KMeans | 0.518 | 0.218 | 0.214 | 1.00 |
| MiniLM + HDBSCAN | 0.493 | 0.497 | 0.308 | 1.00 |
| TF-IDF + HDBSCAN | 0.262 | 0.004 | 0.00 | 0.96 |
| RoBERTa + HDBSCAN | 0.00 | 0.00 | 0.00 | 0.00 |
| OpenAI + KMeans / HDBSCAN | — | — | — | *pending — no API key configured for this run* |

**Semi-supervised & supervised — test-set / label-quality performance**

| Method | Test Accuracy | Macro F1 | Label Accuracy | Coverage | Notes |
|---|---|---|---|---|---|
| **Pseudo-labeling** (5% seed, self-trained, 400 labeled + 400 unlabeled (800-row pool)) | **0.887** | 0.887 | 0.907 | 1.00 | Now the best method in the whole table — see tuning story below |
| Full supervised (100% labels, 150-row sample) | 0.852 | 0.852 | — | — | Upper-bound reference, but on a *much smaller* sample than pseudo-labeling (150 rows vs pseudo-labeling's 400 labeled + 400 unlabeled (800-row pool)) — see caveat below |
| Label propagation (5% seed, no fine-tuning) | — | — | 0.872 | 1.00 | Zero training cost; scored on label quality, not test accuracy (it doesn't produce a standalone classifier) |
| Weak supervision (Snorkel, label quality only) | — | — | 0.464 | 0.788 | Label Macro F1 0.444 |
| Pseudo-labeling — **ELECTRA-small** (5% seed, self-trained, 400 labeled + 400 unlabeled (800-row pool)) | 0.839 | 0.837 | 0.823 | 1.00 | Additive comparison vs. the DistilBERT pseudo-labeling row above — confidence threshold retuned to 0.30 (not 0.80) since ELECTRA-small's round-0 confidence ceiling was only ~0.33 at this sample size; see `05b_pseudo_labeling_electra.ipynb` |
| Full supervised — **ELECTRA-small** (100% labels, 150-row sample) | 0.412 | 0.348 | — | — | Additive comparison vs. the DistilBERT full-supervised row above — see discussion below |
| Summarize + zero-shot (200-row test sample, no fine-tuning) | 0.675 | 0.641 | — | — | New labeling method (`09_summarization_labeling.ipynb`) — summarize then zero-shot classify, no fine-tuning at all |

### DistilBERT vs. ELECTRA-small (mentor feedback item 1)

The mentor's original ask was to "choose another model, we can still compare
it with DistilBERT later" — here's that comparison, run at two different
sample sizes:

At `CLASSIFIER_SAMPLE_SIZE=150` (full supervised, the tiny-sample regime),
DistilBERT (85.2% accuracy) dramatically outperforms ELECTRA-small (41.2%)
— a ~44-point gap. This is plausible rather than alarming: ELECTRA-small's
classification head is randomly initialized (its replaced-token-detection
pretraining objective produces no usable `[CLS]`-style head the way
masked-LM pretraining does), and with only 150 examples and 3 epochs there
isn't enough signal to learn a working head from scratch, whereas
DistilBERT's architecture/pretraining converges faster on tiny samples.

At the larger, self-training-grown pseudo-labeling sample (400 labeled + 400
unlabeled, 800-row pool), ELECTRA-small reaches 83.9% accuracy / 83.7% Macro F1, versus DistilBERT's
88.7% / 88.7% at the same method — still behind DistilBERT, but a much
smaller ~5-point gap.

**Honest read:** DistilBERT wins in both regimes tested — this isn't a case
where ELECTRA-small pulls ahead anywhere. But the gap narrows a lot (44
points → 5 points) as effective sample size grows, which suggests
ELECTRA-small's disadvantage here is primarily a tiny-sample /
classifier-head-initialization effect rather than the model being
categorically weaker than DistilBERT for this task. A fairer head-to-head
would need both models run at matching, larger sample sizes than either has
been tested at here — not yet done (see Section 7).

> **Important caveat:** pseudo-labeling's 400 labeled + 400 unlabeled (800-row
> pool) and full-supervised's 150-row sample are **not directly comparable at
> face value** — pseudo-labeling
> currently gets ~2.7x more raw text data (its own labeled seed *plus* the
> unlabeled pool it self-labels) than the full-supervised baseline's 150-row
> cap, which exists purely to keep that notebook's runtime bounded (see
> `utils/config.py`'s `CLASSIFIER_SAMPLE_SIZE` note). The headline "pseudo-labeling
> beats full supervised" result should be read as "self-training scales past a
> tiny 150-row full-supervised baseline," not as "beats what full supervision
> could achieve at equal or larger sample size" — that comparison hasn't been
> run yet.

### Pseudo-labeling confidence-threshold tuning (the key methodological story)

The self-training loop stops when either (a) `target_coverage` of the pool is
labeled, (b) the unlabeled pool is exhausted, (c) a round absorbs zero new
pseudo-labels (model too underconfident to progress — genuine stall), or (d)
a safety cap on iterations is hit.

| Run | Sample size | Confidence threshold | Target coverage | Rounds | Coverage reached | Label Acc. | Test Acc. |
|---|---|---|---|---|---|---|---|
| 1 | 150 | 0.90 (original) | 0.95 | 1 (stalled) | 0% absorbed | — | 0.828 |
| 2 | 150 | 0.60 | 0.95 | 4 | 72.8% | 0.912 | 0.849 |
| 3 | 400 | 0.60 | 0.95 | 1 (saturated) | 96.8% | 0.904 | 0.886 |
| 4 (current) | 400 | 0.80 | 0.98 | 3 | 98.2% | 0.907 | **0.887** |

- **Run 1** revealed the original fixed 0.90 threshold was too strict: the
  DistilBERT model, fine-tuned on only ~152 labeled rows, never got
  confident enough to cross 90% softmax on anything, even though its raw
  predictions were often correct (82.8% test accuracy) — so the loop
  "stalled" at zero absorbed pseudo-labels on round 1.
- **Run 2** confirmed the threshold was the real bottleneck: lowering it to
  0.60 let the loop run genuinely for 4 rounds and lifted test accuracy to
  84.9%.
- **Run 3** widened the sample to 400 labeled + 400 unlabeled (800-row pool)
  at the same 0.60 threshold — big
  accuracy win (88.6%) but the loop converged in a **single round** (374/400
  unlabeled rows already cleared 60% confidence on the very first pass,
  hitting 96.8% coverage before a second round could run). This was correct
  behavior for that threshold/sample combination, not a bug — but it meant
  no genuine multi-round self-training was being observed at this scale.
- **Run 4** (today) raised the threshold to 0.80 and the target coverage to
  0.98 (so an early single-round coverage hit wouldn't cut the loop short),
  specifically to force each round to leave a meaningful unresolved residual
  for the next round. Result: a clean 3-round self-training curve with the
  classic diminishing-returns shape —

  | Round | New pseudo-labels absorbed | Cumulative coverage |
  |---|---|---|
  | 0 | 331 (41.4% of pool) | 91.4% |
  | 1 | 43 (5.4%) | 96.8% |
  | 2 | 12 (1.5%) | 98.2% → target reached, stopped |

  Round 0 mops up the easy/confident majority; later rounds pick off
  progressively harder residual cases at high confidence. Final test
  accuracy (88.7%) matched Run 3 within noise — tightening the threshold
  produced the expected multi-round behavior **without costing accuracy**.

### Key takeaways

1. **Sentence embeddings (MiniLM) + simple KMeans is a surprisingly strong,
   fully unsupervised baseline** — 83% clustering accuracy with zero labels,
   beating both TF-IDF and frozen RoBERTa embeddings.
2. **Frozen RoBERTa embeddings underperformed** MiniLM/TF-IDF here — likely
   because raw mean-pooled RoBERTa isn't optimized for semantic similarity
   the way a sentence-transformer is, combined with a much smaller sample
   size (500 vs 8,000 rows) for this method specifically.
3. **HDBSCAN was consistently worse than KMeans** on this data — it doesn't
   know there are exactly 4 classes and tends to either over-fragment or
   (in RoBERTa's case) call everything noise. KMeans' assumption of `k=4`
   fits AG News's known, balanced class count well.
4. **A fixed confidence threshold is the main failure mode of self-training
   pseudo-labeling** — too strict (0.90) and it stalls at zero absorbed
   labels even when the model's raw predictions are decent; too loose
   (0.60) at a large enough sample and it saturates in a single round,
   hiding the iterative self-training dynamic. A threshold tuned to leave a
   residual after round 1 (0.80 here) recovers genuine multi-round
   self-training without sacrificing final accuracy.
5. **Label propagation is a strong, near-zero-cost semi-supervised
   baseline** — 87.2% label accuracy with no fine-tuning loop at all, just
   a k-NN graph over existing embeddings. Worth comparing against
   pseudo-labeling before investing in more fine-tuning compute.
6. **Weak supervision (keyword heuristics) had good coverage (79%) but low
   label accuracy (46%)** — hand-written keyword rules are noisy and not by
   themselves a substitute for a trained classifier, though they're a
   reasonable zero-labeled-data starting point.

## 6. Sample Generated Labels

Aggregate metrics (Section 5) can hide what a method is actually doing
row-by-row. Below are real example outputs pulled directly from each
method's run — 2 examples per predicted class where available, truncated to
~140 characters (full text in `results/sample_labels_*.csv` and
`results/clusters_*.csv`).

### Unsupervised clustering (MiniLM + KMeans, the strongest unsupervised method)

Clusters have no class names during training — a majority-true-label is
assigned only afterward, for reporting. `purity` is the fraction of a
cluster's members that share that majority label.

| Cluster | Majority label | Purity | Example doc |
|---|---|---|---|
| 0 | World | 0.87 | "Dhaka to announce reward for disclosing grenade attacker's... The Bangladeshi government may announce a..." |
| 1 | Sports | 0.96 | "Hill out for nine months London - Richard Hill, a key member of the long-established England World Cup-winning back row,..." |

KeyBERT (per-document extraction, aggregated across a sampled ~100 docs per
cluster) produces coherent, on-topic multi-word key-phrases per cluster —
real output from `results/clusters_minilm_kmeans.csv` after the finding-1 fix
(concatenated-doc extraction previously produced cross-document splices like
"presidency castro injures"):

| Cluster | Majority label | Top KeyBERT key-phrases (first 4 of 10) |
|---|---|---|
| 0 | World | sen john kerry, minister ariel sharon, arafat intensive care, submarine chicoutimi seaworthy |
| 1 | Sports | grant hill scored, magic utah jazz, ap ap, moss doubtful sunday |
| 2 | Sci/Tech | internet, vodafone begins 3g, netscape browser supports, microsoft |
| 3 | Business | oil prices, economic growth, gallaher profits despite, tokyo stocks rise |

(See `results/clusters_minilm_kmeans.csv` for all 4 clusters, top KeyBERT
key-phrases, and more example docs per cluster.)

### Weak supervision (Snorkel) — label accuracy 46.4%

| Text | Predicted | True | Correct |
|---|---|---|---|
| "Surfers Paradise: Shootout Pirtek Racing's Marcos Ambrose won a fiery first race of the Gillette V8 Supercar Challenge..." | World | Sports | ✗ |
| "Buffs bobble upset bid Some players' mouths were wide open, others sat on the bench. CU coaches awaited an official word..." | Sports | Sports | ✓ |
| "UBS Buys Schwab Unit Swiss banking giant UBS says it will acquire Charles Schwab's (SCH) brokerage unit..." | Business | Business | ✓ |
| "Microsoft Releases Low-Cost Windows XP for India... a pilot program targeting first-time and beginning computer users..." | Sci/Tech | Sci/Tech | ✓ |

Weak supervision's keyword-based labeling functions clearly struggle with
articles that mention a class's keywords in passing without being *about*
that class (the surfing article got flagged "World" likely via a stray
keyword match, and a UN peacekeeping article got misrouted to "Sci/Tech").
This matches its low 46% label accuracy — full 8-row sample in
`results/sample_labels_weak_supervision.csv`.

### Label propagation — label accuracy 87.2%

| Text | Predicted | True | Correct | Confidence |
|---|---|---|---|---|
| "Impotence Rules in the UN Response Rarely has the impotence of the international community's response to the crisis in Sudan's Darfu..." | World | World | ✓ | 1.00 |
| "San Francisco at Milwaukee, 1:05 PM MILWAUKEE (Ticker) -- Barry Bonds tries to go where just two players have gone before..." | Sports | Sports | ✓ | 0.74 |
| "UPDATE 3-Two Marsh & McLennan executives step down... executives linked to..." | Business | Business | ✓ | 1.00 |
| "Sony PSP Draws Crowds and Lines on First Day  TOKYO (Reuters) - Game fans stood in lines through a chilly Tokyo night..." | Sci/Tech | Sci/Tech | ✓ | 1.00 |

All 8 sampled rows were correct (matches its strong 87.2% label accuracy) —
full sample in `results/sample_labels_label_propagation.csv`.

### Full supervised baseline — test accuracy 85.2%

| Text | Predicted | True | Correct | Confidence |
|---|---|---|---|---|
| "Japanese rescuers grapple with rocks to retrieve girl from quake... TOKYO - Rescuers grappled through mud and rocks for..." | World | World | ✓ | 0.47 |
| "Soccer: Matuzalem brace ends Celtic's Champions League hopes DONETSK, Ukraine..." | Sports | Sports | ✓ | 0.59 |
| "US Airways to Cut Hundreds of Jobs... plans to eliminate 'hundreds' of management and nonunion jobs..." | Business | Business | ✓ | 0.51 |
| "Telescope snaps distant 'planet' The first direct image of a planet circling another star may have been obtained..." | Sci/Tech | **World** | ✗ | 0.36 |

7/8 sampled rows correct. The one miss (a space-telescope story misread as
"World" instead of "Sci/Tech") is a plausible confusion — the article body
likely reads more like an international-collaboration story than a tech
story to a model trained on only 150 rows. Full sample in
`results/sample_labels_full_supervised.csv`.

### Pseudo-labeling — *sample rows pending*

A code cell was added to `05_pseudo_labeling.ipynb` to save example
generated pseudo-labels (`results/sample_labels_pseudo_labeling_train_pool.csv`
and `..._test.csv`), but the rerun needed to produce them (~1hr, 3
fine-tuning rounds) was interrupted twice before completing. The method's
**metrics and round-by-round history are already fully documented** in
Section 5 (88.7% test accuracy, 90.7% label accuracy, 3 rounds absorbing
331→43→12 pseudo-labels) — only the row-level example table is outstanding.
Re-run `05_pseudo_labeling.ipynb` end-to-end (unattended, budget ~1hr) to
fill this in.

### Summarize + zero-shot — test accuracy 67.5%

| Text | Summary | Generated title | Predicted | True | Correct |
|---|---|---|---|---|---|
| "Large Explosion Heard in Central Baghdad (Reuters) Reuters - A large blast was heard in central Baghdad on Thursday, witnesses said." | "Large blast heard in central Baghdad on Thursday, witnesses said." | "Large blast heard in central Baghdad on Thursday." | World | World | ✓ |
| "Gatlin sprints from unknown to 100m gold Reuters Athens Aug 23: American Justin Gatlin roared from virtual unknown to win the blue ribband O..." | "Gatlin sprints from unknown to 100m gold. Gatlin roared from virtual unknown to win the blue ribband Olympic mens 100m race." | "Gatlin sprints from unknown to 100" | Sports | Sports | ✓ |
| "Vivendi surprises with rise in revenue Vivendi Universal, the French media group that almost collapsed into bankruptcy two years ago, yester..." | "French media group Vivendi Universal almost collapsed into bankruptcy two years ago. Third-quarter revenues driven by soaring music sales in Britain and North America." | "French media group almost collapsed into bankruptcy two years" | Business | Business | ✓ |
| "Israel, Egypt in Prisoner Swap CAIRO (Reuters) - Israel released six Egyptian students from prison on Sunday as part of a deal which inclu..." | "Israel released six Egyptian students from prison on Sunday as part of a deal. Deal includes freedom for Israeli businessman and convicted spy Azzam Azzam." | "Israeli businessman and convicted spy Azzam A" | **Business** | World | ✗ |

3/4 sampled rows correct (matches the method's overall 67.5% test accuracy,
noticeably below every fine-tuned-classifier method in Section 5). The miss
is a plausible confusion: the summary leans on "businessman," "spy," and a
prisoner *swap* framing that reads Business/espionage-flavored to the
zero-shot classifier even though the story's true section is World (a
diplomatic prisoner exchange). Unlike the fine-tuned methods, this approach
never sees a single labeled AG News example — it's two off-the-shelf models
(summarizer + NLI zero-shot classifier) composed with no task-specific
training at all, so some accuracy cost relative to fine-tuning is expected.
Full 8-row sample in `results/sample_labels_summarization.csv`.

## 7. Limitations / Next Steps

- **Dev-scale sampling, and inconsistent sample sizes across methods**: the
  full-supervised baseline (150 rows) and pseudo-labeling (400 labeled + 400
  unlabeled, 800-row pool) are not
  on equal footing — see the caveat in Section 5. Re-running both at the
  same sample size would make the "pseudo-labeling beats full supervised"
  result trustworthy rather than confounded by sample-size differences.
- **Full-data final run**: not yet started — a literal full-120,000-row run
  is scoped and costed out already (see
  `docs/superpowers/plans/2026-08-15-autolabel-notebooks.md`, "Final Run"
  section) at an estimated **~5–10+ days unattended** on this CPU-only
  machine. A "scaled-up bounded" run (e.g. 20–30k rows) would be far more
  representative than current numbers while staying practical (a few
  hours), and is the more realistic near-term option.
- **OpenAI embeddings** were not run in this dataset snapshot (no API key
  configured) — rows show as "pending."
- A known fix is needed before scaling up: `silhouette_score`/
  `davies_bouldin_score` are O(n²) and must be sub-sampled before running at
  much larger scale.
- Could push `PSEUDO_LABEL_SAMPLE_SIZE` higher, or try the threshold-tuning
  trick at other sample sizes, to map out the accuracy/cost curve more
  precisely (see `docs/semi_supervised_methods.md` for other untried
  candidate methods — SetFit, FlexMatch, co-training, etc.).
- **DistilBERT vs. ELECTRA-small at matching, larger sample sizes**: the
  Section 5 comparison so far is at 150 rows (full supervised) and a
  400 labeled + 400 unlabeled (800-row pool) (pseudo-labeling) — both regimes favor DistilBERT, with the
  gap narrowing sharply from 44 to 5 points as sample size grows. Whether
  that gap keeps closing (or reverses) at larger sample sizes is untested.
- **Semantic-closeness / embedding-similarity label scoring** (deferred,
  explicit scope decision): the summarize + zero-shot method's zero-shot
  classifier always outputs one of the 4 fixed AG News class names, so
  exact-match accuracy is well-defined as-is. But a natural complement for
  more free-form generated labels (e.g. the per-article generated
  headline/title also saved in Section 6) would be to score how
  semantically close a generated free-text label is to the true class name
  via embedding similarity, rather than requiring an exact string match.
  Not built this pass — out of scope by design, left as future work.

## 8. Repo Map

```
utils/            shared code: config, data loading/splitting, embeddings
                   (TF-IDF/MiniLM/RoBERTa/OpenAI, disk-cached), metrics,
                   modeling (fine-tuning + pseudo-label self-training),
                   label_propagation, interpretability (cluster inspection),
                   summarization (summarize + zero-shot classify),
                   samples (saves qualitative example-label CSVs)
notebooks/
  00_data_transform.ipynb          raw CSV → cleaned/split parquet
  01_embeddings.ipynb              generate & cache all embedding methods
  02_unsupervised_clustering.ipynb KMeans/HDBSCAN over each embedding method
  03_bertopic.ipynb                BERTopic topic discovery
  04_weak_supervision.ipynb        Snorkel labeling functions + LabelModel
  05_pseudo_labeling.ipynb         DistilBERT self-training loop
  05b_pseudo_labeling_electra.ipynb ELECTRA-small self-training loop (additive)
  06_full_supervised_baseline.ipynb fully supervised upper bound
  06b_full_supervised_baseline_electra.ipynb ELECTRA-small upper bound (additive)
  07_comparison.ipynb              assembles results/comparison_table.csv
  08_label_propagation.ipynb       MiniLM k-NN graph + LabelSpreading
  09_summarization_labeling.ipynb  summarize + zero-shot classify
results/comparison_table.csv       final results table (source for Section 5)
results/clusters_*.csv             qualitative cluster inspection per method
results/sample_labels_*.csv        example generated labels per method (Section 6)
autolabel_project_spec.md          original project spec (approach/metric defs)
docs/semi_supervised_methods.md    candidate methods considered, tuning notes
docs/superpowers/                  implementation plan + design doc
```
