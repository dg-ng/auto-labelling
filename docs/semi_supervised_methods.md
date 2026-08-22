# Additional Semi-Supervised Methods — Candidates

Options to extend `04_weak_supervision.ipynb` / `05_pseudo_labeling.ipynb`
beyond the current weak-supervision (Snorkel) and pseudo-labeling
(DistilBERT self-training) approaches. Not implemented yet — pick one (or a
few) and it can be built out as a new `notebooks/0X_*.ipynb` +
`utils/` additions, following the same pattern as the existing methods.

Current pseudo-labeling loop's known weak spot this list responds to: a
**fixed** confidence threshold (0.90) stalled completely at small sample
sizes (`CLASSIFIER_SAMPLE_SIZE=150` absorbed 0 pseudo-labels across 3
rounds) — several options below (FlexMatch, SetFit, label propagation)
sidestep that specific failure mode.

## Options

| Method | Idea | Why it'd fit this project | Rough cost |
|---|---|---|---|
| **SetFit** | Contrastive-fine-tune a sentence-transformer on pairs drawn from the tiny labeled seed, then a lightweight classifier head on top — no large-batch fine-tuning loop | Purpose-built for "few labels, small model"; likely beats DistilBERT self-training at 5% labels, and doesn't depend on a brittle confidence threshold | Low — single short training pass, CPU-friendly |
| **FlexMatch / curriculum pseudo-labeling** | Per-class adaptive confidence threshold instead of one fixed 0.90 (classes the model is currently worse at get a lower bar) | Directly fixes the "threshold too strict → nothing absorbed" stall seen in the dev-scale run | Low — same loop, smarter threshold logic |
| **Noisy Student** | Same self-training loop, but inject data augmentation / dropout noise into the student each round, optionally growing model capacity | More robust to pseudo-label noise than plain self-training | Medium — needs an augmentation step |
| **Label propagation / label spreading** (`sklearn.semi_supervised`) | Build a k-NN graph over embeddings (already computed in `01_embeddings.ipynb`) and propagate the 5% seed labels through the graph algebraically — no fine-tuning loop at all | Cheap, fast, reuses existing embedding cache; good complementary baseline against pseudo-labeling/weak supervision | Low — no model training, just graph algebra |
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
  approach in `05_pseudo_labeling.ipynb` as of 2026-08-21: stop at 95% of
  the pool labeled).
- Label-quality-vs-ground-truth (`evaluate_label_quality`) should stay an
  *evaluation-only* metric computed after the loop, not a loop-control
  signal, for any of the options above too.

## Decision

*Not chosen yet — revisit and pick one (or a small combination) before
building the next notebook.*
