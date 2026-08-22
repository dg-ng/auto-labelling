import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.config import CLASSIFIER_MODEL_NAME, NUM_CLASSES, SEED


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def train_model(labeled_df, model_name=CLASSIFIER_MODEL_NAME, epochs=3,
                 output_dir="./_tmp_trainer", max_length=128, batch_size=16):
    """Fine-tune a sequence classifier on a labeled DataFrame with text/label columns."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)

    encodings = tokenizer(
        labeled_df["text"].tolist(), padding=True, truncation=True,
        max_length=max_length, return_tensors="pt")
    dataset = TextClassificationDataset(encodings, labeled_df["label"].to_numpy())

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        seed=SEED,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    return model, tokenizer


def get_predictions(model, tokenizer, texts, batch_size=64, max_length=128) -> np.ndarray:
    """Return softmax class probabilities, shape (len(texts), NUM_CLASSES)."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        all_probs.append(softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


def pseudo_label_loop(labeled_df, unlabeled_df, model_name=CLASSIFIER_MODEL_NAME,
                       confidence_threshold=0.90, epochs=3, target_coverage=0.95,
                       max_iterations=10, verbose=True):
    """Iteratively: train on labeled set, predict on remaining unlabeled set,
    absorb high-confidence (>= confidence_threshold) predictions, repeat.

    Stops when any of these is true (checked in this order each round):
      1. `target_coverage` of the whole pool (labeled + unlabeled) has been
         labeled (seed labels + absorbed pseudo-labels combined).
      2. No unlabeled rows remain.
      3. A round absorbs zero new pseudo-labels — the model isn't confident
         enough to make further progress at this `confidence_threshold`, so
         repeating would just retrain on the same data for no gain.
      4. `max_iterations` rounds have run (safety cap).

    Returns the final model, tokenizer, the grown labeled DataFrame, and a
    per-iteration history log (also printed live if `verbose`)."""
    current_labeled = labeled_df.copy()
    remaining_unlabeled = unlabeled_df.copy()
    total_sample = len(labeled_df) + len(unlabeled_df)
    history = []

    if verbose:
        print(f"Total sample: {total_sample} | target coverage: {target_coverage:.0%} "
              f"| confidence threshold: {confidence_threshold}")

    for iteration in range(max_iterations):
        coverage = len(current_labeled) / total_sample
        if coverage >= target_coverage:
            if verbose:
                print(f"Reached target coverage ({coverage:.1%} >= {target_coverage:.0%}). Stopping.")
            break
        if len(remaining_unlabeled) == 0:
            if verbose:
                print("No unlabeled rows remain. Stopping.")
            break

        model, tokenizer = train_model(current_labeled, model_name=model_name, epochs=epochs)

        probs = get_predictions(model, tokenizer, remaining_unlabeled["text"].tolist())
        confidence = probs.max(axis=1)
        predicted_labels = probs.argmax(axis=1)

        high_conf_mask = confidence >= confidence_threshold
        n_new = int(high_conf_mask.sum())
        newly_labeled = remaining_unlabeled[high_conf_mask].copy()
        newly_labeled["label"] = predicted_labels[high_conf_mask]

        drop_cols = [c for c in ["true_label"] if c in newly_labeled.columns]
        current_labeled = pd.concat(
            [current_labeled, newly_labeled.drop(columns=drop_cols)],
            ignore_index=True)
        remaining_unlabeled = remaining_unlabeled[~high_conf_mask]

        coverage = len(current_labeled) / total_sample
        record = {
            "iteration": iteration,
            "total_sample": total_sample,
            "new_labels": n_new,
            "new_labels_pct_of_total": n_new / total_sample,
            "labeled_size": len(current_labeled),
            "coverage": coverage,
            "unlabeled_size": len(remaining_unlabeled),
        }
        history.append(record)
        if verbose:
            print(f"Iteration {iteration}: total={total_sample} | "
                  f"new >= {confidence_threshold:.0%} confidence: {n_new} "
                  f"({record['new_labels_pct_of_total']:.1%} of total) | "
                  f"labeled so far: {record['labeled_size']} ({coverage:.1%} of total) | "
                  f"remaining unlabeled: {record['unlabeled_size']}")

        if n_new == 0:
            if verbose:
                print(f"No new pseudo-labels absorbed at threshold={confidence_threshold} "
                      f"— model isn't confident enough to progress further. Stopping "
                      f"({coverage:.1%} of total labeled, short of the {target_coverage:.0%} target).")
            break
    else:
        if verbose and len(current_labeled) / total_sample < target_coverage:
            print(f"Hit max_iterations={max_iterations} without reaching "
                  f"{target_coverage:.0%} coverage ({len(current_labeled) / total_sample:.1%} reached).")

    final_model, final_tokenizer = train_model(current_labeled, model_name=model_name, epochs=epochs)
    return final_model, final_tokenizer, current_labeled, history
