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
                       n_iterations=3, confidence_threshold=0.90, epochs=3):
    """Iteratively: train on labeled set, predict on remaining unlabeled set,
    absorb high-confidence predictions, repeat. Returns the final model,
    tokenizer, the grown labeled DataFrame, and a per-iteration history log."""
    current_labeled = labeled_df.copy()
    remaining_unlabeled = unlabeled_df.copy()
    history = []

    for iteration in range(n_iterations):
        if len(remaining_unlabeled) == 0:
            break

        model, tokenizer = train_model(current_labeled, model_name=model_name, epochs=epochs)

        probs = get_predictions(model, tokenizer, remaining_unlabeled["text"].tolist())
        confidence = probs.max(axis=1)
        predicted_labels = probs.argmax(axis=1)

        high_conf_mask = confidence >= confidence_threshold
        newly_labeled = remaining_unlabeled[high_conf_mask].copy()
        newly_labeled["label"] = predicted_labels[high_conf_mask]

        history.append({
            "iteration": iteration,
            "new_labels": int(high_conf_mask.sum()),
            "labeled_size": len(current_labeled),
            "unlabeled_size": len(remaining_unlabeled),
        })

        drop_cols = [c for c in ["true_label"] if c in newly_labeled.columns]
        current_labeled = pd.concat(
            [current_labeled, newly_labeled.drop(columns=drop_cols)],
            ignore_index=True)
        remaining_unlabeled = remaining_unlabeled[~high_conf_mask]

    final_model, final_tokenizer = train_model(current_labeled, model_name=model_name, epochs=epochs)
    return final_model, final_tokenizer, current_labeled, history
