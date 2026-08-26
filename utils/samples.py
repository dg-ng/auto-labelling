import numpy as np
import pandas as pd


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
