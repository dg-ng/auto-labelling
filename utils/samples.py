"""Utilities for saving per-row output tables from clustering / classification runs.

Each row in the output CSV contains the original text, the model's predicted
label (class name string), the true label (class name string), and any
caller-supplied extra columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_full_output(
    texts: list[str],
    predicted: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str],
    extra_columns: dict[str, Any] | None = None,
    path: str | Path = "results/full_labels.csv",
) -> pd.DataFrame:
    """Build and save a per-row output CSV.

    Args:
        texts: Original document strings, one per row.
        predicted: Integer predicted class indices (after Hungarian remapping),
            shape (N,). -1 entries are written as the string "UNASSIGNED".
        true_labels: Integer ground-truth class indices, shape (N,).
        class_names: List mapping integer index → class name string.
        extra_columns: Optional dict of additional column_name -> list/array
            values to include in the output (e.g. {"summary": summaries}).
        path: File path to write the CSV. Parent directories are created
            automatically.

    Returns:
        The saved DataFrame.
    """
    predicted = np.asarray(predicted)
    true_labels = np.asarray(true_labels)

    def _label_name(idx: int) -> str:
        if idx < 0 or idx >= len(class_names):
            return "UNASSIGNED"
        return class_names[idx]

    rows: dict[str, Any] = {
        "text": texts,
        "predicted_label": [_label_name(int(p)) for p in predicted],
        "true_label": [_label_name(int(t)) for t in true_labels],
    }

    if extra_columns:
        for col_name, values in extra_columns.items():
            rows[col_name] = list(values)

    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
