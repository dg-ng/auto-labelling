import pandas as pd


def load_raw(path) -> pd.DataFrame:
    """Load an AG News CSV. Files have a header row: Class Index,Title,Description."""
    return pd.read_csv(path, header=0, names=["label", "title", "description"])


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title+description into `text` and 0-index the label column."""
    df = df.copy()
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["label"] = df["label"] - 1
    return df


def make_splits(train_df: pd.DataFrame, label_fraction: float, seed: int):
    """Split into a small labeled pool and a large unlabeled pool.

    The unlabeled pool's real label is kept as `true_label` for evaluation
    only — `label` is set to -1 to simulate it being unavailable to any
    training algorithm.
    """
    indexed_df = train_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=label_fraction, random_state=seed))
    sampled_index = indexed_df.index
    labeled_df = train_df.loc[sampled_index]
    unlabeled_df = train_df.drop(sampled_index).copy()
    unlabeled_df["true_label"] = unlabeled_df["label"]
    unlabeled_df["label"] = -1
    return labeled_df.reset_index(drop=True), unlabeled_df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, sample_size, seed: int, label_col: str = "label") -> pd.DataFrame:
    """Return a stratified sample of `sample_size` rows, or the full df if
    sample_size is None or >= len(df)."""
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if label_col not in df.columns:
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    frac = sample_size / len(df)
    indexed = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=seed))
    return df.loc[indexed.index].reset_index(drop=True)
