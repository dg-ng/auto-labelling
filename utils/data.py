import pandas as pd


def load_master_data(path, class_names, min_words: int = 5) -> pd.DataFrame:
    """Load and clean `data/master_data.csv` (columns: category, title,
    summary, text). Returns a DataFrame with `text` (title + body,
    deduplicated), `category` (cleaned string), and `label` (0-indexed
    position into `class_names`).

    Cleaning: keeps the source `summary` column where present (populated
    for only a minority of rows/categories, NaN elsewhere) — it's reused
    as-is instead of being regenerated; only rows missing one get a
    freshly generated summary later, in 00_data_transform. Also combines
    title+text into one `text` field, drops duplicate `text` rows (keep
    first), and drops degenerate rows below `min_words` words (e.g. a
    body of just "(CNN)").
    """
    df = pd.read_csv(path)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df.drop_duplicates(subset="text", keep="first")

    word_count = df["text"].str.split().str.len()
    df = df[word_count >= min_words].copy()

    df["category"] = df["category"].str.strip()
    unknown = set(df["category"].unique()) - set(class_names)
    assert not unknown, f"Unexpected categories not in class_names: {unknown}"

    label_by_name = {name: i for i, name in enumerate(class_names)}
    df["label"] = df["category"].map(label_by_name)
    return df.reset_index(drop=True)


def stratified_train_test_split(df: pd.DataFrame, test_fraction: float, seed: int,
                                 label_col: str = "label"):
    """Stratified train/test split by `label_col` (every class split at the
    same fraction). Returns (train_df, test_df), both index-reset."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_fraction, stratify=df[label_col], random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


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
