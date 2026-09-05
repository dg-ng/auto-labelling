"""Auto-derived weak-supervision labeling functions.

The AG News version of this project hand-wrote one keyword list per class
(4 classes). At 16 classes — several overlapping in subject matter
(POLITICS vs. NEWS vs. WOMEN vs. CRIME vs. MEDIA could all describe the
same story) — hand-authoring doesn't scale. Instead, each class's most
TF-IDF-distinctive terms (relative to the rest of the 5% labeled seed) are
turned into a keyword-vote labeling function automatically.
"""

import numpy as np


def derive_class_keywords(labeled_df, class_names, text_col="text", label_col="label",
                           top_n=15, max_features=20000) -> dict:
    """For each class, return its `top_n` most distinctive keywords: the
    terms with the largest (mean TF-IDF weight within the class) minus
    (mean TF-IDF weight in the rest of the seed).

    Returns {label_int: [term, ...]}.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=1)
    tfidf = vectorizer.fit_transform(labeled_df[text_col])
    vocab = np.array(vectorizer.get_feature_names_out())
    labels = labeled_df[label_col].to_numpy()

    keywords_by_class = {}
    for label in range(len(class_names)):
        mask = labels == label
        if mask.sum() == 0:
            keywords_by_class[label] = []
            continue
        class_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        rest_mask = ~mask
        rest_mean = (np.asarray(tfidf[rest_mask].mean(axis=0)).ravel()
                     if rest_mask.sum() else np.zeros_like(class_mean))
        distinctiveness = class_mean - rest_mean
        top_idx = distinctiveness.argsort()[::-1][:top_n]
        keywords_by_class[label] = vocab[top_idx].tolist()
    return keywords_by_class


def build_keyword_lfs(keywords_by_class):
    """Build one Snorkel `LabelingFunction` per class from its keyword
    list. Each LF votes for its class if any of its keywords appears in
    `x.text` (case-insensitive substring match), else abstains (-1).

    Returns a list of `snorkel.labeling.LabelingFunction`, skipping any
    class with an empty keyword list (e.g. zero seed rows for that class).

    Note: The canonical way to invoke a Snorkel LF is via `lf(x)` (the
    `__call__` method). The `.f` attribute exposed below is a non-standard
    addition for compatibility with certain test patterns; it should not be
    relied upon as part of Snorkel's standard API. Use `lf(x)` for production.
    """
    from snorkel.labeling import LabelingFunction

    ABSTAIN = -1

    def _make_fn(label, keywords):
        def _fn(x):
            text_lower = x.text.lower()
            return label if any(kw in text_lower for kw in keywords) else ABSTAIN
        return _fn

    lfs = []
    for label, keywords in keywords_by_class.items():
        if not keywords:
            continue
        fn = _make_fn(label, keywords)
        lf = LabelingFunction(name=f"lf_class_{label}", f=fn)
        # Non-standard: expose the function as .f for test compatibility.
        # Real Snorkel LFs store callbacks as ._f and invoke via __call__.
        # Use lf(x) in production; .f is for compatibility only.
        lf.f = fn
        lfs.append(lf)
    return lfs
