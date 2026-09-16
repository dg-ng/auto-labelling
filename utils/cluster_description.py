# utils/cluster_description.py
import re
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


def generate_cluster_descriptions(
    summaries: list[str],
    cluster_labels: np.ndarray,
    n_sentences: int = 3,
    min_words: int = 10,
) -> dict[int, str]:
    """Return {cluster_id: description} for all non-noise clusters.

    Concatenates all summaries in a cluster, runs LexRank to pick the
    n_sentences most central sentences, filters out fragments shorter than
    min_words words, and joins the remainder into one description string.
    Falls back to top-n TF-IDF terms if LexRank returns nothing.
    """
    cluster_labels = np.array(cluster_labels)
    summaries = list(summaries)
    cluster_ids = sorted(c for c in np.unique(cluster_labels) if c >= 0)
    summarizer = LexRankSummarizer()

    descriptions = {}
    for c in cluster_ids:
        mask = cluster_labels == c
        cluster_docs = [summaries[i] for i, m in enumerate(mask) if m]

        full_text = " ".join(cluster_docs)
        # Remove zero-width and control characters that confuse sumy's tokenizer
        full_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\xa0]", " ", full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()

        if not full_text:
            descriptions[c] = ""
            continue

        try:
            parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
            result = summarizer(parser.document, sentences_count=n_sentences)
            sentences = [str(s).strip() for s in result
                         if len(str(s).strip().split()) >= min_words]
        except Exception:
            sentences = []

        descriptions[c] = " ".join(sentences) if sentences else full_text[:300]

    return descriptions
