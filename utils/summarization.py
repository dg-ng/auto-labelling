import json
from pathlib import Path

import numpy as np

from utils.config import SUMMARIZATION_MODEL_NAME


def summarize_texts(texts, model_name=SUMMARIZATION_MODEL_NAME, max_length=60,
                     min_length=8, batch_size=8, num_beams=4) -> list:
    """Summarize each text with a CPU-friendly distilled summarization model.

    Uses AutoModelForSeq2SeqLM + generate() directly rather than
    transformers.pipeline("summarization") — the installed transformers
    release (5.x) removed the summarization/translation/text2text-generation
    pipeline task shortcuts; the underlying seq2seq generation API is
    unaffected.

    Returns one summary string per input text, same order.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()

    # Some tokenizers (e.g. bart-large-cnn's) leave model_max_length at the
    # library's huge sentinel default rather than the model's real position
    # limit, so truncation=True alone is a no-op and long inputs overflow
    # the model's learned position embeddings (IndexError). Cap explicitly.
    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    encode_kwargs = {"return_tensors": "pt", "truncation": True, "padding": True}
    if max_position_embeddings:
        encode_kwargs["max_length"] = max_position_embeddings

    summaries = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, **encode_kwargs)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_length=max_length, min_length=min_length,
                num_beams=num_beams, no_repeat_ngram_size=3, early_stopping=True)
        summaries.extend(
            tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids)
    return summaries


def summarize_texts_cached(texts, cache_path, model_name=SUMMARIZATION_MODEL_NAME,
                            max_length=60, min_length=8, batch_size=4,
                            num_beams=2, chunk_size=40) -> list:
    """Summarize each text, reusing a durable on-disk cache keyed by exact
    text content, so re-running at a different sample size (or after an
    interruption) never re-generates a summary already computed for that
    exact text. `cache_path` is a JSON file mapping text -> summary, meant
    to be kept indefinitely across every run/size — NEVER delete it;
    growing it only saves future work. Writes the cache back to disk after
    each chunk so a crash/timeout mid-run only loses the in-progress chunk.
    """
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    texts = list(texts)
    missing = [t for t in texts if t not in cache]
    print(f"{len(texts) - len(missing)} reused from cache, {len(missing)} to generate")

    for i in range(0, len(missing), chunk_size):
        chunk = missing[i:i + chunk_size]
        chunk_summaries = summarize_texts(chunk, model_name=model_name, max_length=max_length,
                                            min_length=min_length, batch_size=batch_size,
                                            num_beams=num_beams)
        for t, s in zip(chunk, chunk_summaries):
            cache[t] = s
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    return [cache[t] for t in texts]


def generate_titles(texts, model_name=SUMMARIZATION_MODEL_NAME, max_length=12,
                     min_length=3, batch_size=8) -> list:
    """Generate a short headline-length title per text.

    Same summarization model/approach as `summarize_texts`, just a much
    shorter output length — reused rather than a separate pipeline.
    """
    return summarize_texts(texts, model_name=model_name, max_length=max_length,
                            min_length=min_length, batch_size=batch_size)
