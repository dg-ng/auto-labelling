import numpy as np

from utils.config import SUMMARIZATION_MODEL_NAME


def summarize_texts(texts, model_name=SUMMARIZATION_MODEL_NAME, max_length=60,
                     min_length=8, batch_size=8) -> list:
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

    summaries = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_length=max_length, min_length=min_length,
                num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
        summaries.extend(
            tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids)
    return summaries


def generate_titles(texts, model_name=SUMMARIZATION_MODEL_NAME, max_length=12,
                     min_length=3, batch_size=8) -> list:
    """Generate a short headline-length title per text.

    Same summarization model/approach as `summarize_texts`, just a much
    shorter output length — reused rather than a separate pipeline.
    """
    return summarize_texts(texts, model_name=model_name, max_length=max_length,
                            min_length=min_length, batch_size=batch_size)
