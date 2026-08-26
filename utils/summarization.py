import numpy as np

from utils.config import SUMMARIZATION_MODEL_NAME, ZERO_SHOT_MODEL_NAME


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


def zero_shot_label(texts, class_names, model_name=ZERO_SHOT_MODEL_NAME,
                     batch_size=8):
    """Zero-shot-classify each text into one of `class_names` via an NLI model.

    Returns (predicted_labels, confidence) as parallel numpy arrays —
    predicted_labels are indices into class_names, confidence is the top
    label's entailment score. Same shape/contract as utils.modeling's
    prediction outputs so callers can reuse utils.metrics unchanged.
    """
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model=model_name)
    results = classifier(list(texts), candidate_labels=list(class_names),
                          batch_size=batch_size)
    if isinstance(results, dict):
        results = [results]

    predicted_labels = np.array([class_names.index(r["labels"][0]) for r in results])
    confidence = np.array([r["scores"][0] for r in results])
    return predicted_labels, confidence
