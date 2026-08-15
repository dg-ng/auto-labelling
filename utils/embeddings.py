from pathlib import Path

import numpy as np

from utils.config import CACHE_DIR, MODEL


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npy"


def load_cached(name: str):
    path = _cache_path(name)
    if path.exists():
        return np.load(path)
    return None


def save_cache(name: str, arr: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_cache_path(name), arr)


def get_tfidf_embeddings(texts, cache_name: str, max_features: int = 5000) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features)
    arr = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    save_cache(cache_name, arr)
    return arr


def get_sentence_embeddings(texts, cache_name: str, model_name: str = "all-MiniLM-L6-v2",
                             batch_size: int = 64) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    arr = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)
    save_cache(cache_name, arr)
    return arr


def get_bert_embeddings(texts, cache_name: str, model_name: str = "roberta-base",
                         batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    arr = np.vstack(all_embeddings).astype(np.float32)
    save_cache(cache_name, arr)
    return arr


def get_openai_embeddings(texts, cache_name: str, model: str = MODEL,
                           batch_size: int = 100) -> np.ndarray:
    cached = load_cached(cache_name)
    if cached is not None:
        return cached

    from openai import OpenAI

    client = OpenAI()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([r.embedding for r in response.data])

    arr = np.array(all_embeddings, dtype=np.float32)
    save_cache(cache_name, arr)
    return arr
