"""Shared embedding utilities using sentence-transformers (CPU).

Provides text and image embedding generation for evidence indexing.
"""

from __future__ import annotations

import logging

import numpy as np

from detective_ai.config import settings
from detective_ai.storage.cache import cache

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_text_model = None


def _get_text_model():
    """Lazy-load the sentence-transformer model."""
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _text_model = SentenceTransformer(settings.embedding_model, device="cpu")
        logger.info("Embedding model loaded.")
    return _text_model


def embed_text(text: str, use_cache: bool = True) -> list[float]:
    """Generate a text embedding using sentence-transformers.

    Args:
        text: Input text to embed.
        use_cache: Whether to cache the result.

    Returns:
        List of floats (384-dim for all-MiniLM-L6-v2).
    """
    if use_cache:
        cache_key = f"emb:text:{hash(text)}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    model = _get_text_model()
    embedding = model.encode(text, normalize_embeddings=True)
    result = embedding.tolist()

    if use_cache:
        cache.set(cache_key, result, ttl=7200)

    return result


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts.

    Args:
        texts: List of input texts.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    model = _get_text_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [e.tolist() for e in embeddings]


def compute_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)
