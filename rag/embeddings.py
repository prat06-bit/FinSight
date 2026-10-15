"""Local embedding via sentence-transformers.
Uses sentence-transformers/all-MiniLM-L6-v2 by default.
No external API key required. Model is loaded locally and cached.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING
from config import RAG_EMBED_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    """Load the sentence-transformer model (cached in memory after first call)."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading local embedding model: %s", RAG_EMBED_MODEL)
    return SentenceTransformer(RAG_EMBED_MODEL)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document texts. Returns list of 384-dim float vectors."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string into a 384-dim vector."""
    if not query.strip():
        return [0.0] * 384
    model = _get_model()
    embedding = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    return embedding.tolist()
