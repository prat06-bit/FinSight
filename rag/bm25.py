"""BM25 sparse retriever for financial document search.
Uses ``rank_bm25.BM25Okapi`` with financial-domain-aware tokenization
that preserves dollar amounts, percentages, fiscal years, and
accounting terms. The index is persisted to disk as a pickle.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any
from rank_bm25 import BM25Okapi
from config import RAG_BM25_INDEX_PATH

logger = logging.getLogger(__name__)

# Accounting / financial terms to keep intact during tokenisation
_FINANCIAL_TERMS = frozenset({
    "10-k", "10-q", "8-k", "10k", "10q", "8k",
    "gaap", "ebitda", "eps", "roe", "roa", "fcf",
    "s&p", "sec", "irs", "fasb", "ifrs",
    "md&a", "p&l", "r&d",
})


def _tokenize(text: str) -> list[str]:
    """Financial-aware tokenizer.
    Preserves:
    - Dollar amounts (``$1.2B``, ``$500M``)
    - Percentages (``15.3%``)
    - Fiscal years (``FY2023``)
    - Accounting terms (``EBITDA``, ``MD&A``)
    - Hyphenated financial terms (``10-K``)
    """
    text_lower = text.lower()

    # Extract financial patterns before general tokenisation
    special_tokens: list[str] = []

    # Dollar amounts: $1.2B, $500M, $1,234
    special_tokens.extend(re.findall(r"\$[\d,.]+[bmkt]?", text_lower))
    # Percentages
    special_tokens.extend(re.findall(r"\d+\.?\d*%", text_lower))
    # Fiscal years
    special_tokens.extend(re.findall(r"fy\d{2,4}", text_lower))

    # General word tokens (letters, digits, hyphens, ampersands)
    word_tokens = re.findall(r"[a-z0-9][a-z0-9&\-]*[a-z0-9]|[a-z0-9]", text_lower)

    # Keep tokens that are meaningful (len >= 2 or in financial terms)
    stop_words = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
        "for", "from", "had", "has", "have", "he", "her", "his", "how",
        "if", "in", "is", "it", "its", "may", "no", "not", "of", "on",
        "or", "our", "per", "she", "so", "such", "than", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "to",
        "us", "was", "we", "were", "what", "when", "which", "who",
        "will", "with", "would", "you", "your",
    })

    tokens = [
        t for t in word_tokens
        if (len(t) >= 2 and t not in stop_words) or t in _FINANCIAL_TERMS
    ]

    # Add special financial tokens
    tokens.extend(special_tokens)

    return tokens


class BM25Index:
    """BM25 sparse index over financial document chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[dict[str, Any]] = []  
        self._tokenized_corpus: list[list[str]] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._documents) > 0

    def build(self, documents: list[dict[str, Any]]) -> None:
        """Build the BM25 index from a list of document dicts.

        Each dict must have: ``id``, ``text``, ``metadata``.
        """
        self._documents = documents
        self._tokenized_corpus = [_tokenize(doc["text"]) for doc in documents]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None
        logger.info("BM25 index built with %d documents", len(documents))

    def search(
        self,
        query: str,
        top_k: int = 20,
        ticker: str | None = None,
        filing_year: int | str | None = None,
        filing_type: str | None = None,
        section: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search with optional metadata pre-filtering.
        Returns list of ``{id, text, metadata, score}``.
        """
        if not self.is_built:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Pre-filter by metadata with normalized types
        norm_ticker = str(ticker).upper() if ticker else None
        norm_year = None
        if filing_year is not None:
            try:
                norm_year = int(filing_year)
            except (ValueError, TypeError):
                norm_year = str(filing_year)
        candidate_indices = list(range(len(self._documents)))
        if norm_ticker or norm_year or filing_type or section:
            filtered = []
            for idx in candidate_indices:
                meta = self._documents[idx].get("metadata", {})
                if norm_ticker and str(meta.get("ticker", "")).upper() != norm_ticker:
                    continue
                if norm_year is not None:
                    doc_year = meta.get("filing_year")
                    try:
                        if doc_year is not None and int(doc_year) != norm_year:
                            continue
                    except (ValueError, TypeError):
                        if str(doc_year) != str(norm_year):
                            continue
                if filing_type and meta.get("filing_type") != filing_type:
                    continue
                if section and meta.get("section") != section:
                    continue
                filtered.append(idx)
            candidate_indices = filtered

        if not candidate_indices:
            return []
        # Score all documents, then filter to candidates
        scores = self._bm25.get_scores(query_tokens)

        # Build (index, score) for candidates and sort
        scored = [(idx, float(scores[idx])) for idx in candidate_indices]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            if score <= 0:
                continue
            doc = self._documents[idx]
            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score,
                }
            )
        return results

    def save(self, path: Path | None = None) -> None:
        """Persist the index to disk."""
        save_path = path or RAG_BM25_INDEX_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self._documents,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )
        logger.info("BM25 index saved to %s (%d docs)", save_path, len(self._documents))

    def load(self, path: Path | None = None) -> bool:
        """Load a persisted index. Returns True on success."""
        load_path = path or RAG_BM25_INDEX_PATH
        if not load_path.exists():
            logger.info("No BM25 index found at %s", load_path)
            return False
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self._documents = data["documents"]
            self._tokenized_corpus = data["tokenized_corpus"]
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info("BM25 index loaded: %d documents", len(self._documents))
            return True
        except Exception as exc:
            logger.error("Failed to load BM25 index: %s", exc)
            return False

    def document_count(self) -> int:
        return len(self._documents)
