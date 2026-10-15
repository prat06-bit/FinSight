"""Reciprocal Rank Fusion (RRF) for combining dense + sparse retrieval.
RRF assigns each document a score of 1 / (k + rank)`from each
retrieval source, then sums across sources. This rank-based method
is robust because it depends only on candidate ordering, not on raw
score scale differences. """

from __future__ import annotations

import logging
from typing import Any
from config import RAG_RRF_K
from rag.models import RetrievalResult

logger = logging.getLogger(__name__)


def rrf_fuse(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int | None = None,
    final_top_k: int = 5,
) -> list[RetrievalResult]:
    """Fuse dense and BM25 result lists via Reciprocal Rank Fusion.
    Parameters
    ----------
    dense_results:
        List of dicts from ChromaDB (keys: ``id``, ``text``, ``metadata``, ``distance``).
    bm25_results:
        List of dicts from BM25 (keys: ``id``, ``text``, ``metadata``, ``score``).
    k:
        RRF smoothing constant (default 60).
    final_top_k:
        Number of results to return after fusion.
    Returns
    -------
    list[RetrievalResult]
        Fused and deduplicated results sorted by RRF score descending.
    """
    if k is None:
        k = RAG_RRF_K

    # Accumulate RRF scores and record candidate ranks
    rrf_scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}
    doc_lookup: dict[str, dict[str, Any]] = {}

    # Dense candidates
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        dense_ranks[doc_id] = rank
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = doc
    # BM25 candidates
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["id"]
        bm25_ranks[doc_id] = rank
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results: list[RetrievalResult] = []
    for doc_id in sorted_ids[:final_top_k]:
        doc = doc_lookup[doc_id]
        meta = doc.get("metadata", {})
        # Determine method contribution tag for debugging
        d_rank = dense_ranks.get(doc_id)
        b_rank = bm25_ranks.get(doc_id)
        if d_rank and b_rank:
            method_tag = "hybrid"
        elif d_rank:
            method_tag = "dense"
        else:
            method_tag = "bm25"
        year_val = meta.get("filing_year", 0)
        try:
            int_year = int(year_val)
        except (ValueError, TypeError):
            int_year = 0

        results.append(
            RetrievalResult(
                text=doc.get("text", ""),
                score=round(rrf_scores[doc_id], 6),
                chunk_id=doc_id,
                ticker=meta.get("ticker", ""),
                company_name=meta.get("company_name", ""),
                filing_year=int_year,
                filing_type=meta.get("filing_type", "10-K"),
                section=meta.get("section", ""),
                item_number=meta.get("item_number", ""),
                accession_number=meta.get("accession_number", ""),
                source_url=meta.get("source_url", ""),
                filing_date=meta.get("filing_date", ""),
                retrieval_method=method_tag,
            )
        )

    return results
