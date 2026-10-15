"""Hybrid retriever combining Dense + BM25 via Reciprocal Rank Fusion.

Public API
----------
``HybridRetriever.search(query, ...)``
    Runs dense retrieval, BM25 retrieval, and fuses via RRF.

``retrieve()`` / ``answer_question()``
    Backward-compatible functions preserved for existing CLI commands.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.financial_data import MoneyRecord, summarize_financials
from config import (
    NVIDIA_API_KEY,
    RAG_BM25_TOP_K,
    RAG_DENSE_TOP_K,
    RAG_TOP_K,
)
from rag.bm25 import BM25Index
from rag.embeddings import embed_query
from rag.fusion import rrf_fuse
from rag.models import RetrievalResult
from rag.vectorstore import FinSightVectorStore

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_FILE = Path(__file__).parent / "knowledge_base.md"


# ── Legacy dataclass (backward compat) ────────────────────────────────────

@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    matched_terms: tuple[str, ...] = ()
    source: str = "knowledge_base.md"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key == "text":
            return self.text
        if key == "score":
            return self.score
        if key == "distance":
            return 1.0 - self.score
        if key == "source":
            return self.source
        if key == "metadata":
            return self.metadata
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default


# ── Hybrid Retriever ──────────────────────────────────────────────────────

class HybridRetriever:
    """Orchestrates Dense + BM25 retrieval with RRF fusion."""

    def __init__(self) -> None:
        self._store = FinSightVectorStore()
        self._bm25 = BM25Index()
        self._bm25.load()  # Load persisted index if available

    def search(
        self,
        query: str,
        ticker: str | None = None,
        filing_year: int | None = None,
        filing_type: str | None = None,
        section: str | None = None,
        top_k: int | None = None,
        method: str = "hybrid",
    ) -> list[RetrievalResult]:
        """Search across indexed SEC filings.

        Parameters
        ----------
        query:
            Natural-language search query.
        ticker, filing_year, filing_type, section:
            Optional metadata filters.
        top_k:
            Number of results to return (default from config).
        method:
            ``"hybrid"`` (default), ``"dense"``, or ``"bm25"``.
        """
        if top_k is None:
            top_k = RAG_TOP_K

        if method == "dense":
            return self._dense_search(
                query, ticker, filing_year, filing_type, section, top_k,
            )
        elif method == "bm25":
            return self._bm25_search(
                query, ticker, filing_year, filing_type, section, top_k,
            )
        else:
            return self._hybrid_search(
                query, ticker, filing_year, filing_type, section, top_k,
            )

    def _dense_search(
        self, query: str, ticker: str | None, filing_year: int | None,
        filing_type: str | None, section: str | None, top_k: int,
    ) -> list[RetrievalResult]:
        query_embedding = embed_query(query)
        results = self._store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            ticker=ticker,
            filing_year=filing_year,
            filing_type=filing_type,
            section=section,
        )
        return [
            self._doc_to_result(doc, method="dense") for doc in results
        ]

    def _bm25_search(
        self, query: str, ticker: str | None, filing_year: int | None,
        filing_type: str | None, section: str | None, top_k: int,
    ) -> list[RetrievalResult]:
        if not self._bm25.is_built:
            self._bm25.load()
        if not self._bm25.is_built:
            logger.warning("BM25 index not available")
            return []

        results = self._bm25.search(
            query=query,
            top_k=top_k,
            ticker=ticker,
            filing_year=filing_year,
            filing_type=filing_type,
            section=section,
        )
        return [
            self._doc_to_result(doc, method="bm25") for doc in results
        ]

    def _hybrid_search(
        self, query: str, ticker: str | None, filing_year: int | None,
        filing_type: str | None, section: str | None, top_k: int,
    ) -> list[RetrievalResult]:
        # Dense retrieval
        query_embedding = embed_query(query)
        dense_results = self._store.query(
            query_embedding=query_embedding,
            top_k=RAG_DENSE_TOP_K,
            ticker=ticker,
            filing_year=filing_year,
            filing_type=filing_type,
            section=section,
        )

        # BM25 retrieval
        bm25_results: list[dict[str, Any]] = []
        if self._bm25.is_built:
            bm25_results = self._bm25.search(
                query=query,
                top_k=RAG_BM25_TOP_K,
                ticker=ticker,
                filing_year=filing_year,
                filing_type=filing_type,
                section=section,
            )

        # Fuse
        if not dense_results and not bm25_results:
            return []

        if not bm25_results:
            return [self._doc_to_result(d, "dense") for d in dense_results[:top_k]]
        if not dense_results:
            return [self._doc_to_result(d, "bm25") for d in bm25_results[:top_k]]

        return rrf_fuse(dense_results, bm25_results, final_top_k=top_k)

    @staticmethod
    def _doc_to_result(doc: dict[str, Any], method: str) -> RetrievalResult:
        meta = doc.get("metadata", {})
        # Dense uses "distance" (lower=better), BM25 uses "score" (higher=better)
        if "distance" in doc:
            score = 1.0 - float(doc["distance"])  # convert cosine distance to similarity
        else:
            score = float(doc.get("score", 0.0))

        year_val = meta.get("filing_year", 0)
        try:
            int_year = int(year_val)
        except (ValueError, TypeError):
            int_year = 0

        return RetrievalResult(
            text=doc.get("text", ""),
            score=round(score, 6),
            chunk_id=doc.get("id", ""),
            ticker=meta.get("ticker", ""),
            company_name=meta.get("company_name", ""),
            filing_year=int_year,
            filing_type=meta.get("filing_type", "10-K"),
            section=meta.get("section", ""),
            item_number=meta.get("item_number", ""),
            accession_number=meta.get("accession_number", ""),
            source_url=meta.get("source_url", ""),
            filing_date=meta.get("filing_date", ""),
            retrieval_method=method,
        )


# ── Backward-compatible functions ─────────────────────────────────────────

def retrieve(
    question: str,
    knowledge_file: str | Path = DEFAULT_KNOWLEDGE_FILE,
    limit: int = 3,
    ticker: str = "",
) -> list[RetrievedChunk] | list[dict]:
    """Backward-compatible retrieve function.

    Uses HybridRetriever if documents are indexed, else falls back
    to keyword matching over the static knowledge base.
    """
    if limit <= 0:
        return []

    # Try hybrid retrieval first
    try:
        retriever = HybridRetriever()
        if retriever._store.count() > 0:
            results = retriever.search(
                query=question,
                ticker=ticker if ticker else None,
                top_k=limit,
            )
            if results:
                return [
                    RetrievedChunk(
                        text=r.text,
                        score=r.score,
                        source=r.section or "vectorstore",
                        metadata={
                            "section_name": r.section,
                            "year": r.filing_year,
                            "ticker": r.ticker,
                            "company_name": r.company_name,
                        },
                    )
                    for r in results
                ]
    except Exception as exc:
        logger.debug("Hybrid retrieval failed, falling back: %s", exc)

    # Keyword fallback over static knowledge base
    terms = _expand_query(_tokenize(question))
    try:
        text_content = Path(knowledge_file).read_text(encoding="utf-8")
    except Exception:
        text_content = ""

    paragraphs = _paragraphs(text_content)
    scored: list[RetrievedChunk] = []
    for paragraph in paragraphs:
        matched = tuple(sorted(terms.intersection(_tokenize(paragraph))))
        score = sum(1.0 + (0.25 if len(term) >= 7 else 0.0) for term in matched)
        if score:
            scored.append(RetrievedChunk(text=paragraph, score=score, matched_terms=matched))
    return sorted(scored, key=lambda item: (-item.score, item.text))[:limit]


def answer_question(
    question: str,
    records: list[MoneyRecord],
    knowledge_file: str | Path = DEFAULT_KNOWLEDGE_FILE,
    ticker: str = "",
) -> str:
    """Answer a financial question using retrieval + financial context."""
    chunks = retrieve(question, knowledge_file, ticker=ticker)
    summary = summarize_financials(records)

    if NVIDIA_API_KEY:
        try:
            from rag.generator import generate_answer
            dict_chunks = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    dict_chunks.append(chunk)
                else:
                    dict_chunks.append({
                        "text": chunk.text,
                        "metadata": {"section_name": chunk.source, "year": summary.get("latest_year")},
                        "distance": chunk.score,
                    })
            return generate_answer(question, dict_chunks, summary, ticker=ticker)
        except Exception:
            pass

    # Fallback: template-based answer
    fallback_chunks = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            fallback_chunks.append(RetrievedChunk(
                text=chunk["text"],
                score=1.0 - chunk.get("distance", 0),
                source=chunk.get("metadata", {}).get("section_name", "vectorstore"),
            ))
        else:
            fallback_chunks.append(chunk)

    context = "\n".join(
        f"- [{chunk.source}; relevance {chunk.score:.2f}] {chunk.text}"
        for chunk in fallback_chunks
    )

    return "\n".join([
        "FinSight RAG Answer",
        f"Question: {question}",
        "",
        "Retrieved context",
        context if context else "- No close local knowledge match found.",
        "",
        "Latest financial context",
        f"- Year: {summary['latest_year']}",
        f"- Revenue: ${summary['revenue']:,.0f}",
        f"- Net margin: {summary['net_margin'] * 100:.1f}%",
        f"- Current ratio: {summary['current_ratio']:.2f}",
        f"- Debt to equity: {summary['debt_to_equity']:.2f}",
        f"- Health signal: {summary['health_signal']}",
        "",
        "Answer",
        _compose_answer(question, fallback_chunks, summary),
    ])


# ── Private helpers ───────────────────────────────────────────────────────

def _compose_answer(question: str, chunks: list[RetrievedChunk], summary: dict) -> str:
    lowered = question.lower()
    if "risk" in lowered:
        return (
            f"The latest health signal is {summary['health_signal']}. "
            f"Net margin is {summary['net_margin'] * 100:.1f}%, current ratio is "
            f"{summary['current_ratio']:.2f}, and debt to equity is {summary['debt_to_equity']:.2f}."
        )
    if "cash" in lowered:
        return "Use free cash flow and operating cash flow to judge whether reported profit is converting into cash."
    if chunks:
        return chunks[0].text
    return "I could not find a close local knowledge match, but the latest financial summary is shown above."


def _paragraphs(text: str) -> list[str]:
    return [
        p.strip().replace("\n", " ")
        for p in text.split("\n\n")
        if p.strip() and not p.startswith("#")
    ]


def _tokenize(text: str) -> set[str]:
    stop_words = {"and", "are", "does", "for", "how", "the", "what", "when", "with"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop_words
    }


def _expand_query(terms: set[str]) -> set[str]:
    aliases = {
        "liquidity": {"current", "ratio", "obligations"},
        "debt": {"leverage", "interest", "coverage"},
        "profitability": {"margin", "profit", "income"},
        "cashflow": {"cash", "flow", "capital"},
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(aliases.get(term, set()))
    return expanded
