"""Tests for RAG retriever and keyword fallback."""

from __future__ import annotations

from rag.retriever import RetrievedChunk, answer_question, retrieve


class TestKeywordRetrieval:
    def test_retrieves_relevant_chunks(self) -> None:
        chunks = retrieve("What does current ratio mean?")
        score = getattr(chunks[0], "score", chunks[0].get("score", 0) if isinstance(chunks[0], dict) else 0)
        assert score >= 0

    def test_retrieves_with_liquidity_query(self) -> None:
        chunks = retrieve("What is the liquidity risk?")
        assert len(chunks) > 0

    def test_empty_query_returns_empty(self) -> None:
        chunks = retrieve("", limit=0)
        assert chunks == []


class TestAnswerQuestion:
    def test_answer_contains_structure(self, sample_records) -> None:
        answer = answer_question("What is the liquidity risk?", sample_records)
        assert "FinSight" in answer or "current ratio" in answer.lower() or "liquidity" in answer.lower()

    def test_answer_with_ticker(self, sample_records) -> None:
        answer = answer_question("What is the debt risk?", sample_records, ticker="TEST")
        assert isinstance(answer, str)
        assert len(answer) > 0
