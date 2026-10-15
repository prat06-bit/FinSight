"""Unit tests for Step 5 — Hybrid RAG over SEC Financial Filings.

Tests cover:
- RAG data models
- BM25 tokenization and indexing
- RRF fusion correctness
- ChromaDB vectorstore operations
- Ingest pipeline flow (mocked SEC fetches)
- HybridRetriever orchestration
- Metadata filtering
- Evaluation framework
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

# ── RAG Models ────────────────────────────────────────────────────────────


class TestRetrievalResult:
    def test_citation_format(self):
        from rag.models import RetrievalResult

        r = RetrievalResult(
            ticker="AAPL", filing_year=2024, filing_type="10-K",
            section="Item 1A - Risk Factors",
        )
        assert r.citation() == "AAPL | 2024 | 10-K | Item 1A - Risk Factors"

    def test_citation_no_section(self):
        from rag.models import RetrievalResult

        r = RetrievalResult(ticker="MSFT", filing_year=2023, filing_type="10-K")
        assert r.citation() == "MSFT | 2023 | 10-K"

    def test_to_dict_keys(self):
        from rag.models import RetrievalResult

        r = RetrievalResult(text="test", score=0.9, ticker="AAPL")
        d = r.to_dict()
        expected_keys = {
            "text", "score", "chunk_id", "ticker", "company_name", "filing_year",
            "filing_type", "section", "item_number", "accession_number", "source_url",
            "filing_date", "retrieval_method",
        }
        assert set(d.keys()) == expected_keys

    def test_default_method_is_hybrid(self):
        from rag.models import RetrievalResult

        r = RetrievalResult()
        assert r.retrieval_method == "hybrid"


# ── BM25 Tokenization ────────────────────────────────────────────────────


class TestBM25Tokenization:
    def test_basic_tokenization(self):
        from rag.bm25 import _tokenize

        tokens = _tokenize("Apple reported strong revenue growth in fiscal 2024")
        assert "apple" in tokens
        assert "revenue" in tokens
        assert "growth" in tokens
        assert "fiscal" in tokens
        # Stop words removed
        assert "in" not in tokens

    def test_financial_terms_preserved(self):
        from rag.bm25 import _tokenize

        tokens = _tokenize("The EBITDA margin improved. GAAP results per 10-K filing.")
        assert "ebitda" in tokens
        assert "gaap" in tokens

    def test_dollar_amounts(self):
        from rag.bm25 import _tokenize

        tokens = _tokenize("Revenue was $1.2B and expenses were $500M")
        assert "$1.2b" in tokens or "$1.2B" in [t for t in tokens]
        assert "$500m" in tokens or "$500M" in [t for t in tokens]

    def test_percentages(self):
        from rag.bm25 import _tokenize

        tokens = _tokenize("Gross margin was 45.3% compared to 42.1%")
        assert "45.3%" in tokens
        assert "42.1%" in tokens

    def test_empty_input(self):
        from rag.bm25 import _tokenize

        assert _tokenize("") == []

    def test_stop_words_removed(self):
        from rag.bm25 import _tokenize

        tokens = _tokenize("The company is a leader in the market")
        assert "the" not in tokens
        assert "is" not in tokens


# ── BM25 Index ────────────────────────────────────────────────────────────


class TestBM25Index:
    def _sample_docs(self):
        return [
            {"id": "1", "text": "Apple revenue grew 15% driven by iPhone sales", "metadata": {"ticker": "AAPL", "filing_year": 2024, "section": "Item 7 - MD&A"}},
            {"id": "2", "text": "Risk factors include supply chain disruption", "metadata": {"ticker": "AAPL", "filing_year": 2024, "section": "Item 1A - Risk Factors"}},
            {"id": "3", "text": "Microsoft cloud revenue from Azure exceeded expectations", "metadata": {"ticker": "MSFT", "filing_year": 2024, "section": "Item 7 - MD&A"}},
            {"id": "4", "text": "Total long-term debt was $50 billion at year end", "metadata": {"ticker": "MSFT", "filing_year": 2024, "section": "Item 8 - Financial Statements"}},
            {"id": "5", "text": "Cybersecurity threats represent significant operational risk", "metadata": {"ticker": "META", "filing_year": 2023, "section": "Item 1A - Risk Factors"}},
        ]

    def test_build_and_search(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        idx.build(self._sample_docs())
        assert idx.is_built
        assert idx.document_count() == 5

        results = idx.search("revenue growth", top_k=3)
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_ticker_filtering(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        idx.build(self._sample_docs())

        results = idx.search("revenue", top_k=5, ticker="AAPL")
        for r in results:
            assert r["metadata"]["ticker"] == "AAPL"

    def test_section_filtering(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        idx.build(self._sample_docs())

        results = idx.search("risk", top_k=5, section="Item 1A - Risk Factors")
        for r in results:
            assert r["metadata"]["section"] == "Item 1A - Risk Factors"

    def test_empty_index(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        assert not idx.is_built
        assert idx.search("anything") == []

    def test_no_results_for_unmatched_query(self):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        idx.build(self._sample_docs())
        results = idx.search("quantum computing nanotechnology", top_k=5)
        # May return 0 results or low-score results
        if results:
            assert results[0]["score"] >= 0

    def test_save_and_load(self, tmp_path):
        from rag.bm25 import BM25Index

        idx = BM25Index()
        idx.build(self._sample_docs())
        save_path = tmp_path / "test_bm25.pkl"
        idx.save(save_path)
        assert save_path.exists()

        idx2 = BM25Index()
        loaded = idx2.load(save_path)
        assert loaded
        assert idx2.is_built
        assert idx2.document_count() == 5

        # Results should be the same
        r1 = idx.search("revenue", top_k=2)
        r2 = idx2.search("revenue", top_k=2)
        assert len(r1) == len(r2)
        assert r1[0]["id"] == r2[0]["id"]


# ── RRF Fusion ────────────────────────────────────────────────────────────


class TestRRFFusion:
    def test_basic_fusion(self):
        from rag.fusion import rrf_fuse

        dense = [
            {"id": "a", "text": "Doc A", "metadata": {"ticker": "AAPL"}, "distance": 0.1},
            {"id": "b", "text": "Doc B", "metadata": {"ticker": "AAPL"}, "distance": 0.2},
            {"id": "c", "text": "Doc C", "metadata": {"ticker": "AAPL"}, "distance": 0.3},
        ]
        bm25 = [
            {"id": "b", "text": "Doc B", "metadata": {"ticker": "AAPL"}, "score": 5.0},
            {"id": "d", "text": "Doc D", "metadata": {"ticker": "AAPL"}, "score": 3.0},
            {"id": "a", "text": "Doc A", "metadata": {"ticker": "AAPL"}, "score": 1.0},
        ]
        results = rrf_fuse(dense, bm25, k=60, final_top_k=5)

        # "a" and "b" appear in both lists → should score higher
        ids = [r.chunk_id for r in results]
        assert "a" in ids
        assert "b" in ids
        # "b" is rank 1 in BM25 and rank 2 in dense → should be top or near top
        assert results[0].chunk_id in ("a", "b")

    def test_deduplication(self):
        from rag.fusion import rrf_fuse

        dense = [
            {"id": "x", "text": "Same doc", "metadata": {}, "distance": 0.1},
        ]
        bm25 = [
            {"id": "x", "text": "Same doc", "metadata": {}, "score": 5.0},
        ]
        results = rrf_fuse(dense, bm25, k=60, final_top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "x"
        assert results[0].retrieval_method == "hybrid"

    def test_empty_inputs(self):
        from rag.fusion import rrf_fuse

        results = rrf_fuse([], [], k=60, final_top_k=5)
        assert results == []

    def test_dense_only(self):
        from rag.fusion import rrf_fuse

        dense = [
            {"id": "a", "text": "Doc A", "metadata": {"ticker": "AAPL"}, "distance": 0.1},
        ]
        results = rrf_fuse(dense, [], k=60, final_top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "a"

    def test_bm25_only(self):
        from rag.fusion import rrf_fuse

        bm25 = [
            {"id": "b", "text": "Doc B", "metadata": {"ticker": "MSFT"}, "score": 3.0},
        ]
        results = rrf_fuse([], bm25, k=60, final_top_k=5)
        assert len(results) == 1
        assert results[0].ticker == "MSFT"

    def test_metadata_preserved(self):
        from rag.fusion import rrf_fuse

        dense = [
            {
                "id": "z", "text": "Content",
                "metadata": {
                    "ticker": "NVDA", "filing_year": 2024,
                    "filing_type": "10-K", "section": "Item 7 - MD&A",
                    "accession_number": "0001-23-456",
                    "source_url": "https://sec.gov/...",
                    "filing_date": "2024-02-15",
                },
                "distance": 0.05,
            },
        ]
        results = rrf_fuse(dense, [], k=60, final_top_k=1)
        r = results[0]
        assert r.ticker == "NVDA"
        assert r.filing_year == 2024
        assert r.section == "Item 7 - MD&A"
        assert r.accession_number == "0001-23-456"

    def test_final_top_k_limit(self):
        from rag.fusion import rrf_fuse

        dense = [{"id": str(i), "text": f"Doc {i}", "metadata": {}, "distance": i * 0.1} for i in range(10)]
        results = rrf_fuse(dense, [], k=60, final_top_k=3)
        assert len(results) == 3


# ── Vectorstore ───────────────────────────────────────────────────────────


class TestVectorStore:
    """Test ChromaDB operations using a temporary collection."""

    def _make_store(self):
        from rag.vectorstore import FinSightVectorStore

        store = FinSightVectorStore(collection_name="test_finsight_step5")
        store.reset()
        return store

    def test_add_and_query(self):
        store = self._make_store()
        try:
            texts = ["Apple revenue increased significantly"]
            # Use a simple fake embedding (384-dim to match model output size)
            embeddings = [[0.1] * 384]
            metadatas = [{"ticker": "AAPL", "filing_year": 2024, "section": "Item 7 - MD&A"}]
            ids = ["test-001"]

            store.add_documents(texts, embeddings, metadatas, ids)
            assert store.count() == 1
            assert store.has_documents("AAPL")

            results = store.query(query_embedding=[0.1] * 384, top_k=1)
            assert len(results) == 1
            assert results[0]["text"] == texts[0]
            assert results[0]["metadata"]["ticker"] == "AAPL"
        finally:
            store.reset()

    def test_metadata_filtering(self):
        store = self._make_store()
        try:
            store.add_documents(
                texts=["Doc A", "Doc B"],
                embeddings=[[0.1] * 384, [0.2] * 384],
                metadatas=[
                    {"ticker": "AAPL", "filing_year": 2024, "section": "Item 1 - Business"},
                    {"ticker": "MSFT", "filing_year": 2023, "section": "Item 7 - MD&A"},
                ],
                ids=["a", "b"],
            )

            # Filter by ticker
            results = store.query(query_embedding=[0.15] * 384, top_k=5, ticker="AAPL")
            assert len(results) == 1
            assert results[0]["metadata"]["ticker"] == "AAPL"

            # Filter by year
            results = store.query(query_embedding=[0.15] * 384, top_k=5, filing_year=2023)
            assert len(results) == 1
            assert results[0]["metadata"]["filing_year"] == 2023
        finally:
            store.reset()

    def test_delete_ticker(self):
        store = self._make_store()
        try:
            store.add_documents(
                texts=["Doc A"],
                embeddings=[[0.1] * 384],
                metadatas=[{"ticker": "AAPL"}],
                ids=["del-1"],
            )
            assert store.has_documents("AAPL")
            store.delete_ticker("AAPL")
            assert not store.has_documents("AAPL")
        finally:
            store.reset()

    def test_get_all_metadata(self):
        store = self._make_store()
        try:
            store.add_documents(
                texts=["Doc 1", "Doc 2"],
                embeddings=[[0.1] * 384, [0.2] * 384],
                metadatas=[{"ticker": "AAPL"}, {"ticker": "MSFT"}],
                ids=["m1", "m2"],
            )
            all_docs = store.get_all_metadata()
            assert len(all_docs) == 2
            tickers = {d["metadata"]["ticker"] for d in all_docs}
            assert tickers == {"AAPL", "MSFT"}
        finally:
            store.reset()


# ── Ingest Pipeline ───────────────────────────────────────────────────────


class TestIngestPipeline:
    """Test ingestion logic with mocked SEC data."""

    def test_ingest_filings_mock(self):
        """Verify ingest correctly chains URL resolution → download → chunk → embed → store."""
        from rag.ingest import ingest_filings
        from data.sources.sec_filing_text import FilingSection

        mock_filing_urls = [
            {"form": "10-K", "date": "2024-02-01", "accession": "0001-24-001", "url": "https://sec.gov/test"},
        ]
        mock_sections = [
            FilingSection(
                ticker="AAPL", year=2024,
                section_name="Item 1 - Business",
                text="Apple designs and manufactures consumer electronics. " * 50,
                filing_date="2024-02-01",
                source_url="https://sec.gov/test",
            ),
        ]

        with (
            patch("rag.ingest.SECEdgarSource") as mock_source_cls,
            patch("rag.ingest.download_filing_text", return_value=mock_sections),
            patch("rag.ingest.embed_texts", return_value=[[0.1] * 384]),
            patch("rag.ingest.FinSightVectorStore") as mock_store_cls,
        ):
            mock_source = MagicMock()
            mock_source.get_filing_urls.return_value = mock_filing_urls
            mock_source_cls.return_value = mock_source

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            count = ingest_filings("AAPL", num_filings=1)

            assert count > 0
            mock_source.get_filing_urls.assert_called_once()
            mock_store.add_documents.assert_called()

    def test_ingest_handles_empty_filings(self):
        from rag.ingest import ingest_filings

        with patch("rag.ingest.SECEdgarSource") as mock_source_cls:
            mock_source = MagicMock()
            mock_source.get_filing_urls.return_value = []
            mock_source_cls.return_value = mock_source

            count = ingest_filings("XXXX", num_filings=1)
            assert count == 0


# ── Retriever Orchestration ───────────────────────────────────────────────


class TestHybridRetriever:
    def test_instantiation(self):
        from rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        assert retriever is not None

    def test_search_empty_store(self):
        """Search on empty store should return empty list, not error."""
        from rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        # Use a temporary collection to avoid interfering with real data
        retriever._store = MagicMock()
        retriever._store.count.return_value = 0
        retriever._store.query.return_value = []
        retriever._bm25 = MagicMock()
        retriever._bm25.is_built = False

        results = retriever.search("test query", method="hybrid")
        assert results == []


# ── Backward Compatibility ────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_retrieve_falls_back_to_keywords(self):
        """Without indexed docs, retrieve() should fall back to keyword matching."""
        from rag.retriever import retrieve, RetrievedChunk

        chunks = retrieve("What is the current ratio benchmark?", limit=2)
        # Should return keyword-matched chunks from knowledge_base.md
        assert isinstance(chunks, list)
        if chunks:
            assert isinstance(chunks[0], RetrievedChunk)

    def test_answer_question_format(self):
        """answer_question should return a string without errors."""
        from rag.retriever import answer_question

        records = [
            {"year": 2024, "revenue": 1000, "cost_of_goods_sold": 400,
             "operating_expenses": 200, "net_income": 200,
             "total_assets": 5000, "total_equity": 2000, "total_debt": 1000,
             "current_assets": 1500, "current_liabilities": 800}
        ]
        answer = answer_question("What is the risk?", records, ticker="TEST")
        assert isinstance(answer, str)
        assert len(answer) > 0


# ── Evaluation Framework ─────────────────────────────────────────────────


class TestEvaluation:
    def test_eval_queries_well_formed(self):
        from rag.evaluation import EVAL_QUERIES

        assert len(EVAL_QUERIES) >= 25
        for eq in EVAL_QUERIES:
            assert eq.query
            assert eq.expected_ticker
            assert len(eq.expected_sections) > 0

    def test_eval_metrics_dataclass(self):
        from rag.evaluation import EvalMetrics

        m = EvalMetrics(method="test", total_queries=10, hit_rate_at_k=0.8, mrr=0.6)
        assert m.method == "test"
        assert m.total_queries == 10
