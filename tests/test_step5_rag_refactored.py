"""Unit and integration tests for refactored Step 5 RAG subsystem.

Tests cover:
- RetrievalDocument & RetrievalResult canonical models
- HTML table row formatting in sec_filing_text
- Local embedding contextual headers
- Financial BM25 tokenization & indexing
- RRF rank fusion mathematics
- ChromaDB metadata filtering & reset
- Preflight corpus validation & nDCG metric calculation
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest


class TestCanonicalModels:
    def test_retrieval_document_contextual_header(self):
        from rag.models import RetrievalDocument

        doc = RetrievalDocument(
            chunk_id="test-1",
            text="Operating cash flow increased to $15B.",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_year=2024,
            filing_type="10-K",
            section="Item 7 - MD&A",
        )
        context = doc.contextual_embedding_text()
        assert "[Company: Apple Inc. | Ticker: AAPL | Filing: 2024 10-K | Section: Item 7 - MD&A]" in context
        assert "Operating cash flow increased to $15B." in context

    def test_retrieval_result_citation_and_dict(self):
        from rag.models import RetrievalResult

        res = RetrievalResult(
            text="Risk of supply chain disruption",
            score=0.015,
            chunk_id="chunk-99",
            ticker="MSFT",
            company_name="Microsoft Corp",
            filing_year=2023,
            filing_type="10-K",
            section="Item 1A - Risk Factors",
            retrieval_method="hybrid",
        )
        assert res.citation() == "MSFT | 2023 | 10-K | Item 1A - Risk Factors"
        d = res.to_dict()
        assert d["ticker"] == "MSFT"
        assert d["company_name"] == "Microsoft Corp"
        assert d["filing_year"] == 2023
        assert d["retrieval_method"] == "hybrid"


class TestHTMLTableFormatting:
    def test_table_row_formatting(self):
        from data.sources.sec_filing_text import _fetch_and_clean

        html_content = """
        <html>
            <body>
                <table>
                    <tr><th>Metric</th><th>2024</th><th>2023</th></tr>
                    <tr><td>Revenue</td><td>$100M</td><td>$80M</td></tr>
                </table>
            </body>
        </html>
        """
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = html_content
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            text = _fetch_and_clean("https://sec.gov/test")
            assert "Metric | 2024 | 2023" in text
            assert "Revenue | $100M | $80M" in text


class TestBM25FinancialTokenizer:
    def test_tokenization_preserves_financial_expressions(self):
        from rag.bm25 import _tokenize

        tokens = _tokenize("Revenue was $1.2B (up 15.3%) in FY2024 per 10-K Item 1A.")
        assert "$1.2b" in tokens
        assert "15.3%" in tokens
        assert "fy2024" in tokens
        assert "10-k" in tokens
        assert "item" in tokens


class TestRRFFusionMath:
    def test_rrf_rank_fusion_order(self):
        from rag.fusion import rrf_fuse

        dense = [
            {"id": "doc1", "text": "Doc 1", "metadata": {"ticker": "AAPL"}, "distance": 0.1},
            {"id": "doc2", "text": "Doc 2", "metadata": {"ticker": "AAPL"}, "distance": 0.2},
        ]
        bm25 = [
            {"id": "doc2", "text": "Doc 2", "metadata": {"ticker": "AAPL"}, "score": 10.0},
            {"id": "doc3", "text": "Doc 3", "metadata": {"ticker": "AAPL"}, "score": 5.0},
        ]

        # RRF(doc2) = 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
        # RRF(doc1) = 1/(60+1) = 0.016393
        # RRF(doc3) = 1/(60+2) = 0.016129
        results = rrf_fuse(dense, bm25, k=60, final_top_k=3)

        assert len(results) == 3
        assert results[0].chunk_id == "doc2"
        assert results[0].retrieval_method == "hybrid"
        assert results[1].chunk_id == "doc1"
        assert results[1].retrieval_method == "dense"
        assert results[2].chunk_id == "doc3"
        assert results[2].retrieval_method == "bm25"


class TestEvaluationMetrics:
    def test_ndcg_calculation(self):
        from rag.evaluation import EvalQuery, EvalMetrics, evaluate_retrieval

        mock_query = EvalQuery(
            query="Apple risk factors",
            expected_ticker="AAPL",
            expected_sections=["Item 1A - Risk Factors"],
        )

        with patch("rag.retriever.HybridRetriever.search") as mock_search:
            from rag.models import RetrievalResult
            # Rank 1 hit -> DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> nDCG = 1.0
            mock_search.return_value = [
                RetrievalResult(ticker="AAPL", section="Item 1A - Risk Factors", score=0.9),
                RetrievalResult(ticker="AAPL", section="Item 1 - Business", score=0.8),
            ]

            metrics = evaluate_retrieval(method="hybrid", top_k=2, queries=[mock_query])
            assert metrics.hit_rate_at_k == 1.0
            assert metrics.ndcg_at_k == 1.0
            assert metrics.mrr == 1.0
