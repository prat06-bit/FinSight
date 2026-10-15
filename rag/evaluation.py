"""Retrieval evaluation for the hybrid RAG system.

Contains a 25-query evaluation dataset, preflight corpus validation,
nDCG@K metrics, and structured failure analysis.
Run as a module: ``python -m rag.evaluation``
"""

from __future__ import annotations

import math
import logging
import sys
import time
from dataclasses import dataclass, field

from rag.vectorstore import FinSightVectorStore

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    """One evaluation query with expected retrieval targets."""

    query: str
    expected_ticker: str
    expected_sections: list[str]  # acceptable section names (any match = hit)
    expected_filing_year: int | None = None  # optional year filter
    description: str = ""


# ── Evaluation Dataset (25 queries across 10 companies) ─────────────────

EVAL_QUERIES: list[EvalQuery] = [
    # Business description queries
    EvalQuery(
        query="What products and services does Apple offer?",
        expected_ticker="AAPL",
        expected_sections=["Item 1 - Business"],
        description="Business overview",
    ),
    EvalQuery(
        query="Describe Microsoft's cloud computing business segments",
        expected_ticker="MSFT",
        expected_sections=["Item 1 - Business"],
        description="Segment description",
    ),
    EvalQuery(
        query="What are Google's main sources of advertising revenue?",
        expected_ticker="GOOGL",
        expected_sections=["Item 1 - Business", "Item 7 - MD&A"],
        description="Revenue sources",
    ),
    # Risk factor queries
    EvalQuery(
        query="What are the key risk factors for Amazon?",
        expected_ticker="AMZN",
        expected_sections=["Item 1A - Risk Factors"],
        description="General risk factors",
    ),
    EvalQuery(
        query="Supply chain disruption risks for Apple",
        expected_ticker="AAPL",
        expected_sections=["Item 1A - Risk Factors"],
        description="Supply chain risks",
    ),
    EvalQuery(
        query="Cybersecurity and data privacy risks at Meta",
        expected_ticker="META",
        expected_sections=["Item 1A - Risk Factors"],
        description="Cyber risks",
    ),
    EvalQuery(
        query="Regulatory and antitrust risks facing Google",
        expected_ticker="GOOGL",
        expected_sections=["Item 1A - Risk Factors"],
        description="Regulatory risks",
    ),
    EvalQuery(
        query="Competition risks in the semiconductor industry for NVIDIA",
        expected_ticker="NVDA",
        expected_sections=["Item 1A - Risk Factors", "Item 1 - Business"],
        description="Competition risks",
    ),
    # MD&A queries (financial performance discussion)
    EvalQuery(
        query="What drove revenue growth for Microsoft?",
        expected_ticker="MSFT",
        expected_sections=["Item 7 - MD&A"],
        description="Revenue drivers",
    ),
    EvalQuery(
        query="Discuss Apple's liquidity and capital resources",
        expected_ticker="AAPL",
        expected_sections=["Item 7 - MD&A"],
        description="Liquidity discussion",
    ),
    EvalQuery(
        query="Operating expenses and cost management at Amazon",
        expected_ticker="AMZN",
        expected_sections=["Item 7 - MD&A"],
        description="OpEx discussion",
    ),
    EvalQuery(
        query="Cash flow from operations for Intel",
        expected_ticker="INTC",
        expected_sections=["Item 7 - MD&A", "Item 8 - Financial Statements"],
        description="Cash flow analysis",
    ),
    EvalQuery(
        query="Capital expenditure and infrastructure investment by AMD",
        expected_ticker="AMD",
        expected_sections=["Item 7 - MD&A"],
        description="CapEx discussion",
    ),
    # Market risk queries
    EvalQuery(
        query="Foreign currency exchange rate risk exposure for Oracle",
        expected_ticker="ORCL",
        expected_sections=["Item 7A - Market Risk", "Item 7 - MD&A"],
        description="FX risk",
    ),
    EvalQuery(
        query="Interest rate sensitivity and debt instruments for Salesforce",
        expected_ticker="CRM",
        expected_sections=["Item 7A - Market Risk", "Item 7 - MD&A"],
        description="Interest rate risk",
    ),
    # Financial statement queries
    EvalQuery(
        query="Total debt and long-term obligations for Intel",
        expected_ticker="INTC",
        expected_sections=["Item 8 - Financial Statements", "Item 7 - MD&A"],
        description="Debt structure",
    ),
    EvalQuery(
        query="Goodwill and intangible assets on the balance sheet of Microsoft",
        expected_ticker="MSFT",
        expected_sections=["Item 8 - Financial Statements"],
        description="Balance sheet items",
    ),
    # Cross-section queries
    EvalQuery(
        query="Stock-based compensation expenses at NVIDIA",
        expected_ticker="NVDA",
        expected_sections=["Item 7 - MD&A", "Item 8 - Financial Statements"],
        description="SBC expenses",
    ),
    EvalQuery(
        query="Research and development spending by AMD",
        expected_ticker="AMD",
        expected_sections=["Item 7 - MD&A", "Item 1 - Business", "Item 8 - Financial Statements"],
        description="R&D spending",
    ),
    EvalQuery(
        query="Share repurchase and buyback program for Apple",
        expected_ticker="AAPL",
        expected_sections=["Item 7 - MD&A", "Item 8 - Financial Statements"],
        description="Buyback program",
    ),
    # Specific financial metrics
    EvalQuery(
        query="Gross margin trends and cost of revenue for Amazon",
        expected_ticker="AMZN",
        expected_sections=["Item 7 - MD&A"],
        description="Margin analysis",
    ),
    EvalQuery(
        query="Deferred revenue and contract liabilities for Salesforce",
        expected_ticker="CRM",
        expected_sections=["Item 8 - Financial Statements", "Item 7 - MD&A"],
        description="Deferred revenue",
    ),
    # Industry-specific
    EvalQuery(
        query="Data center GPU demand and AI training workload growth at NVIDIA",
        expected_ticker="NVDA",
        expected_sections=["Item 1 - Business", "Item 7 - MD&A"],
        description="AI/GPU demand",
    ),
    EvalQuery(
        query="Cloud infrastructure AWS and marketplace at Amazon",
        expected_ticker="AMZN",
        expected_sections=["Item 1 - Business", "Item 7 - MD&A"],
        description="AWS cloud",
    ),
    EvalQuery(
        query="Acquisition strategy and integration risks for Oracle",
        expected_ticker="ORCL",
        expected_sections=["Item 1A - Risk Factors", "Item 7 - MD&A"],
        description="M&A strategy",
    ),
]


@dataclass
class EvalMetrics:
    """Metrics from evaluating a retrieval method."""

    method: str = ""
    total_queries: int = 0
    hit_rate_at_k: float = 0.0
    mrr: float = 0.0
    avg_precision_at_k: float = 0.0
    avg_recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    query_details: list[dict] = field(default_factory=list)


def validate_preflight_corpus(queries: list[EvalQuery] | None = None) -> tuple[bool, dict[str, bool]]:
    """Verify that all target tickers in evaluation queries exist in ChromaDB."""
    store = FinSightVectorStore()
    eval_queries = queries or EVAL_QUERIES
    expected_tickers = sorted({eq.expected_ticker for eq in eval_queries})

    status_map: dict[str, bool] = {}
    for ticker in expected_tickers:
        status_map[ticker] = store.has_documents(ticker)

    all_present = all(status_map.values())
    return all_present, status_map


def evaluate_retrieval(
    method: str = "hybrid",
    top_k: int = 5,
    queries: list[EvalQuery] | None = None,
) -> EvalMetrics:
    """Run evaluation queries and compute retrieval metrics including nDCG@K."""
    from rag.retriever import HybridRetriever

    retriever = HybridRetriever()
    eval_queries = queries or EVAL_QUERIES

    hits = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    details: list[dict] = []

    for eq in eval_queries:
        results = retriever.search(
            query=eq.query,
            ticker=eq.expected_ticker,
            top_k=top_k,
            method=method,
        )

        hit = False
        first_relevant_rank = 0
        relevant_count = 0
        dcg = 0.0

        for rank, r in enumerate(results, 1):
            section_match = r.section in eq.expected_sections
            year_match = (eq.expected_filing_year is None or r.filing_year == eq.expected_filing_year)
            if section_match and year_match:
                relevant_count += 1
                dcg += 1.0 / math.log2(rank + 1)
                if not hit:
                    hit = True
                    first_relevant_rank = rank

        # Calculate Ideal DCG (IDCG) assuming max possible relevant hits at top positions
        max_possible_rel = min(top_k, len(eq.expected_sections))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, max_possible_rel + 1)) if max_possible_rel > 0 else 1.0
        ndcg = dcg / idcg if idcg > 0 else 0.0

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

        precision = relevant_count / len(results) if results else 0.0
        found_sections = {r.section for r in results if r.section in eq.expected_sections}
        recall = len(found_sections) / len(eq.expected_sections) if eq.expected_sections else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

        details.append({
            "query": eq.query,
            "ticker": eq.expected_ticker,
            "expected_sections": eq.expected_sections,
            "hit": hit,
            "relevant_count": relevant_count,
            "total_results": len(results),
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "rr": reciprocal_ranks[-1],
            "top_results": [
                {
                    "rank": i + 1,
                    "section": r.section,
                    "year": r.filing_year,
                    "score": r.score,
                    "method": r.retrieval_method,
                    "preview": r.text[:100].replace("\n", " "),
                }
                for i, r in enumerate(results[:3])
            ],
        })

    n = len(eval_queries)
    return EvalMetrics(
        method=method,
        total_queries=n,
        hit_rate_at_k=hits / n if n else 0.0,
        mrr=sum(reciprocal_ranks) / n if n else 0.0,
        avg_precision_at_k=sum(precisions) / n if n else 0.0,
        avg_recall_at_k=sum(recalls) / n if n else 0.0,
        ndcg_at_k=sum(ndcgs) / n if n else 0.0,
        query_details=details,
    )


def run_full_evaluation(top_k: int = 5) -> tuple[list[EvalMetrics], dict[str, bool]]:
    """Run evaluation for all methods, check preflight status, and print results."""
    all_present, corpus_status = validate_preflight_corpus()
    
    print("=" * 80)
    print("Preflight Corpus Check")
    print("=" * 80)
    for ticker, present in corpus_status.items():
        tag = "[PRESENT]" if present else "[MISSING]"
        print(f"  {tag:<10} {ticker}")
    print("-" * 80)

    if not all_present:
        missing_count = sum(1 for v in corpus_status.values() if not v)
        print(f"WARNING: {missing_count} expected evaluation tickers missing from corpus.")
        print("Run 'python main.py ingest --all' to populate the complete multi-company corpus.\n")

    methods = ["dense", "bm25", "hybrid"]
    all_metrics: list[EvalMetrics] = []

    for method in methods:
        print(f"Evaluating {method}...")
        t0 = time.perf_counter()
        metrics = evaluate_retrieval(method=method, top_k=top_k)
        elapsed = time.perf_counter() - t0
        all_metrics.append(metrics)
        print(f"  {method}: {elapsed:.1f}s")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"Retrieval Evaluation Results (Top-{top_k})")
    print(f"{'='*80}")
    print(f"{'Method':<10} {'Queries':<10} {'Hit@K':<10} {'MRR':<10} {'Prec@K':<10} {'Recall@K':<10} {'nDCG@K':<10}")
    print(f"{'-'*70}")
    for m in all_metrics:
        print(
            f"{m.method:<10} {m.total_queries:<10} "
            f"{m.hit_rate_at_k:<10.3f} {m.mrr:<10.3f} "
            f"{m.avg_precision_at_k:<10.3f} {m.avg_recall_at_k:<10.3f} "
            f"{m.ndcg_at_k:<10.3f}"
        )

    # Print Failure Analysis for Hybrid
    hybrid = next((m for m in all_metrics if m.method == "hybrid"), None)
    if hybrid:
        failed_queries = [d for d in hybrid.query_details if not d["hit"]]
        print(f"\n{'='*80}")
        print(f"Failure Analysis (Hybrid: {len(failed_queries)} / {len(hybrid.query_details)} unretrieved queries)")
        print(f"{'='*80}")
        for d in failed_queries:
            has_ticker = corpus_status.get(d["ticker"], False)
            reason = "Corpus Missing" if not has_ticker else "Section/Ranking Miss"
            print(f"  [-] [{d['ticker']}] [{reason}] Query: {d['query']}")
            if d["top_results"]:
                top1 = d["top_results"][0]
                print(f"      Top Result: {top1['section']} ({top1['method']}, score={top1['score']:.4f})")
                print(f"      Snippet: {top1['preview']}...")
            else:
                print("      No results returned.")

    return all_metrics, corpus_status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_full_evaluation(top_k=top_k)
