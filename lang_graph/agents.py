"""LangGraph agent tools for FinSight.

Defines LangChain @tool functions and a helper to create the NVIDIA LLM,
used by the LangGraph agentic workflow.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from config import NVIDIA_API_KEY, NVIDIA_LLM_MODEL

logger = logging.getLogger(__name__)


def get_llm():
    """Return a configured ChatNVIDIA instance."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://build.nvidia.com"
        )
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=NVIDIA_LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=0.2,
        max_completion_tokens=2048,
    )


@tool
def get_financial_report(ticker: str) -> str:
    """Get a detailed financial health report for a stock ticker."""
    from analysis.financial_data import format_report, load_from_ticker

    records = load_from_ticker(ticker)
    return format_report(records)


@tool
def get_risk_score(ticker: str) -> str:
    """Get ML-based risk score and key drivers for a company."""
    from analysis.financial_data import load_from_ticker
    from ml.risk_model import format_risk_report

    records = load_from_ticker(ticker)
    return format_risk_report(records)


@tool
def get_forecast(ticker: str) -> str:
    """Get next-year financial forecast for a company."""
    from analysis.financial_data import load_from_ticker
    from ml.forecast_model import forecast_next_year

    records = load_from_ticker(ticker)
    fc = forecast_next_year(records)
    return str(fc)


@tool
def ask_about_filings(ticker: str, question: str) -> str:
    """Answer a question using RAG over SEC 10-K filings."""
    from analysis.financial_data import load_from_ticker
    from rag.retriever import answer_question

    records = load_from_ticker(ticker)
    return answer_question(question, records, ticker=ticker)


@tool
def ingest_sec_filings(ticker: str) -> str:
    """Fetch and ingest recent 10-K filings for a company into the RAG knowledge base."""
    from rag.ingest import ingest_filings

    count = ingest_filings(ticker)
    return f"Ingested {count} chunks for {ticker}"


@tool
def get_company_summary(ticker: str) -> str:
    """Get a quick financial summary with key metrics."""
    from analysis.financial_data import load_from_ticker, summarize_financials

    records = load_from_ticker(ticker)
    summary = summarize_financials(records)
    return str(summary)


AGENT_TOOLS = [
    get_financial_report,
    get_risk_score,
    get_forecast,
    ask_about_filings,
    ingest_sec_filings,
    get_company_summary,
]
