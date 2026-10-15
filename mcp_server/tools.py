from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from analysis.financial_data import load_from_ticker, load_financial_data, format_report, summarize_financials
from ml.risk_model import predict_financial_risk, format_risk_report
from ml.forecast_model import forecast_next_year
from rag.retriever import answer_question
from config import DEFAULT_DATA_FILE

def register_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def financial_report(ticker: str) -> str:
        """Generate a financial health report for a given stock ticker.
        Args:
            ticker: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
        """
        records = load_from_ticker(ticker)
        return format_report(records)

    @mcp.tool()
    def risk_score(ticker: str) -> dict:
        """Get the ML-based financial risk score for a company.
        Args:
            ticker: Stock ticker symbol
        """
        records = load_from_ticker(ticker)
        prediction = predict_financial_risk(records)
        return {
            'ticker': ticker,
            'year': prediction.year,
            'score': prediction.score,
            'label': prediction.label,
            'drivers': prediction.drivers,
            'confidence': prediction.confidence,
        }

    @mcp.tool()
    def forecast(ticker: str) -> dict:
        """Return a next-year financial forecast for a company.
        Args:
            ticker: Stock ticker symbol
        """
        records = load_from_ticker(ticker)
        return forecast_next_year(records)

    @mcp.tool()
    def ask_financials(ticker: str, question: str) -> str:
        """Answer a financial question using RAG over SEC filings and financial data.
        Args:
            ticker: Stock ticker symbol for context
            question: The financial question to answer
        """
        records = load_from_ticker(ticker)
        return answer_question(question, records, ticker=ticker)

    @mcp.tool()
    def company_summary(ticker: str) -> dict:
        """Get key financial metrics summary for a company.
        Args:
            ticker: Stock ticker symbol
        """
        records = load_from_ticker(ticker)
        return summarize_financials(records)

    @mcp.tool()
    def ingest_filings(ticker: str, num_filings: int = 3) -> str:
        """Fetch and ingest SEC 10-K filings into the RAG knowledge base.
        Args:
            ticker: Stock ticker symbol
            num_filings: Number of recent 10-K filings to ingest (default 3)
        """
        from rag.ingest import ingest_filings as _ingest
        count = _ingest(ticker, num_filings=num_filings)
        return f'Ingested {count} chunks from {num_filings} filings for {ticker}'
