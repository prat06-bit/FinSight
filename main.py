from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis.financial_data import format_report, load_financial_data, load_from_ticker
from config import DEFAULT_DATA_FILE
from lang_graph.workflow import LangGraphNotInstalledError, run_langgraph_workflow
from ml.risk_model import format_risk_report
from rag.retriever import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)

def _load_records(args: argparse.Namespace) -> tuple[list, str]:
    """Load financial records from ticker or CSV file."""
    if getattr(args, "ticker", None):
        return load_from_ticker(args.ticker, source=args.source), args.ticker.upper()
    return load_financial_data(args.data), ""

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinSight — Financial analysis powered by real data, ML, and AI.",
    )
    parser.add_argument(
        "--ticker", "-t", type=str,
        help="Stock ticker symbol (e.g., AAPL, MSFT). Primary input.",
    )
    parser.add_argument(
        "--data", "-d", default=DEFAULT_DATA_FILE, type=Path,
        help="Path to CSV file (legacy mode).",
    )
    parser.add_argument(
        "--source", "-s", default="sec_edgar",
        choices=["sec_edgar", "yahoo"],
        help="Data source (default: sec_edgar).",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("report", help="Print the financial report.")
    subparsers.add_parser("risk", help="Print the ML risk report.")
    subparsers.add_parser("forecast", help="Print the time-series forecasting report.")
    ask_parser = subparsers.add_parser("ask", help="Ask a question (RAG).")
    ask_parser.add_argument("question", nargs="+", help="Question to answer.")
    workflow_parser = subparsers.add_parser("workflow", help="Run the LangGraph analysis workflow.")
    workflow_parser.add_argument("question", nargs="*", help="Optional question.")
    
    ingest_parser = subparsers.add_parser("ingest", help="Ingest SEC 10-K filings into RAG.")
    ingest_parser.add_argument("--ticker", "-t", type=str, help="Stock ticker symbol (e.g., AAPL).")
    ingest_parser.add_argument("--all", "-a", action="store_true", help="Ingest complete 10-company evaluation corpus.")

    search_parser = subparsers.add_parser("search", help="Search indexed SEC filings (hybrid RAG).")
    search_parser.add_argument("query", nargs="+", help="Search query.")
    search_parser.add_argument("--ticker", "-t", type=str, help="Stock ticker symbol (e.g., AAPL).")
    search_parser.add_argument("--method", default="hybrid", choices=["hybrid", "dense", "bm25"], help="Retrieval method.")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results.")

    subparsers.add_parser("rag-stats", help="Display summary statistics for the RAG index and corpus.")
    subparsers.add_parser("train", help="Train the ML risk model.")
    subparsers.add_parser("mcp-server", help="Start the MCP server (stdio transport).")

    args = parser.parse_args()
    command = args.command or "report"

    if command == "report":
        records, ticker = _load_records(args)
        print(format_report(records))

    elif command == "risk":
        records, ticker = _load_records(args)
        print(format_risk_report(records))

    elif command == "forecast":
        ticker = args.ticker.upper() if getattr(args, "ticker", None) else "AAPL"
        from ml.forecasting import format_forecasting_report
        print(format_forecasting_report(ticker))

    elif command == "ask":
        records, ticker = _load_records(args)
        question = " ".join(args.question)
        print(answer_question(question, records, ticker=ticker))

    elif command == "workflow":
        question = " ".join(args.question) if args.question else None
        identifier = args.ticker if args.ticker else str(args.data)
        try:
            print(run_langgraph_workflow(identifier, question))
        except LangGraphNotInstalledError as exc:
            parser.exit(status=2, message=f"{exc}\n")

    elif command == "ingest":
        from rag.ingest import ingest_filings, ingest_all_tickers, build_bm25_index
        if args.all:
            print("Starting bulk ingestion for 10-company benchmark corpus...")
            res = ingest_all_tickers()
            print("\nCorpus Ingestion Summary:")
            for t, c in res.items():
                print(f"  {t:<8}: {c} chunks")
            print(f"Total Chunks: {sum(res.values())}")
        elif args.ticker:
            count = ingest_filings(args.ticker)
            print(f"Ingested {count} chunks for {args.ticker}")
            print("Rebuilding BM25 index...")
            idx = build_bm25_index()
            print(f"BM25 index: {idx.document_count()} documents")
        else:
            parser.error("Specify --ticker TICKER or --all for ingest command")

    elif command == "search":
        from rag.retriever import HybridRetriever
        query = " ".join(args.query)
        retriever = HybridRetriever()
        ticker_val = args.ticker.upper() if args.ticker else None
        results = retriever.search(
            query=query,
            ticker=ticker_val,
            top_k=args.top_k,
            method=args.method,
        )
        print(f"\n{'='*80}")
        print(f"Search Query: {query}")
        print(f"Filter Ticker: {ticker_val or 'ALL'}  |  Method: {args.method}  |  Results Returned: {len(results)}")
        print(f"{'='*80}")
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r.score:.4f}) ---")
            print(f"Citation: {r.citation()}")
            print(f"Section:  {r.section}")
            print(f"Date:     {r.filing_date}")
            print(f"Method:   {r.retrieval_method}")
            print(f"Source:   {r.source_url}")
            preview = r.text[:300].replace('\n', ' ')
            print(f"Text:     {preview}...")
        if not results:
            print("\nNo results found. Have you ingested filings for this query/ticker?")

    elif command == "rag-stats":
        from rag.vectorstore import FinSightVectorStore
        from rag.bm25 import BM25Index
        store = FinSightVectorStore()
        bm25 = BM25Index()
        bm25.load()
        
        all_metadata = store.get_all_metadata()
        tickers = sorted({m['metadata'].get('ticker') for m in all_metadata if m.get('metadata', {}).get('ticker')})
        sections = sorted({m['metadata'].get('section') for m in all_metadata if m.get('metadata', {}).get('section')})
        years = sorted({m['metadata'].get('filing_year') for m in all_metadata if m.get('metadata', {}).get('filing_year')})

        print(f"\n{'='*80}")
        print("FinSight RAG Corpus & Index Statistics")
        print(f"{'='*80}")
        print(f"ChromaDB Vector Chunks : {store.count()}")
        print(f"BM25 Index Documents   : {bm25.document_count()}")
        print(f"Represented Tickers    : {len(tickers)} ({', '.join(tickers) if tickers else 'None'})")
        print(f"Unique Filing Years    : {len(years)} ({', '.join(map(str, years)) if years else 'None'})")
        print(f"Document Sections      : {len(sections)}")
        for sec in sections:
            count = sum(1 for m in all_metadata if m.get('metadata', {}).get('section') == sec)
            print(f"  - {sec:<30}: {count} chunks")
        print(f"{'='*80}\n")

    elif command == "train":
        from ml.train import train_risk_model
        metrics = train_risk_model()
        print("Training complete. Metrics:", metrics)

    elif command == "mcp-server":
        from mcp_server.server import mcp as mcp_app
        mcp_app.run()

    else:
        parser.error(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
