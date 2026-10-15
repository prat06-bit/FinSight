"""LangGraph workflow with NVIDIA LLM agent for FinSight."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from analysis.financial_data import MoneyRecord, format_report, load_financial_data, load_from_ticker
from config import NVIDIA_API_KEY
from ml.forecast_model import forecast_next_year
from ml.risk_model import format_risk_report, predict_financial_risk
from rag.retriever import answer_question

logger = logging.getLogger(__name__)


class LangGraphNotInstalledError(RuntimeError):
    """Raised when the optional LangGraph dependency is not installed."""


class FinSightState(TypedDict, total=False):
    ticker: str
    data_path: str
    question: str | None
    records: list[MoneyRecord]
    financial_report: str
    risk_report: str
    risk_label: str
    risk_score: int
    forecast: dict
    rag_answer: str
    final_answer: str
    messages: list  # For agent message history


# ---------------------------------------------------------------------------
# Non-graph workflow (backward compat)
# ---------------------------------------------------------------------------


def run_financial_workflow(
    records: list[MoneyRecord],
    question: str | None = None,
    ticker: str = "",
) -> str:
    """Simple non-graph workflow combining report + risk + optional RAG."""
    sections = [format_report(records), format_risk_report(records)]
    if question:
        sections.append(answer_question(question, records, ticker=ticker))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LangGraph — structured pipeline
# ---------------------------------------------------------------------------


def build_finsight_graph() -> object:
    """Build the structured (non-agent) LangGraph state machine."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise LangGraphNotInstalledError(
            "LangGraph is not installed. Run: pip install -r requirements.txt"
        ) from exc

    graph = StateGraph(FinSightState)
    graph.add_node("load_data", _load_data_node)
    graph.add_node("analyze_financials", _analyze_financials_node)
    graph.add_node("score_risk", _score_risk_node)
    graph.add_node("retrieve_answer", _retrieve_answer_node)
    graph.add_node("compose_final", _compose_final_node)

    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "analyze_financials")
    graph.add_edge("analyze_financials", "score_risk")
    graph.add_conditional_edges(
        "score_risk",
        _route_after_risk,
        {"retrieve_answer": "retrieve_answer", "compose_final": "compose_final"},
    )
    graph.add_edge("retrieve_answer", "compose_final")
    graph.add_edge("compose_final", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# LangGraph — agentic (NVIDIA LLM + tools)
# ---------------------------------------------------------------------------


def build_agent_graph() -> object:
    """Build an agentic LangGraph that uses NVIDIA LLM with tool-calling."""
    if not NVIDIA_API_KEY:
        logger.warning("No NVIDIA_API_KEY — falling back to structured graph")
        return build_finsight_graph()

    try:
        from langgraph.prebuilt import create_react_agent

        from lang_graph.agents import AGENT_TOOLS, get_llm
    except ImportError as exc:
        raise LangGraphNotInstalledError(str(exc)) from exc

    llm = get_llm()
    system_message = (
        "You are FinSight, an expert financial analyst assistant. "
        "Use your tools to gather SEC filing data, financial reports, risk scores, "
        "and forecasts. Then synthesize a comprehensive, well-structured analysis. "
        "Always cite specific numbers and data points. "
        "If asked about a specific company, use the ticker to fetch real data."
    )

    return create_react_agent(llm, AGENT_TOOLS, prompt=system_message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_langgraph_workflow(
    data_path_or_ticker: str | Path,
    question: str | None = None,
) -> str:
    """Run the LangGraph workflow.  Accepts a CSV path OR a ticker symbol."""
    data_str = str(data_path_or_ticker)

    # Detect ticker vs path
    is_ticker = (
        len(data_str) <= 5
        and "/" not in data_str
        and "\\" not in data_str
        and "." not in data_str
    )

    # Agent graph for ticker-based queries when NVIDIA key is available
    if is_ticker and NVIDIA_API_KEY:
        try:
            graph = build_agent_graph()
            prompt = f"Analyze the financial health of {data_str.upper()}."
            if question:
                prompt += f" Specifically: {question}"
            result = graph.invoke({"messages": [("user", prompt)]})
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    return str(msg.content)
            return str(result)
        except Exception as exc:
            logger.warning("Agent graph failed: %s — falling back to structured", exc)

    # Structured graph fallback
    if is_ticker:
        records = load_from_ticker(data_str.upper())
        ticker = data_str.upper()
    else:
        records = load_financial_data(data_str)
        ticker = ""

    graph = build_finsight_graph()
    state: FinSightState = {
        "data_path": data_str,
        "ticker": ticker,
        "question": question,
        "records": records,
    }
    result = graph.invoke(state)
    return str(result.get("final_answer", ""))


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _load_data_node(state: FinSightState) -> FinSightState:
    if state.get("records"):
        return state
    ticker = state.get("ticker", "")
    if ticker:
        records = load_from_ticker(ticker)
    else:
        records = load_financial_data(state["data_path"])
    return {**state, "records": records}


def _analyze_financials_node(state: FinSightState) -> FinSightState:
    return {**state, "financial_report": format_report(state["records"])}


def _score_risk_node(state: FinSightState) -> FinSightState:
    records = state["records"]
    prediction = predict_financial_risk(records)
    fc = forecast_next_year(records)
    return {
        **state,
        "risk_report": format_risk_report(records),
        "risk_label": prediction.label,
        "risk_score": prediction.score,
        "forecast": fc,
    }


def _retrieve_answer_node(state: FinSightState) -> FinSightState:
    question = state.get("question")
    if not question:
        return state
    ticker = state.get("ticker", "")
    return {**state, "rag_answer": answer_question(question, state["records"], ticker=ticker)}


def _compose_final_node(state: FinSightState) -> FinSightState:
    sections = [state["financial_report"], state["risk_report"]]
    if state.get("rag_answer"):
        sections.append(state["rag_answer"])
    return {**state, "final_answer": "\n\n".join(sections)}


def _route_after_risk(state: FinSightState) -> str:
    if state.get("question"):
        return "retrieve_answer"
    return "compose_final"
