"""Tests for the LangGraph workflow."""

from __future__ import annotations

from analysis.financial_data import load_financial_data
from lang_graph.workflow import FinSightState, run_financial_workflow


def _has_langgraph() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


class TestWorkflow:
    def test_financial_workflow_combines_report_risk_and_rag(self, sample_records) -> None:
        output = run_financial_workflow(sample_records, "What is debt risk?", ticker="TEST")

        assert "FinSight Financial Report" in output
        assert "FinSight ML Risk Report" in output

    def test_workflow_without_question(self, sample_records) -> None:
        output = run_financial_workflow(sample_records)
        assert "FinSight Financial Report" in output
        assert "FinSight ML Risk Report" in output

    def test_state_contract_has_expected_keys(self) -> None:
        annotations = FinSightState.__annotations__
        assert "ticker" in annotations
        assert "records" in annotations
        assert "financial_report" in annotations
        assert "risk_report" in annotations
        assert "rag_answer" in annotations
        assert "final_answer" in annotations
        assert "messages" in annotations
