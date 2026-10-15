"""Tests for the analysis layer — metrics, summaries, and reports."""

from __future__ import annotations
from analysis.financial_data import (
    MoneyRecord,
    calculate_metrics,
    format_report,
    load_financial_data,
    summarize_financials,
)

class TestLoadFinancialData:
    def test_loads_and_sorts_by_year(self) -> None:
        records = load_financial_data("data/financial_data.csv")
        years = [record["year"] for record in records]
        assert years == sorted(years)
        assert years[0] == 2019
    def test_skips_malformed_rows(self) -> None:
        """The CSV has a corrupted last line — it should be skipped."""
        records = load_financial_data("data/financial_data.csv")
        for r in records:
            assert isinstance(r["year"], int)
            assert isinstance(r["revenue"], (int, float))


class TestCalculateMetrics:
    def test_adds_ratios_and_growth(self, sample_records: list[MoneyRecord]) -> None:
        metrics = calculate_metrics(sample_records)
        assert metrics[0]["revenue_growth"] is None
        assert abs(metrics[1]["revenue_growth"] - 0.10) < 0.01  # 10% growth
        assert abs(metrics[1]["gross_margin"] - (450_000 / 1_100_000)) < 0.001
        assert metrics[1]["current_ratio"] == 660_000 / 330_000

    def test_handles_single_record(self, sample_records: list[MoneyRecord]) -> None:
        metrics = calculate_metrics([sample_records[0]])
        assert len(metrics) == 1
        assert metrics[0]["revenue_growth"] is None


class TestSummarizeFinancials:
    def test_reports_latest_year(self, sample_records: list[MoneyRecord]) -> None:
        summary = summarize_financials(sample_records)
        assert summary["latest_year"] == 2024
        assert summary["revenue"] == 1_100_000.0
        assert "health_signal" in summary
    def test_includes_ticker(self, sample_records: list[MoneyRecord]) -> None:
        summary = summarize_financials(sample_records)
        assert summary["ticker"] == "TEST"


class TestFormatReport:
    def test_contains_key_sections(self, sample_records: list[MoneyRecord]) -> None:
        report = format_report(sample_records)
        assert "FinSight Financial Report" in report
        assert "Summary" in report
        assert "Yearly trend" in report
        assert "2024" in report
