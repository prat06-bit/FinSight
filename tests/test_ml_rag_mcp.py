"""Tests for ML features, risk model, and forecasting."""

from __future__ import annotations

from analysis.financial_data import MoneyRecord
from ml.features import FEATURE_COLUMNS, build_feature_rows, extract_feature_vector, latest_feature_row
from ml.forecast_model import forecast_next_year, forecast_revenue
from ml.risk_model import predict_financial_risk, format_risk_report, RiskPrediction


class TestFeatures:
    def test_feature_columns_complete(self) -> None:
        assert len(FEATURE_COLUMNS) >= 15
        assert "net_margin" in FEATURE_COLUMNS
        assert "net_margin_3yr_avg" in FEATURE_COLUMNS
        assert "revenue_momentum" in FEATURE_COLUMNS
        assert "revenue_volatility" in FEATURE_COLUMNS

    def test_build_feature_rows_adds_all_features(self, sample_records: list[MoneyRecord]) -> None:
        rows = build_feature_rows(sample_records)
        latest = rows[-1]

        for col in FEATURE_COLUMNS:
            assert col in latest, f"Missing feature: {col}"

        assert latest["interest_coverage"] > 0
        assert latest["free_cash_flow_margin"] > 0

    def test_extract_feature_vector_length(self, sample_records: list[MoneyRecord]) -> None:
        row = latest_feature_row(sample_records)
        vector = extract_feature_vector(row)
        assert len(vector) == len(FEATURE_COLUMNS)
        assert all(isinstance(v, float) for v in vector)

    def test_rolling_averages_with_extended_data(self, extended_records: list[MoneyRecord]) -> None:
        rows = build_feature_rows(extended_records)
        latest = rows[-1]
        assert latest["net_margin_3yr_avg"] > 0
        assert latest["current_ratio_3yr_avg"] > 0


class TestRiskModel:
    def test_returns_bounded_score(self, sample_records: list[MoneyRecord]) -> None:
        prediction = predict_financial_risk(sample_records)

        assert isinstance(prediction, RiskPrediction)
        assert 0 <= prediction.score <= 100
        assert prediction.label in ("Low risk", "Moderate risk", "High risk")
        assert prediction.confidence > 0.5
        assert prediction.year == 2024
        assert prediction.model_type in ("rule_based", "xgboost")

    def test_drivers_are_populated(self, sample_records: list[MoneyRecord]) -> None:
        prediction = predict_financial_risk(sample_records)
        assert len(prediction.drivers) > 0

    def test_format_risk_report_contains_sections(self, sample_records: list[MoneyRecord]) -> None:
        report = format_risk_report(sample_records)
        assert "FinSight ML Risk Report" in report
        assert "Risk score:" in report
        assert "Next-year forecast" in report
        assert "Model:" in report


class TestForecastModel:
    def test_forecast_next_year_basic(self, sample_records: list[MoneyRecord]) -> None:
        forecast = forecast_next_year(sample_records)
        assert forecast["year"] == 2025
        assert forecast["revenue"] > 0
        assert "method" in forecast

    def test_forecast_with_extended_data(self, extended_records: list[MoneyRecord]) -> None:
        forecasts = forecast_revenue(extended_records, periods=2)
        assert len(forecasts) == 2
        assert forecasts[0]["year"] == 2025
        assert forecasts[1]["year"] == 2026
        assert forecasts[0]["revenue"] > 0

    def test_forecast_single_record(self, sample_records: list[MoneyRecord]) -> None:
        forecast = forecast_next_year([sample_records[0]])
        assert forecast["year"] == 2024
        assert forecast["revenue"] > 0
