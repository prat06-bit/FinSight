"""Unit tests for Step 3: Financial Forecasting with Walk-Forward Validation."""

from __future__ import annotations
import numpy as np
import pytest
from ml.forecasting import (
    build_time_series_lags,
    calculate_regression_metrics,
    calculate_smape,
    generate_ticker_forecast,
    run_walk_forward_backtest,
)


class TestTimeSeriesFeatures:
    """Verify chronological lag and rolling feature generation with strict leakage prevention."""
    def test_chronological_ordering_and_lags(self) -> None:
        records = [
            {"year": 2018, "revenue": 100.0, "net_income": 10.0, "operating_income": 15.0, "cost_of_goods_sold": 50.0},
            {"year": 2019, "revenue": 120.0, "net_income": 12.0, "operating_income": 18.0, "cost_of_goods_sold": 60.0},
            {"year": 2020, "revenue": 150.0, "net_income": 15.0, "operating_income": 22.0, "cost_of_goods_sold": 75.0},
            {"year": 2021, "revenue": 200.0, "net_income": 20.0, "operating_income": 30.0, "cost_of_goods_sold": 100.0},
        ]
        X, y, years = build_time_series_lags(records, include_macro=False)
        assert len(X) == 1
        assert years[0] == 2021
        assert y[0] == 200.0  # target 2021 revenue
        assert X[0][0] == 150.0  # lag1 (2020)
        assert X[0][1] == 120.0  # lag2 (2019)
        assert X[0][2] == 100.0  # lag3 (2018)
        assert X[0][3] == pytest.approx((100 + 120 + 150) / 3.0)  # rolling3

    def test_no_future_observations_in_features(self) -> None:
        records = [
            {"year": 2018, "revenue": 100.0, "net_income": 10.0},
            {"year": 2019, "revenue": 120.0, "net_income": 12.0},
            {"year": 2020, "revenue": 150.0, "net_income": 15.0},
            {"year": 2021, "revenue": 999.0, "net_income": 99.0},  # Future observation
        ]
        X, y, years = build_time_series_lags(records, include_macro=False)
        assert 999.0 not in X[0]
        assert y[0] == 999.0


class TestRegressionMetrics:
    """Verify MAE, RMSE, and sMAPE metric calculations."""

    def test_smape_zero_handling(self) -> None:
        y_true = [100.0, 0.0, 50.0]
        y_pred = [100.0, 0.0, 25.0]
        sm = calculate_smape(np.array(y_true), np.array(y_pred))
        assert sm >= 0.0
        assert not np.isnan(sm)
    def test_metric_calculations(self) -> None:
        y_true = [100.0, 200.0, 300.0]
        y_pred = [110.0, 190.0, 310.0]
        metrics = calculate_regression_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(10.0)
        assert metrics["rmse"] == pytest.approx(10.0)
        assert 0.0 < metrics["smape"] < 10.0


class TestWalkForwardValidation:
    """Verify out-of-time walk-forward backtesting engine."""
    def test_walk_forward_backtest_execution(self) -> None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
        backtest = run_walk_forward_backtest(tickers=tickers, start_backtest_year=2021, end_backtest_year=2025)
        assert backtest["target_field"] == "revenue"
        assert len(backtest["models_comparison"]) == 4
        assert backtest["best_model"] in [m["model"] for m in backtest["models_comparison"]]

    def test_unequal_company_history_handling(self) -> None:
        backtest = run_walk_forward_backtest(tickers=["GOOGL", "AVGO"], start_backtest_year=2021, end_backtest_year=2025)
        assert backtest["sample_evaluations_count"] > 0


class TestTickerForecastGeneration:
    """Verify point forecast, lower bound, and upper bound generation for AAPL."""
    def test_aapl_2026_forecast_structure(self) -> None:
        res = generate_ticker_forecast("AAPL")
        assert res.ticker == "AAPL"
        assert res.historical_latest_year == 2025
        assert res.forecast_year == 2026
        assert res.point_forecast > 0.0
        assert res.lower_bound_95 <= res.point_forecast <= res.upper_bound_95
    def test_scope_separation_aapl_vs_pooled(self) -> None:
        res = generate_ticker_forecast("AAPL")
        assert "models_comparison" in res.ticker_backtest
        assert "models_comparison" in res.pooled_backtest
        aapl_holt = [m for m in res.ticker_backtest["models_comparison"] if m["model"] == "Holt Exponential Smoothing"][0]
        pooled_holt = [m for m in res.pooled_backtest["models_comparison"] if m["model"] == "Holt Exponential Smoothing"][0]

        assert aapl_holt["smape"] > 0.0
        assert pooled_holt["smape"] > 0.0
