"""Time-series financial forecasting for FinSight.
Uses exponential smoothing (Holt) for 5+ data points, falls back to
simple linear regression for shorter series.
"""

from __future__ import annotations
import logging
import numpy as np
from analysis.financial_data import MoneyRecord

logger = logging.getLogger(__name__)


def forecast_revenue(
    records: list[MoneyRecord],
    periods: int = 1,
) -> list[dict[str, float | int | str]]:
    """Forecast future revenue and net income.
    Uses Holt exponential smoothing when >= 5 data points are available,
    otherwise falls back to simple linear regression.
    """
    revenues = [float(r["revenue"]) for r in records]
    net_incomes = [float(r["net_income"]) for r in records]
    years = [int(r["year"]) for r in records]

    if not revenues:
        return []

    latest_year = years[-1]
    method = "unknown"

    if len(revenues) >= 5:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(
                revenues,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            revenue_forecast = fitted.forecast(periods)
            method = "holt_exponential_smoothing"
        except Exception as exc:
            logger.warning("Exponential smoothing failed: %s — falling back to linear", exc)
            revenue_forecast = _linear_forecast(revenues, periods)
            method = "linear_regression"
    else:
        revenue_forecast = _linear_forecast(revenues, periods)
        method = "linear_regression"

    # Estimate net margin from recent history
    recent = records[-3:] if len(records) >= 3 else records
    avg_margin = np.mean([float(r["net_income"]) / float(r["revenue"])
                          for r in recent if float(r["revenue"]) > 0])

    results: list[dict[str, float | int | str]] = []
    for i, rev in enumerate(revenue_forecast):
        rev = max(0, float(rev))  # No negative revenue
        year = latest_year + i + 1
        net_inc = rev * avg_margin
        growth = (rev / revenues[-1] - 1) if revenues[-1] > 0 else 0.0

        results.append({
            "year": year,
            "revenue": round(rev, 2),
            "net_income": round(net_inc, 2),
            "method": method,
            "assumed_revenue_growth": round(growth, 4),
            "assumed_net_margin": round(float(avg_margin), 4),
        })

    return results


def forecast_next_year(records: list[MoneyRecord]) -> dict[str, float | int | str]:
    """Return a single next-year forecast dict."""
    forecasts = forecast_revenue(records, periods=1)
    if forecasts:
        return forecasts[0]

    # Ultimate fallback
    latest = records[-1] if records else {}
    return {
        "year": int(latest.get("year", 0)) + 1,
        "revenue": 0.0,
        "net_income": 0.0,
        "method": "fallback",
        "assumed_revenue_growth": 0.0,
        "assumed_net_margin": 0.0,
    }


def _linear_forecast(values: list[float], periods: int) -> list[float]:
    """Simple linear regression forecast."""
    n = len(values)
    if n == 0:
        return [0.0] * periods
    if n == 1:
        return [values[0]] * periods

    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)

    # Least squares: y = mx + b
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-10)
    intercept = y_mean - slope * x_mean

    return [float(slope * (n + i) + intercept) for i in range(periods)]
