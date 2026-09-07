from __future__ import annotations

import logging
import statistics
from typing import Any

from analysis.financial_data import MoneyRecord, calculate_metrics

logger = logging.getLogger(__name__)

from data.sources.macro import MacroSource

BASELINE_FEATURE_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "return_on_assets",
    "return_on_equity",
    "debt_to_equity",
    "current_ratio",
    "revenue_growth",
    "free_cash_flow_margin",
    "interest_coverage",
    # Financial Distress & Composite Metrics
    "altman_z_score",
    "asset_turnover",
    "quality_of_earnings",
    "dupont_equity_multiplier",
    # Macroeconomic Regime Signals
    "fed_funds_rate",
    "treasury_10y_yield",
    "yield_curve_slope",
    "cpi_inflation",
    # Trends (1-year deltas)
    "net_margin_trend",
    "current_ratio_trend",
    "debt_to_equity_trend",
    # Rolling 3-year averages
    "net_margin_3yr_avg",
    "current_ratio_3yr_avg",
    "debt_to_equity_3yr_avg",
    # Momentum (2-year rate of change)
    "revenue_momentum",
    "margin_momentum",
    # Volatility
    "revenue_volatility",
]

SECTOR_FEATURE_COLUMNS = [
    "net_margin_sector_delta",
    "gross_margin_sector_delta",
    "operating_margin_sector_delta",
    "roa_sector_delta",
    "current_ratio_sector_delta",
    "debt_to_equity_sector_delta",
]

# Total 33 features (27 baseline + 6 sector-relative)
FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + SECTOR_FEATURE_COLUMNS

_macro_source = MacroSource()


def build_feature_rows(records: list[MoneyRecord]) -> list[MoneyRecord]:
    """Build enriched feature rows with ratios, trends, rolling averages, macro indicators, and momentum."""
    rows: list[MoneyRecord] = []
    previous: MoneyRecord | None = None

    for row in calculate_metrics(records):
        revenue = float(row["revenue"])
        total_assets = float(row.get("total_assets", 0))
        total_liab = float(row.get("total_liabilities", 0))
        total_equity = float(row.get("total_equity", 0))
        operating_income = float(row.get("operating_income", 0))
        net_income = float(row.get("net_income", 0))
        current_assets = float(row.get("current_assets", 0))
        current_liab = float(row.get("current_liabilities", 0))
        interest_expense = float(row.get("interest_expense", 0))
        free_cash_flow = float(row.get("free_cash_flow", 0))
        op_cash_flow = float(row.get("operating_cash_flow", 0))
        yr = int(row.get("year", 2024))

        # Composite domain metrics
        x1 = _safe_divide(current_assets - current_liab, total_assets)
        x3 = _safe_divide(operating_income, total_assets)
        x4 = _safe_divide(total_equity, total_liab)
        x5 = _safe_divide(revenue, total_assets)
        altman_z = 1.2 * x1 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        # Macroeconomic regime indicators for the row's fiscal year
        macro_dict = _macro_source.get_as_dict(yr)

        feature_row: MoneyRecord = {
            **row,
            "altman_z_score": altman_z,
            "asset_turnover": x5,
            "quality_of_earnings": _safe_divide(op_cash_flow, abs(net_income) if net_income != 0 else 1.0),
            "dupont_equity_multiplier": _safe_divide(total_assets, total_equity),
            **macro_dict,
            "free_cash_flow_margin": _safe_divide(free_cash_flow, revenue),
            "interest_coverage": _safe_divide(operating_income, interest_expense),
            "revenue_growth": 0.0 if row["revenue_growth"] is None else float(row["revenue_growth"]),
            **{col: float(row.get(col, 0.0)) for col in SECTOR_FEATURE_COLUMNS},
        }

        # 1-year trends
        feature_row["net_margin_trend"] = _change_from_previous(feature_row, previous, "net_margin")
        feature_row["current_ratio_trend"] = _change_from_previous(feature_row, previous, "current_ratio")
        feature_row["debt_to_equity_trend"] = _change_from_previous(feature_row, previous, "debt_to_equity")

        rows.append(feature_row)
        previous = feature_row

    # Second pass: rolling windows and momentum (need full history)
    for i, row in enumerate(rows):
        # 3-year rolling averages
        window = rows[max(0, i - 2): i + 1]
        row["net_margin_3yr_avg"] = _avg([float(r["net_margin"]) for r in window])
        row["current_ratio_3yr_avg"] = _avg([float(r["current_ratio"]) for r in window])
        row["debt_to_equity_3yr_avg"] = _avg([float(r["debt_to_equity"]) for r in window])

        # 2-year momentum (current vs 2 years ago)
        if i >= 2:
            row["revenue_momentum"] = float(row["revenue_growth"]) - float(rows[i - 2].get("revenue_growth", 0))
            row["margin_momentum"] = float(row["net_margin"]) - float(rows[i - 2]["net_margin"])
        else:
            row["revenue_momentum"] = 0.0
            row["margin_momentum"] = 0.0

        # Revenue growth volatility (std dev over available history)
        growth_values = [float(r["revenue_growth"]) for r in rows[: i + 1] if r.get("revenue_growth") not in (None, 0.0)]
        row["revenue_volatility"] = statistics.stdev(growth_values) if len(growth_values) >= 2 else 0.0

    return rows


def latest_feature_row(records: list[MoneyRecord]) -> MoneyRecord:
    """Return the feature row for the most recent year."""
    return build_feature_rows(records)[-1]


def extract_feature_vector(row: MoneyRecord) -> list[float]:
    """Extract an ordered feature vector from a feature row for ML model input."""
    return [float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _change_from_previous(row: MoneyRecord, previous: MoneyRecord | None, key: str) -> float:
    if previous is None:
        return 0.0
    return float(row[key]) - float(previous[key])


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
