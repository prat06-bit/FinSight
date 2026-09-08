from __future__ import annotations

import logging
from dataclasses import dataclass, field
import numpy as np
from analysis.financial_data import MoneyRecord
from ml.features import FEATURE_COLUMNS, extract_feature_vector, latest_feature_row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskPrediction:
    year: int
    score: int
    label: str
    drivers: list[str]
    confidence: float
    model_type: str = "rule_based"
    feature_importances: dict[str, float] = field(default_factory=dict)


def predict_financial_risk(records: list[MoneyRecord]) -> RiskPrediction:
    """Predict financial risk using trained ML model or rule-based fallback."""
    row = latest_feature_row(records)

    # Try trained ML model first
    try:
        return _predict_with_model(row, records)
    except Exception as exc:
        logger.debug("ML model not available (%s), using rule-based scoring", exc)
        return _predict_with_rules(row, records)


def _predict_with_model(row: MoneyRecord, records: list[MoneyRecord]) -> RiskPrediction:
    """Use trained XGBoost model for prediction."""
    import joblib
    from config import RISK_MODEL_PATH

    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found")

    bundle = joblib.load(RISK_MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle.get("feature_columns", FEATURE_COLUMNS)

    feature_vector = [float(row.get(col, 0.0)) for col in feature_cols]
    X = np.array([feature_vector])

    # Predict probability
    proba = model.predict_proba(X)[0]
    risk_proba = float(proba[1]) if len(proba) > 1 else float(proba[0])

    # Map probability to 0-100 score (inverted: high prob of deterioration = low score)
    score = max(0, min(100, int(100 * (1 - risk_proba))))

    # Extract feature importances
    importances = {}
    if hasattr(model, "feature_importances_"):
        for col, imp in zip(feature_cols, model.feature_importances_):
            if imp > 0.01:
                importances[col] = round(float(imp), 4)

    # Generate driver descriptions from top features
    drivers = _drivers_from_importances(importances, row)

    return RiskPrediction(
        year=int(row["year"]),
        score=score,
        label=_label_for_score(score),
        drivers=drivers,
        confidence=min(0.95, 0.70 + 0.03 * min(len(records), 8)),
        model_type="xgboost",
        feature_importances=importances,
    )


def _predict_with_rules(row: MoneyRecord, records: list[MoneyRecord]) -> RiskPrediction:
    """Rule-based risk scoring (fallback)."""
    score = 50
    drivers: list[str] = []

    score += _score_signal(float(row["net_margin"]), 0.12, 0.08, "net margin", drivers)
    score += _score_signal(float(row["current_ratio"]), 1.6, 1.1, "liquidity", drivers)
    score += _score_inverse(float(row["debt_to_equity"]), 0.6, 1.2, "leverage", drivers)
    score += _score_signal(float(row["revenue_growth"]), 0.10, 0.00, "revenue growth", drivers)
    score += _score_signal(float(row["free_cash_flow_margin"]), 0.08, 0.03, "free cash flow", drivers)
    score += _score_signal(float(row["interest_coverage"]), 6.0, 3.0, "interest coverage", drivers)
    score += _score_trend(float(row["net_margin_trend"]), -0.02, "margin trend", drivers)
    score += _score_trend(float(row["current_ratio_trend"]), -0.10, "liquidity trend", drivers)
    score += _score_inverse_trend(float(row["debt_to_equity_trend"]), 0.10, "leverage trend", drivers)

    bounded_score = max(0, min(100, score))
    return RiskPrediction(
        year=int(row["year"]),
        score=bounded_score,
        label=_label_for_score(bounded_score),
        drivers=drivers,
        confidence=min(0.95, 0.55 + 0.05 * min(len(records), 8)),
        model_type="rule_based",
    )


def format_risk_report(records: list[MoneyRecord]) -> str:
    """Render a human-readable risk report."""
    from ml.forecast_model import forecast_next_year

    prediction = predict_financial_risk(records)
    forecast = forecast_next_year(records)
    drivers = "\n".join(f"- {driver}" for driver in prediction.drivers)

    ticker = str(records[-1].get("ticker", "")) if records else ""
    ticker_str = f" ({ticker})" if ticker else ""

    lines = [
        f"FinSight ML Risk Report{ticker_str}",
        f"Model: {prediction.model_type}",
        f"Scored year: {prediction.year}",
        f"Risk score: {prediction.score}/100",
        f"Risk label: {prediction.label}",
        f"Model confidence: {prediction.confidence * 100:.0f}%",
        "",
        "Key drivers",
        drivers,
        "",
        "Next-year forecast",
        f"- Year: {forecast['year']}",
        f"- Revenue: ${forecast['revenue']:,.0f}",
        f"- Net income: ${forecast['net_income']:,.0f}",
        f"- Method: {forecast.get('method', 'n/a')}",
        f"- Assumed revenue growth: {float(forecast['assumed_revenue_growth']) * 100:.1f}%",
        f"- Assumed net margin: {float(forecast['assumed_net_margin']) * 100:.1f}%",
    ]

    if prediction.feature_importances:
        lines.append("")
        lines.append("Top feature importances")
        sorted_imp = sorted(prediction.feature_importances.items(), key=lambda x: -x[1])
        for name, imp in sorted_imp[:5]:
            lines.append(f"- {name}: {imp:.4f}")

    return "\n".join(lines)


def _drivers_from_importances(importances: dict[str, float], row: MoneyRecord) -> list[str]:
    """Generate human-readable driver descriptions from feature importances."""
    drivers = []
    sorted_features = sorted(importances.items(), key=lambda x: -x[1])

    for feature, importance in sorted_features[:6]:
        value = float(row.get(feature, 0))
        if "margin" in feature:
            status = "Strong" if value >= 0.10 else ("Acceptable" if value >= 0.05 else "Weak")
        elif "ratio" in feature:
            status = "Strong" if value >= 1.5 else ("Acceptable" if value >= 1.0 else "Weak")
        elif "debt" in feature or "leverage" in feature:
            status = "Low" if value <= 0.6 else ("Manageable" if value <= 1.2 else "High")
        elif "trend" in feature:
            status = "Improving" if value > 0 else ("Stable" if value == 0 else "Deteriorating")
        else:
            status = "Positive" if value > 0 else "Negative"

        drivers.append(f"{status} {feature.replace('_', ' ')} ({value:.3f})")

    return drivers if drivers else ["Insufficient data for detailed drivers"]


# Rule-based scoring helpers

def _score_signal(value: float, strong: float, weak: float, name: str, drivers: list[str]) -> int:
    if value >= strong:
        drivers.append(f"Strong {name}")
        return 10
    if value >= weak:
        drivers.append(f"Acceptable {name}")
        return 0
    drivers.append(f"Weak {name}")
    return -10


def _score_inverse(value: float, strong: float, weak: float, name: str, drivers: list[str]) -> int:
    if value <= strong:
        drivers.append(f"Low {name}")
        return 10
    if value <= weak:
        drivers.append(f"Manageable {name}")
        return 0
    drivers.append(f"High {name}")
    return -10


def _score_trend(value: float, weak: float, name: str, drivers: list[str]) -> int:
    if value <= weak:
        drivers.append(f"Deteriorating {name}")
        return -5
    if value > 0:
        drivers.append(f"Improving {name}")
        return 5
    drivers.append(f"Stable {name}")
    return 0


def _score_inverse_trend(value: float, strong: float, name: str, drivers: list[str]) -> int:
    if value >= strong:
        drivers.append(f"Rising {name}")
        return -5
    if value < 0:
        drivers.append(f"Improving {name}")
        return 5
    drivers.append(f"Stable {name}")
    return 0


def _label_for_score(score: int) -> str:
    if score >= 75:
        return "Low risk"
    if score >= 45:
        return "Moderate risk"
    return "High risk"
