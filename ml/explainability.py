"""Explainability module for FinSight ML risk model.

Uses SHAP (SHapley Additive exPlanations) to interpret XGBoost risk model predictions:
- Global feature importance (mean absolute SHAP value across training observations)
- Top 10 feature breakdown with direction of effect (increases risk vs. decreases risk)
- Local feature explanations for individual company records (e.g. Apple / AAPL)

Disclaimer: SHAP metrics capture feature attribution and empirical correlation within
the model's learned structure; they do not imply direct causal influence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import shap

from analysis.financial_data import MoneyRecord
from data.sources.sec_edgar import SECEdgarSource
from ml.features import FEATURE_COLUMNS, latest_feature_row
from ml.train import DEFAULT_TICKERS, collect_training_data

logger = logging.getLogger(__name__)


def compute_global_shap_importance(
    tickers: list[str] | None = None,
    train_end_year: int = 2017,
) -> dict[str, Any]:
    """Compute global SHAP feature importances trained on historical TRAIN set."""
    from xgboost import XGBClassifier

    tickers = tickers or DEFAULT_TICKERS
    features, labels, years = collect_training_data(tickers)

    X = np.array(features)
    y = np.array(labels)
    years_arr = np.array(years)

    train_mask = (years_arr >= 2007) & (years_arr <= train_end_year)
    X_train, y_train = X[train_mask], y[train_mask]

    # Fit model on train set
    num_neg = sum(y_train == 0)
    num_pos = sum(y_train == 1)
    pos_weight = float(num_neg / max(num_pos, 1))

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.5,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_indices = np.argsort(-mean_abs_shap)

    top_features = []
    for idx in sorted_indices[:10]:
        col_name = FEATURE_COLUMNS[idx]
        val = float(mean_abs_shap[idx])
        # Determine average direction (correlation between feature value and SHAP value)
        feat_vals = X_train[:, idx]
        shap_vals_feat = shap_values[:, idx]
        corr = float(np.corrcoef(feat_vals, shap_vals_feat)[0, 1]) if np.std(feat_vals) > 0 else 0.0

        direction = (
            "Higher values increase risk" if corr > 0.1
            else ("Higher values decrease risk" if corr < -0.1 else "Non-linear / Mixed effect")
        )

        top_features.append({
            "feature": col_name,
            "mean_abs_shap": round(val, 4),
            "correlation_with_shap": round(corr, 4),
            "direction": direction,
        })

    return {
        "model_type": "xgboost",
        "sample_count": len(X_train),
        "top_10_features": top_features,
        "causality_disclaimer": (
            "SHAP values quantify statistical feature attribution in the model's predictions. "
            "They do not assert direct structural or causal economic relationships."
        ),
    }


def explain_company_risk(
    ticker: str = "AAPL",
    train_end_year: int = 2017,
) -> dict[str, Any]:
    """Compute local SHAP explanation for a specific company's latest annual filing."""
    from xgboost import XGBClassifier

    source = SECEdgarSource()
    company = source.fetch_financials(ticker)
    records = company.to_dicts()

    if not records:
        raise ValueError(f"No records found for {ticker}")

    row = latest_feature_row(records)
    feat_vector = np.array([[float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]])

    # Train model on historical set for consistency
    features, labels, years = collect_training_data(DEFAULT_TICKERS)
    X = np.array(features)
    y = np.array(labels)
    years_arr = np.array(years)
    train_mask = (years_arr >= 2007) & (years_arr <= train_end_year)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X[train_mask], y[train_mask])

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(feat_vector)[0]
    predicted_risk_prob = float(model.predict_proba(feat_vector)[0, 1])

    # Rank features by absolute SHAP contribution
    sorted_idx = np.argsort(-np.abs(shap_vals))
    local_drivers = []

    for idx in sorted_idx[:8]:
        col = FEATURE_COLUMNS[idx]
        val = float(feat_vector[0, idx])
        sv = float(shap_vals[idx])
        effect = "increases_risk" if sv > 0 else "decreases_risk"

        local_drivers.append({
            "feature": col,
            "value": round(val, 4),
            "shap_value": round(sv, 4),
            "effect": effect,
        })

    return {
        "ticker": ticker.upper(),
        "company_name": company.company_name,
        "year": int(row["year"]),
        "predicted_risk_probability": round(predicted_risk_prob, 4),
        "local_shap_drivers": local_drivers,
        "causality_disclaimer": (
            "SHAP values quantify statistical feature attribution in the model's predictions. "
            "They do not assert direct structural or causal economic relationships."
        ),
    }


def print_explainability_summary() -> None:
    """Print global and local SHAP explainability analysis."""
    global_info = compute_global_shap_importance()
    aapl_info = explain_company_risk("AAPL")

    print("=================================================================")
    print("           FINSIGHT RISK MODEL SHAP EXPLAINABILITY AUDIT         ")
    print("=================================================================")
    print(f"Model       : {global_info['model_type']}")
    print(f"Training N  : {global_info['sample_count']} historical samples")
    print("-----------------------------------------------------------------")
    print("GLOBAL TOP 10 SHAP FEATURES")
    print("Rank | Feature                   | Mean |SHAP| | Direction / Effect")
    print("-----|---------------------------|-------------|--------------------------------")
    for i, f in enumerate(global_info["top_10_features"], 1):
        print(f" {i:<3} | {f['feature']:<25} | {f['mean_abs_shap']:<11.4f} | {f['direction']}")
    print("-----------------------------------------------------------------")
    print(f"LOCAL SHAP EXPLANATION FOR {aapl_info['company_name']} ({aapl_info['ticker']}) — {aapl_info['year']}")
    print(f"Predicted Risk Probability: {aapl_info['predicted_risk_probability']:.1%}")
    print("Top Local Feature Drivers:")
    for d in aapl_info["local_shap_drivers"]:
        direction_str = "+ (increases risk)" if d["effect"] == "increases_risk" else "- (decreases risk)"
        print(f"  - {d['feature']:<25}: value={d['value']:<8.4f} | SHAP={d['shap_value']:+6.4f} {direction_str}")
    print("-----------------------------------------------------------------")
    print("CAUSALITY DISCLAIMER:")
    print(f"  {global_info['causality_disclaimer']}")
    print("=================================================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_explainability_summary()
