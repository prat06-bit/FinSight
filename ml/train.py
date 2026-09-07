"""Training pipeline for FinSight ML risk model.

Fetches historical financials for multiple companies from SEC EDGAR,
engineers features, and trains an XGBoost classifier to predict
financial health deterioration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from config import RISK_MODEL_PATH

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ORCL", "CSCO", "ADBE", "AVGO", "QCOM", "TXN", "IBM", "NOW", "AMAT", "MU",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "DHR", "BMY", "AMGN",
    "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
]


def collect_training_data(
    tickers: list[str],
    feature_columns: list[str] | None = None,
) -> tuple[list[list[float]], list[int], list[int]]:
    """Fetch financials for *tickers*, compute features, and create labels.

    Supports sector-relative feature computation using point-in-time,
    leave-one-out peer medians.
    """
    from data.sources.sec_edgar import SECEdgarSource
    from ml.features import FEATURE_COLUMNS, build_feature_rows
    from ml.sector import compute_sector_relative_features, sic_to_sector

    feature_cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    source = SECEdgarSource()

    raw_records_by_ticker: dict[str, list[dict]] = {}
    all_raw_records: list[dict] = []

    for ticker in tickers:
        try:
            company = source.fetch_financials(ticker)
            recs = company.to_dicts()
            if len(recs) < 3:
                continue
            for r in recs:
                r["sic_code"] = company.sic_code
                r["sector"] = company.sector or sic_to_sector(company.sic_code)
                all_raw_records.append(r)
            raw_records_by_ticker[ticker] = recs
        except Exception as exc:
            logger.warning("Error processing %s: %s", ticker, exc)
            continue

    # Enrich all records with leave-one-out sector deltas
    enriched_records = compute_sector_relative_features(all_raw_records)

    # Group enriched records back by ticker
    enriched_by_ticker: dict[str, list[dict]] = {}
    for r in enriched_records:
        t = str(r["ticker"])
        enriched_by_ticker.setdefault(t, []).append(r)

    all_features: list[list[float]] = []
    all_labels: list[int] = []
    all_years: list[int] = []

    for ticker, recs in enriched_by_ticker.items():
        recs_sorted = sorted(recs, key=lambda x: int(x["year"]))
        feature_rows = build_feature_rows(recs_sorted)

        for i in range(len(feature_rows) - 1):
            current = feature_rows[i]
            next_year = feature_rows[i + 1]
            yr = int(current["year"])

            ni_curr = float(current["net_income"])
            ni_next = float(next_year["net_income"])
            next_cr = float(next_year["current_ratio"])

            ni_drop = (ni_curr - ni_next) / max(abs(ni_curr), 1.0)

            deteriorated = int(
                ni_next < 0
                or ni_drop > 0.15
                or next_cr < 1.0
            )

            feature_vector = [float(current.get(col, 0.0)) for col in feature_cols]

            vec_arr = np.array(feature_vector)
            if np.isnan(vec_arr).any() or np.isinf(vec_arr).any():
                continue

            all_features.append(feature_vector)
            all_labels.append(deteriorated)
            all_years.append(yr)

    logger.info(
        "Collected %d samples from %d tickers (%.0f%% positive, %d features)",
        len(all_labels),
        len(enriched_by_ticker),
        100 * sum(all_labels) / max(len(all_labels), 1),
        len(feature_cols),
    )
    return all_features, all_labels, all_years


def train_risk_model(
    tickers: list[str] | None = None,
    split_strategy: str = "time_based",
    cutoff_year: int = 2020,
) -> dict[str, float]:
    """Train an XGBoost classifier on multi-company data.

    Parameters
    ----------
    tickers:
        List of stock tickers (defaults to 40 non-financial S&P 500 companies).
    split_strategy:
        ``"time_based"`` (default) to train on years <= cutoff_year and test on > cutoff_year,
        or ``"random"`` for stratified random split.
    cutoff_year:
        Split cutoff year for time_based strategy (default 2020).

    Returns a dict of evaluation metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from ml.features import FEATURE_COLUMNS

    tickers = tickers or DEFAULT_TICKERS
    logger.info("Training risk model on %d tickers (split: %s)...", len(tickers), split_strategy)

    features, labels, years = collect_training_data(tickers)

    if len(features) < 20:
        logger.error("Not enough training data (%d samples). Need at least 20.", len(features))
        return {"error": 1.0, "samples": float(len(features))}

    X = np.array(features)
    y = np.array(labels)
    years_arr = np.array(years)

    if split_strategy == "time_based":
        train_mask = years_arr <= cutoff_year
        test_mask = years_arr > cutoff_year

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_test) < 5:  # Fallback if cutoff year leaves too few test samples
            logger.warning("Time-based split created too few test samples (%d), falling back to random", len(X_test))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y,
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

    # Train regularized XGBoost classifier
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_samples": float(len(X_train)),
        "test_samples": float(len(X_test)),
        "positive_rate": float(sum(y) / len(y)),
    }

    # Save model + feature column names
    RISK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, RISK_MODEL_PATH)
    logger.info("Model saved to %s", RISK_MODEL_PATH)
    logger.info("Metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_risk_model()
    print("Training complete. Metrics:", result)
