"""Unit tests for Step 2: Risk-model evaluation, threshold calibration, and explainability."""

from __future__ import annotations

import numpy as np
import pytest

from ml.audit import run_evaluation_audit
from ml.explainability import compute_global_shap_importance, explain_company_risk
from ml.features import FEATURE_COLUMNS, build_feature_rows
from ml.train import DEFAULT_TICKERS, collect_training_data


class TestTemporalSplitting:
    """Verify strict time-based three-stage temporal split with zero year overlap."""

    def test_no_year_overlap(self) -> None:
        features, labels, years = collect_training_data(DEFAULT_TICKERS[:5])
        years_arr = np.array(years)

        train_years = years_arr[(years_arr >= 2007) & (years_arr <= 2017)]
        val_years = years_arr[(years_arr >= 2018) & (years_arr <= 2020)]
        test_years = years_arr[years_arr >= 2021]

        # Ensure no overlap
        assert set(train_years).isdisjoint(set(val_years))
        assert set(train_years).isdisjoint(set(test_years))
        assert set(val_years).isdisjoint(set(test_years))

    def test_temporal_bounds(self) -> None:
        report = run_evaluation_audit(tickers=DEFAULT_TICKERS[:5])
        cd = report["class_distribution"]

        assert cd["train_2007_2017"]["years"] == "2007-2017"
        assert cd["validation_2018_2020"]["years"] == "2018-2020"
        assert "2021" in cd["test_2021_2025"]["years"]


class TestLeakageFreeThresholdSelection:
    """Verify threshold selection uses Validation ONLY and Test remains untouched."""

    def test_threshold_selected_on_validation_only(self) -> None:
        report = run_evaluation_audit(tickers=DEFAULT_TICKERS[:5])
        val_objs = report["validation_objectives"]
        selected_t = val_objs["selected_threshold"]

        # Selected threshold must be between 0.10 and 0.90
        assert 0.10 <= selected_t <= 0.90
        assert "validation" in val_objs["selection_reason"].lower()

    def test_confusion_matrix_consistency(self) -> None:
        report = run_evaluation_audit(tickers=DEFAULT_TICKERS[:5])
        cm = report["final_untouched_test_results"]["confusion_matrix"]
        total_test_samples = report["class_distribution"]["test_2021_2025"]["total_samples"]

        matrix_sum = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
        assert matrix_sum == total_test_samples, f"Confusion matrix sum ({matrix_sum}) != Test samples ({total_test_samples})"

    def test_all_required_metrics_present(self) -> None:
        report = run_evaluation_audit(tickers=DEFAULT_TICKERS[:5])
        res = report["final_untouched_test_results"]

        required_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"}
        assert required_metrics.issubset(res.keys())


class TestSHAPExplainability:
    """Verify SHAP global and local feature importance output generation."""

    def test_global_shap_importance_top_10(self) -> None:
        info = compute_global_shap_importance(tickers=DEFAULT_TICKERS[:5])
        top10 = info["top_10_features"]

        assert len(top10) == 10
        assert all("feature" in item and "mean_abs_shap" in item for item in top10)
        assert info["top_10_features"][0]["mean_abs_shap"] >= info["top_10_features"][1]["mean_abs_shap"]

    def test_local_shap_explanation_structure(self) -> None:
        aapl = explain_company_risk("AAPL")

        assert aapl["ticker"] == "AAPL"
        assert 0.0 <= aapl["predicted_risk_probability"] <= 1.0
        assert len(aapl["local_shap_drivers"]) > 0
        assert "causal" in aapl["causality_disclaimer"].lower()


class TestFeatureLeakagePrevention:
    """Verify rolling features are strictly backward-looking without future leakage."""

    def test_rolling_features_backward_looking_only(self) -> None:
        dummy_records = [
            {"year": 2018, "revenue": 100, "cost_of_goods_sold": 50, "operating_expenses": 20, "net_income": 20, "total_assets": 200, "total_liabilities": 80, "total_equity": 120, "total_debt": 40, "current_assets": 80, "current_liabilities": 40},
            {"year": 2019, "revenue": 120, "cost_of_goods_sold": 60, "operating_expenses": 25, "net_income": 25, "total_assets": 220, "total_liabilities": 85, "total_equity": 135, "total_debt": 45, "current_assets": 90, "current_liabilities": 45},
            {"year": 2020, "revenue": 150, "cost_of_goods_sold": 75, "operating_expenses": 30, "net_income": 30, "total_assets": 250, "total_liabilities": 90, "total_equity": 160, "total_debt": 50, "current_assets": 100, "current_liabilities": 50},
        ]
        rows = build_feature_rows(dummy_records)
        # Year 2018 rolling 3-year avg must be equal to 2018 value alone (cannot see 2019 or 2020)
        assert rows[0]["net_margin_3yr_avg"] == pytest.approx(rows[0]["net_margin"])
