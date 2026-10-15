"""Unit tests for Step 4: Sector-Relative Financial Features & Ratio Normalization."""

from __future__ import annotations

import numpy as np
import pytest

from ml.features import BASELINE_FEATURE_COLUMNS, FEATURE_COLUMNS, SECTOR_FEATURE_COLUMNS
from ml.sector import compute_sector_relative_features, get_sector_coverage_summary, sic_to_sector
from ml.train import DEFAULT_TICKERS, collect_training_data


class TestSectorGrouping:
    """Verify deterministic SIC code to sector group mapping."""

    def test_deterministic_sic_mapping(self) -> None:
        assert sic_to_sector("3571") == "Technology"
        assert sic_to_sector("7372") == "Technology"
        assert sic_to_sector("2834") == "Healthcare & Pharma"
        assert sic_to_sector("5331") == "Retail & Consumer"
        assert sic_to_sector("3711") == "Industrial & Automotive"

    def test_invalid_sic_fallback(self) -> None:
        assert sic_to_sector("") == "General"
        assert sic_to_sector("INVALID") == "General"
        assert sic_to_sector(99999) == "General"


class TestSectorFeatureCalculation:
    """Verify point-in-time, leave-one-out sector median calculation and deltas."""

    def test_leave_one_out_and_minimum_peers(self) -> None:
        # 4 records in Technology for 2020
        records = [
            {"ticker": "AAPL", "year": 2020, "sic_code": "3571", "revenue": 100.0, "net_income": 20.0, "cost_of_goods_sold": 40.0, "operating_income": 25.0, "total_assets": 200.0, "current_assets": 80.0, "current_liabilities": 40.0, "total_debt": 10.0, "total_equity": 100.0},
            {"ticker": "MSFT", "year": 2020, "sic_code": "7372", "revenue": 100.0, "net_income": 30.0, "cost_of_goods_sold": 30.0, "operating_income": 35.0, "total_assets": 200.0, "current_assets": 100.0, "current_liabilities": 40.0, "total_debt": 20.0, "total_equity": 100.0},
            {"ticker": "ADBE", "year": 2020, "sic_code": "7372", "revenue": 100.0, "net_income": 40.0, "cost_of_goods_sold": 20.0, "operating_income": 45.0, "total_assets": 200.0, "current_assets": 120.0, "current_liabilities": 40.0, "total_debt": 30.0, "total_equity": 100.0},
            {"ticker": "ORCL", "year": 2020, "sic_code": "7372", "revenue": 100.0, "net_income": 50.0, "cost_of_goods_sold": 10.0, "operating_income": 55.0, "total_assets": 200.0, "current_assets": 140.0, "current_liabilities": 40.0, "total_debt": 40.0, "total_equity": 100.0},
        ]
        enriched = compute_sector_relative_features(records, min_peers=3, use_leave_one_out=True)

        aapl_rec = [r for r in enriched if r["ticker"] == "AAPL"][0]
        # AAPL net_margin = 0.20
        # Peers for AAPL (MSFT=0.30, ADBE=0.40, ORCL=0.50) -> median peer net margin = 0.40
        # AAPL net_margin_sector_delta = 0.20 - 0.40 = -0.20
        assert aapl_rec["net_margin_sector_delta"] == pytest.approx(-0.20)
        assert aapl_rec["sector_peers_count"] == 3

    def test_insufficient_peers_fallback(self) -> None:
        # Only 2 records (less than min_peers=3)
        records = [
            {"ticker": "TSLA", "year": 2020, "sic_code": "3711", "revenue": 100.0, "net_income": 10.0},
            {"ticker": "F", "year": 2020, "sic_code": "3711", "revenue": 100.0, "net_income": 5.0},
        ]
        enriched = compute_sector_relative_features(records, min_peers=3, use_leave_one_out=True)

        tsla_rec = [r for r in enriched if r["ticker"] == "TSLA"][0]
        # Fewer than 3 peers -> deltas must fall back safely to 0.0
        assert tsla_rec["net_margin_sector_delta"] == 0.0
        assert tsla_rec["sector_peers_count"] == 1

    def test_no_future_year_leakage_in_sector_medians(self) -> None:
        records = [
            {"ticker": "AAPL", "year": 2020, "sic_code": "3571", "revenue": 100.0, "net_income": 10.0},
            {"ticker": "MSFT", "year": 2020, "sic_code": "7372", "revenue": 100.0, "net_income": 10.0},
            {"ticker": "ADBE", "year": 2020, "sic_code": "7372", "revenue": 100.0, "net_income": 10.0},
            {"ticker": "ORCL", "year": 2020, "sic_code": "7372", "revenue": 100.0, "net_income": 10.0},
            # Future 2025 outlier observation
            {"ticker": "NVDA", "year": 2025, "sic_code": "3674", "revenue": 100.0, "net_income": 9999.0},
        ]
        enriched = compute_sector_relative_features(records, min_peers=3, use_leave_one_out=True)

        aapl_2020 = [r for r in enriched if r["ticker"] == "AAPL" and r["year"] == 2020][0]
        # 2025 NVDA net income (9999.0) must NOT affect AAPL's 2020 sector delta
        assert abs(aapl_2020["net_margin_sector_delta"]) < 1.0


class TestFeatureIntegration:
    """Verify feature counts and baseline preservation."""

    def test_feature_counts(self) -> None:
        assert len(BASELINE_FEATURE_COLUMNS) == 27
        assert len(SECTOR_FEATURE_COLUMNS) == 6
        assert len(FEATURE_COLUMNS) == 33

    def test_training_data_collection_with_sector_features(self) -> None:
        features, labels, years = collect_training_data(DEFAULT_TICKERS[:5])

        assert len(features) > 0
        assert len(features[0]) == 33
        assert len(labels) == len(features)
        assert len(years) == len(features)


class TestSectorCoverage:
    """Verify data quality and sector coverage reporting."""

    def test_sector_coverage_summary(self) -> None:
        summary = get_sector_coverage_summary(DEFAULT_TICKERS[:5])

        assert summary["total_tickers_evaluated"] > 0
        assert "sector_group_breakdown" in summary
        assert "coverage_limitation_note" in summary
