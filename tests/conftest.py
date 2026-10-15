"""Shared test fixtures for FinSight."""

from __future__ import annotations

import pytest

from analysis.financial_data import MoneyRecord


@pytest.fixture
def sample_records() -> list[MoneyRecord]:
    """Two years of minimal financial records for testing."""
    return [
        {
            "year": 2023,
            "ticker": "TEST",
            "company_name": "Test Corp",
            "revenue": 1_000_000.0,
            "cost_of_goods_sold": 600_000.0,
            "gross_profit": 400_000.0,
            "operating_expenses": 200_000.0,
            "operating_income": 200_000.0,
            "net_income": 120_000.0,
            "total_assets": 2_000_000.0,
            "total_liabilities": 800_000.0,
            "total_equity": 1_200_000.0,
            "total_debt": 500_000.0,
            "current_assets": 600_000.0,
            "current_liabilities": 300_000.0,
            "cash_and_equivalents": 200_000.0,
            "accounts_receivable": 150_000.0,
            "inventory": 100_000.0,
            "working_capital": 300_000.0,
            "operating_cash_flow": 180_000.0,
            "capital_expenditures": 50_000.0,
            "free_cash_flow": 130_000.0,
            "interest_expense": 30_000.0,
            "tax_expense": 50_000.0,
            "shares_outstanding": 100_000.0,
            "eps": 1.20,
            "dividends_paid": 20_000.0,
            "market_price": 15.0,
            "market_cap": 1_500_000.0,
            "source": "test",
        },
        {
            "year": 2024,
            "ticker": "TEST",
            "company_name": "Test Corp",
            "revenue": 1_100_000.0,
            "cost_of_goods_sold": 650_000.0,
            "gross_profit": 450_000.0,
            "operating_expenses": 220_000.0,
            "operating_income": 230_000.0,
            "net_income": 140_000.0,
            "total_assets": 2_200_000.0,
            "total_liabilities": 880_000.0,
            "total_equity": 1_320_000.0,
            "total_debt": 520_000.0,
            "current_assets": 660_000.0,
            "current_liabilities": 330_000.0,
            "cash_and_equivalents": 220_000.0,
            "accounts_receivable": 165_000.0,
            "inventory": 110_000.0,
            "working_capital": 330_000.0,
            "operating_cash_flow": 200_000.0,
            "capital_expenditures": 60_000.0,
            "free_cash_flow": 140_000.0,
            "interest_expense": 32_000.0,
            "tax_expense": 58_000.0,
            "shares_outstanding": 100_000.0,
            "eps": 1.40,
            "dividends_paid": 22_000.0,
            "market_price": 17.0,
            "market_cap": 1_700_000.0,
            "source": "test",
        },
    ]


@pytest.fixture
def extended_records(sample_records: list[MoneyRecord]) -> list[MoneyRecord]:
    """Five years of records for ML and forecasting tests."""
    base = sample_records[0]
    records = []
    for i, year in enumerate(range(2020, 2025)):
        growth = 1 + 0.08 * i
        records.append({
            **base,
            "year": year,
            "revenue": base["revenue"] * growth,
            "net_income": base["net_income"] * growth * (0.95 + 0.01 * i),
            "total_assets": base["total_assets"] * growth,
            "total_equity": base["total_equity"] * growth,
            "total_debt": base["total_debt"] * (1 + 0.05 * i),
            "current_assets": base["current_assets"] * growth,
            "current_liabilities": base["current_liabilities"] * (1 + 0.06 * i),
            "operating_cash_flow": base["operating_cash_flow"] * growth,
            "free_cash_flow": base["free_cash_flow"] * growth,
            "interest_expense": base["interest_expense"] * (1 + 0.03 * i),
        })
    return records
