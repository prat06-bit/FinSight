"""Pydantic models for structured financial data throughout FinSight."""

from __future__ import annotations
from pydantic import BaseModel, Field


class AnnualRecord(BaseModel):
    """One year of financial data for a single company."""

    year: int
    ticker: str = ""
    company_name: str = ""

    #  Income statement 
    revenue: float = 0.0
    cost_of_goods_sold: float = 0.0
    gross_profit: float = 0.0
    operating_expenses: float = 0.0
    operating_income: float = 0.0
    ebitda: float = 0.0
    depreciation_amortization: float = 0.0
    interest_expense: float = 0.0
    tax_expense: float = 0.0
    net_income: float = 0.0

    # Balance sheet 
    total_assets: float = 0.0
    total_liabilities: float = 0.0
    total_equity: float = 0.0
    total_debt: float = 0.0
    current_assets: float = 0.0
    current_liabilities: float = 0.0
    cash_and_equivalents: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    working_capital: float = 0.0

    # Cash flow 
    operating_cash_flow: float = 0.0
    capital_expenditures: float = 0.0
    free_cash_flow: float = 0.0

    # Per-share / market data 
    shares_outstanding: float = 0.0
    eps: float = 0.0
    dividends_paid: float = 0.0
    market_price: float = 0.0
    market_cap: float = 0.0

    # Metadata 
    sic_code: str = ""
    sector: str = ""
    source: str = ""  # "sec_edgar", "yfinance", "csv"
    filing_date: str = ""

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a plain dict matching the legacy MoneyRecord format."""
        return self.model_dump()


class CompanyFinancials(BaseModel):
    """Complete financial dataset for one company across multiple years."""

    ticker: str
    company_name: str = ""
    cik: str = ""
    sector: str = ""
    sic_code: str = ""
    sic_description: str = ""
    records: list[AnnualRecord] = Field(default_factory=list)

    def sorted_records(self) -> list[AnnualRecord]:
        """Return records sorted by year ascending."""
        return sorted(self.records, key=lambda r: r.year)

    def to_dicts(self) -> list[dict[str, float | int | str]]:
        """Return list of plain dicts for backward compatibility."""
        return [r.to_dict() for r in self.sorted_records()]

    def latest_record(self) -> AnnualRecord | None:
        """Return the most recent year's record."""
        recs = self.sorted_records()
        return recs[-1] if recs else None

    def years_range(self) -> tuple[int, int] | None:
        """Return (first_year, last_year) or None if empty."""
        recs = self.sorted_records()
        if not recs:
            return None
        return recs[0].year, recs[-1].year
