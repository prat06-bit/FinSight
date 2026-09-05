"""Abstract base for financial data sources."""

from __future__ import annotations

from typing import Protocol

from data.models import CompanyFinancials


class DataSource(Protocol):
    """Protocol that every data source must satisfy."""

    def fetch_financials(self, ticker: str) -> CompanyFinancials:
        """Fetch financial records for *ticker* and return structured data."""
        ...

    def supports_ticker(self, ticker: str) -> bool:
        """Return ``True`` if this source can provide data for *ticker*."""
        ...
