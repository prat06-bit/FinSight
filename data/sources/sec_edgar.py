"""SEC EDGAR data source — Company Facts API (XBRL) and Submissions API.
This is the **primary** data source for FinSight.  It pulls structured
annual financials for any US public company using the free, keyless SEC
EDGAR APIs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from config import CACHE_DIR, SEC_BASE_URL, SEC_COMPANY_TICKERS_URL, SEC_RATE_LIMIT, SEC_USER_AGENT
from data.models import AnnualRecord, CompanyFinancials
from ml.sector import sic_to_sector

logger = logging.getLogger(__name__)

# XBRL tag mapping — multiple fallbacks per financial concept
_XBRL_TAG_MAP: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cost_of_goods_sold": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_expenses": [
        "OperatingExpenses",
        "CostsAndExpenses",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "ebitda": [
        "EBITDA",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
    ],
    "tax_expense": [
        "IncomeTaxExpenseBenefit",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "total_assets": ["Assets"],
    "total_liabilities": [
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "total_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryFinishedGoodsAndWorkInProcess",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "capital_expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    "eps": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    "dividends_paid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ],
}


class SECEdgarSource:
    """Fetch structured financials from the SEC Company Facts API (XBRL).
    Rate-limited to comply with SEC's 10 requests/second policy.
    """

    def __init__(self) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json"},
            timeout=30.0,
        )
        self._last_request_time: float = 0.0
        self._ticker_cache: dict[str, dict[str, Any]] | None = None

    #  public interface 

    def fetch_financials(self, ticker: str) -> CompanyFinancials:
        """Return annual financial records for *ticker* from SEC EDGAR."""
        ticker = ticker.upper()
        info = self._resolve_ticker(ticker)
        cik_padded = str(info["cik_str"]).zfill(10)
        company_name = str(info.get("title", ticker))

        facts = self._fetch_company_facts(cik_padded)
        submissions = self._fetch_submissions(cik_padded)

        # Extract SIC metadata
        sic_code = str(submissions.get("sic", ""))
        sic_description = str(submissions.get("sicDescription", ""))

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        # Instant (balance sheet & shares) vs duration (income statement & cash flow)
        INSTANT_FIELDS = {
            "total_assets", "total_liabilities", "total_equity", "total_debt",
            "current_assets", "current_liabilities", "cash_and_equivalents",
            "accounts_receivable", "inventory", "shares_outstanding",
        }

        # Build {year: {field: value}} from XBRL tags
        year_data: dict[int, dict[str, float]] = {}
        for field, tags in _XBRL_TAG_MAP.items():
            is_dur = field not in INSTANT_FIELDS
            values_by_year = self._extract_annual_values(us_gaap, tags, is_duration=is_dur)
            for year, value in values_by_year.items():
                year_data.setdefault(year, {})[field] = value

        # Build AnnualRecord objects
        records: list[AnnualRecord] = []
        for year in sorted(year_data.keys()):
            raw = year_data[year]
            # Compute derived fields if missing
            revenue = raw.get("revenue", 0.0)
            cogs = raw.get("cost_of_goods_sold", 0.0)
            gross_profit = raw.get("gross_profit", revenue - cogs if revenue and cogs else 0.0)
            operating_income = raw.get("operating_income", 0.0)
            operating_expenses = raw.get("operating_expenses", 0.0)
            if not operating_expenses and revenue and operating_income:
                operating_expenses = revenue - operating_income
            ocf = raw.get("operating_cash_flow", 0.0)
            capex = raw.get("capital_expenditures", 0.0)
            fcf = ocf - capex if ocf else 0.0
            ca = raw.get("current_assets", 0.0)
            cl = raw.get("current_liabilities", 0.0)

            rec = AnnualRecord(
                year=year,
                ticker=ticker,
                company_name=company_name,
                sic_code=sic_code,
                sector=sic_to_sector(sic_code),
                revenue=revenue,
                cost_of_goods_sold=cogs,
                gross_profit=gross_profit,
                operating_expenses=operating_expenses,
                operating_income=operating_income,
                ebitda=raw.get("ebitda", 0.0),
                depreciation_amortization=raw.get("depreciation_amortization", 0.0),
                interest_expense=raw.get("interest_expense", 0.0),
                tax_expense=raw.get("tax_expense", 0.0),
                net_income=raw.get("net_income", 0.0),
                total_assets=raw.get("total_assets", 0.0),
                total_liabilities=raw.get("total_liabilities", 0.0),
                total_equity=raw.get("total_equity", 0.0),
                total_debt=raw.get("total_debt", 0.0),
                current_assets=ca,
                current_liabilities=cl,
                cash_and_equivalents=raw.get("cash_and_equivalents", 0.0),
                accounts_receivable=raw.get("accounts_receivable", 0.0),
                inventory=raw.get("inventory", 0.0),
                working_capital=ca - cl,
                operating_cash_flow=ocf,
                capital_expenditures=capex,
                free_cash_flow=fcf,
                shares_outstanding=raw.get("shares_outstanding", 0.0),
                eps=raw.get("eps", 0.0),
                dividends_paid=raw.get("dividends_paid", 0.0),
                source="sec_edgar",
            )
            # Only keep years where we have at least revenue or total_assets
            if rec.revenue > 0 or rec.total_assets > 0:
                records.append(rec)

        logger.info("SEC EDGAR: %s (%s) — %d annual records", ticker, company_name, len(records))

        return CompanyFinancials(
            ticker=ticker,
            company_name=company_name,
            cik=cik_padded,
            sic_code=sic_code,
            sic_description=sic_description,
            records=records,
        )

    def supports_ticker(self, ticker: str) -> bool:
        """Return True if the ticker exists in SEC's company list."""
        try:
            self._resolve_ticker(ticker.upper())
            return True
        except ValueError:
            return False

    def get_filing_urls(self, ticker: str, form_type: str = "10-K", limit: int = 5) -> list[dict[str, str]]:
        """Return URLs for recent filings of the given type."""
        info = self._resolve_ticker(ticker.upper())
        cik_padded = str(info["cik_str"]).zfill(10)
        submissions = self._fetch_submissions(cik_padded)

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results: list[dict[str, str]] = []
        for i, form in enumerate(forms):
            if form == form_type and i < len(accessions):
                accession_dashed = accessions[i].replace("-", "")
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik_padded)}/{accession_dashed}/{primary_docs[i]}"
                )
                results.append({
                    "form": form,
                    "date": dates[i] if i < len(dates) else "",
                    "accession": accessions[i],
                    "url": url,
                })
                if len(results) >= limit:
                    break

        return results

    #  private helpers 

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < SEC_RATE_LIMIT:
            time.sleep(SEC_RATE_LIMIT - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, url: str) -> dict[str, Any]:
        self._rate_limit()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.json()

    def _resolve_ticker(self, ticker: str) -> dict[str, Any]:
        """Convert a ticker symbol to SEC CIK + company info."""
        if self._ticker_cache is None:
            cache_path = CACHE_DIR / "company_tickers.json"
            if cache_path.exists():
                self._ticker_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                self._rate_limit()
                resp = self._client.get(SEC_COMPANY_TICKERS_URL)
                resp.raise_for_status()
                self._ticker_cache = resp.json()
                cache_path.write_text(json.dumps(self._ticker_cache), encoding="utf-8")

        for entry in self._ticker_cache.values():
            if str(entry.get("ticker", "")).upper() == ticker:
                return entry

        raise ValueError(f"Ticker '{ticker}' not found in SEC company list")

    def _fetch_company_facts(self, cik_padded: str) -> dict[str, Any]:
        """GET /api/xbrl/companyfacts/CIK{cik}.json — cached locally."""
        cache_path = CACHE_DIR / f"facts_{cik_padded}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        data = self._get(url)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _fetch_submissions(self, cik_padded: str) -> dict[str, Any]:
        """GET /submissions/CIK{cik}.json — cached locally."""
        cache_path = CACHE_DIR / f"submissions_{cik_padded}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{SEC_BASE_URL}/submissions/CIK{cik_padded}.json"
        data = self._get(url)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _extract_annual_values(
        self,
        us_gaap: dict[str, Any],
        tags: list[str],
        is_duration: bool = True,
    ) -> dict[int, float]:
        """Try multiple XBRL tags and return {fiscal_year: value} for 10-K filings.
        Combines values across all tag fallbacks (prioritizing earlier tags in the list)
        and determines fiscal year from period end date rather than SEC filing fy metadata.
        """
        from datetime import datetime

        results: dict[int, tuple[float, str, int]] = {}  # yr -> (val, filed, tag_priority)

        for priority, tag in enumerate(tags):
            concept = us_gaap.get(tag)
            if not concept:
                continue

            units = concept.get("units", {})
            for unit_key, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    form = str(entry.get("form", ""))
                    fp = str(entry.get("fp", ""))

                    if form != "10-K" or fp not in ("FY", ""):
                        continue

                    end = entry.get("end")
                    if not end:
                        continue

                    yr = int(str(end)[:4])

                    if is_duration:
                        start = entry.get("start")
                        if not start:
                            continue
                        try:
                            d1 = datetime.strptime(str(start), "%Y-%m-%d")
                            d2 = datetime.strptime(str(end), "%Y-%m-%d")
                            days = (d2 - d1).days
                            if not (340 <= days <= 380):
                                continue
                        except ValueError:
                            continue

                    val = float(entry.get("val", 0.0))
                    filed = str(entry.get("filed", ""))

                    # Prioritize: 1) higher tag priority (lower index), 2) latest filing date
                    if yr not in results:
                        results[yr] = (val, filed, priority)
                    else:
                        prev_val, prev_filed, prev_prio = results[yr]
                        if priority < prev_prio or (priority == prev_prio and filed > prev_filed):
                            results[yr] = (val, filed, priority)

        return {yr: val_tuple[0] for yr, val_tuple in sorted(results.items())}
