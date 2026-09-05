"""Yahoo Finance data source via *yfinance*.

Used as a **secondary** source to supplement SEC EDGAR with market prices,
P/E ratios, and real-time data that EDGAR does not provide.
"""

from __future__ import annotations

import logging

from data.models import AnnualRecord, CompanyFinancials

logger = logging.getLogger(__name__)


class YahooFinanceSource:
    """Fetch financials and market data from Yahoo Finance via *yfinance*."""

    def fetch_financials(self, ticker: str) -> CompanyFinancials:
        """Pull annual income statement, balance sheet, and cash flow data."""
        import yfinance as yf

        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        income_stmt = stock.financials  # annual income statement
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        records: list[AnnualRecord] = []

        if income_stmt is not None and not income_stmt.empty:
            for col in income_stmt.columns:
                year = col.year if hasattr(col, "year") else int(str(col)[:4])

                rev = _get(income_stmt, col, "Total Revenue", "Operating Revenue")
                cogs = _get(income_stmt, col, "Cost Of Revenue")
                gross = _get(income_stmt, col, "Gross Profit")
                opex = _get(income_stmt, col, "Operating Expense", "Total Operating Expenses")
                op_income = _get(income_stmt, col, "Operating Income")
                ebitda_val = _get(income_stmt, col, "EBITDA", "Normalized EBITDA")
                da = _get(income_stmt, col, "Reconciled Depreciation")
                interest = _get(income_stmt, col, "Interest Expense", "Net Interest Income")
                tax = _get(income_stmt, col, "Tax Provision")
                net_inc = _get(income_stmt, col, "Net Income", "Net Income Common Stockholders")
                eps_val = _get(income_stmt, col, "Diluted EPS", "Basic EPS")

                # Balance sheet
                total_assets = _get(balance_sheet, col, "Total Assets") if balance_sheet is not None else 0.0
                total_liab = _get(balance_sheet, col, "Total Liabilities Net Minority Interest", "Total Liab") if balance_sheet is not None else 0.0
                total_eq = _get(balance_sheet, col, "Stockholders Equity", "Total Equity Gross Minority Interest") if balance_sheet is not None else 0.0
                total_debt = _get(balance_sheet, col, "Total Debt", "Long Term Debt") if balance_sheet is not None else 0.0
                curr_assets = _get(balance_sheet, col, "Current Assets") if balance_sheet is not None else 0.0
                curr_liab = _get(balance_sheet, col, "Current Liabilities") if balance_sheet is not None else 0.0
                cash = _get(balance_sheet, col, "Cash And Cash Equivalents") if balance_sheet is not None else 0.0
                ar = _get(balance_sheet, col, "Accounts Receivable", "Net Receivables") if balance_sheet is not None else 0.0
                inv = _get(balance_sheet, col, "Inventory") if balance_sheet is not None else 0.0
                shares = _get(balance_sheet, col, "Share Issued", "Ordinary Shares Number") if balance_sheet is not None else 0.0

                # Cash flow
                ocf = _get(cashflow, col, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities") if cashflow is not None else 0.0
                capex = abs(_get(cashflow, col, "Capital Expenditure", "Purchase Of PPE")) if cashflow is not None else 0.0
                divs = abs(_get(cashflow, col, "Common Stock Dividend Paid", "Cash Dividends Paid")) if cashflow is not None else 0.0

                rec = AnnualRecord(
                    year=year,
                    ticker=ticker,
                    company_name=info.get("longName", info.get("shortName", ticker)),
                    revenue=rev,
                    cost_of_goods_sold=cogs,
                    gross_profit=gross,
                    operating_expenses=opex,
                    operating_income=op_income,
                    ebitda=ebitda_val,
                    depreciation_amortization=da,
                    interest_expense=abs(interest) if interest else 0.0,
                    tax_expense=tax,
                    net_income=net_inc,
                    total_assets=total_assets,
                    total_liabilities=total_liab,
                    total_equity=total_eq,
                    total_debt=total_debt,
                    current_assets=curr_assets,
                    current_liabilities=curr_liab,
                    cash_and_equivalents=cash,
                    accounts_receivable=ar,
                    inventory=inv,
                    working_capital=curr_assets - curr_liab,
                    operating_cash_flow=ocf,
                    capital_expenditures=capex,
                    free_cash_flow=ocf - capex,
                    shares_outstanding=shares,
                    eps=eps_val,
                    dividends_paid=divs,
                    market_price=info.get("currentPrice", info.get("previousClose", 0.0)),
                    market_cap=info.get("marketCap", 0.0),
                    source="yfinance",
                )
                records.append(rec)

        logger.info("yfinance: %s — %d annual records", ticker, len(records))

        return CompanyFinancials(
            ticker=ticker,
            company_name=info.get("longName", info.get("shortName", ticker)),
            sector=info.get("sector", ""),
            records=records,
        )

    def supports_ticker(self, ticker: str) -> bool:
        """Check if yfinance can provide data for this ticker."""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker.upper())
            return bool(stock.info and stock.info.get("regularMarketPrice"))
        except Exception:
            return False


def _get(df: object, col: object, *row_names: str) -> float:
    """Safely extract a value from a pandas DataFrame."""
    if df is None:
        return 0.0
    for name in row_names:
        try:
            val = df.loc[name, col]  # type: ignore[index]
            if val is not None and str(val) not in ("nan", "None", ""):
                return float(val)
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0
