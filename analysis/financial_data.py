from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

MoneyRecord = dict[str, float | int | str]

REQUIRED_COLUMNS = {
    "year",
    "revenue",
    "cost_of_goods_sold",
    "operating_expenses",
    "net_income",
    "total_assets",
    "total_equity",
    "total_debt",
    "current_assets",
    "current_liabilities",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_financial_data(filepath: str | Path) -> list[MoneyRecord]:
    """Load yearly financial records from a CSV file (backward compat)."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Financial data file not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("Financial data file is empty.")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Financial data is missing required columns: {missing}")

        rows: list[MoneyRecord] = []
        for row in reader:
            try:
                rows.append(_coerce_record(row))
            except (ValueError, KeyError):
                continue  # skip malformed rows

    if not rows:
        raise ValueError("Financial data file has headers but no records.")

    return sorted(rows, key=lambda row: int(row["year"]))


def load_from_ticker(
    ticker: str,
    source: str = "sec_edgar",
) -> list[MoneyRecord]:
    """Fetch live financial data for *ticker* and return as MoneyRecords.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (e.g. ``"AAPL"``).
    source:
        ``"sec_edgar"`` (default) or ``"yahoo"``.
    """
    if source == "sec_edgar":
        from data.sources.sec_edgar import SECEdgarSource

        src = SECEdgarSource()
    elif source == "yahoo":
        from data.sources.yahoo import YahooFinanceSource

        src = YahooFinanceSource()
    else:
        raise ValueError(f"Unknown data source: {source}")

    company = src.fetch_financials(ticker)
    records = company.to_dicts()

    if not records:
        raise ValueError(f"No financial data found for {ticker} via {source}")

    logger.info(
        "Loaded %d records for %s (%s) via %s",
        len(records),
        ticker,
        company.company_name,
        source,
    )
    return records


# ---------------------------------------------------------------------------
# Metrics & reporting (unchanged logic, kept for backward compat)
# ---------------------------------------------------------------------------


def calculate_metrics(records: Iterable[MoneyRecord]) -> list[MoneyRecord]:
    """Add profitability, liquidity, leverage, and growth metrics."""
    enriched: list[MoneyRecord] = []
    previous_revenue: float | None = None

    for record in records:
        revenue = float(record["revenue"])
        gross_profit = revenue - float(record["cost_of_goods_sold"])
        operating_income = gross_profit - float(record["operating_expenses"])
        current_liabilities = float(record["current_liabilities"])
        total_equity = float(record["total_equity"])

        metrics: MoneyRecord = {
            **record,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "gross_margin": _safe_divide(gross_profit, revenue),
            "operating_margin": _safe_divide(operating_income, revenue),
            "net_margin": _safe_divide(float(record["net_income"]), revenue),
            "return_on_assets": _safe_divide(float(record["net_income"]), float(record["total_assets"])),
            "return_on_equity": _safe_divide(float(record["net_income"]), total_equity),
            "debt_to_equity": _safe_divide(float(record["total_debt"]), total_equity),
            "current_ratio": _safe_divide(float(record["current_assets"]), current_liabilities),
            "revenue_growth": None if previous_revenue is None else _safe_divide(revenue - previous_revenue, previous_revenue),
        }
        enriched.append(metrics)
        previous_revenue = revenue

    return enriched


def summarize_financials(records: list[MoneyRecord]) -> dict[str, float | int | str]:
    """Create a compact summary for the latest year in the dataset."""
    metrics = calculate_metrics(records)
    latest = metrics[-1]
    first = metrics[0]

    return {
        "ticker": str(latest.get("ticker", "")),
        "company_name": str(latest.get("company_name", "")),
        "latest_year": int(latest["year"]),
        "revenue": float(latest["revenue"]),
        "net_income": float(latest["net_income"]),
        "net_margin": float(latest["net_margin"]),
        "current_ratio": float(latest["current_ratio"]),
        "debt_to_equity": float(latest["debt_to_equity"]),
        "revenue_cagr": _compound_annual_growth_rate(
            float(first["revenue"]),
            float(latest["revenue"]),
            int(latest["year"]) - int(first["year"]),
        ),
        "health_signal": _health_signal(latest),
    }


def format_report(records: list[MoneyRecord]) -> str:
    """Render an easy-to-read financial health report."""
    metrics = calculate_metrics(records)
    summary = summarize_financials(records)

    ticker_str = f" ({summary['ticker']})" if summary.get("ticker") else ""
    company_str = str(summary.get("company_name", ""))
    header = f"FinSight Financial Report{' — ' + company_str if company_str else ''}{ticker_str}"

    lines = [
        header,
        f"Latest year: {summary['latest_year']}",
        "",
        "Summary",
        f"- Revenue: {_format_money(summary['revenue'])}",
        f"- Net income: {_format_money(summary['net_income'])}",
        f"- Net margin: {_format_percent(summary['net_margin'])}",
        f"- Current ratio: {summary['current_ratio']:.2f}",
        f"- Debt to equity: {summary['debt_to_equity']:.2f}",
        f"- Revenue CAGR: {_format_percent(summary['revenue_cagr'])}",
        f"- Health signal: {summary['health_signal']}",
        "",
        "Yearly trend",
        "Year  Revenue      Net margin  Revenue growth  Current ratio  D/E",
    ]

    for row in metrics:
        growth = "n/a" if row["revenue_growth"] is None else _format_percent(row["revenue_growth"])
        lines.append(
            f"{int(row['year'])}  "
            f"{_format_money(row['revenue']):>11}  "
            f"{_format_percent(row['net_margin']):>10}  "
            f"{growth:>14}  "
            f"{row['current_ratio']:>13.2f}  "
            f"{row['debt_to_equity']:>4.2f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _coerce_record(row: dict[str, str]) -> MoneyRecord:
    record: MoneyRecord = {}
    for key, value in row.items():
        value = value.strip()
        if not value or not _is_numeric(value):
            if key in REQUIRED_COLUMNS and key != "year":
                record[key] = 0.0
            elif key == "year":
                record[key] = int(float(value))
            else:
                record[key] = value
            continue
        if key == "year":
            record[key] = int(float(value))
        else:
            record[key] = float(value)
    return record


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _compound_annual_growth_rate(start: float, end: float, periods: int) -> float:
    if start <= 0 or periods <= 0:
        return 0.0
    return (end / start) ** (1 / periods) - 1


def _health_signal(record: MoneyRecord) -> str:
    if (
        float(record["net_margin"]) >= 0.10
        and float(record["current_ratio"]) >= 1.2
        and float(record["debt_to_equity"]) <= 1.0
    ):
        return "Strong"
    if float(record["net_margin"]) > 0 and float(record["current_ratio"]) >= 1.0:
        return "Stable"
    return "Needs attention"


def _format_money(value: float | int) -> str:
    return f"${float(value):,.0f}"


def _format_percent(value: float | int) -> str:
    return f"{float(value) * 100:.1f}%"
