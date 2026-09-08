"""Sector-Relative Ratio Normalization Engine for FinSight.
Implements leakage-safe industry ratio benchmarking:
- Deterministic SIC (Standard Industrial Classification) to Sector Grouping
- Leave-One-Out Peer Median Calculation (excludes focal company from its own benchmark)
- Temporal Boundary Enforcement (uses ONLY point-in-time observations at year T)
- Minimum Peer Observation Threshold (N >= 3)
- 6 Sector-Relative Features (difference from sector median):
    1. net_margin_sector_delta
    2. gross_margin_sector_delta
    3. operating_margin_sector_delta
    4. roa_sector_delta
    5. current_ratio_sector_delta
    6. debt_to_equity_sector_delta
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from analysis.financial_data import MoneyRecord

logger = logging.getLogger(__name__)

SECTOR_FEATURE_COLUMNS = [
    "net_margin_sector_delta",
    "gross_margin_sector_delta",
    "operating_margin_sector_delta",
    "roa_sector_delta",
    "current_ratio_sector_delta",
    "debt_to_equity_sector_delta",
]


def sic_to_sector(sic_code: str | int) -> str:
    """Map a 4-digit or 2-digit SEC SIC code to a standard industry sector group.
    Deterministic grouping based on SEC EDGAR classification taxonomy.
    """
    sic_str = str(sic_code).strip()
    if not sic_str or not sic_str.isdigit():
        return "General"

    code = int(sic_str)

    # Specific 4-digit ranges
    if (2830 <= code <= 2836) or (3820 <= code <= 3849) or (6300 <= code <= 6399):
        return "Healthcare & Pharma"
    elif (3570 <= code <= 3579) or (3600 <= code <= 3699) or (7370 <= code <= 7379):
        return "Technology"
    elif (5200 <= code <= 5999) or (2080 <= code <= 2089) or (3000 <= code <= 3099) or (5800 <= code <= 5899):
        return "Retail & Consumer"
    elif 3700 <= code <= 3799:
        return "Industrial & Automotive"

    # 2-digit Major Group fallbacks
    div = code // 100
    if div in (28, 38, 80):
        return "Healthcare & Pharma"
    elif div in (35, 36, 73):
        return "Technology"
    elif div in (20, 30, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59):
        return "Retail & Consumer"
    elif div in (37, 39):
        return "Industrial & Automotive"

    return "General"


def compute_sector_relative_features(
    records: list[MoneyRecord],
    min_peers: int = 3,
    use_leave_one_out: bool = True,
) -> list[MoneyRecord]:
    """Compute leakage-safe sector-relative features across all company-year records.
    Leakage Prevention Rules:
    - Point-In-Time: For an observation at year T, only peer records from year T are evaluated.
    - Leave-One-Out: Excludes focal company C from its own sector median calculation when enabled.
    - Minimum Peer Threshold: Requires >= min_peers eligible peer observations. If < min_peers,
      sector deltas default safely to 0.0 (no distortion).
    """
    if not records:
        return []

    # Map year + sector -> list of peer records
    recs_by_year_sector: dict[tuple[int, str], list[MoneyRecord]] = {}
    for r in records:
        yr = int(r.get("year", 0))
        sic = r.get("sic_code", "")
        sector = r.get("sector") or sic_to_sector(sic)
        recs_by_year_sector.setdefault((yr, sector), []).append(r)

    enriched_records: list[MoneyRecord] = []

    for r in records:
        yr = int(r.get("year", 0))
        ticker = str(r.get("ticker", ""))
        sic = r.get("sic_code", "")
        sector = r.get("sector") or sic_to_sector(sic)

        # Peer set for this year and sector
        sector_peers = recs_by_year_sector.get((yr, sector), [])

        if use_leave_one_out:
            eligible_peers = [p for p in sector_peers if str(p.get("ticker", "")) != ticker]
        else:
            eligible_peers = sector_peers

        # Extract focal company metrics
        rev = float(r.get("revenue", 0.0))
        net_inc = float(r.get("net_income", 0.0))
        cogs = float(r.get("cost_of_goods_sold", 0.0))
        op_inc = float(r.get("operating_income", 0.0))
        assets = float(r.get("total_assets", 0.0))
        equity = float(r.get("total_equity", 0.0))
        debt = float(r.get("total_debt", 0.0))
        ca = float(r.get("current_assets", 0.0))
        cl = float(r.get("current_liabilities", 0.0))

        c_net_margin = net_inc / rev if rev > 0 else 0.0
        c_gross_margin = (rev - cogs) / rev if rev > 0 else 0.0
        c_op_margin = op_inc / rev if rev > 0 else 0.0
        c_roa = net_inc / assets if assets > 0 else 0.0
        c_current_ratio = ca / cl if cl > 0 else 0.0
        c_debt_to_equity = debt / equity if equity > 0 else 0.0

        r_copy = dict(r)
        r_copy["sic_code"] = str(sic)
        r_copy["sector"] = sector
        r_copy["sector_peers_count"] = len(eligible_peers)

        if len(eligible_peers) >= min_peers:
            p_net = [
                float(p.get("net_income", 0.0)) / float(p.get("revenue", 1.0))
                for p in eligible_peers if float(p.get("revenue", 0.0)) > 0
            ]
            p_gross = [
                (float(p.get("revenue", 0.0)) - float(p.get("cost_of_goods_sold", 0.0))) / float(p.get("revenue", 1.0))
                for p in eligible_peers if float(p.get("revenue", 0.0)) > 0
            ]
            p_op = [
                float(p.get("operating_income", 0.0)) / float(p.get("revenue", 1.0))
                for p in eligible_peers if float(p.get("revenue", 0.0)) > 0
            ]
            p_roa = [
                float(p.get("net_income", 0.0)) / float(p.get("total_assets", 1.0))
                for p in eligible_peers if float(p.get("total_assets", 0.0)) > 0
            ]
            p_cr = [
                float(p.get("current_assets", 0.0)) / float(p.get("current_liabilities", 1.0))
                for p in eligible_peers if float(p.get("current_liabilities", 0.0)) > 0
            ]
            p_d2e = [
                float(p.get("total_debt", 0.0)) / float(p.get("total_equity", 1.0))
                for p in eligible_peers if float(p.get("total_equity", 0.0)) > 0
            ]

            med_net = float(np.median(p_net)) if p_net else 0.0
            med_gross = float(np.median(p_gross)) if p_gross else 0.0
            med_op = float(np.median(p_op)) if p_op else 0.0
            med_roa = float(np.median(p_roa)) if p_roa else 0.0
            med_cr = float(np.median(p_cr)) if p_cr else 0.0
            med_d2e = float(np.median(p_d2e)) if p_d2e else 0.0

            r_copy["net_margin_sector_delta"] = c_net_margin - med_net
            r_copy["gross_margin_sector_delta"] = c_gross_margin - med_gross
            r_copy["operating_margin_sector_delta"] = c_op_margin - med_op
            r_copy["roa_sector_delta"] = c_roa - med_roa
            r_copy["current_ratio_sector_delta"] = c_current_ratio - med_cr
            r_copy["debt_to_equity_sector_delta"] = c_debt_to_equity - med_d2e
            r_copy["sector_median_net_margin"] = med_net
        else:
            # Fallback when peer count < min_peers: set deltas safely to 0.0
            r_copy["net_margin_sector_delta"] = 0.0
            r_copy["gross_margin_sector_delta"] = 0.0
            r_copy["operating_margin_sector_delta"] = 0.0
            r_copy["roa_sector_delta"] = 0.0
            r_copy["current_ratio_sector_delta"] = 0.0
            r_copy["debt_to_equity_sector_delta"] = 0.0
            r_copy["sector_median_net_margin"] = c_net_margin

        enriched_records.append(r_copy)

    return enriched_records


def get_sector_coverage_summary(
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze SIC code distribution and data quality/coverage across tickers."""
    from ml.train import DEFAULT_TICKERS
    from data.sources.sec_edgar import SECEdgarSource

    tickers = tickers or DEFAULT_TICKERS
    sec = SECEdgarSource()

    sector_counts: dict[str, list[str]] = {}
    sic_map: dict[str, tuple[str, str]] = {}

    for t in tickers:
        try:
            c = sec.fetch_financials(t)
            sec_group = sic_to_sector(c.sic_code)
            sector_counts.setdefault(sec_group, []).append(t)
            sic_map[t] = (c.sic_code, c.sic_description)
        except Exception:
            continue

    insufficient_groups = {g: len(tks) for g, tks in sector_counts.items() if len(tks) < 3}

    return {
        "total_tickers_evaluated": len(sic_map),
        "unique_sic_codes_count": len({code for code, desc in sic_map.values()}),
        "sector_groups_count": len(sector_counts),
        "sector_group_breakdown": {g: len(tks) for g, tks in sector_counts.items()},
        "groups_with_insufficient_peers_count": len(insufficient_groups),
        "insufficient_peer_groups": insufficient_groups,
        "coverage_limitation_note": (
            "The dataset contains 40 S&P 500 tickers. Major sectors (Technology, Healthcare, Retail) "
            "have 10–18 peer companies per year, meeting the N>=3 peer threshold. Niche sectors "
            "(e.g., Industrial & Automotive with 1 ticker) fall back safely to zero deltas."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = get_sector_coverage_summary()
    print("=== SECTOR COVERAGE SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
