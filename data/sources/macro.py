"""Macro-economic data source for FinSight.
Provides annual US macroeconomic indicators (Federal Funds Rate, 10-Year Treasury Yield,
Yield Curve Slope 10Y-2Y, CPI Inflation) for incorporating macro-regime signals into
financial ML risk models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MacroIndicators:
    year: int
    fed_funds_rate: float        # Effective Federal Funds Rate (%)
    treasury_10y_yield: float    # 10-Year Treasury Constant Maturity Rate (%)
    yield_curve_slope: float     # 10Y minus 2Y Treasury Yield Spread (% points)
    cpi_inflation: float        # Annual CPI Inflation Rate (%)


# Historical US Macro Data (2005 - 2025) compiled from Federal Reserve FRED & US BLS
_HISTORICAL_MACRO: dict[int, MacroIndicators] = {
    2005: MacroIndicators(2005, fed_funds_rate=4.25, treasury_10y_yield=4.39, yield_curve_slope=0.08, cpi_inflation=3.4),
    2006: MacroIndicators(2006, fed_funds_rate=5.25, treasury_10y_yield=4.70, yield_curve_slope=-0.09, cpi_inflation=3.2),
    2007: MacroIndicators(2007, fed_funds_rate=4.25, treasury_10y_yield=4.63, yield_curve_slope=0.09, cpi_inflation=2.85),
    2008: MacroIndicators(2008, fed_funds_rate=0.25, treasury_10y_yield=3.66, yield_curve_slope=1.65, cpi_inflation=3.85),
    2009: MacroIndicators(2009, fed_funds_rate=0.25, treasury_10y_yield=3.26, yield_curve_slope=2.29, cpi_inflation=-0.36),
    2010: MacroIndicators(2010, fed_funds_rate=0.25, treasury_10y_yield=3.22, yield_curve_slope=2.52, cpi_inflation=1.64),
    2011: MacroIndicators(2011, fed_funds_rate=0.25, treasury_10y_yield=2.78, yield_curve_slope=2.33, cpi_inflation=3.16),
    2012: MacroIndicators(2012, fed_funds_rate=0.25, treasury_10y_yield=1.80, yield_curve_slope=1.53, cpi_inflation=2.07),
    2013: MacroIndicators(2013, fed_funds_rate=0.25, treasury_10y_yield=2.35, yield_curve_slope=2.04, cpi_inflation=1.46),
    2014: MacroIndicators(2014, fed_funds_rate=0.25, treasury_10y_yield=2.54, yield_curve_slope=2.08, cpi_inflation=1.62),
    2015: MacroIndicators(2015, fed_funds_rate=0.50, treasury_10y_yield=2.14, yield_curve_slope=1.45, cpi_inflation=0.12),
    2016: MacroIndicators(2016, fed_funds_rate=0.75, treasury_10y_yield=1.84, yield_curve_slope=1.00, cpi_inflation=1.26),
    2017: MacroIndicators(2017, fed_funds_rate=1.50, treasury_10y_yield=2.33, yield_curve_slope=0.93, cpi_inflation=2.13),
    2018: MacroIndicators(2018, fed_funds_rate=2.50, treasury_10y_yield=2.91, yield_curve_slope=0.38, cpi_inflation=2.44),
    2019: MacroIndicators(2019, fed_funds_rate=1.75, treasury_10y_yield=2.14, yield_curve_slope=0.17, cpi_inflation=1.81),
    2020: MacroIndicators(2020, fed_funds_rate=0.25, treasury_10y_yield=0.89, yield_curve_slope=0.53, cpi_inflation=1.23),
    2021: MacroIndicators(2021, fed_funds_rate=0.25, treasury_10y_yield=1.45, yield_curve_slope=1.18, cpi_inflation=4.70),
    2022: MacroIndicators(2022, fed_funds_rate=4.50, treasury_10y_yield=2.95, yield_curve_slope=-0.52, cpi_inflation=8.00),
    2023: MacroIndicators(2023, fed_funds_rate=5.50, treasury_10y_yield=3.96, yield_curve_slope=-0.54, cpi_inflation=4.10),
    2024: MacroIndicators(2024, fed_funds_rate=4.50, treasury_10y_yield=4.22, yield_curve_slope=0.08, cpi_inflation=2.90),
    2025: MacroIndicators(2025, fed_funds_rate=4.25, treasury_10y_yield=4.40, yield_curve_slope=0.15, cpi_inflation=2.60),
}


class MacroSource:
    """Provides macro-economic regime data for any target fiscal year."""
    def get_indicators(self, year: int) -> MacroIndicators:
        """Return macro indicators for *year*. Defaults to nearest year if outside 2005-2025."""
        if year in _HISTORICAL_MACRO:
            return _HISTORICAL_MACRO[year]

        # Clamp to bounds
        min_yr = min(_HISTORICAL_MACRO.keys())
        max_yr = max(_HISTORICAL_MACRO.keys())

        if year < min_yr:
            return _HISTORICAL_MACRO[min_yr]
        return _HISTORICAL_MACRO[max_yr]

    def get_as_dict(self, year: int) -> dict[str, float]:
        """Return macro indicators for *year* as a dictionary."""
        ind = self.get_indicators(year)
        return {
            "fed_funds_rate": ind.fed_funds_rate,
            "treasury_10y_yield": ind.treasury_10y_yield,
            "yield_curve_slope": ind.yield_curve_slope,
            "cpi_inflation": ind.cpi_inflation,
        }
