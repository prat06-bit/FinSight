from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from config import CACHE_DIR, SEC_RATE_LIMIT, SEC_USER_AGENT

logger = logging.getLogger(__name__)

# Section patterns in 10-K filings
_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("Item 1 - Business", r"item\s*1[\.\s\-\—]+business"),
    ("Item 1A - Risk Factors", r"item\s*1a[\.\s\-\—]+risk\s*factors"),
    ("Item 7 - MD&A", r"item\s*7[\.\s\-\—]+(?:management.s?\s*discussion|md\s*&\s*a)"),
    ("Item 7A - Market Risk", r"item\s*7a[\.\s\-\—]+(?:quantitative|market\s*risk)"),
    ("Item 8 - Financial Statements", r"item\s*8[\.\s\-\—]+financial\s*statements"),
]


@dataclass
class FilingSection:
    """One logical section from a 10-K filing."""

    ticker: str
    year: int
    section_name: str
    text: str
    filing_date: str = ""
    source_url: str = ""
    company_name: str = ""


def download_filing_text(
    url: str,
    ticker: str,
    year: int,
    filing_date: str = "",
    company_name: str = "",
) -> list[FilingSection]:
    """Download a 10-K filing and split into sections.

    Parameters
    ----------
    url:
        Full URL to the SEC filing HTML document.
    ticker:
        Company ticker symbol.
    year:
        Fiscal year of the filing.
    filing_date:
        Filing date string for metadata.
    company_name:
        Optional company full name.

    Returns
    -------
    list[FilingSection]
        Extracted sections with cleaned text.
    """
    cache_key = f"filing_{ticker}_{year}.txt"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        raw_text = cache_path.read_text(encoding="utf-8")
    else:
        raw_text = _fetch_and_clean(url)
        cache_path.write_text(raw_text, encoding="utf-8")

    sections = _split_into_sections(raw_text, ticker, year, filing_date, url, company_name)

    if not sections:
        # If section splitting fails, return the whole document as one chunk
        sections = [
            FilingSection(
                ticker=ticker,
                year=year,
                section_name="Full Filing",
                text=raw_text[:50000],  # Cap at 50k chars
                filing_date=filing_date,
                source_url=url,
                company_name=company_name,
            )
        ]

    logger.info("Filing %s/%d: extracted %d sections", ticker, year, len(sections))
    return sections


def _fetch_and_clean(url: str) -> str:
    """Fetch HTML from SEC and strip to plain text, preserving table row structures."""
    time.sleep(SEC_RATE_LIMIT)
    client = httpx.Client(
        headers={"User-Agent": SEC_USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
    )
    try:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text
    finally:
        client.close()

    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup(["script", "style", "meta", "link", "img", "noscript"]):
        tag.decompose()

    # Format HTML table rows cleanly so financial metric cell values stay associated
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        non_empty = [c for c in cells if c]
        if non_empty:
            row_text = " | ".join(non_empty)
            tr.replace_with(soup.new_string(f"\n{row_text}\n"))

    text = soup.get_text(separator="\n")

    # Clean up whitespace line by line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _split_into_sections(
    text: str,
    ticker: str,
    year: int,
    filing_date: str,
    source_url: str,
    company_name: str = "",
) -> list[FilingSection]:
    """Split filing text into known 10-K sections."""
    text_lower = text.lower()
    section_positions: list[tuple[str, int]] = []

    for section_name, pattern in _SECTION_PATTERNS:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            # Use the last match to avoid matching the Table of Contents index
            section_positions.append((section_name, matches[-1].start()))

    if not section_positions:
        return []

    # Sort by position in document
    section_positions.sort(key=lambda x: x[1])

    sections: list[FilingSection] = []
    for i, (name, start) in enumerate(section_positions):
        end = section_positions[i + 1][1] if i + 1 < len(section_positions) else len(text)
        section_text = text[start:end].strip()

        # Skip very short sections
        if len(section_text) < 100:
            continue

        # Cap individual sections at 100k characters
        if len(section_text) > 100_000:
            section_text = section_text[:100_000]

        sections.append(
            FilingSection(
                ticker=ticker,
                year=year,
                section_name=name,
                text=section_text,
                filing_date=filing_date,
                source_url=source_url,
                company_name=company_name,
            )
        )

    return sections
