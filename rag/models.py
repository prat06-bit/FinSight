"""Data models for RAG retrieval results and canonical documents.

Every retrieval document/result carries full provenance metadata so that
downstream consumers (evaluation, CLI, future LLM/MCP) can cite, filter, and trace evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalDocument:
    """Canonical representation of one retrieval chunk in the system."""

    chunk_id: str = ""
    text: str = ""
    ticker: str = ""
    company_name: str = ""
    filing_year: int = 0
    filing_date: str = ""
    filing_type: str = "10-K"
    accession_number: str = ""
    section: str = ""
    item_number: str = ""
    source_url: str = ""

    def contextual_embedding_text(self) -> str:
        """Return header-prefixed text for dense embedding similarity."""
        header_parts = []
        if self.company_name:
            header_parts.append(f"Company: {self.company_name}")
        if self.ticker:
            header_parts.append(f"Ticker: {self.ticker}")
        if self.filing_year or self.filing_type:
            header_parts.append(f"Filing: {self.filing_year} {self.filing_type}".strip())
        if self.section:
            header_parts.append(f"Section: {self.section}")
        
        header = " | ".join(header_parts)
        if header:
            return f"[{header}]\n\n{self.text}"
        return self.text

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return standardized metadata dictionary for storage engines."""
        return {
            "chunk_id": str(self.chunk_id),
            "ticker": str(self.ticker).upper(),
            "company_name": str(self.company_name),
            "filing_year": int(self.filing_year),
            "filing_date": str(self.filing_date),
            "filing_type": str(self.filing_type),
            "accession_number": str(self.accession_number),
            "section": str(self.section),
            "item_number": str(self.item_number),
            "source_url": str(self.source_url),
        }


@dataclass
class RetrievalResult:
    """One retrieved chunk with ranking score and provenance metadata."""

    text: str = ""
    score: float = 0.0
    chunk_id: str = ""
    ticker: str = ""
    company_name: str = ""
    filing_year: int = 0
    filing_type: str = "10-K"
    section: str = ""
    item_number: str = ""
    accession_number: str = ""
    source_url: str = ""
    filing_date: str = ""
    retrieval_method: str = "hybrid"  # "dense", "bm25", or "hybrid"

    def citation(self) -> str:
        """Return a human-readable citation string."""
        parts = [self.ticker or "UNKNOWN", str(self.filing_year) if self.filing_year else "N/A", self.filing_type]
        if self.section:
            parts.append(self.section)
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation."""
        return {
            "text": self.text,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_year": self.filing_year,
            "filing_type": self.filing_type,
            "section": self.section,
            "item_number": self.item_number,
            "accession_number": self.accession_number,
            "source_url": self.source_url,
            "filing_date": self.filing_date,
            "retrieval_method": self.retrieval_method,
        }

    @classmethod
    def from_doc(cls, doc: RetrievalDocument, score: float = 0.0, method: str = "hybrid") -> RetrievalResult:
        """Construct a RetrievalResult from a canonical RetrievalDocument."""
        return cls(
            text=doc.text,
            score=score,
            chunk_id=doc.chunk_id,
            ticker=doc.ticker,
            company_name=doc.company_name,
            filing_year=doc.filing_year,
            filing_type=doc.filing_type,
            section=doc.section,
            item_number=doc.item_number,
            accession_number=doc.accession_number,
            source_url=doc.source_url,
            filing_date=doc.filing_date,
            retrieval_method=method,
        )
