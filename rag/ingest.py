"""Ingest SEC 10-K filings into both dense (ChromaDB) and sparse (BM25) indexes.

Pipeline:
1. Resolve filing URLs via ``SECEdgarSource.get_filing_urls()``
2. Download and extract text sections via ``download_filing_text()``
3. Chunk each section with ``RecursiveCharacterTextSplitter``
4. Generate contextual header-prefixed dense embeddings locally via sentence-transformers
5. Store canonical chunks in ChromaDB (dense) and rebuild BM25 index (sparse)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE
from data.sources.sec_edgar import SECEdgarSource
from data.sources.sec_filing_text import download_filing_text
from rag.bm25 import BM25Index
from rag.embeddings import embed_texts
from rag.models import RetrievalDocument
from rag.vectorstore import FinSightVectorStore

logger = logging.getLogger(__name__)

# Tickers targeted for 10-company evaluation corpus benchmark
RAG_TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "INTC", "CRM", "ORCL",
]


def ingest_filings(
    ticker: str,
    num_filings: int = 3,
    form_type: str = "10-K",
) -> int:
    """Fetch recent 10-K filings, chunk, embed, and index.

    Returns the number of chunks ingested.
    """
    ticker = ticker.upper()
    logger.info("Ingesting %s filings for %s (limit=%d)", form_type, ticker, num_filings)

    source = SECEdgarSource()
    store = FinSightVectorStore()

    # 1. Resolve filing URLs
    try:
        filing_list: list[dict[str, str]] = source.get_filing_urls(
            ticker, form_type=form_type, limit=num_filings,
        )
    except Exception as exc:
        logger.error("Failed to get filing URLs for %s: %s", ticker, exc)
        return 0

    if not filing_list:
        logger.warning("No %s filings found for %s", form_type, ticker)
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )

    all_display_texts: list[str] = []
    all_context_texts: list[str] = []
    all_metadatas: list[dict[str, Any]] = []
    all_ids: list[str] = []

    company_name = source.get_company_name(ticker) if hasattr(source, "get_company_name") else ticker

    for filing_info in filing_list:
        url = filing_info.get("url", "")
        filing_date = filing_info.get("date", "")
        accession = filing_info.get("accession", "")
        form = filing_info.get("form", form_type)

        # Derive fiscal year from filing date
        try:
            filing_year = int(filing_date[:4]) if len(filing_date) >= 4 else 0
        except (ValueError, TypeError):
            filing_year = 0

        # 2. Download and split into sections
        try:
            sections = download_filing_text(
                url=url,
                ticker=ticker,
                year=filing_year,
                filing_date=filing_date,
                company_name=company_name,
            )
        except Exception as exc:
            logger.error("Failed to download %s filing %s: %s", ticker, url, exc)
            continue

        # 3. Chunk each section
        for sec in sections:
            if not sec.text.strip():
                continue

            chunks = splitter.split_text(sec.text)
            if not chunks:
                continue

            for chunk_idx, chunk_text in enumerate(chunks):
                # Hash input metadata and text snippet for deterministic, unique chunk ID
                id_string = f"{ticker}-{filing_year}-{sec.section_name}-{chunk_idx}-{chunk_text[:30]}"
                chunk_id = hashlib.sha256(id_string.encode("utf-8")).hexdigest()[:16]

                doc = RetrievalDocument(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    ticker=ticker,
                    company_name=company_name,
                    filing_year=filing_year,
                    filing_date=filing_date,
                    filing_type=form,
                    accession_number=accession,
                    section=sec.section_name,
                    item_number="",
                    source_url=url,
                )

                all_display_texts.append(chunk_text)
                all_context_texts.append(doc.contextual_embedding_text())
                all_metadatas.append(doc.to_metadata_dict())
                all_ids.append(chunk_id)

    if not all_display_texts:
        logger.warning("No text extracted for %s", ticker)
        return 0

    # 4. Generate dense embeddings using contextual headers for metadata-grounded search
    logger.info("Embedding %d chunks for %s ...", len(all_display_texts), ticker)
    all_embeddings = embed_texts(all_context_texts)

    # 5. Store in ChromaDB (batch size 500)
    batch_size = 500
    for start in range(0, len(all_display_texts), batch_size):
        end = start + batch_size
        store.add_documents(
            texts=all_display_texts[start:end],
            embeddings=all_embeddings[start:end],
            metadatas=all_metadatas[start:end],
            ids=all_ids[start:end],
        )

    logger.info("Ingested %d chunks for %s into ChromaDB", len(all_display_texts), ticker)
    return len(all_display_texts)


def build_bm25_index() -> BM25Index:
    """Rebuild the BM25 index from all documents stored in ChromaDB."""
    store = FinSightVectorStore()
    all_docs = store.get_all_metadata()
    logger.info("Building BM25 index from %d stored chunks", len(all_docs))

    index = BM25Index()
    index.build(all_docs)
    index.save()
    return index


def ingest_all_tickers(
    tickers: list[str] | None = None,
    num_filings: int = 3,
) -> dict[str, int]:
    """Ingest filings for multiple tickers, then build BM25 index.

    Returns ``{ticker: chunk_count}``.
    """
    tickers = tickers or RAG_TICKERS
    results: dict[str, int] = {}

    for ticker in tickers:
        count = ingest_filings(ticker, num_filings=num_filings)
        results[ticker] = count

    # Build the sparse BM25 index over all ingested documents
    build_bm25_index()

    total = sum(results.values())
    logger.info("Total ingestion: %d chunks across %d tickers", total, len(tickers))
    return results
