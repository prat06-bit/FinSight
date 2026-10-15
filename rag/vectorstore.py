"""ChromaDB-backed dense vector store for financial documents.

Stores pre-computed embeddings with rich metadata and supports
filtered queries by ticker, filing year, filing type, and section.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from config import CHROMA_DIR

logger = logging.getLogger(__name__)


class FinSightVectorStore:
    """ChromaDB-backed vector store for financial documents."""

    def __init__(self, collection_name: str = "finsight_filings") -> None:
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- write --------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Upsert documents with pre-computed embeddings."""
        if not ids:
            return
        
        # Ensure metadata types are normalized (int filing_year, upper ticker)
        normalized_metadatas = []
        for m in metadatas:
            norm_m = dict(m)
            if "ticker" in norm_m and norm_m["ticker"]:
                norm_m["ticker"] = str(norm_m["ticker"]).upper()
            if "filing_year" in norm_m and norm_m["filing_year"]:
                try:
                    norm_m["filing_year"] = int(norm_m["filing_year"])
                except (ValueError, TypeError):
                    pass
            normalized_metadatas.append(norm_m)

        self._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=normalized_metadatas,
            ids=ids,
        )

    # -- read ---------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        ticker: str | None = None,
        filing_year: int | str | None = None,
        filing_type: str | None = None,
        section: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense retrieval with optional metadata filtering."""
        where_clauses: list[dict] = []
        if ticker:
            where_clauses.append({"ticker": str(ticker).upper()})
        if filing_year is not None:
            try:
                where_clauses.append({"filing_year": int(filing_year)})
            except (ValueError, TypeError):
                where_clauses.append({"filing_year": filing_year})
        if filing_type:
            where_clauses.append({"filing_type": str(filing_type)})
        if section:
            where_clauses.append({"section": str(section)})

        where_filter: dict | None = None
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        elif len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}

        count = self.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                docs.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
        return docs

    # -- utility ------------------------------------------------------------

    def has_documents(self, ticker: str) -> bool:
        """Check if any documents exist for a ticker."""
        try:
            results = self._collection.get(where={"ticker": str(ticker).upper()}, limit=1)
            return bool(results["ids"])
        except Exception:
            return False

    def count(self) -> int:
        return self._collection.count()

    def delete_ticker(self, ticker: str) -> None:
        """Delete all documents for a ticker."""
        self._collection.delete(where={"ticker": str(ticker).upper()})

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for all stored chunks (useful for BM25 rebuild)."""
        results = self._collection.get(include=["documents", "metadatas"])
        out: list[dict[str, Any]] = []
        for i, doc_id in enumerate(results["ids"]):
            out.append(
                {
                    "id": doc_id,
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )
        return out

    def reset(self) -> None:
        """Delete the collection and recreate it (for testing)."""
        col_name = self._collection.name
        try:
            self._client.delete_collection(col_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )
