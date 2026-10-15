"""Centralized configuration for FinSight.
Loads settings from a ``.env`` file when present and exposes constants used
across every layer (data ingestion, ML, RAG, MCP, LangGraph).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError: 
    pass

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma_db"
ML_MODELS_DIR = PROJECT_ROOT / "ml" / "models"

# Ensure directories exist at import time
for _dir in (CACHE_DIR, CHROMA_DIR, ML_MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# SEC EDGAR
SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "FinSight admin@finsight.dev")
SEC_BASE_URL: str = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_RATE_LIMIT: float = 0.12  # seconds between requests (< 10 req/s)

# NVIDIA LLM
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
NVIDIA_LLM_MODEL: str = os.getenv(
    "NVIDIA_LLM_MODEL", "meta/llama-3.1-70b-instruct"
)
NVIDIA_EMBED_MODEL: str = os.getenv(
    "NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"
)

# RAG
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_EMBED_MODEL: str = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
RAG_BM25_INDEX_PATH = CACHE_DIR / "bm25_index.pkl"
RAG_DENSE_TOP_K: int = int(os.getenv("RAG_DENSE_TOP_K", "20"))
RAG_BM25_TOP_K: int = int(os.getenv("RAG_BM25_TOP_K", "20"))
RAG_RRF_K: int = int(os.getenv("RAG_RRF_K", "60"))

# ML
RISK_MODEL_PATH = ML_MODELS_DIR / "risk_model.joblib"
FORECAST_LOOKBACK_YEARS: int = int(os.getenv("FORECAST_LOOKBACK_YEARS", "5"))

# ---------------------------------------------------------------------------
# Default CSV (backward compatibility)
# ---------------------------------------------------------------------------
DEFAULT_DATA_FILE = DATA_DIR / "financial_data.csv"
