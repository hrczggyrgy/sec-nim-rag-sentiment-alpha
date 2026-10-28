"""Centralized configuration, environment settings, and model hyperparameters."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from sec_nim_alpha.schemas import ExecutionMode

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma_db"
RAW_FILINGS_DIR = DATA_DIR / "raw_filings"


def ensure_data_dirs() -> None:
    """Create data directories on demand (not at import time)."""
    for directory in (DATA_DIR, CACHE_DIR, CHROMA_DIR, RAW_FILINGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and endpoint identifiers for NVIDIA NIM and FinBERT."""

    # Execution mode controls fallback behavior and data provenance
    execution_mode: ExecutionMode = field(
        default_factory=lambda: ExecutionMode(os.getenv("EXECUTION_MODE", "demo"))
    )

    # API credentials
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv(
            "SEC_USER_AGENT", "QuantResearchBot contact@hrczggyrgy-portfolio.io"
        )
    )

    # NVIDIA NIM model endpoints (use currently available models)
    nim_llm_model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    nim_embed_model: str = "nvidia/nemotron-3.5-embed-8b"
    nim_rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

    # Local fallback models
    finbert_model: str = "ProsusAI/finbert"
    local_embed_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking and retrieval parameters
    chunk_size: int = 1200
    chunk_overlap: int = 180
    top_k_initial: int = 30
    top_k_rerank: int = 8
    top_k_final: int = 5
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    max_sentiment_sentences: int = 200

    # Default tickers for demo
    default_tickers: tuple[str, ...] = ("NVDA", "AAPL", "MSFT", "TSLA", "AMZN")

    # Cache and storage
    cache_ttl_hours: int = 168  # 1 week
    chroma_persist: bool = True

    # Network timeouts and retries
    nim_timeout_seconds: int = 30
    nim_max_retries: int = 3
    nim_retry_backoff: float = 1.5

    # SEC rate limiting
    sec_requests_per_second: float = 5.0


CONFIG = ModelConfig()