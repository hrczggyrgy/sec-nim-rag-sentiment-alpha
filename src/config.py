"""Centralized configuration, environment settings, and model hyperparameters."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma_db"

for directory in (DATA_DIR, CACHE_DIR, CHROMA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and endpoint identifiers for NVIDIA NIM and FinBERT."""

    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv(
            "SEC_USER_AGENT", "QuantResearchBot contact@hrczggyrgy-portfolio.io"
        )
    )
    nim_llm_model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    nim_embed_model: str = "nvidia/nemotron-3.5-embed-8b"
    nim_rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    finbert_model: str = "ProsusAI/finbert"
    local_embed_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 1200
    chunk_overlap: int = 180
    top_k_initial: int = 10
    top_k_rerank: int = 4
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    max_sentiment_chunks: int = 40
    default_tickers: tuple[str, ...] = ("NVDA", "AAPL", "MSFT", "TSLA", "AMZN")


CONFIG = ModelConfig()