"""Shared data schemas and execution manifests for the SEC NIM RAG Alpha pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import hashlib
import json


class ExecutionMode(Enum):
    """Execution policy modes controlling data provenance and fallback behavior."""

    DEMO = "demo"
    RESEARCH = "research"
    PRODUCTION = "production"


class ComponentStatus(Enum):
    """Per-component execution status for the execution manifest."""

    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    SKIPPED = "skipped"
    OFFLINE_FALLBACK = "offline_fallback"


@dataclass(frozen=True)
class ExecutionManifest:
    """Immutable record of pipeline execution for reproducibility and audit."""

    data_mode: ExecutionMode
    filing_accession: Optional[str]
    index_version: str
    embedding_model: str
    embedding_status: ComponentStatus
    reranker_model: str
    reranker_status: ComponentStatus
    generator_model: str
    generator_status: ComponentStatus
    sentiment_backend: str
    sentiment_status: ComponentStatus
    market_data_status: ComponentStatus
    warnings: tuple[str, ...] = field(default_factory=tuple)
    execution_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_mode": self.data_mode.value,
            "filing_accession": self.filing_accession,
            "index_version": self.index_version,
            "embedding_model": self.embedding_model,
            "embedding_status": self.embedding_status.value,
            "reranker_model": self.reranker_model,
            "reranker_status": self.reranker_status.value,
            "generator_model": self.generator_model,
            "generator_status": self.generator_status.value,
            "sentiment_backend": self.sentiment_backend,
            "sentiment_status": self.sentiment_status.value,
            "market_data_status": self.market_data_status.value,
            "warnings": list(self.warnings),
            "execution_timestamp": self.execution_timestamp,
        }


@dataclass(frozen=True)
class FilingIdentifier:
    """Immutable SEC filing identifier with full provenance."""

    cik: str
    accession: str
    form: str
    period_end: str
    filed_at: str
    accepted_at: Optional[str] = None
    public_disseminated_at: Optional[str] = None
    ingestion_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    source_url: Optional[str] = None
    content_hash: Optional[str] = None
    parser_version: str = "1.0.0"

    def stable_id(self) -> str:
        """Generate a stable, versioned identifier for this filing."""
        core = f"{self.cik}|{self.accession}|{self.form}|{self.period_end}|{self.filed_at}|{self.parser_version}"
        return hashlib.sha256(core.encode()).hexdigest()[:16]

    def cache_key(self) -> str:
        """Generate cache key including accession for isolation."""
        return f"{self.cik}_{self.accession}_{self.form}"


@dataclass
class MarketMetrics:
    """Market metrics with explicit provenance and completeness flags."""

    ticker: str
    filing_date: str
    post_7d_return_pct: Optional[float]
    post_7d_volatility_pct: Optional[float]
    trading_days_observed: int
    required_trading_days: int = 7
    status: ComponentStatus = ComponentStatus.SUCCESS
    error_message: Optional[str] = None
    data_source: str = "yfinance"
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @property
    def is_complete(self) -> bool:
        return self.trading_days_observed >= self.required_trading_days

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "filing_date": self.filing_date,
            "post_7d_return_pct": self.post_7d_return_pct,
            "post_7d_volatility_pct": self.post_7d_volatility_pct,
            "trading_days_observed": self.trading_days_observed,
            "required_trading_days": self.required_trading_days,
            "status": self.status.value,
            "error_message": self.error_message,
            "data_source": self.data_source,
            "fetched_at": self.fetched_at,
            "is_complete": self.is_complete,
        }


@dataclass
class SectionExtraction:
    """Result of section-aware extraction with status tracking."""

    section_name: str
    content: str
    char_start: int
    char_end: int
    extraction_status: ComponentStatus
    parser_version: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return self.extraction_status == ComponentStatus.SUCCESS and len(self.content.strip()) > 200


@dataclass
class SentenceSentiment:
    """Per-sentence sentiment with full probability distribution."""

    sentence: str
    section: str
    char_offset: int
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    label: str
    confidence: float
    backend: str
    model_version: str


@dataclass
class SentimentResult:
    """Aggregated sentiment with section coverage and backend provenance."""

    optimism_index: Optional[float]
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    analyzed_sentences: int
    eligible_sentences: int = 0
    section_coverage: dict[str, int] = field(default_factory=dict)
    section_sentiment: Optional[dict[str, dict[str, float]]] = None
    backend: str = ""
    model_version: str = ""
    top_bullish_sentence: Optional[str] = None
    top_bearish_sentence: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimism_index": self.optimism_index,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "neutral_ratio": self.neutral_ratio,
            "analyzed_sentences": self.analyzed_sentences,
            "eligible_sentences": self.eligible_sentences,
            "section_coverage": self.section_coverage,
            "section_sentiment": self.section_sentiment,
            "backend": self.backend,
            "model_version": self.model_version,
            "top_bullish_sentence": self.top_bullish_sentence,
            "top_bearish_sentence": self.top_bearish_sentence,
            "warnings": list(self.warnings),
        }


@dataclass
class RAGResponse:
    """Structured RAG response with full provenance."""

    query: str
    answer: str
    citations: list[dict[str, Any]]
    retrieval_latency_ms: float
    generation_latency_ms: float
    groundedness_score: float
    engine_mode: str
    execution_manifest: Optional[ExecutionManifest] = None
    retrieval_count: int = 0
    rerank_count: int = 0
    insufficient_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "groundedness_score": self.groundedness_score,
            "engine_mode": self.engine_mode,
            "execution_manifest": self.execution_manifest.to_dict() if self.execution_manifest else None,
            "retrieval_count": self.retrieval_count,
            "rerank_count": self.rerank_count,
            "insufficient_evidence": self.insufficient_evidence,
        }


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication and integrity."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]