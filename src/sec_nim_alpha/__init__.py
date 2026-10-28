"""SEC NIM RAG Sentiment Alpha - Production-grade SEC filing analysis pipeline."""

from sec_nim_alpha.config import CONFIG
from sec_nim_alpha.schemas import (
    ExecutionMode,
    ComponentStatus,
    ExecutionManifest,
    FilingIdentifier,
    MarketMetrics,
    SectionExtraction,
    SentimentResult,
    SentenceSentiment,
    RAGResponse,
)
from sec_nim_alpha.data_loader import SECDataLoader, FilingRecord
from sec_nim_alpha.features import FinancialFeaturePipeline
from sec_nim_alpha.model import NVIDIANimRAGEngine, StableHashEmbeddings
from sec_nim_alpha.utils import get_logger, compute_lexical_groundedness, set_request_id

__version__ = "1.0.0"
__author__ = "hrczggyrgy"

__all__ = [
    "CONFIG",
    "ExecutionMode",
    "ComponentStatus",
    "ExecutionManifest",
    "FilingIdentifier",
    "MarketMetrics",
    "SectionExtraction",
    "SentimentResult",
    "SentenceSentiment",
    "RAGResponse",
    "SECDataLoader",
    "FilingRecord",
    "FinancialFeaturePipeline",
    "NVIDIANimRAGEngine",
    "StableHashEmbeddings",
    "get_logger",
    "compute_lexical_groundedness",
    "set_request_id",
]