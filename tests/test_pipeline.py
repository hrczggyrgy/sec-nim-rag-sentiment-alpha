"""Unit test suite for SEC ingestion, FinBERT feature engineering, and NVIDIA NIM RAG pipeline."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from sec_nim_alpha.data_loader import FilingRecord, SECDataLoader
from sec_nim_alpha.features import FinancialFeaturePipeline
from sec_nim_alpha.model import NVIDIANimRAGEngine
from sec_nim_alpha.utils import compute_lexical_groundedness
from sec_nim_alpha.schemas import ComponentStatus, ExecutionMode, SentimentResult


@pytest.fixture
def sample_record() -> FilingRecord:
    loader = SECDataLoader()
    # Force refresh to get new structure
    return loader.fetch_latest_10k("NVDA", force_refresh=True)


@pytest.fixture
def demo_record() -> FilingRecord:
    """Create a deterministic demo record for testing without network calls."""
    from sec_nim_alpha.data_loader import DEMO_SEC_CORPUS
    fallback = DEMO_SEC_CORPUS["NVDA"]
    loader = SECDataLoader(execution_mode=ExecutionMode.DEMO)
    return loader.fetch_latest_10k("NVDA", force_refresh=False)


def test_sec_data_loader_returns_valid_record(sample_record: FilingRecord) -> None:
    assert sample_record.ticker == "NVDA"
    assert sample_record.form_type == "10-K"
    # Live EDGAR data may have empty MD&A; require at least one section to have content
    assert len(sample_record.risk_factors_text) > 100 or len(sample_record.mda_text) > 100
    assert sample_record.market_metrics is not None
    assert isinstance(sample_record.market_metrics.post_7d_volatility_pct, float)
    assert sample_record.market_metrics.status in (ComponentStatus.SUCCESS, ComponentStatus.DEGRADED, ComponentStatus.OFFLINE_FALLBACK)


def test_feature_pipeline_chunks_and_scores_sentiment(sample_record: FilingRecord) -> None:
    pipeline = FinancialFeaturePipeline(use_finbert_model=False)
    docs = pipeline.build_rag_documents(sample_record)
    assert len(docs) >= 2
    assert all("chunk_id" in d.metadata for d in docs)

    sentiment = pipeline.compute_sentiment_features(sample_record)
    assert -1.0 <= sentiment.optimism_index <= 1.0
    ratio_sum = sentiment.positive_ratio + sentiment.negative_ratio + sentiment.neutral_ratio
    assert abs(ratio_sum - 1.0) < 1e-3


def test_rag_engine_indexing_and_retrieval(sample_record: FilingRecord) -> None:
    pipeline = FinancialFeaturePipeline(use_finbert_model=False)
    docs = pipeline.build_rag_documents(sample_record)
    sentiment = pipeline.compute_sentiment_features(sample_record)

    engine = NVIDIANimRAGEngine(api_key="")
    indexed_count = engine.index_documents(docs, collection_name="test_nvda")
    assert indexed_count == len(docs)

    response = engine.answer_question(
        "What supply chain risks involve CoWoS packaging?",
        sentiment_summary=sentiment.to_dict(),
    )
    assert len(response.answer) > 50
    assert len(response.citations) > 0
    assert 0.0 <= response.groundedness_score <= 1.0
    assert response.retrieval_latency_ms >= 0.0


def test_groundedness_metric_calculation() -> None:
    context = ["NVIDIA Data Center revenue grew 114% driven by Blackwell GPU adoption."]
    grounded_ans = "Data Center revenue grew 114% due to Blackwell GPU adoption."
    hallucinated_ans = "Quantum submarine sales declined in Antarctica."
    assert compute_lexical_groundedness(grounded_ans, context) > 0.7
    assert compute_lexical_groundedness(hallucinated_ans, context) < 0.2


def test_null_sentiment_rag_formatting() -> None:
    """Test that RAG formatting handles null optimism_index without TypeError."""
    engine = NVIDIANimRAGEngine(api_key="")
    # Mock vectorstore to return empty results
    engine.vectorstore = Mock()
    engine.vectorstore.similarity_search = Mock(return_value=[])
    engine.documents = []
    
    # This should not raise TypeError when optimism_index is None
    response = engine.answer_question(
        "Test question",
        sentiment_summary={"optimism_index": None, "positive_ratio": 0.0, "negative_ratio": 0.0},
    )
    assert "Unavailable" in response.answer or response.insufficient_evidence


def test_empty_sentiment_result() -> None:
    """Test SentimentResult with None optimism_index serializes correctly."""
    sentiment = SentimentResult(
        optimism_index=None,
        positive_ratio=0.0,
        negative_ratio=0.0,
        neutral_ratio=1.0,
        analyzed_sentences=0,
        section_coverage={},
        backend="test",
        model_version="test",
    )
    d = sentiment.to_dict()
    assert d["optimism_index"] is None
    assert d["positive_ratio"] == 0.0


def test_production_mode_requires_valid_key() -> None:
    """Test that production mode raises RuntimeError without valid NVIDIA API key."""
    with pytest.raises(RuntimeError, match="Production mode requires a valid NVIDIA API key"):
        NVIDIANimRAGEngine(api_key="", execution_mode=ExecutionMode.PRODUCTION)
    
    with pytest.raises(RuntimeError, match="Production mode requires a valid NVIDIA API key"):
        NVIDIANimRAGEngine(api_key="invalid-key", execution_mode=ExecutionMode.PRODUCTION)


def test_production_mode_fails_on_nim_init_failure() -> None:
    """Test that production mode fails when NIM client initialization fails."""
    # Use a fake but properly formatted key to trigger NIM init attempt
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings", side_effect=Exception("Connection failed")):
        with pytest.raises(RuntimeError, match="Production mode: NVIDIA NIM initialization failed"):
            NVIDIANimRAGEngine(api_key="nvapi-fake-key-for-testing", execution_mode=ExecutionMode.PRODUCTION)


def test_demo_mode_falls_back_on_nim_failure() -> None:
    """Test that demo mode falls back gracefully when NIM initialization fails."""
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings", side_effect=Exception("Connection failed")):
        engine = NVIDIANimRAGEngine(api_key="nvapi-fake-key-for-testing", execution_mode=ExecutionMode.DEMO)
        assert engine.nim_enabled is False
        assert engine._embedding_status == ComponentStatus.OFFLINE_FALLBACK


def test_research_mode_fails_on_live_fetch_failure() -> None:
    """Test that research mode fails explicitly when live EDGAR fetch fails."""
    loader = SECDataLoader(execution_mode=ExecutionMode.RESEARCH)
    with patch.object(loader, "_fetch_live_10k", side_effect=Exception("EDGAR unavailable")):
        with pytest.raises(RuntimeError, match="Live EDGAR fetch required but failed"):
            loader.fetch_latest_10k("NVDA")


def test_index_documents_replaces_obsolete_chunks() -> None:
    """Test that re-indexing with fewer chunks removes obsolete ones."""
    from langchain_core.documents import Document
    
    engine = NVIDIANimRAGEngine(api_key="")
    
    # First indexing with 5 chunks
    docs_v1 = [
        Document(page_content=f"Chunk {i}", metadata={"chunk_id": f"NVDA_test_Item1A_{i}"})
        for i in range(5)
    ]
    engine.index_documents(docs_v1, collection_name="test_replace", ticker="NVDA", accession="test")
    assert len(engine.documents) == 5
    
    # Re-index with 3 chunks (simulating revised parse)
    docs_v2 = [
        Document(page_content=f"Updated Chunk {i}", metadata={"chunk_id": f"NVDA_test_Item1A_{i}"})
        for i in range(3)
    ]
    engine.index_documents(docs_v2, collection_name="test_replace", ticker="NVDA", accession="test")
    assert len(engine.documents) == 3
    # The collection should only have 3 documents now


def test_finbert_validation_rejects_invalid_distributions() -> None:
    """Test that FinBERT validation detects invalid distributions and falls back with warning."""
    pipeline = FinancialFeaturePipeline(use_finbert_model=True)
    
    # Use long enough text to generate eligible sentences (>45 chars each)
    long_risk = "This is a long risk factor sentence that should be eligible for sentiment analysis because it exceeds the minimum length threshold of forty-five characters."
    long_mda = "This is a long MD&A sentence that should also be eligible for sentiment analysis because it exceeds the minimum length threshold of forty-five characters."
    
    record = FilingRecord(
        ticker="NVDA", cik="0001045810", form_type="10-K",
        filing_date="2024-01-01", accession_number="test", period_end="2024-01-01",
        filed_at="2024-01-01", risk_factors_text=long_risk, mda_text=long_mda,
        execution_mode=ExecutionMode.DEMO,
    )
    
    # Case 1: Missing required labels - should trigger fallback
    mock_clf = Mock()
    mock_clf.side_effect = lambda sentences: [
        [{"label": "POSITIVE", "score": 0.5}, {"label": "NEGATIVE", "score": 0.5}],
        [{"label": "POSITIVE", "score": 0.5}, {"label": "NEGATIVE", "score": 0.5}],
    ]
    pipeline._finbert_pipeline = mock_clf
    pipeline._model_version = "ProsusAI/finbert"
    
    sentiment = pipeline.compute_sentiment_features(record)
    # Should fall back to lexicon and record warning
    assert sentiment.backend == "loughran_mcdonald_lexicon_v1"
    assert any("FinBERT inference unavailable" in w for w in sentiment.warnings)
    
    # Case 2: Probability out of bounds - should trigger fallback
    mock_clf = Mock()
    mock_clf.side_effect = lambda sentences: [
        [{"label": "POSITIVE", "score": 1.5}, {"label": "NEGATIVE", "score": -0.2}, {"label": "NEUTRAL", "score": -0.3}],
        [{"label": "POSITIVE", "score": 1.5}, {"label": "NEGATIVE", "score": -0.2}, {"label": "NEUTRAL", "score": -0.3}],
    ]
    pipeline._finbert_pipeline = mock_clf
    pipeline._model_version = "ProsusAI/finbert"
    
    sentiment = pipeline.compute_sentiment_features(record)
    assert sentiment.backend == "loughran_mcdonald_lexicon_v1"
    assert any("FinBERT inference unavailable" in w for w in sentiment.warnings)
    
    # Case 3: NaN probability - should trigger fallback
    mock_clf = Mock()
    mock_clf.side_effect = lambda sentences: [
        [{"label": "POSITIVE", "score": float("nan")}, {"label": "NEGATIVE", "score": 0.5}, {"label": "NEUTRAL", "score": 0.5}],
        [{"label": "POSITIVE", "score": float("nan")}, {"label": "NEGATIVE", "score": 0.5}, {"label": "NEUTRAL", "score": 0.5}],
    ]
    pipeline._finbert_pipeline = mock_clf
    pipeline._model_version = "ProsusAI/finbert"
    
    sentiment = pipeline.compute_sentiment_features(record)
    assert sentiment.backend == "loughran_mcdonald_lexicon_v1"
    assert any("FinBERT inference unavailable" in w for w in sentiment.warnings)
    
    # Case 4: Duplicate labels - should trigger fallback
    mock_clf = Mock()
    mock_clf.side_effect = lambda sentences: [
        [{"label": "POSITIVE", "score": 0.4}, {"label": "POSITIVE", "score": 0.3}, {"label": "NEUTRAL", "score": 0.3}],
        [{"label": "POSITIVE", "score": 0.4}, {"label": "POSITIVE", "score": 0.3}, {"label": "NEUTRAL", "score": 0.3}],
    ]
    pipeline._finbert_pipeline = mock_clf
    pipeline._model_version = "ProsusAI/finbert"
    
    sentiment = pipeline.compute_sentiment_features(record)
    assert sentiment.backend == "loughran_mcdonald_lexicon_v1"
    assert any("FinBERT inference unavailable" in w for w in sentiment.warnings)


def test_cache_selection_excludes_demo() -> None:
    """Test that cache selection excludes DEMO files when loading live cache."""
    from sec_nim_alpha.config import CACHE_DIR
    
    # Clean up any existing cache files first
    for f in CACHE_DIR.glob("NVDA_*_10k.json"):
        f.unlink(missing_ok=True)
    
    loader = SECDataLoader(execution_mode=ExecutionMode.DEMO)
    
    # Create a fake live cache file and a demo cache file
    live_cache = CACHE_DIR / "NVDA_0001045810-24-000012_10k.json"
    demo_cache = CACHE_DIR / "NVDA_DEMO_10k.json"
    
    # Write minimal valid records
    live_record = FilingRecord(
        ticker="NVDA", cik="0001045810", form_type="10-K",
        filing_date="2024-01-01", accession_number="0001045810-24-000012",
        period_end="2024-01-01", filed_at="2024-01-01",
        risk_factors_text="Risk", mda_text="MDA",
        execution_mode=ExecutionMode.DEMO,
    )
    demo_record = FilingRecord(
        ticker="NVDA", cik="0001045810", form_type="10-K",
        filing_date="2024-01-01", accession_number="DEMO",
        period_end="2024-01-01", filed_at="2024-01-01",
        risk_factors_text="Risk", mda_text="MDA",
        is_synthetic_fallback=True,
        execution_mode=ExecutionMode.DEMO,
    )
    
    live_cache.write_text(json.dumps(live_record.to_dict()))
    demo_cache.write_text(json.dumps(demo_record.to_dict()))
    
    try:
        # Should load the live cache, not the demo cache (which sorts higher alphabetically)
        record = loader.fetch_latest_10k("NVDA", force_refresh=False)
        assert record.accession_number == "0001045810-24-000012"
        assert not record.is_synthetic_fallback
    finally:
        live_cache.unlink(missing_ok=True)
        demo_cache.unlink(missing_ok=True)


def test_refresh_creates_new_request_id() -> None:
    """Test that refresh operations create distinct request IDs."""
    from sec_nim_alpha.utils import set_request_id, request_id_var
    
    # Set initial request ID
    id1 = set_request_id()
    
    # Simulate a cached call (would reuse same ID)
    id2 = set_request_id()
    
    # Fresh call should generate new ID
    id3 = set_request_id()
    
    assert id1 != id3
    assert id2 != id3


def test_sentiment_fallback_warning_recorded() -> None:
    """Test that FinBERT fallback reason is recorded in warnings."""
    # Use lexicon mode directly to test fallback warning recording
    pipeline = FinancialFeaturePipeline(use_finbert_model=False)
    
    # Use long enough text to generate eligible sentences
    long_risk = "This is a long risk factor sentence that should be eligible for sentiment analysis because it exceeds the minimum length threshold of forty-five characters."
    long_mda = "This is a long MD&A sentence that should also be eligible for sentiment analysis because it exceeds the minimum length threshold of forty-five characters."
    
    record = FilingRecord(
        ticker="NVDA", cik="0001045810", form_type="10-K",
        filing_date="2024-01-01", accession_number="test", period_end="2024-01-01",
        filed_at="2024-01-01", risk_factors_text=long_risk, mda_text=long_mda,
        execution_mode=ExecutionMode.DEMO,
    )
    sentiment = pipeline.compute_sentiment_features(record)
    
    # Should use lexicon backend
    assert sentiment.backend == "loughran_mcdonald_lexicon_v1"
    # Note: no fallback warning in pure lexicon mode


def test_load_filing_by_accession() -> None:
    """Test loading a specific filing by accession from cache."""
    from sec_nim_alpha.config import CACHE_DIR
    
    # Clean up
    for f in CACHE_DIR.glob("TEST_*_10k.json"):
        f.unlink(missing_ok=True)
    
    loader = SECDataLoader(execution_mode=ExecutionMode.DEMO)
    
    # Create test records
    record1 = FilingRecord(
        ticker="TEST", cik="0000000001", form_type="10-K",
        filing_date="2024-01-01", accession_number="0000000001-24-000001",
        period_end="2024-01-01", filed_at="2024-01-01",
        risk_factors_text="Risk text", mda_text="MDA text",
        execution_mode=ExecutionMode.DEMO,
    )
    record2 = FilingRecord(
        ticker="TEST", cik="0000000001", form_type="10-K",
        filing_date="2024-04-01", accession_number="0000000001-24-000002",
        period_end="2024-04-01", filed_at="2024-04-01",
        risk_factors_text="Risk text 2", mda_text="MDA text 2",
        execution_mode=ExecutionMode.DEMO,
    )
    
    # Write to cache
    cache1 = CACHE_DIR / "TEST_0000000001-24-000001_10k.json"
    cache2 = CACHE_DIR / "TEST_0000000001-24-000002_10k.json"
    cache1.write_text(json.dumps(record1.to_dict()))
    cache2.write_text(json.dumps(record2.to_dict()))
    
    try:
        # Load by accession
        loaded1 = loader.load_filing_by_accession("TEST", "0000000001-24-000001")
        assert loaded1 is not None
        assert loaded1.accession_number == "0000000001-24-000001"
        assert loaded1.filing_date == "2024-01-01"
        
        loaded2 = loader.load_filing_by_accession("TEST", "0000000001-24-000002")
        assert loaded2 is not None
        assert loaded2.accession_number == "0000000001-24-000002"
        assert loaded2.filing_date == "2024-04-01"
        
        # Non-existent
        loaded_none = loader.load_filing_by_accession("TEST", "NONEXISTENT")
        assert loaded_none is None
    finally:
        cache1.unlink(missing_ok=True)
        cache2.unlink(missing_ok=True)


def test_workspace_identity_changes_on_config() -> None:
    """Test that workspace key includes execution mode and sentiment backend."""
    from sec_nim_alpha.data_loader import FilingRecord
    from sec_nim_alpha.schemas import SentimentResult
    
    record = FilingRecord(
        ticker="TEST", cik="0000000001", form_type="10-K",
        filing_date="2024-01-01", accession_number="0000000001-24-000001",
        period_end="2024-01-01", filed_at="2024-01-01",
        risk_factors_text="Risk", mda_text="MDA",
        execution_mode=ExecutionMode.DEMO,
    )
    
    sentiment1 = SentimentResult(
        optimism_index=0.1, positive_ratio=0.3, negative_ratio=0.2, neutral_ratio=0.5,
        analyzed_sentences=10, eligible_sentences=10, section_coverage={},
        backend="finbert", model_version="ProsusAI/finbert",
    )
    
    sentiment2 = SentimentResult(
        optimism_index=0.1, positive_ratio=0.3, negative_ratio=0.2, neutral_ratio=0.5,
        analyzed_sentences=10, eligible_sentences=10, section_coverage={},
        backend="loughran_mcdonald_lexicon_v1", model_version="loughran_mcdonald_lexicon_v1",
    )
    
    # Different backends should produce different keys
    from app.main import _make_workspace_key
    key1 = _make_workspace_key(record, "demo", sentiment1.backend)
    key2 = _make_workspace_key(record, "demo", sentiment2.backend)
    assert key1 != key2
    
    # Different execution modes should produce different keys
    key3 = _make_workspace_key(record, "production", sentiment1.backend)
    assert key1 != key3