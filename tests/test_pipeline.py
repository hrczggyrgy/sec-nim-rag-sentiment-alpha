"""Unit test suite for SEC ingestion, FinBERT feature engineering, and NVIDIA NIM RAG pipeline."""

import pytest
from src.data_loader import FilingRecord, SECDataLoader
from src.features import FinancialFeaturePipeline
from src.model import NVIDIANimRAGEngine
from src.utils import compute_lexical_groundedness


@pytest.fixture
def sample_record() -> FilingRecord:
    loader = SECDataLoader()
    return loader.fetch_latest_10k("NVDA")


def test_sec_data_loader_returns_valid_record(sample_record: FilingRecord) -> None:
    assert sample_record.ticker == "NVDA"
    assert sample_record.form_type == "10-K"
    # Live EDGAR data may have empty MD&A; require at least one section to have content
    assert len(sample_record.risk_factors_text) > 100 or len(sample_record.mda_text) > 100
    assert isinstance(sample_record.post_7d_volatility_pct, float)


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