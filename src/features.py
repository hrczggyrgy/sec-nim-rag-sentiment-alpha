"""Semantic document chunking and FinBERT quantitative sentiment feature engineering."""

import re
from dataclasses import dataclass
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CONFIG
from src.data_loader import FilingRecord
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Container for FinBERT quantitative tone metrics."""

    optimism_index: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    analyzed_chunks: int
    top_bullish_sentence: str
    top_bearish_sentence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimism_index": self.optimism_index,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "neutral_ratio": self.neutral_ratio,
            "analyzed_chunks": self.analyzed_chunks,
            "top_bullish_sentence": self.top_bullish_sentence,
            "top_bearish_sentence": self.top_bearish_sentence,
        }


class FinancialFeaturePipeline:
    """Transforms raw SEC filing text into semantic RAG Documents and FinBERT sentiment scores."""

    def __init__(self, use_finbert_model: bool = True) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        self.use_finbert_model = use_finbert_model
        self._finbert_pipeline = None

    def _load_finbert(self):
        """Lazy-loads Hugging Face ProsusAI/finbert pipeline with graceful fallback."""
        if self._finbert_pipeline is not None or not self.use_finbert_model:
            return self._finbert_pipeline
        try:
            from transformers import pipeline

            logger.info("Loading Hugging Face model: %s", CONFIG.finbert_model)
            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=CONFIG.finbert_model,
                truncation=True,
                max_length=512,
            )
        except Exception as exc:
            logger.warning("FinBERT model load skipped (%s). Using financial lexicon scorer.", exc)
            self._finbert_pipeline = None
        return self._finbert_pipeline

    def build_rag_documents(self, record: FilingRecord) -> list[Document]:
        """Splits Item 1A and Item 7 into metadata-enriched LangChain Documents."""
        sections = [
            ("Item 1A: Risk Factors", record.risk_factors_text),
            ("Item 7: MD&A", record.mda_text),
        ]
        documents: list[Document] = []
        for section_name, raw_text in sections:
            if not raw_text.strip():
                continue
            chunks = self.splitter.split_text(raw_text)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "ticker": record.ticker,
                        "form_type": record.form_type,
                        "filing_date": record.filing_date,
                        "accession_number": record.accession_number,
                        "section": section_name,
                        "chunk_id": f"{record.ticker}_{section_name[:7].replace(' ', '')}_{idx}",
                    },
                )
                documents.append(doc)
        logger.info("Created %d semantic chunks for %s (%s)", len(documents), record.ticker, record.filing_date)
        return documents

    @staticmethod
    def _lexicon_fallback_score(sentence: str) -> tuple[str, float]:
        """Deterministic Loughran-McDonald financial lexicon fallback when running offline."""
        pos_words = {
            "growth", "increased", "record", "strong", "expanded", "advantages",
            "surpassed", "accelerated", "profit", "margin", "adoption", "leadership",
        }
        neg_words = {
            "risk", "vulnerabilities", "restrict", "adversely", "shortages", "headwinds",
            "scrutiny", "compress", "decelerates", "loss", "litigation", "uncertainties",
        }
        tokens = set(re.findall(r"\b[a-z]+\b", sentence.lower()))
        pos_hits = len(tokens & pos_words)
        neg_hits = len(tokens & neg_words)
        if pos_hits > neg_hits:
            return "positive", min(0.65 + 0.1 * pos_hits, 0.98)
        if neg_hits > pos_hits:
            return "negative", min(0.65 + 0.1 * neg_hits, 0.98)
        return "neutral", 0.75

    def compute_sentiment_features(self, record: FilingRecord) -> SentimentResult:
        """Scores executive commentary across filing sentences to compute Management Optimism Index."""
        combined_text = f"{record.mda_text}\n{record.risk_factors_text}"
        raw_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", combined_text)
            if len(s.strip()) > 45
        ]
        sample_sentences = raw_sentences[: CONFIG.max_sentiment_chunks]
        if not sample_sentences:
            sample_sentences = ["Company reported stable operational performance."]

        clf = self._load_finbert()
        scored_items: list[tuple[str, str, float]] = []

        if clf is not None:
            preds = clf(sample_sentences)
            for sent, pred in zip(sample_sentences, preds):
                scored_items.append((sent, str(pred["label"]).lower(), float(pred["score"])))
        else:
            for sent in sample_sentences:
                label, score = self._lexicon_fallback_score(sent)
                scored_items.append((sent, label, score))

        total = len(scored_items)
        pos_sum = sum(score for _, label, score in scored_items if label == "positive")
        neg_sum = sum(score for _, label, score in scored_items if label == "negative")
        neu_sum = sum(score for _, label, score in scored_items if label == "neutral")

        pos_count = sum(1 for _, label, _ in scored_items if label == "positive")
        neg_count = sum(1 for _, label, _ in scored_items if label == "negative")
        neu_count = sum(1 for _, label, _ in scored_items if label == "neutral")

        optimism_index = round((pos_sum - neg_sum) / total, 4)

        bullish_candidates = [x for x in scored_items if x[1] == "positive"]
        bearish_candidates = [x for x in scored_items if x[1] == "negative"]

        top_bullish = (
            max(bullish_candidates, key=lambda x: x[2])[0]
            if bullish_candidates
            else sample_sentences[0]
        )
        top_bearish = (
            max(bearish_candidates, key=lambda x: x[2])[0]
            if bearish_candidates
            else sample_sentences[-1]
        )

        return SentimentResult(
            optimism_index=optimism_index,
            positive_ratio=round(pos_count / total, 4),
            negative_ratio=round(neg_count / total, 4),
            neutral_ratio=round(neu_count / total, 4),
            analyzed_chunks=total,
            top_bullish_sentence=top_bullish,
            top_bearish_sentence=top_bearish,
        )