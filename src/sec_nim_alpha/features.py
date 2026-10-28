"""Semantic document chunking and FinBERT quantitative sentiment feature engineering."""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sec_nim_alpha.config import CONFIG
from sec_nim_alpha.data_loader import FilingRecord
from sec_nim_alpha.schemas import (
    SentimentResult,
    SentenceSentiment,
    ComponentStatus,
    ExecutionMode,
)
from sec_nim_alpha.utils import get_logger

logger = get_logger(__name__)


# Loughran-McDonald financial lexicon (expanded)
LOUGHRAN_MCDONALD_POSITIVE = {
    "growth", "increased", "record", "strong", "expanded", "advantages",
    "surpassed", "accelerated", "profit", "margin", "adoption", "leadership",
    "improved", "higher", "gain", "positive", "success", "opportunity",
    "robust", "solid", "favorable", "beneficial", "outperform", "exceed",
    "strength", "momentum", "upside", "bullish", "confident", "optimistic",
}

LOUGHRAN_MCDONALD_NEGATIVE = {
    "risk", "vulnerabilities", "restrict", "adversely", "shortages", "headwinds",
    "scrutiny", "compress", "decelerates", "loss", "litigation", "uncertainties",
    "decline", "decrease", "weak", "negative", "challenge", "difficult",
    "impairment", "write-down", "downturn", "adverse", "unfavorable",
    "bearish", "pessimistic", "concern", "threat", "volatile", "pressure",
}

LOUGHRAN_MCDONALD_UNCERTAINTY = {
    "may", "could", "might", "possible", "potential", "uncertain",
    "depends", "subject to", "assuming", "estimate", "approximately",
    "preliminary", "expected", "anticipate", "believe", "intend",
    "plan", "project", "forecast", "outlook", "guidance", "target",
}


class FinancialFeaturePipeline:
    """Transforms raw SEC filing text into semantic RAG Documents and FinBERT sentiment scores."""

    def __init__(
        self,
        use_finbert_model: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.DEMO,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        self.use_finbert_model = use_finbert_model
        self.execution_mode = execution_mode
        self._finbert_pipeline = None
        self._model_version = "ProsusAI/finbert" if use_finbert_model else "loughran_mcdonald_lexicon_v1"

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
                return_all_scores=True,  # Get all class probabilities
            )
        except Exception as exc:
            logger.warning("FinBERT model load skipped (%s). Using financial lexicon scorer.", exc)
            self._finbert_pipeline = None
            self._model_version = "loughran_mcdonald_lexicon_v1"
        return self._finbert_pipeline

    def build_rag_documents(self, record: FilingRecord) -> list[Document]:
        """Splits Item 1A and Item 7 into metadata-enriched LangChain Documents."""
        sections = [
            ("Item 1A: Risk Factors", record.risk_factors_text),
            ("Item 7: MD&A", record.mda_text),
        ]
        documents: list[Document] = []
        # Use full accession hash for unique chunk IDs across filing years
        accession_suffix = hashlib.sha256(record.accession_number.encode()).hexdigest()[:12]
        for section_name, raw_text in sections:
            if not raw_text.strip():
                continue
            chunks = self.splitter.split_text(raw_text)
            section_key = section_name[:7].replace(' ', '').replace(':', '')
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "ticker": record.ticker,
                        "cik": record.cik,
                        "form_type": record.form_type,
                        "filing_date": record.filing_date,
                        "accession_number": record.accession_number,
                        "period_end": record.period_end,
                        "filed_at": record.filed_at,
                        "section": section_name,
                        "chunk_id": f"{record.ticker}_{accession_suffix}_{section_key}_{idx}",
                        "content_hash": record.content_hash,
                        "is_synthetic": record.is_synthetic_fallback,
                    },
                )
                documents.append(doc)
        logger.info("Created %d semantic chunks for %s (%s)", len(documents), record.ticker, record.filing_date)
        return documents

    @staticmethod
    def _lexicon_score_sentence(sentence: str) -> tuple[dict[str, float], str, float]:
        """Scores sentence using Loughran-McDonald lexicon, returns all class probabilities."""
        tokens = set(re.findall(r"\b[a-z]+\b", sentence.lower()))
        pos_hits = len(tokens & LOUGHRAN_MCDONALD_POSITIVE)
        neg_hits = len(tokens & LOUGHRAN_MCDONALD_NEGATIVE)
        unc_hits = len(tokens & LOUGHRAN_MCDONALD_UNCERTAINTY)

        # Simple scoring based on hits
        total_hits = pos_hits + neg_hits + unc_hits
        if total_hits == 0:
            return {"positive": 0.2, "negative": 0.2, "neutral": 0.6}, "neutral", 0.5

        pos_score = min(0.3 + 0.15 * pos_hits, 0.95)
        neg_score = min(0.3 + 0.15 * neg_hits, 0.95)
        neu_score = min(0.4 + 0.1 * unc_hits, 0.85)

        # Normalize
        total = pos_score + neg_score + neu_score
        probs = {
            "positive": pos_score / total,
            "negative": neg_score / total,
            "neutral": neu_score / total,
        }

        label = max(probs, key=probs.get)
        confidence = probs[label]
        return probs, label, confidence

    def compute_sentiment_features(self, record: FilingRecord) -> SentimentResult:
        """
        Scores executive commentary across filing sentences to compute Management Optimism Index.
        Returns full probability distributions and section coverage.
        """
        # Combine with section tags for coverage tracking
        section_sentences = []

        for section_name, text in [("MD&A", record.mda_text), ("Risk Factors", record.risk_factors_text)]:
            if not text.strip():
                continue
            # Better sentence splitting handling abbreviations
            raw_sentences = self._split_sentences(text)
            for sent in raw_sentences:
                if len(sent.strip()) > 45:
                    section_sentences.append((sent.strip(), section_name))

        # Store original counts for accurate warning
        original_mda_count = sum(1 for _, sec in section_sentences if sec == "MD&A")
        original_rf_count = sum(1 for _, sec in section_sentences if sec == "Risk Factors")
        original_total = len(section_sentences)

        if not section_sentences:
            warnings = ["No eligible sentences found in filing"]
            # Return null sentiment for no evidence (proposed contract)
            return SentimentResult(
                optimism_index=None,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=0.0,
                analyzed_sentences=0,
                section_coverage={"MD&A": 0, "Risk Factors": 0},
                backend=self._model_version,
                model_version=self._model_version,
                top_bullish_sentence=None,
                top_bearish_sentence=None,
                warnings=tuple(warnings),
            )

        # Apply max sentence limit for performance (but score all if possible)
        max_sentences = CONFIG.max_sentiment_sentences
        if len(section_sentences) > max_sentences:
            # Stratified sampling: proportional from each section
            # B15 FIX: Don't reserve minimum quota for empty sections
            mda_sents = [s for s, sec in section_sentences if sec == "MD&A"]
            rf_sents = [s for s, sec in section_sentences if sec == "Risk Factors"]
            
            # Calculate proportional quotas without minimum floor
            if original_total > 0:
                mda_quota = int(round(max_sentences * len(mda_sents) / original_total))
                rf_quota = int(round(max_sentences * len(rf_sents) / original_total))
            else:
                mda_quota = 0
                rf_quota = 0
            
            # Adjust to match max_sentences exactly
            while mda_quota + rf_quota > max_sentences:
                if mda_quota > rf_quota:
                    mda_quota -= 1
                else:
                    rf_quota -= 1
            while mda_quota + rf_quota < max_sentences:
                if len(mda_sents) > mda_quota:
                    mda_quota += 1
                elif len(rf_sents) > rf_quota:
                    rf_quota += 1
                else:
                    break
            
            # Ensure quotas don't exceed available sentences
            mda_quota = min(mda_quota, len(mda_sents))
            rf_quota = min(rf_quota, len(rf_sents))

            sampled = mda_sents[:mda_quota] + rf_sents[:rf_quota]
            section_sentences = [(s, "MD&A") for s in sampled[:mda_quota]] + \
                               [(s, "Risk Factors") for s in sampled[mda_quota:]]
            # B16 FIX: Use original counts in warning
            warnings = [f"Sampled {len(section_sentences)} of {original_total} sentences proportionally "
                       f"(MD&A: {mda_quota}/{original_mda_count}, Risk Factors: {rf_quota}/{original_rf_count})"]
        else:
            warnings = []

        sentences = [s for s, _ in section_sentences]
        sections = [sec for _, sec in section_sentences]

        clf = self._load_finbert()
        scored_items: list[SentenceSentiment] = []

        if clf is not None:
            try:
                # FinBERT returns all scores when return_all_scores=True
                all_preds = clf(sentences)
                
                # B17/B18 FIX: Validate prediction batch before accepting any results
                if len(all_preds) != len(sentences):
                    raise ValueError(f"FinBERT returned {len(all_preds)} predictions for {len(sentences)} sentences")
                
                # Validate each prediction has valid probability distributions
                required_labels = {"positive", "negative", "neutral"}
                for preds in all_preds:
                    if not isinstance(preds, list) or len(preds) == 0:
                        raise ValueError("FinBERT returned invalid prediction format")
                    
                    # Check for exact required labels, no duplicates
                    labels = [str(p.get("label", "")).lower() for p in preds]
                    if set(labels) != required_labels:
                        raise ValueError(f"FinBERT prediction missing required labels: got {set(labels)}")
                    if len(labels) != len(set(labels)):
                        raise ValueError("FinBERT prediction contains duplicate labels")
                    
                    # Check each probability is finite and in [0, 1]
                    for p in preds:
                        score = p.get("score")
                        if score is None:
                            raise ValueError("FinBERT prediction missing score")
                        if not isinstance(score, (int, float)):
                            raise ValueError(f"FinBERT score is not numeric: {score}")
                        if not (0.0 <= score <= 1.0):
                            raise ValueError(f"FinBERT probability out of bounds [0,1]: {score}")
                        if not (score == score):  # NaN check
                            raise ValueError("FinBERT probability is NaN")
                        if score == float('inf') or score == float('-inf'):
                            raise ValueError("FinBERT probability is infinite")
                    
                    total_prob = sum(p.get("score", 0) for p in preds)
                    if abs(total_prob - 1.0) > 0.01:  # Allow small floating point error
                        raise ValueError(f"FinBERT probabilities don't sum to 1: {total_prob}")
                
                # All valid - now process
                for sent, section, preds in zip(sentences, sections, all_preds):
                    probs = {str(p["label"]).lower(): p["score"] for p in preds}
                    pos_p = probs.get("positive", 0.0)
                    neg_p = probs.get("negative", 0.0)
                    neu_p = probs.get("neutral", 0.0)

                    # Find winning label
                    label = max(probs, key=probs.get)
                    confidence = probs[label]

                    scored_items.append(SentenceSentiment(
                        sentence=sent,
                        section=section,
                        char_offset=0,
                        positive_prob=pos_p,
                        negative_prob=neg_p,
                        neutral_prob=neu_p,
                        label=label,
                        confidence=confidence,
                        backend="finbert",
                        model_version=self._model_version,
                    ))
            except Exception as exc:
                logger.warning("FinBERT inference failed (%s). Falling back to lexicon for all sentences.", exc)
                clf = None
                scored_items = []  # Clear any partial results
                # Reset model version to lexicon since we're falling back
                self._model_version = "loughran_mcdonald_lexicon_v1"

        if clf is None:
            # Lexicon fallback for all sentences
            fallback_reason = "FinBERT model unavailable or validation failed"
            for sent, section in section_sentences:
                probs, label, confidence = self._lexicon_score_sentence(sent)
                scored_items.append(SentenceSentiment(
                    sentence=sent,
                    section=section,
                    char_offset=0,
                    positive_prob=probs["positive"],
                    negative_prob=probs["negative"],
                    neutral_prob=probs["neutral"],
                    label=label,
                    confidence=confidence,
                    backend="lexicon",
                    model_version="loughran_mcdonald_lexicon_v1",
                ))
            self._model_version = "loughran_mcdonald_lexicon_v1"
            warnings.append(f"FinBERT inference unavailable; using lexicon fallback: {fallback_reason}")

        # Aggregate metrics using probability-weighted approach
        total = len(scored_items)
        if total == 0:
            # Should not happen if we had section_sentences, but guard anyway
            warnings.append("No sentences scored after fallback")
            return SentimentResult(
                optimism_index=None,
                positive_ratio=0.0,
                negative_ratio=0.0,
                neutral_ratio=0.0,
                analyzed_sentences=0,
                section_coverage={},
                backend=self._model_version,
                model_version=self._model_version,
                top_bullish_sentence=None,
                top_bearish_sentence=None,
                warnings=tuple(warnings),
            )

        # Verify we have exactly one score per input sentence (transactional)
        if total != len(sentences):
            raise RuntimeError(f"Sentiment scoring produced {total} results for {len(sentences)} input sentences")

        # Probability-weighted optimism index
        weighted_sum = sum(item.positive_prob - item.negative_prob for item in scored_items)
        optimism_index = round(weighted_sum / total, 4)

        # Class ratios
        pos_count = sum(1 for item in scored_items if item.label == "positive")
        neg_count = sum(1 for item in scored_items if item.label == "negative")
        neu_count = sum(1 for item in scored_items if item.label == "neutral")

        # Section coverage
        section_coverage = {}
        for section in set(sections):
            section_coverage[section] = sum(1 for item in scored_items if item.section == section)

        # Top bullish/bearish by confidence (only if class exists)
        bullish_candidates = [item for item in scored_items if item.label == "positive"]
        bearish_candidates = [item for item in scored_items if item.label == "negative"]

        top_bullish = max(bullish_candidates, key=lambda x: x.confidence).sentence if bullish_candidates else None
        top_bearish = max(bearish_candidates, key=lambda x: x.confidence).sentence if bearish_candidates else None

        # Warn if missing classes
        if not bullish_candidates:
            warnings.append("No positive sentences detected")
        if not bearish_candidates:
            warnings.append("No negative sentences detected")

        return SentimentResult(
            optimism_index=optimism_index,
            positive_ratio=round(pos_count / total, 4),
            negative_ratio=round(neg_count / total, 4),
            neutral_ratio=round(neu_count / total, 4),
            analyzed_sentences=total,
            section_coverage=section_coverage,
            backend=self._model_version.split("/")[-1] if "/" in self._model_version else self._model_version,
            model_version=self._model_version,
            top_bullish_sentence=top_bullish,
            top_bearish_sentence=top_bearish,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences handling common financial abbreviations."""
        # Protect common abbreviations
        protected = {
            "e.g.": "e.g.<PRD>",
            "i.e.": "i.e.<PRD>",
            "U.S.": "U.S.<PRD>",
            "U.K.": "U.K.<PRD>",
            "Inc.": "Inc.<PRD>",
            "Corp.": "Corp.<PRD>",
            "Ltd.": "Ltd.<PRD>",
            "vs.": "vs.<PRD>",
            "Q1.": "Q1<PRD>",
            "Q2.": "Q2<PRD>",
            "Q3.": "Q3<PRD>",
            "Q4.": "Q4<PRD>",
            "FY23.": "FY23<PRD>",
            "FY24.": "FY24<PRD>",
            "FY25.": "FY25<PRD>",
            "FY26.": "FY26<PRD>",
        }
        for k, v in protected.items():
            text = text.replace(k, v)

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Restore abbreviations
        restored = []
        for s in sentences:
            for k, v in protected.items():
                s = s.replace(v, k)
            restored.append(s.strip())

        return [s for s in restored if s]