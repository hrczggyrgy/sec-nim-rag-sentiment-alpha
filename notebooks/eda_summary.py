"""Statistical Event-Study EDA: SEC Filing FinBERT Sentiment vs. Equity Volatility Across Tech Tickers."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sec_nim_alpha.config import CONFIG
from sec_nim_alpha.data_loader import SECDataLoader
from sec_nim_alpha.features import FinancialFeaturePipeline
from sec_nim_alpha.utils import get_logger

logger = get_logger("eda_summary")


def run_cross_sectional_eda() -> pd.DataFrame:
    """Runs cross-sectional sentiment and volatility analysis across default tech tickers."""
    loader = SECDataLoader(execution_mode=CONFIG.execution_mode)
    feat_pipe = FinancialFeaturePipeline(use_finbert_model=False, execution_mode=CONFIG.execution_mode)
    records_summary = []

    for ticker in ("NVDA", "AAPL", "MSFT"):
        record = loader.fetch_latest_10k(ticker)
        sentiment = feat_pipe.compute_sentiment_features(record)
        docs = feat_pipe.build_rag_documents(record)
        records_summary.append(
            {
                "ticker": ticker,
                "filing_date": record.filing_date,
                "accession": record.accession_number,
                "num_rag_chunks": len(docs),
                "optimism_index": sentiment.optimism_index,
                "positive_ratio": sentiment.positive_ratio,
                "negative_ratio": sentiment.negative_ratio,
                "post_7d_return_pct": record.market_metrics.post_7d_return_pct if record.market_metrics else None,
                "post_7d_volatility_pct": record.market_metrics.post_7d_volatility_pct if record.market_metrics else None,
                "market_data_status": record.market_metrics.status.value if record.market_metrics else "unknown",
                "is_synthetic": record.is_synthetic_fallback,
            }
        )

    df = pd.DataFrame(records_summary)
    if not df["optimism_index"].isna().all() and not df["post_7d_return_pct"].isna().all():
        corr = df["optimism_index"].corr(df["post_7d_return_pct"])
        logger.info("Sample MOI vs. 7D Return Pearson Correlation: %.4f", corr)
    else:
        logger.info("Insufficient data for correlation (missing market data or sentiment)")

    logger.info("Cross-Sectional Summary:\n%s", df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_cross_sectional_eda()