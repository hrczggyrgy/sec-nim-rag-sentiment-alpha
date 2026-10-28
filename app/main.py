"""Interactive Streamlit Dashboard for SEC 10-K NVIDIA NIM RAG & FinBERT Volatility Analytics."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.config import CONFIG
from src.data_loader import SECDataLoader
from src.features import FinancialFeaturePipeline
from src.model import NVIDIANimRAGEngine
from src.utils import (
    create_sentiment_distribution_chart,
    create_sentiment_volatility_figure,
)

st.set_page_config(
    page_title="SEC Filing NIM RAG & FinBERT Alpha Terminal",
    page_icon="📈",
    layout="wide",
)

st.title("📈 SEC 10-K NVIDIA NIM RAG & FinBERT Volatility Terminal")
st.caption(
    "Production MLOps Pipeline: Automated SEC EDGAR Ingestion (`edgartools`) • "
    "Two-Stage NVIDIA NIM RAG (`nv-embedqa-e5-v5` + `NVIDIARerank` + `Llama-3.3-70B`) • "
    "Quantitative Tone Alignment (`ProsusAI/finbert` + `yfinance`)"
)

with st.sidebar:
    st.header("⚙️ Control & MLOps Configuration")
    selected_ticker = st.selectbox("Select Equity Ticker", list(CONFIG.default_tickers), index=0)
    custom_ticker = st.text_input("Or Enter Custom Ticker (e.g., AMD, META)", value="").strip().upper()
    active_ticker = custom_ticker if custom_ticker else selected_ticker

    api_key_input = st.text_input(
        "NVIDIA NIM API Key (nvapi-...)",
        value=CONFIG.nvidia_api_key,
        type="password",
        help="Get a free API key at https://build.nvidia.com. Runs in deterministic local fallback mode if blank.",
    )
    use_live_finbert = st.checkbox("Enable HuggingFace FinBERT Weights", value=False)
    force_edgar_refresh = st.checkbox("Force Live SEC EDGAR Refresh", value=False)
    run_btn = st.button("🚀 Run SEC Ingestion & Indexing", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Executing SEC EDGAR Ingestion, FinBERT Scoring, and Vector Indexing...")
def run_full_pipeline(ticker: str, api_key: str, enable_finbert: bool, refresh: bool):
    loader = SECDataLoader()
    record = loader.fetch_latest_10k(ticker, force_refresh=refresh)
    panel_df = loader.load_historical_panel(ticker)

    feat_pipe = FinancialFeaturePipeline(use_finbert_model=enable_finbert)
    docs = feat_pipe.build_rag_documents(record)
    sentiment = feat_pipe.compute_sentiment_features(record)

    panel_df.loc[panel_df.index[-1], "optimism_index"] = sentiment.optimism_index
    panel_df.loc[panel_df.index[-1], "post_7d_volatility_pct"] = record.post_7d_volatility_pct
    panel_df.loc[panel_df.index[-1], "post_7d_return_pct"] = record.post_7d_return_pct

    rag_engine = NVIDIANimRAGEngine(api_key=api_key)
    rag_engine.index_documents(docs, collection_name=f"sec_{ticker.lower()}")
    return record, sentiment, panel_df, rag_engine


record, sentiment, panel_df, rag_engine = run_full_pipeline(
    active_ticker, api_key_input, use_live_finbert, force_edgar_refresh or run_btn
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filing Form & Date", f"{record.form_type} ({record.filing_date})", record.accession_number[:14])
col2.metric(
    "Management Optimism Index (MOI)",
    f"{sentiment.optimism_index:+.3f}",
    f"{sentiment.positive_ratio * 100:.1f}% Bullish Sentences",
)
col3.metric(
    "7-Day Post-Filing Volatility",
    f"{record.post_7d_volatility_pct:.2f}%",
    f"{record.post_7d_return_pct:+.2f}% 7D Return",
)
col4.metric("Active RAG Engine", "NVIDIA NIM" if rag_engine.nim_enabled else "Offline Fallback", f"{len(rag_engine.documents)} Chunks")

st.divider()

left_col, right_col = st.columns([3, 2])
with left_col:
    st.plotly_chart(create_sentiment_volatility_figure(panel_df), use_container_width=True)
with right_col:
    st.plotly_chart(create_sentiment_distribution_chart(sentiment.to_dict()), use_container_width=True)
    st.markdown("**🟢 Top Bullish Executive Statement:**")
    st.info(sentiment.top_bullish_sentence)
    st.markdown("**🔴 Top Risk / Bearish Statement:**")
    st.warning(sentiment.top_bearish_sentence)

st.divider()
st.subheader(f"💬 Interrogate {active_ticker}'s 10-K Filing (Two-Stage NVIDIA NIM RAG)")

preset_questions = [
    f"What are the primary supply chain and geopolitical risks reported by {active_ticker}?",
    f"What drove {active_ticker}'s revenue growth and gross margin performance in MD&A?",
    f"How do capital expenditures and AI infrastructure investments impact {active_ticker}?",
]
selected_q = st.selectbox("Select an Institutional Analyst Prompt:", preset_questions)
user_query = st.text_input("Or ask a custom question:", value=selected_q)

if st.button("🔍 Retrieve & Synthesize Cited Answer", type="primary"):
    response = rag_engine.answer_question(user_query, sentiment_summary=sentiment.to_dict())

    m1, m2, m3 = st.columns(3)
    m1.metric("Stage 1+2 Retrieval Latency", f"{response.retrieval_latency_ms:.1f} ms")
    m2.metric("LLM Synthesis Latency", f"{response.generation_latency_ms:.1f} ms")
    m3.metric("Lexical Groundedness Score", f"{response.groundedness_score * 100:.1f}%")

    st.markdown(response.answer)

    with st.expander("📑 View Reranked SEC 10-K Source Passages & Citations", expanded=True):
        for idx, cite in enumerate(response.citations, 1):
            st.markdown(
                f"**{idx}. [{cite['section']} | `{cite['chunk_id']}`]** *(Filing Date: {cite['filing_date']})*\n\n"
                f"> {cite['snippet']}"
            )