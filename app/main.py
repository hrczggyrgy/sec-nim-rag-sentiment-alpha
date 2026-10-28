"""Interactive Streamlit Dashboard for SEC 10-K NVIDIA NIM RAG & FinBERT Volatility Analytics."""

import streamlit as st
from sec_nim_alpha.config import CONFIG, ExecutionMode
from sec_nim_alpha.data_loader import SECDataLoader
from sec_nim_alpha.features import FinancialFeaturePipeline
from sec_nim_alpha.model import NVIDIANimRAGEngine
from sec_nim_alpha.utils import (
    create_sentiment_distribution_chart,
    create_sentiment_volatility_figure,
    create_execution_manifest_chart,
    set_request_id,
    compute_claim_level_support,
)

st.set_page_config(
    page_title="SEC Filing NIM RAG & FinBERT Alpha Terminal",
    page_icon="📈",
    layout="wide",
)

st.title("📈 SEC 10-K NVIDIA NIM RAG & FinBERT Volatility Terminal")

# Dynamic caption showing actual config
st.caption(
    f"Execution Mode: **{CONFIG.execution_mode.value.upper()}** | "
    f"Embedding: `{CONFIG.nim_embed_model}` | "
    f"LLM: `{CONFIG.nim_llm_model}` | "
    f"Reranker: `{CONFIG.nim_rerank_model}` | "
    f"FinBERT: `{CONFIG.finbert_model}`"
)

with st.sidebar:
    st.header("⚙️ Control & MLOps Configuration")

    # Execution mode selector
    execution_mode = st.selectbox(
        "Execution Mode",
        options=[ExecutionMode.DEMO, ExecutionMode.RESEARCH, ExecutionMode.PRODUCTION],
        index=0,
        format_func=lambda x: x.value.title(),
        help="Demo: synthetic fallbacks allowed. Research: no synthetic data. Production: strict failures.",
    )

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
def run_full_pipeline(
    ticker: str,
    api_key: str,
    enable_finbert: bool,
    refresh: bool,
    execution_mode: ExecutionMode,
):
    # Set request ID for tracing
    request_id = set_request_id()

    loader = SECDataLoader(execution_mode=execution_mode)
    record = loader.fetch_latest_10k(ticker, force_refresh=refresh)
    panel_df = loader.load_historical_panel(ticker)

    feat_pipe = FinancialFeaturePipeline(use_finbert_model=enable_finbert, execution_mode=execution_mode)
    docs = feat_pipe.build_rag_documents(record)
    sentiment = feat_pipe.compute_sentiment_features(record)

    # Update panel with current filing's actual data (not synthetic)
    if record.market_metrics:
        panel_df.loc[panel_df.index[-1] if len(panel_df) > 0 else 0, "optimism_index"] = sentiment.optimism_index
        panel_df.loc[panel_df.index[-1] if len(panel_df) > 0 else 0, "post_7d_volatility_pct"] = record.market_metrics.post_7d_volatility_pct
        panel_df.loc[panel_df.index[-1] if len(panel_df) > 0 else 0, "post_7d_return_pct"] = record.market_metrics.post_7d_return_pct

    rag_engine = NVIDIANimRAGEngine(api_key=api_key, execution_mode=execution_mode)
    rag_engine.index_documents(
        docs,
        ticker=record.ticker,
        accession=record.accession_number,
    )
    return record, sentiment, panel_df, rag_engine, request_id


# Run pipeline
record, sentiment, panel_df, rag_engine, request_id = run_full_pipeline(
    active_ticker, api_key_input, use_live_finbert, force_edgar_refresh or run_btn, execution_mode
)

# --- Header Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Filing Form & Date",
    f"{record.form_type} ({record.filing_date})",
    f"Accession: {record.accession_number[:18]}",
)
col2.metric(
    "Management Optimism Index (MOI)",
    f"{sentiment.optimism_index:+.3f}",
    f"{sentiment.positive_ratio * 100:.1f}% Bullish | {sentiment.analyzed_sentences} sentences",
    help=f"Backend: {sentiment.backend} | Coverage: {sentiment.section_coverage}",
)
col3.metric(
    "7-Day Post-Filing Volatility",
    f"{record.market_metrics.post_7d_volatility_pct:.2f}%" if record.market_metrics and record.market_metrics.post_7d_volatility_pct else "N/A",
    f"{record.market_metrics.post_7d_return_pct:+.2f}% 7D Return" if record.market_metrics and record.market_metrics.post_7d_return_pct else "N/A",
    help=f"Status: {record.market_metrics.status.value if record.market_metrics else 'unknown'} | Days: {record.market_metrics.trading_days_observed if record.market_metrics else 0}/7",
)
engine_label = "NVIDIA NIM" if rag_engine.nim_enabled else "Offline Fallback"
col4.metric("Active RAG Engine", engine_label, f"{len(rag_engine.documents)} Chunks")

# Show synthetic warning
if record.is_synthetic_fallback:
    st.warning("⚠️ **DEMO MODE**: This filing uses curated synthetic data. Not for research use.")

# Market data status
if record.market_metrics and record.market_metrics.status != "success":
    status_color = "🟡" if record.market_metrics.status == "degraded" else "🔴"
    st.warning(f"{status_color} Market Data: {record.market_metrics.status.value} - {record.market_metrics.error_message}")

st.divider()

# --- Charts ---
left_col, right_col = st.columns([3, 2])
with left_col:
    if len(panel_df) > 0:
        st.plotly_chart(create_sentiment_volatility_figure(panel_df), use_container_width=True)
        if panel_df["optimism_index"].isna().all():
            st.info("ℹ️ Historical MOI not available (requires historical sentiment computation)")
    else:
        st.info("No historical filings available for this ticker in research mode.")

with right_col:
    st.plotly_chart(create_sentiment_distribution_chart(sentiment.to_dict()), use_container_width=True)
    st.markdown("**🟢 Top Bullish Executive Statement:**")
    st.info(sentiment.top_bullish_sentence if sentiment.top_bullish_sentence else "No bullish sentences detected")
    st.markdown("**🔴 Top Risk / Bearish Statement:**")
    st.warning(sentiment.top_bearish_sentence if sentiment.top_bearish_sentence else "No bearish sentences detected")

# Show sentiment warnings
for warn in sentiment.warnings:
    st.caption(f"⚠️ {warn}")

st.divider()

# --- RAG Query Interface ---
st.subheader(f"💬 Interrogate {active_ticker}'s 10-K Filing (Two-Stage NVIDIA NIM RAG)")

preset_questions = [
    f"What are the primary supply chain and geopolitical risks reported by {active_ticker}?",
    f"What drove {active_ticker}'s revenue growth and gross margin performance in MD&A?",
    f"How do capital expenditures and AI infrastructure investments impact {active_ticker}?",
]
selected_q = st.selectbox("Select an Institutional Analyst Prompt:", preset_questions)
user_query = st.text_input("Or ask a custom question:", value=selected_q)

if st.button("🔍 Retrieve & Synthesize Cited Answer", type="primary"):
    response = rag_engine.answer_question(
        user_query,
        sentiment_summary=sentiment.to_dict(),
        filing_accession=record.accession_number,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stage 1+2 Retrieval Latency", f"{response.retrieval_latency_ms:.1f} ms")
    m2.metric("LLM Synthesis Latency", f"{response.generation_latency_ms:.1f} ms")
    m3.metric("Lexical Overlap Score", f"{response.groundedness_score * 100:.1f}%")
    m4.metric("Generation Mode", response.engine_mode.split("(")[0].strip())

    st.markdown(response.answer)

    # Claim-level support
    from src.utils import compute_claim_level_support
    claim_support = compute_claim_level_support(response.answer, [c["snippet"] for c in response.citations])
    if claim_support["total_claims"] > 0:
        st.caption(
            f"📊 Numerical Claims: {claim_support['supported_claims']}/{claim_support['total_claims']} supported "
            f"({claim_support['claim_support_rate']*100:.0f}%)"
        )
        if claim_support["unsupported_details"]:
            st.caption(f"⚠️ Unsupported: {', '.join(claim_support['unsupported_details'])}")

    with st.expander("📑 View Reranked SEC 10-K Source Passages & Citations", expanded=True):
        for idx, cite in enumerate(response.citations, 1):
            st.markdown(
                f"**{idx}. [{cite['section']} | `{cite['chunk_id']}`]** "
                f"*(Filing Date: {cite['filing_date']} | Accession: {cite['accession']})*\n\n"
                f"> {cite['snippet']}"
            )

    # Execution Manifest
    with st.expander("🔧 Execution Manifest (MLOps Observability)", expanded=False):
        if response.execution_manifest:
            manifest_dict = response.execution_manifest.to_dict()
            st.json(manifest_dict)
            st.plotly_chart(create_execution_manifest_chart(manifest_dict), use_container_width=True)
        else:
            st.info("Execution manifest not available")

    st.caption(f"Request ID: {request_id}")