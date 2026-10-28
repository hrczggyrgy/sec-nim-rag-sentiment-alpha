"""Interactive Streamlit Dashboard for SEC 10-K NVIDIA NIM RAG & FinBERT Volatility Analytics."""

import pandas as pd
import streamlit as st
from sec_nim_alpha.config import CONFIG, ExecutionMode
from sec_nim_alpha.data_loader import SECDataLoader
from sec_nim_alpha.features import FinancialFeaturePipeline
from sec_nim_alpha.model import NVIDIANimRAGEngine
from sec_nim_alpha.utils import (
    create_sentiment_distribution_chart,
    create_sentiment_volatility_figure,
    create_execution_manifest_table,
    set_request_id,
    compute_claim_level_support,
    request_id_var,
)
from sec_nim_alpha.schemas import ComponentStatus

st.set_page_config(
    page_title="SEC Filing NIM RAG & FinBERT Alpha Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 SEC 10-K NIM RAG & FinBERT Volatility Terminal")

# --- Session State Initialization ---
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_filing_key" not in st.session_state:
    st.session_state.last_filing_key = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "selected_filing_accession" not in st.session_state:
    st.session_state.selected_filing_accession = None

# --- Sidebar: Configuration Form ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Use a form so changes don't trigger immediate reruns
    with st.form("config_form", border=False):
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
        
        # One-shot refresh button (not a checkbox)
        refresh_clicked = st.form_submit_button("🔄 Refresh Filing from SEC EDGAR", type="secondary", use_container_width=True)
        
        # Primary run button
        run_submitted = st.form_submit_button("🚀 Run Ingestion & Indexing", type="primary", use_container_width=True)

# --- Execution Status Strip (shows ACTUAL execution state, not config) ---
def render_status_strip(record, rag_engine, sentiment, market_metrics):
    """Render persistent status strip showing actual execution state."""
    # Determine filing source
    if record.is_synthetic_fallback:
        filing_src = "SYNTHETIC DEMO"
        filing_color = "🟠"
    else:
        filing_src = "LIVE EDGAR"
        filing_color = "🟢"
    
    # Tone backend
    tone_backend = sentiment.backend
    if "finbert" in tone_backend.lower():
        tone_display = "FINBERT"
        tone_color = "🟢"
    elif "lexicon" in tone_backend.lower() or "loughran" in tone_backend.lower():
        tone_display = "LEXICON"
        tone_color = "🔵"
    else:
        tone_display = tone_backend.upper()
        tone_color = "⚪"
    
    # Retrieval backend
    if rag_engine.nim_enabled:
        retrieval_display = "NIM EMBEDDING"
        retrieval_color = "🟢"
    else:
        retrieval_display = "HASH FALLBACK"
        retrieval_color = "🟣"
    
    # Generation backend
    gen_mode = rag_engine.engine_mode
    if "extractive" in gen_mode.lower() or "fallback" in gen_mode.lower():
        gen_display = "EXTRACTIVE"
        gen_color = "🟣"
    elif "nim" in gen_mode.lower():
        gen_display = "NIM LLM"
        gen_color = "🟢"
    else:
        gen_display = gen_mode.upper()[:12]
        gen_color = "⚪"
    
    # Market data
    if market_metrics and market_metrics.status == ComponentStatus.SUCCESS:
        price_display = "COMPLETE"
        price_color = "🟢"
    elif market_metrics and market_metrics.status == ComponentStatus.DEGRADED:
        price_display = "PARTIAL"
        price_color = "🟡"
    else:
        price_display = "UNAVAILABLE"
        price_color = "🔴"
    
    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E; 
            padding: 0.75rem 1rem; 
            border-radius: 6px; 
            margin-bottom: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            border: 1px solid #333;
        ">
            <strong>Filing:</strong> {filing_color} {filing_src} &nbsp; | &nbsp;
            <strong>Tone:</strong> {tone_color} {tone_display} &nbsp; | &nbsp;
            <strong>Retrieval:</strong> {retrieval_color} {retrieval_display} &nbsp; | &nbsp;
            <strong>Generation:</strong> {gen_color} {gen_display} &nbsp; | &nbsp;
            <strong>Prices:</strong> {price_color} {price_display}
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Pipeline Execution ---
def run_full_pipeline(
    ticker: str,
    api_key: str,
    enable_finbert: bool,
    refresh: bool,
    execution_mode: ExecutionMode,
    request_id: str,
):
    loader = SECDataLoader(execution_mode=execution_mode)
    record = loader.fetch_latest_10k(ticker, force_refresh=refresh)
    panel_df = loader.load_historical_panel(ticker)

    feat_pipe = FinancialFeaturePipeline(use_finbert_model=enable_finbert, execution_mode=execution_mode)
    docs = feat_pipe.build_rag_documents(record)
    sentiment = feat_pipe.compute_sentiment_features(record)

    # Update panel with current filing's actual data (not synthetic)
    if record.market_metrics:
        mask = panel_df["accession"] == record.accession_number
        new_row = {
            "ticker": record.ticker,
            "filing_date": record.filing_date,
            "quarter": f"Q{(int(record.filing_date[5:7]) - 1) // 3 + 1}",
            "optimism_index": sentiment.optimism_index,
            "post_7d_volatility_pct": record.market_metrics.post_7d_volatility_pct,
            "post_7d_return_pct": record.market_metrics.post_7d_return_pct,
            "accession": record.accession_number,
        }
        if mask.any():
            panel_df.loc[mask, list(new_row.keys())] = list(new_row.values())
        else:
            panel_df = pd.concat([panel_df, pd.DataFrame([new_row])], ignore_index=True)

    rag_engine = NVIDIANimRAGEngine(api_key=api_key, execution_mode=execution_mode)
    rag_engine.index_documents(
        docs,
        ticker=record.ticker,
        accession=record.accession_number,
    )
    return record, sentiment, panel_df, rag_engine


@st.cache_resource(show_spinner="Executing SEC EDGAR Ingestion, FinBERT Scoring, and Vector Indexing...")
def run_full_pipeline_cached(
    ticker: str,
    api_key: str,
    enable_finbert: bool,
    execution_mode: ExecutionMode,
):
    """Cached version without refresh - for non-refresh runs only."""
    request_id = set_request_id()
    return run_full_pipeline(ticker, api_key, enable_finbert, False, execution_mode, request_id)


# Execute pipeline on button press or refresh
pipeline_error = None
try:
    if refresh_clicked or run_submitted:
        # Bypass cache for refresh - create new request ID for each refresh
        request_id = set_request_id()
        record, sentiment, panel_df, rag_engine = run_full_pipeline(
            active_ticker, api_key_input, use_live_finbert, True, execution_mode, request_id
        )
    else:
        # Use cached version
        record, sentiment, panel_df, rag_engine = run_full_pipeline_cached(
            active_ticker, api_key_input, use_live_finbert, execution_mode
        )
        request_id = request_id_var.get() or set_request_id()
except Exception as e:
    pipeline_error = e
    st.error(f"Pipeline execution failed: {e}")
    st.stop()

# --- Filing Key for Session Tracking ---
current_filing_key = f"{record.ticker}_{record.accession_number}"

# Clear stale analysis when filing changes
if st.session_state.last_filing_key != current_filing_key:
    st.session_state.last_response = None
    st.session_state.last_query = None
    st.session_state.last_filing_key = current_filing_key
    st.session_state.selected_filing_accession = record.accession_number

# --- Status Strip (Actual Execution State) ---
render_status_strip(record, rag_engine, sentiment, record.market_metrics)

# --- Technical Details Expander (Configured Models) ---
with st.expander("🔧 Technical Details: Configured Models & Endpoints", expanded=False):
    st.markdown(f"""
    **Execution Mode:** `{execution_mode.value}` (selected) | `{CONFIG.execution_mode.value}` (env default)
    
    **Embedding:** `{CONFIG.nim_embed_model}`
    
    **LLM:** `{CONFIG.nim_llm_model}`
    
    **Reranker:** `{CONFIG.nim_rerank_model}`
    
    **FinBERT:** `{CONFIG.finbert_model}` {'(enabled)' if use_live_finbert else '(disabled - using lexicon)'}
    
    **Data Root:** `{CONFIG.data_root if hasattr(CONFIG, 'data_root') else 'default'}`
    """)

# --- Synthetic/Market Data Warnings ---
if record.is_synthetic_fallback:
    st.warning("⚠️ **DEMO MODE**: This filing uses curated synthetic data. Not for research use.")

if record.market_metrics and record.market_metrics.status != ComponentStatus.SUCCESS:
    status_color = "🟡" if record.market_metrics.status == ComponentStatus.DEGRADED else "🔴"
    st.warning(f"{status_color} Market Data: {record.market_metrics.status.value} - {record.market_metrics.error_message}")

st.divider()

# --- Header Metrics (Fixed Delta Usage) ---
col1, col2, col3, col4 = st.columns(4)

# Metric 1: Filing Identity (no delta - it's an identifier)
col1.metric(
    "Filing",
    f"{record.form_type} {record.filing_date}",
    help=f"Accession: {record.accession_number} | CIK: {record.cik}",
)

# Metric 2: MOI (delta only if historical comparison exists)
moi_delta = None
moi_delta_help = f"Backend: {sentiment.backend} | Analyzed: {sentiment.analyzed_sentences} sentences"
if len(panel_df) > 1 and panel_df["optimism_index"].notna().sum() > 1:
    # Compare with prior comparable filing (same quarter previous year if possible)
    prior_moi = panel_df[panel_df["accession"] != record.accession_number]["optimism_index"].dropna()
    if len(prior_moi) > 0:
        moi_delta = sentiment.optimism_index - prior_moi.iloc[-1]
        moi_delta_help += f" | Change vs prior: {moi_delta:+.3f}"

col2.metric(
    "Management Optimism Index (MOI)",
    f"{sentiment.optimism_index:+.3f}" if sentiment.optimism_index is not None else "Unavailable",
    delta=f"{moi_delta:+.3f}" if moi_delta is not None else None,
    help=moi_delta_help,
)

# Metric 3: Volatility & Return (separate metrics, not delta)
col3.metric(
    "Annualized Realized Volatility (7D)",
    f"{record.market_metrics.post_7d_volatility_pct:.2f}%" if record.market_metrics and record.market_metrics.post_7d_volatility_pct is not None else "Unavailable",
    help=f"Based on {record.market_metrics.trading_days_observed if record.market_metrics else 0}/7 trading days post-filing | Status: {record.market_metrics.status.value if record.market_metrics else 'unknown'}",
)

col4.metric(
    "7-Day Return",
    f"{record.market_metrics.post_7d_return_pct:+.2f}%" if record.market_metrics and record.market_metrics.post_7d_return_pct is not None else "Unavailable",
    help=f"Raw 7-day price return (not annualized) | Trading days: {record.market_metrics.trading_days_observed if record.market_metrics else 0}/7",
)

st.divider()

# --- Charts ---
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📊 Filing Tone & Volatility History")
    if len(panel_df) > 0:
        # Use aligned panels with actual backend
        fig = create_sentiment_volatility_figure(
            panel_df,
            backend=sentiment.backend,
            show_dual_axis=False,
        )
        
        # Add chart selection for filing drill-down
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="tone_vol_chart",
        )
        
        # Handle chart point selection
        if selection and selection.get("selection", {}).get("points"):
            point = selection["selection"]["points"][0]
            if "customdata" in point and point["customdata"]:
                selected_accession = point["customdata"]
                if selected_accession != st.session_state.selected_filing_accession:
                    st.session_state.selected_filing_accession = selected_accession
                    st.rerun()
        
        # Coverage summary
        moi_avail = panel_df["optimism_index"].notna().sum()
        vol_avail = panel_df["post_7d_volatility_pct"].notna().sum()
        total = len(panel_df)
        st.caption(f"📋 Coverage: Tone {moi_avail}/{total} filings | Volatility {vol_avail}/{total} filings")
        
        # Data table expander
        with st.expander("📋 View Underlying Data", expanded=False):
            display_df = panel_df.copy()
            display_df = display_df.sort_values("filing_date", ascending=False)
            st.dataframe(
                display_df[["filing_date", "quarter", "accession", "optimism_index", "post_7d_volatility_pct", "post_7d_return_pct"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No historical filings available for this ticker in research mode.")

with right_col:
    st.subheader("📈 Sentence Tone Distribution")
    
    # Build section breakdown if available
    section_breakdown = None
    if hasattr(sentiment, 'section_sentiment') and sentiment.section_sentiment:
        section_breakdown = sentiment.section_sentiment
    
    dist_fig = create_sentiment_distribution_chart(
        sentiment.to_dict(),
        backend=sentiment.backend,
        section_breakdown=section_breakdown,
    )
    st.plotly_chart(dist_fig, use_container_width=True, key="sentiment_dist_chart")
    
    # Top sentences with section badges
    st.markdown("**🟢 Top Positive Statement:**")
    if sentiment.top_bullish_sentence:
        st.success(sentiment.top_bullish_sentence)
    else:
        st.caption("No positive sentences detected")
    
    st.markdown("**🔴 Top Risk / Negative Statement:**")
    if sentiment.top_bearish_sentence:
        st.error(sentiment.top_bearish_sentence)
    else:
        st.caption("No negative sentences detected")

# Sentiment warnings
for warn in sentiment.warnings:
    st.caption(f"⚠️ {warn}")

st.divider()

# --- RAG Query Interface ---
st.subheader(f"💬 Interrogate {active_ticker}'s 10-K Filing (Two-Stage NVIDIA NIM RAG)")

# Show current filing context
st.caption(f"📄 Current Filing: **{record.ticker}** | {record.form_type} | {record.filing_date} | Accession: `{record.accession_number}` | Backend: **{sentiment.backend}**")

preset_questions = [
    f"What are the primary supply chain and geopolitical risks reported by {active_ticker}?",
    f"What drove {active_ticker}'s revenue growth and gross margin performance in MD&A?",
    f"How do capital expenditures and AI infrastructure investments impact {active_ticker}?",
]
selected_q = st.selectbox("Select an Institutional Analyst Prompt:", preset_questions, index=0)
user_query = st.text_input("Or ask a custom question:", value=selected_q, key="query_input")

# Query button
if st.button("🔍 Retrieve & Synthesize Cited Answer", type="primary", use_container_width=True):
    response = rag_engine.answer_question(
        user_query,
        sentiment_summary=sentiment.to_dict(),
        filing_accession=record.accession_number,
        sentiment_status=ComponentStatus.SUCCESS if sentiment.analyzed_sentences > 0 else ComponentStatus.FAILED,
        market_data_status=record.market_metrics.status if record.market_metrics else ComponentStatus.FAILED,
    )
    
    # Store in session state for persistence
    st.session_state.last_response = response
    st.session_state.last_query = user_query
    st.session_state.query_history.append({
        "query": user_query,
        "response": response,
        "filing_key": current_filing_key,
        "timestamp": pd.Timestamp.now().isoformat(),
    })
    # Keep last 10 queries
    if len(st.session_state.query_history) > 10:
        st.session_state.query_history = st.session_state.query_history[-10:]

# Display persistent response (survives reruns)
response = st.session_state.last_response
if response and st.session_state.last_filing_key == current_filing_key:
    st.markdown("---")
    st.markdown(f"**Question:** {st.session_state.last_query}")
    
    # Generation metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retrieval Latency", f"{response.retrieval_latency_ms:.1f} ms")
    m2.metric("Synthesis Latency", f"{response.generation_latency_ms:.1f} ms")
    m3.metric("Lexical Overlap", f"{response.groundedness_score * 100:.1f}%")
    m4.metric("Generation Mode", response.engine_mode)
    
    # Answer
    st.markdown(response.answer)
    
    # Numeric occurrence check (renamed from claim support)
    claim_support = compute_claim_level_support(response.answer, [c["snippet"] for c in response.citations])
    if claim_support["total_claims"] > 0:
        st.caption(
            f"🔢 **Numeric Occurrence Check:** {claim_support['supported_claims']}/{claim_support['total_claims']} "
            f"numerical values found in sources ({claim_support['claim_support_rate']*100:.0f}%)"
        )
        if claim_support["unsupported_details"]:
            st.caption(f"⚠️ Not found in sources: {', '.join(claim_support['unsupported_details'])}")
    
    # Evidence & Citations - Enhanced inspection tool
    with st.expander("📑 Evidence & Citations", expanded=True):
        for idx, cite in enumerate(response.citations, 1):
            # Citation header with navigation
            col_cite1, col_cite2 = st.columns([4, 1])
            with col_cite1:
                st.markdown(
                    f"**{idx}. [{cite['section']} | `{cite['chunk_id']}`]** "
                    f"*(Filing: {cite['filing_date']} | Accession: {cite['accession']})*"
                )
            with col_cite2:
                # Copy button
                if st.button("📋 Copy", key=f"copy_{idx}", help="Copy citation text"):
                    st.toast("Copied to clipboard!")
            
            # Full passage (not truncated)
            st.markdown(f"> {cite['snippet']}")
            
            # Expandable full context
            with st.expander(f"📄 Full Passage {idx}", expanded=False):
                st.text(cite.get("full_text", cite["snippet"]))
            
            st.divider()
    
    # Execution Manifest as Table
    with st.expander("🔧 Execution Manifest (MLOps Observability)", expanded=False):
        if response.execution_manifest:
            manifest_dict = response.execution_manifest.to_dict()
            
            # Status table
            manifest_df = create_execution_manifest_table(manifest_dict)
            st.dataframe(
                manifest_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Component": st.column_config.TextColumn("Component", width="small"),
                    "Actual Backend": st.column_config.TextColumn("Actual Backend", width="medium"),
                    "State": st.column_config.TextColumn("State", width="small"),
                    "Explanation": st.column_config.TextColumn("Explanation", width="large"),
                },
            )
            
            # Raw JSON for debugging
            with st.expander("Raw Manifest JSON", expanded=False):
                st.json(manifest_dict)
        else:
            st.info("Execution manifest not available")
    
    st.caption(f"Request ID: {request_id}")

# --- Query History ---
if st.session_state.query_history:
    with st.expander("📜 Query History (This Session)", expanded=False):
        for i, item in enumerate(reversed(st.session_state.query_history)):
            if item["filing_key"] == current_filing_key:
                st.markdown(f"**{len(st.session_state.query_history) - i}.** {item['query']}")
                st.caption(f"⏱️ {item['timestamp'][:19]} | Mode: {item['response'].engine_mode}")

# --- Clear Analysis Action ---
if st.session_state.last_response:
    if st.button("🗑️ Clear Analysis", type="secondary"):
        st.session_state.last_response = None
        st.session_state.last_query = None
        st.rerun()