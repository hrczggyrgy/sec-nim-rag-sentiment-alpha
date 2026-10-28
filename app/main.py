"""Interactive Streamlit Dashboard for SEC 10-K NVIDIA NIM RAG & FinBERT Volatility Analytics."""

import json
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional
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
from sec_nim_alpha.schemas import ComponentStatus, SentimentResult
from sec_nim_alpha.data_loader import FilingRecord

st.set_page_config(
    page_title="SEC Filing NIM RAG & FinBERT Alpha Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 SEC 10-K NIM RAG & FinBERT Volatility Terminal")

# ============================================================
# WORKSPACE MODEL
# ============================================================
@dataclass
class AnalystWorkspace:
    """Complete analyst workspace state for a filing analysis session."""
    ticker: str
    accession: str
    execution_mode: str
    artifact_version: str
    record: FilingRecord
    sentiment: SentimentResult
    history: pd.DataFrame
    engine: NVIDIANimRAGEngine
    request_id: str
    # Per-query state
    last_response: Optional[object] = None
    last_query: Optional[str] = None
    query_history: list = field(default_factory=list)


def _make_workspace_key(record: FilingRecord, execution_mode: str, sentiment_backend: str) -> str:
    """Create a unique workspace identity including provenance."""
    return f"{record.ticker}|{record.accession_number}|{execution_mode}|{sentiment_backend}"


def _load_workspace(
    ticker: str,
    accession: str,
    execution_mode: ExecutionMode,
    api_key: str,
    enable_finbert: bool,
) -> Optional[AnalystWorkspace]:
    """Load a complete workspace for a specific filing."""
    loader = SECDataLoader(execution_mode=execution_mode)
    record = loader.load_filing_by_accession(ticker, accession)
    
    if record is None:
        return None
    
    # Load historical panel
    panel_df = loader.load_historical_panel(ticker)
    
    # Compute sentiment for this filing
    feat_pipe = FinancialFeaturePipeline(use_finbert_model=enable_finbert, execution_mode=execution_mode)
    docs = feat_pipe.build_rag_documents(record)
    sentiment = feat_pipe.compute_sentiment_features(record)
    
    # Build/update retrieval engine
    rag_engine = NVIDIANimRAGEngine(api_key=api_key, execution_mode=execution_mode)
    rag_engine.index_documents(docs, ticker=record.ticker, accession=record.accession_number)
    
    request_id = set_request_id()
    artifact_version = f"{record.content_hash[:8] if record.content_hash else 'unknown'}-{request_id[:8]}"
    
    return AnalystWorkspace(
        ticker=record.ticker,
        accession=record.accession_number,
        execution_mode=execution_mode.value,
        artifact_version=artifact_version,
        record=record,
        sentiment=sentiment,
        history=panel_df,
        engine=rag_engine,
        request_id=request_id,
    )


def _run_full_pipeline(
    ticker: str,
    api_key: str,
    enable_finbert: bool,
    execution_mode: ExecutionMode,
) -> AnalystWorkspace:
    """Run full pipeline and return a new workspace (always fresh)."""
    loader = SECDataLoader(execution_mode=execution_mode)
    record = loader.fetch_latest_10k(ticker, force_refresh=True)
    panel_df = loader.load_historical_panel(ticker)

    feat_pipe = FinancialFeaturePipeline(use_finbert_model=enable_finbert, execution_mode=execution_mode)
    docs = feat_pipe.build_rag_documents(record)
    sentiment = feat_pipe.compute_sentiment_features(record)

    # Update panel with current filing's actual data
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
    rag_engine.index_documents(docs, ticker=record.ticker, accession=record.accession_number)
    
    request_id = set_request_id()
    artifact_version = f"{record.content_hash[:8] if record.content_hash else 'unknown'}-{request_id[:8]}"
    
    return AnalystWorkspace(
        ticker=record.ticker,
        accession=record.accession_number,
        execution_mode=execution_mode.value,
        artifact_version=artifact_version,
        record=record,
        sentiment=sentiment,
        history=panel_df,
        engine=rag_engine,
        request_id=request_id,
    )


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "workspace" not in st.session_state:
    st.session_state.workspace = None
if "last_refresh_error" not in st.session_state:
    st.session_state.last_refresh_error = None
if "config_hash" not in st.session_state:
    st.session_state.config_hash = None


# ============================================================
# SIDEBAR: CONFIGURATION FORM
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
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

# ============================================================
# CONFIG HASH - detect config changes that invalidate workspace
# ============================================================
current_config_hash = f"{active_ticker}|{execution_mode.value}|{use_live_finbert}|{api_key_input[:10] if api_key_input else 'none'}"

if st.session_state.config_hash != current_config_hash:
    # Config changed - invalidate workspace
    st.session_state.workspace = None
    st.session_state.config_hash = current_config_hash


# ============================================================
# PIPELINE EXECUTION & WORKSPACE MANAGEMENT
# ============================================================
pipeline_error = None

if refresh_clicked or run_submitted:
    # Always run fresh pipeline on explicit action
    try:
        with st.spinner("Executing SEC EDGAR Ingestion, FinBERT Scoring, and Vector Indexing..."):
            workspace = _run_full_pipeline(active_ticker, api_key_input, use_live_finbert, execution_mode)
        st.session_state.workspace = workspace
        st.session_state.last_refresh_error = None
        st.toast("✅ Pipeline completed successfully", icon="✅")
    except Exception as e:
        st.session_state.last_refresh_error = str(e)
        # Keep previous valid workspace visible
        if st.session_state.workspace is None:
            st.error(f"Pipeline execution failed: {e}")
            st.stop()
        else:
            st.error(f"Refresh failed: {e} — showing previous workspace")

elif st.session_state.workspace is None:
    # No workspace yet - try to load from cache (latest filing)
    try:
        loader = SECDataLoader(execution_mode=execution_mode)
        latest_record = loader.fetch_latest_10k(active_ticker, force_refresh=False)
        
        with st.spinner("Loading cached workspace..."):
            workspace = _load_workspace(
                active_ticker, 
                latest_record.accession_number, 
                execution_mode, 
                api_key_input, 
                use_live_finbert
            )
        
        if workspace is None:
            st.info("No cached workspace found. Click **Run Ingestion & Indexing** to begin.")
            st.stop()
        
        st.session_state.workspace = workspace
    except Exception as e:
        st.error(f"Failed to load workspace: {e}")
        st.info("Click **Run Ingestion & Indexing** to create a new workspace.")
        st.stop()


# ============================================================
# FILING DRILL-DOWN: Handle chart selection
# ============================================================
workspace = st.session_state.workspace

# Check if user selected a different filing via chart
if st.session_state.get("selected_filing_accession"):
    selected_accession = st.session_state.selected_filing_accession
    if selected_accession != workspace.accession:
        # Load the selected filing's workspace
        with st.spinner(f"Loading filing {selected_accession}..."):
            new_workspace = _load_workspace(
                workspace.ticker,
                selected_accession,
                execution_mode,
                api_key_input,
                use_live_finbert,
            )
        if new_workspace:
            st.session_state.workspace = new_workspace
            workspace = new_workspace
            # Clear selection after successful load
            st.session_state.selected_filing_accession = None
        else:
            st.error(f"Could not load filing {selected_accession}")
            st.session_state.selected_filing_accession = None

# ============================================================
# EXECUTION STATUS STRIP (shows ACTUAL execution state)
# ============================================================
def render_status_strip(record, rag_engine, sentiment, market_metrics):
    """Render persistent status strip showing actual execution state."""
    if record.is_synthetic_fallback:
        filing_src = "SYNTHETIC DEMO"
        filing_color = "🟠"
    else:
        filing_src = "LIVE EDGAR"
        filing_color = "🟢"
    
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
    
    if rag_engine.nim_enabled:
        retrieval_display = "NIM EMBEDDING"
        retrieval_color = "🟢"
    else:
        retrieval_display = "HASH FALLBACK"
        retrieval_color = "🟣"
    
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
            background-color: var(--background-color-secondary, #1E1E1E); 
            padding: 0.75rem 1rem; 
            border-radius: 6px; 
            margin-bottom: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            border: 1px solid var(--border-color, #333);
            color: var(--text-color, inherit);
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

render_status_strip(workspace.record, workspace.engine, workspace.sentiment, workspace.record.market_metrics)

# Show active filing identifier
st.caption(f"📄 **Active Filing:** {workspace.record.ticker} | {workspace.record.form_type} | {workspace.record.filing_date} | Accession: `{workspace.record.accession_number}` | Artifact: `{workspace.artifact_version}`")

# ============================================================
# TECHNICAL DETAILS EXPANDER
# ============================================================
with st.expander("🔧 Technical Details: Configured Models & Endpoints", expanded=False):
    st.markdown(f"""
    **Execution Mode:** `{execution_mode.value}` (selected) | `{CONFIG.execution_mode.value}` (env default)
    
    **Embedding:** `{CONFIG.nim_embed_model}`
    
    **LLM:** `{CONFIG.nim_llm_model}`
    
    **Reranker:** `{CONFIG.nim_rerank_model}`
    
    **FinBERT:** `{CONFIG.finbert_model}` {'(enabled)' if use_live_finbert else '(disabled - using lexicon)'}
    """)

# ============================================================
# WARNINGS
# ============================================================
if workspace.record.is_synthetic_fallback:
    st.warning("⚠️ **DEMO MODE**: This filing uses curated synthetic data. Not for research use.")

if workspace.record.market_metrics and workspace.record.market_metrics.status != ComponentStatus.SUCCESS:
    status_color = "🟡" if workspace.record.market_metrics.status == ComponentStatus.DEGRADED else "🔴"
    st.warning(f"{status_color} Market Data: {workspace.record.market_metrics.status.value} - {workspace.record.market_metrics.error_message}")

if st.session_state.last_refresh_error:
    st.error(f"⚠️ Last refresh failed: {st.session_state.last_refresh_error}")

st.divider()

# ============================================================
# HEADER METRICS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Filing",
    f"{workspace.record.form_type} {workspace.record.filing_date}",
    help=f"Accession: {workspace.record.accession_number} | CIK: {workspace.record.cik}",
)

# MOI delta only with historical comparison
moi_delta = None
moi_delta_help = f"Backend: {workspace.sentiment.backend} | Analyzed: {workspace.sentiment.analyzed_sentences} sentences"
if len(workspace.history) > 1 and workspace.history["optimism_index"].notna().sum() > 1:
    prior_moi = workspace.history[workspace.history["accession"] != workspace.record.accession_number]["optimism_index"].dropna()
    if len(prior_moi) > 0:
        moi_delta = workspace.sentiment.optimism_index - prior_moi.iloc[-1]
        moi_delta_help += f" | Change vs prior: {moi_delta:+.3f}"

col2.metric(
    "Management Optimism Index (MOI)",
    f"{workspace.sentiment.optimism_index:+.3f}" if workspace.sentiment.optimism_index is not None else "Unavailable",
    delta=f"{moi_delta:+.3f}" if moi_delta is not None else None,
    help=moi_delta_help,
)

col3.metric(
    "Annualized Realized Volatility (7D)",
    f"{workspace.record.market_metrics.post_7d_volatility_pct:.2f}%" if workspace.record.market_metrics and workspace.record.market_metrics.post_7d_volatility_pct is not None else "Unavailable",
    help=f"Based on {workspace.record.market_metrics.trading_days_observed if workspace.record.market_metrics else 0}/7 trading days post-filing | Status: {workspace.record.market_metrics.status.value if workspace.record.market_metrics else 'unknown'}",
)

col4.metric(
    "7-Day Return",
    f"{workspace.record.market_metrics.post_7d_return_pct:+.2f}%" if workspace.record.market_metrics and workspace.record.market_metrics.post_7d_return_pct is not None else "Unavailable",
    help=f"Raw 7-day price return (not annualized) | Trading days: {workspace.record.market_metrics.trading_days_observed if workspace.record.market_metrics else 0}/7",
)

st.divider()

# ============================================================
# CHARTS
# ============================================================
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📊 Filing Tone & Volatility History")
    if len(workspace.history) > 0:
        fig = create_sentiment_volatility_figure(
            workspace.history,
            backend=workspace.sentiment.backend,
            show_dual_axis=False,
        )
        
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="tone_vol_chart",
        )
        
        # Handle chart point selection → drill down
        if selection and selection.get("selection", {}).get("points"):
            point = selection["selection"]["points"][0]
            if "customdata" in point and point["customdata"]:
                selected_accession = point["customdata"]
                if selected_accession != workspace.accession:
                    st.session_state.selected_filing_accession = selected_accession
                    st.rerun()
        
        moi_avail = workspace.history["optimism_index"].notna().sum()
        vol_avail = workspace.history["post_7d_volatility_pct"].notna().sum()
        total = len(workspace.history)
        st.caption(f"📋 Coverage: Tone {moi_avail}/{total} filings | Volatility {vol_avail}/{total} filings")
        
        with st.expander("📋 View Underlying Data", expanded=False):
            display_df = workspace.history.copy().sort_values("filing_date", ascending=False)
            st.dataframe(
                display_df[["filing_date", "quarter", "accession", "optimism_index", "post_7d_volatility_pct", "post_7d_return_pct"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No historical filings available for this ticker in research mode.")

with right_col:
    st.subheader("📈 Sentence Tone Distribution")
    
    section_breakdown = None
    if hasattr(workspace.sentiment, 'section_sentiment') and workspace.sentiment.section_sentiment:
        section_breakdown = workspace.sentiment.section_sentiment
    
    dist_fig = create_sentiment_distribution_chart(
        workspace.sentiment.to_dict(),
        backend=workspace.sentiment.backend,
        section_breakdown=section_breakdown,
    )
    st.plotly_chart(dist_fig, use_container_width=True, key="sentiment_dist_chart")
    
    st.markdown("**🟢 Top Positive Statement:**")
    if workspace.sentiment.top_bullish_sentence:
        st.success(workspace.sentiment.top_bullish_sentence)
    else:
        st.caption("No positive sentences detected")
    
    st.markdown("**🔴 Top Risk / Negative Statement:**")
    if workspace.sentiment.top_bearish_sentence:
        st.error(workspace.sentiment.top_bearish_sentence)
    else:
        st.caption("No negative sentences detected")

for warn in workspace.sentiment.warnings:
    st.caption(f"⚠️ {warn}")

st.divider()

# ============================================================
# RAG QUERY INTERFACE
# ============================================================
st.subheader(f"💬 Interrogate {workspace.record.ticker}'s 10-K Filing (Two-Stage NVIDIA NIM RAG)")

preset_questions = [
    f"What are the primary supply chain and geopolitical risks reported by {workspace.record.ticker}?",
    f"What drove {workspace.record.ticker}'s revenue growth and gross margin performance in MD&A?",
    f"How do capital expenditures and AI infrastructure investments impact {workspace.record.ticker}?",
]
selected_q = st.selectbox("Select an Institutional Analyst Prompt:", preset_questions, index=0)
user_query = st.text_input("Or ask a custom question:", value=selected_q, key="query_input")

if st.button("🔍 Retrieve & Synthesize Cited Answer", type="primary", use_container_width=True):
    response = workspace.engine.answer_question(
        user_query,
        sentiment_summary=workspace.sentiment.to_dict(),
        filing_accession=workspace.record.accession_number,
        sentiment_status=ComponentStatus.SUCCESS if workspace.sentiment.analyzed_sentences > 0 else ComponentStatus.FAILED,
        market_data_status=workspace.record.market_metrics.status if workspace.record.market_metrics else ComponentStatus.FAILED,
    )
    
    # Store in workspace (not just session_state)
    workspace.last_response = response
    workspace.last_query = user_query
    workspace.query_history.append({
        "query": user_query,
        "response": response,
        "workspace_key": _make_workspace_key(workspace.record, workspace.execution_mode, workspace.sentiment.backend),
        "timestamp": pd.Timestamp.now().isoformat(),
    })
    # Keep last 10 queries
    if len(workspace.query_history) > 10:
        workspace.query_history = workspace.query_history[-10:]

# Display persistent response (survives reruns, scoped to workspace)
response = workspace.last_response
if response and _make_workspace_key(workspace.record, workspace.execution_mode, workspace.sentiment.backend) == workspace.query_history[-1].get("workspace_key", "") if workspace.query_history else False:
    pass  # Will be displayed below

# Display if we have a response for THIS workspace
if response:
    st.markdown("---")
    st.markdown(f"**Question:** {workspace.last_query}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retrieval Latency", f"{response.retrieval_latency_ms:.1f} ms")
    m2.metric("Synthesis Latency", f"{response.generation_latency_ms:.1f} ms")
    m3.metric("Lexical Overlap", f"{response.groundedness_score * 100:.1f}%")
    m4.metric("Generation Mode", response.engine_mode)
    
    st.markdown(response.answer)
    
    # Numeric occurrence check on FULL passages
    full_passages = [c.get("full_text", c["snippet"]) for c in response.citations]
    claim_support = compute_claim_level_support(response.answer, full_passages)
    if claim_support["total_claims"] > 0:
        st.caption(
            f"🔢 **Numeric Occurrence Check:** {claim_support['supported_claims']}/{claim_support['total_claims']} "
            f"numerical values found in sources ({claim_support['claim_support_rate']*100:.0f}%)"
        )
        if claim_support["unsupported_details"]:
            st.caption(f"⚠️ Not found in sources: {', '.join(claim_support['unsupported_details'])}")
    
    # Evidence & Citations with WORKING copy
    with st.expander("📑 Evidence & Citations", expanded=True):
        for idx, cite in enumerate(response.citations, 1):
            col_cite1, col_cite2 = st.columns([4, 1])
            with col_cite1:
                st.markdown(
                    f"**{idx}. [{cite['section']} | `{cite['chunk_id']}`]** "
                    f"*(Filing: {cite['filing_date']} | Accession: {cite['accession']})*"
                )
            with col_cite2:
                # Actual copy to clipboard using Streamlit's clipboard API
                full_text = cite.get("full_text", cite["snippet"])
                if st.button("📋 Copy", key=f"copy_{idx}", help="Copy full passage to clipboard"):
                    # Use JavaScript to copy to clipboard
                    st.components.v1.html(f"""
                        <script>
                            navigator.clipboard.writeText({json.dumps(full_text)});
                        </script>
                    """, height=0)
                    st.toast("✅ Copied to clipboard!")
            
            st.markdown(f"> {cite['snippet']}")
            
            with st.expander(f"📄 Full Passage {idx}", expanded=False):
                st.text(full_text)
            
            st.divider()
    
    # Execution Manifest
    with st.expander("🔧 Execution Manifest (MLOps Observability)", expanded=False):
        if response.execution_manifest:
            manifest_dict = response.execution_manifest.to_dict()
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
            with st.expander("Raw Manifest JSON", expanded=False):
                st.json(manifest_dict)
        else:
            st.info("Execution manifest not available")
    
    st.caption(f"Request ID: {workspace.request_id}")

# ============================================================
# QUERY HISTORY (scoped to workspace)
# ============================================================
if workspace.query_history:
    with st.expander("📜 Query History (This Workspace)", expanded=False):
        for i, item in enumerate(reversed(workspace.query_history)):
            st.markdown(f"**{len(workspace.query_history) - i}.** {item['query']}")
            st.caption(f"⏱️ {item['timestamp'][:19]} | Mode: {item['response'].engine_mode}")
            if st.button("🔄 Restore", key=f"restore_{i}", help="Reload this answer"):
                workspace.last_response = item["response"]
                workspace.last_query = item["query"]
                st.rerun()

# ============================================================
# CLEAR ANALYSIS
# ============================================================
if workspace.last_response:
    if st.button("🗑️ Clear Analysis", type="secondary"):
        workspace.last_response = None
        workspace.last_query = None
        st.rerun()