"""Structured logging, groundedness metrics, and interactive Plotly chart builders."""

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Context variable for request tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class StructuredLogger:
    """Structured JSON logger for observability."""

    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)

    def _log(self, level: int, message: str, *args: Any, **kwargs: Any) -> None:
        request_id = request_id_var.get() or ""
        # Handle both formatted strings and keyword arguments
        if args:
            message = message % args
        extra = {"request_id": request_id}
        extra.update(kwargs)
        self.logger.log(level, message, extra=extra)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, *args, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON formatter that properly escapes all fields."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
        }
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                          "module", "lineno", "created", "msecs", "relativeCreated", "thread",
                          "threadName", "processName", "process", "message", "asctime",
                          "exc_info", "exc_text", "stack_info", "request_id"):
                log_data[key] = value
        return json.dumps(log_data, ensure_ascii=False)


def get_logger(name: str) -> StructuredLogger:
    """Returns a structured JSON logger."""
    return StructuredLogger(name)


def set_request_id(request_id: Optional[str] = None) -> str:
    """Sets and returns a request ID for tracing."""
    if request_id is None:
        request_id = uuid.uuid4().hex[:12]
    request_id_var.set(request_id)
    return request_id


def compute_lexical_groundedness(answer: str, context_chunks: list[str]) -> float:
    """
    Computes token-level groundedness overlap between LLM response and retrieved SEC chunks.
    Renamed from 'groundedness' to 'lexical_overlap' - this is a proxy, not true groundedness.
    """
    if not answer.strip() or not context_chunks:
        return 0.0
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "have", "are",
        "was", "were", "will", "would", "could", "should", "into", "about",
    }
    answer_tokens = {
        tok for tok in re.findall(r"\b[a-zA-Z0-9]{4,}\b", answer.lower())
        if tok not in stop_words
    }
    if not answer_tokens:
        return 1.0
    combined_context = " ".join(context_chunks).lower()
    supported = sum(1 for tok in answer_tokens if tok in combined_context)
    return round(supported / len(answer_tokens), 4)


def compute_claim_level_support(answer: str, context_chunks: list[str]) -> dict[str, Any]:
    """
    Attempts claim-level support checking by extracting numerical claims from answer
    and verifying presence in context using precise boundary matching.
    """
    import re

    # Extract numerical claims (percentages, dollar amounts, ratios) with surrounding context
    # Pattern captures the full claim including potential sign and unit
    claim_pattern = re.compile(
        r"(?<!\d)(?:[+-]?\d+(?:\.\d+)?\s*%|\$[+-]?\d+(?:\.\d+)?[BMK]?|[+-]?\d+(?:\.\d+)?x)(?!\d)"
    )
    
    # Find claims with their positions for better context matching
    numerical_claims = claim_pattern.findall(answer)

    combined_context = " ".join(context_chunks)
    supported_claims = []
    unsupported_claims = []

    for claim in numerical_claims:
        # Use word-boundary-aware matching: claim must appear as a complete token
        # Escape special regex characters in the claim
        escaped_claim = re.escape(claim.strip())
        # Match with word boundaries or string boundaries
        pattern = rf"(?<![\d\w]){escaped_claim}(?![\d\w])"
        if re.search(pattern, combined_context):
            supported_claims.append(claim)
        else:
            unsupported_claims.append(claim)

    return {
        "total_claims": len(numerical_claims),
        "supported_claims": len(supported_claims),
        "unsupported_claims": len(unsupported_claims),
        "claim_support_rate": len(supported_claims) / len(numerical_claims) if numerical_claims else 1.0,
        "unsupported_details": unsupported_claims,
    }


def create_sentiment_volatility_figure(
    metrics_df: pd.DataFrame,
    backend: str = "unknown",
    show_dual_axis: bool = False,
) -> go.Figure:
    """
    Creates aligned panel chart comparing Management Optimism Index vs Post-Filing Volatility.
    
    By default uses vertically aligned panels sharing the x-axis (filing date) to avoid
    misleading dual-axis correlations. Set show_dual_axis=True for legacy overlay view.
    """
    # Sort by filing date for consistent rendering
    df = metrics_df.sort_values("filing_date").reset_index(drop=True)
    
    # Determine backend for title
    backend_label = backend.replace("_", " ").title()
    if "lexicon" in backend.lower() or "loughran" in backend.lower():
        backend_display = "Lexicon Baseline"
    elif "finbert" in backend.lower():
        backend_display = "FinBERT"
    else:
        backend_display = backend_label
    
    if show_dual_axis:
        # Legacy dual-axis view (not recommended)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        def _color_for_val(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "#6B7280"  # Gray for missing
            return "#10B981" if val >= 0 else "#EF4444"
        
        fig.add_trace(
            go.Bar(
                x=df["filing_date"],
                y=df["optimism_index"],
                name="Management Optimism Index (MOI)",
                marker_color=[_color_for_val(val) for val in df["optimism_index"]],
                opacity=0.85,
                customdata=df["accession"] if "accession" in df.columns else None,
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=df["filing_date"],
                y=df["post_7d_volatility_pct"],
                name="7-Day Post-Filing Annualized Volatility (%)",
                mode="lines+markers",
                line=dict(color="#60A5FA", width=3),
                marker=dict(size=9),
                customdata=df["accession"] if "accession" in df.columns else None,
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            title=(
                f"Executive Sentiment ({backend_display} MOI) vs. Post-Filing Realized Volatility "
                "<br><sup>Descriptive comparison; not evidence of predictive performance</sup>"
            ),
            template="plotly_white",  # Support light/dark via template
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40),
            height=420,
        )
        fig.update_yaxes(title_text="Management Optimism Index (-1 to +1)", secondary_y=False)
        fig.update_yaxes(title_text="Annualized Volatility (%)", secondary_y=True)
        return fig
    
    # Recommended: Two vertically aligned panels sharing x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Filing Tone (MOI) — {backend_display}",
            "Annualized Realized Volatility (7-Day Post-Filing Window)",
        ),
        row_heights=[0.5, 0.5],
    )
    
    # Panel 1: MOI bars
    def _color_for_val(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "#9CA3AF"  # Gray for missing
        return "#10B981" if val >= 0 else "#EF4444"
    
    moi_colors = [_color_for_val(val) for val in df["optimism_index"]]
    
    # Add pattern/shape for missing values (accessibility)
    fig.add_trace(
        go.Bar(
            x=df["filing_date"],
            y=df["optimism_index"],
            name="Management Optimism Index (MOI)",
            marker_color=moi_colors,
            opacity=0.85,
            customdata=df["accession"] if "accession" in df.columns else None,
            hovertemplate=(
                "Filing Date: %{x}<br>"
                "MOI: %{y:.3f}<br>"
                "Accession: %{customdata}<br>"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    
    # Zero reference line for MOI
    fig.add_hline(y=0, line_dash="dot", line_color="#9CA3AF", line_width=1, row=1, col=1)
    
    # Panel 2: Volatility line
    fig.add_trace(
        go.Scatter(
            x=df["filing_date"],
            y=df["post_7d_volatility_pct"],
            name="Annualized Volatility (%)",
            mode="lines+markers",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=8, symbol="circle"),
            connectgaps=False,  # Don't connect across missing values
            customdata=df["accession"] if "accession" in df.columns else None,
            hovertemplate=(
                "Filing Date: %{x}<br>"
                "Annualized Volatility: %{y:.2f}%<br>"
                "Accession: %{customdata}<br>"
                "<extra></extra>"
            ),
        ),
        row=2, col=1,
    )
    
    # Coverage info
    moi_available = df["optimism_index"].notna().sum()
    vol_available = df["post_7d_volatility_pct"].notna().sum()
    total = len(df)
    
    fig.update_layout(
        title=(
            f"Filing Tone & Volatility History — {backend_display} "
            f"<br><sup>Tone available: {moi_available}/{total} | Volatility: {vol_available}/{total} | "
            "Descriptive comparison; not evidence of predictive performance</sup>"
        ),
        template="plotly_white",  # Support both light/dark themes
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=100, b=40),
        height=500,
        showlegend=False,
    )
    fig.update_yaxes(title_text="MOI (-1 to +1)", row=1, col=1, zeroline=True, zerolinecolor="#9CA3AF")
    fig.update_yaxes(title_text="Annualized Volatility (%)", row=2, col=1)
    fig.update_xaxes(title_text="Filing Date", row=2, col=1)
    
    return fig


def create_sentiment_distribution_chart(
    sentiment_summary: dict[str, Any],
    backend: str = "unknown",
    section_breakdown: Optional[dict[str, dict[str, float]]] = None,
) -> go.Figure:
    """
    Creates a horizontal bar chart of sentence tone distribution.
    
    Shows actual backend used, sentence counts, and optional section-level breakdown.
    Handles unavailable/zero-sentence states clearly.
    """
    backend_display = _get_backend_display_name(backend)
    analyzed = sentiment_summary.get("analyzed_sentences", 0)
    eligible = sentiment_summary.get("eligible_sentences", analyzed)
    
    # Check if sentiment is unavailable (no analyzed sentences)
    if analyzed == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentences analyzed — sentiment unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#9CA3AF"),
        )
        fig.update_layout(
            title=f"Sentence Tone — {backend_display} (Unavailable)",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=30),
            height=280,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig
    
    # Determine display mode
    has_sections = section_breakdown and len(section_breakdown) > 0
    
    if has_sections:
        # Section-level 100% stacked bars
        sections = list(section_breakdown.keys())
        fig = go.Figure()
        
        for sentiment_label, color in [
            ("Positive", "#10B981"),
            ("Neutral", "#6B7280"),
            ("Negative", "#EF4444"),
        ]:
            values = [section_breakdown[sec].get(sentiment_label.lower(), 0) * 100 for sec in sections]
            fig.add_trace(
                go.Bar(
                    x=values,
                    y=sections,
                    orientation="h",
                    name=sentiment_label,
                    marker_color=color,
                    text=[f"{v:.1f}%" if v > 0 else "" for v in values],
                    textposition="inside",
                    insidetextanchor="middle",
                )
            )
        
        fig.update_layout(
            barmode="stack",
            title=f"Sentence Tone by Section — {backend_display}",
            template="plotly_white",
            xaxis=dict(title="Percentage of Section Sentences (%)", range=[0, 100]),
            margin=dict(l=120, r=40, t=50, b=30),
            height=max(280, 80 * len(sections) + 80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        # Aggregate horizontal bars with sentence counts
        labels = ["Positive", "Neutral", "Negative"]
        ratios = [
            sentiment_summary.get("positive_ratio", 0.0),
            sentiment_summary.get("neutral_ratio", 0.0),
            sentiment_summary.get("negative_ratio", 0.0),
        ]
        values = [r * 100 for r in ratios]
        counts = [round(r * analyzed) for r in ratios]
        colors = ["#10B981", "#6B7280", "#EF4444"]
        
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}% ({c} sentences)" for v, c in zip(values, counts)],
                textposition="auto",
                hovertemplate=(
                    "Tone: %{y}<br>"
                    "Percentage: %{x:.1f}%<br>"
                    "Sentences: %{customdata}<br>"
                    "<extra></extra>"
                ),
                customdata=counts,
            )
        )
        
        fig.update_layout(
            title=f"Sentence Tone — {backend_display}",
            template="plotly_white",
            xaxis=dict(title="Percentage of Analyzed Sentences (%)", range=[0, 100]),
            margin=dict(l=40, r=40, t=50, b=30),
            height=280,
        )
    
    # Add annotation for coverage
    if analyzed < eligible:
        fig.add_annotation(
            text=f"Analyzed: {analyzed} of {eligible} eligible sentences",
            xref="paper", yref="paper",
            x=1, y=-0.15, showarrow=False,
            font=dict(size=10, color="#9CA3AF"),
            xanchor="right", yanchor="top",
        )
    
    return fig


def _get_backend_display_name(backend: str) -> str:
    """Convert backend identifier to display name."""
    backend_lower = backend.lower()
    if "finbert" in backend_lower:
        return "FinBERT"
    elif "lexicon" in backend_lower or "loughran" in backend_lower:
        return "Lexicon Baseline"
    elif "offline" in backend_lower or "fallback" in backend_lower:
        return "Offline Fallback"
    else:
        return backend.replace("_", " ").title()


def create_execution_manifest_table(manifest: dict[str, Any]) -> pd.DataFrame:
    """
    Creates a DataFrame for rendering execution manifest as a status table.
    
    Returns a DataFrame with columns: Component, Actual Backend, State, Explanation.
    """
    # Map status to display
    status_map = {
        "success": ("✅ Complete", "Component executed successfully"),
        "degraded": ("⚠️ Degraded", "Component completed with reduced capability"),
        "failed": ("❌ Failed", "Component failed to execute"),
        "skipped": ("⏭️ Skipped", "Component not required for this execution"),
        "offline_fallback": ("🔄 Offline Fallback", "Using deterministic local fallback"),
    }
    
    # Extract actual backends from manifest
    backends = {
        "Embedding": manifest.get("embedding_model", "unknown"),
        "Reranker": manifest.get("reranker_model", "unknown"),
        "Generator": manifest.get("generator_model", "unknown"),
        "Sentiment": manifest.get("sentiment_model", "unknown"),
        "Market Data": manifest.get("market_data_source", "Yahoo Finance"),
    }
    
    statuses = {
        "Embedding": manifest.get("embedding_status", "unknown"),
        "Reranker": manifest.get("reranker_status", "unknown"),
        "Generator": manifest.get("generator_status", "unknown"),
        "Sentiment": manifest.get("sentiment_status", "unknown"),
        "Market Data": manifest.get("market_data_status", "unknown"),
    }
    
    rows = []
    for component in ["Embedding", "Reranker", "Generator", "Sentiment", "Market Data"]:
        status = statuses.get(component, "unknown")
        state_label, explanation = status_map.get(status, ("❓ Unknown", "Status not recognized"))
        backend = backends.get(component, "unknown")
        
        # Enhance explanation for fallback states
        if status == "offline_fallback":
            if component == "Embedding":
                explanation = "Using SHA256 deterministic hash embeddings (no semantic search)"
            elif component == "Reranker":
                explanation = "No cross-encoder reranking; using vector similarity only"
            elif component == "Generator":
                explanation = "Using extractive synthesis (no LLM generation)"
            elif component == "Sentiment":
                explanation = "Using Loughran-McDonald lexicon (no neural model)"
        
        rows.append({
            "Component": component,
            "Actual Backend": backend,
            "State": state_label,
            "Explanation": explanation,
        })
    
    return pd.DataFrame(rows)