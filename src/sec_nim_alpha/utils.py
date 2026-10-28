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
            formatter = logging.Formatter(
                fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "request_id": "%(request_id)s"}',
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            handler.setFormatter(formatter)
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
    and verifying presence in context.
    """
    import re

    # Extract numerical claims (percentages, dollar amounts, ratios)
    numerical_claims = re.findall(
        r"\b\d+(?:\.\d+)?\s*%|\$\d+(?:\.\d+)?[BMK]?|\b\d+(?:\.\d+)?x\b",
        answer
    )

    combined_context = " ".join(context_chunks)
    supported_claims = []
    unsupported_claims = []

    for claim in numerical_claims:
        if claim in combined_context:
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


def create_sentiment_volatility_figure(metrics_df: pd.DataFrame) -> go.Figure:
    """Creates a dual-axis Plotly chart comparing Management Optimism Index vs Post-Filing Volatility."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=metrics_df["filing_date"],
            y=metrics_df["optimism_index"],
            name="Management Optimism Index (MOI)",
            marker_color=[
                "#10B981" if val >= 0 else "#EF4444"
                for val in metrics_df["optimism_index"]
            ],
            opacity=0.85,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics_df["filing_date"],
            y=metrics_df["post_7d_volatility_pct"],
            name="7-Day Post-Filing Annualized Volatility (%)",
            mode="lines+markers",
            line=dict(color="#60A5FA", width=3),
            marker=dict(size=9),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Executive Sentiment (FinBERT MOI) vs. Post-Filing Realized Stock Volatility",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420,
    )
    fig.update_yaxes(title_text="Management Optimism Index (-1 to +1)", secondary_y=False)
    fig.update_yaxes(title_text="Annualized Volatility (%)", secondary_y=True)
    return fig


def create_sentiment_distribution_chart(sentiment_summary: dict[str, Any]) -> go.Figure:
    """Creates a horizontal stacked bar chart of FinBERT sentence classifications."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        sentiment_summary.get("positive_ratio", 0.0) * 100,
        sentiment_summary.get("neutral_ratio", 0.0) * 100,
        sentiment_summary.get("negative_ratio", 0.0) * 100,
    ]
    colors = ["#10B981", "#6B7280", "#EF4444"]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="FinBERT Sentence-Level Tone Breakdown (%)",
        template="plotly_dark",
        xaxis=dict(title="Percentage of Filing Sentences (%)", range=[0, 100]),
        margin=dict(l=40, r=40, t=50, b=30),
        height=280,
    )
    return fig


def create_execution_manifest_chart(manifest: dict[str, Any]) -> go.Figure:
    """Creates a status chart for the execution manifest."""
    components = [
        ("Embedding", manifest.get("embedding_status", "unknown")),
        ("Reranker", manifest.get("reranker_status", "unknown")),
        ("Generator", manifest.get("generator_status", "unknown")),
        ("Sentiment", manifest.get("sentiment_status", "unknown")),
        ("Market Data", manifest.get("market_data_status", "unknown")),
    ]

    status_colors = {
        "success": "#10B981",
        "degraded": "#F59E0B",
        "failed": "#EF4444",
        "skipped": "#6B7280",
        "offline_fallback": "#8B5CF6",
    }

    fig = go.Figure(
        go.Bar(
            x=[c[0] for c in components],
            y=[1] * len(components),
            marker_color=[status_colors.get(c[1], "#6B7280") for c in components],
            text=[c[1] for c in components],
            textposition="inside",
            orientation="v",
        )
    )
    fig.update_layout(
        title="Pipeline Component Status",
        template="plotly_dark",
        yaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=40, r=40, t=50, b=30),
        height=250,
    )
    return fig