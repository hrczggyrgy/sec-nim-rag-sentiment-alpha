"""Structured logging, groundedness metrics, and interactive Plotly chart builders."""

import logging
import re
import sys
from typing import Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a structured stdout logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def compute_lexical_groundedness(answer: str, context_chunks: list[str]) -> float:
    """Computes token-level groundedness overlap between LLM response and retrieved SEC chunks."""
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


def create_sentiment_volatility_figure(metrics_df: pd.DataFrame) -> go.Figure:
    """Creates a dual-axis Plotly chart comparing Management Optimism Index vs Post-Filing Volatility."""
    fig = make_subplots(specs=[[{"secondary_axis": True}]])

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
        secondary_axis=False,
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
        secondary_axis=True,
    )

    fig.update_layout(
        title="Executive Sentiment (FinBERT MOI) vs. Post-Filing Realized Stock Volatility",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420,
    )
    fig.update_yaxes(title_text="Management Optimism Index (-1 to +1)", secondary_axis=False)
    fig.update_yaxes(title_text="Annualized Volatility (%)", secondary_axis=True)
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