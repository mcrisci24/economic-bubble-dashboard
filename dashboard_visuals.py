"""Reusable chart builders and visual components for the dashboard.

Everything here either (a) returns a Plotly figure, or (b) renders a small
visual component into Streamlit (interpretation box, regime badge). Pages
should prefer these helpers over inline Plotly so we stay consistent across
the dashboard and avoid copy-pasted styling.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Interpretation boxes. Every important chart should be followed by one of
# these. The four-section contract — what / how / current / do-not-overclaim —
# enforces a discipline of *describing* the result rather than asserting a
# financial conclusion.
# ---------------------------------------------------------------------------
def render_interpretation(
    *,
    what_chart_shows: str,
    how_to_read: str,
    current_result: str,
    do_not_overclaim: str,
    container: Any | None = None,
) -> None:
    """Render the standard 4-section interpretation box under a chart.

    `container` defaults to `st` so the box appears in the current page flow.
    Pass an `st.container()` if you need to nest it inside columns.
    """
    target = container if container is not None else st
    target.markdown(
        "**Interpretation**\n\n"
        f"- **What this chart shows:** {what_chart_shows}\n"
        f"- **How to read it:** {how_to_read}\n"
        f"- **Current result:** {current_result}\n"
        f"- **What *not* to overclaim:** {do_not_overclaim}"
    )


# ---------------------------------------------------------------------------
# Regime badge component. Translates the rule-based phase label / warning
# score into one of the user-facing badges from the spec.
# ---------------------------------------------------------------------------
BADGE_DEFINITIONS: dict[str, dict[str, str]] = {
    "Normal": {
        "color": "#2e7d32",
        "tooltip": "No active warning. Drawdown is shallow, momentum is neutral, valuation/macro stress is unremarkable.",
    },
    "Heating Up": {
        "color": "#fbc02d",
        "tooltip": "Rising warning score and stretched momentum, but no breakdown yet.",
    },
    "Euphoria / Stretched": {
        "color": "#ef6c00",
        "tooltip": "High warning score, large positive distance from 200DMA, hot RSI, or extreme valuation. Historically a higher-risk regime.",
    },
    "Breakdown": {
        "color": "#c62828",
        "tooltip": "Drawdown is accelerating; signals that previously fired are now confirmed by price action.",
    },
    "Capitulation": {
        "color": "#6a1b9a",
        "tooltip": "Very deep drawdown, oversold momentum, fear signals. Historically late in the bust cycle.",
    },
    "Recovery Watch": {
        "color": "#1565c0",
        "tooltip": "Drawdown stabilising and recovery score rising. Historical re-entry rules may begin to fire.",
    },
}


def render_regime_badge(badge: str, *, subtitle: str | None = None) -> None:
    """Render a colored regime badge with optional subtitle.

    Keep this small — it should sit next to a metric card, not dominate the
    page. Uses inline CSS via st.markdown for portability.
    """
    spec = BADGE_DEFINITIONS.get(badge)
    if spec is None:
        st.info(f"Regime: {badge}")
        return
    sub = f"<div style='font-size:0.85em;opacity:0.85;margin-top:4px'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div title="{spec['tooltip']}" style="
            display:inline-block;
            padding:8px 14px;
            border-radius:8px;
            background:{spec['color']};
            color:white;
            font-weight:600;
            letter-spacing:0.02em;
            box-shadow:0 1px 2px rgba(0,0,0,0.15);">
            {badge}
        </div>
        {sub}
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def price_signal_chart(g: pd.DataFrame, ticker: str) -> go.Figure:
    """Adjusted close + 50/200 SMA + exit/re-entry markers for one ticker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["date"], y=g["adjusted_close"], mode="lines", name="Adjusted close"))
    if "sma_50" in g.columns:
        fig.add_trace(go.Scatter(x=g["date"], y=g["sma_50"], mode="lines", name="50-day SMA"))
    if "sma_200" in g.columns:
        fig.add_trace(go.Scatter(x=g["date"], y=g["sma_200"], mode="lines", name="200-day SMA"))
    if "exit_signal" in g.columns:
        exits = g[g["exit_signal"].fillna(False)]
        fig.add_trace(go.Scatter(
            x=exits["date"], y=exits["adjusted_close"], mode="markers", name="Exit warning",
            marker=dict(symbol="triangle-down", size=9),
        ))
    if "reentry_signal" in g.columns:
        entries = g[g["reentry_signal"].fillna(False)]
        fig.add_trace(go.Scatter(
            x=entries["date"], y=entries["adjusted_close"], mode="markers", name="Re-entry signal",
            marker=dict(symbol="triangle-up", size=9),
        ))
    fig.update_layout(title=f"{ticker} price, moving averages, and signal markers", height=520, hovermode="x unified")
    return fig


def confusion_matrix_figure(metrics: dict[str, Any], title: str) -> go.Figure:
    matrix = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Actual no burst", "Actual burst"],
        columns=["Predicted no burst", "Predicted burst"],
    )
    fig = px.imshow(matrix, text_auto=True, title=title)
    fig.update_layout(height=380)
    return fig


def roc_curve_figure(curves: list[dict[str, Any]], title: str) -> go.Figure:
    """Plot one or more ROC curves on a single figure.

    Each entry in `curves` is `{ 'name': str, 'curve': DataFrame[fpr,tpr], 'auc': float }`.
    The chance diagonal is added once at the end.
    """
    fig = go.Figure()
    for c in curves:
        fig.add_trace(go.Scatter(
            x=c["curve"]["fpr"], y=c["curve"]["tpr"], mode="lines",
            name=f"{c['name']} (AUC={c['auc']:.3f})",
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="False positive rate", yaxis_title="True positive rate", height=480)
    return fig


def calibration_curve_figure(cal: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cal["mean_predicted_probability"], y=cal["empirical_event_rate"],
        mode="markers+lines", name="Observed",
    ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Empirical event rate",
        height=420,
    )
    return fig


def grouped_model_comparison_bar(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    color: str | None = "asset_segment",
) -> go.Figure:
    """Grouped (not stacked) bar chart for model comparisons.

    Stacking would imply the metric adds across scopes/segments, which is
    statistically wrong. This helper enforces grouping.
    """
    df_sorted = df.sort_values(metric, ascending=False).copy()
    color_arg = color if color and color in df_sorted.columns else None
    fig = px.bar(df_sorted, x="model", y=metric, color=color_arg, barmode="group", title=title)
    fig.update_layout(height=420)
    return fig


def bucket_event_rate_lineplot(
    bucket_df: pd.DataFrame,
    *,
    bucket_order: list[str],
    title: str,
) -> go.Figure:
    """One line per model showing event rate by risk bucket.

    A line chart answers the actual question — *does each model's empirical
    event rate rise as the bucket gets riskier?* — and avoids the misleading
    impression of stacked bar charts.
    """
    fig = px.line(
        bucket_df,
        x="risk_bucket",
        y="bucket_event_rate",
        color="model",
        markers=True,
        facet_col="model_scope" if "model_scope" in bucket_df.columns else None,
        category_orders={"risk_bucket": bucket_order},
        title=title,
    )
    fig.update_layout(height=520, yaxis_tickformat=".0%")
    return fig


def top_k_lift_bar(top_df: pd.DataFrame, title: str) -> go.Figure:
    """Grouped top-5% / top-10% lift bars — never stacked."""
    fig = px.bar(
        top_df.sort_values("lift_vs_base_rate", ascending=False),
        x="model",
        y="lift_vs_base_rate",
        color="top_label" if "top_label" in top_df.columns else None,
        facet_col="model_scope" if "model_scope" in top_df.columns else None,
        barmode="group",
        title=title,
    )
    fig.update_layout(height=520)
    return fig


def drawdown_area(g: pd.DataFrame) -> go.Figure:
    fig = px.area(g, x="date", y="drawdown_pct", title="Drawdown from trailing peak (%)")
    fig.update_layout(height=360)
    return fig


def weekly_rsi_line(g: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["date"], y=g["weekly_rsi"], mode="lines", name="Weekly RSI"))
    fig.add_hline(y=75, line_dash="dash", annotation_text="Hot zone")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold zone")
    fig.update_layout(title="Weekly RSI", height=360)
    return fig


def feature_importance_bar(imp: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(
        imp.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        title=title,
    )
    fig.update_layout(height=max(420, len(imp) * 26))
    return fig
