"""Streamlit dashboard — Economic Bubble Monitoring & Investment Strategy.

This file is the **page router** for the dashboard. Each page renders one
section of the 14-step narrative laid out in `plan.md`:

    1.  Executive Summary / Story
    2.  Data Overview & EDA
    3.  Bubble Explorer
    4.  Historical Bubble Comparison
    5.  Bubble Vital Signs
    6.  Current AI Cycle Monitor
    7.  Machine Learning Model Lab
    8.  Rare Event & Imbalance Lab
    9.  Event-Level Validation
    10. Hazard & Poisson Models
    11. Investment / Strategy Simulator
    12. Methods & Diagrams
    13. Literature / Background Research
    14. Data Quality & Limitations

Helper logic — loaders, formatters, ROC math, missing-artifact notices,
explanations, chart builders, the regime-badge classifier, and the signal-
confirmation engine — lives in:

- ``dashboard_helpers.py``
- ``dashboard_explanations.py``
- ``dashboard_visuals.py``
- ``interpretation_engine.py``

Run:
    streamlit run dashboard.py

Generate inputs (one-time, in order):
    python data_ingestion.py
    python feature_engineering.py
    python bubble_signals.py
    python backtesting.py
    python ml_dataset.py
    python model_training.py
    python rare_event_analysis.py
    python imbalance_experiments.py
    python event_validation.py
    python cost_threshold_analysis.py
    python firth_logistic_export.py
    python hazard_model.py
    python poisson_count_model.py
    python ml_inference.py
    python model_evaluation.py
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtesting import (
    STRATEGY_RULES,
    compare_strategies,
    prepare_backtest_frame,
    simulate_strategy,
    summarize_result,
)
from config import (
    AI_LEADERS,
    BUBBLES,
    MONTHLY_CONTRIBUTION_DEFAULT,
    STARTING_CAPITAL_DEFAULT,
)
from dashboard_explanations import (
    PROJECT_DISCLAIMER,
    project_purpose_md,
    what_we_can_and_cannot_answer_md,
    what_is_a_bubble_md,
    why_prediction_is_hard_md,
    global_vs_segment_md,
    model_families_md,
    why_simpler_can_beat_complex_md,
    why_accuracy_is_not_enough_md,
    why_pr_auc_matters_md,
    why_calibration_matters_md,
    why_probabilities_should_not_be_overinterpreted_md,
    why_rare_events_are_hard_md,
    why_big_row_count_is_misleading_md,
    smote_caveats_md,
    balanced_rf_vs_rf_md,
    why_event_validation_matters_md,
    why_rule_baselines_matter_md,
    signal_confirmation_md,
    best_model_depends_on_goal_md,
    full_metric_glossary_md,
    render_expander,
    render_project_disclaimer,
    render_global_vs_segment_expander,
    render_metric_glossary_expander,
    render_rare_event_caveats_expander,
    render_signal_confirmation_expander,
    render_best_model_by_goal_expander,
    render_calibration_caveat_expander,
)
from dashboard_helpers import (
    auc_from_curve,
    build_binary_diagnostics,
    choose_binary_target_column,
    choose_prediction_column,
    compute_roc_points,
    feature_theme,
    fmt_num,
    fmt_pct,
    format_pct,
    load_all_data,
    load_ml_summary_text,
    load_report_text,
    missing_artifact_notice,
    missing_artifacts_block,
    ml_dataset_target_columns,
    pretty_metric_label,
    require_core_data,
    risk_bucket_order,
    top_numeric_features,
)
from dashboard_visuals import (
    bucket_event_rate_lineplot,
    calibration_curve_figure,
    confusion_matrix_figure,
    drawdown_area,
    feature_importance_bar,
    grouped_model_comparison_bar,
    price_signal_chart,
    render_interpretation,
    render_regime_badge,
    roc_curve_figure,
    top_k_lift_bar,
    weekly_rsi_line,
)
from interpretation_engine import (
    build_research_story,
    build_signal_confirmation_table,
    classify_regime_badge,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dashboard")

st.set_page_config(
    page_title="Economic Bubble Monitoring Dashboard",
    page_icon="📈",
    layout="wide",
)


# =============================================================================
# Sidebar — narrative-grouped navigation. Each numbered section corresponds
# to one of the 14 narrative steps. Group separators (non-selectable) are
# implemented by using Markdown-styled labels in the radio. Streamlit does
# not support disabled radio items, so we route disabled-looking labels back
# to the previous page.
# =============================================================================
PAGES: list[tuple[str, str]] = [
    # (section_label, route_key)
    ("1. Executive Summary",                     "executive_summary"),
    ("2. Data Overview & EDA",                   "data_overview"),
    ("3. Bubble Explorer",                       "bubble_explorer"),
    ("4. Historical Bubble Comparison",          "historical_comparison"),
    ("5. Bubble Vital Signs",                    "vital_signs"),
    ("6. Current AI Cycle Monitor",              "ai_monitor"),
    ("7. Machine Learning Model Lab",            "ml_lab"),
    ("8. Rare Event & Imbalance Lab",            "rare_event_lab"),
    ("9. Event-Level Validation",                "event_validation"),
    ("10. Hazard & Poisson Models",              "hazard_poisson"),
    ("11. Investment / Strategy Simulator",      "investment_sim"),
    ("12. Methods & Diagrams",                   "methods"),
    ("13. Literature / Background Research",     "literature"),
    ("14. Data Quality & Limitations",           "data_quality"),
]


def sidebar_ticker_filters(signals: pd.DataFrame) -> tuple[str, str, pd.Timestamp, pd.Timestamp]:
    """Sidebar filters used by pages that need a ticker/date window."""
    bubble_name = st.sidebar.selectbox("Bubble period", list(BUBBLES.keys()))
    meta = BUBBLES[bubble_name]
    available = sorted(set(meta.get("tickers", [])).intersection(set(signals["ticker"].unique())))
    if not available:
        available = sorted(signals["ticker"].unique())
    ticker = st.sidebar.selectbox("Ticker / index", available)
    default_start = pd.Timestamp(meta["start"])
    default_end = pd.Timestamp(meta["end"]) if meta.get("end") else signals["date"].max()
    ticker_rows = signals[signals["ticker"].eq(ticker)]
    min_date = max(ticker_rows["date"].min(), default_start)
    max_date = min(ticker_rows["date"].max(), default_end)
    start_date, end_date = st.sidebar.date_input(
        "Date window",
        value=(min_date.date(), max_date.date()),
        min_value=ticker_rows["date"].min().date(),
        max_value=ticker_rows["date"].max().date(),
    )
    return bubble_name, ticker, pd.Timestamp(start_date), pd.Timestamp(end_date)


# =============================================================================
# 1. EXECUTIVE SUMMARY
# =============================================================================
def page_executive_summary(data: dict[str, pd.DataFrame]) -> None:
    st.title("Economic Bubble Monitoring & Investment Strategy Dashboard")
    render_project_disclaimer()

    st.markdown(project_purpose_md())
    st.markdown(what_we_can_and_cannot_answer_md())

    # Top-line dataset counters.
    signals = data.get("signals", pd.DataFrame())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers loaded", f"{signals['ticker'].nunique():,}" if not signals.empty else "0")
    c2.metric("Daily rows",     f"{len(signals):,}"                  if not signals.empty else "0")
    c3.metric("First date",     str(signals["date"].min().date())    if not signals.empty else "n/a")
    c4.metric("Latest date",    str(signals["date"].max().date())    if not signals.empty else "n/a")

    if not data.get("valuations", pd.DataFrame()).empty:
        st.success(
            f"SEC historical valuation rows loaded: {len(data['valuations']):,}. "
            "Valuation warning signals are active for covered corporate tickers."
        )
    else:
        st.info(
            "SEC historical valuation rows are not loaded yet. "
            "Run `python data_ingestion.py` (without `--skip-sec`) or `python sec_fundamentals.py` to activate valuation signals."
        )

    # Dynamic research story.
    st.subheader("Current key findings")
    story = build_research_story(data)
    st.info(story["headline"])
    for bullet in story.get("bullets", []):
        st.markdown(f"- {bullet}")
    if not story.get("bullets"):
        st.warning(
            "No model artifacts found. Run the full pipeline (see README) to populate this page. "
            "The data overview, bubble explorer, and historical comparison pages can still run without ML artifacts."
        )

    render_best_model_by_goal_expander()

    render_expander("📖 What is a bubble?",                     what_is_a_bubble_md())
    render_expander("🧠 Why predicting bubble bursts is hard",  why_prediction_is_hard_md())
    render_global_vs_segment_expander()
    render_signal_confirmation_expander()
    render_calibration_caveat_expander()

    # Supporting tables compiled by the research-story engine.
    tables = story.get("tables", {})
    if tables:
        st.subheader("Key supporting tables")
        for name, df in tables.items():
            st.markdown(f"#### {name}")
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Bubble timeline for context.
    st.subheader("Bubble timeline")
    timeline_rows = []
    for name, meta in BUBBLES.items():
        timeline_rows.append({
            "Bubble": name,
            "Start": meta["start"],
            "Peak hint": meta.get("peak_hint") or "live / unknown",
            "End": meta.get("end") or "present / unresolved",
            "Asset class": meta["asset_class"],
            "Included tickers": ", ".join(meta.get("tickers", [])[:8]),
        })
    st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)
    render_interpretation(
        what_chart_shows="Calendar of historical bubble periods the project studies.",
        how_to_read="Each row is one regime. The 'AI / Mega-Cap Tech Cycle' is the only live, unresolved entry.",
        current_result="Eleven regimes are tracked. Live regimes are intentionally labelled 'unresolved' rather than predicted.",
        do_not_overclaim="A regime's presence here does not mean a bubble existed — labels are applied with hindsight where possible.",
    )


# =============================================================================
# 2. DATA OVERVIEW & EDA
# =============================================================================
def page_data_overview(data: dict[str, pd.DataFrame]) -> None:
    st.title("Data Overview & EDA")
    st.markdown(
        "Exploratory data analysis of the rule-based bubble dataset and the supervised ML dataset. "
        "The goal is to understand **coverage, missingness, class balance, distributions, and basic relationships** "
        "before trusting any model."
    )
    render_rare_event_caveats_expander()

    signals = data.get("signals", pd.DataFrame()).copy()
    ml_dataset = data.get("ml_dataset", pd.DataFrame()).copy()
    if signals.empty and ml_dataset.empty:
        st.error("No EDA source tables are loaded. Run `python data_ingestion.py` and `python feature_engineering.py`.")
        return

    tabs = st.tabs(["Rule-based signal EDA", "ML dataset EDA", "Correlation & missingness"])

    # --- Tab 1: rule-based signal EDA -----------------------------------------
    with tabs[0]:
        if signals.empty:
            missing_artifact_notice("signals")
        else:
            c1, c2 = st.columns(2)
            with c1:
                ticker_counts = (
                    signals.groupby("ticker", as_index=False).size()
                    .rename(columns={"size": "rows"}).sort_values("rows", ascending=False).head(20)
                )
                st.plotly_chart(px.bar(ticker_counts, x="ticker", y="rows", title="Top 20 tickers by signal-row count"), use_container_width=True)
            with c2:
                phase_col = "phase_label" if "phase_label" in signals.columns else "signal_label"
                phase_counts = (
                    signals[phase_col].fillna("Unknown").value_counts()
                    .rename_axis(phase_col).reset_index(name="rows")
                )
                st.plotly_chart(px.bar(phase_counts, x=phase_col, y="rows", title="Distribution of rule-based labels"), use_container_width=True)

            default_tickers = [t for t in ("NVDA", "QQQ", "BTC-USD") if t in set(signals["ticker"].dropna().unique())]
            selected_tickers = st.multiselect(
                "Tickers for warning-score timeline",
                sorted(signals["ticker"].dropna().unique()),
                default=default_tickers,
                key="eda_signal_tickers",
            )
            if selected_tickers:
                tmp = signals[signals["ticker"].isin(selected_tickers)].sort_values("date")
                fig = px.line(tmp, x="date", y="warning_score", color="ticker", title="Warning score over time for selected tickers")
                fig.add_hline(y=55, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)
                render_interpretation(
                    what_chart_shows="Rule-based warning score over time for each chosen ticker.",
                    how_to_read="Higher = more concurrent rule-based warnings firing. 55 is the exit-warning threshold.",
                    current_result="Compare which assets spent the longest time above the 55 line and during which historical periods.",
                    do_not_overclaim="The warning score is a transparent rule, not a model prediction.",
                )

    # --- Tab 2: ML dataset EDA -----------------------------------------------
    with tabs[1]:
        if ml_dataset.empty:
            missing_artifact_notice("ml_dataset")
        else:
            target_col, valid_flag = ml_dataset_target_columns(ml_dataset)
            if target_col and valid_flag and valid_flag in ml_dataset.columns:
                # Use the dataset's own validity flag — fixes the v16 bug where
                # the page silently dropped to empty when the legacy
                # `burst_6m` target was not present.
                valid = ml_dataset[ml_dataset[valid_flag] == True].copy()
            else:
                valid = ml_dataset.copy()

            c1, c2 = st.columns(2)
            with c1:
                if target_col and target_col in valid.columns:
                    target_counts = (
                        valid[target_col].fillna(-1).value_counts()
                        .rename_axis(target_col).reset_index(name="rows")
                    )
                    st.plotly_chart(
                        px.bar(target_counts, x=target_col, y="rows", title=f"Target balance in usable ML rows ({target_col})"),
                        use_container_width=True,
                    )
            with c2:
                if "asset_segment" in valid.columns:
                    seg_counts = (
                        valid["asset_segment"].fillna("Unknown").value_counts()
                        .rename_axis("asset_segment").reset_index(name="rows")
                    )
                    st.plotly_chart(
                        px.bar(seg_counts, x="asset_segment", y="rows", title="Rows by asset segment"),
                        use_container_width=True,
                    )

            if target_col and "asset_segment" in valid.columns:
                seg_rate = valid.groupby("asset_segment", as_index=False)[target_col].mean().rename(columns={target_col: "event_rate"})
                fig_rate = px.bar(seg_rate, x="asset_segment", y="event_rate", title="Historical major-drawdown event rate by segment")
                fig_rate.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_rate, use_container_width=True)
                render_interpretation(
                    what_chart_shows="Empirical share of weekly rows that were followed by a segment-adjusted major drawdown.",
                    how_to_read="Higher bars mean the model has more positive examples to learn from in that segment.",
                    current_result="Speculative segments usually have higher event rates; broad indexes have the rarest events.",
                    do_not_overclaim="A high event rate is not a forecast — it is a historical base rate the model must beat.",
                )

            feature_choices = top_numeric_features(valid, max_features=8, exclude=[target_col] if target_col else [])
            if feature_choices and target_col:
                chosen_feature = st.selectbox("Distribution feature", feature_choices, key="eda_dist_feature")
                fig = px.histogram(valid, x=chosen_feature, color=target_col, barmode="overlay", marginal="box",
                                   title=f"Distribution of {chosen_feature} by target")
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: correlation & missingness ------------------------------------
    with tabs[2]:
        base_df = ml_dataset if not ml_dataset.empty else signals
        if base_df.empty:
            st.info("No table is available for missingness/correlation analysis.")
        else:
            numeric_features = top_numeric_features(base_df, max_features=10, exclude=["burst_6m", "burst_6m_segment"])
            if numeric_features:
                corr = base_df[numeric_features].corr(numeric_only=True)
                fig_corr = px.imshow(corr, text_auto=".2f", title="Correlation heatmap for a high-coverage numeric subset")
                fig_corr.update_layout(height=520)
                st.plotly_chart(fig_corr, use_container_width=True)
            missing = base_df.isna().mean().sort_values(ascending=False).head(25).rename_axis("column").reset_index(name="missing_rate")
            st.plotly_chart(
                px.bar(missing, x="missing_rate", y="column", orientation="h", title="Top 25 columns by missingness"),
                use_container_width=True,
            )


# =============================================================================
# 3. BUBBLE EXPLORER
# =============================================================================
def page_bubble_explorer(data: dict[str, pd.DataFrame]) -> None:
    st.title("Bubble Explorer")
    signals = data["signals"]
    macro = data["macro"]
    bubble_name, ticker, start_date, end_date = sidebar_ticker_filters(signals)
    g = signals[(signals["ticker"].eq(ticker)) & (signals["date"].between(start_date, end_date))].copy()

    st.caption(BUBBLES[bubble_name]["notes"])
    if g.empty:
        st.error("No rows for this ticker/date window.")
        return

    latest = g.tail(1).iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest close",          f"${latest['adjusted_close']:,.2f}")
    c2.metric("Drawdown",              format_pct(latest.get("drawdown_pct")))
    c3.metric("Distance from 200DMA",  format_pct(latest.get("distance_from_200dma")))
    c4.metric("Weekly RSI",            f"{latest.get('weekly_rsi', np.nan):,.1f}")
    c5.metric("Warning score",         f"{latest.get('warning_score', np.nan):,.0f}/100")

    badge = classify_regime_badge(
        warning_score=latest.get("warning_score"),
        recovery_score=latest.get("recovery_score"),
        drawdown_pct=latest.get("drawdown_pct"),
        distance_from_200dma=latest.get("distance_from_200dma"),
        weekly_rsi=latest.get("weekly_rsi"),
        phase_label=latest.get("phase_label", latest.get("signal_label")),
    )
    render_regime_badge(badge, subtitle=f"{ticker} • latest week")

    st.plotly_chart(price_signal_chart(g, ticker), use_container_width=True)
    render_interpretation(
        what_chart_shows="Price, 50/200-day SMAs, and rule-based exit/re-entry markers.",
        how_to_read="Triangle-down = exit warning fired; triangle-up = re-entry signal. SMAs frame the trend.",
        current_result=f"{ticker} is currently flagged as **{badge}** by the rule-based engine.",
        do_not_overclaim="Exit/re-entry markers are rule outputs, not advice. They have backtested edge but also fail in ranging markets.",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(drawdown_area(g), use_container_width=True)
    with c2:
        st.plotly_chart(weekly_rsi_line(g), use_container_width=True)

    st.subheader("Macro overlay")
    if macro.empty:
        missing_artifact_notice("macro")
    else:
        indicators = sorted(macro["indicator_id"].unique())
        default = "FEDFUNDS" if "FEDFUNDS" in indicators else indicators[0]
        indicator = st.selectbox("Macro indicator", indicators, index=indicators.index(default))
        m = macro[(macro["indicator_id"].eq(indicator)) & (macro["date"].between(start_date, end_date))]
        if not m.empty:
            st.plotly_chart(
                px.line(m, x="date", y="value", title=f"{indicator}: {m['indicator_name'].iloc[0]}"),
                use_container_width=True,
            )

    st.subheader("Latest signal details")
    cols = [c for c in [
        "date", "ticker", "signal_label", "phase_label", "warning_score", "recovery_score",
        "drawdown_pct", "distance_from_200dma", "weekly_rsi", "volatility_30d",
        "return_126d", "return_252d",
    ] if c in g.columns]
    st.dataframe(g[cols].tail(20), use_container_width=True, hide_index=True)


# =============================================================================
# 4. HISTORICAL BUBBLE COMPARISON
# =============================================================================
def _align_by_peak(signals: pd.DataFrame, selections: dict[str, str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for bubble_name, ticker in selections.items():
        meta = BUBBLES[bubble_name]
        start = pd.Timestamp(meta["start"])
        end = pd.Timestamp(meta["end"]) if meta.get("end") else signals["date"].max()
        g = signals[(signals["ticker"].eq(ticker)) & (signals["date"].between(start, end))].copy()
        if g.empty:
            continue
        peak_idx = g["adjusted_close"].idxmax()
        peak_date = g.loc[peak_idx, "date"]
        peak_price = g.loc[peak_idx, "adjusted_close"]
        g["relative_day"] = (g["date"] - peak_date).dt.days
        g["normalized_to_peak"] = g["adjusted_close"] / peak_price * 100.0
        g["bubble"] = bubble_name
        rows.append(g[["bubble", "ticker", "date", "relative_day", "normalized_to_peak", "drawdown_pct", "phase_label"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def page_historical_comparison(data: dict[str, pd.DataFrame]) -> None:
    st.title("Historical Bubble Comparison")
    signals = data["signals"]
    st.markdown(
        "Normalize each selected asset to **100 at its retrospective peak** inside the configured bubble window. "
        "This is the canonical way to overlay bubble shapes from different eras."
    )

    selections: dict[str, str] = {}
    default_bubbles = ["Dot-com Bubble", "Housing / Financial Crisis", "Crypto / High-Growth Tech Bubble", "AI / Mega-Cap Tech Cycle"]
    for bubble_name in default_bubbles:
        if bubble_name not in BUBBLES:
            continue
        meta = BUBBLES[bubble_name]
        valid = sorted(set(meta["tickers"]).intersection(signals["ticker"].unique()))
        if valid:
            default = "^IXIC" if "^IXIC" in valid else valid[0]
            selections[bubble_name] = st.selectbox(
                f"{bubble_name} asset", valid,
                index=valid.index(default) if default in valid else 0,
            )

    aligned = _align_by_peak(signals, selections)
    if aligned.empty:
        st.error("No aligned data available for the chosen tickers.")
        return

    fig = px.line(
        aligned, x="relative_day", y="normalized_to_peak", color="bubble",
        title="Normalized price paths aligned by retrospective peak date",
        labels={"relative_day": "Days from peak", "normalized_to_peak": "Price index (peak = 100)"},
    )
    fig.add_vline(x=0, line_dash="dash", annotation_text="Peak")
    fig.update_layout(height=560, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    render_interpretation(
        what_chart_shows="Each bubble's selected asset normalized so its retrospective peak = 100.",
        how_to_read="Time 0 is the peak. Curves to the left show the run-up; curves to the right show the decline.",
        current_result="Live bubbles (e.g. AI / Mega-Cap Tech) keep extending — the 'peak' is provisional and shifts as new highs arrive.",
        do_not_overclaim="Peak alignment uses hindsight. A live regime's peak label is provisional and may shift.",
    )

    summary = (
        aligned.groupby("bubble").agg(
            ticker=("ticker", "first"),
            min_normalized=("normalized_to_peak", "min"),
            worst_drawdown_pct=("drawdown_pct", "min"),
            days_observed=("relative_day", "count"),
        ).reset_index()
    )
    summary["crash_depth_from_peak_pct"] = summary["min_normalized"] - 100
    st.dataframe(summary, use_container_width=True, hide_index=True)


# =============================================================================
# 5. BUBBLE VITAL SIGNS
# =============================================================================
def page_vital_signs(data: dict[str, pd.DataFrame]) -> None:
    st.title("Bubble Vital Signs")
    st.markdown(
        "A single-asset scorecard combining technical and rule-based indicators with an "
        "interpretable **regime badge**. SEC valuation history is shown when available."
    )
    signals = data["signals"]
    valuations = (
        data["valuation_features"] if not data.get("valuation_features", pd.DataFrame()).empty
        else data["valuations"]
    )

    tickers = sorted(signals["ticker"].unique())
    default_idx = tickers.index("QQQ") if "QQQ" in tickers else 0
    ticker = st.selectbox("Ticker", tickers, index=default_idx)
    g = signals[signals["ticker"].eq(ticker)].sort_values("date")
    latest = g.tail(1).iloc[0]

    badge = classify_regime_badge(
        warning_score=latest.get("warning_score"),
        recovery_score=latest.get("recovery_score"),
        drawdown_pct=latest.get("drawdown_pct"),
        distance_from_200dma=latest.get("distance_from_200dma"),
        weekly_rsi=latest.get("weekly_rsi"),
        phase_label=latest.get("phase_label", latest.get("signal_label")),
    )
    render_regime_badge(badge, subtitle=f"{ticker} • latest week")

    st.subheader("Technical vital signs")
    vital = pd.DataFrame(
        [
            ["Warning score",            latest.get("warning_score"),        "0-100 transparent rule score"],
            ["Recovery score",           latest.get("recovery_score"),       "0-100 transparent rule score"],
            ["Drawdown from peak",       latest.get("drawdown_pct"),         "Lower means deeper crash"],
            ["Distance from 200DMA",     latest.get("distance_from_200dma"), "High positive values can show stretched momentum"],
            ["Weekly RSI",               latest.get("weekly_rsi"),           ">75 hot; <30 oversold"],
            ["30-day volatility",        latest.get("volatility_30d"),       "Annualized volatility"],
            ["126-day return",           latest.get("return_126d"),          "Six-month momentum proxy"],
            ["252-day return",           latest.get("return_252d"),          "One-year momentum proxy"],
        ],
        columns=["Metric", "Latest value", "Interpretation"],
    )
    st.dataframe(vital, use_container_width=True, hide_index=True)

    render_expander(
        "🩺 What each vital sign means and how to read it",
        (
            "- **Warning score** — concurrent rule firings (valuation stretch, momentum, macro). Higher = more rules say *risk*.\n"
            "- **Recovery score** — the mirror image. Rises when oversold/stabilising conditions are present.\n"
            "- **Drawdown** — how far below the trailing peak. Used to define the supervised label.\n"
            "- **Distance from 200DMA** — proxy for trend stretch. Very positive values are historically associated with euphoria.\n"
            "- **Weekly RSI** — overbought/oversold momentum. >75 is hot, <30 is oversold.\n"
            "- **30-day volatility** — annualised. A spike usually accompanies regime change.\n"
            "- **126/252-day returns** — momentum proxies. Useful in combination with valuation, not alone."
        ),
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(g.tail(756), x="date", y="warning_score", title="Warning score, trailing ~3 years")
        fig.add_hline(y=55, line_dash="dash", annotation_text="Exit-warning threshold")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(g.tail(756), x="date", y="recovery_score", title="Recovery score, trailing ~3 years")
        fig.add_hline(y=45, line_dash="dash", annotation_text="Recovery threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("SEC historical valuation vital signs")
    if valuations.empty or "ticker" not in valuations.columns or ticker not in valuations["ticker"].values:
        st.info(
            "No SEC historical valuation row for this ticker. That usually means the asset is an ETF/index/crypto, "
            "the ticker lacks SEC CIK coverage, or EDGAR did not expose a clean revenue/share concept."
        )
    else:
        vt = valuations[valuations["ticker"].eq(ticker)].sort_values("date").copy()
        cols = [c for c in [
            "date", "filing_date", "period_end", "price_to_sales", "ps_historical_median",
            "ps_to_historical_median", "pe_ratio", "ev_to_sales", "revenue_ttm",
            "revenue_ttm_yoy_pct", "valuation_warning_score", "source",
        ] if c in vt.columns]
        st.caption("Rows are dated to the first trading day **after** the SEC filing date to reduce look-ahead bias.")
        st.dataframe(vt[cols].tail(40), use_container_width=True, hide_index=True)
        if "price_to_sales" in vt.columns:
            fig_v = px.line(vt, x="date", y="price_to_sales", title=f"{ticker} SEC-derived historical Price/Sales")
            if "ps_historical_median" in vt.columns:
                fig_v.add_trace(go.Scatter(x=vt["date"], y=vt["ps_historical_median"], mode="lines", name="Expanding historical median"))
            st.plotly_chart(fig_v, use_container_width=True)


# =============================================================================
# 6. CURRENT AI CYCLE MONITOR
# =============================================================================
def page_ai_monitor(data: dict[str, pd.DataFrame]) -> None:
    st.title("Current AI Cycle Monitor")
    st.warning(
        "Status labels are rule-based historical-risk labels combined with ML resemblance scores. "
        "They are not predictions and not buy/sell advice."
    )
    ai = data.get("ai_monitor", pd.DataFrame())
    if ai.empty:
        missing_artifact_notice("ai_monitor", severity="error")
        return

    # Signal-confirmation summary (combines rule, ML, hazard).
    st.subheader("Signal confirmation across rule, ML, and hazard layers")
    confirmation = build_signal_confirmation_table(data)
    if confirmation.empty:
        st.info("Run `python ml_inference.py` and `python hazard_model.py` to populate the cross-signal confirmation table.")
    else:
        display_cols = [c for c in [
            "ticker", "regime_badge", "signal_confirmation_level", "signal_confirmation_score",
            "confirmation_reasons", "ml_risk_category", "ml_empirical_bucket_rate", "ml_raw_score",
            "hazard_risk_category", "hazard_score", "phase_label", "warning_score", "recovery_score",
        ] if c in confirmation.columns]
        st.dataframe(
            confirmation[display_cols].sort_values(["signal_confirmation_score", "ticker"], ascending=[False, True]),
            use_container_width=True, hide_index=True,
        )
        render_interpretation(
            what_chart_shows="Per-ticker confluence of rule-based, ML, and hazard signals.",
            how_to_read="`signal_confirmation_level` = None/Low/Medium/High depending on how many layers are simultaneously elevated.",
            current_result="Use this as the primary 'are multiple things saying the same thing?' view.",
            do_not_overclaim="High confirmation ≠ a forecast. It means historical resemblance signals agree right now.",
        )
    render_signal_confirmation_expander()

    st.subheader("Rule-based AI monitor")
    st.dataframe(ai, use_container_width=True, hide_index=True)

    fig = px.scatter(
        ai,
        x="distance_from_200dma", y="weekly_rsi",
        size="warning_score", color="status",
        hover_name="ticker",
        title="AI-exposed assets: momentum heat map",
        labels={"distance_from_200dma": "Distance from 200DMA (%)", "weekly_rsi": "Weekly RSI"},
    )
    fig.add_hline(y=75, line_dash="dash")
    fig.add_vline(x=50, line_dash="dash")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    render_interpretation(
        what_chart_shows="Each AI-exposed asset positioned by its trend stretch (x) and momentum (y).",
        how_to_read="Top-right = stretched + hot. Bubble size = warning-score magnitude. Color = rule status.",
        current_result="Assets clustering top-right are in the historically higher-risk corner.",
        do_not_overclaim="Top-right does not mean a drawdown is imminent. Stretched-and-hot conditions can persist for months.",
    )

    valuations = (
        data["valuation_features"] if not data.get("valuation_features", pd.DataFrame()).empty
        else data["valuations"]
    )
    if not valuations.empty and "ticker" in valuations.columns:
        ai_val = valuations[valuations["ticker"].isin(AI_LEADERS)].copy()
        if not ai_val.empty:
            latest_sec = ai_val.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
            st.subheader("Latest SEC-derived historical valuation rows")
            st.caption("Filing-date-aware rows, not scraped current snapshots.")
            sec_cols = [c for c in [
                "date", "ticker", "filing_date", "period_end", "price_to_sales",
                "ps_to_historical_median", "ev_to_sales", "revenue_ttm_yoy_pct",
                "valuation_warning_score", "source",
            ] if c in latest_sec.columns]
            st.dataframe(latest_sec[sec_cols], use_container_width=True, hide_index=True)

    snapshot = data.get("valuation_snapshot", pd.DataFrame())
    if not snapshot.empty and "ticker" in snapshot.columns:
        snap = snapshot[snapshot["ticker"].isin(AI_LEADERS)].copy()
        if not snap.empty:
            st.subheader("Current valuation snapshot (live context only)")
            st.caption("yfinance snapshot, used for context only — NOT historical evidence.")
            st.dataframe(snap, use_container_width=True, hide_index=True)

    status_counts = ai["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    st.subheader("Status distribution")
    st.plotly_chart(px.bar(status_counts, x="Status", y="Count", title="AI monitor status counts"), use_container_width=True)


# =============================================================================
# 7. MACHINE LEARNING MODEL LAB
# =============================================================================
def page_ml_lab(data: dict[str, pd.DataFrame]) -> None:
    st.title("Machine Learning Model Lab")
    st.warning(
        "Educational research tool only. Models estimate **historical forward drawdown-risk resemblance**, "
        "not exact crash dates."
    )
    st.markdown(
        "Primary target: `burst_6m_segment = 1` if the asset crossed a **segment-adjusted major-drawdown "
        "threshold over the next 26 weekly observations**. Segment-specific thresholds are used because a 20% "
        "decline in SPY/QQQ and a 50% decline in BTC/ARKK do not mean the same thing."
    )

    required = {
        "ML dataset":          data.get("ml_dataset", pd.DataFrame()),
        "Model results":       data.get("ml_results", pd.DataFrame()),
        "Model predictions":   data.get("ml_predictions", pd.DataFrame()),
        "Feature importance":  data.get("ml_feature_importance", pd.DataFrame()),
    }
    if missing_artifacts_block(required, section_label="the ML lab"):
        return

    render_metric_glossary_expander()
    render_global_vs_segment_expander()
    render_best_model_by_goal_expander()
    render_calibration_caveat_expander()
    render_expander("📚 Why simpler models often beat complex ones", why_simpler_can_beat_complex_md())
    render_expander("⚖️ Why accuracy alone is not enough",            why_accuracy_is_not_enough_md())
    render_expander("📈 Why PR-AUC matters for rare events",         why_pr_auc_matters_md())
    render_expander("🧰 Model families in this project",             model_families_md())

    results = data["ml_results"].copy()
    predictions = data["ml_predictions"].copy()
    importance = data["ml_feature_importance"].copy()
    thresholds = data.get("ml_thresholds", pd.DataFrame())
    top_decile = data.get("ml_top_decile", pd.DataFrame())
    calibration = data.get("ml_calibration", pd.DataFrame())
    conclusions = data.get("ml_conclusions", pd.DataFrame())

    tabs = st.tabs([
        "Executive model conclusion",
        "Model comparison",
        "Top-decile risk",
        "Thresholds & calibration",
        "Confusion matrices",
        "ROC curves",
        "False-positive audit",
        "Feature importance",
        "Plain-English summary",
    ])

    # --- Tab: executive model conclusion -------------------------------------
    with tabs[0]:
        test_results = results[results["split"].eq("test")].copy() if "split" in results.columns else pd.DataFrame()
        if not test_results.empty:
            best_cols = st.columns(4)
            for col_obj, metric, label in zip(
                best_cols,
                ["roc_auc", "pr_auc", "recall", "precision"],
                ["Best ROC-AUC", "Best PR-AUC", "Best Recall", "Best Precision"],
            ):
                if metric in test_results.columns and test_results[metric].notna().any():
                    row = test_results.sort_values(metric, ascending=False).iloc[0]
                    scope = row.get("asset_segment", row.get("model_scope", "global"))
                    col_obj.metric(label, f"{row[metric]:.3f}", help=f"{row['model']} / {scope}")
                    col_obj.caption(f"{row['model']} / {scope}")
                else:
                    col_obj.metric(label, "n/a")
        if not conclusions.empty:
            strength = conclusions[conclusions["metric"].eq("overall_strength")]
            explanation = conclusions[conclusions["metric"].eq("overall_strength_explanation")]
            if not strength.empty:
                st.info(f"**Overall strength:** {strength.iloc[0]['model']}")
            if not explanation.empty:
                st.write(str(explanation.iloc[0]["model"]))
        else:
            st.info("Run `python model_evaluation.py` to generate the explicit strength conclusion table.")

    # --- Tab: model comparison -----------------------------------------------
    with tabs[1]:
        st.caption(
            "Metrics from chronological validation/test splits. Rows were not randomly shuffled. "
            "Prediction thresholds are tuned on validation, not hard-coded to 0.50."
        )
        metric_cols = [c for c in [
            "model", "asset_segment", "split", "threshold_value", "n_rows", "positive_rate",
            "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier_score",
            "true_negative", "false_positive", "false_negative", "true_positive",
        ] if c in results.columns]
        sort_cols = [c for c in ["split", "asset_segment", "pr_auc"] if c in results.columns]
        if sort_cols:
            ascending = [True] * (len(sort_cols) - 1) + [False]
            st.dataframe(results[metric_cols].sort_values(sort_cols, ascending=ascending),
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(results[metric_cols], use_container_width=True, hide_index=True)

        test_results = results[results["split"].eq("test")].copy() if "split" in results.columns else pd.DataFrame()
        if not test_results.empty and {"model", "pr_auc", "roc_auc"}.issubset(test_results.columns):
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    grouped_model_comparison_bar(test_results, metric="pr_auc",
                                                 title="Test PR-AUC by model and scope"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    grouped_model_comparison_bar(test_results, metric="roc_auc",
                                                 title="Test ROC-AUC by model and scope"),
                    use_container_width=True,
                )
            render_interpretation(
                what_chart_shows="Test PR-AUC and ROC-AUC for every model × segment.",
                how_to_read="Bars are grouped (not stacked) because metrics do not add across scopes.",
                current_result="Identify the model that leads in *both* PR-AUC and ROC-AUC for the goal you care about.",
                do_not_overclaim="A small lead in AUC can vanish under event-level validation. See page 9.",
            )

    # --- Tab: top-decile risk ------------------------------------------------
    with tabs[2]:
        if top_decile.empty:
            missing_artifact_notice("ml_top_decile")
        else:
            td_cols = [c for c in [
                "model", "model_scope", "split", "base_event_rate", "top_decile_event_rate",
                "lift_vs_base_rate", "top_decile_rows", "top_decile_true_positives", "min_probability_top_decile",
            ] if c in top_decile.columns]
            st.dataframe(
                top_decile[top_decile["split"].eq("test")][td_cols].sort_values("lift_vs_base_rate", ascending=False),
                use_container_width=True, hide_index=True,
            )
            td_test = top_decile[top_decile["split"].eq("test")].copy()
            if not td_test.empty:
                fig_td = px.bar(
                    td_test.sort_values("lift_vs_base_rate", ascending=False),
                    x="model", y="lift_vs_base_rate", color="model_scope",
                    barmode="group", title="Top-decile lift vs base event rate, grouped by scope",
                )
                fig_td.update_layout(height=440)
                st.plotly_chart(fig_td, use_container_width=True)
                render_interpretation(
                    what_chart_shows="How much more likely a major drawdown was when the model placed a week in its top 10% of scores.",
                    how_to_read="A lift of 2.0 means the top 10% had twice the base-rate event frequency.",
                    current_result="Higher bars = stronger risk ranking. Inspect the best model × scope combination for your decision goal.",
                    do_not_overclaim="Lift values are historical. They are not a probability for the *next* drawdown.",
                )

    # --- Tab: thresholds & calibration ---------------------------------------
    with tabs[3]:
        if thresholds.empty:
            missing_artifact_notice("ml_thresholds")
        else:
            st.subheader("Tuned thresholds")
            st.markdown(
                "0.50 is **not** a magic cutoff. Each model's alert threshold is selected on validation by maximising "
                "F1 (with rare-event guardrails) and then applied to the later test period."
            )
            st.dataframe(thresholds.sort_values(["model_scope", "model"]),
                         use_container_width=True, hide_index=True)

        if calibration.empty:
            missing_artifact_notice("ml_calibration")
        else:
            st.subheader("Calibration / reliability")
            cal_splits = sorted(calibration["split"].dropna().unique())
            default_split = "test" if "test" in cal_splits else cal_splits[0]
            cal_split = st.selectbox("Calibration split", cal_splits,
                                     index=cal_splits.index(default_split), key="cal_split")
            cal_models = sorted(calibration["model"].dropna().unique())
            cal_model = st.selectbox("Calibration model", cal_models, key="cal_model")
            cal_scopes = sorted(calibration["model_scope"].dropna().unique())
            cal_scope = st.selectbox("Calibration scope", cal_scopes,
                                     index=cal_scopes.index("global") if "global" in cal_scopes else 0,
                                     key="cal_scope")
            cal = calibration[
                (calibration["split"].eq(cal_split))
                & (calibration["model"].eq(cal_model))
                & (calibration["model_scope"].eq(cal_scope))
            ]
            if cal.empty:
                st.info("No calibration rows for this model/scope/split.")
            else:
                st.dataframe(cal, use_container_width=True, hide_index=True)
                st.plotly_chart(
                    calibration_curve_figure(cal, "Calibration curve by probability decile"),
                    use_container_width=True,
                )
                render_interpretation(
                    what_chart_shows="Mean predicted probability (x) vs empirical event rate (y) for each bucket.",
                    how_to_read="On the dashed line = perfectly calibrated. Above = under-predicting risk. Below = over-predicting risk.",
                    current_result="Prefer the empirical bucket rate over the raw score when calibration drifts from the diagonal.",
                    do_not_overclaim="A well-calibrated model can still be wrong about *which* events happen — calibration ≠ forecast accuracy.",
                )

    # --- Tab: confusion matrices ---------------------------------------------
    with tabs[4]:
        if predictions.empty:
            missing_artifact_notice("ml_predictions")
        else:
            splits = sorted(predictions["split"].dropna().unique()) if "split" in predictions.columns else []
            scopes = sorted(predictions["model_scope"].dropna().unique()) if "model_scope" in predictions.columns else []
            selected_split = st.multiselect("Prediction split(s)", splits,
                                            default=["test"] if "test" in splits else splits, key="diag_cm_splits")
            selected_scope = st.multiselect("Model scope(s)", scopes, default=scopes, key="diag_cm_scopes")
            subset = predictions.copy()
            if selected_split:
                subset = subset[subset["split"].isin(selected_split)]
            if selected_scope and "model_scope" in subset.columns:
                subset = subset[subset["model_scope"].isin(selected_scope)]
            combos = subset[["model", "model_scope", "split"]].drop_duplicates().sort_values(["split", "model_scope", "model"])
            if combos.empty:
                st.info("No confusion-matrix combinations match the current filter.")
            else:
                st.caption("Each matrix uses the tuned threshold saved with the prediction artifact.")
                for split_name in selected_split or splits:
                    split_combos = combos[combos["split"].eq(split_name)]
                    if split_combos.empty:
                        continue
                    st.markdown(f"### Split: {split_name}")
                    cols = st.columns(2)
                    for idx, combo in enumerate(split_combos.to_dict("records")):
                        pred = subset[
                            (subset["model"].eq(combo["model"]))
                            & (subset["split"].eq(combo["split"]))
                            & (subset["model_scope"].eq(combo["model_scope"]))
                        ]
                        metrics = build_binary_diagnostics(pred)
                        if not metrics:
                            continue
                        with cols[idx % 2]:
                            title = f"{combo['model']} / {combo['model_scope']}"
                            st.plotly_chart(confusion_matrix_figure(metrics, title), use_container_width=True)
                            st.caption(
                                f"TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']} | "
                                f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
                                f"FPR={metrics['false_positive_rate']:.3f}"
                            )

    # --- Tab: ROC curves -----------------------------------------------------
    with tabs[5]:
        if predictions.empty or "probability" not in predictions.columns:
            st.info("Prediction probabilities are missing, so ROC curves cannot be drawn.")
        else:
            available_splits = sorted(predictions["split"].dropna().unique())
            split_choice = st.selectbox("ROC split", available_splits,
                                        index=available_splits.index("test") if "test" in available_splits else 0,
                                        key="roc_split")
            roc_subset = predictions[predictions["split"].eq(split_choice)].copy()
            available_scopes = sorted(roc_subset["model_scope"].dropna().unique()) if "model_scope" in roc_subset.columns else ["global"]
            scope_choice = st.multiselect("ROC scope(s)", available_scopes, default=available_scopes, key="roc_scopes")
            if scope_choice and "model_scope" in roc_subset.columns:
                roc_subset = roc_subset[roc_subset["model_scope"].isin(scope_choice)]
            if roc_subset.empty:
                st.info("No ROC data for this selection.")
            else:
                for scope_name in scope_choice or available_scopes:
                    scope_df = roc_subset[roc_subset["model_scope"].eq(scope_name)] if "model_scope" in roc_subset.columns else roc_subset
                    if scope_df.empty:
                        continue
                    curves = []
                    auc_rows = []
                    for model_name in sorted(scope_df["model"].dropna().unique()):
                        pred = scope_df[scope_df["model"].eq(model_name)]
                        target_col = choose_binary_target_column(pred)
                        if target_col is None:
                            continue
                        curve = compute_roc_points(pred[target_col], pred["probability"])
                        if curve.empty:
                            continue
                        auc_value = auc_from_curve(curve)
                        curves.append({"name": model_name, "curve": curve, "auc": auc_value})
                        auc_rows.append({"model": model_name, "model_scope": scope_name, "split": split_choice, "roc_auc_from_curve": auc_value})
                    if curves:
                        st.plotly_chart(
                            roc_curve_figure(curves, title=f"ROC curves: {scope_name} / {split_choice}"),
                            use_container_width=True,
                        )
                        st.dataframe(
                            pd.DataFrame(auc_rows).sort_values("roc_auc_from_curve", ascending=False),
                            use_container_width=True, hide_index=True,
                        )

    # --- Tab: false-positive audit ------------------------------------------
    with tabs[6]:
        test_results = results[results["split"].eq("test")].copy() if "split" in results.columns else results.copy()
        if test_results.empty:
            st.info("No model results available for false-positive auditing.")
        else:
            cols = [c for c in [
                "model", "asset_segment", "model_scope", "split", "false_positive",
                "false_positive_rate", "false_positives_per_true_positive",
                "precision", "recall", "alert_rate", "mcc", "threshold_value",
            ] if c in test_results.columns]
            audit_df = test_results[cols].copy()
            sort_cols = [c for c in ["false_positives_per_true_positive", "false_positive"] if c in audit_df.columns]
            if sort_cols:
                audit_df = audit_df.sort_values(sort_cols)
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                if {"model", "false_positive"}.issubset(audit_df.columns):
                    fig_fp = px.bar(
                        audit_df.sort_values("false_positive", ascending=False),
                        x="model", y="false_positive",
                        color="model_scope" if "model_scope" in audit_df.columns else None,
                        title="False positives by model",
                    )
                    fig_fp.update_layout(height=420)
                    st.plotly_chart(fig_fp, use_container_width=True)
            with c2:
                metric = "false_positives_per_true_positive" if "false_positives_per_true_positive" in audit_df.columns else "false_positive"
                fig_burden = px.bar(
                    audit_df.sort_values(metric, ascending=False),
                    x="model", y=metric,
                    color="model_scope" if "model_scope" in audit_df.columns else None,
                    title=f"{pretty_metric_label(metric)} by model",
                )
                fig_burden.update_layout(height=420)
                st.plotly_chart(fig_burden, use_container_width=True)
            render_interpretation(
                what_chart_shows="False-positive burden — total FPs and FPs per true positive — for each model on test.",
                how_to_read="Lower = fewer noisy alerts per real detection.",
                current_result="Models with very high recall often pay for it with FP/TP > 10.",
                do_not_overclaim="A model with low FP burden may simply be conservative; check it has non-trivial recall too.",
            )

    # --- Tab: feature importance ---------------------------------------------
    with tabs[7]:
        if importance.empty:
            st.info("Feature importance table is empty.")
        else:
            importance_scopes = sorted(importance["model_scope"].dropna().unique()) if "model_scope" in importance.columns else ["global"]
            imp_scope = st.selectbox(
                "Feature-importance scope", importance_scopes,
                index=importance_scopes.index("global") if "global" in importance_scopes else 0,
                key="imp_scope",
            )
            imp_for_scope = importance[importance["model_scope"].eq(imp_scope)] if "model_scope" in importance.columns else importance
            model_for_importance = st.selectbox(
                "Feature-importance model",
                sorted(imp_for_scope["model"].dropna().unique()),
                key="imp_model",
            )
            top_n = st.slider("Number of features", 5, 30, 15, 5)
            imp = imp_for_scope[imp_for_scope["model"].eq(model_for_importance)].sort_values("importance", ascending=False).head(top_n)
            st.plotly_chart(
                feature_importance_bar(imp, title=f"Top features: {model_for_importance} / {imp_scope}"),
                use_container_width=True,
            )
            if "signed_value" in imp.columns and model_for_importance == "Logistic Regression":
                st.caption(
                    "Logistic Regression signed coefficients indicate direction after scaling. "
                    "Positive = raises estimated risk; negative = lowers it."
                )
            # Themed roll-up.
            themed = importance.copy()
            themed["theme"] = themed["feature"].map(feature_theme)
            agg = themed.groupby(["model_scope", "model", "theme"], as_index=False)["importance"].sum()
            fig = px.bar(agg, x="theme", y="importance", color="model", facet_col="model_scope",
                         barmode="group", title="Importance aggregated into interpretable feature themes")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab: plain-English summary -----------------------------------------
    with tabs[8]:
        summary_text = load_ml_summary_text()
        if summary_text:
            st.markdown(summary_text)
        else:
            missing_artifact_notice("ml_conclusions", script_hint="model_evaluation.py")
        st.markdown(
            "**Bottom line.** This model stack does not predict the exact timing of a crash. It asks whether "
            "today's conditions *resemble* historical pre-drawdown regimes. High scores mean similar conditions "
            "were historically followed by major drawdowns more often than baseline — but they can still be wrong."
        )


# =============================================================================
# 8. RARE EVENT & IMBALANCE LAB
# =============================================================================
def page_rare_event_lab(data: dict[str, pd.DataFrame]) -> None:
    st.title("Rare Event & Imbalance Lab")
    st.markdown(
        "This page collects rare-event diagnostics: stricter thresholds, percentile risk buckets, top-k lift, "
        "rare-event calibration, imbalance experiments (Balanced RF, SMOTE/SMOTE-ENN), and cost-threshold stress tests."
    )
    render_rare_event_caveats_expander()
    render_expander("🧪 SMOTE / SMOTE-ENN: useful but easy to misread", smote_caveats_md())
    render_expander("🌳 Balanced Random Forest vs Random Forest",      balanced_rf_vs_rf_md())
    render_expander("📏 Why rule-based baselines matter",              why_rule_baselines_matter_md())

    rare_metrics = data.get("ml_rare_event_metrics", pd.DataFrame())
    rare_thresholds = data.get("ml_rare_event_thresholds", pd.DataFrame())
    risk_buckets = data.get("ml_risk_buckets", pd.DataFrame())
    top_k_lift = data.get("ml_top_k_lift", pd.DataFrame())
    rare_calibration = data.get("ml_rare_event_calibration", pd.DataFrame())
    imbalance_results = data.get("imbalance_results", pd.DataFrame())
    imbalance_topk = data.get("imbalance_topk", pd.DataFrame())
    cost_results = data.get("cost_threshold_results", pd.DataFrame())

    if all(df.empty for df in [rare_metrics, rare_thresholds, risk_buckets, top_k_lift, imbalance_results, cost_results]):
        st.error("Run `python rare_event_analysis.py`, `python imbalance_experiments.py`, and `python cost_threshold_analysis.py` to populate this page.")
        return

    tabs = st.tabs([
        "Rare-event thresholds & metrics",
        "Percentile risk buckets",
        "Top-5% / Top-10% lift",
        "Rare-event calibration",
        "Imbalance experiments (Balanced RF, SMOTE)",
        "Cost-ratio threshold stress test",
    ])

    with tabs[0]:
        st.markdown(
            "Rare-event thresholds enforce minimum precision and a cap on flagged rows. This prevents fake "
            "improvements where recall rises only because the model labels nearly everything as risky."
        )
        if rare_thresholds.empty:
            missing_artifact_notice("ml_rare_event_thresholds")
        else:
            st.dataframe(rare_thresholds.sort_values(["model_scope", "model"]),
                         use_container_width=True, hide_index=True)
        if not rare_metrics.empty:
            st.markdown("**Rare-event metrics using constrained thresholds:**")
            st.dataframe(
                rare_metrics[rare_metrics["split"].eq("test")].sort_values("pr_auc", ascending=False),
                use_container_width=True, hide_index=True,
            )

    with tabs[1]:
        if risk_buckets.empty:
            missing_artifact_notice("ml_risk_buckets")
        else:
            bucket_test = risk_buckets[risk_buckets["split"].eq("test")].copy()
            st.dataframe(
                bucket_test.sort_values(["model_scope", "model", "risk_bucket"]),
                use_container_width=True, hide_index=True,
            )
            st.plotly_chart(
                bucket_event_rate_lineplot(
                    bucket_test,
                    bucket_order=risk_bucket_order(),
                    title="Empirical event rate by model risk bucket, test split",
                ),
                use_container_width=True,
            )
            render_interpretation(
                what_chart_shows="One line per model showing event rate across risk buckets (Bottom 50% → Top 5%).",
                how_to_read="A useful model has lines that rise monotonically from left to right.",
                current_result="Models whose lines flatten or invert are NOT producing useful risk rankings.",
                do_not_overclaim="A steep line is risk *enrichment*, not certainty. The top bucket still has many non-events.",
            )

    with tabs[2]:
        if top_k_lift.empty:
            missing_artifact_notice("ml_top_k_lift")
        else:
            top_test = top_k_lift[top_k_lift["split"].eq("test")].copy()
            st.dataframe(top_test.sort_values("lift_vs_base_rate", ascending=False),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(
                top_k_lift_bar(top_test, title="Top 10% and Top 5% lift vs base event rate (grouped, not stacked)"),
                use_container_width=True,
            )

    with tabs[3]:
        if rare_calibration.empty:
            missing_artifact_notice("ml_rare_event_calibration")
        else:
            rc_splits = sorted(rare_calibration["split"].dropna().unique())
            rc_split = st.selectbox("Calibration split", rc_splits,
                                    index=rc_splits.index("test") if "test" in rc_splits else 0,
                                    key="rare_cal_split")
            rc_models = sorted(rare_calibration["model"].dropna().unique())
            rc_model = st.selectbox("Calibration model", rc_models, key="rare_cal_model")
            rc_scopes = sorted(rare_calibration["model_scope"].dropna().unique())
            rc_scope = st.selectbox("Calibration scope", rc_scopes,
                                    index=rc_scopes.index("global") if "global" in rc_scopes else 0,
                                    key="rare_cal_scope")
            rc = rare_calibration[
                (rare_calibration["split"].eq(rc_split))
                & (rare_calibration["model"].eq(rc_model))
                & (rare_calibration["model_scope"].eq(rc_scope))
            ]
            if rc.empty:
                st.info("No rare-event calibration rows for this selection.")
            else:
                st.dataframe(rc, use_container_width=True, hide_index=True)
                st.plotly_chart(
                    calibration_curve_figure(rc, "Validation-calibrated reliability curve"),
                    use_container_width=True,
                )

    with tabs[4]:
        if imbalance_results.empty:
            missing_artifact_notice("imbalance_results")
        else:
            imb_test = imbalance_results[imbalance_results["split"].eq("test")].copy() if "split" in imbalance_results.columns else imbalance_results
            st.dataframe(imb_test.sort_values("pr_auc", ascending=False),
                         use_container_width=True, hide_index=True)
            if not imbalance_topk.empty:
                st.markdown("**Imbalance-method top-risk lift:**")
                topk = imbalance_topk[imbalance_topk["split"].eq("test")].copy() if "split" in imbalance_topk.columns else imbalance_topk
                st.dataframe(topk.sort_values("lift_vs_base_rate", ascending=False),
                             use_container_width=True, hide_index=True)
            report = load_report_text("imbalance_experiment_summary.md")
            if report:
                with st.expander("Imbalance experiment — written interpretation"):
                    st.markdown(report)

    with tabs[5]:
        if cost_results.empty:
            missing_artifact_notice("cost_threshold_results")
        else:
            cost_test = cost_results[cost_results["split"].eq("test")].copy() if "split" in cost_results.columns else cost_results
            st.dataframe(cost_test.sort_values("cost_per_row"),
                         use_container_width=True, hide_index=True)
            report = load_report_text("cost_threshold_summary.md")
            if report:
                with st.expander("Cost-threshold — written interpretation"):
                    st.markdown(report)


# =============================================================================
# 9. EVENT-LEVEL VALIDATION
# =============================================================================
def page_event_validation(data: dict[str, pd.DataFrame]) -> None:
    st.title("Event-Level Validation")
    st.markdown(
        "**Why this page exists.** Row-level chronological validation can overstate evidence because one historical "
        "crisis creates many correlated positive ticker-weeks. Event-level (leave-one-crisis-style) validation "
        "trains *before* a crisis and tests the pre-event/event window. Metrics are usually less flattering — which "
        "is exactly why this is the honest test."
    )
    render_expander("📚 Why event-level validation matters", why_event_validation_matters_md())

    event_results = data.get("event_validation_results", pd.DataFrame())
    event_topk = data.get("event_validation_topk", pd.DataFrame())

    if event_results.empty:
        missing_artifact_notice("event_validation_results", severity="error")
        return

    st.subheader("Event-level metrics by historical regime")
    st.dataframe(
        event_results.sort_values(["event_name", "pr_auc"], ascending=[True, False]),
        use_container_width=True, hide_index=True,
    )

    if "pr_auc" in event_results.columns and "event_name" in event_results.columns:
        fig = px.bar(
            event_results.sort_values("pr_auc", ascending=False),
            x="event_name", y="pr_auc",
            color="model" if "model" in event_results.columns else None,
            barmode="group", title="Event-level PR-AUC by historical regime",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        render_interpretation(
            what_chart_shows="PR-AUC per historical regime when trained on data preceding that regime.",
            how_to_read="Tall bars = the model generalised to that regime. Short bars = it did not.",
            current_result="Look for regimes where every model is weak — those are honest blind spots.",
            do_not_overclaim="A model that handled one historical crisis well is not guaranteed to handle the next.",
        )

    if not event_topk.empty:
        st.subheader("Event-level top-risk lift")
        st.dataframe(
            event_topk.sort_values("lift_vs_base_rate", ascending=False),
            use_container_width=True, hide_index=True,
        )

    report = load_report_text("event_validation_summary.md")
    if report:
        with st.expander("Written interpretation"):
            st.markdown(report)


# =============================================================================
# 10. HAZARD & POISSON MODELS
# =============================================================================
def page_hazard_poisson(data: dict[str, pd.DataFrame]) -> None:
    st.title("Hazard & Poisson Models")
    st.markdown(
        "Two complementary event-style models:\n\n"
        "- **Discrete-time hazard model**: 'did the asset enter a major drawdown state within the next 4 weeks?'\n"
        "- **Poisson count model**: 'how many assets in a segment may experience a drawdown event?'\n\n"
        "Both are experimental relative to the main supervised stack."
    )

    hazard_results = data.get("hazard_results", pd.DataFrame())
    hazard_audit = data.get("hazard_audit", pd.DataFrame())
    hazard_current = data.get("hazard_current_ai", pd.DataFrame())
    poisson_results = data.get("poisson_results", pd.DataFrame())
    poisson_current = data.get("poisson_current", pd.DataFrame())

    tabs = st.tabs(["Hazard model + leakage audit", "Poisson count model"])

    with tabs[0]:
        st.markdown(
            "Because the 4-week question can be accidentally easy if the model sees current drawdown-state "
            "variables, **two feature sets are trained**: a full set and a *leakage-restricted* set without "
            "drawdown-state variables. Use the restricted result as the safer research estimate."
        )
        if hazard_results.empty:
            missing_artifact_notice("hazard_results", severity="error")
        else:
            st.dataframe(hazard_results, use_container_width=True, hide_index=True)
            if not hazard_audit.empty:
                st.subheader("Leakage audit")
                st.dataframe(hazard_audit, use_container_width=True, hide_index=True)
                if "audit_warning" in hazard_audit.columns and \
                   hazard_audit["audit_warning"].astype(str).str.contains("SUSPICIOUS", case=False, na=False).any():
                    st.warning(
                        "The full hazard model looks suspiciously stronger than the leakage-restricted model. "
                        "**Use restricted-feature hazard scores** for research interpretation."
                    )
            else:
                missing_artifact_notice("hazard_audit")
            if not hazard_current.empty:
                st.subheader("Current AI-exposed hazard scores")
                sort_col = "hazard_score_4w" if "hazard_score_4w" in hazard_current.columns else "hazard_probability_4w"
                st.dataframe(
                    hazard_current.sort_values(sort_col, ascending=False),
                    use_container_width=True, hide_index=True,
                )

    with tabs[1]:
        if poisson_results.empty:
            missing_artifact_notice("poisson_results", severity="error")
        else:
            st.dataframe(poisson_results, use_container_width=True, hide_index=True)
            if not poisson_current.empty:
                st.subheader("Current segment-level count forecast")
                st.dataframe(
                    poisson_current.sort_values("predicted_event_rate_next_6m", ascending=False),
                    use_container_width=True, hide_index=True,
                )
            render_interpretation(
                what_chart_shows="Poisson regression on counts of segment-level events per period.",
                how_to_read="Compare 'mean actual count' vs 'mean predicted count' and RMSE to gauge fit.",
                current_result="Current Poisson fit is weak — treat as a stress-count *experiment*, not a forecast.",
                do_not_overclaim="A Poisson count is not a probability for any individual ticker.",
            )


# =============================================================================
# 11. INVESTMENT / STRATEGY SIMULATOR
# =============================================================================
def page_investment_simulator(data: dict[str, pd.DataFrame]) -> None:
    st.title("Investment / Strategy Simulator")
    st.warning(
        "Backtests only. Retrospective peak strategies use hindsight and are labelled accordingly. "
        "**Nothing on this page is investment advice.**"
    )
    signals = data["signals"]
    bubble_name, ticker, start_date, end_date = sidebar_ticker_filters(signals)

    c1, c2, c3 = st.columns(3)
    starting_capital = c1.number_input("Starting capital", min_value=1.0,
                                       value=float(STARTING_CAPITAL_DEFAULT), step=500.0)
    monthly_contribution = c2.number_input("Monthly contribution", min_value=0.0,
                                            value=float(MONTHLY_CONTRIBUTION_DEFAULT), step=100.0)
    default_strategy = "sell_exit_buy_recovery" if "sell_exit_buy_recovery" in STRATEGY_RULES else list(STRATEGY_RULES.keys())[0]
    strategy = c3.selectbox("Primary strategy", list(STRATEGY_RULES.keys()),
                            index=list(STRATEGY_RULES.keys()).index(default_strategy))

    g = prepare_backtest_frame(signals, ticker, str(start_date.date()), str(end_date.date()))
    result = simulate_strategy(g, ticker, strategy, starting_capital, monthly_contribution)
    summary = summarize_result(result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ending value",  f"${summary['ending_value']:,.2f}")
    c2.metric("Total return",  format_pct(summary["total_return_pct"]))
    c3.metric("Max drawdown",  format_pct(summary["max_drawdown_pct"]))
    c4.metric("Sharpe",        f"{summary['sharpe_ratio']:,.2f}" if pd.notna(summary["sharpe_ratio"]) else "n/a")
    st.caption(summary["notes"])

    st.plotly_chart(
        px.line(result.equity_curve, x="date", y="portfolio_value", title=f"Equity curve: {ticker} / {strategy}"),
        use_container_width=True,
    )
    render_interpretation(
        what_chart_shows="Equity curve of the chosen strategy applied historically to the chosen ticker/window.",
        how_to_read="Y-axis is portfolio value over time. Flat regions = cash-out periods for risk-off strategies.",
        current_result="Compare to buy-and-hold on the same chart further down for missed-upside vs drawdown reduction.",
        do_not_overclaim="Past backtest performance is not a forecast. Rules that worked historically can break in new regimes.",
    )

    st.subheader("Compare strategies")
    default_comparison = [s for s in [
        "buy_and_hold", "sell_exit_buy_recovery", "cash_until_40pct_drawdown", "cash_until_reclaim_200dma",
    ] if s in STRATEGY_RULES]
    selected_strategies = st.multiselect("Strategies to compare", list(STRATEGY_RULES.keys()), default=default_comparison)
    if selected_strategies:
        comparison, curves = compare_strategies(
            signals, ticker=ticker, strategies=selected_strategies,
            start=str(start_date.date()), end=str(end_date.date()),
            starting_capital=starting_capital, monthly_contribution=monthly_contribution,
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        curve_rows: list[pd.DataFrame] = []
        for name, curve in curves.items():
            tmp = curve[["date", "portfolio_value"]].copy()
            tmp["strategy"] = name
            curve_rows.append(tmp)
        if curve_rows:
            all_curves = pd.concat(curve_rows, ignore_index=True)
            st.plotly_chart(
                px.line(all_curves, x="date", y="portfolio_value", color="strategy",
                        title="Strategy comparison equity curves"),
                use_container_width=True,
            )


# =============================================================================
# 12. METHODS & DIAGRAMS
# =============================================================================
def page_methods(data: dict[str, pd.DataFrame]) -> None:
    st.title("Methods, Diagrams, and Formula Walkthrough")
    st.markdown(
        "Three questions this page answers:\n\n"
        "1. What do the scripts do, and in what order?\n"
        "2. How are labels, scores, thresholds, and metrics calculated?\n"
        "3. How does the dashboard combine rule-based monitoring with supervised ML?"
    )

    tabs = st.tabs(["Pipeline diagram", "Label creation", "Thresholds & metrics", "Combining signals"])

    with tabs[0]:
        sankey = go.Figure(data=[go.Sankey(
            node=dict(label=[
                "data_ingestion.py", "feature_engineering.py", "bubble_signals.py", "ml_dataset.py",
                "model_training.py", "rare_event_analysis.py", "hazard_model.py", "poisson_count_model.py",
                "ml_inference.py", "dashboard.py",
            ]),
            link=dict(
                source=[0, 1, 2, 3, 4, 4, 4, 6, 8],
                target=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                value=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
        )])
        sankey.update_layout(title="End-to-end project flow", height=520)
        st.plotly_chart(sankey, use_container_width=True)
        st.markdown(
            "- **data_ingestion.py** — downloads raw market, macro, and valuation context.\n"
            "- **feature_engineering.py** — turns raw prices/macro into technical, valuation, and macro features.\n"
            "- **bubble_signals.py** — transparent rule-based warning and recovery scores.\n"
            "- **ml_dataset.py** — reshapes weekly rows + forward labels for supervised learning.\n"
            "- **model_training.py** — trains LR, Elastic Net LR, RF, Balanced RF, XGBoost.\n"
            "- **rare_event_analysis.py** — top-k lift, calibration, percentile buckets.\n"
            "- **imbalance_experiments.py** — SMOTE / SMOTE-ENN / Balanced RF stress tests.\n"
            "- **event_validation.py** — event-level / leave-one-crisis-style validation.\n"
            "- **cost_threshold_analysis.py** — alert-threshold tradeoffs under cost ratios.\n"
            "- **firth_logistic_export.py** — exports clean CSVs for the optional R Firth-logistic benchmark.\n"
            "- **hazard_model.py** — 4-week onset model with full and restricted feature sets + leakage audit.\n"
            "- **poisson_count_model.py** — segment-level count regression.\n"
            "- **ml_inference.py** — scores current AI-exposed assets with trained models.\n"
            "- **model_evaluation.py** — final research story / written summaries.\n"
            "- **dashboard.py / app.py** — this app."
        )

    with tabs[1]:
        st.markdown(
            "### How the main supervised label is created\n\n"
            "**burst_6m_segment = 1** if the asset experiences a segment-adjusted major drawdown over the **next 26 weekly observations**.\n\n"
            "1. Start at week **t**.\n"
            "2. Look forward **t+1 … t+26**.\n"
            "3. Compute worst forward drawdown in that window.\n"
            "4. Compare to a segment-specific threshold.\n"
            "5. Label week **t** as 0 or 1.\n\n"
            "This is why the model is a **historical resemblance model**: it asks whether today looks like weeks "
            "that were later followed by a major drawdown."
        )
        steps_df = pd.DataFrame({
            "step": [1, 2, 3, 4, 5],
            "operation": [
                "Choose week t",
                "Look ahead 26 weekly observations",
                "Find worst forward drawdown",
                "Compare to segment threshold",
                "Assign 0/1 label to week t",
            ],
        })
        st.dataframe(steps_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown(
            "### Threshold tuning and metric calculation\n\n"
            "0.50 is **not** the warning threshold. It is tuned on validation, subject to rare-event guardrails.\n\n"
            "**Confusion-matrix terms**\n"
            "- TP — warned, drawdown happened.\n"
            "- FP — warned, drawdown did not happen.\n"
            "- FN — no warning, drawdown happened.\n"
            "- TN — no warning, no drawdown.\n\n"
            "**Key formulas**\n"
            "- Precision = TP / (TP + FP)\n"
            "- Recall = TP / (TP + FN)\n"
            "- Specificity = TN / (TN + FP)\n"
            "- False positive rate = FP / (FP + TN)\n"
            "- F1 = 2 × Precision × Recall / (Precision + Recall)\n"
            "- MCC — uses all four cells; harder to game with imbalance.\n"
            "- ROC-AUC — overall ranking quality.\n"
            "- PR-AUC — emphasises the rare positive class."
        )
        st.markdown(full_metric_glossary_md())

    with tabs[3]:
        st.markdown(signal_confirmation_md())
        if not data.get("current_ai_ml_risk", pd.DataFrame()).empty:
            st.dataframe(data["current_ai_ml_risk"].head(15), use_container_width=True, hide_index=True)


# =============================================================================
# 13. LITERATURE / BACKGROUND
# =============================================================================
def page_literature() -> None:
    st.title("Background Research and Literature Review")
    st.markdown(
        "Background references for the research topic — bubbles, crashes, valuation excess, speculative behavior, "
        "and market fragility. These frame the dashboard. They are **not** claims that the code replicates each paper."
    )
    refs: dict[str, list[str]] = {
        "Classic bubble and instability frameworks": [
            "Kindleberger, Charles P., and Robert Aliber. *Manias, Panics, and Crashes*.",
            "Minsky, Hyman P. *Stabilizing an Unstable Economy*.",
            "Shiller, Robert J. *Irrational Exuberance*.",
        ],
        "Empirical asset pricing and crash-risk style evidence": [
            "Greenwood, Robin, Andrei Shleifer, and Yang You (2019). *Bubbles for Fama*.",
            "Brunnermeier, Markus, and Martin Oehmke. Literature on bubbles and financial crises.",
            "Baron, Matthew, Emil Verner, and Wei Xiong. Research on fragility and crises.",
        ],
        "Valuation, momentum, and speculative episodes": [
            "Asness, Clifford and coauthors on valuation and momentum interactions.",
            "De Bondt and Thaler on overreaction and reversals.",
            "Jegadeesh and Titman on momentum effects.",
        ],
        "Machine learning and financial prediction caveats": [
            "Gu, Kelly, and Xiu (2020). Empirical asset pricing via machine learning.",
            "Lopez de Prado, Marcos. *Advances in Financial Machine Learning*.",
            "Campbell, Giglio, and Polk style work on return forecasting and predictor instability.",
        ],
        "Financial crises, leverage, and housing/credit": [
            "Reinhart, Carmen, and Kenneth Rogoff. *This Time Is Different*.",
            "Mian, Atif, and Amir Sufi. Housing, leverage, and credit-cycle research.",
            "Jordà, Schularick, and Taylor on credit booms and macro-financial fragility.",
        ],
        "Rare-event modelling and imbalanced classification": [
            "King, Gary, and Langche Zeng. Logistic regression in rare events data.",
            "Firth, David. Bias reduction of maximum likelihood estimates (Biometrika, 1993).",
            "Chawla et al. SMOTE: synthetic minority over-sampling technique (JAIR 2002).",
        ],
    }
    for section, items in refs.items():
        st.subheader(section)
        for item in items:
            st.markdown(f"- {item}")
    st.info(
        "Use this tab as a starting point for a formal written literature review. "
        "Citation formatting (APA/MLA/Chicago) can be added later."
    )


# =============================================================================
# 14. DATA QUALITY & LIMITATIONS
# =============================================================================
def page_data_quality(data: dict[str, pd.DataFrame]) -> None:
    st.title("Data Quality and Limitations")
    st.markdown(
        "**Core risks and limits:**\n\n"
        "- **Survivorship bias** — delisted names (Bear Stearns, Lehman, many SPACs) may not be available through free APIs.\n"
        "- **SEC valuation mapping risk** — XBRL concepts differ by company and era. Rows are kept only when revenue/shares/prices trace cleanly.\n"
        "- **Macro frequency mismatch** — daily prices vs quarterly GDP / monthly CPI. Frequencies are tracked explicitly.\n"
        "- **Ticker history issues** — QQQ begins in 1999; earlier dot-com requires Nasdaq Composite proxies.\n"
        "- **False positives** — hot RSI / stretched 200DMA can persist for long periods.\n"
        "- **False negatives** — some crashes precede slow macro deterioration.\n"
        "- **Different bubble anatomy** — AI infra firms with real earnings ≠ unprofitable dot-coms, ≠ crypto tokens, ≠ housing/credit products.\n"
        "- **No exact top/bottom detection** — the dashboard tests rules. It does not promise perfect timing.\n"
        "- **Few independent events** — even with hundreds of thousands of rows, only ~5–10 distinct historical crisis regimes are usable.\n"
        "- **Model uncertainty** — calibration is imperfect; raw probabilities should be treated as resemblance scores."
    )

    st.subheader("Loaded table checks")
    checks: list[dict[str, Any]] = []
    for name, df in data.items():
        checks.append({
            "Table":     name,
            "Rows":      len(df),
            "Columns":   len(df.columns) if not df.empty else 0,
            "Min date":  df["date"].min().date() if (not df.empty and "date" in df.columns) else "n/a",
            "Max date":  df["date"].max().date() if (not df.empty and "date" in df.columns) else "n/a",
        })
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)


# =============================================================================
# Routing
# =============================================================================
ROUTES = {
    "executive_summary":     page_executive_summary,
    "data_overview":         page_data_overview,
    "bubble_explorer":       page_bubble_explorer,
    "historical_comparison": page_historical_comparison,
    "vital_signs":           page_vital_signs,
    "ai_monitor":            page_ai_monitor,
    "ml_lab":                page_ml_lab,
    "rare_event_lab":        page_rare_event_lab,
    "event_validation":      page_event_validation,
    "hazard_poisson":        page_hazard_poisson,
    "investment_sim":        page_investment_simulator,
    "methods":               page_methods,
    "literature":            None,   # page_literature takes no data
    "data_quality":          page_data_quality,
}

PAGES_NEEDING_CORE_DATA = {
    "data_overview", "bubble_explorer", "historical_comparison", "vital_signs",
    "ai_monitor", "ml_lab", "rare_event_lab", "event_validation", "hazard_poisson",
    "investment_sim", "methods", "data_quality",
    # executive_summary degrades gracefully even without core data.
}


def main() -> None:
    data = load_all_data()

    st.sidebar.markdown("### 📊 Dashboard navigation")
    labels = [label for label, _ in PAGES]
    chosen_label = st.sidebar.radio("Section", labels, label_visibility="collapsed")
    chosen_route = next(route for label, route in PAGES if label == chosen_label)

    # Executive summary degrades nicely; everything else needs core data.
    if chosen_route in PAGES_NEEDING_CORE_DATA and not require_core_data(data):
        return

    if chosen_route == "literature":
        page_literature()
        return

    page_fn = ROUTES[chosen_route]
    page_fn(data)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Research signal & model-interpretation tool. "
        "Not a trading oracle. Not financial advice."
    )


if __name__ == "__main__":
    main()
