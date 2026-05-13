"""
Streamlit dashboard for Economic Bubble Monitoring & Investment Strategy Dashboard.

Run:
    streamlit run dashboard.py

Before launching, run:
    python data_ingestion.py
    python feature_engineering.py
    python bubble_signals.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from backtesting import STRATEGY_RULES, compare_strategies, load_scored_data, prepare_backtest_frame, simulate_strategy, summarize_result
from config import AI_LEADERS, BASE_DIR, BUBBLES, PROCESSED_DIR, STARTING_CAPITAL_DEFAULT, MONTHLY_CONTRIBUTION_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dashboard")

st.set_page_config(
    page_title="Economic Bubble Monitoring Dashboard",
    page_icon="📈",
    layout="wide",
)


def debug_print(message: str) -> None:
    print(f"[DASHBOARD DEBUG] {message}")
    logger.info(message)


@st.cache_data(show_spinner=False)
def load_parquet_file(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_csv_file(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=True)
def load_all_data() -> dict[str, pd.DataFrame]:
    return {
        "prices": load_parquet_file("asset_prices.parquet"),
        "macro": load_parquet_file("macro_indicators.parquet"),
        "macro_features": load_parquet_file("macro_features.parquet"),
        "signals": load_parquet_file("bubble_signal_scores.parquet"),
        "ai_monitor": load_parquet_file("current_ai_monitor.parquet"),
        "valuations": load_parquet_file("valuation_metrics.parquet"),
        "valuation_features": load_parquet_file("valuation_features.parquet"),
        "valuation_snapshot": load_parquet_file("current_valuation_snapshot.parquet"),
        "ml_dataset": load_parquet_file("ml_burst_dataset.parquet"),
        "ml_results": load_parquet_file("ml_model_results.parquet"),
        "ml_predictions": load_parquet_file("ml_test_predictions.parquet"),
        "ml_feature_importance": load_parquet_file("ml_feature_importance.parquet"),
        "ml_thresholds": load_parquet_file("ml_thresholds.parquet"),
        "ml_top_decile": load_parquet_file("ml_top_decile_analysis.parquet"),
        "ml_calibration": load_parquet_file("ml_calibration_table.parquet"),
        "ml_conclusions": load_parquet_file("ml_dashboard_conclusions.parquet"),
        "ml_rare_event_metrics": load_parquet_file("ml_rare_event_metrics.parquet"),
        "ml_rare_event_thresholds": load_parquet_file("ml_rare_event_thresholds.parquet"),
        "ml_risk_buckets": load_parquet_file("ml_risk_bucket_analysis.parquet"),
        "ml_top_k_lift": load_parquet_file("ml_top_k_lift_analysis.parquet"),
        "ml_calibrated_predictions": load_parquet_file("ml_calibrated_predictions.parquet"),
        "ml_rare_event_calibration": load_parquet_file("ml_rare_event_calibration_curves.parquet"),
        "ml_rare_event_conclusions": load_parquet_file("ml_rare_event_conclusions.parquet"),
        "hazard_results": load_parquet_file("hazard_model_results.parquet"),
        "hazard_predictions": load_parquet_file("hazard_model_predictions.parquet"),
        "hazard_audit": load_parquet_file("hazard_leakage_audit.parquet"),
        "hazard_current_ai": load_parquet_file("current_ai_hazard_scores.parquet"),
        "poisson_results": load_parquet_file("poisson_count_model_results.parquet"),
        "poisson_predictions": load_parquet_file("poisson_count_predictions.parquet"),
        "poisson_current": load_parquet_file("current_poisson_count_forecast.parquet"),
        "current_ai_ml_risk": load_parquet_file("current_ai_ml_risk_scores.parquet"),
        "imbalance_results": load_parquet_file("imbalance_experiment_results.parquet"),
        "imbalance_predictions": load_parquet_file("imbalance_experiment_predictions.parquet"),
        "imbalance_thresholds": load_parquet_file("imbalance_experiment_thresholds.parquet"),
        "imbalance_topk": load_parquet_file("imbalance_experiment_topk.parquet"),
        "event_validation_results": load_parquet_file("event_validation_results.parquet"),
        "event_validation_topk": load_parquet_file("event_validation_topk.parquet"),
        "cost_threshold_results": load_parquet_file("ml_cost_threshold_results.parquet"),
        "cost_thresholds": load_parquet_file("ml_cost_thresholds.parquet"),
        "firth_coefficients": load_csv_file("firth_logistic_coefficients.csv"),
    }




def load_ml_summary_text() -> str:
    """Load the markdown summary produced by model_evaluation.py.

    This is intentionally not cached as parquet because the report is written for
    humans, not as a structured table. If the file is missing, the ML dashboard
    page shows exact commands to generate it.
    """
    path = BASE_DIR / "reports" / "ml_model_summary.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_report_text(filename: str) -> str:
    path = BASE_DIR / "reports" / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def choose_binary_target_column(df: pd.DataFrame) -> str | None:
    """Return the best available target column for confusion matrices.

    v12/v13 introduced a segment-adjusted label (`burst_6m_segment`) while
    older artifacts may still contain `burst_6m`. Some experiment tables also
    use generic names such as `target` or `y_true`. The dashboard should not
    crash just because the selected artifact uses the newer target name.
    """
    candidates = ["burst_6m_segment", "burst_6m", "target", "y_true", "actual"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def choose_prediction_column(df: pd.DataFrame) -> str | None:
    """Return the best available binary prediction column."""
    candidates = ["prediction", "predicted_label", "y_pred"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def risk_bucket_order() -> list[str]:
    return ["Bottom 50%", "50-75%", "75-90%", "90-95%", "Top 5%"]


def compute_roc_points(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> pd.DataFrame:
    """Build ROC curve points without requiring sklearn plotting helpers."""
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)
    tmp = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna().sort_values("y_score", ascending=False)
    if tmp.empty or tmp["y_true"].nunique() < 2:
        return pd.DataFrame()
    pos = int((tmp["y_true"] == 1).sum())
    neg = int((tmp["y_true"] == 0).sum())
    if pos == 0 or neg == 0:
        return pd.DataFrame()
    tmp["tp"] = (tmp["y_true"] == 1).cumsum()
    tmp["fp"] = (tmp["y_true"] == 0).cumsum()
    curve = tmp[["y_score", "tp", "fp"]].drop_duplicates(subset=["y_score"], keep="last").copy()
    curve["tpr"] = curve["tp"] / pos
    curve["fpr"] = curve["fp"] / neg
    curve = pd.concat([
        pd.DataFrame({"y_score": [np.inf], "tp": [0], "fp": [0], "tpr": [0.0], "fpr": [0.0]}),
        curve,
        pd.DataFrame({"y_score": [-np.inf], "tp": [pos], "fp": [neg], "tpr": [1.0], "fpr": [1.0]}),
    ], ignore_index=True)
    return curve.sort_values(["fpr", "tpr"]).reset_index(drop=True)


def auc_from_curve(curve: pd.DataFrame) -> float:
    if curve.empty:
        return np.nan
    return float(np.trapezoid(curve["tpr"], curve["fpr"]))


def build_binary_diagnostics(pred: pd.DataFrame) -> dict[str, Any]:
    target_col = choose_binary_target_column(pred)
    prediction_col = choose_prediction_column(pred)
    prob_col = "probability" if "probability" in pred.columns else None
    if target_col is None or prediction_col is None:
        return {}
    y_true = pred[target_col].astype(int)
    y_pred = pred[prediction_col].astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    out = {
        "target_col": target_col,
        "prediction_col": prediction_col,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_positives_per_true_positive": float(fp / tp) if tp > 0 else np.inf,
    }
    if prob_col is not None:
        out["roc_curve"] = compute_roc_points(y_true, pred[prob_col])
    else:
        out["roc_curve"] = pd.DataFrame()
    return out


def plot_confusion_matrix_from_metrics(metrics: dict[str, Any], title: str) -> go.Figure:
    matrix = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Actual no burst", "Actual burst"],
        columns=["Predicted no burst", "Predicted burst"],
    )
    fig = px.imshow(matrix, text_auto=True, title=title)
    fig.update_layout(height=380)
    return fig


def pretty_metric_label(metric: str) -> str:
    return {
        "false_positive": "False positives",
        "false_positive_rate": "False positive rate",
        "false_positives_per_true_positive": "False positives per true positive",
        "precision": "Precision",
        "recall": "Recall",
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "mcc": "MCC",
        "alert_rate": "Alert rate",
    }.get(metric, metric.replace("_", " ").title())


def top_numeric_features(df: pd.DataFrame, max_features: int = 12, exclude: list[str] | None = None) -> list[str]:
    exclude = set(exclude or [])
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    if not numeric:
        return []
    completeness = df[numeric].notna().mean().sort_values(ascending=False)
    return completeness.head(max_features).index.tolist()


def feature_theme(feature: str) -> str:
    f = str(feature).lower()
    if any(x in f for x in ["rsi", "sma", "drawdown", "volatility", "return", "200dma", "price_log", "zscore", "parabolic"]):
        return "Technical / price behavior"
    if any(x in f for x in ["price_to_sales", "pe_", "ev_", "market_cap", "enterprise_value", "revenue", "income", "valuation"]):
        return "Valuation / fundamentals"
    if any(x in f for x in ["federal", "treasury", "yield_curve", "cpi", "unemployment", "mortgage", "gdp", "wilshire", "stress", "recession"]):
        return "Macro / rates"
    if any(x in f for x in ["warning_score", "recovery_score"]):
        return "Rule-based scores"
    return "Other"


def build_signal_confirmation_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a current AI signal-confirmation table from available outputs.

    This table intentionally avoids saying "buy" or "sell." It combines the
    rule-based AI monitor, ML risk ranking, and hazard score into a compact
    confirmation view. The goal is to show whether different research signals
    point in the same direction.
    """
    ai = data.get("current_ai_ml_risk", pd.DataFrame()).copy()
    hazard = data.get("hazard_current_ai", pd.DataFrame()).copy()
    monitor = data.get("ai_monitor", pd.DataFrame()).copy()

    if ai.empty and hazard.empty and monitor.empty:
        return pd.DataFrame()

    tickers = set()
    for df in [ai, hazard, monitor]:
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].dropna().astype(str).tolist())

    rows = []
    for ticker in sorted(tickers):
        row = {"ticker": ticker}
        if not ai.empty and "ticker" in ai.columns:
            a = ai[ai["ticker"].astype(str).eq(ticker)].tail(1)
            if not a.empty:
                ar = a.iloc[0]
                row["ml_model"] = ar.get("primary_model", ar.get("model", "n/a"))
                row["ml_scope"] = ar.get("primary_model_scope", ar.get("model_scope", "n/a"))
                row["ml_raw_score"] = ar.get("primary_ml_risk_probability", np.nan)
                row["ml_empirical_bucket_rate"] = ar.get("calibrated_empirical_risk", np.nan)
                row["ml_risk_category"] = ar.get("risk_category", "n/a")
        if not hazard.empty and "ticker" in hazard.columns:
            h = hazard[hazard["ticker"].astype(str).eq(ticker)].tail(1)
            if not h.empty:
                hr = h.iloc[0]
                row["hazard_model"] = hr.get("hazard_model", "n/a")
                row["hazard_feature_set"] = hr.get("hazard_feature_set", "n/a")
                row["hazard_score"] = hr.get("hazard_score_4w", hr.get("hazard_probability_4w", np.nan))
                row["hazard_risk_category"] = hr.get("hazard_risk_category", "n/a")
        if not monitor.empty and "ticker" in monitor.columns:
            m = monitor[monitor["ticker"].astype(str).eq(ticker)].tail(1)
            if not m.empty:
                mr = m.iloc[0]
                for c in ["phase_label", "signal_label", "warning_score", "recovery_score", "drawdown_pct", "distance_from_200dma"]:
                    if c in m.columns:
                        row[c] = mr.get(c)

        # A simple confluence score based on ranks/categories that exist.
        score = 0
        reasons = []
        ml_cat = str(row.get("ml_risk_category", "")).lower()
        hazard_cat = str(row.get("hazard_risk_category", "")).lower()
        phase = str(row.get("phase_label", row.get("signal_label", ""))).lower()
        warning = pd.to_numeric(row.get("warning_score", np.nan), errors="coerce")
        if any(x in ml_cat for x in ["high", "extreme", "elevated"]):
            score += 1; reasons.append("ML risk bucket elevated")
        if any(x in hazard_cat for x in ["high", "extreme", "elevated"]):
            score += 1; reasons.append("Hazard/onset score elevated")
        if pd.notna(warning) and warning >= 2:
            score += 1; reasons.append("Rule warning score active")
        if any(x in phase for x in ["euphoria", "breakdown", "risk", "crash"]):
            score += 1; reasons.append("Rule phase label is risk-oriented")
        row["signal_confirmation_score"] = score
        row["signal_confirmation_level"] = pd.cut([score], [-1, 0, 1, 2, 4], labels=["None", "Low", "Medium", "High"])[0]
        row["confirmation_reasons"] = "; ".join(reasons) if reasons else "No strong cross-signal confirmation"
        rows.append(row)
    return pd.DataFrame(rows)


def require_data(data: dict[str, pd.DataFrame]) -> bool:
    missing = [name for name in ["prices", "signals"] if data.get(name, pd.DataFrame()).empty]
    if missing:
        st.error(
            "Required processed data is missing. Run these commands from the project folder:\n\n"
            "python data_ingestion.py\n"
            "python feature_engineering.py\n"
            "python bubble_signals.py\n\n"
            f"Missing: {', '.join(missing)}"
        )
        return False
    return True


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:,.2f}%"


def sidebar_filters(signals: pd.DataFrame) -> tuple[str, str, pd.Timestamp, pd.Timestamp]:
    bubble_name = st.sidebar.selectbox("Bubble period", list(BUBBLES.keys()))
    meta = BUBBLES[bubble_name]
    available_tickers = sorted(set(meta.get("tickers", [])).intersection(set(signals["ticker"].unique())))
    if not available_tickers:
        available_tickers = sorted(signals["ticker"].unique())
    ticker = st.sidebar.selectbox("Ticker / index", available_tickers)
    default_start = pd.Timestamp(meta["start"])
    default_end = pd.Timestamp(meta["end"]) if meta.get("end") else signals["date"].max()
    min_date = max(signals[signals["ticker"].eq(ticker)]["date"].min(), default_start)
    max_date = min(signals[signals["ticker"].eq(ticker)]["date"].max(), default_end)
    start_date, end_date = st.sidebar.date_input(
        "Date window",
        value=(min_date.date(), max_date.date()),
        min_value=signals[signals["ticker"].eq(ticker)]["date"].min().date(),
        max_value=signals[signals["ticker"].eq(ticker)]["date"].max().date(),
    )
    return bubble_name, ticker, pd.Timestamp(start_date), pd.Timestamp(end_date)


def page_overview(data: dict[str, pd.DataFrame]) -> None:
    st.title("Economic Bubble Monitoring & Investment Strategy Dashboard")
    st.warning(
        "Educational research tool only. This dashboard backtests historical signal rules. "
        "It does not predict exact tops or bottoms and is not personalized financial advice."
    )

    st.markdown(
        """
        This project compares historical bubble-like cycles across technology, housing/credit, crypto,
        commodities, Japan equities, and the current AI/mega-cap technology cycle. The core question is
        whether transparent historical warning and recovery rules can help us study market behavior before,
        during, and after major speculative episodes.

        **The dashboard is built around four research moves:**
        1. Normalize price and macro data into reusable tables.
        2. Compute valuation, technical, macro, and speculation proxies where data is available.
        3. Backtest rules without look-ahead bias whenever the rule can be evaluated in real time.
        4. Compare the current AI cycle against prior bubbles as a risk-monitoring exercise, not prophecy.
        """
    )

    signals = data["signals"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers loaded", f"{signals['ticker'].nunique():,}" if not signals.empty else "0")
    c2.metric("Daily rows", f"{len(signals):,}" if not signals.empty else "0")
    c3.metric("First date", str(signals["date"].min().date()) if not signals.empty else "n/a")
    c4.metric("Latest date", str(signals["date"].max().date()) if not signals.empty else "n/a")
    if not data.get("valuations", pd.DataFrame()).empty:
        st.success(f"SEC historical valuation rows loaded: {len(data['valuations']):,}. Valuation warning signals are active for covered corporate tickers.")
    else:
        st.info("SEC historical valuation rows are not loaded yet. Run data_ingestion.py without --skip-sec to activate valuation warning signals.")

    timeline_rows = []
    for name, meta in BUBBLES.items():
        timeline_rows.append(
            {
                "Bubble": name,
                "Start": meta["start"],
                "Peak hint": meta.get("peak_hint") or "live / unknown",
                "End": meta.get("end") or "present / unresolved",
                "Asset class": meta["asset_class"],
                "Included tickers": ", ".join(meta.get("tickers", [])[:8]),
            }
        )
    st.subheader("Bubble timeline")
    st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)


def price_signal_chart(g: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["date"], y=g["adjusted_close"], mode="lines", name="Adjusted close"))
    fig.add_trace(go.Scatter(x=g["date"], y=g["sma_50"], mode="lines", name="50-day SMA"))
    fig.add_trace(go.Scatter(x=g["date"], y=g["sma_200"], mode="lines", name="200-day SMA"))
    exits = g[g["exit_signal"].fillna(False)]
    entries = g[g["reentry_signal"].fillna(False)]
    fig.add_trace(
        go.Scatter(
            x=exits["date"],
            y=exits["adjusted_close"],
            mode="markers",
            name="Exit warning",
            marker=dict(symbol="triangle-down", size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=entries["date"],
            y=entries["adjusted_close"],
            mode="markers",
            name="Re-entry signal",
            marker=dict(symbol="triangle-up", size=9),
        )
    )
    fig.update_layout(title=f"{ticker} price, moving averages, and signal markers", height=520, hovermode="x unified")
    return fig


def page_bubble_explorer(data: dict[str, pd.DataFrame]) -> None:
    st.title("Bubble Explorer")
    signals = data["signals"]
    macro = data["macro"]
    bubble_name, ticker, start_date, end_date = sidebar_filters(signals)
    g = signals[(signals["ticker"].eq(ticker)) & (signals["date"].between(start_date, end_date))].copy()

    st.caption(BUBBLES[bubble_name]["notes"])
    if g.empty:
        st.error("No rows for this ticker/date window.")
        return

    latest = g.tail(1).iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest close", f"${latest['adjusted_close']:,.2f}")
    c2.metric("Drawdown", format_pct(latest.get("drawdown_pct")))
    c3.metric("Distance from 200DMA", format_pct(latest.get("distance_from_200dma")))
    c4.metric("Weekly RSI", f"{latest.get('weekly_rsi', np.nan):,.1f}")
    c5.metric("Warning score", f"{latest.get('warning_score', np.nan):,.0f}/100")

    st.plotly_chart(price_signal_chart(g, ticker), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_dd = px.area(g, x="date", y="drawdown_pct", title="Drawdown from trailing peak (%)")
        fig_dd.update_layout(height=360)
        st.plotly_chart(fig_dd, use_container_width=True)
    with c2:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=g["date"], y=g["weekly_rsi"], mode="lines", name="Weekly RSI"))
        fig_rsi.add_hline(y=75, line_dash="dash", annotation_text="Hot zone")
        fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold zone")
        fig_rsi.update_layout(title="Weekly RSI", height=360)
        st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("Macro overlay")
    if macro.empty:
        st.info("Macro data not loaded.")
    else:
        indicators = sorted(macro["indicator_id"].unique())
        indicator = st.selectbox("Macro indicator", indicators, index=indicators.index("FEDFUNDS") if "FEDFUNDS" in indicators else 0)
        m = macro[(macro["indicator_id"].eq(indicator)) & (macro["date"].between(start_date, end_date))]
        if not m.empty:
            fig_m = px.line(m, x="date", y="value", title=f"{indicator}: {m['indicator_name'].iloc[0]}")
            st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Latest signal details")
    cols = [
        "date",
        "ticker",
        "signal_label",
        "phase_label",
        "warning_score",
        "recovery_score",
        "drawdown_pct",
        "distance_from_200dma",
        "weekly_rsi",
        "volatility_30d",
        "return_126d",
        "return_252d",
    ]
    st.dataframe(g[cols].tail(20), use_container_width=True, hide_index=True)


def align_by_peak(signals: pd.DataFrame, selections: dict[str, str]) -> pd.DataFrame:
    rows = []
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
    st.markdown("Normalize each selected asset to 100 at its retrospective peak inside the configured bubble window.")

    selections: dict[str, str] = {}
    default_bubbles = ["Dot-com Bubble", "Housing / Financial Crisis", "Crypto / High-Growth Tech Bubble", "AI / Mega-Cap Tech Cycle"]
    for bubble_name in default_bubbles:
        meta = BUBBLES[bubble_name]
        valid = sorted(set(meta["tickers"]).intersection(signals["ticker"].unique()))
        if valid:
            default = "^IXIC" if "^IXIC" in valid else valid[0]
            selections[bubble_name] = st.selectbox(f"{bubble_name} asset", valid, index=valid.index(default) if default in valid else 0)

    aligned = align_by_peak(signals, selections)
    if aligned.empty:
        st.error("No aligned data available.")
        return

    fig = px.line(
        aligned,
        x="relative_day",
        y="normalized_to_peak",
        color="bubble",
        title="Normalized price paths aligned by retrospective peak date",
        labels={"relative_day": "Days from peak", "normalized_to_peak": "Price index (peak = 100)"},
    )
    fig.add_vline(x=0, line_dash="dash", annotation_text="Peak")
    fig.update_layout(height=560, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    summary = (
        aligned.groupby("bubble")
        .agg(
            ticker=("ticker", "first"),
            min_normalized=("normalized_to_peak", "min"),
            worst_drawdown_pct=("drawdown_pct", "min"),
            days_observed=("relative_day", "count"),
        )
        .reset_index()
    )
    summary["crash_depth_from_peak_pct"] = summary["min_normalized"] - 100
    st.dataframe(summary, use_container_width=True, hide_index=True)


def page_vital_signs(data: dict[str, pd.DataFrame]) -> None:
    st.title("Bubble Vital Signs")
    signals = data["signals"]
    valuations = data["valuation_features"] if not data.get("valuation_features", pd.DataFrame()).empty else data["valuations"]
    ticker = st.selectbox("Ticker", sorted(signals["ticker"].unique()), index=sorted(signals["ticker"].unique()).index("QQQ") if "QQQ" in signals["ticker"].unique() else 0)
    g = signals[signals["ticker"].eq(ticker)].sort_values("date")
    latest = g.tail(1).iloc[0]

    st.subheader("Technical vital signs")
    vital = pd.DataFrame(
        [
            ["Warning score", latest.get("warning_score"), "0-100 transparent rule score"],
            ["Recovery score", latest.get("recovery_score"), "0-100 transparent rule score"],
            ["Drawdown from peak", latest.get("drawdown_pct"), "Lower means deeper crash"],
            ["Distance from 200DMA", latest.get("distance_from_200dma"), "High positive values can show stretched momentum"],
            ["Weekly RSI", latest.get("weekly_rsi"), ">75 hot; <30 oversold"],
            ["30-day volatility", latest.get("volatility_30d"), "Annualized volatility"],
            ["126-day return", latest.get("return_126d"), "Six-month momentum proxy"],
            ["252-day return", latest.get("return_252d"), "One-year momentum proxy"],
        ],
        columns=["Metric", "Latest value", "Interpretation"],
    )
    st.dataframe(vital, use_container_width=True, hide_index=True)

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
            "No SEC historical valuation row is available for this ticker. That usually means the asset is an ETF/index/crypto, the ticker lacks SEC CIK coverage, or EDGAR did not expose a clean revenue/share concept for that company."
        )
    else:
        vt = valuations[valuations["ticker"].eq(ticker)].sort_values("date").copy()
        cols = [
            "date", "filing_date", "period_end", "price_to_sales", "ps_historical_median",
            "ps_to_historical_median", "pe_ratio", "ev_to_sales", "revenue_ttm",
            "revenue_ttm_yoy_pct", "valuation_warning_score", "source"
        ]
        cols = [c for c in cols if c in vt.columns]
        st.caption("Rows are dated to the first trading day after the SEC filing date to reduce look-ahead bias.")
        st.dataframe(vt[cols].tail(40), use_container_width=True, hide_index=True)
        if "price_to_sales" in vt.columns:
            fig_v = px.line(vt, x="date", y="price_to_sales", title=f"{ticker} SEC-derived historical Price/Sales")
            if "ps_historical_median" in vt.columns:
                fig_v.add_trace(go.Scatter(x=vt["date"], y=vt["ps_historical_median"], mode="lines", name="Expanding historical median"))
            st.plotly_chart(fig_v, use_container_width=True)


def page_ai_monitor(data: dict[str, pd.DataFrame]) -> None:
    st.title("Current AI Cycle Monitor")
    st.warning(
        "Status labels are rule-based historical-risk labels. They are not predictions and not buy/sell advice."
    )
    ai = data["ai_monitor"]
    valuations = data["valuation_features"] if not data.get("valuation_features", pd.DataFrame()).empty else data["valuations"]
    snapshot = data.get("valuation_snapshot", pd.DataFrame())
    if ai.empty:
        st.error("AI monitor data missing. Run python bubble_signals.py after ingestion and feature engineering.")
        return

    st.dataframe(ai, use_container_width=True, hide_index=True)

    fig = px.scatter(
        ai,
        x="distance_from_200dma",
        y="weekly_rsi",
        size="warning_score",
        color="status",
        hover_name="ticker",
        title="AI-exposed assets: momentum heat map",
        labels={"distance_from_200dma": "Distance from 200DMA (%)", "weekly_rsi": "Weekly RSI"},
    )
    fig.add_hline(y=75, line_dash="dash")
    fig.add_vline(x=50, line_dash="dash")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    if not valuations.empty and "ticker" in valuations.columns:
        ai_val = valuations[valuations["ticker"].isin(AI_LEADERS)].copy()
        if not ai_val.empty:
            latest_sec = ai_val.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
            st.subheader("Latest SEC-derived historical valuation signal rows")
            st.caption("These are filing-date-aware rows, not scraped current-only snapshots.")
            sec_cols = [
                "date", "ticker", "filing_date", "period_end", "price_to_sales",
                "ps_to_historical_median", "ev_to_sales", "revenue_ttm_yoy_pct",
                "valuation_warning_score", "source"
            ]
            sec_cols = [c for c in sec_cols if c in latest_sec.columns]
            st.dataframe(latest_sec[sec_cols], use_container_width=True, hide_index=True)

    if not snapshot.empty and "ticker" in snapshot.columns:
        snap = snapshot[snapshot["ticker"].isin(AI_LEADERS)].copy()
        if not snap.empty:
            st.subheader("Current valuation snapshot context")
            st.caption("This yfinance snapshot is current context only; it is not used as historical point-in-time evidence.")
            st.dataframe(snap, use_container_width=True, hide_index=True)

    status_counts = ai["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    st.subheader("Status distribution")
    st.plotly_chart(px.bar(status_counts, x="Status", y="Count", title="AI monitor status counts"), use_container_width=True)


def page_investment_simulator(data: dict[str, pd.DataFrame]) -> None:
    st.title("Investment Simulator")
    st.warning(
        "Backtests are historical simulations. Retrospective peak strategies use hindsight and are clearly labeled."
    )
    signals = data["signals"]
    bubble_name, ticker, start_date, end_date = sidebar_filters(signals)

    c1, c2, c3 = st.columns(3)
    starting_capital = c1.number_input("Starting capital", min_value=1.0, value=float(STARTING_CAPITAL_DEFAULT), step=500.0)
    monthly_contribution = c2.number_input("Monthly contribution", min_value=0.0, value=float(MONTHLY_CONTRIBUTION_DEFAULT), step=100.0)
    strategy = c3.selectbox("Primary strategy", list(STRATEGY_RULES.keys()), index=list(STRATEGY_RULES.keys()).index("sell_exit_buy_recovery"))

    g = prepare_backtest_frame(signals, ticker, str(start_date.date()), str(end_date.date()))
    result = simulate_strategy(g, ticker, strategy, starting_capital, monthly_contribution)
    summary = summarize_result(result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ending value", f"${summary['ending_value']:,.2f}")
    c2.metric("Total return", format_pct(summary["total_return_pct"]))
    c3.metric("Max drawdown", format_pct(summary["max_drawdown_pct"]))
    c4.metric("Sharpe", f"{summary['sharpe_ratio']:,.2f}" if pd.notna(summary["sharpe_ratio"]) else "n/a")
    st.caption(summary["notes"])

    fig = px.line(result.equity_curve, x="date", y="portfolio_value", title=f"Equity curve: {ticker} / {strategy}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Compare strategies")
    selected_strategies = st.multiselect(
        "Strategies to compare",
        list(STRATEGY_RULES.keys()),
        default=["buy_and_hold", "sell_exit_buy_recovery", "cash_until_40pct_drawdown", "cash_until_reclaim_200dma"],
    )
    if selected_strategies:
        comparison, curves = compare_strategies(
            signals,
            ticker=ticker,
            strategies=selected_strategies,
            start=str(start_date.date()),
            end=str(end_date.date()),
            starting_capital=starting_capital,
            monthly_contribution=monthly_contribution,
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        curve_rows = []
        for name, curve in curves.items():
            tmp = curve[["date", "portfolio_value"]].copy()
            tmp["strategy"] = name
            curve_rows.append(tmp)
        if curve_rows:
            all_curves = pd.concat(curve_rows, ignore_index=True)
            st.plotly_chart(
                px.line(all_curves, x="date", y="portfolio_value", color="strategy", title="Strategy comparison equity curves"),
                use_container_width=True,
            )



def page_machine_learning_crash_risk(data: dict[str, pd.DataFrame]) -> None:
    """Dashboard page for supervised ML forward drawdown-risk modeling.

    This upgraded page shows not only raw model metrics, but also the pieces that
    matter for a rare-event financial risk model: tuned thresholds, top-decile
    analysis, calibration, segment-specific models, and an explicit conclusion on
    whether the model is strong, weak, or still experimental.
    """
    st.title("Machine Learning Signal Confirmation & Risk-Ranking Lab")
    st.warning(
        "Educational research tool only. These models estimate historical forward drawdown-risk patterns. "
        "They do not predict exact bubble-burst dates and are not financial advice."
    )

    st.markdown(
        """
        This page upgrades the rule-based bubble dashboard into a supervised machine-learning and signal-confirmation experiment.
        The primary target is:

        **burst_6m_segment = 1 if the asset crossed a segment-adjusted major-drawdown threshold over the next 26 weekly observations.**

        The segment thresholds are intentionally different: broad indexes/ETFs use a lower major-drawdown threshold,
        mega-cap AI/tech uses a middle threshold, and speculative/high-volatility assets use a harsher threshold.
        In plain English, the model asks: **given the information available during this week, did a major drawdown
        occur over the next six months relative to what counts as severe for that asset segment?** Future prices are
        used only to create historical labels. They are not used as input features when scoring current AI-exposed assets.
        """
    )

    required_ml = {
        "ML dataset": data.get("ml_dataset", pd.DataFrame()),
        "Model results": data.get("ml_results", pd.DataFrame()),
        "Model predictions": data.get("ml_predictions", pd.DataFrame()),
        "Feature importance": data.get("ml_feature_importance", pd.DataFrame()),
        "Current AI ML risk": data.get("current_ai_ml_risk", pd.DataFrame()),
    }
    missing = [name for name, df in required_ml.items() if df.empty]
    if missing:
        st.error(
            "Some ML outputs are missing. Run these commands from the project folder:\n\n"
            "python ml_dataset.py\n"
            "python model_training.py\n"
            "python model_evaluation.py\n"
            "python ml_inference.py\n"
            "streamlit run dashboard.py\n\n"
            f"Missing or empty: {', '.join(missing)}"
        )
        st.info(
            "The rest of the dashboard can still work without the ML layer, but this page needs those generated files. "
            "This prevents fake placeholder charts and keeps the research trail honest."
        )
        return

    results = data["ml_results"].copy()
    predictions = data["ml_predictions"].copy()
    importance = data["ml_feature_importance"].copy()
    ai_risk = data["current_ai_ml_risk"].copy()
    ml_dataset = data["ml_dataset"].copy()
    thresholds = data.get("ml_thresholds", pd.DataFrame()).copy()
    top_decile = data.get("ml_top_decile", pd.DataFrame()).copy()
    calibration = data.get("ml_calibration", pd.DataFrame()).copy()
    conclusions = data.get("ml_conclusions", pd.DataFrame()).copy()
    rare_metrics = data.get("ml_rare_event_metrics", pd.DataFrame()).copy()
    rare_thresholds = data.get("ml_rare_event_thresholds", pd.DataFrame()).copy()
    risk_buckets = data.get("ml_risk_buckets", pd.DataFrame()).copy()
    top_k_lift = data.get("ml_top_k_lift", pd.DataFrame()).copy()
    rare_calibration = data.get("ml_rare_event_calibration", pd.DataFrame()).copy()
    rare_conclusions = data.get("ml_rare_event_conclusions", pd.DataFrame()).copy()
    hazard_results = data.get("hazard_results", pd.DataFrame()).copy()
    hazard_predictions = data.get("hazard_predictions", pd.DataFrame()).copy()
    hazard_audit = data.get("hazard_audit", pd.DataFrame()).copy()
    hazard_current_ai = data.get("hazard_current_ai", pd.DataFrame()).copy()
    imbalance_results = data.get("imbalance_results", pd.DataFrame()).copy()
    imbalance_topk = data.get("imbalance_topk", pd.DataFrame()).copy()
    event_results = data.get("event_validation_results", pd.DataFrame()).copy()
    event_topk = data.get("event_validation_topk", pd.DataFrame()).copy()
    cost_results = data.get("cost_threshold_results", pd.DataFrame()).copy()
    firth_coefficients = data.get("firth_coefficients", pd.DataFrame()).copy()
    poisson_results = data.get("poisson_results", pd.DataFrame()).copy()
    poisson_current = data.get("poisson_current", pd.DataFrame()).copy()

    st.subheader("1. Executive Model Conclusion")
    test_results = results[results["split"].eq("test")].copy() if "split" in results.columns else pd.DataFrame()
    if not test_results.empty:
        best_cols = st.columns(4)
        for col_obj, metric, label in zip(best_cols, ["roc_auc", "pr_auc", "recall", "precision"], ["Best ROC-AUC", "Best PR-AUC", "Best Recall", "Best Precision"]):
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

    st.subheader("2. Signal Confirmation View")
    st.markdown(
        """
        This section reframes the output as **risk ranking and signal confirmation**, not literal prophecy.
        A useful warning is stronger when multiple independent signals agree: rule-based bubble warnings,
        ML top-risk buckets, hazard/onset scores, and valuation/macro context.
        """
    )
    confirmation = build_signal_confirmation_table(data)
    if confirmation.empty:
        st.info("Current signal-confirmation table is unavailable. Run `python ml_inference.py` and `python hazard_model.py`.")
    else:
        display_cols = [c for c in [
            "ticker", "signal_confirmation_level", "signal_confirmation_score", "confirmation_reasons",
            "ml_risk_category", "ml_empirical_bucket_rate", "ml_raw_score",
            "hazard_risk_category", "hazard_score", "phase_label", "warning_score", "recovery_score"
        ] if c in confirmation.columns]
        st.dataframe(confirmation[display_cols].sort_values(["signal_confirmation_score", "ticker"], ascending=[False, True]), use_container_width=True, hide_index=True)
        st.caption("Signal confirmation is a research ranking summary. It is not a personalized buy/sell recommendation.")

    st.subheader("3. Model Comparison with Tuned Thresholds")
    st.caption(
        "Metrics are from chronological validation/test splits. The rows were not randomly shuffled. "
        "The prediction threshold is tuned on validation data rather than hard-coded to 0.50."
    )
    metric_cols = [
        "model", "asset_segment", "split", "threshold_value", "n_rows", "positive_rate", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier_score",
        "true_negative", "false_positive", "false_negative", "true_positive",
    ]
    metric_cols = [c for c in metric_cols if c in results.columns]
    sort_cols = [c for c in ["split", "asset_segment", "pr_auc"] if c in results.columns]
    st.dataframe(results[metric_cols].sort_values(sort_cols, ascending=[True] * (len(sort_cols)-1) + [False] if sort_cols else True), use_container_width=True, hide_index=True)

    if not test_results.empty and {"model", "pr_auc", "roc_auc"}.issubset(test_results.columns):
        c1, c2 = st.columns(2)
        with c1:
            fig_pr = px.bar(test_results.sort_values("pr_auc", ascending=False), x="model", y="pr_auc", color="asset_segment" if "asset_segment" in test_results.columns else None, title="Test PR-AUC by model and scope")
            fig_pr.update_layout(height=400)
            st.plotly_chart(fig_pr, use_container_width=True)
        with c2:
            fig_roc = px.bar(test_results.sort_values("roc_auc", ascending=False), x="model", y="roc_auc", color="asset_segment" if "asset_segment" in test_results.columns else None, title="Test ROC-AUC by model and scope")
            fig_roc.update_layout(height=400)
            st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown(
        """
        **Why PR-AUC matters:** major drawdowns are rare. Accuracy can look good if a model mostly predicts
        "no burst." PR-AUC focuses more directly on whether high-risk alerts are informative for the rare
        positive class.
        """
    )

    st.subheader("3. Top-Decile Risk Analysis")
    st.markdown(
        """
        This is one of the most important upgrades. Instead of asking only whether the model crossed a single
        warning threshold, this asks: **when the model ranked a row in the highest-risk 10% of historical scores,
        how often did a 30% forward drawdown actually occur?**
        """
    )
    if top_decile.empty:
        st.info("Top-decile file is missing. Run `python model_training.py` again with the upgraded script.")
    else:
        td_cols = ["model", "model_scope", "split", "base_event_rate", "top_decile_event_rate", "lift_vs_base_rate", "top_decile_rows", "top_decile_true_positives", "min_probability_top_decile"]
        td_cols = [c for c in td_cols if c in top_decile.columns]
        st.dataframe(top_decile[top_decile["split"].eq("test")][td_cols].sort_values("lift_vs_base_rate", ascending=False), use_container_width=True, hide_index=True)
        td_test = top_decile[top_decile["split"].eq("test")].copy()
        if not td_test.empty:
            fig_td = px.bar(
                td_test.sort_values("lift_vs_base_rate", ascending=False),
                x="model",
                y="lift_vs_base_rate",
                color="model_scope",
                barmode="group",
                title="Top-decile lift vs base event rate, grouped by scope",
            )
            fig_td.update_layout(height=440)
            st.plotly_chart(fig_td, use_container_width=True)
            st.caption("Bars are grouped rather than stacked because lift values do not add across model scopes.")

    st.subheader("4. Tuned Thresholds")
    st.markdown(
        """
        The project no longer treats **0.50** as a magic cutoff. Each model's alert threshold is selected on
        validation data by maximizing F1, then applied to the later test period. This is more appropriate when
        the base event rate is low and a 15%-25% modeled risk may already be historically elevated.
        """
    )
    if thresholds.empty:
        st.info("Threshold table is missing. Run `python model_training.py` again.")
    else:
        st.dataframe(thresholds.sort_values(["model_scope", "model"]), use_container_width=True, hide_index=True)

    st.subheader("5. Calibration / Reliability")
    st.markdown(
        """
        Calibration asks whether model scores behave like probabilities. In the current project, raw model scores
        are **not trusted as literal probabilities** unless the empirical calibration curve supports that interpretation.
        Prefer the empirical event rate of the matching score bucket over the raw score itself.
        """
    )
    if calibration.empty:
        st.info("Calibration table is missing. Run `python model_training.py` again.")
    else:
        cal_splits = sorted(calibration["split"].dropna().unique())
        default_split = "test" if "test" in cal_splits else cal_splits[0]
        cal_split = st.selectbox("Calibration split", cal_splits, index=cal_splits.index(default_split), key="cal_split")
        cal_models = sorted(calibration["model"].dropna().unique())
        cal_model = st.selectbox("Calibration model", cal_models, key="cal_model")
        cal_scopes = sorted(calibration["model_scope"].dropna().unique())
        cal_scope = st.selectbox("Calibration scope", cal_scopes, index=cal_scopes.index("global") if "global" in cal_scopes else 0, key="cal_scope")
        cal = calibration[(calibration["split"].eq(cal_split)) & (calibration["model"].eq(cal_model)) & (calibration["model_scope"].eq(cal_scope))].copy()
        if cal.empty:
            st.info("No calibration rows for this model/scope/split.")
        else:
            st.dataframe(cal, use_container_width=True, hide_index=True)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(x=cal["mean_predicted_probability"], y=cal["empirical_event_rate"], mode="markers+lines", name="Observed"))
            fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
            fig_cal.update_layout(title="Calibration curve by probability decile", xaxis_title="Mean predicted probability", yaxis_title="Empirical event rate", height=420)
            st.plotly_chart(fig_cal, use_container_width=True)

    st.subheader("6. Rare-Event Threshold Audit")
    st.markdown(
        """
        This section applies stricter rare-event threshold rules. The threshold must clear a minimum precision
        requirement and cannot flag an unreasonable share of rows. This prevents fake improvements where recall
        rises only because the model labels almost everything as risky.
        """
    )
    if rare_thresholds.empty:
        st.info("Rare-event threshold table is missing. Run `python rare_event_analysis.py` after model_training.py.")
    else:
        st.dataframe(rare_thresholds.sort_values(["model_scope", "model"]), use_container_width=True, hide_index=True)

    if not rare_metrics.empty:
        st.markdown("**Rare-event metrics using constrained thresholds:**")
        st.dataframe(rare_metrics[rare_metrics["split"].eq("test")].sort_values("pr_auc", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("7. Percentile Risk Buckets")
    st.markdown(
        """
        Risk buckets often communicate rare-event models better than a yes/no prediction. The key question is whether
        higher score buckets actually contain higher empirical drawdown rates.
        """
    )
    if risk_buckets.empty:
        st.info("Risk bucket analysis is missing. Run `python rare_event_analysis.py`.")
    else:
        bucket_test = risk_buckets[risk_buckets["split"].eq("test")].copy()
        st.dataframe(bucket_test.sort_values(["model_scope", "model", "risk_bucket"]), use_container_width=True, hide_index=True)
        # Do not stack model event rates. Stacking makes the chart look like event rates add up
        # across models, which has no statistical interpretation. A line chart answers the real
        # question: does each model's empirical event rate rise as the score bucket becomes riskier?
        bucket_order = risk_bucket_order()
        fig_bucket = px.line(
            bucket_test,
            x="risk_bucket",
            y="bucket_event_rate",
            color="model",
            markers=True,
            facet_col="model_scope" if "model_scope" in bucket_test.columns else None,
            category_orders={"risk_bucket": bucket_order},
            title="Empirical event rate by model risk bucket, test split",
        )
        fig_bucket.update_layout(height=520, yaxis_tickformat=".0%")
        st.plotly_chart(fig_bucket, use_container_width=True)

    st.subheader("8. Top-Decile and Top-5% Lift")
    if top_k_lift.empty:
        st.info("Top-k lift analysis is missing. Run `python rare_event_analysis.py`.")
    else:
        top_test = top_k_lift[top_k_lift["split"].eq("test")].copy()
        st.dataframe(top_test.sort_values("lift_vs_base_rate", ascending=False), use_container_width=True, hide_index=True)
        fig_topk = px.bar(
            top_test.sort_values("lift_vs_base_rate", ascending=False),
            x="model",
            y="lift_vs_base_rate",
            color="top_label",
            facet_col="model_scope",
            barmode="group",
            title="Top 10% and Top 5% lift vs base event rate, grouped not stacked",
        )
        fig_topk.update_layout(height=520)
        st.plotly_chart(fig_topk, use_container_width=True)
        st.caption("Top-5% and Top-10% lift bars are grouped. Stacking them would imply the lifts add together, which is not meaningful.")

    st.subheader("9. Rare-Event Calibration Curves")
    if rare_calibration.empty:
        st.info("Rare-event calibration curves are missing. Run `python rare_event_analysis.py`.")
    else:
        rc_splits = sorted(rare_calibration["split"].dropna().unique())
        rc_split = st.selectbox("Rare-event calibration split", rc_splits, index=rc_splits.index("test") if "test" in rc_splits else 0, key="rare_cal_split")
        rc_models = sorted(rare_calibration["model"].dropna().unique())
        rc_model = st.selectbox("Rare-event calibration model", rc_models, key="rare_cal_model")
        rc_scopes = sorted(rare_calibration["model_scope"].dropna().unique())
        rc_scope = st.selectbox("Rare-event calibration scope", rc_scopes, index=rc_scopes.index("global") if "global" in rc_scopes else 0, key="rare_cal_scope")
        rc = rare_calibration[(rare_calibration["split"].eq(rc_split)) & (rare_calibration["model"].eq(rc_model)) & (rare_calibration["model_scope"].eq(rc_scope))].copy()
        if rc.empty:
            st.info("No rare-event calibration rows for this selection.")
        else:
            st.dataframe(rc, use_container_width=True, hide_index=True)
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=rc["mean_predicted_probability"], y=rc["empirical_event_rate"], mode="markers+lines", name="Observed"))
            fig_rc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
            fig_rc.update_layout(title="Validation-calibrated reliability curve", xaxis_title="Mean model score or bucket-calibrated risk", yaxis_title="Empirical event rate", height=420)
            st.plotly_chart(fig_rc, use_container_width=True)

    st.subheader("10. Discrete-Time Hazard Model + Leakage Audit")
    st.markdown(
        """
        The hazard model asks a shorter-horizon timing question: **did the asset enter a major drawdown state within
        the next four weeks?** Because that question can be accidentally easy if the model sees current drawdown-state
        variables, v14 trains both a full feature set and a leakage-restricted feature set. Treat the restricted result
        as the safer research estimate.
        """
    )
    if hazard_results.empty:
        st.info("Hazard results are missing. Run `python hazard_model.py`.")
    else:
        st.dataframe(hazard_results, use_container_width=True, hide_index=True)
        if not hazard_audit.empty:
            st.markdown("**Hazard leakage audit:**")
            st.dataframe(hazard_audit, use_container_width=True, hide_index=True)
            if "audit_warning" in hazard_audit.columns and hazard_audit["audit_warning"].astype(str).str.contains("SUSPICIOUS", case=False, na=False).any():
                st.warning("The full hazard model looks suspiciously stronger than the leakage-restricted model. Use restricted-feature hazard scores for research interpretation.")
        else:
            st.info("Hazard leakage audit is missing. Re-run `python hazard_model.py` from the v14 package.")
        if not hazard_current_ai.empty:
            st.markdown("**Current AI-exposed hazard/onset scores:**")
            sort_col = "hazard_score_4w" if "hazard_score_4w" in hazard_current_ai.columns else "hazard_probability_4w"
            st.dataframe(hazard_current_ai.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

    st.subheader("11. Poisson Count Model")
    st.markdown(
        """
        Poisson regression is kept separate because it models **counts**, not individual yes/no asset labels. Here it estimates
        how many assets within a segment are expected to experience a major drawdown event, rather than whether one ticker will.
        """
    )
    if poisson_results.empty:
        st.info("Poisson count model results are missing. Run `python poisson_count_model.py`.")
    else:
        st.dataframe(poisson_results, use_container_width=True, hide_index=True)
        if not poisson_current.empty:
            st.markdown("**Current segment-level count forecast:**")
            st.dataframe(poisson_current.sort_values("predicted_event_rate_next_6m", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("12. Confusion Matrix")
    available_models = sorted(predictions["model"].dropna().unique()) if "model" in predictions.columns else []
    available_splits = sorted(predictions["split"].dropna().unique()) if "split" in predictions.columns else []
    available_scopes = sorted(predictions["model_scope"].dropna().unique()) if "model_scope" in predictions.columns else ["global"]
    c1, c2, c3 = st.columns(3)
    selected_model = c1.selectbox("Model", available_models, index=available_models.index("XGBoost") if "XGBoost" in available_models else 0, key="cm_model")
    selected_scope = c2.selectbox("Model scope", available_scopes, index=available_scopes.index("global") if "global" in available_scopes else 0, key="cm_scope")
    selected_split = c3.selectbox("Split", available_splits, index=available_splits.index("test") if "test" in available_splits else 0, key="cm_split")

    pred = predictions[(predictions["model"].eq(selected_model)) & (predictions["split"].eq(selected_split))].copy()
    if "model_scope" in pred.columns:
        pred = pred[pred["model_scope"].eq(selected_scope)]
    if pred.empty:
        st.info("No prediction rows are available for this model/scope/split.")
    else:
        target_col = choose_binary_target_column(pred)
        prediction_col = choose_prediction_column(pred)
        if target_col is None or prediction_col is None:
            st.warning(
                "This prediction artifact does not contain the expected target/prediction columns. "
                f"Available columns: {list(pred.columns)}"
            )
            return
        y_true = pred[target_col].astype(int)
        y_pred = pred[prediction_col].astype(int)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        matrix = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual no burst", "Actual burst"], columns=["Predicted no burst", "Predicted burst"])
        fig_cm = px.imshow(matrix, text_auto=True, title=f"Confusion matrix: {selected_model} / {selected_scope} / {selected_split}")
        fig_cm.update_layout(height=420)
        st.plotly_chart(fig_cm, use_container_width=True)
        threshold_msg = f"Tuned threshold used: {pred['threshold_value'].iloc[0]:.3f}." if "threshold_value" in pred.columns else ""
        st.markdown(f"{threshold_msg} Target column used: `{target_col}`. False positives are research warnings that did not lead to the target drawdown; false negatives are missed historical drawdowns.")

    st.subheader("13. Feature Importance")
    if importance.empty:
        st.info("Feature importance table is empty.")
    else:
        importance_scopes = sorted(importance["model_scope"].dropna().unique()) if "model_scope" in importance.columns else ["global"]
        imp_scope = st.selectbox("Feature-importance scope", importance_scopes, index=importance_scopes.index("global") if "global" in importance_scopes else 0, key="imp_scope")
        imp_for_scope = importance[importance["model_scope"].eq(imp_scope)] if "model_scope" in importance.columns else importance
        model_for_importance = st.selectbox("Feature-importance model", sorted(imp_for_scope["model"].dropna().unique()), key="imp_model")
        top_n = st.slider("Number of features to show", min_value=5, max_value=30, value=15, step=5)
        imp = imp_for_scope[imp_for_scope["model"].eq(model_for_importance)].sort_values("importance", ascending=False).head(top_n)
        fig_imp = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h", title=f"Top features: {model_for_importance} / {imp_scope}")
        fig_imp.update_layout(height=max(420, top_n * 28))
        st.plotly_chart(fig_imp, use_container_width=True)
        if "signed_value" in imp.columns and model_for_importance == "Logistic Regression":
            st.caption("For logistic regression, signed coefficients indicate direction after scaling. Positive values increase estimated drawdown risk; negative values reduce it.")

    st.subheader("14. Imbalance Methods Lab")
    st.markdown(
        """
        This section checks whether imbalance-specific methods make the model harder to fool.
        Balanced Random Forest does **not** invent new market rows; SMOTE and SMOTE-ENN are treated as controlled experiments only.
        Keep a method only if it improves PR-AUC, MCC, top-5%/top-10% lift, calibration, and false-positive burden.
        """
    )
    if imbalance_results.empty:
        st.info("Imbalance experiment results are missing. Run `python imbalance_experiments.py` after `python ml_dataset.py`.")
    else:
        imb_test = imbalance_results[imbalance_results["split"].eq("test")].copy() if "split" in imbalance_results.columns else imbalance_results
        st.dataframe(imb_test.sort_values("pr_auc", ascending=False), use_container_width=True, hide_index=True)
        if not imbalance_topk.empty:
            st.markdown("**Imbalance-method top-risk lift:**")
            top = imbalance_topk[imbalance_topk["split"].eq("test")].copy() if "split" in imbalance_topk.columns else imbalance_topk
            st.dataframe(top.sort_values("lift_vs_base_rate", ascending=False), use_container_width=True, hide_index=True)
        report = load_report_text("imbalance_experiment_summary.md")
        if report:
            with st.expander("Imbalance experiment written interpretation"):
                st.markdown(report)

    st.subheader("15. Event-Level / Leave-One-Crisis-Style Validation")
    st.markdown(
        """
        Row-level validation can overstate evidence because one crisis creates many correlated positive ticker-weeks.
        Event validation tests whether models trained before a crisis window can rank the pre-event/event window as risky.
        This is stricter and usually less flattering, which is exactly why it matters.
        """
    )
    if event_results.empty:
        st.info("Event-validation results are missing. Run `python event_validation.py`.")
    else:
        st.dataframe(event_results.sort_values(["event_name", "pr_auc"], ascending=[True, False]), use_container_width=True, hide_index=True)
        if not event_topk.empty:
            st.markdown("**Event-level top-risk lift:**")
            st.dataframe(event_topk.sort_values("lift_vs_base_rate", ascending=False), use_container_width=True, hide_index=True)
        report = load_report_text("event_validation_summary.md")
        if report:
            with st.expander("Event-validation written interpretation"):
                st.markdown(report)

    st.subheader("16. Cost-Ratio Threshold Stress Test")
    st.markdown(
        """
        This section asks how the alert threshold changes when missed drawdowns are treated as 2x, 5x, 10x, or 20x more costly than false alarms.
        It should reveal tradeoffs, not manufacture certainty. A threshold is useful only if it improves warning quality without creating oceans of false positives.
        """
    )
    if cost_results.empty:
        st.info("Cost-threshold results are missing. Run `python cost_threshold_analysis.py`.")
    else:
        cost_test = cost_results[cost_results["split"].eq("test")].copy() if "split" in cost_results.columns else cost_results
        st.dataframe(cost_test.sort_values("cost_per_row"), use_container_width=True, hide_index=True)
        report = load_report_text("cost_threshold_summary.md")
        if report:
            with st.expander("Cost-threshold written interpretation"):
                st.markdown(report)

    st.subheader("17. Optional Firth Logistic Regression Benchmark")
    st.markdown(
        """
        Firth logistic regression is useful for small-success or separation-prone logistic models.
        Python support is uneven, so this project exports clean CSVs and an optional R script (`firth_logistic_optional.R`).
        Treat it as a statistical benchmark, not a replacement for the time-aware ML pipeline.
        """
    )
    if firth_coefficients.empty:
        st.info("Firth coefficient output is missing. Run `python firth_logistic_export.py`, then run `firth_logistic_optional.R` in R if you want this optional benchmark.")
    else:
        st.dataframe(firth_coefficients, use_container_width=True, hide_index=True)

    st.subheader("18. Current AI Cycle ML Risk Scores")
    st.caption("The primary score uses a segment-specific model when available, otherwise it falls back to the global model.")
    st.dataframe(ai_risk, use_container_width=True, hide_index=True)

    risk_col = "calibrated_empirical_risk" if "calibrated_empirical_risk" in ai_risk.columns else "primary_ml_risk_probability"
    if risk_col in ai_risk.columns:
        fig_ai = px.bar(
            ai_risk.sort_values(risk_col, ascending=True),
            x=risk_col,
            y="ticker",
            color="risk_category" if "risk_category" in ai_risk.columns else None,
            orientation="h",
            title="Current AI-exposed assets: empirical risk-ranking score",
        )
        fig_ai.update_layout(height=520, xaxis_tickformat=".0%")
        st.plotly_chart(fig_ai, use_container_width=True)

    if "ticker" in ai_risk.columns:
        ticker = st.selectbox("Inspect AI ticker", sorted(ai_risk["ticker"].dropna().unique()), key="ai_inspect")
        row = ai_risk[ai_risk["ticker"].eq(ticker)].tail(1)
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Raw model score", f"{r.get('primary_ml_risk_probability', np.nan):.1%}")
            c2.metric("Empirical bucket rate", f"{r.get('calibrated_empirical_risk', np.nan):.1%}")
            c3.metric("Primary model", str(r.get("primary_model", "n/a")))
            c4.metric("Scope", str(r.get("primary_model_scope", "n/a")))
            st.write(str(r.get("top_decile_context", "No top-decile context available.")))
            st.markdown(f"**Interpretation for {ticker}:** the ML score is a historical resemblance score. It is not a promise about what happens next.")

    st.subheader("19. Dataset Diagnostics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML weekly rows", f"{len(ml_dataset):,}")
    c2.metric("Tickers", f"{ml_dataset['ticker'].nunique():,}" if "ticker" in ml_dataset.columns else "n/a")
    c3.metric("First week", str(ml_dataset["date"].min().date()) if "date" in ml_dataset.columns else "n/a")
    c4.metric("Latest week", str(ml_dataset["date"].max().date()) if "date" in ml_dataset.columns else "n/a")
    if "asset_segment" in ml_dataset.columns:
        st.dataframe(ml_dataset["asset_segment"].value_counts().rename_axis("asset_segment").reset_index(name="rows"), use_container_width=True, hide_index=True)
    if "burst_6m" in ml_dataset.columns and "target_valid_burst_6m" in ml_dataset.columns:
        valid = ml_dataset[ml_dataset["target_valid_burst_6m"] == True]
        balance = valid["burst_6m"].value_counts().rename_axis("burst_6m").reset_index(name="rows")
        st.dataframe(balance, use_container_width=True, hide_index=True)

    st.subheader("20. Plain-English Research Summary")
    summary_text = load_ml_summary_text()
    if summary_text:
        st.markdown(summary_text)
    else:
        st.info("No reports/ml_model_summary.md file found. Run python model_evaluation.py to generate the written interpretation.")

    st.markdown(
        """
        **Bottom line:** this model does not predict the exact timing of a crash. Instead, it asks whether
        today’s conditions resemble historical pre-drawdown regimes. A high score means similar conditions
        were followed by large forward drawdowns more often than normal in the backtest, but it can still be wrong.
        """
    )


def page_ml_visual_diagnostics(data: dict[str, pd.DataFrame]) -> None:
    st.title("ML Visual Diagnostics")
    st.markdown(
        """
        This page gathers the **visual diagnostics** for the supervised crash-risk models:
        confusion matrices, ROC curves, and false-positive burden summaries. These visuals help us
        judge whether a model is merely noisy or whether it actually ranks risky historical weeks
        better than chance.
        """
    )
    predictions = data.get("ml_predictions", pd.DataFrame()).copy()
    results = data.get("ml_results", pd.DataFrame()).copy()
    if predictions.empty or results.empty:
        st.error("This page needs `ml_test_predictions.parquet` and `ml_model_results.parquet`. Re-run model_training.py and model_evaluation.py.")
        return

    tabs = st.tabs(["All confusion matrices", "ROC-AUC curves", "False positive audit"])

    with tabs[0]:
        splits = sorted(predictions["split"].dropna().unique()) if "split" in predictions.columns else []
        scopes = sorted(predictions["model_scope"].dropna().unique()) if "model_scope" in predictions.columns else []
        selected_split = st.multiselect("Prediction split(s)", splits, default=["test"] if "test" in splits else splits, key="diag_cm_splits")
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
                        (subset["model"].eq(combo["model"])) &
                        (subset["split"].eq(combo["split"])) &
                        (subset["model_scope"].eq(combo["model_scope"]))
                    ].copy()
                    metrics = build_binary_diagnostics(pred)
                    if not metrics:
                        continue
                    with cols[idx % 2]:
                        title = f"{combo['model']} / {combo['model_scope']}"
                        st.plotly_chart(plot_confusion_matrix_from_metrics(metrics, title), use_container_width=True)
                        st.caption(
                            f"TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']} | "
                            f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
                            f"FPR={metrics['false_positive_rate']:.3f}"
                        )

    with tabs[1]:
        if "probability" not in predictions.columns:
            st.info("Prediction probabilities are missing, so ROC curves cannot be drawn.")
        else:
            available_splits = sorted(predictions["split"].dropna().unique()) if "split" in predictions.columns else []
            split_choice = st.selectbox("ROC split", available_splits, index=available_splits.index("test") if "test" in available_splits else 0, key="roc_split")
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
                    fig = go.Figure()
                    auc_rows = []
                    for model_name in sorted(scope_df["model"].dropna().unique()):
                        pred = scope_df[scope_df["model"].eq(model_name)].copy()
                        target_col = choose_binary_target_column(pred)
                        if target_col is None:
                            continue
                        curve = compute_roc_points(pred[target_col], pred["probability"])
                        if curve.empty:
                            continue
                        auc_value = auc_from_curve(curve)
                        auc_rows.append({"model": model_name, "model_scope": scope_name, "split": split_choice, "roc_auc_from_curve": auc_value})
                        fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{model_name} (AUC={auc_value:.3f})"))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
                    fig.update_layout(title=f"ROC curves: {scope_name} / {split_choice}", xaxis_title="False positive rate", yaxis_title="True positive rate", height=480)
                    st.plotly_chart(fig, use_container_width=True)
                    if auc_rows:
                        st.dataframe(pd.DataFrame(auc_rows).sort_values("roc_auc_from_curve", ascending=False), use_container_width=True, hide_index=True)

    with tabs[2]:
        test_results = results[results["split"].eq("test")].copy() if "split" in results.columns else results.copy()
        if test_results.empty:
            st.info("No model results available for false-positive auditing.")
        else:
            cols = [c for c in ["model", "asset_segment", "model_scope", "split", "false_positive", "false_positive_rate", "false_positives_per_true_positive", "precision", "recall", "alert_rate", "mcc", "threshold_value"] if c in test_results.columns]
            audit_df = test_results[cols].copy()
            st.dataframe(audit_df.sort_values([c for c in ["false_positives_per_true_positive", "false_positive"] if c in audit_df.columns]), use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                if {"model", "false_positive"}.issubset(audit_df.columns):
                    fig_fp = px.bar(audit_df.sort_values("false_positive", ascending=False), x="model", y="false_positive", color="model_scope" if "model_scope" in audit_df.columns else None, title="False positives by model")
                    fig_fp.update_layout(height=420)
                    st.plotly_chart(fig_fp, use_container_width=True)
            with c2:
                metric = "false_positives_per_true_positive" if "false_positives_per_true_positive" in audit_df.columns else "false_positive"
                fig_burden = px.bar(audit_df.sort_values(metric, ascending=False), x="model", y=metric, color="model_scope" if "model_scope" in audit_df.columns else None, title=f"{pretty_metric_label(metric)} by model")
                fig_burden.update_layout(height=420)
                st.plotly_chart(fig_burden, use_container_width=True)
            burden_metrics = [m for m in ["false_positive", "false_positives_per_true_positive", "precision", "recall", "mcc", "alert_rate"] if m in audit_df.columns]
            if burden_metrics:
                melt = audit_df.melt(id_vars=[c for c in ["model", "model_scope"] if c in audit_df.columns], value_vars=burden_metrics, var_name="metric", value_name="value")
                fig_heat = px.density_heatmap(melt, x="model", y="metric", z="value", histfunc="avg", facet_col="model_scope" if "model_scope" in melt.columns else None, title="False-positive burden and related metrics heatmap")
                fig_heat.update_layout(height=500)
                st.plotly_chart(fig_heat, use_container_width=True)


def page_eda_visuals(data: dict[str, pd.DataFrame]) -> None:
    st.title("EDA Visuals")
    st.markdown(
        """
        This page performs exploratory data analysis on both the rule-based bubble dataset and the supervised ML dataset.
        The goal is to understand **coverage, missingness, class balance, distributions, and basic relationships** before trusting any model.
        """
    )
    signals = data.get("signals", pd.DataFrame()).copy()
    ml_dataset = data.get("ml_dataset", pd.DataFrame()).copy()
    if signals.empty and ml_dataset.empty:
        st.error("No EDA source tables are loaded.")
        return

    tabs = st.tabs(["Rule-based signal EDA", "ML dataset EDA", "Correlation and missingness"])

    with tabs[0]:
        if signals.empty:
            st.info("Signal table is unavailable.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                ticker_counts = signals.groupby("ticker", as_index=False).size().rename(columns={"size": "rows"}).sort_values("rows", ascending=False).head(20)
                st.plotly_chart(px.bar(ticker_counts, x="ticker", y="rows", title="Top 20 tickers by number of signal rows"), use_container_width=True)
            with c2:
                phase_col = "phase_label" if "phase_label" in signals.columns else "signal_label"
                phase_counts = signals[phase_col].fillna("Unknown").value_counts().rename_axis(phase_col).reset_index(name="rows")
                st.plotly_chart(px.bar(phase_counts, x=phase_col, y="rows", title="Distribution of rule-based labels"), use_container_width=True)

            selected_tickers = st.multiselect("Tickers for warning-score timeline", sorted(signals["ticker"].dropna().unique()), default=[t for t in ["NVDA", "QQQ", "BTC-USD"] if t in set(signals["ticker"].dropna().unique())], key="eda_signal_tickers")
            if selected_tickers:
                tmp = signals[signals["ticker"].isin(selected_tickers)].copy().sort_values("date")
                fig = px.line(tmp, x="date", y="warning_score", color="ticker", title="Warning score over time for selected tickers")
                fig.add_hline(y=55, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if ml_dataset.empty:
            st.info("ML dataset is unavailable.")
        else:
            target_col = "burst_6m_segment" if "burst_6m_segment" in ml_dataset.columns else "burst_6m" if "burst_6m" in ml_dataset.columns else None
            if target_col and f"target_valid_{target_col}" in ml_dataset.columns:
                valid = ml_dataset[ml_dataset[f"target_valid_{target_col}"] == True].copy()
            else:
                valid = ml_dataset.copy()
            c1, c2 = st.columns(2)
            with c1:
                if target_col and target_col in valid.columns:
                    target_counts = valid[target_col].fillna(-1).value_counts().rename_axis(target_col).reset_index(name="rows")
                    st.plotly_chart(px.bar(target_counts, x=target_col, y="rows", title="Target balance in usable ML rows"), use_container_width=True)
            with c2:
                if "asset_segment" in valid.columns:
                    seg_counts = valid["asset_segment"].fillna("Unknown").value_counts().rename_axis("asset_segment").reset_index(name="rows")
                    st.plotly_chart(px.bar(seg_counts, x="asset_segment", y="rows", title="Rows by asset segment"), use_container_width=True)

            if target_col and "asset_segment" in valid.columns:
                seg_rate = valid.groupby("asset_segment", as_index=False)[target_col].mean().rename(columns={target_col: "event_rate"})
                st.plotly_chart(px.bar(seg_rate, x="asset_segment", y="event_rate", title="Historical major-drawdown event rate by segment"), use_container_width=True)

            feature_choices = top_numeric_features(valid, max_features=8, exclude=[target_col] if target_col else [])
            if feature_choices and target_col:
                chosen_feature = st.selectbox("Distribution feature", feature_choices, key="eda_dist_feature")
                fig = px.histogram(valid, x=chosen_feature, color=target_col, barmode="overlay", marginal="box", title=f"Distribution of {chosen_feature} by target")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        base_df = ml_dataset if not ml_dataset.empty else signals
        if base_df.empty:
            st.info("No table is available for missingness/correlation analysis.")
        else:
            numeric_features = top_numeric_features(base_df, max_features=10, exclude=["burst_6m", "burst_6m_segment"]) 
            if numeric_features:
                corr = base_df[numeric_features].corr(numeric_only=True)
                fig_corr = px.imshow(corr, text_auto='.2f', title="Correlation heatmap for a high-coverage numeric subset")
                fig_corr.update_layout(height=520)
                st.plotly_chart(fig_corr, use_container_width=True)
            missing = base_df.isna().mean().sort_values(ascending=False).head(25).rename_axis("column").reset_index(name="missing_rate")
            st.plotly_chart(px.bar(missing, x="missing_rate", y="column", orientation="h", title="Top 25 columns by missingness"), use_container_width=True)


def page_feature_importance_visuals(data: dict[str, pd.DataFrame]) -> None:
    st.title("Feature Importance Visuals")
    st.markdown(
        """
        This page expands feature importance into several complementary visual views. Different plots answer different questions:
        **which features matter most, which ones push risk up or down, how concentrated importance is, and whether importance changes by model or segment.**
        """
    )
    importance = data.get("ml_feature_importance", pd.DataFrame()).copy()
    if importance.empty:
        st.error("Feature importance output is missing. Run model_training.py first.")
        return

    tabs = st.tabs(["Ranked bars", "Signed effects", "Heatmaps", "Cumulative importance", "Feature themes"])
    scopes = sorted(importance["model_scope"].dropna().unique()) if "model_scope" in importance.columns else ["global"]
    models = sorted(importance["model"].dropna().unique())

    with tabs[0]:
        scope = st.selectbox("Scope", scopes, index=scopes.index("global") if "global" in scopes else 0, key="fi_scope_bar")
        subset = importance[importance["model_scope"].eq(scope)] if "model_scope" in importance.columns else importance
        model_name = st.selectbox("Model", sorted(subset["model"].dropna().unique()), key="fi_model_bar")
        top_n = st.slider("Top N features", 5, 30, 15, 1, key="fi_topn_bar")
        imp = subset[subset["model"].eq(model_name)].sort_values("importance", ascending=False).head(top_n).copy()
        fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance", title=f"Top features: {model_name} / {scope}")
        fig.update_layout(height=max(420, top_n * 26))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(imp, use_container_width=True, hide_index=True)

    with tabs[1]:
        signed = importance[importance["signed_value"].notna()].copy() if "signed_value" in importance.columns else pd.DataFrame()
        if signed.empty:
            st.info("Signed coefficient plots are only available for linear models such as Logistic Regression and Elastic Net Logistic.")
        else:
            signed_scope = st.selectbox("Signed-effect scope", sorted(signed["model_scope"].dropna().unique()), index=0, key="fi_scope_signed")
            signed_models = sorted(signed[signed["model_scope"].eq(signed_scope)]["model"].dropna().unique())
            signed_model = st.selectbox("Signed-effect model", signed_models, key="fi_model_signed")
            n_signed = st.slider("Number of signed features", 5, 25, 12, 1, key="fi_topn_signed")
            sub = signed[(signed["model_scope"].eq(signed_scope)) & (signed["model"].eq(signed_model))].copy()
            sub["direction"] = np.where(sub["signed_value"] >= 0, "Raises estimated risk", "Lowers estimated risk")
            sub = sub.assign(abs_signed=np.abs(sub["signed_value"])).sort_values("abs_signed", ascending=False).head(n_signed)
            fig = px.bar(sub.sort_values("signed_value"), x="signed_value", y="feature", orientation="h", color="direction", title=f"Signed coefficient effects: {signed_model} / {signed_scope}")
            fig.update_layout(height=max(420, n_signed * 28))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        top_k = st.slider("Top features per model for heatmap", 5, 20, 10, 1, key="fi_heatmap_topk")
        heat_df = (
            importance.sort_values("importance", ascending=False)
            .groupby(["model_scope", "model"], as_index=False)
            .head(top_k)
            .copy()
        )
        heat_df["model_label"] = heat_df["model"] + " | " + heat_df["model_scope"].astype(str)
        pivot = heat_df.pivot_table(index="feature", columns="model_label", values="importance", aggfunc="max", fill_value=0.0)
        if not pivot.empty:
            fig = px.imshow(pivot, aspect="auto", title="Feature-importance heatmap across models and scopes")
            fig.update_layout(height=max(500, len(pivot) * 22))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough feature-importance data for a heatmap.")

    with tabs[3]:
        cum_scope = st.selectbox("Cumulative scope", scopes, index=scopes.index("global") if "global" in scopes else 0, key="fi_scope_cum")
        cum_subset = importance[importance["model_scope"].eq(cum_scope)] if "model_scope" in importance.columns else importance
        cum_model = st.selectbox("Cumulative model", sorted(cum_subset["model"].dropna().unique()), key="fi_model_cum")
        sub = cum_subset[cum_subset["model"].eq(cum_model)].sort_values("importance", ascending=False).copy()
        sub["rank"] = np.arange(1, len(sub) + 1)
        total = sub["importance"].sum()
        sub["importance_share"] = sub["importance"] / total if total else 0.0
        sub["cumulative_share"] = sub["importance_share"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sub["rank"], y=sub["importance_share"], name="Importance share"))
        fig.add_trace(go.Scatter(x=sub["rank"], y=sub["cumulative_share"], mode="lines+markers", name="Cumulative share", yaxis="y2"))
        fig.update_layout(
            title=f"How concentrated is importance? {cum_model} / {cum_scope}",
            xaxis_title="Feature rank",
            yaxis_title="Per-feature importance share",
            yaxis2=dict(title="Cumulative share", overlaying="y", side="right", range=[0, 1]),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        themed = importance.copy()
        themed["theme"] = themed["feature"].map(feature_theme)
        agg = themed.groupby(["model_scope", "model", "theme"], as_index=False)["importance"].sum()
        fig = px.bar(agg, x="theme", y="importance", color="model", facet_col="model_scope", barmode="group", title="Importance aggregated into interpretable feature themes")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(agg.sort_values(["model_scope", "importance"], ascending=[True, False]), use_container_width=True, hide_index=True)


def page_methods_and_diagrams(data: dict[str, pd.DataFrame]) -> None:
    st.title("Methods, Diagrams, and Formula Walkthrough")
    st.markdown(
        """
        This page explains the **entire research and code process** in plain English but with enough detail to defend the project.
        It answers three questions:
        1. **What do the scripts do, and in what order?**
        2. **How are labels, scores, thresholds, and metrics calculated?**
        3. **How does the dashboard combine rule-based monitoring with supervised ML?**
        """
    )
    tabs = st.tabs(["Pipeline diagram", "Label creation", "Threshold + metrics", "How the dashboard combines signals"])

    with tabs[0]:
        sankey = go.Figure(data=[go.Sankey(
            node=dict(label=[
                "data_ingestion.py", "feature_engineering.py", "bubble_signals.py", "ml_dataset.py",
                "model_training.py", "rare_event_analysis.py", "hazard_model.py", "poisson_count_model.py",
                "ml_inference.py", "dashboard.py"
            ]),
            link=dict(
                source=[0, 1, 2, 3, 4, 4, 4, 6, 8],
                target=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                value=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            )
        )])
        sankey.update_layout(title="End-to-end project flow", height=520)
        st.plotly_chart(sankey, use_container_width=True)
        st.markdown(
            """
            **Script order and purpose**

            - **data_ingestion.py**: downloads raw market, macro, and valuation context.
            - **feature_engineering.py**: converts raw prices and macro data into technical, valuation, and macro features.
            - **bubble_signals.py**: creates the transparent rule-based warning and recovery scores.
            - **ml_dataset.py**: reshapes weekly rows into a supervised-learning dataset and creates forward-looking historical labels.
            - **model_training.py**: trains Logistic Regression, Elastic Net Logistic, Random Forest, Balanced Random Forest, and XGBoost.
            - **rare_event_analysis.py**: audits thresholding, top-k lift, and calibration in a rare-event setting.
            - **hazard_model.py**: asks a shorter-horizon event-onset question.
            - **poisson_count_model.py**: forecasts counts of future drawdown events at the segment level.
            - **ml_inference.py**: scores the latest AI-exposed assets with trained historical-risk models.
            - **dashboard.py / app.py**: turns all outputs into an explorable research dashboard.
            """
        )

    with tabs[1]:
        st.markdown(
            """
            ### How the main supervised label is created

            The main target is a **future-looking historical label**, not an input feature:

            **burst_6m_segment = 1** if the asset experiences a segment-adjusted major drawdown over the **next 26 weekly observations**.

            Conceptually:

            1. Start at week **t**.
            2. Look forward from **t+1** through **t+26**.
            3. Compute the worst forward drawdown in that window.
            4. Compare it to a threshold that depends on the asset segment.
            5. Label week **t** as 1 if the threshold is crossed, else 0.

            This is why the model is a **historical resemblance model**. It asks whether today looks like weeks that were later followed by a major drawdown.
            """
        )
        steps_df = pd.DataFrame({
            "step": [1, 2, 3, 4, 5],
            "operation": [
                "Choose week t",
                "Look ahead 26 weekly observations",
                "Find worst forward drawdown",
                "Compare to segment threshold",
                "Assign 0/1 label to week t"
            ]
        })
        st.dataframe(steps_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown(
            """
            ### Threshold tuning and metric calculation

            After a model outputs a risk score, the project does **not** blindly use 0.50 as the warning threshold.
            Instead, it tunes the threshold on the **validation split** subject to rare-event guardrails.

            **Confusion matrix terms**
            - **TP**: warned, and a major forward drawdown happened.
            - **FP**: warned, but the drawdown did not happen.
            - **FN**: no warning, but a drawdown happened.
            - **TN**: no warning, and no drawdown happened.

            **Key formulas**
            - Precision = TP / (TP + FP)
            - Recall = TP / (TP + FN)
            - Specificity = TN / (TN + FP)
            - False positive rate = FP / (FP + TN)
            - F1 = 2 × Precision × Recall / (Precision + Recall)
            - MCC uses all four confusion-matrix cells and is harder to game with class imbalance.
            - ROC-AUC summarizes ranking quality over all cutoffs.
            - PR-AUC is especially useful when positive events are rare.
            """
        )
        metric_demo = pd.DataFrame({
            "metric": ["Precision", "Recall", "Specificity", "False positive rate", "F1"],
            "formula": [
                "TP / (TP + FP)",
                "TP / (TP + FN)",
                "TN / (TN + FP)",
                "FP / (FP + TN)",
                "2PR / (P + R)",
            ],
            "interpretation": [
                "Of the warnings, how many were historically correct?",
                "Of the true events, how many were caught?",
                "Of the calm periods, how many were correctly left alone?",
                "How often calm periods were falsely flagged.",
                "Balance between precision and recall.",
            ]
        })
        st.dataframe(metric_demo, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown(
            """
            ### How the dashboard combines rule-based and ML signals

            The system is no longer a naive “predict the exact crash date” dashboard.
            It is a **signal-confirmation and risk-ranking dashboard**.

            The logic is:
            1. Use **rule-based transparent indicators** to summarize technical stress or recovery.
            2. Use **supervised ML** to rank which historical states resembled later drawdown setups.
            3. Use **hazard modeling** for short-horizon event-onset risk.
            4. Use **Poisson count modeling** for segment-level event volume.
            5. Present the output as **confirmation**, **risk buckets**, and **historical analog evidence** instead of pretending to know the future.
            """
        )
        if not data.get("current_ai_ml_risk", pd.DataFrame()).empty:
            st.dataframe(data["current_ai_ml_risk"].head(15), use_container_width=True, hide_index=True)


def page_literature_review() -> None:
    st.title("Background Research and Literature Review")
    st.markdown(
        """
        This page provides a curated reading list for the research topic: bubbles, crashes, valuation excess,
        speculative behavior, and market fragility. These are **background references** for framing the dashboard,
        not claims that the code replicates each paper exactly.
        """
    )
    refs = {
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
    }
    for section, items in refs.items():
        st.subheader(section)
        for item in items:
            st.markdown(f"- {item}")
    st.info(
        "Use this tab as a starting point for a written literature review section in your report or presentation. "
        "You can expand it later with formal citation formatting (APA/MLA/Chicago) depending on your program requirements."
    )




def _fmt_pct(value: float | int | None, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}%}"
    except Exception:
        return "n/a"


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def build_research_story(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build a dynamic interpretation from whatever model artifacts exist.

    The dashboard should tell a coherent research story, but it must not invent
    results. This helper only uses values already saved by the pipeline.
    """
    story: dict[str, Any] = {
        "headline": "No complete model story is available yet.",
        "bullets": [],
        "tables": {},
    }

    results = data.get("ml_results", pd.DataFrame()).copy()
    top_decile = data.get("ml_top_decile", pd.DataFrame()).copy()
    top_k = data.get("ml_top_k_lift", pd.DataFrame()).copy()
    imbalance = data.get("imbalance_results", pd.DataFrame()).copy()
    imbalance_topk = data.get("imbalance_topk", pd.DataFrame()).copy()
    ai = data.get("current_ai_ml_risk", pd.DataFrame()).copy()
    hazard_current = data.get("hazard_current_ai", pd.DataFrame()).copy()
    hazard_audit = data.get("hazard_audit", pd.DataFrame()).copy()
    poisson = data.get("poisson_results", pd.DataFrame()).copy()
    cost = data.get("cost_threshold_results", pd.DataFrame()).copy()
    importance = data.get("ml_feature_importance", pd.DataFrame()).copy()

    bullets: list[str] = []

    if not results.empty and "split" in results.columns:
        test = results[results["split"].eq("test")].copy()
        if not test.empty and "pr_auc" in test.columns:
            best_pr = test.sort_values("pr_auc", ascending=False).iloc[0]
            scope = best_pr.get("asset_segment", best_pr.get("model_scope", "unknown scope"))
            bullets.append(
                f"The strongest ordinary test-set classifier by PR-AUC is **{best_pr['model']} / {scope}** "
                f"with PR-AUC {_fmt_num(best_pr.get('pr_auc'), 3)}, ROC-AUC {_fmt_num(best_pr.get('roc_auc'), 3)}, "
                f"precision {_fmt_pct(best_pr.get('precision'))}, and recall {_fmt_pct(best_pr.get('recall'))}."
            )
        if not test.empty and {"model", "asset_segment", "false_positive", "true_positive"}.issubset(test.columns):
            fp_table = test.copy()
            fp_table["false_positives_per_true_positive"] = fp_table.apply(
                lambda r: (r["false_positive"] / r["true_positive"]) if r["true_positive"] else np.inf,
                axis=1,
            )
            clean = fp_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["false_positives_per_true_positive"])
            if not clean.empty:
                best_clean = clean.sort_values("false_positives_per_true_positive").iloc[0]
                bullets.append(
                    f"The cleanest alert burden among ordinary test models is **{best_clean['model']} / {best_clean.get('asset_segment', best_clean.get('model_scope', 'unknown'))}**, "
                    f"with about {_fmt_num(best_clean['false_positives_per_true_positive'], 2)} false positives per true positive. "
                    "This is useful context because high recall alone can be fake-good when the model floods the dashboard with alerts."
                )

    if not top_decile.empty and "split" in top_decile.columns:
        td_test = top_decile[top_decile["split"].eq("test")].copy()
        if not td_test.empty:
            best_td = td_test.sort_values("lift_vs_base_rate", ascending=False).iloc[0]
            bullets.append(
                f"The clearest risk-ranking evidence comes from top-risk buckets: **{best_td['model']} / {best_td['model_scope']}** "
                f"has a top-decile event rate of {_fmt_pct(best_td.get('top_decile_event_rate'))} versus a base rate of {_fmt_pct(best_td.get('base_event_rate'))}, "
                f"a lift of {_fmt_num(best_td.get('lift_vs_base_rate'), 2)}x. That is risk enrichment, not certainty."
            )
            story["tables"]["Top-decile leaders"] = td_test.sort_values("lift_vs_base_rate", ascending=False).head(8)

    if not top_k.empty and "split" in top_k.columns:
        tk_test = top_k[top_k["split"].eq("test")].copy()
        if not tk_test.empty:
            best_tk = tk_test.sort_values("lift_vs_base_rate", ascending=False).iloc[0]
            bullets.append(
                f"The highest top-k lift is **{best_tk['model']} / {best_tk['model_scope']} / {best_tk['top_label']}**, "
                f"where the event rate is {_fmt_pct(best_tk.get('top_event_rate'))} versus base {_fmt_pct(best_tk.get('base_event_rate'))}, "
                f"for {_fmt_num(best_tk.get('lift_vs_base_rate'), 2)}x lift. This supports using the model as a ranking tool."
            )
            story["tables"]["Top-k lift leaders"] = tk_test.sort_values("lift_vs_base_rate", ascending=False).head(8)

    if not imbalance.empty and "split" in imbalance.columns:
        imb_test = imbalance[imbalance["split"].eq("test")].copy()
        if not imb_test.empty:
            best_imb = imb_test.sort_values("pr_auc", ascending=False).iloc[0]
            burden = best_imb.get("false_positives_per_true_positive", np.nan)
            bullets.append(
                f"The imbalance experiments help but are not magic. The best imbalance method by PR-AUC is **{best_imb['model']} / {best_imb['model_scope']}** "
                f"with PR-AUC {_fmt_num(best_imb.get('pr_auc'), 3)}, precision {_fmt_pct(best_imb.get('precision'))}, recall {_fmt_pct(best_imb.get('recall'))}, "
                f"and false-positive burden {_fmt_num(burden, 2)} FP per TP. Keep it only as a controlled experiment unless lift improves without alert flooding."
            )
            story["tables"]["Imbalance method leaders"] = imb_test.sort_values("pr_auc", ascending=False).head(8)

    if not ai.empty:
        if "calibrated_empirical_risk" in ai.columns:
            ai_sorted = ai.sort_values("calibrated_empirical_risk", ascending=False)
            leader = ai_sorted.iloc[0]
            bullets.append(
                f"Among current AI-exposed assets, the highest empirical bucket risk is **{leader['ticker']}** at {_fmt_pct(leader.get('calibrated_empirical_risk'))}. "
                f"Its raw model score is {_fmt_pct(leader.get('primary_ml_risk_probability'))}, but the dashboard should emphasize the empirical bucket rate because calibration is imperfect."
            )
            story["tables"]["Current AI empirical-risk ranking"] = ai_sorted[[c for c in [
                "ticker", "asset_segment", "warning_score", "drawdown_pct", "distance_from_200dma", "weekly_rsi",
                "primary_model", "primary_model_scope", "primary_ml_risk_probability", "calibrated_empirical_risk",
                "above_tuned_threshold", "risk_category"
            ] if c in ai_sorted.columns]].head(12)
        hot_cols = [c for c in ["ticker", "warning_score", "distance_from_200dma", "weekly_rsi", "drawdown_pct"] if c in ai.columns]
        if hot_cols:
            hot = ai.sort_values("warning_score", ascending=False).head(5)
            story["tables"]["Current technical warning leaders"] = hot[hot_cols]

    if not hazard_audit.empty:
        suspicious = hazard_audit[hazard_audit.get("audit_warning", "").astype(str).str.contains("SUSPICIOUS", na=False)] if "audit_warning" in hazard_audit.columns else pd.DataFrame()
        if not suspicious.empty:
            bullets.append(
                "The hazard model audit shows a major drop from full features to leakage-restricted features. "
                "That means the original full hazard model is probably partly detecting current drawdown-state variables. Treat the restricted hazard model as the safer research signal."
            )
            story["tables"]["Hazard leakage audit"] = hazard_audit
    if not hazard_current.empty:
        haz = hazard_current.sort_values("hazard_score_4w", ascending=False).head(8)
        story["tables"]["Current restricted hazard leaders"] = haz

    if not poisson.empty and {"split", "mean_actual_count", "mean_predicted_count"}.issubset(poisson.columns):
        ptest = poisson[poisson["split"].eq("test")]
        if not ptest.empty:
            r = ptest.iloc[0]
            bullets.append(
                f"The Poisson count model is conceptually useful but currently weak: in the test split, mean actual event count is {_fmt_num(r.get('mean_actual_count'), 2)} "
                f"while mean predicted count is {_fmt_num(r.get('mean_predicted_count'), 2)}, with RMSE {_fmt_num(r.get('rmse'), 2)}. Treat it as a stress-count experiment, not a reliable forecast."
            )
            story["tables"]["Poisson count model summary"] = poisson

    if not cost.empty and {"precision", "recall"}.issubset(cost.columns):
        clean_cost = cost.replace([np.inf, -np.inf], np.nan).dropna(subset=["precision", "recall"])
        if not clean_cost.empty:
            precise = clean_cost.sort_values("precision", ascending=False).iloc[0]
            bullets.append(
                f"The cost-threshold stress test confirms the tradeoff: the cleanest high-precision alerts can reach precision {_fmt_pct(precise.get('precision'))}, "
                f"but recall is only {_fmt_pct(precise.get('recall'))}. Clean alerts miss many events, while broad alerts create noise."
            )
            story["tables"]["High-precision cost-threshold examples"] = clean_cost.sort_values("precision", ascending=False).head(8)

    if not importance.empty:
        if {"feature", "importance"}.issubset(importance.columns):
            imp = importance.sort_values("importance", ascending=False).head(10)
            top_features = ", ".join(imp["feature"].astype(str).head(5).tolist())
            bullets.append(
                f"Feature importance shows the model story is a mix of valuation, trend, volatility, and macro stress rather than one magic variable. Top fields include: {top_features}."
            )
            story["tables"]["Top feature-importance rows"] = imp

    if bullets:
        story["headline"] = (
            "The dashboard shows a coherent but modest story: the models are weak as exact crash predictors, "
            "but useful as risk-ranking and signal-confirmation tools when top-risk buckets, false-positive burden, and leakage checks are read together."
        )
        story["bullets"] = bullets
    return story


def page_research_story(data: dict[str, pd.DataFrame]) -> None:
    st.title("Research Story & Interpretation")
    st.warning(
        "This page interprets saved model artifacts as historical research signals. It does not provide personal financial advice, "
        "and it does not claim to predict exact market tops or bottoms."
    )
    story = build_research_story(data)
    st.subheader("Executive story")
    st.info(story["headline"])

    st.subheader("What the results are saying")
    for bullet in story.get("bullets", []):
        st.markdown(f"- {bullet}")

    st.subheader("The actual story, in plain English")
    st.markdown(
        """
        The model stack is not telling us that it can forecast crashes with surgical precision. The stronger pattern is narrower and more useful:

        1. **Top-risk buckets matter more than yes/no alerts.** When the model ranks historical weeks in the highest-risk buckets, event rates often rise above the base rate. That is evidence of risk enrichment.
        2. **Different segments behave differently.** Speculative/high-volatility assets show stronger top-bucket lift, but also much noisier false-positive behavior. Broad indexes are cleaner conceptually but harder to move because major index drawdowns are rarer and more macro-driven.
        3. **Calibration remains fragile.** Raw model scores can be much higher than the empirical event rate. That is why the dashboard now emphasizes empirical bucket rates and risk ranks instead of treating raw scores as literal probabilities.
        4. **Hazard modeling is useful only after leakage control.** The full hazard model looked too strong. The restricted model is safer because it removes target-adjacent state variables.
        5. **Poisson count modeling is experimental.** It asks the right count question, but current test error is too high to trust as a live stress-count forecast.
        6. **Feature importance supports a multi-cause narrative.** The models are reacting to a combination of valuation stretch, trend behavior, volatility, rates, curve behavior, and macro/financial stress. That is exactly what we would expect from bubble-cycle monitoring.

        The story is therefore **not** “the model predicts the next AI crash.” The story is:

        > Current and historical market states can be ranked by resemblance to prior forward-drawdown environments, but the signal is probabilistic, noisy, and segment-dependent. The dashboard is strongest as a research cockpit for signal confirmation, not a trade oracle.
        """
    )

    st.subheader("Key supporting tables")
    tables = story.get("tables", {})
    if not tables:
        st.info("No supporting tables were found. Run the full pipeline first.")
    for name, df in tables.items():
        st.markdown(f"### {name}")
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Recommended interpretation rules")
    st.markdown(
        """
        - Treat **top-5% and top-10% lift** as stronger evidence than a single binary prediction.
        - Treat **false positives per true positive** as the honesty meter for any alert system.
        - Treat **raw probabilities** as model scores unless calibration proves they behave like real probabilities.
        - Trust **restricted hazard** more than full hazard when the leakage audit says the full model is suspicious.
        - Use **feature importance** to explain what kind of market condition the model is reacting to.
        - Do not use any one model alone. The research value comes from agreement among technical rules, ML risk ranking, hazard/onset scores, and historical bucket evidence.
        """
    )



def page_data_quality(data: dict[str, pd.DataFrame]) -> None:
    st.title("Data Quality and Limitations")
    st.markdown(
        """
        **Core risks and limits:**

        - **Survivorship bias:** delisted names such as Yahoo, AOL, Bear Stearns, Lehman, and some SPACs may not be available through free APIs.
        - **SEC valuation mapping risk:** the pipeline now uses SEC EDGAR companyfacts for filing-date-aware historical valuation rows, but XBRL concepts differ by company and era. Rows are kept only when the code can trace revenue/shares/prices cleanly.
        - **Macro frequency mismatch:** market data is daily, while GDP is quarterly and many macro indicators are monthly or weekly. The code stores frequency explicitly.
        - **Ticker history issues:** ETFs like QQQ begin in 1999, so earlier dot-com buildup requires Nasdaq Composite proxies.
        - **False positives:** a hot RSI or extreme distance from the 200DMA can stay hot for a long time.
        - **False negatives:** some crashes happen before slow macro indicators clearly deteriorate.
        - **Different bubble anatomy:** AI infrastructure firms with real earnings are not identical to unprofitable dot-com firms, crypto tokens, or housing-credit products.
        - **No exact top/bottom detection:** the dashboard tests rules. It does not promise perfect timing.
        """
    )

    st.subheader("Loaded table checks")
    checks = []
    for name, df in data.items():
        checks.append(
            {
                "Table": name,
                "Rows": len(df),
                "Columns": len(df.columns) if not df.empty else 0,
                "Min date": df["date"].min().date() if not df.empty and "date" in df.columns else "n/a",
                "Max date": df["date"].max().date() if not df.empty and "date" in df.columns else "n/a",
            }
        )
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)


def main() -> None:
    data = load_all_data()
    page = st.sidebar.radio(
        "Dashboard page",
        [
            "Overview",
            "Bubble Explorer",
            "Historical Bubble Comparison",
            "Bubble Vital Signs",
            "Current AI Cycle Monitor",
            "ML Signal Confirmation Lab",
            "Research Story & Interpretation",
            "ML Visual Diagnostics",
            "EDA Visuals",
            "Feature Importance Visuals",
            "Methods, Diagrams, and Formula Walkthrough",
            "Background Research and Literature Review",
            "Investment Simulator",
            "Data Quality and Limitations",
        ],
    )

    if page == "Overview":
        page_overview(data)
        return

    pages_that_need_core_data = {
        "Bubble Explorer",
        "Historical Bubble Comparison",
        "Bubble Vital Signs",
        "Current AI Cycle Monitor",
        "ML Signal Confirmation Lab",
        "Research Story & Interpretation",
        "ML Visual Diagnostics",
        "EDA Visuals",
        "Feature Importance Visuals",
        "Methods, Diagrams, and Formula Walkthrough",
        "Investment Simulator",
        "Data Quality and Limitations",
    }
    if page in pages_that_need_core_data and not require_data(data):
        return

    if page == "Bubble Explorer":
        page_bubble_explorer(data)
    elif page == "Historical Bubble Comparison":
        page_historical_comparison(data)
    elif page == "Bubble Vital Signs":
        page_vital_signs(data)
    elif page == "Current AI Cycle Monitor":
        page_ai_monitor(data)
    elif page == "ML Signal Confirmation Lab":
        page_machine_learning_crash_risk(data)
    elif page == "Research Story & Interpretation":
        page_research_story(data)
    elif page == "ML Visual Diagnostics":
        page_ml_visual_diagnostics(data)
    elif page == "EDA Visuals":
        page_eda_visuals(data)
    elif page == "Feature Importance Visuals":
        page_feature_importance_visuals(data)
    elif page == "Methods, Diagrams, and Formula Walkthrough":
        page_methods_and_diagrams(data)
    elif page == "Background Research and Literature Review":
        page_literature_review()
    elif page == "Investment Simulator":
        page_investment_simulator(data)
    elif page == "Data Quality and Limitations":
        page_data_quality(data)


if __name__ == "__main__":
    main()
