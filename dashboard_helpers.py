"""Shared helpers for the Economic Bubble Monitoring dashboard.

This module extracts the small, reusable utility functions that previously lived
inside `dashboard.py`. Keeping them here makes `dashboard.py` itself a thinner
page-routing shell, makes each helper unit-testable, and gives every page a
single source of truth for things like ROC math, column resolution, and the
"this artifact is missing, run this script" notice.

The functions here intentionally avoid Streamlit dependencies where possible.
Anything that calls Streamlit (e.g. `missing_artifact_notice`) is clearly
labelled as a UI helper and left isolated at the bottom of the file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st

from config import BASE_DIR, PROCESSED_DIR

logger = logging.getLogger("dashboard.helpers")


# ---------------------------------------------------------------------------
# Generation-script registry. Used by `missing_artifact_notice` to tell users
# exactly which command to run to produce a missing parquet/csv/markdown file.
# Keep keys aligned with `load_all_data` keys in dashboard.py.
# ---------------------------------------------------------------------------
ARTIFACT_TO_SCRIPT: dict[str, str] = {
    # Core data
    "prices": "data_ingestion.py and feature_engineering.py",
    "macro": "data_ingestion.py",
    "macro_features": "feature_engineering.py",
    "signals": "bubble_signals.py",
    "ai_monitor": "bubble_signals.py",
    "valuations": "data_ingestion.py (without --skip-sec) or sec_fundamentals.py",
    "valuation_features": "feature_engineering.py",
    "valuation_snapshot": "data_ingestion.py",
    # ML core
    "ml_dataset": "ml_dataset.py",
    "ml_results": "model_training.py",
    "ml_predictions": "model_training.py",
    "ml_feature_importance": "model_training.py",
    "ml_thresholds": "model_training.py",
    "ml_top_decile": "model_training.py",
    "ml_calibration": "model_training.py",
    "ml_conclusions": "model_evaluation.py",
    # Rare-event layer
    "ml_rare_event_metrics": "rare_event_analysis.py",
    "ml_rare_event_thresholds": "rare_event_analysis.py",
    "ml_risk_buckets": "rare_event_analysis.py",
    "ml_top_k_lift": "rare_event_analysis.py",
    "ml_calibrated_predictions": "rare_event_analysis.py",
    "ml_rare_event_calibration": "rare_event_analysis.py",
    "ml_rare_event_conclusions": "rare_event_analysis.py",
    # Hazard / Poisson
    "hazard_results": "hazard_model.py",
    "hazard_predictions": "hazard_model.py",
    "hazard_audit": "hazard_model.py",
    "hazard_current_ai": "hazard_model.py",
    "poisson_results": "poisson_count_model.py",
    "poisson_predictions": "poisson_count_model.py",
    "poisson_current": "poisson_count_model.py",
    # Current AI scoring
    "current_ai_ml_risk": "ml_inference.py",
    # Imbalance / event / cost / firth
    "imbalance_results": "imbalance_experiments.py",
    "imbalance_predictions": "imbalance_experiments.py",
    "imbalance_thresholds": "imbalance_experiments.py",
    "imbalance_topk": "imbalance_experiments.py",
    "event_validation_results": "event_validation.py",
    "event_validation_topk": "event_validation.py",
    "cost_threshold_results": "cost_threshold_analysis.py",
    "cost_thresholds": "cost_threshold_analysis.py",
    "firth_coefficients": "firth_logistic_export.py (then firth_logistic_optional.R in R)",
    # Phase 8 — transformed-feature experiment (optional)
    "transformed_results":      "feature_transformation_experiments.py",
    "transformed_predictions":  "feature_transformation_experiments.py",
    "transformed_importance":   "feature_transformation_experiments.py",
    "transformed_top_k_lift":   "feature_transformation_experiments.py",
    "transformed_calibration":  "feature_transformation_experiments.py",
    "transformed_feature_list": "feature_transformation_experiments.py",
}


# ---------------------------------------------------------------------------
# File loading. Streamlit's `@st.cache_data` lets each parquet/CSV load once
# per session. `load_all_data` returns a dict keyed by short artifact name so
# every page can look up what it needs without rediscovering paths.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_parquet_file(filename: str) -> pd.DataFrame:
    """Load a parquet from data/processed; return empty df if absent.

    Returning an empty frame (instead of raising) lets each page decide whether
    it can degrade gracefully or whether it should show a missing-artifact
    notice via `missing_artifact_notice`.
    """
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
    """Single entry point each page calls. New artifacts go here, not inline."""
    return {
        # Core market + macro
        "prices": load_parquet_file("asset_prices.parquet"),
        "macro": load_parquet_file("macro_indicators.parquet"),
        "macro_features": load_parquet_file("macro_features.parquet"),
        "signals": load_parquet_file("bubble_signal_scores.parquet"),
        "ai_monitor": load_parquet_file("current_ai_monitor.parquet"),
        "valuations": load_parquet_file("valuation_metrics.parquet"),
        "valuation_features": load_parquet_file("valuation_features.parquet"),
        "valuation_snapshot": load_parquet_file("current_valuation_snapshot.parquet"),
        # ML core
        "ml_dataset": load_parquet_file("ml_burst_dataset.parquet"),
        "ml_results": load_parquet_file("ml_model_results.parquet"),
        "ml_predictions": load_parquet_file("ml_test_predictions.parquet"),
        "ml_feature_importance": load_parquet_file("ml_feature_importance.parquet"),
        "ml_thresholds": load_parquet_file("ml_thresholds.parquet"),
        "ml_top_decile": load_parquet_file("ml_top_decile_analysis.parquet"),
        "ml_calibration": load_parquet_file("ml_calibration_table.parquet"),
        "ml_conclusions": load_parquet_file("ml_dashboard_conclusions.parquet"),
        # Rare-event layer
        "ml_rare_event_metrics": load_parquet_file("ml_rare_event_metrics.parquet"),
        "ml_rare_event_thresholds": load_parquet_file("ml_rare_event_thresholds.parquet"),
        "ml_risk_buckets": load_parquet_file("ml_risk_bucket_analysis.parquet"),
        "ml_top_k_lift": load_parquet_file("ml_top_k_lift_analysis.parquet"),
        "ml_calibrated_predictions": load_parquet_file("ml_calibrated_predictions.parquet"),
        "ml_rare_event_calibration": load_parquet_file("ml_rare_event_calibration_curves.parquet"),
        "ml_rare_event_conclusions": load_parquet_file("ml_rare_event_conclusions.parquet"),
        # Hazard / Poisson
        "hazard_results": load_parquet_file("hazard_model_results.parquet"),
        "hazard_predictions": load_parquet_file("hazard_model_predictions.parquet"),
        "hazard_audit": load_parquet_file("hazard_leakage_audit.parquet"),
        "hazard_current_ai": load_parquet_file("current_ai_hazard_scores.parquet"),
        "poisson_results": load_parquet_file("poisson_count_model_results.parquet"),
        "poisson_predictions": load_parquet_file("poisson_count_predictions.parquet"),
        "poisson_current": load_parquet_file("current_poisson_count_forecast.parquet"),
        # Current AI ML scoring
        "current_ai_ml_risk": load_parquet_file("current_ai_ml_risk_scores.parquet"),
        # Imbalance / event-level / cost / firth
        "imbalance_results": load_parquet_file("imbalance_experiment_results.parquet"),
        "imbalance_predictions": load_parquet_file("imbalance_experiment_predictions.parquet"),
        "imbalance_thresholds": load_parquet_file("imbalance_experiment_thresholds.parquet"),
        "imbalance_topk": load_parquet_file("imbalance_experiment_topk.parquet"),
        "event_validation_results": load_parquet_file("event_validation_results.parquet"),
        "event_validation_topk": load_parquet_file("event_validation_topk.parquet"),
        "cost_threshold_results": load_parquet_file("ml_cost_threshold_results.parquet"),
        "cost_thresholds": load_parquet_file("ml_cost_thresholds.parquet"),
        "firth_coefficients": load_csv_file("firth_logistic_coefficients.csv"),
        # Phase 8 — transformed-feature experiment outputs (optional)
        "transformed_results":      load_parquet_file("transformed_feature_model_results.parquet"),
        "transformed_predictions":  load_parquet_file("transformed_feature_predictions.parquet"),
        "transformed_importance":   load_parquet_file("transformed_feature_importance.parquet"),
        "transformed_top_k_lift":   load_parquet_file("transformed_feature_top_k_lift.parquet"),
        "transformed_calibration":  load_parquet_file("transformed_feature_calibration.parquet"),
        "transformed_feature_list": load_parquet_file("transformed_feature_list.parquet"),
    }


def load_report_text(filename: str) -> str:
    """Read a markdown research report from /reports. Empty string if absent."""
    path = BASE_DIR / "reports" / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_ml_summary_text() -> str:
    return load_report_text("ml_model_summary.md")


# ---------------------------------------------------------------------------
# Target / prediction column resolution. The supervised target was renamed
# from `burst_6m` to `burst_6m_segment` in v12+. Different artifact tables
# (predictions, imbalance experiments, event-validation) use slightly
# different column names. These helpers pick the right one at runtime.
# ---------------------------------------------------------------------------
def choose_binary_target_column(df: pd.DataFrame) -> str | None:
    """Return the best available target column for confusion-matrix logic.

    Order of preference encodes our modeling decision: prefer the
    segment-adjusted label, fall back to the legacy uniform label, then any
    generic 'target'/'y_true' columns used by experiment tables.
    """
    candidates = ["burst_6m_segment", "burst_6m", "target", "y_true", "actual"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def choose_prediction_column(df: pd.DataFrame) -> str | None:
    candidates = ["prediction", "predicted_label", "y_pred"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def ml_dataset_target_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """For the ML training dataset, return (target_col, target_valid_flag_col).

    The previous dashboard hard-coded `burst_6m` + `target_valid_burst_6m`.
    That silently produced empty diagnostics when the dataset only carried
    the newer `burst_6m_segment` label. This helper finds whichever pair is
    actually present in the file.
    """
    for tgt in ("burst_6m_segment", "burst_6m"):
        if tgt in df.columns:
            flag = f"target_valid_{tgt}"
            return tgt, (flag if flag in df.columns else None)
    return None, None


def risk_bucket_order() -> list[str]:
    return ["Bottom 50%", "50-75%", "75-90%", "90-95%", "Top 5%"]


# ---------------------------------------------------------------------------
# ROC / AUC math. We replace the previous `np.trapezoid` call with sklearn's
# `auc`, which is already a project dependency and works identically across
# NumPy 1.x and 2.x. The trapz/trapezoid shim exists only as a defensive
# fallback in case sklearn ever fails to import for some reason.
# ---------------------------------------------------------------------------
def _trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    """NumPy-version-safe trapezoid integration.

    NumPy 2.0 renamed `np.trapz` to `np.trapezoid` and started emitting a
    DeprecationWarning for `np.trapz`. NumPy 1.x has only `np.trapz`. Using
    `getattr` lets the same code work in both worlds without a try/except.
    """
    func = getattr(np, "trapezoid", np.trapz)
    return float(func(y, x))


def compute_roc_points(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Build a ROC curve without depending on sklearn plotting helpers."""
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)
    tmp = (
        pd.DataFrame({"y_true": y_true, "y_score": y_score})
        .dropna()
        .sort_values("y_score", ascending=False)
    )
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
    curve = pd.concat(
        [
            pd.DataFrame({"y_score": [np.inf], "tp": [0], "fp": [0], "tpr": [0.0], "fpr": [0.0]}),
            curve,
            pd.DataFrame({"y_score": [-np.inf], "tp": [pos], "fp": [neg], "tpr": [1.0], "fpr": [1.0]}),
        ],
        ignore_index=True,
    )
    return curve.sort_values(["fpr", "tpr"]).reset_index(drop=True)


def auc_from_curve(curve: pd.DataFrame) -> float:
    """ROC-AUC from a curve returned by `compute_roc_points`.

    Prefers `sklearn.metrics.auc` for numerical stability; falls back to the
    NumPy trapezoid shim if sklearn is somehow unavailable. Both return
    NaN-safe floats so callers can chart them without extra guards.
    """
    if curve.empty:
        return float("nan")
    try:
        from sklearn.metrics import auc as _sk_auc  # local import keeps import-time light

        return float(_sk_auc(curve["fpr"].to_numpy(), curve["tpr"].to_numpy()))
    except Exception:  # pragma: no cover - defensive fallback
        return _trapezoid_auc(curve["fpr"].to_numpy(), curve["tpr"].to_numpy())


# ---------------------------------------------------------------------------
# Binary confusion-matrix diagnostics.
# ---------------------------------------------------------------------------
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
        "false_positives_per_true_positive": float(fp / tp) if tp > 0 else float("inf"),
    }
    if prob_col is not None:
        out["roc_curve"] = compute_roc_points(y_true, pred[prob_col])
    else:
        out["roc_curve"] = pd.DataFrame()
    return out


# ---------------------------------------------------------------------------
# Feature inspection helpers.
# ---------------------------------------------------------------------------
def top_numeric_features(
    df: pd.DataFrame,
    max_features: int = 12,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Return the most populated numeric columns; high-coverage cols come first."""
    exclude_set = set(exclude or [])
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_set]
    if not numeric:
        return []
    completeness = df[numeric].notna().mean().sort_values(ascending=False)
    return completeness.head(max_features).index.tolist()


def feature_theme(feature: str) -> str:
    """Bucket a feature name into one of five interpretable themes.

    Used by the Feature Importance "themes" tab to roll up per-feature
    importances into something a non-expert can read at a glance.
    """
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


# ---------------------------------------------------------------------------
# Formatting helpers used across pages.
# ---------------------------------------------------------------------------
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


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:,.2f}%"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    """Pandas-friendly percent formatter used by the research-story builder."""
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}%}"
    except Exception:
        return "n/a"


def fmt_num(value: float | int | None, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# UI helpers — these touch Streamlit and are intended to be called from page
# functions only.
# ---------------------------------------------------------------------------
def missing_artifact_notice(
    artifact_key_or_name: str,
    *,
    severity: str = "warning",
    script_hint: str | None = None,
) -> None:
    """Render a consistent 'this artifact is missing, run this script' message.

    `artifact_key_or_name` is either a load_all_data key (preferred), or a
    plain description for ad-hoc notices. `script_hint` lets the caller override
    the default script string when the artifact comes from multiple stages.
    """
    script = script_hint or ARTIFACT_TO_SCRIPT.get(artifact_key_or_name)
    pretty_name = artifact_key_or_name.replace("_", " ")
    if script:
        msg = (
            f"**Required artifact `{pretty_name}` is missing.** "
            f"Run `python {script}` from the project folder to generate it. "
            f"The rest of the dashboard still works without this section."
        )
    else:
        msg = (
            f"**`{pretty_name}` is missing.** Re-run the relevant pipeline step. "
            f"The dashboard will not invent placeholder values."
        )
    if severity == "error":
        st.error(msg)
    elif severity == "info":
        st.info(msg)
    else:
        st.warning(msg)


def missing_artifacts_block(
    required: dict[str, pd.DataFrame],
    *,
    section_label: str = "this section",
) -> bool:
    """If any required artifact is empty, render a grouped notice and return True.

    Returns True when the page should stop early. Pages that have many
    optional sub-tables can call this once for the truly required ones, then
    use `missing_artifact_notice` per-section for optional extras.
    """
    missing = [name for name, df in required.items() if df.empty]
    if not missing:
        return False
    scripts = sorted(
        {ARTIFACT_TO_SCRIPT.get(name, "the upstream pipeline step") for name in missing}
    )
    st.error(
        f"**{section_label.capitalize()} cannot render** because these artifacts are missing: "
        f"{', '.join(missing)}.\n\n"
        f"Run these scripts (in order) to produce them:\n\n"
        + "\n".join(f"- `python {s}`" for s in scripts)
        + "\n\nThis dashboard intentionally refuses to invent placeholder charts."
    )
    return True


def require_core_data(data: dict[str, pd.DataFrame]) -> bool:
    """Guard used by pages that need at least prices+signals to function."""
    missing = [name for name in ["prices", "signals"] if data.get(name, pd.DataFrame()).empty]
    if missing:
        st.error(
            "Required processed data is missing. Run these commands from the project folder:\n\n"
            "- `python data_ingestion.py`\n"
            "- `python feature_engineering.py`\n"
            "- `python bubble_signals.py`\n\n"
            f"Missing: {', '.join(missing)}"
        )
        return False
    return True
