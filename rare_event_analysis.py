"""
Rare-event post-processing for the ML crash-risk models.

Run after:
    python model_training.py

This script does not retrain the base classifiers. It takes their saved
validation/test predictions and adds rare-event analytics that are more useful
than a default 0.50 cutoff:

1. stricter threshold tuning with minimum precision constraints;
2. percentile risk buckets;
3. calibrated probabilities using validation-bin empirical event rates;
4. top-decile and top-5% lift analysis;
5. dashboard-ready rare-event conclusions.

The reason this exists separately from model_training.py is engineering hygiene:
model_training.py fits models; this script audits whether the model scores are
useful as rare-event risk rankings.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import BASE_DIR, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "rare_event_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rare_event_analysis")

PREDICTIONS_PATH = PROCESSED_DIR / "ml_test_predictions.parquet"
OUTPUT_METRICS = PROCESSED_DIR / "ml_rare_event_metrics.parquet"
OUTPUT_THRESHOLDS = PROCESSED_DIR / "ml_rare_event_thresholds.parquet"
OUTPUT_BUCKETS = PROCESSED_DIR / "ml_risk_bucket_analysis.parquet"
OUTPUT_TOPK = PROCESSED_DIR / "ml_top_k_lift_analysis.parquet"
OUTPUT_CALIBRATED = PROCESSED_DIR / "ml_calibrated_predictions.parquet"
OUTPUT_CALIBRATION_CURVES = PROCESSED_DIR / "ml_rare_event_calibration_curves.parquet"
OUTPUT_CONCLUSIONS = PROCESSED_DIR / "ml_rare_event_conclusions.parquet"
REPORTS_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORTS_DIR / "rare_event_model_summary.md"

TARGET_CANDIDATES = ["burst_6m_segment", "burst_6m"]


def debug_print(message: str) -> None:
    print(f"[RARE_EVENT_ANALYSIS DEBUG] {message}")
    logger.info(message)


def detect_target(df: pd.DataFrame) -> str:
    for target in TARGET_CANDIDATES:
        if target in df.columns:
            return target
    raise ValueError(f"Prediction table does not contain any target column from {TARGET_CANDIDATES}.")


def load_predictions() -> tuple[pd.DataFrame, str]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}. Run python model_training.py first.")
    df = pd.read_parquet(PREDICTIONS_PATH)
    if df.empty:
        raise ValueError("ml_test_predictions.parquet is empty. Re-run model_training.py.")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    target = detect_target(df)
    df = df[df[target].notna()].copy()
    df[target] = df[target].astype(int)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["probability", target])
    debug_print(f"Loaded predictions: shape={df.shape}, target={target}")
    return df, target


def scope_min_precision(scope: str, base_rate: float) -> float:
    """Minimum precision floor by segment.

    The floor prevents degenerate thresholds that flag almost everything. The
    floor is intentionally relative to the base rate because 8% precision can be
    meaningful for broad-index crashes but weak for speculative assets where the
    base event rate may already be 30%.
    """
    if not np.isfinite(base_rate):
        base_rate = 0.10
    if scope == "broad_index_etf":
        return float(max(0.05, base_rate * 1.75))
    if scope == "mega_cap_ai_tech":
        return float(max(0.10, base_rate * 1.50))
    if scope == "speculative_high_vol":
        return float(max(0.35, base_rate * 1.10))
    if scope == "global":
        return float(max(0.12, base_rate * 1.50))
    return float(max(0.10, base_rate * 1.50))


def scope_max_alert_rate(scope: str, base_rate: float) -> float:
    """Maximum fraction of rows that may be flagged as positive."""
    if scope == "broad_index_etf":
        return 0.20
    if scope == "mega_cap_ai_tech":
        return 0.30
    if scope == "speculative_high_vol":
        return 0.60
    return 0.35 if base_rate < 0.20 else 0.60


def choose_constrained_threshold(y_true: np.ndarray, y_prob: np.ndarray, scope: str) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    finite = np.isfinite(y_prob)
    y_true = y_true[finite]
    y_prob = y_prob[finite]
    if len(y_true) == 0:
        return {"threshold_value": 0.50, "threshold_strategy": "fallback_empty", "minimum_precision_required": np.nan}

    base_rate = float(np.mean(y_true))
    min_precision = scope_min_precision(scope, base_rate)
    max_alert = scope_max_alert_rate(scope, base_rate)

    candidates = set(np.linspace(0.005, 0.995, 199).round(4))
    for q in [50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99]:
        candidates.add(round(float(np.nanpercentile(y_prob, q)), 4))
    candidates = sorted(c for c in candidates if 0 <= c <= 1)

    best = None
    best_unconstrained = None
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        alert_rate = float(y_pred.mean())
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        beta = 1.25
        f_beta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall) if precision + recall else 0.0
        row = {
            "threshold_value": float(threshold),
            "validation_precision": float(precision),
            "validation_recall": float(recall),
            "validation_f1": float(f1),
            "validation_f_beta_1_25": float(f_beta),
            "validation_alert_rate": float(alert_rate),
            "validation_base_event_rate": float(base_rate),
            "minimum_precision_required": float(min_precision),
            "maximum_alert_rate_allowed": float(max_alert),
        }
        if best_unconstrained is None or f1 > best_unconstrained[0]:
            best_unconstrained = (f1, row)
        if precision >= min_precision and alert_rate <= max_alert:
            if best is None or f_beta > best[0] or (np.isclose(f_beta, best[0]) and precision > best[1]["validation_precision"]):
                best = (f_beta, row)

    if best is not None:
        out = best[1]
        out["threshold_strategy"] = "constrained_precision_and_alert_rate"
        return out

    conservative_threshold = float(np.nanpercentile(y_prob, 90))
    out = best_unconstrained[1] if best_unconstrained else {}
    out.update({
        "threshold_value": conservative_threshold,
        "threshold_strategy": "fallback_90th_percentile_constraint_unmet",
        "minimum_precision_required": float(min_precision),
        "maximum_alert_rate_allowed": float(max_alert),
        "validation_base_event_rate": float(base_rate),
    })
    return out


def metric_row(df: pd.DataFrame, target: str, threshold: float) -> dict[str, Any]:
    y_true = df[target].astype(int).to_numpy()
    y_prob = df["probability"].astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_rows": int(len(df)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "threshold_value": float(threshold),
        "alert_rate": float(np.mean(y_pred)) if len(y_pred) else np.nan,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) >= 2 and len(np.unique(y_true)) >= 2 else 0.0,
        "brier_score": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positives_per_true_positive": float(fp / tp) if tp > 0 else np.inf,
        "false_negatives_per_true_positive": float(fn / tp) if tp > 0 else np.inf,
        "precision_lift_vs_base_rate": float(precision_score(y_true, y_pred, zero_division=0) / np.mean(y_true)) if np.mean(y_true) > 0 else np.nan,
    }


def fit_validation_calibration_map(val: pd.DataFrame, target: str, n_bins: int = 10) -> pd.DataFrame:
    if val.empty or val["probability"].nunique(dropna=True) < 2:
        return pd.DataFrame()
    tmp = val.copy()
    try:
        tmp["probability_bin"] = pd.qcut(tmp["probability"], q=min(n_bins, tmp["probability"].nunique()), duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    global_rate = float(tmp[target].mean())
    for bin_value, g in tmp.groupby("probability_bin", observed=True):
        # Laplace smoothing keeps tiny bins from mapping to exact 0 or 1.
        positives = float(g[target].sum())
        n = float(len(g))
        empirical = (positives + global_rate) / (n + 1.0)
        rows.append({
            "bin_left": float(bin_value.left),
            "bin_right": float(bin_value.right),
            "n_rows": int(len(g)),
            "mean_predicted_probability": float(g["probability"].mean()),
            "empirical_event_rate": float(empirical),
            "raw_empirical_event_rate": float(g[target].mean()),
        })
    return pd.DataFrame(rows)


def apply_calibration_map(probabilities: pd.Series, cal_map: pd.DataFrame) -> pd.Series:
    if cal_map.empty:
        return probabilities.astype(float)
    centers = (cal_map["bin_left"] + cal_map["bin_right"]) / 2.0
    out = []
    for value in probabilities.astype(float):
        containing = cal_map[(cal_map["bin_left"] <= value) & (value <= cal_map["bin_right"])]
        if containing.empty:
            idx = (centers - value).abs().idxmin()
            out.append(float(cal_map.loc[idx, "empirical_event_rate"]))
        else:
            out.append(float(containing.iloc[-1]["empirical_event_rate"]))
    return pd.Series(out, index=probabilities.index, dtype=float)


def add_risk_buckets(df: pd.DataFrame, prob_col: str = "probability") -> pd.DataFrame:
    out = df.copy()
    out["risk_percentile"] = out[prob_col].rank(pct=True, method="average")
    out["risk_bucket"] = pd.cut(
        out["risk_percentile"],
        bins=[-np.inf, 0.50, 0.75, 0.90, 0.95, np.inf],
        labels=["Bottom 50%", "50-75%", "75-90%", "90-95%", "Top 5%"],
    ).astype(str)
    return out


def bucket_analysis(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    rows = []
    base = float(df[target].mean()) if len(df) else np.nan
    for bucket, g in df.groupby("risk_bucket", observed=False):
        rate = float(g[target].mean()) if len(g) else np.nan
        rows.append({
            "model": df["model"].iloc[0],
            "model_scope": df["model_scope"].iloc[0],
            "split": df["split"].iloc[0],
            "risk_bucket": str(bucket),
            "n_rows": int(len(g)),
            "base_event_rate": base,
            "bucket_event_rate": rate,
            "lift_vs_base_rate": float(rate / base) if base and np.isfinite(rate) else np.nan,
            "mean_probability": float(g["probability"].mean()) if len(g) else np.nan,
            "mean_calibrated_probability": float(g["calibrated_probability"].mean()) if len(g) and "calibrated_probability" in g.columns else np.nan,
        })
    return rows


def top_k_lift(df: pd.DataFrame, target: str, top_fraction: float) -> dict[str, Any]:
    if df.empty:
        return {}
    n_top = max(1, int(np.ceil(len(df) * top_fraction)))
    top = df.sort_values("probability", ascending=False).head(n_top)
    base = float(df[target].mean())
    rate = float(top[target].mean()) if len(top) else np.nan
    return {
        "model": df["model"].iloc[0],
        "model_scope": df["model_scope"].iloc[0],
        "split": df["split"].iloc[0],
        "top_fraction": float(top_fraction),
        "top_label": "Top 5%" if np.isclose(top_fraction, 0.05) else "Top 10%",
        "n_rows": int(len(df)),
        "top_rows": int(len(top)),
        "base_event_rate": base,
        "top_event_rate": rate,
        "lift_vs_base_rate": float(rate / base) if base and np.isfinite(rate) else np.nan,
        "true_positives_in_top": int(top[target].sum()) if len(top) else 0,
        "min_probability_in_top": float(top["probability"].min()) if len(top) else np.nan,
    }


def build_markdown(conclusions: pd.DataFrame, topk: pd.DataFrame, buckets: pd.DataFrame) -> str:
    lines = [
        "# Rare-Event Model Audit",
        "",
        "This report checks whether the crash-risk models behave like useful rare-event rankers rather than simple yes/no classifiers.",
        "",
        "## Key idea",
        "The dashboard should not trust raw accuracy or unconstrained recall. Major drawdowns are rare, so the useful questions are: do high-risk score buckets contain more future drawdowns than normal, and do calibrated probabilities roughly match empirical event rates?",
        "",
    ]
    if not conclusions.empty:
        lines += ["## Executive rare-event conclusion"]
        for _, row in conclusions.iterrows():
            lines.append(f"- **{row['item']}**: {row['value']}")
        lines.append("")
    if not topk.empty:
        best = topk[topk["split"].eq("test")].sort_values("lift_vs_base_rate", ascending=False).head(8)
        lines.append("## Best test-split top-risk lift")
        for _, row in best.iterrows():
            lines.append(f"- {row['model']} / {row['model_scope']} / {row['top_label']}: event rate {row['top_event_rate']:.1%} vs base {row['base_event_rate']:.1%}, lift {row['lift_vs_base_rate']:.2f}x.")
    return "\n".join(lines)


def main() -> None:
    debug_print("Rare-event analysis start")
    predictions, target = load_predictions()
    metrics_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    calibration_curve_rows: list[pd.DataFrame] = []
    calibrated_frames: list[pd.DataFrame] = []

    group_cols = ["model", "model_scope"]
    for (model, scope), group in predictions.groupby(group_cols, dropna=False):
        val = group[group["split"].eq("validation")].copy()
        if val.empty:
            debug_print(f"Skipping {model}/{scope}: no validation split available for threshold/calibration.")
            continue
        threshold_info = choose_constrained_threshold(val[target].to_numpy(), val["probability"].to_numpy(), str(scope))
        threshold_rows.append({"model": model, "model_scope": scope, **threshold_info})
        cal_map = fit_validation_calibration_map(val, target)
        if not cal_map.empty:
            tmp_map = cal_map.copy()
            tmp_map["model"] = model
            tmp_map["model_scope"] = scope
            tmp_map["split"] = "validation_map"
            calibration_curve_rows.append(tmp_map)

        for split, split_df in group.groupby("split"):
            split_df = split_df.copy()
            split_df["calibrated_probability"] = apply_calibration_map(split_df["probability"], cal_map)
            split_df = add_risk_buckets(split_df, "probability")
            split_df["rare_event_threshold_value"] = float(threshold_info["threshold_value"])
            split_df["rare_event_prediction"] = (split_df["probability"] >= float(threshold_info["threshold_value"])).astype(int)
            calibrated_frames.append(split_df)

            m = metric_row(split_df, target, float(threshold_info["threshold_value"]))
            metrics_rows.append({"model": model, "model_scope": scope, "split": split, "target": target, **m})
            bucket_rows.extend(bucket_analysis(split_df, target))
            topk_rows.append(top_k_lift(split_df, target, 0.10))
            topk_rows.append(top_k_lift(split_df, target, 0.05))

            # Calibration curve for the split after applying the validation map.
            cal_split = fit_validation_calibration_map(split_df.rename(columns={"calibrated_probability":"probability_original"}).assign(probability=split_df["calibrated_probability"]), target)
            if not cal_split.empty:
                cal_split["model"] = model
                cal_split["model_scope"] = scope
                cal_split["split"] = split
                calibration_curve_rows.append(cal_split)

    metrics = pd.DataFrame(metrics_rows)
    thresholds = pd.DataFrame(threshold_rows)
    buckets = pd.DataFrame(bucket_rows)
    topk = pd.DataFrame([r for r in topk_rows if r])
    calibrated = pd.concat(calibrated_frames, ignore_index=True) if calibrated_frames else pd.DataFrame()
    calibration_curves = pd.concat(calibration_curve_rows, ignore_index=True) if calibration_curve_rows else pd.DataFrame()

    conclusions = []
    test = metrics[metrics["split"].eq("test")].copy() if not metrics.empty else pd.DataFrame()
    if not test.empty:
        for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "mcc"]:
            row = test.sort_values(metric, ascending=False).head(1).iloc[0]
            conclusions.append({"item": f"best_{metric}", "value": f"{row['model']} / {row['model_scope']} = {row[metric]:.3f}"})
    top_test = topk[topk["split"].eq("test")].copy() if not topk.empty else pd.DataFrame()
    if not top_test.empty:
        row = top_test.sort_values("lift_vs_base_rate", ascending=False).head(1).iloc[0]
        conclusions.append({"item": "best_top_risk_lift", "value": f"{row['model']} / {row['model_scope']} / {row['top_label']} lift {row['lift_vs_base_rate']:.2f}x"})
    conclusions.append({"item": "interpretation", "value": "Use this as a rare-event risk-ranking audit. Strong top-bucket lift is more meaningful than high accuracy or unconstrained recall."})
    conclusions_df = pd.DataFrame(conclusions)

    metrics.to_parquet(OUTPUT_METRICS, index=False)
    thresholds.to_parquet(OUTPUT_THRESHOLDS, index=False)
    buckets.to_parquet(OUTPUT_BUCKETS, index=False)
    topk.to_parquet(OUTPUT_TOPK, index=False)
    calibrated.to_parquet(OUTPUT_CALIBRATED, index=False)
    calibration_curves.to_parquet(OUTPUT_CALIBRATION_CURVES, index=False)
    conclusions_df.to_parquet(OUTPUT_CONCLUSIONS, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_markdown(conclusions_df, topk, buckets), encoding="utf-8")

    debug_print(f"Saved rare-event metrics to {OUTPUT_METRICS}")
    debug_print("Rare-event analysis complete")


if __name__ == "__main__":
    main()
