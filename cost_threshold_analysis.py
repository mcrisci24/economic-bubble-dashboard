"""
Cost-ratio threshold stress tests for crash-risk predictions.

Run after:
    python model_training.py

Why this exists:
    Rare-event models often force a tradeoff between false negatives and false
    positives. In market-risk monitoring, missing a major drawdown can be costly,
    but too many false alarms can also make a signal unusable. This script tests
    several false-negative cost ratios on validation predictions and applies the
    selected thresholds to the test split.

Outputs:
    data/processed/ml_cost_threshold_results.parquet
    data/processed/ml_cost_thresholds.parquet
    reports/cost_threshold_summary.md
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score

from config import BASE_DIR, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "cost_threshold_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cost_threshold_analysis")

PREDICTIONS_PATH = PROCESSED_DIR / "ml_test_predictions.parquet"
OUTPUT_RESULTS = PROCESSED_DIR / "ml_cost_threshold_results.parquet"
OUTPUT_THRESHOLDS = PROCESSED_DIR / "ml_cost_thresholds.parquet"
REPORT_PATH = BASE_DIR / "reports" / "cost_threshold_summary.md"
TARGET_CANDIDATES = ["burst_6m_segment", "burst_6m"]
COST_RATIOS = [1, 2, 5, 10, 20]


def debug_print(message: str) -> None:
    print(f"[COST_THRESHOLD DEBUG] {message}")
    logger.info(message)


def detect_target(df: pd.DataFrame) -> str:
    for target in TARGET_CANDIDATES:
        if target in df.columns:
            return target
    raise ValueError(f"Prediction table does not contain one of {TARGET_CANDIDATES}")


def load_predictions() -> tuple[pd.DataFrame, str]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}. Run python model_training.py first.")
    df = pd.read_parquet(PREDICTIONS_PATH)
    target = detect_target(df)
    df = df[df[target].notna()].copy()
    df[target] = df[target].astype(int)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["probability", target])
    return df, target


def choose_cost_threshold(val: pd.DataFrame, target: str, fn_cost_ratio: float) -> dict[str, Any]:
    y_true = val[target].astype(int).to_numpy()
    y_prob = val["probability"].astype(float).to_numpy()
    candidates = sorted(set(np.linspace(0.005, 0.995, 199).round(4)) | {round(float(np.nanpercentile(y_prob, q)), 4) for q in [50, 60, 70, 75, 80, 85, 90, 95, 97.5, 99]})
    best = None
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        total_cost = float(fp + fn_cost_ratio * fn)
        cost_per_row = total_cost / len(y_true)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        alert_rate = float(pred.mean())
        row = {
            "fn_cost_ratio": float(fn_cost_ratio),
            "threshold_value": float(threshold),
            "validation_total_cost": total_cost,
            "validation_cost_per_row": cost_per_row,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_alert_rate": alert_rate,
            "validation_false_positive": int(fp),
            "validation_false_negative": int(fn),
            "validation_true_positive": int(tp),
        }
        # Tie-break: lower cost, then higher precision, then lower alert rate.
        score = (cost_per_row, -precision, alert_rate)
        if best is None or score < best[0]:
            best = (score, row)
    return best[1]


def evaluate_cost_threshold(df: pd.DataFrame, target: str, threshold: float, fn_cost_ratio: float) -> dict[str, Any]:
    y_true = df[target].astype(int).to_numpy()
    y_prob = df["probability"].astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = float(fp + fn_cost_ratio * fn)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    return {
        "n_rows": int(len(df)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "threshold_value": float(threshold),
        "fn_cost_ratio": float(fn_cost_ratio),
        "alert_rate": float(np.mean(y_pred)) if len(y_pred) else np.nan,
        "precision": precision,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) >= 2 and len(np.unique(y_pred)) >= 2 else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positives_per_true_positive": float(fp / tp) if tp > 0 else np.inf,
        "total_cost": total_cost,
        "cost_per_row": total_cost / len(df) if len(df) else np.nan,
        "precision_lift_vs_base_rate": float(precision / np.mean(y_true)) if np.mean(y_true) > 0 else np.nan,
    }


def build_report(results: pd.DataFrame) -> str:
    lines = [
        "# Cost-Ratio Threshold Stress Test",
        "",
        "This report tests how model alerts change when false negatives are treated as 1x, 2x, 5x, 10x, or 20x as costly as false positives.",
        "The goal is not to find a magical threshold. The goal is to expose the tradeoff between missed drawdowns and false alarms.",
        "",
    ]
    if not results.empty:
        test = results[results["split"].eq("test")].copy()
        lines.append("## Lowest test cost per row")
        for _, row in test.sort_values("cost_per_row").head(15).iterrows():
            lines.append(f"- {row['model']} / {row['model_scope']} / FN cost {row['fn_cost_ratio']:.0f}x: threshold={row['threshold_value']:.3f}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, FP/TP={row['false_positives_per_true_positive']:.2f}, cost/row={row['cost_per_row']:.3f}")
    return "\n".join(lines)


def main() -> None:
    debug_print("Cost-threshold analysis start")
    predictions, target = load_predictions()
    results: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for (model, scope), group in predictions.groupby(["model", "model_scope"], dropna=False):
        val = group[group["split"].eq("validation")].copy()
        test = group[group["split"].eq("test")].copy()
        if val.empty or test.empty:
            continue
        for ratio in COST_RATIOS:
            threshold_info = choose_cost_threshold(val, target, ratio)
            thresholds.append({"model": model, "model_scope": scope, **threshold_info})
            for split_name, split_df in [("validation", val), ("test", test)]:
                results.append({
                    "model": model,
                    "model_scope": scope,
                    "split": split_name,
                    **evaluate_cost_threshold(split_df, target, float(threshold_info["threshold_value"]), ratio),
                })
    results_df = pd.DataFrame(results)
    thresholds_df = pd.DataFrame(thresholds)
    results_df.to_parquet(OUTPUT_RESULTS, index=False)
    thresholds_df.to_parquet(OUTPUT_THRESHOLDS, index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results_df), encoding="utf-8")
    debug_print("Cost-threshold analysis complete")


if __name__ == "__main__":
    main()
