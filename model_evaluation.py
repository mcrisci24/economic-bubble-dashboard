"""
Create human-readable evaluation summaries for the ML crash-risk models.

Run after:
    python model_training.py

Outputs:
    reports/ml_model_summary.md
    data/processed/ml_dashboard_conclusions.parquet

This script is deliberately more interpretive than model_training.py. The goal is
not only to calculate metrics, but to explain what they mean for a rare-event
financial risk model.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_DIR, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "model_evaluation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model_evaluation")

RESULTS_PATH = PROCESSED_DIR / "ml_model_results.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "ml_test_predictions.parquet"
IMPORTANCE_PATH = PROCESSED_DIR / "ml_feature_importance.parquet"
TOP_DECILE_PATH = PROCESSED_DIR / "ml_top_decile_analysis.parquet"
THRESHOLDS_PATH = PROCESSED_DIR / "ml_thresholds.parquet"
CALIBRATION_PATH = PROCESSED_DIR / "ml_calibration_table.parquet"
RARE_EVENT_METRICS_PATH = PROCESSED_DIR / "ml_rare_event_metrics.parquet"
RISK_BUCKET_ANALYSIS_PATH = PROCESSED_DIR / "ml_risk_bucket_analysis.parquet"
TOP_K_LIFT_PATH = PROCESSED_DIR / "ml_top_k_lift_analysis.parquet"
RARE_EVENT_CONCLUSIONS_PATH = PROCESSED_DIR / "ml_rare_event_conclusions.parquet"
CONCLUSIONS_PATH = PROCESSED_DIR / "ml_dashboard_conclusions.parquet"
REPORTS_DIR = BASE_DIR / "reports"
SUMMARY_PATH = REPORTS_DIR / "ml_model_summary.md"


def debug_print(message: str) -> None:
    print(f"[MODEL_EVALUATION DEBUG] {message}")
    logger.info(message)


def load_parquet(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {path}. Run python model_training.py first.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    debug_print(f"Loaded {path.name}: shape={df.shape}")
    return df


def pick_best(test_results: pd.DataFrame, metric: str) -> dict[str, object]:
    if test_results.empty or metric not in test_results.columns:
        return {"metric": metric, "model": "n/a", "model_scope": "n/a", "value": np.nan}
    usable = test_results.dropna(subset=[metric]).copy()
    if usable.empty:
        return {"metric": metric, "model": "n/a", "model_scope": "n/a", "value": np.nan}
    row = usable.sort_values(metric, ascending=False).iloc[0]
    return {"metric": metric, "model": row["model"], "model_scope": row.get("asset_segment", row.get("model_scope", "global")), "value": float(row[metric])}


def classify_model_strength(test_results: pd.DataFrame) -> tuple[str, str]:
    """Classify overall model quality using PR-AUC lift and ROC-AUC.

    PR-AUC must be judged relative to the positive class rate. A PR-AUC of 0.15
    can be weak if the event rate is 0.12, but meaningful if the event rate is
    0.02. This function writes the dashboard conclusion honestly rather than
    pretending every fitted model is useful.
    """
    if test_results.empty:
        return "Unavailable", "No test results were available."
    global_test = test_results[test_results.get("asset_segment", "global").eq("global")] if "asset_segment" in test_results.columns else test_results
    scope = global_test if not global_test.empty else test_results
    if scope.empty or "positive_rate" not in scope.columns or "pr_auc" not in scope.columns:
        return "Unavailable", "The results table is missing positive_rate or PR-AUC."

    best_pr = scope.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=False).head(1)
    best_roc = scope.dropna(subset=["roc_auc"]).sort_values("roc_auc", ascending=False).head(1)
    if best_pr.empty:
        return "Unavailable", "No usable PR-AUC values were available."

    pr = float(best_pr["pr_auc"].iloc[0])
    base = float(best_pr["positive_rate"].iloc[0]) if pd.notna(best_pr["positive_rate"].iloc[0]) else np.nan
    roc = float(best_roc["roc_auc"].iloc[0]) if not best_roc.empty else np.nan
    lift = pr / base if base and np.isfinite(base) else np.nan

    if np.isfinite(lift) and lift >= 2.0 and np.isfinite(roc) and roc >= 0.70:
        label = "Promising research signal"
    elif np.isfinite(lift) and lift >= 1.35 and np.isfinite(roc) and roc >= 0.60:
        label = "Modest but usable experimental signal"
    elif np.isfinite(lift) and lift > 1.05:
        label = "Weak experimental signal"
    else:
        label = "Very weak / not reliable yet"

    explanation = (
        f"Best global test PR-AUC is {pr:.3f} versus a base event rate of {base:.3f}, "
        f"for a PR-AUC lift of {lift:.2f}x. Best global test ROC-AUC is {roc:.3f}."
    )
    return label, explanation


def summarize_top_decile(top_decile: pd.DataFrame) -> str:
    if top_decile.empty:
        return "Top-decile analysis was not available."
    test = top_decile[top_decile["split"].eq("test")].copy()
    if test.empty:
        test = top_decile.copy()
    best = test.dropna(subset=["lift_vs_base_rate"]).sort_values("lift_vs_base_rate", ascending=False).head(5)
    if best.empty:
        return "Top-decile analysis did not find usable lift values."
    lines = ["When the models rank observations in their highest-risk 10%, the strongest test-split lifts were:"]
    for _, row in best.iterrows():
        lines.append(
            f"- {row['model']} ({row.get('model_scope', 'global')}): top-decile event rate "
            f"{row['top_decile_event_rate']:.3f} vs base {row['base_event_rate']:.3f}, "
            f"lift {row['lift_vs_base_rate']:.2f}x."
        )
    return "\n".join(lines)


def summarize_thresholds(thresholds: pd.DataFrame) -> str:
    if thresholds.empty:
        return "Tuned thresholds were not available."
    lines = [
        "The project no longer uses 0.50 as the automatic warning cutoff. Each model threshold is selected on the validation split by maximizing F1, then applied to the later test split. This makes the warning threshold appropriate for a rare-event setting where a probability far below 50% may still be historically elevated."
    ]
    display = thresholds.sort_values(["model_scope", "model"]).head(20)
    for _, row in display.iterrows():
        lines.append(
            f"- {row['model']} ({row['model_scope']}): threshold {row['threshold_value']:.3f}, "
            f"validation F1 {row.get('validation_f1_at_threshold', np.nan):.3f}."
        )
    return "\n".join(lines)


def summarize_calibration(calibration: pd.DataFrame) -> str:
    if calibration.empty:
        return "Calibration tables were not available."
    test = calibration[calibration["split"].eq("test")].copy()
    if test.empty:
        test = calibration[calibration["split"].eq("validation")].copy()
    if test.empty:
        return "Calibration data exists, but no validation/test rows were available for summary."
    # Use the highest mean-probability bins, because those are the risk warnings
    # a dashboard user will care about most.
    high = test.sort_values("mean_predicted_probability", ascending=False).head(5)
    lines = [
        "Calibration check: high predicted probabilities should correspond to high empirical event rates. The highest-score bins show:"
    ]
    for _, row in high.iterrows():
        lines.append(
            f"- {row['model']} ({row['model_scope']}, {row['split']}): mean predicted "
            f"{row['mean_predicted_probability']:.3f}, empirical event rate {row['empirical_event_rate']:.3f}, rows {int(row['n_rows'])}."
        )
    return "\n".join(lines)


def make_conclusions(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"].eq("test")].copy() if "split" in results.columns else pd.DataFrame()
    rows = []
    for metric in ["roc_auc", "pr_auc", "recall", "precision", "f1", "mcc"]:
        rows.append(pick_best(test, metric))
    strength_label, strength_explanation = classify_model_strength(test)
    rows.append({"metric": "overall_strength", "model": strength_label, "model_scope": "dashboard_conclusion", "value": np.nan})
    rows.append({"metric": "overall_strength_explanation", "model": strength_explanation, "model_scope": "dashboard_conclusion", "value": np.nan})
    return pd.DataFrame(rows)


def build_markdown_report(
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_decile: pd.DataFrame,
    calibration: pd.DataFrame,
    importance: pd.DataFrame,
    conclusions: pd.DataFrame,
) -> str:
    test = results[results["split"].eq("test")].copy() if "split" in results.columns else pd.DataFrame()
    strength_label, strength_explanation = classify_model_strength(test)

    best_roc = pick_best(test, "roc_auc")
    best_pr = pick_best(test, "pr_auc")
    best_recall = pick_best(test, "recall")
    best_precision = pick_best(test, "precision")
    best_mcc = pick_best(test, "mcc")

    lines = [
        "# Machine Learning Crash-Risk Model Summary",
        "",
        "## Research framing",
        "The models do not predict the exact timing of a bubble burst. They estimate whether today's feature pattern resembles historical weeks that were followed by a 30% or larger forward drawdown over roughly six months.",
        "",
        "## Dashboard conclusion",
        f"**Overall model strength:** {strength_label}.",
        strength_explanation,
        "",
        "## Best test-split models",
        f"- Best ROC-AUC: {best_roc['model']} ({best_roc['model_scope']}), value={best_roc['value']:.3f}" if pd.notna(best_roc["value"]) else "- Best ROC-AUC: unavailable",
        f"- Best PR-AUC: {best_pr['model']} ({best_pr['model_scope']}), value={best_pr['value']:.3f}" if pd.notna(best_pr["value"]) else "- Best PR-AUC: unavailable",
        f"- Best recall: {best_recall['model']} ({best_recall['model_scope']}), value={best_recall['value']:.3f}" if pd.notna(best_recall["value"]) else "- Best recall: unavailable",
        f"- Best precision: {best_precision['model']} ({best_precision['model_scope']}), value={best_precision['value']:.3f}" if pd.notna(best_precision["value"]) else "- Best precision: unavailable",
        f"- Best MCC: {best_mcc['model']} ({best_mcc['model_scope']}), value={best_mcc['value']:.3f}" if pd.notna(best_mcc["value"]) else "- Best MCC: unavailable",
        "",
        "## Top-decile risk analysis",
        summarize_top_decile(top_decile),
        "",
        "## Tuned thresholds",
        summarize_thresholds(thresholds),
        "",
        "## Calibration check",
        summarize_calibration(calibration),
        "",
        "## Why accuracy alone is misleading",
        "Major forward drawdowns are rare. A model can achieve high accuracy by predicting 'no burst' for nearly everything, while still missing most actual drawdowns. PR-AUC, MCC, false positives per true positive, top-risk-bucket lift, and calibration are more useful for this type of risk-monitoring task.",
        "",
        "## Feature interpretation",
    ]

    if importance.empty:
        lines.append("Feature-importance data was not available.")
    else:
        for (scope, model), g in importance.groupby(["model_scope", "model"]):
            top = g.sort_values("importance", ascending=False).head(8)
            features = ", ".join(top["feature"].astype(str).tolist())
            lines.append(f"- {model} ({scope}) top features: {features}")

    lines.extend([
        "",
        "## Bottom line",
        "Treat the ML system as an experimental research layer. It can rank historical resemblance to past pre-drawdown regimes, but it is not a stand-alone trading system and it is not financial advice.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    debug_print("Model evaluation summary start")
    results = load_parquet(RESULTS_PATH)
    thresholds = load_parquet(THRESHOLDS_PATH, required=False)
    top_decile = load_parquet(TOP_DECILE_PATH, required=False)
    calibration = load_parquet(CALIBRATION_PATH, required=False)
    rare_metrics = load_parquet(RARE_EVENT_METRICS_PATH, required=False)
    risk_buckets = load_parquet(RISK_BUCKET_ANALYSIS_PATH, required=False)
    top_k_lift = load_parquet(TOP_K_LIFT_PATH, required=False)
    rare_conclusions = load_parquet(RARE_EVENT_CONCLUSIONS_PATH, required=False)
    importance = load_parquet(IMPORTANCE_PATH, required=False)

    conclusions = make_conclusions(results)
    CONCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conclusions.to_parquet(CONCLUSIONS_PATH, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown_report(results, thresholds, top_decile, calibration, importance, conclusions)
    SUMMARY_PATH.write_text(markdown, encoding="utf-8")
    debug_print(f"Saved dashboard conclusions to {CONCLUSIONS_PATH}")
    debug_print(f"Saved ML summary to {SUMMARY_PATH}")
    debug_print("Model evaluation summary complete")


if __name__ == "__main__":
    main()
