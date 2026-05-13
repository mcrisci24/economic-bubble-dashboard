"""
Event-level validation for historical bubble/crash regimes.

Run after:
    python ml_dataset.py

Why this exists:
    Row-level validation can overstate sample size because one historical crash
    creates many positive ticker-week labels across many correlated assets.
    Event-level validation asks a stricter question: can a model trained on data
    before a historical episode identify elevated risk during the pre-event and
    event window?

Design:
    For each crisis window, the test set is the 26 weeks leading into the event
    plus the event itself. The training set uses only rows before that pre-event
    window. This is stricter than random validation and closer to the research
    question: "would this have warned us before/during a new regime?"

Outputs:
    data/processed/event_validation_results.parquet
    data/processed/event_validation_predictions.parquet
    data/processed/event_validation_topk.parquet
    reports/event_validation_summary.md
"""
from __future__ import annotations

import logging
import sys
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
from model_training import (
    TARGET,
    clean_feature_matrix,
    clean_model_dataset,
    infer_asset_segment,
    make_model_set,
    select_feature_columns,
    choose_threshold_on_validation,
)

LOG_FILE = LOG_DIR / "event_validation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("event_validation")

DATASET_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"
RESULTS_PATH = PROCESSED_DIR / "event_validation_results.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "event_validation_predictions.parquet"
TOPK_PATH = PROCESSED_DIR / "event_validation_topk.parquet"
REPORT_PATH = BASE_DIR / "reports" / "event_validation_summary.md"

CRISIS_WINDOWS = [
    {"event_name": "1929_crash_great_depression", "start": "1929-09-01", "end": "1933-06-30"},
    {"event_name": "nifty_fifty_1973_1974", "start": "1973-01-01", "end": "1974-12-31"},
    {"event_name": "japan_asset_bubble_unwind", "start": "1990-01-01", "end": "1992-12-31"},
    {"event_name": "dotcom_bust", "start": "2000-03-01", "end": "2002-10-31"},
    {"event_name": "global_financial_crisis", "start": "2007-10-01", "end": "2009-03-31"},
    {"event_name": "crypto_2017_2018", "start": "2017-12-01", "end": "2018-12-31"},
    {"event_name": "covid_crash", "start": "2020-02-01", "end": "2020-04-30"},
    {"event_name": "speculative_tech_crypto_2021_2022", "start": "2021-11-01", "end": "2022-12-31"},
    {"event_name": "rate_hike_tech_drawdown_2022", "start": "2022-01-01", "end": "2022-12-31"},
]


def debug_print(message: str) -> None:
    print(f"[EVENT_VALIDATION DEBUG] {message}")
    logger.info(message)


def load_labeled_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python ml_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["asset_segment"] = df["ticker"].astype(str).map(infer_asset_segment)
    valid_col = f"target_valid_{TARGET}"
    labeled = df[(df[valid_col] == True) & df[TARGET].notna()].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)
    feature_cols = select_feature_columns(labeled)
    labeled = clean_model_dataset(labeled, feature_cols, context="event_validation labeled dataset")
    return labeled.sort_values(["date", "ticker"]).reset_index(drop=True), feature_cols


def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def metric_row(event: str, model: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_train: int) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    base = float(np.mean(y_true)) if len(y_true) else np.nan
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    return {
        "event_name": event,
        "model": model,
        "n_train_rows": int(n_train),
        "n_test_rows": int(len(y_true)),
        "positive_rate": base,
        "threshold_value": float(threshold),
        "alert_rate": float(np.mean(y_pred)) if len(y_pred) else np.nan,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) >= 2 and len(np.unique(y_true)) >= 2 else 0.0,
        "brier_score": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positives_per_true_positive": float(fp / tp) if tp > 0 else np.inf,
        "precision_lift_vs_base_rate": float(precision / base) if base else np.nan,
    }


def topk_for_event(pred: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = float(pred[TARGET].mean()) if len(pred) else np.nan
    for frac, label in [(0.10, "Top 10%"), (0.05, "Top 5%")]:
        top_n = max(1, int(np.ceil(len(pred) * frac)))
        top = pred.sort_values("probability", ascending=False).head(top_n)
        rate = float(top[TARGET].mean()) if len(top) else np.nan
        rows.append({
            "event_name": pred["event_name"].iloc[0],
            "model": pred["model"].iloc[0],
            "top_label": label,
            "base_event_rate": base,
            "top_event_rate": rate,
            "lift_vs_base_rate": float(rate / base) if base and np.isfinite(rate) else np.nan,
            "top_rows": int(len(top)),
            "true_positives_in_top": int(top[TARGET].sum()) if len(top) else 0,
        })
    return rows


def run_event(df: pd.DataFrame, feature_cols: list[str], event: dict[str, str]) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    event_name = event["event_name"]
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])
    pre_start = start - pd.Timedelta(days=183)

    train = df[df["date"] < pre_start].copy()
    # A small validation slice is taken from the final 20% of the training rows by date.
    # This avoids using the test crisis window to pick a threshold.
    if train["date"].nunique() < 50 or train[TARGET].nunique() < 2:
        debug_print(f"Skipping event={event_name}: not enough pre-event training history.")
        return [], [], []
    unique_train_dates = np.array(sorted(train["date"].unique()))
    val_cut = unique_train_dates[int(len(unique_train_dates) * 0.80)]
    train_fit = train[train["date"] <= val_cut].copy()
    val = train[train["date"] > val_cut].copy()
    test = df[(df["date"] >= pre_start) & (df["date"] <= end)].copy()

    if train_fit[TARGET].nunique() < 2 or val.empty or val[TARGET].nunique() < 2 or test.empty or test[TARGET].nunique() < 2:
        debug_print(f"Skipping event={event_name}: train/validation/test class coverage insufficient.")
        return [], [], []

    debug_print(f"Event={event_name}; train_fit={len(train_fit):,}; val={len(val):,}; test={len(test):,}; test positives={int(test[TARGET].sum())}")
    X_train = clean_feature_matrix(train_fit[feature_cols], feature_cols, context=f"event {event_name} train")
    y_train = train_fit[TARGET].astype(int)
    X_val = clean_feature_matrix(val[feature_cols], feature_cols, context=f"event {event_name} validation")
    y_val = val[TARGET].astype(int).to_numpy()
    X_test = clean_feature_matrix(test[feature_cols], feature_cols, context=f"event {event_name} test")
    y_test = test[TARGET].astype(int).to_numpy()

    results: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []
    topk: list[dict[str, Any]] = []
    models = make_model_set(feature_cols, y_train)
    for name, model in models.items():
        try:
            debug_print(f"Training event-validation model={name}; event={event_name}")
            model.fit(X_train, y_train)
            val_prob = predict_probability(model, X_val)
            threshold = float(choose_threshold_on_validation(y_val, val_prob)["threshold_value"])
            prob = predict_probability(model, X_test)
            results.append(metric_row(event_name, name, y_test, prob, threshold, len(train_fit)))
            pred = test[["date", "ticker", "asset_segment", TARGET]].copy()
            pred["event_name"] = event_name
            pred["model"] = name
            pred["probability"] = prob
            pred["threshold_value"] = threshold
            pred["prediction"] = (prob >= threshold).astype(int)
            preds.append(pred)
            topk.extend(topk_for_event(pred))
        except Exception as exc:  # noqa: BLE001
            debug_print(f"Skipping model={name}; event={event_name}; error={exc}")
    return results, preds, topk


def build_report(results: pd.DataFrame, topk: pd.DataFrame) -> str:
    lines = [
        "# Event-Level Validation Summary",
        "",
        "This report tests whether models trained before a historical event could rank the pre-event/event window as risky.",
        "It is intentionally stricter than row-level validation because many positive ticker-week labels are clustered inside the same crises.",
        "",
    ]
    if not results.empty:
        lines.append("## Best event-validation rows by PR-AUC")
        for _, row in results.sort_values("pr_auc", ascending=False).head(12).iterrows():
            lines.append(f"- {row['event_name']} / {row['model']}: PR-AUC={row['pr_auc']:.3f}, MCC={row['mcc']:.3f}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, FP/TP={row['false_positives_per_true_positive']:.2f}")
    if not topk.empty:
        lines += ["", "## Best event top-risk lift"]
        for _, row in topk.sort_values("lift_vs_base_rate", ascending=False).head(12).iterrows():
            lines.append(f"- {row['event_name']} / {row['model']} / {row['top_label']}: event rate {row['top_event_rate']:.1%} vs base {row['base_event_rate']:.1%}, lift {row['lift_vs_base_rate']:.2f}x")
    return "\n".join(lines)


def main() -> None:
    debug_print("Event-level validation start")
    df, feature_cols = load_labeled_dataset()
    all_results: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    all_topk: list[dict[str, Any]] = []
    for event in CRISIS_WINDOWS:
        results, preds, topk = run_event(df, feature_cols, event)
        all_results.extend(results)
        all_preds.extend(preds)
        all_topk.extend(topk)

    results_df = pd.DataFrame(all_results)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    topk_df = pd.DataFrame(all_topk)
    results_df.to_parquet(RESULTS_PATH, index=False)
    preds_df.to_parquet(PREDICTIONS_PATH, index=False)
    topk_df.to_parquet(TOPK_PATH, index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results_df, topk_df), encoding="utf-8")
    debug_print("Event-level validation complete")


if __name__ == "__main__":
    main()
