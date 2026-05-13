"""
Controlled imbalance-method experiments for the bubble crash-risk project.

Run after:
    python ml_dataset.py

Why this exists:
    The main v12 pipeline already uses class weights, scale_pos_weight,
    threshold tuning, risk buckets, calibration, hazard modeling, and a Poisson
    count model. This script adds a *controlled experiment layer* for imbalance
    methods that can help rare-event classification but can also fool us:

    1. Balanced Random Forest
       Uses balanced bootstrap samples inside each tree. This is usually safer
       than synthetic oversampling because it does not invent market rows.

    2. SMOTE + Logistic Regression / Random Forest
       Creates synthetic minority-class rows only inside the training split.
       Validation and test rows remain untouched future data.

    3. SMOTE-ENN + Logistic Regression / Random Forest
       Creates synthetic minority rows and then removes ambiguous/noisy rows.
       This is powerful, but potentially dangerous in finance because ambiguous
       pre-crash periods may be economically important.

Important leakage rule:
    Resampling is applied ONLY after chronological train/validation/test splits.
    We never SMOTE the full dataset before splitting, because that would let
    synthetic versions of future regimes leak backward into training.

Outputs:
    data/processed/imbalance_experiment_results.parquet
    data/processed/imbalance_experiment_predictions.parquet
    data/processed/imbalance_experiment_thresholds.parquet
    data/processed/imbalance_experiment_topk.parquet
    reports/imbalance_experiment_summary.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError as exc:  # pragma: no cover
    BalancedRandomForestClassifier = None
    SMOTE = None
    SMOTEENN = None
    ImbPipeline = None

from config import BASE_DIR, LOG_DIR, PROCESSED_DIR
from model_training import (
    MIN_SEGMENT_POSITIVES,
    MIN_SEGMENT_ROWS,
    TARGET,
    build_chronological_splits,
    choose_threshold_on_validation,
    clean_feature_matrix,
    clean_model_dataset,
    infer_asset_segment,
    select_feature_columns,
)

LOG_FILE = LOG_DIR / "imbalance_experiments.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("imbalance_experiments")

DATASET_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"
RESULTS_PATH = PROCESSED_DIR / "imbalance_experiment_results.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "imbalance_experiment_predictions.parquet"
THRESHOLDS_PATH = PROCESSED_DIR / "imbalance_experiment_thresholds.parquet"
TOPK_PATH = PROCESSED_DIR / "imbalance_experiment_topk.parquet"
REPORT_PATH = BASE_DIR / "reports" / "imbalance_experiment_summary.md"


def debug_print(message: str) -> None:
    print(f"[IMBALANCE_EXPERIMENTS DEBUG] {message}")
    logger.info(message)


def require_imblearn() -> None:
    if any(x is None for x in [BalancedRandomForestClassifier, SMOTE, SMOTEENN, ImbPipeline]):
        raise ImportError(
            "imbalanced-learn is required for imbalance_experiments.py. "
            "Install with: pip install imbalanced-learn"
        )


def load_labeled_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python ml_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["asset_segment"] = df["ticker"].astype(str).map(infer_asset_segment)
    valid_col = f"target_valid_{TARGET}"
    if TARGET not in df.columns or valid_col not in df.columns:
        raise ValueError(f"Dataset must contain {TARGET} and {valid_col}. Re-run ml_dataset.py.")
    labeled = df[(df[valid_col] == True) & df[TARGET].notna()].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)
    features = select_feature_columns(labeled)
    labeled = clean_model_dataset(labeled, features, context="imbalance_experiments labeled dataset")
    debug_print(f"Loaded labeled dataset for imbalance experiments: rows={len(labeled):,}, features={len(features)}")
    return labeled.sort_values(["date", "ticker"]).reset_index(drop=True), features


def numeric_preprocessor(feature_cols: list[str], scale: bool) -> ColumnTransformer:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[("numeric", SklearnPipeline(steps=steps), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_experiment_models(feature_cols: list[str]) -> dict[str, Any]:
    require_imblearn()
    return {
        "Balanced Random Forest": ImbPipeline(steps=[
            ("preprocessor", numeric_preprocessor(feature_cols, scale=False)),
            ("model", BalancedRandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1,
                replacement=True,
                sampling_strategy="all",
            )),
        ]),
        "SMOTE + Logistic Regression": ImbPipeline(steps=[
            ("preprocessor", numeric_preprocessor(feature_cols, scale=True)),
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ]),
        "SMOTE + Random Forest": ImbPipeline(steps=[
            ("preprocessor", numeric_preprocessor(feature_cols, scale=False)),
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "SMOTE-ENN + Logistic Regression": ImbPipeline(steps=[
            ("preprocessor", numeric_preprocessor(feature_cols, scale=True)),
            ("smote_enn", SMOTEENN(random_state=42)),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ]),
        "SMOTE-ENN + Random Forest": ImbPipeline(steps=[
            ("preprocessor", numeric_preprocessor(feature_cols, scale=False)),
            ("smote_enn", SMOTEENN(random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def metric_row(model: str, scope: str, split: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    base = float(np.mean(y_true)) if len(y_true) else np.nan
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    return {
        "model": model,
        "model_scope": scope,
        "split": split,
        "threshold_value": float(threshold),
        "n_rows": int(len(y_true)),
        "positive_rate": base,
        "alert_rate": float(y_pred.mean()) if len(y_pred) else np.nan,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) >= 2 and len(np.unique(y_pred)) >= 2 else 0.0,
        "brier_score": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positives_per_true_positive": float(fp / tp) if tp > 0 else np.inf,
        "precision_lift_vs_base_rate": float(precision / base) if base else np.nan,
    }


def prediction_frame(split_df: pd.DataFrame, model: str, scope: str, split: str, y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    cols = [c for c in ["date", "ticker", "asset_segment", TARGET] if c in split_df.columns]
    out = split_df[cols].copy()
    out["model"] = model
    out["model_scope"] = scope
    out["split"] = split
    out["probability"] = y_prob
    out["threshold_value"] = threshold
    out["prediction"] = (y_prob >= threshold).astype(int)
    return out


def topk_rows(pred: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = float(pred[target].mean()) if len(pred) else np.nan
    for frac, label in [(0.10, "Top 10%"), (0.05, "Top 5%")]:
        top_n = max(1, int(np.ceil(len(pred) * frac)))
        top = pred.sort_values("probability", ascending=False).head(top_n)
        rate = float(top[target].mean()) if len(top) else np.nan
        rows.append({
            "model": pred["model"].iloc[0],
            "model_scope": pred["model_scope"].iloc[0],
            "split": pred["split"].iloc[0],
            "top_label": label,
            "top_fraction": frac,
            "n_rows": int(len(pred)),
            "top_rows": int(len(top)),
            "base_event_rate": base,
            "top_event_rate": rate,
            "lift_vs_base_rate": float(rate / base) if base and np.isfinite(rate) else np.nan,
            "true_positives_in_top": int(top[target].sum()) if len(top) else 0,
        })
    return rows


def usable_scopes(labeled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scopes = {"global": labeled}
    for segment in ["broad_index_etf", "mega_cap_ai_tech", "speculative_high_vol"]:
        g = labeled[labeled["asset_segment"].eq(segment)].copy()
        if len(g) >= MIN_SEGMENT_ROWS and g[TARGET].sum() >= MIN_SEGMENT_POSITIVES and g[TARGET].nunique() == 2:
            scopes[segment] = g
        else:
            debug_print(f"Skipping imbalance scope={segment}: rows={len(g):,}, positives={int(g[TARGET].sum()) if not g.empty else 0}")
    return scopes


def run_scope(scope: str, df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    train, val, test = build_chronological_splits(df)
    X_train = clean_feature_matrix(train[feature_cols], feature_cols, context=f"imbalance {scope} train")
    y_train = train[TARGET].astype(int)
    X_val = clean_feature_matrix(val[feature_cols], feature_cols, context=f"imbalance {scope} validation")
    y_val = val[TARGET].astype(int).to_numpy()
    X_test = clean_feature_matrix(test[feature_cols], feature_cols, context=f"imbalance {scope} test")
    y_test = test[TARGET].astype(int).to_numpy()

    results: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []
    thresholds: list[dict[str, Any]] = []
    topk: list[dict[str, Any]] = []

    for name, model in build_experiment_models(feature_cols).items():
        try:
            debug_print(f"Training imbalance model={name}; scope={scope}")
            model.fit(X_train, y_train)
            val_prob = predict_probability(model, X_val)
            threshold_info = choose_threshold_on_validation(y_val, val_prob)
            threshold = float(threshold_info["threshold_value"])
            thresholds.append({"model": name, "model_scope": scope, **threshold_info})
            for split_name, split_df, X_split, y_true in [("validation", val, X_val, y_val), ("test", test, X_test, y_test)]:
                prob = predict_probability(model, X_split)
                results.append(metric_row(name, scope, split_name, y_true, prob, threshold))
                pred = prediction_frame(split_df, name, scope, split_name, prob, threshold)
                preds.append(pred)
                topk.extend(topk_rows(pred, TARGET))
        except Exception as exc:  # noqa: BLE001
            debug_print(f"Skipping imbalance model={name}; scope={scope}; error={exc}")
    return results, preds, thresholds, topk


def build_report(results: pd.DataFrame, topk: pd.DataFrame) -> str:
    lines = [
        "# Imbalance Method Experiment Summary",
        "",
        "These experiments test Balanced Random Forest, SMOTE, and SMOTE-ENN as controlled rare-event methods.",
        "Resampling is applied only to the training split. Validation and test splits remain untouched future data.",
        "",
        "## Interpretation rule",
        "Keep a method only if it improves PR-AUC, MCC, top-5%/top-10% lift, and false-positive burden. Reject methods that merely inflate recall by flooding the dashboard with false alarms.",
        "",
    ]
    if not results.empty:
        test = results[results["split"].eq("test")].copy()
        if not test.empty:
            lines.append("## Best test PR-AUC by experiment")
            for _, row in test.sort_values("pr_auc", ascending=False).head(10).iterrows():
                lines.append(f"- {row['model']} / {row['model_scope']}: PR-AUC={row['pr_auc']:.3f}, MCC={row['mcc']:.3f}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, FP/TP={row['false_positives_per_true_positive']:.2f}")
    if not topk.empty:
        tt = topk[topk["split"].eq("test")].copy()
        if not tt.empty:
            lines += ["", "## Best top-risk lift"]
            for _, row in tt.sort_values("lift_vs_base_rate", ascending=False).head(10).iterrows():
                lines.append(f"- {row['model']} / {row['model_scope']} / {row['top_label']}: event rate {row['top_event_rate']:.1%} vs base {row['base_event_rate']:.1%}, lift {row['lift_vs_base_rate']:.2f}x")
    return "\n".join(lines)


def main() -> None:
    debug_print("Imbalance experiments start")
    require_imblearn()
    labeled, feature_cols = load_labeled_dataset()
    all_results: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    all_thresholds: list[dict[str, Any]] = []
    all_topk: list[dict[str, Any]] = []

    for scope, scope_df in usable_scopes(labeled).items():
        results, preds, thresholds, topk = run_scope(scope, scope_df, feature_cols)
        all_results.extend(results)
        all_preds.extend(preds)
        all_thresholds.extend(thresholds)
        all_topk.extend(topk)

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    thresholds_df = pd.DataFrame(all_thresholds)
    topk_df = pd.DataFrame(all_topk)

    results_df.to_parquet(RESULTS_PATH, index=False)
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    thresholds_df.to_parquet(THRESHOLDS_PATH, index=False)
    topk_df.to_parquet(TOPK_PATH, index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results_df, topk_df), encoding="utf-8")

    debug_print(f"Saved imbalance experiment results to {RESULTS_PATH}")
    debug_print("Imbalance experiments complete")


if __name__ == "__main__":
    main()
