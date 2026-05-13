
"""
Discrete-time hazard model for major-drawdown onset, with leakage audit.

Run after:
    python ml_dataset.py

The normal six-month classifier asks whether a major drawdown happened at any
point in a long forward window. The hazard model asks a shorter timing question:

    Did the asset ENTER a segment-adjusted major-drawdown state within the next
    four weekly observations?

v14 trains two versions:
    1. full_features: includes current drawdown/trend-state variables.
    2. restricted_no_state_features: removes the most target-adjacent state
       variables, making the result harder to fool.

If full_features looks dramatically better than restricted_no_state_features,
that is a leakage/same-state warning. This is still educational research, not a
trading system.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

from config import AI_LEADERS, BASE_DIR, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "hazard_model.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hazard_model")

DATASET_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"
RESULTS_PATH = PROCESSED_DIR / "hazard_model_results.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "hazard_model_predictions.parquet"
AUDIT_PATH = PROCESSED_DIR / "hazard_leakage_audit.parquet"
CURRENT_AI_PATH = PROCESSED_DIR / "current_ai_hazard_scores.parquet"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET = "hazard_event_4w_segment"
VALID = "target_valid_hazard_event_4w_segment"

FULL_FEATURES = [
    "price_log", "sma_50_over_200", "distance_from_200dma", "weekly_rsi", "drawdown_pct",
    "volatility_30d", "return_21d", "return_63d", "return_126d", "zscore_price",
    "warning_score", "recovery_score", "price_to_sales", "pe_ratio", "ev_to_sales",
    "revenue_ttm_yoy_pct", "valuation_warning_score", "federal_funds_rate", "cpi_yoy",
    "unemployment_rate", "yield_curve_spread", "financial_stress_index", "mortgage_rate",
]

# These variables may encode that the asset is already inside or very near a drawdown state.
# Removing them gives a more conservative onset model.
TARGET_ADJACENT_FEATURES = {
    "drawdown_pct", "distance_from_200dma", "zscore_price", "recovery_score", "warning_score",
}
RESTRICTED_FEATURES = [c for c in FULL_FEATURES if c not in TARGET_ADJACENT_FEATURES]


def debug_print(message: str) -> None:
    print(f"[HAZARD_MODEL DEBUG] {message}")
    logger.info(message)


def clean_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    arr = out.to_numpy(dtype=float)
    bad = ~np.isfinite(arr) | (np.abs(arr) > 1e100)
    if bad.any():
        debug_print(f"Replacing {int(bad.sum()):,} non-finite/extreme hazard feature values with NaN.")
        arr[bad] = np.nan
    return pd.DataFrame(arr, columns=cols, index=df.index)


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python ml_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} missing. Re-run the upgraded ml_dataset.py.")
    labeled = df[(df[VALID] == True) & df[TARGET].notna()].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)
    debug_print(f"Loaded hazard labeled rows={len(labeled):,}; positives={int(labeled[TARGET].sum()):,}")
    return labeled.sort_values(["date", "ticker"])


def available_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    candidates = FULL_FEATURES if feature_set == "full_features" else RESTRICTED_FEATURES
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        raise ValueError(f"No hazard model features available for {feature_set}.")
    return cols


def split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    train_cut = dates[int(len(dates) * 0.70)]
    val_cut = dates[int(len(dates) * 0.85)]
    train = df[df["date"] <= train_cut].copy()
    val = df[(df["date"] > train_cut) & (df["date"] <= val_cut)].copy()
    test = df[df["date"] > val_cut].copy()
    return train, val, test


def make_models(cols: list[str], y_train: pd.Series) -> dict[str, Pipeline]:
    pre_scaled = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols)], remainder="drop")
    pre_tree = ColumnTransformer([("num", SimpleImputer(strategy="median"), cols)], remainder="drop")
    models: dict[str, Pipeline] = {
        "Hazard Logistic Regression": Pipeline([("pre", pre_scaled), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Hazard Random Forest": Pipeline([("pre", pre_tree), ("model", RandomForestClassifier(n_estimators=350, max_depth=7, min_samples_leaf=10, class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
    }
    if XGBClassifier is not None:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        models["Hazard XGBoost"] = Pipeline([("pre", pre_tree), ("model", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=(neg/pos if pos else 1), eval_metric="logloss", random_state=42, n_jobs=-1))])
    return models


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    threshold = np.nanpercentile(p, 90)
    pred = (p >= threshold).astype(int)
    return {
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "top10_score_threshold": float(threshold),
        "precision_top10_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_top10_threshold": float(recall_score(y, pred, zero_division=0)),
        "f1_top10_threshold": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }


def model_filename(name: str, feature_set: str) -> Path:
    safe = str(name).lower().replace(" ", "_").replace("/", "_")
    return MODELS_DIR / f"{safe}_{feature_set}.pkl"


def build_leakage_audit(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"].eq("test")].copy()
    rows = []
    for model in sorted(test["model"].dropna().unique()):
        full = test[(test["model"].eq(model)) & (test["feature_set"].eq("full_features"))]
        restricted = test[(test["model"].eq(model)) & (test["feature_set"].eq("restricted_no_state_features"))]
        if full.empty or restricted.empty:
            continue
        f = full.iloc[0]
        r = restricted.iloc[0]
        roc_gap = float(f.get("roc_auc", np.nan) - r.get("roc_auc", np.nan))
        pr_gap = float(f.get("pr_auc", np.nan) - r.get("pr_auc", np.nan))
        suspicious = (f.get("roc_auc", 0) >= 0.90 and roc_gap >= 0.10) or pr_gap >= 0.20
        rows.append({
            "model": model,
            "full_roc_auc": f.get("roc_auc", np.nan),
            "restricted_roc_auc": r.get("roc_auc", np.nan),
            "roc_auc_gap_full_minus_restricted": roc_gap,
            "full_pr_auc": f.get("pr_auc", np.nan),
            "restricted_pr_auc": r.get("pr_auc", np.nan),
            "pr_auc_gap_full_minus_restricted": pr_gap,
            "audit_warning": "SUSPICIOUS: full model may be using target-adjacent state variables" if suspicious else "OK: restricted model remains comparable",
        })
    return pd.DataFrame(rows)


def write_audit_report(audit: pd.DataFrame) -> None:
    path = REPORTS_DIR / "hazard_leakage_audit.md"
    lines = ["# Hazard Model Leakage Audit", "", "The hazard model is trained with both full and leakage-restricted feature sets.", "The restricted version removes current drawdown-state variables such as drawdown_pct, distance_from_200dma, zscore_price, warning_score, and recovery_score.", ""]
    if audit.empty:
        lines.append("No audit rows were produced.")
    else:
        for _, row in audit.iterrows():
            lines.append(f"- {row['model']}: full PR-AUC={row['full_pr_auc']:.3f}, restricted PR-AUC={row['restricted_pr_auc']:.3f}, gap={row['pr_auc_gap_full_minus_restricted']:.3f}. {row['audit_warning']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    debug_print("Hazard model start")
    df = load_data()
    train, val, test = split_time(df)
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    fitted_models: dict[tuple[str, str], Pipeline] = {}
    feature_sets = {
        "full_features": available_features(df, "full_features"),
        "restricted_no_state_features": available_features(df, "restricted_no_state_features"),
    }

    for feature_set, cols in feature_sets.items():
        debug_print(f"Training hazard feature_set={feature_set}; features={cols}")
        X_train = clean_X(train, cols)
        y_train = train[TARGET].astype(int)
        models = make_models(cols, y_train)
        for name, model in models.items():
            debug_print(f"Training {name} / {feature_set}")
            model.fit(X_train, y_train)
            joblib.dump({"model": model, "features": cols, "feature_set": feature_set}, model_filename(name, feature_set))
            fitted_models[(name, feature_set)] = model
            for split_name, part in [("validation", val), ("test", test)]:
                X = clean_X(part, cols)
                p = model.predict_proba(X)[:, 1]
                y = part[TARGET].astype(int).to_numpy()
                rows.append({"model": name, "feature_set": feature_set, "split": split_name, "n_rows": len(part), **metrics(y, p)})
                tmp = part[["date", "ticker", "asset_segment", TARGET]].copy()
                tmp["model"] = name
                tmp["feature_set"] = feature_set
                tmp["split"] = split_name
                tmp["hazard_score"] = p
                tmp["hazard_risk_percentile"] = tmp["hazard_score"].rank(pct=True)
                pred_frames.append(tmp)

    results = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    results.to_parquet(RESULTS_PATH, index=False)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    audit = build_leakage_audit(results)
    audit.to_parquet(AUDIT_PATH, index=False)
    write_audit_report(audit)

    # Current AI onset hazard scores use the best restricted-feature test PR-AUC model.
    test_restricted = results[(results["split"].eq("test")) & (results["feature_set"].eq("restricted_no_state_features"))].copy()
    if test_restricted.empty:
        test_restricted = results[results["split"].eq("test")].copy()
    best = test_restricted.sort_values("pr_auc", ascending=False).iloc[0]
    best_model_name = best["model"]
    best_feature_set = best["feature_set"]
    payload = joblib.load(model_filename(best_model_name, best_feature_set))
    best_model = payload["model"]
    cols = payload["features"]
    latest = df[df["ticker"].isin(AI_LEADERS)].sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).copy()
    if not latest.empty:
        latest["hazard_model"] = best_model_name
        latest["hazard_feature_set"] = best_feature_set
        latest["hazard_score_4w"] = best_model.predict_proba(clean_X(latest, cols))[:, 1]
        # Keep old column name for backward compatibility but label it as score in dashboard.
        latest["hazard_probability_4w"] = latest["hazard_score_4w"]
        latest["hazard_risk_category"] = pd.cut(latest["hazard_score_4w"].rank(pct=True), [-np.inf, .5, .75, .9, .95, np.inf], labels=["Low", "Moderate", "Elevated", "High", "Extreme"]).astype(str)
        latest[["date", "ticker", "asset_segment", "hazard_model", "hazard_feature_set", "hazard_score_4w", "hazard_probability_4w", "hazard_risk_category"]].to_parquet(CURRENT_AI_PATH, index=False)
    debug_print("Hazard model complete")


if __name__ == "__main__":
    main()
