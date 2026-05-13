"""
Score latest AI-exposed ticker rows with trained ML crash-risk models.

Run after:
    python model_training.py

Output:
    data/processed/current_ai_ml_risk_scores.parquet

This upgraded version uses segment-specific models when possible. For example,
PLTR can be scored with the speculative/high-volatility model if that model was
trained, while NVDA can be scored with the mega-cap AI/tech model. If a segment
model is unavailable, the script falls back to the global model.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import AI_LEADERS, BASE_DIR, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "ml_inference.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ml_inference")

MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"
OUTPUT_PATH = PROCESSED_DIR / "current_ai_ml_risk_scores.parquet"
PREPROCESSOR_PATH = MODELS_DIR / "ml_preprocessor.pkl"
CALIBRATION_PATH = PROCESSED_DIR / "ml_calibration_table.parquet"
THRESHOLDS_PATH = PROCESSED_DIR / "ml_thresholds.parquet"
TOP_DECILE_PATH = PROCESSED_DIR / "ml_top_decile_analysis.parquet"
RARE_CALIBRATION_PATH = PROCESSED_DIR / "ml_rare_event_calibration_curves.parquet"
RARE_THRESHOLDS_PATH = PROCESSED_DIR / "ml_rare_event_thresholds.parquet"
TOP_K_LIFT_PATH = PROCESSED_DIR / "ml_top_k_lift_analysis.parquet"

PREFERRED_MODEL_ORDER = ["XGBoost", "Random Forest", "Elastic Net Logistic", "Logistic Regression"]


def debug_print(message: str) -> None:
    print(f"[ML_INFERENCE DEBUG] {message}")
    logger.info(message)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_bundle() -> dict[str, Any]:
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing {PREPROCESSOR_PATH}. Run python model_training.py first.")
    bundle = joblib.load(PREPROCESSOR_PATH)
    debug_print(f"Loaded ML metadata bundle with target={bundle.get('target')}")
    return bundle


def infer_asset_segment_from_bundle(ticker: str, bundle: dict[str, Any]) -> str:
    segment_tickers = bundle.get("segment_tickers", {})
    for segment, tickers in segment_tickers.items():
        if ticker in set(tickers):
            return segment
    if ticker.endswith("-USD"):
        return "speculative_high_vol"
    if ticker.startswith("^"):
        return "broad_index_etf"
    return "other_single_name"


def load_models(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for scope in bundle.get("model_scopes", ["global"]):
        scope_slug = slugify(scope)
        models[scope] = {}
        for model_name in bundle.get("available_models_by_scope", {}).get(scope, PREFERRED_MODEL_ORDER):
            model_slug = slugify(model_name)
            path = MODELS_DIR / f"{scope_slug}_{model_slug}_burst_6m.pkl"
            if path.exists():
                models[scope][model_name] = joblib.load(path)
                debug_print(f"Loaded {model_name} model for scope={scope}: {path.name}")
    # Legacy fallback in case the user has older model files.
    if not models.get("global"):
        models["global"] = {}
        legacy_files = {
            "Elastic Net Logistic": MODELS_DIR / "elastic_net_logistic_burst_6m.pkl",
            "Logistic Regression": MODELS_DIR / "logistic_regression_burst_6m.pkl",
            "Random Forest": MODELS_DIR / "random_forest_burst_6m.pkl",
            "XGBoost": MODELS_DIR / "xgboost_burst_6m.pkl",
        }
        for name, path in legacy_files.items():
            if path.exists():
                models["global"][name] = joblib.load(path)
                debug_print(f"Loaded legacy global model: {name}")
    if not any(scope_models for scope_models in models.values()):
        raise FileNotFoundError("No trained model files found. Run python model_training.py first.")
    return models


def load_latest_ai_rows(feature_cols: list[str], bundle: dict[str, Any]) -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python ml_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if "asset_segment" not in df.columns:
        df["asset_segment"] = df["ticker"].astype(str).apply(lambda t: infer_asset_segment_from_bundle(t, bundle))

    ai = df[df["ticker"].isin(AI_LEADERS)].sort_values(["ticker", "date"]).copy()
    if ai.empty:
        raise ValueError(f"No AI leader rows found in {DATASET_PATH}. Available tickers: {sorted(df['ticker'].dropna().unique())[:30]}")
    latest = ai.groupby("ticker", as_index=False).tail(1).copy()
    latest["asset_segment"] = latest["ticker"].astype(str).apply(lambda t: infer_asset_segment_from_bundle(t, bundle))

    missing_features = [c for c in feature_cols if c not in latest.columns]
    if missing_features:
        raise ValueError(f"Latest AI rows are missing trained feature columns: {missing_features}")
    debug_print(f"Latest AI inference rows shape={latest.shape}")
    return latest


def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def load_optional_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def lookup_threshold(thresholds: pd.DataFrame, model: str, scope: str) -> float:
    if thresholds.empty:
        return 0.50
    match = thresholds[(thresholds["model"].eq(model)) & (thresholds["model_scope"].eq(scope))]
    if match.empty and scope != "global":
        match = thresholds[(thresholds["model"].eq(model)) & (thresholds["model_scope"].eq("global"))]
    if match.empty:
        return 0.50
    return float(match.iloc[0]["threshold_value"])


def empirical_calibrated_probability(calibration: pd.DataFrame, model: str, scope: str, probability: float) -> float:
    """Map raw model probability to empirical event rate from calibration bins.

    If the exact segment calibration is unavailable, fall back to global. If even
    that is unavailable, return the raw probability. This keeps inference robust
    rather than pretending calibration exists when it does not.
    """
    if calibration.empty or pd.isna(probability):
        return float(probability)
    usable = calibration[calibration["split"].isin(["historical_in_sample_final", "validation", "test", "validation_map"])].copy()
    match = usable[(usable["model"].eq(model)) & (usable["model_scope"].eq(scope))]
    if match.empty and scope != "global":
        match = usable[(usable["model"].eq(model)) & (usable["model_scope"].eq("global"))]
    if match.empty:
        return float(probability)
    containing = match[(match["bin_left"] <= probability) & (probability <= match["bin_right"])]
    if containing.empty:
        # Use closest bin center for probabilities outside observed range.
        centers = (match["bin_left"] + match["bin_right"]) / 2.0
        idx = (centers - probability).abs().idxmin()
        return float(match.loc[idx, "empirical_event_rate"])
    return float(containing.iloc[-1]["empirical_event_rate"])


def categorize_risk_from_history(probability: float, historical_probs: pd.Series | None = None) -> str:
    if historical_probs is None or historical_probs.dropna().empty:
        if probability < 0.10:
            return "Low Historical Drawdown Risk"
        if probability < 0.20:
            return "Moderate Historical Drawdown Risk"
        if probability < 0.35:
            return "Elevated Historical Drawdown Risk"
        if probability < 0.50:
            return "High Historical Drawdown Risk"
        return "Extreme Historical Drawdown Risk"
    p50, p75, p90, p95 = np.nanpercentile(historical_probs, [50, 75, 90, 95])
    if probability <= p50:
        return "Low Historical Drawdown Risk"
    if probability <= p75:
        return "Moderate Historical Drawdown Risk"
    if probability <= p90:
        return "Elevated Historical Drawdown Risk"
    if probability <= p95:
        return "High Historical Drawdown Risk"
    return "Extreme Historical Drawdown Risk"


def choose_primary_model_for_row(scope: str, available_scope_models: dict[str, Any], global_models: dict[str, Any]) -> tuple[str, str, Any]:
    """Pick segment model when possible, otherwise global fallback."""
    for model_name in PREFERRED_MODEL_ORDER:
        if model_name in available_scope_models:
            return model_name, scope, available_scope_models[model_name]
    for model_name in PREFERRED_MODEL_ORDER:
        if model_name in global_models:
            return model_name, "global", global_models[model_name]
    # Last-resort deterministic fallback.
    if available_scope_models:
        model_name, model = next(iter(available_scope_models.items()))
        return model_name, scope, model
    model_name, model = next(iter(global_models.items()))
    return model_name, "global", model


def main() -> None:
    debug_print("Current AI ML inference start")
    bundle = load_bundle()
    models_by_scope = load_models(bundle)
    feature_cols = bundle["feature_columns"]
    latest = load_latest_ai_rows(feature_cols, bundle)
    rare_thresholds = load_optional_table(RARE_THRESHOLDS_PATH)
    thresholds = rare_thresholds if not rare_thresholds.empty else load_optional_table(THRESHOLDS_PATH)
    rare_calibration = load_optional_table(RARE_CALIBRATION_PATH)
    calibration = rare_calibration if not rare_calibration.empty else load_optional_table(CALIBRATION_PATH)
    top_k_lift = load_optional_table(TOP_K_LIFT_PATH)
    top_decile = top_k_lift if not top_k_lift.empty else load_optional_table(TOP_DECILE_PATH)

    base_cols = ["date", "ticker", "asset_segment", "adjusted_close", "warning_score", "recovery_score", "drawdown_pct", "distance_from_200dma", "weekly_rsi"]
    base_cols = [c for c in base_cols if c in latest.columns]
    output = latest[base_cols].copy()

    # Score every available model/scope pair where possible. The dashboard can
    # display these columns, while the primary score uses the most relevant segment
    # model for each ticker.
    for scope, scope_models in models_by_scope.items():
        for model_name, model in scope_models.items():
            col = f"{slugify(scope)}_{slugify(model_name)}_probability"
            output[col] = predict_probability(model, latest[feature_cols])
            debug_print(f"Scored latest AI rows with {model_name} / {scope}")

    primary_rows: list[dict[str, Any]] = []
    for idx, row in latest.iterrows():
        ticker = row["ticker"]
        segment = row.get("asset_segment") or infer_asset_segment_from_bundle(str(ticker), bundle)
        available_scope_models = models_by_scope.get(segment, {})
        global_models = models_by_scope.get("global", {})
        model_name, model_scope, model = choose_primary_model_for_row(segment, available_scope_models, global_models)
        raw_prob = float(predict_probability(model, latest.loc[[idx], feature_cols])[0])
        calibrated_prob = empirical_calibrated_probability(calibration, model_name, model_scope, raw_prob)
        threshold = lookup_threshold(thresholds, model_name, model_scope)

        historical_col = f"{slugify(model_scope)}_{slugify(model_name)}_probability"
        historical_probs = output[historical_col] if historical_col in output.columns else None
        risk_category = categorize_risk_from_history(raw_prob, historical_probs)

        top_decile_note = "Top-decile history unavailable."
        if not top_decile.empty:
            match = top_decile[(top_decile["model"].eq(model_name)) & (top_decile["model_scope"].eq(model_scope)) & (top_decile["split"].eq("test"))]
            if match.empty and model_scope != "global":
                match = top_decile[(top_decile["model"].eq(model_name)) & (top_decile["model_scope"].eq("global")) & (top_decile["split"].eq("test"))]
            if not match.empty:
                m = match.iloc[0]
                if "top_event_rate" in m.index:
                    top_label = m.get("top_label", "top-risk bucket")
                    top_decile_note = f"In test data, this model's {top_label} had event rate {m['top_event_rate']:.1%} vs base {m['base_event_rate']:.1%}."
                else:
                    top_decile_note = f"In test data, this model's top-risk decile had event rate {m['top_decile_event_rate']:.1%} vs base {m['base_event_rate']:.1%}."

        primary_rows.append({
            "ticker": ticker,
            "primary_model": model_name,
            "primary_model_scope": model_scope,
            "primary_ml_risk_probability": raw_prob,
            "calibrated_empirical_risk": calibrated_prob,
            "tuned_alert_threshold": threshold,
            "above_tuned_threshold": bool(raw_prob >= threshold),
            "risk_category": risk_category,
            "top_decile_context": top_decile_note,
            "interpretation": (
                "This is a historically backtested forward drawdown-risk score. It compares today's features with past weeks. "
                "It is not a guaranteed forecast or financial advice."
            ),
        })

    primary_df = pd.DataFrame(primary_rows)
    output = output.merge(primary_df, on="ticker", how="left")
    output = output.sort_values("primary_ml_risk_probability", ascending=False).reset_index(drop=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    debug_print(f"Saved current AI ML risk scores to {OUTPUT_PATH}")
    debug_print("Current AI ML inference complete")


if __name__ == "__main__":
    main()
