"""
Poisson count model for market-wide drawdown clustering.

Run after:
    python ml_dataset.py

This is deliberately separate from the main binary ticker-week classifier.
Poisson regression is appropriate here because the target is a count:

    How many assets in a segment later experienced a segment-adjusted major
    drawdown?

It is not used to replace Logistic Regression, Random Forest, or XGBoost for the
row-level burst classifier.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BASE_DIR, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "poisson_count_model.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("poisson_count_model")

DATASET_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"
COUNT_DATASET_PATH = PROCESSED_DIR / "poisson_count_dataset.parquet"
RESULTS_PATH = PROCESSED_DIR / "poisson_count_model_results.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "poisson_count_predictions.parquet"
CURRENT_PATH = PROCESSED_DIR / "current_poisson_count_forecast.parquet"
MODEL_PATH = BASE_DIR / "models" / "poisson_count_model.pkl"

TARGET = "burst_6m_segment"
VALID = "target_valid_burst_6m_segment"
NUMERIC_FEATURES = [
    "n_assets", "mean_warning_score", "mean_recovery_score", "mean_drawdown_pct", "mean_distance_from_200dma",
    "mean_weekly_rsi", "mean_volatility_30d", "mean_return_63d", "mean_price_to_sales",
    "mean_valuation_warning_score", "federal_funds_rate", "cpi_yoy", "unemployment_rate",
    "yield_curve_spread", "financial_stress_index",
]
CAT_FEATURES = ["asset_segment"]


def debug_print(message: str) -> None:
    print(f"[POISSON_COUNT_MODEL DEBUG] {message}")
    logger.info(message)


def load_and_aggregate() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python ml_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} missing. Re-run upgraded ml_dataset.py.")
    valid = df[(df[VALID] == True) & df[TARGET].notna()].copy()
    if valid.empty:
        raise ValueError("No valid rows for Poisson count modeling.")
    for col in ["warning_score", "recovery_score", "drawdown_pct", "distance_from_200dma", "weekly_rsi", "volatility_30d", "return_63d", "price_to_sales", "valuation_warning_score"]:
        if col not in valid.columns:
            valid[col] = np.nan
    macro_cols = [c for c in ["federal_funds_rate", "cpi_yoy", "unemployment_rate", "yield_curve_spread", "financial_stress_index"] if c in valid.columns]
    rows = []
    for (date, segment), g in valid.groupby(["date", "asset_segment"], dropna=False):
        row = {
            "date": date,
            "asset_segment": str(segment),
            "n_assets": int(len(g)),
            "event_count": int(g[TARGET].sum()),
            "event_rate": float(g[TARGET].mean()),
            "mean_warning_score": float(pd.to_numeric(g["warning_score"], errors="coerce").mean()),
            "mean_recovery_score": float(pd.to_numeric(g["recovery_score"], errors="coerce").mean()),
            "mean_drawdown_pct": float(pd.to_numeric(g["drawdown_pct"], errors="coerce").mean()),
            "mean_distance_from_200dma": float(pd.to_numeric(g["distance_from_200dma"], errors="coerce").mean()),
            "mean_weekly_rsi": float(pd.to_numeric(g["weekly_rsi"], errors="coerce").mean()),
            "mean_volatility_30d": float(pd.to_numeric(g["volatility_30d"], errors="coerce").mean()),
            "mean_return_63d": float(pd.to_numeric(g["return_63d"], errors="coerce").mean()) if "return_63d" in g else np.nan,
            "mean_price_to_sales": float(pd.to_numeric(g["price_to_sales"], errors="coerce").mean()),
            "mean_valuation_warning_score": float(pd.to_numeric(g["valuation_warning_score"], errors="coerce").mean()),
        }
        for col in macro_cols:
            row[col] = float(pd.to_numeric(g[col], errors="coerce").dropna().tail(1).mean()) if g[col].notna().any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["date", "asset_segment"]).reset_index(drop=True)
    out.to_parquet(COUNT_DATASET_PATH, index=False)
    debug_print(f"Saved Poisson count dataset: shape={out.shape}")
    return out


def make_pipeline(num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", PoissonRegressor(alpha=0.1, max_iter=1000))])


def split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(df["date"].unique()))
    train_cut = dates[int(len(dates)*.70)]
    val_cut = dates[int(len(dates)*.85)]
    return df[df["date"] <= train_cut], df[(df["date"] > train_cut) & (df["date"] <= val_cut)], df[df["date"] > val_cut]


def eval_split(name: str, part: pd.DataFrame, pred: np.ndarray) -> dict[str, float | str | int]:
    y = part["event_count"].astype(float).to_numpy()
    pred = np.clip(pred, 1e-9, None)
    return {
        "split": name,
        "n_rows": int(len(part)),
        "mean_actual_count": float(np.mean(y)),
        "mean_predicted_count": float(np.mean(pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mean_poisson_deviance": float(mean_poisson_deviance(y, pred)),
    }


def main() -> None:
    debug_print("Poisson count model start")
    df = load_and_aggregate()
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    train, val, test = split_time(df)
    model = make_pipeline(num_cols)
    X_train = train[num_cols + CAT_FEATURES]
    y_train = train["event_count"].astype(float)
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    rows = []
    pred_frames = []
    for name, part in [("validation", val), ("test", test)]:
        pred = model.predict(part[num_cols + CAT_FEATURES])
        rows.append(eval_split(name, part, pred))
        tmp = part[["date", "asset_segment", "n_assets", "event_count", "event_rate"]].copy()
        tmp["predicted_event_count"] = pred
        tmp["predicted_event_rate"] = pred / tmp["n_assets"].replace(0, np.nan)
        tmp["split"] = name
        pred_frames.append(tmp)
    pd.DataFrame(rows).to_parquet(RESULTS_PATH, index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(PREDICTIONS_PATH, index=False)
    latest = df.sort_values("date").groupby("asset_segment", as_index=False).tail(1).copy()
    latest["predicted_event_count_next_6m"] = model.predict(latest[num_cols + CAT_FEATURES])
    latest["predicted_event_rate_next_6m"] = latest["predicted_event_count_next_6m"] / latest["n_assets"].replace(0, np.nan)
    latest.to_parquet(CURRENT_PATH, index=False)
    debug_print("Poisson count model complete")


if __name__ == "__main__":
    main()
