"""
Export compact datasets for optional Firth logistic regression in R.

Run after:
    python ml_dataset.py

Why this exists:
    Firth logistic regression is useful when a binary model has very few
    successes or near-separation. Python support is uneven, especially on modern
    Python versions, while R has mature packages such as logistf and brglm2.

This script does not pretend to run Firth in Python. It creates clean CSV files
for global and segment-specific Firth benchmarks, plus an R script in this
project that can fit the models and export results back to CSV.

Outputs:
    data/processed/firth_benchmark_global.csv
    data/processed/firth_benchmark_<segment>.csv
    firth_logistic_optional.R
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_DIR, LOG_DIR, PROCESSED_DIR
from model_training import TARGET, clean_feature_matrix, clean_model_dataset, infer_asset_segment, select_feature_columns

LOG_FILE = LOG_DIR / "firth_logistic_export.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("firth_logistic_export")

DATASET_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"

# Firth logistic can become slow with hundreds of thousands of rows and many
# features. We export a compact, interpretable subset designed for a statistical
# benchmark rather than a production model.
PREFERRED_FEATURES = [
    "distance_from_200dma",
    "weekly_rsi",
    "drawdown_pct",
    "volatility_30d",
    "return_63d",
    "return_126d",
    "zscore_price",
    "warning_score",
    "recovery_score",
    "price_to_sales",
    "ev_to_sales",
    "ps_to_historical_median",
    "revenue_ttm_yoy_pct",
    "federal_funds_rate",
    "yield_curve_spread",
    "cpi_yoy",
    "unemployment_rate",
    "financial_stress_index",
]


def debug_print(message: str) -> None:
    print(f"[FIRTH_EXPORT DEBUG] {message}")
    logger.info(message)


def load_labeled() -> tuple[pd.DataFrame, list[str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run python ml_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["asset_segment"] = df["ticker"].astype(str).map(infer_asset_segment)
    valid_col = f"target_valid_{TARGET}"
    labeled = df[(df[valid_col] == True) & df[TARGET].notna()].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)
    all_features = select_feature_columns(labeled)
    features = [f for f in PREFERRED_FEATURES if f in all_features]
    if not features:
        features = all_features[:12]
    labeled = clean_model_dataset(labeled, features, context="firth export")
    return labeled, features


def export_scope(df: pd.DataFrame, features: list[str], scope: str) -> None:
    if scope == "global":
        out = df.copy()
    else:
        out = df[df["asset_segment"].eq(scope)].copy()
    if out.empty or out[TARGET].nunique() < 2:
        debug_print(f"Skipping Firth export for {scope}: insufficient rows/classes.")
        return
    cols = ["date", "ticker", "asset_segment", TARGET] + features
    out = out[cols].copy()
    out[features] = clean_feature_matrix(out[features], features, context=f"firth {scope}")
    # Median-fill for R simplicity; this is only an optional benchmark export.
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())
    path = PROCESSED_DIR / f"firth_benchmark_{scope}.csv"
    out.to_csv(path, index=False)
    debug_print(f"Wrote {path}: rows={len(out):,}, positives={int(out[TARGET].sum())}")


def write_r_script(features: list[str]) -> None:
    feature_text = ", ".join([f'"{f}"' for f in features])
    script = f'''# Optional Firth logistic regression benchmark
# Run in R after Python creates data/processed/firth_benchmark_*.csv
# install.packages("logistf")

library(logistf)

base_dir <- getwd()
processed_dir <- file.path(base_dir, "data", "processed")
features <- c({feature_text})
target <- "{TARGET}"
files <- list.files(processed_dir, pattern="^firth_benchmark_.*\\.csv$", full.names=TRUE)
results <- data.frame()

for (file in files) {{
  df <- read.csv(file)
  scope <- sub("firth_benchmark_", "", tools::file_path_sans_ext(basename(file)))
  formula_text <- paste(target, "~", paste(features[features %in% names(df)], collapse=" + "))
  cat("Fitting Firth logistic for", scope, "with formula", formula_text, "\n")
  fit <- logistf(as.formula(formula_text), data=df)
  coefs <- data.frame(scope=scope, feature=names(coef(fit)), coefficient=as.numeric(coef(fit)))
  results <- rbind(results, coefs)
}}

write.csv(results, file.path(processed_dir, "firth_logistic_coefficients.csv"), row.names=FALSE)
cat("Saved Firth coefficients to data/processed/firth_logistic_coefficients.csv\n")
'''
    (BASE_DIR / "firth_logistic_optional.R").write_text(script, encoding="utf-8")
    debug_print("Wrote firth_logistic_optional.R")


def main() -> None:
    debug_print("Firth logistic export start")
    df, features = load_labeled()
    for scope in ["global", "broad_index_etf", "mega_cap_ai_tech", "speculative_high_vol"]:
        export_scope(df, features, scope)
    write_r_script(features)
    debug_print("Firth logistic export complete")


if __name__ == "__main__":
    main()
