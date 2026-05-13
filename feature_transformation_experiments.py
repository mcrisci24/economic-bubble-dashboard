"""
Feature-transformation experiment — Phase 8 (optional, does not replace the baseline).

This script answers ONE controlled research question:

    "Do economically meaningful predictor transformations improve out-of-time
     rare-event model performance compared with the baseline feature set?"

It is intentionally a SEPARATE experiment. It does not overwrite any baseline
artifact. The dashboard renders both side by side so a reader can judge
whether transformations actually helped.

Run AFTER `python model_training.py`:

    python feature_transformation_experiments.py

Inputs
------
    data/processed/ml_burst_dataset.parquet  (produced by ml_dataset.py)

Outputs (all NEW files — none of these overwrite a baseline file)
----------------------------------------------------------------
    data/processed/transformed_feature_model_results.parquet
    data/processed/transformed_feature_predictions.parquet
    data/processed/transformed_feature_importance.parquet
    data/processed/transformed_feature_top_k_lift.parquet
    data/processed/transformed_feature_calibration.parquet
    data/processed/transformed_feature_list.parquet      # which features were used + which family
    reports/transformed_feature_experiment_summary.md
    models/transformed/*.pkl                              # trained transformed models

Leakage discipline (strictly enforced)
--------------------------------------
- Winsorization quantiles are fit on the TRAIN split only and reused on
  validation/test.
- Regime-flag thresholds (cpi_yoy, fed funds change, volatility regimes,
  financial stress) are fit on the TRAIN split only.
- Rolling z-scores by ticker and per-ticker percentile ranks use ONLY past
  observations of that ticker — computed with expanding windows.
- The supervised target `burst_6m_segment` is never used as a feature, never
  transformed, and `future_*` / `target_valid_*` / legacy `burst_*` columns
  are explicitly blacklisted from the feature pool.
- Splits are CHRONOLOGICAL (70/85 by unique date), identical to the baseline.

Framing for the dashboard
-------------------------
"Feature transformations may improve feature geometry and make patterns
 easier for models to learn, but they do not create new independent
 historical bubbles. They improve signal extraction; they do not remove
 regime uncertainty."
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:  # pragma: no cover
    BalancedRandomForestClassifier = None

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:  # pragma: no cover
    LGBMClassifier = None

from config import BASE_DIR, LOG_DIR, PROCESSED_DIR

# Reuse a few helpers from baseline training to guarantee apples-to-apples
# comparability. We import functions, not classes, to keep the dependency
# surface tiny.
from model_training import (
    SEGMENT_TICKERS,
    CANDIDATE_FEATURES as BASELINE_FEATURE_CANDIDATES,
    MIN_SEGMENT_POSITIVES,
    MIN_SEGMENT_ROWS,
    add_prediction_rows as _baseline_add_prediction_rows,  # unused — kept for type parity
    build_chronological_splits,
    build_rule_based_probabilities,
    calibration_table_from_predictions as _baseline_calibration,  # not reused; we replicate to control the TARGET name
    choose_threshold_on_validation,
    clean_feature_matrix,
    extract_feature_importance,
    infer_asset_segment,
    predict_probability,
    safe_metric,
    slugify,
)


LOG_FILE = LOG_DIR / "feature_transformation_experiments.log"

# On Windows the default stdout is cp1252 which chokes on common research
# symbols (arrows, en-dashes). Reconfigure to UTF-8 with `errors='replace'`
# so any stray glyph degrades to '?' instead of killing the whole run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_transformation_experiments")


TARGET = "burst_6m_segment"
TARGET_VALID = f"target_valid_{TARGET}"

DATASET_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"

# Output paths — every single one is namespaced so the baseline cannot be
# touched accidentally.
RESULTS_PATH      = PROCESSED_DIR / "transformed_feature_model_results.parquet"
PREDICTIONS_PATH  = PROCESSED_DIR / "transformed_feature_predictions.parquet"
IMPORTANCE_PATH   = PROCESSED_DIR / "transformed_feature_importance.parquet"
TOP_K_LIFT_PATH   = PROCESSED_DIR / "transformed_feature_top_k_lift.parquet"
CALIBRATION_PATH  = PROCESSED_DIR / "transformed_feature_calibration.parquet"
FEATURE_LIST_PATH = PROCESSED_DIR / "transformed_feature_list.parquet"
SUMMARY_REPORT    = BASE_DIR / "reports" / "transformed_feature_experiment_summary.md"

TRANSFORMED_MODELS_DIR = BASE_DIR / "models" / "transformed"
TRANSFORMED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def debug_print(message: str) -> None:
    print(f"[FEATURE_TRANSFORM DEBUG] {message}")
    logger.info(message)


# ---------------------------------------------------------------------------
# Feature transformation primitives — small, interpretable, and no-look-ahead.
# ---------------------------------------------------------------------------
NONNEG_LOG1P_CANDIDATES = [
    "market_cap", "enterprise_value", "revenue_ttm", "volume",
    "price_to_sales", "ev_to_sales",
]

SIGNED_LOG_CANDIDATES = [
    "daily_return", "return_21d", "return_63d", "return_126d", "return_252d",
    "distance_from_200dma", "drawdown_pct",
    "revenue_ttm_yoy_pct", "net_income_ttm",
    "federal_funds_change_3p", "federal_funds_change_12p",
    "ten_year_treasury_change_3p", "ten_year_treasury_change_12p",
    "yield_curve_change_3p", "yield_curve_change_12p",
    "cpi_change_12p", "unemployment_change_3p", "unemployment_change_12p",
]

WINSORIZE_CANDIDATES = [
    "pe_ratio", "price_to_sales", "ev_to_sales",
    "return_63d", "return_126d", "return_252d",
    "volatility_30d", "drawdown_pct", "distance_from_200dma",
    "federal_funds_change_3p", "federal_funds_change_12p",
    "yield_curve_change_3p", "yield_curve_change_12p",
    "ps_to_historical_median", "pe_to_historical_median",
    "ev_sales_to_historical_median", "valuation_warning_score",
]

ROLLING_ZSCORE_CANDIDATES = [
    "price_to_sales", "ev_to_sales", "volatility_30d",
    "distance_from_200dma", "return_126d", "return_252d",
    "warning_score", "recovery_score",
]

PERCENTILE_CANDIDATES = [
    "price_to_sales", "ev_to_sales", "volatility_30d",
    "distance_from_200dma", "warning_score", "drawdown_pct",
]

INTERACTION_PAIRS = [
    # (name, left, right, description)
    ("interact_valuation_x_momentum",       "valuation_warning_score", "return_252d",       "Valuation stretch × 12m momentum"),
    ("interact_valuation_x_distance_200",   "valuation_warning_score", "distance_from_200dma", "Valuation stretch × trend stretch"),
    ("interact_valuation_x_rsi",            "valuation_warning_score", "weekly_rsi",        "Valuation stretch × weekly RSI"),
    ("interact_volatility_x_drawdown",      "volatility_30d",          "drawdown_pct",      "Volatility expansion × drawdown"),
    ("interact_macro_tightening_x_dist200", "federal_funds_change_12p", "distance_from_200dma", "Rate tightening × trend stretch"),
    ("interact_warning_x_volatility",       "warning_score",           "volatility_30d",    "Warning × volatility"),
    ("interact_warning_x_dist200",          "warning_score",           "distance_from_200dma", "Warning × trend stretch"),
]

SQUARED_CANDIDATES = [
    "distance_from_200dma", "weekly_rsi", "volatility_30d",
    "drawdown_pct", "warning_score",
]

VOL_ADJ_RETURNS = [
    ("vol_adj_return_63d",  "return_63d",  "volatility_30d"),
    ("vol_adj_return_126d", "return_126d", "volatility_30d"),
    ("vol_adj_return_252d", "return_252d", "volatility_30d"),
]


@dataclass
class TransformFitState:
    """Statistics fit on the TRAIN split — applied to validation/test rows.

    Anything past-only / per-ticker (rolling z-score, percentile ranks) does
    not live in here — those are recomputed from the row's own ticker history.
    """
    winsor_bounds: dict[str, tuple[float, float]]
    regime_thresholds: dict[str, float]


def _signed_log1p(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.sign(s) * np.log1p(np.abs(s))


def _safe_log1p_nonneg(s: pd.Series) -> pd.Series:
    """log1p with negative values masked to NaN (so a strictly nonneg variable
    stays interpretable). Sentinels like -1 in volume become NaN."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s >= 0, np.nan)
    return np.log1p(s)


def fit_transform_state(train_df: pd.DataFrame) -> TransformFitState:
    """Fit winsorisation bounds and regime thresholds on TRAIN rows only."""
    winsor: dict[str, tuple[float, float]] = {}
    for col in WINSORIZE_CANDIDATES:
        if col not in train_df.columns:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            continue
        lo = float(np.nanpercentile(finite, 1.0))
        hi = float(np.nanpercentile(finite, 99.0))
        if lo == hi:
            continue
        winsor[col] = (lo, hi)

    regime: dict[str, float] = {}
    if "cpi_yoy" in train_df.columns:
        cpi = pd.to_numeric(train_df["cpi_yoy"], errors="coerce").dropna()
        if len(cpi) > 0:
            regime["cpi_high_q75"] = float(np.nanpercentile(cpi, 75))
            regime["cpi_low_q25"]  = float(np.nanpercentile(cpi, 25))
    if "federal_funds_change_12p" in train_df.columns:
        rate_chg = pd.to_numeric(train_df["federal_funds_change_12p"], errors="coerce").dropna()
        if len(rate_chg) > 0:
            regime["fed_funds_rising_q75"]  = float(np.nanpercentile(rate_chg, 75))
            regime["fed_funds_falling_q25"] = float(np.nanpercentile(rate_chg, 25))
    if "volatility_30d" in train_df.columns:
        vol = pd.to_numeric(train_df["volatility_30d"], errors="coerce").dropna()
        if len(vol) > 0:
            regime["volatility_high_q75"] = float(np.nanpercentile(vol, 75))
    if "financial_stress_index" in train_df.columns:
        stress = pd.to_numeric(train_df["financial_stress_index"], errors="coerce").dropna()
        if len(stress) > 0:
            regime["financial_stress_high_q75"] = float(np.nanpercentile(stress, 75))

    debug_print(f"Fit winsor bounds on {len(winsor)} cols; regime thresholds: {regime}")
    return TransformFitState(winsor_bounds=winsor, regime_thresholds=regime)


def apply_transformations(
    df: pd.DataFrame,
    state: TransformFitState,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Apply every transformation to a dataframe, return (df_with_new_cols, registry).

    The registry lists each generated column with its family + a short
    description for the dashboard's methodology view.
    """
    out = df.copy()
    registry: list[dict[str, str]] = []

    def _register(name: str, family: str, description: str) -> None:
        registry.append({"feature": name, "family": family, "description": description})

    # 1. log / signed-log -----------------------------------------------------
    for col in NONNEG_LOG1P_CANDIDATES:
        if col in out.columns:
            new_col = f"log1p_{col}"
            out[new_col] = _safe_log1p_nonneg(out[col])
            _register(new_col, "log1p", f"log1p({col}) — compresses right-skew of nonneg variable")

    for col in SIGNED_LOG_CANDIDATES:
        if col in out.columns:
            new_col = f"signed_log_{col}"
            out[new_col] = _signed_log1p(out[col])
            _register(new_col, "signed_log", f"sign(x)·log1p|x| of {col} — compresses tails while preserving sign")

    # 2. winsorisation (training-fit bounds) ----------------------------------
    for col, (lo, hi) in state.winsor_bounds.items():
        if col not in out.columns:
            continue
        new_col = f"winsor_{col}"
        series = pd.to_numeric(out[col], errors="coerce")
        out[new_col] = series.clip(lower=lo, upper=hi)
        _register(new_col, "winsor", f"{col} clipped to TRAIN-fit [{lo:.4g}, {hi:.4g}]")

    # 3. rolling z-scores BY TICKER (past-only expanding) ---------------------
    if "ticker" in out.columns:
        sorted_idx = out.sort_values(["ticker", "date"]).index
        for col in ROLLING_ZSCORE_CANDIDATES:
            if col not in out.columns:
                continue
            new_col = f"z_{col}_by_ticker"
            # `shift(1)` ensures the current row's value is NOT in its own
            # reference distribution — no look-ahead within a row.
            grp = out.loc[sorted_idx].groupby("ticker", group_keys=False)[col]
            past = grp.apply(lambda s: s.shift(1))
            mean = grp.apply(lambda s: s.shift(1).expanding(min_periods=12).mean())
            std  = grp.apply(lambda s: s.shift(1).expanding(min_periods=12).std())
            z = (out.loc[sorted_idx, col] - mean) / std.replace(0, np.nan)
            # Re-align to the original index ordering.
            z = z.reindex(out.index)
            out[new_col] = z
            _register(new_col, "rolling_zscore_by_ticker",
                      f"Past-only expanding z-score of {col} within each ticker; min 12 weeks")
            # Suppress unused-var warning while keeping the intermediate
            # `past` value around for review.
            _ = past

    # 4. expanding percentile ranks BY TICKER (past-only) ---------------------
    if "ticker" in out.columns:
        sorted_idx = out.sort_values(["ticker", "date"]).index
        for col in PERCENTILE_CANDIDATES:
            if col not in out.columns:
                continue
            new_col = f"ticker_percentile_{col}"
            ser = out.loc[sorted_idx, col]

            def _expanding_rank(s: pd.Series) -> pd.Series:
                # Past-only rank of the latest value vs all prior values
                # within the same ticker. Implementation: for each position
                # i, rank value at i against values at 0..i-1 (excluding i),
                # divided by count of priors. Returns NaN until at least 12
                # prior observations exist.
                arr = s.to_numpy(dtype=float)
                out_arr = np.full(len(arr), np.nan)
                for i in range(len(arr)):
                    if i < 12:
                        continue
                    prior = arr[:i]
                    prior = prior[np.isfinite(prior)]
                    if len(prior) < 12:
                        continue
                    if not np.isfinite(arr[i]):
                        continue
                    out_arr[i] = float((prior < arr[i]).sum()) / len(prior)
                return pd.Series(out_arr, index=s.index)

            ranks = ser.groupby(out.loc[sorted_idx, "ticker"]).apply(_expanding_rank)
            # `apply` on grouped Series adds a level — drop it
            if isinstance(ranks.index, pd.MultiIndex):
                ranks = ranks.droplevel(0)
            out[new_col] = ranks.reindex(out.index)
            _register(new_col, "expanding_percentile_by_ticker",
                      f"Past-only expanding percentile rank of {col} within each ticker")

    # 5. interactions ---------------------------------------------------------
    for name, left, right, desc in INTERACTION_PAIRS:
        if left in out.columns and right in out.columns:
            out[name] = pd.to_numeric(out[left], errors="coerce") * pd.to_numeric(out[right], errors="coerce")
            _register(name, "interaction", desc)

    # 6. regime flags (training-fit thresholds) -------------------------------
    if "cpi_high_q75" in state.regime_thresholds and "cpi_yoy" in out.columns:
        out["regime_high_inflation"] = (pd.to_numeric(out["cpi_yoy"], errors="coerce")
                                        > state.regime_thresholds["cpi_high_q75"]).astype(int)
        _register("regime_high_inflation", "regime",
                  f"cpi_yoy > train Q75 ({state.regime_thresholds['cpi_high_q75']:.3g})")
    if "cpi_low_q25" in state.regime_thresholds and "cpi_yoy" in out.columns:
        out["regime_low_inflation"] = (pd.to_numeric(out["cpi_yoy"], errors="coerce")
                                       < state.regime_thresholds["cpi_low_q25"]).astype(int)
        _register("regime_low_inflation", "regime",
                  f"cpi_yoy < train Q25 ({state.regime_thresholds['cpi_low_q25']:.3g})")
    if "fed_funds_rising_q75" in state.regime_thresholds and "federal_funds_change_12p" in out.columns:
        out["regime_rising_rates"] = (pd.to_numeric(out["federal_funds_change_12p"], errors="coerce")
                                      > state.regime_thresholds["fed_funds_rising_q75"]).astype(int)
        _register("regime_rising_rates", "regime",
                  f"fed funds change > train Q75 ({state.regime_thresholds['fed_funds_rising_q75']:.3g})")
    if "fed_funds_falling_q25" in state.regime_thresholds and "federal_funds_change_12p" in out.columns:
        out["regime_falling_rates"] = (pd.to_numeric(out["federal_funds_change_12p"], errors="coerce")
                                       < state.regime_thresholds["fed_funds_falling_q25"]).astype(int)
        _register("regime_falling_rates", "regime",
                  f"fed funds change < train Q25 ({state.regime_thresholds['fed_funds_falling_q25']:.3g})")
    if "recession_indicator" in out.columns:
        out["regime_recession"] = (pd.to_numeric(out["recession_indicator"], errors="coerce") > 0.5).astype(int)
        _register("regime_recession", "regime", "NBER recession indicator > 0.5")
        out["regime_non_recession"] = 1 - out["regime_recession"]
        _register("regime_non_recession", "regime", "1 - regime_recession")
    if "volatility_high_q75" in state.regime_thresholds and "volatility_30d" in out.columns:
        out["regime_high_volatility"] = (pd.to_numeric(out["volatility_30d"], errors="coerce")
                                         > state.regime_thresholds["volatility_high_q75"]).astype(int)
        _register("regime_high_volatility", "regime",
                  f"volatility_30d > train Q75 ({state.regime_thresholds['volatility_high_q75']:.3g})")
    if "financial_stress_high_q75" in state.regime_thresholds and "financial_stress_index" in out.columns:
        out["regime_financial_stress"] = (pd.to_numeric(out["financial_stress_index"], errors="coerce")
                                          > state.regime_thresholds["financial_stress_high_q75"]).astype(int)
        _register("regime_financial_stress", "regime",
                  f"financial_stress_index > train Q75 ({state.regime_thresholds['financial_stress_high_q75']:.3g})")

    # 7. volatility-adjusted returns -----------------------------------------
    for name, ret_col, vol_col in VOL_ADJ_RETURNS:
        if ret_col in out.columns and vol_col in out.columns:
            ret = pd.to_numeric(out[ret_col], errors="coerce")
            vol = pd.to_numeric(out[vol_col], errors="coerce")
            # Avoid divide-by-zero / divide-by-tiny: gate on vol > eps
            denom = vol.where(vol > 1e-6, np.nan)
            out[name] = ret / denom
            _register(name, "vol_adjusted_return",
                      f"{ret_col} / {vol_col}, gated to vol > 1e-6")

    # 8. drawdown-state flags -------------------------------------------------
    if "drawdown_pct" in out.columns:
        dd = pd.to_numeric(out["drawdown_pct"], errors="coerce")
        # `drawdown_pct` is stored in percent space (e.g. -25.0 == -25%)
        out["in_20pct_drawdown"] = (dd <= -20).astype(int)
        out["in_40pct_drawdown"] = (dd <= -40).astype(int)
        out["in_60pct_drawdown"] = (dd <= -60).astype(int)
        for col in ("in_20pct_drawdown", "in_40pct_drawdown", "in_60pct_drawdown"):
            _register(col, "drawdown_state", "Drawdown depth flag")
    if "distance_from_200dma" in out.columns:
        dist = pd.to_numeric(out["distance_from_200dma"], errors="coerce")
        out["above_200dma"] = (dist > 0).astype(int)
        out["below_200dma"] = (dist <= 0).astype(int)
        out["extreme_above_200dma"] = (dist > 30).astype(int)
        out["extreme_below_200dma"] = (dist < -30).astype(int)
        for col in ("above_200dma", "below_200dma", "extreme_above_200dma", "extreme_below_200dma"):
            _register(col, "drawdown_state", "Trend-position flag (% relative to 200DMA)")
    if "reclaim_200dma" in out.columns:
        out["reclaimed_200dma"] = pd.to_numeric(out["reclaim_200dma"], errors="coerce").fillna(0).astype(int)
        _register("reclaimed_200dma", "drawdown_state", "Recovery flag — reclaimed 200DMA from below")
    if "weekly_rsi" in out.columns:
        rsi = pd.to_numeric(out["weekly_rsi"], errors="coerce")
        out["rsi_overbought"] = (rsi >= 70).astype(int)
        out["rsi_oversold"]   = (rsi <= 30).astype(int)
        _register("rsi_overbought", "drawdown_state", "Weekly RSI ≥ 70")
        _register("rsi_oversold",   "drawdown_state", "Weekly RSI ≤ 30")

    # 9. squared terms --------------------------------------------------------
    for col in SQUARED_CANDIDATES:
        if col in out.columns:
            new_col = f"sq_{col}"
            series = pd.to_numeric(out[col], errors="coerce")
            out[new_col] = series * series
            _register(new_col, "squared", f"{col} squared — captures curvature")

    debug_print(f"apply_transformations: produced {len(registry)} new columns")
    return out, registry


# ---------------------------------------------------------------------------
# Feature column selection. Strategy: baseline candidates (intersected with
# what exists) ∪ new transformed columns from the registry.
# ---------------------------------------------------------------------------
LEAKAGE_MARKERS = ("future_", "target_valid_", "burst_3m", "burst_6m", "burst_12m",
                   "hazard_event_4w", "threshold_burst_")


def assemble_feature_set(
    df: pd.DataFrame,
    transform_registry: list[dict[str, str]],
) -> list[str]:
    """Final feature column list for the transformed experiment.

    Includes:
        - every original numeric candidate from baseline (so the comparison
          isolates "transformations" rather than "different feature universe");
        - every generated transformed column.
    Excludes any column that matches the leakage marker list.
    """
    baseline_present = [
        c for c in BASELINE_FEATURE_CANDIDATES
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    transformed = [r["feature"] for r in transform_registry if r["feature"] in df.columns]
    combined: list[str] = []
    seen: set[str] = set()
    for col in baseline_present + transformed:
        if col in seen:
            continue
        if any(marker in col for marker in LEAKAGE_MARKERS):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        combined.append(col)
        seen.add(col)
    debug_print(f"assemble_feature_set: {len(baseline_present)} baseline + "
                f"{len(transformed)} transformed -> {len(combined)} unique")
    return combined


# ---------------------------------------------------------------------------
# Model factory — mirrors baseline pipelines but uses NEW feature_cols.
# LightGBM is added when the import succeeded; never when it didn't.
# ---------------------------------------------------------------------------
def _logistic_pipeline(feature_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("numeric",
                       Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler",  StandardScaler())]),
                       feature_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", pre),
                     ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"))])


def _elastic_net_pipeline(feature_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("numeric",
                       Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler",  StandardScaler())]),
                       feature_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", pre),
                     ("model", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                   solver="saga", penalty="elasticnet", l1_ratio=0.50))])


def _random_forest_pipeline(feature_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", pre),
                     ("model", RandomForestClassifier(
                         n_estimators=500, max_depth=8, min_samples_leaf=8,
                         random_state=42, n_jobs=-1, class_weight="balanced_subsample"))])


def _balanced_rf_pipeline(feature_cols: list[str]) -> Pipeline | None:
    if BalancedRandomForestClassifier is None:
        return None
    pre = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", pre),
                     ("model", BalancedRandomForestClassifier(
                         n_estimators=500, max_depth=8, min_samples_leaf=8,
                         random_state=42, n_jobs=-1, replacement=True, sampling_strategy="all"))])


def _xgboost_pipeline(feature_cols: list[str], y_train: pd.Series) -> Pipeline | None:
    if XGBClassifier is None:
        return None
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0
    pre = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", pre),
                     ("model", XGBClassifier(
                         n_estimators=450, max_depth=3, learning_rate=0.035,
                         subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
                         reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
                         random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight))])


def _lightgbm_pipeline(feature_cols: list[str], y_train: pd.Series) -> Pipeline | None:
    """LightGBM with class imbalance handled via class_weight='balanced'.

    Returns None when lightgbm is not installed — we don't crash the
    experiment when an optional dependency is missing.
    """
    if LGBMClassifier is None:
        return None
    pre = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", pre),
                     ("model", LGBMClassifier(
                         n_estimators=600, max_depth=-1, num_leaves=31,
                         learning_rate=0.03, min_child_samples=20,
                         reg_lambda=1.0, class_weight="balanced",
                         random_state=42, n_jobs=-1, verbosity=-1))])


def build_models(feature_cols: list[str], y_train: pd.Series) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "Logistic Regression":   _logistic_pipeline(feature_cols),
        "Elastic Net Logistic":  _elastic_net_pipeline(feature_cols),
        "Random Forest":         _random_forest_pipeline(feature_cols),
    }
    brf = _balanced_rf_pipeline(feature_cols)
    if brf is not None:
        models["Balanced Random Forest"] = brf
    xgb = _xgboost_pipeline(feature_cols, y_train)
    if xgb is not None:
        models["XGBoost"] = xgb
    lgbm = _lightgbm_pipeline(feature_cols, y_train)
    if lgbm is not None:
        models["LightGBM"] = lgbm
    debug_print(f"Model families this run: {sorted(models.keys())}")
    return models


# ---------------------------------------------------------------------------
# Evaluation helpers — replicate the baseline metric shape so comparison is
# apples-to-apples in the dashboard.
# ---------------------------------------------------------------------------
def evaluate_predictions(model_name: str, segment: str, split_name: str,
                         y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positive_rate = float(np.mean(y_true)) if len(y_true) else np.nan
    alert_rate = float(np.mean(y_pred)) if len(y_pred) else np.nan
    precision = safe_metric("precision", y_true, y_prob, y_pred)
    recall = safe_metric("recall", y_true, y_prob, y_pred)
    return {
        "feature_set": "transformed",
        "model": model_name,
        "asset_segment": segment,
        "model_scope": segment,
        "split": split_name,
        "threshold_value": float(threshold),
        "n_rows": int(len(y_true)),
        "positive_rate": positive_rate,
        "alert_rate": alert_rate,
        "roc_auc": safe_metric("roc_auc", y_true, y_prob, y_pred),
        "pr_auc": safe_metric("pr_auc", y_true, y_prob, y_pred),
        "accuracy": safe_metric("accuracy", y_true, y_prob, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": safe_metric("f1", y_true, y_prob, y_pred),
        "mcc": safe_metric("mcc", y_true, y_prob, y_pred),
        "brier_score": safe_metric("brier_score", y_true, y_prob, y_pred),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positives_per_true_positive": float(fp / tp) if tp > 0 else float("inf"),
    }


def add_prediction_rows(split_df: pd.DataFrame, model_name: str, segment: str,
                        split_name: str, y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    cols = [c for c in ("date", "ticker", "asset_segment", TARGET, "warning_score") if c in split_df.columns]
    out = split_df[cols].copy()
    out["feature_set"] = "transformed"
    out["model"] = model_name
    out["model_scope"] = segment
    out["split"] = split_name
    out["probability"] = y_prob
    out["threshold_value"] = float(threshold)
    out["prediction"] = (out["probability"] >= threshold).astype(int)
    out["risk_rank_pct"] = out["probability"].rank(pct=True, method="average")
    out["top_decile_flag"] = out["risk_rank_pct"] >= 0.90
    out["top_5pct_flag"] = out["risk_rank_pct"] >= 0.95
    out["risk_bucket"] = pd.cut(
        out["risk_rank_pct"],
        bins=[-np.inf, 0.50, 0.75, 0.90, 0.95, np.inf],
        labels=["Bottom 50%", "50-75%", "75-90%", "90-95%", "Top 5%"],
    ).astype(str)
    return out


def top_k_lift_rows(pred_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Top-5% and top-10% lift relative to base event rate."""
    if pred_df.empty or pred_df[TARGET].isna().all():
        return []
    base = float(pred_df[TARGET].astype(int).mean())
    out: list[dict[str, Any]] = []
    for flag_col, label in (("top_decile_flag", "Top 10%"), ("top_5pct_flag", "Top 5%")):
        sub = pred_df[pred_df[flag_col]]
        n = len(sub)
        if n == 0:
            continue
        top_rate = float(sub[TARGET].astype(int).mean())
        out.append({
            "feature_set":  "transformed",
            "model":        pred_df["model"].iloc[0],
            "model_scope":  pred_df["model_scope"].iloc[0],
            "split":        pred_df["split"].iloc[0],
            "top_label":    label,
            "n_rows":       n,
            "top_event_rate": top_rate,
            "base_event_rate": base,
            "lift_vs_base_rate": float(top_rate / base) if base > 0 else np.nan,
        })
    return out


def calibration_table(pred_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if pred_df.empty or pred_df["probability"].nunique(dropna=True) < 2:
        return pd.DataFrame()
    tmp = pred_df.copy()
    try:
        tmp["probability_bin"] = pd.qcut(tmp["probability"],
                                          q=min(n_bins, tmp["probability"].nunique()),
                                          duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bin_value, g in tmp.groupby("probability_bin", observed=True):
        rows.append({
            "feature_set": "transformed",
            "model":       g["model"].iloc[0],
            "model_scope": g["model_scope"].iloc[0],
            "split":       g["split"].iloc[0],
            "probability_bin": str(bin_value),
            "bin_left":  float(bin_value.left),
            "bin_right": float(bin_value.right),
            "n_rows":    int(len(g)),
            "mean_predicted_probability": float(g["probability"].mean()),
            "empirical_event_rate":        float(g[TARGET].astype(int).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Run `python ml_dataset.py` first.")
    df = pd.read_parquet(DATASET_PATH)
    if df.empty:
        raise ValueError(f"{DATASET_PATH} is empty.")
    df["date"] = pd.to_datetime(df["date"])
    df["asset_segment"] = df["ticker"].astype(str).map(infer_asset_segment)
    debug_print(f"Loaded baseline ML dataset: shape={df.shape}")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def filter_labeled(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns or TARGET_VALID not in df.columns:
        raise ValueError(f"Dataset must contain {TARGET} and {TARGET_VALID}. Re-run ml_dataset.py.")
    labeled = df[(df[TARGET_VALID] == True) & df[TARGET].notna()].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)
    debug_print(f"Labeled rows: {len(labeled):,}; class balance:\n"
                f"{labeled[TARGET].value_counts().sort_index().to_string()}")
    return labeled


def train_evaluate_scope_transformed(
    scope_name: str,
    scope_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]],
           list[pd.DataFrame],
           list[pd.DataFrame],
           list[dict[str, Any]],
           list[dict[str, Any]],
           list[dict[str, str]]]:
    """Mirror of model_training.train_evaluate_scope, using transformed features."""
    debug_print(f"=== scope: {scope_name} — rows={len(scope_df):,} ===")
    train, val, test = build_chronological_splits(scope_df)

    # Fit transform statistics on TRAIN only, then apply identically to all splits.
    state = fit_transform_state(train)
    train_t, train_registry = apply_transformations(train, state)
    val_t,   _ = apply_transformations(val, state)
    test_t,  _ = apply_transformations(test, state)

    feature_cols = assemble_feature_set(train_t, train_registry)
    debug_print(f"Final feature count for scope {scope_name}: {len(feature_cols)}")

    X_train = clean_feature_matrix(train_t[feature_cols], feature_cols, context=f"{scope_name} train (transformed)")
    y_train = train_t[TARGET].astype(int)

    models = build_models(feature_cols, y_train)

    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    calibrations: list[pd.DataFrame] = []
    top_k_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    # Rule-based baseline carried as a reference inside the transformed-only
    # output set, so the dashboard can show "even after transformations, the
    # rule-based baseline still wins on precision X% of the time."
    rule_val_prob = build_rule_based_probabilities(val_t)
    rule_thresh = choose_threshold_on_validation(val_t[TARGET].astype(int).to_numpy(), rule_val_prob)
    threshold_rows.append({"feature_set": "transformed", "model": "Rule-Based Warning Score",
                            "model_scope": scope_name, **rule_thresh})
    for split_name, split_df in (("validation", val_t), ("test", test_t)):
        y_true = split_df[TARGET].astype(int).to_numpy()
        y_prob = build_rule_based_probabilities(split_df)
        thr = float(rule_thresh["threshold_value"])
        results.append(evaluate_predictions("Rule-Based Warning Score", scope_name, split_name, y_true, y_prob, thr))
        pred_df = add_prediction_rows(split_df, "Rule-Based Warning Score", scope_name, split_name, y_prob, thr)
        predictions.append(pred_df)
        top_k_rows.extend(top_k_lift_rows(pred_df))
        cal = calibration_table(pred_df)
        if not cal.empty:
            calibrations.append(cal)

    for model_name, pipeline in models.items():
        debug_print(f"Training {model_name} / {scope_name}")
        try:
            pipeline.fit(X_train, y_train)
        except Exception as exc:  # noqa: BLE001
            debug_print(f"  fit failed: {exc} — skipping")
            continue
        X_val = clean_feature_matrix(val_t[feature_cols], feature_cols,
                                     context=f"{scope_name} val (transformed) {model_name}")
        val_prob = predict_probability(pipeline, X_val)
        thresh_info = choose_threshold_on_validation(val_t[TARGET].astype(int).to_numpy(), val_prob)
        threshold_rows.append({"feature_set": "transformed", "model": model_name,
                                "model_scope": scope_name, **thresh_info})
        thr = float(thresh_info["threshold_value"])
        for split_name, split_df in (("validation", val_t), ("test", test_t)):
            y_true = split_df[TARGET].astype(int).to_numpy()
            X_split = clean_feature_matrix(split_df[feature_cols], feature_cols,
                                           context=f"{scope_name} {split_name} (transformed) {model_name}")
            y_prob = predict_probability(pipeline, X_split)
            results.append(evaluate_predictions(model_name, scope_name, split_name, y_true, y_prob, thr))
            pred_df = add_prediction_rows(split_df, model_name, scope_name, split_name, y_prob, thr)
            predictions.append(pred_df)
            top_k_rows.extend(top_k_lift_rows(pred_df))
            cal = calibration_table(pred_df)
            if not cal.empty:
                calibrations.append(cal)

    return results, predictions, calibrations, top_k_rows, threshold_rows, train_registry


def refit_and_save_final_models(
    labeled: pd.DataFrame,
    state_global: TransformFitState,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Refit transformed models on all completed labels (global + segments).

    Saves trained pipelines under models/transformed/. Returns combined
    feature-importance frame and the registry list.
    """
    importance_frames: list[pd.DataFrame] = []
    full_registry: list[dict[str, str]] = []
    registry_seen: set[str] = set()

    labeled_t, full_registry_local = apply_transformations(labeled, state_global)
    for r in full_registry_local:
        if r["feature"] not in registry_seen:
            full_registry.append(r)
            registry_seen.add(r["feature"])
    feature_cols = assemble_feature_set(labeled_t, full_registry_local)

    scope_frames: dict[str, pd.DataFrame] = {"global": labeled_t}
    for segment in ("broad_index_etf", "mega_cap_ai_tech", "speculative_high_vol"):
        sub = labeled_t[labeled_t["asset_segment"].eq(segment)].copy()
        if len(sub) >= MIN_SEGMENT_ROWS and sub[TARGET].sum() >= MIN_SEGMENT_POSITIVES and sub[TARGET].nunique() == 2:
            scope_frames[segment] = sub
        else:
            debug_print(f"Skipping final transformed segment {segment} (rows={len(sub):,}, pos={int(sub[TARGET].sum()) if not sub.empty else 0})")

    for scope_name, scope_df in scope_frames.items():
        X = clean_feature_matrix(scope_df[feature_cols], feature_cols, context=f"{scope_name} final refit (transformed)")
        y = scope_df[TARGET].astype(int)
        for model_name, pipeline in build_models(feature_cols, y).items():
            try:
                pipeline.fit(X, y)
            except Exception as exc:  # noqa: BLE001
                debug_print(f"refit failed: {model_name}/{scope_name}: {exc}")
                continue
            slug = f"{slugify(scope_name)}_{slugify(model_name)}_burst_6m.pkl"
            joblib.dump(pipeline, TRANSFORMED_MODELS_DIR / slug)
            importance_frames.append(extract_feature_importance(model_name, scope_name, pipeline, feature_cols))

    importance_df = (
        pd.concat([f for f in importance_frames if not f.empty], ignore_index=True)
        if importance_frames else pd.DataFrame()
    )
    if not importance_df.empty:
        importance_df.insert(0, "feature_set", "transformed")
    return importance_df, full_registry


def write_summary_markdown(
    results_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    registry: list[dict[str, str]],
) -> None:
    SUMMARY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Transformed-Feature Experiment — Summary")
    lines.append("")
    lines.append(
        "This experiment fits the same model families on a transformed-feature "
        "pipeline and saves results separately from the baseline so the dashboard "
        "can render a controlled, leakage-respecting comparison."
    )
    lines.append("")
    lines.append("> **Framing.** Feature transformations may improve feature geometry and "
                 "make patterns easier for models to learn, but they do not create new "
                 "independent historical bubbles. They improve signal extraction; they do "
                 "not remove regime uncertainty.")
    lines.append("")
    lines.append("## Transformations applied")
    by_family: dict[str, list[dict[str, str]]] = {}
    for r in registry:
        by_family.setdefault(r["family"], []).append(r)
    for family in sorted(by_family.keys()):
        rows = by_family[family]
        lines.append(f"### {family} ({len(rows)} features)")
        for r in rows[:25]:
            lines.append(f"- `{r['feature']}` — {r['description']}")
        if len(rows) > 25:
            lines.append(f"- … and {len(rows) - 25} more in this family")
        lines.append("")

    lines.append("## Test-split top results by model and scope")
    test = results_df[results_df["split"].eq("test")].copy() if "split" in results_df.columns else pd.DataFrame()
    if test.empty:
        lines.append("_No test-split results produced._")
    else:
        sort_metric = "pr_auc" if "pr_auc" in test.columns else "roc_auc"
        for scope in sorted(test["model_scope"].dropna().unique()):
            sub = test[test["model_scope"].eq(scope)].sort_values(sort_metric, ascending=False)
            lines.append(f"### Scope: `{scope}`")
            lines.append("")
            lines.append("| Model | PR-AUC | ROC-AUC | MCC | Precision | Recall | FP/TP |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for _, row in sub.iterrows():
                pr_a = f"{row.get('pr_auc', float('nan')):.3f}" if pd.notna(row.get("pr_auc")) else "n/a"
                ro_a = f"{row.get('roc_auc', float('nan')):.3f}" if pd.notna(row.get("roc_auc")) else "n/a"
                mcc  = f"{row.get('mcc',     float('nan')):.3f}" if pd.notna(row.get("mcc"))     else "n/a"
                pre  = f"{row.get('precision', float('nan')):.3f}" if pd.notna(row.get("precision")) else "n/a"
                rec  = f"{row.get('recall', float('nan')):.3f}" if pd.notna(row.get("recall")) else "n/a"
                fp_tp = row.get("false_positives_per_true_positive", float("inf"))
                fp_tp_s = "inf" if not np.isfinite(fp_tp) else f"{fp_tp:.2f}"
                lines.append(f"| {row['model']} | {pr_a} | {ro_a} | {mcc} | {pre} | {rec} | {fp_tp_s} |")
            lines.append("")

    lines.append("## Notes / honest caveats")
    lines.append("")
    lines.append("- Compare these to `data/processed/ml_model_results.parquet` (baseline) before "
                 "concluding that transformations help. The dashboard provides a side-by-side view.")
    lines.append("- Improvements only count if they hold on the **test** split AND do not increase "
                 "false-positive burden meaningfully.")
    lines.append("- Linear models (Logistic, Elastic Net) usually benefit most from these "
                 "transformations because they cannot internally rescale or interact features.")
    lines.append("- Tree-based models (RF, Balanced RF, XGBoost, LightGBM) are less sensitive — "
                 "if a transformation 'helps' a tree model dramatically, suspect leakage or "
                 "over-correlation with the target before celebrating.")
    SUMMARY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    debug_print(f"Wrote summary report -> {SUMMARY_REPORT}")


def main() -> None:
    debug_print("Feature-transformation experiment start")
    df = load_dataset()
    labeled = filter_labeled(df)

    # Scope: global + segment-specific (same rules as baseline training).
    scope_frames: dict[str, pd.DataFrame] = {"global": labeled}
    for segment in ("broad_index_etf", "mega_cap_ai_tech", "speculative_high_vol"):
        sub = labeled[labeled["asset_segment"].eq(segment)].copy()
        if len(sub) >= MIN_SEGMENT_ROWS and sub[TARGET].sum() >= MIN_SEGMENT_POSITIVES and sub[TARGET].nunique() == 2:
            scope_frames[segment] = sub
        else:
            debug_print(f"Skipping evaluation segment {segment}: rows={len(sub):,}, "
                        f"positives={int(sub[TARGET].sum()) if not sub.empty else 0}")

    all_results: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_calibrations: list[pd.DataFrame] = []
    all_top_k: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    full_registry: list[dict[str, str]] = []
    registry_seen: set[str] = set()

    for scope_name, scope_df in scope_frames.items():
        try:
            r, p, c, tk, th, reg = train_evaluate_scope_transformed(scope_name, scope_df)
            all_results.extend(r)
            all_predictions.extend(p)
            all_calibrations.extend(c)
            all_top_k.extend(tk)
            threshold_rows.extend(th)
            for entry in reg:
                if entry["feature"] not in registry_seen:
                    full_registry.append(entry)
                    registry_seen.add(entry["feature"])
        except Exception as exc:  # noqa: BLE001
            debug_print(f"Skipping scope {scope_name} after error: {exc}")

    # Final refit on all labeled data for inference + feature-importance export.
    # Fit transform state on the FULL labeled training-period rows.
    labeled_sorted = labeled.sort_values("date")
    final_train_state = fit_transform_state(labeled_sorted)
    importance_df, registry_full = refit_and_save_final_models(labeled, final_train_state)
    for entry in registry_full:
        if entry["feature"] not in registry_seen:
            full_registry.append(entry)
            registry_seen.add(entry["feature"])

    # Persist outputs (every path namespaced — no baseline file is touched).
    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    calibration_df = pd.concat(all_calibrations, ignore_index=True) if all_calibrations else pd.DataFrame()
    top_k_df = pd.DataFrame(all_top_k)
    registry_df = pd.DataFrame(full_registry)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(RESULTS_PATH,         index=False)
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    importance_df.to_parquet(IMPORTANCE_PATH,   index=False) if not importance_df.empty else None
    top_k_df.to_parquet(TOP_K_LIFT_PATH,        index=False)
    calibration_df.to_parquet(CALIBRATION_PATH, index=False)
    registry_df.to_parquet(FEATURE_LIST_PATH,   index=False)

    write_summary_markdown(results_df, importance_df, full_registry)

    debug_print(f"Saved: {RESULTS_PATH}")
    debug_print(f"Saved: {PREDICTIONS_PATH}")
    debug_print(f"Saved: {IMPORTANCE_PATH}")
    debug_print(f"Saved: {TOP_K_LIFT_PATH}")
    debug_print(f"Saved: {CALIBRATION_PATH}")
    debug_print(f"Saved: {FEATURE_LIST_PATH}")
    debug_print(f"Saved: {SUMMARY_REPORT}")
    debug_print("Feature-transformation experiment complete.")


if __name__ == "__main__":
    main()
