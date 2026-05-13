"""
Build the supervised machine-learning dataset for forward drawdown-risk modeling.

Run after the normal signal pipeline:
    python data_ingestion.py
    python feature_engineering.py
    python bubble_signals.py
    python ml_dataset.py

Output:
    data/processed/ml_burst_dataset.parquet

Research framing:
    This script does NOT try to label vague "bubble" opinions. Instead, it
    creates measurable target variables from realized future drawdowns. The
    primary label is burst_6m, which equals 1 if the asset later falls at least
    30% from the current weekly close at any point over the next 26 weeks.

Leakage rule:
    Features are current/past information only. Future prices are used only to
    create historical labels for supervised training. Current/live rows near the
    end of the dataset will naturally have missing target labels because the full
    future horizon has not happened yet; those rows are still useful for inference.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "ml_dataset.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ml_dataset")

OUTPUT_PATH = PROCESSED_DIR / "ml_burst_dataset.parquet"

# The target horizons are expressed in weekly observations because this ML layer
# intentionally compresses daily market data to weekly rows. A weekly row is far
# less noisy than a daily row and reduces the chance that the model memorizes
# tiny daily wiggles rather than meaningful market regimes.
TARGET_SPECS = {
    "burst_3m": {"horizon_weeks": 13, "drawdown_threshold": -0.20},
    "burst_6m": {"horizon_weeks": 26, "drawdown_threshold": -0.30},
    "burst_12m": {"horizon_weeks": 52, "drawdown_threshold": -0.40},
}

# Segment-specific drawdown thresholds make the label more economically realistic.
# A 30% drawdown means something very different for SPY than it does for BTC or
# PLTR. The uniform burst_6m target is kept for comparison, but the upgraded ML
# training script uses burst_6m_segment by default.
SEGMENT_TARGET_SPECS = {
    "broad_index_etf": {
        "burst_3m_segment": {"horizon_weeks": 13, "drawdown_threshold": -0.12},
        "burst_6m_segment": {"horizon_weeks": 26, "drawdown_threshold": -0.20},
        "burst_12m_segment": {"horizon_weeks": 52, "drawdown_threshold": -0.30},
    },
    "mega_cap_ai_tech": {
        "burst_3m_segment": {"horizon_weeks": 13, "drawdown_threshold": -0.20},
        "burst_6m_segment": {"horizon_weeks": 26, "drawdown_threshold": -0.30},
        "burst_12m_segment": {"horizon_weeks": 52, "drawdown_threshold": -0.40},
    },
    "speculative_high_vol": {
        "burst_3m_segment": {"horizon_weeks": 13, "drawdown_threshold": -0.30},
        "burst_6m_segment": {"horizon_weeks": 26, "drawdown_threshold": -0.50},
        "burst_12m_segment": {"horizon_weeks": 52, "drawdown_threshold": -0.60},
    },
    "other_single_name": {
        "burst_3m_segment": {"horizon_weeks": 13, "drawdown_threshold": -0.20},
        "burst_6m_segment": {"horizon_weeks": 26, "drawdown_threshold": -0.30},
        "burst_12m_segment": {"horizon_weeks": 52, "drawdown_threshold": -0.40},
    },
}

MACRO_RENAME_MAP = {
    "FEDFUNDS_value": "federal_funds_rate",
    "FEDFUNDS_change_3p": "federal_funds_change_3p",
    "FEDFUNDS_change_12p": "federal_funds_change_12p",
    "DGS10_value": "ten_year_treasury_yield",
    "DGS10_change_3p": "ten_year_treasury_change_3p",
    "DGS10_change_12p": "ten_year_treasury_change_12p",
    "T10Y2Y_value": "yield_curve_spread",
    "T10Y2Y_change_3p": "yield_curve_change_3p",
    "T10Y2Y_change_12p": "yield_curve_change_12p",
    "CPIAUCSL_pct_change_12p": "cpi_yoy",
    "CPIAUCSL_change_12p": "cpi_change_12p",
    "UNRATE_value": "unemployment_rate",
    "UNRATE_change_3p": "unemployment_change_3p",
    "UNRATE_change_12p": "unemployment_change_12p",
    "USREC_value": "recession_indicator",
    "STLFSI4_value": "financial_stress_index",
    "MORTGAGE30US_value": "mortgage_rate",
    "MORTGAGE30US_change_12p": "mortgage_rate_change_12p",
    "CSUSHPINSA_pct_change_12p": "case_shiller_yoy",
    "GDP_pct_change_12p": "gdp_yoy_proxy",
    "WILL5000INDFC_pct_change_12p": "wilshire_5000_yoy",
}


def debug_print(message: str) -> None:
    """Print and log the same message so debugging works in both terminal and logs."""
    print(f"[ML_DATASET DEBUG] {message}")
    logger.info(message)


def load_required_parquet(filename: str, run_hint: str) -> pd.DataFrame:
    """Load a required processed parquet file with a helpful error message."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {run_hint} first.")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{path} exists but is empty. Re-run {run_hint} and inspect the logs.")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    debug_print(f"Loaded {filename}: shape={df.shape}")
    return df


def load_optional_parquet(filename: str) -> pd.DataFrame:
    """Load an optional parquet file and return an empty DataFrame if unavailable."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        debug_print(f"Optional file missing: {filename}. The ML dataset will continue without it.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    debug_print(f"Loaded optional {filename}: shape={df.shape}")
    return df


def ensure_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML helper columns when existing pipeline names differ or are absent.

    The dashboard project has evolved across versions, so this function avoids
    brittle column assumptions. It never invents market data; it only derives
    features from columns that are already present in the scored signal table.
    """
    g = df.copy()

    if "adjusted_close" not in g.columns:
        raise ValueError("bubble_signal_scores.parquet must contain adjusted_close for target construction.")

    # A ratio version of the moving-average relationship is easier for models to
    # learn from than two raw moving average levels, especially across tickers
    # with very different price scales.
    if "sma_50_over_200" not in g.columns and {"sma_50", "sma_200"}.issubset(g.columns):
        g["sma_50_over_200"] = g["sma_50"] / g["sma_200"].replace(0, np.nan)

    # Some earlier project versions used return_63d/126d/252d but not return_21d.
    # The 21-trading-day return is a rough one-month momentum feature created
    # using only past prices.
    if "return_21d" not in g.columns:
        g["return_21d"] = g.groupby("ticker")["adjusted_close"].pct_change(21) * 100.0

    if "price_log" not in g.columns:
        g["price_log"] = np.log(g["adjusted_close"].where(g["adjusted_close"] > 0))

    # Standardize the acceleration column name expected by the prompt while
    # preserving the original project column.
    if "parabolic_acceleration_proxy" not in g.columns:
        if "parabolic_acceleration" in g.columns:
            g["parabolic_acceleration_proxy"] = g["parabolic_acceleration"]
        else:
            g["parabolic_acceleration_proxy"] = np.nan

    # Warning/recovery scores should exist after bubble_signals.py. If not, we
    # keep the columns with NaN rather than fabricating values.
    for col in ["warning_score", "recovery_score", "valuation_warning_score"]:
        if col not in g.columns:
            g[col] = np.nan

    return g


def resample_scored_signals_to_weekly(scored: pd.DataFrame) -> pd.DataFrame:
    """Convert daily ticker rows into one row per ticker per week.

    Why weekly? Financial crash labels are rare and noisy. Daily rows create a
    huge number of nearly duplicate observations, which can make the model look
    more confident than it really is. Weekly rows still preserve trend, valuation,
    RSI, and drawdown conditions while reducing autocorrelation.
    """
    required = {"date", "ticker", "adjusted_close"}
    missing = required.difference(scored.columns)
    if missing:
        raise ValueError(f"Scored signals are missing required columns: {sorted(missing)}")

    scored = ensure_engineered_columns(scored)
    frames: list[pd.DataFrame] = []

    for ticker, group in scored.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        g = g.drop_duplicates(subset=["date"], keep="last")
        g = g.set_index("date")

        # .last() takes the final observed market row in each week. If Friday is
        # a holiday, the row may be Thursday; the index label remains the week-end
        # Friday, which is useful for aligning assets consistently.
        weekly = g.resample("W-FRI").last().dropna(subset=["adjusted_close"])
        weekly = weekly.reset_index()
        weekly["ticker"] = ticker
        frames.append(weekly)
        debug_print(f"Weekly rows for {ticker}: daily={len(group):,}, weekly={len(weekly):,}")

    if not frames:
        raise ValueError("No weekly rows were created. Check ticker/date coverage in bubble_signal_scores.parquet.")

    weekly_all = pd.concat(frames, ignore_index=True)
    weekly_all = weekly_all.sort_values(["ticker", "date"]).reset_index(drop=True)
    debug_print(f"Weekly scored-signal dataset shape={weekly_all.shape}")
    return weekly_all


def build_macro_wide(macro_features: pd.DataFrame) -> pd.DataFrame:
    """Pivot long FRED macro features into a wide table suitable for ML.

    The raw macro table has one row per indicator/date. Machine-learning rows
    need one row per date with many macro columns. We pivot values and trend
    measures, then later as-of merge so each market week receives the latest
    macro observation available at or before that week.
    """
    if macro_features.empty:
        debug_print("No macro features available. The ML dataset will use market/valuation features only.")
        return pd.DataFrame()

    keep_metrics = ["value", "change_1p", "change_3p", "change_12p", "pct_change_12p"]
    available_metrics = [c for c in keep_metrics if c in macro_features.columns]
    if not available_metrics:
        debug_print("Macro features file has no recognized value/change columns.")
        return pd.DataFrame()

    macro = macro_features[["date", "indicator_id"] + available_metrics].copy()
    macro["indicator_id"] = macro["indicator_id"].astype(str)

    wide_parts: list[pd.DataFrame] = []
    for metric in available_metrics:
        pivot = macro.pivot_table(index="date", columns="indicator_id", values=metric, aggfunc="last")
        pivot.columns = [f"{indicator}_{metric}" for indicator in pivot.columns]
        wide_parts.append(pivot)

    wide = pd.concat(wide_parts, axis=1).sort_index().reset_index()
    wide = wide.rename(columns=MACRO_RENAME_MAP)
    debug_print(f"Macro wide feature table shape={wide.shape}")
    return wide


def normalize_datetime_column(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """Force a datetime column into one consistent pandas dtype.

    Pandas merge_asof is stricter than a normal merge: both join keys must have
    the exact same datetime unit. On some Windows/Python/pyarrow combinations,
    one parquet file may load as datetime64[ms] while another loads as
    datetime64[us]. Those both look like datetimes to humans, but pandas refuses
    to as-of merge them because the internal precision differs.

    We convert both sides to timezone-naive datetime64[ns], pandas' normal
    high-precision datetime representation, before the merge. This does not add
    future information. It only standardizes the storage precision of the same
    dates so the no-look-ahead macro merge can run safely.
    """
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")

    # If any timezone-aware datetimes sneak in from an API, remove the timezone.
    # The dashboard compares market and macro dates by calendar date, not by
    # intraday timestamps, so timezone-aware values would only create noisy joins.
    try:
        if getattr(out[column].dt, "tz", None) is not None:
            out[column] = out[column].dt.tz_localize(None)
    except AttributeError:
        # If the column is empty or not datetimelike after coercion, the validation
        # below will catch the problem.
        pass

    # This is the key compatibility fix for the user's pandas error:
    # dtype('<M8[ms]') and dtype('<M8[us]') must become the same dtype.
    out[column] = out[column].astype("datetime64[ns]")
    out = out.dropna(subset=[column])
    return out


def merge_macro_asof(weekly: pd.DataFrame, macro_wide: pd.DataFrame) -> pd.DataFrame:
    """Attach latest available macro row to each weekly market row without leakage."""
    if macro_wide.empty:
        debug_print("Macro feature table is empty. Continuing with market/signal features only.")
        return weekly

    # IMPORTANT COMPATIBILITY FIX:
    # merge_asof requires matching datetime dtypes on both sides. Normalizing here
    # prevents pandas from failing when parquet loads one date column as
    # datetime64[ms] and another as datetime64[us].
    left = normalize_datetime_column(weekly, "date").sort_values("date").copy()
    right = normalize_datetime_column(macro_wide, "date").sort_values("date").copy()

    debug_print(f"Weekly date dtype before as-of merge: {left['date'].dtype}")
    debug_print(f"Macro date dtype before as-of merge: {right['date'].dtype}")

    if left.empty:
        raise ValueError("Weekly feature table became empty after date normalization.")
    if right.empty:
        debug_print("Macro feature table became empty after date normalization. Returning weekly features only.")
        return weekly

    # merge_asof(direction='backward') is essential here. It means the ML row for
    # a market week can only see macro data dated on or before that week. It does
    # not peek at a later CPI, GDP, unemployment, or stress-index release.
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    debug_print(f"Weekly dataset after macro as-of merge shape={merged.shape}")
    return merged


def compute_forward_min_close(prices: pd.Series, horizon_weeks: int) -> pd.Series:
    """Return the lowest future close over the next N weekly rows.

    For row i, the function looks at rows i+1 through i+horizon. It intentionally
    excludes the current row so a current drawdown does not label itself as a
    future crash. Rows near the end without a complete future window receive NaN.
    """
    values = prices.to_numpy(dtype=float)
    future_min = np.full(len(values), np.nan)

    for i in range(len(values)):
        start = i + 1
        stop = i + horizon_weeks + 1
        if stop <= len(values):
            window = values[start:stop]
            if np.isfinite(window).any():
                future_min[i] = np.nanmin(window)

    return pd.Series(future_min, index=prices.index)


def add_forward_drawdown_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    """Create supervised labels from future drawdowns.

    This is the only place where future data is allowed. The future close is used
    to create historical labels so the model can learn. These target columns must
    never be used as input features.
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in weekly.groupby("ticker", sort=False):
        g = group.sort_values("date").copy().reset_index(drop=True)
        current_close = g["adjusted_close"].astype(float)

        for target_name, spec in TARGET_SPECS.items():
            horizon = int(spec["horizon_weeks"])
            threshold = float(spec["drawdown_threshold"])
            future_min = compute_forward_min_close(current_close, horizon_weeks=horizon)
            forward_drawdown = future_min / current_close - 1.0

            g[f"future_min_close_{target_name}"] = future_min
            g[f"future_drawdown_{target_name}"] = forward_drawdown * 100.0
            g[f"target_valid_{target_name}"] = forward_drawdown.notna()
            g[f"threshold_{target_name}"] = threshold
            g[target_name] = np.where(forward_drawdown.notna(), (forward_drawdown <= threshold).astype(int), np.nan)

        # Segment-adjusted targets use different drawdown thresholds by asset type.
        # This prevents the model from treating a normal crypto decline like an
        # extraordinary broad-index crash. These columns become the recommended
        # labels for rare-event modeling while the original uniform targets remain
        # available for sensitivity checks.
        segment = infer_asset_segment(str(ticker))
        g["asset_segment"] = segment
        for target_name, spec in SEGMENT_TARGET_SPECS.get(segment, SEGMENT_TARGET_SPECS["other_single_name"]).items():
            horizon = int(spec["horizon_weeks"])
            threshold = float(spec["drawdown_threshold"])
            future_min = compute_forward_min_close(current_close, horizon_weeks=horizon)
            forward_drawdown = future_min / current_close - 1.0

            g[f"future_min_close_{target_name}"] = future_min
            g[f"future_drawdown_{target_name}"] = forward_drawdown * 100.0
            g[f"target_valid_{target_name}"] = forward_drawdown.notna()
            g[f"threshold_{target_name}"] = threshold
            g[target_name] = np.where(forward_drawdown.notna(), (forward_drawdown <= threshold).astype(int), np.nan)

        # Discrete-time hazard label: did the asset ENTER a major drawdown state
        # over the next four weeks? This is different from the six-month burst
        # label because it focuses on event onset, not whether the future minimum
        # ever crosses a threshold.
        major_threshold_pct = SEGMENT_TARGET_SPECS.get(segment, SEGMENT_TARGET_SPECS["other_single_name"])["burst_6m_segment"]["drawdown_threshold"] * 100.0
        current_state = pd.to_numeric(g.get("drawdown_pct", np.nan), errors="coerce") <= major_threshold_pct
        onset = current_state & (~current_state.shift(1).fillna(False))
        future_onset = []
        for i in range(len(g)):
            window = onset.iloc[i + 1: i + 5]
            future_onset.append(np.nan if len(window) < 4 else int(window.any()))
        g["hazard_event_4w_segment"] = future_onset
        g["target_valid_hazard_event_4w_segment"] = pd.Series(future_onset).notna().to_numpy()

        frames.append(g)
        valid_6m = int(g["target_valid_burst_6m"].sum()) if "target_valid_burst_6m" in g.columns else 0
        positives_6m = int(g["burst_6m"].sum(skipna=True)) if "burst_6m" in g.columns else 0
        debug_print(f"Targets for {ticker}: valid_6m={valid_6m:,}, burst_6m_positives={positives_6m:,}")

    labeled = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    debug_print(f"ML dataset with targets shape={labeled.shape}")
    return labeled



def infer_asset_segment(ticker: str) -> str:
    """Add the same broad asset segment used by model_training.py.

    Keeping this label in the saved ML dataset makes the dashboard easier to
    interpret and lets later scripts train segment-specific models without
    guessing from scratch. It is not a future-looking label; it is static ticker
    metadata.
    """
    broad = {"SPY", "QQQ", "^GSPC", "^IXIC", "^DJI", "IWM", "XLK", "SOXX", "SMH", "XLF", "XHB", "EWJ", "FXI", "ASHR", "MCHI", "EWH", "XLE", "DBC", "GLD", "SLV", "USO", "ARKK", "IPO"}
    mega = {"NVDA", "MSFT", "AMD", "AVGO", "META", "GOOGL", "GOOG", "AMZN", "AAPL", "TSM", "ASML", "CSCO", "INTC", "ORCL", "QCOM", "IBM", "TXN", "MU", "LRCX", "KLAC", "ADBE", "CRM"}
    speculative = {"PLTR", "BTC-USD", "ETH-USD", "COIN", "TSLA", "GME", "AMC", "SPCE", "NKLA", "RIVN", "LCID", "ROKU", "ZM", "SHOP", "SNOW", "NET", "DDOG", "MDB", "CRWD", "U", "PYPL", "SQ", "XYZ"}
    if ticker in broad or ticker.startswith("^"):
        return "broad_index_etf"
    if ticker in mega:
        return "mega_cap_ai_tech"
    if ticker in speculative or ticker.endswith("-USD"):
        return "speculative_high_vol"
    return "other_single_name"

def validate_ml_dataset(df: pd.DataFrame) -> None:
    """Print data-quality diagnostics before saving the ML dataset."""
    debug_print("Validating ML dataset")

    duplicate_count = int(df.duplicated(subset=["ticker", "date"]).sum())
    debug_print(f"Duplicate ticker/date rows: {duplicate_count:,}")
    if duplicate_count > 0:
        examples = df[df.duplicated(subset=["ticker", "date"], keep=False)][["ticker", "date"]].head(10)
        debug_print(f"Duplicate examples:\n{examples.to_string(index=False)}")
        raise ValueError("Duplicate ticker/date rows found in ML dataset. Stop and inspect weekly resampling.")

    debug_print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    debug_print(f"Ticker count: {df['ticker'].nunique():,}")
    debug_print("Rows per ticker:\n" + df.groupby("ticker").size().sort_values(ascending=False).head(25).to_string())

    for target in TARGET_SPECS:
        if target in df.columns:
            valid = df[df[f"target_valid_{target}"] == True]
            counts = valid[target].value_counts(dropna=False).sort_index()
            debug_print(f"Class balance for {target} among valid rows:\n{counts.to_string()}")
            if len(counts) < 2:
                debug_print(f"WARNING: {target} has only one class in valid rows. Some ML metrics will be unavailable.")

    # Missingness is not automatically fatal because real financial data is
    # uneven, especially valuation data. The model pipeline uses median imputation.
    missing_summary = df.isna().mean().sort_values(ascending=False).head(30) * 100.0
    debug_print("Top missing-value percentages:\n" + missing_summary.to_string())



def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean mixed Python/object columns before saving to Parquet.

    Why this exists:
    ----------------
    PyArrow, the engine pandas uses to write Parquet, needs each column to have
    one stable data type. During this project, some indicator columns can become
    mixed object columns because one upstream script represents a signal as True/False
    while another represents the same signal as 1.0/0.0 after missing-value handling.

    Humans see these as the same binary feature, but PyArrow sees a column that is
    partly boolean and partly float and refuses to guess. We fix that explicitly:

    - Boolean-like signal columns become numeric 1.0/0.0. This keeps them usable
      for scikit-learn and XGBoost.
    - Datetime columns are standardized to datetime64[ns]. This prevents the
      merge_asof datetime precision bugs that appeared elsewhere in the pipeline.
    - Remaining object columns are kept as strings only when they are categorical
      identifiers such as ticker or asset_segment.

    This function does not change the economic meaning of the data. It only makes
    the saved dataset type-stable so the ML pipeline can read it consistently.
    """
    clean = df.copy()

    for col in clean.columns:
        series = clean[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            clean[col] = pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")
            continue

        if pd.api.types.is_bool_dtype(series):
            clean[col] = series.astype(float)
            continue

        if series.dtype == "object":
            non_null = series.dropna()

            # If an object column only contains booleans and numeric 0/1 values,
            # make it a numeric feature. This directly fixes columns like
            # golden_cross and death_cross when they contain True/False plus 0.0.
            if not non_null.empty:
                normalized_values = set()
                for value in non_null.unique():
                    if isinstance(value, (bool, np.bool_)):
                        normalized_values.add(int(value))
                    elif isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                        if float(value) in (0.0, 1.0):
                            normalized_values.add(int(float(value)))
                        else:
                            normalized_values.add("non_binary_number")
                    elif isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1", "0.0", "1.0"}:
                        raw = value.strip().lower()
                        normalized_values.add(1 if raw == "true" or raw in {"1", "1.0"} else 0)
                    else:
                        normalized_values.add("non_binary_object")

                if normalized_values.issubset({0, 1}):
                    clean[col] = series.map(
                        lambda x: np.nan if pd.isna(x) else (
                            1.0 if (x is True or str(x).strip().lower() in {"true", "1", "1.0"}) else 0.0
                        )
                    ).astype(float)
                    continue

            # Known identifier/categorical columns should remain strings. They are
            # not used directly as numeric model features, but they are necessary
            # for filtering, dashboard display, and segment-specific training.
            clean[col] = series.astype("string")

    return clean

def save_dataset(df: pd.DataFrame) -> None:
    """Persist the ML dataset to parquet and DuckDB after dtype cleanup."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned = sanitize_for_parquet(df)

    # Print the remaining object/string columns so debugging is transparent.
    # Numeric model features should be numeric at this point; string columns should
    # mostly be identifiers such as ticker, asset name, or asset segment.
    remaining_object_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
    debug_print(f"Object/string columns before parquet save: {remaining_object_cols}")

    cleaned.to_parquet(OUTPUT_PATH, index=False)

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE ml_burst_dataset AS SELECT * FROM cleaned")
        con.execute("CHECKPOINT")
    finally:
        con.close()

    debug_print(f"Saved ML dataset to {OUTPUT_PATH}; shape={cleaned.shape}")


def main() -> None:
    debug_print("ML dataset build start")
    scored = load_required_parquet("bubble_signal_scores.parquet", "python bubble_signals.py")
    macro_features = load_optional_parquet("macro_features.parquet")

    weekly = resample_scored_signals_to_weekly(scored)
    macro_wide = build_macro_wide(macro_features)
    weekly_with_macro = merge_macro_asof(weekly, macro_wide)
    labeled = add_forward_drawdown_targets(weekly_with_macro)
    # add_forward_drawdown_targets already assigns asset_segment before creating
    # segment-adjusted targets; this line is kept as a safety refresh in case a
    # future edit creates rows without the segment column.
    labeled["asset_segment"] = labeled["ticker"].astype(str).map(infer_asset_segment)
    debug_print("ML rows by asset segment:\n" + labeled["asset_segment"].value_counts().to_string())
    validate_ml_dataset(labeled)
    save_dataset(labeled)
    debug_print("ML dataset build complete")


if __name__ == "__main__":
    main()
