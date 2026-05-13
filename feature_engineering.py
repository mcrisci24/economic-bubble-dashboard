"""
Feature engineering for bubble vital signs.

Run:
    python feature_engineering.py

Inputs:
    data/processed/asset_prices.parquet
    data/processed/macro_indicators.parquet
    data/processed/valuation_metrics.parquet

Outputs:
    data/processed/technical_signals.parquet
    data/processed/macro_features.parquet
    data/processed/valuation_features.parquet
    DuckDB tables technical_signals, macro_features, and valuation_features
"""
from __future__ import annotations

import logging
import sys

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH, LOG_DIR, PROCESSED_DIR, TRADING_DAYS_PER_YEAR

LOG_FILE = LOG_DIR / "feature_engineering.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_engineering")


def debug_print(message: str) -> None:
    print(f"[FEATURE_ENGINEERING DEBUG] {message}")
    logger.info(message)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    price_path = PROCESSED_DIR / "asset_prices.parquet"
    macro_path = PROCESSED_DIR / "macro_indicators.parquet"
    if not price_path.exists():
        raise FileNotFoundError(f"Missing {price_path}. Run python data_ingestion.py first.")
    if not macro_path.exists():
        raise FileNotFoundError(f"Missing {macro_path}. Run python data_ingestion.py first.")
    prices = pd.read_parquet(price_path)
    macro = pd.read_parquet(macro_path)
    prices["date"] = pd.to_datetime(prices["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    debug_print(f"Loaded prices={prices.shape}, macro={macro.shape}")
    return prices, macro


def load_valuation_metrics() -> pd.DataFrame:
    """Load SEC historical valuation metrics if available."""
    valuation_path = PROCESSED_DIR / "valuation_metrics.parquet"
    if not valuation_path.exists():
        debug_print("No valuation_metrics.parquet found. Valuation features will be empty.")
        return pd.DataFrame()
    valuations = pd.read_parquet(valuation_path)
    if valuations.empty:
        debug_print("valuation_metrics.parquet exists but is empty.")
        return valuations
    valuations["date"] = pd.to_datetime(valuations["date"])
    for col in ["filing_date", "period_end", "price_date"]:
        if col in valuations.columns:
            valuations[col] = pd.to_datetime(valuations[col], errors="coerce")
    debug_print(f"Loaded valuations={valuations.shape}")
    return valuations


def build_valuation_features(valuations: pd.DataFrame) -> pd.DataFrame:
    """Create valuation vital signs from point-in-time SEC valuation rows.

    Expanding medians are shifted one row so the current filing cannot become
    its own historical benchmark. That keeps the signal logic honest.
    """
    if valuations.empty:
        return valuations
    debug_print("Computing valuation features from SEC historical metrics")
    frames = []
    for ticker, group in valuations.groupby("ticker"):
        g = group.sort_values("date").copy()
        for col in ["price_to_sales", "pe_ratio", "ev_to_sales", "revenue_ttm", "net_income_ttm"]:
            if col not in g.columns:
                g[col] = np.nan
            g[col] = pd.to_numeric(g[col], errors="coerce")

        g["ps_historical_median"] = g["price_to_sales"].expanding(min_periods=4).median().shift(1)
        g["pe_historical_median"] = g["pe_ratio"].where(g["pe_ratio"] > 0).expanding(min_periods=4).median().shift(1)
        g["ev_sales_historical_median"] = g["ev_to_sales"].expanding(min_periods=4).median().shift(1)

        g["ps_to_historical_median"] = g["price_to_sales"] / g["ps_historical_median"].replace(0, np.nan)
        g["pe_to_historical_median"] = g["pe_ratio"] / g["pe_historical_median"].replace(0, np.nan)
        g["ev_sales_to_historical_median"] = g["ev_to_sales"] / g["ev_sales_historical_median"].replace(0, np.nan)

        # Filing-row growth measures, based only on previously filed rows.
        g["revenue_ttm_yoy_pct"] = g["revenue_ttm"].pct_change(4) * 100.0
        g["net_income_ttm_yoy_pct"] = g["net_income_ttm"].pct_change(4) * 100.0
        g["valuation_warning_ps_2x"] = g["ps_to_historical_median"] >= 2.0
        g["valuation_warning_ps_3x"] = g["ps_to_historical_median"] >= 3.0
        g["valuation_warning_ev_sales_2x"] = g["ev_sales_to_historical_median"] >= 2.0
        g["valuation_warning_pe_high"] = g["pe_ratio"] >= 80
        g["valuation_warning_growth_gap"] = (g["ps_to_historical_median"] >= 2.0) & (g["revenue_ttm_yoy_pct"] < 10)

        g["valuation_warning_score"] = (
            g["valuation_warning_ps_2x"].fillna(False).astype(int) * 20
            + g["valuation_warning_ps_3x"].fillna(False).astype(int) * 20
            + g["valuation_warning_ev_sales_2x"].fillna(False).astype(int) * 15
            + g["valuation_warning_pe_high"].fillna(False).astype(int) * 15
            + g["valuation_warning_growth_gap"].fillna(False).astype(int) * 20
        ).clip(0, 100)
        frames.append(g)

    features = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    debug_print(f"Valuation features shape={features.shape}")
    return features


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI using exponential smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_weekly_rsi(daily_df: pd.DataFrame) -> pd.Series:
    """Compute RSI from weekly closes, then forward-fill back to daily rows."""
    df = daily_df[["date", "adjusted_close"]].dropna().copy().set_index("date")
    if df.empty:
        return pd.Series(index=daily_df.index, dtype=float)
    weekly_close = df["adjusted_close"].resample("W-FRI").last().dropna()
    weekly_rsi = compute_rsi(weekly_close, window=14)
    daily_rsi = weekly_rsi.reindex(df.index, method="ffill")
    return daily_rsi.reindex(daily_df["date"]).to_numpy()


def compute_asset_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for one ticker with no look-ahead."""
    g = group.sort_values("date").copy()
    price = g["adjusted_close"].astype(float)
    returns = price.pct_change()

    g["daily_return"] = returns
    g["sma_50"] = price.rolling(50, min_periods=20).mean()
    g["sma_200"] = price.rolling(200, min_periods=80).mean()
    g["sma_50_prev"] = g["sma_50"].shift(1)
    g["sma_200_prev"] = g["sma_200"].shift(1)
    g["golden_cross"] = (g["sma_50_prev"] <= g["sma_200_prev"]) & (g["sma_50"] > g["sma_200"])
    g["death_cross"] = (g["sma_50_prev"] >= g["sma_200_prev"]) & (g["sma_50"] < g["sma_200"])

    cumulative_peak = price.cummax()
    g["drawdown_pct"] = (price / cumulative_peak - 1.0) * 100.0
    g["distance_from_200dma"] = (price / g["sma_200"] - 1.0) * 100.0

    rolling_mean = price.rolling(252, min_periods=80).mean()
    rolling_std = price.rolling(252, min_periods=80).std()
    g["zscore_price"] = (price - rolling_mean) / rolling_std.replace(0, np.nan)
    g["volatility_30d"] = returns.rolling(30, min_periods=15).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    g["return_63d"] = price.pct_change(63) * 100.0
    g["return_126d"] = price.pct_change(126) * 100.0
    g["return_252d"] = price.pct_change(252) * 100.0
    g["parabolic_acceleration"] = g["return_63d"] - (g["return_252d"] / 4.0)

    g["daily_rsi"] = compute_rsi(price, window=14)
    g["weekly_rsi"] = compute_weekly_rsi(g)
    g["weekly_rsi_prev"] = pd.Series(g["weekly_rsi"]).shift(1).to_numpy()
    g["rsi_euphoria_break"] = (g["weekly_rsi_prev"] > 75) & (g["weekly_rsi"] < 70)
    g["rsi_recovery"] = (g["weekly_rsi_prev"] < 40) & (g["weekly_rsi"] >= 40)
    g["major_drawdown_40"] = g["drawdown_pct"] <= -40
    g["major_drawdown_60"] = g["drawdown_pct"] <= -60
    g["reclaim_200dma"] = (price.shift(1) <= g["sma_200"].shift(1)) & (price > g["sma_200"])

    # Keep raw booleans as bool for backtester and dashboard filtering.
    return g


def build_technical_signals(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "adjusted_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Prices missing required columns: {missing}")

    debug_print("Computing technical indicators by ticker")

    # Do NOT use DataFrameGroupBy.apply(include_groups=True).
    # pandas 3.x removed include_groups=True, which caused the pipeline to fail on
    # modern Python/PyCharm environments. This explicit loop is more stable, easier
    # to debug, and preserves the ticker column for downstream joins/dashboard pages.
    frames: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker", sort=False):
        debug_print(f"Computing technical indicators for {ticker}: rows={len(group)}")
        engineered = compute_asset_features(group)
        engineered["ticker"] = ticker
        frames.append(engineered)

    if not frames:
        raise ValueError("No ticker groups were available to compute technical signals.")

    signals = pd.concat(frames, ignore_index=True)
    signals = signals.sort_values(["ticker", "date"]).reset_index(drop=True)
    debug_print(f"Technical signals shape={signals.shape}")
    return signals


def build_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Create macro feature table with YoY changes and short trend measures by indicator."""
    debug_print("Computing macro feature trends")
    rows = []
    for indicator_id, group in macro.groupby("indicator_id"):
        g = group.sort_values("date").copy()
        g["value_lag_1"] = g["value"].shift(1)
        g["value_lag_3"] = g["value"].shift(3)
        g["value_lag_12"] = g["value"].shift(12)
        g["change_1p"] = g["value"] - g["value_lag_1"]
        g["change_3p"] = g["value"] - g["value_lag_3"]
        g["change_12p"] = g["value"] - g["value_lag_12"]
        g["pct_change_12p"] = (g["value"] / g["value_lag_12"] - 1.0) * 100.0
        rows.append(g)
    features = pd.concat(rows, ignore_index=True).sort_values(["indicator_id", "date"])
    debug_print(f"Macro features shape={features.shape}")
    return features


def save_outputs(technical: pd.DataFrame, macro_features: pd.DataFrame, valuation_features: pd.DataFrame) -> None:
    technical_path = PROCESSED_DIR / "technical_signals.parquet"
    macro_features_path = PROCESSED_DIR / "macro_features.parquet"
    valuation_features_path = PROCESSED_DIR / "valuation_features.parquet"
    technical.to_parquet(technical_path, index=False)
    macro_features.to_parquet(macro_features_path, index=False)
    valuation_features.to_parquet(valuation_features_path, index=False)

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE technical_signals AS SELECT * FROM technical")
        con.execute("CREATE OR REPLACE TABLE macro_features AS SELECT * FROM macro_features")
        con.execute("CREATE OR REPLACE TABLE valuation_features AS SELECT * FROM valuation_features")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    debug_print("Feature engineering outputs saved.")


def main() -> None:
    debug_print("Feature engineering pipeline start")
    prices, macro = load_inputs()
    valuations = load_valuation_metrics()
    technical = build_technical_signals(prices)
    macro_features = build_macro_features(macro)
    valuation_features = build_valuation_features(valuations)
    save_outputs(technical, macro_features, valuation_features)


if __name__ == "__main__":
    main()
