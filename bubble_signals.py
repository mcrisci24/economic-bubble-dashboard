"""
Bubble signal rules and phase classification.

Run:
    python bubble_signals.py

Outputs:
    data/processed/bubble_signal_scores.parquet
    data/processed/current_ai_monitor.parquet

Rule-based, transparent scoring is used because this is easier to audit than a black-box model.
"""
from __future__ import annotations

import logging
import sys

import duckdb
import numpy as np
import pandas as pd

from config import AI_LEADERS, DB_PATH, LOG_DIR, PROCESSED_DIR

LOG_FILE = LOG_DIR / "bubble_signals.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bubble_signals")


def debug_print(message: str) -> None:
    print(f"[BUBBLE_SIGNALS DEBUG] {message}")
    logger.info(message)


def normalize_datetime_ns(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """Normalize a datetime column to pandas datetime64[ns].

    Pandas merge_asof is intentionally strict: both join keys must have the
    exact same internal datetime precision. Parquet/yfinance/SEC data can load
    as datetime64[ms], datetime64[us], datetime64[s], or datetime64[ns] on
    different machines. They all look like ordinary dates, but merge_asof will
    reject them unless the dtypes match exactly.

    This helper standardizes date columns before no-look-ahead as-of merges.
    It does not change the date meaning; it only fixes the storage precision.
    """
    out = df.copy()
    if column not in out.columns:
        return out

    out[column] = pd.to_datetime(out[column], errors="coerce")

    try:
        if getattr(out[column].dt, "tz", None) is not None:
            out[column] = out[column].dt.tz_localize(None)
    except AttributeError:
        pass

    out[column] = out[column].astype("datetime64[ns]")
    out = out.dropna(subset=[column])
    return out


def load_technical() -> pd.DataFrame:
    path = PROCESSED_DIR / "technical_signals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python feature_engineering.py first.")
    df = pd.read_parquet(path)
    df = normalize_datetime_ns(df, "date")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)

def load_valuation_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "valuation_features.parquet"
    if not path.exists():
        debug_print("No valuation_features.parquet found. Valuation scoring will use technical signals only.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        debug_print("valuation_features.parquet is empty. Valuation scoring will use technical signals only.")
        return df
    df = normalize_datetime_ns(df, "date")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)

def merge_valuation_asof(technical: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent filed valuation row to each daily technical row.

    The valuation rows were dated to the first trading day after the filing date,
    so an as-of merge does not leak future fundamentals into prior dates.

    We normalize datetime precision before merging because pandas merge_asof
    rejects joins where one side is datetime64[ms] and the other is datetime64[ns].
    """
    if valuations.empty:
        return technical

    technical = normalize_datetime_ns(technical, "date")
    valuations = normalize_datetime_ns(valuations, "date")

    value_cols = [
        "date",
        "ticker",
        "filing_date",
        "period_end",
        "market_cap",
        "enterprise_value",
        "revenue_ttm",
        "net_income_ttm",
        "price_to_sales",
        "pe_ratio",
        "ev_to_sales",
        "ps_historical_median",
        "pe_historical_median",
        "ev_sales_historical_median",
        "ps_to_historical_median",
        "pe_to_historical_median",
        "ev_sales_to_historical_median",
        "revenue_ttm_yoy_pct",
        "valuation_warning_score",
        "valuation_warning_ps_2x",
        "valuation_warning_ps_3x",
        "valuation_warning_ev_sales_2x",
        "valuation_warning_pe_high",
        "valuation_warning_growth_gap",
    ]
    value_cols = [c for c in value_cols if c in valuations.columns]

    frames = []
    for ticker, g in technical.groupby("ticker", sort=False):
        left = normalize_datetime_ns(g, "date").sort_values("date").copy()
        right = valuations[valuations["ticker"].eq(ticker)][value_cols].copy()
        right = normalize_datetime_ns(right, "date").sort_values("date")

        if right.empty:
            frames.append(left)
            continue

        debug_print(
            f"Valuation as-of merge for {ticker}: "
            f"technical_date_dtype={left['date'].dtype}, "
            f"valuation_date_dtype={right['date'].dtype}, "
            f"technical_rows={len(left)}, valuation_rows={len(right)}"
        )

        # We are already inside one ticker at a time, so merging on date only
        # avoids pandas `by=` edge cases while still preserving ticker identity.
        merged = pd.merge_asof(left, right, on="date", direction="backward", suffixes=("", "_valuation"))

        if "ticker_valuation" in merged.columns:
            merged = merged.drop(columns=["ticker_valuation"])

        frames.append(merged)

    merged_all = pd.concat(frames, ignore_index=True) if frames else technical
    debug_print(f"Merged valuation features into daily signals: {merged_all.shape}")
    return merged_all.sort_values(["ticker", "date"]).reset_index(drop=True)

def add_signal_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create transparent warning/recovery scores from technical indicators only.

    All rules use current or past values. No future values are used.
    """
    g = df.copy()

    # Warning components. These are intentionally simple, inspectable rules.
    g["warn_dist_200_extreme"] = g["distance_from_200dma"] > 50
    g["warn_dist_200_high"] = g["distance_from_200dma"] > 25
    g["warn_weekly_rsi_hot"] = g["weekly_rsi"] > 75
    g["warn_price_zscore_hot"] = g["zscore_price"] > 2.5
    g["warn_parabolic"] = g["parabolic_acceleration"] > 30
    g["warn_volatility_high"] = g["volatility_30d"] > g.groupby("ticker")["volatility_30d"].transform(
        lambda s: s.rolling(756, min_periods=120).quantile(0.85)
    )
    g["warn_euphoria_break"] = g["rsi_euphoria_break"].fillna(False)
    g["warn_death_cross"] = g["death_cross"].fillna(False)
    g["warn_20pct_peak_break"] = g["drawdown_pct"] <= -20

    # Valuation components are present only for SEC-covered companies. ETFs, indexes,
    # crypto, and missing EDGAR histories simply receive zero valuation contribution.
    for col in [
        "valuation_warning_ps_2x",
        "valuation_warning_ps_3x",
        "valuation_warning_ev_sales_2x",
        "valuation_warning_pe_high",
        "valuation_warning_growth_gap",
    ]:
        if col not in g.columns:
            g[col] = False

    g["warning_score"] = (
        g["warn_dist_200_extreme"].astype(int) * 16
        + g["warn_dist_200_high"].astype(int) * 8
        + g["warn_weekly_rsi_hot"].astype(int) * 12
        + g["warn_price_zscore_hot"].astype(int) * 12
        + g["warn_parabolic"].astype(int) * 8
        + g["warn_volatility_high"].fillna(False).astype(int) * 8
        + g["warn_euphoria_break"].astype(int) * 8
        + g["warn_death_cross"].astype(int) * 12
        + g["warn_20pct_peak_break"].astype(int) * 8
        + g["valuation_warning_ps_2x"].fillna(False).astype(int) * 8
        + g["valuation_warning_ps_3x"].fillna(False).astype(int) * 10
        + g["valuation_warning_ev_sales_2x"].fillna(False).astype(int) * 7
        + g["valuation_warning_pe_high"].fillna(False).astype(int) * 5
        + g["valuation_warning_growth_gap"].fillna(False).astype(int) * 10
    ).clip(0, 100)

    # Recovery components.
    g["rec_rsi_recovery"] = g["rsi_recovery"].fillna(False)
    g["rec_reclaim_200dma"] = g["reclaim_200dma"].fillna(False)
    g["rec_above_200dma"] = g["adjusted_close"] > g["sma_200"]
    g["rec_drawdown_stabilizing"] = (g["drawdown_pct"] > g.groupby("ticker")["drawdown_pct"].shift(20)) & (
        g["drawdown_pct"] < -20
    )
    g["rec_positive_63d"] = g["return_63d"] > 0
    g["rec_golden_cross"] = g["golden_cross"].fillna(False)

    g["recovery_score"] = (
        g["rec_rsi_recovery"].astype(int) * 20
        + g["rec_reclaim_200dma"].astype(int) * 25
        + g["rec_above_200dma"].fillna(False).astype(int) * 15
        + g["rec_drawdown_stabilizing"].fillna(False).astype(int) * 15
        + g["rec_positive_63d"].fillna(False).astype(int) * 10
        + g["rec_golden_cross"].astype(int) * 15
    ).clip(0, 100)

    g["exit_signal"] = (
        (g["warning_score"] >= 55)
        | g["warn_euphoria_break"]
        | ((g["warn_20pct_peak_break"]) & (g["warning_score"] >= 35))
        | g["warn_death_cross"]
    )
    g["reentry_signal"] = (
        (g["recovery_score"] >= 45)
        | g["rec_reclaim_200dma"]
        | (g["rec_rsi_recovery"] & (g["drawdown_pct"] <= -30))
    )

    g["signal_label"] = np.select(
        [
            g["exit_signal"] & (g["drawdown_pct"] > -20),
            g["exit_signal"] & (g["drawdown_pct"] <= -20),
            g["reentry_signal"],
            g["warning_score"] >= 50,
            g["recovery_score"] >= 45,
        ],
        [
            "Exit / Reduce Exposure Warning",
            "Breakdown / Risk Control",
            "Re-Entry / Recovery Confirmation",
            "Heating Up Warning",
            "Recovery Watch",
        ],
        default="Neutral / Monitor",
    )
    g["phase_label"] = classify_phase(g)
    return g


def classify_phase(g: pd.DataFrame) -> pd.Series:
    """Map vital signs into readable bubble phase labels."""
    return pd.Series(
        np.select(
            [
                (g["drawdown_pct"] <= -60),
                (g["drawdown_pct"] <= -35) & (g["weekly_rsi"] < 40),
                (g["drawdown_pct"] <= -20) & (g["adjusted_close"] < g["sma_200"]),
                (g["distance_from_200dma"] > 50) & (g["weekly_rsi"] > 75),
                (g["return_126d"] > 50) & (g["distance_from_200dma"] > 25),
                (g["return_252d"] > 20) & (g["adjusted_close"] > g["sma_200"]),
                (g["reentry_signal"] if "reentry_signal" in g.columns else False),
            ],
            [
                "Capitulation",
                "Crash / Deep Drawdown",
                "Breakdown",
                "Euphoria",
                "Acceleration",
                "Accumulation / Early Growth",
                "Recovery",
            ],
            default="Normal Growth / Monitor",
        ),
        index=g.index,
    )


def build_current_ai_monitor(scored: pd.DataFrame) -> pd.DataFrame:
    """Latest status for current AI-exposed tickers."""
    latest = (
        scored[scored["ticker"].isin(AI_LEADERS)]
        .sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )
    if latest.empty:
        return pd.DataFrame()

    latest["status"] = np.select(
        [
            latest["phase_label"].eq("Euphoria"),
            latest["phase_label"].eq("Acceleration"),
            latest["exit_signal"],
            latest["phase_label"].isin(["Breakdown", "Crash / Deep Drawdown"]),
            latest["phase_label"].eq("Capitulation"),
            latest["reentry_signal"],
            latest["warning_score"].between(35, 54),
        ],
        [
            "Euphoria",
            "Heating Up",
            "Profit-Taking / Breakdown Risk",
            "Breakdown",
            "Capitulation",
            "Recovery",
            "Heating Up",
        ],
        default="Normal Growth",
    )
    keep = [
        "date",
        "ticker",
        "adjusted_close",
        "drawdown_pct",
        "distance_from_200dma",
        "weekly_rsi",
        "volatility_30d",
        "price_to_sales",
        "ps_to_historical_median",
        "ev_to_sales",
        "revenue_ttm_yoy_pct",
        "valuation_warning_score",
        "warning_score",
        "recovery_score",
        "phase_label",
        "status",
        "signal_label",
    ]
    keep = [c for c in keep if c in latest.columns]
    return latest[keep].sort_values("warning_score", ascending=False)


def save_outputs(scored: pd.DataFrame, ai_monitor: pd.DataFrame) -> None:
    scored_path = PROCESSED_DIR / "bubble_signal_scores.parquet"
    ai_path = PROCESSED_DIR / "current_ai_monitor.parquet"
    scored.to_parquet(scored_path, index=False)
    ai_monitor.to_parquet(ai_path, index=False)

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE bubble_signal_scores AS SELECT * FROM scored")
        con.execute("CREATE OR REPLACE TABLE current_ai_monitor AS SELECT * FROM ai_monitor")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    debug_print("Signal outputs saved.")


def main() -> None:
    debug_print("Signal scoring start")
    technical = load_technical()
    valuations = load_valuation_features()
    technical_with_valuation = merge_valuation_asof(technical, valuations)
    scored = add_signal_scores(technical_with_valuation)
    ai_monitor = build_current_ai_monitor(scored)
    debug_print(f"Scored rows={len(scored):,}; AI monitor rows={len(ai_monitor):,}")
    save_outputs(scored, ai_monitor)


if __name__ == "__main__":
    main()
