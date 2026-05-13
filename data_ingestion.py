"""
Data ingestion layer.

Run:
    python data_ingestion.py

Outputs:
    data/processed/asset_prices.parquet
    data/processed/macro_indicators.parquet
    data/processed/valuation_metrics.parquet        # SEC historical point-in-time fundamentals
    data/processed/current_valuation_snapshot.parquet # latest yfinance context only
    data/bubble_dashboard.duckdb

No investment advice. Research/education only.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    ASSET_METADATA,
    BUBBLES,
    CORE_ASSETS,
    DB_PATH,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    FRED_SERIES,
    LOG_DIR,
    PROCESSED_DIR,
)

from sec_fundamentals import empty_valuation_frame, fetch_sec_historical_valuation_metrics

LOG_FILE = LOG_DIR / "data_ingestion.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_ingestion")


def debug_print(message: str) -> None:
    """Print and log a debug-style message so failures are visible in terminals and logs."""
    print(f"[DATA_INGESTION DEBUG] {message}")
    logger.info(message)


def unique_tickers() -> list[str]:
    """Collect tickers from all configured asset groups and bubble definitions."""
    tickers: set[str] = set()
    for group in CORE_ASSETS.values():
        tickers.update(group)
    for bubble in BUBBLES.values():
        tickers.update(bubble.get("tickers", []))
    return sorted(tickers)


def identify_bubble_period(date_value: pd.Timestamp, ticker: str) -> str | None:
    """Return a pipe-separated bubble label if ticker/date belongs to one or more configured bubble windows."""
    labels: list[str] = []
    for name, meta in BUBBLES.items():
        if ticker not in meta.get("tickers", []):
            continue
        start = pd.Timestamp(meta["start"])
        end = pd.Timestamp(meta["end"]) if meta.get("end") else pd.Timestamp.today().normalize()
        if start <= date_value <= end:
            labels.append(name)
    return " | ".join(labels) if labels else None


def _flatten_single_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output for a single ticker.

    yfinance versions differ: some return plain columns and newer versions can
    return a MultiIndex even for a single ticker. This function handles both so
    the pipeline does not break on harmless yfinance shape changes.
    """
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # Handle yfinance MultiIndex columns, usually like (Price, Ticker) or
    # (Ticker, Price). For a one-ticker download, we keep the price field level.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            level_values = [str(v).upper() for v in df.columns.get_level_values(-1)]
            if ticker.upper() in level_values:
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(-1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not clean MultiIndex columns for %s: %s", ticker, exc)
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "adj_close": "adjusted_close",
        "datetime": "date",
        "price_date": "date",
    }
    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        debug_print(f"Ticker {ticker} returned no date column. Columns: {list(df.columns)}")
        return pd.DataFrame()

    expected = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # If adjusted_close is missing but close is available, use close as the
    # fallback so older indexes/futures do not disappear unnecessarily.
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adjusted_close"] = df["adjusted_close"].fillna(df["close"])

    asset_name, asset_class = ASSET_METADATA.get(ticker, (ticker, "Security"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"])
    df["asset_id"] = ticker.replace("^", "INDEX_").replace("-", "_").replace("=", "_")
    df["ticker"] = ticker
    df["asset_name"] = asset_name
    df["asset_class"] = asset_class
    df["bubble_period"] = df["date"].apply(lambda d: identify_bubble_period(pd.Timestamp(d), ticker))
    df["source"] = "yfinance"

    ordered = [
        "date",
        "asset_id",
        "ticker",
        "asset_name",
        "asset_class",
        "bubble_period",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
    ]
    return df[ordered]


def fetch_yfinance_prices(
    tickers: Iterable[str],
    start: str = DEFAULT_START_DATE,
    end: str | None = DEFAULT_END_DATE,
    pause_seconds: float = 0.25,
) -> pd.DataFrame:
    """Fetch daily OHLCV data from yfinance one ticker at a time for safer error handling."""
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    tickers = list(dict.fromkeys(tickers))
    debug_print(f"Fetching {len(tickers)} tickers from yfinance. Start={start}, End={end or 'latest'}")

    for i, ticker in enumerate(tickers, start=1):
        try:
            debug_print(f"[{i}/{len(tickers)}] Downloading {ticker}")
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            df = _flatten_single_download(raw, ticker)
            if df.empty:
                failures.append(ticker)
                logger.warning("No data returned for ticker=%s", ticker)
                continue
            # Drop rows where the usable price is missing.
            df = df.dropna(subset=["adjusted_close"], how="all")
            if df.empty:
                failures.append(ticker)
                logger.warning("Adjusted close missing for ticker=%s", ticker)
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001 - ingestion should continue across ticker failures
            failures.append(ticker)
            logger.exception("Failed to fetch ticker=%s: %s", ticker, exc)
        time.sleep(pause_seconds)

    if failures:
        debug_print(
            "Ticker failures or unavailable/delisted symbols skipped: " + ", ".join(sorted(set(failures)))
        )

    if not frames:
        raise RuntimeError("No market data downloaded. Check internet connection, yfinance availability, and tickers.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    validate_price_data(prices)
    return prices


def validate_price_data(prices: pd.DataFrame) -> None:
    """Basic integrity checks for normalized price data."""
    required = {"date", "ticker", "adjusted_close", "source"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Price data missing required columns: {missing}")
    if prices.empty:
        raise ValueError("Price data is empty.")
    if prices["adjusted_close"].isna().mean() > 0.25:
        logger.warning("More than 25%% of adjusted_close values are missing.")
    duplicate_count = prices.duplicated(["ticker", "date"]).sum()
    if duplicate_count:
        raise ValueError(f"Found duplicate ticker/date rows in price data: {duplicate_count}")
    debug_print(f"Validated price data: {len(prices):,} rows, {prices['ticker'].nunique():,} tickers")


def _fetch_fred_csv_series(series_id: str) -> pd.DataFrame:
    """Fetch one FRED series through the public FRED graph CSV endpoint.

    This avoids pandas_datareader, which currently imports distutils and breaks
    under Python 3.12+. The CSV endpoint does not require a key for public series.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw = pd.read_csv(BytesIO(response.content))
    if raw.empty:
        return pd.DataFrame()
    raw.columns = [str(c).strip() for c in raw.columns]
    if "observation_date" in raw.columns:
        raw = raw.rename(columns={"observation_date": "date"})
    elif "DATE" in raw.columns:
        raw = raw.rename(columns={"DATE": "date"})
    if series_id not in raw.columns:
        # FRED normally names the value column as the series id. If not, use the
        # first non-date column and log the adjustment.
        value_candidates = [c for c in raw.columns if c.lower() != "date"]
        if not value_candidates:
            return pd.DataFrame()
        logger.warning("FRED %s returned value column %s instead of series id", series_id, value_candidates[0])
        raw = raw.rename(columns={value_candidates[0]: series_id})
    return raw[["date", series_id]]


def fetch_fred_macro(start: str = DEFAULT_START_DATE, end: str | None = DEFAULT_END_DATE) -> pd.DataFrame:
    """Fetch macro indicators from FRED without pandas_datareader.

    Python 3.12 removed distutils from the standard library, and current
    pandas_datareader releases can still import distutils internally. To keep
    this project Python-3.12-safe, this function pulls public FRED CSV data
    directly and then normalizes it into the project schema.
    """
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end or datetime.today().strftime("%Y-%m-%d"))

    debug_print(f"Fetching FRED macro data from {start_ts.date()} to {end_ts.date()}")
    for series_id, meta in FRED_SERIES.items():
        try:
            debug_print(f"Fetching FRED series {series_id}: {meta['name']}")
            raw = _fetch_fred_csv_series(series_id)
            if raw.empty:
                failures.append(series_id)
                continue
            df = raw.rename(columns={series_id: "value"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # FRED CSV missing values are often represented as '.'.
            df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
            df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
            df = df.dropna(subset=["date"])
            if df.empty:
                failures.append(series_id)
                continue
            df["indicator_id"] = series_id
            df["indicator_name"] = meta["name"]
            df["frequency"] = meta.get("frequency", "unknown")
            df["source"] = "FRED_public_csv"
            df = df[["date", "indicator_id", "indicator_name", "value", "frequency", "source"]]
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            failures.append(series_id)
            logger.exception("Failed FRED series=%s: %s", series_id, exc)

    if failures:
        debug_print("FRED failures skipped: " + ", ".join(sorted(set(failures))))
    if not frames:
        raise RuntimeError("No FRED macro data downloaded.")

    macro = pd.concat(frames, ignore_index=True).sort_values(["indicator_id", "date"])
    validate_macro_data(macro)
    return macro


def validate_macro_data(macro: pd.DataFrame) -> None:
    required = {"date", "indicator_id", "value", "frequency", "source"}
    missing = required.difference(macro.columns)
    if missing:
        raise ValueError(f"Macro data missing required columns: {missing}")
    duplicate_count = macro.duplicated(["indicator_id", "date"]).sum()
    if duplicate_count:
        raise ValueError(f"Found duplicate indicator/date rows in macro data: {duplicate_count}")
    debug_print(f"Validated macro data: {len(macro):,} rows, {macro['indicator_id'].nunique():,} indicators")


def fetch_current_valuation_snapshot(tickers: Iterable[str]) -> pd.DataFrame:
    """Fetch latest point-in-time snapshot metrics from yfinance.

    Important: these are latest available snapshots, not historical point-in-time valuations.
    The historical valuation table is now built from SEC EDGAR companyfacts where possible.
    This snapshot table is current-context only.
    """
    rows: list[dict] = []
    failures: list[str] = []
    tickers = [t for t in dict.fromkeys(tickers) if not t.startswith("^") and "=" not in t and not t.endswith("-USD")]
    debug_print(f"Fetching latest valuation snapshots for {len(tickers)} non-index/non-crypto tickers")

    for i, ticker in enumerate(tickers, start=1):
        try:
            debug_print(f"[{i}/{len(tickers)}] yfinance info for {ticker}")
            info = yf.Ticker(ticker).get_info()
            row = {
                "date": pd.Timestamp.today().normalize(),
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "revenue_ttm": info.get("totalRevenue"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "ev_to_sales": info.get("enterpriseToRevenue"),
                "profit_margins": info.get("profitMargins"),
                "source": "yfinance_info_latest_snapshot_not_historical",
            }
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failures.append(ticker)
            logger.exception("Valuation snapshot failed for ticker=%s: %s", ticker, exc)
        time.sleep(0.25)

    if failures:
        debug_print("Valuation snapshot failures skipped: " + ", ".join(sorted(set(failures))))

    if not rows:
        logger.warning("No valuation snapshot data returned. Continuing with empty valuation_metrics table.")
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "market_cap",
                "revenue_ttm",
                "price_to_sales",
                "pe_ratio",
                "forward_pe",
                "ev_to_sales",
                "profit_margins",
                "source",
            ]
        )
    return pd.DataFrame(rows)


def save_outputs(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    valuations: pd.DataFrame,
    current_snapshots: pd.DataFrame,
) -> None:
    """Persist normalized outputs to Parquet and DuckDB."""
    prices_path = PROCESSED_DIR / "asset_prices.parquet"
    macro_path = PROCESSED_DIR / "macro_indicators.parquet"
    valuations_path = PROCESSED_DIR / "valuation_metrics.parquet"
    snapshots_path = PROCESSED_DIR / "current_valuation_snapshot.parquet"

    debug_print(f"Writing Parquet files to {PROCESSED_DIR}")
    prices.to_parquet(prices_path, index=False)
    macro.to_parquet(macro_path, index=False)
    valuations.to_parquet(valuations_path, index=False)
    current_snapshots.to_parquet(snapshots_path, index=False)

    debug_print(f"Writing DuckDB database: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE asset_prices AS SELECT * FROM prices")
        con.execute("CREATE OR REPLACE TABLE macro_indicators AS SELECT * FROM macro")
        con.execute("CREATE OR REPLACE TABLE valuation_metrics AS SELECT * FROM valuations")
        con.execute("CREATE OR REPLACE TABLE current_valuation_snapshot AS SELECT * FROM current_snapshots")
        con.execute("CHECKPOINT")
    finally:
        con.close()

    debug_print("Data ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize market/macro data.")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="End date, YYYY-MM-DD. Default latest.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional custom ticker list. If omitted, uses project universe.",
    )
    parser.add_argument(
        "--skip-sec",
        action="store_true",
        help="Skip SEC EDGAR historical fundamentals. Use only when debugging price/macro ingestion.",
    )
    parser.add_argument(
        "--refresh-sec",
        action="store_true",
        help="Refetch SEC companyfacts instead of using the local cache.",
    )
    args = parser.parse_args()

    tickers = args.tickers or unique_tickers()
    debug_print(f"Pipeline start. Python file={Path(__file__).name}")
    prices = fetch_yfinance_prices(tickers=tickers, start=args.start, end=args.end)
    macro = fetch_fred_macro(start=args.start, end=args.end)

    if args.skip_sec:
        debug_print("Skipping SEC historical fundamentals because --skip-sec was provided.")
        valuations = empty_valuation_frame()
    else:
        valuations = fetch_sec_historical_valuation_metrics(tickers=tickers, prices=prices, refresh=args.refresh_sec)
        if valuations.empty:
            debug_print("SEC historical valuation table is empty. The dashboard will still run, but valuation signals will be unavailable.")

    current_snapshots = fetch_current_valuation_snapshot(tickers=tickers)
    save_outputs(prices, macro, valuations, current_snapshots)


if __name__ == "__main__":
    main()
