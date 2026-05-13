"""
SEC EDGAR historical fundamentals ingestion.

This module pulls point-in-time financial facts from the SEC's official EDGAR
JSON APIs and converts them into valuation rows that can be joined to daily
market prices without look-ahead leakage.

Important design choice:
    The valuation signal date is the first trading day AFTER the filing date.
    That is conservative. It avoids pretending that a model could always trade
    on the same closing price as a filing that may have arrived after market
    hours.

Run directly for a focused test after price ingestion:
    python sec_fundamentals.py --tickers NVDA MSFT AMD AVGO PLTR META GOOGL AMZN
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import requests

from config import (
    DB_PATH,
    LOG_DIR,
    PROCESSED_DIR,
    SEC_CONCEPTS,
    SEC_FUNDAMENTAL_TICKERS,
    SEC_REQUEST_PAUSE_SECONDS,
    SEC_TICKER_CIK_URL,
    SEC_USER_AGENT,
)

LOG_FILE = LOG_DIR / "sec_fundamentals.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sec_fundamentals")

SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
HTTP_TIMEOUT = 30


@dataclass(frozen=True)
class ConceptChoice:
    metric: str
    taxonomy: str
    concept: str
    unit: str


def debug_print(message: str) -> None:
    print(f"[SEC_FUNDAMENTALS DEBUG] {message}")
    logger.info(message)


def normalize_datetime_ns(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """Normalize a datetime column to pandas datetime64[ns].

    Several parts of this project use pd.merge_asof because we want valuation
    rows and macro rows to join to the nearest available market date without
    looking into the future. Pandas is strict about as-of joins: both date keys
    must use the exact same internal datetime precision. On Windows + pyarrow,
    one parquet/dataframe can arrive as datetime64[us] while another arrives as
    datetime64[s]. They represent ordinary dates to us, but pandas refuses to
    merge them.

    This helper converts both sides to timezone-naive datetime64[ns], which is
    pandas' standard high-precision datetime type. This fixes the storage-unit
    mismatch without changing the economic meaning of the dates.
    """
    out = df.copy()
    if column not in out.columns:
        return out

    out[column] = pd.to_datetime(out[column], errors="coerce")

    # If a source ever returns timezone-aware timestamps, strip the timezone so
    # all valuation and price dates share one simple comparison scale.
    try:
        if getattr(out[column].dt, "tz", None) is not None:
            out[column] = out[column].dt.tz_localize(None)
    except AttributeError:
        pass

    out[column] = out[column].astype("datetime64[ns]")
    out = out.dropna(subset=[column])
    return out


def _headers() -> dict[str, str]:
    """SEC asks automated users to identify themselves. Use .env to set this."""
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def _sec_get_json(url: str, host: str | None = None) -> dict[str, Any]:
    headers = _headers().copy()
    if host:
        headers["Host"] = host
    response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    if response.status_code == 429:
        raise RuntimeError("SEC rate limit reached. Slow the request pause or retry later.")
    response.raise_for_status()
    return response.json()


def load_or_fetch_ticker_cik_map(cache_path: Path | None = None) -> dict[str, str]:
    """Download/cache the SEC ticker-to-CIK mapping."""
    cache_path = cache_path or (PROCESSED_DIR / "sec_ticker_cik_map.parquet")
    if cache_path.exists():
        mapping_df = pd.read_parquet(cache_path)
        mapping = dict(zip(mapping_df["ticker"], mapping_df["cik10"]))
        debug_print(f"Loaded cached SEC ticker/CIK map with {len(mapping):,} rows")
        return mapping

    debug_print("Downloading SEC ticker/CIK mapping")
    # company_tickers.json lives on www.sec.gov, not data.sec.gov.
    headers = _headers().copy()
    headers["Host"] = "www.sec.gov"
    response = requests.get(SEC_TICKER_CIK_URL, headers=headers, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for item in payload.values():
        ticker = str(item.get("ticker", "")).upper().replace(".", "-")
        cik = str(item.get("cik_str", "")).strip()
        if ticker and cik:
            rows.append({"ticker": ticker, "cik10": cik.zfill(10), "title": item.get("title")})
    mapping_df = pd.DataFrame(rows).drop_duplicates("ticker")
    mapping_df.to_parquet(cache_path, index=False)
    debug_print(f"Cached SEC ticker/CIK map with {len(mapping_df):,} rows")
    return dict(zip(mapping_df["ticker"], mapping_df["cik10"]))


def get_company_facts(cik10: str, cache_dir: Path | None = None, refresh: bool = False) -> dict[str, Any]:
    """Fetch/cache one company's full XBRL companyfacts JSON."""
    cache_dir = cache_dir or (PROCESSED_DIR / "sec_companyfacts_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"CIK{cik10}.json"

    if cache_file.exists() and not refresh:
        try:
            import json

            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading cache for CIK %s: %s. Refetching.", cik10, exc)

    url = SEC_BASE.format(cik10=cik10)
    debug_print(f"Fetching SEC companyfacts for CIK{cik10}")
    payload = _sec_get_json(url)
    try:
        import json

        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not cache SEC companyfacts for CIK %s: %s", cik10, exc)
    time.sleep(SEC_REQUEST_PAUSE_SECONDS)
    return payload


def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _duration_days(row: pd.Series) -> float:
    if pd.isna(row.get("start")) or pd.isna(row.get("end")):
        return np.nan
    return float((row["end"] - row["start"]).days)


def extract_concept_records(
    facts: dict[str, Any],
    metric: str,
    candidate_concepts: list[str],
    unit: str,
    ticker: str,
    cik10: str,
) -> pd.DataFrame:
    """Extract rows for a metric from a list of possible us-gaap concepts."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    frames: list[pd.DataFrame] = []

    for concept in candidate_concepts:
        concept_payload = us_gaap.get(concept)
        if not concept_payload:
            continue
        units = concept_payload.get("units", {})
        if unit not in units:
            continue
        rows = units[unit]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df["ticker"] = ticker
        df["cik"] = cik10
        df["metric"] = metric
        df["concept"] = concept
        df["unit"] = unit
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    for col in ["start", "end", "filed"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        else:
            out[col] = pd.NaT
    for col in ["fy", "fp", "form", "accn", "frame"]:
        if col not in out.columns:
            out[col] = None
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    out["duration_days"] = out.apply(_duration_days, axis=1)
    out = out.dropna(subset=["filed", "end", "val"])
    out = out[out["form"].isin(["10-K", "10-Q"])]
    out = out.sort_values(["filed", "end", "concept"])
    return out


def choose_best_records(records: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Deduplicate competing concepts and repeated amendments by accession/end date.

    The concept order in config is the priority. This keeps the first available
    fact after sorting by concept priority and latest filing metadata.
    """
    if records.empty:
        return records
    concept_priority = {c: i for i, c in enumerate(SEC_CONCEPTS[metric]["concepts"])}
    g = records.copy()
    g["concept_priority"] = g["concept"].map(concept_priority).fillna(999).astype(int)
    g = g.sort_values(
        ["ticker", "end", "filed", "form", "duration_days", "concept_priority"],
        ascending=[True, True, True, True, True, True],
    )
    # Same metric/period can appear under several concepts. Keep the preferred concept.
    g = g.drop_duplicates(["ticker", "metric", "end", "filed", "form", "accn"], keep="first")
    # For the same period and form, later amended filings supersede earlier ones.
    g = g.sort_values(["ticker", "end", "form", "filed"])
    g = g.drop_duplicates(["ticker", "metric", "end", "form"], keep="last")
    return g


def build_flow_ttm(records: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Build TTM values for flow metrics such as revenue and net income.

    Uses two evidence paths:
        1. Annual 10-K duration facts as direct TTM values.
        2. Rolling four-quarter sum when true quarterly facts are available.

    This avoids using fiscal-period values before the filing date.
    """
    if records.empty:
        return pd.DataFrame(columns=["ticker", "date", "filing_date", "period_end", metric_name, f"{metric_name}_source"])

    r = choose_best_records(records, metric_name.replace("_ttm", ""))
    annual = r[(r["form"].eq("10-K")) & (r["duration_days"].between(330, 390))].copy()
    annual_rows = pd.DataFrame()
    if not annual.empty:
        annual_rows = annual[["ticker", "cik", "filed", "end", "val", "form", "accn", "concept"]].copy()
        annual_rows = annual_rows.rename(columns={"filed": "filing_date", "end": "period_end", "val": metric_name})
        annual_rows[f"{metric_name}_source"] = "SEC 10-K annual fact used as TTM"

    quarterly = r[(r["duration_days"].between(70, 110))].copy()
    quarter_rows = []
    if not quarterly.empty:
        quarterly = quarterly.sort_values(["ticker", "end", "filed"])
        # Deduplicate to one quarterly value per period end.
        quarterly = quarterly.drop_duplicates(["ticker", "end"], keep="last")
        for ticker, g in quarterly.groupby("ticker"):
            g = g.sort_values("end").copy()
            g["rolling_4q"] = g["val"].rolling(4, min_periods=4).sum()
            g["span_days_4q"] = (g["end"] - g["end"].shift(3)).dt.days
            valid = g.dropna(subset=["rolling_4q"])
            # Four quarter period-end span is commonly ~270-370 days depending on fiscal calendar.
            valid = valid[valid["span_days_4q"].between(240, 390)]
            if valid.empty:
                continue
            tmp = valid[["ticker", "cik", "filed", "end", "rolling_4q", "form", "accn", "concept"]].copy()
            tmp = tmp.rename(columns={"filed": "filing_date", "end": "period_end", "rolling_4q": metric_name})
            tmp[f"{metric_name}_source"] = "SEC rolling four quarterly facts"
            quarter_rows.append(tmp)
    quarterly_rows = pd.concat(quarter_rows, ignore_index=True) if quarter_rows else pd.DataFrame()

    combined = pd.concat([annual_rows, quarterly_rows], ignore_index=True) if not annual_rows.empty or not quarterly_rows.empty else pd.DataFrame()
    if combined.empty:
        return pd.DataFrame(columns=["ticker", "date", "filing_date", "period_end", metric_name, f"{metric_name}_source"])

    combined["filing_date"] = pd.to_datetime(combined["filing_date"])
    combined["period_end"] = pd.to_datetime(combined["period_end"])
    # One valuation row becomes tradable the next calendar day at the earliest.
    combined["date"] = combined["filing_date"] + pd.Timedelta(days=1)
    combined = combined.sort_values(["ticker", "date", "period_end", f"{metric_name}_source"])
    combined = combined.drop_duplicates(["ticker", "date"], keep="last")
    return combined


def build_instant_metric(records: pd.DataFrame, output_col: str) -> pd.DataFrame:
    """Build point-in-time balance sheet/share metrics keyed by post-filing signal date."""
    if records.empty:
        return pd.DataFrame(columns=["ticker", "date", output_col, f"{output_col}_source"])
    metric_key = output_col
    if output_col == "shares_outstanding":
        metric_key = "shares_outstanding"
    elif output_col == "cash_and_equivalents":
        metric_key = "cash"
    elif output_col == "total_debt":
        metric_key = "debt"

    r = choose_best_records(records, metric_key).copy()
    # Instant facts normally have no start. Keep non-flow rows regardless of duration.
    r = r.sort_values(["ticker", "end", "filed"])
    r = r.drop_duplicates(["ticker", "end"], keep="last")
    out = r[["ticker", "cik", "filed", "end", "val", "form", "accn", "concept"]].copy()
    out = out.rename(columns={"filed": "filing_date", "end": "period_end", "val": output_col})
    out["date"] = out["filing_date"] + pd.Timedelta(days=1)
    out[f"{output_col}_source"] = "SEC instant fact"
    return out




def build_debt_metric(records: pd.DataFrame) -> pd.DataFrame:
    """Build total debt from SEC instant concepts without double-counting.

    Companies often report debt as current/noncurrent pieces or as a single
    aggregate. This function prefers summed current+noncurrent pairs and falls
    back to the single aggregate concept when split facts are unavailable.
    """
    if records.empty:
        return pd.DataFrame(columns=["ticker", "date", "total_debt", "total_debt_source"])

    r = records.copy().sort_values(["ticker", "end", "filed", "concept"])
    rows = []
    group_cols = ["ticker", "cik", "filed", "end", "form", "accn"]
    for keys, g in r.groupby(group_cols, dropna=False):
        values = g.dropna(subset=["val"]).drop_duplicates("concept", keep="last").set_index("concept")["val"].to_dict()
        debt_val = np.nan
        source = None
        if (
            "LongTermDebtAndFinanceLeaseObligationsCurrent" in values
            and "LongTermDebtAndFinanceLeaseObligationsNoncurrent" in values
        ):
            debt_val = values["LongTermDebtAndFinanceLeaseObligationsCurrent"] + values["LongTermDebtAndFinanceLeaseObligationsNoncurrent"]
            source = "SEC current + noncurrent finance lease debt facts"
        elif "LongTermDebtCurrent" in values and "LongTermDebtNoncurrent" in values:
            debt_val = values["LongTermDebtCurrent"] + values["LongTermDebtNoncurrent"]
            source = "SEC current + noncurrent long-term debt facts"
        elif "LongTermDebt" in values:
            debt_val = values["LongTermDebt"]
            source = "SEC aggregate long-term debt fact"
        elif "ShortTermBorrowings" in values:
            debt_val = values["ShortTermBorrowings"]
            source = "SEC short-term borrowings fact only"
        if pd.isna(debt_val):
            continue
        ticker, cik, filed, end, form, accn = keys
        rows.append({
            "ticker": ticker,
            "cik": cik,
            "filing_date": filed,
            "period_end": end,
            "date": pd.to_datetime(filed) + pd.Timedelta(days=1),
            "total_debt": debt_val,
            "form": form,
            "accn": accn,
            "total_debt_source": source,
        })

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "total_debt", "total_debt_source"])
    out = pd.DataFrame(rows).sort_values(["ticker", "date", "period_end"])
    out = out.drop_duplicates(["ticker", "date"], keep="last")
    return out

def _price_asof_after_filing(prices: pd.DataFrame, ticker: str, dates: pd.Series) -> pd.DataFrame:
    """Return the first adjusted close on/after each metric date for one ticker.

    SEC valuation rows become usable only after a filing is available. This
    function finds the first market price on or after that filing-available
    date. The as-of join direction is "forward" because we are moving from a
    filing date to the next actual trading day, not from a market row into the
    future for a fundamental value.
    """
    p = prices[prices["ticker"].eq(ticker)][["date", "adjusted_close"]].dropna().copy()
    p = normalize_datetime_ns(p, "date").sort_values("date")

    base = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce")})
    base = normalize_datetime_ns(base, "date").sort_values("date")

    if p.empty or base.empty:
        return pd.DataFrame({"date": dates, "price_date": pd.NaT, "price": np.nan})

    right = p.rename(columns={"date": "price_date", "adjusted_close": "price"}).copy()
    right = normalize_datetime_ns(right, "price_date").sort_values("price_date")

    debug_print(
        f"Price as-of merge for {ticker}: filing_date_dtype={base['date'].dtype}, "
        f"price_date_dtype={right['price_date'].dtype}, filings={len(base)}, prices={len(right)}"
    )

    merged = pd.merge_asof(
        base,
        right,
        left_on="date",
        right_on="price_date",
        direction="forward",
    )
    return merged


def _merge_metric_asof(base: pd.DataFrame, metric: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Merge fundamentals backward by ticker while keeping datetime precision stable."""
    if base.empty or metric.empty:
        for col in value_cols:
            if col not in base.columns:
                base[col] = np.nan
        return base

    base = normalize_datetime_ns(base, "date")
    metric = normalize_datetime_ns(metric, "date")

    output_frames = []
    for ticker, g in base.groupby("ticker"):
        left = normalize_datetime_ns(g, "date").sort_values("date").copy()
        right = normalize_datetime_ns(metric[metric["ticker"].eq(ticker)], "date").sort_values("date").copy()
        if right.empty:
            for col in value_cols:
                left[col] = np.nan
            output_frames.append(left)
            continue
        keep = ["date"] + [c for c in value_cols if c in right.columns]
        merged = pd.merge_asof(left, right[keep], on="date", direction="backward")
        output_frames.append(merged)
    return pd.concat(output_frames, ignore_index=True) if output_frames else base


def compute_historical_valuations_for_ticker(
    ticker: str,
    cik10: str,
    facts: dict[str, Any],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical valuation rows for one ticker using EDGAR facts plus prices."""
    concept_frames: dict[str, pd.DataFrame] = {}
    for metric, spec in SEC_CONCEPTS.items():
        concept_frames[metric] = extract_concept_records(
            facts=facts,
            metric=metric,
            candidate_concepts=spec["concepts"],
            unit=spec["unit"],
            ticker=ticker,
            cik10=cik10,
        )
        debug_print(f"{ticker}: extracted {len(concept_frames[metric]):,} SEC rows for {metric}")

    revenue_ttm = build_flow_ttm(concept_frames["revenue"], "revenue_ttm")
    net_income_ttm = build_flow_ttm(concept_frames["net_income"], "net_income_ttm")
    shares = build_instant_metric(concept_frames["shares_outstanding"], "shares_outstanding")
    cash = build_instant_metric(concept_frames["cash"], "cash_and_equivalents")

    # Debt can be split across multiple concepts. Sum same-date preferred current/noncurrent rows when available.
    debt_records = concept_frames["debt"]
    debt = build_debt_metric(debt_records)

    if revenue_ttm.empty:
        debug_print(f"{ticker}: no usable SEC revenue TTM rows. Skipping historical valuation.")
        return pd.DataFrame()

    base = revenue_ttm[["ticker", "cik", "date", "filing_date", "period_end", "revenue_ttm", "revenue_ttm_source"]].copy()
    base = _merge_metric_asof(base, net_income_ttm, ["net_income_ttm", "net_income_ttm_source"])
    base = _merge_metric_asof(base, shares, ["shares_outstanding", "shares_outstanding_source"])
    base = _merge_metric_asof(base, cash, ["cash_and_equivalents", "cash_and_equivalents_source"])
    base = _merge_metric_asof(base, debt, ["total_debt", "total_debt_source"])

    price_lookup = _price_asof_after_filing(prices, ticker, base["date"])
    base = base.sort_values("date").reset_index(drop=True)
    price_lookup = price_lookup.sort_values("date").reset_index(drop=True)
    base["price_date"] = price_lookup["price_date"]
    base["price"] = price_lookup["price"]

    base["market_cap"] = base["price"] * base["shares_outstanding"]
    base["enterprise_value"] = base["market_cap"] + base["total_debt"].fillna(0) - base["cash_and_equivalents"].fillna(0)
    base["price_to_sales"] = base["market_cap"] / base["revenue_ttm"].replace(0, np.nan)
    base["pe_ratio"] = base["market_cap"] / base["net_income_ttm"].replace(0, np.nan)
    base["ev_to_sales"] = base["enterprise_value"] / base["revenue_ttm"].replace(0, np.nan)
    base["source"] = "SEC EDGAR companyfacts + yfinance adjusted close"

    # Clean clearly impossible values without hiding the row.
    ratio_cols = ["price_to_sales", "pe_ratio", "ev_to_sales"]
    for col in ratio_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")
        base.loc[~np.isfinite(base[col]), col] = np.nan

    ordered = [
        "date",
        "ticker",
        "cik",
        "filing_date",
        "period_end",
        "price_date",
        "price",
        "market_cap",
        "enterprise_value",
        "revenue_ttm",
        "net_income_ttm",
        "shares_outstanding",
        "cash_and_equivalents",
        "total_debt",
        "price_to_sales",
        "pe_ratio",
        "ev_to_sales",
        "revenue_ttm_source",
        "net_income_ttm_source",
        "shares_outstanding_source",
        "cash_and_equivalents_source",
        "total_debt_source",
        "source",
    ]
    for col in ordered:
        if col not in base.columns:
            base[col] = np.nan
    return base[ordered].sort_values("date")


def fetch_sec_historical_valuation_metrics(
    tickers: Iterable[str],
    prices: pd.DataFrame,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch historical valuation metrics for tickers with SEC CIK coverage."""
    mapping = load_or_fetch_ticker_cik_map()
    input_tickers = [t.upper().replace(".", "-") for t in dict.fromkeys(tickers)]
    eligible = [t for t in input_tickers if t in SEC_FUNDAMENTAL_TICKERS and t in mapping]
    if not eligible:
        debug_print("No eligible SEC tickers found for historical fundamentals.")
        return empty_valuation_frame()

    debug_print(f"Fetching SEC historical fundamentals for {len(eligible)} tickers: {', '.join(eligible)}")
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for i, ticker in enumerate(eligible, start=1):
        cik10 = mapping[ticker]
        try:
            debug_print(f"[{i}/{len(eligible)}] SEC valuation build for {ticker} / CIK{cik10}")
            facts = get_company_facts(cik10, refresh=refresh)
            df = compute_historical_valuations_for_ticker(ticker, cik10, facts, prices)
            if df.empty:
                failures.append(ticker)
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            failures.append(ticker)
            logger.exception("SEC historical valuation failed for %s: %s", ticker, exc)

    if failures:
        debug_print("SEC valuation failures or no usable rows: " + ", ".join(sorted(set(failures))))
    if not frames:
        return empty_valuation_frame()

    valuations = pd.concat(frames, ignore_index=True)
    valuations = valuations.dropna(subset=["date", "ticker", "revenue_ttm"], how="any")
    valuations = valuations.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    validate_sec_valuations(valuations)
    return valuations


def empty_valuation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "cik",
            "filing_date",
            "period_end",
            "price_date",
            "price",
            "market_cap",
            "enterprise_value",
            "revenue_ttm",
            "net_income_ttm",
            "shares_outstanding",
            "cash_and_equivalents",
            "total_debt",
            "price_to_sales",
            "pe_ratio",
            "ev_to_sales",
            "revenue_ttm_source",
            "net_income_ttm_source",
            "shares_outstanding_source",
            "cash_and_equivalents_source",
            "total_debt_source",
            "source",
        ]
    )


def validate_sec_valuations(valuations: pd.DataFrame) -> None:
    required = {"date", "ticker", "filing_date", "period_end", "revenue_ttm", "source"}
    missing = required.difference(valuations.columns)
    if missing:
        raise ValueError(f"SEC valuation data missing columns: {missing}")
    duplicates = valuations.duplicated(["ticker", "date"]).sum()
    if duplicates:
        raise ValueError(f"Duplicate SEC valuation ticker/date rows: {duplicates}")
    coverage = valuations.groupby("ticker")["date"].agg(["min", "max", "count"]).reset_index()
    debug_print("SEC valuation coverage:\n" + coverage.to_string(index=False))


def save_sec_valuations(valuations: pd.DataFrame) -> None:
    path = PROCESSED_DIR / "valuation_metrics.parquet"
    valuations.to_parquet(path, index=False)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE valuation_metrics AS SELECT * FROM valuations")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    debug_print(f"Saved SEC valuation metrics to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR historical fundamentals and build valuation metrics.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker list. Defaults to configured SEC universe.")
    parser.add_argument("--refresh", action="store_true", help="Refetch SEC companyfacts instead of using local cache.")
    args = parser.parse_args()

    price_path = PROCESSED_DIR / "asset_prices.parquet"
    if not price_path.exists():
        raise FileNotFoundError(f"Missing {price_path}. Run python data_ingestion.py first.")
    prices = pd.read_parquet(price_path)
    prices["date"] = pd.to_datetime(prices["date"])
    tickers = args.tickers or SEC_FUNDAMENTAL_TICKERS
    valuations = fetch_sec_historical_valuation_metrics(tickers, prices, refresh=args.refresh)
    save_sec_valuations(valuations)


if __name__ == "__main__":
    main()
