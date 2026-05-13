"""
Project-wide configuration for the Economic Bubble Monitoring & Investment Strategy Dashboard.

This project is for education/research only. It does not provide financial advice.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "bubble_dashboard.duckdb"
LOG_DIR = BASE_DIR / "logs"

for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()

# SEC EDGAR access settings. SEC fair-access guidance limits automated users
# to no more than 10 requests/second; this project defaults far below that.
# Set SEC_USER_AGENT in .env to something identifiable, e.g.
# SEC_USER_AGENT="EconomicBubbleDashboard/1.0 your.email@example.com"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "EconomicBubbleDashboard/1.0 research@example.com").strip()
SEC_REQUEST_PAUSE_SECONDS = float(os.getenv("SEC_REQUEST_PAUSE_SECONDS", "0.25"))
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

DEFAULT_START_DATE = "1927-01-01"
DEFAULT_END_DATE = None  # yfinance interprets None as latest available
STARTING_CAPITAL_DEFAULT = 10_000.0
MONTHLY_CONTRIBUTION_DEFAULT = 0.0
TRADING_DAYS_PER_YEAR = 252

# Core asset universe. Delisted tickers are intentionally included where useful;
# the ingestion layer logs failures and continues rather than inventing data.
CORE_ASSETS = {
    "broad_indexes": ["^GSPC", "SPY", "^IXIC", "QQQ", "^DJI", "IWM"],
    "sector_etfs": ["XLK", "SOXX", "SMH", "XLF", "XHB", "ARKK", "XLE", "XOP", "IYW", "IGV"],
    "dotcom_leaders": ["CSCO", "INTC", "MSFT", "AMZN", "ORCL", "QCOM", "AAPL", "IBM", "TXN", "YHOO", "AOL"],
    "housing_financial": ["XLF", "JPM", "BAC", "C", "AIG", "GS", "MS", "LEN", "DHI", "TOL", "XHB"],
    "crypto_tech": ["BTC-USD", "ETH-USD", "COIN", "ARKK", "QQQ", "TSLA", "NVDA", "ROKU", "ZM", "SHOP", "PYPL"],
    "ai_cycle": ["NVDA", "MSFT", "AMD", "AVGO", "PLTR", "META", "GOOGL", "AMZN", "AAPL", "TSM", "ASML", "MU", "LRCX", "KLAC", "SMH", "SOXX"],
    "global_bubbles": ["^N225", "EWJ", "FXI", "ASHR", "USO", "XLE", "DBC", "GLD", "SLV"],
    "speculative_2020": ["SPCE", "NKLA", "RIVN", "LCID", "GME", "AMC", "ARKK", "IPO", "ROKU", "ZM", "SHOP", "SNOW", "NET", "DDOG", "MDB", "CRWD", "U", "PYPL", "SQ", "XYZ"],
}

BUBBLES = {
    "Dot-com Bubble": {
        "start": "1995-01-01",
        "peak_hint": "2000-03-10",
        "end": "2003-12-31",
        "asset_class": "Equities / Internet / Telecom / Semiconductors",
        "tickers": ["^IXIC", "QQQ", "CSCO", "INTC", "MSFT", "AMZN", "ORCL", "QCOM"],
        "notes": "Primary research anchor for comparing AI/tech enthusiasm with a historically documented tech mania.",
    },
    "Housing / Financial Crisis": {
        "start": "2004-01-01",
        "peak_hint": "2007-10-09",
        "end": "2012-12-31",
        "asset_class": "Financials / Housing / Credit",
        "tickers": ["SPY", "XLF", "XHB", "JPM", "BAC", "C", "AIG", "LEN", "DHI"],
        "notes": "Best cycle for connecting prices, credit stress, mortgage rates, home prices, and recession timing.",
    },
    "2017 Crypto Bubble": {
        "start": "2015-01-01",
        "peak_hint": "2017-12-17",
        "end": "2019-12-31",
        "asset_class": "Crypto assets",
        "tickers": ["BTC-USD", "ETH-USD"],
        "notes": "Useful high-volatility case with severe drawdowns and rapid reflexive sentiment cycles.",
    },
    "COVID Speculative / SPAC / Meme Cycle": {
        "start": "2020-01-01",
        "peak_hint": "2021-02-12",
        "end": "2023-12-31",
        "asset_class": "Speculative growth / SPACs / meme stocks",
        "tickers": ["ARKK", "IPO", "GME", "AMC", "SPCE", "NKLA", "TSLA", "QQQ"],
        "notes": "Useful for teaching liquidity-driven speculation, retail flows, and drawdown dispersion.",
    },
    "Crypto / High-Growth Tech Bubble": {
        "start": "2020-01-01",
        "peak_hint": "2021-11-09",
        "end": "2023-12-31",
        "asset_class": "Crypto / High-growth tech",
        "tickers": ["BTC-USD", "ETH-USD", "COIN", "ARKK", "QQQ", "TSLA", "NVDA"],
        "notes": "Cross-asset cycle covering crypto, unprofitable tech, high-duration equities, and tightening macro conditions.",
    },
    "AI / Mega-Cap Tech Cycle": {
        "start": "2022-01-01",
        "peak_hint": None,
        "end": None,
        "asset_class": "AI infrastructure / semiconductors / mega-cap software",
        "tickers": ["NVDA", "MSFT", "AMD", "AVGO", "PLTR", "META", "GOOGL", "AMZN", "AAPL", "TSM", "ASML", "MU", "LRCX", "KLAC", "SMH", "SOXX", "QQQ"],
        "notes": "Current live monitor. Status is an evidence score, not a prediction.",
    },
    "1929 Stock Market Crash": {
        "start": "1927-01-01",
        "peak_hint": "1929-09-03",
        "end": "1933-12-31",
        "asset_class": "Broad US equities",
        "tickers": ["^DJI", "^GSPC"],
        "notes": "Historically important, but free daily data is patchier than modern cycles.",
    },
    "1970s Nifty Fifty": {
        "start": "1969-01-01",
        "peak_hint": "1972-12-31",
        "end": "1978-12-31",
        "asset_class": "Large-cap growth equities",
        "tickers": ["^GSPC", "IBM", "KO", "MCD", "JNJ", "PG", "DIS"],
        "notes": "Useful valuation regime case, but exact constituent-level historical valuation data is harder to reconstruct.",
    },
    "Japanese Asset Bubble": {
        "start": "1985-01-01",
        "peak_hint": "1989-12-29",
        "end": "1995-12-31",
        "asset_class": "Japanese equities / real estate proxy",
        "tickers": ["^N225", "EWJ"],
        "notes": "Excellent non-US comparison; equity data feasible, real estate data requires extra sources.",
    },
    "Commodity / Oil Bubble": {
        "start": "2003-01-01",
        "peak_hint": "2008-07-03",
        "end": "2011-12-31",
        "asset_class": "Commodities / energy",
        "tickers": ["USO", "XLE", "DBC", "CL=F"],
        "notes": "Useful because macro/inflation/rate dynamics differ from equity bubbles.",
    },
    "China Real Estate / Credit Cycle": {
        "start": "2015-01-01",
        "peak_hint": "2021-02-17",
        "end": None,
        "asset_class": "China real estate / equities proxy",
        "tickers": ["FXI", "ASHR", "MCHI", "EWH"],
        "notes": "Feasible as proxy ETF analysis. Direct property developer and housing data require care.",
    },
}

# FRED series. The ingestion layer skips unavailable series and records failures.
FRED_SERIES = {
    "FEDFUNDS": {"name": "Federal Funds Effective Rate", "frequency": "monthly"},
    "DGS10": {"name": "10-Year Treasury Constant Maturity Rate", "frequency": "daily"},
    "T10Y2Y": {"name": "10-Year Minus 2-Year Treasury Spread", "frequency": "daily"},
    "CPIAUCSL": {"name": "CPI: All Urban Consumers", "frequency": "monthly"},
    "UNRATE": {"name": "Unemployment Rate", "frequency": "monthly"},
    "GDP": {"name": "Gross Domestic Product", "frequency": "quarterly"},
    "MORTGAGE30US": {"name": "30-Year Fixed Mortgage Rate", "frequency": "weekly"},
    "CSUSHPINSA": {"name": "Case-Shiller U.S. National Home Price Index", "frequency": "monthly"},
    "USREC": {"name": "NBER Recession Indicator", "frequency": "monthly"},
    "STLFSI4": {"name": "St. Louis Fed Financial Stress Index", "frequency": "weekly"},
    "WILL5000INDFC": {"name": "Wilshire 5000 Total Market Full Cap Index", "frequency": "daily"},
}

# Hand-built metadata for cleaner dashboards.
ASSET_METADATA = {
    "^GSPC": ("S&P 500 Index", "Index"),
    "SPY": ("SPDR S&P 500 ETF", "ETF"),
    "^IXIC": ("Nasdaq Composite", "Index"),
    "QQQ": ("Invesco QQQ Trust", "ETF"),
    "^DJI": ("Dow Jones Industrial Average", "Index"),
    "IWM": ("Russell 2000 ETF", "ETF"),
    "XLK": ("Technology Select Sector SPDR", "ETF"),
    "SOXX": ("iShares Semiconductor ETF", "ETF"),
    "SMH": ("VanEck Semiconductor ETF", "ETF"),
    "XLF": ("Financial Select Sector SPDR", "ETF"),
    "XHB": ("SPDR S&P Homebuilders ETF", "ETF"),
    "ARKK": ("ARK Innovation ETF", "ETF"),
    "BTC-USD": ("Bitcoin USD", "Crypto"),
    "ETH-USD": ("Ethereum USD", "Crypto"),
    "CL=F": ("WTI Crude Oil Futures", "Commodity"),
}

AI_LEADERS = ["NVDA", "MSFT", "AMD", "AVGO", "PLTR", "META", "GOOGL", "AMZN", "AAPL", "TSM", "ASML", "MU", "LRCX", "KLAC", "SMH", "SOXX", "QQQ"]
DOTCOM_LEADERS = ["CSCO", "INTC", "MSFT", "AMZN", "ORCL", "QCOM", "^IXIC", "QQQ"]

REAL_TIME_STRATEGIES = [
    "buy_and_hold",
    "sell_exit_buy_recovery",
    "cash_until_40pct_drawdown",
    "cash_until_60pct_drawdown",
    "cash_until_rsi_recovery",
    "cash_until_reclaim_200dma",
    "dca_after_40pct_drawdown",
]

RETROSPECTIVE_STRATEGIES = [
    "buy_6_months_after_retrospective_peak",
    "buy_12_months_after_retrospective_peak",
]

# Tickers where SEC fundamentals are meaningful. ETFs, indexes, crypto, and futures
# are intentionally excluded because they do not file corporate 10-K/10-Q XBRL facts.
SEC_FUNDAMENTAL_TICKERS = sorted(set([
    "NVDA", "MSFT", "AMD", "AVGO", "PLTR", "META", "GOOGL", "AMZN", "AAPL", "TSM", "ASML", "MU", "LRCX", "KLAC",
    "CSCO", "INTC", "ORCL", "QCOM",
    "JPM", "BAC", "C", "AIG", "GS", "MS", "LEN", "DHI", "TOL",
    "TSLA", "COIN", "SPCE", "NKLA", "RIVN", "LCID", "GME", "AMC", "ROKU", "ZM", "SHOP", "SNOW", "NET", "DDOG", "MDB", "CRWD", "U", "PYPL", "SQ", "XYZ",
    "IBM", "KO", "MCD", "JNJ", "PG", "DIS",
]))

# SEC XBRL concept candidates. The first concept in each list is preferred,
# but the ingestion layer falls through to alternatives because companies use
# different tags over time.
SEC_CONCEPTS = {
    "revenue": {
        "unit": "USD",
        "concepts": [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
        ],
    },
    "net_income": {
        "unit": "USD",
        "concepts": [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
    },
    "shares_outstanding": {
        "unit": "shares",
        "concepts": [
            "EntityCommonStockSharesOutstanding",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
        ],
    },
    "cash": {
        "unit": "USD",
        "concepts": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashCashEquivalentsAndShortTermInvestments",
        ],
    },
    "debt": {
        "unit": "USD",
        "concepts": [
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "ShortTermBorrowings",
            "LongTermDebtCurrent",
            "LongTermDebtNoncurrent",
            "LongTermDebt",
        ],
    },
}
