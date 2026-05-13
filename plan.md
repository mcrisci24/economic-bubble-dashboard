# Economic Bubble Monitoring & Investment Strategy Dashboard

Educational/research project only. It is not financial advice and does not predict exact market tops or bottoms.

## 1. Executive Summary

This project is feasible as a transparent, rules-based research system over daily market data, macro indicators, and filing-date-aware company fundamentals. The project now treats SEC EDGAR companyfacts as a core data source for historical valuation rows, not a future add-on. Current yfinance valuation snapshots are kept only as live context and are not used as historical evidence.

The dashboard can answer:

- How did different assets behave before, during, and after historical bubble peaks?
- Which technical and macro warning signals appeared before or during breakdowns?
- Which re-entry rules historically worked better or worse after large drawdowns?
- How do current AI-exposed assets compare to prior technology/manic cycles on transparent technical signals?

The dashboard cannot answer with certainty:

- The exact market top or bottom.
- Whether AI is definitely a bubble.
- Whether any specific user should buy, sell, or hold an asset.
- Whether a historical signal will work in the future.

## 2. Historical Bubble Dataset Feasibility Table

| Bubble | Approx. range | Asset class | Representative tickers/indexes | Data sources | Feasibility | Include? | Why useful |
|---|---:|---|---|---|---|---|---|
| Dot-com Bubble | 1995-2003 | Internet, telecom, semis, broad tech | ^IXIC, QQQ, CSCO, INTC, MSFT, AMZN, ORCL, QCOM | yfinance/Yahoo, SEC, FRED | High for prices; medium for valuations | Yes | Best historical comparison for AI/tech enthusiasm. |
| Housing / Financial Crisis | 2004-2012 | Financials, homebuilders, credit, housing | SPY, XLF, XHB, JPM, BAC, C, AIG, LEN, DHI | yfinance, FRED, NBER/FRED USREC | High | Yes | Strong macro + credit + housing linkage. |
| 2017 Crypto Bubble | 2015-2019 | Crypto | BTC-USD, ETH-USD | yfinance, CoinMarketCap/CoinGecko optional | High for BTC/ETH | Yes | Clean high-volatility bubble and drawdown case. |
| Crypto / High-Growth Tech | 2020-2023 | Crypto, ARKK, high-duration equities | BTC-USD, ETH-USD, COIN, ARKK, QQQ, TSLA, NVDA | yfinance, FRED | High | Yes | Good modern rate-sensitivity case. |
| COVID SPAC/Meme/Speculative Cycle | 2020-2023 | SPACs, meme stocks, unprofitable growth | ARKK, IPO, GME, AMC, SPCE, NKLA, TSLA | yfinance; SPAC CSV fallback | Medium | Yes, with caveats | Useful retail/speculation regime. |
| AI / Mega-Cap Tech Cycle | 2022-present | AI semis, cloud, software, mega-cap tech | NVDA, MSFT, AMD, AVGO, PLTR, META, GOOGL, AMZN, SMH, SOXX | yfinance, FRED, current fundamentals | High for monitoring; unresolved phase | Yes | Current live research target. |
| 1929 Crash | 1927-1933 | Broad US equities | ^DJI, ^GSPC if available | Yahoo/Stooq/manual CSV | Medium/low | Optional | Historically important, data patchier. |
| 1970s Nifty Fifty | 1969-1978 | Large-cap growth | ^GSPC, IBM, KO, MCD, JNJ, PG, DIS | Yahoo/Stooq/SEC/manual valuations | Medium/low | Optional | Great valuation lesson, harder point-in-time fundamentals. |
| Japanese Asset Bubble | 1985-1995 | Japan equities/real estate proxy | ^N225, EWJ | Yahoo, FRED/OECD optional | Medium/high for equities | Yes/optional | Non-US template for long post-bubble stagnation. |
| Commodity/Oil Bubble | 2003-2011 | Oil/commodities/energy equities | USO, XLE, DBC, CL=F | yfinance, FRED | Medium/high | Optional | Shows macro/inflation bubble dynamics. |
| China Real Estate / Credit Cycle | 2015-present | China property/equities proxy | FXI, ASHR, MCHI, EWH | yfinance, BIS/OECD/World Bank optional | Medium | Optional | Useful but harder direct real estate data. |

## 3. Architecture

```text
External sources
  ├─ yfinance/Yahoo Finance: daily OHLCV prices
  ├─ FRED: macro indicators
  ├─ SEC EDGAR: historical companyfacts fundamentals from 10-K/10-Q filings
  └─ CSV fallbacks: delisted tickers, SPAC lists, IPO/margin/sentiment proxies

Python pipeline
  ├─ data_ingestion.py
  ├─ feature_engineering.py
  ├─ bubble_signals.py
  ├─ backtesting.py
  └─ dashboard.py

Storage
  ├─ Parquet files: fast local analytics and Streamlit loading
  └─ DuckDB: SQL layer over normalized research tables
```

Recommended storage: **Parquet + DuckDB**.

Why:

- CSV is simple but slow and loses types.
- SQLite is fine for small relational data, but less pleasant for column analytics.
- PostgreSQL is excellent but unnecessary for a local research dashboard unless this becomes multi-user.
- DuckDB + Parquet gives fast local analytics, SQL querying, and easy portability.

## 4. Pipeline Steps

1. Download daily market prices from yfinance for configured tickers.
2. Download FRED macro indicators using public FRED CSV endpoints, avoiding pandas_datareader for Python 3.12 compatibility.
3. Fetch SEC EDGAR companyfacts for eligible corporate tickers, cache the raw JSON, and build filing-date-aware historical valuation rows.
4. Fetch latest yfinance valuation snapshots separately for current context only.
5. Save normalized `asset_prices`, `macro_indicators`, `valuation_metrics`, and `current_valuation_snapshot` tables.
6. Compute technical indicators: SMA, RSI, drawdown, volatility, z-score, acceleration.
7. Compute valuation features: P/S vs expanding historical median, EV/S vs expanding historical median, P/E heat, and revenue-growth/valuation gap.
8. Score warning and recovery rules.
9. Run backtests using next-day execution to avoid same-close look-ahead.
10. Launch Streamlit dashboard.

## 5. Gold Standard Schema

### asset_prices

| Column | Type | Frequency | Notes |
|---|---|---|---|
| date | datetime | daily | Trading date. |
| asset_id | string | daily | Normalized asset identifier. |
| ticker | string | daily | Market ticker. |
| asset_name | string | daily | Human-readable name. |
| asset_class | string | daily | Index, ETF, equity, crypto, commodity. |
| bubble_period | string/null | daily | Bubble window tag. |
| open | float | daily | OHLCV. |
| high | float | daily | OHLCV. |
| low | float | daily | OHLCV. |
| close | float | daily | Raw close. |
| adjusted_close | float | daily | Split/dividend adjusted where available. |
| volume | float | daily | Trading volume. |
| source | string | daily | yfinance, CSV, etc. |

### macro_indicators

| Column | Type | Frequency | Notes |
|---|---|---|---|
| date | datetime | mixed | Observation date. |
| indicator_id | string | mixed | FRED series ID. |
| indicator_name | string | mixed | Human-readable name. |
| value | float | mixed | Indicator value. |
| frequency | string | mixed | daily, weekly, monthly, quarterly. |
| source | string | mixed | FRED. |

### valuation_metrics

Historical valuation rows are built from SEC EDGAR companyfacts plus the first available market price after the filing date. The `date` column is intentionally the first tradable day after the filing date to reduce look-ahead bias.

| Column | Type | Frequency | Notes |
|---|---|---|---|
| date | datetime | filing/quarterly-ish | Conservative signal date: first trading day after filing date. |
| ticker | string | filing/quarterly-ish | Corporate ticker. |
| cik | string | filing/quarterly-ish | SEC Central Index Key. |
| filing_date | datetime | filing/quarterly-ish | Date the 10-K/10-Q fact was filed. |
| period_end | datetime | filing/quarterly-ish | Fiscal period end. |
| price_date | datetime | daily | Trading date used for valuation price. |
| price | float | daily | Adjusted close used for market cap proxy. |
| market_cap | float | filing/quarterly-ish | Price × SEC shares outstanding. |
| enterprise_value | float | filing/quarterly-ish | Market cap + debt - cash where available. |
| revenue_ttm | float | filing/quarterly-ish | SEC annual revenue or rolling four-quarter revenue. |
| net_income_ttm | float | filing/quarterly-ish | SEC annual net income or rolling four-quarter net income. |
| shares_outstanding | float | filing/quarterly-ish | SEC share fact. |
| cash_and_equivalents | float | filing/quarterly-ish | SEC cash concept where available. |
| total_debt | float | filing/quarterly-ish | SEC debt concept where available. |
| price_to_sales | float | filing/quarterly-ish | Market cap / revenue TTM. |
| pe_ratio | float | filing/quarterly-ish | Market cap / net income TTM. |
| ev_to_sales | float | filing/quarterly-ish | Enterprise value / revenue TTM. |
| source | string | filing/quarterly-ish | SEC EDGAR companyfacts + yfinance adjusted close. |

### valuation_features

| Column | Type | Frequency | Notes |
|---|---|---|---|
| date | datetime | filing/quarterly-ish | Same conservative signal date. |
| ps_historical_median | float | expanding | Prior-row expanding median, shifted to avoid leakage. |
| ps_to_historical_median | float | expanding | Current P/S divided by prior historical median. |
| ev_sales_to_historical_median | float | expanding | Current EV/S divided by prior historical median. |
| revenue_ttm_yoy_pct | float | filing-row | Approximate YoY TTM revenue growth from prior filed rows. |
| valuation_warning_score | int | filing/quarterly-ish | 0-100 transparent valuation heat score. |
| valuation_warning_* | bool | filing/quarterly-ish | Rule flags used in bubble_signals.py. |

### technical_signals / bubble_signal_scores

| Column | Type | Frequency | Notes |
|---|---|---|---|
| date | datetime | daily | Trading date. |
| ticker | string | daily | Asset ticker. |
| sma_50 | float | daily | 50-day moving average. |
| sma_200 | float | daily | 200-day moving average. |
| weekly_rsi | float | daily forward-filled | RSI computed from weekly closes. |
| drawdown_pct | float | daily | Drawdown from trailing peak. |
| distance_from_200dma | float | daily | Percent above/below 200DMA. |
| zscore_price | float | daily | Price z-score vs 252-day mean/std. |
| volatility_30d | float | daily | Annualized 30-day volatility. |
| warning_score | int | daily | 0-100 rule score. |
| recovery_score | int | daily | 0-100 rule score. |
| signal_label | string | daily | Human-readable signal. |
| phase_label | string | daily | Bubble phase classification. |

### strategy_backtests

The simulator returns this structure in memory and dashboard tables:

| Column | Type | Notes |
|---|---|---|
| ticker | string | Asset. |
| strategy | string | Rule name. |
| start_date | datetime | Backtest start. |
| end_date | datetime | Backtest end. |
| starting_capital | float | Initial cash. |
| monthly_contribution | float | Monthly contribution. |
| ending_value | float | Final portfolio value. |
| total_return_pct | float | Return relative to contributed capital. |
| annualized_return_pct | float | Annualized return. |
| max_drawdown_pct | float | Worst portfolio drawdown. |
| volatility_pct | float | Annualized volatility. |
| sharpe_ratio | float | Simple zero-risk-free Sharpe approximation. |
| time_to_recover_days | int/null | Days from max trough to recovery. |
| trades | int | Number of buy/sell executions. |
| notes | string | Strategy caveats. |

## 6. Signal Definitions

### Technical exit/reduce exposure signals

- Price > 50% above 200-day moving average.
- Price > 25% above 200-day moving average.
- Weekly RSI > 75.
- Price z-score > 2.5.
- Parabolic acceleration: 63-day return materially exceeds one-quarter of 252-day return.
- Weekly RSI was >75 and then falls below 70.
- 50-day moving average crosses below 200-day moving average.
- Price falls more than 20% from prior peak after elevated warning score.

### Technical entry/re-entry signals

- Weekly RSI recovers above 40 after weakness.
- Price reclaims the 200-day moving average after a major drawdown.
- Drawdown begins stabilizing after being deeper than 20%.
- 63-day return turns positive after crash conditions.
- 50-day moving average crosses above 200-day moving average.

### Macro signals

Tracked but not forced into the first score:

- Federal Funds Rate trend.
- 10-year Treasury yield.
- 10-year minus 2-year yield spread.
- CPI inflation.
- Unemployment rate.
- GDP.
- 30-year mortgage rates.
- Case-Shiller home prices.
- NBER recession indicator.
- Financial stress index.

### Valuation signals

Supported now:

- SEC-derived historical P/S, P/E, EV/S, revenue TTM, net income TTM, shares outstanding, cash, and debt where companyfacts exposes clean concepts.
- P/S ratio relative to the asset's own prior expanding historical median.
- EV/S ratio relative to the asset's own prior expanding historical median.
- High P/E flag for profitable companies.
- Revenue-growth versus valuation-growth gap.
- Current yfinance snapshots are stored separately as context only, not used as historical evidence.

## 7. Backtesting Rules

Real-time-safe strategies:

- `buy_and_hold`
- `sell_exit_buy_recovery`
- `cash_until_40pct_drawdown`
- `cash_until_60pct_drawdown`
- `cash_until_rsi_recovery`
- `cash_until_reclaim_200dma`
- `dca_after_40pct_drawdown`

Retrospective benchmark strategies:

- `buy_6_months_after_retrospective_peak`
- `buy_12_months_after_retrospective_peak`

The retrospective peak rules are explicitly labeled as hindsight benchmarks. They are useful for teaching but not live trading.

## 8. Machine Learning Extension

Do not force ML in version 1. Rule-based analysis is more transparent and easier to defend.

If ML is later added, suitable tasks are:

1. Bubble phase classification using known historical regimes.
2. Crash-risk scoring using features available at time t.
3. Recovery detection after severe drawdowns.
4. Clustering bubble paths by acceleration/crash/recovery shape.

Rules:

- Use time-series split, not random split.
- Avoid using future peak/trough labels as current features.
- Compare ML against the rule-based benchmark.
- Use interpretable features and show feature importance.

## 9. Local Run Instructions

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python data_ingestion.py
python feature_engineering.py
python bubble_signals.py
streamlit run dashboard.py
```

Optional narrower ingestion for faster testing:

```bash
python data_ingestion.py --start 1995-01-01 --tickers QQQ ^IXIC CSCO INTC MSFT AMZN NVDA AMD AVGO PLTR SMH SOXX SPY XLF XHB BTC-USD ETH-USD ARKK
```

## 10. Strategic Research Summary

The current AI cycle should be treated as unresolved. The dashboard classifies it through evidence labels such as Normal Growth, Heating Up, Euphoria, Profit-Taking / Breakdown Risk, Breakdown, Capitulation, and Recovery.

The status should be interpreted as a research signal. For example, a stock can be in Euphoria based on technical heat while still having strong business fundamentals. Conversely, a strong company can still deliver poor forward returns if expectations and valuation become too extreme.

The highest-quality conclusion this project can produce is not "buy" or "sell." It is a structured statement such as:

> Historically, assets with similar levels of price extension, RSI heat, valuation expansion, and macro tightening experienced higher drawdown risk, but outcomes varied widely by profitability, rates, and earnings durability.

## 11. Limitations and Risks

- Free historical data can be incomplete for delisted companies.
- Point-in-time fundamentals require careful filing-date alignment.
- Yahoo/yfinance is convenient but unofficial and can change behavior.
- FRED macro data may be revised after initial release.
- Macro indicators are lower frequency than daily markets.
- Strategy rules can produce false positives and false negatives.
- Historical bubbles differ structurally from the AI cycle.
- A dashboard can organize evidence, but it cannot eliminate uncertainty.


## Supervised Machine Learning Crash-Risk Upgrade

### Objective

The ML layer estimates historical forward drawdown risk. It does not predict exact tops, bottoms, or calendar dates. The central supervised target is `burst_6m`, which equals 1 if an asset falls at least 30% from the current weekly close at any point over the next 26 weekly observations.

### Row Grain

One row equals one ticker during one weekly period. Weekly rows reduce daily noise and autocorrelation while preserving trend, RSI, drawdown, valuation, and macro conditions.

### Target Construction

Future prices are used only to construct labels for historical training rows:

- `burst_3m`: 20% or worse forward drawdown within 13 weeks.
- `burst_6m`: 30% or worse forward drawdown within 26 weeks.
- `burst_12m`: 40% or worse forward drawdown within 52 weeks.

Rows near the end of the dataset have missing labels because the future window has not completed. Those rows are valid for inference but not for model training.

### Feature Groups

The model uses date-t features only:

- Technical: moving-average distance, RSI, drawdown, volatility, trailing returns, z-score, acceleration.
- Rule-based scores: `warning_score`, `recovery_score`.
- Valuation: SEC-derived valuation metrics when available. The pipeline still runs if valuation coverage is missing.
- Macro: FRED indicators merged with backward as-of joins to avoid seeing future macro data.

### Models

- Rule-based baseline using `warning_score`.
- Logistic Regression / GLM-style baseline for interpretability.
- Random Forest for nonlinear threshold interactions.
- XGBoost for boosted-tree tabular performance.

No neural network is included because the number of true historical crash regimes is limited and interpretability matters more here.

### Validation

The training script uses chronological splitting, not random splitting. Random splits are invalid for this project because they can let later market regimes influence training before evaluating on earlier rows.

### Dashboard Page

The Streamlit app includes a `Machine Learning Crash-Risk Lab` page with model comparison metrics, confusion matrix, feature importance, current AI-cycle ML risk scores, dataset diagnostics, and plain-English interpretation.

### Run Order

```bash
python data_ingestion.py
python feature_engineering.py
python bubble_signals.py
python ml_dataset.py
python model_training.py
python model_evaluation.py
python ml_inference.py
streamlit run dashboard.py
```

### Interpretation

A high score means the current setup historically resembles pre-drawdown regimes more than normal. It does not guarantee a crash, does not identify an exact date, and is not personalized financial advice.

## v7 Machine-Learning Upgrades

The crash-risk layer now includes the second-pass improvements requested after inspecting the first model results.

### 1. Top-decile risk analysis
The project now saves `data/processed/ml_top_decile_analysis.parquet`. For each model, model scope, and split, it asks: when the model ranked an observation in its highest-risk 10% of scores, what was the actual `burst_6m` event rate? This is often more useful than the default confusion matrix because a financial risk monitor may be used to rank risk rather than issue binary predictions.

### 2. Tuned thresholds
The project no longer uses `0.50` as the automatic alert threshold. Each model chooses a threshold on the validation split by maximizing F1. That threshold is then applied to the later chronological test split. This prevents the dashboard from pretending that 50% is a natural cutoff in a rare-event problem.

### 3. SEC valuation features
The pipeline is designed to include SEC EDGAR valuation features when run without `--skip-sec`. If EDGAR concepts are unavailable for a company/year, the ML system still runs on technical, macro, and rule-signal features. The dashboard surfaces this limitation instead of inventing missing valuation data.

### 4. Expanded historical ticker universe
The default universe now covers more historical bubble families: dot-com leaders, AI/semiconductor leaders, financial/housing-crisis names, commodities/energy proxies, crypto, and COVID-era speculative growth names. Free APIs may not return delisted symbols; those failures are logged and skipped.

### 5. Separate model scopes
In addition to the global model, `model_training.py` trains segment-specific models when enough data exists:

- `broad_index_etf`
- `mega_cap_ai_tech`
- `speculative_high_vol`

During inference, `ml_inference.py` uses the segment-specific model for the ticker when available. If not, it falls back to the global model.

### 6. Calibration
The project now saves `data/processed/ml_calibration_table.parquet`, which compares mean predicted probabilities to empirical event rates by probability decile. This helps answer whether a model score of 20% behaves like approximately 20% historical risk, rather than just being an arbitrary number.

### 7. Dashboard conclusion
`model_evaluation.py` now creates `data/processed/ml_dashboard_conclusions.parquet` and `reports/ml_model_summary.md`. The dashboard reports:

- best ROC-AUC model
- best PR-AUC model
- best recall model
- best precision model
- an overall strength label: promising, modest, weak, or very weak/unreliable

The model remains an experimental research tool, not a trading recommendation.

## v12 Rare-Event Modeling Extension

The v12 pipeline adds stricter rare-event model evaluation and two additional modeling frames.

### Binary classifier target

The main row-level model now uses `burst_6m_segment`, a segment-adjusted target. This preserves the original `burst_6m` but avoids treating SPY, NVDA, PLTR, and BTC as if they share the same natural drawdown anatomy.

### Rare-event audit

`rare_event_analysis.py` audits saved model predictions using constrained thresholds, percentile buckets, top-k lift, and validation-based calibration. This prevents misleading improvements from low thresholds that produce high recall by flagging nearly every row.

### Hazard model

`hazard_model.py` implements a discrete-time hazard-style classifier for near-term entry into a major drawdown state. It asks whether an asset enters a segment-adjusted major drawdown within the next four weeks.

### Poisson count model

`poisson_count_model.py` models counts by date and segment. It answers a separate question: how many assets in a segment are expected to experience major future drawdowns? It does not replace the binary ticker-week models.

## v13 Rare-Event Robustness Plan

The v13 upgrade adds a final audit layer designed to test whether rare-event methods genuinely improve historical drawdown-risk detection or merely inflate recall.

### Additions

1. **MCC and false-positive burden metrics**
   - Added to model-training and rare-event result tables.
   - Makes degenerate threshold behavior easier to detect.

2. **Balanced Random Forest**
   - Added to the main model set when `imbalanced-learn` is available.
   - Preferred before SMOTE because it does not synthesize artificial market states.

3. **SMOTE / SMOTE-ENN controlled experiments**
   - Implemented in `imbalance_experiments.py`.
   - Applied only inside the training split after chronological splitting.
   - Validation/test data are never resampled.

4. **Event-level validation**
   - Implemented in `event_validation.py`.
   - Tests whether models trained before a crisis can rank the pre-event/event window as risky.
   - Reduces the illusion that many correlated ticker-week rows equal many independent crises.

5. **Cost-ratio threshold stress tests**
   - Implemented in `cost_threshold_analysis.py`.
   - Tests false-negative costs of 1x, 2x, 5x, 10x, and 20x.

6. **Optional Firth logistic regression**
   - Implemented as an export workflow in `firth_logistic_export.py` and `firth_logistic_optional.R`.
   - Kept optional because mature Firth logistic tooling is more reliable in R than Python.

### Acceptance standard

A method is useful only if it improves out-of-time and event-level risk ranking without creating an unacceptable false-positive burden. Stronger recall alone is not considered an improvement.


## v14 Reliability and Signal-Confirmation Patch

The v14 package changes the dashboard interpretation layer from binary prediction toward signal confirmation:

1. The confusion matrix is target-column aware and no longer assumes `burst_6m` exists in every artifact.
2. Risk-bucket visuals use non-stacked lines so model event rates are not added together.
3. Hazard modeling includes a leakage audit comparing full state-aware features against restricted no-state features.
4. Raw probabilities are described as raw model scores unless calibration supports probability interpretation.
5. The dashboard includes a current AI signal-confirmation table that combines ML risk bucket, hazard/onset score, rule phase labels, warning score, and recovery score.

This keeps the project honest: the model is useful only when it improves hard-to-fake ranking evidence such as PR-AUC, MCC, top-risk lift, event-level validation, and calibrated empirical event rates.
