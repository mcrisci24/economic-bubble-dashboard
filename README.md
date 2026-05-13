# Economic Bubble Monitoring & Investment Strategy Dashboard

A Python + Streamlit research dashboard for studying historical bubble cycles, bubble warning/recovery signals, SEC-derived valuation history, and educational strategy backtests.

**This is not financial advice.** It is an educational and research-oriented backtesting project.

## Files

- `data_ingestion.py`

Downloads market prices, macro indicators, and SEC valuation/fundamental data where available.


- `feature_engineering.py`

Builds technical indicators, returns, drawdowns, volatility, moving averages, valuation features, and macro-aligned features.

- `bubble_signals.py`

Creates transparent rule-based warning scores, recovery scores, and phase labels.

- `backtesting.py`

Runs historical strategy/backtesting logic used by the Investment Simulator and strategy comparison outputs.

- `ml_dataset.py`

Builds the supervised machine-learning dataset and creates forward drawdown target variables such as burst_6m_segment.

- `model_training.py`

Trains the main supervised models: Logistic Regression, Elastic Net Logistic, Random Forest, Balanced Random Forest, XGBoost, and the rule-based baseline.

- `rare_event_analysis.py`

Adds rare-event threshold tuning, risk buckets, calibration curves, and top-5% / top-10% lift analysis.

- `imbalance_experiments.py`

Tests imbalance-specific methods such as Balanced Random Forest, SMOTE, and SMOTE-ENN.

- `event_validation.py`

Runs event-level / leave-one-crisis-style validation to test whether models generalize across major historical market episodes.

- `cost_threshold_analysis.py`

Tests threshold choices under different false-positive and false-negative cost assumptions.

- `firth_logistic_export.py`

Exports datasets for optional Firth Logistic Regression benchmarking in R.

- `hazard_model.py`

Trains the discrete-time hazard/onset model and performs leakage auditing.

- `poisson_count_model.py`

Trains the segment-level Poisson count model for estimating how many assets in a segment may enter major drawdown conditions.

- `ml_inference.py`

Scores the latest/current AI-exposed assets using the trained models.

- `model_evaluation.py`

Creates final model summaries, dashboard conclusions, interpretation tables, and research-story outputs.

- `dashboard.py`

Launches the Streamlit dashboard.



## Quickstart

```bash
py -3.12 -m venv .venv_v12

 or 
 
python -m venv .venv  (if py doesn't work)
 
.\.venv_v12\Scripts\Activate.ps1 
 
  or
 
.venv\Scripts\Activate.ps1     # Windows PowerShell

# source .venv/bin/activate     # macOS/Linux


pip install -r requirements.txt

python data_ingestion.py
python feature_engineering.py
python bubble_signals.py
python backtesting.py
python ml_dataset.py
python model_training.py
python rare_event_analysis.py
python imbalance_experiments.py
python event_validation.py
python cost_threshold_analysis.py
python firth_logistic_export.py
python hazard_model.py
python poisson_count_model.py
python ml_inference.py
python model_evaluation.py
streamlit run dashboard.py
```


## Machine Learning Crash-Risk Layer

The machine-learning layer turns the dashboard into a supervised historical risk-modeling project. It does **not** predict the exact date of a bubble burst. It estimates whether current weekly conditions resemble historical conditions that were followed by a major forward drawdown.

Primary target:

```text
burst_6m = 1 if the asset falls at least 30% from the current weekly close at any point over the next 26 weekly observations; otherwise 0.
```

Optional targets created by `ml_dataset.py`:

- `burst_3m`: 20% or worse forward drawdown within 13 weeks.
- `burst_6m`: 30% or worse forward drawdown within 26 weeks.
- `burst_12m`: 40% or worse forward drawdown within 52 weeks.

Models compared:

1. Rule-based baseline using `warning_score`.
2. Logistic Regression as the interpretable GLM-style baseline.
3. Elastic Net Logistic Regression as a sparse/shrinkage linear benchmark.
4. Random Forest Classifier for nonlinear threshold interactions.
5. XGBoost Classifier for boosted-tree tabular modeling.

The model uses a chronological train/validation/test split rather than random splitting. Random splitting would leak future market regimes into training and overstate performance. Current AI-cycle rows are treated as inference/monitoring rows when their future six-month outcomes are not fully known.

Run the ML pipeline after the existing data/signals pipeline:

```bash
python ml_dataset.py
python model_training.py
python model_evaluation.py
python ml_inference.py
streamlit run dashboard.py
```

Outputs:

- `data/processed/ml_burst_dataset.parquet`
- `data/processed/ml_model_results.parquet`
- `data/processed/ml_test_predictions.parquet`
- `data/processed/ml_feature_importance.parquet`
- `data/processed/ml_thresholds.parquet`
- `data/processed/ml_top_decile_analysis.parquet`
- `data/processed/ml_calibration_table.parquet`
- `data/processed/ml_dashboard_conclusions.parquet`
- `data/processed/current_ai_ml_risk_scores.parquet`
- `reports/ml_model_summary.md`
- trained model files under `models/`

Interpretation rule:

A high ML score means the current feature pattern historically appeared before large forward drawdowns more often than normal. It does **not** mean a crash is guaranteed, and it is not financial advice.


### v7 ML upgrades

The ML layer now includes the requested second-pass improvements:

- **Top-decile risk analysis:** shows what happened historically when a model ranked a row in its highest-risk 10% of scores.
- **Tuned thresholds:** models no longer use `0.50` as the automatic warning cutoff. Each threshold is selected on validation data by maximizing F1, then applied to the later test split.
- **SEC valuation features:** run `data_ingestion.py` without `--skip-sec` to activate filing-date-aware SEC valuation features where EDGAR concepts are available.
- **Expanded ticker universe:** the default project universe now includes more dot-com, AI/semi, financial/housing, commodity, and speculative growth tickers. Some delisted/unavailable symbols will be logged and skipped.
- **Separate model scopes:** the project trains a global model and, when enough labeled rows exist, separate models for broad index/ETF, mega-cap AI/tech, and speculative/high-volatility assets.
- **Calibration tables:** the dashboard compares predicted probability bins with empirical event rates, so you can see whether a 20% predicted risk behaves like roughly 20% historically.
- **Explicit dashboard conclusion:** the ML page now identifies best ROC-AUC, best PR-AUC, best recall, best precision, and whether the model strength is strong, weak, or still experimental.


## New dashboard pages in this version
- **Research Story & Interpretation**: dynamic narrative interpretation of the saved results, including whether the model story is meaningful or weak.

- **ML Visual Diagnostics**: all confusion matrices, ROC-AUC curve views, and false-positive audit visuals.
- **EDA Visuals**: target balance, segment balance, distributions, correlations, and missingness charts.
- **Feature Importance Visuals**: ranked bars, signed coefficient plots, heatmaps, cumulative-importance charts, and feature-theme summaries.
- **Methods, Diagrams, and Formula Walkthrough**: process diagrams, script flow, label construction, and metric explanations.
- **Background Research and Literature Review**: a curated list of bubble/crisis/financial-ML references.

## SEC EDGAR setup

The project now pulls historical valuation rows from SEC EDGAR companyfacts for eligible corporate tickers. Set a polite identifiable user agent in `.env`:

```text
SEC_USER_AGENT="EconomicBubbleDashboard/1.0 your.email@example.com"
SEC_REQUEST_PAUSE_SECONDS=0.25
```

The code caches SEC companyfacts JSON under `data/processed/sec_companyfacts_cache/`, so repeated runs do not hammer SEC servers.

If you only want to debug price/macro ingestion, you can skip SEC temporarily:

```bash
python data_ingestion.py --skip-sec
```

For a focused SEC valuation refresh after prices already exist:

```bash
python sec_fundamentals.py --tickers NVDA MSFT AMD AVGO PLTR META GOOGL AMZN --refresh
python feature_engineering.py
python bubble_signals.py
python ml_dataset.py
python model_training.py
python model_evaluation.py
python ml_inference.py
streamlit run dashboard.py
```

## Optional FRED API key

The default pipeline fetches FRED data through public CSV endpoints, so a FRED key is not required. If you later add `fredapi`, create a `.env` file:

```text
FRED_API_KEY=your_key_here
```

Get a key from FRED's API page.

## Faster test run

```bash
python data_ingestion.py --start 1995-01-01 --tickers QQQ ^IXIC CSCO INTC MSFT AMZN NVDA AMD AVGO PLTR SMH SOXX SPY XLF XHB BTC-USD ETH-USD ARKK
python feature_engineering.py
python bubble_signals.py
python ml_dataset.py
python model_training.py
python model_evaluation.py
python ml_inference.py
streamlit run dashboard.py
```

## Notes

- Delisted tickers such as Yahoo/AOL may fail through free APIs. The ingestion script logs and skips unavailable symbols.
- SEC valuation rows are filing-date-aware and designed to reduce look-ahead bias.
- Current yfinance valuation snapshots are stored separately as current context only, not as historical evidence.
- Retrospective peak strategies are hindsight benchmarks, not live trading signals.

## v12 Rare-Event Upgrade

This version adds a rare-event modeling layer on top of the existing crash-risk ML system.

### What changed

1. **Models preserved:** Logistic Regression, Random Forest, and XGBoost remain the main row-level classifiers.
2. **Stricter threshold tuning:** `rare_event_analysis.py` applies minimum precision constraints and maximum alert-rate guards so a model cannot look good merely by predicting almost everything as risky.
3. **Percentile risk buckets:** the dashboard now evaluates Bottom 50%, 50-75%, 75-90%, 90-95%, and Top 5% risk buckets.
4. **Calibration:** validation-bin empirical event rates are used to create calibrated probability estimates and reliability curves.
5. **Top-decile and top-5% lift:** the dashboard now shows whether the highest-risk 10% and 5% of model scores contain more realized future drawdowns than the base event rate.
6. **Segment-specific targets:** the dataset now keeps the original uniform targets and adds segment-adjusted targets. The main ML training target is now `burst_6m_segment`:
   - broad indexes/ETFs: 20% six-month forward drawdown threshold
   - mega-cap AI/tech: 30% six-month forward drawdown threshold
   - speculative/high-volatility assets: 50% six-month forward drawdown threshold
   - other single names: 30% six-month forward drawdown threshold
7. **Discrete-time hazard model:** `hazard_model.py` models whether an asset enters a segment-adjusted major-drawdown state within the next four weeks.
8. **Poisson count model:** `poisson_count_model.py` is separate from the binary classifier and models the number of assets in a segment that later experience major drawdowns.

### Full v12 run order

```powershell
python data_ingestion.py
python feature_engineering.py
python bubble_signals.py
python ml_dataset.py
python model_training.py
python rare_event_analysis.py
python model_evaluation.py
python hazard_model.py
python poisson_count_model.py
python ml_inference.py
streamlit run dashboard.py
```

### Fast first run without SEC valuation

```powershell
python data_ingestion.py --tickers QQQ SPY NVDA MSFT AMD AVGO PLTR META GOOGL AMZN AAPL TSM ASML MU LRCX KLAC SMH SOXX BTC-USD ETH-USD ARKK TSLA COIN ROKU ZM SHOP XLF XHB JPM BAC C AIG --skip-sec
python feature_engineering.py
python bubble_signals.py
python ml_dataset.py
python model_training.py
python rare_event_analysis.py
python model_evaluation.py
python hazard_model.py
python poisson_count_model.py
python ml_inference.py
streamlit run dashboard.py
```

### How to interpret v12

The dashboard should be treated as a **historical risk-ranking research tool**, not a prediction machine. The most important outputs are PR-AUC, calibration, top-10% lift, top-5% lift, and risk-bucket event rates. A high score means the current pattern resembles historical higher-risk setups. It does not mean a crash will occur.

## v13 Rare-Event Robustness Upgrade

This version adds one final rare-event modeling audit layer on top of the v12 architecture.

### What is new

- **MCC and false-positive burden metrics** in model results so high recall cannot hide oceans of false alarms.
- **Balanced Random Forest** added to the main model set when `imbalanced-learn` is installed.
- **Controlled imbalance experiments** in `imbalance_experiments.py`:
  - Balanced Random Forest
  - SMOTE + Logistic Regression
  - SMOTE + Random Forest
  - SMOTE-ENN + Logistic Regression
  - SMOTE-ENN + Random Forest
- **Event-level validation** in `event_validation.py`, which tests pre-event/event windows such as dot-com, the global financial crisis, COVID crash, and 2021 speculative-tech unwind.
- **Cost-ratio threshold stress testing** in `cost_threshold_analysis.py`, which evaluates what happens when false negatives are treated as 2x, 5x, 10x, or 20x as costly as false positives.
- **Optional Firth logistic regression export** in `firth_logistic_export.py` plus `firth_logistic_optional.R` for users who want a statistical rare-event logistic benchmark in R.

### Full run order

```powershell
python data_ingestion.py
python feature_engineering.py
python bubble_signals.py
python ml_dataset.py
python model_training.py
python rare_event_analysis.py
python imbalance_experiments.py
python event_validation.py
python cost_threshold_analysis.py
python firth_logistic_export.py
python model_evaluation.py
python hazard_model.py
python poisson_count_model.py
python ml_inference.py
streamlit run dashboard.py
```

### Interpretation rule

Do **not** keep a method simply because it increases recall. Keep it only if it improves several hard-to-fake metrics:

- PR-AUC
- MCC
- top-5% and top-10% lift
- false positives per true positive
- calibration / reliability
- event-level validation behavior

The goal is not to make the model look better. The goal is to make the model harder to fool.


## v14 Signal-Confirmation and Reliability Fixes

v14 fixes several dashboard and modeling issues discovered during the v13 review:

- Fixes the dashboard `KeyError: 'burst_6m'` by automatically selecting `burst_6m_segment`, `burst_6m`, `target`, or `y_true` depending on the artifact.
- Replaces the misleading stacked risk-bucket bar chart with a line chart so model event rates are not visually added together.
- Adds a hazard-model leakage audit. `hazard_model.py` now trains both `full_features` and `restricted_no_state_features`. The restricted model removes target-adjacent state variables such as `drawdown_pct`, `distance_from_200dma`, `zscore_price`, `warning_score`, and `recovery_score`.
- Stops treating raw model scores as literal probabilities. The dashboard now emphasizes empirical bucket event rates, top-risk lift, and risk percentiles.
- Reframes the ML page as a signal-confirmation and risk-ranking lab rather than a binary crash prediction page.

### Recommended v14 run order

```powershell
python data_ingestion.py
python feature_engineering.py
python bubble_signals.py
python ml_dataset.py
python model_training.py
python rare_event_analysis.py
python imbalance_experiments.py
python event_validation.py
python cost_threshold_analysis.py
python firth_logistic_export.py
python hazard_model.py
python poisson_count_model.py
python ml_inference.py
python model_evaluation.py
streamlit run dashboard.py
```

`model_evaluation.py` is intentionally near the end so the final written interpretation can see the outputs from the rare-event, imbalance, event-validation, hazard, Poisson, and inference modules.

### v14 interpretation rule

Use the dashboard as a **historical signal-confirmation and risk-ranking system**. The strongest evidence is cross-signal agreement plus top-5%/top-10% lift and event-level validation. Do not interpret raw model scores as literal probabilities unless the calibration curve supports that interpretation.

## v16 interpretation and diagnostics patch

This version fixes a NumPy 2.x compatibility issue in the ROC visual diagnostics page by replacing the deprecated/removed `np.trapz` call with `np.trapezoid`.

It also adds a new dashboard page:

- **Research Story & Interpretation**: a dynamic narrative page that reads the saved model artifacts and explains whether there is a real story in the results. It summarizes top-risk bucket lift, ordinary model comparison, imbalance experiments, current AI empirical-risk rankings, hazard leakage audit findings, Poisson count model reliability, cost-threshold tradeoffs, and feature-importance themes.

The lift charts were also changed to grouped bars instead of stacked bars because lift values do not add across scopes or top-k buckets.



