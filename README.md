# Economic Bubble Monitoring & Investment Strategy Dashboard

A Python + Streamlit research dashboard for studying historical bubble cycles, rule-based bubble warning/recovery signals, SEC-derived valuation history, supervised machine-learning resemblance models, rare-event diagnostics, hazard/onset models, segment-level Poisson count models, and educational strategy backtests.

> **This dashboard studies historical bubble-like market regimes and estimates whether current or past asset conditions resemble historically higher-risk drawdown regimes. It is a research signal and model-interpretation tool, not a trading oracle. Nothing in this repository is personalized financial advice.**

---

## 1. Project overview

This project began as a historical bubble analysis dashboard. It has grown into a richer research stack that combines:

- Public market and macro data ingestion
- SEC EDGAR filing-date-aware valuation/fundamental features (where coverage exists)
- Transparent rule-based bubble warning and recovery scores
- Bubble phase labels and historical bubble comparison tools
- Strategy/backtesting utilities
- Supervised machine learning models for forward drawdown risk
- Rare-event diagnostics, top-k lift, percentile risk buckets, calibration audits
- Imbalance-method experiments (Balanced RF, SMOTE, SMOTE-ENN)
- Event-level (leave-one-crisis-style) validation
- Cost-threshold stress tests
- Discrete-time hazard / onset modeling with leakage audit
- Segment-level Poisson count modeling
- Current AI-cycle ML risk scoring
- A Streamlit dashboard that surfaces every layer with interpretations, expanders, and signal confirmation

## 2. Research question

> **Can historical bubble-like market patterns help identify data-driven warning, risk-ranking, and re-entry signals for a possible current or future AI-related market cycle?**

The dashboard studies past regimes including the dot-com bubble, the housing/credit crisis, crypto cycles, COVID-era speculation, the AI / mega-cap technology cycle, and (optionally) commodity, Japan, and China cycles.

The framing is intentionally narrow: **ranking and resemblance, not prophecy**.

## 3. What the dashboard *does*

- Compares historical bubble episodes on a normalized basis (peak = 100).
- Renders transparent rule-based warning and recovery scores per ticker.
- Surfaces an ML "resemblance" score with explicit `model_scope` (global vs segment).
- Reports top-5% / top-10% bucket lift, calibration curves, false-positive burden, and event-level validation.
- Combines rule, ML, and hazard layers into a per-ticker **signal confirmation table**.
- Lets the user backtest several historical exit/re-entry strategies.

## 4. What the dashboard explicitly *does not* claim

- It does **not** predict the exact top, bottom, or crash date of any asset.
- It does **not** claim that the current AI cycle is or is not a bubble.
- It does **not** issue personalized buy/sell recommendations.
- It does **not** guarantee that historical signals will continue to work in the future.

## 5. Repository structure

```
.
├── dashboard.py                   # Streamlit page router (14 narrative sections)
├── app.py                         # Thin wrapper — `streamlit run app.py` also works
├── dashboard_helpers.py           # Loaders, ROC math, formatters, missing-artifact notice
├── dashboard_explanations.py      # All long-form explanations + metric glossary
├── dashboard_visuals.py           # Chart builders + interpretation-box + regime badge
├── interpretation_engine.py       # Regime classifier + signal confirmation + research story
│
├── config.py                      # Bubble definitions, AI leaders, FRED series, paths
├── data_ingestion.py              # yfinance + FRED + SEC EDGAR ingestion
├── feature_engineering.py         # Technicals, returns, drawdowns, macro features
├── bubble_signals.py              # Rule-based warning/recovery scores + phase labels
├── backtesting.py                 # Strategy simulator (used by Investment Simulator page)
├── ml_dataset.py                  # Weekly ML dataset + segment-adjusted `burst_6m_segment`
├── model_training.py              # Trains LR, Elastic Net LR, RF, Balanced RF, XGBoost
├── rare_event_analysis.py         # Top-k lift, percentile buckets, rare-event calibration
├── imbalance_experiments.py       # Balanced RF / SMOTE / SMOTE-ENN stress tests
├── event_validation.py            # Leave-one-crisis-style event validation
├── cost_threshold_analysis.py     # Cost-ratio threshold stress tests
├── firth_logistic_export.py       # Exports clean CSVs for optional R Firth-logistic
├── firth_logistic_optional.R      # Optional R script — Firth logistic benchmark
├── hazard_model.py                # 4-week onset model + leakage audit
├── poisson_count_model.py         # Segment-level Poisson count regression
├── ml_inference.py                # Scores current AI-exposed assets with trained models
├── model_evaluation.py            # Final research-story / written summaries
├── sec_fundamentals.py            # Standalone SEC EDGAR refresh utility
│
├── data/                          # Generated — git-ignored
│   ├── raw/                       #   raw downloads
│   ├── processed/                 #   parquet artifacts consumed by the dashboard
│   └── bubble_dashboard.duckdb    #   optional DuckDB layer
├── models/                        # Generated — git-ignored — trained .pkl models
├── logs/                          # Generated — git-ignored
├── reports/                       # Generated markdown research summaries (tracked)
├── requirements.txt
├── .env.example                   # Template for SEC_USER_AGENT and optional API keys
└── README.md                      # This file
```

## 6. Full pipeline run order

Run from the project folder, in this order (each script consumes outputs from the previous step):

```bash
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

Optional / standalone:

```bash
python sec_fundamentals.py     # Refresh SEC EDGAR fundamentals only
```

Primary dashboard entrypoint:

```bash
streamlit run dashboard.py
# or (equivalent)
streamlit run app.py
```

A future, optional second-pass experiment (`python feature_transformation_experiments.py`) is **not yet implemented** in this branch and will be added in a separate Phase 8 commit. When added, the recommended position is **after `python model_training.py`**.

## 7. Virtual environment setup

Python 3.12 recommended:

```bash
py -3.12 -m venv .venv_v16
.venv_v16\Scripts\activate          # Windows PowerShell: .\.venv_v16\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 8. `.env` setup for SEC EDGAR access

SEC EDGAR requires a user-agent string identifying your tool and contact email. Copy `.env.example` to `.env` and edit:

```dotenv
SEC_USER_AGENT=EconomicBubbleDashboard/1.0 your_email@example.com
SEC_REQUEST_PAUSE_SECONDS=0.25
# Optional:
FRED_API_KEY=
FMP_API_KEY=
```

The dashboard will load the file via `python-dotenv` at startup.

## 9. Generated folders ignored by Git

The repository ships **only the code and the small markdown reports**. Everything below is generated locally and excluded from version control:

| Path                              | Generated by                        | Why ignored |
|-----------------------------------|-------------------------------------|-------------|
| `data/raw/`                       | `data_ingestion.py`, `sec_fundamentals.py` | Large, regenerable, may contain dated upstream snapshots |
| `data/processed/`                 | the rest of the pipeline            | Tens of parquets, large, regenerable |
| `data/bubble_dashboard.duckdb`    | `data_ingestion.py` and friends     | Local SQL convenience |
| `models/`                         | `model_training.py`, `hazard_model.py`, etc. | Pickle binaries |
| `logs/`                           | every script                        | Run-time logs |
| `.venv*/`, `__pycache__/`, `.env` | local                                | Environment / secrets |

If you clone this repo on a new machine, you will see a clean tree with only source files. Run the pipeline above to populate `data/processed/`, `models/`, and `reports/`.

## 10. Dashboard tab guide

The sidebar follows a narrative flow of 14 numbered sections:

| # | Section | Purpose |
|---|---|---|
| 1 | **Executive Summary** | Top-line story, key findings drawn from `ml_dashboard_conclusions.parquet`, supporting tables, project disclaimer. |
| 2 | **Data Overview & EDA** | Row counts, date coverage, ticker coverage, segment distribution, target balance, distributions, correlation heatmap, missingness. |
| 3 | **Bubble Explorer** | Per-ticker price + SMAs + rule-based exit/re-entry markers, drawdown, RSI, macro overlay. Includes the regime badge. |
| 4 | **Historical Bubble Comparison** | Normalized price paths aligned by retrospective peak across selected bubbles. |
| 5 | **Bubble Vital Signs** | Single-asset scorecard (warning/recovery/drawdown/RSI/volatility/momentum) with regime badge and SEC valuation history. |
| 6 | **Current AI Cycle Monitor** | Cross-signal confirmation table, rule-based monitor, momentum heatmap, latest SEC valuation rows for AI leaders. |
| 7 | **Machine Learning Model Lab** | Best-by-metric cards; model comparison; top-decile risk; thresholds & calibration; confusion matrices; ROC; FP audit; feature importance; plain-English summary. |
| 8 | **Rare Event & Imbalance Lab** | Rare-event thresholds + metrics; percentile risk buckets; top-5/10% lift; rare-event calibration; SMOTE / Balanced RF experiments; cost-threshold stress test. |
| 9 | **Event-Level Validation** | PR-AUC per historical regime trained-before / tested-on the regime window. |
| 10 | **Hazard & Poisson Models** | 4-week hazard with full + restricted feature sets + leakage audit; segment-level Poisson count. |
| 11 | **Investment / Strategy Simulator** | Buy-and-hold vs rule-based exit/re-entry strategies. Backtest only. |
| 12 | **Methods & Diagrams** | Pipeline Sankey, label construction, threshold + metric formulas, signal-combination logic. |
| 13 | **Literature / Background Research** | Curated reference list — bubbles, crashes, asset pricing, ML caveats, rare-event modelling. |
| 14 | **Data Quality & Limitations** | Survivorship bias, SEC mapping risk, macro frequency mismatch, false positives/negatives, model uncertainty, table-by-table check. |

Every chart on every page is followed by an **interpretation box** with four parts: what the chart shows, how to read it, what the current result suggests, and what *not* to overclaim.

## 11. Model explanation summary

The project includes several model families because the "best" depends on the decision goal:

| Family | Type | Where it tends to lead |
|---|---|---|
| **Rule-Based Warning Score** | Hand-crafted scorecard | Highest precision on `broad_index_etf`. Transparent — you can read the rule that fired. |
| **Logistic Regression** | Linear classifier | Usually the strongest broad risk-ranker by ROC-AUC and PR-AUC on `global`. |
| **Elastic Net Logistic** | Regularized linear | Best recall on `mega_cap_ai_tech` in current runs; better feature selection. |
| **Random Forest** | Tree ensemble | Captures non-linear interactions but rarely dominates here. |
| **Balanced Random Forest** | Class-balanced bootstraps | Useful for top-risk bucket detection in AI/tech and speculative segments. |
| **XGBoost** | Gradient-boosted trees | Strong benchmark; included to demonstrate it does *not* automatically win on this data. |
| **Firth Logistic (optional, R)** | Penalized logistic | Useful for small-sample / separation-prone configurations. |
| **Discrete-time Hazard** | Event-onset | Asks 'did the asset enter a major drawdown state within the next 4 weeks?' — requires the leakage audit. |
| **Poisson Count** | Count regression | Estimates *how many* assets in a segment may experience a drawdown; currently experimental. |

**Why no neural network is central.** The bottleneck is not model capacity — it is data scarcity (few independent crises), non-stationarity, and label noise. A deep model would over-fit unless validation is much stricter than row-level chronological splits.

## 12. Interpretation of metrics

Every metric used in the dashboard has a glossary entry inside the app (see the **Methods & Diagrams → Thresholds & metrics** tab, and the "📐 Metric glossary" expander on the ML Lab page). Headline reminders:

- **PR-AUC** matters more than ROC-AUC when positive events are rare.
- **Accuracy** is misleading when the positive class is under 10% of rows.
- **MCC** is the most imbalance-robust headline metric.
- **Top-5% / Top-10% lift** is often more decision-useful than any binary metric.
- **False positives per true positive** is the honesty meter for alert systems.
- **Empirical bucket event rate** should be quoted instead of the raw model probability when calibration is imperfect.

## 13. Current key findings

Findings change with each pipeline re-run, but the general pattern is:

1. **Logistic Regression / `global`** is usually the most stable broad risk-ranker by ROC-AUC and PR-AUC.
2. **Elastic Net Logistic / `mega_cap_ai_tech`** shows the best recall in the AI / mega-cap segment.
3. **Rule-Based Warning Score / `broad_index_etf`** often shows the best precision (cleanest alerts).
4. **Balanced Random Forest** helps top-risk bucket detection in some AI/tech and speculative segments.
5. **XGBoost** is competitive as a benchmark but does not dominate.
6. **Top-bucket lift** is often the clearest evidence — empirical event rates rise from bottom to top buckets when the model is doing useful work.
7. **Calibration is imperfect.** Raw scores should be treated as resemblance scores, not literal probabilities. Prefer the empirical bucket event rate.
8. **Hazard model** looks unusually strong before the leakage audit and noticeably weaker after — use the **restricted** hazard score.
9. **Poisson count model** is conceptually useful but currently experimental; do not overclaim.
10. **"Best model"** depends on the decision goal — see the in-app "🏆 Which model is best?" expander.

## 14. Limitations

- **Few independent crises.** Even with hundreds of thousands of rows, only ~5–10 distinct historical crisis regimes are usable. Most positive rows are correlated neighbors of the same event.
- **Survivorship bias.** Delisted names (Lehman, Bear Stearns, many SPACs) may be missing.
- **SEC valuation mapping risk.** XBRL concepts differ by company and era — coverage is partial.
- **Macro frequency mismatch.** Daily prices vs quarterly GDP vs monthly CPI; resampling can hide leading/lagging behaviour.
- **Non-stationarity.** Features that worked in past crises (curve inversion, leverage, valuation extremes) may stop working when policy or market structure changes.
- **Calibration drift.** Raw probabilities should not be treated as literal forecasts.
- **No exact top/bottom detection.** This dashboard tests rules and ranks resemblance; it does not promise timing.

## 15. Future improvements

- Optional **`feature_transformation_experiments.py`** (Phase 8, separate branch): controlled experiment testing log/signed-log transforms, winsorization, rolling z-scores by ticker, percentile ranks, economically meaningful interactions, regime-adjusted features, volatility-adjusted returns, and drawdown-state flags — all with strict no-look-ahead discipline.
- Richer event-level validation with bootstrapped confidence intervals.
- A second R-language benchmark using `survival` for the hazard layer.
- Expanded SEC coverage via fallback XBRL concept maps.
- Optional fine-tuned text sentiment features for the AI-cycle monitor.

## 16. Optional transformed-feature experiment (planned, not yet implemented)

To be added in a later commit as `feature_transformation_experiments.py`. Goals (per the project specification):

- Test economically meaningful transformations (log / signed-log, winsorization, rolling z-scores by ticker, percentile ranks, interactions, regime-adjusted features, volatility-adjusted returns, drawdown-state flags).
- Fit the same model families on transformed features without overwriting baseline artifacts.
- Save outputs separately:
  - `data/processed/transformed_feature_model_results.parquet`
  - `data/processed/transformed_feature_predictions.parquet`
  - `data/processed/transformed_feature_importance.parquet`
  - `reports/transformed_feature_experiment_summary.md`
- Compare baseline vs transformed on: PR-AUC, ROC-AUC, MCC, top-5/10% lift, calibration, false-positive burden, and event-level validation.
- Keep the framing honest: "Transformations may improve feature geometry and make patterns easier for models to learn, but they do not create new independent historical bubbles."

This experiment is **deferred** until the dashboard refactor and bug-fix phases are stable.

---

## Quick links

- **Run the dashboard:** `streamlit run dashboard.py`
- **Project disclaimer:** see `dashboard_explanations.PROJECT_DISCLAIMER` (rendered on the Executive Summary page).
- **Regime badge logic:** see `interpretation_engine.classify_regime_badge`.
- **Signal confirmation logic:** see `interpretation_engine.build_signal_confirmation_table`.
- **Metric glossary:** see `dashboard_explanations.METRIC_GLOSSARY`.
