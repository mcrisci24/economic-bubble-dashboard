# Speaker Script — Economic Bubble Monitoring Dashboard
## Timing guide: 20–25 minutes total + 5–10 minutes Q&A

---

### SLIDE 1 — Title (0:00–0:30)

"Thank you. Today I'm presenting the Economic Bubble Monitoring and Investment Strategy Dashboard — an end-to-end machine learning research pipeline built to study whether historical market data can detect conditions that precede major economic drawdowns.

Before I go further: everything here is an educational research project. Nothing I show constitutes investment advice, and I'll be honest throughout about what the models can and cannot do."

---

### SLIDE 2 — Research Question (0:30–2:00)

"The core question is: can we build a detector — not a predictor — that identifies when current market conditions resemble historical pre-crash regimes?

The key framing choice is the word 'detector.' Forecasting the exact timing of a crash is widely considered near-impossible. A more tractable goal is to ask: right now, does the data pattern look like what we saw before the dot-com collapse, the 2008 crisis, or the 2020 drawdown?

Our target label, burst_6m_segment, equals 1 if the asset crosses a segment-adjusted major drawdown threshold within 26 weekly observations — roughly 6 months. The segment-adjusted part is important and I'll explain it in a moment."

---

### SLIDE 3 — Dataset Architecture (2:00–3:30)

"The dataset spans 81 tickers, 597,806 weekly rows, from 1927 to May 2026. That sounds large, but I want to be upfront about a key limitation: the number of truly *independent* bubble events is much smaller — perhaps 15 to 30 globally. The rows are highly autocorrelated. One bubble regime can generate hundreds of positive-label rows for dozens of tickers simultaneously.

We use four asset segments because a −20% drawdown in SPY, the S&P 500 ETF, is a major market correction. The same −20% move in Bitcoin or a speculative tech fund is an ordinary week. Pooling them would corrupt the label."

---

### SLIDE 4 — Target Label (3:30–5:00)

"The segment thresholds are −20% for broad ETFs, −35% for mega-cap AI/tech, −50% for speculative assets, and −25% for the global pooled scope.

The split design is strictly chronological. The first 70% of unique dates becomes training, the next 15% validation, the last 15% test. No row is randomly shuffled. This is non-negotiable for time series — random shuffling would leak future information into training and produce absurdly optimistic metrics.

The positive rate ranges from 6.8% in mega-cap AI/tech to 11% in broad ETFs. This imbalance is the central challenge for our classifiers."

---

### SLIDE 5 — ML Pipeline (5:00–6:30)

"The pipeline has 6 stages. Raw price data from Yahoo Finance is ingested, then technically engineered into momentum, volatility, volume, and valuation features. The ML dataset is constructed with the segment-adjusted labels. The chronological split is applied. We train five model families per scope — Logistic Regression, Elastic Net, Random Forest, Balanced Random Forest, and XGBoost — and evaluate on PR-AUC as the primary metric because it is not corrupted by the true-negative majority."

---

### SLIDE 6 — Model Performance (6:30–8:30)

"Here is the headline performance chart. The primary metric is PR-AUC — precision-recall area under curve. The baseline is the positive rate: 8.5% for the global scope. A useless random classifier would score roughly 0.085.

Our best models achieve 0.128. That's a ~50% improvement over random — real signal, but modest. And here is the first 'wow' moment: Logistic Regression matches or beats XGBoost on PR-AUC. A simple linear model with L2 regularisation is as good as our most complex gradient-boosted ensemble.

I want you to take this seriously as a finding, not an embarrassment. It tells us something important about the data."

---

### SLIDE 7 — Top-K Lift (8:30–10:00)

"The second wow moment is top-5% lift. When we rank all ticker-weeks by predicted probability and isolate the highest-risk 5%, that bucket contains real drawdown events at **twice the baseline rate** — 2.15× for Balanced Random Forest on mega-cap AI/tech, 2.05× for Logistic Regression globally.

This means: if you used these scores to prioritise which assets to examine more closely, you'd be looking at a pool that is twice as rich in real pre-drawdown conditions as a random selection.

I need to be equally clear about what this is NOT: 82.5% of 'high-risk' rows had no drawdown. This is risk enrichment, not certainty."

---

### SLIDE 8 — Why Simple Beats Complex (10:00–11:30)

"Why does Logistic Regression win? Four reasons. First, the true event count is small — around 500 globally in the test set. Tree ensembles need more signal. Second, autocorrelation means the effective sample size is far smaller than 597K rows. Third, many bubble precursors are nearly linear — RSI extremes, momentum ratios, yield curve inversions. Fourth, L2 regularisation is very effective when your number of true events is small relative to the feature count.

The lesson: always train a regularised linear baseline first. Justify complexity by demonstrably improving out-of-time metrics."

---

### SLIDE 9 — Rare Event Challenge (11:30–13:00)

"With an 8.5% positive rate, standard classifiers are tempted to predict 'no crash' for everything and score 91.5% accuracy. Accuracy is misleading here.

We use three tools to address this: SMOTE oversampling in training (though this inflates recall at the cost of precision), Balanced Random Forest with class-weight adjustment, and cost-ratio threshold analysis where we explicitly model the relative cost of a false negative versus a false positive.

We also enforce a constrained rare-event threshold: models must achieve minimum precision of 12% and an alert rate no higher than 20% — preventing models from achieving high recall simply by flagging nearly everything."

---

### SLIDE 10 — Feature Transformations (13:00–14:30)

"Phase 8 adds a controlled feature transformation experiment. We run the exact same model families, the same chronological splits, and the same metrics — the only thing that changes is the feature pipeline.

Nine transformation families produce 88 additional columns: log and signed-log compressions for skewed valuation metrics, winsorisation to remove extreme outliers fitted only on the training split, rolling z-scores by ticker with a shift(1) lookback guard to prevent leakage, percentile ranks, regime flags, interaction features, volatility-adjusted returns, drawdown-state indicators, and squared terms.

The leakage guard is critical: winsorisation bounds and regime thresholds are fitted exclusively on the training split and applied unchanged to validation and test."

---

### SLIDE 11 — Transformation Results (14:30–15:30)

"The results are honest. Tree models gain +0.01 to +0.015 PR-AUC from transformations. Linear models gain almost nothing — they already handle scale via regularisation.

We also benchmarked LightGBM as an additional model family in Phase 8. Its PR-AUC of 0.113 on the broad ETF scope is competitive with XGBoost but does not surpass it.

No model crosses what I'd call a 'practically useful' bar of PR-AUC > 0.20. The story is consistent positive deltas across tree models — which is the honest interpretation — not a dramatic improvement."

---

### SLIDE 12 — Multi-Layer Validation (15:30–17:00)

"One of the project's architectural strengths is that we don't rely on a single evaluation method. We use five layers.

Chronological ML splits are the primary method. Event-level validation trains before a historical crisis and tests on the pre-event window — it's less flattering but the most honest generalization test. The Cox proportional hazard model asks a different question: given time-in-regime, what's the conditional probability of drawdown? The Poisson count model asks: how many drawdowns are expected in the next N weeks? And the rule-based baseline — Hindenburg Omen, RSI extremes, CAPE excess — must be beaten by any ML model to justify its complexity. The baseline achieves PR-AUC 0.096."

---

### SLIDE 13 — Dashboard Overview (17:00–18:00)

"The dashboard has 14 sections organized as a narrative. You start at the Executive Summary, proceed through data exploration, the bubble explorer and historical comparisons, vital signs, the live AI cycle monitor, and then the ML Lab — which is where most of the research lives.

The ML Lab now has 12 sub-tabs: model comparison, ROC curves, calibration, confusion matrices, false-positive audit, feature importance, phase 8 toggle and comparison, feature selection audit, and the recommended overall model composite score.

Every page includes interpretation boxes with 'do not overclaim' guidance, and missing-artifact notices that tell the user exactly which script to run if outputs are absent."

---

### SLIDE 14 — Limitations (18:00–19:30)

"I want to spend a slide on limitations because they're important.

True event count: 597K rows but perhaps 15–30 independent bubble events. Confidence intervals on PR-AUC are wide. Survivorship and selection bias: the ticker list reflects assets that survived and were liquid. Many 1927–1960 assets are missing. Data snooping risk: even with chronological splits, iterative feature engineering on the same dataset can implicitly overfit. Modest absolute performance: PR-AUC 0.128 versus 0.085 base rate. Real but modest. No live deployment validation was attempted."

---

### SLIDE 15 — Technical Contributions (19:30–20:30)

"The technical contributions are: a 14-script reproducible pipeline, segment-adjusted labels, a leakage-proof transformation framework, multi-layer validation from four independent angles, a composite model scoring formula that's pre-specified and auditable, a 14-tab narrative dashboard, and honest framing throughout — every chart includes a 'do not overclaim' box."

---

### SLIDE 16 — Conclusion (20:30–21:30)

"To conclude: Logistic Regression with chronological splits is the most defensible model. Top-5% risk ranking enriches events ~2× — real but modest. Feature transformations help tree models by ~0.01 PR-AUC. The honest answer is that predicting bubble bursts is hard and our metrics reflect that.

The dashboard infrastructure — interpretation boxes, regime badges, multi-layer validation — may be the project's most durable contribution. It gives any future researcher a framework for honest rare-event evaluation.

Thank you. I'm happy to take questions."

---

## Closing transition to Q&A

*"I'm going to leave the dashboard on screen during questions. If there's anything you'd like to see live, I can navigate to it."*
