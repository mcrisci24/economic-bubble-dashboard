# Speaker Script — Supervised ML: Future Drawdown Risk Classification
## Timing guide: 10–12 minutes presentation + 5–8 minutes Q&A

---

### SLIDE 1 — Title (0:00–0:30)

"Thank you. Today I'm presenting a supervised machine learning project: classifying future segment-adjusted drawdown risk from financial market data.

This is a binary classification pipeline applied to a rare-event problem — predicting whether assets will cross major drawdown thresholds within six months. The interactive dashboard you'll see is the interpretation layer. The ML pipeline is the main deliverable.

Nothing here is investment advice."

---

### SLIDE 2 — Why This Problem (0:30–1:30)

"Why study financial drawdowns with ML?

Bubble episodes appear in every market cycle. Practitioners want early-warning signal — but predicting exact crash timing is widely regarded as impossible. A more tractable ML formulation is: does today's feature vector resemble the conditions that historically preceded major drawdowns?

This turns a speculative forecasting problem into a supervised rare-event classification problem — which is well-defined, evaluable, and honest about its limits."

---

### SLIDE 3 — Dataset (1:30–2:30)

"The data is original — not a Kaggle dataset. I collected 81 tickers via Yahoo Finance and FRED: broad ETFs, mega-cap tech, speculative assets, and macro indicators, going back to 1927 where available.

After feature engineering, the dataset has 597,806 weekly rows. The key design choice is segment-specific drawdown thresholds: −20% for broad ETFs, −35% for mega-cap tech, −50% for speculative assets. This prevents labelling a Bitcoin −20% move the same way as an S&P 500 −20% move."

---

### SLIDE 4 — Target Label (2:30–3:15)

"The target, burst_6m_segment, equals 1 if the asset hits its segment-adjusted threshold within 26 weekly observations — roughly six months forward.

This creates an imbalanced binary classification problem: the positive rate is 6.8% to 11% depending on scope. The chronological 70/15/15 split is strict — no random shuffling. Shuffling would leak future market states into training and produce inflated metrics."

---

### SLIDE 5 — EDA & Class Imbalance (3:15–4:00)

"EDA confirmed several things: strong class imbalance, high momentum autocorrelation within bubble regimes, wide valuation dispersion across asset segments, and meaningful distributional differences between pre-burst and non-burst periods for RSI, price-to-sales, and drawdown velocity.

We also confirmed that a naive majority-class classifier would score 91.5% accuracy with 0% recall on positive cases — which is why accuracy is not our metric."

---

### SLIDE 6 — Feature Engineering (4:00–4:45)

"Features cover six families: price momentum and RSI, volume ratios, valuation multiples, macro indicators, rule-based bubble scores, and drawdown velocity.

Phase 8 added 88 transformed features: log and signed-log compressions, winsorisation fitted on train only, rolling z-scores by ticker with shift(1) lookback guard, regime flags, interaction terms, and squared terms. Leakage prevention was enforced through a TransformFitState object that fits bounds on training data only."

---

### SLIDE 7 — Benchmark Model (4:45–5:30)

"The benchmark is Logistic Regression with L2 regularisation — the simplest defensible baseline for imbalanced binary classification.

Test-set results: PR-AUC 0.128 versus a base rate of 0.085. ROC-AUC 0.615. Top-5% lift 2.05×.

This is modest signal — but real. And it is the honest baseline every more-complex model must beat."

---

### SLIDE 8 — All Models Tested (5:30–6:15)

"We trained five model families per scope — four asset segments each. That is 24 baseline model × scope combinations.

The five families: Logistic Regression as benchmark, Elastic Net as the regularised linear comparison, Random Forest and Balanced Random Forest as nonlinear and imbalance-aware tree models, and XGBoost as the boosted-tree challenger.

The Rule-Based Warning Score — Hindenburg Omen, CAPE excess, RSI extremes — is the interpretable sanity-check baseline. Any ML model must beat its PR-AUC of 0.096.

LightGBM was added as a Phase 8 transformed-feature benchmark only. It is not part of the original baseline comparison."

---

### SLIDE 9 — Train/Val/Test & Metrics (6:15–7:00)

"The chronological split: first 70% of unique dates for training, next 15% for validation, final 15% for test. No shuffling.

Primary metric: PR-AUC — precision-recall area under curve, which is not corrupted by the true-negative majority. Secondary metrics: ROC-AUC, MCC, false-positives per true positive, Brier score, and top-5% / top-10% lift.

Prediction thresholds are tuned on validation, not hard-coded at 0.50."

---

### SLIDE 10 — Model Results (7:00–8:00)

"Logistic Regression and Elastic Net share the top PR-AUC of 0.128 on the global test split. XGBoost reaches 0.109. Balanced Random Forest achieves 0.108 with higher recall.

The main finding: a regularised linear model equals or beats the boosted ensemble. This is expected — with roughly 500 independent positive-class events in the test set, XGBoost's additional capacity is not justified by the data.

The composite scoring formula — PR-AUC 35%, ROC-AUC 20%, MCC 20%, FP/TP burden −15%, Brier −10% — selects Logistic Regression / global as the recommended overall model."

---

### SLIDE 11 — Feature Transformations: Three Model Groups (8:00–8:45)

"Feature transformations change how predictor variables are represented before training. The model family stays the same — Logistic Regression is still Logistic Regression — but the input feature geometry changes.

Before comparing results, I want to be precise about three groups of models.

Group 1: Models present in both the baseline and the transformed pipeline — Logistic Regression, Elastic Net, Random Forest, XGBoost, and the Rule-Based score. These are the only models with valid before/after deltas. Tree models gained +0.010 to +0.015 PR-AUC. Linear models gained near zero.

Group 2: Balanced Random Forest exists only in the baseline. It was not carried into Phase 8. There is no transformed version, so no delta is calculated.

Group 3: LightGBM exists only in the transformed pipeline. It was added as a Phase 8 benchmark. It has no baseline result, so comparing it to baseline numbers would be misleading. LightGBM scored PR-AUC 0.119 on broad ETF scope — competitive with XGBoost but not a before/after result.

No model in either pipeline crosses PR-AUC 0.20."

---

### SLIDE 12 — Feature Importance (8:45–9:20)

"Logistic Regression coefficients and tree-model feature importance both point to the same families: price momentum, RSI distance from extremes, rule-based bubble scores, and short-term volatility.

Valuation features contribute less in isolation. Transformed features — particularly rolling z-scores and regime flags — rank highly for XGBoost and Random Forest in the Phase 8 results."

---

### SLIDE 13 — Limitations (9:20–10:00)

"Honest limitations. The number of truly independent bubble events is 15–30 globally — confidence intervals on PR-AUC are wide. Survivorship bias in the ticker universe. Iterative feature engineering creates implicit overfitting risk even with chronological splits. The top-5% bucket is 82.5% false positives. No live paper-trading validation was performed."

---

### SLIDE 14 — Conclusion (10:00–10:30)

"What this project shows: you can build a principled rare-event classification pipeline on financial market data, evaluate it honestly with PR-AUC and top-k lift rather than accuracy, and show that a regularised linear model is the most defensible baseline.

The result is modest signal — PR-AUC 0.128 versus 0.085 base rate. Modest signal from a genuinely difficult problem is a legitimate and honest ML conclusion.

Thank you."
