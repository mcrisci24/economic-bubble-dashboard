# Demo Script — Live ML Evidence Walkthrough
## Target: 5–8 minutes at http://localhost:8502

---

## Pre-demo checklist

- [ ] Dashboard running: `streamlit run dashboard.py --server.port=8502`
- [ ] Browser at `http://localhost:8502`
- [ ] Sidebar visible

---

## FRAMING (say before opening the browser)

"The dashboard is the interpretation layer for the ML pipeline — it is not the deliverable. The deliverable is the supervised classification pipeline. I'll walk you through the ML evidence."

---

## Stop 1 — Executive Summary (0:30)

**Navigate to:** `1. Executive Summary`

> "This landing page shows dataset scale: 81 tickers, 597K rows, 1927–2026. The dynamic story at the bottom is built from ML artifacts on disk. The important framing: we are classifying **future segment-adjusted drawdown risk** — a binary rare-event classification problem, not a price forecasting problem."

---

## Stop 2 — Data Overview & EDA (1:00)

**Navigate to:** `2. Data Overview & EDA`

> "EDA confirmed three things relevant to model design: strong class imbalance (~8.5% positive globally), high autocorrelation within bubble regimes, and meaningful feature separation between pre-burst and non-burst periods. A naive majority-class classifier scores 91.5% accuracy with 0% recall — this is why accuracy is not our metric."

Point out the class balance chart if visible.

---

## Stop 3 — ML Model Lab: Model Comparison (2:00)

**Navigate to:** `7. Machine Learning Model Lab`

> "This is the core experiment. The toggle at the top switches between baseline and transformed-feature results. I'll start with baseline — the default."

1. Click **"Model comparison"** tab
2. Point to PR-AUC column: "Best is 0.128 for Logistic Regression — our benchmark. XGBoost is 0.109. The regularised linear model matches the boosted ensemble."
3. Point to Rule-Based Warning Score row: "PR-AUC 0.096 — the interpretable sanity check every ML model must beat."
4. Point to any weak result (e.g., Logistic Regression on speculative_high_vol, PR-AUC 0.078): "We do not hide weak results. ROC-AUC near chance and low PR-AUC on speculative assets is part of the honest story — this scope is the hardest."

> "Note: **LightGBM does not appear here.** LightGBM was added as a transformed-feature benchmark. Check the toggle to see it."

---

## Stop 4 — Recommended Overall Model (1:00)

**Click:** "Recommended Overall Model" tab

> "Rather than cherry-picking the best metric, I apply a pre-specified composite score across PR-AUC, ROC-AUC, MCC, false-positive burden, and Brier score. Logistic Regression / global wins. The formula is fully documented — you can see the weights. The winner can be re-evaluated if you weight differently."

Point to the winner card: "PR-AUC 0.128, composite score shown, honest caveat that 82.5% of top-5% rows had no drawdown."

---

## Stop 5 — Feature Selection Audit (0:45)

**Click:** "Feature Selection Audit" tab

> "This documents leakage prevention: target-adjacent columns are blacklisted, winsorisation bounds are fit on training only. The feature families driving the model are momentum, RSI, and rule-based scores — not valuation alone."

---

## Stop 6 — Transformed vs Baseline: Three Groups (1:00)

**Click:** "Transformed vs Baseline" tab

> "This tab splits models into three groups. Point to the comparability notice at the top."

> "**Group 1** — models present in both pipelines: Logistic Regression, Elastic Net, Random Forest, XGBoost, Rule-Based Score. These are the only models with valid before/after deltas. Tree models gained +0.01–0.015 PR-AUC. Linear models near zero."

> "**Group 2** — Balanced Random Forest is baseline-only. No transformed version exists. No delta is calculated — that is correct, not a bug."

> "**Group 3** — LightGBM is transformed-only. It was added as a Phase 8 benchmark. It has no baseline counterpart, so its PR-AUC 0.119 is a standalone benchmark result, not a before/after comparison."

---

## Stop 7 — Feature Importance (0:30)

**Click:** "Feature importance" tab (ensure baseline toggle is unchecked)

> "Logistic Regression coefficients and tree importance both converge on the same families: momentum, RSI extremes, rule-based scores, short-term volatility. This cross-model consistency is one reason we trust these features."

---

## Stop 8 — Limitations (0:30)

**Navigate to:** `14. Data Quality & Limitations`

> "The limitations page is intentional. Roughly 15–30 independent bubble events globally. Wide confidence intervals on PR-AUC. Survivorship bias. No live deployment. The models provide risk-ranking signal, not crash certainty."

---

## Closing

> "The dashboard lets us inspect ML evidence interactively, but the project is the pipeline: novel dataset, segment-adjusted labels, chronological splits, multi-model evaluation, rare-event metrics, and honest framing. The modest PR-AUC improvement over base rate is a real result for a genuinely hard problem."
