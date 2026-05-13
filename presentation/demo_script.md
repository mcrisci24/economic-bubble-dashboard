# Demo Script — Live Dashboard Walkthrough
## Target: 5–8 minutes at http://localhost:8502

---

## Pre-demo checklist

- [ ] Dashboard running: `streamlit run dashboard.py --server.port=8502`
- [ ] Browser at `http://localhost:8502`
- [ ] Sidebar visible (not collapsed)
- [ ] Monitor resolution readable from projector distance

---

## DEMO FLOW

### Stop 1 — Executive Summary (1 min)

**Navigate to:** `1. Executive Summary`

> "This is the landing page. You can see four headline metrics at the top — 81 tickers, 597K rows, data back to 1927. Below that, the dynamic research story is built from whatever artifacts exist on disk."

- Point out the **signal confirmation table** if visible
- Click to expand **"Why predicting bubble bursts is hard"** expander
- Read the first sentence: "We build a detector, not a forecaster."

---

### Stop 2 — ML Model Lab: Model Comparison (2 min)

**Navigate to:** `7. Machine Learning Model Lab`

> "This is the core of the research. Notice the toggle at the top — unchecked shows baseline, checked shows the Phase 8 transformed-feature results."

1. Click the **"Model comparison"** tab
2. Point to PR-AUC column: "Best is 0.128 for Logistic Regression globally. XGBoost is 0.109."
3. Show the grouped bar chart: "Logistic Regression and XGBoost almost identical — this is the main finding."

**Navigate to:** Feature importance tab

> "If we look at feature importance, we see momentum, RSI, and rule-based scores dominate. Valuation features matter less in isolation."

---

### Stop 3 — Rare Event Lab: Top-K Lift (1 min)

**Navigate to:** `8. Rare Event & Imbalance Lab`  
**Click:** "Top-5% / Top-10% lift" tab

> "This is the actionable result. The highest-scoring 5% of ticker-weeks contains drawdown events at 2× the base rate. Not a guarantee — 82% of the top bucket had no drawdown — but it's real risk enrichment."

- Point out the lift column vs base_event_rate column

---

### Stop 4 — Transformed vs Baseline (1 min)

**Back to:** `7. Machine Learning Model Lab`  
**Click:** "Transformed vs Baseline" tab

> "Phase 8 ran the same models with 88 additional engineered features. The delta PR-AUC chart shows tree models gained +0.010 to +0.015. Linear models gained almost nothing — they already handle scale. This is the honest result."

---

### Stop 5 — Feature Selection Audit (30 sec)

**Click:** "Feature Selection Audit" tab

> "This documents every feature that entered the model and what was excluded for leakage. The transformation family bar chart shows which families produced the most features — signed-log and winsorisation are the most prolific."

---

### Stop 6 — Recommended Overall Model (30 sec)

**Click:** "Recommended Overall Model" tab

> "Rather than cherry-picking the best metric, we apply a pre-specified composite score across PR-AUC, ROC-AUC, MCC, false-positive burden, and Brier score. Logistic Regression / global wins. The formula is fully documented and you can re-weight it for a different use case."

---

### Stop 7 — Hazard & Poisson (30 sec, if time allows)

**Navigate to:** `10. Hazard & Poisson Models`

> "Two additional model layers beyond the point-in-time classifier. The hazard model estimates conditional probability given time-in-regime. The Poisson model estimates expected drawdown count in the next N weeks. These ask different questions than the ML classifier and triangulate the answer."

---

### Stop 8 — Data Quality (30 sec, if time allows)

**Navigate to:** `14. Data Quality & Limitations`

> "The project doesn't hide its limitations. This page documents missingness, ticker coverage gaps, segment distribution, and an explicit discussion of small event count. We know this page is often overlooked — we put it in the navigation intentionally."

---

## Demo fallback

If the dashboard fails to load or artifacts are missing, use the PDF screenshots in the presentation as backup. Missing-artifact notices will be visible — point to them: "This is exactly what we designed — the dashboard tells you which script to run rather than crashing or silently returning empty charts."
