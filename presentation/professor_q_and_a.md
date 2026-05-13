# Professor Q&A Guide — Anticipated Questions & Honest Answers
## Supervised ML: Future Drawdown Risk Classification
### Note: The dashboard is the interpretation layer. The ML pipeline is the main deliverable.

---

## Category 1: Statistical Validity

**Q: Your PR-AUC is only 0.128. How do you know this isn't just noise?**

The base positive rate is 0.085 (global scope). A random classifier has expected PR-AUC ≈ 0.085. Our best model achieves 0.128, approximately 50% above random. Given the test set has ~54,000 rows with ~4,600 positive cases, the improvement is statistically significant, but confidence intervals are wide. We do not claim high effect size — we acknowledge modest signal and present it honestly. The top-5% lift (2.05×) provides an additional, more intuitive metric that corroborates the PR-AUC gain.

**Q: You have 597,806 rows. Why are your metrics so modest?**

Two reasons. First, the positive-label rows are highly autocorrelated — a single bubble regime generates hundreds of consecutive positive rows for dozens of tickers simultaneously. The number of truly *independent* bubble events is perhaps 15–30 globally. Second, the chronological split means we train on 1927–~2007 and test on 2007–2026. The test period includes only a handful of major crises: the 2008 financial crisis, the 2020 COVID crash, and the 2022 rate-shock drawdown. Variance is high over so few events.

**Q: Couldn't you just be overfitting despite the chronological split?**

Yes, partially. We mitigate this with: (1) strict no-shuffle chronological split, (2) event-level leave-one-crisis validation, (3) an explicit rule-based baseline that ML must outperform, (4) regularised linear models rather than over-parameterised ensembles. But iterative feature engineering on the same dataset constitutes a form of implicit overfitting even with chronological splits. We document this as a limitation. PR-AUC 0.128 on this data should be treated as an upper bound for a well-tuned pipeline, not as a precise population estimate.

**Q: Why not use walk-forward cross-validation instead of a single 70/85 split?**

Walk-forward CV would give more robust variance estimates but at the cost of dramatically increased compute and code complexity. More importantly, our positive-class rate is ~8.5% and the test window covers only ~3 major crises. Walk-forward folds would produce even fewer positive test cases per fold, making fold-level metrics noisier, not cleaner. The single chronological split gives one honest held-out test set that covers the most recent market conditions.

---

## Category 2: Methodology

**Q: Why use PR-AUC rather than ROC-AUC as the primary metric?**

ROC-AUC is a function of TPR and FPR. FPR is computed against all true negatives — in our dataset, ~91.5% of rows. Even a model with very low precision can achieve high ROC-AUC because the denominator of FPR is enormous. PR-AUC uses precision (TP / (TP + FP)), which is directly relevant to how useful the model is in practice — of the rows we flag, how many are true events? For rare events, PR-AUC is the standard recommendation (Davis & Goadrich 2006; Saito & Rehmsmeier 2015).

**Q: Why does Logistic Regression beat XGBoost? Isn't that suspicious?**

No — it is actually expected in this setting. We discuss it explicitly in the presentation. Short answer: the number of truly independent training events is small. Tree ensembles require more signal to justify their additional parameters. Logistic Regression with L2 regularisation trades capacity for variance reduction, and on sparse, autocorrelated rare-event data, that trade-off typically wins. This pattern appears in medical rare-event literature, fraud detection, and financial crisis prediction consistently.

**Q: How did you prevent leakage in the feature transformations (Phase 8)?**

`TransformFitState` stores winsorisation bounds and regime thresholds that are computed exclusively on the training split during `fit_transform_state(train_df)`. The `apply_transformations(df, state)` function then applies the same pre-computed bounds to validation and test without re-fitting. Rolling z-scores use `shift(1)` before the expanding-window calculation so no future observation enters the current row's normalisation. Target-adjacent columns (`burst_6m*`, `future_*`, `target_valid_*`) are blacklisted at the feature assembly step.

**Q: Why segment-specific drawdown thresholds rather than a single threshold?**

A single −25% threshold applied to Bitcoin and SPY simultaneously would label almost every Bitcoin week as positive (Bitcoin routinely drops 25%+ without constituting a "crisis" in the sense we study) and would rarely flag SPY (which experiences −25%+ only in major crises). The result would be a label distribution that is meaningless for Bitcoin and over-conservative for SPY. Segment-specific thresholds ensure the label captures genuinely unusual drawdowns relative to each asset class's typical volatility.

**Q: Your model has data back to 1927. Does a 1929 Dow Jones week add useful information for predicting AI stock crashes in 2025?**

Probably not directly. The 1927–1940 data contributes primarily to the global pooled scope during training. The model presumably learns that price-momentum divergence and valuation excess are persistent bubble precursors regardless of era. Whether the 1929 regime is informative is empirically testable by comparing a model trained only post-1990. We did not run this ablation — it is a fair criticism and a direction for future work.

---

## Category 3: Practical Implications

**Q: Is this system deployable for real investment decisions?**

No, and we say so explicitly. The system was not paper-traded. No out-of-sample data beyond the original split has been evaluated. Real deployment would require: continuous data pipeline hardening, model retraining cadence, and position sizing rules that account for the ~82% false-positive rate in the top-5% bucket. This is a research tool, not a trading system.

**Q: The Poisson and Hazard models seem redundant with the ML classifier. Why include them?**

They answer different questions. The ML classifier asks: "does this week's feature vector resemble a pre-drawdown regime?" The hazard model asks: "conditional on being in a given regime for N weeks, what is the instantaneous probability of a drawdown event today?" The Poisson model asks: "how many discrete drawdown events should we expect in the next 13 weeks across all tickers?" Triangulating from three different modeling assumptions is stronger than relying on one. When all three agree, the signal is more credible.

**Q: How do you handle the fact that bubbles can last for years, creating very long runs of consecutive positive labels?**

We use `target_valid_burst_6m_segment` to exclude the 26-week post-burst overlap windows — rows where the burst has already started. This prevents the model from learning to predict an ongoing burst rather than a pre-burst regime. Even with this, consecutive pre-burst weeks within a regime will all carry positive labels, which is appropriate: those weeks genuinely did precede the drawdown.

---

## Category 4: Phase 8 — Feature Transformations

**Q: Why didn't transformations help linear models but they helped tree models?**

Logistic Regression with L2 regularisation already adapts to feature scale through the regularisation penalty — features with larger raw values receive proportionally larger penalties, producing an effect similar to standardisation. Tree models split on raw feature values, so log-compressing right-skewed valuations or normalising by rolling z-score genuinely changes which thresholds produce informative splits. The result — linear models indifferent, tree models improved — is exactly what feature transformation theory predicts.

**Q: Why does LightGBM only appear in the transformed-feature results, not the baseline?**

LightGBM was added as a transformed-feature benchmark in Phase 8. It was not trained in the baseline pipeline (`model_training.py`). The baseline models are: Logistic Regression, Elastic Net, Random Forest, Balanced Random Forest, XGBoost, and the Rule-Based Warning Score. LightGBM appears only when the transformed-feature toggle is activated in the ML Lab, and this is made explicit in the dashboard text. The dashboard shows "LightGBM was added as a transformed-feature benchmark. It is not part of the original baseline model comparison unless explicitly trained in the baseline pipeline." Note also that Balanced Random Forest is in the baseline but was not carried into Phase 8 — the two pipelines are not symmetric, and both facts are documented.

---

## Category 5: Dashboard Design

**Q: Why build a Streamlit dashboard instead of a Jupyter notebook?**

A dashboard serves a different audience than a notebook. The notebook is the analysis scratchpad; the dashboard is the shareable research tool. With Streamlit, a professor, a fellow student, or a non-technical reviewer can explore the full pipeline, change parameters, and see results without executing code. The interpretation boxes and regime badges make the results accessible to someone who is not deeply familiar with PR-AUC or confusion matrices.

**Q: You have 14 sections. Is that too complex?**

It is comprehensive by design. Each section addresses a different research question: the raw data layer (sections 1–5), the ML modeling layer (sections 7–9), the alternative models layer (section 10), the application layer (section 11), and the meta-layer (sections 12–14). Users who only care about ML results can go directly to section 7. The section count reflects the genuine breadth of the analysis, not feature bloat.

---

## Category 6: Honest Self-Assessment

**Q: What would you do differently if you had more time?**

1. Walk-forward cross-validation with bootstrap confidence intervals on PR-AUC
2. Ticker-held-out validation (analogous to patient-held-out in medical ML)
3. Calibrated probability outputs with proper Platt scaling or isotonic regression on a true holdout
4. Firth logistic regression for all linear models (we have the R export but did not fully integrate it)
5. A more systematic ablation study removing each transformation family to isolate its contribution

**Q: What's the most important result from the project?**

That Logistic Regression matches XGBoost on PR-AUC. It forces you to confront that complexity is not always the answer. In rare-event time-series forecasting on small true-event counts, the regularised linear model is genuinely competitive and much easier to interpret, audit, and deploy. This lesson generalises beyond bubble detection.
