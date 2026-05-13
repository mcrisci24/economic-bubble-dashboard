# Hazard Model Leakage Audit

The hazard model is trained with both full and leakage-restricted feature sets.
The restricted version removes current drawdown-state variables such as drawdown_pct, distance_from_200dma, zscore_price, warning_score, and recovery_score.

- Hazard Logistic Regression: full PR-AUC=0.961, restricted PR-AUC=0.701, gap=0.260. SUSPICIOUS: full model may be using target-adjacent state variables
- Hazard Random Forest: full PR-AUC=0.935, restricted PR-AUC=0.734, gap=0.201. SUSPICIOUS: full model may be using target-adjacent state variables
- Hazard XGBoost: full PR-AUC=0.937, restricted PR-AUC=0.746, gap=0.190. SUSPICIOUS: full model may be using target-adjacent state variables