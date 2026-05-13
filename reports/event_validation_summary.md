# Event-Level Validation Summary

This report tests whether models trained before a historical event could rank the pre-event/event window as risky.
It is intentionally stricter than row-level validation because many positive ticker-week labels are clustered inside the same crises.

## Best event-validation rows by PR-AUC
- dotcom_bust / Logistic Regression: PR-AUC=0.480, MCC=0.170, precision=0.418, recall=0.692, FP/TP=1.39
- dotcom_bust / Elastic Net Logistic: PR-AUC=0.480, MCC=0.171, precision=0.419, recall=0.691, FP/TP=1.39
- global_financial_crisis / Elastic Net Logistic: PR-AUC=0.461, MCC=0.067, precision=0.432, recall=0.218, FP/TP=1.32
- global_financial_crisis / Logistic Regression: PR-AUC=0.461, MCC=0.068, precision=0.433, recall=0.218, FP/TP=1.31
- global_financial_crisis / Balanced Random Forest: PR-AUC=0.425, MCC=0.032, precision=0.553, recall=0.010, FP/TP=0.81
- global_financial_crisis / Random Forest: PR-AUC=0.408, MCC=0.026, precision=0.531, recall=0.008, FP/TP=0.88
- dotcom_bust / Balanced Random Forest: PR-AUC=0.403, MCC=0.152, precision=0.430, recall=0.544, FP/TP=1.32
- global_financial_crisis / XGBoost: PR-AUC=0.382, MCC=0.000, precision=0.000, recall=0.000, FP/TP=inf
- dotcom_bust / Random Forest: PR-AUC=0.353, MCC=-0.022, precision=0.326, recall=0.168, FP/TP=2.07
- dotcom_bust / XGBoost: PR-AUC=0.337, MCC=-0.078, precision=0.217, recall=0.045, FP/TP=3.61
- covid_crash / Elastic Net Logistic: PR-AUC=0.308, MCC=-0.055, precision=0.212, recall=0.055, FP/TP=3.71
- covid_crash / Logistic Regression: PR-AUC=0.308, MCC=-0.051, precision=0.224, recall=0.065, FP/TP=3.46

## Best event top-risk lift
- crypto_2017_2018 / Elastic Net Logistic / Top 5%: event rate 24.1% vs base 6.7%, lift 3.62x
- crypto_2017_2018 / Logistic Regression / Top 5%: event rate 24.1% vs base 6.7%, lift 3.62x
- crypto_2017_2018 / Elastic Net Logistic / Top 10%: event rate 16.5% vs base 6.7%, lift 2.48x
- crypto_2017_2018 / Logistic Regression / Top 10%: event rate 16.5% vs base 6.7%, lift 2.48x
- japan_asset_bubble_unwind / Balanced Random Forest / Top 10%: event rate 26.0% vs base 12.9%, lift 2.01x
- speculative_tech_crypto_2021_2022 / Random Forest / Top 5%: event rate 38.2% vs base 20.4%, lift 1.87x
- rate_hike_tech_drawdown_2022 / Random Forest / Top 5%: event rate 41.4% vs base 22.2%, lift 1.86x
- speculative_tech_crypto_2021_2022 / Balanced Random Forest / Top 5%: event rate 37.6% vs base 20.4%, lift 1.84x
- speculative_tech_crypto_2021_2022 / Elastic Net Logistic / Top 5%: event rate 37.3% vs base 20.4%, lift 1.83x
- speculative_tech_crypto_2021_2022 / Logistic Regression / Top 5%: event rate 37.3% vs base 20.4%, lift 1.83x
- rate_hike_tech_drawdown_2022 / Elastic Net Logistic / Top 5%: event rate 38.9% vs base 22.2%, lift 1.75x
- rate_hike_tech_drawdown_2022 / Logistic Regression / Top 5%: event rate 38.9% vs base 22.2%, lift 1.75x