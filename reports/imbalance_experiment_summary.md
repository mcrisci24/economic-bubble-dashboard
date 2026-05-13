# Imbalance Method Experiment Summary

These experiments test Balanced Random Forest, SMOTE, and SMOTE-ENN as controlled rare-event methods.
Resampling is applied only to the training split. Validation and test splits remain untouched future data.

## Interpretation rule
Keep a method only if it improves PR-AUC, MCC, top-5%/top-10% lift, and false-positive burden. Reject methods that merely inflate recall by flooding the dashboard with false alarms.

## Best test PR-AUC by experiment
- SMOTE + Logistic Regression / speculative_high_vol: PR-AUC=0.166, MCC=-0.022, precision=0.097, recall=0.576, FP/TP=9.26
- SMOTE-ENN + Logistic Regression / speculative_high_vol: PR-AUC=0.157, MCC=-0.024, precision=0.097, recall=0.597, FP/TP=9.29
- SMOTE + Logistic Regression / global: PR-AUC=0.127, MCC=0.080, precision=0.125, recall=0.349, FP/TP=6.97
- SMOTE-ENN + Logistic Regression / global: PR-AUC=0.127, MCC=0.081, precision=0.128, recall=0.328, FP/TP=6.83
- Balanced Random Forest / broad_index_etf: PR-AUC=0.117, MCC=0.018, precision=0.120, recall=0.288, FP/TP=7.34
- SMOTE + Logistic Regression / broad_index_etf: PR-AUC=0.115, MCC=-0.020, precision=0.090, recall=0.072, FP/TP=10.07
- SMOTE-ENN + Logistic Regression / broad_index_etf: PR-AUC=0.115, MCC=-0.017, precision=0.093, recall=0.069, FP/TP=9.81
- SMOTE + Random Forest / broad_index_etf: PR-AUC=0.110, MCC=-0.037, precision=0.054, recall=0.020, FP/TP=17.68
- SMOTE-ENN + Random Forest / broad_index_etf: PR-AUC=0.110, MCC=-0.042, precision=0.049, recall=0.020, FP/TP=19.43
- Balanced Random Forest / global: PR-AUC=0.108, MCC=0.071, precision=0.121, recall=0.331, FP/TP=7.27

## Best top-risk lift
- SMOTE + Logistic Regression / speculative_high_vol / Top 10%: event rate 27.0% vs base 10.3%, lift 2.63x
- SMOTE-ENN + Random Forest / mega_cap_ai_tech / Top 5%: event rate 16.4% vs base 6.8%, lift 2.40x
- SMOTE-ENN + Logistic Regression / speculative_high_vol / Top 10%: event rate 24.3% vs base 10.3%, lift 2.37x
- SMOTE + Random Forest / mega_cap_ai_tech / Top 5%: event rate 14.9% vs base 6.8%, lift 2.18x
- Balanced Random Forest / mega_cap_ai_tech / Top 5%: event rate 14.7% vs base 6.8%, lift 2.15x
- SMOTE-ENN + Logistic Regression / global / Top 5%: event rate 17.4% vs base 8.5%, lift 2.04x
- SMOTE + Logistic Regression / global / Top 5%: event rate 17.4% vs base 8.5%, lift 2.04x
- SMOTE-ENN + Random Forest / mega_cap_ai_tech / Top 10%: event rate 13.6% vs base 6.8%, lift 1.99x
- Balanced Random Forest / mega_cap_ai_tech / Top 10%: event rate 13.3% vs base 6.8%, lift 1.94x
- SMOTE + Random Forest / mega_cap_ai_tech / Top 10%: event rate 12.9% vs base 6.8%, lift 1.88x