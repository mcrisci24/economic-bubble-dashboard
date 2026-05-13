# Rare-Event Model Audit

This report checks whether the crash-risk models behave like useful rare-event rankers rather than simple yes/no classifiers.

## Key idea
The dashboard should not trust raw accuracy or unconstrained recall. Major drawdowns are rare, so the useful questions are: do high-risk score buckets contain more future drawdowns than normal, and do calibrated probabilities roughly match empirical event rates?

## Executive rare-event conclusion
- **best_roc_auc**: Logistic Regression / global = 0.615
- **best_pr_auc**: Logistic Regression / global = 0.128
- **best_precision**: Rule-Based Warning Score / broad_index_etf = 0.194
- **best_recall**: Elastic Net Logistic / mega_cap_ai_tech = 0.790
- **best_f1**: Logistic Regression / global = 0.178
- **best_mcc**: Elastic Net Logistic / global = 0.077
- **best_top_risk_lift**: Balanced Random Forest / mega_cap_ai_tech / Top 5% lift 2.15x
- **interpretation**: Use this as a rare-event risk-ranking audit. Strong top-bucket lift is more meaningful than high accuracy or unconstrained recall.

## Best test-split top-risk lift
- Balanced Random Forest / mega_cap_ai_tech / Top 5%: event rate 14.7% vs base 6.8%, lift 2.15x.
- Logistic Regression / global / Top 5%: event rate 17.5% vs base 8.5%, lift 2.05x.
- Elastic Net Logistic / global / Top 5%: event rate 17.4% vs base 8.5%, lift 2.05x.
- Random Forest / mega_cap_ai_tech / Top 10%: event rate 13.7% vs base 6.8%, lift 2.00x.
- Random Forest / mega_cap_ai_tech / Top 5%: event rate 13.7% vs base 6.8%, lift 2.00x.
- Rule-Based Warning Score / mega_cap_ai_tech / Top 10%: event rate 13.3% vs base 6.8%, lift 1.94x.
- Balanced Random Forest / mega_cap_ai_tech / Top 10%: event rate 13.3% vs base 6.8%, lift 1.94x.
- Rule-Based Warning Score / mega_cap_ai_tech / Top 5%: event rate 13.1% vs base 6.8%, lift 1.91x.