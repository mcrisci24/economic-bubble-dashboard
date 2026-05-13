# Cost-Ratio Threshold Stress Test

This report tests how model alerts change when false negatives are treated as 1x, 2x, 5x, 10x, or 20x as costly as false positives.
The goal is not to find a magical threshold. The goal is to expose the tradeoff between missed drawdowns and false alarms.

## Lowest test cost per row
- Rule-Based Warning Score / mega_cap_ai_tech / FN cost 1x: threshold=0.725, precision=0.279, recall=0.026, FP/TP=2.59, cost/row=0.071
- XGBoost / mega_cap_ai_tech / FN cost 1x: threshold=0.880, precision=0.079, recall=0.005, FP/TP=11.67, cost/row=0.072
- Balanced Random Forest / mega_cap_ai_tech / FN cost 1x: threshold=0.705, precision=0.150, recall=0.014, FP/TP=5.67, cost/row=0.073
- Random Forest / mega_cap_ai_tech / FN cost 1x: threshold=0.690, precision=0.143, recall=0.014, FP/TP=6.00, cost/row=0.073
- Random Forest / global / FN cost 1x: threshold=0.795, precision=0.000, recall=0.000, FP/TP=inf, cost/row=0.085
- Balanced Random Forest / global / FN cost 1x: threshold=0.835, precision=0.000, recall=0.000, FP/TP=inf, cost/row=0.085
- Rule-Based Warning Score / global / FN cost 1x: threshold=0.725, precision=0.209, recall=0.004, FP/TP=3.79, cost/row=0.086
- XGBoost / global / FN cost 1x: threshold=0.905, precision=0.123, recall=0.003, FP/TP=7.13, cost/row=0.087
- Elastic Net Logistic / global / FN cost 1x: threshold=0.990, precision=0.267, recall=0.020, FP/TP=2.74, cost/row=0.088
- Logistic Regression / global / FN cost 1x: threshold=0.990, precision=0.266, recall=0.020, FP/TP=2.76, cost/row=0.088
- Rule-Based Warning Score / speculative_high_vol / FN cost 1x: threshold=0.815, precision=0.250, recall=0.003, FP/TP=3.00, cost/row=0.103
- Logistic Regression / broad_index_etf / FN cost 1x: threshold=0.990, precision=0.714, recall=0.005, FP/TP=0.40, cost/row=0.110
- Elastic Net Logistic / broad_index_etf / FN cost 1x: threshold=0.990, precision=0.667, recall=0.004, FP/TP=0.50, cost/row=0.110
- Rule-Based Warning Score / broad_index_etf / FN cost 1x: threshold=0.485, precision=0.250, recall=0.002, FP/TP=3.00, cost/row=0.111
- XGBoost / broad_index_etf / FN cost 1x: threshold=0.780, precision=0.000, recall=0.000, FP/TP=inf, cost/row=0.117