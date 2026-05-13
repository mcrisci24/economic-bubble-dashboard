# Machine Learning Crash-Risk Model Summary

## Research framing
The models do not predict the exact timing of a bubble burst. They estimate whether today's feature pattern resembles historical weeks that were followed by a 30% or larger forward drawdown over roughly six months.

## Dashboard conclusion
**Overall model strength:** Modest but usable experimental signal.
Best global test PR-AUC is 0.128 versus a base event rate of 0.085, for a PR-AUC lift of 1.50x. Best global test ROC-AUC is 0.615.

## Best test-split models
- Best ROC-AUC: Logistic Regression (global), value=0.615
- Best PR-AUC: Logistic Regression (global), value=0.128
- Best recall: Elastic Net Logistic (mega_cap_ai_tech), value=0.817
- Best precision: Rule-Based Warning Score (broad_index_etf), value=0.156
- Best MCC: Elastic Net Logistic (global), value=0.077

## Top-decile risk analysis
When the models rank observations in their highest-risk 10%, the strongest test-split lifts were:
- Random Forest (mega_cap_ai_tech): top-decile event rate 0.137 vs base 0.068, lift 2.00x.
- Balanced Random Forest (mega_cap_ai_tech): top-decile event rate 0.133 vs base 0.068, lift 1.94x.
- Rule-Based Warning Score (mega_cap_ai_tech): top-decile event rate 0.130 vs base 0.068, lift 1.91x.
- Elastic Net Logistic (global): top-decile event rate 0.146 vs base 0.085, lift 1.72x.
- Logistic Regression (global): top-decile event rate 0.146 vs base 0.085, lift 1.71x.

## Tuned thresholds
The project no longer uses 0.50 as the automatic warning cutoff. Each model threshold is selected on the validation split by maximizing F1, then applied to the later test split. This makes the warning threshold appropriate for a rare-event setting where a probability far below 50% may still be historically elevated.
- Balanced Random Forest (broad_index_etf): threshold 0.440, validation F1 0.296.
- Elastic Net Logistic (broad_index_etf): threshold 0.755, validation F1 0.280.
- Logistic Regression (broad_index_etf): threshold 0.757, validation F1 0.280.
- Random Forest (broad_index_etf): threshold 0.410, validation F1 0.104.
- Rule-Based Warning Score (broad_index_etf): threshold 0.125, validation F1 0.290.
- XGBoost (broad_index_etf): threshold 0.475, validation F1 0.021.
- Balanced Random Forest (global): threshold 0.485, validation F1 0.335.
- Elastic Net Logistic (global): threshold 0.755, validation F1 0.216.
- Logistic Regression (global): threshold 0.755, validation F1 0.216.
- Random Forest (global): threshold 0.465, validation F1 0.284.
- Rule-Based Warning Score (global): threshold 0.125, validation F1 0.300.
- XGBoost (global): threshold 0.410, validation F1 0.333.
- Balanced Random Forest (mega_cap_ai_tech): threshold 0.705, validation F1 0.002.
- Elastic Net Logistic (mega_cap_ai_tech): threshold 0.405, validation F1 0.286.
- Logistic Regression (mega_cap_ai_tech): threshold 0.415, validation F1 0.286.
- Random Forest (mega_cap_ai_tech): threshold 0.670, validation F1 0.002.
- Rule-Based Warning Score (mega_cap_ai_tech): threshold 0.625, validation F1 0.009.
- XGBoost (mega_cap_ai_tech): threshold 0.514, validation F1 0.180.
- Balanced Random Forest (speculative_high_vol): threshold 0.380, validation F1 0.374.
- Elastic Net Logistic (speculative_high_vol): threshold 0.315, validation F1 0.245.

## Calibration check
Calibration check: high predicted probabilities should correspond to high empirical event rates. The highest-score bins show:
- Logistic Regression (speculative_high_vol, test): mean predicted 1.000, empirical event rate 0.073, rows 370.
- Elastic Net Logistic (speculative_high_vol, test): mean predicted 1.000, empirical event rate 0.076, rows 370.
- Elastic Net Logistic (mega_cap_ai_tech, test): mean predicted 1.000, empirical event rate 0.099, rows 948.
- Logistic Regression (mega_cap_ai_tech, test): mean predicted 1.000, empirical event rate 0.099, rows 948.
- Logistic Regression (speculative_high_vol, test): mean predicted 1.000, empirical event rate 0.068, rows 370.

## Why accuracy alone is misleading
Major forward drawdowns are rare. A model can achieve high accuracy by predicting 'no burst' for nearly everything, while still missing most actual drawdowns. PR-AUC, MCC, false positives per true positive, top-risk-bucket lift, and calibration are more useful for this type of risk-monitoring task.

## Feature interpretation
- Balanced Random Forest (broad_index_etf) top features: volatility_30d, enterprise_value, revenue_ttm, federal_funds_change_12p, return_252d, ten_year_treasury_yield, zscore_price, distance_from_200dma
- Elastic Net Logistic (broad_index_etf) top features: sma_50, sma_200, adjusted_close, sma_50_over_200, return_252d, volatility_30d, federal_funds_rate, distance_from_200dma
- Logistic Regression (broad_index_etf) top features: sma_50, sma_200, adjusted_close, sma_50_over_200, return_252d, volatility_30d, federal_funds_rate, distance_from_200dma
- Random Forest (broad_index_etf) top features: volatility_30d, enterprise_value, revenue_ttm, federal_funds_change_12p, return_252d, ten_year_treasury_yield, distance_from_200dma, zscore_price
- XGBoost (broad_index_etf) top features: volatility_30d, zscore_price, distance_from_200dma, enterprise_value, revenue_ttm, federal_funds_change_12p, ten_year_treasury_yield, return_252d
- Balanced Random Forest (global) top features: volatility_30d, ten_year_treasury_yield, yield_curve_spread, financial_stress_index, return_252d, distance_from_200dma, mortgage_rate, drawdown_pct
- Elastic Net Logistic (global) top features: sma_200, sma_50, volatility_30d, ev_to_sales, enterprise_value, price_to_sales, market_cap, unemployment_change_12p
- Logistic Regression (global) top features: volatility_30d, sma_200, enterprise_value, market_cap, adjusted_close, sma_50, unemployment_change_12p, distance_from_200dma
- Random Forest (global) top features: volatility_30d, ten_year_treasury_yield, financial_stress_index, yield_curve_spread, return_252d, distance_from_200dma, mortgage_rate, drawdown_pct
- XGBoost (global) top features: volatility_30d, distance_from_200dma, financial_stress_index, ten_year_treasury_yield, yield_curve_spread, net_income_ttm, ev_sales_to_historical_median, mortgage_rate
- Balanced Random Forest (mega_cap_ai_tech) top features: volatility_30d, ten_year_treasury_yield, financial_stress_index, mortgage_rate, yield_curve_spread, sma_200, revenue_ttm, drawdown_pct
- Elastic Net Logistic (mega_cap_ai_tech) top features: ev_to_sales, price_to_sales, sma_50, sma_200, volatility_30d, ten_year_treasury_yield, drawdown_pct, adjusted_close
- Logistic Regression (mega_cap_ai_tech) top features: ev_to_sales, price_to_sales, sma_50, sma_200, volatility_30d, ten_year_treasury_yield, adjusted_close, market_cap
- Random Forest (mega_cap_ai_tech) top features: volatility_30d, ten_year_treasury_yield, financial_stress_index, mortgage_rate, yield_curve_spread, drawdown_pct, sma_200, revenue_ttm
- XGBoost (mega_cap_ai_tech) top features: volatility_30d, ten_year_treasury_yield, financial_stress_index, net_income_ttm, ps_to_historical_median, yield_curve_spread, mortgage_rate, pe_ratio
- Balanced Random Forest (speculative_high_vol) top features: sma_200, ten_year_treasury_yield, volatility_30d, sma_50, yield_curve_spread, sma_50_over_200, revenue_ttm, distance_from_200dma
- Elastic Net Logistic (speculative_high_vol) top features: pe_to_historical_median, ev_sales_to_historical_median, ps_to_historical_median, price_log, return_63d, revenue_ttm, enterprise_value, zscore_price
- Logistic Regression (speculative_high_vol) top features: pe_to_historical_median, enterprise_value, ev_sales_to_historical_median, ps_to_historical_median, price_log, price_to_sales, return_63d, ev_to_sales
- Random Forest (speculative_high_vol) top features: ten_year_treasury_yield, sma_200, volatility_30d, yield_curve_spread, distance_from_200dma, sma_50, revenue_ttm, price_to_sales
- XGBoost (speculative_high_vol) top features: distance_from_200dma, ev_to_sales, pe_to_historical_median, ten_year_treasury_yield, volatility_30d, sma_50_over_200, price_to_sales, sma_50

## Bottom line
Treat the ML system as an experimental research layer. It can rank historical resemblance to past pre-drawdown regimes, but it is not a stand-alone trading system and it is not financial advice.
