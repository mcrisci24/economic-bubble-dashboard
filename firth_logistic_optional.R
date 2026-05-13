# Optional Firth logistic regression benchmark
# Run in R after Python creates data/processed/firth_benchmark_*.csv
# install.packages("logistf")

library(logistf)

base_dir <- getwd()
processed_dir <- file.path(base_dir, "data", "processed")
features <- c("distance_from_200dma", "weekly_rsi", "drawdown_pct", "volatility_30d", "return_63d", "return_126d", "zscore_price", "warning_score", "recovery_score", "price_to_sales", "ev_to_sales", "ps_to_historical_median", "revenue_ttm_yoy_pct", "federal_funds_rate", "yield_curve_spread", "cpi_yoy", "unemployment_rate", "financial_stress_index")
target <- "burst_6m_segment"
files <- list.files(processed_dir, pattern="^firth_benchmark_.*\.csv$", full.names=TRUE)
results <- data.frame()

for (file in files) {
  df <- read.csv(file)
  scope <- sub("firth_benchmark_", "", tools::file_path_sans_ext(basename(file)))
  formula_text <- paste(target, "~", paste(features[features %in% names(df)], collapse=" + "))
  cat("Fitting Firth logistic for", scope, "with formula", formula_text, "
")
  fit <- logistf(as.formula(formula_text), data=df)
  coefs <- data.frame(scope=scope, feature=names(coef(fit)), coefficient=as.numeric(coef(fit)))
  results <- rbind(results, coefs)
}

write.csv(results, file.path(processed_dir, "firth_logistic_coefficients.csv"), row.names=FALSE)
cat("Saved Firth coefficients to data/processed/firth_logistic_coefficients.csv
")
