"""
Backtesting and investment simulator.

Run quick example:
    python backtesting.py --ticker QQQ --start 1999-01-01 --end 2003-12-31 --strategy sell_exit_buy_recovery

All strategy results are historical backtests, not financial advice.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from config import LOG_DIR, PROCESSED_DIR, STARTING_CAPITAL_DEFAULT, MONTHLY_CONTRIBUTION_DEFAULT, TRADING_DAYS_PER_YEAR

LOG_FILE = LOG_DIR / "backtesting.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtesting")


def debug_print(message: str) -> None:
    print(f"[BACKTEST DEBUG] {message}")
    logger.info(message)


@dataclass
class BacktestResult:
    ticker: str
    strategy: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    starting_capital: float
    monthly_contribution: float
    ending_value: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    volatility_pct: float
    sharpe_ratio: float
    time_to_recover_days: int | None
    trades: int
    notes: str
    equity_curve: pd.DataFrame


def load_scored_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "bubble_signal_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run data_ingestion.py, feature_engineering.py, bubble_signals.py first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def prepare_backtest_frame(df: pd.DataFrame, ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    g = df[df["ticker"].eq(ticker)].sort_values("date").copy()
    if start:
        g = g[g["date"] >= pd.Timestamp(start)]
    if end:
        g = g[g["date"] <= pd.Timestamp(end)]
    g = g.dropna(subset=["adjusted_close"])
    if len(g) < 30:
        raise ValueError(f"Not enough rows for {ticker} in selected window. Rows={len(g)}")
    g["adjusted_close"] = g["adjusted_close"].astype(float)
    return g.reset_index(drop=True)


def signal_buy_and_hold(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    buy = pd.Series(False, index=g.index)
    sell = pd.Series(False, index=g.index)
    buy.iloc[0] = True
    return buy, sell, "Invest all available cash from first available date."


def signal_sell_exit_buy_recovery(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    buy = g["reentry_signal"].fillna(False).astype(bool).copy()
    sell = g["exit_signal"].fillna(False).astype(bool).copy()
    buy.iloc[0] = True
    return buy, sell, "Buy initially, sell on exit warnings, re-enter on recovery confirmations. Signals execute next trading day."


def signal_cash_until_drawdown(threshold: float) -> Callable[[pd.DataFrame], tuple[pd.Series, pd.Series, str]]:
    def _rule(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
        buy = (g["drawdown_pct"] <= -abs(threshold)).fillna(False)
        sell = pd.Series(False, index=g.index)
        return buy, sell, f"Hold cash until trailing peak drawdown reaches {threshold:.0f}%."

    return _rule


def signal_cash_until_rsi_recovery(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    buy = (g["rsi_recovery"].fillna(False) & (g["drawdown_pct"] <= -30)).astype(bool)
    sell = pd.Series(False, index=g.index)
    return buy, sell, "Hold cash until weekly RSI recovers above 40 after a drawdown of at least 30%."


def signal_cash_until_reclaim_200dma(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    buy = (g["reclaim_200dma"].fillna(False) & (g["drawdown_pct"] <= -20)).astype(bool)
    sell = pd.Series(False, index=g.index)
    return buy, sell, "Hold cash until price reclaims the 200-day moving average after a drawdown of at least 20%."


def signal_dca_after_40pct_drawdown(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    buy = (g["drawdown_pct"] <= -40).fillna(False).astype(bool)
    sell = pd.Series(False, index=g.index)
    return buy, sell, "Start investing contributions after a 40% drawdown from prior peak."


def signal_buy_after_retrospective_peak(months: int) -> Callable[[pd.DataFrame], tuple[pd.Series, pd.Series, str]]:
    def _rule(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
        # Retrospective benchmark: uses the highest price inside the selected historical window.
        # This is intentionally labeled as an oracle-style comparison, not a live trading signal.
        peak_date = g.loc[g["adjusted_close"].idxmax(), "date"]
        target_date = peak_date + pd.DateOffset(months=months)
        buy = pd.Series(False, index=g.index)
        valid = g[g["date"] >= target_date]
        if not valid.empty:
            buy.loc[valid.index[0]] = True
        sell = pd.Series(False, index=g.index)
        return buy, sell, f"Retrospective benchmark: buy {months} months after selected-window peak. This uses hindsight."

    return _rule


STRATEGY_RULES: dict[str, Callable[[pd.DataFrame], tuple[pd.Series, pd.Series, str]]] = {
    "buy_and_hold": signal_buy_and_hold,
    "sell_exit_buy_recovery": signal_sell_exit_buy_recovery,
    "cash_until_40pct_drawdown": signal_cash_until_drawdown(40),
    "cash_until_60pct_drawdown": signal_cash_until_drawdown(60),
    "cash_until_rsi_recovery": signal_cash_until_rsi_recovery,
    "cash_until_reclaim_200dma": signal_cash_until_reclaim_200dma,
    "dca_after_40pct_drawdown": signal_dca_after_40pct_drawdown,
    "buy_6_months_after_retrospective_peak": signal_buy_after_retrospective_peak(6),
    "buy_12_months_after_retrospective_peak": signal_buy_after_retrospective_peak(12),
}


def simulate_strategy(
    g: pd.DataFrame,
    ticker: str,
    strategy: str,
    starting_capital: float = STARTING_CAPITAL_DEFAULT,
    monthly_contribution: float = MONTHLY_CONTRIBUTION_DEFAULT,
    transaction_cost_bps: float = 0.0,
) -> BacktestResult:
    """Simulate a strategy using next-day execution to avoid same-close look-ahead.

    Signal at date t is executed at date t+1 close. The strategy cannot trade on information
    from the future. Retrospective peak strategies are explicitly marked as hindsight benchmarks.
    """
    if strategy not in STRATEGY_RULES:
        raise ValueError(f"Unknown strategy {strategy}. Available={list(STRATEGY_RULES)}")

    buy_signal_raw, sell_signal_raw, notes = STRATEGY_RULES[strategy](g)
    buy_exec = buy_signal_raw.shift(1).fillna(False).astype(bool)
    sell_exec = sell_signal_raw.shift(1).fillna(False).astype(bool)

    # For buy-and-hold, allow the first row to invest because no future information is involved.
    if strategy == "buy_and_hold":
        buy_exec.iloc[0] = True

    cash = float(starting_capital)
    shares = 0.0
    invested = False
    trade_count = 0
    records: list[dict] = []
    last_month = None
    cost_rate = transaction_cost_bps / 10_000

    for i, row in g.iterrows():
        date = row["date"]
        price = float(row["adjusted_close"])
        current_month = (date.year, date.month)

        if last_month is not None and current_month != last_month and monthly_contribution > 0:
            cash += monthly_contribution
        last_month = current_month

        if bool(sell_exec.iloc[i]) and invested and shares > 0:
            proceeds = shares * price * (1 - cost_rate)
            cash += proceeds
            shares = 0.0
            invested = False
            trade_count += 1
            debug_print(f"SELL {ticker} on {date.date()} @ {price:.2f} via {strategy}")

        if bool(buy_exec.iloc[i]) and not invested and cash > 0:
            shares = cash * (1 - cost_rate) / price
            cash = 0.0
            invested = True
            trade_count += 1
            debug_print(f"BUY {ticker} on {date.date()} @ {price:.2f} via {strategy}")

        # DCA strategy: after trigger, monthly contributions are invested as they arrive if already invested.
        if strategy == "dca_after_40pct_drawdown" and bool(buy_exec.iloc[i]) and cash > 0 and price > 0:
            add_shares = cash * (1 - cost_rate) / price
            shares += add_shares
            cash = 0.0
            invested = shares > 0

        portfolio_value = cash + shares * price
        records.append(
            {
                "date": date,
                "ticker": ticker,
                "strategy": strategy,
                "price": price,
                "cash": cash,
                "shares": shares,
                "invested": invested,
                "portfolio_value": portfolio_value,
                "buy_executed": bool(buy_exec.iloc[i]),
                "sell_executed": bool(sell_exec.iloc[i]),
                "signal_label": row.get("signal_label", ""),
                "phase_label": row.get("phase_label", ""),
            }
        )

    equity = pd.DataFrame(records)
    metrics = calculate_performance(equity, starting_capital, monthly_contribution)

    return BacktestResult(
        ticker=ticker,
        strategy=strategy,
        start_date=equity["date"].iloc[0],
        end_date=equity["date"].iloc[-1],
        starting_capital=starting_capital,
        monthly_contribution=monthly_contribution,
        ending_value=float(equity["portfolio_value"].iloc[-1]),
        total_return_pct=metrics["total_return_pct"],
        annualized_return_pct=metrics["annualized_return_pct"],
        max_drawdown_pct=metrics["max_drawdown_pct"],
        volatility_pct=metrics["volatility_pct"],
        sharpe_ratio=metrics["sharpe_ratio"],
        time_to_recover_days=metrics["time_to_recover_days"],
        trades=trade_count,
        notes=notes,
        equity_curve=equity,
    )


def calculate_performance(equity: pd.DataFrame, starting_capital: float, monthly_contribution: float) -> dict:
    values = equity["portfolio_value"].astype(float)
    dates = pd.to_datetime(equity["date"])
    days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
    years = days / 365.25

    # Approximate contributed capital. This keeps the metric honest for monthly contribution scenarios.
    month_count = dates.dt.to_period("M").nunique() - 1
    contributed = starting_capital + max(month_count, 0) * monthly_contribution

    ending_value = float(values.iloc[-1])
    total_return_pct = (ending_value / contributed - 1.0) * 100.0 if contributed > 0 else np.nan
    annualized_return_pct = ((ending_value / contributed) ** (1 / years) - 1.0) * 100.0 if contributed > 0 else np.nan

    rolling_peak = values.cummax()
    drawdown = values / rolling_peak - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)

    daily_returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volatility_pct = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0) if not daily_returns.empty else np.nan
    sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)) if daily_returns.std() > 0 else np.nan

    time_to_recover_days = compute_time_to_recover(equity)
    return {
        "total_return_pct": float(total_return_pct),
        "annualized_return_pct": float(annualized_return_pct),
        "max_drawdown_pct": max_drawdown_pct,
        "volatility_pct": volatility_pct,
        "sharpe_ratio": sharpe_ratio,
        "time_to_recover_days": time_to_recover_days,
    }


def compute_time_to_recover(equity: pd.DataFrame) -> int | None:
    values = equity["portfolio_value"].astype(float).to_numpy()
    dates = pd.to_datetime(equity["date"]).reset_index(drop=True)
    running_peak = np.maximum.accumulate(values)
    drawdowns = values / running_peak - 1.0
    trough_idx = int(np.argmin(drawdowns))
    pre_trough_peak_value = running_peak[trough_idx]
    after = np.where(values[trough_idx:] >= pre_trough_peak_value)[0]
    if len(after) == 0:
        return None
    recovery_idx = trough_idx + int(after[0])
    return int((dates.iloc[recovery_idx] - dates.iloc[trough_idx]).days)


def summarize_result(result: BacktestResult) -> dict:
    return {
        "ticker": result.ticker,
        "strategy": result.strategy,
        "start_date": result.start_date.date(),
        "end_date": result.end_date.date(),
        "starting_capital": result.starting_capital,
        "monthly_contribution": result.monthly_contribution,
        "ending_value": result.ending_value,
        "total_return_pct": result.total_return_pct,
        "annualized_return_pct": result.annualized_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "volatility_pct": result.volatility_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "time_to_recover_days": result.time_to_recover_days,
        "trades": result.trades,
        "notes": result.notes,
    }


def compare_strategies(
    df: pd.DataFrame,
    ticker: str,
    strategies: list[str],
    start: str | None,
    end: str | None,
    starting_capital: float,
    monthly_contribution: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    g = prepare_backtest_frame(df, ticker, start, end)
    summaries = []
    curves = {}
    for strategy in strategies:
        try:
            result = simulate_strategy(g, ticker, strategy, starting_capital, monthly_contribution)
            summaries.append(summarize_result(result))
            curves[strategy] = result.equity_curve
        except Exception as exc:  # noqa: BLE001
            logger.exception("Strategy failed: %s", strategy)
            summaries.append({"ticker": ticker, "strategy": strategy, "error": str(exc)})
    return pd.DataFrame(summaries), curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a historical strategy backtest.")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--start", default="1999-01-01")
    parser.add_argument("--end", default="2003-12-31")
    parser.add_argument("--strategy", default="sell_exit_buy_recovery", choices=list(STRATEGY_RULES))
    parser.add_argument("--capital", type=float, default=STARTING_CAPITAL_DEFAULT)
    parser.add_argument("--monthly", type=float, default=MONTHLY_CONTRIBUTION_DEFAULT)
    args = parser.parse_args()

    df = load_scored_data()
    g = prepare_backtest_frame(df, args.ticker, args.start, args.end)
    result = simulate_strategy(g, args.ticker, args.strategy, args.capital, args.monthly)
    summary = summarize_result(result)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
