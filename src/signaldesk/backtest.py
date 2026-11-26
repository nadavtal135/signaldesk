"""Backtest plumbing and performance metrics.

Metric definitions (finance terms, in plain words):
- Annualised return: the constant yearly growth rate that would turn the
  starting money into the ending money ("compound annual growth rate").
- Annualised volatility: the standard deviation of daily returns scaled to a
  year (x sqrt(252)) -- the size of the account's typical swings.
- Sharpe ratio: return earned per unit of volatility (mean daily return /
  std of daily return, x sqrt(252)). In DSP terms, a signal-to-noise ratio of
  the P&L. Cash earns 0% here, so no risk-free rate is subtracted.
- Max drawdown: the largest drop from a previous peak of the account value to
  a later low -- the worst loss someone who bought at the top would have seen.
- Worst day: the single largest one-day loss.
- Turnover: how much of the account is traded per year (see strategy.turnover).
"""
import numpy as np
import pandas as pd

from signaldesk.returns import log_returns
from signaldesk.strategy import (COST_RATE, TRADING_DAYS, exposure_from_vol, forecast_vol_annual,
                                 strategy_returns, turnover)


def run_strategy(price: pd.Series, target_vol: float, max_exposure: float,
                 cost_rate: float = COST_RATE) -> pd.DataFrame:
    """Full causal pipeline for one stock. Returns a frame with the stock's
    simple return R, the exposure decided at each close, and the net P&L.
    The series starts on the first decision day: at that close we buy (and
    pay the entry cost) but earn nothing yet. Earlier days are dropped."""
    vol = forecast_vol_annual(log_returns(price))
    return run_from_vol(price, vol, target_vol, max_exposure, cost_rate)


def run_from_vol(price: pd.Series, vol: pd.Series, target_vol: float, max_exposure: float,
                 cost_rate: float = COST_RATE) -> pd.DataFrame:
    """Same as run_strategy, but with the volatility forecast supplied by the
    caller (used by the null test to plug in a shuffled forecast)."""
    e = exposure_from_vol(vol, target_vol, max_exposure)
    R = price.pct_change()
    # "trade" is computed on the FULL history, so slicing a period later does
    # not mistake an existing position for a new purchase.
    df = pd.DataFrame({"R": R, "exposure": e, "trade": turnover(e),
                       "net": strategy_returns(R, e, cost_rate)})
    return df.loc[vol.first_valid_index():]


def constant_strategy(R: pd.Series, level: float, cost_rate: float = COST_RATE) -> pd.DataFrame:
    """Hold a fixed exposure every day (buy-and-hold = level 1.0).
    Same convention as run_strategy: buy at the close of the first day
    (paying the entry cost), earn returns from the second day on."""
    e = pd.Series(level, index=R.index)
    return pd.DataFrame({"R": R, "exposure": e, "trade": turnover(e),
                         "net": strategy_returns(R, e, cost_rate)})


def metrics(bt: pd.DataFrame) -> dict:
    """Performance summary of a backtest frame (columns net, exposure, trade)
    over its own span: the account restarts at 1.0 on its first day."""
    net = bt["net"]
    equity = (1 + net).cumprod()
    years = len(net) / TRADING_DAYS
    peak = np.maximum.accumulate(np.r_[1.0, equity.values])[1:]
    return {
        "ann_return_%": 100 * (equity.iloc[-1] ** (1 / years) - 1),
        "ann_vol_%": 100 * net.std() * np.sqrt(TRADING_DAYS),
        "sharpe": net.mean() / net.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown_%": 100 * (equity.values / peak - 1).min(),
        "worst_day_%": 100 * net.min(),
        "turnover_per_yr": bt["trade"].sum() / years,
        "avg_exposure": bt["exposure"].mean(),
    }
