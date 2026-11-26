"""Phase 3, step 3: trading costs and turnover.

Uses an ILLUSTRATIVE target of 15%/yr (tuning is step 4) with caps 1.0 and 1.5,
over the full 2015-2026 sample, to show how much the strategy trades and what
0.05% per trade costs per year. Returns here are simple averages x 252 (not
compounded) -- good enough to measure the cost drag; full metrics come in step 4.

Run:  PYTHONPATH=src .venv/bin/python scripts/10_costs_turnover.py
Output: figures/costs_turnover.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.returns import log_returns
from signaldesk.strategy import (COST_RATE, TRADING_DAYS, exposure_from_vol, forecast_vol_annual,
                                 strategy_returns, turnover_per_year)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TARGET = 0.15

rows = []
for t in TICKERS:
    price = load_prices(t)["AdjClose"]
    R = price.pct_change()
    vol = forecast_vol_annual(log_returns(price))
    start = vol.first_valid_index()          # first day with a forecast
    R = R.loc[start:]
    for cap in [1.0, 1.5]:
        e = exposure_from_vol(vol, TARGET, cap).loc[start:]
        gross = strategy_returns(R, e, cost_rate=0.0)
        net = strategy_returns(R, e, cost_rate=COST_RATE)
        # Average daily change in exposure: how jittery the gain is.
        rows.append({
            "ticker": t, "cap": cap,
            "avg_exposure": e.mean(),
            "turnover_per_year": turnover_per_year(e),
            "trading_days_with_a_trade_%": 100 * (e.diff().abs() > 1e-12).mean(),
            "gross_mean_return_%/yr": 100 * gross.mean() * TRADING_DAYS,
            "net_mean_return_%/yr": 100 * net.mean() * TRADING_DAYS,
            "cost_drag_%/yr": 100 * (gross.mean() - net.mean()) * TRADING_DAYS,
        })
    # Buy-and-hold reference: one purchase at the start, then nothing.
    bh = pd.Series(1.0, index=R.index)
    rows.append({"ticker": t, "cap": "buy&hold", "avg_exposure": 1.0,
                 "turnover_per_year": turnover_per_year(bh),
                 "trading_days_with_a_trade_%": 0.0,
                 "gross_mean_return_%/yr": 100 * strategy_returns(R, bh).mean() * TRADING_DAYS,
                 "net_mean_return_%/yr": 100 * strategy_returns(R, bh, COST_RATE).mean() * TRADING_DAYS,
                 "cost_drag_%/yr": 100 * (strategy_returns(R, bh).mean()
                                          - strategy_returns(R, bh, COST_RATE).mean()) * TRADING_DAYS})

table = pd.DataFrame(rows).set_index(["ticker", "cap"])
table.to_csv(FIG_DIR / "costs_turnover.csv", float_format="%.4f")
pd.set_option("display.width", 200)
print(table.round(3).to_string())
