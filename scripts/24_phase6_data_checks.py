"""Phase 6, step 2b: data-quality checks against the pre-registered drop rules.

Checks per ticker: first date, missing values, non-positive prices, alignment
with the US trading calendar (SPY's days), longest run of missing US trading
days, and the largest one-day moves (|log return| > 50% would flag a split or
data error). Also Close vs AdjClose jumps (unadjusted split).

Run:  PYTHONPATH=src .venv/bin/python scripts/24_phase6_data_checks.py
Output: figures/phase6_data_checks.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.data import load_prices
from signaldesk.universe import ASSET_CLASS, RAW6_DIR, UNIVERSE, load_asset, us_trading_days

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
cal = us_trading_days()
rows = []
for t in UNIVERSE:
    raw = load_prices(t, RAW6_DIR)
    p = load_asset(t)                               # on the US calendar
    missing = cal.difference(p.index)
    # Longest run of consecutive missing US trading days.
    run, longest = 0, 0
    pos = set(p.index)
    for d in cal:
        run = run + 1 if d not in pos else 0
        longest = max(longest, run)
    r = 100 * np.log(p / p.shift(1)).dropna()
    big = r.abs().nlargest(3)
    # Split check: day-to-day change in AdjClose/Close ratio far from 1 = an
    # unadjusted split or a dividend error.
    ratio = (raw["AdjClose"] / raw["Close"])
    ratio_jump = (ratio / ratio.shift(1) - 1).abs().max()
    rows.append({
        "ticker": t, "class": ASSET_CLASS[t],
        "first_date": raw.index[0].date(), "rows_raw": len(raw), "rows_on_US_calendar": len(p),
        "nan_values": int(raw.isna().sum().sum()),
        "nonpositive_prices": int((raw[["Open", "High", "Low", "Close", "AdjClose"]] <= 0).sum().sum()),
        "US_days_missing": len(missing), "longest_missing_run": longest,
        "max_abs_daily_move_%": round(float(big.iloc[0]), 2),
        "max_move_date": big.index[0].date(),
        "2nd_3rd_moves": "; ".join(f"{d.date()} {v:+.1f}%" for d, v in r.loc[big.index[1:]].items()),
        "max_adj_ratio_jump_%": round(100 * float(ratio_jump), 2),
        "ann_vol_2015_21_%": round(float(r.loc[:"2021"].std() * np.sqrt(252)), 1),
    })
T = pd.DataFrame(rows)
T.to_csv(FIG_DIR / "phase6_data_checks.csv", index=False)
pd.set_option("display.width", 250)
print(T.to_string(index=False))

btc = load_prices("BTC-USD", RAW6_DIR)
wk = btc.index.day_of_week >= 5
print(f"\nBTC-USD raw: {len(btc)} calendar days, {wk.sum()} on weekends; "
      f"{len(btc) - len(load_asset('BTC-USD'))} days fold into the next US trading day's return.")
