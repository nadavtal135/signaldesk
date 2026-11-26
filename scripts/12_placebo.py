"""Phase 3, step 5: the placebo test -- is it the TIMING, or just holding less?

DSP view: the strategy is a time-varying gain g[t] applied to the stock's
returns. Any constant gain g0 < 1 also lowers volatility and drawdowns, simply
by attenuating everything. To credit the *timing* (turning the gain down when
the input is loud), compare against a constant gain with the same average:
  placebo exposure = mean of the strategy's exposure over the same period.
Same average amount invested, zero timing.

Second control (stricter for risk): a constant gain chosen so the account has
the SAME VOLATILITY as the strategy. At equal risk, does timing earn more
return or suffer smaller drawdowns?

Note: both controls use the period's own average (so they peek inside the
period). That's fine for a *diagnostic* control -- they are not strategies
anyone could have traded -- and it can only make the placebo look better.

Note: a constant exposure's Sharpe ratio equals buy-and-hold's (scaling a
signal does not change its SNR), apart from tiny cost differences.

Run:  PYTHONPATH=src .venv/bin/python scripts/12_placebo.py
Output: figures/placebo_metrics.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.backtest import constant_strategy, metrics, run_strategy
from signaldesk.data import TICKERS, load_prices

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
PERIODS = {"train": slice(None, TRAIN_END), "test": slice(TEST_START, None)}
best = {float(c): tv for c, tv in
        json.loads((FIG_DIR / "risk_target_tuning.json").read_text())["best_target_by_cap"].items()}

rows = []
for t in TICKERS:
    price = load_prices(t)["AdjClose"]
    for cap, tv in best.items():
        bt = run_strategy(price, tv, cap)
        R = bt["R"]
        name = f"RiskTarget {int(100*tv)}% cap {cap}"
        for per, sl in PERIODS.items():
            strat = bt.loc[sl]
            m_s = metrics(strat)
            # Placebo 1: same average exposure. Placebo 2: same volatility.
            # Built over the full span then sliced, so the test period carries
            # an existing position (no fake entry trade), like the strategy.
            avg_e = strat["exposure"].mean()
            vol_e = m_s["ann_vol_%"] / metrics(constant_strategy(R, 1.0).loc[sl])["ann_vol_%"]
            m_p1 = metrics(constant_strategy(R, avg_e).loc[sl])
            m_p2 = metrics(constant_strategy(R, vol_e).loc[sl])
            for label, m in [(name, m_s),
                             (f"Placebo: const {avg_e:.2f} (same avg exposure)", m_p1),
                             (f"Placebo: const {vol_e:.2f} (same volatility)", m_p2)]:
                rows.append({"ticker": t, "cap": cap, "period": per, "strategy": label, **m})

table = pd.DataFrame(rows)
table.to_csv(FIG_DIR / "placebo_metrics.csv", index=False, float_format="%.4f")

# Compact view: strategy minus placebo, per ticker (positive return/drawdown diff = strategy better).
pd.set_option("display.width", 220)
cols = ["ann_return_%", "ann_vol_%", "sharpe", "max_drawdown_%", "worst_day_%"]
for cap in best:
    for per in PERIODS:
        print(f"\n===== cap {cap}, target {int(100*best[cap])}%  |  {per.upper()} =====")
        sub = table[(table.cap == cap) & (table.period == per)]
        print(sub.set_index(["ticker", "strategy"])[cols].round(2).to_string())

# ---- Verdict table: strategy MINUS same-volatility placebo -------------------
# At equal risk: positive return diff = timing earned more; positive drawdown
# diff = timing had a SMALLER max drawdown (drawdowns are negative numbers).
diff_rows = []
for (t, cap, per), g in table.groupby(["ticker", "cap", "period"]):
    s = g[g.strategy.str.startswith("RiskTarget")].iloc[0]
    p = g[g.strategy.str.contains("same volatility")].iloc[0]
    diff_rows.append({"cap": cap, "period": per, "ticker": t,
                      "return_diff_pts": s["ann_return_%"] - p["ann_return_%"],
                      "drawdown_diff_pts": s["max_drawdown_%"] - p["max_drawdown_%"]})
diff = pd.DataFrame(diff_rows)
diff.to_csv(FIG_DIR / "placebo_verdict.csv", index=False, float_format="%.3f")
print("\n===== Strategy minus same-volatility placebo (percentage points) =====")
for (cap, per), g in diff.groupby(["cap", "period"], sort=False):
    r, d = g["return_diff_pts"], g["drawdown_diff_pts"]
    print(f"cap {cap} {per:5s}: return diff " + " ".join(f"{v:+5.2f}" for v in r)
          + f"  | wins {int((r > 0).sum())}/6, mean {r.mean():+.2f}"
          + f"   || drawdown diff " + " ".join(f"{v:+5.2f}" for v in d)
          + f"  | wins {int((d > 0).sum())}/6, mean {d.mean():+.2f}")
print("(ticker order:", ", ".join(sorted(TICKERS)) + ")")
