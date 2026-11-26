"""Phase 3, step 4: tune target volatility on 2015-2021 only, then evaluate.

Rules:
- Grid: target vol in {10, 12, 15, 20}%/yr, separately for cap 1.0 and 1.5.
- Selection metric: Sharpe ratio averaged over the six tickers, TRAIN only,
  net of costs. The winner is frozen; TEST (2022-today) is reported once.
- Buy-and-hold is scored on exactly the same days.
- Metrics are computed within each period (equity restarts at 1.0 on the
  first day of the period, so drawdowns don't carry over).

Run:  PYTHONPATH=src .venv/bin/python scripts/11_risk_target_eval.py
Outputs: figures/risk_target_tuning.json, figures/risk_target_metrics.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.backtest import constant_strategy, metrics, run_strategy
from signaldesk.data import TICKERS, load_prices

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
TARGETS = [0.10, 0.12, 0.15, 0.20]
CAPS = [1.0, 1.5]
PERIODS = {"train": slice(None, TRAIN_END), "test": slice(TEST_START, None)}

prices = {t: load_prices(t)["AdjClose"] for t in TICKERS}

# ---- 1. Tune on TRAIN only --------------------------------------------------
grid = {}
for cap in CAPS:
    for tv in TARGETS:
        sharpes = []
        for t in TICKERS:
            bt = run_strategy(prices[t], tv, cap).loc[PERIODS["train"]]
            sharpes.append(metrics(bt)["sharpe"])
        grid[(cap, tv)] = float(np.mean(sharpes))
best = {cap: max(TARGETS, key=lambda tv: grid[(cap, tv)]) for cap in CAPS}
tuning = {"best_target_by_cap": {str(c): best[c] for c in CAPS},
          "train_mean_sharpe_grid": {f"cap={c} target={tv}": round(v, 4) for (c, tv), v in grid.items()}}
(FIG_DIR / "risk_target_tuning.json").write_text(json.dumps(tuning, indent=2))
print("TRAIN mean Sharpe over 6 tickers:")
for cap in CAPS:
    print(f"  cap {cap}: " + "  ".join(f"{int(100*tv)}%: {grid[(cap, tv)]:.3f}" for tv in TARGETS)
          + f"   -> chosen {int(100*best[cap])}%")

# ---- 2. Evaluate the frozen choices + buy-and-hold on train and test ----------
rows = []
for t in TICKERS:
    for cap in CAPS:
        bt = run_strategy(prices[t], best[cap], cap)
        for per, sl in PERIODS.items():
            b = bt.loc[sl]
            rows.append({"ticker": t, "period": per,
                         "strategy": f"RiskTarget {int(100*best[cap])}% cap {cap}",
                         **metrics(b)})
    # Buy-and-hold on the same days as the strategy.
    days = run_strategy(prices[t], best[1.0], 1.0).index
    bh = constant_strategy(prices[t].pct_change().loc[days], 1.0)
    for per, sl in PERIODS.items():
        b = bh.loc[sl]
        rows.append({"ticker": t, "period": per, "strategy": "Buy & hold",
                     **metrics(b)})

table = pd.DataFrame(rows)
table.to_csv(FIG_DIR / "risk_target_metrics.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 220)
for per in PERIODS:
    print(f"\n===== {per.upper()} =====")
    print(table[table.period == per].drop(columns="period")
          .set_index(["ticker", "strategy"]).round(2).to_string())
