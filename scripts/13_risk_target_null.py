"""Phase 3, step 6: null test -- does the forecast's TIMING matter?

DSP view: keep the exact same set of gain values, but scramble WHEN each one is
applied. If the result on the real (time-aligned) forecast is no better than on
scrambled copies, the alignment between the gain and the market's loud periods
carries no value.

Details:
- Shuffle the annualised volatility forecast WITHIN the train period and WITHIN
  the test period (each period keeps its own set of values), rebuild the
  exposure with the frozen target/cap, and rerun the backtest. 200 times.
- Statistic: TEST-period Sharpe ratio (and max drawdown as a second view).
- Costs: a shuffled forecast jumps around every day, so it trades far more
  than the real one. Charging costs would handicap the shuffles for a reason
  unrelated to timing, so the comparison uses GROSS (cost-free) results for
  both real and shuffled. (Net-of-cost real results are in step 4.)
- p = (1 + #shuffles >= real) / (1 + 200), one-sided ("real is better").

Run:  PYTHONPATH=src .venv/bin/python scripts/13_risk_target_null.py
Outputs: figures/risk_target_null.png, figures/risk_target_null.csv
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signaldesk.backtest import metrics, run_from_vol
from signaldesk.data import TICKERS, load_prices
from signaldesk.returns import log_returns
from signaldesk.strategy import forecast_vol_annual

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
N_SHUFFLES = 200
best = {float(c): tv for c, tv in
        json.loads((FIG_DIR / "risk_target_tuning.json").read_text())["best_target_by_cap"].items()}

INK, INK2, GRID, BAR, BLUE = "#191919", "#666666", "#e6e6e6", "#b3b3b3", "#257de0"
plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})


def shuffle_within_periods(vol: pd.Series, rng) -> pd.Series:
    v = vol.dropna()
    tr, te = v.loc[:TRAIN_END].values, v.loc[TEST_START:].values
    out = pd.Series(np.concatenate([rng.permutation(tr), rng.permutation(te)]), index=v.index)
    return out.reindex(vol.index)


def test_stats(price, vol, tv, cap):
    m = metrics(run_from_vol(price, vol, tv, cap, cost_rate=0.0).loc[TEST_START:])
    return m["sharpe"], m["max_drawdown_%"]


rng = np.random.default_rng(7)
rows, null_sharpe = [], {}
for t in TICKERS:
    price = load_prices(t)["AdjClose"]
    vol = forecast_vol_annual(log_returns(price))
    for cap, tv in best.items():
        real_s, real_dd = test_stats(price, vol, tv, cap)
        sims = np.array([test_stats(price, shuffle_within_periods(vol, rng), tv, cap)
                         for _ in range(N_SHUFFLES)])
        null_sharpe[(t, cap)] = sims[:, 0]
        rows.append({
            "ticker": t, "cap": cap, "target": tv,
            "real_sharpe": real_s, "null_sharpe_median": np.median(sims[:, 0]),
            "p_sharpe": (1 + np.sum(sims[:, 0] >= real_s)) / (1 + N_SHUFFLES),
            "real_maxdd_%": real_dd, "null_maxdd_median_%": np.median(sims[:, 1]),
            # Drawdowns are negative: "better" = closer to zero = larger.
            "p_maxdd": (1 + np.sum(sims[:, 1] >= real_dd)) / (1 + N_SHUFFLES),
        })
table = pd.DataFrame(rows)
table.to_csv(FIG_DIR / "risk_target_null.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 200)
print(table.round(3).to_string(index=False))

# Figure: cap 1.0 (the main configuration), Sharpe histograms.
cap = 1.0
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
for ax, t in zip(axes.flat, TICKERS):
    r = table[(table.ticker == t) & (table.cap == cap)].iloc[0]
    ax.hist(null_sharpe[(t, cap)], bins=25, color=BAR, edgecolor="white", linewidth=0.5,
            label=f"{N_SHUFFLES} shuffled forecasts (timing scrambled)")
    ax.axvline(r.real_sharpe, color=BLUE, linewidth=2, label="real forecast")
    ax.set_title(t, loc="left")
    ax.text(0.02, 0.95, f"p = {r.p_sharpe:.2f}", transform=ax.transAxes,
            ha="left", va="top", color=INK, fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("Test-period Sharpe ratio (gross of costs)")
for ax in axes[:, 0]:
    ax.set_ylabel("Number of shuffles")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2,
           fontsize=8, frameon=False)
fig.suptitle(f"Null test, 2022-today: real volatility timing vs. shuffled timing "
             f"(target {int(100*best[cap])}%, cap {cap})", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG_DIR / "risk_target_null.png", dpi=150)
