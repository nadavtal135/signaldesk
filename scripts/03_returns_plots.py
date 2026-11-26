"""Step 3: compute daily log returns and plot price vs. returns per ticker.

Run:  PYTHONPATH=src .venv/bin/python scripts/03_returns_plots.py
Outputs: figures/price_vs_returns_<TICKER>.png and figures/returns_stats.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to files, no screen needed
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

from signaldesk.data import TICKERS, load_prices
from signaldesk.returns import log_returns

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
SERIES = "#257de0"    # one series per panel -> one colour
INK = "#191919"
INK2 = "#666666"
GRID = "#e6e6e6"

plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})

rows = []
for t in TICKERS:
    price = load_prices(t)["AdjClose"]
    r = log_returns(price)

    # Summary statistics. Annualising assumes ~252 trading days/year:
    # mean scales with N, standard deviation (volatility) with sqrt(N),
    # exactly like summing N independent noise samples.
    rows.append({
        "ticker": t,
        "n_days": len(r),
        "mean_daily_%": 100 * r.mean(),
        "std_daily_%": 100 * r.std(),
        "ann_return_%": 100 * r.mean() * 252,
        "ann_vol_%": 100 * r.std() * np.sqrt(252),
        "skew": stats.skew(r),
        # Excess kurtosis: 0 for a Gaussian. >0 means "heavy tails":
        # large spikes are far more common than Gaussian noise would give.
        "excess_kurtosis": stats.kurtosis(r),
        "worst_day_%": 100 * r.min(),
        "worst_day": r.idxmin().date(),
        "best_day_%": 100 * r.max(),
        "best_day": r.idxmax().date(),
        # How many days exceed 4 sigma, vs. what a Gaussian predicts.
        "days_beyond_4sigma": int((np.abs(r - r.mean()) > 4 * r.std()).sum()),
        "gaussian_expect_4sigma": round(len(r) * 2 * stats.norm.sf(4), 2),
    })

    # Two stacked panels, shared time axis. Separate y-axes on purpose:
    # price and returns are different quantities, so they never share a scale.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    # Log scale for price: a constant % growth rate becomes a straight line,
    # the same reason we plot power in dB.
    ax1.plot(price.index, price.values, color=SERIES, linewidth=1.2)
    ax1.set_yscale("log")
    # Plain dollar tick labels instead of 6x10^2-style scientific notation.
    dollars = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    # Label only 1-2-5 steps (like a scope's V/div), so labels never collide.
    ax1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax1.yaxis.set_major_formatter(dollars)
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_ylabel("Adjusted price, $ (log scale)")
    ax1.set_title(f"{t}: price (the integrated signal) vs. daily log return (its increments)",
                  loc="left", fontsize=11)

    ax2.plot(r.index, 100 * r.values, color=SERIES, linewidth=0.5)
    ax2.axhline(0, color=INK2, linewidth=0.8)
    ax2.set_ylabel("Daily log return, %")
    ax2.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"price_vs_returns_{t}.png", dpi=150)
    plt.close(fig)

table = pd.DataFrame(rows).set_index("ticker")
table.round(3).to_csv(FIG_DIR / "returns_stats.csv")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(table.round(2))
