"""Phase 2, step 2: compute the volatility estimators and plot them for SPY.

Run:  PYTHONPATH=src .venv/bin/python scripts/05_vol_estimators.py
Output: figures/vol_estimators_SPY.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaldesk.data import load_prices
from signaldesk.returns import log_returns
from signaldesk.volatility import constant_variance, ewma_variance, rolling_variance

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END = "2021-12-31"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})

r = log_returns(load_prices("SPY")["AdjClose"])

# Fixed colour per estimator (colour follows the estimator, never its rank).
estimators = {
    "Boxcar N=10":  ("#257de0", rolling_variance(r, 10)),
    "Boxcar N=20":  ("#00b894", rolling_variance(r, 20)),
    "Boxcar N=60":  ("#6a4ce2", rolling_variance(r, 60)),
    "EWMA λ=0.94":  ("#f47230", ewma_variance(r, 0.94)),
    "EWMA λ=0.97":  ("#ec4647", ewma_variance(r, 0.97)),
    "Constant (train avg)": ("#191919", constant_variance(r, TRAIN_END)),
}

# Units: daily variance in %^2, i.e. (100*r)^2. A value of 1 means a typical
# move of about 1% per day.
r2 = (100 * r) ** 2
r2_plot = r2[r2 > 0]  # log axis can't show exact zeros (a handful of flat days)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5),
                               gridspec_kw={"height_ratios": [1.1, 1]})
for ax, (lo, hi) in [(ax1, (None, None)), (ax2, ("2020-01-15", "2020-07-31"))]:
    rr = r2_plot.loc[lo:hi]
    ax.scatter(rr.index, rr.values, s=3, color="#b3b3b3", linewidths=0,
               label="r² (one noisy sample of the power per day)", zorder=1)
    for name, (color, h) in estimators.items():
        hh = (1e4 * h).loc[lo:hi]
        style = "--" if name.startswith("Constant") else "-"
        ax.plot(hh.index, hh.values, style, color=color, linewidth=1.3, label=name, zorder=2)
    ax.set_yscale("log")
    ax.set_ylabel("Daily variance, %²  (log scale)")
ax1.set_ylim(1e-3, 3e2)  # a few near-zero r² days fall below; they carry no extra information
ax1.axvline(np.datetime64(TRAIN_END), color=INK2, linewidth=0.8, linestyle=":")
ax1.text(np.datetime64(TRAIN_END), ax1.get_ylim()[1], " test period →", va="top",
         color=INK2, fontsize=8)
ax1.text(np.datetime64(TRAIN_END), ax1.get_ylim()[1], "← train period ", va="top",
         ha="right", color=INK2, fontsize=8)
ax1.set_title("All data, 2015-2026", loc="left")
ax2.set_title("Zoom: COVID crash, Feb-Jul 2020 (shows how fast each filter reacts)", loc="left")
ax2.set_ylim(1e-2, 2e2)
ax2.set_xlabel("Date (estimate made at the close of day t = forecast for day t+1)")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=4,
           fontsize=8, frameon=False, markerscale=3)
fig.suptitle("SPY: squared daily returns and causal low-pass estimates of their power",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(FIG_DIR / "vol_estimators_SPY.png", dpi=150)

# A few numbers to quote: reaction to the crash and the "ghost" step of the boxcar.
for name, (_, h) in estimators.items():
    hh = 1e4 * h
    print(f"{name:22s} peak 2020 = {hh.loc['2020'].max():6.2f} %² on {hh.loc['2020'].idxmax().date()}"
          f"   level on 2020-02-21 = {hh.loc['2020-02-21']:.2f}")
