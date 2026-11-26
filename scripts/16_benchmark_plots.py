"""Phase 4, step 7: summary figures for the model benchmark.

Run (after 15_model_benchmark.py):  PYTHONPATH=src .venv/bin/python scripts/16_benchmark_plots.py
Outputs: figures/benchmark_summary.png, figures/benchmark_spy_2025.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signaldesk.data import TICKERS

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
COLORS = {"GARCH(1,1)-t": "#257de0", "GJR-GARCH(1,1)-t": "#f47230", "HAR-style": "#00b894",
          "EWMA λ=0.94": "#6a4ce2", "Constant (train avg)": "#191919"}
MARKERS = {"GARCH(1,1)-t": "o", "GJR-GARCH(1,1)-t": "^", "HAR-style": "s"}
CHALLENGERS = list(MARKERS)
plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})

test = pd.read_csv(FIG_DIR / "benchmark_test.csv")
by_year = pd.read_csv(FIG_DIR / "benchmark_by_year.csv")
cx = pd.read_csv(FIG_DIR / "benchmark_complexity.csv").set_index("model")

# ---- Figure 1: test-period gain per ticker + year-by-year mean gain ---------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
y = np.arange(len(TICKERS))[::-1]
for off, m in zip([-0.2, 0.0, 0.2], CHALLENGERS):
    s = test[test.model == m].set_index("ticker").loc[TICKERS]
    sig = s["DM_t_QLIKE_vs_EWMA"] < -2
    ax1.scatter(s.QLIKE_gain_vs_EWMA, y + off, color=COLORS[m], marker=MARKERS[m], s=42,
                edgecolor="white", linewidth=0.7, zorder=3,
                label=f"{m}  (pooled DM t = {cx.loc[m, 'pooled_DM_t_QLIKE_vs_EWMA']:+.1f})")
    # Ring around per-ticker significant wins (DM t < -2).
    ax1.scatter(s.QLIKE_gain_vs_EWMA[sig], (y + off)[sig.values], s=150, facecolor="none",
                edgecolor=COLORS[m], linewidth=1.0, zorder=2)
ax1.axvline(0, color=INK2, linestyle="--", linewidth=1)
ax1.set_yticks(y, TICKERS)
ax1.set_xlabel("QLIKE gain vs EWMA  (> 0 = model beats EWMA)")
ax1.set_title("TEST 2022-today, per ticker (ring = significant, DM t < -2)", loc="left")
ax1.legend(loc="lower right", fontsize=7.5, frameon=False)

yr = by_year[by_year.model.isin(CHALLENGERS)].groupby(["model", "year"]).QLIKE_gain_vs_EWMA.mean().unstack(0)
ax2.axvspan(2021.5, yr.index.max() + 0.5, color="#f2f2f2", zorder=0)
ax2.text(2021.6, 0.0, "test period", va="bottom", color=INK2, fontsize=8, transform=ax2.get_xaxis_transform())
for m in CHALLENGERS:
    ax2.plot(yr.index, yr[m], color=COLORS[m], marker=MARKERS[m], linewidth=1.4, markersize=5, label=m)
ax2.axhline(0, color=INK2, linestyle="--", linewidth=1)
ax2.set_xticks(yr.index)
ax2.set_ylabel("QLIKE gain vs EWMA, mean of 6 tickers")
ax2.set_title("Year by year (2026 = Jan-Nov): GARCH-type models beat EWMA in 6 of 9 years, lost 2021-2023", loc="left")
fig.suptitle("Do GARCH, GJR-GARCH or HAR forecast tomorrow's variance better than EWMA (λ=0.94)?",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG_DIR / "benchmark_summary.png", dpi=150)
plt.close(fig)

# ---- Figure 2: SPY forecasts around the April 2025 spike ---------------------
losses_path = FIG_DIR / "benchmark_losses.csv"
if losses_path.exists():
    L = pd.read_csv(losses_path, parse_dates=["date"])
    spy = L[(L.ticker == "SPY") & (L.date >= "2025-02-15") & (L.date <= "2025-08-31")]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    y_ = spy[spy.model == "EWMA λ=0.94"].set_index("date")["y"]
    ax.scatter(y_.index, y_.clip(lower=1e-3), s=6, color="#b3b3b3", label="realised r² (target)", zorder=1)
    for m in ["EWMA λ=0.94", "GARCH(1,1)-t", "GJR-GARCH(1,1)-t", "HAR-style", "Constant (train avg)"]:
        s = spy[spy.model == m].set_index("date")["h"]
        ax.plot(s.index, s.values, color=COLORS[m], linewidth=1.4,
                linestyle="--" if m.startswith("Constant") else "-", label=m, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 2e2)
    ax.set_ylabel("Daily variance, %² (log scale)")
    ax.set_xlabel("Target day (forecast made the evening before)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.set_title("SPY, spring 2025: GARCH-type models jump faster on the shock and relax "
                 "toward their long-run level; EWMA lags on both", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "benchmark_spy_2025.png", dpi=150)
