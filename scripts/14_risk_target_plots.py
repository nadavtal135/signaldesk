"""Phase 3, step 8: equity curves and drawdowns.

Figures (cap 1.0 with the frozen target from step 4, net of costs):
- figures/risk_target_test_equity_drawdown.png: TEST period (2022-today),
  strategy vs buy-and-hold vs placebo (constant exposure = strategy's average).
  Equity restarts at 1.0 on the first test day.
- figures/risk_target_full_equity.png: full 2015-2026 history, strategy vs
  buy-and-hold, log scale, train/test boundary marked.

Run:  PYTHONPATH=src .venv/bin/python scripts/14_risk_target_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from signaldesk.backtest import constant_strategy, run_strategy
from signaldesk.data import TICKERS, load_prices

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
CAP = 1.0
TARGET = json.loads((FIG_DIR / "risk_target_tuning.json").read_text())["best_target_by_cap"][str(CAP)]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
COL = {"Buy & hold": "#257de0", "Risk target": "#f47230", "Placebo": "#00b894"}
plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 8.5, "axes.titlecolor": INK,
})


def equity(net):
    return (1 + net).cumprod()


def drawdown(net):
    eq = equity(net)
    return 100 * (eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1)


# ---- Figure 1: test period, equity + drawdown, 4 rows x 3 cols ---------------
fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [1.3, 1, 1.3, 1]})
for i, t in enumerate(TICKERS):
    price = load_prices(t)["AdjClose"]
    bt = run_strategy(price, TARGET, CAP)
    R = bt["R"]
    avg_e = bt.loc[TEST_START:, "exposure"].mean()
    series = {
        "Buy & hold": constant_strategy(R, 1.0).loc[TEST_START:, "net"],
        "Risk target": bt.loc[TEST_START:, "net"],
        "Placebo": constant_strategy(R, avg_e).loc[TEST_START:, "net"],
    }
    row, col = 2 * (i // 3), i % 3
    ax_eq, ax_dd = axes[row, col], axes[row + 1, col]
    for name, net in series.items():
        label = {"Buy & hold": "Buy & hold (exposure 1.0)",
                 "Risk target": f"Risk target {int(100*TARGET)}%, cap {CAP}",
                 "Placebo": "Placebo: constant = strategy's avg exposure"}[name]
        ax_eq.plot(net.index, equity(net), color=COL[name], linewidth=1.2, label=label)
        ax_dd.plot(net.index, drawdown(net), color=COL[name], linewidth=1.0)
    ax_eq.set_title(f"{t}  (placebo exposure {avg_e:.2f})", loc="left")
    ax_eq.axhline(1.0, color=INK2, linewidth=0.6)
    if col == 0:
        ax_eq.set_ylabel("Account value\n(start = 1.0)")
        ax_dd.set_ylabel("Drawdown, %")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=3,
           fontsize=8.5, frameon=False)
fig.suptitle("TEST period 2022-today, net of 0.05% costs: equity (top) and drawdown from peak (bottom)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG_DIR / "risk_target_test_equity_drawdown.png", dpi=150)
plt.close(fig)

# ---- Figure 2: full history, log scale --------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for ax, t in zip(axes.flat, TICKERS):
    price = load_prices(t)["AdjClose"]
    bt = run_strategy(price, TARGET, CAP)
    bh = constant_strategy(bt["R"], 1.0)
    ax.plot(bh.index, equity(bh["net"]), color=COL["Buy & hold"], linewidth=1.1, label="Buy & hold")
    ax.plot(bt.index, equity(bt["net"]), color=COL["Risk target"], linewidth=1.1,
            label=f"Risk target {int(100*TARGET)}%, cap {CAP}")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvline(np.datetime64(TEST_START), color=INK2, linestyle=":", linewidth=0.9)
    ax.set_title(t, loc="left")
for ax in axes[:, 0]:
    ax.set_ylabel("Account value (start = 1, log scale)")
axes[0, 0].text(np.datetime64(TEST_START), axes[0, 0].get_ylim()[1], " test →",
                va="top", color=INK2, fontsize=8)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2,
           fontsize=8.5, frameon=False)
fig.suptitle("Full history 2015-2026, net of costs (dotted line = start of the test period)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG_DIR / "risk_target_full_equity.png", dpi=150)
