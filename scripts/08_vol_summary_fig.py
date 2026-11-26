"""Phase 2, step 6: one summary figure of the train/test scores from step 3.

Run (after 06_vol_evaluation.py):  PYTHONPATH=src .venv/bin/python scripts/08_vol_summary_fig.py
Output: figures/vol_scores_summary.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signaldesk.data import TICKERS

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
# Same colour per estimator as in vol_estimators_SPY.png.
COLORS = {"Boxcar N=10": "#257de0", "Boxcar N=20": "#00b894", "Boxcar N=60": "#6a4ce2",
          "EWMA λ=0.94": "#f47230", "EWMA λ=0.97": "#ec4647"}
MARKERS = {"Boxcar N=10": "o", "Boxcar N=20": "s", "Boxcar N=60": "D",
           "EWMA λ=0.94": "^", "EWMA λ=0.97": "v"}   # shape too, so colour is never the only cue

plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})

scores = pd.read_csv(FIG_DIR / "vol_scores.csv")
tuning = json.loads((FIG_DIR / "vol_tuning.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
y = np.arange(len(TICKERS))[::-1]
offsets = np.linspace(-0.28, 0.28, len(COLORS))
for ax, period, title in [(axes[0], "train", "TRAIN 2015-2021 (used for tuning)"),
                          (axes[1], "test", "TEST 2022-today (never used for tuning)")]:
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    for off, (name, color) in zip(offsets, COLORS.items()):
        s = scores[(scores.period == period) & (scores.estimator == name)].set_index("ticker").loc[TICKERS]
        picked = name in (tuning["best_boxcar"], tuning["best_ewma"])
        ax.scatter(s["QLIKE_gain"], y + off, color=color, marker=MARKERS[name],
                   s=55 if picked else 30, edgecolor="white", linewidth=0.8, zorder=3,
                   label=name + ("  (tuned pick)" if picked else ""))
    ax.set_title(title, loc="left")
    ax.set_xlabel("QLIKE gain vs. constant baseline  (> 0 = better than baseline)")
axes[0].set_yticks(y, TICKERS)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.935), ncol=5,
           fontsize=8, frameon=False)
fig.suptitle("Volatility filters vs. a constant baseline: in-sample edge shrinks out of sample, "
             "smooth filters keep a small one", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(FIG_DIR / "vol_scores_summary.png", dpi=150)
