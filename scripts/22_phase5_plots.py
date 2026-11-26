"""Phase 5 summary figure: final test (2024-today) per ticker and year by year.

Run (after 19_phase5_final_test.py):  PYTHONPATH=src .venv/bin/python scripts/22_phase5_plots.py
Output: figures/phase5_final_test.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
COL = {"C3 asymmetric": "#f47230", "C1 EWMA 0.94": "#6a4ce2", "Constant (2015-21 avg)": "#191919"}
MRK = {"C3 asymmetric": "^", "C1 EWMA 0.94": "o", "Constant (2015-21 avg)": "s"}
plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})
T = pd.read_csv(FIG_DIR / "phase5_final_test.csv")
Y = pd.read_csv(FIG_DIR / "phase5_final_by_year.csv").set_index("year")
V = json.loads((FIG_DIR / "phase5_final_verdict.json").read_text())
tick = [t for t in T.ticker.unique() if t != "POOLED"] + ["POOLED"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
y = np.arange(len(tick))[::-1]
for off, m in zip([-0.18, 0.0, 0.18], COL):
    s = T[T.model == m].set_index("ticker").loc[tick]
    ax1.scatter(s.QLIKE_gain_vs_GJR, y + off, color=COL[m], marker=MRK[m], s=45,
                edgecolor="white", linewidth=0.7, zorder=3, label=m)
ax1.axvline(0, color=INK2, linestyle="--", linewidth=1)
ax1.axhspan(-0.5, 0.5, color="#f2f2f2", zorder=0)
ax1.set_yticks(y, tick)
ax1.set_xlabel("QLIKE gain vs GJR-GARCH(1,1)-t  (> 0 = beats GJR)")
ax1.set_title("Final test 2024-2026: per ticker and pooled", loc="left")


gain = Y.rsub(Y["GJR-GARCH(1,1)-t"], axis=0)     # GJR loss - model loss
x = np.arange(len(gain.index))
for off, m in zip([-0.25, 0.0, 0.25], COL):
    ax2.bar(x + off, gain[m], width=0.24, color=COL[m], label=m)
ax2.axhline(0, color=INK2, linewidth=1)
ax2.set_xticks(x, [f"{yr}" + ("*" if yr == 2026 else "") for yr in gain.index])
ax2.set_ylabel("Pooled QLIKE gain vs GJR (> 0 = beats GJR)")
ax2.legend(loc="lower right", fontsize=8.5, frameon=False)  # one legend for both panels
ax2.set_title("Year by year (*2026 = Jan-Nov): GJR wins every year", loc="left")
fig.suptitle(f"Phase 5 pre-registered test: frozen {V['frozen_winner']} vs GJR-GARCH -> "
             f"{V['verdict'].upper()}  (pooled DM t = {V['pooled_DM_t_vs_GJR']:+.2f}, "
             f"{V['years_won_vs_GJR']}/3 years)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG_DIR / "phase5_final_test.png", dpi=150)
