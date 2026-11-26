"""Phase 3, step 2: show what risk targeting does, on SPY, with an
ILLUSTRATIVE target of 15%/yr and cap 1.0 (not tuned -- tuning is step 4).

Run:  PYTHONPATH=src .venv/bin/python scripts/09_risk_target_demo.py
Output: figures/risk_target_demo_SPY.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaldesk.data import load_prices
from signaldesk.returns import log_returns
from signaldesk.strategy import exposure_from_vol, forecast_vol_annual, strategy_returns

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TARGET, CAP = 0.15, 1.0
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE = "#257de0", "#f47230"
plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})

price = load_prices("SPY")["AdjClose"]
R = price.pct_change()                      # simple returns: these scale with exposure
vol = forecast_vol_annual(log_returns(price))
e = exposure_from_vol(vol, TARGET, CAP)
s = strategy_returns(R, e)

# Realised volatility over a trailing 60-day window, annualised -- only used
# to SHOW the output level, not to make any decision.
def realised(x):
    return x.rolling(60).std() * np.sqrt(252)

fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
ax = axes[0]
ax.plot(vol.index, 100 * vol, color=BLUE, linewidth=1)
ax.axhline(100 * TARGET, color=INK2, linestyle="--", linewidth=1)
ax.text(vol.index[5], 100 * TARGET + 1, f"target {100*TARGET:.0f}%", color=INK2, fontsize=8)
ax.set_ylabel("Forecast vol, %/yr")
ax.set_title("1. Measured input level: EWMA volatility forecast (λ=0.94)", loc="left")

ax = axes[1]
ax.plot(e.index, e, color=ORANGE, linewidth=1)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Exposure")
ax.set_title(f"2. Gain = min({CAP:.1f}, target / forecast): fraction of the account in SPY", loc="left")

ax = axes[2]
ax.plot(R.index, 100 * realised(R), color=BLUE, linewidth=1, label="SPY (buy-and-hold)")
ax.plot(s.index, 100 * realised(s), color=ORANGE, linewidth=1, label="Risk-targeted")
ax.axhline(100 * TARGET, color=INK2, linestyle="--", linewidth=1)
ax.set_ylabel("Realised vol, %/yr\n(trailing 60 days)")
ax.set_title("3. Output level: realised volatility of each account", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle("Risk targeting as an AGC loop on SPY (illustrative target 15%, cap 1.0, not tuned)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG_DIR / "risk_target_demo_SPY.png", dpi=150)

print(f"average exposure {e.mean():.2f}, time at cap {100*(e >= CAP - 1e-12).mean():.0f}%")
print(f"realised vol, full period: SPY {100*R.std()*np.sqrt(252):.1f}%  strategy {100*s.std()*np.sqrt(252):.1f}%")
rv_b, rv_s = realised(R).dropna(), realised(s).dropna()
print(f"spread of 60-day realised vol (10th-90th pct): SPY {100*rv_b.quantile(.1):.0f}-{100*rv_b.quantile(.9):.0f}%  "
      f"strategy {100*rv_s.quantile(.1):.0f}-{100*rv_s.quantile(.9):.0f}%")
