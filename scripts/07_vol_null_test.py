"""Phase 2, step 4: null test -- does the EWMA's edge come from volatility clustering?

DSP view: this is like checking a detector's false-alarm behaviour by feeding
it pure noise. We randomly permute each return series. A permutation keeps
exactly the same set of values (same heavy tails, same total power) but
destroys all time order -- so the noise power no longer varies slowly; there
is no envelope left to track. If the EWMA's score on the real data is no
better than on these shuffled copies, its "edge" was luck.

Details (chosen to keep the comparison fair):
- Shuffle WITHIN the train period and WITHIN the test period separately, so
  each period keeps its own set of values, as in the real data. Only the
  ordering inside each period is destroyed.
- Re-run the tuned estimator (EWMA lambda=0.94, frozen from step 3) and the
  constant baseline on every shuffled series, score them on the same days, and
  record the statistic: test-period QLIKE gain of EWMA over the baseline.
- 200 shuffles per ticker. p-value = (1 + #shuffles scoring >= real) / (1 + 200).

Run:  PYTHONPATH=src .venv/bin/python scripts/07_vol_null_test.py
Outputs: figures/vol_null_test.png, figures/vol_null_test.csv
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.evaluation import align_forecast, qlike_loss
from signaldesk.returns import log_returns
from signaldesk.volatility import constant_variance, ewma_variance

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
WARMUP = 60          # same scored days as step 3 (after the longest warm-up)
N_SHUFFLES = 200
BEST = json.loads((FIG_DIR / "vol_tuning.json").read_text())["best_overall"]
LAM = float(BEST.split("=")[1])   # frozen from step 3: tuned on train only

INK, INK2, GRID, BAR, BLUE = "#191919", "#666666", "#e6e6e6", "#b3b3b3", "#257de0"
plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})


def qlike_gains(r: pd.Series) -> tuple[float, float]:
    """(train gain, test gain) of EWMA over the constant baseline, QLIKE units."""
    m = align_forecast(r, ewma_variance(r, LAM)).iloc[WARMUP:]
    b = align_forecast(r, constant_variance(r, TRAIN_END)).loc[m.index]
    gain = qlike_loss(b.y, b.h) - qlike_loss(m.y, m.h)    # per-day, positive = EWMA better
    return gain.loc[:TRAIN_END].mean(), gain.loc[TEST_START:].mean()


def shuffle_within_periods(r: pd.Series, rng) -> pd.Series:
    train, test = r.loc[:TRAIN_END].values, r.loc[TEST_START:].values
    return pd.Series(np.concatenate([rng.permutation(train), rng.permutation(test)]), index=r.index)


rng = np.random.default_rng(42)
rows, nulls = [], {}
for t in TICKERS:
    r = 100 * log_returns(load_prices(t)["AdjClose"])
    real_train, real_test = qlike_gains(r)
    null_test = np.array([qlike_gains(shuffle_within_periods(r, rng))[1] for _ in range(N_SHUFFLES)])
    nulls[t] = null_test
    p = (1 + np.sum(null_test >= real_test)) / (1 + N_SHUFFLES)
    rows.append({"ticker": t, "real_test_gain": real_test,
                 "null_mean": null_test.mean(), "null_p95": np.percentile(null_test, 95),
                 "null_max": null_test.max(), "p_value": p,
                 # How many null standard deviations the real score sits above the null mean.
                 "z_vs_null": (real_test - null_test.mean()) / null_test.std()})
    print(f"{t}: real={real_test:+.3f}  null mean={null_test.mean():+.3f}  "
          f"null 95th pct={np.percentile(null_test, 95):+.3f}  p={p:.3f}")

table = pd.DataFrame(rows).set_index("ticker")
table.to_csv(FIG_DIR / "vol_null_test.csv", float_format="%.4f")

# Small multiples: one histogram per ticker, real score as a vertical line.
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
bins = np.linspace(-0.24, 0.24, 49)   # same bin edges in every panel
for ax, t in zip(axes.flat, TICKERS):
    ax.hist(nulls[t], bins=bins, color=BAR, edgecolor="white", linewidth=0.5,
            label=f"{N_SHUFFLES} shuffled series (no clustering)")
    ax.axvline(0, color=INK2, linewidth=0.8, linestyle="--")
    real = table.loc[t, "real_test_gain"]
    ax.axvline(real, color=BLUE, linewidth=2, label="real data")
    ax.set_title(t, loc="left")
    p = table.loc[t, "p_value"]
    # With 200 shuffles the smallest possible p is 1/201 ~ 0.005 ("beats every shuffle").
    label = f"p = {p:.3f}" if p > 1 / (N_SHUFFLES + 1) else f"p ≤ {1/(N_SHUFFLES+1):.3f}\n(beats all {N_SHUFFLES})"
    ax.text(0.02, 0.95, label,
            transform=ax.transAxes, ha="left", va="top", color=INK, fontsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("Test QLIKE gain over baseline\n(> 0 = EWMA better)")
for ax in axes[:, 0]:
    ax.set_ylabel("Number of shuffles")
axes.flat[0].set_ylim(0, 100)   # headroom above the bars for the p-value label
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2,
           fontsize=8, frameon=False)
fig.suptitle(f"Null test: EWMA λ={LAM} edge over the baseline, 2022-today: real returns vs. the same returns shuffled",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG_DIR / "vol_null_test.png", dpi=150)
