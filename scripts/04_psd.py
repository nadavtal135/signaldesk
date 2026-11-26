"""Step 4: is the return series white noise? PSD + autocorrelation vs. a white reference.

Run:  PYTHONPATH=src .venv/bin/python scripts/04_psd.py
Outputs: figures/psd_returns.png, figures/psd_squared_returns.png,
         figures/acf.png, figures/whiteness_tests.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.returns import log_returns
from signaldesk.spectral import acf, ljung_box, robust_acf_band, shuffled_null_band, welch_psd

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
BLUE, ORANGE = "#257de0", "#f47230"
INK, INK2, GRID, BAND = "#191919", "#666666", "#e6e6e6", "#d9d9d9"
MAX_LAG = 20

plt.rcParams.update({
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlecolor": INK,
})

series = {t: log_returns(load_prices(t)["AdjClose"]) for t in TICKERS}
returns = {t: s.values for t, s in series.items()}
# The COVID crash window, used for a robustness check: does any structure
# survive when this one extreme episode is removed?
CRASH = ("2020-02-01", "2020-06-30")


def psd_grid(transform, title, fname, ylim, show_ex_crash=False):
    """Small multiples: one panel per ticker, PSD in dB relative to the
    white-noise level (median of the shuffled reference). 0 dB = white."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    results = {}
    for ax, t in zip(axes.flat, TICKERS):
        x = transform(returns[t])
        f, p = welch_psd(x)
        med, lo, hi = shuffled_null_band(x)
        f, p, med, lo, hi = f[1:], p[1:], med[1:], lo[1:], hi[1:]  # drop DC: mean was removed
        db = lambda v: 10 * np.log10(v / med)
        out_ex = np.nan
        ax.fill_between(f, db(lo), db(hi), color=BAND, linewidth=0,
                        label="95% band for white noise (500 shuffles of the same data)")
        ax.axhline(0, color=INK2, linewidth=0.8, linestyle="--")
        ax.plot(f, db(p), color=BLUE, linewidth=1.2, label="Welch PSD, all data")
        if show_ex_crash:
            # Robustness check: drop Feb-Jun 2020 and splice the rest together.
            # One splice point barely affects a 256-sample Welch average.
            s_ = series[t]
            x_ex = transform(s_[(s_.index < CRASH[0]) | (s_.index > CRASH[1])].values)
            _, p_ex = welch_psd(x_ex)
            med_ex, lo_ex, hi_ex = (b[1:] for b in shuffled_null_band(x_ex, n_shuffles=200))
            out_ex = np.mean((p_ex[1:] < lo_ex) | (p_ex[1:] > hi_ex))
            ax.plot(f, 10 * np.log10(p_ex[1:] / med_ex), color=ORANGE, linewidth=1.0,
                    label="Welch PSD, excluding Feb-Jun 2020")
        ax.set_title(t, loc="left")
        ax.set_ylim(*ylim)
        outside = np.mean((p < lo) | (p > hi))
        ax.text(0.98, 0.95, f"{100*outside:.0f}% of bins outside band",
                transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)
        # Low-frequency excess: mean dB over the lowest 5 bins (periods ~50-256 days).
        results[t] = (outside, float(np.mean(db(p)[:5])), out_ex)
    for ax in axes[1]:
        ax.set_xlabel("Frequency (cycles per trading day)\n0.5 = Nyquist (2-day period)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Power vs. white level (dB)")
    # One shared legend above the panels, so it never covers data.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    return results


psd_r = psd_grid(lambda r: r,
                 "PSD of daily returns, relative to white noise of the same power (0 dB = white)",
                 "psd_returns.png", (-6, 6), show_ex_crash=True)
# Squared returns ~ instantaneous power (an envelope detector, like |x|^2 in a
# receiver). If the *loudness* of the noise is predictable, it shows up here.
psd_r2 = psd_grid(lambda r: (r - r.mean())**2,
                  "PSD of SQUARED returns (the noise power envelope), relative to white noise",
                  "psd_squared_returns.png", (-10, 16))

# Autocorrelation: the time-domain twin of the PSD.
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
lags = np.arange(1, MAX_LAG + 1)
rows = []
for ax, t in zip(axes.flat, TICKERS):
    r = returns[t]
    n = len(r)
    a_r = acf(r, MAX_LAG)[1:]
    a_r2 = acf((r - r.mean())**2, MAX_LAG)[1:]
    ci = 1.96 / np.sqrt(n)   # textbook 95% band (assumes constant noise power)
    rb = robust_acf_band(r, MAX_LAG)  # band that allows for volatility bursts
    ax.axhspan(-ci, ci, color=BAND, linewidth=0, label="95% band, white noise (textbook)")
    ax.plot(lags, rb, color=INK2, linewidth=0.9, linestyle=":",
            label="95% band, white noise with volatility bursts (robust)")
    ax.plot(lags, -rb, color=INK2, linewidth=0.9, linestyle=":")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.plot(lags - 0.15, a_r, "o", color=BLUE, markersize=4, label="returns")
    ax.vlines(lags - 0.15, 0, a_r, color=BLUE, linewidth=1)
    ax.plot(lags + 0.15, a_r2, "o", color=ORANGE, markersize=4, label="squared returns")
    ax.vlines(lags + 0.15, 0, a_r2, color=ORANGE, linewidth=1)
    ax.set_title(t, loc="left")
    s_ex = series[t][(series[t].index < CRASH[0]) | (series[t].index > CRASH[1])].values
    rows.append({
        "ticker": t,
        "acf_lag1_returns": a_r[0],
        "acf_lag1_robust_t": a_r[0] / (rb[0] / 1.96),   # |t| > 2 ~ significant
        "acf_lag1_excl_crash": acf(s_ex, 1)[1],
        "acf_lag1_excl_crash_robust_t": acf(s_ex, 1)[1] / (robust_acf_band(s_ex, 1)[0] / 1.96),
        "ljungbox_p_returns_10lags": ljung_box(r, 10)[1],
        "psd_bins_outside_band_returns_%": 100 * psd_r[t][0],
        "psd_bins_outside_band_returns_excl_crash_%": 100 * psd_r[t][2],
        "acf_lag1_squared": a_r2[0],
        "ljungbox_p_squared_10lags": ljung_box((r - r.mean())**2, 10)[1],
        "psd_bins_outside_band_squared_%": 100 * psd_r2[t][0],
        "low_freq_excess_squared_dB": psd_r2[t][1],
    })
for ax in axes[1]:
    ax.set_xlabel("Lag (trading days)")
    ax.set_xticks([1, 5, 10, 15, 20])
for ax in axes[:, 0]:
    ax.set_ylabel("Autocorrelation")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=4, fontsize=8, frameon=False)
fig.suptitle("Autocorrelation of returns and squared returns, lags 1-20",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG_DIR / "acf.png", dpi=150)
plt.close(fig)

table = pd.DataFrame(rows).set_index("ticker")
table.to_csv(FIG_DIR / "whiteness_tests.csv", float_format="%.4g")
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(table.round(4))
