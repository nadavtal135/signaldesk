"""Turning prices into returns.

DSP view: the price P[n] is a random walk -- an integrator driven by noise.
It is non-stationary (its mean and variance drift over time), so its spectrum
is dominated by a huge 1/f^2 low-frequency pile-up and tells us little.

Taking the log and then a first difference,

    r[n] = ln(P[n]) - ln(P[n-1]) = ln(P[n] / P[n-1])

undoes the integrator. A first difference is a simple high-pass FIR filter,
h = [1, -1], with |H(f)| = 2|sin(pi f)|. What's left -- the "innovation" or
driving noise -- is what we'll analyse.

Finance view: r[n] is the daily *log return*, i.e. roughly the percent
gain/loss on day n (0.01 ~ +1%). Log returns add across days, which is why
we prefer them over simple percent changes.

Causality: r[n] uses only P[n] and P[n-1], never anything after day n.
"""
import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns. The first sample is dropped (it has no previous day)."""
    return np.log(prices).diff().dropna()
