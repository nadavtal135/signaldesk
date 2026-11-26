"""Volatility estimators: causal low-pass filters on squared returns.

DSP view
--------
Phase 1 showed that returns r[t] behave like white noise whose *power* drifts
slowly over time. The squared return r[t]^2 is a one-sample, very noisy
measurement of that instantaneous power (like |x|^2 at an envelope detector).
To estimate the underlying power we smooth r^2 with a low-pass filter.
Every filter here is CAUSAL: the output on day t uses r[0..t] only.

Finance view
------------
*Variance* = the noise power of daily returns. *Volatility* = its square root
(the standard deviation), usually quoted per year. A volatility forecast says
"how big are tomorrow's moves likely to be", not which way they will go.

Timing convention (the no-look-ahead rule)
------------------------------------------
Every function returns a Series h indexed by day t. h[t] is computed at the
close of day t using returns up to and including t, and it is the FORECAST
for day t+1. It is scored against the realised r[t+1]^2, never against r[t]^2.
"""
import numpy as np
import pandas as pd

EWMA_INIT_WINDOW = 20  # days used to seed the EWMA's starting state


def rolling_variance(r: pd.Series, window: int) -> pd.Series:
    """Boxcar (moving-average) filter on r^2: h[t] = mean(r[t-N+1..t]^2).

    FIR filter with N equal taps of 1/N. Every day in the window counts the
    same and a day drops out completely after N days, so a big shock causes
    a step up and then, exactly N days later, a step down ("ghost" effect).
    The first N-1 outputs are NaN because the window isn't full yet.
    """
    return (r**2).rolling(window).mean()


def ewma_variance(r: pd.Series, lam: float, init_window: int = EWMA_INIT_WINDOW) -> pd.Series:
    """Exponentially weighted moving average of r^2 (finance name: RiskMetrics EWMA).

        h[t] = lam * h[t-1] + (1 - lam) * r[t]^2

    A one-pole IIR low-pass filter with pole at z = lam and unity gain at DC.
    The impulse response is (1-lam) * lam^k: the weight on a day k days ago
    decays geometrically, so old data fades out smoothly instead of falling
    off a cliff like the boxcar.

    Memory in numbers:
      time constant    tau = -1/ln(lam) ~ 1/(1-lam)   (0.94 -> ~16 days, 0.97 -> ~33 days)
      half-life        ln(0.5)/ln(lam)                (0.94 -> ~11 days, 0.97 -> ~23 days)
      equivalent boxcar length N = (1+lam)/(1-lam)    (0.94 -> ~32,      0.97 -> ~66)
    The last line matches the boxcar with the same average delay AND the same
    noise-reduction factor; for this filter pair the two criteria agree.

    Initial state: the filter needs h[-1]. We seed it with the plain average
    of the first `init_window` squared returns (a causal choice: that value is
    only used from day init_window-1 on). Earlier outputs are NaN.
    """
    r2 = (r**2).to_numpy()
    h = np.full(len(r2), np.nan)
    start = init_window - 1
    h[start] = r2[:init_window].mean()
    for t in range(start + 1, len(r2)):
        h[t] = lam * h[t - 1] + (1 - lam) * r2[t]
    return pd.Series(h, index=r.index)


def constant_variance(r: pd.Series, train_end: str) -> pd.Series:
    """Naive baseline: one fixed number, the average r^2 over the training period.

    DSP view: a filter with infinite memory and no adaptation; it assumes the
    noise power never changes. Caveat: on training days this uses training
    data from *after* day t, so it is slightly in-sample there. On test days
    (after train_end) it is fully honest.
    """
    level = (r.loc[:train_end] ** 2).mean()
    return pd.Series(level, index=r.index)
