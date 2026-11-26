"""Phase 5: EWMA-family volatility filters (the pre-registered candidates C1-C5).

See notes/phase5_plan.md for the frozen specification. All filters are causal:
h[t] uses r[0..t] only and is the variance forecast for day t+1.

DSP view of each candidate
- C1 EWMA:        one-pole low-pass filter on r^2 (pole at lambda).
- C2 two-speed:   two one-pole filters in parallel (a fast and a slow pole),
                  mixed with weight w -- a crude two-time-constant envelope
                  detector: reacts fast to shocks, yet remembers the slow level.
- C3 asymmetric:  one pole, but a polarity-dependent input gain: a down-day's
                  r^2 is amplified by (1 + a). A level-correction factor c
                  (computed from past data, not fitted) removes the DC bias
                  this would otherwise add.
- C4:             C2 and C3 combined.
- C5 pull-to-level: the EWMA output mixed with a very slow reference (the
                  expanding mean of all past r^2) -- like adding a set-point,
                  the feature that GARCH has and the EWMA lacks.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

INIT_WINDOW = 20
FIRST_FIT = 756
REFIT_EVERY = 63
LAMBDA_BASE = 0.94
LS_MAX = 0.999


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def ewma_filter(x: np.ndarray, lam: float, init_window: int = INIT_WINDOW) -> np.ndarray:
    """y[t] = lam*y[t-1] + (1-lam)*x[t], seeded with mean(x[:init_window]) at
    t = init_window-1 (NaN before). Implemented with lfilter (fast), and
    identical to signaldesk.volatility.ewma_variance when x = r^2."""
    y = np.full(len(x), np.nan)
    s = init_window - 1
    y[s] = np.mean(x[:init_window])
    if len(x) > init_window:
        # Direct-form filter (1-lam) / (1 - lam z^-1) with initial state lam*y[s].
        y[s + 1:], _ = lfilter([1 - lam], [1, -lam], x[init_window:], zi=[lam * y[s]])
    return y


def asym_input(r: np.ndarray, a: float) -> np.ndarray:
    """x = r^2 * (1 + a * [r < 0]): down-days count (1 + a) times."""
    return r ** 2 * (1 + a * (r < 0))


def level_correction(r: np.ndarray, a: float) -> float:
    """c = mean(r^2) / mean(x) over the given (past) window, so c*x has the same
    average level as r^2. Not a free parameter."""
    return float(np.mean(r ** 2) / np.mean(asym_input(r, a)))


def expanding_mean(x: np.ndarray) -> np.ndarray:
    """m[t] = mean(x[0..t]) -- uses data up to t only."""
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def _two_speed(lf, gap):
    return lf, min(lf + gap, LS_MAX)


# ---------------------------------------------------------------------------
# The candidates: forecast path for fixed parameters
# ---------------------------------------------------------------------------

def c2_path(r, lf, gap, w):
    lf, ls = _two_speed(lf, gap)
    r2 = r ** 2
    return w * ewma_filter(r2, lf) + (1 - w) * ewma_filter(r2, ls)


def c3_path(r, lam, a, c):
    return c * ewma_filter(asym_input(r, a), lam)


def c4_path(r, lf, gap, w, a, c):
    lf, ls = _two_speed(lf, gap)
    x = asym_input(r, a)
    return c * (w * ewma_filter(x, lf) + (1 - w) * ewma_filter(x, ls))


def c5_path(r, lam, k):
    r2 = r ** 2
    return (1 - k) * ewma_filter(r2, lam) + k * expanding_mean(r2)


# name -> (parameter names, fixed start, bounds, uses level correction?, path fn)
CANDIDATES = {
    "C2 two-speed": (["lf", "gap", "w"], [0.90, 0.08, 0.5],
                     [(0.50, 0.97), (0.001, 0.499), (0.0, 1.0)], False, c2_path),
    "C3 asymmetric": (["lam", "a"], [0.94, 0.5],
                      [(0.80, 0.999), (0.0, 3.0)], True, c3_path),
    "C4 two-speed+asym": (["lf", "gap", "w", "a"], [0.90, 0.08, 0.5, 0.5],
                          [(0.50, 0.97), (0.001, 0.499), (0.0, 1.0), (0.0, 3.0)], True, c4_path),
    "C5 pull-to-level": (["lam", "k"], [0.94, 0.1],
                         [(0.80, 0.999), (0.0, 1.0)], False, c5_path),
}
N_PARAMS = {"C1 EWMA 0.94": 0, "C2 two-speed": 3, "C3 asymmetric": 2,
            "C4 two-speed+asym": 4, "C5 pull-to-level": 2}


def path(name: str, r: np.ndarray, params, c: float = 1.0) -> np.ndarray:
    _, _, _, uses_c, fn = CANDIDATES[name]
    return fn(r, *params, c) if uses_c else fn(r, *params)


# ---------------------------------------------------------------------------
# Fitting and walk-forward
# ---------------------------------------------------------------------------

def mean_qlike(r: np.ndarray, h: np.ndarray) -> float:
    """Mean QLIKE of forecasts h[s] against r[s+1]^2, over all valid s."""
    hh, y = h[:-1], r[1:] ** 2
    ok = ~np.isnan(hh)
    return float(np.mean(y[ok] / hh[ok] + np.log(hh[ok])))


def fit(name: str, r_past: np.ndarray) -> dict:
    """Fit one candidate by minimising QLIKE on past data only (pre-registered:
    L-BFGS-B from a fixed start, fixed bounds). The level correction c (C3/C4)
    is recomputed inside the objective from the same past window."""
    pnames, start, bounds, uses_c, _ = CANDIDATES[name]

    def c_for(p):
        return level_correction(r_past, p[pnames.index("a")]) if uses_c else 1.0

    def objective(p):
        h = path(name, r_past, p, c_for(p))
        if np.nanmin(h) <= 0:
            return 1e6
        return mean_qlike(r_past, h)

    res = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
    out = dict(zip(pnames, res.x))
    out["c"] = c_for(res.x)
    out["train_qlike"] = res.fun
    return out


def refit_days(n: int, first_fit: int = FIRST_FIT, every: int = REFIT_EVERY) -> list[int]:
    return list(range(first_fit - 1, n, every))


def walk_forward(name: str, r: pd.Series, first_fit: int = FIRST_FIT,
                 every: int = REFIT_EVERY):
    """Walk-forward forecasts for one candidate.

    Fit k uses r[0..T_k] only; its parameters (and its c) produce the forecasts
    for days T_k .. T_{k+1}-1 by running the fixed filter over data up to each day.
    Returns (h Series, DataFrame of fits).
    """
    x = r.to_numpy(dtype=float)
    pnames = CANDIDATES[name][0]
    h = np.full(len(x), np.nan)
    fits = []
    days = refit_days(len(x), first_fit, every)
    for k, T in enumerate(days):
        f = fit(name, x[: T + 1])
        end = days[k + 1] if k + 1 < len(days) else len(x)
        p = [f[n] for n in pnames]
        h[T:end] = path(name, x[:end], p, f["c"])[T:end]
        fits.append({"refit_day": r.index[T], **f})
    return pd.Series(h, index=r.index), pd.DataFrame(fits)


def c1_forecast(r: pd.Series) -> pd.Series:
    """C1: the frozen Phase 2 EWMA (lambda = 0.94); nothing is fitted."""
    return pd.Series(ewma_filter(r.to_numpy(dtype=float) ** 2, LAMBDA_BASE), index=r.index)
