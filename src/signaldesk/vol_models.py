"""Phase 4 volatility forecasters: GARCH(1,1), GJR-GARCH(1,1) and HAR-style.

All are causal and follow the Phase 2 convention: the returned Series h is
indexed by day t, and h[t] (made at the close of day t from returns up to t)
is the forecast of the variance on day t+1.

Walk-forward schedule (shared by all three):
- First fit after FIRST_FIT trading days; re-fit every REFIT_EVERY days.
- Fit number k uses returns r[0..T_k] only (T_k = the re-fit day, inclusive).
- Its parameters produce forecasts for days T_k .. T_{k+1}-1. Between re-fits
  the forecast updates recursively with those frozen parameters.
See notes/phase4_models.md for plain-English descriptions.
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import nnls

FIRST_FIT = 756      # 3 years of trading days
REFIT_EVERY = 63     # about one quarter
INIT_WINDOW = 20     # days used to seed the GARCH recursion (same as the EWMA)
HAR_FLOOR = 1e-4     # %^2: safety floor for HAR forecasts (reported if ever used)


def refit_days(n: int, first_fit: int = FIRST_FIT, every: int = REFIT_EVERY) -> list[int]:
    """Indices T_k of the re-fit days: first_fit-1, then every `every` days."""
    return list(range(first_fit - 1, n, every))


# ---------------------------------------------------------------------------
# GARCH / GJR-GARCH
# ---------------------------------------------------------------------------

def garch_recursion(r: np.ndarray, omega: float, alpha: float, beta: float,
                    gamma: float = 0.0, init_window: int = INIT_WINDOW) -> np.ndarray:
    """One-step-ahead variance path with FIXED parameters.

        h[t] = omega + (alpha + gamma * 1[r[t] < 0]) * r[t]^2 + beta * h[t-1]

    h[t] uses r[0..t] only. The filter state before the first update,
    h[init_window-1], is seeded with the mean r^2 of the first init_window
    days (identical to the EWMA seed); earlier entries are NaN.
    gamma = 0 gives plain GARCH(1,1).
    """
    h = np.full(len(r), np.nan)
    start = init_window - 1
    h[start] = np.mean(r[:init_window] ** 2)
    for t in range(start + 1, len(r)):
        gain = alpha + (gamma if r[t] < 0 else 0.0)
        h[t] = omega + gain * r[t] ** 2 + beta * h[t - 1]
    return h


def _fit_garch(r_past: np.ndarray, asymmetric: bool) -> dict:
    """Maximum-likelihood fit (Student-t errors, zero mean) on past data only."""
    am = arch_model(r_past, mean="Zero", vol="GARCH", p=1, o=1 if asymmetric else 0, q=1,
                    dist="t", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # optimiser chatter; convergence is checked below
        res = am.fit(disp="off", show_warning=False)
    p = res.params
    return {"omega": p["omega"], "alpha": p["alpha[1]"], "beta": p["beta[1]"],
            "gamma": p["gamma[1]"] if asymmetric else 0.0, "nu": p["nu"],
            "converged": res.convergence_flag == 0}


def garch_forecast(r: pd.Series, asymmetric: bool = False,
                   first_fit: int = FIRST_FIT, every: int = REFIT_EVERY):
    """Walk-forward GARCH(1,1) (asymmetric=False) or GJR-GARCH(1,1) forecasts.

    Returns (h, params): h is the forecast Series; params lists each fit.
    If a fit fails to converge, the previous parameters are kept (and the
    failure is recorded) -- never data from the future.
    """
    x = r.to_numpy(dtype=float)
    h = np.full(len(x), np.nan)
    fits, prev = [], None
    days = refit_days(len(x), first_fit, every)
    for k, T in enumerate(days):
        fit = _fit_garch(x[: T + 1], asymmetric)
        if not fit["converged"] and prev is not None:
            fit = {**prev, "converged": False}
        prev = fit
        end = days[k + 1] if k + 1 < len(days) else len(x)
        # Run the fixed-parameter filter over data up to the end of this block
        # (all at or before each forecast day) and keep the block's forecasts.
        path = garch_recursion(x[:end], fit["omega"], fit["alpha"], fit["beta"], fit["gamma"])
        h[T:end] = path[T:end]
        fits.append({"refit_day": r.index[T], **fit})
    return pd.Series(h, index=r.index), pd.DataFrame(fits)


# ---------------------------------------------------------------------------
# HAR-style regression
# ---------------------------------------------------------------------------

def har_features(r: pd.Series) -> pd.DataFrame:
    """Daily, weekly (5-day) and monthly (22-day) averages of r^2, each ending at day t."""
    r2 = r ** 2
    return pd.DataFrame({"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()})


def har_forecast(r: pd.Series, first_fit: int = FIRST_FIT, every: int = REFIT_EVERY):
    """Walk-forward HAR-style forecasts, fitted by non-negative least squares.

    Training pairs for re-fit day T: features at day s, target r^2[s+1], for
    every s with s+1 <= T (so the target itself is already in the past).
    NNLS keeps all four weights >= 0, so the forecast is a non-negative mix of
    non-negative averages and cannot go negative.

    Returns (h, params): params lists each fit and how often the floor was used.
    """
    X = har_features(r)
    X.insert(0, "const", 1.0)
    target = (r ** 2).shift(-1)          # target for row s is r^2[s+1]
    Xv, yv = X.to_numpy(), target.to_numpy()
    h = np.full(len(r), np.nan)
    fits = []
    days = refit_days(len(r), first_fit, every)
    for k, T in enumerate(days):
        rows = np.arange(T)                              # s = 0..T-1, so s+1 <= T
        ok = rows[~np.isnan(Xv[rows]).any(axis=1)]       # drop the 21-day warm-up
        b, _ = nnls(Xv[ok], yv[ok])
        end = days[k + 1] if k + 1 < len(days) else len(r)
        raw = Xv[T:end] @ b
        h[T:end] = np.maximum(raw, HAR_FLOOR)
        fits.append({"refit_day": r.index[T], "b0": b[0], "b_day": b[1], "b_week": b[2],
                     "b_month": b[3], "floor_used": int((raw < HAR_FLOOR).sum())})
    return pd.Series(h, index=r.index), pd.DataFrame(fits)
