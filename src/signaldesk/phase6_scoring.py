"""Phase 6 scoring: per-day losses on identical target days for every model, per asset."""
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.evaluation import diebold_mariano

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
MODELS = ["EWMA 0.94", "C3 asymmetric", "GJR-GARCH(1,1)-t", "Constant (2015-21 avg)",
          "C3 with a=0 (diagnostic)"]


def losses_from_forecasts(F: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """Long table of QLIKE/MSE per (ticker, model, target date).

    The forecast made on day t (row t) is paired with r[t+1]^2 on the next row.
    Per asset, only target days where EVERY model has a forecast are kept, so
    all models are scored on exactly the same days.
    """
    out = []
    for t, g in F.groupby("ticker", sort=False):
        g = g.set_index("date").sort_index()
        y = g["r"] ** 2
        h = g[MODELS].shift(1)
        ok = h.notna().all(axis=1) & y.notna() & (h.index >= start)
        if end is not None:
            ok &= h.index <= end
        hh, yy = h[ok], y[ok]
        for m in MODELS:
            out.append(pd.DataFrame({"ticker": t, "model": m, "date": hh.index,
                                     "qlike": (yy / hh[m] + np.log(hh[m])).values,
                                     "mse": ((yy - hh[m]) ** 2).values}))
    return pd.concat(out, ignore_index=True)


def load_losses(start: str, end: str | None = None) -> pd.DataFrame:
    F = pd.read_csv(FIG_DIR / "phase6_forecasts.csv", parse_dates=["date"])
    return losses_from_forecasts(F, start, end)


def pooled_daily(L: pd.DataFrame, model: str, col: str = "qlike") -> pd.Series:
    return L[L.model == model].groupby("date")[col].mean()


def pooled_dm(L: pd.DataFrame, a: str, b: str, col: str = "qlike") -> float:
    """DM t of a vs b on the per-day average over assets (< -2: a significantly better)."""
    return diebold_mariano(pooled_daily(L, a, col).values, pooled_daily(L, b, col).values)[0]


def mean_gain(L: pd.DataFrame, model: str, ref: str, col: str = "qlike") -> pd.Series:
    """Per-asset mean loss of ref minus model (> 0 = model better)."""
    m = L.groupby(["ticker", "model"])[col].mean().unstack()
    return m[ref] - m[model]
