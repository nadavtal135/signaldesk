"""Shared scoring helpers for Phase 5 (validation and final test use the same code)."""
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.evaluation import diebold_mariano

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
MODELS = ["C1 EWMA 0.94", "C2 two-speed", "C3 asymmetric", "C4 two-speed+asym",
          "C5 pull-to-level", "GJR-GARCH(1,1)-t", "Constant (2015-21 avg)"]


def load_losses(start: str, end: str) -> pd.DataFrame:
    """Per-day QLIKE and MSE for every model, on TARGET days in [start, end],
    restricted to days where every model has a forecast (same days for all)."""
    F = pd.read_csv(FIG_DIR / "phase5_forecasts.csv", parse_dates=["date"])
    out = []
    for t, g in F.groupby("ticker", sort=False):
        g = g.set_index("date")
        y = g["r"] ** 2                       # realised r^2 on the target day
        h = g[MODELS].shift(1)                # forecast made the evening before
        ok = h.notna().all(axis=1) & y.notna()
        ok &= (h.index >= start) & (h.index <= end)
        hh, yy = h[ok], y[ok]
        for m in MODELS:
            out.append(pd.DataFrame({"ticker": t, "model": m, "date": hh.index,
                                     "qlike": (yy / hh[m] + np.log(hh[m])).values,
                                     "mse": ((yy - hh[m]) ** 2).values}))
    return pd.concat(out, ignore_index=True)


def pooled_daily(L: pd.DataFrame, model: str, col: str = "qlike") -> pd.Series:
    """Average loss over the six tickers on each day."""
    return L[L.model == model].groupby("date")[col].mean()


def pooled_dm(L, a, b, col="qlike"):
    """DM t of model a vs model b on the pooled daily loss (< -2: a significantly better)."""
    return diebold_mariano(pooled_daily(L, a, col).values, pooled_daily(L, b, col).values)[0]
