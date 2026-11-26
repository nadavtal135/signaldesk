"""Forward test: frozen models, forecasts for days that did not exist at freeze time.

Why this is the strongest evidence: every backtest, however careful, is scored
on data that existed when the model was designed -- and in this project the
2022-today period had been looked at repeatedly. A forward test scores
forecasts for days AFTER the model was frozen and committed, so no choice could
have been influenced by them. It is the only test that is immune to
look-ahead, data snooping and multiple testing by construction.

The frozen models run with FIXED parameters (no more re-fits), as an
append-only log: forecasts are written once, never edited, and scored later.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.ewma_family import LAMBDA_BASE, asym_input, ewma_filter
from signaldesk.vol_models import garch_recursion

FWD_DIR = Path(__file__).resolve().parents[2] / "forward"
FROZEN = FWD_DIR / "frozen_model.json"
FORECASTS = FWD_DIR / "forecasts.csv"
SCORES = FWD_DIR / "score_log.csv"
# Phase 6 assets: a separate frozen set with its own append-only logs.
FROZEN6 = FWD_DIR / "frozen_model_phase6.json"
FORECASTS6 = FWD_DIR / "forecasts_phase6.csv"
SCORES6 = FWD_DIR / "score_log_phase6.csv"
MODELS = ["C3 asymmetric", "C1 EWMA 0.94", "GJR-GARCH(1,1)-t"]


def model_paths(r: pd.Series, p: dict) -> pd.DataFrame:
    """Forecast paths for one ticker with FROZEN parameters p (no fitting).
    h[t] uses r[0..t] only and forecasts day t+1's variance."""
    x = r.to_numpy(dtype=float)
    c3, gjr = p["C3 asymmetric"], p["GJR-GARCH(1,1)-t"]
    return pd.DataFrame({
        "C3 asymmetric": c3["c"] * ewma_filter(asym_input(x, c3["a"]), c3["lam"]),
        "C1 EWMA 0.94": ewma_filter(x ** 2, LAMBDA_BASE),
        "GJR-GARCH(1,1)-t": garch_recursion(x, gjr["omega"], gjr["alpha"], gjr["beta"], gjr["gamma"]),
    }, index=r.index)


def qlike(y, h):
    return y / h + np.log(h)
