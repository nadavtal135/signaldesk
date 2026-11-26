"""Phase 5, step 6a: freeze the forward-test models and write the first forecasts.

Freezes, per ticker, the parameters from the LAST walk-forward re-fit on or
before the freeze date (2026-11-25, the last day of data):
- C3 asymmetric (the frozen Phase 5 winner): lambda, a, c
- GJR-GARCH(1,1)-t (the benchmark to beat): omega, alpha, beta, gamma, nu
- C1 EWMA 0.94 (no parameters)
Then writes forecasts made on the freeze date (for the next trading day).
Refuses to run if a frozen model already exists: freezing happens once.

Run once:  PYTHONPATH=src .venv/bin/python scripts/21_phase5_freeze_forward.py
"""
import json
import subprocess

import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.ewma_family import walk_forward
from signaldesk.forward import FORECASTS, FROZEN, MODELS, model_paths
from signaldesk.returns import log_returns
from signaldesk.vol_models import garch_forecast

assert not FROZEN.exists(), f"{FROZEN} exists: the forward test is already frozen."
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

frozen = {"freeze_date": None, "code_commit_at_freeze": commit,
          "note": "Parameters are fixed from here on: no re-fits in the forward test.",
          "params": {}}
rows = []
for t in TICKERS:
    r = 100 * log_returns(load_prices(t)["AdjClose"])
    _, f3 = walk_forward("C3 asymmetric", r)
    _, fg = garch_forecast(r, asymmetric=True)
    last3, lastg = f3.iloc[-1], fg.iloc[-1]
    p = {"C3 asymmetric": {"lam": float(last3.lam), "a": float(last3.a), "c": float(last3.c),
                           "fitted_on_data_up_to": str(pd.Timestamp(last3.refit_day).date())},
         "GJR-GARCH(1,1)-t": {k: float(lastg[k]) for k in ["omega", "alpha", "beta", "gamma", "nu"]}
                             | {"fitted_on_data_up_to": str(pd.Timestamp(lastg.refit_day).date())}}
    frozen["params"][t] = p
    h = model_paths(r, p)
    d = r.index[-1]
    rows.append({"ticker": t, "made_on": d.date(), **{m: h.loc[d, m] for m in MODELS}})
    frozen["freeze_date"] = str(d.date())

FROZEN.write_text(json.dumps(frozen, indent=2))
pd.DataFrame(rows).to_csv(FORECASTS, index=False, float_format="%.8g")
print(json.dumps(frozen, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
