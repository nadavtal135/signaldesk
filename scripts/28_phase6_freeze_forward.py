"""Phase 6, step 6: freeze the forward test for the Phase 6 assets.

Takes each asset's LAST walk-forward fit (data up to 2026-11-25) from
figures/phase6_fits.csv -- no re-fitting -- for C3 and GJR-GARCH(1,1)-t, and
logs forecasts made on 2026-11-25. Refuses to run if already frozen.

Run once:  PYTHONPATH=src .venv/bin/python scripts/28_phase6_freeze_forward.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.forward import FORECASTS6, FROZEN6, MODELS, model_paths
from signaldesk.universe import UNIVERSE, load_asset

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
assert not FROZEN6.exists(), f"{FROZEN6} exists: the Phase 6 forward test is already frozen."

fits = pd.read_csv(FIG_DIR / "phase6_fits.csv", parse_dates=["refit_day"])
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
frozen = {"freeze_date": None, "code_commit_at_freeze": commit,
          "note": "Phase 6 assets. Parameters fixed from here on: no re-fits.", "params": {}}
rows = []
for t in UNIVERSE:
    f3 = fits[(fits.ticker == t) & (fits.model == "C3 asymmetric")].iloc[-1]
    fg = fits[(fits.ticker == t) & (fits.model == "GJR-GARCH(1,1)-t")].iloc[-1]
    p = {"C3 asymmetric": {"lam": float(f3.lam), "a": float(f3.a), "c": float(f3.c),
                           "fitted_on_data_up_to": str(f3.refit_day.date())},
         "GJR-GARCH(1,1)-t": {k: float(fg[k]) for k in ["omega", "alpha", "beta", "gamma", "nu"]}
                             | {"fitted_on_data_up_to": str(fg.refit_day.date())}}
    frozen["params"][t] = p
    r = 100 * np.log(load_asset(t)).diff().dropna()
    h = model_paths(r, p)
    d = r.index[-1]
    rows.append({"ticker": t, "made_on": d.date(), **{m: h.loc[d, m] for m in MODELS}})
    frozen["freeze_date"] = str(d.date())

FROZEN6.write_text(json.dumps(frozen, indent=2))
pd.DataFrame(rows).to_csv(FORECASTS6, index=False, float_format="%.8g")
print(f"frozen {len(rows)} assets on {frozen['freeze_date']}")
