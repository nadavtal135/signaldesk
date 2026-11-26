"""Phase 6, step 3: run the FROZEN models on every new asset (no code or parameter changes).

Models (notes/phase6_plan.md):
- EWMA lambda=0.94                 (signaldesk.ewma_family.c1_forecast)
- C3 asymmetric EWMA               (walk_forward, re-fitted per asset on its own past)
- GJR-GARCH(1,1)-t                 (garch_forecast(asymmetric=True), walk-forward)
- Constant baseline                (asset's own mean r^2 over 2015-2021)
- Diagnostic for H2 only: C3 with a fixed at 0 (only lambda fitted), as in Phase 5.

Run:  PYTHONPATH=src .venv/bin/python scripts/25_phase6_run_models.py   (~1-2 min)
Outputs: figures/phase6_forecasts.csv (gitignored, regenerable), figures/phase6_fits.csv
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.ewma_family import CANDIDATES, c1_forecast, c3_path, walk_forward
from signaldesk.universe import UNIVERSE, load_asset
from signaldesk.vol_models import garch_forecast
from signaldesk.volatility import constant_variance

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
LAM_ONLY = "C3 with a=0 (diagnostic)"
CANDIDATES[LAM_ONLY] = (["lam", "a"], [0.94, 0.0], [(0.80, 0.999), (0.0, 0.0)], True, c3_path)
MODELS = ["EWMA 0.94", "C3 asymmetric", "GJR-GARCH(1,1)-t", "Constant (2015-21 avg)", LAM_ONLY]


def run_one(ticker: str):
    # Re-register in the worker process (spawned workers re-import this module,
    # which runs the CANDIDATES line above, so this is just for clarity).
    r = 100 * np.log(load_asset(ticker)).diff().dropna()
    h_c3, f_c3 = walk_forward("C3 asymmetric", r)
    h_l, f_l = walk_forward(LAM_ONLY, r)
    h_g, f_g = garch_forecast(r, asymmetric=True)
    df = pd.DataFrame({"EWMA 0.94": c1_forecast(r), "C3 asymmetric": h_c3,
                       "GJR-GARCH(1,1)-t": h_g,
                       "Constant (2015-21 avg)": constant_variance(r, "2021-12-31"),
                       LAM_ONLY: h_l, "r": r})
    df["ticker"] = ticker
    fits = pd.concat([f_c3.assign(model="C3 asymmetric"), f_l.assign(model=LAM_ONLY),
                      f_g.assign(model="GJR-GARCH(1,1)-t")]).assign(ticker=ticker)
    return df.rename_axis("date").reset_index(), fits


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as ex:
        out = list(ex.map(run_one, UNIVERSE))
    F = pd.concat([o[0] for o in out], ignore_index=True)
    fits = pd.concat([o[1] for o in out], ignore_index=True)
    F.to_csv(FIG_DIR / "phase6_forecasts.csv", index=False, float_format="%.8g")
    fits.to_csv(FIG_DIR / "phase6_fits.csv", index=False, float_format="%.5f")
    bad = (F[MODELS] <= 0).sum().sum()
    nonconv = int((~fits.loc[fits.model == "GJR-GARCH(1,1)-t", "converged"].astype(bool)).sum())
    print(f"{len(UNIVERSE)} assets, {len(F)} rows; non-positive forecasts: {bad}; "
          f"GJR fits not converged: {nonconv} of {int((fits.model == 'GJR-GARCH(1,1)-t').sum())}")
