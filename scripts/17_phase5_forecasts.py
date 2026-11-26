"""Phase 5: compute walk-forward forecasts for all models and cache them.

This script only COMPUTES forecasts (each causal, walk-forward); it does not
score anything. Scoring is split into a validation script (2022-2023) and a
final-test script (2024-today), so the test period cannot leak into the choice.

Run:  PYTHONPATH=src .venv/bin/python scripts/17_phase5_forecasts.py
Output: figures/phase5_forecasts.csv (gitignored, regenerable),
        figures/phase5_fits.csv (all fitted parameters per re-fit)
"""
from pathlib import Path

import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.ewma_family import CANDIDATES, c1_forecast, walk_forward
from signaldesk.returns import log_returns
from signaldesk.vol_models import garch_forecast
from signaldesk.volatility import constant_variance

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END_BASELINE = "2021-12-31"   # Phase 2 constant baseline (2015-2021 average)

rows, fits = [], []
for t in TICKERS:
    r = 100 * log_returns(load_prices(t)["AdjClose"])
    fc = {"C1 EWMA 0.94": c1_forecast(r)}
    for name in CANDIDATES:
        h, f = walk_forward(name, r)
        fc[name] = h
        fits.append(f.assign(ticker=t, model=name))
    fc["GJR-GARCH(1,1)-t"] = garch_forecast(r, asymmetric=True)[0]
    fc["Constant (2015-21 avg)"] = constant_variance(r, TRAIN_END_BASELINE)
    df = pd.DataFrame(fc)
    df["r"] = r
    df["ticker"] = t
    rows.append(df.rename_axis("date").reset_index())
    print(f"{t}: done")

pd.concat(rows).to_csv(FIG_DIR / "phase5_forecasts.csv", index=False, float_format="%.8g")
pd.concat(fits).to_csv(FIG_DIR / "phase5_fits.csv", index=False, float_format="%.5f")
