"""Phase 4, steps 4-5: walk-forward benchmark of EWMA vs GARCH, GJR-GARCH, HAR-style.

- Every forecast h[t] (made at the close of day t) is scored against r[t+1]^2.
- All models are scored on exactly the same target days: from the first day
  every model has a forecast (after the 3-year GARCH/HAR warm-up) to today.
- Test period = 2022-today; plus a year-by-year breakdown.
- Diebold-Mariano (Newey-West) tests: each model vs EWMA, per ticker, and once
  on the average loss across the six tickers (a single test that respects the
  fact that the tickers move together).

Run:  PYTHONPATH=src .venv/bin/python scripts/15_model_benchmark.py
Outputs (figures/): benchmark_test.csv, benchmark_by_year.csv,
                    benchmark_complexity.csv, benchmark_fit_diagnostics.csv,
                    benchmark_losses.csv (per-day losses, for the plots)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.evaluation import align_forecast, diebold_mariano, mse_loss, qlike_loss
from signaldesk.returns import log_returns
from signaldesk.vol_models import garch_forecast, har_forecast
from signaldesk.volatility import constant_variance, ewma_variance

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
MODELS = ["EWMA λ=0.94", "GARCH(1,1)-t", "GJR-GARCH(1,1)-t", "HAR-style", "Constant (train avg)"]
N_PARAMS = {"EWMA λ=0.94": 0, "GARCH(1,1)-t": 4, "GJR-GARCH(1,1)-t": 5, "HAR-style": 4,
            "Constant (train avg)": 1}


def forecasts_for(r: pd.Series):
    """All five forecasts for one ticker, with wall-clock runtime and fit info."""
    out, runtime, diag = {}, {}, {}
    jobs = {
        "EWMA λ=0.94": lambda: (ewma_variance(r, 0.94), None),
        "GARCH(1,1)-t": lambda: garch_forecast(r, asymmetric=False),
        "GJR-GARCH(1,1)-t": lambda: garch_forecast(r, asymmetric=True),
        "HAR-style": lambda: har_forecast(r),
        "Constant (train avg)": lambda: (constant_variance(r, TRAIN_END), None),
    }
    for name, job in jobs.items():
        t0 = time.perf_counter()
        h, fits = job()
        runtime[name] = time.perf_counter() - t0
        out[name] = h
        diag[name] = fits
    return out, runtime, diag


# ---- 1. Per-day losses on common days -------------------------------------
loss_rows, runtimes, diag_rows = [], [], []
for t in TICKERS:
    r = 100 * log_returns(load_prices(t)["AdjClose"])
    fc, rt, diag = forecasts_for(r)
    aligned = {m: align_forecast(r, h) for m, h in fc.items()}
    common = sorted(set.intersection(*(set(a.index) for a in aligned.values())))
    for m, a in aligned.items():
        a = a.loc[common]
        assert (a["h"] > 0).all(), f"non-positive forecast: {t} {m}"
        loss_rows.append(pd.DataFrame({"ticker": t, "model": m, "date": a.index,
                                       "h": a["h"].values, "y": a["y"].values,
                                       "qlike": qlike_loss(a.y, a.h).values,
                                       "mse": mse_loss(a.y, a.h).values}))
    runtimes += [{"ticker": t, "model": m, "seconds": s} for m, s in rt.items()]
    for m in ["GARCH(1,1)-t", "GJR-GARCH(1,1)-t"]:
        d = diag[m]
        diag_rows.append({"ticker": t, "model": m, "n_fits": len(d),
                          "non_converged": int((~d["converged"]).sum()),
                          "mean_persistence": float((d["alpha"] + d["gamma"] / 2 + d["beta"]).mean()),
                          "mean_nu": float(d["nu"].mean()),
                          "mean_gamma": float(d["gamma"].mean())})
    d = diag["HAR-style"]
    diag_rows.append({"ticker": t, "model": "HAR-style", "n_fits": len(d),
                      "har_floor_days": int(d["floor_used"].sum()),
                      "mean_b_day": d["b_day"].mean(), "mean_b_week": d["b_week"].mean(),
                      "mean_b_month": d["b_month"].mean()})
    print(f"{t}: scored days {common[0].date()} .. {common[-1].date()} ({len(common)})")

L = pd.concat(loss_rows, ignore_index=True)
L.to_csv(FIG_DIR / "benchmark_losses.csv", index=False, float_format="%.6g")
pd.DataFrame(diag_rows).to_csv(FIG_DIR / "benchmark_fit_diagnostics.csv", index=False,
                               float_format="%.4f")
RT = pd.DataFrame(runtimes)
test = L[L.date >= TEST_START]


# ---- 2. Test period table --------------------------------------------------
def by_model(df, t, m):
    return df[(df.ticker == t) & (df.model == m)].set_index("date")


rows = []
for t in TICKERS:
    base, ewma = by_model(test, t, "Constant (train avg)"), by_model(test, t, "EWMA λ=0.94")
    for m in MODELS:
        d = by_model(test, t, m)
        dm_q = dm_m = np.nan
        if m != "EWMA λ=0.94":
            dm_q, _ = diebold_mariano(d.qlike.values, ewma.qlike.values)
            dm_m, _ = diebold_mariano(d.mse.values, ewma.mse.values)
        rows.append({"ticker": t, "model": m, "QLIKE": d.qlike.mean(), "MSE": d.mse.mean(),
                     "QLIKE_gain_vs_baseline": base.qlike.mean() - d.qlike.mean(),
                     "QLIKE_gain_vs_EWMA": ewma.qlike.mean() - d.qlike.mean(),
                     "MSE_change_vs_EWMA_%": 100 * (d.mse.mean() / ewma.mse.mean() - 1),
                     "DM_t_QLIKE_vs_EWMA": dm_q, "DM_t_MSE_vs_EWMA": dm_m})
T = pd.DataFrame(rows)
T.to_csv(FIG_DIR / "benchmark_test.csv", index=False, float_format="%.4f")

# Pooled DM: average each day's loss over the six tickers, then one test.
pooled = {}
for m in MODELS:
    if m == "EWMA λ=0.94":
        continue
    a = test[test.model == m].groupby("date")[["qlike", "mse"]].mean()
    e = test[test.model == "EWMA λ=0.94"].groupby("date")[["qlike", "mse"]].mean()
    pooled[m] = (diebold_mariano(a.qlike.values, e.qlike.values)[0],
                 diebold_mariano(a.mse.values, e.mse.values)[0])

pd.set_option("display.width", 220)
for col, fmt in [("QLIKE_gain_vs_EWMA", "{:+.3f}"), ("DM_t_QLIKE_vs_EWMA", "{:+.1f}"),
                 ("MSE_change_vs_EWMA_%", "{:+.1f}"), ("DM_t_MSE_vs_EWMA", "{:+.1f}"),
                 ("QLIKE_gain_vs_baseline", "{:+.3f}")]:
    pv = T.pivot(index="model", columns="ticker", values=col).loc[MODELS, TICKERS]
    print(f"\nTEST 2022-today: {col}")
    print(pv.map(lambda v: fmt.format(v) if pd.notna(v) else "   -").to_string())
print("\nPooled DM t vs EWMA (average loss over 6 tickers; < -2 = model significantly better):")
for m, (q, s) in pooled.items():
    print(f"  {m:22s} QLIKE {q:+.2f}   MSE {s:+.2f}")

# ---- 3. Year by year: QLIKE gain vs EWMA (positive = model better than EWMA) --
L["year"] = pd.DatetimeIndex(L.date).year
yr = L.groupby(["model", "ticker", "year"]).qlike.mean().unstack("model")
gain = yr.rsub(yr["EWMA λ=0.94"], axis=0)        # EWMA loss - model loss
gain_long = gain.stack().rename("QLIKE_gain_vs_EWMA").reset_index()
gain_long.to_csv(FIG_DIR / "benchmark_by_year.csv", index=False, float_format="%.4f")
avg_year = gain.groupby("year").mean()[MODELS[1:]]
print("\nQLIKE gain vs EWMA by year, averaged over 6 tickers (positive = beats EWMA):")
print(avg_year.round(3).to_string())
wins = (gain[MODELS[1:]] > 0).groupby("year").sum()
print("\nNumber of tickers (of 6) where the model beats EWMA, by year:")
print(wins.to_string())

# ---- 4. Complexity vs benefit ---------------------------------------------
C = pd.DataFrame({
    "parameters_fitted": pd.Series(N_PARAMS),
    "test_QLIKE_gain_vs_baseline_mean": T.groupby("model").QLIKE_gain_vs_baseline.mean(),
    "test_QLIKE_gain_vs_EWMA_mean": T.groupby("model").QLIKE_gain_vs_EWMA.mean(),
    "tickers_beating_EWMA_on_test": T.groupby("model").QLIKE_gain_vs_EWMA.apply(lambda s: int((s > 0).sum())),
    "pooled_DM_t_QLIKE_vs_EWMA": pd.Series({m: v[0] for m, v in pooled.items()}),
    "runtime_s_per_ticker": RT.groupby("model").seconds.mean(),
}).loc[MODELS]
C.index.name = "model"
C.to_csv(FIG_DIR / "benchmark_complexity.csv", float_format="%.4f")
print("\nComplexity vs benefit (test 2022-today):")
print(C.round(3).to_string())
print("\nFit diagnostics:")
print(pd.DataFrame(diag_rows).round(3).to_string(index=False))
