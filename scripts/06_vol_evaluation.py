"""Phase 2, step 3: honest out-of-sample evaluation of the volatility filters.

Rules:
- Forecast h[t] (made at the close of day t) is scored against r[t+1]^2.
- Train = target days 2015..2021, Test = target days 2022..today.
- Window length N and lambda are chosen on TRAIN only (pooled QLIKE over the
  six tickers), then frozen. The test period is never used for any choice.
- All estimators are scored on exactly the same days (after the longest
  warm-up, 60 days), so the comparison is like-for-like.

Run:  PYTHONPATH=src .venv/bin/python scripts/06_vol_evaluation.py
Outputs: figures/vol_scores.csv (long table), figures/vol_tuning.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.evaluation import align_forecast, diebold_mariano, mse_loss, qlike_loss
from signaldesk.returns import log_returns
from signaldesk.volatility import constant_variance, ewma_variance, rolling_variance

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
WINDOWS = [10, 20, 60]
LAMBDAS = [0.94, 0.97]


def all_forecasts(r: pd.Series) -> dict[str, pd.Series]:
    f = {f"Boxcar N={n}": rolling_variance(r, n) for n in WINDOWS}
    f |= {f"EWMA λ={lam}": ewma_variance(r, lam) for lam in LAMBDAS}
    f["Constant (train avg)"] = constant_variance(r, TRAIN_END)
    return f


# ---- 1. Per-day losses for every (ticker, estimator) --------------------------
losses = {}   # (ticker, estimator) -> DataFrame[mse, qlike] indexed by target day
for t in TICKERS:
    r = 100 * log_returns(load_prices(t)["AdjClose"])   # percent units: variance in %^2
    fc = all_forecasts(r)
    aligned = {name: align_forecast(r, h) for name, h in fc.items()}
    common = sorted(set.intersection(*(set(a.index) for a in aligned.values())))
    for name, a in aligned.items():
        a = a.loc[common]
        losses[(t, name)] = pd.DataFrame({"mse": mse_loss(a.y, a.h), "qlike": qlike_loss(a.y, a.h)},
                                         index=a.index)

ESTIMATORS = list(all_forecasts(pd.Series([0.0] * 100, index=pd.bdate_range("2015", periods=100))))


def period(df, which):
    return df.loc[:TRAIN_END] if which == "train" else df.loc[TEST_START:]


# ---- 2. Tune on TRAIN only ---------------------------------------------------
def pooled_train_qlike(name):
    return np.mean([period(losses[(t, name)], "train")["qlike"].mean() for t in TICKERS])

best_box = min((f"Boxcar N={n}" for n in WINDOWS), key=pooled_train_qlike)
best_ewma = min((f"EWMA λ={lam}" for lam in LAMBDAS), key=pooled_train_qlike)
best_overall = min([best_box, best_ewma], key=pooled_train_qlike)
tuning = {"best_boxcar": best_box, "best_ewma": best_ewma, "best_overall": best_overall,
          "pooled_train_qlike": {e: round(pooled_train_qlike(e), 4) for e in ESTIMATORS}}
(FIG_DIR / "vol_tuning.json").write_text(json.dumps(tuning, indent=2, ensure_ascii=False))
print("Chosen on TRAIN only:", best_box, "|", best_ewma, "| best overall:", best_overall)

# ---- 3. Score table: estimator x ticker x (train, test) ----------------------
rows = []
base = "Constant (train avg)"
for (t, name), df in losses.items():
    for which in ["train", "test"]:
        d, b = period(df, which), period(losses[(t, base)], which)
        if name == base:   # comparing the baseline with itself is 0/0
            dm_q = dm_m = np.nan
        else:
            dm_q, _ = diebold_mariano(d["qlike"].values, b["qlike"].values)
            dm_m, _ = diebold_mariano(d["mse"].values, b["mse"].values)
        rows.append({
            "estimator": name, "ticker": t, "period": which, "n_days": len(d),
            "tuned_pick": name in (best_box, best_ewma),
            "MSE": d["mse"].mean(), "QLIKE": d["qlike"].mean(),
            # Positive = better than the constant baseline.
            "MSE_improvement_%": 100 * (1 - d["mse"].mean() / b["mse"].mean()),
            "QLIKE_gain": b["qlike"].mean() - d["qlike"].mean(),
            # Diebold-Mariano t-stat vs baseline: < -2 means significantly better.
            "DM_t_QLIKE": dm_q,
            "DM_t_MSE": dm_m,
        })
table = pd.DataFrame(rows)
table.to_csv(FIG_DIR / "vol_scores.csv", index=False, float_format="%.4f")

pd.set_option("display.width", 200)
order = ESTIMATORS
for metric, fmt in [("QLIKE_gain", "{:+.3f}"), ("MSE_improvement_%", "{:+.1f}"),
                    ("DM_t_QLIKE", "{:+.1f}"), ("DM_t_MSE", "{:+.1f}")]:
    for which in ["train", "test"]:
        pv = (table[table.period == which]
              .pivot(index="estimator", columns="ticker", values=metric)
              .loc[order, TICKERS])
        print(f"\n{metric}  [{which.upper()}]   (vs constant baseline)")
        print(pv.map(lambda v: fmt.format(v) if pd.notna(v) else "  -").to_string())
