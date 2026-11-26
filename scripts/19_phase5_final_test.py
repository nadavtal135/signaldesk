"""Phase 5, step 4: the FINAL TEST (2024-01-01 .. 2026-11-25), run once.

Frozen winner (figures/phase5_frozen_choice.json) vs plain EWMA (C1) vs
GJR-GARCH(1,1)-t vs the constant baseline. Per ticker and pooled, with
Diebold-Mariano tests and a year-by-year table. The pre-registered pass/fail
rule is applied literally at the end. Nothing is tuned here.

Run:  PYTHONPATH=src .venv/bin/python scripts/19_phase5_final_test.py
Outputs: figures/phase5_final_test.csv, figures/phase5_final_by_year.csv,
         figures/phase5_final_verdict.json
"""
import json

import numpy as np
import pandas as pd

from signaldesk.evaluation import diebold_mariano
from signaldesk.phase5_scoring import FIG_DIR, load_losses, pooled_daily, pooled_dm

TEST_START, TEST_END = "2024-01-01", "2026-11-25"
WIN = json.loads((FIG_DIR / "phase5_frozen_choice.json").read_text())["frozen_choice"]
GJR, C1, CONST = "GJR-GARCH(1,1)-t", "C1 EWMA 0.94", "Constant (2015-21 avg)"
SHOW = [WIN, C1, GJR, CONST]

L = load_losses(TEST_START, TEST_END)
L = L[L.model.isin(SHOW)]
print(f"FINAL TEST {TEST_START}..{TEST_END}: {L.date.nunique()} days x 6 tickers; frozen winner = {WIN}\n")

# ---- Per ticker ----------------------------------------------------------
rows = []
for t, g in L.groupby("ticker", sort=False):
    loss = {m: g[g.model == m].set_index("date") for m in SHOW}
    for m in SHOW:
        rows.append({
            "ticker": t, "model": m,
            "QLIKE": loss[m].qlike.mean(), "MSE": loss[m].mse.mean(),
            "QLIKE_gain_vs_GJR": loss[GJR].qlike.mean() - loss[m].qlike.mean(),
            "QLIKE_gain_vs_C1": loss[C1].qlike.mean() - loss[m].qlike.mean(),
            "DM_t_QLIKE_vs_GJR": np.nan if m == GJR else
                diebold_mariano(loss[m].qlike.values, loss[GJR].qlike.values)[0],
            "DM_t_QLIKE_vs_C1": np.nan if m == C1 else
                diebold_mariano(loss[m].qlike.values, loss[C1].qlike.values)[0],
        })
T = pd.DataFrame(rows)
pooled = []
for m in SHOW:
    pooled.append({"ticker": "POOLED", "model": m,
                   "QLIKE": pooled_daily(L, m).mean(), "MSE": pooled_daily(L, m, "mse").mean(),
                   "QLIKE_gain_vs_GJR": pooled_daily(L, GJR).mean() - pooled_daily(L, m).mean(),
                   "QLIKE_gain_vs_C1": pooled_daily(L, C1).mean() - pooled_daily(L, m).mean(),
                   "DM_t_QLIKE_vs_GJR": np.nan if m == GJR else pooled_dm(L, m, GJR),
                   "DM_t_QLIKE_vs_C1": np.nan if m == C1 else pooled_dm(L, m, C1),
                   "DM_t_MSE_vs_GJR": np.nan if m == GJR else pooled_dm(L, m, GJR, "mse")})
T = pd.concat([T, pd.DataFrame(pooled)], ignore_index=True)
T.to_csv(FIG_DIR / "phase5_final_test.csv", index=False, float_format="%.5f")
pd.set_option("display.width", 220)
print(T.set_index(["ticker", "model"]).round(4).to_string())

# ---- Year by year (pooled) ----------------------------------------------------
L["year"] = L.date.dt.year
Y = (L.groupby(["year", "model", "date"]).qlike.mean()    # pooled over tickers per day
       .groupby(["year", "model"]).mean().unstack("model")[SHOW])
Y.to_csv(FIG_DIR / "phase5_final_by_year.csv", float_format="%.5f")
print("\nPooled QLIKE by year (lower = better):")
print(Y.round(4).to_string())
yr_dm = {y: pooled_dm(L[L.year == y], WIN, GJR) for y in Y.index}
print("Per-year pooled DM t (winner vs GJR):", {y: round(v, 2) for y, v in yr_dm.items()})

# ---- Pre-registered pass/fail rule -----------------------------------------
dm_t = pooled_dm(L, WIN, GJR)
years_won = int((Y[WIN] < Y[GJR]).sum())
cond1 = dm_t < -2
cond2 = years_won >= 2
verdict = "PASS" if (cond1 and cond2) else "did not beat GJR-GARCH"
print(f"\nRULE: (1) pooled DM t vs GJR < -2: t = {dm_t:+.2f} -> {cond1}")
print(f"      (2) lower pooled QLIKE than GJR in >= 2 of 3 years: {years_won}/3 -> {cond2}")
print(f"VERDICT: {verdict}")
(FIG_DIR / "phase5_final_verdict.json").write_text(json.dumps({
    "frozen_winner": WIN, "test_period": [TEST_START, TEST_END],
    "pooled_DM_t_vs_GJR": dm_t, "years_won_vs_GJR": years_won,
    "per_year_DM_t_vs_GJR": yr_dm, "condition1": bool(cond1), "condition2": bool(cond2),
    "verdict": verdict}, indent=2))
