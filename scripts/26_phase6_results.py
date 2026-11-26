"""Phase 6, step 4: results per asset, per class and pooled; H1-H3 judged by the
pre-registered rules in notes/phase6_plan.md (applied literally, no changes).

Run (after 25_phase6_run_models.py):  PYTHONPATH=src .venv/bin/python scripts/26_phase6_results.py
Outputs: figures/phase6_per_asset.csv, figures/phase6_per_class.csv,
         figures/phase6_pooled.csv, figures/phase6_hypotheses.json, figures/phase6_gains.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from signaldesk.evaluation import diebold_mariano
from signaldesk.phase6_scoring import FIG_DIR, load_losses, mean_gain, pooled_daily, pooled_dm
from signaldesk.universe import ASSET_CLASS, CLASSES, EQUITY_TYPE, UNIVERSE

EW, C3, GJR, CONST, A0 = ("EWMA 0.94", "C3 asymmetric", "GJR-GARCH(1,1)-t",
                          "Constant (2015-21 avg)", "C3 with a=0 (diagnostic)")
PERIODS = {"2018-today": "2018-01-01", "2024-today": "2024-01-01"}
L = {p: load_losses(s, "2026-11-25") for p, s in PERIODS.items()}
checks = pd.read_csv(FIG_DIR / "phase6_data_checks.csv").set_index("ticker")

# ---- Per asset ---------------------------------------------------------------
rows = []
for p, Lp in L.items():
    g_gjr, g_c3 = mean_gain(Lp, GJR, EW), mean_gain(Lp, C3, EW)
    g_const, g_asym = mean_gain(Lp, CONST, EW), mean_gain(Lp, C3, A0)
    mse = Lp.groupby(["ticker", "model"]).mse.mean().unstack()
    for t in UNIVERSE:
        d = Lp[Lp.ticker == t]
        loss = {m: d[d.model == m].set_index("date") for m in [EW, C3, GJR, A0]}
        rows.append({
            "period": p, "ticker": t, "class": ASSET_CLASS[t], "n_days": len(loss[EW]),
            "ann_vol_2015_21_%": checks.loc[t, "ann_vol_2015_21_%"],
            "GJR_gain_vs_EWMA": g_gjr[t], "C3_gain_vs_EWMA": g_c3[t],
            "Const_gain_vs_EWMA": g_const[t], "asym_gain_C3_vs_a0": g_asym[t],
            "DM_t_GJR_vs_EWMA": diebold_mariano(loss[GJR].qlike.values, loss[EW].qlike.values)[0],
            "DM_t_C3_vs_EWMA": diebold_mariano(loss[C3].qlike.values, loss[EW].qlike.values)[0],
            "DM_t_C3_vs_a0": diebold_mariano(loss[C3].qlike.values, loss[A0].qlike.values)[0],
            "MSE_change_GJR_vs_EWMA_%": 100 * (mse.loc[t, GJR] / mse.loc[t, EW] - 1),
            "MSE_change_C3_vs_EWMA_%": 100 * (mse.loc[t, C3] / mse.loc[t, EW] - 1),
        })
A = pd.DataFrame(rows)
A.to_csv(FIG_DIR / "phase6_per_asset.csv", index=False, float_format="%.4f")

# ---- Per class (mean of per-asset gains; class-pooled DM where >1 asset) ---------
crow = []
for p, Lp in L.items():
    for c, ts in CLASSES.items():
        sub = Lp[Lp.ticker.isin(ts)]
        a = A[(A.period == p) & (A["class"] == c)]
        crow.append({"period": p, "class": c, "n_assets": len(ts),
                     "GJR_gain_vs_EWMA": a.GJR_gain_vs_EWMA.mean(),
                     "C3_gain_vs_EWMA": a.C3_gain_vs_EWMA.mean(),
                     "asym_gain_C3_vs_a0": a.asym_gain_C3_vs_a0.mean(),
                     "GJR_beats_EWMA": f"{int((a.GJR_gain_vs_EWMA > 0).sum())}/{len(ts)}",
                     "pooled_DM_t_GJR_vs_EWMA": pooled_dm(sub, GJR, EW),
                     "pooled_DM_t_C3_vs_EWMA": pooled_dm(sub, C3, EW)})
C = pd.DataFrame(crow)
C.to_csv(FIG_DIR / "phase6_per_class.csv", index=False, float_format="%.4f")

# ---- Pooled over all 26 ------------------------------------------------------
prow = []
for p, Lp in L.items():
    base = pooled_daily(Lp, EW).mean()
    for m in [C3, GJR, CONST, A0]:
        prow.append({"period": p, "model": m,
                     "pooled_QLIKE": pooled_daily(Lp, m).mean(), "pooled_QLIKE_EWMA": base,
                     "gain_vs_EWMA": base - pooled_daily(Lp, m).mean(),
                     "DM_t_QLIKE_vs_EWMA": pooled_dm(Lp, m, EW),
                     "DM_t_MSE_vs_EWMA": pooled_dm(Lp, m, EW, "mse")})
    prow.append({"period": p, "model": "C3 vs GJR", "DM_t_QLIKE_vs_EWMA": np.nan,
                 "gain_vs_EWMA": pooled_daily(Lp, GJR).mean() - pooled_daily(Lp, C3).mean(),
                 "DM_t_MSE_vs_EWMA": np.nan, "pooled_QLIKE": np.nan, "pooled_QLIKE_EWMA": np.nan,
                 "note": f"DM t C3 vs GJR = {pooled_dm(Lp, C3, GJR):+.2f} (gain = GJR loss - C3 loss)"})
P = pd.DataFrame(prow)
P.to_csv(FIG_DIR / "phase6_pooled.csv", index=False, float_format="%.4f")

# ---- Hypotheses, judged literally -----------------------------------------------
def judge_h1(t):
    return "HOLDS" if t < -2 else ("FAILS" if t > 2 else "INCONCLUSIVE")

h1 = {p: {"pooled_DM_t_GJR_vs_EWMA": pooled_dm(Lp, GJR, EW),
          "verdict": judge_h1(pooled_dm(Lp, GJR, EW))} for p, Lp in L.items()}

a18 = A[A.period == "2018-today"].set_index("ticker")
eq_median = a18.loc[EQUITY_TYPE, "asym_gain_C3_vs_a0"].median()
non_eq = {t: a18.loc[t, "asym_gain_C3_vs_a0"] for t in ["GLD", "TLT", "BTC-USD"]}
n_above = sum(v > eq_median for v in non_eq.values())
h2_verdict = "HOLDS" if n_above == 0 else ("FAILS" if n_above >= 2 else "INCONCLUSIVE")

g = a18["GJR_gain_vs_EWMA"]
vol_hi, calm = ["BTC-USD", "EEM"], ["TLT", "GLD"]
if min(g[vol_hi]) > max(g[calm]):
    h3_verdict = "HOLDS"
elif min(g[calm]) > max(g[vol_hi]):
    h3_verdict = "FAILS"
else:
    h3_verdict = "INCONCLUSIVE"
rho, rho_p = spearmanr(a18["ann_vol_2015_21_%"], g)

H = {"H1": h1,
     "H2": {"equity_type_median_asym_gain": eq_median, "GLD_TLT_BTC_asym_gain": non_eq,
            "n_above_median": n_above, "verdict": h2_verdict},
     "H3": {"GJR_gain_vs_EWMA": {t: g[t] for t in vol_hi + calm}, "verdict": h3_verdict,
            "descriptive_spearman_gain_vs_vol": {"rho": rho, "p": rho_p}}}
(FIG_DIR / "phase6_hypotheses.json").write_text(json.dumps(H, indent=2, default=float))

pd.set_option("display.width", 230)
cols = ["ticker", "class", "ann_vol_2015_21_%", "GJR_gain_vs_EWMA", "DM_t_GJR_vs_EWMA",
        "C3_gain_vs_EWMA", "DM_t_C3_vs_EWMA", "asym_gain_C3_vs_a0", "DM_t_C3_vs_a0"]
for p in PERIODS:
    print(f"\n===== PER ASSET {p} =====")
    print(A[A.period == p][cols].round(3).to_string(index=False))
print("\n===== PER CLASS =====")
print(C.round(3).to_string(index=False))
print("\n===== POOLED (26 assets) =====")
print(P.round(4).to_string(index=False))
print("\n===== HYPOTHESES =====")
print(json.dumps(H, indent=2, default=lambda x: round(float(x), 4)))

# ---- Figure: gain vs plain EWMA per asset, GJR and C3, both periods ----------------
order = [t for c in CLASSES for t in CLASSES[c]]
fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
INK2 = "#666666"
for ax, p in zip(axes, PERIODS):
    a = A[A.period == p].set_index("ticker").loc[order]
    y = np.arange(len(order))[::-1]
    ax.axvline(0, color=INK2, linestyle="--", linewidth=1)
    ax.scatter(a.GJR_gain_vs_EWMA, y + 0.15, color="#f47230", marker="^", s=40, label="GJR-GARCH(1,1)-t",
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(a.C3_gain_vs_EWMA, y - 0.15, color="#257de0", marker="o", s=36, label="C3 asymmetric EWMA",
               edgecolor="white", linewidth=0.6, zorder=3)
    # Shade alternate asset classes for readability.
    start = 0
    for i, (c, ts) in enumerate(CLASSES.items()):
        if i % 2 == 0:
            ax.axhspan(len(order) - start - len(ts) - 0.5, len(order) - start - 0.5,
                       color="#f4f4f4", zorder=0)
        ax.text(1.0, len(order) - start - 1, c, transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=7.5, color=INK2)
        start += len(ts)
    ax.set_title(f"{p}", loc="left", fontsize=10)
    ax.set_xlabel("QLIKE gain vs plain EWMA λ=0.94  (> 0 = better than EWMA)")
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_yticks(np.arange(len(order))[::-1], order)
axes[0].legend(loc="lower left", fontsize=8, frameon=False)
fig.suptitle("Phase 6: frozen models on 26 new assets: gain over the plain EWMA",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG_DIR / "phase6_gains.png", dpi=150)
