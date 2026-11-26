"""Phase 5, step 3: validation on 2022-2023 ONLY, and the pre-registered selection rule.

Nothing from 2024 onward is read or printed here.

Rule (notes/phase5_plan.md):
1. Among C2-C5, find the lowest pooled QLIKE on validation.
2. If a candidate with FEWER parameters is within noise of it (pooled DM |t| < 2),
   pick the simpler one (ties on parameter count -> lower QLIKE).
3. If the pick is not better than C1 on validation, the frozen choice is C1.

Run:  PYTHONPATH=src .venv/bin/python scripts/18_phase5_validation.py
Output: figures/phase5_validation.csv, figures/phase5_frozen_choice.json
"""
import json

import pandas as pd

from signaldesk.ewma_family import N_PARAMS
from signaldesk.phase5_scoring import FIG_DIR, MODELS, load_losses, pooled_daily, pooled_dm

VAL_START, VAL_END = "2022-01-01", "2023-12-31"
L = load_losses(VAL_START, VAL_END)
assert L.date.max() <= pd.Timestamp(VAL_END)

rows = []
for m in MODELS:
    rows.append({"model": m, "params": N_PARAMS.get(m, {"GJR-GARCH(1,1)-t": 5}.get(m, 1)),
                 "pooled_QLIKE": pooled_daily(L, m).mean(),
                 "pooled_MSE": pooled_daily(L, m, "mse").mean(),
                 "QLIKE_gain_vs_C1": pooled_daily(L, "C1 EWMA 0.94").mean() - pooled_daily(L, m).mean(),
                 "DM_t_vs_C1": pooled_dm(L, m, "C1 EWMA 0.94") if m != "C1 EWMA 0.94" else float("nan"),
                 "DM_t_vs_GJR": pooled_dm(L, m, "GJR-GARCH(1,1)-t") if m != "GJR-GARCH(1,1)-t" else float("nan")})
V = pd.DataFrame(rows).set_index("model")
V.to_csv(FIG_DIR / "phase5_validation.csv", float_format="%.5f")
pd.set_option("display.width", 200)
print(f"VALIDATION {VAL_START}..{VAL_END}  ({L.date.nunique()} days x 6 tickers)")
print(V.round(4).to_string())

# ---- Apply the selection rule ---------------------------------------------
cands = ["C2 two-speed", "C3 asymmetric", "C4 two-speed+asym", "C5 pull-to-level"]
best = min(cands, key=lambda m: V.loc[m, "pooled_QLIKE"])
log = [f"lowest validation QLIKE among C2-C5: {best}"]
pick = best
for m in sorted(cands, key=lambda m: (N_PARAMS[m], V.loc[m, "pooled_QLIKE"])):
    if N_PARAMS[m] >= N_PARAMS[best] or m == best:
        continue
    t = pooled_dm(L, m, best)
    log.append(f"  simpler {m} vs {best}: DM t = {t:+.2f}")
    if abs(t) < 2:
        pick = m
        log.append(f"  -> within noise: prefer simpler {m}")
        break
if V.loc[pick, "pooled_QLIKE"] >= V.loc["C1 EWMA 0.94", "pooled_QLIKE"]:
    log.append(f"{pick} is not better than C1 on validation -> frozen choice is C1")
    pick = "C1 EWMA 0.94"
log.append(f"FROZEN CHOICE: {pick}")
print("\n" + "\n".join(log))
(FIG_DIR / "phase5_frozen_choice.json").write_text(json.dumps(
    {"frozen_choice": pick, "selection_log": log, "validation_period": [VAL_START, VAL_END]},
    indent=2))
