"""Forward test scorer: run this any time after new trading days exist.

What it does (append-only, never edits saved rows):
1. Downloads fresh daily prices (yfinance) into data/forward/ -- separate from
   the frozen research CSVs in data/raw/, which stay untouched.
   Use --no-download to reuse what is already in data/forward/.
2. For each trading day AFTER the last logged forecast, computes the frozen
   models' forecasts (fixed parameters from forward/frozen_model.json, data up
   to that day only) and APPENDS them to forward/forecasts.csv.
3. Scores every logged forecast whose target day (the next trading day) now
   has a return and has not been scored yet, APPENDING to forward/score_log.csv.
4. Checks that previously logged forecasts are reproduced by the current data
   (warns if the data vendor revised history; the logged value always stands).
5. Prints the running pooled QLIKE of each model since the freeze date.

Run:  PYTHONPATH=src .venv/bin/python scripts/score_forward.py [--no-download]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.data import TICKERS, download_prices, load_prices, save_prices
from signaldesk.evaluation import diebold_mariano
from signaldesk.forward import FORECASTS, FROZEN, MODELS, SCORES, model_paths, qlike
from signaldesk.returns import log_returns

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "forward"


def append(path: Path, df: pd.DataFrame):
    """Append rows; write a header only if the file is new. Never rewrites."""
    if len(df):
        df.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.8g")


def main(download: bool):
    frozen = json.loads(FROZEN.read_text())
    freeze = pd.Timestamp(frozen["freeze_date"])
    logged = pd.read_csv(FORECASTS, parse_dates=["made_on"])
    scored = pd.read_csv(SCORES, parse_dates=["made_on"]) if SCORES.exists() else None

    new_fc, new_sc = [], []
    for t in TICKERS:
        if download:
            save_prices(download_prices(t), t, DATA_DIR)
        r = 100 * log_returns(load_prices(t, DATA_DIR)["AdjClose"])
        h = model_paths(r, frozen["params"][t])
        mine = logged[logged.ticker == t]

        # (4) Integrity check against already-logged forecasts.
        for _, row in mine.iterrows():
            d = row.made_on
            if d in h.index and not np.allclose(h.loc[d, MODELS].values.astype(float),
                                                row[MODELS].values.astype(float), rtol=1e-3):
                print(f"WARNING {t} {d.date()}: data revision changes this forecast; "
                      f"the logged value is kept.")

        # (2) New forecasts: days after the last logged one.
        last = mine.made_on.max()
        for d in h.index[h.index > last]:
            new_fc.append({"ticker": t, "made_on": d.date(), **{m: h.loc[d, m] for m in MODELS}})

        # (3) Score logged forecasts whose next trading day now exists.
        done = set() if scored is None else set(scored[scored.ticker == t].made_on)
        all_fc = pd.concat([mine, pd.DataFrame([x for x in new_fc if x["ticker"] == t])])
        for _, row in all_fc.iterrows():
            d = pd.Timestamp(row.made_on)
            if d in done or d < freeze:
                continue
            nxt = r.index[r.index > d]
            if len(nxt) == 0:
                continue                       # target day not available yet
            y = r.loc[nxt[0]] ** 2
            new_sc.append({"ticker": t, "made_on": d.date(), "target_date": nxt[0].date(),
                           "r2": y, **{f"qlike {m}": qlike(y, float(row[m])) for m in MODELS}})

    append(FORECASTS, pd.DataFrame(new_fc))
    append(SCORES, pd.DataFrame(new_sc))
    print(f"appended {len(new_fc)} forecasts and {len(new_sc)} scores")

    # (5) Running summary since the freeze.
    if SCORES.exists():
        S = pd.read_csv(SCORES, parse_dates=["target_date"])
        P = S.groupby("target_date")[[f"qlike {m}" for m in MODELS]].mean()   # pooled per day
        print(f"\nForward test since {freeze.date()}: {len(P)} target days")
        print(P.mean().round(4).to_string())
        if len(P) >= 30:
            tstat = diebold_mariano(P["qlike C3 asymmetric"].values,
                                    P["qlike GJR-GARCH(1,1)-t"].values)[0]
            print(f"pooled DM t, C3 vs GJR: {tstat:+.2f}  (< -2: C3 significantly better)")
        else:
            print("(fewer than 30 days: too early for a significance test)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    main(download=not ap.parse_args().no_download)
